import os
import json
import torch
from pathlib import Path
from typing import Dict
import matplotlib.pyplot as plt

from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.utilities import rank_zero_only

from utils.configs import ExperimentConfigs
from data.particle_clouds.particles import ParticleClouds
from data.particle_clouds.jets import JetClassHighLevelFeatures
from model.multimodal_bridge_matching import HybridState


class ModelCheckpointCallback(ModelCheckpoint):
    """
    A wrapper around Lightning's ModelCheckpoint to initialize using ExperimentConfigs.
    """

    def __init__(self, config: ExperimentConfigs):
        """
        Initialize the callback using a configuration object.

        Args:
            config (ExperimentConfigs): The configuration object containing checkpoint settings.
        """
        checkpoint_config = config.checkpoints.to_dict()
        super().__init__(**checkpoint_config)


class ExperimentLoggerCallback(Callback):
    """
    Callback to log epoch-level metrics dynamically during training and validation,
    supporting additional custom metrics beyond loss.
    """

    def __init__(self, config: ExperimentConfigs):
        super().__init__()
        self.config = config
        self.sync_dist = False
        self.epoch_metrics = {"train": {}, "val": {}}

    def setup(self, trainer, pl_module, stage=None):
        """Set up distributed synchronization if required."""
        self.sync_dist = trainer.world_size > 1
        self.checkpoint_dir_created = False if hasattr(trainer, "config") else True

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Accumulate metrics for epoch-level logging during training."""
        self._track_metrics("train", outputs)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Accumulate metrics for epoch-level logging during validation."""
        self._track_metrics("val", outputs)

    def on_train_epoch_end(self, trainer, pl_module):
        """Log metrics at the end of a training epoch."""
        self._log_epoch_metrics("train", pl_module, trainer)
        self._save_metada_to_dir(trainer)

    def on_validation_epoch_end(self, trainer, pl_module):
        """Log metrics at the end of a validation epoch."""
        self._log_epoch_metrics("val", pl_module, trainer)

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        return super().on_train_end(trainer, pl_module)

    @rank_zero_only
    def _save_metada_to_dir(self, trainer):
        """Save metadata to the checkpoint directory"""
        while not self.checkpoint_dir_created and Path(trainer.config.path).exists():
            self.checkpoint_dir_created = True
            trainer.config.save(Path(trainer.config.path))
            with open(Path(trainer.config.path) / "metadata.json", "w") as f:
                json.dump(trainer.metadata, f, indent=4)

    def _track_metrics(self, stage: str, outputs: Dict[str, torch.Tensor]):
        """
        Accumulate metrics for epoch-level logging.
        Args:
            stage (str): Either "train" or "val".
            outputs (Dict[str, Any]): Dictionary of metrics from the batch.
        """
        for key, value in outputs.items():
            if key not in self.epoch_metrics[stage]:
                self.epoch_metrics[stage][key] = []
            if isinstance(value, torch.Tensor):  # Handle tensor values
                self.epoch_metrics[stage][key].append(value.detach().cpu().item())
            elif isinstance(value, (float, int)):  # Handle float or int values
                self.epoch_metrics[stage][key].append(value)
            else:
                raise TypeError(
                    f"Unsupported metric type for key '{key}': {type(value)}"
                )

    def _log_epoch_metrics(self, stage: str, pl_module, trainer):
        """
        Compute and log metrics for the epoch, and log them using the Comet logger if available.
        Args:
            stage (str): Either "train" or "val".
            pl_module: LightningModule to log metrics.
            trainer: The Lightning Trainer instance.
        """
        epoch_metrics = {}
        for key, values in self.epoch_metrics[stage].items():
            epoch_metric = sum(values) / len(values)
            self.log(
                key,
                epoch_metric,
                on_epoch=True,
                logger=True,
                sync_dist=self.sync_dist,
            )
            epoch_metrics[key] = epoch_metric
        self.epoch_metrics[stage].clear()  # Reset for next epoch


class JetsGenerativeCallback(Callback):
    def __init__(self, config: ExperimentConfigs):
        super().__init__()
        self.config = config
        self.batched_gen_states = []
        self.batched_source_states = []
        self.batched_target_states = []

    def on_predict_start(self, trainer, pl_module):
        self.data_dir = os.path.join(self.config.path, "data")
        self.metric_dir = os.path.join(self.config.path, "metrics")
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.metric_dir, exist_ok=True)
        #...load metadata for postprocessing
        # with open(os.path.join(self.config.path, "metadata.json"), "r") as f:


    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if outputs is not None:
            self.batched_gen_states.append(outputs[0])
            self.batched_source_states.append(outputs[1])
            self.batched_target_states.append(outputs[2])

    def on_predict_end(self, trainer, pl_module):
        # ...generated sample
        gen_sample = HybridState.cat(self.batched_gen_states)
        self.gen_sample = ParticleClouds(dataset=gen_sample)
        self.gen_sample.stats = self.config.data.target.preprocess.stats.to_dict()
        self.gen_sample.postprocess()
        gen_jets = JetClassHighLevelFeatures(constituents=self.gen_sample)

        # ...test sample
        test_sample = HybridState.cat(self.batched_target_states)
        self.test_sample = ParticleClouds(dataset=test_sample)
        test_jets = JetClassHighLevelFeatures(constituents=self.test_sample)

        # ...log results

        metrics = self.compute_performance_metrics(gen_jets, test_jets)
        self.gen_sample.save_to(path=os.path.join(self.data_dir, "generated_sample.h5"))
        self.test_sample.save_to(path=os.path.join(self.data_dir, "test_sample.h5"))
        with open(os.path.join(self.metric_dir, "performance_metrics.h5"), "w") as f:
            json.dump(metrics, f, indent=4)

        if hasattr(self.config.experiment, "comet_logger"):
            figures = self.get_results_plots(gen_jets, test_jets)
            trainer.logger.experiment.log_metrics(metrics)
            for key in figures.keys():
                trainer.logger.experiment.log_figure(
                    figure=figures[key], figure_name=key
                )

    def compute_performance_metrics(self, gen_jets, test_jets):
        return {
            "Wasserstein1D_pt": test_jets.Wassertein1D("pt", gen_jets),
            "Wasserstein1D_mass": test_jets.Wassertein1D("m", gen_jets),
            "Wasserstein1D_tau21": test_jets.Wassertein1D("tau21", gen_jets),
            "Wasserstein1D_tau23": test_jets.Wassertein1D("tau32", gen_jets),
            "Wasserstein1D_d2": test_jets.Wassertein1D("d2", gen_jets),
        }

    def get_results_plots(self, gen_jets, test_jets):
        continuous_plots = {
            "particle transverse momentum": self.plot_feature(
                "pt",
                gen_jets.constituents,
                test_jets.constituents,
                xlabel=r"$p_t$ [GeV]",
                log=True,
            ),
            "particle rapidity": self.plot_feature(
                "eta_rel",
                gen_jets.constituents,
                test_jets.constituents,
                xlabel=r"$\eta^{\rm rel}$",
                log=True,
            ),
            "particle azimuth": self.plot_feature(
                "phi_rel",
                gen_jets.constituents,
                test_jets.constituents,
                xlabel=r"$\phi^{\rm rel}$",
                log=True,
            ),
            "jet transverse momentum": self.plot_feature(
                "pt",
                gen_jets,
                test_jets,
                xlabel=r"$p_t$ [GeV]",
            ),
            "jet rapidity": self.plot_feature(
                "eta",
                gen_jets,
                test_jets,
                xlabel=r"$\eta$",
            ),
            "jet azimuth": self.plot_feature(
                "phi",
                gen_jets,
                test_jets,
                xlabel=r"$\phi$",
            ),
            "jet mass": self.plot_feature(
                "m",
                gen_jets,
                test_jets,
                xlabel=r"$m$ [GeV]",
            ),
            "energy correlation function": self.plot_feature(
                "d2",
                gen_jets,
                test_jets,
                xlabel=r"$D_2$",
            ),
            "21-subjetiness ratio": self.plot_feature(
                "tau21",
                gen_jets,
                test_jets,
                xlabel=r"$\tau_{21}$",
            ),
            "23-subjetiness ratio": self.plot_feature(
                "tau32",
                gen_jets,
                test_jets,
                xlabel=r"$\tau_{32}$",
            ),
        }
        discrete_plots = {
            "total charge": self.plot_feature(
                "Q_total",
                gen_jets,
                test_jets,
                xlabel=r"$Q_{\rm jet}^{\kappa=0}$",
                discrete=True,
            ),
            "jet charge": self.plot_feature(
                "Q_jet",
                gen_jets,
                test_jets,
                xlabel=r"$Q_{\rm jet}^{\kappa=1}$",
            ),
            "photon multiplicity": self.plot_multiplicity(
                0,
                gen_jets,
                test_jets,
                xlabel=r"$N_\gamma$",
            ),
            "neutral hadron multiplicity": self.plot_multiplicity(
                1,
                gen_jets,
                test_jets,
                xlabel=r"$N_{\rm h^0}$",
            ),
            "negative hadron multiplicity": self.plot_multiplicity(
                2,
                gen_jets,
                test_jets,
                xlabel=r"$N_{\rm h^-}$",
            ),
            "positive hadron multiplicity": self.plot_multiplicity(
                2,
                gen_jets,
                test_jets,
                xlabel=r"$N_{\rm h^+}$",
            ),
            "electron multiplicity": self.plot_multiplicity(
                4,
                gen_jets,
                test_jets,
                xlabel=r"$N_{e^-}$",
            ),
            "positron multiplicity": self.plot_multiplicity(
                5,
                gen_jets,
                test_jets,
                xlabel=r"$N_{e^+}$",
            ),
            "muon multiplicity": self.plot_multiplicity(
                6,
                gen_jets,
                test_jets,
                xlabel=r"$N_{\mu^-}$",
            ),
            "antimuon multiplicity": self.plot_multiplicity(
                7,
                gen_jets,
                test_jets,
                xlabel=r"$N_{\mu^+}$",
            ),
        }
        return (
            {**continuous_plots, **discrete_plots}
            if hasattr(test_jets.constituents, "discrete")
            else continuous_plots
        )

    def plot_feature(self, feat, gen, test, xlabel=None, log=False, discrete=False):
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        gen.histplot(
            feat,
            xlabel=xlabel,
            ax=ax,
            stat="density",
            color="crimson",
            log_scale=(False, log),
            fill=False,
            label="gen",
            discrete=discrete,
        )
        test.histplot(
            feat,
            xlabel=xlabel,
            ax=ax,
            stat="density",
            color="k",
            log_scale=(False, log),
            fill=False,
            label="target",
            discrete=discrete,
        )
        plt.legend(fontsize=10)
        plt.tight_layout()
        return fig

    def plot_multiplicity(self, state, gen, test, xlabel=None, log=False):
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        gen.histplot_multiplicities(
            state,
            xlabel=xlabel,
            ax=ax,
            stat="density",
            color="crimson",
            log_scale=(False, log),
            fill=False,
            label="gen",
        )
        test.histplot_multiplicities(
            state,
            xlabel=xlabel,
            ax=ax,
            stat="density",
            color="k",
            log_scale=(False, log),
            fill=False,
            label="target",
        )
        plt.legend(fontsize=7)
        plt.tight_layout()
        return fig
