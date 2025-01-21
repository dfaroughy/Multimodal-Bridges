import os
import json
import torch
import glob
import numpy as np
from pathlib import Path
from typing import Dict
import matplotlib.pyplot as plt

from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.utilities import rank_zero_only

from pipeline.configs import ExperimentConfigs
from pipeline.helpers import get_from_json
from pipeline.helpers import SimpleLogger as log
from data.particle_clouds.particles import ParticleClouds
from data.particle_clouds.jets import ParticleCloudsHighLevelFeatures
from model.multimodal_bridge_matching import MultiModeState


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
        super().__init__(**config.checkpoints.__dict__)


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
            log.info("Config file and metadata save to experiment path.")

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
        self.data_dir = Path(self.config.path) / "data"
        self.metric_dir = Path(self.config.path) / "metrics"
        self.plots_dir = Path(self.config.path) / "plots"
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.metric_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)

    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if outputs is not None:
            self.batched_gen_states.append(outputs[0])
            self.batched_source_states.append(outputs[1])
            self.batched_target_states.append(outputs[2])

    def on_predict_end(self, trainer, pl_module):
        rank = trainer.global_rank

        self._save_results_local(rank)
        trainer.strategy.barrier()  # wait for all ranks to finish

        if trainer.is_global_zero:
            self._gather_results_global(trainer)

        # self._clean_temp_files()

    def _save_results_local(self, rank):
        random = np.random.randint(0, 1000)

        src_raw = MultiModeState.cat(self.batched_source_states)
        gen_raw = MultiModeState.cat(self.batched_gen_states)
        test_raw = MultiModeState.cat(self.batched_target_states)

        src_raw.save_to(f"{self.data_dir}/temp_src_{rank}_{random}.h5")
        gen_raw.save_to(f"{self.data_dir}/temp_gen_{rank}_{random}.h5")
        test_raw.save_to(f"{self.data_dir}/temp_test_{rank}_{random}.h5")

    @rank_zero_only
    def _gather_results_global(self, trainer):
        stats = get_from_json("target_data_stats", self.config.path, "metadata.json")
        prep_continuous = self.config.data.target_preprocess_continuous
        prep_discrete = self.config.data.target_preprocess_discrete

        src_files = sorted(self.data_dir.glob("temp_src_*_*.h5"))
        gen_files = sorted(self.data_dir.glob("temp_gen_*_*.h5"))
        test_files = sorted(self.data_dir.glob("temp_test_*_*.h5"))

        src_states = [MultiModeState.load_from(str(f)) for f in src_files]
        gen_states = [MultiModeState.load_from(str(f)) for f in gen_files]
        test_states = [MultiModeState.load_from(str(f)) for f in test_files]

        src_merged = MultiModeState.cat(src_states)
        gen_merged = MultiModeState.cat(gen_states)
        test_merged = MultiModeState.cat(test_states)

        src_clouds = ParticleClouds(dataset=src_merged)
        src_clouds.save_to(f"{self.data_dir}/source_sample.h5")

        gen_clouds = ParticleClouds(dataset=gen_merged)
        gen_clouds.postprocess(prep_continuous, prep_discrete, **stats)
        gen_clouds.save_to(f"{self.data_dir}/generated_sample.h5")

        test_clouds = ParticleClouds(dataset=test_merged)
        test_clouds.save_to(f"{self.data_dir}/test_sample.h5")

        gen_jets = ParticleCloudsHighLevelFeatures(constituents=gen_clouds)
        test_jets = ParticleCloudsHighLevelFeatures(constituents=test_clouds)

        metrics = self.compute_performance_metrics(gen_jets, test_jets)

        with open(self.metric_dir / "performance_metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)

        if hasattr(self.config, "comet_logger"):
            figures = self.get_results_plots(gen_jets, test_jets)
            trainer.logger.experiment.log_metrics(metrics)

            for key in figures.keys():
                trainer.logger.experiment.log_figure(
                    figure=figures[key], figure_name=key
                )

    def _clean_temp_files(self):
        for f in self.data_dir.glob("temp_src_*.h5"):
            f.unlink()
        for f in self.data_dir.glob("temp_gen_*.h5"):
            f.unlink()
        for f in self.data_dir.glob("temp_test_*.h5"):
            f.unlink()

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
                binrange=(0,400),
                binwidth=5,
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
                binrange=(0,800),
                binwidth=8,
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
                binrange=(0,250),
                binwidth=2,
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
        if hasattr(gen_jets.constituents, "discrete"):
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
            return {**continuous_plots, **discrete_plots}
        return continuous_plots

    def plot_feature(self, feat, gen, test, xlabel=None, log=False, binwidth=None, binrange=None, discrete=False):
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        gen.histplot(
            feat,
            xlabel=xlabel,
            ax=ax,
            stat="density",
            color="crimson",
            log_scale=(False, log),
            binrange=binrange,
            binwidth=binwidth,
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
            binrange=binrange,
            binwidth=binwidth,
            fill=False,
            label="target",
            discrete=discrete,
        )
        plt.legend(fontsize=10)
        plt.tight_layout()
        plt.savefig(self.plots_dir / f"{xlabel}.png")
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
        plt.savefig(self.plots_dir / f"{xlabel}.png")
        return fig
