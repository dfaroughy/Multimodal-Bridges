from comet_ml import ExistingExperiment

import os
import json
from pathlib import Path
import matplotlib.pyplot as plt
from lightning.pytorch.callbacks import Callback

from multimodal_bridge_matching import HybridState
from utils.configs import ExperimentConfigs
from data.particle_clouds.particles import ParticleClouds
from data.particle_clouds.jets import JetClassHighLevelFeatures


class MetricLoggerCallback(Callback):
    """
    Custom MLflow Callback to handle logging and saving checkpoints as MLflow artifacts.
    """

    def __init__(self, sync_dist=False):
        super().__init__()
        self.sync_dist = sync_dist

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """
        Logs metrics at the end of each training batch.
        """
        if outputs is not None:
            loss = outputs["loss"]
            loss_0 = outputs["loss_individual"][0]
            loss_1 = outputs["loss_individual"][1]
            weights_0 = outputs["weights"][0]
            weights_1 = outputs["weights"][1]
            pl_module.log("train_loss", loss, on_epoch=True, sync_dist=self.sync_dist)
            pl_module.log(
                "train_loss_continuous", loss_0, on_epoch=True, sync_dist=self.sync_dist
            )
            pl_module.log(
                "train_loss_discrete", loss_1, on_epoch=True, sync_dist=self.sync_dist
            )
            pl_module.log(
                "train_weights_continuous",
                weights_0,
                on_epoch=True,
                sync_dist=self.sync_dist,
            )
            pl_module.log(
                "train_weights_discrete",
                weights_1,
                on_epoch=True,
                sync_dist=self.sync_dist,
            )

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """
        Logs metrics at the end of each validation batch.
        """
        if outputs is not None:
            loss = outputs["loss"]
            loss_0 = outputs["loss_individual"][0]
            loss_1 = outputs["loss_individual"][1]
            weights_0 = outputs["weights"][0]
            weights_1 = outputs["weights"][1]
            pl_module.log("val_loss", loss, on_epoch=True, sync_dist=self.sync_dist)
            pl_module.log(
                "val_loss_continuous", loss_0, on_epoch=True, sync_dist=self.sync_dist
            )
            pl_module.log(
                "val_loss_discrete", loss_1, on_epoch=True, sync_dist=self.sync_dist
            )
            pl_module.log(
                "val_weights_continuous",
                weights_0,
                on_epoch=True,
                sync_dist=self.sync_dist,
            )
            pl_module.log(
                "val_weights_discrete",
                weights_1,
                on_epoch=True,
                sync_dist=self.sync_dist,
            )


class JetsGenerativeCallback(Callback):
    def __init__(self, config: ExperimentConfigs):
        super().__init__()
        self.config = config
        self.batched_gen_states = []
        self.batched_source_states = []
        self.batched_target_states = []

    def on_predict_start(self, trainer, pl_module):
        self.data_dir = os.path.join(self.config.experiment.dir_path, "data")
        self.metric_dir = os.path.join(self.config.experiment.dir_path, "metrics")
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.metric_dir, exist_ok=True)

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
        figs = self.get_results_plots(gen_jets, test_jets)
        self.log_results(metrics, figs)

    def log_results(self, metrics=None, figures=None):
        self.gen_sample.save_to(path=os.path.join(self.data_dir, "generated_sample.h5"))
        self.test_sample.save_to(path=os.path.join(self.data_dir, "test_sample.h5"))
        with open(os.path.join(self.metric_dir, "performance_metrics.h5"), "w") as f:
            json.dump(metrics, f, indent=4)
        if hasattr(self.config.experiment, "comet_logger"):
            api_key = self.config.experiment.comet_logger.api_key
            exp_key = self.config.experiment.comet_logger.experiment_key
            experiment = ExistingExperiment(
                api_key=api_key, previous_experiment=exp_key
            )
            if metrics:
                experiment.log_metrics(metrics)
            if figures:
                for key in figures.keys():
                    experiment.log_figure(figure=figures[key], figure_name=key)
            experiment.end()
        else:
            pass

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
