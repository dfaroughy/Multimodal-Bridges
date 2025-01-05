import os
import tempfile
import mlflow
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
            pl_module.log(
                "train_loss", 
                loss, 
                on_epoch=True, 
                sync_dist=self.sync_dist)
            pl_module.log(
                "train_loss_continuous", 
                loss_0, 
                on_epoch=True, 
                sync_dist=self.sync_dist
            )
            pl_module.log(
                "train_loss_discrete", 
                loss_1, 
                on_epoch=True, 
                sync_dist=self.sync_dist
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
            pl_module.log(
                "val_loss", 
                loss, 
                on_epoch=True, 
                sync_dist=self.sync_dist)
            pl_module.log(
                "val_loss_continuous", 
                loss_0, 
                on_epoch=True, 
                sync_dist=self.sync_dist
            )
            pl_module.log(
                "val_loss_discrete", 
                loss_1, 
                on_epoch=True, 
                sync_dist=self.sync_dist
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
    """Callback to save generated data as MLflow artifacts after prediction."""

    def __init__(self, config: ExperimentConfigs):
        super().__init__()
        self.config = config
        self.batched_gen_states = []
        self.batched_source_states = []
        self.batched_target_states = []

    def on_predict_start(self, trainer, pl_module):
        run_id = self.config.experiment.logger.run_id
        if not mlflow.active_run():
            mlflow.start_run(run_id=run_id, nested=True)

    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if outputs is not None:
            self.batched_gen_states.append(outputs[0])
            self.batched_source_states.append(outputs[1])
            self.batched_target_states.append(outputs[2])

    def on_predict_end(self, trainer, pl_module):
        # ...generated sample
        gen_sample = HybridState.cat(self.batched_gen_states)
        gen_sample = ParticleClouds(dataset=gen_sample)
        gen_sample.stats = self.config.data.target.preprocess.stats.to_dict()
        gen_sample.postprocess()
        gen_jets = JetClassHighLevelFeatures(constituents=gen_sample)

        # ...test sample
        test_sample = HybridState.cat(self.batched_target_states)
        test_sample = ParticleClouds(dataset=test_sample)
        test_jets = JetClassHighLevelFeatures(constituents=test_sample)

        # ...results
        self.plot_histograms(gen_jets, test_jets)
        self.compute_performance_metrics(gen_jets, test_jets)
        mlflow.end_run()

    def compute_performance_metrics(self, gen_jets, test_jets):
        mlflow.log_metric(
            "Wasserstein1D_pt", f"{test_jets.Wassertein1D('pt', gen_jets):3f}"
        )
        mlflow.log_metric(
            "Wasserstein1D_mass", f"{test_jets.Wassertein1D('m', gen_jets):3f}"
        )
        mlflow.log_metric(
            "Wasserstein1D_tau21", f"{test_jets.Wassertein1D('tau21', gen_jets):3f}"
        )
        mlflow.log_metric(
            "Wasserstein1D_d2", f"{test_jets.Wassertein1D('d2', gen_jets):3f}"
        )

    def plot_histograms(self, gen_jets, test_jets):
        _, ax = plt.subplots(4, 4, figsize=(15, 12))

        arg_test = dict(
            stat="density",
            fill=False,
            log_scale=(False, False),
            color="k",
            lw=1.0,
            label="AOJ",
        )
        arg_gen = dict(
            stat="density",
            fill=False,
            log_scale=(False, False),
            color="crimson",
            lw=1.0,
            label="gen",
        )
        arg_test_log = dict(
            stat="density",
            fill=False,
            log_scale=(False, True),
            color="k",
            lw=1.0,
            label="AOJ",
        )
        arg_gen_log = dict(
            stat="density",
            fill=False,
            log_scale=(False, True),
            color="crimson",
            lw=1.0,
            label="gen",
        )

        binrange, binwidth = (0, 500), 5
        test_jets.constituents.histplot(
            "pt",
            binrange=binrange,
            binwidth=binwidth,
            xlabel=r"particle $p_t^{\rm rel}$",
            ax=ax[0, 0],
            **arg_test_log,
        )
        gen_jets.constituents.histplot(
            "pt",
            binrange=binrange,
            binwidth=binwidth,
            xlabel=r"particle $p_t^{\rm rel}$",
            ax=ax[0, 0],
            **arg_gen_log,
        )

        binrange, binwidth = (-2, 2), 0.04
        test_jets.constituents.histplot(
            "eta_rel",
            binrange=binrange,
            binwidth=binwidth,
            xlabel=r"particle $\Delta \eta$",
            ax=ax[0, 1],
            **arg_test_log,
        )
        gen_jets.constituents.histplot(
            "eta_rel",
            binrange=binrange,
            binwidth=binwidth,
            xlabel=r"particle $\Delta \eta$",
            ax=ax[0, 1],
            **arg_gen_log,
        )

        binrange, binwidth = (-2, 2), 0.04
        test_jets.constituents.histplot(
            "phi_rel",
            binrange=binrange,
            binwidth=binwidth,
            xlabel=r"particle $\Delta \phi$",
            ax=ax[0, 2],
            **arg_test_log,
        )
        gen_jets.constituents.histplot(
            "phi_rel",
            binrange=binrange,
            binwidth=binwidth,
            xlabel=r"particle $\Delta \phi$",
            ax=ax[0, 2],
            **arg_gen_log,
        )

        test_jets.histplot_multiplicities(
            xlabel="jet multiplicity", ax=ax[0, 3], **arg_test
        )
        gen_jets.histplot_multiplicities(
            xlabel="jet multiplicity", ax=ax[0, 3], **arg_gen
        )

        # ------------------------------

        binrange, binwidth = (450, 1100), 10
        test_jets.histplot(
            "pt",
            xlabel=r"jet $p_T$ [GeV]",
            ylabel="density",
            ax=ax[1, 0],
            binrange=binrange,
            binwidth=binwidth,
            **arg_test,
        )
        gen_jets.histplot(
            "pt",
            xlabel=r"jet $p_T$ [GeV]",
            ylabel="density",
            ax=ax[1, 0],
            binrange=binrange,
            binwidth=binwidth,
            **arg_gen,
        )

        binrange, binwidth = (0, 300), 5
        test_jets.histplot(
            "m",
            xlabel=r"jet mass [GeV]",
            ax=ax[1, 1],
            binrange=binrange,
            binwidth=binwidth,
            **arg_test,
        )
        gen_jets.histplot(
            "m",
            xlabel=r"jet mass [GeV]",
            ax=ax[1, 1],
            binrange=binrange,
            binwidth=binwidth,
            **arg_gen,
        )

        binrange, binwidth, ylim = (0, 1.25), 0.025, (0, 4.0)
        test_jets.histplot(
            "tau21",
            xlabel=r"$\tau_{21}$",
            ylabel="density",
            ax=ax[1, 2],
            binrange=binrange,
            binwidth=binwidth,
            **arg_test,
        )
        gen_jets.histplot(
            "tau21",
            xlabel=r"$\tau_{21}$",
            ylabel="density",
            ax=ax[1, 2],
            binrange=binrange,
            binwidth=binwidth,
            **arg_gen,
        )

        test_jets.histplot(
            "tau32",
            xlabel=r"$\tau_{32}$",
            ax=ax[1, 3],
            binrange=binrange,
            binwidth=binwidth,
            **arg_test,
        )
        gen_jets.histplot(
            "tau32",
            xlabel=r"$\tau_{32}$",
            ax=ax[1, 3],
            binrange=binrange,
            binwidth=binwidth,
            **arg_gen,
        )

        # ------------------------------

        binrange, binwidth, ylim = (0, 10.0), 0.1, (0, 0.6)
        test_jets.histplot(
            "d2",
            xlabel=r"$D_2$",
            ylabel="density",
            ax=ax[2, 0],
            binrange=binrange,
            binwidth=binwidth,
            **arg_test,
        )
        gen_jets.histplot(
            "d2",
            xlabel=r"$D_2$",
            ylabel="density",
            ax=ax[2, 0],
            ylim=ylim,
            binrange=binrange,
            binwidth=binwidth,
            **arg_gen,
        )

        xlim, ylim = (-20, 20), (0, 0.2)
        test_jets.histplot(
            "Q_total",
            xlabel=r"$Q_{\rm jet}^{\kappa=0}$",
            discrete=True,
            ax=ax[2, 1],
            **arg_test,
        )
        gen_jets.histplot(
            "Q_total",
            xlabel=r"$Q_{\rm jet}^{\kappa=0}$",
            ylim=ylim,
            xlim=xlim,
            ylabel="density",
            discrete=True,
            ax=ax[2, 1],
            **arg_gen,
        )

        binrange, binwidth, ylim = (-1, 1), 0.03, (0.0, 4.0)
        test_jets.histplot(
            "Q_jet",
            xlabel=r"$Q_{\rm jet}^{\kappa=1}$",
            binrange=binrange,
            binwidth=binwidth,
            ax=ax[2, 2],
            **arg_test,
        )
        gen_jets.histplot(
            "Q_jet",
            xlabel=r"$Q_{\rm jet}^{\kappa=1}$",
            ylim=ylim,
            binrange=binrange,
            binwidth=binwidth,
            ax=ax[2, 2],
            **arg_gen,
        )

        xlim, ylim = (0, 70), (0, 0.08)
        test_jets.histplot_multiplicities(
            state=[2, 3, 4, 5, 6, 7],
            xlabel="charged multiplicity",
            ax=ax[2, 3],
            **arg_test,
        )
        gen_jets.histplot_multiplicities(
            state=[2, 3, 4, 5, 6, 7],
            xlabel="charged multiplicity",
            xlim=xlim,
            ylim=ylim,
            ax=ax[2, 3],
            **arg_gen,
        )

        xlim, ylim = (0, 70), (0, 0.08)
        test_jets.histplot_multiplicities(
            state=[0, 1], xlabel="neutral multiplicity", ax=ax[3, 0], **arg_test
        )
        gen_jets.histplot_multiplicities(
            state=[0, 1],
            xlabel="neutral multiplicity",
            xlim=xlim,
            ylim=ylim,
            ax=ax[3, 0],
            **arg_gen,
        )

        xlim, ylim = (0, 70), (0, 0.08)
        test_jets.histplot_multiplicities(
            state=[1, 2, 3], xlabel="hadron multiplicity", ax=ax[3, 1], **arg_test
        )
        gen_jets.histplot_multiplicities(
            state=[1, 2, 3],
            xlabel="hadron multiplicity",
            xlim=xlim,
            ylim=ylim,
            ax=ax[3, 1],
            **arg_gen,
        )

        xlim, ylim = (-0.5, 7), (0, 1.25)
        test_jets.histplot_multiplicities(
            state=[4, 5, 6, 7], xlabel="lepton multiplicity", ax=ax[3, 2], **arg_test
        )
        gen_jets.histplot_multiplicities(
            state=[4, 5, 6, 7],
            xlabel="lepton multiplicity",
            xlim=xlim,
            ax=ax[3, 2],
            **arg_gen,
        )

        xlim, ylim = (0, 50), (0, 0.08)
        test_jets.histplot_multiplicities(
            state=0, xlabel="photon multiplicity", ax=ax[3, 3], **arg_test
        )
        gen_jets.histplot_multiplicities(
            state=0,
            xlabel="photon multiplicity",
            xlim=xlim,
            ylim=ylim,
            ax=ax[3, 3],
            **arg_gen,
        )

        plt.legend(fontsize=7)
        plt.tight_layout()
        plt.show()
        plt.savefig("results_plots.png")
        mlflow.log_artifact("results_plots.png", artifact_path="plots")
