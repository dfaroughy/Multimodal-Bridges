import os
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities import rank_zero_only

from pipeline.configs import ExperimentConfigs
from pipeline.helpers import SimpleLogger as log
from datamodules.utils import JetFeatures
from tensorclass import TensorMultiModal


class JetGeneratorCallback(Callback):
    def __init__(self, config: ExperimentConfigs):
        super().__init__()

        self.config = config
        self.transform = config.data.transform
        self.batched_gen_states = []
        self.batched_source_states = []
        self.batched_target_states = []

    def on_predict_start(self, trainer, pl_module):
        self.data_dir = Path(self.config.path) / "data"
        self.metric_dir = os.path.join(self.config.path, "metrics")
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
            self._clean_temp_files()

    ############
    ### helpers:
    ############

    def _save_results_local(self, rank):
        random = np.random.randint(0, 1000)

        src_raw = TensorMultiModal.cat(self.batched_source_states)
        gen_raw = TensorMultiModal.cat(self.batched_gen_states)
        test_raw = TensorMultiModal.cat(self.batched_target_states)

        src_raw.save_to(f"{self.data_dir}/temp_src_{rank}_{random}.h5")
        gen_raw.save_to(f"{self.data_dir}/temp_gen_{rank}_{random}.h5")
        test_raw.save_to(f"{self.data_dir}/temp_test_{rank}_{random}.h5")

    @rank_zero_only
    def _gather_results_global(self, trainer):
        src_files = self.data_dir.glob("temp_src_*_*.h5")
        gen_files = self.data_dir.glob("temp_gen_*_*.h5")
        test_files = self.data_dir.glob("temp_test_*_*.h5")

        src_states = [TensorMultiModal.load_from(str(f)) for f in src_files]
        src_merged = TensorMultiModal.cat(src_states)
        src_merged = self._postprocess(src_merged, transform=None)
        src_merged.save_to(f"{self.data_dir}/source_sample.h5")

        gen_states = [TensorMultiModal.load_from(str(f)) for f in gen_files]
        gen_merged = TensorMultiModal.cat(gen_states)
        gen_merged = self._postprocess(gen_merged, transform=self.transform)
        gen_merged.save_to(f"{self.data_dir}/generated_sample.h5")
        gen_jets = JetFeatures(gen_merged)

        test_states = [TensorMultiModal.load_from(str(f)) for f in test_files]
        test_merged = TensorMultiModal.cat(test_states)
        test_merged = self._postprocess(test_merged, transform=None)
        test_merged.save_to(f"{self.data_dir}/test_sample.h5")
        test_jets = JetFeatures(test_merged)

        metrics = self.compute_performance_metrics(gen_jets, test_jets)

        # with open(self.metric_dir / "performance_metrics.json", "w") as f:
        #     json.dump(metrics, f, indent=4)

        if hasattr(self.config, "comet_logger"):
            figures = self.get_results_plots(gen_jets, test_jets)
            for key in figures.keys():
                trainer.logger.experiment.log_figure(
                    figure=figures[key], figure_name=key
                )
            df = pd.DataFrame(metrics)
            trainer.logger.experiment.log_table(
                f"{self.metric_dir}/performance_metrics.csv", df
            )

    def _postprocess(self, x: TensorMultiModal, transform=None):
        metadata = self._load_metadata(self.config.path)

        if transform == "standardize":
            mean = torch.tensor(metadata["target"]["mean"])
            std = torch.tensor(metadata["target"]["std"])
            x.continuous = x.continuous * std + mean

        elif transform == "normalize":
            min_val = torch.tensor(metadata["target"]["min"])
            max_val = torch.tensor(metadata["target"]["max"])
            x.continuous = x.continuous * (max_val - min_val) + min_val

        if transform == "log_pt":
            x.continuous[:, :, 0] = torch.exp(x.continuous[:, :, 0]) - 1e-6

        if self.config.data.discrete_features == "onehot":
            x.discrete = x.continuous[:, :, -self.config.data.vocab_size :]
            x.continuous = x.continuous[:, :, : -self.config.data.vocab_size]
            x.discrete = torch.argmax(x.discrete, dim=-1).unsqueeze(-1)

        x.apply_mask()

        return x

    def _clean_temp_files(self):
        for f in self.data_dir.glob("temp_src_*_*.h5"):
            f.unlink()
        for f in self.data_dir.glob("temp_gen_*_*.h5"):
            f.unlink()
        for f in self.data_dir.glob("temp_test_*_*.h5"):
            f.unlink()

    def _load_metadata(self, path):
        metadata_file = os.path.join(path, "metadata.json")
        log.info(f"Loading metadata from {metadata_file}.")
        with open(metadata_file, "r") as f:
            metadata = json.load(f)
        return metadata

    def compute_performance_metrics(self, gen_jets, test_jets):
        return {
            'obs':['pt', 'm', 'tau21', 'tau32', 'd2'],
            'W1':[test_jets.Wassertein1D("pt", gen_jets),
            test_jets.Wassertein1D("m", gen_jets),
            test_jets.Wassertein1D("tau21", gen_jets),
            test_jets.Wassertein1D("tau32", gen_jets),
            test_jets.Wassertein1D("d2", gen_jets)]
        }

    def get_results_plots(self, gen_jets, test_jets):
        continuous_plots = {
            "particle transverse momentum": self.plot_feature(
                "pt",
                gen_jets.constituents,
                test_jets.constituents,
                apply_map='mask_bool',
                xlabel=r"$p_t$ [GeV]",
                binrange=(0, 400),
                binwidth=5,
                log=True,
                suffix_file="_part",
            ),
            "particle rapidity": self.plot_feature(
                "eta_rel",
                gen_jets.constituents,
                test_jets.constituents,
                apply_map='mask_bool',
                xlabel=r"$\eta^{\rm rel}$",
                log=True,
                suffix_file="_part",
            ),
            "particle azimuth": self.plot_feature(
                "phi_rel",
                gen_jets.constituents,
                test_jets.constituents,
                apply_map='mask_bool',
                xlabel=r"$\phi^{\rm rel}$",
                log=True,
                suffix_file="_part",
            ),
            "jet transverse momentum": self.plot_feature(
                "pt",
                gen_jets,
                test_jets,
                binrange=(0, 800),
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
                binrange=(0, 250),
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
            "particle multipicity": self.plot_feature(
                "multiplicity",
                gen_jets.constituents,
                test_jets.constituents,
                apply_map=lambda x: x.squeeze(-1),
                discrete=True,
                xlabel=r"$N$",
            ),
        }
        if gen_jets.constituents.has_discrete:
            discrete_plots = {
                "all hadrons": self.plot_feature(
                    "numHadrons",
                    gen_jets,
                    test_jets,
                    xlabel=r"$N_{\rm h}$",
                    discrete=True,
                ),
                "all leptons": self.plot_feature(
                    "numLeptons",
                    gen_jets,
                    test_jets,
                    xlabel=r"$N_{\ell}$",
                    discrete=True,
                ),
                "electric charges": self.plot_charges(gen_jets, test_jets),
                "flavor counts": self.plot_flavor_counts_per_jet(gen_jets, test_jets),
                "photon kin": self.plot_flavored_kinematics(
                    "Photon", gen_jets, test_jets
                ),
                "neutral hadron kin": self.plot_flavored_kinematics(
                    "NeutralHadron", gen_jets, test_jets
                ),
                "negative hadron kin": self.plot_flavored_kinematics(
                    "NegativeHadron", gen_jets, test_jets
                ),
                "positive hadron kin": self.plot_flavored_kinematics(
                    "PositiveHadron", gen_jets, test_jets
                ),
                "electron kin": self.plot_flavored_kinematics(
                    "Electron", gen_jets, test_jets
                ),
                "positron kin": self.plot_flavored_kinematics(
                    "Positron", gen_jets, test_jets
                ),
                "muon kin": self.plot_flavored_kinematics(
                    "Muon", gen_jets, test_jets),
                "antimuon kin": self.plot_flavored_kinematics(
                    "AntiMuon", gen_jets, test_jets
                ),
            }
            return {**continuous_plots, **discrete_plots}
        return continuous_plots

    def plot_feature(
        self,
        feat,
        gen,
        test,
        apply_map=None,
        xlabel=None,
        log=False,
        binwidth=None,
        binrange=None,
        discrete=False,
        suffix_file="",
    ):
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        gen.histplot(
            feat,
            apply_map=apply_map,
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
            apply_map=apply_map,
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
        plt.savefig(self.plots_dir / f"{feat}{suffix_file}.png")
        return fig

    def plot_flavor_counts_per_jet(
        self,
        gen,
        test,
    ):
        fig, ax = plt.subplots(1, 1, figsize=(4, 3))
        gen.plot_flavor_count_per_jet(ax=ax, color="crimson")
        test.plot_flavor_count_per_jet(ax=ax, color="k")
        plt.legend(fontsize=10)
        plt.tight_layout()
        plt.savefig(self.plots_dir / "flavor_counts.png")
        return fig

    def plot_flavored_kinematics(self, flavor, gen, test):
        flavor_labels = {
            "Electron": "{\,e^-}",
            "Positron": "{\,e^+}",
            "Muon": "{\,\mu^-}",
            "AntiMuon": "{\,\mu^+}",
            "Photon": "\gamma",
            "NeutralHadron": "{\,h^0}",
            "NegativeHadron": "{\,h^-}",
            "PositiveHadron": "{\,h^+}",
        }

        fig, ax = plt.subplots(1, 4, figsize=(8, 2))
        test.constituents.histplot(
            f"pt_{flavor}",
            apply_map=None,
            ax=ax[0],
            fill=False,
            bins=100,
            lw=1,
            color="k",
            log_scale=(True, True),
            stat="density",
            xlim=(1e-2, 800),
            label="AOJ",
        )
        gen.constituents.histplot(
            f"pt_{flavor}",
            apply_map=None,
            ax=ax[0],
            fill=False,
            bins=100,
            lw=1,
            color="crimson",
            log_scale=(True, True),
            stat="density",
            xlim=(1e-2, 800),
            label="generated",
        )
        test.constituents.histplot(
            f"eta_{flavor}",
            apply_map=None,
            ax=ax[1],
            fill=False,
            bins=100,
            color="k",
            lw=1,
            log_scale=(False, True),
            stat="density",
            xlim=(-1.2, 1.2),
        )
        gen.constituents.histplot(
            f"eta_{flavor}",
            apply_map=None,
            ax=ax[1],
            fill=False,
            bins=100,
            color="crimson",
            lw=1,
            log_scale=(False, True),
            stat="density",
            xlim=(-1.2, 1.2),
        )
        test.constituents.histplot(
            f"phi_{flavor}",
            apply_map=None,
            ax=ax[2],
            fill=False,
            bins=100,
            color="k",
            lw=1,
            log_scale=(False, True),
            stat="density",
            xlim=(-1.2, 1.2),
        )
        gen.constituents.histplot(
            f"phi_{flavor}",
            apply_map=None,
            ax=ax[2],
            fill=False,
            bins=100,
            color="crimson",
            lw=1,
            log_scale=(False, True),
            stat="density",
            xlim=(-1.2, 1.2),
        )
        test.histplot(
            f"num{flavor}s",
            ax=ax[3],
            fill=False,
            discrete=True,
            lw=1,
            color="k",
            log_scale=(False, False),
            stat="density",
        )
        gen.histplot(
            f"num{flavor}s",
            ax=ax[3],
            fill=False,
            discrete=True,
            lw=1,
            color="crimson",
            log_scale=(False, False),
            stat="density",
        )
        ax[0].set_xlabel(rf"$p_T^{flavor_labels[flavor]}$")
        ax[1].set_xlabel(rf"$\eta^{flavor_labels[flavor]}$")
        ax[2].set_xlabel(rf"$\phi^{flavor_labels[flavor]}$")
        ax[3].set_xlabel(rf"$N_{flavor_labels[flavor]}$")
        ax[0].set_ylabel("density")

        plt.tight_layout()
        plt.legend(fontsize=10)
        plt.savefig(self.plots_dir / f"kinematics_{flavor}.png")
        return fig

    def plot_charges(self, gen, test):
        fig, ax = plt.subplots(1, 4, figsize=(8, 2))
        test.histplot(
            "numNeutrals",
            ax=ax[0],
            fill=False,
            discrete=True,
            lw=1,
            color="k",
            stat="density",
            xlabel=r"$N_{Q=0}$",
        )
        gen.histplot(
            "numNeutrals",
            ax=ax[0],
            fill=False,
            discrete=True,
            lw=1,
            color="crimson",
            stat="density",
            xlabel=r"$N_{Q=0}$",
        )
        test.histplot(
            "numCharged",
            ax=ax[1],
            fill=False,
            discrete=True,
            color="k",
            lw=1,
            stat="density",
            xlabel=r"$N_{Q=\pm1}$",
        )
        gen.histplot(
            "numCharged",
            ax=ax[1],
            fill=False,
            discrete=True,
            color="crimson",
            lw=1,
            stat="density",
            xlabel=r"$N_{Q=\pm1}$",
        )
        test.histplot(
            "charge",
            ax=ax[2],
            fill=False,
            discrete=True,
            color="k",
            lw=1,
            stat="density",
            xlabel=r"$Q_{\rm jet}^{\kappa=0}$",
        )
        gen.histplot(
            "charge",
            ax=ax[2],
            fill=False,
            discrete=True,
            color="crimson",
            lw=1,
            stat="density",
            xlabel=r"$Q_{\rm jet}^{\kappa=0}$",
        )
        test.histplot(
            "jet_charge",
            ax=ax[3],
            fill=False,
            color="k",
            lw=1,
            stat="density",
            xlabel=r"$Q_{\rm jet}^{\kappa=1}$",
        )
        gen.histplot(
            "jet_charge",
            ax=ax[3],
            fill=False,
            color="crimson",
            lw=1,
            stat="density",
            xlabel=r"$Q_{\rm jet}^{\kappa=1}$",
        )
        ax[0].set_xticks([0, 20, 40, 60])
        ax[1].set_xticks([0, 20, 40, 60, 80])
        ax[2].set_xticks([-20, -10, 0, 10, 20])
        ax[3].set_xticks([-0.75, 0, 0.75])
        ax[0].set_ylabel("density")
        plt.tight_layout()
        plt.savefig(self.plots_dir / "charges.png")
        return fig
