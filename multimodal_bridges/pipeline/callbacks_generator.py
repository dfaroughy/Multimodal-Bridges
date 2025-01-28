import os
import json
import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities import rank_zero_only

from pipeline.configs import ExperimentConfigs
from pipeline.helpers import SimpleLogger as log
from datamodules.particle_clouds.utils import JetFeatures
from tensorclass import TensorMultiModal


class JetGeneratorCallback(Callback):
    def __init__(self, config: ExperimentConfigs, transform=None):
        super().__init__()

        self.config = config
        self.transform = transform
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

        # load metadata for post-processing

        metadata = self._load_metadata(self.config.path)
        mean = torch.tensor(metadata["target"]["mean"])
        std = torch.tensor(metadata["target"]["std"])
        min_val = torch.tensor(metadata["target"]["min"])
        max_val = torch.tensor(metadata["target"]["max"])

        if self.transform == 'standardize':
            self.transform = lambda x: x * std + mean

        elif self.transform == 'normalize':
            self.transform = lambda x: x * (max_val - min_val) + min_val
        
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
        src_merged.save_to(f"{self.data_dir}/source_sample.h5")

        gen_states = [TensorMultiModal.load_from(str(f)) for f in gen_files]
        gen_merged = TensorMultiModal.cat(gen_states)

        if self.transform:
            gen_merged.continuous = self.transform(gen_merged.continuous)
            gen_merged.apply_mask()

        gen_merged.save_to(f"{self.data_dir}/generated_sample.h5")
        gen_jets = JetFeatures(gen_merged)

        test_states = [TensorMultiModal.load_from(str(f)) for f in test_files]
        test_merged = TensorMultiModal.cat(test_states)
        test_merged.save_to(f"{self.data_dir}/test_sample.h5")
        test_jets = JetFeatures(test_merged)

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
                binrange=(0, 400),
                binwidth=5,
                log=True,
                suffix_file="_part",
            ),
            "particle rapidity": self.plot_feature(
                "eta_rel",
                gen_jets.constituents,
                test_jets.constituents,
                xlabel=r"$\eta^{\rm rel}$",
                log=True,
                suffix_file="_part",
            ),
            "particle azimuth": self.plot_feature(
                "phi_rel",
                gen_jets.constituents,
                test_jets.constituents,
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
                discrete=True,
                xlabel=r"$N$",
            ),
        }
        if gen_jets.constituents.has_discrete:
            discrete_plots = {
                "total charge": self.plot_feature(
                    "charge",
                    gen_jets,
                    test_jets,
                    xlabel=r"$Q_{\rm jet}^{\kappa=0}$",
                    discrete=True,
                ),
                "jet charge": self.plot_feature(
                    "jet_charge",
                    gen_jets,
                    test_jets,
                    xlabel=r"$Q_{\rm jet}^{\kappa=1}$",
                ),
                "photon multiplicity": self.plot_feature(
                    "numPhotons",
                    gen_jets,
                    test_jets,
                    xlabel=r"$N_\gamma$",
                    discrete=True,
                ),
                "neutral hadron multiplicity": self.plot_feature(
                    "numNeutralHadrons",
                    gen_jets,
                    test_jets,
                    xlabel=r"$N_{\rm h^0}$",
                    discrete=True,
                ),
                "negative hadron multiplicity": self.plot_feature(
                    "numNegativeHadrons",
                    gen_jets,
                    test_jets,
                    xlabel=r"$N_{\rm h^-}$",
                    discrete=True,
                ),
                "positive hadron multiplicity": self.plot_feature(
                    "numPositiveHadrons",
                    gen_jets,
                    test_jets,
                    xlabel=r"$N_{\rm h^+}$",
                    discrete=True,
                ),
                "electron multiplicity": self.plot_feature(
                    "numElectrons",
                    gen_jets,
                    test_jets,
                    xlabel=r"$N_{e^-}$",
                    discrete=True,
                ),
                "positron multiplicity": self.plot_feature(
                    "numPositrons",
                    gen_jets,
                    test_jets,
                    xlabel=r"$N_{e^+}$",
                    discrete=True,
                ),
                "muon multiplicity": self.plot_feature(
                    "numMuons",
                    gen_jets,
                    test_jets,
                    xlabel=r"$N_{\mu^-}$",
                    discrete=True,
                ),
                "antimuon multiplicity": self.plot_feature(
                    "numAntiMuons",
                    gen_jets,
                    test_jets,
                    xlabel=r"$N_{\mu^+}$",
                    discrete=True,
                ),
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
                "all charged": self.plot_feature(
                    "numCharged",
                    gen_jets,
                    test_jets,
                    xlabel=r"$N_{charged}$",
                    discrete=True,
                ),
                "all neutral": self.plot_feature(
                    "numNeutrals",
                    gen_jets,
                    test_jets,
                    xlabel=r"$N_{neutrals}$",
                    discrete=True,
                ),
                "flavor counts": self.plot_flavor_counts_per_jet(gen_jets, test_jets),
            }
            return {**continuous_plots, **discrete_plots}
        return continuous_plots

    def plot_feature(
        self,
        feat,
        gen,
        test,
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
