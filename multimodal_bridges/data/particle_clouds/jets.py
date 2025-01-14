import torch
import numpy as np
import awkward as ak
import fastjet
import scipy
import json
import os
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split, Subset
import lightning.pytorch as L

plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["figure.autolayout"] = False

from utils.helpers import SimpleLogger as log
from data.particle_clouds.particles import ParticleClouds
from data.datasets import HybridDataset, hybrid_collate_fn


class JetDataModule(L.LightningDataModule):
    """DataModule for handling source-target-context coupling for particle cloud data.
    """
    def __init__(
        self,
        config,
        preprocess=False,
        metadata_path=None,
    ):
        super().__init__()
        self.config = config
        self.batch_size = config.datamodule.batch_size
        self.num_workers = config.datamodule.num_workers
        self.pin_memory = config.datamodule.pin_memory
        self.split_ratios = config.datamodule.split_ratios
        self.metadata_path = metadata_path
        self.metadata = {}

        # Metadata definitions
        self.metadata["continuous_features"] = {'pt': 0, 'eta_rel': 1, 'phi_rel': 2}
        self.metadata["discrete_features"] = {
            "isPhoton": [1, 0, 0, 0, 0],
            "isNeutralHadron": [0, 1, 0, 0, 0],
            "isChargedHadron": [0, 0, 1, 0, 0],
            "isElectron": [0, 0, 0, 1, 0],
            "isMuon": [0, 0, 0, 0, 1],
        }
        self.metadata["electric_charges"] = [-1, 0, 1]
        self.metadata["particles_info"] = {
            0: {"name": "photon", "color": "gold", "marker": "o", "tex": r"\gamma"},
            1: {"name": "neutral hadron", "color": "darkred", "marker": "o", "tex": r"$\rm h^0$"},
            2: {"name": "negative hadron", "color": "darkred", "marker": "v", "tex": r"$\rm h^-$"},
            3: {"name": "positive hadron", "color": "darkred", "marker": "^", "tex": r"$\rm h^+$"},
            4: {"name": "electron", "color": "blue", "marker": "v", "tex": r"e^-"},
            5: {"name": "positron", "color": "blue", "marker": "^", "tex": r"e^+"},
            6: {"name": "muon", "color": "green", "marker": "v", "tex": r"\mu^-"},
            7: {"name": "antimuon", "color": "green", "marker": "^", "tex": r"\mu^+"},
        }

        self._prepare_datasets()

        if preprocess:
            self._preprocess_datasets()
            self._store_metadata()

    def _prepare_datasets(self):
        """Prepare source and target datasets based on configuration."""
        self.target = ParticleClouds(
            dataset=self.config.data.target_name,
            path=self.config.data.target_path,
            num_jets=self.config.data.num_jets,
            min_num_particles=self.config.data.min_num_particles,
            max_num_particles=self.config.data.max_num_particles,
        )
        self.source = ParticleClouds(
            dataset=self.config.data.source_name,
            path=self.config.data.source_path,
            num_jets=self.config.data.num_jets,
            min_num_particles=self.config.data.min_num_particles,
            max_num_particles=self.config.data.max_num_particles,
            multiplicity_dist=self.target.multiplicity
            if "Noise" in self.config.data.source_name
            else None,
        )

        self.metadata["source_data_stats"] = self.source.get_data_stats()
        self.metadata["target_data_stats"] = self.target.get_data_stats()

    def _preprocess_datasets(self):
        """Preprocess source and target datasets."""
        log.info("Preprocessing datasets...")
        self.source.preprocess(
            continuous=self.config.data.source_preprocess_continuous,
            discrete=self.config.data.source_preprocess_discrete,
            **self.metadata["source_data_stats"],
        )
        self.target.preprocess(
            continuous=self.config.data.target_preprocess_continuous,
            discrete=self.config.data.target_preprocess_discrete,
            **self.metadata["target_data_stats"],
        )

    def _store_metadata(self):
        if self.metadata_path:
            log.info("Storing metadata at: {path}")
            path = os.path.join(self.metadata_path, "metadata.json")
            if not os.path.exists(path):
                with open(path, "w") as f:
                    json.dump(self.metadata, f, indent=4)

    def _get_dataloader(self, dataset, shuffle):
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=shuffle,
            collate_fn=hybrid_collate_fn,
        )

    #####################
    # Lightning methods #
    #####################

    def setup(self, stage=None):
        """Setup datasets for train, validation, and test splits."""

        dataset = HybridDataset(self)
        assert np.abs(1.0 - sum(self.split_ratios)) < 1e-3, (
            "Split fractions do not sum to 1!"
        )
        total_size = len(dataset)
        train_size = int(total_size * self.split_ratios[0])
        valid_size = int(total_size * self.split_ratios[1])
        test_size = int(total_size * self.split_ratios[2])

        # ...define splitting indices

        idx = torch.arange(total_size)
        idx_train = idx[:train_size].tolist()
        idx_valid = idx[train_size : train_size + valid_size].tolist()
        idx_test = idx[train_size + valid_size :].tolist()

        # ...Create Subset for each split

        self.train_dataset = Subset(dataset, idx_train) if train_size > 0 else None
        self.val_dataset = Subset(dataset, idx_valid) if valid_size > 0 else None
        self.test_dataset = Subset(dataset, idx_test) if test_size > 0 else None

    def train_dataloader(self):
        return self._get_dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self):
        return self._get_dataloader(self.val_dataset, shuffle=False)

    def test_dataloader(self):
        return self._get_dataloader(self.test_dataset, shuffle=False)

    # def predict_dataloader(self):
    #     """Dataloader for inference/prediction."""
    #     dataset = HybridDataset(self)
    #     return self._get_dataloader(dataset, shuffle=False)


class JetClassHighLevelFeatures:
    def __init__(self, constituents: ParticleClouds):
        self.constituents = constituents

        # ...compute jet kinematics:
        self.constituents.compute_4mom()
        self.px = self.constituents.px.sum(axis=-1)
        self.py = self.constituents.py.sum(axis=-1)
        self.pz = self.constituents.pz.sum(axis=-1)
        self.e = self.constituents.e.sum(axis=-1)
        self.pt = torch.clamp_min(self.px**2 + self.py**2, 0).sqrt()
        self.m = torch.clamp_min(
            self.e**2 - self.px**2 - self.py**2 - self.pz**2, 0
        ).sqrt()
        self.eta = 0.5 * torch.log((self.pt + self.pz) / (self.pt - self.pz))
        self.phi = torch.atan2(self.py, self.px)

        # discrete jet features
        self.multiplicity = torch.sum(self.constituents.mask, dim=1)
        if hasattr(self.constituents, "discrete"):
            self.Q_total = self.jet_charge(kappa=0.0)
            self.Q_jet = self.jet_charge(kappa=1.0)

        # ...subsstructure
        self.R = 0.8
        self.beta = 1.0
        self.use_wta_scheme = False
        self.substructure()

    def histplot(
        self,
        features="pt",
        xlim=None,
        ylim=None,
        xlabel=None,
        ylabel=None,
        figsize=(3, 3),
        fontsize=12,
        ax=None,
        **kwargs,
    ):
        x = getattr(self, features)
        if isinstance(x, torch.Tensor):
            x.cpu().numpy()
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)
        sns.histplot(x=x, element="step", ax=ax, **kwargs)
        ax.set_xlabel(features if xlabel is None else xlabel, fontsize=fontsize)
        ax.set_ylabel(ylabel, fontsize=fontsize)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    def jet_charge(self, kappa):
        """jet charge defined as Q_j^kappa = Sum_i Q_i * (pT_i / pT_jet)^kappa"""
        Qjet = self.constituents.charge.squeeze(-1) * (self.constituents.pt) ** kappa
        return Qjet.sum(axis=1) / (self.pt**kappa)

    def histplot_multiplicities(
        self,
        state=None,
        xlim=None,
        ylim=None,
        xlabel=None,
        ylabel=None,
        figsize=(3, 3),
        fontsize=12,
        ax=None,
        **kwargs,
    ):
        if state is not None:
            if isinstance(state, int):
                state = [state]
            multiplicity = torch.zeros(self.constituents.discrete.shape[0], 1)
            for s in state:
                x = (
                    torch.argmax(self.constituents.discrete, dim=-1).unsqueeze(-1) == s
                ) * self.constituents.mask
                multiplicity += x.sum(dim=1)
        else:
            multiplicity = self.multiplicity

        if ax is None:
            _, ax = plt.subplots(figsize=figsize)
        sns.histplot(
            x=multiplicity.squeeze(-1), element="step", ax=ax, discrete=True, **kwargs
        )
        ax.set_xlabel(xlabel, fontsize=fontsize)
        ax.set_ylabel(ylabel, fontsize=fontsize)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    def flavor_fractions(self, figsize=(3, 3), fontsize=12, ax=None, **kwargs):
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)
        sns.histplot(
            self.constituents.discrete[self.constituents.mask].squeeze(),
            binrange=(-0.1, 7.1),
            element="step",
            ax=ax,
            discrete=True,
            **kwargs,
        )
        ax.legend(loc="upper right", fontsize=8)
        ax.set_xlabel("Particle flavor", fontsize=fontsize)
        ax.set_xticks(np.arange(8))
        ax.set_xticklabels(
            [
                r"$\gamma$",
                r"$h^0$",
                r"$h^-$",
                r"$h^+$",
                r"$e^-$",
                r"$e^+$",
                r"$\mu^-$",
                r"$\mu^+$",
            ]
        )

    def substructure(self):
        constituents_ak = ak.zip(
            {
                "pt": np.array(self.constituents.pt),
                "eta": np.array(self.constituents.eta_rel),
                "phi": np.array(self.constituents.phi_rel),
                "mass": np.zeros_like(np.array(self.constituents.pt)),
            },
            with_name="Momentum4D",
        )

        constituents_ak = ak.mask(constituents_ak, constituents_ak.pt > 0)
        constituents_ak = ak.drop_none(constituents_ak)
        self.constituents_ak = constituents_ak[ak.num(constituents_ak) >= 3]
        if self.use_wta_scheme:
            jetdef = fastjet.JetDefinition(
                fastjet.kt_algorithm, self.R, fastjet.WTA_pt_scheme
            )
        else:
            jetdef = fastjet.JetDefinition(fastjet.kt_algorithm, self.R)
        print("Clustering jets with fastjet")
        print("Jet definition:", jetdef)
        self.cluster = fastjet.ClusterSequence(self.constituents_ak, jetdef)
        self.inclusive_jets = self.cluster.inclusive_jets()
        self.exclusive_jets_1 = self.cluster.exclusive_jets(n_jets=1)
        self.exclusive_jets_2 = self.cluster.exclusive_jets(n_jets=2)
        self.exclusive_jets_3 = self.cluster.exclusive_jets(n_jets=3)
        print("Calculating N-subjettiness")
        self._calc_d0()
        self._calc_tau1()
        self._calc_tau2()
        self._calc_tau3()
        self.tau21 = np.ma.divide(self.tau2, self.tau1)
        self.tau32 = np.ma.divide(self.tau3, self.tau2)
        print("Calculating D2")
        # D2 as defined in https://arxiv.org/pdf/1409.6298.pdf
        self.d2 = self.cluster.exclusive_jets_energy_correlator(njets=1, func="d2")

    def _calc_deltaR(self, particles, jet):
        jet = ak.unflatten(ak.flatten(jet), counts=1)
        return particles.deltaR(jet)

    def _calc_d0(self):
        """Calculate the d0 values."""
        self.d0 = ak.sum(self.constituents_ak.pt * self.R**self.beta, axis=1)

    def _calc_tau1(self):
        """Calculate the tau1 values."""
        self.delta_r_1i = self._calc_deltaR(
            self.constituents_ak, self.exclusive_jets_1[:, :1]
        )
        self.pt_i = self.constituents_ak.pt
        self.tau1 = ak.sum(self.pt_i * self.delta_r_1i**self.beta, axis=1) / self.d0

    def _calc_tau2(self):
        """Calculate the tau2 values."""
        delta_r_1i = self._calc_deltaR(
            self.constituents_ak, self.exclusive_jets_2[:, :1]
        )
        delta_r_2i = self._calc_deltaR(
            self.constituents_ak, self.exclusive_jets_2[:, 1:2]
        )
        self.pt_i = self.constituents_ak.pt
        # add new axis to make it broadcastable
        min_delta_r = ak.min(
            ak.concatenate(
                [
                    delta_r_1i[..., np.newaxis] ** self.beta,
                    delta_r_2i[..., np.newaxis] ** self.beta,
                ],
                axis=-1,
            ),
            axis=-1,
        )
        self.tau2 = ak.sum(self.pt_i * min_delta_r, axis=1) / self.d0

    def _calc_tau3(self):
        """Calculate the tau3 values."""
        delta_r_1i = self._calc_deltaR(
            self.constituents_ak, self.exclusive_jets_3[:, :1]
        )
        delta_r_2i = self._calc_deltaR(
            self.constituents_ak, self.exclusive_jets_3[:, 1:2]
        )
        delta_r_3i = self._calc_deltaR(
            self.constituents_ak, self.exclusive_jets_3[:, 2:3]
        )
        self.pt_i = self.constituents_ak.pt
        min_delta_r = ak.min(
            ak.concatenate(
                [
                    delta_r_1i[..., np.newaxis] ** self.beta,
                    delta_r_2i[..., np.newaxis] ** self.beta,
                    delta_r_3i[..., np.newaxis] ** self.beta,
                ],
                axis=-1,
            ),
            axis=-1,
        )
        self.tau3 = ak.sum(self.pt_i * min_delta_r, axis=1) / self.d0

    def histogram(self, features="pt", density=True, num_bins=100, use_quantiles=False):
        x = getattr(self, features)
        bins = (
            np.quantile(x, np.linspace(0.001, 0.999, num_bins))
            if use_quantiles
            else num_bins
        )
        return np.histogram(x, density=density, bins=bins)[0]

    def KLmetric1D(self, feature, reference, num_bins=100, use_quantiles=True):
        h1 = (
            self.histogram(
                feature, density=True, num_bins=num_bins, use_quantiles=use_quantiles
            )
            + 1e-8
        )
        h2 = (
            reference.histogram(
                feature, density=True, num_bins=num_bins, use_quantiles=use_quantiles
            )
            + 1e-8
        )
        return scipy.stats.entropy(h1, h2)

    def Wassertein1D(self, feature, reference):
        x = getattr(self, feature)
        y = getattr(reference, feature)
        return scipy.stats.wasserstein_distance(x, y)



# class JetDataModule:
#     """class that prepares the source-target coupling"""

#     def __init__(self, config, preprocess=False, metadata_path=None):

#         self.metadata = {}
#         self.metadata["continuous_features"] = {'pt': 0, 'eta_rel': 1, 'phi_rel': 2}
#         self.metadata["discrete_featsures"] = {
#             "isPhoton": [1, 0, 0, 0, 0],
#             "isNeutralHadron": [0, 1, 0, 0, 0],
#             "isChargedHadron": [0, 0, 1, 0, 0],
#             "isElectron": [0, 0, 0, 1, 0],
#             "isMuon": [0, 0, 0, 0, 1],
#         }
#         self.metadata["electric_charges"] = [-1, 0, 1]
#         self.metadata["particles_info"] = {
#             0: {"name": "photon", "color": "gold", "marker": "o", "tex": r"\gamma"},
#             1: {
#                 "name": "neutral hadron",
#                 "color": "darkred",
#                 "marker": "o",
#                 "tex": r"$\rm h^0$",
#             },
#             2: {
#                 "name": "negative hadron",
#                 "color": "darkred",
#                 "marker": "v",
#                 "tex": r"$\rm h^-$",
#             },
#             3: {
#                 "name": "positive hadron",
#                 "color": "darkred",
#                 "marker": "^",
#                 "tex": r"$\rm h^+$",
#             },
#             4: {"name": "electron", "color": "blue", "marker": "v", "tex": r"e^-"},
#             5: {"name": "positron", "color": "blue", "marker": "^", "tex": r"e^+"},
#             6: {"name": "muon", "color": "green", "marker": "v", "tex": r"\mu^-"},
#             7: {"name": "antimuon", "color": "green", "marker": "^", "tex": r"\mu^+"},
#         }

#         # ...define target and source:

#         self.target = ParticleClouds(
#             dataset=config.data.target_name,
#             path=config.data.target_path,
#             num_jets=config.data.num_jets,
#             min_num_particles=config.data.min_num_particles,
#             max_num_particles=config.data.max_num_particles,
#         )
#         self.source = ParticleClouds(
#             dataset=config.data.source_name,
#             path=config.data.source_path,
#             num_jets=config.data.num_jets,
#             min_num_particles=config.data.min_num_particles,
#             max_num_particles=config.data.max_num_particles,
#             multiplicity_dist=self.target.multiplicity
#             if "Noise" in config.data.source_name
#             else None,
#         )

#         self.metadata["source_data_stats"] = self.source.get_data_stats()
#         self.metadata["target_data_stats"] = self.target.get_data_stats()

#         # ...preprocess if needed:

#         if preprocess:
#             self.source.preprocess(
#                 continuous=config.data.source_preprocess_continuous,
#                 discrete=config.data.source_preprocess_discrete,
#                 **self.metadata["source_data_stats"],
#             )
#             self.target.preprocess(
#                 continuous=config.data.target_preprocess_continuous,
#                 discrete=config.data.target_preprocess_discrete,
#                 **self.metadata["target_data_stats"],
#             )
#             self.store_metadata(path=metadata_path)

#     def store_metadata(self, path):
#         if path:
#             log.info(f"Storing metadata at: {path}")
#             self.metadata_path = os.path.join(path, "metadata.json")
#             if not os.path.exists(self.metadata_path):
#                 with open(self.metadata_path, "w") as f:
#                     json.dump(self.metadata, f, indent=4)

