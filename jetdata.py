import torch
import numpy as np
import awkward as ak
import fastjet
import vector
import scipy
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from dataclasses import dataclass
from typing import List

plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["figure.autolayout"] = False

vector.register_awkward()

from utils import (
    extract_jetclass_features,
    extract_aoj_features,
    sample_noise,
    sample_masks,
    flavor_to_onehot,
    states_to_flavor,
)


@dataclass
class BridgeState:
    time: torch.Tensor = None
    continuous: torch.Tensor = None
    discrete: torch.Tensor = None
    absorbing: torch.Tensor = None

    def to(self, device):
        return BridgeState(
            time=self.time.to(device) if isinstance(self.time, torch.Tensor) else None,
            continuous=self.continuous.to(device)
            if isinstance(self.continuous, torch.Tensor)
            else None,
            discrete=self.discrete.to(device)
            if isinstance(self.discrete, torch.Tensor)
            else None,
            absorbing=self.absorbing.to(device)
            if isinstance(self.absorbing, torch.Tensor)
            else None,
        )

    @staticmethod
    def cat(states: List["BridgeState"], dim=0) -> "BridgeState":
        # Helper function to cat a specific attribute if not None
        def cat_attr(attr_name):
            attrs = [getattr(s, attr_name) for s in states]
            # Check if all are None
            if all(a is None for a in attrs):
                return None
            # Otherwise, all should be Tensors
            # Filter out None if needed (or assert if None is found)
            attrs = [a for a in attrs if a is not None]
            return torch.cat(attrs, dim=dim)

        return BridgeState(
            time=cat_attr("time"),
            continuous=cat_attr("continuous"),
            discrete=cat_attr("discrete"),
            absorbing=cat_attr("absorbing"),
        )


@dataclass
class OutputHeads:
    continuous: torch.Tensor = None
    discrete: torch.Tensor = None
    absorbing: torch.Tensor = None


class JetDataclass:
    """class that prepares the source-target coupling"""

    def __init__(self, config):
        self.config = config
        kwargs_target = config.data.target.params.to_dict()
        kwargs_source = config.data.source.params.to_dict()

        # ...target:

        self.target = ParticleClouds(
            dataset=config.data.target.name,
            data_paths=getattr(config.data.target, "path", None),
            **kwargs_target,
        )

        # ... noise source:

        kwargs_source["set_masks_like"] = (
            self.target.multiplicity
            if config.data.target.params.max_num_particles
            > config.data.target.params.min_num_particles
            else None
        )

        kwargs_source["num_jets"] = getattr(
            config.data.source.params, "num_jets", len(self.target)
        )

        self.source = ParticleClouds(
            dataset=config.data.source.name,
            data_paths=getattr(config.data.source, "path", None),
            **kwargs_source,
        )

    def preprocess(self, source_stats=None, target_stats=None):
        if hasattr(self.config.data.source, "preprocess"):
            self.source.preprocess(
                output_continuous=self.config.data.source.preprocess.continuous,
                output_discrete=self.config.data.source.preprocess.discrete,
                stats=source_stats,
            )
            self.config.data.source.preprocess.stats = (
                self.source.stats if hasattr(self.source, "stats") else target_stats
            )
        if hasattr(self.config.data.target, "preprocess"):
            self.target.preprocess(
                output_continuous=self.config.data.target.preprocess.continuous,
                output_discrete=self.config.data.target.preprocess.discrete,
                stats=target_stats,
            )
            self.config.data.target.preprocess.stats = (
                self.target.stats if hasattr(self.target, "stats") else source_stats
            )

    def postprocess(self, source_stats=None, target_stats=None):
        if hasattr(self.config.data.source, "preprocess"):
            self.source.postprocess(
                input_continuous=self.config.data.source.preprocess.continuous,
                input_discrete=self.config.data.source.preprocess.discrete,
                stats=self.config.data.source.preprocess.stats
                if source_stats is None
                else source_stats,
            )
            self.config.data.source.preprocess.stats = (
                self.source.stats if hasattr(self.source, "stats") else target_stats
            )
        if hasattr(self.config.data.target, "preprocess"):
            self.target.postprocess(
                input_continuous=self.config.data.target.preprocess.continuous,
                input_discrete=self.config.data.target.preprocess.discrete,
                stats=self.config.data.target.preprocess.stats
                if target_stats is None
                else target_stats,
            )
            self.config.data.target.preprocess.stats = (
                self.target.stats if hasattr(self.target, "stats") else source_stats
            )


class ParticleClouds:
    def __init__(self, dataset="JetClass", data_paths=None, **data_params):
        if isinstance(dataset, torch.Tensor):
            self.continuous, self.discrete, self.mask = (
                dataset[..., :3],
                dataset[..., 3:-1].long(),
                dataset[..., -1].unsqueeze(-1).long(),
            )
            if not self.discrete.nelement():
                del self.discrete

        elif isinstance(dataset, BridgeState):
            self.continuous = dataset.continuous
            self.discrete = dataset.discrete
            self.mask = dataset.absorbing
            if not self.discrete.nelement():
                del self.discrete

        elif "JetClass" in dataset:
            assert data_paths is not None, "Specify the path to the JetClass dataset"
            self.continuous, self.discrete, self.mask = extract_jetclass_features(
                data_paths, **data_params
            )

        elif "AspenOpenJets" in dataset:
            assert data_paths is not None, "Specify the path to the AOJ dataset"
            self.continuous, self.discrete, self.mask = extract_aoj_features(
                data_paths, **data_params
            )

        elif "Noise" in dataset:
            self.continuous, self.discrete = sample_noise(dataset, **data_params)
            self.mask = sample_masks(**data_params)
            self.continuous *= self.mask
            self.discrete *= self.mask

        # ...attributes:

        self.pt = self.continuous[..., 0]
        self.eta_rel = self.continuous[..., 1]
        self.phi_rel = self.continuous[..., 2]
        self.multiplicity = torch.sum(self.mask, dim=1)

        if hasattr(self, "discrete"):
            self.flavor = self.discrete[..., :-1]
            self.charge = self.discrete[..., -1]

    def __len__(self):
        return self.continuous.shape[0]

    def compute_4mom(self):
        self.px = self.pt * torch.cos(self.phi_rel)
        self.py = self.pt * torch.sin(self.phi_rel)
        self.pz = self.pt * torch.sinh(self.eta_rel)
        self.e = self.pt * torch.cosh(self.eta_rel)

    # ...data processing methods

    def summary_stats(self):
        mask = self.mask.squeeze(-1) > 0
        data = self.continuous[mask]
        return {
            "mean": data.mean(0).tolist(),
            "std": data.std(0).tolist(),
            "min": data.min(0).values.tolist(),
            "max": data.max(0).values.tolist(),
        }

    def preprocess(
        self, output_continuous="standardize", output_discrete="tokens", stats=None
    ):
        if output_discrete == "onehot_dequantize":
            one_hot = flavor_to_onehot(self.discrete[..., :-1], self.discrete[..., -1])
            self.continuous = torch.cat([self.continuous, one_hot], dim=-1)
            del self.discrete
        elif output_discrete == "tokens":
            one_hot = flavor_to_onehot(self.discrete[..., :-1], self.discrete[..., -1])
            self.discrete = torch.argmax(one_hot, dim=-1).unsqueeze(-1).long()

        if output_continuous == "standardize":
            self.stats = self.summary_stats() if stats is None else stats
            self.continuous = (self.continuous - torch.tensor(self.stats["mean"])) / (
                torch.tensor(self.stats["std"])
            )
            self.continuous *= self.mask
            self.pt = self.continuous[..., 0]
            self.eta_rel = self.continuous[..., 1]
            self.phi_rel = self.continuous[..., 2]

    def postprocess(
        self, input_continuous="standardize", input_discrete="tokens", stats=None
    ):
        if input_continuous == "standardize":
            if input_discrete == "onehot_dequantize":
                self.continuous = torch.cat([self.continuous, self.discrete], dim=-1)

            stats = getattr(self, "stats", stats)
            self.continuous = (
                self.continuous * torch.tensor(stats["std"])
            ) + torch.tensor(stats["mean"])
            self.continuous *= self.mask
            self.pt = self.continuous[..., 0]
            self.eta_rel = self.continuous[..., 1]
            self.phi_rel = self.continuous[..., 2]

        if input_discrete == "onehot_dequantize":
            discrete = (
                torch.argmax(self.continuous[..., 3:], dim=-1).unsqueeze(-1).long()
            )
            self.flavor, self.charge = states_to_flavor(discrete)
            self.discrete = torch.cat([self.flavor, self.charge], dim=-1)
            self.flavor *= self.mask
            self.charge *= self.mask
            self.discrete *= self.mask
            self.continuous = self.continuous[..., :3]

        if input_discrete == "tokens":
            self.flavor, self.charge = states_to_flavor(self.discrete)
            self.discrete = torch.cat([self.flavor, self.charge], dim=-1)
            self.flavor *= self.mask
            self.charge *= self.mask
            self.discrete *= self.mask

    # ...data visualization methods

    def histplot(
        self,
        feature="pt",
        idx=None,
        xlim=None,
        ylim=None,
        xlabel=None,
        ylabel=None,
        figsize=(3, 3),
        fontsize=12,
        ax=None,
        **kwargs,
    ):
        mask = self.mask.squeeze(-1) > 0
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)
        x = (
            getattr(self, feature)[mask]
            if idx is None
            else getattr(self, feature)[:, idx]
        )
        sns.histplot(x=x, element="step", ax=ax, **kwargs)
        ax.set_xlabel(feature if xlabel is None else xlabel, fontsize=fontsize)
        ax.set_ylabel(ylabel, fontsize=fontsize)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    def display_cloud(
        self,
        idx,
        scale_marker=1.0,
        ax=None,
        figsize=(3, 3),
        facecolor="whitesmoke",
        color="darkblue",
        title_box_anchor=(1.025, 1.125),
        savefig=None,
    ):
        eta = self.eta_rel[idx]
        phi = self.phi_rel[idx]
        pt = self.pt[idx] * scale_marker
        flavor = torch.argmax(self.flavor[idx], dim=-1)
        q = self.charge[idx]
        mask = self.mask[idx]
        pt = pt[mask.squeeze(-1) > 0]
        eta = eta[mask.squeeze(-1) > 0]
        phi = phi[mask.squeeze(-1) > 0]
        flavor = flavor[mask.squeeze(-1) > 0]
        charge = q[mask.squeeze(-1) > 0]

        if ax is None:
            _, ax = plt.subplots(figsize=figsize)

        ax.scatter(
            eta[flavor == 0],
            phi[flavor == 0],
            marker="o",
            s=pt[flavor == 0],
            color="gold",
            alpha=0.5,
            label=r"$\gamma$",
        )
        ax.scatter(
            eta[flavor == 1],
            phi[flavor == 1],
            marker="o",
            s=pt[flavor == 1],
            color="darkred",
            alpha=0.5,
            label=r"$h^{0}$",
        )
        ax.scatter(
            eta[(flavor == 2) & (charge < 0)],
            phi[(flavor == 2) & (charge < 0)],
            marker="^",
            s=pt[(flavor == 2) & (charge < 0)],
            color="darkred",
            alpha=0.5,
            label=r"$h^{-}$",
        )
        ax.scatter(
            eta[(flavor == 2) & (charge > 0)],
            phi[(flavor == 2) & (charge > 0)],
            marker="v",
            s=pt[(flavor == 2) & (charge > 0)],
            color="darkred",
            alpha=0.5,
            label=r"$h^{+}$",
        )
        ax.scatter(
            eta[(flavor == 3) & (charge < 0)],
            phi[(flavor == 3) & (charge < 0)],
            marker="^",
            s=pt[(flavor == 3) & (charge < 0)],
            color="blue",
            alpha=0.5,
            label=r"$e^{-}$",
        )
        ax.scatter(
            eta[(flavor == 3) & (charge > 0)],
            phi[(flavor == 3) & (charge > 0)],
            marker="v",
            s=pt[(flavor == 3) & (charge > 0)],
            color="blue",
            alpha=0.5,
            label=r"$e^{+}$",
        )
        ax.scatter(
            eta[(flavor == 4) & (charge < 0)],
            phi[(flavor == 4) & (charge < 0)],
            marker="^",
            s=pt[(flavor == 4) & (charge < 0)],
            color="green",
            alpha=0.5,
            label=r"$\mu^{-}$",
        )
        ax.scatter(
            eta[(flavor == 4) & (charge > 0)],
            phi[(flavor == 4) & (charge > 0)],
            marker="v",
            s=pt[(flavor == 4) & (charge > 0)],
            color="green",
            alpha=0.5,
            label=r"$\mu^{+}$",
        )

        # Define custom legend markers
        h1 = Line2D(
            [0],
            [0],
            marker="o",
            markersize=2,
            alpha=0.5,
            color="gold",
            linestyle="None",
        )
        h2 = Line2D(
            [0],
            [0],
            marker="o",
            markersize=2,
            alpha=0.5,
            color="darkred",
            linestyle="None",
        )
        h3 = Line2D(
            [0],
            [0],
            marker="^",
            markersize=2,
            alpha=0.5,
            color="darkred",
            linestyle="None",
        )
        h4 = Line2D(
            [0],
            [0],
            marker="v",
            markersize=2,
            alpha=0.5,
            color="darkred",
            linestyle="None",
        )
        h5 = Line2D(
            [0],
            [0],
            marker="^",
            markersize=2,
            alpha=0.5,
            color="blue",
            linestyle="None",
        )
        h6 = Line2D(
            [0],
            [0],
            marker="v",
            markersize=2,
            alpha=0.5,
            color="blue",
            linestyle="None",
        )
        h7 = Line2D(
            [0],
            [0],
            marker="^",
            markersize=2,
            alpha=0.5,
            color="green",
            linestyle="None",
        )
        h8 = Line2D(
            [0],
            [0],
            marker="v",
            markersize=2,
            alpha=0.5,
            color="green",
            linestyle="None",
        )

        plt.legend(
            [h1, h2, h3, h4, h5, h6, h7, h8],
            [
                r"$\gamma$",
                r"$h^0$",
                r"$h^-$",
                r"$h^+$",
                r"$e^-$",
                r"$e^+$",
                r"$\mu^{-}$",
                r"$\mu^{+}$",
            ],
            loc="upper right",
            markerscale=2,
            scatterpoints=1,
            fontsize=8,
            frameon=False,
            ncol=8,
            bbox_to_anchor=title_box_anchor,
            handletextpad=-0.5,
            columnspacing=0.1,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor(facecolor)  # Set the same color for the axis background
        if savefig is not None:
            plt.savefig(savefig)


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
