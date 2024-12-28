import torch
import numpy as np
import awkward as ak
import fastjet
import vector
import scipy
import seaborn as sns
import matplotlib.pyplot as plt

plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["figure.autolayout"] = False

vector.register_awkward()

from data.particle_clouds.particles import ParticleClouds


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
