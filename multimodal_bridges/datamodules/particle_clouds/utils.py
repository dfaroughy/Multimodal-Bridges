import torch
import numpy as np
import awkward as ak
import vector
from torch.nn import functional as F
import seaborn as sns
import matplotlib.pyplot as plt
from dataclasses import dataclass
import fastjet
import scipy

from tensorclass import TensorMultiModal

plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["figure.autolayout"] = False
vector.register_awkward()


@dataclass
class ParticleClouds:
    """
    A dataclass to hold particle clouds data.
    """

    data: TensorMultiModal = None

    def __post_init__(self):
        self.continuous = self.data.continuous
        self.discrete = self.data.discrete
        self.mask = self.data.mask
        self.pt = self.continuous[..., 0]
        self.eta_rel = self.continuous[..., 1]
        self.phi_rel = self.continuous[..., 2]
        self.px = self.pt * torch.cos(self.phi_rel)
        self.py = self.pt * torch.sin(self.phi_rel)
        self.pz = self.pt * torch.sinh(self.eta_rel)
        self.e = self.pt * torch.cosh(self.eta_rel)
        self.mask_bool = self.mask.squeeze(-1) > 0
        self.multiplicity = torch.sum(self.mask, dim=1)

        if self.continuous.shape[-1] > 3:
            self.d0 = self.continuous[..., 3]
            self.d0Err = self.continuous[..., 4]
            self.dz = self.continuous[..., 5] 
            self.dzErr = self.continuous[..., 6]
            self.d0_ratio = np.divide(self.d0, self.d0Err, out=np.zeros_like(self.d0), where=self.d0Err!=0)
            self.dz_ratio = np.divide(self.dz, self.dzErr, out=np.zeros_like(self.dz), where=self.dzErr!=0)


    def __len__(self):
        return len(self.data)

    @property
    def has_continuous(self):
        if self.data.has_continuous:
            return True
        return False
    
    @property
    def has_discrete(self):
        if self.data.has_discrete:
            return True
        return False

    def histplot(
        self,
        feature="pt",
        idx=None,
        apply_mask=None,
        xlim=None,
        ylim=None,
        xlabel=None,
        ylabel=None,
        figsize=(3, 3),
        fontsize=10,
        ax=None,
        **kwargs,
    ):
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)

        if feature == "multiplicity":
            x = self.multiplicity.squeeze(-1)
        else:
            if apply_mask is None:
                apply_mask = self.mask_bool

            x = (
                getattr(self, feature)[apply_mask]
                if idx is None
                else getattr(self, feature)[:, idx]
            )

        if isinstance(x, torch.Tensor):
            x.cpu().numpy()

        sns.histplot(x, element="step", ax=ax, **kwargs)
        ax.set_xlabel(
            "particle-level " + feature if xlabel is None else xlabel,
            fontsize=fontsize,
        )
        ax.set_ylabel(ylabel, fontsize=fontsize)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    def display_cloud(self, idx):
        """TODO
        Display the particle cloud image for a given index.
        """
        pass


@dataclass
class JetFeatures:
    """
    A dataclass to hold jet features and substructure.
    """

    data: TensorMultiModal = None

    def __post_init__(self):
        self.constituents = ParticleClouds(self.data)
        self.px = self.constituents.px.sum(axis=-1)
        self.py = self.constituents.py.sum(axis=-1)
        self.pz = self.constituents.pz.sum(axis=-1)
        self.e = self.constituents.e.sum(axis=-1)
        self.pt = torch.sqrt(self.px**2 + self.py**2)
        self.m = torch.sqrt(self.e**2 - self.pt**2 - self.pz**2)
        self.eta = 0.5 * torch.log((self.pt + self.pz) / (self.pt - self.pz))
        self.phi = torch.atan2(self.py, self.px)
        self.numParticles = torch.sum(self.constituents.mask, dim=1)

        self._substructure(R=0.8, beta=1.0, use_wta_scheme=True)

        if self.constituents.has_discrete:
            counts = self._get_flavor_counts()
            self.numPhotons = counts[..., 0]
            self.numNeutralHadrons = counts[..., 1]
            self.numNegativeHadrons = counts[..., 2]
            self.numPositiveHadrons = counts[..., 3]
            self.numElectrons = counts[..., 4]
            self.numPositrons = counts[..., 5]
            self.numMuons = counts[..., 6]
            self.numAntiMuons = counts[..., 7]
            self.numChargedHadrons = self.numPositiveHadrons + self.numNegativeHadrons
            self.numHadrons = self.numNeutralHadrons + self.numChargedHadrons
            self.numLeptons = (
                self.numElectrons
                + self.numPositrons
                + self.numMuons
                + self.numAntiMuons
            )
            self.numNeutrals = self.numPhotons + self.numNeutralHadrons
            self.numCharged = self.numChargedHadrons + self.numLeptons

            self.charge = self._jet_charge(kappa=0.0)
            self.jet_charge = self._jet_charge(kappa=1.0)

    # plotting methods:

    def histplot(
        self,
        features="pt",
        xlim=None,
        ylim=None,
        xlabel=None,
        ylabel=None,
        figsize=(3, 3),
        fontsize=10,
        ax=None,
        **kwargs,
    ):
        x = getattr(self, features)
        if isinstance(x, torch.Tensor):
            x.cpu().numpy()
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)
        sns.histplot(x=x, element="step", ax=ax, **kwargs)
        ax.set_xlabel(
            "jet-level " + features if xlabel is None else xlabel,
            fontsize=fontsize,
        )
        ax.set_ylabel(ylabel, fontsize=fontsize)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    # metrics:

    def KLmetric1D(self, feature, reference, num_bins=100, use_quantiles=True):
        h1 = (
            self._histogram(
                feature, density=True, num_bins=num_bins, use_quantiles=use_quantiles
            )
            + 1e-8
        )
        h2 = (
            reference._histogram(
                feature, density=True, num_bins=num_bins, use_quantiles=use_quantiles
            )
            + 1e-8
        )
        return scipy.stats.entropy(h1, h2)

    def Wassertein1D(self, feature, reference):
        x = getattr(self, feature)
        y = getattr(reference, feature)
        return scipy.stats.wasserstein_distance(x, y)

    # helper methods:

    def _histogram(
        self, features="pt", density=True, num_bins=100, use_quantiles=False
    ):
        x = getattr(self, features)
        bins = (
            np.quantile(x, np.linspace(0.001, 0.999, num_bins))
            if use_quantiles
            else num_bins
        )
        return np.histogram(x, density=density, bins=bins)[0]

    def _jet_charge(self, kappa):
        """jet charge defined as Q_j^kappa = Sum_i Q_i * (pT_i / pT_jet)^kappa"""
        charge = map_tokens_to_basis(self.constituents.discrete)[..., -1]
        jet_charge = charge * (self.constituents.pt) ** kappa
        return jet_charge.sum(axis=1) / (self.pt**kappa)

    def _get_flavor_counts(self, return_fracs=False, vocab_size=8):
        num_jets = len(self.constituents)
        tokens = self.constituents.discrete.squeeze(-1)
        mask = self.constituents.mask_bool

        flat_indices = (
            torch.arange(num_jets, device=tokens.device).unsqueeze(1) * (vocab_size + 1)
        ) + tokens * mask
        flat_indices = flat_indices[
            mask
        ]  # Use the mask to remove invalid (padded) values

        token_counts = torch.bincount(
            flat_indices, minlength=num_jets * (vocab_size + 1)
        )
        count = token_counts.view(num_jets, vocab_size + 1)  # Reshape to (B, n + 1)
        fracs = token_counts / mask.sum(
            dim=1, keepdim=True
        )  # Broadcast along the last dimension

        if return_fracs:
            return fracs

        return count

    def plot_flavor_count_per_jet(
        self,
        marker=".",
        color="darkred",
        markersize=2,
        ax=None,
        figsize=(3, 2),
        label=None,
    ):
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)

        count = self._get_flavor_counts(return_fracs=False).float()
        mean = count.mean(dim=0).cpu().numpy().tolist()
        std = count.std(dim=0).cpu().numpy().tolist()

        labels = [
            r"$\gamma$",
            r"$h^0$",
            r"$h^-$",
            r"$h^+$",
            r"$e^{-}$",
            r"$e^{+}$",
            r"$\mu^{-}$",
            r"$\mu^{+}$",
        ]
        binwidth = 1
        x_positions = np.arange(len(labels))

        for i, (mu, sig) in enumerate(zip(mean, std)):
            ax.add_patch(
                plt.Rectangle(
                    (i - binwidth / 2, mu - sig),
                    binwidth,
                    2 * sig,
                    color=color,
                    alpha=0.15,
                    linewidth=0.0,
                )
            )
            ax.plot(i, mu, marker, color=color, markersize=markersize, label=label)

        ax.set_xticks(x_positions)
        ax.set_xticklabels(labels)
        ax.set_yscale("log")
        ax.set_ylim(1e-2, 5000)
        ax.set_ylabel(r"$\langle N \rangle \pm 1\sigma$", fontsize=13)
        ax.set_xlabel(r"particle flavor", fontsize=13)

    def _substructure(self, R=0.8, beta=1.0, use_wta_scheme=True):
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

        self._constituents_ak = constituents_ak[ak.num(constituents_ak) >= 3]

        if use_wta_scheme:
            jetdef = fastjet.JetDefinition(
                fastjet.kt_algorithm, R, fastjet.WTA_pt_scheme
            )
        else:
            jetdef = fastjet.JetDefinition(fastjet.kt_algorithm, R)

        print("Clustering jets with fastjet")
        print("Jet definition:", jetdef)
        print("Calculating N-subjettiness")

        self._cluster = fastjet.ClusterSequence(self._constituents_ak, jetdef)
        self.d0 = self._calc_d0(R, beta)
        self.d2 = self._cluster.exclusive_jets_energy_correlator(njets=1, func="d2")
        self.tau1 = self._calc_tau1(beta)
        self.tau2 = self._calc_tau2(beta)
        self.tau3 = self._calc_tau3(beta)
        self.tau21 = np.ma.divide(self.tau2, self.tau1)
        self.tau32 = np.ma.divide(self.tau3, self.tau2)

    def _calc_deltaR(self, particles, jet):
        jet = ak.unflatten(ak.flatten(jet), counts=1)
        return particles.deltaR(jet)

    def _calc_d0(self, R, beta):
        """Calculate the d0 values."""
        return ak.sum(self._constituents_ak.pt * R**beta, axis=1)

    def _calc_tau1(self, beta):
        """Calculate the tau1 values."""
        excl_jets_1 = self._cluster.exclusive_jets(n_jets=1)
        delta_r_1i = self._calc_deltaR(self._constituents_ak, excl_jets_1[:, :1])
        pt_i = self._constituents_ak.pt
        return ak.sum(pt_i * delta_r_1i**beta, axis=1) / self.d0

    def _calc_tau2(self, beta):
        """Calculate the tau2 values."""
        excl_jets_2 = self._cluster.exclusive_jets(n_jets=2)
        delta_r_1i = self._calc_deltaR(self._constituents_ak, excl_jets_2[:, :1])
        delta_r_2i = self._calc_deltaR(self._constituents_ak, excl_jets_2[:, 1:2])
        pt_i = self._constituents_ak.pt

        # add new axis to make it broadcastable
        min_delta_r = ak.min(
            ak.concatenate(
                [
                    delta_r_1i[..., np.newaxis] ** beta,
                    delta_r_2i[..., np.newaxis] ** beta,
                ],
                axis=-1,
            ),
            axis=-1,
        )
        return ak.sum(pt_i * min_delta_r, axis=1) / self.d0

    def _calc_tau3(self, beta):
        """Calculate the tau3 values."""
        excl_jets_3 = self._cluster.exclusive_jets(n_jets=3)
        delta_r_1i = self._calc_deltaR(self._constituents_ak, excl_jets_3[:, :1])
        delta_r_2i = self._calc_deltaR(self._constituents_ak, excl_jets_3[:, 1:2])
        delta_r_3i = self._calc_deltaR(self._constituents_ak, excl_jets_3[:, 2:3])
        pt_i = self._constituents_ak.pt

        min_delta_r = ak.min(
            ak.concatenate(
                [
                    delta_r_1i[..., np.newaxis] ** beta,
                    delta_r_2i[..., np.newaxis] ** beta,
                    delta_r_3i[..., np.newaxis] ** beta,
                ],
                axis=-1,
            ),
            axis=-1,
        )
        return ak.sum(pt_i * min_delta_r, axis=1) / self.d0


def map_basis_to_tokens(tensor):
    """
    works for jwetclass & aoj datastes. Maps a tensor with shape (N, M, D)
    to a space of 8 tokens based on particle type and charge.

    Args:
        tensor (torch.Tensor): Input tensor of shape (N, M, D) where D=6.

    Returns:
        torch.Tensor: A tensor of shape (N, M) containing the token mappings.
    """

    if tensor.shape[-1] != 6:
        raise ValueError("The last dimension of the input tensor must be 6.")
    one_hot = tensor[..., :-1]  # Shape: (N, M, 5)
    charge = tensor[..., -1]  # Shape: (N, M)
    flavor_charge_combined = one_hot.argmax(dim=-1) * 10 + charge  # Shape: (N, M)
    map_rules = {
        0: 0,  # Photon (1, 0, 0, 0, 0; 0)
        10: 1,  # Neutral hadron (0, 1, 0, 0, 0; 0)
        19: 2,  # Negatively charged hadron (0, 0, 1, 0, 0; -1)
        21: 3,  # Positively charged hadron (0, 0, 1, 0, 0; 1)
        29: 4,  # Negatively charged electron (0, 0, 0, 1, 0; -1)
        31: 5,  # Positively charged electron (0, 0, 0, 1, 0; 1)
        39: 6,  # Negatively charged muon (0, 0, 0, 0, 1; -1)
        41: 7,  # Positively charged muon (0, 0, 0, 0, 1; 1)
    }
    tokens = torch.full_like(
        flavor_charge_combined, -1, dtype=torch.int64
    )  # Initialize with invalid token
    for key, value in map_rules.items():
        tokens[flavor_charge_combined == key] = value
    return tokens.unsqueeze(-1)


def map_tokens_to_basis(tokens):
    """
    Maps a tensor of tokens (integers 0-7) back to the original basis representation.

    Args:
        tokens (torch.Tensor): A tensor of shape (N, M) containing token values (0-7).

    Returns:
        torch.Tensor: A tensor of shape (N, M, 6) with the original basis representation.
    """
    token_to_basis = torch.tensor(
        [
            [1, 0, 0, 0, 0, 0],  # Photon 0
            [0, 1, 0, 0, 0, 0],  # Neutral hadron 1
            [0, 0, 1, 0, 0, -1],  # Negatively charged hadron 2
            [0, 0, 1, 0, 0, 1],  # Positively charged hadron 3
            [0, 0, 0, 1, 0, -1],  # Negatively charged electron 4
            [0, 0, 0, 1, 0, 1],  # Positively charged electron 5
            [0, 0, 0, 0, 1, -1],  # Negatively charged muon 6
            [0, 0, 0, 0, 1, 1],  # Positively charged muon 7
        ],
        dtype=torch.float32,
    )
    basis_tensor = token_to_basis[tokens.squeeze(-1)]
    return basis_tensor


def map_basis_to_onehot(tensor):
    return F.one_hot(map_basis_to_tokens(tensor).squeeze(-1), num_classes=8)


def map_onehot_to_basis(onehot):
    return map_tokens_to_basis(onehot.argmax(dim=-1).unsqueeze(-1))
