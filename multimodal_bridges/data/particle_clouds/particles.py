import torch
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["figure.autolayout"] = False

from data.states import HybridState
from data.particle_clouds.utils import (
    extract_jetclass_features,
    extract_aoj_features,
    sample_noise,
    sample_masks,
    map_basis_to_tokens,
    map_tokens_to_basis,
)


class ParticleClouds:
    def __init__(
        self,
        dataset="JetClass",
        path=None,
        num_jets=100_000,
        min_num_particles=0,
        max_num_particles=128,
        multiplicity_dist=None,
    ):
        self.max_num_particles = max_num_particles

        if isinstance(dataset, torch.Tensor):
            self.continuous, self.discrete, self.mask = (
                dataset[..., :3],
                dataset[..., 3:-1].long(),
                dataset[..., -1].unsqueeze(-1).long(),
            )
            if not self.discrete.nelement():
                del self.discrete

        elif isinstance(dataset, HybridState):
            self.continuous = dataset.continuous
            self.discrete = dataset.discrete
            self.mask = dataset.mask
            if not self.discrete.nelement():
                del self.discrete

        elif "JetClass" in dataset:
            assert path is not None, "Specify the path to the JetClass dataset"
            self.continuous, self.discrete, self.mask = extract_jetclass_features(
                path,
                num_jets,
                min_num_particles,
                max_num_particles,
            )

        elif "AspenOpenJets" in dataset:
            assert path is not None, "Specify the path to the AOJ dataset"
            self.continuous, self.discrete, self.mask = extract_aoj_features(
                path,
                num_jets,
                min_num_particles,
                max_num_particles,
            )

        elif "Noise" in dataset:
            self.continuous, self.discrete = sample_noise(num_jets, max_num_particles)
            self.mask = sample_masks(
                multiplicity_dist,
                num_jets,
                min_num_particles,
                max_num_particles,
            )
            self.continuous *= self.mask
            self.discrete *= self.mask

        # ... useful attributes:

        self.pt = self.continuous[..., 0]
        self.eta_rel = self.continuous[..., 1]
        self.phi_rel = self.continuous[..., 2]
        self.multiplicity = torch.sum(self.mask, dim=1)
        self.mask_bool = self.mask.squeeze(-1) > 0

    def __len__(self):
        return self.continuous.shape[0]

    def compute_4mom(self):
        self.px = self.pt * torch.cos(self.phi_rel)
        self.py = self.pt * torch.sin(self.phi_rel)
        self.pz = self.pt * torch.sinh(self.eta_rel)
        self.e = self.pt * torch.cosh(self.eta_rel)

    def get_data_stats(self):
        hist, _ = np.histogram(
            self.multiplicity,
            bins=np.arange(0, self.max_num_particles + 2, 1),
            density=True,
        )
        return {
            "mean": self.continuous[self.mask_bool].mean(0).tolist(),
            "std": self.continuous[self.mask_bool].std(0).tolist(),
            "min": self.continuous[self.mask_bool].min(0).values.tolist(),
            "max": self.continuous[self.mask_bool].max(0).values.tolist(),
            "multinomial_num_particles": hist.tolist(),
            "num_particles_mean": torch.mean(
                self.multiplicity.squeeze(-1).float()
            ).item(),
            "num_particles_std": torch.std(
                self.multiplicity.squeeze(-1).float()
            ).item(),
        }

    #############################
    # ...data processing methods
    #############################

    def preprocess(self, continuous, discrete, **kwargs):
        if discrete == "tokenize":
            self.discrete = map_basis_to_tokens(self.discrete)
            self.discrete *= self.mask
        if continuous == "standardize":
            mean = torch.tensor(kwargs["mean"])
            std = torch.tensor(kwargs["std"])
            self.continuous = (self.continuous - torch.tensor(mean)) / (
                torch.tensor(std)
            )
            self.continuous *= self.mask
            self.pt = self.continuous[..., 0]
            self.eta_rel = self.continuous[..., 1]
            self.phi_rel = self.continuous[..., 2]

    def postprocess(self, continuous, discrete, **kwargs):
        if continuous == "standardize":
            mean = torch.tensor(kwargs["mean"])
            std = torch.tensor(kwargs["std"])
            self.continuous = (self.continuous * torch.tensor(std)) + torch.tensor(mean)
            self.continuous *= self.mask
            self.pt = self.continuous[..., 0]
            self.eta_rel = self.continuous[..., 1]
            self.phi_rel = self.continuous[..., 2]
        if discrete == "tokenize":
            self.discrete = map_tokens_to_basis(self.discrete)
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
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)
        x = (
            getattr(self, feature)[self.mask_bool]
            if idx is None
            else getattr(self, feature)[:, idx]
        )
        sns.histplot(x=x, element="step", ax=ax, **kwargs)
        ax.set_xlabel(feature if xlabel is None else xlabel, fontsize=fontsize)
        ax.set_ylabel(ylabel, fontsize=fontsize)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
