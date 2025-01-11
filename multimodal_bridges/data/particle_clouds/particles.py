import torch
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import h5py

plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["figure.autolayout"] = False

from multimodal_bridge_matching import HybridState
from data.particle_clouds.utils import (
    extract_jetclass_features,
    extract_aoj_features,
    sample_noise,
    sample_masks,
    map_basis_to_tokens,
    map_tokens_to_basis,
    map_basis_to_onehot,
    map_onehot_to_basis,
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

        elif isinstance(dataset, HybridState):
            self.continuous = dataset.continuous
            self.discrete = dataset.discrete
            self.mask = dataset.mask
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
            if data_params.get("fill_target_with_noise", False):
                print("INFO: Filling target with noise")
                # ...sample noise
                noise_continuous = torch.randn_like(self.continuous)
                noise_discrete = torch.randint_like(self.mask, 0, 8)
                noise_discrete = map_tokens_to_basis(noise_discrete)
                # ...fill target with noise:
                self.continuous += noise_continuous * ~(self.mask > 0)
                self.discrete += noise_discrete * ~(self.mask > 0)

        elif "Noise" in dataset:
            self.continuous, self.discrete = sample_noise(dataset, **data_params)
            self.mask = sample_masks(**data_params)
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
        return {
            "mean": self.continuous[self.mask_bool].mean(0).tolist(),
            "std": self.continuous[self.mask_bool].std(0).tolist(),
            "min": self.continuous[self.mask_bool].min(0).values.tolist(),
            "max": self.continuous[self.mask_bool].max(0).values.tolist(),
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
