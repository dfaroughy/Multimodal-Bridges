import torch
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import h5py
from torch.distributions.categorical import Categorical

plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["figure.autolayout"] = False

from data.dataclasses import MultiModeState
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

        elif isinstance(dataset, MultiModeState):
            if "continuous" in dataset.available_modes():
                self.continuous = dataset.continuous
            if "discrete" in dataset.available_modes():
                self.discrete = dataset.discrete
            self.mask = dataset.mask

        elif "jetclass" in dataset:
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

        # ... useful attributes:

        self.pt = self.continuous[..., 0]
        self.eta_rel = self.continuous[..., 1]
        self.phi_rel = self.continuous[..., 2]
        self.multiplicity = torch.sum(self.mask, dim=1)
        self.mask_bool = self.mask.squeeze(-1) > 0

    def __len__(self):
        return self.mask.size(0)

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

    def save_to(self, path):
        with h5py.File(path, "w") as f:
            f.create_dataset("continuous", data=self.continuous)
            if hasattr(self, "discrete"):
                f.create_dataset("discrete", data=self.discrete)
            f.create_dataset("mask", data=self.mask)

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



class MultiModalNoise:
    
    def __init__(self, config):
        self.config = config

    def sample(self, shape):
        """
        Sample multi-modal random source:
        - time: None
        - continuous: N(0, 1) for each particle feature
        - discrete: random token in [0, vocab_size) for each particle
        - mask: number of active particles from a categorical distribution with `categorial_probs`

        """

        num_jets, max_num_particles = shape
        vocab_size = self.config.data.vocab_size

        if self.config.data.target_name == "AspenOpenJets":
            categorial_probs = [
            0.0, 0.0, 2.3e-05, 4.2e-05, 4.5e-05, 7.6e-05, 0.000126, 0.000139, 0.000295, 
            0.000411, 0.000711, 0.001076, 0.001498, 0.002125, 0.002676, 0.003562, 0.004589, 
            0.005568, 0.006726, 0.00745, 0.008848, 0.009891, 0.010794, 0.011616, 0.012913, 
            0.014028, 0.014461, 0.01527, 0.016087, 0.017046, 0.017625, 0.018257, 0.018313, 
            0.018739, 0.019904, 0.01987, 0.020017, 0.020573, 0.020321, 0.020915, 0.021078, 
            0.021068, 0.021238, 0.021362, 0.021184, 0.020843, 0.021087, 0.020803, 0.020675, 
            0.02012, 0.019687, 0.019767, 0.019179, 0.018818, 0.018088, 0.017954, 0.017493, 
            0.01666, 0.015898, 0.015648, 0.015328, 0.014447, 0.013704, 0.0133, 0.012902, 
            0.011917, 0.011454, 0.010991, 0.010148, 0.009768, 0.009205, 0.008871, 0.008108, 
            0.007705, 0.007257, 0.006812, 0.006552, 0.005785, 0.005534, 0.004937, 0.004508, 
            0.004359, 0.003957, 0.003599, 0.003272, 0.003055, 0.002747, 0.002811, 0.002423, 
            0.002255, 0.001991, 0.001862, 0.001687, 0.001502, 0.001489, 0.001202, 0.001192, 
            0.001114, 0.0009, 0.000844, 0.000768, 0.000693, 0.000636, 0.000547, 0.000447, 
            0.000453, 0.000381, 0.000422, 0.000332, 0.000267, 0.000254, 0.000234, 0.000208, 
            0.000167, 0.000152, 0.000135, 0.000108, 9.5e-05, 0.000117, 9.9e-05, 9.4e-05, 
            8e-05, 5e-05, 6.3e-05, 2.7e-05, 3.2e-05, 4.9e-05, 4.3e-05,
            ]

        elif self.config.data.target.name == "JetClass":
            # TODO
            pass

        if categorial_probs:
            cat = Categorical(torch.tensor(categorial_probs))
            multiplicity = cat.sample((num_jets,))
            mask = torch.zeros((num_jets, max_num_particles))
            for i, n in enumerate(multiplicity):
                mask[i, :n] = 1
            mask = mask.long().unsqueeze(-1)
        else:
            mask = torch.ones((num_jets, max_num_particles, 1)).long()

        time = None
        continuous = torch.randn((num_jets, max_num_particles, 3)) * mask

        if vocab_size > 0:
            discrete = torch.randint(0, vocab_size, (num_jets, max_num_particles, 1)) * mask
        else:
            discrete = None

        return time, continuous, discrete, mask