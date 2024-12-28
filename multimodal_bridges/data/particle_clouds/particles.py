import torch
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["figure.autolayout"] = False

from multimodal_bridge_matching import BridgeState
from data.particle_clouds.utils import (
    extract_jetclass_features,
    extract_aoj_features,
    sample_noise,
    sample_masks,
    flavor_to_onehot,
    states_to_flavor,
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
