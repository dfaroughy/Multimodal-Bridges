import seaborn as sns
import numpy as np
import torch
import tqdm as tqdm
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import jensenshannon


# Function to plot histograms and ratios
def plot_hist_and_ratio(test, 
                        gen1,  
                        gen2,
                        ax_hist, 
                        ax_ratio, 
                        xlabel=None, 
                        xlim=None,
                        ylim=None,
                        ylabel='',
                        feat=None,
                        num_bins=100, 
                        log_scale=False, 
                        color1=None,
                        color2=None,
                        color3='dodgerblue',
                        color_test='darkslategrey',
                        legend1=None,
                        legend2=None,
                        legend3=None,
                        discrete=False,
                        fill=False,
                        lw=0.75,
                        ls='-',):

    if xlim is not None:    
        bins = np.linspace(xlim[0], xlim[1], num_bins)  # Adjust binning if necessary
    else:
        bins = num_bins
        
    if feat is not None:
        test.histplot(feat, apply_map=lambda test: test[test!=0],  stat='density', fill=fill,bins=bins, alpha=1 if not fill else 0.2, ax=ax_hist, lw=1.25 if not fill else 0.35, color=color_test, label="AOJ", log_scale=log_scale, discrete=discrete)
        gen1.histplot(feat, apply_map=lambda gen1: gen1[gen1!=0],  stat='density', fill=False, bins=bins, ax=ax_hist, ls=ls, lw=lw, color=color1, label=legend1, log_scale=log_scale, discrete=discrete)
        gen2.histplot(feat, apply_map=lambda gen2: gen2[gen2!=0],  stat='density', fill=False, bins=bins, ax=ax_hist, ls=ls, lw=lw, color=color2, label=legend2, log_scale=log_scale, discrete=discrete)
        x = getattr(test, feat)
        x = x[x!=0]
        y1 = getattr(gen1, feat)
        y2 = getattr(gen2, feat)
        y1 = y1[y1!=0]
        y2 = y2[y2!=0]

    else:
        sns.histplot(test[test!=0], stat='density', fill=fill, element='step', bins=bins, ax=ax_hist, alpha=1 if not fill else 0.3, lw=1.25 if not fill else 0.35, color=color_test, label="AOJ", log_scale=log_scale, discrete=discrete)
        sns.histplot(gen1[gen1!=0], stat='density', fill=False, element='step' ,bins=bins, ax=ax_hist, ls=ls, lw=lw, color=color1, label=legend1, log_scale=log_scale, discrete=discrete)
        sns.histplot(gen2[gen2!=0], stat='density', fill=False, element='step' ,bins=bins, ax=ax_hist, ls=ls, lw=lw, color=color2, label=legend2, log_scale=log_scale, discrete=discrete)
        x = test[test!=0]
        y1 = gen1[gen1!=0]
        y2 = gen2[gen2!=0]

    if isinstance(x, torch.Tensor):
        x = x.cpu().numpy()

    if isinstance(y1, torch.Tensor):
        y1 = y1.cpu().numpy()
        y2 = y2.cpu().numpy()

    hist, _ = np.histogram(x, bins=bins, density=True)
    hist1, _ = np.histogram(y1, bins=bins, density=True)
    hist2, _ = np.histogram(y2, bins=bins, density=True)
    ratio1 = np.divide(hist1, hist, out=np.ones_like(hist1, dtype=float), where=hist > 0)
    ratio2 = np.divide(hist2, hist, out=np.ones_like(hist2, dtype=float), where=hist > 0)

    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # Plot ratio
    ax_ratio.hist(bin_centers, bins=bins, weights=ratio1, histtype="step", color=color1, ls=ls)
    ax_ratio.hist(bin_centers, bins=bins, weights=ratio2, histtype="step", color=color2, ls=ls)
    ax_ratio.axhline(1.0, color='gray', linestyle='--', lw=0.5)

    # Format axes
    ax_hist.set_ylabel(ylabel, fontsize=10)
    
    if legend1 is not None:
        ax_hist.legend(fontsize=8)

    ax_ratio.set_ylabel("Ratio", fontsize=8)

    if xlabel is not None:
        ax_ratio.set_xlabel(xlabel, fontsize=15)

    ax_ratio.set_ylim(0.5, 1.5)
    ax_ratio.set_yticks([0.7, 1.0, 1.3])
    ax_ratio.set_yticklabels([0.7, 1.0, 1.3], fontsize=7)

    if xlim is not None:
        ax_hist.set_xlim(*xlim)
        ax_ratio.set_xlim(*xlim)

    if ylim is not None:
        ax_hist.set_ylim(*ylim)



class EnergyCorrelationFunctions:

    def __init__(self, data):
        self.data = data
        self.mask_3_parts = data.mask.sum(dim=1).squeeze(-1) >= 3 

    def get_flavor(self, token):
        flavor = self.data.clone()
        if token == 4567:
            flavor_mask = flavor.discrete >= 4
        else:
            flavor_mask = flavor.discrete == token
        flavor.apply_mask(flavor_mask)
        flavor.mask *= flavor_mask 
        del flavor.discrete
        return flavor
        
    def compute_ecf(self, flavor_i, flavor_j=None, beta=1.0):

        flavor = {'photon': 0, 
                  'h0': 1,
                  'h-': 2,
                  'h+': 3,
                  'e-': 4,
                  'e+': 5,
                  'mu-': 6,
                  'mu+': 7,
                  'hadron': 123,
                  'lepton': 4567,
                  'positive':357,
                  'negative':246,
                  'charged': 234567,
                  'h+/-': 23,
                  'e+/-': 45,
                  'mu+/-': 67,
                  'neutral': 10,
                  }
                  
        i = flavor[flavor_i]
        j = flavor[flavor_j] if flavor_j is not None else None

        if flavor_j is None:
            jets = self.get_flavor(i).continuous
            return self._auto_ecf(jets, beta)
        else:
            jets_i = self.get_flavor(i).continuous
            jets_j = self.get_flavor(j).continuous
            return self._cross_ecf(jets_i, jets_j, beta)

    def _auto_ecf(self, tensor, beta=1.0):

        auto_ecf = []
        jet_pT2 = []

        for jet in tensor:
            
            jet = jet[jet[:, 0] != 0]

            if len(jet) < 2:
                auto_ecf.append(0.0)
                jet_pT2.append(0.0)
                continue

            pT = jet[:, 0]
            eta = jet[:, 1]
            phi = jet[:, 2]

            pT2 = pT.sum() ** 2

            # Compute pairwise differences
            delta_eta = eta.unsqueeze(1) - eta.unsqueeze(0)
            delta_phi = phi.unsqueeze(1) - phi.unsqueeze(0)
            delta_phi = torch.remainder(delta_phi + torch.pi, 2 * torch.pi) - torch.pi

            # Compute pairwise distances
            R_ij = torch.sqrt(delta_eta**2 + delta_phi**2) ** beta

            ecf_matrix = (pT.unsqueeze(1) * pT.unsqueeze(0)) * R_ij / 2.0
            ecf = ecf_matrix.sum() / pT2

            auto_ecf.append(ecf.item())
            jet_pT2.append(pT2.item())

        auto_ecf = torch.tensor(auto_ecf)[self.mask_3_parts]
        jet_pT2 = torch.tensor(jet_pT2)[self.mask_3_parts]

        return auto_ecf, jet_pT2


    def _cross_ecf(self, tensor_1, tensor_2, beta=1.0):

        cross_ecf = []
        jet_pT2 = []

        for idx, jet in enumerate(tensor_1):

            j0 = jet[jet[:, 0] != 0]
            j1 = tensor_2[idx][tensor_2[idx][:, 0] != 0]

            if len(jet) == 0  or len(jet) == 0:
                cross_ecf.append(0.0)
                jet_pT2.append(0.0)
                continue

            pT_0, eta_0, phi_0 = j0[:, 0], j0[:, 1], j0[:, 2]
            pT_1, eta_1, phi_1 = j1[:, 0], j1[:, 1], j1[:, 2]

            pT2 = pT_0.sum() * pT_1.sum()

            delta_eta = eta_0.unsqueeze(1) - eta_1.unsqueeze(0)
            delta_phi = phi_0.unsqueeze(1) - phi_1.unsqueeze(0)
            delta_phi = torch.remainder(delta_phi + np.pi, 2 * np.pi) - np.pi

            R_ij = torch.sqrt(delta_eta**2 + delta_phi**2) ** beta

            ecf_matrix = (pT_0.unsqueeze(1) * pT_1.unsqueeze(0)) * R_ij
            ecf = ecf_matrix.sum() / pT2

            cross_ecf.append(ecf.item())
            jet_pT2.append(pT2.item())

        cross_ecf = torch.tensor(cross_ecf)[self.mask_3_parts]
        jet_pT2 = torch.tensor(jet_pT2)[self.mask_3_parts]

        return cross_ecf, jet_pT2



def chi2_metric(target, ref, bins, use_density=True, eps=1e-10):
    # Compute histograms
    hist_ref = np.histogram(ref, density=use_density, bins=bins)[0]
    hist = np.histogram(target, density=use_density, bins=bins)[0]
    ratio = (hist_ref - hist)**2 / (hist_ref + hist + eps)
    return np.sum(ratio)


def wasserstein_metric(target, ref, bins=None, use_density=False):
    # Compute histograms and bin edges
    if bins is not None:
        hist_ref, bin_edges = np.histogram(ref, density=use_density, bins=bins)
        hist, _ = np.histogram(target, density=use_density, bins=bins)
        
        # Compute bin centers
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # If using density=True, convert densities to probability masses by multiplying by bin width.
        if use_density:
            bin_widths = np.diff(bin_edges)
            hist_ref = hist_ref * bin_widths
            hist = hist * bin_widths
        else:
            # Normalize raw counts to convert to probabilities.
            hist_ref = hist_ref / np.sum(hist_ref) if np.sum(hist_ref) > 0 else hist_ref
            hist = hist / np.sum(hist) if np.sum(hist) > 0 else hist

        # Compute Wasserstein distance using the weighted bin centers.
        return wasserstein_distance(bin_centers, bin_centers, u_weights=hist, v_weights=hist_ref)
    else:
        target = target[np.isnan(target)==0]
        ref = ref[np.isnan(ref)==0]        
        return wasserstein_distance(target, ref)


def jensen_shannon_divergence(target, ref, bins, use_density=False, return_squared=False):
    # Compute histograms
    hist_ref, _ = np.histogram(ref, density=use_density, bins=bins)
    hist, _ = np.histogram(target, density=use_density, bins=bins)
    
    # Convert to probability distributions.
    if use_density:
        # When density=True, multiply by bin width to get probabilities.
        # Note: This assumes all bins are of equal width.
        bin_edges = np.histogram_bin_edges(ref, bins=bins)
        bin_width = np.diff(bin_edges)[0]
        hist_ref = hist_ref * bin_width
        hist = hist * bin_width
    else:
        total_ref = np.sum(hist_ref)
        total = np.sum(hist)
        if total_ref > 0:
            hist_ref = hist_ref / total_ref
        if total > 0:
            hist = hist / total
    
    # Compute the Jensen-Shannon divergence.
    # Note: jensenshannon returns the square root of the divergence.
    js_distance = jensenshannon(hist_ref, hist, base=2)
    
    if return_squared:
        return js_distance**2
    else:
        return js_distance

    
def kl_divergence(target, ref, bins, use_density=False, epsilon=1e-10):
    # Compute histograms and bin edges
    hist_ref, bin_edges = np.histogram(ref, density=use_density, bins=bins)
    hist_target, _ = np.histogram(target, density=use_density, bins=bins)
    
    # If using density, convert to probability masses (assumes equal bin widths)
    if use_density:
        bin_width = np.diff(bin_edges)[0]
        hist_ref = hist_ref * bin_width
        hist_target = hist_target * bin_width
    else:
        # Normalize raw counts to get probability distributions
        total_ref = np.sum(hist_ref)
        total_target = np.sum(hist_target)
        if total_ref > 0:
            hist_ref = hist_ref / total_ref
        if total_target > 0:
            hist_target = hist_target / total_target

    # Add epsilon to avoid division by zero and log of zero
    p = hist_ref + epsilon
    q = hist_target + epsilon

    # Compute the KL divergence D_KL(p || q)
    kl_div = np.sum(p * np.log(p / q))
    return kl_div
