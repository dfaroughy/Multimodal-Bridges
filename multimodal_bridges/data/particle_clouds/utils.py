import torch
import numpy as np
import awkward as ak
import vector
import uproot
import h5py
from sklearn.preprocessing import OneHotEncoder
from torch.distributions.categorical import Categorical
from torch.nn import functional as F

vector.register_awkward()


def read_root_file(filepath):
    """Loads a single .root file from the JetClass dataset."""
    x = uproot.open(filepath)["tree"].arrays()
    mom = ak.zip(
        {"px": x.part_px, "py": x.part_py, "pz": x.part_pz, "energy": x.part_energy},
        with_name="Momentum4D",
    )
    mom_jet = ak.sum(mom, axis=1)
    x["part_pt"] = mom.pt
    x["part_eta"] = mom.eta
    x["part_phi"] = mom.phi
    x["part_etarel"] = mom.deltaeta(mom_jet)
    x["part_phirel"] = mom.deltaphi(mom_jet)
    x["mask"] = np.ones_like(x["part_energy"])
    return x


def read_aoj_file(filepath):
    """Loads a single .h5 file from the AOJ dataset."""

    def np_to_ak(x: np.ndarray, names: list, mask: np.ndarray = None, dtype="float32"):
        if mask is None:
            mask = np.ones_like(x[..., 0], dtype="bool")
        return ak.Array(
            {
                name: ak.values_astype(
                    ak.drop_none(ak.mask(ak.Array(x[..., i]), mask != 0)),
                    dtype,
                )
                for i, name in enumerate(names)
            }
        )

    with h5py.File(filepath, "r") as f:
        PFCands = f["PFCands"][:]
        mask_bad_pids = np.abs(PFCands[:, :, -2]) < 11  # remove weird "quark" pid's
        PFCands[mask_bad_pids] = np.zeros_like(PFCands[mask_bad_pids])
        feature_to_encode = np.abs(PFCands[:, :, -2])
        feature_to_encode[feature_to_encode == 11] = 0
        feature_to_encode[feature_to_encode == 13] = 1
        feature_to_encode[feature_to_encode == 22] = 2
        feature_to_encode[feature_to_encode == 130] = 3
        feature_to_encode[feature_to_encode == 211] = 4
        encoder = OneHotEncoder(sparse_output=False)
        one_hot_encoded = encoder.fit_transform(
            feature_to_encode.flatten().reshape(-1, 1)
        )
        one_hot_encoded = one_hot_encoded.reshape(
            PFCands.shape[0], PFCands.shape[1], -1
        )
        PFCands = np.concatenate(
            (PFCands[:, :, :-2], one_hot_encoded, PFCands[:, :, -1:]), axis=-1
        )
        x = np_to_ak(
            x=PFCands,
            names=[
                "part_px",
                "part_py",
                "part_pz",
                "part_energy",
                "part_d0",
                "part_d0Err",
                "part_dz",
                "part_dzErr",
                "part_charge",
                "part_isElectron",
                "part_isMuon",
                "part_isPhoton",
                "part_isNeutralHadron",
                "part_isChargedHadron",
                "part_PUPPI",
            ],
            mask=PFCands[:, :, 3] > 0,
        )
        mom = ak.zip(
            {
                "px": x.part_px,
                "py": x.part_py,
                "pz": x.part_pz,
                "energy": x.part_energy,
            },
            with_name="Momentum4D",
        )
        mom_jet = ak.sum(mom, axis=1)
        x["part_pt"] = mom.pt
        x["part_eta"] = mom.eta
        x["part_phi"] = mom.phi
        x["part_etarel"] = mom.deltaeta(mom_jet)
        x["part_phirel"] = mom.deltaphi(mom_jet)
        x["mask"] = PFCands[:, :, 3] > 0
    return x


def pad(a, min_num, max_num, value=0, dtype="float32"):
    assert max_num >= min_num, "max_num must be >= min_num"
    assert isinstance(a, ak.Array), "Input must be an awkward array"
    a = a[ak.num(a) >= min_num]
    a = ak.fill_none(ak.pad_none(a, max_num, clip=True), value)
    return ak.values_astype(a, dtype)


def extract_jetclass_features(
    dataset, num_jets=100_000, min_num_particles=0, max_num_particles=128
):
    if isinstance(dataset, str):
        dataset = [dataset]
    all_data = []
    for data in dataset:
        assert ".root" in data, "Input should be a path to a .root file"
        data = read_root_file(data)
        features = [
            "part_pt",
            "part_etarel",
            "part_phirel",
            "part_isPhoton",
            "part_isNeutralHadron",
            "part_isChargedHadron",
            "part_isElectron",
            "part_isMuon",
            "part_charge",
            "mask",
        ]
        data = torch.tensor(
            np.stack(
                [
                    ak.to_numpy(
                        pad(
                            data[feat],
                            min_num=min_num_particles,
                            max_num=max_num_particles,
                        )
                    )
                    for feat in features
                ],
                axis=1,
            )
        )
        data = torch.permute(data, (0, 2, 1))
        all_data.append(data)
    data = torch.cat(all_data, dim=0)
    idx = torch.argsort(data[..., 0], dim=1, descending=True)
    data_pt_sorted = torch.gather(
        data, 1, idx.unsqueeze(-1).expand(-1, -1, data.size(2))
    )
    data_pt_sorted = data_pt_sorted[:num_jets]
    return (
        data_pt_sorted[..., :3],
        data_pt_sorted[..., 3:-1],
        data_pt_sorted[..., -1].unsqueeze(-1).long(),
    )


def extract_aoj_features(
    dataset, num_jets=100_000, min_num_particles=0, max_num_particles=128
):
    if isinstance(dataset, str):
        dataset = [dataset]
    all_data = []
    for data in dataset:
        assert ".h5" in data, "Input should be a path to a .h5 file"
        d = read_aoj_file(data)
        features = [
            "part_pt",
            "part_etarel",
            "part_phirel",
            "part_isPhoton",
            "part_isNeutralHadron",
            "part_isChargedHadron",
            "part_isElectron",
            "part_isMuon",
            "part_charge",
            "mask",
        ]
        data = torch.tensor(
            np.stack(
                [
                    ak.to_numpy(
                        pad(
                            d[feat],
                            min_num=min_num_particles,
                            max_num=max_num_particles,
                        )
                    )
                    for feat in features
                ],
                axis=1,
            )
        )
        data = torch.permute(data, (0, 2, 1))
        all_data.append(data)
    data = torch.cat(all_data, dim=0)
    idx = torch.argsort(data[..., 0], dim=1, descending=True)
    data_pt_sorted = torch.gather(
        data, 1, idx.unsqueeze(-1).expand(-1, -1, data.size(2))
    )
    data_pt_sorted = data_pt_sorted[:num_jets]
    return (
        data_pt_sorted[..., :3],
        data_pt_sorted[..., 3:-1],
        data_pt_sorted[..., -1].unsqueeze(-1).long(),
    )


def sample_noise(num_jets=100_000, max_num_particles=128):
    continuous = torch.randn((num_jets, max_num_particles, 3))
    flavor = np.random.choice(
        [0, 1, 2, 3, 4], size=(num_jets, max_num_particles), p=[0.2, 0.2, 0.2, 0.2, 0.2]
    )
    charge = np.random.choice([-1, 1], size=(num_jets, max_num_particles))
    charge[flavor == 0] = charge[flavor == 1] = 0
    flavor = F.one_hot(torch.tensor(flavor), num_classes=5)
    charge = torch.tensor(charge).unsqueeze(-1)
    discrete = torch.cat([flavor, charge], dim=-1).long()
    return continuous, discrete


def sample_masks(
    multiplicity_dist=None, num_jets=100_000, min_num_particles=0, max_num_particles=128
):
    """Sample masks from empirical multiplicity distribution `hist`."""

    if multiplicity_dist is None:
        return torch.ones((num_jets, max_num_particles, 1)).long()
    elif min_num_particles == max_num_particles:
        return torch.ones((num_jets, max_num_particles, 1)).long()
    else:
        hist_values, _ = np.histogram(
            multiplicity_dist, bins=np.arange(0, max_num_particles + 2, 1), density=True
        )
        probs = torch.tensor(hist_values, dtype=torch.float)
        cat = Categorical(probs)
        multiplicity = cat.sample((num_jets,))
        masks = torch.zeros((num_jets, max_num_particles))
        for i, n in enumerate(multiplicity):
            masks[i, :n] = 1
        return masks.unsqueeze(-1).long()


def map_basis_to_tokens(tensor):
    """
    Maps a tensor with shape (N, M, D) to a space of 8 tokens based on particle type and charge.

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


def physics_to_onehot(flavor_tensor, charge_tensor):
    """inputs:
        - flavor in one-hot (isPhoton, isNeutralHadron, isChargedHadron, isElectron, isMuon)
        - charge (-1, 0, +1)
    outputs:
        - 8-dim discrete state vector (isPhoton:0, isNeutralHadron:1, isNegHadron:2, isPosHadron:3, isElectron:4, isAntiElectron:5, isMuon:6, isAntiMuon:7)

    """
    neutrals = flavor_tensor[..., :2].clone()
    charged = flavor_tensor[..., 2:].clone() * charge_tensor.unsqueeze(-1)
    charged = charged.repeat_interleave(2, dim=-1)
    for idx in [0, 2, 4]:
        pos = charged[..., idx] == 1
        neg = charged[..., idx + 1] == -1
        charged[..., idx][pos] = 0
        charged[..., idx + 1][neg] = 0
        charged[..., idx][neg] = 1
    one_hot = torch.cat([neutrals, charged], dim=-1)
    return one_hot


def tokens_to_physics(states):
    """Get the (flavor, charge) representation of each particle:
    0 -> (0, 0)   photon
    1 -> (1, 0)   neutral hadron
    2 -> (2,-1)   charged hadron - negative
    3 -> (2, 1),  charged hadron - positive
    4 -> (3,-1),  electron
    5 -> (3, 1),  positron
    6 -> (4,-1)   muon
    7 -> (4, 1)   antimuon
    """
    flavor, charge = states.clone(), states.clone()

    flavor[flavor == 3] = 2
    for n in [4, 5]:
        flavor[flavor == n] = 3
    for n in [6, 7]:
        flavor[flavor == n] = 4

    for n in [0, 1]:
        charge[charge == n] = 0
    for n in [2, 4, 6]:
        charge[charge == n] = -1
    for n in [3, 5, 7]:
        charge[charge == n] = 1

    flavor = F.one_hot(flavor.squeeze(-1), num_classes=5)
    return flavor, charge
