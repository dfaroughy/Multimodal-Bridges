import torch
import numpy as np
import awkward as ak
import vector
import uproot
import h5py
import yaml
import random
from datetime import datetime
from typing import Union
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder
from torch.distributions.categorical import Categorical
from torch.distributions.beta import Beta
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from torch.utils.data import Dataset
from collections import namedtuple

vector.register_awkward()

# ...DATASET & DATALOADER UTILS


class DataSetModule(Dataset):
    def __init__(self, data):
        self.data = data
        self.attributes = []

        # ...source

        if hasattr(self.data.source, "continuous"):
            self.attributes.append("source_continuous")
            self.source_continuous = self.data.source.continuous

        if hasattr(self.data.source, "discrete"):
            self.attributes.append("source_discrete")
            self.source_discrete = self.data.source.discrete

        if hasattr(self.data.source, "mask"):
            self.attributes.append("source_mask")
            self.source_mask = self.data.source.mask

        # ...target

        if hasattr(self.data.target, "continuous"):
            self.attributes.append("target_continuous")
            self.target_continuous = self.data.target.continuous

        if hasattr(self.data.target, "discrete"):
            self.attributes.append("target_discrete")
            self.target_discrete = self.data.target.discrete

        if hasattr(self.data.target, "mask"):
            self.attributes.append("target_mask")
            self.target_mask = self.data.target.mask

        # ...context

        if hasattr(self.data, "context_continuous"):
            self.attributes.append("context_continuous")
            self.context_continuous = self.data.context_continuous

        if hasattr(self.data, "context_discrete"):
            self.attributes.append("context_discrete")
            self.context_discrete = self.data.context_discrete

        self.databatch = namedtuple("databatch", self.attributes)

    def __getitem__(self, idx):
        return self.databatch(*[getattr(self, attr)[idx] for attr in self.attributes])

    def __len__(self):
        return len(self.data.target)

    def __iter__(self):
        for idx in range(len(self)):
            yield self[idx]


class DataloaderModule:
    def __init__(
        self, config, dataclass, batch_size: int = None, data_split_frac: tuple = None
    ):
        self.dataclass = dataclass
        self.config = config
        self.dataset = DataSetModule(dataclass)
        self.data_split = (
            self.config.train.data_split_frac
            if data_split_frac is None
            else data_split_frac
        )
        self.batch_size = (
            self.config.train.batch_size if batch_size is None else batch_size
        )
        self.dataloader()

    def train_val_test_split(self, shuffle=False):
        assert (
            np.abs(1.0 - sum(self.data_split)) < 1e-3
        ), "Split fractions do not sum to 1!"
        total_size = len(self.dataset)
        train_size = int(total_size * self.data_split[0])
        valid_size = int(total_size * self.data_split[1])

        # ...define splitting indices

        idx = torch.randperm(total_size) if shuffle else torch.arange(total_size)
        idx_train = idx[:train_size].tolist()
        idx_valid = idx[train_size : train_size + valid_size].tolist()
        idx_test = idx[train_size + valid_size :].tolist()

        # ...Create Subset for each split

        train_set = Subset(self.dataset, idx_train)
        valid_set = Subset(self.dataset, idx_valid) if valid_size > 0 else None
        test_set = Subset(self.dataset, idx_test) if self.data_split[2] > 0 else None

        return train_set, valid_set, test_set

    def dataloader(self):
        print("INFO: building dataloaders...")
        print(
            "INFO: train/val/test split ratios: {}/{}/{}".format(
                self.data_split[0], self.data_split[1], self.data_split[2]
            )
        )

        train, valid, test = self.train_val_test_split(shuffle=False)
        self.train = DataLoader(dataset=train, batch_size=self.batch_size, shuffle=True)
        self.valid = (
            DataLoader(dataset=valid, batch_size=self.batch_size, shuffle=False)
            if valid is not None
            else None
        )
        self.test = (
            DataLoader(dataset=test, batch_size=self.batch_size, shuffle=False)
            if test is not None
            else None
        )

        print(
            "INFO: train size: {}, validation size: {}, testing sizes: {}".format(
                len(self.train.dataset),  # type: ignore
                len(self.valid.dataset if valid is not None else []),  # type: ignore
                len(self.test.dataset if test is not None else []),
            )
        )  # type: ignore


# ...JET DATA UTILS


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


def extract_jetclass_features(dataset, **args):
    max_num_particles = args.get("max_num_particles", 128)
    min_num_particles = args.get("min_num_particles", 0)
    num_jets = args.get("num_jets", 100_000)

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


def extract_aoj_features(dataset, **args):
    max_num_particles = args.get("max_num_particles", 150)
    min_num_particles = args.get("min_num_particles", 0)
    num_jets = args.get("num_jets", 100_000)

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


def sample_noise(noise="GaussNoise", **args):
    max_num_particles = args.get("max_num_particles", 128)
    num_jets = args.get("num_jets", 100_000)

    scale = args.get("scale", 1.0)
    cat_probs = args.get("cat_probs", [0.2, 0.2, 0.2, 0.2, 0.2])

    if noise == "BetaNoise":
        concentration = args.get("concentration", [0.1, 10])
        a, b = torch.tensor(concentration[0]), torch.tensor(concentration[1])
        pt = Beta(a, b).sample((num_jets, max_num_particles, 1))
        eta_phi = torch.randn((num_jets, max_num_particles, 2)) * scale
        continuous = torch.cat([pt, eta_phi], dim=2)
    elif noise == "GaussNoise":
        continuous = torch.randn((num_jets, max_num_particles, 3)) * scale
    else:
        raise ValueError(
            'Noise type not recognized. Choose between "GaussNoise" and "BetaNoise".'
        )

    # idx = torch.argsort(continuous[..., 0], dim=1, descending=True)
    # continuous = torch.gather(
    #     continuous, 1, idx.unsqueeze(-1).expand(-1, -1, continuous.size(2))
    # )
    flavor = np.random.choice(
        [0, 1, 2, 3, 4], size=(num_jets, max_num_particles), p=cat_probs
    )
    charge = np.random.choice([-1, 1], size=(num_jets, max_num_particles))
    charge[flavor == 0] = charge[flavor == 1] = 0
    flavor = F.one_hot(torch.tensor(flavor), num_classes=5)
    charge = torch.tensor(charge).unsqueeze(-1)
    discrete = torch.cat([flavor, charge], dim=-1)
    discrete = torch.tensor(discrete).long()
    return continuous, discrete


def sample_masks(**args):
    """Sample masks from empirical multiplicity distribution `hist`."""
    hist = args.get("set_masks_like", None)
    max_num_particles = args.get("max_num_particles", 128)
    num_jets = args.get("num_jets", 100_000)

    if hist is None:
        return torch.ones((num_jets, max_num_particles, 1)).long()
    else:
        hist_values, bin_edges = np.histogram(
            hist, bins=np.arange(0, max_num_particles + 2, 1), density=True
        )
        bin_edges[0] = np.floor(
            bin_edges[0]
        )  # Ensure lower bin includes the floor of the lowest value
        bin_edges[-1] = np.ceil(
            bin_edges[-1]
        )  # Extend the upper bin edge to capture all values

        h = torch.tensor(hist_values, dtype=torch.float)
        probs = h / h.sum()
        cat = Categorical(probs)
        multiplicity = cat.sample((num_jets,))

        # Initialize masks and apply the sampled multiplicities
        masks = torch.zeros((num_jets, max_num_particles))
        for i, n in enumerate(multiplicity):
            masks[i, :n] = 1
        return masks.unsqueeze(-1).long()


def flavor_to_onehot(flavor_tensor, charge_tensor):
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


def states_to_flavor(states):
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


# ...CONFIG UTILS


class Configs:
    def __init__(self, config_source):
        if isinstance(config_source, str):
            with open(config_source, "r") as f:
                config_dict = yaml.safe_load(f)

        elif isinstance(config_source, dict):
            config_dict = config_source
        else:
            raise ValueError("config_source must be a file path or a dictionary")

        self._set_attributes(config_dict)  # set attributes recursively

        if hasattr(self, "experiment"):
            if not hasattr(self.experiment, "type"):
                if hasattr(self.dynamics, "discrete") and not hasattr(
                    self.dynamics, "continuous"
                ):
                    self.experiment.type = "discrete"
                    assert self.data.dim.features_discrete > 0
                    self.data.dim.features_continuous = 0

                elif hasattr(self.dynamics, "continuous") and not hasattr(
                    self.dynamics, "discrete"
                ):
                    assert self.data.dim.features_continuous > 0
                    self.experiment.type = "continuous"
                    self.data.dim.features_discrete = 0

                else:
                    self.experiment.type = "multimodal"
                    assert (
                        self.data.dim.features_continuous > 0
                        and self.data.dim.features_discrete > 0
                    )

            if not hasattr(self.experiment, "name"):
                self.experiment.name = (
                    f"{self.data.source.name}_to_{self.data.target.name}"
                )

                if hasattr(self.dynamics, "continuous"):
                    self.experiment.name += f"_{self.dynamics.continuous.bridge}"

                if hasattr(self.dynamics, "discrete"):
                    self.experiment.name += f"_{self.dynamics.discrete.bridge}"

                self.experiment.name += f"_{self.model.name}"

                time = datetime.now().strftime("%Y.%m.%d_%Hh%M")
                rnd = random.randint(0, 10000)
                self.experiment.name += f"_{time}_{rnd}"
                print(
                    "INFO: created experiment instance {}".format(self.experiment.name)
                )

            if self.experiment.type == "classifier":
                if len(self.data.train.path) > 1:
                    self.experiment.name = "multi_model"
                else:
                    self.experiment.name = "binary"

    def _set_attributes(self, config_dict):
        for key, value in config_dict.items():
            if isinstance(value, dict):  # create a sub-config object
                sub_config = Configs(value)
                setattr(self, key, sub_config)
            else:
                setattr(self, key, value)

    def to_dict(self):
        """
        Recursively converts the Configs object into a dictionary.
        """
        config_dict = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Configs):
                config_dict[key] = value.to_dict()
            else:
                config_dict[key] = value
        return config_dict

    def print(self):
        """
        Prints the configuration parameters in a structured format.
        """
        config_dict = self.to_dict()
        self._print_dict(config_dict)

    def _print_dict(self, config_dict, indent=0):
        """
        Helper method to recursively print the config dictionary.
        """
        for key, value in config_dict.items():
            prefix = " " * indent
            if isinstance(value, dict):
                print(f"{prefix}{key}:")
                self._print_dict(value, indent + 4)
            else:
                print(f"{prefix}{key}: {value}")

    def log_config(self, logger):
        """
        Logs the configuration parameters using the provided logger.
        """
        config_dict = self.to_dict()
        self._log_dict(config_dict, logger)

    def _log_dict(self, config_dict, logger, indent=0):
        """
        Helper method to recursively log the config dictionary.
        """
        for key, value in config_dict.items():
            prefix = " " * indent
            if isinstance(value, dict):
                logger.logfile.info(f"{prefix}{key}:")
                self._log_dict(value, logger, indent + 4)
            else:
                logger.logfile.info(f"{prefix}{key}: {value}")

    def save(self, path):
        """
        Saves the configuration parameters to a YAML file.
        """
        config_dict = self.to_dict()
        with open(path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False)
