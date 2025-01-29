import numpy as np
import torch
import h5py
import os
import urllib.request
import json
from pipeline.helpers import SimpleLogger as log
from tensorclass import TensorMultiModal


class AspenOpenJets:
    """data constructor for the Aspen OpenJets dataset.
    """
    
    def __init__(
        self,
        data_dir,
        data_files=None,
        url="https://www.fdr.uni-hamburg.de/record/16505/files",
    ):
        self.url = url
        self.data_dir = data_dir
        self.data_files = data_files

    def __call__(
        self,
        num_jets=None,
        max_num_particles=150,
        download=False,
        transform=None,
        features={"continuous": ["pt", "eta_rel", "phi_rel"], "discrete": ["tokens"]},
    ) -> TensorMultiModal:
        """
        Fetch the data, either from the provided files or by downloading it.
        Args:
            file_paths (list of str): List of absolute paths to the data files.
            download (bool): If True, downloads the file if not found.
        Returns:
            time, continuous, discrete, mask: Parsed data components.
        """

        if isinstance(self.data_files, str):
            self.data_files = [self.data_files]

        list_continuous_feats = []
        list_discrete_feats = []
        list_masks = []
        jet_count = 0

        for datafile in self.data_files:
            path = os.path.join(self.data_dir, datafile)

            if download:
                if not os.path.exists(path):
                    log.warn(f"File {datafile} not found. Downloading from {self.url}")
                    self._download_file(path)
                else:
                    log.info(f"File {datafile} already exists. Skipping download.")

            if not os.path.isfile(path):
                raise FileNotFoundError(
                    f"File {datafile} not found in {self.data_dir}."
                )

            feats, mask = self._read_aoj_file(path, num_jets)

            list_continuous_feats.append(
                torch.cat([feats[x] for x in features["continuous"]], dim=-1)
            )
            list_discrete_feats.append(torch.tensor(feats[features["discrete"]]))
            list_masks.append(mask)

            # halt if enough jets:
            if num_jets:
                jet_count += len(list_masks[-1])
                if jet_count > num_jets:
                    break

        continuous = torch.cat(list_continuous_feats, dim=0)[
            :num_jets, :max_num_particles, :
        ]
        discrete = torch.cat(list_discrete_feats, dim=0)[
            :num_jets, :max_num_particles, :
        ]
        mask = torch.cat(list_masks, dim=0)[:num_jets, :max_num_particles, :]

        metadata = self._extract_metadata(continuous, mask)

        if transform == "standardize":
            mean = torch.tensor(metadata["mean"])
            std = torch.tensor(metadata["std"])
            continuous = (continuous - mean) / std

        if transform == "normalize":
            min_val = torch.tensor(metadata["min"])
            max_val = torch.tensor(metadata["max"])
            continuous = (continuous - min_val) / (max_val - min_val)

        if transform == "log_pt":
            continuous[:, :, 0] = torch.log(continuous[:, :, 0] + 1e-6)

        # create a multimodal tensor output

        output = TensorMultiModal(None, continuous, discrete, mask)
        output.apply_mask()

        return output, metadata

    def _read_aoj_file(self, filepath, num_jets=None):
        """Reads and processes a single .h5 file from the AOJ dataset."""

        try:
            with h5py.File(filepath, "r") as f:
                PFCands = f["PFCands"][:num_jets] if num_jets else f["PFCands"][:]
        except (OSError, KeyError) as e:
            raise ValueError(f"Error reading file {filepath}: {e}")

        PFCands = self._filter_particles(PFCands)
        feats, mask = self._compute_continuous_coordinates(PFCands)
        feats["tokens"] = self._map_pid_to_tokens(PFCands[:, :, -2])[:, :, None]
        feats["onehot"] = np.eye(8)[feats["tokens"]].squeeze(2)
        mask = torch.tensor(mask[:, :, None], dtype=torch.long)

        for key in feats:
            feats[key] = torch.tensor(feats[key], dtype=torch.float32)

        return feats, mask

    def _download_file(self, target_file):
        """Download a file from a URL to a local path."""

        filename = os.path.basename(target_file)
        full_url = f"{self.url}/{filename}"  # Append filename to the base URL

        try:
            urllib.request.urlretrieve(full_url, target_file)
            log.info(f"Downloaded {target_file} successfully.")
        except Exception as e:
            raise RuntimeError(f"Failed to download file from {full_url}. Error: {e}")

    def _filter_particles(self, PFCands):
        """Filter and preprocess particle candidates."""

        mask_bad_pids = np.abs(PFCands[:, :, -2]) < 11
        PFCands[mask_bad_pids] = np.zeros_like(PFCands[mask_bad_pids])
        return PFCands

    def _map_pid_to_tokens(self, pid):
        """Map particle IDs to predefined values."""

        pid_map = {
            22: 0,  # Photon
            130: 1,  # Neutral Hadron
            -211: 2,  # Charged Hadron
            211: 3,  # Charged Hadron
            -11: 4,  # Electron
            11: 5,  # Positron
            -13: 6,  # Muon
            13: 7,  # Antimuon
        }
        pid = np.vectorize(lambda x: pid_map.get(x, 0))(pid)
        return pid

    def _compute_continuous_coordinates(self, PFCands):
        """Compute relative kinematic and spatial coordinates."""

        px, py, pz, e = PFCands[:, :, 0], PFCands[:, :, 1], PFCands[:, :, 2], PFCands[:, :, 3]
        pt = np.sqrt(px**2 + py**2)
        eta = np.arcsinh(np.divide(pz, pt, out=np.zeros_like(pz), where=pt != 0))
        phi = np.arctan2(py, px)

        jet = PFCands[:, :, :4].sum(axis=1)
        jet_eta = np.arcsinh(jet[:, 2] / np.sqrt(jet[:, 0] ** 2 + jet[:, 1] ** 2))
        jet_phi = np.arctan2(jet[:, 1], jet[:, 0])

        eta_rel = eta - jet_eta[:, None]
        phi_rel = (phi - jet_phi[:, None] + np.pi) % (2 * np.pi) - np.pi

        mask = PFCands[:, :, 3] > 0
        
        feats = {}
        feats['px'] = (px * mask)[:,:,None]
        feats['py'] = (py * mask)[:,:,None]
        feats['pz'] = (pz * mask)[:,:,None]
        feats['e'] = (e * mask)[:,:,None]
        feats["pt"] = (pt * mask)[:,:,None]
        feats["eta"] = (eta * mask)[:,:,None]
        feats["phi"] = (phi * mask)[:,:,None]
        feats["eta_rel"] = (eta_rel * mask)[:,:,None]
        feats["phi_rel"] = (phi_rel * mask)[:,:,None]
    
        return feats, mask

    def _load_metadata(self, path):
        metadata_file = os.path.join(path, "metadata.json")
        log.info(f"Loading metadata from {metadata_file}.")
        with open(metadata_file, "r") as f:
            metadata = json.load(f)
        return metadata

    def _extract_metadata(self, continuous, mask):
        mask_bool = mask.squeeze(-1) > 0
        nums = mask.sum(dim=1).squeeze()
        hist, _ = np.histogram(
            nums, bins=np.arange(0, continuous.shape[1] + 2, 1), density=True
        )
        return {
            "num_jets_sample": continuous.shape[0],
            "num_particles_sample": nums.sum().item(),
            "max_num_particles_per_jet": continuous.shape[1],
            "mean": continuous[mask_bool].mean(0).tolist(),
            "std": continuous[mask_bool].std(0).tolist(),
            "min": continuous[mask_bool].min(0).values.tolist(),
            "max": continuous[mask_bool].max(0).values.tolist(),
            "categorical_probs": hist.tolist(),
        }
