import torch
import numpy as np
import awkward as ak
import vector
import uproot

vector.register_awkward()

class JetClass:
    """
    JetClass dataset class.
    """
    def __init__(self, config):
        self.config = config

    def get_data(self, path_dir):

        num_jets = self.config.data.num_jets,
        min_num_particles = self.config.data.min_num_particles,
        max_num_particles = self.config.data.max_num_particles,

        if isinstance(path_dir, str):
            path_dir = [path_dir]
        
        all_data = []
        
        for data in path_dir:
            assert ".root" in data, "Input should be a path to a .root file"

            data = self._read_root_file(data)

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
        
        time = None
        continuous = data_pt_sorted[..., :3]
        discrete = map_basis_to_tokens(data_pt_sorted[..., 3:-1])
        mask = data_pt_sorted[..., -1].unsqueeze(-1).long()

        return time, continuous, discrete, mask

    def _read_root_file(self, filepath):
        """Loads a single .root file from the JetClass path."""
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