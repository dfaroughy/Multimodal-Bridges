import torch
import os
import json

from torch.distributions.categorical import Categorical
from tensorclass import TensorMultiModal


class MultiModalNoise:
    def __init__(self, config):
        self.config = config

        self.vocab_size = self.config.data.vocab_size
        self.dim_continuous = self.config.data.dim_continuous
        self.metadata = self._load_metadata(self.config.data.target_path)

    def __call__(self, shape, device, sample_masks=False) -> TensorMultiModal:
        """
        Sample multi-modal random source:
        - time: None
        - continuous: N(0, 1) for each particle feature
        - discrete: random token in [0, vocab_size) for each particle
        - mask: number of active particles from a categorical distribution with `categorial_probs`

        """

        num_jets, max_num_particles = shape

        if sample_masks:
            probs = self.metadata["categorical_probs"]
            cat = Categorical(torch.tensor(probs))
            multiplicity = cat.sample((num_jets,))
            mask = torch.zeros((num_jets, max_num_particles))
            
            for i, n in enumerate(multiplicity):
                mask[i, :n] = 1
                idx = torch.randperm(max_num_particles)
                mask[i] = mask[i][idx] 
                  
            mask = mask.long().unsqueeze(-1)
        else:
            mask = torch.ones((num_jets, max_num_particles, 1)).long()

        time = None
        continuous = torch.randn((num_jets, max_num_particles, self.dim_continuous))
        discrete = None

        if self.config.data.discrete_features == "tokens":
            discrete = torch.randint(
                0, self.vocab_size, (num_jets, max_num_particles, 1)
            )
            
        elif self.config.data.discrete_features == "onehot":
            continuous = torch.randn(
                (num_jets, max_num_particles, self.dim_continuous - self.vocab_size)
            )
            pids = torch.randint(0, self.vocab_size, (num_jets, max_num_particles))
            pids = torch.nn.functional.one_hot(pids, self.vocab_size).float()
            continuous = torch.cat((continuous, pids), dim=-1)

        sample = TensorMultiModal(time, continuous, discrete, mask)

        return sample.to(device)

    def _load_metadata(self, path):
        metadata_file = os.path.join(path, "metadata.json")
        with open(metadata_file, "r") as f:
            metadata = json.load(f)
        return metadata
