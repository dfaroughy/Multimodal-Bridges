import torch
import os
import json

from torch.distributions.categorical import Categorical
from tensorclass import TensorMultiModal

class MultiModalNoise:
    def __init__(self, config):
        self.config = config

    def __call__(self, shape, device) -> TensorMultiModal:
        """
        Sample multi-modal random source:
        - time: None
        - continuous: N(0, 1) for each particle feature
        - discrete: random token in [0, vocab_size) for each particle
        - mask: number of active particles from a categorical distribution with `categorial_probs`

        """

        num_jets, max_num_particles = shape
        vocab_size = self.config.data.vocab_size
        metadata = self._load_metadata(self.config.data.target_path)
            
        if "categorical_probs" in metadata.keys():
            
            probs = metadata["categorical_probs"]
            cat = Categorical(torch.tensor(probs))
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
            discrete = (
                torch.randint(0, vocab_size, (num_jets, max_num_particles, 1)) * mask
            )
        else:
            discrete = None

        sample = TensorMultiModal(time, continuous, discrete, mask)
        return sample.to(device)

    def _load_metadata(self, path):
        metadata_file = os.path.join(path, "metadata.json")
        with open(metadata_file, "r") as f:
            metadata = json.load(f)
        return metadata
