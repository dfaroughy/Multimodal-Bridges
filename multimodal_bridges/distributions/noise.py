import torch
import os
import json

from torch.distributions.categorical import Categorical
from tensorclass import TensorMultiModal


class Noise:
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

        if self.config.data.discrete_features == "onehot":
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


class SourceUniform:
    def __init__(self, config):
        self.config = config

        self.vocab_size = self.config.data.vocab_size
        self.dim_continuous = self.config.data.dim_continuous
        self.metadata = self._load_metadata(self.config.data.target_path)

    def __call__(self, shape, device, sample_masks=False) -> TensorMultiModal:
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

        if self.config.data.modality == "discrete":
            continuous = None
            discrete = torch.randint(0, self.vocab_size, (num_jets, max_num_particles, 1))

        elif self.config.data.modality == "continuous":
            continuous = torch.randn((num_jets, max_num_particles, self.dim_continuous))
            discrete = None

        else:
            continuous = torch.randn((num_jets, max_num_particles, self.dim_continuous))
            discrete = torch.randint(0, self.vocab_size, (num_jets, max_num_particles, 1))

        sample = TensorMultiModal(None, continuous, discrete, mask)

        return sample.to(device)

    def _load_metadata(self, path):
        metadata_file = os.path.join(path, "metadata.json")
        with open(metadata_file, "r") as f:
            metadata = json.load(f)
        return metadata


class SourceDirichlet:
    def __init__(self, config):
        self.config = config

        self.vocab_size = self.config.data.vocab_size
        self.dim_continuous = self.config.data.dim_continuous
        self.metadata = self._load_metadata(self.config.data.target_path)

    def __call__(self, shape, device, sample_masks=False) -> TensorMultiModal:
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

        if self.config.data.modality == "continuous":
            discrete = None
            continuous = torch.distributions.dirichlet.Dirichlet(torch.ones(self.vocab_size)).sample((num_jets, max_num_particles))
        else:
            raise ValueError("Dirichlet source only supports continuous features")

        sample = TensorMultiModal(None, continuous, discrete, mask)

        return sample.to(device)

    def _load_metadata(self, path):
        metadata_file = os.path.join(path, "metadata.json")
        with open(metadata_file, "r") as f:
            metadata = json.load(f)
        return metadata



class SourceUniform:
    def __init__(self, config):
        self.config = config

        self.vocab_size = self.config.data.vocab_size
        self.dim_continuous = self.config.data.dim_continuous
        self.metadata = self._load_metadata(self.config.data.target_path)

    def __call__(self, shape, device, sample_masks=False) -> TensorMultiModal:
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

        if self.config.data.modality == "discrete":
            continuous = None
            discrete = torch.randint(0, self.vocab_size, (num_jets, max_num_particles, 1))

        elif self.config.data.modality == "continuous":
            continuous = torch.randn((num_jets, max_num_particles, self.dim_continuous))
            discrete = None
        else:
            continuous = torch.randn((num_jets, max_num_particles, self.dim_continuous))
            discrete = torch.randint(0, self.vocab_size, (num_jets, max_num_particles, 1))

        sample = TensorMultiModal(None, continuous, discrete, mask)

        return sample.to(device)

    def _load_metadata(self, path):
        metadata_file = os.path.join(path, "metadata.json")
        with open(metadata_file, "r") as f:
            metadata = json.load(f)
        return metadata







class SourceDegenerate:
    def __init__(self, config):
        self.config = config

        self.vocab_size = self.config.data.vocab_size
        self.dim_continuous = self.config.data.dim_continuous
        self.metadata = self._load_metadata(self.config.data.target_path)

    def __call__(self, shape, device, sample_masks=False) -> TensorMultiModal:
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

        if self.config.data.modality == "discrete":
            continuous = None
            discrete = torch.zeros((num_jets, max_num_particles, 1), dtype=torch.long)

        elif self.config.data.modality == "continuous":
            continuous = torch.randn((num_jets, max_num_particles, self.dim_continuous))
            discrete = None
            
        else:
            continuous = torch.randn((num_jets, max_num_particles, self.dim_continuous))
            discrete = torch.zeros((num_jets, max_num_particles, 1), dtype=torch.long)

        sample = TensorMultiModal(None, continuous, discrete, mask)

        return sample.to(device)

    def _load_metadata(self, path):
        metadata_file = os.path.join(path, "metadata.json")
        with open(metadata_file, "r") as f:
            metadata = json.load(f)
        return metadata


class SourceMasked:
    def __init__(self, config):
        self.config = config
        self.vocab_size = self.config.data.vocab_size
        self.dim_continuous = self.config.data.dim_continuous
        self.metadata = self._load_metadata(self.config.data.target_path)

    def __call__(self, shape, device, sample_masks=False, inference_mode=False) -> TensorMultiModal:
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

        if self.config.data.modality == "discrete":
            continuous = None
            discrete = -1 * torch.ones((num_jets, max_num_particles, 1))
            if inference_mode:
                discrete = torch.zeros((num_jets, max_num_particles, 1))

        elif self.config.data.modality == "continuous":
            continuous = torch.randn((num_jets, max_num_particles, self.dim_continuous))
            discrete = None
            
        else:
            continuous = torch.randn((num_jets, max_num_particles, self.dim_continuous))
            discrete = -1 * torch.ones((num_jets, max_num_particles, 1))

        sample = TensorMultiModal(None, continuous, discrete.long(), mask)

        return sample.to(device)

    def _load_metadata(self, path):
        metadata_file = os.path.join(path, "metadata.json")
        with open(metadata_file, "r") as f:
            metadata = json.load(f)
        return metadata


class SourceDataDriven:
    def __init__(self, config):
        self.config = config

        self.vocab_size = self.config.data.vocab_size
        self.dim_continuous = self.config.data.dim_continuous
        self.metadata = self._load_metadata(self.config.data.target_path)

    def __call__(self, shape, device, sample_masks=False) -> TensorMultiModal:
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

        datadriven_probs = [
            0.304,
            0.0847,
            0.2986,
            0.3078,
            0.0013,
            0.0014,
            0.0011,
            0.001,
        ]

        if self.config.data.modality == "discrete":
            continuous = None
            cat = Categorical(torch.tensor(datadriven_probs),)
            discrete = cat.sample((num_jets, max_num_particles, 1))

        elif self.config.data.modality == "continuous":
            continuous = torch.randn((num_jets, max_num_particles, self.dim_continuous))
            discrete = None
            
        else:
            continuous = torch.randn((num_jets, max_num_particles, self.dim_continuous))
            cat = Categorical(torch.tensor(datadriven_probs),)
            discrete = cat.sample((num_jets, max_num_particles, 1))


        sample = TensorMultiModal(None, continuous, discrete, mask)

        return sample.to(device)

    def _load_metadata(self, path):
        metadata_file = os.path.join(path, "metadata.json")
        with open(metadata_file, "r") as f:
            metadata = json.load(f)
        return metadata
