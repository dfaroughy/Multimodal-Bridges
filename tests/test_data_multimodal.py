import os
import pytest
import torch
import torch.nn.functional as F

from utils.helpers import get_from_json
from utils.helpers import SimpleLogger as log
log.warnings_off()

from utils.configs import ExperimentConfigs
from data.dataclasses import MultiModeState, DataCoupling
from data.particle_clouds.jets import JetDataModule
from data.particle_clouds.utils import (
    map_basis_to_tokens,
    map_basis_to_onehot,
    map_tokens_to_basis,
    map_onehot_to_basis,
)

RESOURCE_PATH = "/home/df630/Multimodal-Bridges/tests/resources"
OUTPUT_PATH = "/home/df630/Multimodal-Bridges/tests/output"
CONFIG_PATH = os.path.join(RESOURCE_PATH, "config.yaml")


def test_configs():
    config = ExperimentConfigs(CONFIG_PATH)
    assert config is not None


def test_data_shapes():
    config = ExperimentConfigs(CONFIG_PATH)
    jets = JetDataModule(config=config)
    jets.setup()
    N = config.data.num_jets
    M = config.data.max_num_particles
    D0 = config.data.dim_continuous
    D1 = config.data.dim_discrete
    assert jets.source.continuous.shape == torch.Size([N, M, D0])
    assert jets.source.discrete.shape == torch.Size([N, M, D1])
    assert jets.source.mask.shape == torch.Size([N, M, 1])
    assert jets.target.continuous.shape == torch.Size([N, M, D0])
    assert jets.target.discrete.shape == torch.Size([N, M, D1])
    assert jets.target.mask.shape == torch.Size([N, M, 1])


def test_databatch():
    config = ExperimentConfigs(CONFIG_PATH)
    jets = JetDataModule(config=config)

    jets.setup(stage="fit")
    train_dataloader = jets.train_dataloader()
    
    assert (
        len(train_dataloader)
        == jets.config.data.num_jets
        * config.data.split_ratios[0]
        // config.data.batch_size
    )

    for batch in train_dataloader:
        
        assert isinstance(batch.source, MultiModeState) 
        assert batch.source.ndim == 3
        
        assert isinstance(batch.target, MultiModeState) 
        assert batch.target.ndim == 3

        assert batch.source.continuous.shape[0] == config.data.batch_size
        assert batch.source.continuous.shape[1] == config.data.max_num_particles
        assert batch.source.continuous.shape[2] == config.data.dim_continuous
        assert batch.source.discrete.shape[2] == config.data.dim_discrete
        assert batch.source.mask.shape[2] == 1

        assert batch.target.continuous.shape[0] == config.data.batch_size
        assert batch.target.continuous.shape[1] == config.data.max_num_particles
        assert batch.target.continuous.shape[2] == config.data.dim_continuous
        assert batch.target.discrete.shape[2] == config.data.dim_discrete
        assert batch.target.mask.shape[2] == 1

    jets.config.data.num_jets = 90
    jets.setup(stage="predict")
    predict_dataloader = jets.predict_dataloader()

    assert (
        len(predict_dataloader)
        == jets.config.data.num_jets // config.data.batch_size
    )

    for batch in predict_dataloader:
        assert batch.source.continuous.shape[0] == config.data.batch_size
        assert batch.source.continuous.shape[1] == config.data.max_num_particles
        assert batch.source.continuous.shape[2] == config.data.dim_continuous
        assert batch.source.discrete.shape[2] == config.data.dim_discrete
        assert batch.source.mask.shape[2] == 1
        assert batch.target.continuous.shape[0] == config.data.batch_size
        assert batch.target.continuous.shape[1] == config.data.max_num_particles
        assert batch.target.continuous.shape[2] == config.data.dim_continuous
        assert batch.target.discrete.shape[2] == 6
        assert batch.target.mask.shape[2] == 1


def test_data_discrete_bases():
    input_tensor = torch.tensor(
        [
            [
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
            ],
            [
                [0, 0, 1, 0, 0, -1],
                [0, 0, 1, 0, 0, 1],
            ],
            [
                [0, 0, 0, 1, 0, -1],
                [0, 0, 0, 1, 0, 1],
            ],
            [
                [0, 0, 0, 0, 1, -1],
                [0, 0, 0, 0, 1, 1],
            ],
        ],
    ).long()

    test_tokens = torch.tensor([[[0], [1]], [[2], [3]], [[4], [5]], [[6], [7]]]).long()
    test_onehot = F.one_hot(test_tokens.squeeze(-1), num_classes=8)

    tokens = map_basis_to_tokens(input_tensor)
    onehot = map_basis_to_onehot(input_tensor)
    assert torch.equal(tokens, test_tokens)
    assert torch.equal(onehot, test_onehot)
    assert torch.equal(input_tensor, map_tokens_to_basis(tokens))
    assert torch.equal(input_tensor, map_onehot_to_basis(onehot))


def test_data_processing_closure():
    config = ExperimentConfigs(CONFIG_PATH)
    jets = JetDataModule(config=config, preprocess=False)
    jets_preprocessed = JetDataModule( config=config, metadata_path=OUTPUT_PATH)

    jets.setup()
    jets_preprocessed.setup()
    
    N = config.data.num_jets
    M = config.data.max_num_particles
    V = config.data.vocab_size

    assert jets_preprocessed.source.discrete.shape == torch.Size([N, M, 1])
    assert jets_preprocessed.target.discrete.shape == torch.Size([N, M, 1])
    assert torch.max(jets_preprocessed.source.discrete.squeeze(-1)).item() == V - 1
    assert torch.max(jets_preprocessed.target.discrete.squeeze(-1)).item() == V - 1

    # closure test:

    prep_continuous = config.data.target_preprocess_continuous
    prep_discrete = config.data.target_preprocess_discrete
    stats = get_from_json("target_data_stats", OUTPUT_PATH, "metadata.json")

    jets_preprocessed.target.postprocess(prep_continuous, prep_discrete, **stats)

    assert torch.equal(jets.target.discrete, jets_preprocessed.target.discrete)
    assert torch.allclose(jets.target.continuous, jets_preprocessed.target.continuous)


if __name__ == "__main__":
    test_data_processing_closure()