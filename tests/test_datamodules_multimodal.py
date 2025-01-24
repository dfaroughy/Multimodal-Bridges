import os
import pytest
import torch
import torch.nn.functional as F

from pipeline.helpers import get_from_json
from pipeline.helpers import SimpleLogger as log
from pipeline.configs import ExperimentConfigs
from tensorclass import TensorMultiModal

from datamodules.particle_clouds.jetmodule import JetDataModule
from datamodules.particle_clouds.utils import (
    map_basis_to_tokens,
    map_basis_to_onehot,
    map_tokens_to_basis,
    map_onehot_to_basis,
)

log.warnings_off()

RESOURCE_PATH = "/home/df630/Multimodal-Bridges/tests/resources"
OUTPUT_PATH = "/home/df630/Multimodal-Bridges/tests/output"
CONFIG_PATH = os.path.join(RESOURCE_PATH, "config_multi-modal.yaml")


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
        
        assert isinstance(batch.source, TensorMultiModal) 
        assert batch.source.ndim == 0
        
        assert isinstance(batch.target, TensorMultiModal) 
        assert batch.target.ndim == 3

        assert batch.source.continuous == None
        assert batch.source.continuous == None
        assert batch.source.continuous == None
        assert batch.source.discrete == None
        assert batch.source.mask == None

        assert batch.target.continuous.shape[0] == config.data.batch_size
        assert batch.target.continuous.shape[1] == config.data.max_num_particles
        assert batch.target.continuous.shape[2] == config.data.dim_continuous
        assert batch.target.discrete.shape[2] == config.data.dim_discrete
        assert batch.target.mask.shape[2] == 1

    jets.setup(stage="predict")
    predict_dataloader = jets.predict_dataloader()

    assert (
        len(predict_dataloader)
        == jets.config.data.num_jets // config.data.batch_size
    )

    for batch in predict_dataloader:
        assert batch.target.continuous.shape[0] == config.data.batch_size
        assert batch.target.continuous.shape[1] == config.data.max_num_particles
        assert batch.target.continuous.shape[2] == config.data.dim_continuous
        assert batch.target.discrete.shape[2] == config.data.dim_discrete
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

if __name__ == "__main__":
    test_data_processing_closure()
