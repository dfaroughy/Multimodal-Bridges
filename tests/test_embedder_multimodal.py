import os
import pytest
import torch

from pipeline.helpers import SimpleLogger as log
from pipeline.configs import ExperimentConfigs
from tensorclass import TensorMultiModal
from datamodules.datasets import DataCoupling
from encoders.embedder import MultiModalEmbedder

log.warnings_off()

RESOURCE_PATH = "/home/df630/Multimodal-Bridges/tests/resources"
CONFIG_PATH_MULTIMODAL = os.path.join(RESOURCE_PATH, "config_multi-modal.yaml")
CONFIG_PATH_DISCRETE = os.path.join(RESOURCE_PATH, "config_discrete.yaml")

#...Multimodal 

@pytest.fixture
def dummy_batch_multimodal():
    source = TensorMultiModal(
        continuous=torch.randn(100, 128, 3),
        discrete=torch.randint(0, 8, (100, 128, 1)),
        mask=torch.ones(100, 128, 1),
    )
    target = TensorMultiModal(
        continuous=torch.randn(100, 128, 3),
        discrete=torch.randint(0, 8, (100, 128, 1)),
        mask=torch.ones(100, 128, 1),
    )
    context = None
    return DataCoupling(source, target, context)

@pytest.fixture
def dummy_state_multimodal():
    return TensorMultiModal(
        time=torch.randn(100),
        continuous=torch.randn(100, 128, 3),
        discrete=torch.randint(0, 8, (100, 128, 1)),
        mask=torch.ones(100, 128, 1),
    )

#...Discrete 

@pytest.fixture
def dummy_batch_discrete():
    source = TensorMultiModal(
        discrete=torch.randint(0, 8, (100, 128, 1)),
        mask=torch.ones(100, 128, 1),
    )
    target = TensorMultiModal(
        discrete=torch.randint(0, 8, (100, 128, 1)),
        mask=torch.ones(100, 128, 1),
    )
    context = None
    return DataCoupling(source, target, context)


@pytest.fixture
def dummy_state_discrete():
    return TensorMultiModal(
        time=torch.randn(100),
        discrete=torch.randint(0, 8, (100, 128, 1)),
        mask=torch.ones(100, 128, 1),
    )


#...TESTS

def test_multimodal_embedder(dummy_batch_multimodal, dummy_state_multimodal):
    config = ExperimentConfigs(CONFIG_PATH_MULTIMODAL)
    embedder = MultiModalEmbedder(config)
    state_loc, state_glob = embedder(dummy_state_multimodal, dummy_batch_multimodal)

    aug_factor = 2 if config.encoder.data_augmentation else 1

    assert state_loc.time.shape == (
        config.data.num_jets,
        config.data.max_num_particles,
        config.encoder.dim_emb_time,
    )
    assert state_loc.continuous.shape == (
        config.data.num_jets,
        config.data.max_num_particles,
        config.encoder.dim_emb_continuous,
    )
    assert state_loc.discrete.shape == (
        config.data.num_jets,
        config.data.max_num_particles,
        config.data.dim_discrete * config.encoder.dim_emb_discrete * aug_factor,
    )


def test_discrete_embedder(dummy_batch_discrete, dummy_state_discrete):
    config = ExperimentConfigs(CONFIG_PATH_DISCRETE)
    embedder = MultiModalEmbedder(config)
    state_loc, state_glob = embedder(dummy_state_discrete, dummy_batch_discrete)

    aug_factor = 2 if config.encoder.data_augmentation else 1

    assert state_loc.time.shape == (
        config.data.num_jets,
        config.data.max_num_particles,
        config.encoder.dim_emb_time,
    )
    assert state_loc.continuous == None

    assert state_loc.discrete.shape == (
        config.data.num_jets,
        config.data.max_num_particles,
        config.data.dim_discrete * config.encoder.dim_emb_discrete * aug_factor,
    )
