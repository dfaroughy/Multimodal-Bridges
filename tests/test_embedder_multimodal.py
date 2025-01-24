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
CONFIG_PATH = os.path.join(RESOURCE_PATH, "config_model.yaml")


@pytest.fixture
def dummy_batch():
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
    context = TensorMultiModal(
        continuous=torch.randn(100, 5),
        discrete=torch.randint(0, 7, (100, 4)),
    )
    return DataCoupling(source, target, context)


@pytest.fixture
def dummy_state():
    return TensorMultiModal(
        time=torch.randn(100),
        continuous=torch.randn(100, 128, 3),
        discrete=torch.randint(0, 8, (100, 128, 1)),
        mask=torch.ones(100, 128, 1),
    )


def test_multimodal_embedder(dummy_batch, dummy_state):
    config = ExperimentConfigs(CONFIG_PATH)
    embedder = MultiModalEmbedder(config)
    state_loc, state_glob = embedder(dummy_state, dummy_batch)

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
    assert state_glob.continuous.shape == (
        config.data.num_jets,
        config.encoder.dim_emb_context_continuous,
    )
    assert state_glob.discrete.shape == (
        config.data.num_jets,
        config.data.dim_context_discrete * config.encoder.dim_emb_context_discrete,
    )
