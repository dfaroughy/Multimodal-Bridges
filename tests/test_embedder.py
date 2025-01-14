import os
import pytest
import torch
from utils.configs import ExperimentConfigs
from data.states import HybridState
from data.datasets import DataBatch
from encoders.embedder import MultiModalParticleCloudEmbedder
from utils.helpers import SimpleLogger as log
log.warnings_off()

RESOURCE_PATH = "/home/df630/Multimodal-Bridges/tests/resources"
CONFIG_PATH = os.path.join(RESOURCE_PATH, "config_model.yaml")


@pytest.fixture
def dummy_batch():

    source = HybridState(
        time=None,
        continuous=torch.randn(100, 128, 3),
        discrete=torch.randint(0, 8, (100, 128, 2)),
        mask=torch.ones(100, 128, 1),
    )
    target = HybridState(
        time=None,
        continuous=torch.randn(100, 128, 3),
        discrete=torch.randint(0, 8, (100, 128, 2)),
        mask=torch.ones(100, 128, 1),
    )
    context = HybridState(
        time=None,
        continuous=torch.randn(100, 5),
        discrete=torch.randint(0, 7, (100, 4)),
        mask=None,
    )
    return DataBatch(source, target, context)

@pytest.fixture
def dummy_state():
    return HybridState(
        time=torch.randn(100, 1),
        continuous=torch.randn(100, 128, 3),
        discrete=torch.randint(0, 8, (100, 128, 2)),
        mask=torch.ones(100, 128, 1),
    )


def test_multimodal_embedder(dummy_batch, dummy_state):
    config = ExperimentConfigs(CONFIG_PATH)
    embedder = MultiModalParticleCloudEmbedder(config)
    state_loc, state_glob = embedder(dummy_state, dummy_batch.source, dummy_batch.context)

    assert state_loc.time.shape == (
        config.data.num_jets,
        config.data.max_num_particles,
        config.encoder.dim_emb_time,
    )
    assert state_loc.continuous.shape == (
        config.data.num_jets,
        config.data.max_num_particles,
        config.encoder.dim_emb_continuous + config.encoder.dim_emb_augment_continuous,
    )
    assert state_loc.discrete.shape == (
        config.data.num_jets,
        config.data.max_num_particles,
        config.data.dim_discrete
        * (config.encoder.dim_emb_discrete + config.encoder.dim_emb_augment_discrete),
    )
    assert state_glob.continuous.shape == (
        config.data.num_jets,
        config.encoder.dim_emb_context_continuous,
    )
    assert state_glob.discrete.shape == (
        config.data.num_jets,
        config.data.dim_context_discrete * config.encoder.dim_emb_context_discrete,
    )
