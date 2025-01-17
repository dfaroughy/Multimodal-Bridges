import os
import pytest
import torch
from pipeline.helpers import SimpleLogger as log
from pipeline.configs import ExperimentConfigs
from data.dataclasses import MultiModeState, DataCoupling
from encoders.embedder import MultiModalParticleCloudEmbedder

log.warnings_off()

RESOURCE_PATH = "/home/df630/Multimodal-Bridges/tests/resources"
CONFIG_PATH = os.path.join(RESOURCE_PATH, "config_model_continuous.yaml")


@pytest.fixture
def dummy_batch():
    source = MultiModeState(
        continuous=torch.randn(100, 128, 3),
        mask=torch.ones(100, 128, 1),
    )
    target = MultiModeState(
        continuous=torch.randn(100, 128, 3),
        mask=torch.ones(100, 128, 1),
    )
    context = MultiModeState(
        continuous=torch.randn(100, 5),
        discrete=torch.randint(0, 7, (100, 4)),
    )
    return DataCoupling(source, target, context)

@pytest.fixture
def dummy_state():
    return MultiModeState(
        time=torch.randn(100, 1, 1),
        continuous=torch.randn(100, 128, 3),
        mask=torch.ones(100, 128, 1),
    )


def test_continuous_embedder(dummy_batch, dummy_state):
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
        config.encoder.dim_emb_continuous,
    )
    assert state_glob.continuous.shape == (
        config.data.num_jets,
        config.encoder.dim_emb_context_continuous,
    )
    assert state_glob.discrete.shape == (
        config.data.num_jets,
        config.data.dim_context_discrete * config.encoder.dim_emb_context_discrete,
    )
