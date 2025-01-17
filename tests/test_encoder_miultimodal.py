import os
import pytest
import torch
from utils.configs import ExperimentConfigs
from data.dataclasses import MultiModeState, DataCoupling
from encoders.embedder import MultiModalParticleCloudEmbedder
from encoders.particle_transformer import MultiModalParticleTransformer
from encoders.epic import MultiModalEPiC
from utils.helpers import SimpleLogger as log

log.warnings_off()

RESOURCE_PATH = "/home/df630/Multimodal-Bridges/tests/resources"
CONFIG_PATH = os.path.join(RESOURCE_PATH, "config_model.yaml")


@pytest.fixture
def dummy_batch():
    source = MultiModeState(
        continuous=torch.randn(100, 128, 3),
        discrete=torch.randint(0, 8, (100, 128, 1)),
        mask=torch.ones(100, 128, 1),
    )
    target = MultiModeState(
        continuous=torch.randn(100, 128, 3),
        discrete=torch.randint(0, 8, (100, 128, 1)),
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
        time=torch.randn(100,1,1),
        continuous=torch.randn(100, 128, 3),
        discrete=torch.randint(0, 8, (100, 128, 1)),
        mask=torch.ones(100, 128, 1),
    )


def test_multimodal_encoder_ParT(dummy_batch, dummy_state):
    config = ExperimentConfigs(CONFIG_PATH)
    config.encoder.num_heads = 2
    config.encoder.dim_hidden_continuous = 32
    config.encoder.dim_hidden_discrete = 16
    config.encoder.dropout = 0.2
    embedder = MultiModalParticleCloudEmbedder(config)
    encoder = MultiModalParticleTransformer(config)
    state_loc, state_glob = embedder(
        dummy_state, dummy_batch.source, dummy_batch.context
    )
    head = encoder(state_loc, state_glob)
    assert head.continuous is not None
    assert head.discrete is not None


def test_multimodal_encoder_EPiC(dummy_batch, dummy_state):
    config = ExperimentConfigs(CONFIG_PATH)
    embedder = MultiModalParticleCloudEmbedder(config)
    encoder = MultiModalEPiC(config)
    state_loc, state_glob = embedder(
        dummy_state, dummy_batch.source, dummy_batch.context
    )
    head = encoder(state_loc, state_glob)
    assert head.continuous is not None
    assert head.discrete is not None
