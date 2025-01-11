import os
import pytest
import torch
from collections import namedtuple

from utils.configs import ExperimentConfigs
from states import HybridState
from encoders.embed import MultiModalPointCloudEmbedder
from encoders.particle_transformer import MultiModalParticleTransformer, ParticleAttentionBlock

RESOURCE_PATH = "/home/df630/Multimodal-Bridges/tests/resources"
CONFIG_PATH = os.path.join(RESOURCE_PATH, "config_model.yaml")


@pytest.fixture
def dummy_batch():
    DummyBatch = namedtuple(
        "DummyBatch",
        [
            "source_continuous",
            "source_discrete",
            "context_continuous",
            "context_discrete",
        ],
    )
    return DummyBatch(
        source_continuous=torch.randn(100, 128, 3),
        source_discrete=torch.randint(0, 8, (100, 128, 2)),
        context_continuous=torch.randn(100, 5),
        context_discrete=torch.randint(0, 7, (100, 4)),
    )


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
    embedder = MultiModalPointCloudEmbedder(config)
    state_loc, state_glob = embedder(dummy_state, dummy_batch)

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


def test_multimodal_encoder(dummy_batch, dummy_state):
    config = ExperimentConfigs(CONFIG_PATH)
    embedder = MultiModalPointCloudEmbedder(config)
    encoder = MultiModalParticleTransformer(config)
    assert encoder is not None

    state_loc, state_glob = embedder(dummy_state, dummy_batch)
    continuous, discrete = encoder(state_loc, state_glob)