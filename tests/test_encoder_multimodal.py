import os
import pytest
import torch
from pipeline.helpers import SimpleLogger as log
from pipeline.configs import ExperimentConfigs
from tensorclass import TensorMultiModal
from datamodules.datasets import DataCoupling
from encoders.embedder import MultiModalEmbedder
from encoders.multimodal_epic import UniModalEPiC, MultiModalEPiC

log.warnings_off()

RESOURCE_PATH = "/home/df630/Multimodal-Bridges/tests/resources"
CONFIG_PATH_DISCRETE = os.path.join(RESOURCE_PATH, "config_discrete.yaml")
CONFIG_PATH_DISCRETE_ONEHOT = os.path.join(RESOURCE_PATH, "config_discrete_onehot.yaml")
CONFIG_PATH_CONTINUOUS = os.path.join(RESOURCE_PATH, "config_continuous.yaml")
CONFIG_PATH_CONTINUOUS_ONEHOT = os.path.join(RESOURCE_PATH, "config_continuous_onehot.yaml")
CONFIG_PATH_MULTIMODAL = os.path.join(RESOURCE_PATH, "config_multi-modal.yaml")


#...Multimodal 

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
    context = None
    return DataCoupling(source, target, context)

@pytest.fixture
def dummy_state():
    return TensorMultiModal(
        time=torch.randn(100),
        continuous=torch.randn(100, 128, 3),
        discrete=torch.randint(0, 8, (100, 128, 1)),
        mask=torch.ones(100, 128, 1),
    )

#...TESTS

def test_EPiC_unimodal_discrete(dummy_batch, dummy_state):
    
    dummy_batch.source.continuous = None
    dummy_batch.target.continuous = None
    dummy_state.continuous = None

    config = ExperimentConfigs(CONFIG_PATH_DISCRETE)
    embedder = MultiModalEmbedder(config)
    state_loc, state_glob = embedder(dummy_state, dummy_batch)

    encoder = UniModalEPiC(config)
    head = encoder(state_loc, state_glob)

    N = config.data.num_jets
    D = config.data.max_num_particles
    dim_logits = config.data.dim_discrete * config.data.vocab_size

    assert head.continuous is None
    assert head.discrete.shape == (N, D, dim_logits)

def test_EPiC_unimodal_discrete_onehot(dummy_batch, dummy_state):
    
    dummy_batch.source.discrete = None
    dummy_batch.source.continuous = torch.randn(100, 128, 8)
    dummy_batch.target.discrete = None
    dummy_batch.target.continuous = torch.randn(100, 128, 8)
    dummy_state.discrete = None
    dummy_state.continuous = torch.randn(100, 128, 8)

    config = ExperimentConfigs(CONFIG_PATH_DISCRETE_ONEHOT)
    embedder = MultiModalEmbedder(config)
    state_loc, state_glob = embedder(dummy_state, dummy_batch)

    encoder = UniModalEPiC(config)
    head = encoder(state_loc, state_glob)

    N = config.data.num_jets
    D = config.data.max_num_particles
    V = config.data.vocab_size

    assert head.discrete is None
    assert head.continuous.shape == (N, D, V)



def test_EPiC_unimodal_continuous(dummy_batch, dummy_state):
    
    dummy_batch.source.discrete = None
    dummy_batch.target.discrete = None
    dummy_state.discrete = None

    config = ExperimentConfigs(CONFIG_PATH_CONTINUOUS)
    embedder = MultiModalEmbedder(config)
    state_loc, state_glob = embedder(dummy_state, dummy_batch)

    encoder = UniModalEPiC(config)
    head = encoder(state_loc, state_glob)

    N = config.data.num_jets
    D = config.data.max_num_particles

    assert head.continuous.shape == (N, D, config.data.dim_continuous)
    assert head.discrete is None


def test_EPiC_unimodal_continuous_onehot(dummy_batch, dummy_state):
    
    dummy_batch.source.discrete = None
    dummy_batch.source.continuous = torch.randn(100, 128, 8+3)
    dummy_batch.target.discrete = None
    dummy_batch.target.continuous = torch.randn(100, 128, 8+3)
    dummy_state.discrete = None
    dummy_state.continuous = torch.randn(100, 128, 8+3)

    config = ExperimentConfigs(CONFIG_PATH_CONTINUOUS_ONEHOT)
    embedder = MultiModalEmbedder(config)
    state_loc, state_glob = embedder(dummy_state, dummy_batch)

    encoder = UniModalEPiC(config)
    head = encoder(state_loc, state_glob)

    N = config.data.num_jets
    D = config.data.max_num_particles
    DIM = config.data.dim_continuous

    assert head.discrete is None
    assert head.continuous.shape == (N, D, DIM)



def test_multimodal_encoder_EPiC(dummy_batch, dummy_state):
    config = ExperimentConfigs(CONFIG_PATH_MULTIMODAL)

    embedder = MultiModalEmbedder(config)
    state_loc, state_glob = embedder(dummy_state, dummy_batch)

    encoder = MultiModalEPiC(config)
    head = encoder(state_loc, state_glob)

    N = config.data.num_jets
    D = config.data.max_num_particles
    dim_vector = config.data.dim_continuous
    dim_logits = config.data.dim_discrete * config.data.vocab_size

    assert head.continuous.shape == (N, D, dim_vector)
    assert head.discrete.shape == (N, D, dim_logits)


if __name__ == "__main__":

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

    state = TensorMultiModal(
        time=torch.randn(100,1,1),
        continuous=torch.randn(100, 128, 3),
        discrete=torch.randint(0, 8, (100, 128, 1)),
        mask=torch.ones(100, 128, 1),
    )

    batch = DataCoupling(source, target, None)

    test_multimodal_encoder_EPiC(batch, state)
