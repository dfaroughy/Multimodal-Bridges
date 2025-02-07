import os
import pytest
import torch
from pipeline.helpers import SimpleLogger as log
from pipeline.configs import ExperimentConfigs
from tensorclass import TensorMultiModal
from datamodules.datasets import DataCoupling
from encoders.embedder import MultiModalEmbedder
from encoders.epic import MultiModalEPiC

log.warnings_off()

RESOURCE_PATH = "/home/df630/Multimodal-Bridges/tests/resources"
CONFIG_PATH = os.path.join(RESOURCE_PATH, "config_multi-modal.yaml")


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
        time=torch.randn(100,1,1),
        continuous=torch.randn(100, 128, 3),
        discrete=torch.randint(0, 8, (100, 128, 1)),
        mask=torch.ones(100, 128, 1),
    )


def test_multimodal_encoder_EPiC(dummy_batch, dummy_state):
    config = ExperimentConfigs(CONFIG_PATH)

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
