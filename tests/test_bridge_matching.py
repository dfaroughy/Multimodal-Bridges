import os
import pytest
import torch
from pipeline.helpers import SimpleLogger as log
from pipeline.configs import ExperimentConfigs
from datamodules.datasets import DataCoupling
from tensorclass import TensorMultiModal
from model.multimodal_bridge_matching import MultiModalBridgeMatching

log.warnings_off()

RESOURCE_PATH = "/home/df630/Multimodal-Bridges/tests/resources"
CONFIG_PATH = os.path.join(RESOURCE_PATH, "config_multi-modal.yaml")

@pytest.fixture
def dummy_batch():
    source = TensorMultiModal()
    target = TensorMultiModal(
        continuous=torch.randn(100, 128, 3),
        discrete=torch.randint(0, 8, (100, 128, 1)),
        mask=torch.ones(100, 128, 1),
    )
    context = None
    return DataCoupling(source, target, context)


def test_multimodal_bridge_matching(dummy_batch):

    config = ExperimentConfigs(CONFIG_PATH)
    model = MultiModalBridgeMatching(config)
    state = model.sample_bridges(dummy_batch)
    print(state.time.shape)
    assert state.shape == dummy_batch.shape
    assert state.ndim == 3
    assert state.time.ndim == 3


