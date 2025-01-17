import os
import pytest
import torch
from utils.configs import ExperimentConfigs
from data.dataclasses import MultiModeState, DataCoupling
from model.multimodal_bridge_matching import MultiModalBridgeMatching
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


def test_multimodal_bridge_matching(dummy_batch):

    config = ExperimentConfigs(CONFIG_PATH)
    model = MultiModalBridgeMatching(config)
    state = model.sample_bridges(dummy_batch)
    print(state.time.shape)
    assert state.shape == dummy_batch.shape
    assert state.ndim == 3
    assert state.time.ndim == 3


if __name__ == "__main__":

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
    
    batch = DataCoupling(source, target, context)


    test_multimodal_bridge_matching(batch)
