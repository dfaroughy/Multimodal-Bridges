import pytest
import torch
from tensorclass import TensorMultiModal

def test_multimodal_state():
    state = TensorMultiModal(
        time=torch.randn(100),
        continuous=torch.randn(100, 128, 3),
        discrete=torch.randint(0, 8, (100, 128, 2)),
        mask=torch.cat([torch.ones(100, 64, 1), torch.zeros(100, 64, 1)], dim=1),
    )

    assert state.available_modes() == ["time", "continuous", "discrete"]
    assert state.ndim == 3
    assert state.shape == (100, 128)
    assert len(state) == 100


if __name__ == "__main__":
    test_multimodal_state()
