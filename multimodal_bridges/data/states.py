import torch
from dataclasses import dataclass
from typing import List


@dataclass
class HybridState:
    """time-dependent hybrid state (t, x, k, mask)"""

    time: torch.Tensor = None
    continuous: torch.Tensor = None
    discrete: torch.Tensor = None
    mask: torch.Tensor = None

    def mask_modes(self):
        return self._apply(lambda tensor: tensor[self.mask.squeeze(-1) > 0])

    def to(self, device: str):
        return self._apply(lambda tensor: tensor.to(device))

    def detach(self):
        return self._apply(lambda tensor: tensor.detach())

    def cpu(self):
        return self._apply(lambda tensor: tensor.cpu())

    def clone(self):
        return self._apply(lambda tensor: tensor.clone())

    @property
    def device(self):
        return (
            self.continuous.device
            if self.continuous is not None
            else self.discrete.device
        )

    @staticmethod
    def cat(states: List["HybridState"], dim=0) -> "HybridState":
        def cat_attr(attr_name):
            attrs = [getattr(s, attr_name, None) for s in states]
            if all(a is None for a in attrs):
                return None
            attrs = [a for a in attrs if a is not None]
            return torch.cat(attrs, dim=dim)

        return HybridState(
            time=cat_attr("time"),
            continuous=cat_attr("continuous"),
            discrete=cat_attr("discrete"),
            mask=cat_attr("mask"),
        )

    def _apply(self, func):
        """apply transformation function to all attributes."""
        return HybridState(
            time=func(self.time) if isinstance(self.time, torch.Tensor) else None,
            continuous=func(self.continuous)
            if isinstance(self.continuous, torch.Tensor)
            else None,
            discrete=func(self.discrete)
            if isinstance(self.discrete, torch.Tensor)
            else None,
            mask=func(self.mask) if isinstance(self.mask, torch.Tensor) else None,
        )

    def modes(
        self,
    ) -> List[str]:
        """Return a list of non-None modes in the state."""
        available_modes = []
        if self.time is not None:
            available_modes.append("time")
        if self.continuous is not None:
            available_modes.append("continuous")
        if self.discrete is not None:
            available_modes.append("discrete")
        return available_modes

