import torch
from dataclasses import dataclass
from typing import List

""" Data classes for multi-modal states and data couplings. 
"""


@dataclass
class MultiModeState:
    time: torch.Tensor = None
    continuous: torch.Tensor = None
    discrete: torch.Tensor = None
    mask: torch.Tensor = None

    # Methods to manipulate the state

    def to(self, device: str):
        return self._apply_fn(lambda tensor: tensor.to(device))

    def detach(self):
        return self._apply_fn(lambda tensor: tensor.detach())

    def cpu(self):
        return self._apply_fn(lambda tensor: tensor.cpu())

    def clone(self):
        return self._apply_fn(lambda tensor: tensor.clone())

    def apply_mask(self):
        self.time = self.time * self.mask
        self.continuous = self.continuous * self.mask
        self.discrete = (self.discrete * self.mask).long()

    @property
    def ndim(self):
        if not len(self.available_modes()):
            return 0
        return len(getattr(self, self.available_modes()[-1]).shape)

    @property
    def shape(self):
        if self.ndim > 0:
            return getattr(self, self.available_modes()[-1]).shape[:-1]
        return None

    def __len__(self):
        if self.ndim > 0:
            return len(getattr(self, self.available_modes()[-1]))
        return 0

    @staticmethod
    def cat(states: List["MultiModeState"], dim=0) -> "MultiModeState":
        def cat_attr(attr_name):
            attrs = [getattr(s, attr_name, None) for s in states]
            if all(a is None for a in attrs):
                return None
            attrs = [a for a in attrs if a is not None]
            return torch.cat(attrs, dim=dim)

        return MultiModeState(
            time=cat_attr("time"),
            continuous=cat_attr("continuous"),
            discrete=cat_attr("discrete"),
            mask=cat_attr("mask"),
        )

    def available_modes(
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

    def _apply_fn(self, func):
        """apply transformation function to all attributes."""
        return MultiModeState(
            time=func(self.time) if isinstance(self.time, torch.Tensor) else None,
            continuous=func(self.continuous)
            if isinstance(self.continuous, torch.Tensor)
            else None,
            discrete=func(self.discrete)
            if isinstance(self.discrete, torch.Tensor)
            else None,
            mask=func(self.mask) if isinstance(self.mask, torch.Tensor) else None,
        )


@dataclass
class DataCoupling:
    source: MultiModeState = None
    target: MultiModeState = None
    context: MultiModeState = None

    def __len__(self):
        return len(self.target)

    @property
    def ndim(self):
        return self.target.ndim

    @property
    def shape(self):
        return self.target.shape