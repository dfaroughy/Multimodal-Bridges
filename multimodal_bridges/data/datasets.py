import torch
from collections import namedtuple
from torch.utils.data import Dataset
from data.states import HybridState


class HybridDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.attribute = []

        # ...source

        if hasattr(self.data.source, "continuous"):
            self.attribute.append("source_continuous")
            self.source_continuous = self.data.source.continuous

        if hasattr(self.data.source, "discrete"):
            self.attribute.append("source_discrete")
            self.source_discrete = self.data.source.discrete

        if hasattr(self.data.source, "mask"):
            self.attribute.append("source_mask")
            self.source_mask = self.data.source.mask

        # ...target

        if hasattr(self.data.target, "continuous"):
            self.attribute.append("target_continuous")
            self.target_continuous = self.data.target.continuous

        if hasattr(self.data.target, "discrete"):
            self.attribute.append("target_discrete")
            self.target_discrete = self.data.target.discrete

        if hasattr(self.data.target, "mask"):
            self.attribute.append("target_mask")
            self.target_mask = self.data.target.mask

        # ...context

        if hasattr(self.data, "context_continuous"):
            self.attribute.append("context_continuous")
            self.context_continuous = self.data.context_continuous

        if hasattr(self.data, "context_discrete"):
            self.attribute.append("context_discrete")
            self.context_discrete = self.data.context_discrete

        self.databatch = namedtuple("databatch", self.attribute)

    def __getitem__(self, idx):
        return self.databatch(*[getattr(self, attr)[idx] for attr in self.attribute])

    def __len__(self):
        return len(self.data.target)

    def __iter__(self):
        for idx in range(len(self)):
            yield self[idx]


DataBatch = namedtuple("DataBatch", ["source", "target", "context"])


def hybrid_collate_fn(batch):
    """Custom collate function for HybridState."""
    source = HybridState()
    target = HybridState()
    context = HybridState()

    source.continuous = (
        torch.stack([data.source_continuous for data in batch])
        if hasattr(batch[0], "source_continuous")
        else None
    )
    source.discrete = (
        torch.stack([data.source_discrete for data in batch])
        if hasattr(batch[0], "source_discrete")
        else None
    )
    source.mask = (
        torch.stack([data.source_mask for data in batch])
        if hasattr(batch[0], "source_mask")
        else None
    )
    target.continuous = (
        torch.stack([data.target_continuous for data in batch])
        if hasattr(batch[0], "target_continuous")
        else None
    )
    target.discrete = (
        torch.stack([data.target_discrete for data in batch])
        if hasattr(batch[0], "target_discrete")
        else None
    )
    target.mask = (
        torch.stack([data.target_mask for data in batch])
        if hasattr(batch[0], "target_mask")
        else None
    )
    context.continuous = (
        torch.stack([data.context_continuous for data in batch])
        if hasattr(batch[0], "context_continuous")
        else None
    )
    context.discrete = (
        torch.stack([data.context_discrete for data in batch])
        if hasattr(batch[0], "context_discrete")
        else None
    )

    return DataBatch(source=source, target=target, context=context)
