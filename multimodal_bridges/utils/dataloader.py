import torch
import numpy as np

from torch.utils.data import DataLoader, Subset
from torch.utils.data import Dataset
from collections import namedtuple

from utils.configs import ExperimentConfigs
from utils.misc import SimpleLogger as log

class DataSetModule(Dataset):
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


class DataloaderModule:
    def __init__(
        self,
        datamodule,
        config: ExperimentConfigs,
        batch_size: int = None,
        data_split_frac: tuple = None,
    ):
        self.datamodule = datamodule
        self.config = config
        self.dataset = DataSetModule(datamodule)
        self.data_split = (
            self.config.dataloader.data_split_frac
            if data_split_frac is None
            else data_split_frac
        )
        self.batch_size = (
            self.config.dataloader.batch_size if batch_size is None else batch_size
        )
        self.dataloader()

    def train_val_test_split(self, shuffle=False):
        assert np.abs(1.0 - sum(self.data_split)) < 1e-3, (
            "Split fractions do not sum to 1!"
        )
        total_size = len(self.dataset)
        train_size = int(total_size * self.data_split[0])
        valid_size = int(total_size * self.data_split[1])
        test_size = int(total_size * self.data_split[2])

        # ...define splitting indices

        idx = torch.randperm(total_size) if shuffle else torch.arange(total_size)
        idx_train = idx[:train_size].tolist()
        idx_valid = idx[train_size : train_size + valid_size].tolist()
        idx_test = idx[train_size + valid_size :].tolist()

        # ...Create Subset for each split

        train_set = Subset(self.dataset, idx_train) if train_size > 0 else None
        valid_set = Subset(self.dataset, idx_valid) if valid_size > 0 else None
        test_set = Subset(self.dataset, idx_test) if test_size > 0 else None

        return train_set, valid_set, test_set

    def dataloader(self):
        log.info("building dataloaders...")
        log.info(
            "train/val/test split ratios: {}/{}/{}".format(
                self.data_split[0], self.data_split[1], self.data_split[2]
            )
        )

        train, valid, test = self.train_val_test_split(shuffle=False)
        self.train = (
            DataLoader(dataset=train, batch_size=self.batch_size, shuffle=True)
            if train is not None
            else None
        )
        self.valid = (
            DataLoader(dataset=valid, batch_size=self.batch_size, shuffle=False)
            if valid is not None
            else None
        )
        self.test = (
            DataLoader(dataset=test, batch_size=self.batch_size, shuffle=False)
            if test is not None
            else None
        )

        log.info("train size: {}, validation size: {}, testing sizes: {}".format(
                len(self.train.dataset if train is not None else []),
                len(self.valid.dataset if valid is not None else []),
                len(self.test.dataset if test is not None else []),
            )
        )
