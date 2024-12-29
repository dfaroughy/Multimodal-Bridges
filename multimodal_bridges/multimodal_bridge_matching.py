import torch
import torch.nn as nn
import lightning as L
from dataclasses import dataclass
from typing import List

from utils.registry import registered_models as Model
from utils.registry import registered_bridges as Bridge
from utils.registry import registered_optimizers as Optimizer
from utils.registry import registered_schedulers as Scheduler


@dataclass
class BridgeState:
    """time-dependent bridge state (x_t,k_t)"""

    time: torch.Tensor = None
    continuous: torch.Tensor = None
    discrete: torch.Tensor = None
    mask: torch.Tensor = None

    def to(self, device):
        return BridgeState(
            time=self.time.to(device) if isinstance(self.time, torch.Tensor) else None,
            continuous=self.continuous.to(device)
            if isinstance(self.continuous, torch.Tensor)
            else None,
            discrete=self.discrete.to(device)
            if isinstance(self.discrete, torch.Tensor)
            else None,
            mask=self.mask.to(device) if isinstance(self.mask, torch.Tensor) else None,
        )

    @staticmethod
    def cat(states: List["BridgeState"], dim=0) -> "BridgeState":
        # function to concat list of states into a single state
        def cat_attr(attr_name):
            attrs = [getattr(s, attr_name) for s in states]
            if all(a is None for a in attrs):
                return None
            attrs = [a for a in attrs if a is not None]
            return torch.cat(attrs, dim=dim)

        return BridgeState(
            time=cat_attr("time"),
            continuous=cat_attr("continuous"),
            discrete=cat_attr("discrete"),
            mask=cat_attr("mask"),
        )


@dataclass
class OutputHeads:
    """model output heads"""

    continuous: torch.Tensor = None
    discrete: torch.Tensor = None


class MultiModalBridgeMatching(L.LightningModule):
    """Bridge-Matching model for hybrid data"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.vocab_size = config.data.vocab_size.features
        self.encoder = Model[config.model.name](config)

        if hasattr(config.dynamics, "bridge_continuous"):
            self.bridge_continuous = Bridge[config.dynamics.bridge_continuous](config)
            self.loss_continuous_fn = nn.MSELoss(reduction="none")

        if hasattr(config.dynamics, "bridge_discrete"):
            self.bridge_discrete = Bridge[config.dynamics.bridge_discrete](config)
            self.loss_discrete_fn = nn.CrossEntropyLoss(reduction="none")

        self.save_hyperparameters()

    def forward(self, state: BridgeState, batch):
        continuous, discrete = self.encoder(
            t=state.time,
            x=state.continuous,
            k=state.discrete,
            mask=state.mask,
            context_continuous=getattr(batch, "context_continuous", None),
            context_discrete=getattr(batch, "context_discrete", None),
        )
        return OutputHeads(continuous, discrete)

    def sample_bridges(self, batch):
        """sample stochastic bridges"""
        t = torch.rand(
            batch.target_continuous.shape[0], device=batch.target_continuous.device
        ).type_as(batch.target_continuous)

        time = self.reshape_time(t, batch.target_continuous)

        continuous = self.bridge_continuous.sample(
            time, batch.source_continuous, batch.target_continuous
        )
        discrete = self.bridge_discrete.sample(
            time, batch.source_discrete, batch.target_discrete
        )
        mask = batch.target_mask
        return BridgeState(time, continuous, discrete, mask)

    def loss_continuous(self, heads: OutputHeads, state: BridgeState, batch):
        """mean square error loss for drift matching"""
        if hasattr(self, "bridge_continuous"):
            vector = heads.continuous
            targets = self.bridge_continuous.drift(
                t=state.time,
                x=state.continuous,
                x0=batch.source_continuous,
                x1=batch.target_continuous,
            ).to(vector.device)
            loss_mse = self.loss_continuous_fn(vector, targets) * state.mask
            return loss_mse.sum() / state.mask.sum()
        else:
            return torch.tensor(0.0, device=heads.discrete.device)

    def loss_discrete(self, heads: OutputHeads, state: BridgeState, batch):
        """cross-entropy loss for discrete state classifier"""
        if hasattr(self, "bridge_discrete"):
            logits = heads.discrete.reshape(-1, self.vocab_size)
            targets = batch.target_discrete.reshape(-1).long()
            targets = targets.to(logits.device)
            mask = state.mask.reshape(-1)
            loss_ce = self.loss_discrete_fn(logits, targets) * mask
            return loss_ce.sum() / mask.sum()
        else:
            return torch.tensor(0.0, device=heads.continuous.device)

    def reshape_time(self, t, x):
        if isinstance(t, (float, int)):
            return t
        else:
            return t.reshape(-1, *([1] * (x.dim() - 1)))

    ###########################
    ### Lightning functions ###
    ###########################

    def training_step(self, batch, batch_idx):
        state = self.sample_bridges(batch)
        state = state.to(self.device)
        heads = self.forward(state, batch)
        loss_continous = self.loss_continuous(heads, state, batch)
        loss_discrete = self.loss_discrete(heads, state, batch)
        loss = loss_continous + loss_discrete
        self.log("train_loss", loss, on_epoch=True)
        self.log("train_loss_continuous", loss_continous, on_epoch=True)
        self.log("train_loss_discrete", loss_discrete, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        state = self.sample_bridges(batch)
        state = state.to(self.device)
        heads = self.forward(state, batch)
        loss_continous = self.loss_continuous(heads, state, batch)
        loss_discrete = self.loss_discrete(heads, state, batch)
        loss = loss_continous + loss_discrete
        self.log("val_loss", loss, on_epoch=True)
        self.log("val_loss_continuous", loss_continous, on_epoch=True)
        self.log("val_loss_discrete", loss_discrete, on_epoch=True)
        return loss

    def predict_step(self, batch, batch_idx):
        """generate target data from source data using trained dynamics"""
        time_steps = torch.linspace(
            0.0,
            1.0 - self.config.pipeline.time_eps,
            self.config.pipeline.num_timesteps,
            device=self.device,
        )
        delta_t = (time_steps[-1] - time_steps[0]) / (len(time_steps) - 1)
        delta_t = delta_t.to(self.device)
        state = BridgeState(
            None, batch.source_continuous, batch.source_discrete, batch.target_mask
        )
        state = state.to(self.device)
        for time in time_steps[1:]:
            state.time = torch.full((len(batch[0]), 1), time.item(), device=self.device)
            heads = self.forward(state, batch)
            state = self.bridge_continuous.solver_step(state, heads, delta_t)
            state = self.bridge_discrete.solver_step(state, heads, delta_t)
        return state

    def configure_optimizers(self):
        conf = self.config.train.optimizer
        optimizer = Optimizer[conf.name](self.parameters(), **conf.params.to_dict())
        conf = self.config.train.scheduler
        scheduler = Scheduler[conf.name](optimizer, **conf.params.to_dict())
        return [optimizer], [scheduler]
