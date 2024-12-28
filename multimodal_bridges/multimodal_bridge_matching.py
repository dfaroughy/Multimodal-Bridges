import torch
import torch.nn as nn
import lightning as L
from dataclasses import dataclass
from typing import List

from models.epic import MultiModalEPiC
from bridges import LinearUniformBridge, TelegraphBridge


@dataclass
class BridgeState:
    time: torch.Tensor = None
    continuous: torch.Tensor = None
    discrete: torch.Tensor = None
    absorbing: torch.Tensor = None

    def to(self, device):
        return BridgeState(
            time=self.time.to(device) if isinstance(self.time, torch.Tensor) else None,
            continuous=self.continuous.to(device)
            if isinstance(self.continuous, torch.Tensor)
            else None,
            discrete=self.discrete.to(device)
            if isinstance(self.discrete, torch.Tensor)
            else None,
            absorbing=self.absorbing.to(device)
            if isinstance(self.absorbing, torch.Tensor)
            else None,
        )

    @staticmethod
    def cat(states: List["BridgeState"], dim=0) -> "BridgeState":
        # function to concat list of states int a single state
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
            absorbing=cat_attr("absorbing"),
        )


@dataclass
class OutputHeads:
    continuous: torch.Tensor = None
    discrete: torch.Tensor = None
    absorbing: torch.Tensor = None


class MultiModalBridgeMatching(L.LightningModule):
    """Model for hybrid data with varying size"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.vocab_size = config.data.vocab_size.features

        self.encoder = MultiModalEPiC(config)

        self.bridge_continuous = LinearUniformBridge(config)
        self.bridge_discrete = TelegraphBridge(config)
        self.bridge_absorbing = None  # implement absorbing bridge

        self.loss_continuous_fn = nn.MSELoss(reduction="none")
        self.loss_discrete_fn = nn.CrossEntropyLoss(reduction="none")
        self.loss_absorbing_fn = None # implement absorbing loss

        self.save_hyperparameters()

    def forward(self, state: BridgeState, batch):
        continuous, discrete, absorbing = self.encoder(
            t=state.time,
            x=state.continuous,
            k=state.discrete,
            mask=state.absorbing,
            context_continuous=getattr(batch, "context_continuous", None),
            context_discrete=getattr(batch, "context_discrete", None),
        )
        return OutputHeads(continuous, discrete, absorbing)

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

        absorbing = batch.target_mask  # replace with absorbing bridge when implemented

        return BridgeState(time, continuous, discrete, absorbing)

    def loss_continuous(self, heads: OutputHeads, state: BridgeState, batch):
        """mean square error loss for velocity field"""
        vector = heads.continuous
        mask = heads.absorbing

        ut = self.bridge_continuous.drift(
            t=state.time,
            x=state.continuous,
            x0=batch.source_continuous,
            x1=batch.target_continuous,
        ).to(vector.device)
        loss_mse = self.loss_continuous_fn(vector, ut) * mask
        return loss_mse.sum() / mask.sum()

    def loss_discrete(self, heads: OutputHeads, batch):
        """cross-entropy loss for discrete state classifier"""
        logits = heads.discrete
        targets = batch.target_discrete
        mask = heads.absorbing
        logits = heads.discrete.reshape(-1, self.vocab_size)
        targets = batch.target_discrete.reshape(-1).long()
        targets = targets.to(logits.device)
        mask = mask.reshape(-1)
        loss_ce = self.loss_discrete_fn(logits, targets) * mask
        return loss_ce.sum() / mask.sum()

    def loss_absorbing(self, heads: OutputHeads, batch):
        # TODO
        pass

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
        loss_discrete = self.loss_discrete(heads, batch)
        loss = loss_continous + loss_discrete
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        state = self.sample_bridges(batch)
        state = state.to(self.device)
        heads = self.forward(state, batch)
        loss_continous = self.loss_continuous(heads, state, batch)
        loss_discrete = self.loss_discrete(heads, batch)
        loss = loss_continous + loss_discrete
        self.log("val_loss", loss, on_step=True, on_epoch=True)
        return loss

    def predict_step(self, batch, batch_idx):
        """generates target data from source data using trained dynamics"""
        time_steps = torch.linspace(
            0.0,
            1.0 - self.config.pipeline.time_eps,
            self.config.pipeline.num_timesteps,
            device=self.device,
        )
        delta_t = (time_steps[-1] - time_steps[0]) / (len(time_steps) - 1)
        delta_t = delta_t.to(self.device)
        state = BridgeState(
            None,
            batch.source_continuous,
            batch.source_discrete,
            batch.source_mask,
        )
        state = state.to(self.device)
        for time in time_steps[1:]:
            state.time = torch.full((len(batch[0]), 1), time.item(), device=self.device)
            heads = self.forward(state, batch)
            state = self.bridge_continuous.solver_step(state, heads, delta_t)
            state = self.bridge_discrete.solver_step(state, heads, delta_t)
        return state

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.config.train.optimizer.params.lr
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.config.train.scheduler.params.T_max,
            eta_min=self.config.train.scheduler.params.eta_min,
            last_epoch=self.config.train.scheduler.params.last_epoch,
        )
        return [optimizer], [scheduler]
