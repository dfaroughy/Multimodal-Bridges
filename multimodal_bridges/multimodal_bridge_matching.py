import torch
import torch.nn as nn
import lightning as L
from dataclasses import dataclass
from typing import List
import h5py

from utils.registry import registered_models as Model
from utils.registry import registered_bridges as Bridge
from utils.registry import registered_optimizers as Optimizer
from utils.registry import registered_schedulers as Scheduler


@dataclass
class HybridState:
    """time-dependent hybrid bridge state (t, x, k, mask)"""

    time: torch.Tensor = None
    continuous: torch.Tensor = None
    discrete: torch.Tensor = None
    mask: torch.Tensor = None

    def to(self, device):
        return HybridState(
            time=self.time.to(device) if isinstance(self.time, torch.Tensor) else None,
            continuous=self.continuous.to(device)
            if isinstance(self.continuous, torch.Tensor)
            else None,
            discrete=self.discrete.to(device)
            if isinstance(self.discrete, torch.Tensor)
            else None,
            mask=self.mask.to(device) if isinstance(self.mask, torch.Tensor) else None,
        )

    def clone(self):
        return HybridState(
            time=self.time.clone() if isinstance(self.time, torch.Tensor) else None,
            continuous=self.continuous.clone()
            if isinstance(self.continuous, torch.Tensor)
            else None,
            discrete=self.discrete.clone()
            if isinstance(self.discrete, torch.Tensor)
            else None,
            mask=self.mask.clone() if isinstance(self.mask, torch.Tensor) else None,
        )

    def save_to(self, file_path):
        with h5py.File(file_path, "w") as f:
            if self.time is not None:
                f.create_dataset("time", data=self.time.cpu().numpy())
            if self.continuous is not None:
                f.create_dataset("continuous", data=self.continuous.cpu().numpy())
            if self.discrete is not None:
                f.create_dataset("discrete", data=self.discrete.cpu().numpy())
            if self.mask is not None:
                f.create_dataset("mask", data=self.mask.cpu().numpy())

    @staticmethod
    def cat(states: List["HybridState"], dim=0) -> "HybridState":
        # concat list of HybridState into a single HybridState
        def cat_attr(attr_name):
            attrs = [getattr(s, attr_name) for s in states]
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

    @staticmethod
    def load(file_path, device="cpu"):
        with h5py.File(file_path, "r") as f:
            time = torch.tensor(f["time"][:]) if "time" in f else None
            continuous = torch.tensor(f["continuous"][:]) if "continuous" in f else None
            discrete = torch.tensor(f["discrete"][:]) if "discrete" in f else None
            mask = torch.tensor(f["mask"][:]) if "mask" in f else None

        return HybridState(
            time=time.to(device) if time is not None else None,
            continuous=continuous.to(device) if continuous is not None else None,
            discrete=discrete.to(device) if discrete is not None else None,
            mask=mask.to(device) if mask is not None else None,
        )


@dataclass
class MultiHeadOutput:
    """model output heads"""

    continuous: torch.Tensor = None
    discrete: torch.Tensor = None


class MultiModalBridgeMatching(L.LightningModule):
    """Bridge-Matching model for hybrid data"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.vocab_size = config.data.vocab_size.features
        self.weight = getattr(config.dynamics.params, "loss_weight", "fixed")
        self.encoder = Model[config.model.name](config)

        if hasattr(config.dynamics, "bridge_continuous"):
            self.bridge_continuous = Bridge[config.dynamics.bridge_continuous](config)
            self.loss_continuous_fn = nn.MSELoss(reduction="none")

        if hasattr(config.dynamics, "bridge_discrete"):
            self.bridge_discrete = Bridge[config.dynamics.bridge_discrete](config)
            self.loss_discrete_fn = nn.CrossEntropyLoss(reduction="none")

        self.loss_multihead = MultiHeadLoss(mode=config.dynamics.params.loss_weights)
        self.save_hyperparameters()

    def forward(self, state: HybridState, batch):
        continuous, discrete = self.encoder(
            t=state.time,
            x=state.continuous,
            k=state.discrete,
            mask=state.mask,
            context_continuous=getattr(batch, "context_continuous", None),
            context_discrete=getattr(batch, "context_discrete", None),
        )
        return MultiHeadOutput(continuous, discrete)

    def sample_bridges(self, batch):
        """sample stochastic bridges"""
        continuous, discrete = None, None
        t = torch.rand(batch.target_continuous.shape[0], device=self.device).type_as(
            batch.target_continuous
        )

        time = self.reshape_time(t, batch.target_continuous)
        if hasattr(self, "bridge_continuous"):
            continuous = self.bridge_continuous.sample(
                time, batch.source_continuous, batch.target_continuous
            )
        if hasattr(self, "bridge_discrete"):
            discrete = self.bridge_discrete.sample(
                time, batch.source_discrete, batch.target_discrete
            )
        mask = batch.target_mask
        return HybridState(time, continuous, discrete, mask)

    def loss_continuous(self, heads: MultiHeadOutput, state: HybridState, batch):
        """mean square error loss for drift matching"""
        if hasattr(self, "bridge_continuous"):
            vector = heads.continuous
            targets = self.bridge_continuous.drift(
                t=state.time,
                x=state.continuous,
                x0=batch.source_continuous,
                x1=batch.target_continuous,
            ).to(self.device)
            loss_mse = self.loss_continuous_fn(vector, targets) * state.mask
            return loss_mse.sum() / state.mask.sum()
        else:
            return torch.tensor(0.0, device=self.device)

    def loss_discrete(self, heads: MultiHeadOutput, state: HybridState, batch):
        """cross-entropy loss for discrete state classifier"""
        if hasattr(self, "bridge_discrete"):
            logits = heads.discrete.reshape(-1, self.vocab_size)
            targets = batch.target_discrete.reshape(-1).long()
            targets = targets.to(self.device)
            mask = state.mask.reshape(-1)
            loss_ce = self.loss_discrete_fn(logits, targets) * mask
            return loss_ce.sum() / mask.sum()
        else:
            return torch.tensor(0.0, device=self.device)

    def simulate_dynamics(self, state: HybridState, batch):
        """generate target data from source input using trained dynamics
           returns the final state of the bridge at the end of the time interval
        """
        time_steps = torch.linspace(
            0.0,
            1.0 - self.config.pipeline.time_eps,
            self.config.pipeline.num_timesteps,
            device=self.device,
        )
        delta_t = (time_steps[-1] - time_steps[0]) / (len(time_steps) - 1)
        delta_t = delta_t.to(self.device)
        state = state.to(self.device)
        for time in time_steps[1:]:
            state.time = torch.full((len(batch[0]), 1), time.item(), device=self.device)
            heads = self.forward(state, batch)
            state = self.bridge_continuous.solver_step(state, heads, delta_t)
            state = self.bridge_discrete.solver_step(state, heads, delta_t)
        return state

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
        loss_0 = self.loss_continuous(heads, state, batch)
        loss_1 = self.loss_discrete(heads, state, batch)
        loss, loss_individual = self.loss_multihead([loss_0, loss_1])
        weights = self.loss_multihead.get_weights()
        return {"loss": loss, "loss_individual": loss_individual, "weights": weights}

    def validation_step(self, batch, batch_idx):
        state = self.sample_bridges(batch)
        state = state.to(self.device)
        heads = self.forward(state, batch)
        loss_0 = self.loss_continuous(heads, state, batch)
        loss_1 = self.loss_discrete(heads, state, batch)
        loss, loss_individual = self.loss_multihead([loss_0, loss_1])
        weights = self.loss_multihead.get_weights()
        return {"loss": loss, "loss_individual": loss_individual, "weights": weights}

    def predict_step(self, batch, batch_idx):
        source_state = HybridState(
            None, batch.source_continuous, batch.source_discrete, batch.source_mask
        )
        target_state = HybridState(
            None, batch.target_continuous, batch.target_discrete, batch.target_mask
        )
        initial_state = source_state.clone()
        final_state = self.simulate_dynamics(initial_state, batch)
        return final_state, source_state, target_state


    def configure_optimizers(self):
        conf = self.config.train.optimizer
        optimizer = Optimizer[conf.name](self.parameters(), **conf.params.to_dict())
        conf = self.config.train.scheduler
        scheduler = Scheduler[conf.name](optimizer, **conf.params.to_dict())
        return [optimizer], [scheduler]


class MultiHeadLoss(nn.Module):
    """
    Combines multiple losses with learnable weights.
    The weights are directly parameterized in the weight space (w = 1 / sigma^2).
    The combined loss includes additional log-weight terms for proper uncertainty weighting.
    """

    def __init__(self, weights=None, mode=None):
        super().__init__()
        self.mode = mode
        if mode == "learnable":
            self.weights = nn.Parameter(torch.tensor([0.0, 0.0]))
        elif mode == "fixed":
            self.weights = torch.tensor(weights if weights else [1.0, 1.0])

    def forward(self, losses):
        if self.mode == "learnable":
            combined_loss = sum(
                torch.exp(-self.weights[i]) * losses[i] + self.weights[i]
                for i in range(len(losses))
            )
            # individual_losses = [
            #     torch.exp(-self.weights[i]).item() * losses[i].item()
            #     for i in range(len(losses))
            # ]
        elif self.mode == "fixed":
            combined_loss = sum(self.weights[i] * losses[i] for i in range(len(losses)))
            # individual_losses = [
            #     self.weights[i] * losses[i].item() for i in range(len(losses))
            # ]
        return combined_loss, losses

    def get_weights(self):
        if self.mode == "learnable":
            return [torch.exp(-weight).item() for weight in self.weights]
        elif self.mode == "fixed":
            return self.weights.tolist()
