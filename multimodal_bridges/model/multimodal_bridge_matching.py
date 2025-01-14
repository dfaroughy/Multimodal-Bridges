import torch
import torch.nn as nn
import lightning as L
from dataclasses import dataclass
from typing import List, Tuple, Dict
import h5py

from utils.configs import ExperimentConfigs
from utils.registry import registered_models as Encoder
from utils.registry import registered_bridges as Bridge
from utils.registry import registered_optimizers as Optimizer
from utils.registry import registered_schedulers as Scheduler

from data.states import HybridState
from data.datasets import DataBatch

from encoders.embedder import MultiModalParticleCloudEmbedder


class MultiModalBridgeMatching(L.LightningModule):
    """Bridge-Matching model for hybrid data"""

    def __init__(self, config: ExperimentConfigs):
        super().__init__()
        self.config = config
        self.vocab_size = config.data.vocab_size
        self.weight = getattr(config.model, "loss_weights", "fixed")
        self.embedder = MultiModalParticleCloudEmbedder(config)
        self.encoder = Encoder[config.encoder.name](config)

        if hasattr(config.model, "bridge_continuous"):
            self.bridge_continuous = Bridge[config.model.bridge_continuous](config)
            self.loss_continuous_fn = nn.MSELoss(reduction="none")

        if hasattr(config.model, "bridge_discrete"):
            self.bridge_discrete = Bridge[config.model.bridge_discrete](config)
            self.loss_discrete_fn = nn.CrossEntropyLoss(reduction="none")

        self.loss_multimode = MultiModeLoss(mode=self.weight)
        self.save_hyperparameters()

    def forward(self, state: HybridState, batch: DataBatch) -> HybridState:
        h_local, h_global = self.embedder(state, batch.source, batch.context)
        return self.encoder(h_local, h_global)

    # ...Lightning functions

    def training_step(self, batch: DataBatch, batch_idx) -> Dict[str, torch.Tensor]:
        state = self.sample_bridges(batch)
        state = state.to(self.device)
        heads = self.forward(state, batch)
        loss_0 = self.loss_continuous(heads, state, batch)
        loss_1 = self.loss_discrete(heads, state, batch)
        loss, loss_individual = self.loss_multimode([loss_0, loss_1])
        weights = self.loss_multimode.get_weights()
        return {
            "loss": loss,
            "train_loss_continuous": loss_individual[0],
            "train_loss_discrete": loss_individual[1],
            "weights_continuous": weights[0],
            "weights_discrete": weights[1],
        }

    def validation_step(self, batch: DataBatch, batch_idx) -> Dict[str, torch.Tensor]:
        state = self.sample_bridges(batch)
        state = state.to(self.device)
        heads = self.forward(state, batch)
        loss_0 = self.loss_continuous(heads, state, batch)
        loss_1 = self.loss_discrete(heads, state, batch)
        loss, loss_individual = self.loss_multimode([loss_0, loss_1])
        weights = self.loss_multimode.get_weights()
        return {
            "val_loss": loss,
            "val_loss_continuous": loss_individual[0],
            "val_loss_discrete": loss_individual[1],
            "weights_continuous": weights[0],
            "weights_discrete": weights[1],
        }

    def predict_step(
        self, batch: DataBatch, batch_idx
    ) -> Tuple[HybridState, HybridState, HybridState]:
        source_state = HybridState(
            None, batch.source.continuous, batch.source.discrete, batch.source.mask
        )
        target_state = HybridState(
            None, batch.target.continuous, batch.target.discrete, batch.target.mask
        )
        initial_state = source_state.clone()
        final_state = self.simulate_dynamics(initial_state, batch)
        return final_state, source_state.detach().cpu(), target_state.detach().cpu()

    def configure_optimizers(self):
        name = self.config.trainer.optimizer_name
        params = self.config.trainer.optimizer_params
        optimizer = Optimizer[name](self.parameters(), **params.to_dict())
        name = self.config.trainer.scheduler_name
        params = self.config.trainer.scheduler_params
        scheduler = Scheduler[name](optimizer, **params.to_dict())
        return [optimizer], [scheduler]

    # ...Model functions

    def sample_bridges(self, batch: DataBatch) -> HybridState:
        """sample stochastic bridges"""
        continuous, discrete = None, None
        t = torch.rand(batch.target.continuous.shape[0], device=self.device).type_as(
            batch.target.continuous
        )

        time = self.reshape_time(t, batch.target.continuous)
        if hasattr(self, "bridge_continuous"):
            continuous = self.bridge_continuous.sample(
                time, batch.source.continuous, batch.target.continuous
            )
        if hasattr(self, "bridge_discrete"):
            discrete = self.bridge_discrete.sample(
                time, batch.source.discrete, batch.target.discrete
            )
        mask = batch.target.mask
        return HybridState(time, continuous, discrete, mask)

    def loss_continuous(
        self, heads: HybridState, state: HybridState, batch: DataBatch
    ) -> torch.Tensor:
        """mean square error loss for drift matching"""
        if hasattr(self, "bridge_continuous"):
            vector = heads.continuous
            targets = self.bridge_continuous.drift(
                t=state.time,
                x=state.continuous,
                x0=batch.source.continuous,
                x1=batch.target.continuous,
            ).to(self.device)
            loss_mse = self.loss_continuous_fn(vector, targets) * state.mask
            return loss_mse.sum() / state.mask.sum()
        else:
            return torch.tensor(0.0, device=self.device)

    def loss_discrete(
        self, heads: HybridState, state: HybridState, batch: DataBatch
    ) -> torch.Tensor:
        """cross-entropy loss for discrete state classifier"""
        if hasattr(self, "bridge_discrete"):
            logits = heads.discrete.reshape(-1, self.vocab_size)
            targets = batch.target.discrete.reshape(-1).long()
            targets = targets.to(self.device)
            mask = state.mask.reshape(-1)
            loss_ce = self.loss_discrete_fn(logits, targets) * mask
            return loss_ce.sum() / mask.sum()
        else:
            return torch.tensor(0.0, device=self.device)

    def simulate_dynamics(self, state: HybridState, batch: DataBatch) -> HybridState:
        """generate target data from source input using trained dynamics
        returns the final state of the bridge at the end of the time interval
        """
        time_steps = torch.linspace(
            0.0,
            1.0 - self.config.model.time_eps,
            self.config.model.num_timesteps,
            device=self.device,
        )
        delta_t = (time_steps[-1] - time_steps[0]) / (len(time_steps) - 1)
        for time in time_steps[1:]:
            state.time = torch.full((len(batch[0]), 1), time.item(), device=self.device)
            heads = self.forward(state, batch)
            state = self.bridge_continuous.solver_step(state, heads, delta_t)
            state = self.bridge_discrete.solver_step(state, heads, delta_t)
        return state.detach().cpu()

    def reshape_time(self, t, x):
        if isinstance(t, (float, int)):
            return t
        else:
            return t.reshape(-1, *([1] * (x.dim() - 1)))


class MultiModeLoss(nn.Module):
    """
    Combines multiple losses with learnable weights.
    The weights are directly parameterized in the weight space (w = 1 / sigma^2).
    The combined loss includes additional log-weight terms for proper uncertainty weighting.
    """

    def __init__(self, mode=None, weights=None):
        super().__init__()
        self.mode = mode
        if mode == "learnable":
            self.weights = nn.Parameter(torch.tensor([0.0, 0.0]))
        elif mode == "fixed":
            self.weights = torch.tensor(weights if weights else [1.0, 1.0])

    def forward(self, losses) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        if self.mode == "learnable":
            combined_loss = sum(
                torch.exp(-self.weights[i]) * losses[i] + self.weights[i]
                for i in range(len(losses))
            )
        elif self.mode == "fixed":
            combined_loss = sum(self.weights[i] * losses[i] for i in range(len(losses)))
        return combined_loss, losses

    def get_weights(self) -> List[float]:
        if self.mode == "learnable":
            return [torch.exp(-weight).item() for weight in self.weights]
        elif self.mode == "fixed":
            return self.weights.tolist()
