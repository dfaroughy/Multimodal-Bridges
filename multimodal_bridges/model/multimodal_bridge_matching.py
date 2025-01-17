import torch
import torch.nn as nn
import lightning as L
from typing import List, Tuple, Dict, Union

from pipeline.configs import ExperimentConfigs
from pipeline.registry import registered_models as Encoder
from pipeline.registry import registered_bridges as Bridge
from pipeline.registry import registered_optimizers as Optimizer
from pipeline.registry import registered_schedulers as Scheduler

from data.dataclasses import MultiModeState, DataCoupling
from encoders.embedder import MultiModalParticleCloudEmbedder


class MultiModalBridgeMatching(L.LightningModule):
    """Bridge-Matching model for multi-modal data"""

    def __init__(self, config: ExperimentConfigs):
        super().__init__()
        self.config = config
        self.embedder = MultiModalParticleCloudEmbedder(config)
        self.encoder = Encoder[config.encoder.name](config)

        if config.data.modality in ["continuous", "multi-modal"]:
            self.bridge_continuous = Bridge[config.model.bridge_continuous](config)
            self.loss_continuous_fn = nn.MSELoss(reduction="none")

        if config.data.modality in ["discrete", "multi-modal"]:
            self.vocab_size = config.data.vocab_size
            self.bridge_discrete = Bridge[config.model.bridge_discrete](config)
            self.loss_discrete_fn = nn.CrossEntropyLoss(reduction="none")

        self.loss_multimode = MultiModeLoss(mode=config.model.loss_weights)
        self.save_hyperparameters()

    def forward(self, state: MultiModeState, batch: DataCoupling) -> MultiModeState:
        h_local, h_global = self.embedder(state, batch.source, batch.context)
        return self.encoder(h_local, h_global)

    # ...Lightning functions

    def training_step(self, batch: DataCoupling, batch_idx) -> Dict[str, torch.Tensor]:
        state = self.sample_bridges(batch)
        state = state.to(self.device)
        heads = self.forward(state, batch)
        loss_0 = self.loss_continuous(heads, state, batch)
        loss_1 = self.loss_discrete(heads, state, batch)
        loss, loss_modes = self.loss_multimode([loss_0, loss_1])
        weights = self.loss_multimode.get_weights()
        return {
            "loss": loss,
            "train_loss_continuous": loss_modes[0],
            "train_loss_discrete": loss_modes[1],
            "weights_continuous": weights[0],
            "weights_discrete": weights[1],
        }

    def validation_step(
        self, batch: DataCoupling, batch_idx
    ) -> Dict[str, torch.Tensor]:
        state = self.sample_bridges(batch)
        state = state.to(self.device)
        heads = self.forward(state, batch)
        loss_0 = self.loss_continuous(heads, state, batch)
        loss_1 = self.loss_discrete(heads, state, batch)
        loss, loss_modes = self.loss_multimode([loss_0, loss_1])
        weights = self.loss_multimode.get_weights()
        return {
            "val_loss": loss,
            "val_loss_continuous": loss_modes[0],
            "val_loss_discrete": loss_modes[1],
            "weights_continuous": weights[0],
            "weights_discrete": weights[1],
        }

    def predict_step(
        self, batch: DataCoupling, batch_idx
    ) -> Tuple[MultiModeState, MultiModeState, MultiModeState]:
        source_state = MultiModeState(
            None, batch.source.continuous, batch.source.discrete, batch.source.mask
        )
        target_state = MultiModeState(
            None, batch.target.continuous, batch.target.discrete, batch.target.mask
        )
        initial_state = source_state.clone()
        final_state = self.simulate_dynamics(initial_state, batch)
        return final_state, source_state.detach().cpu(), target_state.detach().cpu()

    def configure_optimizers(self):
        optimizer = Optimizer[self.config.trainer.optimizer_name](
            self.parameters(), **self.config.trainer.optimizer_params
        )
        scheduler = Scheduler[self.config.trainer.scheduler_name](
            optimizer, **self.config.trainer.scheduler_params
        )
        return [optimizer], [scheduler]

    # ...Model functions

    def sample_bridges(self, batch: DataCoupling) -> MultiModeState:
        """sample stochastic bridges"""
        t = torch.rand(len(batch), device=self.device)
        time = self.reshape_time_dim_like(t, batch)
        continuous = (
            self.bridge_continuous.sample(time, batch)
            if hasattr(self, "bridge_continuous")
            else None
        )
        discrete = (
            self.bridge_discrete.sample(time, batch)
            if hasattr(self, "bridge_discrete")
            else None
        )
        mask = batch.target.mask
        return MultiModeState(time, continuous, discrete, mask)

    def loss_continuous(
        self, heads: MultiModeState, state: MultiModeState, batch: DataCoupling
    ) -> torch.Tensor:
        """mean square error loss for drift matching"""
        if "continuous" in heads.available_modes():
            vector = heads.continuous
            targets = self.bridge_continuous.drift(state, batch).to(self.device)
            loss_mse = self.loss_continuous_fn(vector, targets) * state.mask
            return loss_mse.sum() / state.mask.sum()
        return torch.tensor(0.0, device=self.device)

    def loss_discrete(
        self, heads: MultiModeState, state: MultiModeState, batch: DataCoupling
    ) -> torch.Tensor:
        """cross-entropy loss for discrete state classifier"""
        if "discrete" in heads.available_modes():
            logits = heads.discrete.reshape(-1, self.vocab_size)
            targets = batch.target.discrete.reshape(-1).long()
            targets = targets.to(self.device)
            mask = state.mask.reshape(-1)
            loss_ce = self.loss_discrete_fn(logits, targets) * mask
            return loss_ce.sum() / mask.sum()
        return torch.tensor(0.0, device=self.device)

    def simulate_dynamics(
        self, state: MultiModeState, batch: DataCoupling
    ) -> MultiModeState:
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
            state.time = torch.full((len(batch), 1), time.item(), device=self.device)
            heads = self.forward(state, batch)
            if "continuous" in heads.available_modes():
                state = self.bridge_continuous.solver_step(state, heads, delta_t)
            if "discrete" in heads.available_modes():
                state = self.bridge_discrete.solver_step(state, heads, delta_t)
        return state.detach().cpu()

    def reshape_time_dim_like(self, t, state: Union[MultiModeState, DataCoupling]):
        if isinstance(t, (float, int)):
            return t
        else:
            return t.reshape(-1, *([1] * (state.ndim - 1)))


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
