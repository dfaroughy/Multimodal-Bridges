import torch
import torch.nn as nn
import lightning as L
from typing import List, Tuple, Dict, Union

from pipeline.configs import ExperimentConfigs

from pipeline.registry import registered_distributions as Distribution
from pipeline.registry import registered_models as Encoder
from pipeline.registry import registered_bridges as Bridge
from pipeline.registry import registered_optimizers as Optimizer
from pipeline.registry import registered_schedulers as Scheduler
from pipeline.registry import registered_thermostats as Thermostat

from tensorclass import TensorMultiModal
from datamodules.datasets import DataCoupling
from encoders.embedder import MultiModalEmbedder
from model.solvers import ContinuousSolver, DiscreteSolver


class MultiModalBridgeMatching(L.LightningModule):
    """Bridge-Matching model for multi-modal data"""

    def __init__(self, config: ExperimentConfigs):
        super().__init__()

        self.config = config
        self.embedder = MultiModalEmbedder(config)
        self.encoder = Encoder[config.encoder.name](config)

        if config.data.modality in ["continuous", "multi-modal"]:
            
            self.bridge_continuous = Bridge[config.model.bridge_continuous](
                sigma=config.model.sigma
            )
            
            self.loss_continuous_fn = nn.MSELoss(reduction="none")

        if config.data.modality in ["discrete", "multi-modal"]:
            
            self.vocab_size = config.data.vocab_size

            thermostat = Thermostat[config.model.thermostat_fn](
                    config.model.gamma, self.vocab_size
                )

            self.bridge_discrete = Bridge[config.model.bridge_discrete](
                gamma=config.model.gamma,
                vocab_size=self.vocab_size,
                thermostat_fn=thermostat,
            )

            freqs = (
                torch.tensor(config.data.vocab_freq) if config.data.vocab_freq else None
            )

            self.loss_discrete_fn = nn.CrossEntropyLoss(weight=freqs, reduction="none")

        self.loss_multimode = MultiModeLoss(weights=config.model.loss_weights)
        self.save_hyperparameters()
        self.path_snapshots_idx = config.model.save_path_history_snapshots

        # if source/target data not provided, sample from provided distributions:

        if not config.data.source_path:
            self.sample_source = Distribution[config.data.source_name](config)

        if not config.data.target_path:
            self.sample_target = Distribution[config.data.target_name](config)
            
    # ...Lightning functions

    def forward(self, state: TensorMultiModal, batch: DataCoupling) -> TensorMultiModal:
        h_local, h_global = self.embedder(state, batch)
        return self.encoder(h_local, h_global)

    def training_step(self, batch: DataCoupling, batch_idx) -> Dict[str, torch.Tensor]:
        
        state = self.sample_bridges(batch)
        state = state.to(self.device)

        heads = self.forward(state, batch)

        loss_continuous, loss_discrete = self.loss_fn(heads, state, batch)
        loss = self.loss_multimode([loss_continuous, loss_discrete])
        weights = self.loss_multimode.weight_vals()

        return {
            "loss": loss,
            "train_loss_continuous": loss_continuous,
            "train_loss_discrete": loss_discrete,
            "train_weights_continuous": weights[0],
            "train_weights_discrete": weights[1],
        }

    def validation_step(self, batch: DataCoupling, batch_idx) -> Dict[str, torch.Tensor]:
        
        state = self.sample_bridges(batch)
        state = state.to(self.device)

        heads = self.forward(state, batch)

        loss_continuous, loss_discrete = self.loss_fn(heads, state, batch)
        loss = self.loss_multimode([loss_continuous, loss_discrete])
        weights = self.loss_multimode.weight_vals()

        return {
            "val_loss": loss,
            "val_loss_continuous": loss_continuous,
            "val_loss_discrete": loss_discrete,
            "val_weights_continuous": weights[0],
            "val_weights_discrete": weights[1],
        }

    def predict_step(
        self, batch: DataCoupling, batch_idx
    ) -> Tuple[TensorMultiModal, TensorMultiModal, TensorMultiModal]:
        """generate target data from source by solving EOMs"""

        if not batch.has_source:
            batch.source = self.sample_source(
                shape=batch.target.shape, device=self.device, sample_masks=True
            )

        if not batch.has_target:
            batch.target = self.sample_target(
                shape=batch.target.shape, device=self.device
            )

        source_state = TensorMultiModal(
            torch.zeros_like(batch.source.mask),
            batch.source.continuous,
            batch.source.discrete,
            batch.source.mask,
        )  # t=0

        target_state = TensorMultiModal(
            torch.ones_like(batch.source.mask),
            batch.target.continuous,
            batch.target.discrete,
            batch.target.mask,
        )  # t=1

        paths = self.simulate_dynamics(source_state, batch)  # still preprocessed!

        return (
            paths.detach().cpu(),
            target_state.detach().cpu(),
        )

    def configure_optimizers(self):
        optimizer = Optimizer[self.config.trainer.optimizer_name](
            self.parameters(), **self.config.trainer.optimizer_params
        )
        if self.config.trainer.scheduler_name:
            scheduler = Scheduler[self.config.trainer.scheduler_name](
                optimizer, **self.config.trainer.scheduler_params
            )
            return [optimizer], [scheduler]
        else:
            return optimizer

    # ...Model functions

    def loss_fn(
        self, heads: TensorMultiModal, state: TensorMultiModal, batch: DataCoupling
    ) -> torch.Tensor:

        loss_continuous = torch.tensor(0.0, device=self.device)
        loss_discrete = torch.tensor(0.0, device=self.device)

        if heads.has_continuous:
            """mean square error loss for drift matching
            """

            vector = heads.continuous
            targets = self.bridge_continuous.drift(state, batch).to(self.device)
            loss_mse = self.loss_continuous_fn(vector, targets) 
            loss_mse *= state.mask
            loss_continuous = loss_mse.sum() / state.mask.sum()

        if heads.has_discrete:
            """cross-entropy loss for discrete state classifier
            """

            logits = heads.discrete.transpose(1, 2)
            targets = batch.target.discrete.squeeze(-1).to(self.device)
            loss_ce = self.loss_discrete_fn(logits, targets.long()).unsqueeze(-1)
            loss_ce *= state.mask
            loss_discrete = loss_ce.sum() / state.mask.sum()

        return loss_continuous, loss_discrete

    def sample_bridges(self, batch: DataCoupling) -> TensorMultiModal:
        """sample stochastic bridges"""

        continuous, discrete = None, None

        # sample time:

        eps = self.config.model.time_eps  # min time resolution
        t = eps + (1 - eps) * torch.rand(len(batch), device=self.device)
        time = self.reshape_time_dim_like(t, batch)

        # sample source/target data if necessary:
        if not batch.has_source:
            batch.source = self.sample_source(
                shape=batch.target.shape, device=self.device
            )

        if not batch.has_target:
            batch.target = self.sample_target(
                shape=batch.target.shape, device=self.device
            )

        # sample bridge paths:
        if batch.target.has_continuous:
            continuous = self.bridge_continuous.sample(time, batch)

        if batch.target.has_discrete:
            discrete = self.bridge_discrete.sample(time, batch)

        mask = batch.target.mask

        return TensorMultiModal(time, continuous, discrete, mask)

    def simulate_dynamics(
        self, state: TensorMultiModal, batch: DataCoupling
    ) -> TensorMultiModal:
        """generate target data from source input using trained dynamics
        returns the final state of the bridge at the end of the time interval
        """
        eps = self.config.model.time_eps  # min time resolution
        steps = self.config.model.num_timesteps
        time_steps = torch.linspace(eps, 1.0 - eps, steps, device=self.device)
        delta_t = (time_steps[-1] - time_steps[0]) / (len(time_steps) - 1)

        solver_continuous = ContinuousSolver(self.config, model=self)
        solver_discrete = DiscreteSolver(self.config, model=self)

        paths = [state.clone()]  # append t=0 source

        for i, t in enumerate(time_steps):
            state.time = torch.full((len(batch), 1), t.item(), device=self.device)
            state = solver_continuous.fwd_step(state, batch, delta_t)
            state, rates = solver_discrete.fwd_step(state, batch, delta_t)
            state.broadcast_time()  # (B,1) -> (B,D,1)

            if isinstance(self.path_snapshots_idx, list):
                for i in self.path_history_idx:
                    paths.append(state.clone())

        if state.has_discrete:
            max_rate = torch.max(rates, dim=2)[1]
            state.discrete = max_rate.unsqueeze(-1)

        paths.append(state)  # append t=1 generated target
        paths = TensorMultiModal.stack(paths, dim=0)
        return paths

    def reshape_time_dim_like(self, t, state: Union[TensorMultiModal, DataCoupling]):
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

    def __init__(self, weights=None):
        super().__init__()

        weights = [1.0, 1.0] if weights == "fixed" else weights

        if weights == "learnable":
            self.learnable = True
            self.loss_weights = nn.Parameter(torch.tensor([0.0, 0.0]))
        else:
            self.learnable = False
            self.loss_weights = torch.tensor(weights)

    def forward(self, losses) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        if self.learnable:
            combined_loss = sum(
                0.5 * torch.exp(-self.loss_weights[i]) * losses[i]
                + 0.5 * self.loss_weights[i]
                for i in range(len(losses))
            )

        else:
            combined_loss = sum(
                self.loss_weights[i] * losses[i] for i in range(len(losses))
            )

        return combined_loss

    def weight_vals(self) -> List[float]:
        if self.learnable:
            return [torch.exp(-weight).item() for weight in self.loss_weights]
        else:
            return self.loss_weights.tolist()
