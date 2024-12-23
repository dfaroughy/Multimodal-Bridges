import torch
import torch.nn as nn
import lightning as L
from dataclasses import dataclass
from torch.nn.functional import softmax
from torch.distributions import Categorical
from tqdm.auto import tqdm

from architecture import MultiModalEPiC
from jetdata import BridgeState, OutputHeads


class AbsorbingBridgeMatching(L.LightningModule):
    """Model for hybrid data with varying size"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.vocab_size = config.data.vocab_size.features

        self.encoder = MultiModalEPiC(config)

        self.bridge_continuous = LinearUniformBridge(config)
        self.bridge_discrete = TelegraphBridge(config)
        self.bridge_absorbing = None

        self.loss_continuous_fn = nn.MSELoss(reduction="none")
        self.loss_discrete_fn = nn.CrossEntropyLoss(reduction="none")
        self.loss_absorbing_fn = None

        self.save_hyperparameters()

    def forward(self, state, batch):
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
            T_max=self.config.train.scheduler.params.T_max,  # Adjust as needed
            eta_min=self.config.train.scheduler.params.eta_min,  # Adjust as needed
        )
        return [optimizer], [scheduler]


class LinearUniformBridge:
    """Conditional OT Flow-Matching for continuous states.
    This bridge is a linear interpolation between boundaries states at t=0 and t=1.
    notation:
      - t: time
      - x0: continuous source state at t=0
      - x1: continuous  target state at t=1
      - x: continuous state at time t
      - z: delta function regularizer
    """

    def __init__(self, config: dataclass):
        self.sigma = config.dynamics.params.sigma

    def sample(self, t, x0, x1):
        x = t * x1 + (1.0 - t) * x0
        z = torch.randn_like(x)
        std = self.sigma
        return x + std * z

    def drift(self, t, x, x0, x1):
        A = 0.0
        B = 1.0
        C = -1.0
        return A * x + B * x1 + C * x0

    def diffusion(self, t):
        return 0.0

    def solver_step(self, state, heads, delta_t):
        """Euler step for ODE solver"""
        state.continuous += delta_t * heads.continuous
        state.continuous *= heads.absorbing
        return state


class SchrodingerBridge:
    """Schrodinger bridge for continuous states
    notation:
      - t: time
      - x0: continuous source state at t=0
      - x1: continuous  target state at t=1
      - x: continuous state at time t
      - z: noise
    """

    def __init__(self, config: dataclass):
        self.sigma = config.dynamics.params.sigma

    def sample(self, t, x0, x1):
        x = t * x1 + (1.0 - t) * x0
        z = torch.randn_like(x)
        std = self.sigma * torch.sqrt(t * (1.0 - t))
        return x + std * z

    def drift(self, t, x, x0, x1):
        A = (1 - 2 * t) / (t * (1 - t))
        B = t**2 / (t * (1 - t))
        C = -1 * (1 - t) ** 2 / (t * (1 - t))
        return A * x + B * x1 + C * x0

    def diffusion(self, t):
        return self.sigma * torch.sqrt(t * (1.0 - t))

    def solver_step(self, state, heads, delta_t):
        """Euler-Maruyama step for SDE solver"""
        diffusion = self.diffusion(delta_t)
        delta_w = torch.randn_like(state.continuous)
        state.continuous += delta_t * state.continuous + diffusion * delta_w
        state.continuous *= heads.absorbing
        return state


class TelegraphBridge:
    """Multivariate Telegraph bridge for discrete states
    - t: time
    - k0: discrete source state at t=0
    - k1: discrete  target state at t=1
    - k: discrete state at time t
    """

    def __init__(self, config: dataclass):
        self.gamma = config.dynamics.params.gamma
        self.time_epsilon = config.pipeline.time_eps
        self.vocab_size = config.data.vocab_size.features

    def sample(self, t, k0, k1):
        transition_probs = self.transition_probability(t, k0, k1)
        state = Categorical(transition_probs).sample().to(k1.device)
        if state.dim() == 2:
            state = state.unsqueeze(-1)
        return state

    def rate(self, t, k, logits):
        """t: (b, 1) time tensor
        k: (b, n, 1) current state tensor
        logits: (b, n, vocab_size) logits tensor
        """
        assert (k >= 0).all() and (
            k < self.vocab_size
        ).all(), "Values in `k` outside of bound! k_min={}, k_max={}".format(
            k.min(), k.max()
        )

        qx = softmax(
            logits, dim=2
        )  # softmax to get the transition probabilities for all states
        qy = torch.gather(
            qx, 2, k.long()
        )  # get probabilities for the current state `k`

        # ...Telegraph process rates:

        S = self.vocab_size
        t, t1 = t.squeeze(), 1.0
        wt = torch.exp(-S * self.gamma * (t1 - t))
        A = 1.0
        B = (wt * S) / (1.0 - wt)
        C = wt
        rate = A + B[:, None, None] * qx + C[:, None, None] * qy
        return rate

    def transition_probability(self, t, k0, k1):
        """
        \begin{equation}
        P(x_t=x|x_0,x_1) = \frac{p(x_1|x_t=x) p(x_t = x|x_0)}{p(x_1|x_0)}
        \end{equation}
        """
        # ...reshape input tenors:
        t = t.squeeze()
        if k0.dim() == 1:
            k0 = k0.unsqueeze(1)  # Add an extra dimension if needed
        if k1.dim() == 1:
            k1 = k1.unsqueeze(1)

        # ...set state configurations:
        k = torch.arange(0, self.vocab_size)  # shape: (vocab_size,)
        k = k[None, None, :].repeat(k0.size(0), k0.size(1), 1).float()
        k = k.to(k0.device)

        # ...compute probabilities:
        p_k_to_k1 = self.conditional_probability(t, 1.0, k, k1)
        p_k0_to_k = self.conditional_probability(0.0, t, k0, k)
        p_k0_to_k1 = self.conditional_probability(0.0, 1.0, k0, k1)

        return (p_k_to_k1 * p_k0_to_k) / p_k0_to_k1

    def conditional_probability(self, t_in, t_out, k_in, k_out):
        """
        \begin{equation}
        P(x(t) = i|x(t_0)) = \frac{1}{s} + w_{t,t_0}\left(-\frac{1}{s} + \delta_{i,x(t_0)}\right)
        \end{equation}

        \begin{equation}
        w_{t,t_0} = e^{-S \int_{t_0}^{t} \beta(r)dr}
        \end{equation}

        """
        S = self.vocab_size
        t_out = right_time_size(t_out, k_out).to(k_in.device)
        t_in = right_time_size(t_in, k_out).to(k_in.device)
        w_t = torch.exp(-S * self.gamma * (t_out - t_in))
        k_out, k_in = right_shape(k_out), right_shape(k_in)
        kronecker = (k_out == k_in).float()
        prob = 1.0 / S + w_t[:, None, None] * ((-1.0 / S) + kronecker)
        return prob

    def solver_step(self, state, heads, delta_t):
        """tau-leaping step for master equation solver"""
        rates = self.rate(t=state.time, k=state.discrete, logits=heads.discrete)
        assert (rates >= 0).all(), "Negative rates!"
        state.discrete = state.discrete.squeeze(-1)
        # max_rate = torch.max(rates, dim=2)[1]
        all_jumps = torch.poisson(rates * delta_t).to(state.time.device)
        jump_mask = torch.sum(all_jumps, dim=-1).type_as(state.discrete) <= 1
        diff = (
            torch.arange(self.vocab_size, device=state.time.device).view(
                1, 1, self.vocab_size
            )
            - state.discrete[:, :, None]
        )
        net_jumps = torch.sum(all_jumps * diff, dim=-1).type_as(state.discrete)
        state.discrete += net_jumps * jump_mask
        state.discrete = torch.clamp(state.discrete, min=0, max=self.vocab_size - 1)
        state.discrete = state.discrete.unsqueeze(-1)
        state.discrete *= heads.absorbing
        return state


right_shape = lambda x: x if len(x.shape) == 3 else x[:, :, None]
right_time_size = (
    lambda t, x: t
    if isinstance(t, torch.Tensor)
    else torch.full((x.size(0),), t).to(x.device)
)
