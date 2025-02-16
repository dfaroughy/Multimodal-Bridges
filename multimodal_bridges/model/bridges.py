import torch
from torch.nn.functional import softmax
from torch.distributions import Categorical
from dataclasses import dataclass

from tensorclass import TensorMultiModal
from datamodules.datasets import DataCoupling


class UniformLinearFlow:
    """Conditional OT Flow-Matching for continuous states.
    This bridge is a linear interpolation between boundaries states at t=0 and t=1.
    notation:
      - t: time
      - x0: continuous source state at t=0
      - x1: continuous target state at t=1
      - x: continuous state at time t
      - z: delta function regularizer
    """

    def __init__(self, sigma):
        self.sigma = sigma

    def sample(self, t, batch: DataCoupling):
        x0 = batch.source.continuous
        x1 = batch.target.continuous
        xt = t * x1 + (1.0 - t) * x0
        z = torch.randn_like(xt)
        std = self.sigma
        return xt + std * z

    def drift(self, state: TensorMultiModal, batch: DataCoupling):
        x0 = batch.source.continuous
        x1 = batch.target.continuous
        xt = state.continuous
        A = 0.0
        B = 1.0
        C = -1.0
        return A * xt + B * x1 + C * x0

    def diffusion(self, state: TensorMultiModal):
        return 0.0

    def forward_step(self, state, heads, delta_t):
        """Euler step for ODE solver"""
        state.continuous += delta_t * heads.continuous
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

    def __init__(self, sigma):
        self.sigma = sigma

    def sample(self, t, batch: DataCoupling):
        x0 = batch.source.continuous
        x1 = batch.target.continuous
        x = t * x1 + (1.0 - t) * x0
        z = torch.randn_like(x)
        std = self.sigma * torch.sqrt(t * (1.0 - t))
        return x + std * z

    def drift(self, state: TensorMultiModal, batch: DataCoupling):
        x0 = batch.source.continuous
        x1 = batch.target.continuous
        xt = state.continuous
        t = state.time
        A = (1 - 2 * t) / (t * (1 - t))
        B = t**2 / (t * (1 - t))
        C = -1 * (1 - t) ** 2 / (t * (1 - t))
        return A * xt + B * x1 + C * x0

    def diffusion(self, state: TensorMultiModal):
        t = state.time
        return self.sigma * torch.sqrt(t * (1.0 - t))

    def forward_step(self, state: TensorMultiModal, heads: TensorMultiModal, delta_t):
        """Euler-Maruyama step for SDE solver"""
        diffusion = self.diffusion(delta_t)
        delta_w = torch.randn_like(state.continuous)
        state.continuous += delta_t * state.continuous + diffusion * delta_w
        return state


class TelegraphBridge:
    """Multivariate Telegraph bridge for discrete states
    - t: time
    - k0: discrete source state at t=0
    - k1: discrete  target state at t=1
    - k: discrete state at time t
    """

    def __init__(self, gamma, vocab_size):
        self.gamma = gamma
        self.vocab_size = vocab_size

    def sample(self, t, batch: DataCoupling):
        k0 = batch.source.discrete
        k1 = batch.target.discrete
        transition_probs = self.transition_probability(t, k0, k1)
        drwan_state = Categorical(transition_probs).sample().to(k1.device)
        if drwan_state.dim() == 2:
            drwan_state = drwan_state.unsqueeze(-1)
        return drwan_state

    def rate(self, state: TensorMultiModal, heads: TensorMultiModal):
        """t: (b, 1) time tensor
        k: (b, n, 1) current state tensor
        logits: (b, n, vocab_size) logits tensor
        """
        t = state.time
        k = state.discrete
        logits = heads.discrete

        assert (k >= 0).all() and (k < self.vocab_size).all(), (
            "Values in `k` outside of bound! k_min={}, k_max={}".format(
                k.min(), k.max()
            )
        )

        qx = softmax(logits, dim=2) # transition probabilities to all states
        qy = torch.gather(qx, 2, k.long())  # current state prob

        # ...Telegraph process rates:

        S = self.vocab_size
        t, t1 = t.squeeze(), 1.0
        wt = torch.exp(-S * self.gamma * (t1 - t))
        A = 1.0
        B = (wt * S) / (1.0 - wt)
        C = wt
        return A + B[:, None, None] * qx + C[:, None, None] * qy

    def transition_probability(self, t, k0, k1):
        """
        \begin{equation}
        P(x_t=x|x_0,x_1) = \frac{p(x_1|x_t=x) p(x_t = x|x_0)}{p(x_1|x_0)}
        \end{equation}
        """
        # ...reshape input tenors:
        t = t.clone().squeeze()  # shape: (B)
        if k0.dim() == 1:
            k0 = k0.unsqueeze(1)  # Add an extra dimension if needed
        if k1.dim() == 1:
            k1 = k1.unsqueeze(1)

        # ...set state configurations:
        k = torch.arange(0, self.vocab_size)  # shape: (vocab_size,)
        k = (
            k[None, None, :].repeat(k0.size(0), k0.size(1), 1).float()
        )  # shape: (B, N, vocab_size)
        # k = k.expand(k0.size(0), k0.size(1), -1)  # shape: (B, N, vocab_size)
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

    def forward_step(self, state, heads, delta_t):
        """tau-leaping step for master equation solver"""
        rates = self.rate(state, heads)
        assert (rates >= 0).all(), "Negative rates!"
        state.discrete = state.discrete.squeeze(-1)
        max_rate = torch.max(rates, dim=2)[1]
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
        return state, max_rate


right_shape = lambda x: x if len(x.shape) == 3 else x[:, :, None]
right_time_size = (
    lambda t, x: t
    if isinstance(t, torch.Tensor)
    else torch.full((x.size(0),), t).to(x.device)
)
