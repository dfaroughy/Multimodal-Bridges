import torch
import numpy as np
import math
from torch import nn
from typing import Tuple
import torch.nn.utils.weight_norm as wn

from tensorclass import TensorMultiModal


class MultiModalEmbedder(nn.Module):
    """
    Module that embeds multimodal point cloud data into latent vector spaces for downstream tasks.
    Includes non-markovian source augmentation (optional) as in https://arxiv.org/abs/2311.06978

    Output Dimensions:
        local state:
            - state_loc.time: (B, N, dim_emb_time)
            - state_loc.continuous: (B, N, dim_emb_continuous )
            - state_loc.discrete: (B, N, dim_discrete * dim_emb_discrete)
        global state:
            - state_glob.time: (B, dim_emb_time)
            - state_glob.continuous: (B, dim_context_continuous_emb)
            - state_glob.discrete: (B, dim_context_discrete * dim_emb_context_discrete)
    """

    def __init__(self, config):
        super(MultiModalEmbedder, self).__init__()

        self.config = config

        # feats dimensions
        dim_x = config.data.dim_continuous
        dim_k = config.data.vocab_size
        dim_emb_t = config.encoder.dim_emb_time
        dim_emb_x = config.encoder.dim_emb_continuous
        dim_emb_k = config.encoder.dim_emb_discrete

        # Time embeddings

        # if config.encoder.time_embedding == 'SinusoidalPositionalEncoding':
        #     self.time_embedding = SinusoidalPositionalEncoding(dim_emb_t, max_period=10000)
        
        # elif config.encoder.time_embedding == 'GaussianFourierProjection':
        #     self.time_embedding = nn.Sequential(
        #         GaussianFourierProjection(embed_dim=dim_emb_t, scale=2.0),
        #         nn.Linear(dim_emb_t, dim_emb_t),
        #     )

        self.time_embedding = nn.Sequential(
            GaussianFourierProjection(embed_dim=dim_emb_t, scale=30.0),
            nn.Linear(dim_emb_t, dim_emb_t),
        )

        # Continuous embeddings
        if config.data.modality in ["continuous", "multi-modal"]:
            self.continuous_embedding = nn.Linear(dim_x, dim_emb_x)

        # Discrete embeddings
        if config.data.modality in ["discrete", "multi-modal"]:
            self.discrete_embedding = wn(nn.Embedding(dim_k, dim_emb_k))

    def forward(
        self, state: TensorMultiModal, batch: TensorMultiModal
    ) -> Tuple[TensorMultiModal, TensorMultiModal]:
        continuous_feats, discrete_feats = None, None

        t_emb = self.time_embedding(state.time.squeeze(-1))
        # t_emb = self.time_embedding(state.time)
        time_context = t_emb.clone().to(t_emb.device)  # (B, dim_time_emb)
        time = t_emb.unsqueeze(1).repeat(1, state.shape[-1], 1)  # (B, N, dim_time_emb)

        if state.has_continuous:
            continuous_feats = self.continuous_embedding(state.continuous)

        if state.has_discrete:
            discrete_feats = self.discrete_embedding(state.discrete).squeeze(-2)

        state_loc = TensorMultiModal(time, continuous_feats, discrete_feats, state.mask)
        state_loc.apply_mask()
        state_glob = TensorMultiModal(time=time_context)

        return state_loc, state_glob


class GaussianFourierProjection(nn.Module):
    """
    Gaussian random features for encoding time steps.
    """

    def __init__(self, embed_dim, scale=1.0):
        super().__init__()
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1).squeeze()


class SinusoidalPositionalEncoding(nn.Module):
    """Positional encoding with log-linear spaced frequencies for each dimension"""

    def __init__(self, dim, max_period=10000):
        super(SinusoidalPositionalEncoding, self).__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, timesteps):
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(
                start=0, end=half, dtype=torch.float32, device=timesteps.device
            )
            / half
        )
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding.squeeze()
