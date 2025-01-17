import torch
import math
from torch import nn
from typing import Tuple

from data.datasets import MultiModeState
from utils.helpers import SimpleLogger as log


class EmbedMode(nn.Module):
    """Embedding module for a single data mode.

    Args:
        embedding: Type of embedding ('Linear', 'MLP', etc.).
        dim_input: Input dimension size.
        dim_hidden: Hidden/output dimension size.
    """

    def __init__(self, embedding_type, dim_input=None, dim_hidden=None):
        super().__init__()

        if embedding_type == "Linear":
            self.embedding = nn.Linear(dim_input, dim_hidden)

        elif embedding_type == "MLP":
            self.embedding = nn.Sequential(
                nn.Linear(dim_input, dim_hidden),
                nn.ReLU(),
                nn.Linear(dim_hidden, dim_hidden),
            )

        elif embedding_type == "LookupTable":
            self.embedding = nn.Embedding(dim_input, dim_hidden)

        elif embedding_type == "LookupTableMLP":
            self.embedding = nn.Sequential(
                nn.Embedding(dim_input, dim_hidden),
                nn.ReLU(),
                nn.Linear(dim_hidden, dim_hidden),
            )

        elif embedding_type == "SinusoidalPositionalEncoding":
            self.embedding = SinusoidalPositionalEncoding(dim_hidden, max_period=10000)

        else:
            NotImplementedError(
                "Mode embedding not implemented, use `Linear`, `MLP`, `LookupTable`, `LookupTableMLP`, or `Sinusoidal`"
            )

    def forward(self, x):
        return self.embedding(x)


class MultiModalParticleCloudEmbedder(nn.Module):
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
        super(MultiModalParticleCloudEmbedder, self).__init__()

        self.augmentation = config.encoder.data_augmentation
        log.info("Augmenting local state with source data", self.augmentation)

        # feats dimensions
        dim_x = config.data.dim_continuous * (2 if self.augmentation else 1)
        dim_k = config.data.dim_discrete * (2 if self.augmentation else 1)
        dim_emb_t = config.encoder.dim_emb_time
        dim_emb_x = config.encoder.dim_emb_continuous
        dim_emb_k = config.encoder.dim_emb_discrete

        # context dimensions
        dim_context_x = config.data.dim_context_continuous
        dim_context_k = config.data.dim_context_discrete
        dim_emb_context_x = config.encoder.dim_emb_context_continuous
        dim_emb_context_k = config.encoder.dim_emb_context_discrete

        # Time embeddings
        self.embedding_time = EmbedMode(
            config.encoder.embed_type_time,
            dim_hidden=dim_emb_t,
        )

        # Continuous embeddings
        if config.data.modality in ["continuous", "multi-modal"]:
            if config.encoder.embed_type_continuous:
                self.embedding_continuous = EmbedMode(
                    config.encoder.embed_type_continuous,
                    dim_input=dim_x,
                    dim_hidden=dim_x if dim_emb_x == 0 else dim_emb_x,
                )

                log.warn(f"setting dim_emb_continuous = {dim_x}", dim_emb_x == 0)

        # Discrete embeddings
        if config.data.modality in ["discrete", "multi-modal"]:
            if config.encoder.embed_type_discrete:
                self.embedding_discrete = EmbedMode(
                    config.encoder.embed_type_discrete,
                    dim_input=config.data.vocab_size,
                    dim_hidden=dim_k if dim_emb_k == 0 else dim_emb_k,
                )

                log.warn(f"setting dim_emb_discrete = {dim_k}", dim_emb_k == 0)

        # Context embeddings
        if config.encoder.embed_type_context_continuous:
            self.embedding_context_continuous = EmbedMode(
                config.encoder.embed_type_context_continuous,
                dim_input=dim_context_x,
                dim_hidden=dim_context_x
                if dim_emb_context_x == 0
                else dim_emb_context_x,
            )

            log.warn(
                f"setting dim_emb_context_continuous = {dim_context_x}",
                dim_emb_context_x == 0,
            )

        if config.encoder.embed_type_context_discrete:
            self.embedding_context_discrete = EmbedMode(
                config.encoder.embed_type_context_discrete,
                dim_input=config.data.vocab_size_context,
                dim_hidden=dim_context_k
                if dim_emb_context_k == 0
                else dim_emb_context_k,
            )

            log.warn(
                f"setting dim_emb_context_dicsrete = {dim_context_k}",
                dim_emb_context_k == 0,
            )

    def forward(
        self, state: MultiModeState, source: MultiModeState, context: MultiModeState
    ) -> Tuple[MultiModeState, MultiModeState]:
    
        reshape = (*tuple(state.shape), -1)

        # Initialize states
        state_loc = MultiModeState(mask=state.mask)
        state_glob = MultiModeState()

        # Embed time
        t_emb = self.embedding_time(state.time.squeeze(-1))

        state_glob.time = t_emb.clone().to(t_emb.device)  # (B, dim_time_emb)
        state_loc.time = t_emb.unsqueeze(1).repeat(1, state.shape[-1], 1)  # (B, N, dim_time_emb)
        state_loc.time *= state.mask

        # Augment with source data:
        if self.augmentation:
            state.continuous = torch.cat([state.continuous, source.continuous], dim=-1)
            state.discrete = torch.cat([state.discrete, source.discrete], dim=-1)

        # Embed features
        x_feats, k_feats = [], []

        if hasattr(self, "embedding_continuous"):
            x_feats.append(self.embedding_continuous(state.continuous))
        else:
            if "continuous" in state.available_modes():
                x_feats.append(state.continuous)

        if hasattr(self, "embedding_discrete"):
            k_feats.append(self.embedding_discrete(state.discrete).view(*reshape))
        else:
            if "discrete" in state.available_modes():
                k_feats.append(state.discrete.view(*reshape))

        # Assign embedded features
        if x_feats:
            state_loc.continuous = torch.cat(x_feats, dim=-1) * state.mask
        if k_feats:
            state_loc.discrete = torch.cat(k_feats, dim=-1) * state.mask

        # Embed context
        if hasattr(self, "embedding_context_continuous"):
            state_glob.continuous = self.embedding_context_continuous(
                context.continuous
            )
        else:
            state_glob.continuous = context.continuous

        if hasattr(self, "embedding_context_discrete"):
            state_glob.discrete = self.embedding_context_discrete(
                context.discrete
            ).view(len(context), -1)
        else:
            state_glob.discrete = context.discrete

        return state_loc, state_glob


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
