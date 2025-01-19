import torch
import math
from torch import nn
from typing import Tuple

from data.datasets import MultiModeState
from pipeline.helpers import SimpleLogger as log


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
        self, state: MultiModeState, batch: MultiModeState) -> Tuple[MultiModeState, MultiModeState]:
    
        continuous_feats, discrete_feats = None, None
        continuous_context, discrete_context = None, None

        reshape = (*tuple(state.shape), -1)

        # Embed time
        t_emb = self.embedding_time(state.time.squeeze(-1))
        time_context = t_emb.clone().to(t_emb.device)  # (B, dim_time_emb)
        time = t_emb.unsqueeze(1).repeat(1, state.shape[-1], 1)  # (B, N, dim_time_emb)

        # Embed features
        if state.has_continuous:
            if self.augmentation:
                continuous_feats = torch.cat([state.continuous, batch.source.continuous], dim=-1)
            else:
                continuous_feats = state.continuous
            if hasattr(self, "embedding_continuous"):
                continuous_feats = self.embedding_continuous(continuous_feats)

        if state.has_discrete:
            discrete_feats = state.discrete
            if hasattr(self, "embedding_discrete"):
                discrete_feats = self.embedding_discrete(discrete_feats).view(*reshape)

        state_loc = MultiModeState(time, continuous_feats, discrete_feats, state.mask)
        state_loc.apply_mask()

        # Embed context
        if batch.has_context: 

            reshape = (*tuple(batch.context), -1)

            if batch.context.has_continuous:
                continuous_context = batch.context.continuous
                if hasattr(self, "embedding_context_continuous"):
                    continuous_context = self.embedding_context_continuous(continuous_context)

            if batch.context.has_discrete:
                discrete_context = batch.context.discrete
                if hasattr(self, "embedding_context_discrete"):
                    discrete_context = self.embedding_context_discrete(discrete_context).view(reshape)

        state_glob = MultiModeState(time_context, continuous_context, discrete_context, None)

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
