import torch
import math
from torch import nn
from typing import Tuple
from states import HybridState


class MultiModalPointCloudEmbedder(nn.Module):
    """
    Module that embeds multimodal point cloud data into latent vector spaces for downstream tasks.
    Includes non-markovian source augmentation (optional) as in https://arxiv.org/abs/2311.06978

    Output Dimensions:
        local state:
            - state_loc.time: (B, N, dim_emb_time)
            - state_loc.continuous: (B, N, dim_emb_continuous + dim_emb_augment_continuous)
            - state_loc.discrete: (B, N, dim_discrete * (dim_emb_discrete + dim_emb_augment_discrete))
        global state:
            - state_glob.time: (B, dim_emb_time)
            - state_glob.continuous: (B, dim_context_continuous_emb)
            - state_glob.discrete: (B, dim_context_discrete * dim_emb_context_discrete)
    """

    def __init__(self, config):
        super(MultiModalPointCloudEmbedder, self).__init__()

        self.embedding_time = EmbedMode(
            config.encoder.embed_type_time,
            dim_hidden=config.encoder.dim_emb_time,
        )

        # Continuous embeddings
        if hasattr(config.model, "bridge_continuous"):
            self.embedding_continuous = EmbedMode(
                config.encoder.embed_type_continuous,
                dim_input=config.data.dim_continuous,
                dim_hidden=config.encoder.dim_emb_continuous,
            )

            if config.encoder.embed_type_augment_continuous:
                self.embedding_augment_continuous = EmbedMode(
                    config.encoder.embed_type_augment_continuous,
                    dim_input=config.data.dim_continuous,
                    dim_hidden=config.encoder.dim_emb_augment_continuous,
                )

            if config.encoder.embed_type_context_continuous:
                self.embedding_context_continuous = EmbedMode(
                    config.encoder.embed_type_context_continuous,
                    dim_input=config.data.dim_context_continuous,
                    dim_hidden=config.encoder.dim_emb_context_continuous,
                )

        # Discrete embeddings
        if hasattr(config.model, "bridge_discrete"):
            self.embedding_discrete = EmbedMode(
                config.encoder.embed_type_discrete,
                dim_input=config.data.vocab_size,
                dim_hidden=config.encoder.dim_emb_discrete,
            )

            if config.encoder.embed_type_augment_discrete:
                self.embedding_augment_discrete = EmbedMode(
                    config.encoder.embed_type_augment_discrete,
                    dim_input=config.data.vocab_size,
                    dim_hidden=config.encoder.dim_emb_augment_discrete,
                )

            if config.encoder.embed_type_context_discrete:
                self.embedding_context_discrete = EmbedMode(
                    config.encoder.embed_type_context_discrete,
                    dim_input=config.data.vocab_size_context,
                    dim_hidden=config.encoder.dim_emb_context_discrete,
                )

    def forward(self, state: HybridState, batch) -> Tuple[HybridState, HybridState]:
        B = state.mask.shape[0]
        N = state.mask.shape[1]
        shape = (B, N, -1) if state.mask.ndim == 3 else (B, -1)

        # Initialize states
        state_loc = HybridState(mask=state.mask)
        state_glob = HybridState()

        # Embed time
        t_emb = self.embedding_time(state.time.squeeze(-1))
        state_glob.time = t_emb.clone().to(t_emb.device)  # (B, dim_time_emb)
        state_loc.time = t_emb.unsqueeze(1).repeat(1, N, 1)  # (B, N, dim_time_emb)
        state_loc.time *= state.mask

        # Embed features
        x_feats, k_feats = [], []
        if hasattr(self, "embedding_continuous"):
            x_feats.append(self.embedding_continuous(state.continuous))
        if hasattr(self, "embedding_augment_continuous"):
            x_feats.append(self.embedding_augment_continuous(batch.source_continuous))
        if hasattr(self, "embedding_discrete"):
            k_feats.append(self.embedding_discrete(state.discrete).view(shape))
        if hasattr(self, "embedding_augment_discrete"):
            k_feats.append(
                self.embedding_augment_discrete(batch.source_discrete).view(shape)
            )

        # Assign embedded features
        if x_feats:
            state_loc.continuous = torch.cat(x_feats, dim=-1) * state.mask
        if k_feats:
            state_loc.discrete = torch.cat(k_feats, dim=-1) * state.mask

        # Embed context
        if hasattr(self, "embedding_context_continuous"):
            state_glob.continuous = self.embedding_context_continuous(
                batch.context_continuous
            )
        if hasattr(self, "embedding_context_discrete"):
            state_glob.discrete = self.embedding_context_discrete(
                batch.context_discrete
            ).view(B, -1)

        return state_loc, state_glob


class EmbedMode(nn.Module):
    """Embedding module for various embedding strategies.

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
            self.embedding = nn.Embedding(dim_input, dim_hidden) # output=
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
