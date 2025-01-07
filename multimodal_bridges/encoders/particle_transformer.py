import torch
import torch.nn as nn
import torch.nn.functional as F

from cmb.models.architectures.utils import (
    fc_block,
    get_activation_function,
    KANLinear,
    KAN,
    SinusoidalPositionalEncoding,
    GaussianFourierFeatures,
)


class ParticleTransformer(nn.Module):
    """ Intertaction-less particle-cloud transformer
    """
    def __init__(self, config):
        super().__init__()
        self.device = config.device
        self.particle_embedding = ParticleEmbedding(config)
        self.particle_attention_blocks = []
        self.particle_attention_blocks = nn.ModuleList(
            [
                ParticleAttentionBlock(
                    dim_input=config.dim_hidden,
                    dim_output=config.dim_hidden,
                    dim_hidden=config.dim_hidden,
                    num_heads=config.num_heads,
                    activation=get_activation_function(config.activation),
                    dropout=config.dropout,
                    attention_embedding="linear",
                )
                for _ in range(config.num_attention_blocks)
            ]
        )
        self.projection = nn.Linear(config.dim_hidden, config.dim_continuous)

    def forward(self, t, x, k=None, context=None, mask=None):
        t = t.to(self.device)  # time
        x = x.to(self.device)  # continuous feature (b, n, dim_continuous)
        k = (
            k.to(self.device) if k is not None else None
        )  # discrete feature (b, n, dim_discrete)
        mask = mask.to(self.device)
        h = self.particle_embedding(t=t, x=x, mask=mask)
        for block in self.particle_attention_blocks:
            h = block(h, mask)
        h = self.projection(h) * mask
        return h


class ParticleAttentionBlock(nn.Module):
    def __init__(
        self,
        dim_input,
        dim_output,
        dim_hidden=128,
        num_heads=4,
        activation=nn.GELU(),
        dropout=0.0,
        attention_embedding="linear",
    ):
        super().__init__()

        self.layernorm_0 = nn.LayerNorm(dim_input)
        self.mha_block = MultiHeadAttention(
            dim_input,
            dim_hidden,
            dim_hidden,
            num_heads,
            dropout,
            attention_embedding=attention_embedding,
        )
        self.layernorm_1 = nn.LayerNorm(dim_hidden)
        self.layernorm_2 = nn.LayerNorm(dim_hidden)
        self.fc_block = nn.Sequential(
            nn.Linear(dim_hidden, dim_hidden),
            activation,
            nn.LayerNorm(dim_hidden),
            nn.Linear(dim_hidden, dim_output),
        )

    def forward(self, x, mask):
        h = self.layernorm_0(x)
        h = self.mha_block(h, mask=mask)  # Masking in MHA
        h = self.layernorm_1(h)
        h += x
        f = self.layernorm_2(h)
        f = self.fc_block(f)
        f += h
        f *= mask
        return f


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        dim_input,
        dim_output,
        dim_hidden=128,
        num_heads=4,
        dropout=0.0,
        attention_embedding="linear",
    ):
        super().__init__()

        assert (
            dim_hidden % num_heads == 0
        ), "hidden dimension must be divisible by number of heads"
        self.dim_head = dim_hidden // num_heads
        self.num_heads = num_heads
        self.dim_hidden = dim_hidden

        self.register_buffer("tril", torch.tril(torch.ones(dim_hidden, dim_hidden)))

        # Key, Query, and Value linear transformations
        self.k = nn.Linear(dim_input, dim_hidden, bias=False)
        self.q = nn.Linear(dim_input, dim_hidden, bias=False)
        self.v = nn.Linear(dim_input, dim_hidden, bias=False)
        self.proj = nn.Linear(dim_hidden, dim_output)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        b, n, dim = x.shape
        K, Q, V = self.k(x), self.q(x), self.v(x)  # Compute Key, Query, and Value

        # Reshape for multi-head attention
        K = K.view(b, n, self.num_heads, self.dim_head).transpose(
            1, 2
        )  # (b, num_heads, n, head_dim)
        Q = Q.view(b, n, self.num_heads, self.dim_head).transpose(1, 2)
        V = V.view(b, n, self.num_heads, self.dim_head).transpose(1, 2)

        # Scaled dot-product attention
        QK = (Q @ K.transpose(-2, -1)) * (self.dim_head**-0.5)  # (b, num_heads, n, n)

        if mask is not None:
            # Expand mask from (b, n, 1) to (b, num_heads, n, n) for broadcasting
            mask = mask.unsqueeze(1).expand(b, self.num_heads, n, n)
            QK = QK.masked_fill(mask == 0, float("-1e9"))  # Mask out invalid positions

        # Apply the causal mask (if applicable)
        QK = QK.masked_fill(self.tril[:n, :n] == 0, float("-inf"))

        # Attention weights
        A = F.softmax(QK, dim=-1)  # (b, num_heads, n, n)
        A = self.dropout(A)

        # Compute context vector
        h = (A @ V).transpose(1, 2).contiguous().view(b, n, self.dim_hidden)
        h = self.proj(h)
        return h


# class ParticleEmbedding(nn.Module):
#     def __init__(self, config):
#         super().__init__()

#         # Define the time embedding based on the provided configuration
#         if config.time_embedding == 'sinusoidal':
#             self.time_embedding = nn.Sequential(
#                 SinusoidalPositionalEncoding(config.dim_time_emb, max_period=10000),
#                 nn.Linear(config.dim_time_emb, config.dim_time_emb)
#             )
#         elif config.time_embedding == 'kolmogorov-arnold':
#             self.time_embedding = nn.Sequential(
#                 KANLinear(1, config.dim_time_emb),
#                 nn.Linear(config.dim_time_emb, config.dim_time_emb)
#             )
#         elif config.time_embedding is None:
#             self.time_embedding = nn.Identity()
#         else:
#             raise NotImplementedError

#         # Define the main embedding layer
#         self.embedding = nn.Linear(config.dim_continuous + config.dim_time_emb, config.dim_hidden)

#     def forward(self, t, x, k=None, mask=None):
#         """
#         Forward pass of the particle embedding.

#         Arguments:
#         - t: Time input of shape (batch_size, 1) or (batch_size, 1, 1)
#         - x: Particle continuous features of shape (batch_size, max_num_particles, dim_continuous)
#         - mask: Binary mask of shape (batch_size, max_num_particles, 1) indicating valid particles (1) or masked particles (0)

#         Returns:
#         - h: Embedded particles of shape (batch_size, N, dim_hidden), masked appropriately
#         """

#         t_emb = self.time_embedding(t.squeeze(-1))  # (batch_size, dim_time_emb)
#         t_emb = t_emb.unsqueeze(1).repeat(1, x.shape[1], 1)  # Repeat for each particle -> (batch_size, N, dim_time_emb)
#         t_emb *= mask
#         x *= mask
#         h = torch.cat([t_emb, x], dim=-1)  # (batch_size, N, dim_continuous + dim_time_emb)
#         h = self.embedding(h)  # (batch_size, N, dim_hidden)
#         h *= mask
#         return h
