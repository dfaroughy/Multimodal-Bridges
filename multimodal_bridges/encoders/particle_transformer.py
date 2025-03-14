import torch
import torch.nn as nn

from tensorclass import TensorMultiModal


class MultiModalParticleTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        aug_factor = 2 if config.encoder.data_augmentation else 1
        
        dim_time = config.encoder.dim_emb_time
        dim_input_continuous = config.encoder.dim_emb_continuous
        dim_input_discrete = config.data.dim_discrete * (
            config.encoder.dim_emb_discrete * aug_factor
        )
        dim_hidden_continuous = config.encoder.dim_hidden_continuous
        dim_hidden_discrete = config.encoder.dim_hidden_discrete
        dim_output_continuous = config.data.dim_continuous
        dim_output_discrete = config.data.dim_discrete * config.data.vocab_size
        num_heads = config.encoder.num_heads

        # ...continuous pipeline

        self.attn_continuous_0 = ParticleAttentionBlock(
            dim_input_continuous,  # key, values
            dim_input_continuous,  # query
            dim_hidden_continuous,
            dim_hidden_continuous,
            num_heads=num_heads,
            dropout=config.encoder.dropout,
        )

        self.attn_continuous_1 = ParticleAttentionBlock(
            dim_hidden_continuous,
            dim_hidden_continuous,
            dim_hidden_continuous,
            dim_hidden_continuous,
            dim_time=dim_time,
            num_heads=num_heads,
            dropout=config.encoder.dropout,
        )

        self.attn_continuous_2 = ParticleAttentionBlock(
            dim_hidden_continuous,
            dim_hidden_continuous,
            dim_hidden_continuous,
            dim_hidden_continuous,
            dim_time=dim_time,
            num_heads=num_heads,
            dropout=config.encoder.dropout,
        )

        self.cross_attn_continuous_0 = ParticleAttentionBlock(
            dim_hidden_continuous,
            dim_hidden_discrete,
            dim_hidden_continuous,
            dim_hidden_continuous,
            dim_time=dim_time,
            num_heads=num_heads,
            dropout=config.encoder.dropout,
        )

        self.cross_attn_continuous_1 = ParticleAttentionBlock(
            dim_hidden_continuous,
            dim_hidden_discrete,
            dim_hidden_continuous,
            dim_output_continuous,
            dim_time=dim_time,
            num_heads=num_heads,
            dropout=config.encoder.dropout,
        )
        # ...discrete pipeline

        self.attn_discrete_0 = ParticleAttentionBlock(
            dim_input_discrete,
            dim_input_discrete,
            dim_hidden_discrete,
            dim_hidden_discrete,
            num_heads=num_heads,
            dropout=config.encoder.dropout,
        )

        self.attn_discrete_1 = ParticleAttentionBlock(
            dim_hidden_discrete,
            dim_hidden_discrete,
            dim_hidden_discrete,
            dim_hidden_discrete,
            dim_time=dim_time,
            num_heads=num_heads,
            dropout=config.encoder.dropout,
        )

        self.attn_discrete_2 = ParticleAttentionBlock(
            dim_hidden_discrete,
            dim_hidden_discrete,
            dim_hidden_discrete,
            dim_hidden_discrete,
            dim_time=dim_time,
            num_heads=num_heads,
            dropout=config.encoder.dropout,
        )

        self.cross_attn_discrete_0 = ParticleAttentionBlock(
            dim_hidden_discrete,
            dim_hidden_continuous,
            dim_hidden_discrete,
            dim_hidden_discrete,
            dim_time=dim_time,
            num_heads=num_heads,
            dropout=config.encoder.dropout,
        )

        self.cross_attn_discrete_1 = ParticleAttentionBlock(
            dim_hidden_discrete,
            dim_hidden_continuous,
            dim_hidden_discrete,
            dim_output_discrete,
            dim_time=dim_time,
            num_heads=num_heads,
            dropout=config.encoder.dropout,
        )

    def forward(
        self, state_local: TensorMultiModal, state_global: TensorMultiModal
    ) -> TensorMultiModal:
        time = state_local.time
        continuous = state_local.continuous
        discrete = state_local.discrete
        mask = state_local.mask

        h = self.attn_continuous_0(continuous, continuous, None, mask)
        f = self.attn_discrete_0(discrete, discrete, None, mask)

        h_res, f_res = h.clone(), f.clone()

        h = self.attn_continuous_1(h, h, time, mask)
        f = self.attn_discrete_1(f, f, time, mask)

        h = self.attn_continuous_2(h, h, time, mask)
        f = self.attn_discrete_2(f, f, time, mask)

        h = self.cross_attn_continuous_0(h, f, time, mask)
        f = self.cross_attn_discrete_0(f, h, time, mask)

        h += h_res
        f += f_res

        head_continuous = self.cross_attn_continuous_1(h, f, time, mask)
        head_discrete = self.cross_attn_discrete_1(f, h, time, mask)

        return TensorMultiModal(
            continuous=head_continuous, discrete=head_discrete, mask=mask
        )


class ParticleAttentionBlock(nn.Module):
    def __init__(
        self,
        dim_key,
        dim_query,
        dim_hidden,
        dim_output,
        dim_time=0,
        num_heads=1,
        activation=nn.GELU(),
        dropout=0.0,
    ):
        super().__init__()

        dim_query += dim_time
        dim_key += dim_time

        self.norm_0 = nn.LayerNorm(dim_key)
        self.norm_1 = nn.LayerNorm(dim_query)
        self.norm_2 = nn.LayerNorm(dim_hidden)

        self.query_proj = nn.Linear(dim_query, dim_hidden, bias=False)
        self.key_proj = nn.Linear(dim_key, dim_hidden, bias=False)
        self.value_proj = nn.Linear(dim_key, dim_hidden, bias=False)

        self.attention = nn.MultiheadAttention(
            dim_hidden, num_heads, dropout=dropout, batch_first=True, bias=False
        )
        self.feedfwd = nn.Sequential(
            nn.Linear(dim_hidden, dim_hidden),
            activation,
            nn.LayerNorm(dim_hidden),
            nn.Linear(dim_hidden, dim_output),
        )

    def forward(self, key, query, time, mask):
        if time is not None:
            query = torch.cat([query, time], dim=-1)
            key = torch.cat([key, time], dim=-1)
        key = self.norm_0(key)
        query = self.norm_1(query)
        Wq = self.query_proj(query)
        Wk = self.key_proj(key)
        Wv = self.value_proj(key)
        attn, _ = self.attention(Wq, Wk, Wv, key_padding_mask=mask.float().squeeze(-1))
        attn = self.norm_2(attn)
        return self.feedfwd(attn)
