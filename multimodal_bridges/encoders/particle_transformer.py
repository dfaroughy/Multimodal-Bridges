import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.weight_norm as wn

from tensorclass import TensorMultiModal

class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_input, d_model, d_time, n_layers, n_heads, dim_feedforward, dropout):
        """
        d_input: Dimension of input features per particle.
        d_model: Dimension to which inputs are projected for the transformer.
        d_time: Dimension of the pre-embedded time vector.
        n_layers: Number of transformer encoder layers.
        n_heads: Number of attention heads.
        dim_feedforward: Hidden dimension for the feed-forward networks.
        dropout: Dropout rate.
        """
        super().__init__()


        self.input_proj = nn.Linear(d_input, d_model)

        if d_time != d_model:
            self.time_proj = nn.Linear(d_time, d_model)
        else:
            self.time_proj = nn.Identity()
            
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, t_emb, x, mask):
        """
        Args:
            x: Tensor of shape (B, N, d_input) containing particle features.
            t_emb: Pre-embedded time tensor. Ideally shape (B, d_time) or (B, 1, d_time).
            mask: Tensor of shape (B, N) where 1 indicates a valid particle and 0 indicates padding.
        Returns:
            Tensor of shape (B, N, d_model).
        """
        h = self.input_proj(x.float())
        t_emb_proj = self.time_proj(t_emb) 
        h += t_emb_proj
        return self.transformer(h, src_key_padding_mask=mask.squeeze(-1) > 0)

class MultiModalParticleTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.dim_time = config.encoder.dim_emb_time
        self.dim_cont_in = config.encoder.dim_emb_continuous  # input dim for continuous branch
        self.dim_disc_in = config.encoder.dim_emb_discrete * config.data.dim_discrete  # input for discrete branch

        self.dim_out_cont = config.data.dim_continuous
        self.dim_out_disc = config.data.vocab_size * config.data.dim_discrete

        # Hidden dimensions for each branch 
        self.dim_hid_cont = config.encoder.dim_hidden_local[0]
        self.dim_hid_disc = config.encoder.dim_hidden_local[1]

        num_blocks = config.encoder.num_blocks  # e.g. (num_blocks_cont, num_blocks_disc, num_blocks_fusion)

        self.num_blocks_cont = num_blocks[0]
        self.num_blocks_disc = num_blocks[1]
        self.num_blocks_fusion = num_blocks[2]

        # Transformer hyperparameters 

        self.n_heads = config.encoder.num_heads
        self.dim_feedforward = 4 * self.dim_hid_cont
        self.dropout = config.encoder.dropout

        # Create transformer encoders for each modality.

        self.continuous_encoder = TransformerEncoderBlock(
            d_input=self.dim_cont_in,
            d_model=self.dim_hid_cont,
            d_time=self.dim_time,
            n_layers=self.num_blocks_cont,
            n_heads=self.n_heads,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
        )

        self.discrete_encoder = TransformerEncoderBlock(
            d_input=self.dim_disc_in,
            d_model=self.dim_hid_disc,
            d_time=self.dim_time,
            n_layers=self.num_blocks_disc,
            n_heads=self.n_heads,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
        )

        # Fusion stage if enabled (when num_blocks_fusion > 0)

        self.mode_fusion = self.num_blocks_fusion > 0

        if self.mode_fusion:
            fusion_input_dim = self.dim_hid_cont + self.dim_hid_disc
            self.dim_hid_fusion = config.encoder.dim_hidden_local[2]
            self.fused_encoder = TransformerEncoderBlock(
                d_input=fusion_input_dim,
                d_model=self.dim_hid_fusion,
                d_time=self.dim_time,
                n_layers=self.num_blocks_fusion,
                n_heads=self.n_heads,
                dim_feedforward=self.dim_feedforward,
                dropout=self.dropout,
            )

        # Heads are built similarly to your EPiC heads.
        # For continuous: input = time + (continuous branch output) [+ half fusion output if fused]
        if self.mode_fusion:
            head_cont_in = self.dim_time + self.dim_hid_cont + (self.dim_hid_fusion // 2)
            head_disc_in = self.dim_time + self.dim_hid_disc + (self.dim_hid_fusion // 2)
        else:
            head_cont_in = self.dim_time + self.dim_hid_cont
            head_disc_in = self.dim_time + self.dim_hid_disc

        self.continuous_head = nn.Sequential(
            wn(nn.Linear(head_cont_in, self.dim_hid_cont)),
            nn.GELU(),
            wn(nn.Linear(self.dim_hid_cont, self.dim_out_cont)),
        )
        self.discrete_head = nn.Sequential(
            wn(nn.Linear(head_disc_in, self.dim_hid_disc)),
            nn.GELU(),
            wn(nn.Linear(self.dim_hid_disc, self.dim_out_disc)),
        )

    def forward(self, state_local, state_global):
        """
        Args:
            state_local: a TensorMultiModal with attributes time, continuous, discrete, mask.
            state_global: a TensorMultiModal with global context (e.g., continuous and/or discrete features).
        Returns:
            TensorMultiModal with updated continuous and discrete outputs.
        """
        mask = state_local.mask
        t = state_local.time  
        h1 = self.continuous_encoder(t, state_local.continuous, mask)  
        h2 = self.discrete_encoder(t, state_local.discrete, mask) 

        if self.mode_fusion:
            # Fuse features from both modalities.
            h_cat = torch.cat([h1, h2], dim=-1)  # (B, N, dim_hid_cont + dim_hid_disc)
            fused = self.fused_encoder(t, h_cat, mask)  # (B, N, dim_hid_fusion)
            f1, f2 = torch.chunk(fused, 2, dim=-1)
            head_cont_input = torch.cat([t, h1, f1], dim=-1)
            head_disc_input = torch.cat([t, h2, f2], dim=-1)
        else:
            head_cont_input = torch.cat([t, h1], dim=-1)
            head_disc_input = torch.cat([t, h2], dim=-1)

        # Compute outputs via modality-specific heads.
        out_cont = self.continuous_head(head_cont_input)  # (B, N, dim_out_cont)
        out_disc = self.discrete_head(head_disc_input)      # (B, N, dim_out_disc)

        return TensorMultiModal(time=t, continuous=out_cont, discrete=out_disc, mask=mask)




# class MultiModalParticleTransformer(nn.Module):
#     def __init__(self, config):
#         super().__init__()
        
#         dim_time = config.encoder.dim_emb_time
#         dim_input_continuous = config.encoder.dim_emb_continuous
#         dim_input_discrete = config.data.dim_discrete * (
#             config.encoder.dim_emb_discrete
#         )
#         dim_hidden_continuous = config.encoder.dim_hidden_continuous
#         dim_hidden_discrete = config.encoder.dim_hidden_discrete
#         dim_output_continuous = config.data.dim_continuous
#         dim_output_discrete = config.data.dim_discrete * config.data.vocab_size
#         num_heads = config.encoder.num_heads

#         # ...continuous pipeline

#         self.attn_continuous_0 = ParticleAttentionBlock(
#             dim_input_continuous,  # key, values
#             dim_input_continuous,  # query
#             dim_hidden_continuous,
#             dim_hidden_continuous,
#             num_heads=num_heads,
#             dropout=config.encoder.dropout,
#         )

#         self.attn_continuous_1 = ParticleAttentionBlock(
#             dim_hidden_continuous,
#             dim_hidden_continuous,
#             dim_hidden_continuous,
#             dim_hidden_continuous,
#             dim_time=dim_time,
#             num_heads=num_heads,
#             dropout=config.encoder.dropout,
#         )

#         self.attn_continuous_2 = ParticleAttentionBlock(
#             dim_hidden_continuous,
#             dim_hidden_continuous,
#             dim_hidden_continuous,
#             dim_hidden_continuous,
#             dim_time=dim_time,
#             num_heads=num_heads,
#             dropout=config.encoder.dropout,
#         )

#         self.cross_attn_continuous_0 = ParticleAttentionBlock(
#             dim_hidden_continuous,
#             dim_hidden_discrete,
#             dim_hidden_continuous,
#             dim_hidden_continuous,
#             dim_time=dim_time,
#             num_heads=num_heads,
#             dropout=config.encoder.dropout,
#         )

#         self.cross_attn_continuous_1 = ParticleAttentionBlock(
#             dim_hidden_continuous,
#             dim_hidden_discrete,
#             dim_hidden_continuous,
#             dim_output_continuous,
#             dim_time=dim_time,
#             num_heads=num_heads,
#             dropout=config.encoder.dropout,
#         )
#         # ...discrete pipeline

#         self.attn_discrete_0 = ParticleAttentionBlock(
#             dim_input_discrete,
#             dim_input_discrete,
#             dim_hidden_discrete,
#             dim_hidden_discrete,
#             num_heads=num_heads,
#             dropout=config.encoder.dropout,
#         )

#         self.attn_discrete_1 = ParticleAttentionBlock(
#             dim_hidden_discrete,
#             dim_hidden_discrete,
#             dim_hidden_discrete,
#             dim_hidden_discrete,
#             dim_time=dim_time,
#             num_heads=num_heads,
#             dropout=config.encoder.dropout,
#         )

#         self.attn_discrete_2 = ParticleAttentionBlock(
#             dim_hidden_discrete,
#             dim_hidden_discrete,
#             dim_hidden_discrete,
#             dim_hidden_discrete,
#             dim_time=dim_time,
#             num_heads=num_heads,
#             dropout=config.encoder.dropout,
#         )

#         self.cross_attn_discrete_0 = ParticleAttentionBlock(
#             dim_hidden_discrete,
#             dim_hidden_continuous,
#             dim_hidden_discrete,
#             dim_hidden_discrete,
#             dim_time=dim_time,
#             num_heads=num_heads,
#             dropout=config.encoder.dropout,
#         )

#         self.cross_attn_discrete_1 = ParticleAttentionBlock(
#             dim_hidden_discrete,
#             dim_hidden_continuous,
#             dim_hidden_discrete,
#             dim_output_discrete,
#             dim_time=dim_time,
#             num_heads=num_heads,
#             dropout=config.encoder.dropout,
#         )

#     def forward(
#         self, state_local: TensorMultiModal, state_global: TensorMultiModal
#     ) -> TensorMultiModal:
#         time = state_local.time
#         continuous = state_local.continuous
#         discrete = state_local.discrete
#         mask = state_local.mask

#         h = self.attn_continuous_0(continuous, continuous, None, mask)
#         f = self.attn_discrete_0(discrete, discrete, None, mask)

#         h_res, f_res = h.clone(), f.clone()

#         h = self.attn_continuous_1(h, h, time, mask)
#         f = self.attn_discrete_1(f, f, time, mask)

#         h = self.attn_continuous_2(h, h, time, mask)
#         f = self.attn_discrete_2(f, f, time, mask)

#         h = self.cross_attn_continuous_0(h, f, time, mask)
#         f = self.cross_attn_discrete_0(f, h, time, mask)

#         h += h_res
#         f += f_res

#         head_continuous = self.cross_attn_continuous_1(h, f, time, mask)
#         head_discrete = self.cross_attn_discrete_1(f, h, time, mask)

#         return TensorMultiModal(
#             continuous=head_continuous, discrete=head_discrete, mask=mask
#         )


# class ParticleAttentionBlock(nn.Module):
#     def __init__(
#         self,
#         dim_key,
#         dim_query,
#         dim_hidden,
#         dim_output,
#         dim_time=0,
#         num_heads=1,
#         activation=nn.GELU(),
#         dropout=0.0,
#     ):
#         super().__init__()

#         dim_query += dim_time
#         dim_key += dim_time

#         self.norm_0 = nn.LayerNorm(dim_key)
#         self.norm_1 = nn.LayerNorm(dim_query)
#         self.norm_2 = nn.LayerNorm(dim_hidden)

#         self.query_proj = nn.Linear(dim_query, dim_hidden, bias=False)
#         self.key_proj = nn.Linear(dim_key, dim_hidden, bias=False)
#         self.value_proj = nn.Linear(dim_key, dim_hidden, bias=False)

#         self.attention = nn.MultiheadAttention(
#             dim_hidden, num_heads, dropout=dropout, batch_first=True, bias=False
#         )
#         self.feedfwd = nn.Sequential(
#             nn.Linear(dim_hidden, dim_hidden),
#             activation,
#             nn.LayerNorm(dim_hidden),
#             nn.Linear(dim_hidden, dim_output),
#         )

#     def forward(self, key, query, time, mask):
#         if time is not None:
#             query = torch.cat([query, time], dim=-1)
#             key = torch.cat([key, time], dim=-1)
#         key = self.norm_0(key)
#         query = self.norm_1(query)
#         Wq = self.query_proj(query)
#         Wk = self.key_proj(key)
#         Wv = self.value_proj(key)
#         attn, _ = self.attention(Wq, Wk, Wv, key_padding_mask=mask.float().squeeze(-1))
#         attn = self.norm_2(attn)
#         return self.feedfwd(attn)
