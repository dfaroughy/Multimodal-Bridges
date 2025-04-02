import torch
from torch import nn
import torch.nn.utils.weight_norm as wn

from tensorclass import TensorMultiModal
from encoders.epic import EPiCEncoder


class UniModalEPiC(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        dim_input = (
            config.encoder.dim_emb_time
            + config.encoder.dim_emb_continuous
            + config.encoder.dim_emb_discrete * config.data.dim_discrete
        )

        dim_output = (
            config.data.dim_continuous
            + config.data.vocab_size * config.data.dim_discrete
        )

        dim_context = (
            config.encoder.dim_emb_time
            + config.encoder.dim_emb_context_continuous
            + config.encoder.dim_emb_context_discrete * config.data.dim_context_discrete
        )

        self.epic = EPiCEncoder(
            dim_time=config.encoder.dim_emb_time,
            dim_input_loc=dim_input,
            dim_input_glob=dim_context,
            dim_output_loc=dim_output,
            dim_hid_loc=config.encoder.dim_hidden_local,
            dim_hid_glob=config.encoder.dim_hidden_glob,
            num_blocks=config.encoder.num_blocks,
            use_skip_connection=config.encoder.skip_connection,
            dropout=config.encoder.dropout,
        )

    def forward(
        self, state_local: TensorMultiModal, state_global: TensorMultiModal
    ) -> TensorMultiModal:
        local_modes = [
            getattr(state_local, mode) for mode in state_local.available_modes()
        ]
        global_modes = [
            getattr(state_global, mode) for mode in state_global.available_modes()
        ]

        local_cat = torch.cat(local_modes, dim=-1)
        global_cat = torch.cat(global_modes, dim=-1)
        mask = state_local.mask

        h_loc, h_glob = self.epic(state_local.time, local_cat, global_cat, mask)

        if self.config.data.modality == "continuous":
            return TensorMultiModal(continuous=h_loc, mask=mask)

        elif self.config.data.modality == "discrete":
            return TensorMultiModal(discrete=h_loc, mask=mask)


class MultiModalEPiC(nn.Module):
    """Permutation equivariant architecture for multi-modal continuous-discrete models"""

    def __init__(self, config):
        super().__init__()

        self.config = config
        dim_time = config.encoder.dim_emb_time
        dim_cont = config.encoder.dim_emb_continuous
        dim_disc = config.encoder.dim_emb_discrete * config.data.dim_discrete
        dim_hid_loc = config.encoder.dim_hidden_local
        dim_hid_glob = config.encoder.dim_hidden_glob
        dim_out_continuous = config.data.dim_continuous
        dim_out_discrete = config.data.vocab_size * config.data.dim_discrete

        num_blocks = config.encoder.num_blocks
        self.mode_fused = num_blocks[2] > 0
        self.mode_branched = num_blocks[0] > 0

        dim_context = (
            config.encoder.dim_emb_time
            + config.encoder.dim_emb_context_continuous
            + config.encoder.dim_emb_context_discrete * config.data.dim_context_discrete
        )

        if self.mode_branched:

            self.continuous_encoder = EPiCEncoder(
                dim_time=dim_time,
                dim_input_loc=dim_cont,
                dim_input_glob=dim_context,
                dim_output_loc=dim_hid_loc[0],
                dim_hid_loc=dim_hid_loc[0],
                dim_hid_glob=dim_hid_glob[0],
                num_blocks=num_blocks[0],
                use_skip_connection=config.encoder.skip_connection,
                dropout=config.encoder.dropout,
            )

            self.discrete_encoder = EPiCEncoder(
                dim_time=dim_time,
                dim_input_loc=dim_disc,
                dim_input_glob=dim_context,
                dim_output_loc=dim_hid_loc[1],
                dim_hid_loc=dim_hid_loc[1],
                dim_hid_glob=dim_hid_glob[1],
                num_blocks=num_blocks[1],
                use_skip_connection=config.encoder.skip_connection,
                dropout=config.encoder.dropout,
            )
        else:
            dim_hid_loc[0] = dim_cont
            dim_hid_loc[1] = dim_disc
            dim_hid_glob[0] = dim_context
            dim_hid_glob[1] = 0

        if self.mode_fused:
            self.fused_encoder = EPiCEncoder(
                dim_time=dim_time,
                dim_input_loc=dim_hid_loc[0] + dim_hid_loc[1],
                dim_input_glob=dim_hid_glob[0] + dim_hid_glob[1],
                dim_output_loc=dim_hid_loc[2],
                dim_hid_loc=dim_hid_loc[2],
                dim_hid_glob=dim_hid_glob[2],
                num_blocks=num_blocks[2],
                project_input=False if self.mode_branched else True,
                use_skip_connection=config.encoder.skip_connection,
                dropout=config.encoder.dropout,
            )

        # ...mode heads:

        dim_head_cont = dim_hid_loc[0] + (
            dim_hid_loc[2] // 2 if self.mode_fused else 0
        )
        dim_head_disc = dim_hid_loc[1] + (
            dim_hid_loc[2] // 2 if self.mode_fused else 0
        )

        self.continuous_head = nn.Sequential(
            wn(nn.Linear(dim_time + dim_head_cont, dim_hid_loc[0])),
            nn.GELU(),
            wn(nn.Linear(dim_hid_loc[0], dim_out_continuous)),
        )

        self.discrete_head = nn.Sequential(
            wn(nn.Linear(dim_time + dim_head_disc, dim_hid_loc[1])),
            nn.GELU(),
            wn(nn.Linear(dim_hid_loc[1], dim_out_discrete)),
        )

    def forward(
        self, state_local: TensorMultiModal, state_global: TensorMultiModal
    ) -> TensorMultiModal:

        global_modes = [
            getattr(state_global, mode) for mode in state_global.available_modes()
        ]
        
        global_cat = torch.cat(global_modes, dim=-1)
        mask = state_local.mask
        t = state_local.time

        # ...branches

        if self.mode_branched:
            h1, g1 = self.continuous_encoder(t, state_local.continuous, global_cat, mask)
            h2, g2 = self.discrete_encoder(t, state_local.discrete, global_cat, mask)
            g = torch.cat([g1, g2], dim=-1)
        else:
            h1 = state_local.continuous
            h2 = state_local.discrete
            g = global_cat


        # ...fusion

        if self.mode_fused:
            h = torch.cat([h1, h2], dim=-1)
            fused, _ = self.fused_encoder(t, h, g, mask)
            f1, f2 = torch.tensor_split(fused, 2, dim=-1)
            h_continuous = self.continuous_head(torch.cat([t, f1, h1], dim=-1))
            h_discrete = self.discrete_head(torch.cat([t, f2, h2], dim=-1))
        else:
            h_continuous = self.continuous_head(torch.cat([t, h1], dim=-1))
            h_discrete = self.discrete_head(torch.cat([t, h2], dim=-1))

        return TensorMultiModal(None, h_continuous, h_discrete, mask)
