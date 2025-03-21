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
            dim_input=dim_input,
            dim_output=dim_output,
            dim_context=dim_context,
            num_blocks=config.encoder.num_blocks,
            dim_hid_loc=config.encoder.dim_hidden_local,
            dim_hid_glob=config.encoder.dim_hidden_glob,
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

        h = self.epic(state_local.time, local_cat, global_cat, mask)

        if self.config.data.modality == "continuous":
            return TensorMultiModal(continuous=h, mask=mask)

        elif self.config.data.modality == "discrete":
            return TensorMultiModal(discrete=h, mask=mask)


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
        self.mode_fusion = num_blocks[2] > 0

        dim_context = (
            config.encoder.dim_emb_time
            + config.encoder.dim_emb_context_continuous
            + config.encoder.dim_emb_context_discrete * config.data.dim_context_discrete
        )

        self.continuous_leg = EPiCEncoder(
            dim_time=dim_time,
            dim_input=dim_cont,
            dim_output=dim_hid_loc[0],
            dim_hid_loc=dim_hid_loc[0],
            dim_hid_glob=dim_hid_glob[0],
            dim_context=dim_context,
            num_blocks=num_blocks[0],
            use_skip_connection=config.encoder.skip_connection,
            dropout=config.encoder.dropout,
        )

        self.discrete_leg = EPiCEncoder(
            dim_time=dim_time,
            dim_input=dim_disc,
            dim_output=dim_hid_loc[1],
            dim_hid_loc=dim_hid_loc[1],
            dim_hid_glob=dim_hid_glob[1],
            dim_context=dim_context,
            num_blocks=num_blocks[1],
            use_skip_connection=config.encoder.skip_connection,
            dropout=config.encoder.dropout,
        )

        if self.mode_fusion:
            self.fused_body = EPiCEncoder(
                dim_time=dim_time,
                dim_input=dim_hid_loc[0] + dim_hid_loc[1],
                dim_output=dim_hid_loc[2],
                dim_hid_loc=dim_hid_loc[2],
                dim_hid_glob=dim_hid_glob[2],
                dim_context=dim_context,
                num_blocks=num_blocks[2],
                use_skip_connection=config.encoder.skip_connection,
                dropout=config.encoder.dropout,
            )

        # ...mode heads:

        dim_head_cont = dim_hid_loc[2] // 2
        dim_head_disc = dim_hid_loc[2] // 2

        self.continuous_head = nn.Sequential(
            wn(nn.Linear(dim_time + dim_head_cont + dim_head_disc, dim_head_cont)),
            nn.GELU(),
            wn(nn.Linear(dim_head_cont, dim_out_continuous)),
        )

        self.discrete_head = nn.Sequential(
            wn(nn.Linear(dim_time + dim_head_disc + dim_head_cont, dim_head_disc)),
            nn.GELU(),
            wn(nn.Linear(dim_head_disc, dim_out_discrete)),
        )
        
        self.dim_hid_loc_cont = dim_hid_loc[0]

    def forward(
        self, state_local: TensorMultiModal, state_global: TensorMultiModal
    ) -> TensorMultiModal:

        global_modes = [getattr(state_global, mode) for mode in state_global.available_modes()]
        global_cat = torch.cat(global_modes, dim=-1)
        mask = state_local.mask

        # ...legs

        h1 = self.continuous_leg(state_local.time, state_local.continuous, global_cat, mask)
        h2 = self.discrete_leg(state_local.time, state_local.discrete, global_cat, mask)

        # ...fusion

        f = torch.cat([h1, h2], dim=-1)
        fused = self.fused_body(state_local.time, f, global_cat, mask)
        f1, f2 = torch.tensor_split(fused, 2, dim=-1)
        
        h_continuous = self.continuous_head(torch.cat([state_local.time, f1, h1 + h2], dim=-1))
        h_discrete = self.discrete_head(torch.cat([state_local.time, f2, h1 + h2], dim=-1))

        return TensorMultiModal(None, h_continuous, h_discrete, mask)


class MultiModalFusedEPiC(nn.Module):
    """Permutation equivariant architecture for multi-modal continuous-discrete models"""

    def __init__(self, config):
        super().__init__()

        self.config = config
        aug_factor = 2 if config.encoder.data_augmentation else 1

        dim_input = (
            config.encoder.dim_emb_time
            + config.encoder.dim_emb_continuous
            + config.encoder.dim_emb_discrete * config.data.dim_discrete * aug_factor
        )

        dim_context = (
            config.encoder.dim_emb_time
            + config.encoder.dim_emb_context_continuous
            + config.encoder.dim_emb_context_discrete * config.data.dim_context_discrete
        )

        self.multimode_epic = EPiCEncoderFused(
            dim_time=config.encoder.dim_emb_time,
            dim_input=dim_input,
            dim_output_continuous=config.data.dim_continuous,
            dim_output_discrete=config.data.vocab_size * config.data.dim_discrete,
            dim_context=dim_context,
            num_blocks=config.encoder.num_blocks,
            dim_hid_loc=config.encoder.dim_hidden_local,
            dim_hid_glob=config.encoder.dim_hidden_glob,
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
        head_continuous, head_discrete = self.multimode_epic(
            state_local.time, local_cat, global_cat, mask
        )
        return TensorMultiModal(None, head_continuous, head_discrete, mask)
