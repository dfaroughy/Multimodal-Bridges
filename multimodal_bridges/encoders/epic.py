import torch
from torch import nn
from torch.nn import functional as F
import torch.nn.utils.weight_norm as wn

from tensorclass import TensorMultiModal


class MultiModalEPiC(nn.Module):
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

        self.multimode_epic = EPiCEncoder(
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


class EPiCEncoder(nn.Module):
    def __init__(
        self,
        dim_time: int,
        dim_input: int,
        dim_output_continuous: int = 3,
        dim_output_discrete: int = 0,
        dim_context: int = 0,
        num_blocks: int = 6,
        dim_hid_loc: int = 128,
        dim_hid_glob: int = 10,
        use_skip_connection: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()

        # ...model params:
        self.num_blocks = num_blocks
        self.use_skip_connection = use_skip_connection

        # ...components:
        self.epic_proj = EPiCProjection(
            dim_time=dim_time,
            dim_loc=dim_input,
            dim_glob=dim_context,
            dim_hid_loc=dim_hid_loc,
            dim_hid_glob=dim_hid_glob,
            pooling_fn=self._meansum_pool,
            dropout=dropout,
        )

        self.epic_layers = nn.ModuleList()

        for _ in range(self.num_blocks):
            self.epic_layers.append(
                EPiCLayer(
                    dim_loc=dim_hid_loc,
                    dim_glob=dim_hid_glob,
                    dim_hid=dim_hid_loc,
                    dim_cond=dim_context,
                    pooling_fn=self._meansum_pool,
                    dropout=dropout,
                )
            )

        self.output_layer = wn(
            nn.Linear(dim_time + dim_hid_loc + dim_hid_glob, dim_hid_loc)
        )

        # ...mode heads:

        self.continuous_head = nn.Sequential(
            wn(nn.Linear(dim_time + dim_hid_loc // 2, dim_hid_loc // 2)),
            nn.GELU(),
            wn(nn.Linear(dim_hid_loc // 2, dim_output_continuous)),
            nn.Dropout(dropout),
        )

        self.discrete_head = nn.Sequential(
            wn(nn.Linear(dim_time + dim_hid_loc // 2, dim_hid_loc // 2)),
            nn.GELU(),
            wn(nn.Linear(dim_hid_loc // 2, dim_output_discrete)),
            nn.Dropout(dropout),
        )

    def forward(self, time_local, x_local, context=None, mask=None):
        # ...Projection network:
        x_local, x_global = self.epic_proj(time_local, x_local, context, mask)
        x_local_skip = x_local.clone() if self.use_skip_connection else 0
        x_global_skip = x_global.clone() if self.use_skip_connection else 0

        # ...EPiC layers:
        for i in range(self.num_blocks):
            x_local, x_global = self.epic_layers[i](x_local, x_global, context, mask)
            x_local += x_local_skip
            x_global += x_global_skip

        # ... final layer
        x_global = self._broadcast_global(x_global, local=x_local)
        h = torch.cat([time_local, x_local, x_global], dim=-1)
        h = self.output_layer(h)

        # ...output heads:
        h1, h2 = torch.tensor_split(x_local, 2, dim=-1)
        h_continuous = self.continuous_head(torch.cat([time_local, h1], dim=-1))
        h_discrete = self.discrete_head(torch.cat([time_local, h2], dim=-1))

        return h_continuous * mask, h_discrete * mask  # [batch, points, feats]

    def _meansum_pool(self, mask, x_local, *x_global, scale=0.01):
        """masked pooling local features with mean and sum
        the concat with global features
        """
        x_sum = (x_local * mask).sum(1, keepdim=False)
        x_mean = x_sum / mask.sum(1, keepdim=False)
        x_pool = torch.cat([x_mean, x_sum * scale, *x_global], 1)
        return x_pool

    def _broadcast_global(self, x, local):
        dim = x.size(1)
        D = local.size(1)
        return x.view(-1, 1, dim).repeat(1, D, 1)


class EPiCProjection(nn.Module):
    def __init__(
        self,
        dim_time: int,
        dim_loc: int,
        dim_glob: int,
        dim_hid_loc: int,
        dim_hid_glob: int,
        pooling_fn: callable,
        dropout: float = 0.0,
    ):
        super(EPiCProjection, self).__init__()

        self.pooling_fn = pooling_fn

        self.mlp_local = nn.Sequential(
            wn(nn.Linear(dim_time + dim_loc, dim_hid_loc)),
            nn.GELU(),
            wn(nn.Linear(dim_hid_loc, dim_hid_loc)),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.mlp_global = nn.Sequential(
            wn(nn.Linear(2 * dim_hid_loc + dim_glob, dim_hid_loc)),
            nn.GELU(),
            wn(nn.Linear(dim_hid_loc, dim_hid_glob)),
            nn.GELU(),
        )

    def forward(self, time, x_local, x_global, mask):
        """Input shapes:
         - x_local: (B, D, dim_local)
         - x_global = (B, dim_global)
        Out shapes:
         - x_local: (B, D, dim_hidden_local)
         - x_global = (B, dim_hidden_global)
        """

        x_local = self.mlp_local(torch.cat([time, x_local], dim=-1))
        x_global = self.pooling_fn(mask, x_local, x_global)
        x_global = self.mlp_global(x_global)

        return x_local, x_global


class EPiCLayer(nn.Module):
    # based on https://github.com/uhh-pd-ml/EPiC-GAN/blob/main/models.py
    def __init__(
        self,
        dim_loc: int,
        dim_glob: int,
        dim_hid: int,
        dim_cond: int,
        pooling_fn: callable,
        activation_fn: callable = F.leaky_relu,
        dropout: float = 0.0,
    ):
        super(EPiCLayer, self).__init__()

        self.pooling_fn = pooling_fn
        self.act_fn = activation_fn

        self.fc_glob1 = wn(nn.Linear(2 * dim_loc + dim_glob + dim_cond, dim_hid))
        self.fc_glob2 = wn(nn.Linear(dim_hid, dim_glob))
        self.fc_loc1 = wn(nn.Linear(dim_loc + dim_glob + dim_cond, dim_hid))
        self.fc_loc2 = wn(nn.Linear(dim_hid, dim_loc))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x_local, x_global, context, mask):
        """Input/Output shapes:
        - x_local: (b, num_points, dim_loc)
        - x_global = [b, dim_glob]
        - context = [b, dim_cond]
        """
        # ...global features
        global_hidden = self.pooling_fn(mask, x_local, x_global, context)
        global_hidden = self.act_fn(self.fc_glob1(global_hidden))
        x_global += self.fc_glob2(global_hidden)  # skip connection
        global_hidden = self.dropout(self.act_fn(x_global))

        # ...broadcast global/context features to each particle
        global2local = self._broadcast_global(x_global, local=x_local)
        context2local = self._broadcast_global(context, local=x_local)

        # ...local features
        local_hidden = torch.cat([x_local, global2local, context2local], 2)
        local_hidden = self.act_fn(self.fc_loc1(local_hidden))
        x_local += self.fc_loc2(local_hidden)  # skip connection
        local_hidden = self.dropout(self.act_fn(x_local))

        return local_hidden, global_hidden

    def _broadcast_global(self, x, local):
        dim = x.size(1)
        D = local.size(1)
        return x.view(-1, 1, dim).repeat(1, D, 1)
