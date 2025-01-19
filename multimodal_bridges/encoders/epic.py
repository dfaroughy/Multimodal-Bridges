import torch
from torch import nn
from torch.nn import functional as F
import torch.nn.utils.weight_norm as weight_norm

from data.datasets import MultiModeState


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
        dim_output = (
            config.data.dim_continuous
            + config.data.vocab_size * config.data.dim_discrete 
        )
        dim_context = (
            config.encoder.dim_emb_time
            + config.encoder.dim_emb_context_continuous
            + config.encoder.dim_emb_context_discrete * config.data.dim_context_discrete
        )

        self.epic = EPiCNetwork(
            dim_input=dim_input,
            dim_output=dim_output,
            dim_context=dim_context,
            num_blocks=config.encoder.num_blocks,
            dim_hidden_local=config.encoder.dim_hidden_local,
            dim_hidden_global=config.encoder.dim_hidden_glob,
            use_skip_connection=config.encoder.skip_connection,
        )

    def forward(
        self, state_local: MultiModeState, state_global: MultiModeState
    ) -> MultiModeState:

        local_modes = [getattr(state_local, mode) for mode in state_local.available_modes()]
        global_modes = [getattr(state_global, mode) for mode in state_global.available_modes()]
        
        local_cat = torch.cat(local_modes, dim=-1)
        global_cat = torch.cat(global_modes, dim=-1)

        mask = state_local.mask

        h = self.epic(local_cat, global_cat, mask)
        head_continuous = h[..., : self.config.data.dim_continuous] if state_local.has_continuous else None
        head_discrete = h[..., self.config.data.dim_continuous :] if state_local.has_discrete else None
        return MultiModeState(None, head_continuous, head_discrete, mask)


class EPiCNetwork(nn.Module):
    def __init__(
        self,
        dim_input,
        dim_output=3,
        dim_context=0,
        num_blocks=6,
        dim_hidden_local=128,
        dim_hidden_global=10,
        use_skip_connection=False,
    ):
        super().__init__()

        # ...model params:
        self.num_blocks = num_blocks
        self.use_skip_connection = use_skip_connection

        # ...components:
        self.epic_proj = EPiC_Projection(
            dim_local=dim_input,
            dim_global=dim_context,
            dim_hidden_local=dim_hidden_local,
            dim_hidden_global=dim_hidden_global,
            pooling_fn=self.meansum_pool,
        )

        self.epic_layers = nn.ModuleList()

        for _ in range(self.num_blocks):
            self.epic_layers.append(
                EPiC_layer(
                    dim_local=dim_hidden_local,
                    dim_global=dim_hidden_global,
                    dim_hidden=dim_hidden_local,
                    dim_context=dim_context,
                    pooling_fn=self.meansum_pool,
                )
            )

        # ...output layer:

        self.output_layer = weight_norm(nn.Linear(dim_hidden_local, dim_output))

    def meansum_pool(self, mask, x_local, *x_global):
        """masked pooling local features with mean and sum
        the concat with global features
        """
        x_sum = (x_local * mask).sum(1, keepdim=False)
        x_mean = x_sum / mask.sum(1, keepdim=False)
        x_pool = torch.cat([x_mean, x_sum, *x_global], 1)
        return x_pool

    def forward(self, x_local, context=None, mask=None):
        # ...Projection network:
        x_local, x_global = self.epic_proj(x_local, context, mask)
        x_local_skip = x_local.clone() if self.use_skip_connection else 0
        x_global_skip = x_global.clone() if self.use_skip_connection else 0

        # ...EPiC layers:
        for i in range(self.num_blocks):
            x_local, x_global = self.epic_layers[i](x_local, x_global, context, mask)
            x_local += x_local_skip
            x_global += x_global_skip

        # ...output layer:
        h = self.output_layer(x_local)

        return h * mask  # [batch, points, feats]


class EPiC_Projection(nn.Module):
    def __init__(
        self, dim_local, dim_global, dim_hidden_local, dim_hidden_global, pooling_fn
    ):
        super(EPiC_Projection, self).__init__()

        self.pooling_fn = pooling_fn
        self.local_0 = weight_norm(nn.Linear(dim_local, dim_hidden_local))
        self.global_0 = weight_norm(
            nn.Linear(2 * dim_hidden_local + dim_global, dim_hidden_local)
        )  # local 2 global
        self.global_1 = weight_norm(nn.Linear(dim_hidden_local, dim_hidden_local))
        self.global_2 = weight_norm(nn.Linear(dim_hidden_local, dim_hidden_global))

    def forward(self, x_local, x_global, mask):
        """Input shapes:
         - x_local: (b, num_points, dim_local)
         - x_global = [b, dim_global]
        Out shapes:
         - x_local: (b, num_points, dim_hidden_local)
         - x_global = [b, dim_hidden_global]
        """
        x_local = F.leaky_relu(self.local_0(x_local))
        x_global = self.pooling_fn(mask, x_local, x_global)
        x_global = F.leaky_relu(self.global_0(x_global))
        x_global = F.leaky_relu(self.global_1(x_global))
        x_global = F.leaky_relu(self.global_2(x_global))
        return x_local * mask, x_global


class EPiC_layer(nn.Module):
    # based on https://github.com/uhh-pd-ml/EPiC-GAN/blob/main/models.py
    def __init__(
        self,
        dim_local,
        dim_global,
        dim_hidden,
        dim_context,
        pooling_fn,
        activation_fn=F.leaky_relu,
    ):
        super(EPiC_layer, self).__init__()

        self.pooling_fn = pooling_fn
        self.activation_fn = activation_fn
        self.fc_global1 = weight_norm(
            nn.Linear(int(2 * dim_local) + dim_global + dim_context, dim_hidden)
        )
        self.fc_global2 = weight_norm(nn.Linear(dim_hidden, dim_global))
        self.fc_local1 = weight_norm(
            nn.Linear(dim_local + dim_global + dim_context, dim_hidden)
        )
        self.fc_local2 = weight_norm(nn.Linear(dim_hidden, dim_local))

    def forward(self, x_local, x_global, context, mask):
        """Input/Output shapes:
        - x_local: (b, num_points, dim_local)
        - x_global = [b, dim_global]
        - context = [b, dim_context]
        """
        num_points, dim_global, dim_context = (
            x_local.size(1),
            x_global.size(1),
            context.size(1),
        )
        x_pooledCATglobal = self.pooling_fn(mask, x_local, x_global, context)
        x_global1 = self.activation_fn(self.fc_global1(x_pooledCATglobal))
        x_global = self.activation_fn(
            self.fc_global2(x_global1) + x_global
        )  # with residual connection before AF
        x_global2local = x_global.view(-1, 1, dim_global).repeat(
            1, num_points, 1
        )  # first add dimension, than expand it
        x_context2local = context.view(-1, 1, dim_context).repeat(1, num_points, 1)
        x_localCATglobal = torch.cat([x_local, x_global2local, x_context2local], 2)
        x_local1 = self.activation_fn(self.fc_local1(x_localCATglobal))
        x_local = self.activation_fn(self.fc_local2(x_local1) + x_local)

        return x_local * mask, x_global
