import torch
import math
from torch import nn
from torch.nn import functional as F
import torch.nn.utils.weight_norm as weight_norm

from models.utils import InputEmbeddings

class MultiModalEPiC(nn.Module):
    """Permutation equivariant architecture for multi-modal continuous-discrete models"""

    def __init__(self, config):
        super().__init__()
        self.dim_features_continuous = config.data.dim.features_continuous
        self.dim_features_discrete = config.data.dim.features_discrete
        self.vocab_size = config.data.vocab_size.features
        self.epic = EPiC(config)
        self.add_discrete_head = config.model.add_discrete_head
        if self.add_discrete_head:
            self.fc_layer = nn.Sequential(
                nn.Linear(
                    self.dim_features_discrete * self.vocab_size,
                    self.dim_features_discrete * self.vocab_size,
                ),
                nn.SELU(),
                nn.Linear(
                    self.dim_features_discrete * self.vocab_size,
                    self.dim_features_discrete * self.vocab_size,
                ),
            )

    def forward(
        self, t, x, k, mask=None, context_continuous=None, context_discrete=None
    ):
        h = self.epic(t, x, k, mask, context_continuous, context_discrete)
        continuous_head = h[..., : self.dim_features_continuous]
        discrete_head = h[..., self.dim_features_continuous :]
        absorbing_head = mask  # TODO

        if self.add_discrete_head:
            return continuous_head, self.fc_layer(discrete_head), absorbing_head
        else:
            return continuous_head, discrete_head, absorbing_head


class EPiC(nn.Module):
    """Model wrapper for EPiC Network

    Forward pass:
        - t: time input of shape (b, 1)
        - x: continuous features of shape (b, n, dim_continuous)
        - k: discrete features of shape (b,  n, dim_discrete)
        - context: context features of shape (b, dim_context)
        - mask: binary mask of shape (b, n, 1) indicating valid particles (1) or masked particles (0)
    """

    def __init__(self, config):
        super().__init__()

        # ...data dimensions:
        self.dim_features_continuous = config.data.dim.features_continuous
        self.dim_features_discrete = config.data.dim.features_discrete
        dim_context_continuous = config.data.dim.context_continuous
        self.vocab_size = config.data.vocab_size.features

        # ...embedding dimensions:
        dim_time_emb = config.model.dim.emb_time
        dim_features_continuous_emb = (
            config.model.dim.emb_features_continuous
            if config.model.dim.emb_features_continuous
            else self.dim_features_continuous
        )
        dim_features_discrete_emb = config.model.dim.emb_features_discrete
        dim_context_continuous_emb = (
            config.model.dim.emb_context_continuous
            if config.model.dim.emb_context_continuous
            else dim_context_continuous
        )
        dim_context_discrete_emb = config.model.dim.emb_context_discrete

        # ...components:
        self.embedding = InputEmbeddings(config)
        self.epic = EPiCNetwork(
            dim_input=dim_time_emb
            + dim_features_continuous_emb
            + dim_features_discrete_emb,
            dim_output=self.dim_features_continuous
            + self.dim_features_discrete * self.vocab_size,
            dim_context=dim_time_emb
            + dim_context_continuous_emb
            + dim_context_discrete_emb,
            num_blocks=config.model.num_blocks,
            dim_hidden_local=config.model.dim.hidden_local,
            dim_hidden_global=config.model.dim.hidden_glob,
            use_skip_connection=config.model.skip_connection,
        )

    def forward(
        self, t, x, k=None, mask=None, context_continuous=None, context_discrete=None
    ):
        context_continuous = (
            context_continuous.to(t.device)
            if isinstance(context_continuous, torch.Tensor)
            else None
        )
        context_discrete = (
            context_discrete.to(t.device)
            if isinstance(context_discrete, torch.Tensor)
            else None
        )

        x_local_emb, context_emb = self.embedding(
            t, x, k, mask, context_continuous, context_discrete
        )
        h = self.epic(x_local_emb, context_emb, mask)
        return h


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
