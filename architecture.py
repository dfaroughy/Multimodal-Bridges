import torch
import math
from torch import nn
from torch.nn import functional as F
import torch.nn.utils.weight_norm as weight_norm


class HybridEPiC(nn.Module):
    """Permutation equivariant architecture for hybrid continuous-discrete models"""

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
        h = self.epic(t, x, k, context_continuous, context_discrete, mask)
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

        self.device = config.train.device

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
        self, t, x, k=None, context_continuous=None, context_discrete=None, mask=None
    ):
        t = t.to(self.device)
        x = x.to(self.device)
        k = k.to(self.device) if isinstance(k, torch.Tensor) else None

        context_continuous = (
            context_continuous.to(self.device)
            if isinstance(context_continuous, torch.Tensor)
            else None
        )
        context_discrete = (
            context_discrete.to(self.device)
            if isinstance(context_discrete, torch.Tensor)
            else None
        )
        mask = mask.to(self.device)

        x_local_emb, context_emb = self.embedding(
            t, x, k, context_continuous, context_discrete, mask
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


class InputEmbeddings(nn.Module):
    def __init__(self, config):
        super(InputEmbeddings, self).__init__()

        # ...dimensions:
        dim_features_continuous = config.data.dim.features_continuous
        dim_features_discrete = config.data.dim.features_discrete
        dim_context_continuous = config.data.dim.context_continuous
        dim_context_discrete = config.data.dim.context_discrete

        # ...vocab sizes for discrete data:
        vocab_size = config.data.vocab_size.features
        vocab_size_context = config.data.vocab_size.context

        # ...embedding types:
        embed_type_time = config.model.embedding_time
        embed_type_features_continuous = config.model.embedding_features_continuous
        embed_type_features_discrete = config.model.embedding_features_discrete
        embed_type_context_continuous = config.model.embedding_context_continuous
        embed_type_context_discrete = config.model.embedding_context_discrete

        # ...embedding dimensions:
        dim_time_emb = config.model.dim.emb_time
        dim_features_continuous_emb = (
            config.model.dim.emb_features_continuous
            if config.model.dim.emb_features_continuous
            else dim_features_continuous
        )
        dim_features_discrete_emb = config.model.dim.emb_features_discrete
        dim_context_continuous_emb = (
            config.model.dim.emb_context_continuous
            if config.model.dim.emb_context_continuous
            else dim_context_continuous
        )
        dim_context_discrete_emb = config.model.dim.emb_context_discrete

        # ...Time embeddings:

        if embed_type_time == "SinusoidalPositionalEncoding":
            self.embedding_time = SinusoidalPositionalEncoding(
                dim_time_emb, max_period=10000
            )
        elif embed_type_time == "Linear":
            self.embedding_time = nn.Linear(1, dim_time_emb)
        else:
            NotImplementedError(
                "Time embedding not implemented, choose from `SinusoidalPositionalEncoding`, `KANLinear` or `Linear`"
            )

        # ...Feature embeddings:

        if dim_features_continuous_emb:
            if embed_type_features_continuous == "Linear":
                self.embedding_continuous = nn.Linear(
                    dim_features_continuous, dim_features_continuous_emb
                )
            elif embed_type_features_continuous is None:
                self.embedding_continuous = nn.Identity()
            else:
                NotImplementedError(
                    "Continuous features embedding not implemented, choose from `kolmogorov-arnold`, `linear` or None"
                )

        if dim_features_discrete:
            if embed_type_features_discrete == "Embedding":
                self.embedding_discrete = nn.Embedding(
                    vocab_size, dim_features_discrete_emb
                )
            elif embed_type_features_discrete == "Linear":
                self.embedding_discrete = nn.Linear(
                    dim_features_discrete, dim_features_continuous_emb
                )
            else:
                NotImplementedError(
                    "Discrete context embedding not implemented, use `Linear` or KANLinear"
                )

        # ...Context embeddings:

        if dim_context_continuous:
            if embed_type_context_continuous == "Embedding":
                self.embedding_context_continuous = nn.Linear(
                    dim_context_continuous, dim_context_continuous_emb
                )
            elif embed_type_context_continuous is None:
                self.embedding_context_continuous = nn.Identity()
            else:
                NotImplementedError(
                    "Continuous context embedding not implemented, use `embedding` or None"
                )

        if dim_context_discrete:
            if embed_type_context_discrete == "Embedding":
                self.embedding_context_discrete = nn.Embedding(
                    vocab_size_context, dim_context_discrete_emb
                )
            elif embed_type_context_discrete == "Linear":
                self.embedding_context_discrete = nn.Linear(
                    dim_context_discrete, dim_features_continuous_emb
                )
            else:
                NotImplementedError(
                    "Discrete context embedding not implemented, use `Linear` or KANLinear"
                )

    def forward(
        self, t, x, k, context_continuous=None, context_discrete=None, mask=None
    ):
        """
        Forward pass of the particle embedding.

        Arguments:
        - t: Time input of shape (batch_size, 1) or (batch_size, 1, 1)
        - x: Particle continuous features of shape (batch_size, max_num_particles, dim_continuous)
        - k: Particle discrete features of shape (batch_size, max_num_particles, dim_discrete)
        - context_continuous: Continuous context features of shape (batch_size, dim_context_continuous)
        - context_discrete: Discrete context features of shape (batch_size, dim_context_discrete)
        - mask: Binary mask of shape (batch_size, max_num_particles, 1) indicating valid particles (1) or masked particles (0)

        Returns:
        - h: Embedded particles of shape (batch_size, N, dim_hidden), masked appropriately
        - context: Embedded context of shape (batch_size, dim_context)
        """

        # ...time:

        t_emb = self.embedding_time(t.squeeze(-1))
        t_context_emb = t_emb.clone().to(t_emb.device)
        if x.ndim == 3:
            t_emb = t_emb.unsqueeze(1).repeat(
                1, x.shape[1], 1
            )  # (b, dim_time_emb) -> (b, n, dim_time_emb)

        features = [t_emb]
        context = [t_context_emb]

        # ...features:

        if hasattr(self, "embedding_continuous"):
            emb = self.embedding_continuous(x)
            features.append(emb)

        if hasattr(self, "embedding_discrete"):
            emb = self.embedding_discrete(k.squeeze(-1))
            if x.ndim == 2:
                emb = emb.squeeze(1)
            features.append(emb)

        # ...context:

        if hasattr(self, "embedding_context_continuous"):
            emb = self.embedding_context_continuous(context_continuous)
            context.append(emb)

        if hasattr(self, "embedding_context_discrete"):
            emb = self.embedding_context_discrete(context_discrete).squeeze(1)
            context.append(emb)

        features = torch.cat(
            features, dim=-1
        )  # (b, n, dim_continuous_emb + dim_discrete_emb + dim_time_emb)
        context = torch.cat(
            context, dim=-1
        )  # (b, dim_context_continuous_emb + dim_context_discrete_emb + dim_time_emb)

        return features * mask, context


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


class InputEmbeddings(nn.Module):
    def __init__(self, config):
        super(InputEmbeddings, self).__init__()

        # ...dimensions:
        dim_features_continuous = config.data.dim.features_continuous
        dim_features_discrete = config.data.dim.features_discrete
        dim_context_continuous = config.data.dim.context_continuous
        dim_context_discrete = config.data.dim.context_discrete

        # ...vocab sizes for discrete data:
        vocab_size = config.data.vocab_size.features
        vocab_size_context = config.data.vocab_size.context

        # ...embedding types:
        embed_type_time = config.model.embedding.time
        embed_type_features_continuous = config.model.embedding.features_continuous
        embed_type_features_discrete = config.model.embedding.features_discrete
        embed_type_context_continuous = config.model.embedding.context_continuous
        embed_type_context_discrete = config.model.embedding.context_discrete

        # ...embedding dimensions:
        dim_time_emb = config.model.dim.emb_time
        dim_features_continuous_emb = (
            config.model.dim.emb_features_continuous
            if config.model.dim.emb_features_continuous
            else dim_features_continuous
        )
        dim_features_discrete_emb = config.model.dim.emb_features_discrete
        dim_context_continuous_emb = (
            config.model.dim.emb_context_continuous
            if config.model.dim.emb_context_continuous
            else dim_context_continuous
        )
        dim_context_discrete_emb = config.model.dim.emb_context_discrete

        # ...Time embeddings:

        if embed_type_time == "SinusoidalPositionalEncoding":
            self.time_embedding = SinusoidalPositionalEncoding(
                dim_time_emb, max_period=10000
            )
        elif embed_type_time == "KANLinear":
            self.time_embedding = KANLinear(1, dim_time_emb)
        elif embed_type_time == "Linear":
            self.time_embedding = nn.Linear(1, dim_time_emb)
        else:
            NotImplementedError(
                "Time embedding not implemented, choose from `SinusoidalPositionalEncoding`, `KANLinear` or `Linear`"
            )

        # ...Feature embeddings:

        if dim_features_continuous_emb:
            if embed_type_features_continuous == "Linear":
                self.embedding_continuous = nn.Linear(
                    dim_features_continuous, dim_features_continuous_emb
                )
            elif embed_type_features_continuous is None:
                self.embedding_continuous = nn.Identity()
            else:
                NotImplementedError(
                    "Continuous features embedding not implemented, choose from `kolmogorov-arnold`, `linear` or None"
                )

        if dim_features_discrete:
            if embed_type_features_discrete == "Embedding":
                self.embedding_discrete = nn.Embedding(
                    vocab_size, dim_features_discrete_emb
                )
            elif embed_type_features_discrete == "Linear":
                self.embedding_discrete = nn.Linear(
                    dim_features_discrete, dim_features_continuous_emb
                )
            else:
                NotImplementedError(
                    "Discrete context embedding not implemented, use `Linear` or KANLinear"
                )

        # ...Context embeddings:

        if dim_context_continuous:
            if embed_type_context_continuous == "Embedding":
                self.embedding_context_continuous = nn.Linear(
                    dim_context_continuous, dim_context_continuous_emb
                )
            elif embed_type_context_continuous is None:
                self.embedding_context_continuous = nn.Identity()
            else:
                NotImplementedError(
                    "Continuous context embedding not implemented, use `embedding` or None"
                )

        if dim_context_discrete:
            if embed_type_context_discrete == "Embedding":
                self.embedding_context_discrete = nn.Embedding(
                    vocab_size_context, dim_context_discrete_emb
                )
            elif embed_type_context_discrete == "Linear":
                self.embedding_context_discrete = nn.Linear(
                    dim_context_discrete, dim_features_continuous_emb
                )
            else:
                NotImplementedError(
                    "Discrete context embedding not implemented, use `Linear` or KANLinear"
                )

    def forward(
        self, t, x, k, context_continuous=None, context_discrete=None, mask=None
    ):
        """
        Forward pass of the particle embedding.

        Arguments:
        - t: Time input of shape (batch_size, 1) or (batch_size, 1, 1)
        - x: Particle continuous features of shape (batch_size, max_num_particles, dim_continuous)
        - k: Particle discrete features of shape (batch_size, max_num_particles, dim_discrete)
        - context_continuous: Continuous context features of shape (batch_size, dim_context_continuous)
        - context_discrete: Discrete context features of shape (batch_size, dim_context_discrete)
        - mask: Binary mask of shape (batch_size, max_num_particles, 1) indicating valid particles (1) or masked particles (0)

        Returns:
        - h: Embedded particles of shape (batch_size, N, dim_hidden), masked appropriately
        - context: Embedded context of shape (batch_size, dim_context)
        """

        # ...time:

        t_emb = self.time_embedding(t.squeeze(-1))
        t_context_emb = t_emb.clone().to(t_emb.device)
        if x.ndim == 3:
            t_emb = t_emb.unsqueeze(1).repeat(
                1, x.shape[1], 1
            )  # (b, dim_time_emb) -> (b, n, dim_time_emb)

        features = [t_emb]
        context = [t_context_emb]

        # ...features:

        if hasattr(self, "embedding_continuous"):
            emb = self.embedding_continuous(x)
            features.append(emb)

        if hasattr(self, "embedding_discrete"):
            emb = self.embedding_discrete(k.squeeze(-1))
            if x.ndim == 2:
                emb = emb.squeeze(1)
            features.append(emb)

        # ...context:

        if hasattr(self, "embedding_context_continuous"):
            emb = self.embedding_context_continuous(context_continuous)
            context.append(emb)

        if hasattr(self, "embedding_context_discrete"):
            emb = self.embedding_context_discrete(context_discrete).squeeze(1)
            context.append(emb)

        features = torch.cat(
            features, dim=-1
        )  # (b, n, dim_continuous_emb + dim_discrete_emb + dim_time_emb)
        context = torch.cat(
            context, dim=-1
        )  # (b, dim_context_continuous_emb + dim_context_discrete_emb + dim_time_emb)

        return features * mask, context


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
