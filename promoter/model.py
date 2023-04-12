import numpy as np
import mlxu

import jax
import jax.numpy as jnp
import flax
from flax import linen as nn
import einops
import mlxu.jax_utils as jax_utils


class MLP(nn.Module):
    output_dim: int
    hidden_dim: int = 512
    num_layers: int = 2

    @nn.compact
    def __call__(self, input_tensor):
        x = input_tensor
        for _ in range(self.num_layers):
            x = nn.Dense(self.hidden_dim)(x)
            x = jax.nn.gelu(x)
        return nn.Dense(self.output_dim)(x)


class ConvBlock(nn.Module):
    channels: int
    kernel_size: int = 5
    group_norm_group_size: int = 16
    dropout: float = 0.1

    @nn.compact
    def __call__(self, inputs, deterministic=False):
        x = nn.Conv(
            features=self.channels,
            kernel_size=(self.kernel_size, ),
            padding="SAME",
        )(inputs)
        x = jax.nn.gelu(x)
        x = nn.GroupNorm(
            num_groups=None,
            group_size=self.group_norm_group_size,
        )(x)
        x = nn.Dropout(
            rate=self.dropout,
            deterministic=deterministic,
        )(x)
        return x


class TransformerBlock(nn.Module):
    embedding_dim: int
    num_heads: int
    mlp_dim: int
    dropout: float = 0.1

    @nn.compact
    def __call__(self, inputs, deterministic=False):
        assert self.embedding_dim % self.num_heads == 0
        x = nn.LayerNorm()(inputs)
        xk = nn.Dense(features=self.embedding_dim, use_bias=False)(x)
        xq = nn.Dense(features=self.embedding_dim, use_bias=False)(x)
        xv = nn.Dense(features=self.embedding_dim, use_bias=False)(x)

        xk = einops.rearrange(xk, '... (h d) -> ... h d', h=self.num_heads)
        xq = einops.rearrange(xq, '... (h d) -> ... h d', h=self.num_heads)
        xv = einops.rearrange(xv, '... (h d) -> ... h d', h=self.num_heads)
        attention_weights = einops.einsum(xq, xk, '... q h d, ... k h d -> ... h q k')

        attention_weights = attention_weights / jnp.sqrt(self.embedding_dim // self.num_heads)
        attention_weights = jax.nn.softmax(attention_weights, axis=-1)
        attention_weights = nn.Dropout(
            rate=self.dropout, deterministic=deterministic
        )(attention_weights)
        attention_output = einops.einsum(attention_weights, xv, '... h q k, ... k h d -> ... q h d')
        attention_output = einops.rearrange(attention_output, '... h d -> ... (h d)')
        attention_output = nn.Dense(features=self.embedding_dim, use_bias=False)(attention_output)
        attention_output = nn.Dropout(
            rate=self.dropout, deterministic=deterministic
        )(attention_output)

        mlp_inputs = attention_output + inputs
        x = nn.LayerNorm()(mlp_inputs)
        x = nn.Dense(features=self.mlp_dim)(x)
        x = jax.nn.gelu(x)
        x = nn.Dense(features=self.embedding_dim)(x)
        x = nn.Dropout(
            rate=self.dropout, deterministic=deterministic
        )(x)
        x = x + mlp_inputs
        return x


class Backbone(nn.Module):
    config_updates: ... = None

    @staticmethod
    @nn.nowrap
    def get_default_config(updates=None):
        config = mlxu.config_dict()
        config.embedding_dim = 1024
        config.transformer_blocks = 5
        config.n_heads = 8
        config.dropout = 0.1
        if updates is not None:
            config.update(mlxu.config_dict(updates).copy_and_resolve_references())
        return config

    def setup(self):
        self.config = self.get_default_config(self.config_updates)

    @nn.compact
    def __call__(self, x, deterministic=False):
        x = ConvBlock(channels=256, dropout=self.config.dropout)(x, deterministic=deterministic)
        x = ConvBlock(channels=512, dropout=self.config.dropout)(x, deterministic=deterministic)
        x = ConvBlock(channels=self.config.embedding_dim, dropout=self.config.dropout)(x, deterministic=deterministic)
        cls_embedding = self.param(
            "cls_embedding",
            nn.initializers.normal(stddev=0.02), (1, 1, self.config.embedding_dim)
        )
        cls_embedding = jnp.broadcast_to(cls_embedding, (x.shape[0], 1, self.config.embedding_dim))
        x = jnp.concatenate([cls_embedding, x], axis=1)

        for _ in range(self.config.transformer_blocks):
            x = TransformerBlock(
                embedding_dim=self.config.embedding_dim,
                num_heads=self.config.n_heads,
                mlp_dim=4 * self.config.embedding_dim,
                dropout=self.config.dropout,
            )(x, deterministic=deterministic)

        return x[..., 0, :]


class PretrainNetwork(nn.Module):
    config_updates: ... = None

    @staticmethod
    @nn.nowrap
    def get_default_config(updates=None):
        config = mlxu.config_dict()
        config.output_head_num_layers = 2
        config.output_head_hidden_dim = 512
        config.backbone = Backbone.get_default_config()
        if updates is not None:
            config.update(mlxu.config_dict(updates).copy_and_resolve_references())
        return config

    def setup(self):
        self.config = self.get_default_config(self.config_updates)
        self.backbone = Backbone(self.config.backbone)
        self.output_ln = nn.LayerNorm()
        self.k562_head = MLP(
            output_dim=20,
            hidden_dim=self.config.output_head_hidden_dim,
            num_layers=self.config.output_head_num_layers
        )
        self.hepg2_head = MLP(
            output_dim=20,
            hidden_dim=self.config.output_head_hidden_dim,
            num_layers=self.config.output_head_num_layers
        )
        self.mpra_head = MLP(
            output_dim=12,
            hidden_dim=self.config.output_head_hidden_dim,
            num_layers=self.config.output_head_num_layers
        )

    @nn.compact
    def __call__(self, sure_inputs, mpra_inputs, deterministic=False):
        sure_x = self.output_ln(
            self.backbone(sure_inputs, deterministic=deterministic)
        )
        sure_k562_ouptut = self.k562_head(sure_x)
        sure_hepg2_ouptut = self.hepg2_head(sure_x)

        mpra_x = self.output_ln(
            self.backbone(mpra_inputs, deterministic=deterministic)
        )
        mpra_output = self.mpra_head(mpra_x)

        return sure_k562_ouptut, sure_hepg2_ouptut, mpra_output

    @nn.nowrap
    def rng_keys(self):
        return ('params', 'dropout')


class FinetuneNetwork(nn.Module):
    config_updates: ... = None

    @staticmethod
    @nn.nowrap
    def get_default_config(updates=None):
        config = mlxu.config_dict()
        config.output_head_num_layers = 2
        config.output_head_hidden_dim = 512
        config.backbone = Backbone.get_default_config()
        if updates is not None:
            config.update(mlxu.config_dict(updates).copy_and_resolve_references())
        return config

    def setup(self):
        self.config = self.get_default_config(self.config_updates)
        self.backbone = Backbone(self.config.backbone)
        self.output_ln = nn.LayerNorm()
        self.thp1_head = MLP(
            output_dim=1,
            hidden_dim=self.config.output_head_hidden_dim,
            num_layers=self.config.output_head_num_layers
        )
        self.jurkat_head = MLP(
            output_dim=1,
            hidden_dim=self.config.output_head_hidden_dim,
            num_layers=self.config.output_head_num_layers
        )
        self.k562_head = MLP(
            output_dim=1,
            hidden_dim=self.config.output_head_hidden_dim,
            num_layers=self.config.output_head_num_layers
        )

    @nn.compact
    def __call__(self, inputs, deterministic=False):
        x = self.backbone(inputs, deterministic=deterministic)
        x = self.output_ln(x)
        thp1_output = self.thp1_head(x).squeeze(-1)
        jurkat_output = self.jurkat_head(x).squeeze(-1)
        k562_output = self.k562_head(x).squeeze(-1)
        return thp1_output, jurkat_output, k562_output

    @nn.nowrap
    def rng_keys(self):
        return ('params', 'dropout')
