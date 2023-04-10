import re
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
    embedding_dim: int = 1024
    transformer_blocks: int = 5
    n_heads: int = 8
    dropout: float = 0.1

    @nn.compact
    def __call__(self, x, deterministic=False):
        x = ConvBlock(channels=256, dropout=self.dropout)(x, deterministic=deterministic)
        x = ConvBlock(channels=512, dropout=self.dropout)(x, deterministic=deterministic)
        x = ConvBlock(channels=self.embedding_dim, dropout=self.dropout)(x, deterministic=deterministic)
        cls_embedding = self.param(
            "cls_embedding",
            nn.initializers.normal(stddev=0.02), (1, 1, self.embedding_dim)
        )
        cls_embedding = jnp.broadcast_to(cls_embedding, (x.shape[0], 1, self.embedding_dim))
        x = jnp.concatenate([cls_embedding, x], axis=1)

        for _ in range(self.transformer_blocks):
            x = TransformerBlock(
                embedding_dim=self.embedding_dim,
                num_heads=self.n_heads,
                mlp_dim=4 * self.embedding_dim,
                dropout=self.dropout,
            )(x, deterministic=deterministic)

        return x[..., 0, :]


class PretrainNetwork(nn.Module):
    embedding_dim: int = 1024
    transformer_blocks: int = 5
    n_heads: int = 8
    dropout: float = 0.1
    output_head_num_layers: int = 2
    output_head_hidden_dim: int = 512

    @nn.compact
    def __call__(self, sure_inputs, mpra_inputs, deterministic=False):
        backbone = Backbone(
            embedding_dim=self.embedding_dim,
            transformer_blocks=self.transformer_blocks,
            n_heads=self.n_heads,
            dropout=self.dropout,
        )
        sure_x = nn.LayerNorm()(
            backbone(sure_inputs, deterministic=deterministic)
        )
        sure_k562_ouptut = MLP(
            output_dim=20,
            hidden_dim=self.output_head_hidden_dim,
            num_layers=self.output_head_num_layers
        )(sure_x)
        sure_hepg2_ouptut = MLP(
            output_dim=20,
            hidden_dim=self.output_head_hidden_dim,
            num_layers=self.output_head_num_layers
        )(sure_x)

        mpra_x = nn.LayerNorm()(
            backbone(mpra_inputs, deterministic=deterministic)
        )

        mpra_output = MLP(
            output_dim=12,
            hidden_dim=self.output_head_hidden_dim,
            num_layers=self.output_head_num_layers
        )(mpra_x)

        return sure_k562_ouptut, sure_hepg2_ouptut, mpra_output

    @nn.nowrap
    def rng_keys(self):
        return ('params', 'dropout')


def get_weight_decay_mask(exclusions):
    """ Return a weight decay mask function that computes the pytree masks
        according to the given exclusion rules.
    """
    def decay(name, _):
        for rule in exclusions:
            if re.search(rule, name) is not None:
                return False
        return True

    def weight_decay_mask(params):
        return jax_utils.named_tree_map(decay, params, sep='/')

    return weight_decay_mask