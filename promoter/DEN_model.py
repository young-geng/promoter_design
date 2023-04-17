import numpy as np
import mlxu

import jax
import jax.numpy as jnp
import flax
from flax import linen as nn
import einops
import mlxu.jax_utils as jax_utils

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

# UNet architecture for sequence generation
class UNet(nn.Module):
    seq_length: int
    alphabet_size: int
    latent_size: int
    num_classes: int
    num_samples: int

    @nn.compact
    def __call__(self, inputs, deterministic=False, temperature=1.0):
        assert inputs.shape[1] == self.latent_size

        seq_length_rounded_to_power_of_2 = int(2 ** np.ceil(np.log2(self.seq_length)))

        # initial dense layer
        x = nn.Dense(features=self.latent_size + self.num_classes)(inputs)
        x = jax.nn.gelu(x).reshape((inputs.shape[0], 1, self.latent_size + self.num_classes))

        # downsize
        x1 = ConvBlock(channels=128)(x, deterministic=deterministic)
        x = nn.max_pool(x1, window_shape=(2, ), strides=(2, ), padding="SAME")

        x2 = ConvBlock(channels=256)(x, deterministic=deterministic)
        x = nn.max_pool(x2, window_shape=(2, ), strides=(2, ), padding="SAME")

        x3 = ConvBlock(channels=512)(x, deterministic=deterministic)
        x = nn.max_pool(x3, window_shape=(2, ), strides=(2, ), padding="SAME")

        x4 = ConvBlock(channels=1024)(x, deterministic=deterministic)
        x = nn.max_pool(x4, window_shape=(2, ), strides=(2, ), padding="SAME")

        x5 = ConvBlock(channels=2048)(x, deterministic=deterministic)
        x = nn.max_pool(x5, window_shape=(2, ), strides=(2, ), padding="SAME")

        x6 = ConvBlock(channels=4096)(x, deterministic=deterministic)
        
        # upsize
        x = jax.image.resize(x6, (x6.shape[0], x6.shape[1] * 2, x6.shape[2]), method="nearest")
        x = jnp.concatenate([x, x5], axis=2)
        x = ConvBlock(channels=2048)(x, deterministic=deterministic)

        x = jax.image.resize(x, (x.shape[0], x.shape[1] * 2, x.shape[2]), method="nearest")
        x = jnp.concatenate([x, x4], axis=2)
        x = ConvBlock(channels=1024)(x, deterministic=deterministic)

        x = jax.image.resize(x, (x.shape[0], x.shape[1] * 2, x.shape[2]), method="nearest")
        x = jnp.concatenate([x, x3], axis=2)
        x = ConvBlock(channels=512)(x, deterministic=deterministic)

        x = jax.image.resize(x, (x.shape[0], x.shape[1] * 2, x.shape[2]), method="nearest")
        x = jnp.concatenate([x, x2], axis=2)
        x = ConvBlock(channels=256)(x, deterministic=deterministic)

        x = jax.image.resize(x, (x.shape[0], x.shape[1] * 2, x.shape[2]), method="nearest")
        x = jnp.concatenate([x, x1], axis=2)
        x = ConvBlock(channels=128)(x, deterministic=deterministic)

        # final conv
        x = nn.Conv(
            features=self.alphabet_size,
            kernel_size=(1, ),
            padding="SAME",
        )(x)

        # crop to original size
        x = x[:, :, :self.seq_length]

        # get num_samples samples using gumbel softmax + straight through estimator
        x_rep = jnp.repeat(x, self.num_samples, axis=0).reshape((inputs.shape[0], self.num_samples, self.seq_length, self.alphabet_size))
        gumbel_rng = self.make_rng("gumbel")
        gumbels = jax.random.gumbel(gumbel_rng, x_rep.shape)
        gumbels = jax.nn.softmax((x_rep + gumbels) / temperature, axis=-1)
        y_soft = jax.nn.softmax(gumbels, axis=-1)
        y_hard = jax.nn.one_hot(jnp.argmax(y_soft, axis=-1), self.alphabet_size)
        samples = jax.lax.stop_gradient(y_hard - y_soft) + y_soft

        # get PWM by taking softmax of logits
        x = jnp.nn.softmax(x, axis=-1)

        return x, samples        