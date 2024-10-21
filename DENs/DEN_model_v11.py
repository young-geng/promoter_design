import numpy as np
import pdb
import mlxu

import jax
import jax.numpy as jnp
import flax
from flax import linen as nn
import einops
import mlxu.jax_utils as jax_utils

from .model import FinetuneNetwork
from .DEN_loss_v11 import DEN_loss

class ConvBlock(nn.Module):
    channels: int
    kernel_size: int = 5
    group_norm_group_size: int = 16
    dropout: float = 0.1
    padding: str = "SAME"

    @nn.compact
    def __call__(self, inputs, deterministic=False):
        x = nn.Conv(
            features=self.channels,
            kernel_size=(self.kernel_size, ),
            padding=self.padding,
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
    
class DeconvBlock(nn.Module):
    channels: int
    kernel_size: int = 7
    stride: int = 2
    group_norm_group_size: int = 16
    dropout: float = 0.1
    padding: str = "VALID"

    @nn.compact
    def __call__(self, inputs, deterministic=False):
        x = nn.ConvTranspose(
            features=self.channels,
            kernel_size=(self.kernel_size, ),
            strides=(self.stride, ),
            padding=self.padding,
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
    config_updates: ... = None

    @staticmethod
    @nn.nowrap
    def get_default_config(updates=None):
        config = mlxu.config_dict()
        config.seq_length = 250
        config.alphabet_size = 4
        config.latent_size = 100
        config.num_classes = 3
        config.num_samples = 10
        if updates is not None:
            config.update(mlxu.config_dict(updates).copy_and_resolve_references())
        return config

    def setup(self):
        self.config = self.get_default_config(self.config_updates)
        self.seq_length = self.config.seq_length
        self.alphabet_size = self.config.alphabet_size
        self.latent_size = self.config.latent_size
        self.num_classes = self.config.num_classes
        self.num_samples = self.config.num_samples

    @nn.compact
    def __call__(self, inputs, deterministic=False, temperature=1.0):
        assert inputs.shape[1] == self.latent_size

        seq_length_rounded_to_power_of_2 = int(2 ** np.ceil(np.log2(self.seq_length)))

        # initial dense layer
        x = nn.Dense(features=self.alphabet_size * seq_length_rounded_to_power_of_2)(inputs)
        x = jax.nn.gelu(x).reshape((inputs.shape[0], seq_length_rounded_to_power_of_2, self.alphabet_size))

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
        x = x[:, :self.seq_length]

        # get num_samples samples using gumbel softmax + straight through estimator
        x_rep = jnp.repeat(x, self.num_samples, axis=0).reshape((inputs.shape[0], self.num_samples, self.seq_length, self.alphabet_size))
        gumbel_rng = self.make_rng("gumbel")
        gumbels = jax.random.gumbel(gumbel_rng, x_rep.shape)
        gumbels = jax.nn.softmax((x_rep + gumbels) / temperature, axis=-1)
        y_soft = jax.nn.softmax(gumbels, axis=-1)
        y_hard = jax.nn.one_hot(jnp.argmax(y_soft, axis=-1), self.alphabet_size)
        samples = jax.lax.stop_gradient(y_hard - y_soft) + y_soft

        # get PWM by taking softmax of logits
        x = jax.nn.softmax(x, axis=-1)

        return x, samples

class DeconvConvNet(nn.Module):
    config_updates: ... = None

    @staticmethod
    @nn.nowrap
    def get_default_config(updates=None):
        config = mlxu.config_dict()
        config.seq_length = 250
        config.alphabet_size = 4
        config.latent_size = 200
        config.num_classes = 3
        config.num_samples = 10
        if updates is not None:
            config.update(mlxu.config_dict(updates).copy_and_resolve_references())
        return config

    def setup(self):
        self.config = self.get_default_config(self.config_updates)
        self.seq_length = self.config.seq_length
        self.alphabet_size = self.config.alphabet_size
        self.latent_size = self.config.latent_size
        self.num_classes = self.config.num_classes
        self.num_samples = self.config.num_samples

    @nn.compact
    def __call__(self, inputs, deterministic=False, temperature=1.0):
        assert inputs.shape[1] == self.latent_size
        assert self.seq_length == 250 # structure works properly only for 250bp seq generation
        
        # initial dense layer
        x = nn.Dense(features=12 * 384)(inputs)
        x = jax.nn.gelu(x).reshape((inputs.shape[0], 12, 384)) # x is (12, 384)
        
        # deconv layers
        x = DeconvBlock(channels=256, kernel_size=7)(x, deterministic=deterministic) # x is (29, 256)
        x = DeconvBlock(channels=192, kernel_size=5)(x, deterministic=deterministic) # x is (61, 192)
        x = DeconvBlock(channels=192, kernel_size=5)(x, deterministic=deterministic) # x is (125, 192)
        x = DeconvBlock(channels=128, kernel_size=7)(x, deterministic=deterministic) # x is (255, 128)
        
        x = x[:, :self.seq_length] # x is (250, 128)
        
        # conv layers
        x = ConvBlock(channels=128, kernel_size=7)(x, deterministic=deterministic)
        x = ConvBlock(channels=96, kernel_size=7)(x, deterministic=deterministic)
        x = ConvBlock(channels=64, kernel_size=7)(x, deterministic=deterministic)
        
        # final conv
        x = nn.Conv(
            features=self.alphabet_size,
            kernel_size=(1, ),
            padding="SAME",
        )(x)
        
        # get num_samples samples using gumbel softmax + straight through estimator
        x_rep = jnp.repeat(x, self.num_samples, axis=0).reshape((inputs.shape[0], self.num_samples, self.seq_length, self.alphabet_size))
        gumbel_rng = self.make_rng("gumbel")
        gumbels = jax.random.gumbel(gumbel_rng, x_rep.shape)
        gumbels = jax.nn.softmax((x_rep + gumbels) / temperature, axis=-1)
        y_soft = jax.nn.softmax(gumbels, axis=-1)
        y_hard = jax.nn.one_hot(jnp.argmax(y_soft, axis=-1), self.alphabet_size)
        samples = jax.lax.stop_gradient(y_hard - y_soft) + y_soft

        # get PWM by taking softmax of logits
        x = jax.nn.softmax(x, axis=-1)

        return x, samples
        
    
class DEN(nn.Module):
    generator_config_updates: ... = None
    predictor_config_updates: ... = None
    loss_updates: ... = None

    @staticmethod
    @nn.nowrap
    def get_default_config(generator_config_updates=None, predictor_config_updates=None, loss_updates=None):
        config = mlxu.config_dict()
#         config.generator = UNet.get_default_config(generator_config_updates)
        config.generator = DeconvConvNet.get_default_config(generator_config_updates)
        config.predictor = FinetuneNetwork.get_default_config(predictor_config_updates)
        config.loss = DEN_loss.get_default_config(loss_updates)
        return config
    
    def setup(self):
        self.config = self.get_default_config(self.generator_config_updates, self.predictor_config_updates, self.loss_updates)
#         self.generator = UNet(self.config.generator)
        self.generator = DeconvConvNet(self.config.generator)
        self.predictor = FinetuneNetwork(self.config.predictor)
        self.loss = DEN_loss(self.config.loss)

    def run_on_batch(self, inputs, deterministic=False, temperature=1.0):
        # inputs: (batch_size, latent_size)
        
        # generate sequence
        x, samples = self.generator(inputs, deterministic=deterministic, temperature=temperature)
        samples_flat = samples.reshape((samples.shape[0] * samples.shape[1], samples.shape[2], samples.shape[3]))

        # predict expression
        # using PWM
        intermediate_pwm, pwm_predictions_thp1, pwm_predictions_jurkat, pwm_predictions_k562 = self.predictor(x, deterministic=True)
        pwm_predictions = jnp.stack([pwm_predictions_thp1, pwm_predictions_jurkat, pwm_predictions_k562], axis=1)

        # using samples
        intermediate_samples, samples_predictions_thp1, samples_predictions_jurkat, samples_predictions_k562 = self.predictor(samples_flat, deterministic=True)
        intermediate_samples = intermediate_samples.reshape((samples.shape[0], samples.shape[1], intermediate_samples.shape[1]))
        samples_predictions_thp1 = samples_predictions_thp1.reshape((samples.shape[0], samples.shape[1]))
        samples_predictions_jurkat = samples_predictions_jurkat.reshape((samples.shape[0], samples.shape[1]))
        samples_predictions_k562 = samples_predictions_k562.reshape((samples.shape[0], samples.shape[1]))
        samples_predictions = jnp.stack([samples_predictions_thp1, samples_predictions_jurkat, samples_predictions_k562], axis=2)

        return x, samples, \
               pwm_predictions, samples_predictions, \
               intermediate_pwm, intermediate_samples

    def __call__(self, inputs1, inputs2, deterministic=False, temperature=1.0, return_samples=False):
        pwm1, samples1, pwm_predictions1, samples_predictions1, intermediate_pwm1, intermediate_samples1 = self.run_on_batch(inputs1, deterministic=deterministic, temperature=temperature)
        pwm2, samples2, pwm_predictions2, samples_predictions2, intermediate_pwm2, intermediate_samples2 = self.run_on_batch(inputs2, deterministic=deterministic, temperature=temperature)

        # compute loss
        total_loss, \
        fitness_loss, \
        total_diversity_loss, \
        entropy_loss, \
        base_entropy_loss, \
        diversity_loss, \
        intermediate_repr_loss = self.loss(pwm1, pwm2, \
                                           samples1, samples2, \
                                           pwm_predictions1, pwm_predictions2, \
                                           samples_predictions1, samples_predictions2, \
                                           intermediate_samples1, intermediate_samples2)
        
        if return_samples:
            return total_loss, fitness_loss, total_diversity_loss, entropy_loss, base_entropy_loss, diversity_loss, intermediate_repr_loss, \
                   samples_predictions1, samples_predictions2, \
                   pwm1, pwm2, samples1, samples2
        
        return total_loss, fitness_loss, total_diversity_loss, entropy_loss, base_entropy_loss, diversity_loss, intermediate_repr_loss, \
               samples_predictions1, samples_predictions2

    @nn.nowrap
    def rng_keys(self):
        return ('params', 'dropout', 'gumbel')