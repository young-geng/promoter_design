from functools import partial
import numpy as np
import mlxu

import jax
import jax.numpy as jnp
import optax
import einops
import mlxu.jax_utils as jax_utils


class SequenceOptimizer(object):

    @staticmethod
    def get_default_config(updates=None):
        config = mlxu.config_dict()
        config.gd_steps = 5
        config.gd_step_size = 0.1
        config.gd_b1 = 0.9
        config.gd_b2 = 0.999
        config.gd_presoftmax_scale = 2.7

        config.mutation_steps = 5
        config.mutation_rate = 0.1
        config.mutation_pool_size = 5

        config.total_rounds = 2

        if updates is not None:
            config.update(mlxu.config_dict(updates).copy_and_resolve_references())

        return config


    def __init__(self, config):
        self.config = self.get_default_config(config)

    def gradient_descent_round(self, func, seq, rng, **kwargs):
        if self.config.gd_steps == 0:
            return seq

        optimizer = optax.adam(self.config.gd_step_size)
        seq = seq * self.config.gd_presoftmax_scale
        opt_state = optimizer.init(seq)

        @jax.grad
        def grad_fn(x, rng, **kwargs):
            return -jnp.sum(func(jax.nn.softmax(x, axis=-1), rng, **kwargs))

        def loop_body(i, state):
            seq, opt_state, rng = state
            rng_generator = jax_utils.JaxRNG(rng)
            grads = grad_fn(seq, rng_generator(), **kwargs)
            updates, opt_state = optimizer.update(grads, opt_state)
            seq = jnp.clip(
                optax.apply_updates(seq, updates),
                -self.config.gd_presoftmax_scale,
                self.config.gd_presoftmax_scale
            )
            return seq, opt_state, rng_generator()

        seq, opt_state, rng = jax.lax.fori_loop(
            0, self.config.gd_steps, loop_body,
            (seq, opt_state, rng)
        )
        seq = jax.nn.one_hot(jnp.argmax(seq, axis=-1), 4)
        return seq

    def random_mutation_round(self, func, seq, rng, **kwargs):
        if self.config.mutation_steps == 0:
            return seq

        def loop_body(i, state):
            seq, rng = state
            rng_generator = jax_utils.JaxRNG(rng)
            old_seq = jnp.argmax(seq, axis=-1)
            new_seq = einops.repeat(
                old_seq, 'b s -> b n s', n=self.config.mutation_pool_size
            )
            old_seq = einops.rearrange(old_seq, 'b s -> b 1 s')
            random_seq = jax.random.randint(
                rng_generator(), new_seq.shape, 0, 4
            )
            new_seq = jnp.where(
                jax.random.uniform(rng_generator(), new_seq.shape) < self.config.mutation_rate,
                random_seq,
                new_seq
            )
            combined_seq = einops.rearrange(
                jnp.concatenate([old_seq, new_seq], axis=1),
                'b n s -> (b n) s'
            )
            scores = func(jax.nn.one_hot(combined_seq, 4), rng_generator(), **kwargs)
            scores = einops.rearrange(
                scores, '(b n) -> b n', n=self.config.mutation_pool_size + 1
            )
            best_seq_index = jnp.argmax(scores, axis=-1)
            combined_seq = einops.rearrange(
                combined_seq, '(b n) s -> b n s', n=self.config.mutation_pool_size + 1
            )
            new_seq = combined_seq[jnp.arange(combined_seq.shape[0]), best_seq_index, :]
            new_seq = jax.nn.one_hot(new_seq, 4)
            return new_seq, rng_generator()

        seq, rng = jax.lax.fori_loop(
            0, self.config.mutation_steps, loop_body, (seq, rng)
        )
        return seq

    def __call__(self, func, seq, rng, **kwargs):
        assert len(seq.shape) == 3
        def loop_body(i, state):
            seq, rng = state
            rng_generator = jax_utils.JaxRNG(rng)
            seq = self.gradient_descent_round(func, seq, rng_generator(), **kwargs)
            seq = self.random_mutation_round(func, seq, rng_generator(), **kwargs)
            return seq, rng_generator()

        seq, rng = jax.lax.fori_loop(
            0, self.config.total_rounds, loop_body, (seq, rng)
        )
        return jax.lax.stop_gradient(seq)
