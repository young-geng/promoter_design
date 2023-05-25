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
        config.gd_steps = 60
        config.gd_step_size = 0.5
        config.gd_b1 = 0.9
        config.gd_b2 = 0.999
        config.gd_presoftmax_scale = 2.7

        config.mutation_steps = 15
        config.mutation_rate = 0.1
        config.mutation_pool_size = 5

        config.total_rounds = 5

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


class ExpressionObjective(object):

    @staticmethod
    def get_default_config(updates=None):
        config = mlxu.config_dict()

        config.exp_clip_max = 5.0
        config.exp_clip_min = -1.0

        config.thp1_exp_multiplier = 1.0
        config.jurkat_exp_multiplier = 1.0
        config.k562_exp_multiplier = 1.0

        config.type = 'target'

        config.linear_thp1_weight = 1.0
        config.linear_jurkat_weight = 1.0
        config.linear_k562_weight = 1.0

        config.target_thp1_positive = 5.0
        config.target_thp1_negative = -1.0
        config.target_thp1_loss = 'l1'

        config.target_jurkat_positive = 5.0
        config.target_jurkat_negative = -1.0
        config.target_jurkat_loss = 'l1'

        config.target_k562_positive = 5.0
        config.target_k562_negative = -1.0
        config.target_k562_loss = 'l1'

        if updates is not None:
            config.update(mlxu.config_dict(updates).copy_and_resolve_references())

        return config

    def __init__(self, config):
        self.config = self.get_default_config(config)

    def __call__(self, thp1_exp, jurkat_exp, k562_exp):
        thp1_exp = jnp.clip(thp1_exp, self.config.exp_clip_min, self.config.exp_clip_max)
        jurkat_exp = jnp.clip(jurkat_exp, self.config.exp_clip_min, self.config.exp_clip_max)
        k562_exp = jnp.clip(k562_exp, self.config.exp_clip_min, self.config.exp_clip_max)
        return {
            'linear': self.linear_objective_fn,
            'target': self.target_objective_fn,
        }[self.config.type](
            self.config.thp1_exp_multiplier * thp1_exp,
            self.config.jurkat_exp_multiplier * jurkat_exp,
            self.config.k562_exp_multiplier * k562_exp,
        )

    def linear_objective_fn(self, thp1_exp, jurkat_exp, k562_exp):
        thp1_diff = (
            self.config.linear_thp1_weight * thp1_exp
            - 0.5 * jurkat_exp - 0.5 * k562_exp
        )
        jurkat_diff = (
            self.config.linear_jurkat_weight * jurkat_exp
            - 0.5 * thp1_exp - 0.5 * k562_exp
        )
        k562_diff = (
            self.config.linear_k562_weight * k562_exp
            - 0.5 * thp1_exp - 0.5 * jurkat_exp
        )
        return thp1_diff, jurkat_diff, k562_diff

    def target_objective_fn(self, thp1_exp, jurkat_exp, k562_exp):
        def obj_fn(pos, neg1, neg2, pos_target, neg_target, loss):
            if loss == 'l2':
                return (
                    -2 * jnp.square(pos - pos_target)
                    - jnp.square(neg1 - neg_target)
                    - jnp.square(neg2 - neg_target)
                )
            elif loss == 'l1':
                return (
                    -2 * jnp.abs(pos - pos_target)
                    - jnp.abs(neg1 - neg_target)
                    - jnp.abs(neg2 - neg_target)
                )
            else:
                raise ValueError('Unknown loss function: %s' % loss)

        thp1_diff = obj_fn(
            thp1_exp, jurkat_exp, k562_exp,
            self.config.target_thp1_positive, self.config.target_thp1_negative,
            self.config.target_thp1_loss
        )
        jurkat_diff = obj_fn(
            jurkat_exp, thp1_exp, k562_exp,
            self.config.target_jurkat_positive, self.config.target_jurkat_negative,
            self.config.target_jurkat_loss
        )
        k562_diff = obj_fn(
            k562_exp, thp1_exp, jurkat_exp,
            self.config.target_k562_positive, self.config.target_k562_negative,
            self.config.target_k562_loss
        )
        return thp1_diff, jurkat_diff, k562_diff
