from functools import partial
import numpy as np
import mlxu
from tqdm import tqdm, trange
from pprint import pprint, pformat

import jax
import jax.numpy as jnp
import flax
from flax.jax_utils import replicate, unreplicate
from flax.training.train_state import TrainState
import optax
import einops
import mlxu.jax_utils as jax_utils

from .data import FinetuneDataset
from .model import FinetuneNetwork
from .utils import average_metrics, global_norm, get_weight_decay_mask


FLAGS, FLAGS_DEF = mlxu.define_flags_with_default(
    seed=42,
    total_steps=100000,
    log_freq=20,
    eval_freq=1000,
    eval_steps=30,
    save_model=False,
    accumulate_gradient_steps=1,
    lr=1e-4,
    lr_warmup_steps=1000,
    weight_decay=1e-3,
    k562_loss_weight=1.0,
    hepg2_loss_weight=1.0,
    mpra_loss_weight=1.0,
    clip_gradient=10.0,
    load_pretrained='',
    finetune_network=FinetuneNetwork.get_default_config(),
    train_data=FinetuneDataset.get_default_config(),
    eval_data=FinetuneDataset.get_default_config(),
    logger=mlxu.WandBLogger.get_default_config(),
)


def main(argv):
    logger = mlxu.WandBLogger(
        config=FLAGS.logger,
        variant=mlxu.get_user_flags(FLAGS, FLAGS_DEF),
    )
    jax_utils.set_random_seed(FLAGS.seed)
    jax_device_count = jax.device_count()

    train_dataset = FinetuneDataset(FLAGS.train_data)
    eval_dataset = FinetuneDataset(FLAGS.eval_data)

    model = FinetuneNetwork(FLAGS.finetune_network)
    params = model.init(
        inputs=jnp.zeros((1, 1000, 5)),
        deterministic=False,
        rngs=jax_utils.next_rng(model.rng_keys()),
    )

    if FLAGS.load_pretrained != '':
        params = flax.core.unfreeze(params)
        params['params']['backbone'] = jax.device_put(
            mlxu.load_pickle(FLAGS.load_pretrained)['params']['backbone']
        )
        params = flax.core.freeze(params)

    learning_rate_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=FLAGS.lr,
        warmup_steps=FLAGS.lr_warmup_steps,
        decay_steps=FLAGS.total_steps,
        end_value=0.0,
    )

    optimizer = optax.chain(
        optax.clip_by_global_norm(FLAGS.clip_gradient),
        optax.adamw(
            learning_rate=learning_rate_schedule,
            weight_decay=FLAGS.weight_decay,
            mask=get_weight_decay_mask(['bias']),
        )
    )
    if FLAGS.accumulate_gradient_steps > 1:
        optimizer = optax.MultiSteps(optimizer, FLAGS.accumulate_gradient_steps)

    train_state = TrainState.create(
        params=params,
        tx=optimizer,
        apply_fn=None
    )

    def compute_loss(batch, thp1_output, jurkat_output, k562_output):
        thp1_loss = jnp.mean(jnp.square(thp1_output - batch['thp1_output']))
        jurkat_loss = jnp.mean(jnp.square(jurkat_output - batch['jurkat_output']))
        k562_loss = jnp.mean(jnp.square(k562_output - batch['k562_output']))
        loss = thp1_loss + jurkat_loss + k562_loss
        return loss, locals()

    @partial(jax.pmap, axis_name='dp', donate_argnums=(0, 1))
    def train_step(train_state, rng, batch):
        rng_generator = jax_utils.JaxRNG(rng)

        def loss_fn(params):
            thp1_output, jurkat_output, k562_output = model.apply(
                params,
                inputs=jax.nn.one_hot(batch['sequences'], 5, dtype=jnp.float32),
                deterministic=False,
                rngs=rng_generator(model.rng_keys()),
            )
            loss, aux_values = compute_loss(
                batch, thp1_output, jurkat_output, k562_output
            )
            return loss, aux_values

        (_, aux_values), grads = jax.value_and_grad(
            loss_fn, has_aux=True
        )(train_state.params)
        grads = jax.lax.pmean(grads, axis_name='dp')

        aux_values['learning_rate'] = learning_rate_schedule(train_state.step)
        aux_values['grad_norm'] = global_norm(grads)
        aux_values['param_norm'] = global_norm(train_state.params)

        metrics = jax_utils.collect_metrics(
            aux_values,
            ['thp1_loss', 'jurkat_loss', 'k562_loss', 'loss',
             'learning_rate', 'grad_norm', 'param_norm'],
            prefix='train',
        )
        metrics = jax.lax.pmean(metrics, axis_name='dp')

        train_state = train_state.apply_gradients(grads=grads)
        return train_state, rng_generator(), metrics

    @partial(jax.pmap, axis_name='dp', donate_argnums=1)
    def eval_step(train_state, rng, batch):
        rng_generator = jax_utils.JaxRNG(rng)

        thp1_output, jurkat_output, k562_output = model.apply(
            train_state.params,
            inputs=jax.nn.one_hot(batch['sequences'], 5, dtype=jnp.float32),
            deterministic=False,
            rngs=rng_generator(model.rng_keys()),
        )
        loss, aux_values = compute_loss(
            batch, thp1_output, jurkat_output, k562_output
        )

        metrics = jax_utils.collect_metrics(
            aux_values,
            ['thp1_loss', 'jurkat_loss', 'k562_loss', 'loss'],
            prefix='eval',
        )
        metrics = jax.lax.pmean(metrics, axis_name='dp')
        return metrics, rng_generator()

    train_iterator = train_dataset.batch_iterator(pmap_axis_dim=jax_device_count)
    eval_iterator = eval_dataset.batch_iterator(pmap_axis_dim=jax_device_count)
    rng = jax.device_put_sharded(
        list(jax_utils.next_rng(jax.device_count())),
        jax.devices(),
    )
    train_state = replicate(train_state)

    best_eval_loss = np.inf

    for step in trange(FLAGS.total_steps, ncols=0):
        train_state, rng, train_metrics = train_step(
            train_state, rng, next(train_iterator)
        )
        if step % FLAGS.log_freq == 0:
            train_metrics = jax.device_get(unreplicate(train_metrics))
            train_metrics['step'] = step
            logger.log(train_metrics)
            tqdm.write(pformat(train_metrics))

        if step % FLAGS.eval_freq == 0:
            eval_metrics = []
            for _ in range(FLAGS.eval_steps):
                metrics, rng = eval_step(
                    train_state, rng, next(eval_iterator)
                )
                eval_metrics.append(unreplicate(metrics))
            eval_metrics = average_metrics(jax.device_get(eval_metrics))

            if eval_metrics['eval/loss'] < best_eval_loss:
                best_eval_loss = eval_metrics['eval/loss']
                if FLAGS.save_model:
                    logger.save_pickle(
                        jax.device_get(unreplicate(train_state).params),
                        'best_params.pkl',
                    )

            eval_metrics['eval/best_loss'] = best_eval_loss
            eval_metrics['step'] = step
            logger.log(eval_metrics)
            tqdm.write(pformat(eval_metrics))


if __name__ == '__main__':
    mlxu.run(main)