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

from promoter_design.workflow.data import PretrainDataset
from promoter_design.workflow.model import PretrainNetwork
from promoter_design.utils import (
    average_metrics, global_norm, get_weight_decay_mask, compute_corr_metrics
)


FLAGS, FLAGS_DEF = mlxu.define_flags_with_default(
    seed=42,
    total_steps=20000,
    log_freq=20,
    eval_freq=200,
    eval_steps=100,
    save_model=True,
    remat=True,
    accumulate_gradient_steps=1,
    lr=1e-4,
    lr_warmup_steps=1000,
    weight_decay=3e-3,
    k562_loss_weight=1.0,
    hepg2_loss_weight=1.0,
    mpra_loss_weight=1.0,
    clip_gradient=10.0,
    pretrain_network=PretrainNetwork.get_default_config(),
    train_data=PretrainDataset.get_default_config(),
    eval_data=PretrainDataset.get_default_config(),
    logger=mlxu.WandBLogger.get_default_config(),
)


def main(argv):
    logger = mlxu.WandBLogger(
        config=FLAGS.logger,
        variant=mlxu.get_user_flags(FLAGS, FLAGS_DEF),
    )
    jax_utils.set_random_seed(FLAGS.seed)
    jax_device_count = jax.device_count()

    train_dataset = PretrainDataset(FLAGS.train_data)
    eval_dataset = PretrainDataset(FLAGS.eval_data)

    model = PretrainNetwork(FLAGS.pretrain_network)
    params = model.init(
        sure_inputs=jnp.zeros((1, 1000, 4)),
        mpra_inputs=jnp.zeros((1, 1000, 4)),
        deterministic=False,
        rngs=jax_utils.next_rng(model.rng_keys()),
    )

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

    def compute_loss(batch, sure_k562_logits, sure_hepg2_logits, mpra_prediction):
        sure_k562_labels = batch['sure_k562_labels']
        sure_hepg2_labels = batch['sure_hepg2_labels']
        mpra_output = batch['mpra_output']

        sure_k562_accuracy = jnp.mean(
            (jnp.argmax(sure_k562_logits, axis=-1) == sure_k562_labels).astype(jnp.float32)
        ) * 4
        sure_hepg2_accuracy = jnp.mean(
            (jnp.argmax(sure_hepg2_logits, axis=-1) == sure_hepg2_labels).astype(jnp.float32)
        ) * 4

        sure_k562_logits = einops.rearrange(sure_k562_logits, '... (h d) -> ... h d', h=4)
        sure_hepg2_logits = einops.rearrange(sure_hepg2_logits, '... (h d) -> ... h d', h=4)
        sure_k562_logp = jax.nn.log_softmax(sure_k562_logits, axis=-1)
        sure_hepg2_logp = jax.nn.log_softmax(sure_hepg2_logits, axis=-1)
        sure_k562_logp = einops.rearrange(sure_k562_logp, '... h d -> ... (h d)')
        sure_hepg2_logp = einops.rearrange(sure_hepg2_logp, '... h d -> ... (h d)')

        sure_k562_onehot = jax.nn.one_hot(sure_k562_labels, 20)
        sure_hepg2_onehot = jax.nn.one_hot(sure_hepg2_labels, 20)

        sure_k562_loss = -jnp.mean(jnp.sum(sure_k562_onehot * sure_k562_logp, axis=-1))
        sure_hepg2_loss = -jnp.mean(jnp.sum(sure_hepg2_onehot * sure_hepg2_logp, axis=-1))
        mpra_loss = jnp.mean(jnp.square(mpra_prediction - mpra_output))

        gathered_mpra_prediction = jax.lax.all_gather(
            mpra_prediction, axis_name='dp'
        ).reshape(-1)
        gathered_mpra_output = jax.lax.all_gather(
            mpra_output, axis_name='dp'
        ).reshape(-1)
        mpra_corr, mpra_rank_corr, mpra_r2 = compute_corr_metrics(
            gathered_mpra_prediction, gathered_mpra_output
        )

        loss = (
            sure_k562_loss * FLAGS.k562_loss_weight +
            sure_hepg2_loss * FLAGS.hepg2_loss_weight +
            mpra_loss * FLAGS.mpra_loss_weight
        )
        return loss, locals()

    metric_keys = [
        'sure_k562_loss', 'sure_hepg2_loss', 'mpra_loss', 'loss',
        'sure_k562_accuracy', 'sure_hepg2_accuracy',
        'mpra_corr', 'mpra_rank_corr', 'mpra_r2',
    ]

    @partial(jax.pmap, axis_name='dp', donate_argnums=(0, 1))
    def train_step(train_state, rng, batch):
        rng_generator = jax_utils.JaxRNG(rng)

        def loss_fn(params, rng, batch):
            rng_generator = jax_utils.JaxRNG(rng)
            sure_inputs = jax.nn.one_hot(batch['sure_sequences'], 5, dtype=jnp.float32)[:, :, :4]
            mpra_inputs = jax.nn.one_hot(batch['mpra_sequences'], 5, dtype=jnp.float32)[:, :, :4]
            sure_k562_logits, sure_hepg2_logits, mpra_prediction = model.apply(
                params,
                sure_inputs=sure_inputs,
                mpra_inputs=mpra_inputs,
                deterministic=False,
                rngs=rng_generator(model.rng_keys()),
            )
            loss, aux_values = compute_loss(
                batch, sure_k562_logits, sure_hepg2_logits, mpra_prediction
            )
            return loss, aux_values

        if FLAGS.remat:
            loss_fn = jax.checkpoint(
                loss_fn, policy=jax.checkpoint_policies.checkpoint_dots
            )

        (_, aux_values), grads = jax.value_and_grad(
            loss_fn, has_aux=True
        )(train_state.params, rng_generator(), batch)
        grads = jax.lax.pmean(grads, axis_name='dp')

        aux_values['learning_rate'] = learning_rate_schedule(train_state.step)
        aux_values['grad_norm'] = global_norm(grads)
        aux_values['param_norm'] = global_norm(train_state.params)

        metrics = jax_utils.collect_metrics(
            aux_values,
            metric_keys + ['learning_rate', 'grad_norm', 'param_norm'],
            prefix='train',
        )
        metrics = jax.lax.pmean(metrics, axis_name='dp')

        train_state = train_state.apply_gradients(grads=grads)
        return train_state, rng_generator(), metrics

    @partial(jax.pmap, axis_name='dp', donate_argnums=1)
    def eval_step(train_state, rng, batch):
        rng_generator = jax_utils.JaxRNG(rng)
        sure_sequences = batch['sure_sequences']
        mpra_sequences = batch['mpra_sequences']

        sure_inputs = jax.nn.one_hot(sure_sequences, 5, dtype=jnp.float32)[:, :, :4]
        mpra_inputs = jax.nn.one_hot(mpra_sequences, 5, dtype=jnp.float32)[:, :, :4]
        sure_k562_logits, sure_hepg2_logits, mpra_prediction = model.apply(
            train_state.params,
            sure_inputs=sure_inputs,
            mpra_inputs=mpra_inputs,
            deterministic=True,
            rngs=rng_generator(model.rng_keys()),
        )
        loss, aux_values = compute_loss(
            batch, sure_k562_logits, sure_hepg2_logits, mpra_prediction
        )

        metrics = jax_utils.collect_metrics(
            aux_values, metric_keys, prefix='eval'
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