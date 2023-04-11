from functools import partial
import numpy as np
import mlxu
from tqdm import tqdm, trange
from pprint import pprint, pformat

import jax
import jax.numpy as jnp
from jax.experimental import PartitionSpec as PS
from jax.experimental.maps import Mesh
from jax.experimental.pjit import pjit
import flax
from flax import linen as nn
from flax.training.train_state import TrainState
import optax
import einops
import mlxu.jax_utils as jax_utils

from .data import PretrainDataset
from .model import PretrainNetwork, get_weight_decay_mask


FLAGS, FLAGS_DEF = mlxu.define_flags_with_default(
    seed=42,
    total_steps=100000,
    log_freq=50,
    eval_freq=1000,
    lr=1e-4,
    lr_warmup_steps=1000,
    weight_decay=1e-5,
    k562_loss_weight=1.0,
    hepg2_loss_weight=1.0,
    mpra_loss_weight=1.0,
    clip_gradient=10.0,
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

    train_dataset = PretrainDataset(FLAGS.train_data)
    eval_dataset = PretrainDataset(FLAGS.eval_data)

    model = PretrainNetwork()
    params = model.init(
        sure_inputs=jnp.zeros((1, 1000, 5)),
        mpra_inputs=jnp.zeros((1, 1000, 5)),
        deterministic=False,
        rngs=jax_utils.next_rng(model.rng_keys()),
    )

    learning_rate = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=FLAGS.lr,
        warmup_steps=FLAGS.lr_warmup_steps,
        decay_steps=FLAGS.total_steps,
        end_value=0.0,
    )

    optimizer = optax.chain(
        optax.clip_by_global_norm(FLAGS.clip_gradient),
        optax.adamw(
            learning_rate=learning_rate,
            weight_decay=FLAGS.weight_decay,
            mask=get_weight_decay_mask(['bias']),
        )
    )

    train_state = TrainState.create(
        params=params,
        tx=optimizer,
        apply_fn=None
    )

    def compute_loss(batch, sure_k562_logits, sure_hepg2_logits, mpra_prediction):
        sure_k562_labels = batch['sure_k562_labels']
        sure_hepg2_labels = batch['sure_hepg2_labels']
        mpra_output = batch['mpra_output']

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

        loss = (
            sure_k562_loss * FLAGS.k562_loss_weight +
            sure_hepg2_loss * FLAGS.hepg2_loss_weight +
            mpra_loss * FLAGS.mpra_loss_weight
        )
        return loss, sure_k562_loss, sure_hepg2_loss, mpra_loss

    @partial(
        pjit,
        in_axis_resources=(None, None, PS('dp')),
        out_axis_resources=(None, None, None),
        donate_argnums=(0,),
    )
    def train_step(train_state, rng, batch):
        rng_generator = jax_utils.JaxRNG(rng)
        sure_sequences = batch['sure_sequences']
        mpra_sequences = batch['mpra_sequences']

        def loss_fn(params):
            sure_inputs = jax.nn.one_hot(sure_sequences, 5, dtype=jnp.float32)
            mpra_inputs = jax.nn.one_hot(mpra_sequences, 5, dtype=jnp.float32)
            sure_k562_logits, sure_hepg2_logits, mpra_prediction = model.apply(
                params,
                sure_inputs=sure_inputs,
                mpra_inputs=mpra_inputs,
                deterministic=False,
                rngs=rng_generator(model.rng_keys()),
            )
            loss, sure_k562_loss, sure_hepg2_loss, mpra_loss = compute_loss(
                batch, sure_k562_logits, sure_hepg2_logits, mpra_prediction
            )
            return loss, locals()

        (_, aux_values), grads = jax.value_and_grad(
            loss_fn, has_aux=True
        )(train_state.params)

        metrics = jax_utils.collect_metrics(
            aux_values,
            ['sure_k562_loss', 'sure_hepg2_loss', 'mpra_loss', 'loss'],
            prefix='train',
        )

        train_state = train_state.apply_gradients(grads=grads)
        return train_state, rng_generator(), metrics

    @partial(
        pjit,
        in_axis_resources=(None, None, PS('dp')),
        out_axis_resources=(None, None),
    )
    def eval_step(train_state, rng, batch):
        rng_generator = jax_utils.JaxRNG(rng)
        sure_sequences = batch['sure_sequences']
        mpra_sequences = batch['mpra_sequences']

        sure_inputs = jax.nn.one_hot(sure_sequences, 5, dtype=jnp.float32)
        mpra_inputs = jax.nn.one_hot(mpra_sequences, 5, dtype=jnp.float32)
        sure_k562_logits, sure_hepg2_logits, mpra_prediction = model.apply(
            train_state.params,
            sure_inputs=sure_inputs,
            mpra_inputs=mpra_inputs,
            deterministic=True,
            rngs=rng_generator(model.rng_keys()),
        )
        loss, sure_k562_loss, sure_hepg2_loss, mpra_loss = compute_loss(
            batch, sure_k562_logits, sure_hepg2_logits, mpra_prediction
        )

        metrics = jax_utils.collect_metrics(
            locals(),
            ['sure_k562_loss', 'sure_hepg2_loss', 'mpra_loss', 'loss'],
            prefix='eval',
        )
        return metrics, rng_generator()

    train_iterator = iter(train_dataset)
    eval_iterator = iter(eval_dataset)
    rng = jax_utils.next_rng()

    with Mesh(np.array(jax.devices()), ['dp']):
        for step in trange(FLAGS.total_steps, ncols=0):
            train_state, rng, train_metrics = train_step(train_state, rng, next(train_iterator))
            if step % FLAGS.log_freq == 0:
                train_metrics = jax.device_get(train_metrics)
                train_metrics['step'] = step
                logger.log(train_metrics)
                tqdm.write(pformat(train_metrics))

            if step % FLAGS.eval_freq == 0:
                eval_metrics, rng = eval_step(train_state, rng, next(eval_iterator))
                eval_metrics = jax.device_get(eval_metrics)
                eval_metrics['step'] = step
                logger.log(eval_metrics)
                tqdm.write(pformat(eval_metrics))


if __name__ == '__main__':
    mlxu.run(main)