from functools import partial
import numpy as np
import mlxu
from tqdm import tqdm, trange
from pprint import pprint, pformat
import matplotlib.pyplot as plt
import wandb

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
from .seq_opt import SequenceOptimizer, ExpressionObjective
from .utils import (
    average_metrics, global_norm, get_weight_decay_mask, compute_corr_metrics
)


FLAGS, FLAGS_DEF = mlxu.define_flags_with_default(
    seed=42,
    total_steps=100000,
    log_freq=20,
    eval_freq=1000,
    val_steps=0,
    test_steps=0,
    save_model=False,
    generate_sequences=True,
    remat=True,
    accumulate_gradient_steps=1,
    lr=1e-4,
    lr_warmup_steps=1000,
    weight_decay=1e-3,
    use_coms_loss=False,
    coms_loss_weight=0.0,
    clip_coms_loss=True,
    clip_gradient=10.0,
    load_pretrained='',
    expression_objective=ExpressionObjective.get_default_config(),
    sequence_optimizer=SequenceOptimizer.get_default_config(),
    finetune_network=FinetuneNetwork.get_default_config(),
    train_data=FinetuneDataset.get_default_config(),
    val_data=FinetuneDataset.get_default_config(),
    test_data=FinetuneDataset.get_default_config(),
    generation_data=FinetuneDataset.get_default_config(),
    logger=mlxu.WandBLogger.get_default_config(),
)


def plot_diffs(results):
    def plot_diff(target, base):
        plt.scatter(
            results[f'{target}_output'],
            results[f'{base}_output'],
            label='dataset values',
            color='blue', alpha=0.2, s=2
        )
        plt.scatter(
            results[f'ds_{target}_pred'],
            results[f'ds_{base}_pred'],
            label='dataset predicted values',
            color='green', alpha=0.2, s=2
        )
        plt.scatter(
            results[f'{target}_opt_seq_{target}_pred'],
            results[f'{target}_opt_seq_{base}_pred'],
            label='optimized predicted values',
            color='orange', alpha=0.2, s=2
        )
        plt.xlim(-1, 5)
        plt.ylim(-1, 5)
        plt.title(f'Optimization target: {target}')
        plt.xlabel(target)
        plt.ylabel(base)
        plt.plot(np.linspace(-1, 5, 10), np.linspace(-1, 5, 10), color='red')

    figure = plt.figure(figsize=(9, 15))

    plt.subplot(3, 2, 1)
    plot_diff('thp1', 'jurkat')
    plt.subplot(3, 2, 2)
    plot_diff('thp1', 'k562')

    plt.subplot(3, 2, 3)
    plot_diff('jurkat', 'thp1')
    plt.subplot(3, 2, 4)
    plot_diff('jurkat', 'k562')

    plt.subplot(3, 2, 5)
    plot_diff('k562', 'thp1')
    plt.subplot(3, 2, 6)
    plot_diff('k562', 'jurkat')

    plt.legend(loc='lower center', bbox_to_anchor=(-0.1, -0.35), markerscale=4)
    return figure


def main(argv):
    logger = mlxu.WandBLogger(
        config=FLAGS.logger,
        variant=mlxu.get_user_flags(FLAGS, FLAGS_DEF),
    )
    jax_utils.set_random_seed(FLAGS.seed)
    jax_device_count = jax.device_count()

    train_dataset = FinetuneDataset(FLAGS.train_data)

    if FLAGS.val_steps > 0:
        val_dataset = FinetuneDataset(FLAGS.val_data)

    if FLAGS.test_steps > 0:
        test_dataset = FinetuneDataset(FLAGS.test_data)

    model = FinetuneNetwork(FLAGS.finetune_network)
    params = model.init(
        inputs=jnp.zeros((1, 1000, 4)),
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
    del params

    sequence_optimizer = SequenceOptimizer(FLAGS.sequence_optimizer)
    expression_objective = ExpressionObjective(FLAGS.expression_objective)

    def compute_loss(batch, thp1_output, jurkat_output, k562_output):
        thp1_loss = jnp.mean(jnp.square(thp1_output - batch['thp1_output']))
        jurkat_loss = jnp.mean(jnp.square(jurkat_output - batch['jurkat_output']))
        k562_loss = jnp.mean(jnp.square(k562_output - batch['k562_output']))

        thp1_corr, thp1_rank_corr, thp1_r2 = compute_corr_metrics(
            jax.lax.all_gather(thp1_output, axis_name='dp').reshape(-1),
            jax.lax.all_gather(batch['thp1_output'], axis_name='dp').reshape(-1),
        )
        jurkat_corr, jurkat_rank_corr, jurkat_r2 = compute_corr_metrics(
            jax.lax.all_gather(jurkat_output, axis_name='dp').reshape(-1),
            jax.lax.all_gather(batch['jurkat_output'], axis_name='dp').reshape(-1),
        )
        k562_corr, k562_rank_corr, k562_r2 = compute_corr_metrics(
            jax.lax.all_gather(k562_output, axis_name='dp').reshape(-1),
            jax.lax.all_gather(batch['k562_output'], axis_name='dp').reshape(-1),
        )

        supervised_loss = thp1_loss + jurkat_loss + k562_loss

        aux_values = dict(
            thp1_loss=thp1_loss,
            jurkat_loss=jurkat_loss,
            k562_loss=k562_loss,
            thp1_corr=thp1_corr,
            thp1_rank_corr=thp1_rank_corr,
            thp1_r2=thp1_r2,
            jurkat_corr=jurkat_corr,
            jurkat_rank_corr=jurkat_rank_corr,
            jurkat_r2=jurkat_r2,
            k562_corr=k562_corr,
            k562_rank_corr=k562_rank_corr,
            k562_r2=k562_r2,
            supervised_loss=supervised_loss,
        )

        return supervised_loss, aux_values

    def compute_coms_loss(params, rng, batch):
        rng_generator = jax_utils.JaxRNG(rng)
        starting_seq = jax.nn.one_hot(batch['sequences'], 5, dtype=jnp.float32)[:, :, :4]

        def objectve_funtion(seq, rng, params, target='all'):
            rng_generator = jax_utils.JaxRNG(rng)
            thp1_pred, jurkat_pred, k562_pred = model.apply(
                params,
                inputs=seq,
                deterministic=False,
                rngs=rng_generator(model.rng_keys()),
            )
            thp1_diff, jurkat_diff, k562_diff = expression_objective(
                thp1_pred, jurkat_pred, k562_pred
            )

            if target == 'thp1':
                return thp1_diff
            elif target == 'jurkat':
                return jurkat_diff
            elif target == 'k562':
                return k562_diff
            elif target == 'all':
                return dict(
                    thp1_pred=thp1_pred,
                    jurkat_pred=jurkat_pred,
                    k562_pred=k562_pred,
                    thp1_diff=thp1_diff,
                    jurkat_diff=jurkat_diff,
                    k562_diff=k562_diff,
                )
            else:
                raise ValueError(f'Unknown target {target}')

        def count_mutations(start, end):
            return jnp.sum(
                jnp.argmax(start, axis=-1) != jnp.argmax(end, axis=-1),
                axis=-1,
            ).astype(jnp.float32)

        starting_seq_metrics = objectve_funtion(
            starting_seq, rng_generator(),
            params=params, target='all'
        )
        ds_thp1_pred, ds_jurkat_pred, ds_k562_pred = (
            starting_seq_metrics['thp1_pred'],
            starting_seq_metrics['jurkat_pred'],
            starting_seq_metrics['k562_pred'],
        )
        ds_thp1_diff, ds_jurkat_diff, ds_k562_diff = (
            starting_seq_metrics['thp1_diff'],
            starting_seq_metrics['jurkat_diff'],
            starting_seq_metrics['k562_diff'],
        )

        thp1_optimized_seq = sequence_optimizer(
            objectve_funtion,
            starting_seq,
            rng_generator(),
            params=params,
            target='thp1',
        )
        thp1_n_mutations = count_mutations(starting_seq, thp1_optimized_seq).mean()
        opt_thp1_metrics = objectve_funtion(
            thp1_optimized_seq, rng_generator(),
            params=params, target='all'
        )
        opt_thp1_pred, opt_thp1_diff = (
            opt_thp1_metrics['thp1_pred'],
            opt_thp1_metrics['thp1_diff'],
        )
        opt_thp1_metrics = {
            f'thp1_opt_seq_{key}': val for key, val in opt_thp1_metrics.items()
        }

        jurkat_optimized_seq = sequence_optimizer(
            objectve_funtion,
            starting_seq,
            rng_generator(),
            params=params,
            target='jurkat',
        )
        jurkat_n_mutations = count_mutations(starting_seq, jurkat_optimized_seq)
        opt_jurkat_metrics = objectve_funtion(
            jurkat_optimized_seq, rng_generator(),
            params=params, target='all'
        )
        opt_jurkat_pred, opt_jurkat_diff = (
            opt_jurkat_metrics['jurkat_pred'],
            opt_jurkat_metrics['jurkat_diff'],
        )
        opt_jurkat_metrics = {
            f'jurkat_opt_seq_{key}': val for key, val in opt_jurkat_metrics.items()
        }

        k562_optimized_seq = sequence_optimizer(
            objectve_funtion,
            starting_seq,
            rng_generator(),
            params=params,
            target='k562',
        )
        k562_n_mutations = count_mutations(starting_seq, k562_optimized_seq).mean()
        opt_k562_metrics = objectve_funtion(
            k562_optimized_seq, rng_generator(),
            params=params, target='all'
        )
        opt_k562_pred, opt_k562_diff = (
            opt_k562_metrics['k562_pred'],
            opt_k562_metrics['k562_diff'],
        )
        opt_k562_metrics = {
            f'k562_opt_seq_{key}': val for key, val in opt_k562_metrics.items()
        }

        thp1_gap = opt_thp1_diff - ds_thp1_diff
        jurkat_gap = opt_jurkat_diff - ds_jurkat_diff
        k562_gap = opt_k562_diff - ds_k562_diff

        if FLAGS.clip_coms_loss:
            coms_loss = FLAGS.coms_loss_weight * jnp.mean(
                jnp.clip(thp1_gap, a_min=0.0) +
                jnp.clip(jurkat_gap, a_min=0.0) +
                jnp.clip(k562_gap, a_min=0.0)
            )
        else:
            coms_loss = FLAGS.coms_loss_weight * jnp.mean(
                thp1_gap + jurkat_gap + k562_gap
            )

        aux_values = dict(
            coms_loss=coms_loss,

            original_seq=batch['sequences'],
            thp1_output=batch['thp1_output'],
            jurkat_output=batch['jurkat_output'],
            k562_output=batch['k562_output'],

            ds_thp1_pred=ds_thp1_pred,
            ds_jurkat_pred=ds_jurkat_pred,
            ds_k562_pred=ds_k562_pred,
            ds_thp1_diff=ds_thp1_diff,
            ds_jurkat_diff=ds_jurkat_diff,
            ds_k562_diff=ds_k562_diff,

            thp1_optimized_seq=jnp.argmax(thp1_optimized_seq, axis=-1),
            jurkat_optimized_seq=jnp.argmax(jurkat_optimized_seq, axis=-1),
            k562_optimized_seq=jnp.argmax(k562_optimized_seq, axis=-1),

            opt_thp1_pred=opt_thp1_pred,
            opt_jurkat_pred=opt_jurkat_pred,
            opt_k562_pred=opt_k562_pred,
            opt_thp1_diff=opt_thp1_diff,
            opt_jurkat_diff=opt_jurkat_diff,
            opt_k562_diff=opt_k562_diff,
            thp1_gap=thp1_gap,
            jurkat_gap=jurkat_gap,
            k562_gap=k562_gap,
            thp1_n_mutations=thp1_n_mutations,
            jurkat_n_mutations=jurkat_n_mutations,
            k562_n_mutations=k562_n_mutations,

            **opt_thp1_metrics,
            **opt_jurkat_metrics,
            **opt_k562_metrics,
        )

        return coms_loss, aux_values

    metric_keys = [
        'thp1_loss', 'jurkat_loss', 'k562_loss', 'supervised_loss',
        'thp1_corr', 'thp1_rank_corr', 'thp1_r2',
        'jurkat_corr', 'jurkat_rank_corr', 'jurkat_r2',
        'k562_corr', 'k562_rank_corr', 'k562_r2',
        'ds_thp1_diff', 'ds_jurkat_diff', 'ds_k562_diff',
        'opt_thp1_diff', 'opt_jurkat_diff', 'opt_k562_diff',
        'thp1_gap', 'jurkat_gap', 'k562_gap',
        'thp1_n_mutations', 'jurkat_n_mutations', 'k562_n_mutations',
        'coms_loss', 'loss'
    ]

    @partial(jax.pmap, axis_name='dp')
    def optimization_step(params, rng, batch):
        rng_generator = jax_utils.JaxRNG(rng)
        _, reults = compute_coms_loss(params, rng_generator(), batch)
        return rng_generator(), reults

    @partial(jax.pmap, axis_name='dp', donate_argnums=(0, 1))
    def train_step(train_state, rng, batch):
        rng_generator = jax_utils.JaxRNG(rng)

        def loss_fn(params, rng, batch):
            rng_generator = jax_utils.JaxRNG(rng)
            thp1_output, jurkat_output, k562_output = model.apply(
                params,
                inputs=jax.nn.one_hot(batch['sequences'], 5, dtype=jnp.float32)[:, :, :4],
                deterministic=False,
                rngs=rng_generator(model.rng_keys()),
            )
            loss, aux_values = compute_loss(
                batch, thp1_output, jurkat_output, k562_output
            )
            if FLAGS.use_coms_loss:
                coms_loss, coms_aux_values = compute_coms_loss(params, rng_generator(), batch)
                loss += coms_loss
                aux_values.update(coms_aux_values)

            aux_values['loss'] = loss
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

    @partial(jax.pmap, axis_name='dp', donate_argnums=1, static_broadcasted_argnums=(3,))
    def eval_step(train_state, rng, batch, prefix='val'):
        rng_generator = jax_utils.JaxRNG(rng)

        thp1_output, jurkat_output, k562_output = model.apply(
            train_state.params,
            inputs=jax.nn.one_hot(batch['sequences'], 5, dtype=jnp.float32)[:, :, :4],
            deterministic=True,
            rngs=rng_generator(model.rng_keys()),
        )
        loss, aux_values = compute_loss(
            batch, thp1_output, jurkat_output, k562_output
        )
        if FLAGS.use_coms_loss:
            coms_loss, coms_aux_values = compute_coms_loss(
                train_state.params, rng_generator(), batch
            )
            loss += coms_loss
            aux_values.update(coms_aux_values)

        aux_values['loss'] = loss
        metrics = jax_utils.collect_metrics(
            aux_values, metric_keys, prefix=prefix,
        )
        metrics = jax.lax.pmean(metrics, axis_name='dp')
        return metrics, rng_generator()

    train_iterator = train_dataset.batch_iterator(pmap_axis_dim=jax_device_count)

    if FLAGS.val_steps > 0:
        val_iterator = val_dataset.batch_iterator(pmap_axis_dim=jax_device_count)

    if FLAGS.test_steps > 0:
        test_iterator = test_dataset.batch_iterator(pmap_axis_dim=jax_device_count)

    rng = jax.device_put_sharded(
        list(jax_utils.next_rng(jax.device_count())),
        jax.devices(),
    )
    train_state = replicate(train_state)

    best_val_loss = np.inf
    best_params = None

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
            if FLAGS.val_steps > 0:
                eval_metrics = []
                for _ in range(FLAGS.val_steps):
                    metrics, rng = eval_step(
                        train_state, rng, next(val_iterator), 'val'
                    )
                    eval_metrics.append(unreplicate(metrics))
                eval_metrics = average_metrics(jax.device_get(eval_metrics))

                if eval_metrics['val/loss'] < best_val_loss:
                    best_val_loss = eval_metrics['val/loss']
                    best_params = jax.device_get(unreplicate(train_state).params)
                    if FLAGS.save_model:
                        logger.save_pickle(best_params, 'best_params.pkl',)

                eval_metrics['val/best_loss'] = best_val_loss
                eval_metrics['step'] = step
                logger.log(eval_metrics)
                tqdm.write(pformat(eval_metrics))

            if FLAGS.test_steps > 0:
                eval_metrics = []
                for _ in range(FLAGS.test_steps):
                    metrics, rng = eval_step(
                        train_state, rng, next(test_iterator), 'test'
                    )
                    eval_metrics.append(unreplicate(metrics))
                eval_metrics = average_metrics(jax.device_get(eval_metrics))
                eval_metrics['step'] = step
                logger.log(eval_metrics)
                tqdm.write(pformat(eval_metrics))

    if best_params is None:
        best_params = jax.device_get(unreplicate(train_state).params)

    del train_state

    if FLAGS.generate_sequences:
        FLAGS.generation_data.sequential_sample = True  # Ensure we iterate over the whole dataset
        dataset = FinetuneDataset(FLAGS.generation_data)
        best_params = replicate(best_params)

        data_iterator = dataset.batch_iterator(pmap_axis_dim=jax_device_count)
        steps = len(dataset) // FLAGS.generation_data.batch_size

        results = []

        for _, batch in zip(trange(steps, ncols=0), data_iterator):
            rng, r = optimization_step(best_params, rng, batch)

            results.append({
                key: einops.rearrange(val, 'd b ... -> (d b) ...')
                for key, val in jax.device_get(r).items()
                if len(val.shape) > 1
            })

        results = {
            key: np.concatenate([r[key] for r in results], axis=0)
            for key in results[0].keys()
        }
        logger.save_pickle(results, 'optimized_seqs.pkl')
        logger.log({f'opt_seq_diffs': wandb.Image(plot_diffs(results))})


if __name__ == '__main__':
    mlxu.run(main)