from functools import partial
import numpy as np
import mlxu
from tqdm import tqdm, trange
from pprint import pprint, pformat
import scipy.stats as stats
from sklearn.metrics import r2_score
import os
import pdb

import jax
import jax.numpy as jnp
import flax
from flax.jax_utils import replicate, unreplicate
from flax.training.train_state import TrainState
import optax
import einops
import mlxu.jax_utils as jax_utils

from data import lentiMPRADataset
from model import lentiMPRAPretrainNetwork
from utils import (
    average_metrics, global_norm, get_weight_decay_mask
)


FLAGS, FLAGS_DEF = mlxu.define_flags_with_default(
    seed=42,
    total_steps=100000,
    log_freq=20,
    eval_freq=1000,
    save_model=True,
    remat=True,
    accumulate_gradient_steps=1,
    lr=1e-4,
    lr_warmup_steps=1000,
    weight_decay=1e-3,
    clip_gradient=10.0,
    pretrain_network=lentiMPRAPretrainNetwork.get_default_config(),
    train_data=lentiMPRADataset.get_default_config({"split": "train", "path": "/global/scratch/users/aniketh/promoter_modelling/jax_data/lentiMPRA_data.pkl", "batch_size": 96}),
    val_data=lentiMPRADataset.get_default_config({"split": "val", "path": "/global/scratch/users/aniketh/promoter_modelling/jax_data/lentiMPRA_data.pkl", "batch_size": 96, "sequential_sample": True}),
    test_data=lentiMPRADataset.get_default_config({"split": "test", "path": "/global/scratch/users/aniketh/promoter_modelling/jax_data/lentiMPRA_data.pkl", "batch_size": 96, "sequential_sample": True}),
    logger=mlxu.WandBLogger.get_default_config({"output_dir": "/global/scratch/users/aniketh/promoter_modelling/jax_data/saved_models", "project": "promoter_design_jax", "wandb_dir": "/global/scratch/users/aniketh/promoter_modelling/jax_data/wandb", "online": True, \
                                                "experiment_id": "lentiMPRA_pretraining_savio_16L_b96_lr1e-4_rerun"}),
)


def main(argv):
    logger = mlxu.WandBLogger(
        config=FLAGS.logger,
        variant=mlxu.get_user_flags(FLAGS, FLAGS_DEF),
    )
    jax_utils.set_random_seed(FLAGS.seed)
    jax_device_count = jax.device_count()

    train_dataset = lentiMPRADataset(FLAGS.train_data)
    val_dataset = lentiMPRADataset(FLAGS.val_data)

    model = lentiMPRAPretrainNetwork(FLAGS.pretrain_network)
    params = model.init(
        lentiMPRA_inputs=jnp.zeros((1, 1000, 4)),
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

    def compute_loss(batch, lentiMPRA_k562_prediction, lentiMPRA_hepg2_prediction, lentiMPRA_wtc11_prediction):
        lentiMPRA_k562_outputs = batch['lentiMPRA_k562_outputs']
        lentiMPRA_hepg2_outputs = batch['lentiMPRA_hepg2_outputs']
        lentiMPRA_wtc11_outputs = batch['lentiMPRA_wtc11_outputs']
        lentiMPRA_valid_outputs_mask = batch['lentiMPRA_valid_outputs_mask']

        lentiMPRA_k562_sqerr = jnp.square(lentiMPRA_k562_prediction - lentiMPRA_k562_outputs)
        lentiMPRA_hepg2_sqerr = jnp.square(lentiMPRA_hepg2_prediction - lentiMPRA_hepg2_outputs)
        lentiMPRA_wtc11_sqerr = jnp.square(lentiMPRA_wtc11_prediction - lentiMPRA_wtc11_outputs)

        stacked_loss = jnp.stack([lentiMPRA_k562_sqerr, lentiMPRA_hepg2_sqerr, lentiMPRA_wtc11_sqerr], axis=1)
        stacked_loss = jnp.where(lentiMPRA_valid_outputs_mask, stacked_loss, 0.0)
        
        loss = jnp.mean(stacked_loss)

        return loss, locals()

    metric_keys = [
        'loss'
    ]

    @partial(jax.pmap, axis_name='dp', donate_argnums=(0, 1))
    def train_step(train_state, rng, batch):
        rng_generator = jax_utils.JaxRNG(rng)

        def loss_fn(params, rng, batch):
            rng_generator = jax_utils.JaxRNG(rng)
            lentiMPRA_inputs = jax.nn.one_hot(batch['lentiMPRA_sequences'], 4, dtype=jnp.float32)
            lentiMPRA_k562_prediction, lentiMPRA_hepg2_prediction, lentiMPRA_wtc11_prediction = model.apply(
                params,
                lentiMPRA_inputs=lentiMPRA_inputs,
                deterministic=False,
                rngs=rng_generator(model.rng_keys()),
            )
            loss, aux_values = compute_loss(
                batch, lentiMPRA_k562_prediction, lentiMPRA_hepg2_prediction, lentiMPRA_wtc11_prediction
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
        lentiMPRA_inputs = jax.nn.one_hot(batch['lentiMPRA_sequences'], 4, dtype=jnp.float32)
        
        lentiMPRA_k562_prediction, lentiMPRA_hepg2_prediction, lentiMPRA_wtc11_prediction = model.apply(
            train_state.params,
            lentiMPRA_inputs=lentiMPRA_inputs,
            deterministic=True,
            rngs=rng_generator(model.rng_keys()),
        )
        loss, aux_values = compute_loss(
            batch, lentiMPRA_k562_prediction, lentiMPRA_hepg2_prediction, lentiMPRA_wtc11_prediction
        )
        metrics = jax_utils.collect_metrics(
            aux_values, metric_keys, prefix='eval'
        )
        metrics = jax.lax.pmean(metrics, axis_name='dp')
        return metrics, \
               lentiMPRA_k562_prediction, lentiMPRA_hepg2_prediction, lentiMPRA_wtc11_prediction, \
               batch["lentiMPRA_k562_outputs"], batch["lentiMPRA_hepg2_outputs"], batch["lentiMPRA_wtc11_outputs"], \
               batch["lentiMPRA_valid_outputs_mask"], \
               rng_generator()

    train_iterator = train_dataset.batch_iterator(pmap_axis_dim=jax_device_count)
    rng = jax.device_put_sharded(
        list(jax_utils.next_rng(jax.device_count())),
        jax.devices(),
    )
    train_state = replicate(train_state)

    best_eval_avg_SpearmanR = -np.inf

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
            all_y = {"k562": [], "hepg2": [], "wtc11": []}
            all_yhat = {"k562": [], "hepg2": [], "wtc11": []}

            eval_iterator = val_dataset.batch_iterator(pmap_axis_dim=jax_device_count)
            batch = next(eval_iterator)
            while batch is not None:
                metrics, \
                lentiMPRA_k562_prediction, lentiMPRA_hepg2_prediction, lentiMPRA_wtc11_prediction, \
                lentiMPRA_k562_outputs, lentiMPRA_hepg2_outputs, lentiMPRA_wtc11_outputs, \
                lentiMPRA_valid_outputs_mask, rng = eval_step(
                    train_state, rng, batch
                )
                eval_metrics.append(unreplicate(metrics))

                lentiMPRA_k562_prediction = jax.device_get(lentiMPRA_k562_prediction)
                lentiMPRA_hepg2_prediction = jax.device_get(lentiMPRA_hepg2_prediction)
                lentiMPRA_wtc11_prediction = jax.device_get(lentiMPRA_wtc11_prediction)
                lentiMPRA_k562_outputs = jax.device_get(lentiMPRA_k562_outputs)
                lentiMPRA_hepg2_outputs = jax.device_get(lentiMPRA_hepg2_outputs)
                lentiMPRA_wtc11_outputs = jax.device_get(lentiMPRA_wtc11_outputs)
                lentiMPRA_valid_outputs_mask = jax.device_get(lentiMPRA_valid_outputs_mask)

                lentiMPRA_k562_prediction = einops.rearrange(lentiMPRA_k562_prediction, 'd b -> (d b)')
                lentiMPRA_hepg2_prediction = einops.rearrange(lentiMPRA_hepg2_prediction, 'd b -> (d b)')
                lentiMPRA_wtc11_prediction = einops.rearrange(lentiMPRA_wtc11_prediction, 'd b -> (d b)')
                lentiMPRA_k562_outputs = einops.rearrange(lentiMPRA_k562_outputs, 'd b -> (d b)')
                lentiMPRA_hepg2_outputs = einops.rearrange(lentiMPRA_hepg2_outputs, 'd b -> (d b)')
                lentiMPRA_wtc11_outputs = einops.rearrange(lentiMPRA_wtc11_outputs, 'd b -> (d b)')
                lentiMPRA_valid_outputs_mask = einops.rearrange(lentiMPRA_valid_outputs_mask, 'd b ... -> (d b) ...')

                # only keep valid outputs
                lentiMPRA_k562_prediction = lentiMPRA_k562_prediction[lentiMPRA_valid_outputs_mask[:, 0]]
                lentiMPRA_hepg2_prediction = lentiMPRA_hepg2_prediction[lentiMPRA_valid_outputs_mask[:, 1]]
                lentiMPRA_wtc11_prediction = lentiMPRA_wtc11_prediction[lentiMPRA_valid_outputs_mask[:, 2]]
                lentiMPRA_k562_outputs = lentiMPRA_k562_outputs[lentiMPRA_valid_outputs_mask[:, 0]]
                lentiMPRA_hepg2_outputs = lentiMPRA_hepg2_outputs[lentiMPRA_valid_outputs_mask[:, 1]]
                lentiMPRA_wtc11_outputs = lentiMPRA_wtc11_outputs[lentiMPRA_valid_outputs_mask[:, 2]]

                all_y["k562"].append(lentiMPRA_k562_outputs)
                all_y["hepg2"].append(lentiMPRA_hepg2_outputs)
                all_y["wtc11"].append(lentiMPRA_wtc11_outputs)
                all_yhat["k562"].append(lentiMPRA_k562_prediction)
                all_yhat["hepg2"].append(lentiMPRA_hepg2_prediction)
                all_yhat["wtc11"].append(lentiMPRA_wtc11_prediction)

                batch = next(eval_iterator)

            eval_metrics = average_metrics(jax.device_get(eval_metrics))

            all_y = {k: np.hstack(v).reshape(-1) for k, v in all_y.items()}
            all_yhat = {k: np.hstack(v).reshape(-1) for k, v in all_yhat.items()}

            for k in all_y:
                print("Num of valid outputs for {}: {}".format(k, len(all_y[k])))

                # Compute Pearson correlation
                eval_metrics[f'eval/{k}_PearsonR'] = stats.pearsonr(
                    all_y[k], all_yhat[k]
                )[0]
                # Compute Spearman correlation
                eval_metrics[f'eval/{k}_SpearmanR'] = stats.spearmanr(
                    all_y[k], all_yhat[k]
                )[0]
                # Compute R2
                eval_metrics[f'eval/{k}_R2'] = r2_score(
                    all_y[k], all_yhat[k]
                )
            
            # Compute average Pearson correlation
            eval_metrics['eval/avg_PearsonR'] = np.mean([
                eval_metrics[f'eval/{k}_PearsonR'] for k in all_y
            ])
            # Compute average Spearman correlation
            eval_metrics['eval/avg_SpearmanR'] = np.mean([
                eval_metrics[f'eval/{k}_SpearmanR'] for k in all_y
            ])
            # Compute average R2
            eval_metrics['eval/avg_R2'] = np.mean([
                eval_metrics[f'eval/{k}_R2'] for k in all_y
            ])

            if eval_metrics['eval/avg_SpearmanR'] > best_eval_avg_SpearmanR:
                best_eval_avg_SpearmanR = eval_metrics['eval/avg_SpearmanR']
                if FLAGS.save_model:
                    logger.save_pickle(
                        jax.device_get(unreplicate(train_state).params),
                        'best_params.pkl',
                    )

            eval_metrics['eval/best_avg_SpearmanR'] = best_eval_avg_SpearmanR
            eval_metrics['step'] = step
            logger.log(eval_metrics)
            tqdm.write(pformat(eval_metrics))


if __name__ == '__main__':
    mlxu.run(main)