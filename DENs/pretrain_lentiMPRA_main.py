from functools import partial
import numpy as np
import mlxu
from tqdm import tqdm, trange
from pprint import pprint, pformat
import scipy.stats as stats
from sklearn.metrics import r2_score
import os
import pdb
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import pandas as pd

import jax
import jax.numpy as jnp
import flax
from flax.jax_utils import replicate, unreplicate
from flax.training.train_state import TrainState
import optax
import einops
import mlxu.jax_utils as jax_utils
from ml_collections import ConfigDict

from data import lentiMPRADataset, diffusionDataset
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
    train_data=lentiMPRADataset.get_default_config({"split": "train", "path": "./data/lentiMPRA_data.pkl", "batch_size": 96}),
    val_data=lentiMPRADataset.get_default_config({"split": "val", "path": "./data/lentiMPRA_data.pkl", "batch_size": 96, "sequential_sample": True}),
    test_data=lentiMPRADataset.get_default_config({"split": "test", "path": "./data/lentiMPRA_data.pkl", "batch_size": 96, "sequential_sample": True, "ignore_last_batch": False}),
    diffusion_data=diffusionDataset.get_default_config({"path": "./data/diffusion_data.pkl", "batch_size": 96, "sequential_sample": True, "ignore_last_batch": False}),
    logger=mlxu.WandBLogger.get_default_config({"output_dir": "./saved_models", "project": "promoter_design_jax", "wandb_dir": "./wandb", "online": True, \
                                                "experiment_id": "lentiMPRA_pretraining_savio"}),
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
    test_dataset = lentiMPRADataset(FLAGS.test_data)

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

    @partial(jax.pmap, axis_name='dp', donate_argnums=1)
    def get_predictions(train_state, rng, batch):
        rng_generator = jax_utils.JaxRNG(rng)
        lentiMPRA_inputs = jax.nn.one_hot(batch['sequences'], 4, dtype=jnp.float32)
        
        lentiMPRA_k562_prediction, lentiMPRA_hepg2_prediction, lentiMPRA_wtc11_prediction = model.apply(
            train_state.params,
            lentiMPRA_inputs=lentiMPRA_inputs,
            deterministic=True,
            rngs=rng_generator(model.rng_keys()),
        )
        return lentiMPRA_k562_prediction, lentiMPRA_hepg2_prediction, lentiMPRA_wtc11_prediction, rng_generator()

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

    # load best params
    if FLAGS.save_model:
        print("Loading best params...")
        train_state = train_state.replace(
            params=mlxu.utils.load_pickle(os.path.join(logger.output_dir, 'best_params.pkl'))
        )
        train_state = replicate(train_state)
    
    # best val metrics
    print("Best val metrics:")
    val_iterator = val_dataset.batch_iterator(pmap_axis_dim=jax_device_count)
    batch = next(val_iterator)
    
    val_metrics = []
    all_y = {"k562": [], "hepg2": [], "wtc11": []}
    all_yhat = {"k562": [], "hepg2": [], "wtc11": []}

    while batch is not None:
        metrics, \
        lentiMPRA_k562_prediction, lentiMPRA_hepg2_prediction, lentiMPRA_wtc11_prediction, \
        lentiMPRA_k562_outputs, lentiMPRA_hepg2_outputs, lentiMPRA_wtc11_outputs, \
        lentiMPRA_valid_outputs_mask, rng = eval_step(
            train_state, rng, batch
        )
        val_metrics.append(unreplicate(metrics))

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
        
        batch = next(val_iterator)
    
    val_metrics = average_metrics(jax.device_get(val_metrics))

    all_y = {k: np.hstack(v).reshape(-1) for k, v in all_y.items()}
    all_yhat = {k: np.hstack(v).reshape(-1) for k, v in all_yhat.items()}

    for k in all_y:
        print("Num of valid outputs for {}: {}".format(k, len(all_y[k])))
        # Compute Pearson correlation
        val_metrics[f'val/{k}_PearsonR'] = stats.pearsonr(
            all_y[k], all_yhat[k]
        )[0]
        # Compute Spearman correlation
        val_metrics[f'val/{k}_SpearmanR'] = stats.spearmanr(
            all_y[k], all_yhat[k]
        )[0]
        # Compute R2
        val_metrics[f'val/{k}_R2'] = r2_score(
            all_y[k], all_yhat[k]
        )
    
    # Compute average Pearson correlation
    val_metrics['val/avg_PearsonR'] = np.mean([
        val_metrics[f'val/{k}_PearsonR'] for k in all_y
    ])
    # Compute average Spearman correlation
    val_metrics['val/avg_SpearmanR'] = np.mean([
        val_metrics[f'val/{k}_SpearmanR'] for k in all_y
    ])
    # Compute average R2
    val_metrics['val/avg_R2'] = np.mean([
        val_metrics[f'val/{k}_R2'] for k in all_y
    ])
    tqdm.write(pformat(val_metrics))

    # test metrics
    print("Test metrics:")
    test_iterator = test_dataset.batch_iterator(pmap_axis_dim=jax_device_count)
    batch = next(test_iterator)

    test_metrics = []
    all_y = {"k562": [], "hepg2": [], "wtc11": []}
    all_yhat = {"k562": [], "hepg2": [], "wtc11": []}
    all_test_set_predictions = {"k562": [], "hepg2": [], "wtc11": []}

    if os.path.exists(os.path.join(logger.output_dir, "all_test_set_predictions.pkl")):
        all_y = pickle.load(open(os.path.join(logger.output_dir, "all_y.pkl"), "rb"))
        all_yhat = pickle.load(open(os.path.join(logger.output_dir, "all_yhat.pkl"), "rb"))
        all_test_set_predictions = pickle.load(open(os.path.join(logger.output_dir, "all_test_set_predictions.pkl"), "rb"))
        print("Loaded cached test set predictions")
        test_metrics = {}
    else:
        while batch is not None:
            metrics, \
            lentiMPRA_k562_prediction, lentiMPRA_hepg2_prediction, lentiMPRA_wtc11_prediction, \
            lentiMPRA_k562_outputs, lentiMPRA_hepg2_outputs, lentiMPRA_wtc11_outputs, \
            lentiMPRA_valid_outputs_mask, rng = eval_step(
                train_state, rng, batch
            )
            test_metrics.append(unreplicate(metrics))

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

            all_test_set_predictions["k562"].append(lentiMPRA_k562_prediction)
            all_test_set_predictions["hepg2"].append(lentiMPRA_hepg2_prediction)
            all_test_set_predictions["wtc11"].append(lentiMPRA_wtc11_prediction)

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

            batch = next(test_iterator)
        
        test_metrics = average_metrics(jax.device_get(test_metrics))

        all_y = {k: np.hstack(v).reshape(-1) for k, v in all_y.items()}
        all_yhat = {k: np.hstack(v).reshape(-1) for k, v in all_yhat.items()}
        all_test_set_predictions = {k: np.hstack(v).reshape(-1) for k, v in all_test_set_predictions.items()}

        # save y and yhat
        pickle.dump(all_y, open(os.path.join(logger.output_dir, "all_y.pkl"), "wb"))
        pickle.dump(all_yhat, open(os.path.join(logger.output_dir, "all_yhat.pkl"), "wb"))
        pickle.dump(all_test_set_predictions, open(os.path.join(logger.output_dir, "all_test_set_predictions.pkl"), "wb"))

    for k in all_y:
        print("Num of valid outputs for {}: {}".format(k, len(all_y[k])))
        print("Num of all outputs for {}: {}".format(k, len(all_test_set_predictions[k])))

        # Compute Pearson correlation
        test_metrics[f'test/{k}_PearsonR'] = stats.pearsonr(
            all_y[k], all_yhat[k]
        )[0]
        # Compute Spearman correlation
        test_metrics[f'test/{k}_SpearmanR'] = stats.spearmanr(
            all_y[k], all_yhat[k]
        )[0]
        # Compute R2
        test_metrics[f'test/{k}_R2'] = r2_score(
            all_y[k], all_yhat[k]
        )

    # Compute average Pearson correlation
    test_metrics['test/avg_PearsonR'] = np.mean([
        test_metrics[f'test/{k}_PearsonR'] for k in all_y
    ])
    # Compute average Spearman correlation
    test_metrics['test/avg_SpearmanR'] = np.mean([
        test_metrics[f'test/{k}_SpearmanR'] for k in all_y
    ])
    # Compute average R2
    test_metrics['test/avg_R2'] = np.mean([
        test_metrics[f'test/{k}_R2'] for k in all_y
    ])
    tqdm.write(pformat(test_metrics))

    # create scatter plots
    fig, ax = plt.subplots(1, 3, figsize=(21, 7))
    for i, k in enumerate(all_y):
        ax[i].scatter(all_y[k], all_yhat[k], s=1.5)
        ax[i].set_xlabel("True")
        ax[i].set_ylabel("Predicted")
        
        # plot x = y line
        # add x=y line
        ax[i].plot(
            [all_yhat[k].min(), all_yhat[k].max()],
            [all_yhat[k].min(), all_yhat[k].max()],
            'k--', lw=1, label='x=y'
        )

        ax[i].set_title(k + "\n SpearmanR: {:.3f} \n PearsonR: {:.3f} \n R2: {:.3f}".format(test_metrics[f'test/{k}_SpearmanR'], test_metrics[f'test/{k}_PearsonR'], test_metrics[f'test/{k}_R2']))

    plt.savefig(os.path.join(logger.output_dir, "test_set_prediction_scatter_plots.png"))
    plt.show()

    diffusion_cells = ['HepG2', 'GM12878', 'hESCT0', 'K562']
    for cell in ['HepG2', 'GM12878', 'hESCT0', 'K562']:
        diffusion_cells.append(f"{cell}_training_sequences")
        diffusion_cells.append(f"{cell}_shuffled_training_sequences")        
    diffusion_cells = diffusion_cells + ["training_sequences", "shuffled_training_sequences"]
    if not os.path.exists(os.path.join(logger.output_dir, "all_diffusion_predictions.pkl")):
        all_diffusion_predictions = {}
        for cell in diffusion_cells:
            print("Testing diffusion for cell type: {}".format(cell))
            # load diffusion data
            diffusion_dataset = diffusionDataset(ConfigDict({"path": "./data/diffusion_data.pkl", "batch_size": 96, "sequential_sample": True, "cell": cell, "ignore_last_batch": False}))
            diffusion_iterator = diffusion_dataset.batch_iterator(pmap_axis_dim=jax_device_count)
            
            all_predictions = {"k562": [], "hepg2": [], "wtc11": []}

            batch = next(diffusion_iterator)
            while batch is not None:
                lentiMPRA_k562_prediction, lentiMPRA_hepg2_prediction, lentiMPRA_wtc11_prediction, rng = get_predictions(
                    train_state, rng, batch
                )
                
                lentiMPRA_k562_prediction = jax.device_get(lentiMPRA_k562_prediction)
                lentiMPRA_hepg2_prediction = jax.device_get(lentiMPRA_hepg2_prediction)
                lentiMPRA_wtc11_prediction = jax.device_get(lentiMPRA_wtc11_prediction)

                lentiMPRA_k562_prediction = einops.rearrange(lentiMPRA_k562_prediction, 'd b -> (d b)')
                lentiMPRA_hepg2_prediction = einops.rearrange(lentiMPRA_hepg2_prediction, 'd b -> (d b)')
                lentiMPRA_wtc11_prediction = einops.rearrange(lentiMPRA_wtc11_prediction, 'd b -> (d b)')
                
                all_predictions["k562"].append(lentiMPRA_k562_prediction)
                all_predictions["hepg2"].append(lentiMPRA_hepg2_prediction)
                all_predictions["wtc11"].append(lentiMPRA_wtc11_prediction)

                batch = next(diffusion_iterator)
            
            all_predictions = {k: np.hstack(v).reshape(-1) for k, v in all_predictions.items()}
            all_diffusion_predictions[cell] = all_predictions
        
        # save predictions
        pickle.dump(all_diffusion_predictions, open(os.path.join(logger.output_dir, "all_diffusion_predictions.pkl"), "wb"))
    else:
        print("Loading diffusion predictions from file")
        all_diffusion_predictions = pickle.load(open(os.path.join(logger.output_dir, "all_diffusion_predictions.pkl"), "rb"))

    # compare predictions on test set and diffusion data by making pairwise scatter plots
    fig, ax = plt.subplots(1, 3, figsize=(21, 7))
    count = 0
    for i, cell1 in enumerate(all_yhat):
        for j, cell2 in enumerate(all_yhat):
            if i >= j:
                continue
            
            # first plot test set predictions
            ax[count].scatter(all_test_set_predictions[cell1], all_test_set_predictions[cell2], s=1.5, label="lentiMPRA test set")

            # plot diffusion data predictions
            for diffusion_cell in diffusion_cells:
                ax[count].scatter(all_diffusion_predictions[diffusion_cell][cell1], \
                                  all_diffusion_predictions[diffusion_cell][cell2], s=1.5, label=diffusion_cell)
            # for diffusion_cell in ["training_sequences", "shuffled_training_sequences"]:
            #     ax[count].scatter(all_diffusion_predictions[diffusion_cell][cell1], \
            #                       all_diffusion_predictions[diffusion_cell][cell2], s=1.5, label=diffusion_cell)
            
            # plot x = y line
            ax[count].plot(
                [np.min(all_test_set_predictions[cell1]), np.max(all_test_set_predictions[cell1])],
                [np.min(all_test_set_predictions[cell1]), np.max(all_test_set_predictions[cell1])],
                'k--', lw=1, label='x=y'
            )

            ax[count].set_xlabel(cell1)
            ax[count].set_ylabel(cell2)
            ax[count].set_title("{} vs {}".format(cell1, cell2))
            ax[count].legend()

            count += 1
    fig.suptitle("Predictions on lentiMPRA test set and generated sequences")
    plt.savefig(os.path.join(logger.output_dir, "test_set_and_diffusion_data_scatter_plots_v2.png"))
    plt.show()

    # create violin plots for each cell type and diffusion data class
    fig, ax = plt.subplots(1, 3, figsize=(63, 15))
    count = 0
    for i, cell in enumerate(all_yhat):
        pos = 0
        # first plot test set predictions
        ax[count].violinplot([all_test_set_predictions[cell]], positions=[pos], showmeans=False, showextrema=True, showmedians=True, quantiles=[0.25, 0.75])
        pos += 1
        
        # plot diffusion data predictions
        for diffusion_cell in diffusion_cells:
            ax[count].violinplot([all_diffusion_predictions[diffusion_cell][cell]], positions=[pos], showmeans=False, showextrema=True, showmedians=True, quantiles=[0.25, 0.75])
            pos += 1
        # for diffusion_cell in ['HepG2', 'GM12878', 'hESCT0', 'K562']:
        #     ax[count].violinplot([all_diffusion_predictions[diffusion_cell][cell]], positions=[pos], showmeans=False, showextrema=True, showmedians=True, quantiles=[0.25, 0.75])
        #     pos += 1
        # for diffusion_cell in ["training_sequences", "shuffled_training_sequences"]:
        #     ax[count].violinplot([all_diffusion_predictions[diffusion_cell][cell]], positions=[pos], showmeans=False, showextrema=True, showmedians=True, quantiles=[0.25, 0.75])
        #     pos += 1
        
        ax[count].set_xticks(np.arange(pos))
        ax[count].set_xticklabels(["lentiMPRA test set"] + diffusion_cells)
        ax[count].set_title(cell)
        # rotate x axis labels by 90 degrees
        ax[count].tick_params(axis='x', rotation=90)
        
        count += 1
    fig.suptitle("Distribution of predictions on lentiMPRA test set and diffusion model training data")
    plt.savefig(os.path.join(logger.output_dir, "test_set_and_diffusion_data_violin_plots.png"))
    plt.show()

    # put predictions into a dataframe
    diffusion_data_df = pd.read_csv("./data/diffusion/K562_hESCT0_HepG2_GM12878_12k_sequences_per_group_with_shuffled.txt", sep="\t")
    for cell in ["k562", "hepg2", "wtc11"]:
        diffusion_data_df[cell + "_lentiMPRA_prediction"] = all_diffusion_predictions["training_sequences"][cell]
        diffusion_data_df[cell + "_shuffled_sequence_lentiMPRA_prediction"] = all_diffusion_predictions["shuffled_training_sequences"][cell]
    diffusion_data_df.to_csv(os.path.join(logger.output_dir, "diffusion_data_with_predictions.txt"), sep="\t", index=False)

if __name__ == '__main__':
    mlxu.run(main)