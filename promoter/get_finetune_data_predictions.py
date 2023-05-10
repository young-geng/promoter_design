from functools import partial
import numpy as np
import scipy.stats as stats
from sklearn.metrics import r2_score
import mlxu
from tqdm import tqdm, trange
from pprint import pprint, pformat
import os
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import flax
from flax.jax_utils import replicate, unreplicate
from flax.training.train_state import TrainState
import optax
import einops
import mlxu.jax_utils as jax_utils
from ml_collections import ConfigDict

import pdb

from .data import FinetuneDataset
from .model import FinetuneNetwork
from .utils import average_metrics, global_norm, get_weight_decay_mask, get_generic_mask


FLAGS, FLAGS_DEF = mlxu.define_flags_with_default(
    seed=42,
    pretrained_predictor_path="./data/finetune_coms_0.0.pkl",
    oracle_test_data=FinetuneDataset.get_default_config({"split": "test", "path": "./data/finetune_data.pkl", "sequential_sample": True, "batch_size": 192, "ignore_last_batch": True}),
    predictions_save_dir="./predictions",
)

def reshape_batch_for_pmap(batch, pmap_axis_dim):
    return einops.rearrange(batch, '(p b) ... -> p b ...', p=pmap_axis_dim)

def main(argv):
    jax_utils.set_random_seed(FLAGS.seed)
    jax_device_count = jax.device_count()

    # create dirs
    os.makedirs(FLAGS.predictions_save_dir, exist_ok=True)
    output_dir = os.path.join(FLAGS.predictions_save_dir, FLAGS.pretrained_predictor_path.split('/')[-1])
    os.makedirs(output_dir, exist_ok=True)
    
    # create predictor
    predictor = FinetuneNetwork(FLAGS.predictor_config_updates)

    # init model
    predictor_params = predictor.init(
        inputs=jnp.zeros((16, 1000, 4)),
        deterministic=False,
        rngs=jax_utils.next_rng(predictor.rng_keys()),
    )

    # load pretrained predictor
    predictor_params = flax.core.unfreeze(predictor_params)
    predictor_params['params'] = jax.device_put(
        mlxu.load_pickle(FLAGS.pretrained_predictor_path)['params']
    )
    predictor_params = flax.core.freeze(predictor_params)

    # create train state
    predictor_train_state = TrainState.create(
        params=predictor_params,
        tx=optax.set_to_zero(),
        apply_fn=None
    )
    
    # create RNGs
    rng = jax.device_put_sharded(
        list(jax_utils.next_rng(jax.device_count())),
        jax.devices(),
    )

    # replicate train state across devices
    predictor_train_state = replicate(predictor_train_state)

    # function to get predictions of the pretrained predictor
    @partial(jax.pmap, axis_name='dp', donate_argnums=1)
    def eval_predictor_step(predictor_train_state, rng, batch):
        rng_generator = jax_utils.JaxRNG(rng)

        _, thp1_output, jurkat_output, k562_output = predictor.apply(
            predictor_train_state.params,
            inputs=jax.nn.one_hot(batch['sequences'], 5, dtype=jnp.float32)[:, :, :4],
            deterministic=True,
            rngs=rng_generator(predictor.rng_keys()),
        )

        return batch['thp1_output'], batch['jurkat_output'], batch['k562_output'], \
               thp1_output, jurkat_output, k562_output, \
               rng_generator()

    # first get predictions of the pretrained predictor on the oracle data
    oracle_test_data = FinetuneDataset(FLAGS.oracle_test_data)
    oracle_test_iterator = oracle_test_data.batch_iterator(pmap_axis_dim=jax_device_count)
    batch = next(oracle_test_iterator)
    all_y = {'THP1': [], 'Jurkat': [], 'K562': []}
    all_yhat = {'THP1': [], 'Jurkat': [], 'K562': []}
    while batch is not None:
        thp1_y, jurkat_y, k562_y, \
        thp1_output, jurkat_output, k562_output, \
        rng = eval_predictor_step(
            predictor_train_state, rng, batch
        )
        all_y['THP1'].append(jax.device_get(thp1_y))
        all_y['Jurkat'].append(jax.device_get(jurkat_y))
        all_y['K562'].append(jax.device_get(k562_y))

        all_yhat['THP1'].append(jax.device_get(thp1_output))
        all_yhat['Jurkat'].append(jax.device_get(jurkat_output))
        all_yhat['K562'].append(jax.device_get(k562_output))

        batch = next(oracle_test_iterator)

    all_y = {k: np.hstack(v).reshape(-1) for k, v in all_y.items()}
    all_yhat = {k: np.hstack(v).reshape(-1) for k, v in all_yhat.items()}
    print("y shape: {}".format(all_y["THP1"].shape))
    print("yhat shape: {}".format(all_yhat["THP1"].shape))
    
    test_metrics = {}
    for k in all_y:
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

    # print test metrics
    print('Oracle dataset test metrics:')
    print(pformat(test_metrics))

    # create plots of predictions vs. ground truth
    fig, axes = plt.subplots(1, 3, figsize=(15, 7))
    for i, k in enumerate(all_y):
        axes[i].scatter(all_y[k], all_yhat[k], s=1)

        # draw x=y line
        axes[i].plot(
            [all_y[k].min(), all_y[k].max()],
            [all_y[k].min(), all_y[k].max()],
            'k--',
            lw=1
        )

        axes[i].set_xlabel('ground truth')
        axes[i].set_ylabel('prediction')
        axes[i].set_title(k + '\nSpearmanR: {:.3f}'.format(test_metrics[f'test/{k}_SpearmanR']) + '\nPearsonR: {:.3f}'.format(test_metrics[f'test/{k}_PearsonR']))
    plt.savefig(os.path.join(output_dir, 'oracle_test_predictions.png'))
    plt.close()
    
    # create pairwise plots of predictions and ground truth
    fig, axes = plt.subplots(1, 3, figsize=(15, 7))
    count = 0
    for i, k in enumerate(all_y):
        for j, k2 in enumerate(all_y):
            if i >= j:
                continue
            axes[count].scatter(all_y[k], all_y[k2], s=1, label="ground truth")
            axes[count].scatter(all_yhat[k], all_yhat[k2], s=1, label="prediction")

            # draw x=y line
            axes[count].plot(
                [all_y[k].min(), all_y[k].max()],
                [all_y[k].min(), all_y[k].max()],
                'k--',
                lw=1
            )

            axes[count].set_xlabel(k)
            axes[count].set_ylabel(k2)
            axes[count].legend()
            axes[count].set_title(k + ' vs. ' + k2)
            count += 1
    plt.savefig(os.path.join(output_dir, 'oracle_test_predictions_pairwise.png'))
    plt.close()

    # save predictions and ground truth
    np.save(os.path.join(output_dir, 'oracle_test_predictions.npy'), all_yhat)
    np.save(os.path.join(output_dir, 'oracle_test_ground_truth.npy'), all_y)