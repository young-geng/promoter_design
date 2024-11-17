import os
from functools import partial
import numpy as np
import mlxu
from tqdm import tqdm, trange
from pprint import pprint, pformat
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
from flax.jax_utils import replicate, unreplicate
import einops
import mlxu.jax_utils as jax_utils

from promoter_design.workflow.model import FinetuneNetwork


FLAGS, FLAGS_DEF = mlxu.define_flags_with_default(
    seed=42,
    batch_size=128,
    load_sequence='',
    sequence_dict_keys='sequences',
    load_model_dir='',
    output_file='',
    model_file_name='best_model.pkl',
    logger=mlxu.WandBLogger.get_default_config(),
)


def predict_model(model_path, sequences):
    model_data = mlxu.load_pickle(model_path)
    model = FinetuneNetwork(model_data['model_config'])
    params = replicate(model_data['params'])

    @partial(jax.pmap, axis_name='dp')
    def predict_fn(params, rng, seq):
        rng_generator = jax_utils.JaxRNG(rng)
        seq = jax.nn.one_hot(seq, 5, dtype=jnp.float32)[:, :, :4]
        thp1_pred, jurkat_pred, k562_pred = model.apply(
            params,
            inputs=seq,
            deterministic=True,
            rngs=rng_generator(model.rng_keys()),
        )
        return thp1_pred, jurkat_pred, k562_pred

    num_sequences = sequences.shape[0]
    if num_sequences % FLAGS.batch_size != 0:
        pad_length = FLAGS.batch_size - num_sequences % FLAGS.batch_size
        sequences = np.concatenate(
            [sequences, np.zeros((pad_length, sequences.shape[1]), dtype=np.int32)],
            axis=0
        )
    sequences = einops.rearrange(sequences, '(n b) l -> n b l', b=FLAGS.batch_size)

    n_devices = jax.device_count()
    rng = jax.device_put_sharded(
        list(jax_utils.next_rng(n_devices)),
        jax.devices(),
    )

    thp1_preds = []
    jurkat_preds = []
    k562_preds = []

    for i in range(sequences.shape[0]):
        batch = sequences[i, ...]
        batch = einops.rearrange(batch, '(k b) l -> k b l', k=n_devices)
        thp1_pred, jurkat_pred, k562_pred = predict_fn(
            params=params,
            rng=rng,
            seq=batch
        )
        thp1_pred = einops.rearrange(jax.device_get(thp1_pred), 'k b -> (k b)')
        jurkat_pred = einops.rearrange(jax.device_get(jurkat_pred), 'k b -> (k b)')
        k562_pred = einops.rearrange(jax.device_get(k562_pred), 'k b-> (k b)')
        thp1_preds.append(thp1_pred)
        jurkat_preds.append(jurkat_pred)
        k562_preds.append(k562_pred)

    thp1_preds = np.concatenate(thp1_preds, axis=0)[:num_sequences, ...]
    jurkat_preds = np.concatenate(jurkat_preds, axis=0)[:num_sequences, ...]
    k562_preds = np.concatenate(k562_preds, axis=0)[:num_sequences, ...]

    return thp1_preds, jurkat_preds, k562_preds


def predict_all(sequences):
    models = []
    for d in os.listdir(FLAGS.load_model_dir):
        model_full_path = os.path.join(
            FLAGS.load_model_dir, d, FLAGS.model_file_name
        )
        if os.path.isfile(model_full_path):
            models.append(model_full_path)

    thp1_preds = []
    jurkat_preds = []
    k562_preds = []

    for model in tqdm(models, ncols=0):
        thp1_pred, jurkat_pred, k562_pred = predict_model(model, sequences)
        thp1_preds.append(thp1_pred)
        jurkat_preds.append(jurkat_pred)
        k562_preds.append(k562_pred)

    thp1_preds = np.stack(thp1_preds, axis=0)
    jurkat_preds = np.stack(jurkat_preds, axis=0)
    k562_preds = np.stack(k562_preds, axis=0)

    return thp1_preds, jurkat_preds, k562_preds


def main(argv):
    assert FLAGS.load_model_dir != ''
    assert FLAGS.load_sequence != ''
    assert FLAGS.output_file != ''
    jax_utils.set_random_seed(FLAGS.seed)

    sequence_data = mlxu.load_pickle(FLAGS.load_sequence)

    for key in tqdm(FLAGS.sequence_dict_keys.split(','), ncols=0):
        thp1_preds, jurkat_preds, k562_preds = predict_all(sequence_data[key])
        sequence_data[f'ensemble_{key}_thp1_pred'] = thp1_preds
        sequence_data[f'ensemble_{key}_jurkat_pred'] = jurkat_preds
        sequence_data[f'ensemble_{key}_k562_pred'] = k562_preds

    mlxu.save_pickle(sequence_data, FLAGS.output_file)



if __name__ == '__main__':
    mlxu.run(main)