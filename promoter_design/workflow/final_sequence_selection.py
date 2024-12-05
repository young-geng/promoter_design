from functools import partial
import numpy as np
import pandas as pd
import os
import pdb
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from tqdm.notebook import tqdm
from sklearn.metrics import pairwise_distances

import mlxu

import jax
import jax.numpy as jnp
from jax.sharding import PositionalSharding
from jax.experimental import mesh_utils
import mlxu.jax_utils as jax_utils


jax_utils.set_random_seed(42)
np.random.seed(42)


FLAGS, FLAGS_DEF = mlxu.define_flags_with_default(
    input_file='',
    output_file='',
    distance_coef=10,
    n_selections=4000,
)


@jax.jit
def select_sequence_set(distance, diff_exp, distance_coef, n_selections):
    distance = distance.astype(jnp.bfloat16)
    diff_exp = diff_exp.astype(jnp.bfloat16)
    current_selection = jnp.zeros((distance.shape[0],), dtype=jnp.bfloat16)

    def select_one(i, current_selection):
        select_target = (
            distance_coef * distance @ current_selection / (i + 1)
            + diff_exp
        )
        select_target = jnp.where(
            current_selection > 0,
            -1e8,
            select_target,
        )
        selected = jnp.argmax(select_target, axis=-1)
        return current_selection.at[selected].set(1.0)
    
    selected = jax.lax.fori_loop(
        0, n_selections, select_one, current_selection
    )
    average_diff = jnp.mean(selected * diff_exp)
    average_distance = selected @ distance @ selected / (n_selections ** 2)
    return (
        selected.astype(jnp.float32),
        average_diff.astype(jnp.float32),
        average_distance.astype(jnp.float32),
    )


def greedy_select_sequences(data, distance_coef, n_selections):
    device_count = jax.device_count()
    distance = data['distance_matrix']
    diff_lcb = data['target_diff_mean'] - data['target_diff_std']
    size = distance.shape[0]
    if size % device_count != 0:
        padded_size = int(size / device_count + 1) * device_count
        padded_distance = np.zeros(
            (padded_size, padded_size),
            dtype=distance.dtype
        )
        padded_distance[:size, :size] = distance
        distance = padded_distance
        
        padded_diff_lcb = np.zeros(padded_size, dtype=diff_lcb.dtype)
        padded_diff_lcb[:size] = diff_lcb
        diff_lcb = padded_diff_lcb
        
    sharded_distance = jax.device_put(
        distance,
        PositionalSharding(
            mesh_utils.create_device_mesh((jax.device_count(), 1))
        )
    )
    selected, average_diff, average_distance = jax.device_get(
        select_sequence_set(
            sharded_distance,
            diff_lcb,
            distance_coef,
            n_selections,
        )
    )
    selected = selected[:size]
    return selected, average_diff, average_distance


def select_sequences(data, distance_coef, n_selections):
    new_data = {}
    for target in ['jurkat', 'thp1', 'k562']:
        selected, average_diff, average_distance = greedy_select_sequences(
            data[target],
            distance_coef,
            n_selections,
        )
        selected = selected > 0
        new_data[target] = {}
        
        print(f'{target} diff: {average_diff}, distance: {average_distance}')
        
        for key in data[target]:
            if key == 'distance_matrix':
                continue
            new_data[target][key] = data[target][key][selected, ...]
    
    return new_data


def main(argv):
    data = mlxu.load_pickle(FLAGS.input_file)
    selected_data = select_sequences(data, FLAGS.distance_coef, FLAGS.n_selections)
    mlxu.save_pickle(selected_data, FLAGS.output_file)


if __name__ == '__main__':
    mlxu.run(main)