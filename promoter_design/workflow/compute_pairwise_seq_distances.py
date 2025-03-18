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
from mlxu import jax_utils

from promoter_design.workflow.data import tokenize_sequences

np.random.seed(97)
jax_utils.set_random_seed(97)

FLAGS, FLAGS_DEF = mlxu.define_flags_with_default(
    sequences_file='',
    summary_ensemble_data_file='',
    output_file='',
    kmer_k=6,
)

def get_all_kmers(k):
    bases = ["A", "C", "G", "T"]
    all_kmers = [""] * (len(bases)**k)

    for i in range(k):
        for j in range(int(len(bases)**i)):
            for b, base in enumerate(bases):
                for l in range(len(bases)**(k - i - 1)):
                    ind = int(l + (j*len(bases) + b)*(len(bases)**(k - i - 1)))
                    all_kmers[ind] = all_kmers[ind][:i] + base
    
    assert len(set(all_kmers)) == len(bases)**k
    
    kmer_to_ind = {}
    for i, kmer in enumerate(all_kmers):
        kmer_to_ind[kmer] = i
    
    return all_kmers, kmer_to_ind


def get_kmer_counts(seq, kmer_size, kmer_to_ind):
    assert len(seq) >= kmer_size
    kmer_counts = np.zeros(4**kmer_size)
    for i in range(len(seq) - kmer_size + 1):
        kmer_counts[kmer_to_ind[seq[i: i+kmer_size]]] += 1
    return kmer_counts


def get_kmer_features(seqs, kmer_k):
    all_kmers, kmer_to_ind = get_all_kmers(kmer_k)
    counts = []
    for seq in seqs:
        counts.append(get_kmer_counts(seq, kmer_k, kmer_to_ind))
    return np.stack(counts, axis=0)


@jax.jit
def jax_pairwise_edit_distance(seqs):
    def edit_distance(s):
        return jnp.not_equal(s[None, :], seqs).sum(-1).astype(jnp.float16)
    
    return jax.lax.map(edit_distance, seqs)


def pairwise_edit_distance(seqs):
    seqs = np.array(tokenize_sequences(seqs))
    return jax.device_get(jax_pairwise_edit_distance(seqs))


@jax.jit
def jax_pairwise_distance(features):
    def distance(x):
        return jnp.linalg.norm(x[None, :] - features, axis=-1).astype(jnp.float16)
    return jax.lax.map(distance, features)


def pairwise_distance(features):
    return jax.device_get(jax_pairwise_distance(features))


def extract_seq_data_from_df(table, ensemble_data):
    extracted_data = {}
    for key in ['THP1', 'Jurkat', 'K562']:
        sequences = table[table['designed_for'] == key]['sequence'].values
        kmer_features = get_kmer_features(sequences, FLAGS.kmer_k)

        key = key.lower()
        extracted_data[key] = {
            'sequences': sequences,
            'kmer_features': kmer_features
        }
        
        thp1_mean = []
        thp1_std = []
        jurkat_mean = []
        jurkat_std = []
        k562_mean = []
        k562_std = []

        thp1_diff_mean = []
        thp1_diff_std = []
        jurkat_diff_mean = []
        jurkat_diff_std = []
        k562_diff_mean = []
        k562_diff_std = []

        for seq in sequences:
            ensemble_result = ensemble_data[seq]
            thp1_mean.append(ensemble_result['thp1_mean'])
            thp1_std.append(ensemble_result['thp1_std'])

            jurkat_mean.append(ensemble_result['jurkat_mean'])
            jurkat_std.append(ensemble_result['jurkat_std'])

            k562_mean.append(ensemble_result['k562_mean'])
            k562_std.append(ensemble_result['k562_std'])

            thp1_diff_mean.append(ensemble_result['thp1_diff_mean'])
            thp1_diff_std.append(ensemble_result['thp1_diff_std'])

            jurkat_diff_mean.append(ensemble_result['jurkat_diff_mean'])
            jurkat_diff_std.append(ensemble_result['jurkat_diff_std'])

            k562_diff_mean.append(ensemble_result['k562_diff_mean'])
            k562_diff_std.append(ensemble_result['k562_diff_std'])
            
        extracted_data[key][f'thp1_mean'] = np.array(thp1_mean)
        extracted_data[key][f'thp1_std'] = np.array(thp1_std)

        extracted_data[key][f'jurkat_mean'] = np.array(jurkat_mean)
        extracted_data[key][f'jurkat_std'] = np.array(jurkat_std)

        extracted_data[key][f'k562_mean'] = np.array(k562_mean)
        extracted_data[key][f'k562_std'] = np.array(k562_std)


        extracted_data[key][f'thp1_diff_mean'] = np.array(thp1_diff_mean)
        extracted_data[key][f'thp1_diff_std'] = np.array(thp1_diff_std)

        extracted_data[key][f'jurkat_diff_mean'] = np.array(jurkat_diff_mean)
        extracted_data[key][f'jurkat_diff_std'] = np.array(jurkat_diff_std)

        extracted_data[key][f'k562_diff_mean'] = np.array(k562_diff_mean)
        extracted_data[key][f'k562_diff_std'] = np.array(k562_diff_std)

        extracted_data[key][f'target_diff_mean'] = (
            extracted_data[key][f'{key.lower()}_diff_mean']
        )
        extracted_data[key][f'target_diff_std'] = (
            extracted_data[key][f'{key.lower()}_diff_std']
        )
        
    return extracted_data 

def main(argv):
    assert FLAGS.sequences_file != ''
    assert FLAGS.summary_ensemble_data_file != ''
    assert FLAGS.output_file != ''

    sequences = pd.read_parquet(FLAGS.sequences_file)
    ensemble_data = mlxu.load_pickle(FLAGS.summary_ensemble_data_file)
    seq_data = extract_seq_data_from_df(sequences, ensemble_data)

    seqlen = len(seq_data['k562']['sequences'][0])

    for key in ['k562', 'jurkat', 'thp1']:
        seq_data[key]['distance_matrix'] = 0.5 * (
            pairwise_edit_distance(seq_data[key]['sequences']) / float(seqlen)
            + pairwise_distance(seq_data[key]['kmer_features']) / float(seqlen - FLAGS.kmer_k + 1) / np.sqrt(2)
        )
    
    mlxu.save_pickle(seq_data, FLAGS.output_file)
    

if __name__ == '__main__':
    mlxu.run(main)