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

from promoter_design.workflow.data import decode_sequences

np.random.seed(97)

FLAGS, FLAGS_DEF = mlxu.define_flags_with_default(
    ensemble_data_file='',
    summary_output_file='',
)

def summarize_ensemble_preds(filepath):
    seq_evals = {}
    
    data = mlxu.load_pickle(filepath)

    if 'sequences' in data or 'sequence' in data:
        key = 'sequences' if 'sequences' in data else 'sequence'
        sequences = decode_sequences(data[key], progress=True)

        for i, seq in enumerate(sequences):
            seq_evals[seq] = {
                'thp1_mean': np.mean(data[f'ensemble_{key}_thp1_pred'][:, i]),
                'thp1_std': np.std(data[f'ensemble_{key}_thp1_pred'][:, i]),

                'jurkat_mean': np.mean(data[f'ensemble_{key}_jurkat_pred'][:, i]),
                'jurkat_std': np.std(data[f'ensemble_{key}_jurkat_pred'][:, i]),

                'k562_mean': np.mean(data[f'ensemble_{key}_k562_pred'][:, i]),
                'k562_std': np.std(data[f'ensemble_{key}_k562_pred'][:, i]),

                'thp1_diff_mean': np.mean(
                    data[f'ensemble_{key}_thp1_pred'][:, i]
                    - 0.5 * data[f'ensemble_{key}_jurkat_pred'][:, i]
                    - 0.5 * data[f'ensemble_{key}_k562_pred'][:, i]
                ),
                'thp1_diff_std': np.std(
                    data[f'ensemble_{key}_thp1_pred'][:, i]
                    - 0.5 * data[f'ensemble_{key}_jurkat_pred'][:, i]
                    - 0.5 * data[f'ensemble_{key}_k562_pred'][:, i]
                ),

                'jurkat_diff_mean': np.mean(
                    data[f'ensemble_{key}_jurkat_pred'][:, i]
                    - 0.5 * data[f'ensemble_{key}_thp1_pred'][:, i]
                    - 0.5 * data[f'ensemble_{key}_k562_pred'][:, i]
                ),
                'jurkat_diff_std': np.std(
                    data[f'ensemble_{key}_jurkat_pred'][:, i]
                    - 0.5 * data[f'ensemble_{key}_thp1_pred'][:, i]
                    - 0.5 * data[f'ensemble_{key}_k562_pred'][:, i]
                ),

                'k562_diff_mean': np.mean(
                    data[f'ensemble_{key}_k562_pred'][:, i]
                    - 0.5 * data[f'ensemble_{key}_thp1_pred'][:, i]
                    - 0.5 * data[f'ensemble_{key}_jurkat_pred'][:, i]
                ),
                'k562_diff_std': np.std(
                    data[f'ensemble_{key}_k562_pred'][:, i]
                    - 0.5 * data[f'ensemble_{key}_thp1_pred'][:, i]
                    - 0.5 * data[f'ensemble_{key}_jurkat_pred'][:, i]
                ),


            }
    else:
        for key in ['thp1_optimized_seq', 'jurkat_optimized_seq', 'k562_optimized_seq']:
            sequences = decode_sequences(data[key], progress=True)
            for i, seq in enumerate(sequences):
                seq_evals[seq] = {
                    'thp1_mean': np.mean(data[f'ensemble_{key}_thp1_pred'][:, i]),
                    'thp1_std': np.std(data[f'ensemble_{key}_thp1_pred'][:, i]),

                    'jurkat_mean': np.mean(data[f'ensemble_{key}_jurkat_pred'][:, i]),
                    'jurkat_std': np.std(data[f'ensemble_{key}_jurkat_pred'][:, i]),

                    'k562_mean': np.mean(data[f'ensemble_{key}_k562_pred'][:, i]),
                    'k562_std': np.std(data[f'ensemble_{key}_k562_pred'][:, i]),

                    'thp1_diff_mean': np.mean(
                        data[f'ensemble_{key}_thp1_pred'][:, i]
                        - 0.5 * data[f'ensemble_{key}_jurkat_pred'][:, i]
                        - 0.5 * data[f'ensemble_{key}_k562_pred'][:, i]
                    ),
                    'thp1_diff_std': np.std(
                        data[f'ensemble_{key}_thp1_pred'][:, i]
                        - 0.5 * data[f'ensemble_{key}_jurkat_pred'][:, i]
                        - 0.5 * data[f'ensemble_{key}_k562_pred'][:, i]
                    ),

                    'jurkat_diff_mean': np.mean(
                        data[f'ensemble_{key}_jurkat_pred'][:, i]
                        - 0.5 * data[f'ensemble_{key}_thp1_pred'][:, i]
                        - 0.5 * data[f'ensemble_{key}_k562_pred'][:, i]
                    ),
                    'jurkat_diff_std': np.std(
                        data[f'ensemble_{key}_jurkat_pred'][:, i]
                        - 0.5 * data[f'ensemble_{key}_thp1_pred'][:, i]
                        - 0.5 * data[f'ensemble_{key}_k562_pred'][:, i]
                    ),

                    'k562_diff_mean': np.mean(
                        data[f'ensemble_{key}_k562_pred'][:, i]
                        - 0.5 * data[f'ensemble_{key}_thp1_pred'][:, i]
                        - 0.5 * data[f'ensemble_{key}_jurkat_pred'][:, i]
                    ),
                    'k562_diff_std': np.std(
                        data[f'ensemble_{key}_k562_pred'][:, i]
                        - 0.5 * data[f'ensemble_{key}_thp1_pred'][:, i]
                        - 0.5 * data[f'ensemble_{key}_jurkat_pred'][:, i]
                    ),
                }
    return seq_evals
    

def main(argv):
    assert FLAGS.ensemble_data_file != ''
    assert FLAGS.summary_output_file != ''
    
    seq_evals = summarize_ensemble_preds(FLAGS.ensemble_data_file)
    mlxu.save_pickle(seq_evals, FLAGS.summary_output_file)


if __name__ == '__main__':
    mlxu.run(main)