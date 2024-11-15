from tqdm import tqdm, trange
import numpy as np
import pandas as pd
import h5py
import mlxu


FLAGS, FLAGS_DEF = mlxu.define_flags_with_default(
    sure_dir='./data/SuRE',
    mpra_dir='./data/Sharpr_MPRA',
    sure_max_len=1000,
    output_file='./data/all_data.pkl'
)


def process_sure_data(split):
    dna_tokens = {
        'A': 0,
        'C': 1,
        'G': 2,
        'T': 3,
        'N': 4,
    }
    sequences = []
    k562_labels = []
    hepg2_labels = []
    for i, dataset in enumerate(['SuRE42_HG02601', 'SuRE43_GM18983', 'SuRE44_HG01241', 'SuRE45_HG03464']):
        data = pd.read_csv(f'{FLAGS.sure_dir}/{dataset}/combined_bins_{split}_set.tsv', sep='\t', header=0)
        sequences.extend(data['sequence'].values)
        k562_labels.extend(data['K562_bin'].values + 5 * i)
        hepg2_labels.extend(data['HepG2_bin'].values + 5 * i)

    tokenized_sequences = []
    for seq in tqdm(sequences, ncols=0):
        tokens = [dna_tokens[base] for base in seq]
        tokens.extend([4 for _ in range(FLAGS.sure_max_len - len(seq))])
        tokenized_sequences.append(np.array(tokens, dtype=np.uint8))

    tokenized_sequences = np.stack(tokenized_sequences, axis=0).astype(np.uint8)
    k562_labels = np.array(k562_labels, dtype=np.int32)
    hepg2_labels = np.array(hepg2_labels, dtype=np.int32)

    return tokenized_sequences, k562_labels, hepg2_labels


def process_mpra_data(split):
    h5_file = h5py.File(f'{FLAGS.mpra_dir}/{split}.hdf5', 'r')
    seq_array = np.array(h5_file['X']['sequence'])
    sequences = np.argmax(seq_array, axis=-1).astype(np.uint8)
    output = np.array(h5_file['Y']['output']).astype('float32')
    return sequences, output


def main(argv):
    sure_train_sequences, sure_train_k562_labels, sure_train_hepg2_labels = process_sure_data('train')
    sure_val_sequences, sure_val_k562_labels, sure_val_hepg2_labels = process_sure_data('val')
    sure_test_sequences, sure_test_k562_labels, sure_test_hepg2_labels = process_sure_data('test')

    mpra_train_sequences, mpra_train_output = process_mpra_data('train')
    mpra_val_sequences, mpra_val_output = process_mpra_data('val')
    mpra_test_sequences, mpra_test_output = process_mpra_data('test')

    all_data = {
        'train': {
            'sure_sequences': sure_train_sequences,
            'sure_k562_labels': sure_train_k562_labels,
            'sure_hepg2_labels': sure_train_hepg2_labels,
            'mpra_sequences': mpra_train_sequences,
            'mpra_output': mpra_train_output,
        },
        'val': {
            'sure_sequences': sure_val_sequences,
            'sure_k562_labels': sure_val_k562_labels,
            'sure_hepg2_labels': sure_val_hepg2_labels,
            'mpra_sequences': mpra_val_sequences,
            'mpra_output': mpra_val_output,
        },
        'test': {
            'sure_sequences': sure_test_sequences,
            'sure_k562_labels': sure_test_k562_labels,
            'sure_hepg2_labels': sure_test_hepg2_labels,
            'mpra_sequences': mpra_test_sequences,
            'mpra_output': mpra_test_output,
        },
    }

    mlxu.save_pickle(all_data, FLAGS.output_file)

if __name__ == '__main__':
    mlxu.run(main)
