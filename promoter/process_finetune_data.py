from tqdm import tqdm, trange
import numpy as np
import pandas as pd
import h5py
import mlxu


FLAGS, FLAGS_DEF = mlxu.define_flags_with_default(
    input_file='./data/final.csv',
    output_file='./data/finetune_data.pkl'
)


def tokenize_sequences(sequences):
    dna_tokens = {
        'A': 0,
        'C': 1,
        'G': 2,
        'T': 3,
        'N': 4,
    }
    tokenized_sequences = []
    for seq in tqdm(sequences, ncols=0):
        tokens = [dna_tokens[base] for base in seq]
        tokenized_sequences.append(np.array(tokens, dtype=np.uint8))

    return np.stack(tokenized_sequences, axis=0).astype(np.uint8)


def main(argv):
    data = pd.read_csv(FLAGS.input_file, header=0)
    train_sequences = tokenize_sequences(data[data['is_train']]['sequence'].values)
    train_k562_output = data[data['is_train']]['K562'].values.astype('float32')
    train_jurkat_output = data[data['is_train']]['JURKAT'].values.astype('float32')
    train_thp1_output = data[data['is_train']]['THP1'].values.astype('float32')

    val_sequences = tokenize_sequences(data[data['is_val']]['sequence'].values)
    val_k562_output = data[data['is_val']]['K562'].values.astype('float32')
    val_jurkat_output = data[data['is_val']]['JURKAT'].values.astype('float32')
    val_thp1_output = data[data['is_val']]['THP1'].values.astype('float32')

    test_sequences = tokenize_sequences(data[data['is_test']]['sequence'].values)
    test_k562_output = data[data['is_test']]['K562'].values.astype('float32')
    test_jurkat_output = data[data['is_test']]['JURKAT'].values.astype('float32')
    test_thp1_output = data[data['is_test']]['THP1'].values.astype('float32')

    all_data = {
        'train': {
            'sequences': train_sequences,
            'thp1_output': train_thp1_output,
            'jurkat_output': train_jurkat_output,
            'k562_output': train_k562_output,
        },
        'val': {
            'sequences': val_sequences,
            'thp1_output': val_thp1_output,
            'jurkat_output': val_jurkat_output,
            'k562_output': val_k562_output,
        },
        'test': {
            'sequences': test_sequences,
            'thp1_output': test_thp1_output,
            'jurkat_output': test_jurkat_output,
            'k562_output': test_k562_output,
        },
    }

    mlxu.save_pickle(all_data, FLAGS.output_file)

if __name__ == '__main__':
    mlxu.run(main)
