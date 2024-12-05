from tqdm import tqdm, trange
import numpy as np
import pandas as pd
import h5py
import mlxu


FLAGS, FLAGS_DEF = mlxu.define_flags_with_default(
    input_file='',
    output_file='',
    sequence_column='sequence',
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
    sequences = tokenize_sequences(data[FLAGS.sequence_column].values)

    all_data = {
        'sequences': sequences,
    }

    mlxu.save_pickle(all_data, FLAGS.output_file)

if __name__ == '__main__':
    mlxu.run(main)
