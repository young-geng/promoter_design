from tqdm import tqdm, trange
import numpy as np
import pandas as pd
import h5py
import mlxu

FLAGS, FLAGS_DEF = mlxu.define_flags_with_default(
    lentiMPRA_dir='./data/lentiMPRA',
    output_file='./data/lentiMPRA_data.pkl'
)

def process_lentiMPRA_data(split):
    dna_tokens = {
        'A': 0,
        'C': 1,
        'G': 2,
        'T': 3
    }
    sequences = []
    all_cells = ["K562", "HepG2", "WTC11"]
    data = pd.read_csv(f'{FLAGS.lentiMPRA_dir}/final_dataset.tsv', sep='\t', header=0)
    data = data[data["is_{}".format(split)]].reset_index(drop=True) # only keep the data from the split

    sequences.extend(data['sequence'].values)
    tokenized_sequences = []
    for seq in tqdm(sequences, ncols=0):
        tokens = [dna_tokens[base] for base in seq]
        tokenized_sequences.append(np.array(tokens, dtype=np.uint8))
    tokenized_sequences = np.stack(tokenized_sequences, axis=0).astype(np.uint8)

    all_outputs = {}
    for cell in all_cells:
        all_outputs[cell] = np.array(data[cell].values, dtype=np.float32)

    valid_outputs_mask = []
    for cell in all_cells:
        valid_outputs_mask.append(~np.isnan(all_outputs[cell]))
    valid_outputs_mask = np.stack(valid_outputs_mask, axis=1)
    print(valid_outputs_mask.shape)

    print("Number of valid outputs for each cell type:")
    for cell, mask in zip(all_cells, valid_outputs_mask.T):
        print(cell, np.sum(mask))

    for cell in all_cells:
        all_outputs[cell][np.isnan(all_outputs[cell])] = -100000
    
    return tokenized_sequences, all_outputs, valid_outputs_mask

def main(argv):
    lentiMPRA_train_sequences, lentiMPRA_train_outputs, lentiMPRA_train_mask = process_lentiMPRA_data('train')
    lentiMPRA_val_sequences, lentiMPRA_val_outputs, lentiMPRA_val_mask = process_lentiMPRA_data('val')
    lentiMPRA_test_sequences, lentiMPRA_test_outputs, lentiMPRA_test_mask = process_lentiMPRA_data('test')

    all_data = {
        'train': {
            'lentiMPRA_sequences': lentiMPRA_train_sequences,
            'lentiMPRA_k562_outputs': lentiMPRA_train_outputs['K562'],
            'lentiMPRA_hepg2_outputs': lentiMPRA_train_outputs['HepG2'],
            'lentiMPRA_wtc11_outputs': lentiMPRA_train_outputs['WTC11'],
            'lentiMPRA_valid_outputs_mask': lentiMPRA_train_mask,
        },
        'val': {
            'lentiMPRA_sequences': lentiMPRA_val_sequences,
            'lentiMPRA_k562_outputs': lentiMPRA_val_outputs['K562'],
            'lentiMPRA_hepg2_outputs': lentiMPRA_val_outputs['HepG2'],
            'lentiMPRA_wtc11_outputs': lentiMPRA_val_outputs['WTC11'],   
            'lentiMPRA_valid_outputs_mask': lentiMPRA_val_mask,
        },
        'test': {
            'lentiMPRA_sequences': lentiMPRA_test_sequences,
            'lentiMPRA_k562_outputs': lentiMPRA_test_outputs['K562'],
            'lentiMPRA_hepg2_outputs': lentiMPRA_test_outputs['HepG2'],
            'lentiMPRA_wtc11_outputs': lentiMPRA_test_outputs['WTC11'],
            'lentiMPRA_valid_outputs_mask': lentiMPRA_test_mask,
        },
    }

    mlxu.save_pickle(all_data, FLAGS.output_file)

if __name__ == '__main__':
    mlxu.run(main)
