from tqdm import tqdm, trange
import numpy as np
import pandas as pd
import h5py
import mlxu

FLAGS, FLAGS_DEF = mlxu.define_flags_with_default(
    diffusion_dir='./data/diffusion',
    output_file='./data/diffusion_data.pkl'
)

def process_diffusion_data(cell):
    dna_tokens = {
        'A': 0,
        'C': 1,
        'G': 2,
        'T': 3
    }
    sequences = []
    
    data = pd.read_csv(f'{FLAGS.diffusion_dir}/{cell}.txt', sep='\t', header=0)

    sequences.extend(data["49"].values)
    tokenized_sequences = []
    for seq in tqdm(sequences, ncols=0):
        tokens = [dna_tokens[base] for base in seq]
        tokenized_sequences.append(np.array(tokens, dtype=np.uint8))
    tokenized_sequences = np.stack(tokenized_sequences, axis=0).astype(np.uint8)

    return tokenized_sequences

def process_diffusion_training_data():
    dna_tokens = {
        'A': 0,
        'C': 1,
        'G': 2,
        'T': 3
    }
    sequences = []
    
    data = pd.read_csv(f'{FLAGS.diffusion_dir}/K562_hESCT0_HepG2_GM12878_12k_sequences_per_group.txt', sep='\t', header=0)

    sequences.extend(data["sequence"].values)
    tokenized_sequences = []
    for seq in tqdm(sequences, ncols=0):
        tokens = [dna_tokens[base] for base in seq]
        tokenized_sequences.append(np.array(tokens, dtype=np.uint8))
    tokenized_sequences = np.stack(tokenized_sequences, axis=0).astype(np.uint8)

    # shuffle sequences and generate tokenized sequences
    data["shuffled_sequence"] = data.apply(lambda x: "".join(np.random.permutation(list(x["sequence"]))), axis=1)

    shuffled_sequences = data["shuffled_sequence"].values
    shuffled_tokenized_sequences = []
    for seq in tqdm(shuffled_sequences, ncols=0):
        tokens = [dna_tokens[base] for base in seq]
        shuffled_tokenized_sequences.append(np.array(tokens, dtype=np.uint8))
    shuffled_tokenized_sequences = np.stack(shuffled_tokenized_sequences, axis=0).astype(np.uint8)

    data.to_csv(f'{FLAGS.diffusion_dir}/K562_hESCT0_HepG2_GM12878_12k_sequences_per_group_with_shuffled.txt', sep='\t', index=False)

    all_seqs_specific_to_cell = {}
    all_shuffled_seqs_specific_to_cell = {}
    for cell in ['K562', 'hESCT0', 'HepG2', 'GM12878']:
        print(cell)
        if cell == 'GM12878':
            cell_data = data[data["{}_ENCLB441ZZZ".format(cell)] == 1].reset_index(drop=True)
        elif cell == 'hESCT0':
            cell_data = data[data["{}_ENCLB449ZZZ".format(cell)] == 1].reset_index(drop=True)
        elif cell == 'K562':
            cell_data = data[data["{}_ENCLB843GMH".format(cell)] == 1].reset_index(drop=True)
        elif cell == 'HepG2':
            cell_data = data[data["{}_ENCLB029COU".format(cell)] == 1].reset_index(drop=True)
        
        seqs_specific_to_cell = cell_data["sequence"].values
        tokenized_seqs_specific_to_cell = []
        for seq in tqdm(seqs_specific_to_cell, ncols=0):
            tokens = [dna_tokens[base] for base in seq]
            tokenized_seqs_specific_to_cell.append(np.array(tokens, dtype=np.uint8))
        tokenized_seqs_specific_to_cell = np.stack(tokenized_seqs_specific_to_cell, axis=0).astype(np.uint8)
        all_seqs_specific_to_cell[cell] = tokenized_seqs_specific_to_cell

        shuffled_seqs_specific_to_cell = cell_data["shuffled_sequence"].values
        shuffled_tokenized_seqs_specific_to_cell = []
        for seq in tqdm(shuffled_seqs_specific_to_cell, ncols=0):
            tokens = [dna_tokens[base] for base in seq]
            shuffled_tokenized_seqs_specific_to_cell.append(np.array(tokens, dtype=np.uint8))
        shuffled_tokenized_seqs_specific_to_cell = np.stack(shuffled_tokenized_seqs_specific_to_cell, axis=0).astype(np.uint8)
        all_shuffled_seqs_specific_to_cell[cell] = shuffled_tokenized_seqs_specific_to_cell

    return tokenized_sequences, shuffled_tokenized_sequences, all_seqs_specific_to_cell, all_shuffled_seqs_specific_to_cell

def main(argv):
    all_cells = ['HepG2', 'GM12878', 'hESCT0', 'K562']
    all_data = {}
    for cell in all_cells:
        all_data[cell] = {
            "sequences": process_diffusion_data(cell)
        }
    
    training_sequences, shuffled_training_sequences, all_seqs_specific_to_cell, all_shuffled_seqs_specific_to_cell = process_diffusion_training_data()
    all_data["training_sequences"] = {
        "sequences": training_sequences
    }
    all_data["shuffled_training_sequences"] = {
        "sequences": shuffled_training_sequences
    }
    for cell in all_cells:
        all_data["{}_training_sequences".format(cell)] = {
            "sequences": all_seqs_specific_to_cell[cell]
        }
        all_data["{}_shuffled_training_sequences".format(cell)] = {
            "sequences": all_shuffled_seqs_specific_to_cell[cell]
        }

    mlxu.save_pickle(all_data, FLAGS.output_file)

if __name__ == '__main__':
    mlxu.run(main)
