import tqdm
import numpy as np
import pandas as pd
import einops
import mlxu


DNA_TOKENS = {
    'A': 0,
    'C': 1,
    'G': 2,
    'T': 3,
    'N': 4,
}


class PretrainDataset(object):

    @staticmethod
    def get_default_config(updates=None):
        config = mlxu.config_dict()
        config.path = ''
        config.split = 'train'
        config.batch_size = 32
        config.sequential_sample = False
        return mlxu.update_config_dict(config, updates)

    def __init__(self, config):
        self.config = self.get_default_config(config)
        assert self.config.path != ''
        self.data = mlxu.load_pickle(self.config.path)[self.config.split]

    def batch_iterator(self, pmap_axis_dim=None):
        sure_size = self.data['sure_sequences'].shape[0]
        mpra_size = self.data['mpra_sequences'].shape[0]
        max_size = max(sure_size, mpra_size)
        index = 0

        while True:
            if self.config.sequential_sample:
                sure_indices = np.arange(index, index + self.config.batch_size) % sure_size
                mpra_indices = np.arange(index, index + self.config.batch_size) % mpra_size
                index = (index + self.config.batch_size) % max_size
            else:
                sure_indices = np.random.choice(sure_size, self.config.batch_size)
                mpra_indices = np.random.choice(mpra_size, self.config.batch_size)
            batch = {
                'sure_sequences': self.data['sure_sequences'][sure_indices].astype(np.int32),
                'sure_k562_labels': self.data['sure_k562_labels'][sure_indices].astype(np.int32),
                'sure_hepg2_labels': self.data['sure_hepg2_labels'][sure_indices].astype(np.int32),
                'mpra_sequences': self.data['mpra_sequences'][mpra_indices].astype(np.int32),
                'mpra_output': self.data['mpra_output'][mpra_indices].astype(np.float32),
            }
            if pmap_axis_dim is not None:
                batch = reshape_batch_for_pmap(batch, pmap_axis_dim)
            yield batch


class FinetuneDataset(object):

    @staticmethod
    def get_default_config(updates=None):
        config = mlxu.config_dict()
        config.path = ''
        config.split = 'train'
        config.batch_size = 32
        config.sequential_sample = False
        return mlxu.update_config_dict(config, updates)

    def __init__(self, config):
        self.config = self.get_default_config(config)
        assert self.config.path != ''
        all_data = mlxu.load_pickle(self.config.path)
        if self.config.split == 'all':
            self.data = {
                'sequences': np.concatenate([
                    all_data['train']['sequences'],
                    all_data['val']['sequences'],
                    all_data['test']['sequences'],
                ], axis=0),
                'thp1_output': np.concatenate([
                    all_data['train']['thp1_output'],
                    all_data['val']['thp1_output'],
                    all_data['test']['thp1_output'],
                ], axis=0),
                'jurkat_output': np.concatenate([
                    all_data['train']['jurkat_output'],
                    all_data['val']['jurkat_output'],
                    all_data['test']['jurkat_output'],
                ], axis=0),
                'k562_output': np.concatenate([
                    all_data['train']['k562_output'],
                    all_data['val']['k562_output'],
                    all_data['test']['k562_output'],
                ], axis=0),
            }
        else:
            self.data = all_data[self.config.split]

    def __len__(self):
        return self.data['sequences'].shape[0]

    def batch_iterator(self, pmap_axis_dim=None):
        size = self.data['sequences'].shape[0]
        index = 0
        while True:
            if self.config.sequential_sample:
                indices = np.arange(index, index + self.config.batch_size) % size
                index = (index + self.config.batch_size) % size
            else:
                indices = np.random.choice(size, self.config.batch_size)
            batch = {
                'sequences': self.data['sequences'][indices].astype(np.int32),
                'thp1_output': self.data['thp1_output'][indices].astype(np.float32),
                'jurkat_output': self.data['jurkat_output'][indices].astype(np.float32),
                'k562_output': self.data['k562_output'][indices].astype(np.float32),
            }
            if pmap_axis_dim is not None:
                batch = reshape_batch_for_pmap(batch, pmap_axis_dim)
            yield batch


def reshape_batch_for_pmap(batch, pmap_axis_dim):
    return {
        key: einops.rearrange(value, '(p b) ... -> p b ...', p=pmap_axis_dim)
        for key, value in batch.items()
    }


def tokenize_sequences(sequences, progress=False):
    output = []
    if progress:
        sequences = tqdm.tqdm(sequences, ncols=0)
    for seq in sequences:
        output.append(np.array([DNA_TOKENS[x] for x in seq], dtype=np.int32))

    return output


def decode_sequences(sequences, progress=False):
    output = []
    codes = np.array(['A', 'C', 'G', 'T', 'N'])
    if progress:
        sequences = tqdm.tqdm(sequences, ncols=0)
    for seq in sequences:
        output.append(''.join(codes[seq]))
    return output