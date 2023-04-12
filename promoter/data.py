import numpy as np
import pandas as pd
import einops
import mlxu


class PretrainDataset(object):

    @staticmethod
    def get_default_config(updates=None):
        config = mlxu.config_dict()
        config.path = ''
        config.split = 'train'
        config.batch_size = 32

        if updates is not None:
            config.update(mlxu.config_dict(updates).copy_and_resolve_references())

        return config

    def __init__(self, config):
        self.config = self.get_default_config(config)
        assert self.config.path != ''
        self.data = mlxu.load_pickle(self.config.path)[self.config.split]

    def batch_iterator(self, pmap_axis_dim=None):
        sure_size = self.data['sure_sequences'].shape[0]
        mpra_size = self.data['mpra_sequences'].shape[0]
        while True:
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

        if updates is not None:
            config.update(mlxu.config_dict(updates).copy_and_resolve_references())

        return config

    def __init__(self, config):
        self.config = self.get_default_config(config)
        assert self.config.path != ''
        self.data = mlxu.load_pickle(self.config.path)[self.config.split]


    def batch_iterator(self, pmap_axis_dim=None):
        size = self.data['sequences'].shape[0]
        while True:
            indices = np.random.choice(size, self.config.batch_size)
            batch = {
                'sequences': self.data['sequences'][indices].astype(np.int32),
                'thp1_output': self.data['thp1_output'][indices].astype(np.int32),
                'jurkat_output': self.data['jurkat_output'][indices].astype(np.int32),
                'k562_output': self.data['k562_output'][indices].astype(np.int32),
            }
            if pmap_axis_dim is not None:
                batch = reshape_batch_for_pmap(batch, pmap_axis_dim)
            yield batch


def reshape_batch_for_pmap(batch, pmap_axis_dim):
    return {
        key: einops.rearrange(value, '(p b) ... -> p b ...', p=pmap_axis_dim)
        for key, value in batch.items()
    }
