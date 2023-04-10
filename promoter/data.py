import numpy as np
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

    def __iter__(self):
        sure_size = self.data['sure_sequences'].shape[0]
        mpra_size = self.data['mpra_sequences'].shape[0]
        while True:
            sure_indices = np.random.choice(sure_size, self.config.batch_size)
            mpra_indices = np.random.choice(mpra_size, self.config.batch_size)
            yield {
                'sure_sequences': self.data['sure_sequences'][sure_indices].astype(np.int32),
                'sure_k562_labels': self.data['sure_k562_labels'][sure_indices],
                'sure_hepg2_labels': self.data['sure_hepg2_labels'][sure_indices],
                'mpra_sequences': self.data['mpra_sequences'][mpra_indices].astype(np.int32),
                'mpra_output': self.data['mpra_output'][mpra_indices],
            }
