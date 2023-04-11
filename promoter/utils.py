import numpy as np
import einops
import mlxu

import jax
import jax.numpy as jnp


def average_metrics(metrics):
    averaged = {}
    for key in metrics[0].keys():
        averaged[key] = np.mean([m[key] for m in metrics])
    return averaged
