import re
import numpy as np
import einops
import mlxu

import jax
import jax.numpy as jnp
import mlxu.jax_utils as jax_utils



def average_metrics(metrics):
    averaged = {}
    for key in metrics[0].keys():
        averaged[key] = np.mean([m[key] for m in metrics])
    return averaged


def get_weight_decay_mask(exclusions):
    """ Return a weight decay mask function that computes the pytree masks
        according to the given exclusion rules.
    """
    def decay(name, _):
        for rule in exclusions:
            if re.search(rule, name) is not None:
                return False
        return True

    def weight_decay_mask(params):
        return jax_utils.named_tree_map(decay, params, sep='/')

    return weight_decay_mask


def global_norm(tree):
    """ Return the global L2 norm of a pytree. """
    squared = jax.tree_util.tree_map(lambda x: jnp.sum(jnp.square(x)), tree)
    flattened, _ = jax.flatten_util.ravel_pytree(squared)
    return jnp.sqrt(jnp.sum(flattened))
