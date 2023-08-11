import numpy as np
import pdb
import mlxu

import jax
import jax.numpy as jnp
import flax
from flax import linen as nn
import einops
import mlxu.jax_utils as jax_utils

# Jax implementation of the cosine similarity function
def cosine_similarity(x1, x2, axis=1, eps=1e-8):
    x1_norm = x1 / (jnp.linalg.norm(x1, axis=axis, keepdims=True) + eps)
    x2_norm = x2 / (jnp.linalg.norm(x2, axis=axis, keepdims=True) + eps)
    return jnp.sum(x1_norm * x2_norm, axis=axis)

def fitness_loss_fn(expression_vals, diff_exp_cell_ind, other_cell_inds, max_clip=5.0, min_clip=-1.0, target_cell_weight=1.0):
    expression_vals = jnp.clip(expression_vals, min_clip, max_clip)
    fitness = target_cell_weight * expression_vals[:, :, diff_exp_cell_ind]
    for ind in other_cell_inds:
        fitness -= (0.5 * expression_vals[:, :, ind])
    return -fitness

class DEN_loss(nn.Module):
    config_updates: ... = None

    @staticmethod
    @nn.nowrap
    def get_default_config(updates=None):
        config = mlxu.config_dict()
        config.diversity_loss_epsilon = 0.3
        # config.diversity_loss_sigma = 1
        config.diversity_loss_intermediate_repr_epsilon = 0.3
        config.entropy_loss_m_bits = 1.8
        config.all_motifs = None
        config.all_motif_coefs = None
        config.use_intermediate_repr = True
        config.eps = 1e-8

        config.fitness_loss_coef = 1.0
        config.diversity_loss_coef = 1.0
        config.entropy_loss_coef = 1.0
        config.base_entropy_loss_coef = 1.0
        config.motif_loss_coef = 1.0
        config.motif_loss_intermediate_repr_coef = 1.0
        
        if updates is not None:
            config.update(mlxu.config_dict(updates).copy_and_resolve_references())
        assert config.diff_exp_cell_ind in [0, 1, 2], 'diff_exp_cell_ind must be 0, 1, or 2 (corresponding to THP1, Jurkat or K562 respectively)'
        return config
    
    def setup(self):
        self.config = self.get_default_config(self.config_updates)
        self.diversity_loss_epsilon = self.config.diversity_loss_epsilon
        # self.diversity_loss_sigma = self.config.diversity_loss_sigma
        self.diversity_loss_intermediate_repr_epsilon = self.config.diversity_loss_intermediate_repr_epsilon
        self.entropy_loss_m_bits = self.config.entropy_loss_m_bits
        self.all_motifs = self.config.all_motifs
        self.all_motif_coefs = self.config.all_motif_coefs
        self.use_intermediate_repr = self.config.use_intermediate_repr
        self.eps = self.config.eps
        self.diff_exp_cell_ind = self.config.diff_exp_cell_ind
        
        self.fitness_loss_coef = self.config.fitness_loss_coef
        self.diversity_loss_coef = self.config.diversity_loss_coef
        self.entropy_loss_coef = self.config.entropy_loss_coef
        self.base_entropy_loss_coef = self.config.base_entropy_loss_coef
        self.motif_loss_coef = self.config.motif_loss_coef
        self.motif_loss_intermediate_repr_coef = self.config.motif_loss_intermediate_repr_coef

#         self.fitness_loss_coef = self.param('fitness_loss_coef', lambda key, shape: jnp.zeros(shape), (1, ))
#         self.diversity_loss_coef = self.param('diversity_loss_coef', lambda key, shape: jnp.zeros(shape), (1, ))
#         self.entropy_loss_coef = self.param('entropy_loss_coef', lambda key, shape: jnp.zeros(shape), (1, ))
#         self.motif_loss_coef = self.param('motif_loss_coef', lambda key, shape: jnp.zeros(shape), (1, ))
#         self.motif_loss_intermediate_repr_coef = self.param('motif_loss_intermediate_repr_coef', lambda key, shape: jnp.zeros(shape), (1, ))

    def __call__(self, \
                 seq1_pwm, seq2_pwm, \
                 seq1_samples, seq2_samples, \
                 predicted_expression_seq1_pwm, predicted_expression_seq2_pwm, \
                 predicted_expression_seq1_samples, predicted_expression_seq2_samples, \
                 intermediate_repr_seq1_samples=None, intermediate_repr_seq2_samples=None):
        # seq1_pwm and seq2_pwm are the two generated sequences' PWMs - shape (batch_size, seq_length, alphabet_size)
        # seq1_samples and seq2_samples are the two generated sequences - shape (batch_size, num_samples, seq_length, alphabet_size)
        # predicted_fitness_seq1_pwm and predicted_fitness_seq2_pwm are the predicted fitnesses of the two generated sequences' PWMs - shape (batch_size)
        # predicted_fitness_seq1_samples and predicted_fitness_seq2_samples are the predicted fitnesses of the two generated sequences - shape (batch_size, num_samples)
        # intermediate_repr_seq1_samples and intermediate_repr_seq2_samples are the intermediate representations of the two generated sequences - shape (batch_size, num_samples, intermediate_repr_size)

        assert self.use_intermediate_repr == (intermediate_repr_seq1_samples is not None and intermediate_repr_seq2_samples is not None)

        # compute fitness
        other_cell_inds = [i for i in range(3) if i != self.diff_exp_cell_ind]

        # fitness loss, computed as the average predicted fitness loss of the two generated sequences in samples mode
        if self.diff_exp_cell_ind == 0: # THP1
            fitness_loss = jnp.mean(jnp.mean(fitness_loss_fn(predicted_expression_seq1_samples, self.diff_exp_cell_ind, other_cell_inds, target_cell_weight=1.5), axis=1))
            fitness_loss += jnp.mean(jnp.mean(fitness_loss_fn(predicted_expression_seq2_samples, self.diff_exp_cell_ind, other_cell_inds, target_cell_weight=1.5), axis=1))
        else:
            fitness_loss = jnp.mean(jnp.mean(fitness_loss_fn(predicted_expression_seq1_samples, self.diff_exp_cell_ind, other_cell_inds, target_cell_weight=1.0), axis=1))
            fitness_loss += jnp.mean(jnp.mean(fitness_loss_fn(predicted_expression_seq2_samples, self.diff_exp_cell_ind, other_cell_inds, target_cell_weight=1.0), axis=1))
            
        fitness_loss = fitness_loss * self.fitness_loss_coef

#         std = jnp.exp(self.fitness_loss_coef)**(1/2)
#         coeff = 1 / (2*(std**2))
#         fitness_loss = coeff * fitness_loss + jnp.log(std)
#         fitness_loss = fitness_loss[0]

        # diversity-based loss function that computes sequence-level cosine similarity between the two generated sequences
        # optionally, an additional loss term can be added to penalize the similarity between the intermediate representations
        diversity_loss = nn.relu(-self.diversity_loss_epsilon + jnp.mean(cosine_similarity(seq1_samples, seq2_samples, axis=3), axis=2))
        diversity_loss = jnp.maximum(nn.relu(-self.diversity_loss_epsilon + jnp.mean(cosine_similarity(seq1_samples[:, :, 1:], seq2_samples[:, :, :-1], axis=3), axis=2)), diversity_loss)
        diversity_loss = jnp.maximum(nn.relu(-self.diversity_loss_epsilon + jnp.mean(cosine_similarity(seq1_samples[:, :, :-1], seq2_samples[:, :, 1:], axis=3), axis=2)), diversity_loss)
        diversity_loss = jnp.maximum(nn.relu(-self.diversity_loss_epsilon + jnp.mean(cosine_similarity(seq1_samples[:, :, 2:], seq2_samples[:, :, :-2], axis=3), axis=2)), diversity_loss)
        diversity_loss = jnp.maximum(nn.relu(-self.diversity_loss_epsilon + jnp.mean(cosine_similarity(seq1_samples[:, :, :-2], seq2_samples[:, :, 2:], axis=3), axis=2)), diversity_loss)
        diversity_loss = jnp.maximum(nn.relu(-self.diversity_loss_epsilon + jnp.mean(cosine_similarity(seq1_samples[:, :, 3:], seq2_samples[:, :, :-3], axis=3), axis=2)), diversity_loss)
        diversity_loss = jnp.maximum(nn.relu(-self.diversity_loss_epsilon + jnp.mean(cosine_similarity(seq1_samples[:, :, :-3], seq2_samples[:, :, 3:], axis=3), axis=2)), diversity_loss)
        diversity_loss = jnp.maximum(nn.relu(-self.diversity_loss_epsilon + jnp.mean(cosine_similarity(seq1_samples[:, :, 4:], seq2_samples[:, :, :-4], axis=3), axis=2)), diversity_loss)
        diversity_loss = jnp.maximum(nn.relu(-self.diversity_loss_epsilon + jnp.mean(cosine_similarity(seq1_samples[:, :, :-4], seq2_samples[:, :, 4:], axis=3), axis=2)), diversity_loss)
        diversity_loss = jnp.maximum(nn.relu(-self.diversity_loss_epsilon + jnp.mean(cosine_similarity(seq1_samples[:, :, 5:], seq2_samples[:, :, :-5], axis=3), axis=2)), diversity_loss)
        diversity_loss = jnp.maximum(nn.relu(-self.diversity_loss_epsilon + jnp.mean(cosine_similarity(seq1_samples[:, :, :-5], seq2_samples[:, :, 5:], axis=3), axis=2)), diversity_loss)
        diversity_loss = jnp.maximum(nn.relu(-self.diversity_loss_epsilon + jnp.mean(cosine_similarity(seq1_samples[:, :, 6:], seq2_samples[:, :, :-6], axis=3), axis=2)), diversity_loss)
        diversity_loss = jnp.maximum(nn.relu(-self.diversity_loss_epsilon + jnp.mean(cosine_similarity(seq1_samples[:, :, :-6], seq2_samples[:, :, 6:], axis=3), axis=2)), diversity_loss)
        diversity_loss = jnp.maximum(nn.relu(-self.diversity_loss_epsilon + jnp.mean(cosine_similarity(seq1_samples[:, :, 7:], seq2_samples[:, :, :-7], axis=3), axis=2)), diversity_loss)
        diversity_loss = jnp.maximum(nn.relu(-self.diversity_loss_epsilon + jnp.mean(cosine_similarity(seq1_samples[:, :, :-7], seq2_samples[:, :, 7:], axis=3), axis=2)), diversity_loss)
        diversity_loss = jnp.maximum(nn.relu(-self.diversity_loss_epsilon + jnp.mean(cosine_similarity(seq1_samples[:, :, 8:], seq2_samples[:, :, :-8], axis=3), axis=2)), diversity_loss)
        diversity_loss = jnp.maximum(nn.relu(-self.diversity_loss_epsilon + jnp.mean(cosine_similarity(seq1_samples[:, :, :-8], seq2_samples[:, :, 8:], axis=3), axis=2)), diversity_loss)
        diversity_loss = jnp.maximum(nn.relu(-self.diversity_loss_epsilon + jnp.mean(cosine_similarity(seq1_samples[:, :, 9:], seq2_samples[:, :, :-9], axis=3), axis=2)), diversity_loss)
        diversity_loss = jnp.maximum(nn.relu(-self.diversity_loss_epsilon + jnp.mean(cosine_similarity(seq1_samples[:, :, :-9], seq2_samples[:, :, 9:], axis=3), axis=2)), diversity_loss)
        diversity_loss = jnp.maximum(nn.relu(-self.diversity_loss_epsilon + jnp.mean(cosine_similarity(seq1_samples[:, :, 10:], seq2_samples[:, :, :-10], axis=3), axis=2)), diversity_loss)
        diversity_loss = jnp.maximum(nn.relu(-self.diversity_loss_epsilon + jnp.mean(cosine_similarity(seq1_samples[:, :, :-10], seq2_samples[:, :, 10:], axis=3), axis=2)), diversity_loss)
        
        
        # diversity_loss = jax.lax.fori_loop(0, self.diversity_loss_sigma, \
        #                                    lambda x, y: jnp.max(nn.relu(-self.diversity_loss_epsilon + jnp.mean(cosine_similarity(seq1_samples[:, x:], seq2_samples[:, :-x], axis=2), axis=1)), y), \
        #                                    diversity_loss)
        # diversity_loss = jax.lax.fori_loop(0, self.diversity_loss_sigma, \
        #                                    lambda x, y: jnp.max(nn.relu(-self.diversity_loss_epsilon + jnp.mean(cosine_similarity(seq1_samples[:, :-x], seq2_samples[:, x:], axis=2), axis=1)), y), \
        #                                    diversity_loss)
        diversity_loss = jnp.mean(jnp.mean(diversity_loss, axis=1))

        if self.use_intermediate_repr:
            intermediate_repr_loss = jnp.mean(jnp.mean(nn.relu(-self.diversity_loss_intermediate_repr_epsilon + cosine_similarity(intermediate_repr_seq1_samples, intermediate_repr_seq2_samples, axis=2)), axis=1))
        else:
            intermediate_repr_loss = jnp.zeros(1)
        
        total_diversity_loss = diversity_loss + intermediate_repr_loss
        total_diversity_loss = total_diversity_loss * self.diversity_loss_coef

#         std = jnp.exp(self.diversity_loss_coef)**(1/2)
#         coeff = 1 / (2*(std**2))
#         total_diversity_loss = coeff * total_diversity_loss + jnp.log(std)
#         total_diversity_loss = total_diversity_loss[0]

        # entropy-based loss function that computes the entropy of the generated sequence and enforces a minimum average conservation
        entropy_loss = nn.relu(self.entropy_loss_m_bits - jnp.mean(jnp.log2(seq1_pwm.shape[2]) - \
                                                                   jnp.sum(-seq1_pwm * jnp.log2(seq1_pwm + self.eps), axis=2), axis=1))
        entropy_loss = jnp.mean(entropy_loss) * self.entropy_loss_coef
        
        # entropy-based loss function that ensures that all bases are used uniformly
        mean_base_usage = jnp.mean(seq1_pwm, axis=1)
        base_entropy_loss = - jnp.sum(-mean_base_usage * jnp.log2(mean_base_usage), axis=1)
        base_entropy_loss = jnp.mean(base_entropy_loss) * self.base_entropy_loss_coef

#         std = jnp.exp(self.entropy_loss_coef)**(1/2)
#         coeff = 1 / (2*(std**2))
#         entropy_loss = coeff * entropy_loss + jnp.log(std)
#         entropy_loss = entropy_loss[0]

        # total loss
        total_loss = fitness_loss + total_diversity_loss + entropy_loss + base_entropy_loss
        
        return total_loss, fitness_loss, total_diversity_loss, entropy_loss, base_entropy_loss, diversity_loss, intermediate_repr_loss