import numpy as np
import mlxu

import jax
import jax.numpy as jnp
import flax
from flax import linen as nn
import einops
import mlxu.jax_utils as jax_utils

# Jax implementation of the cosine similarity function
def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    x1_norm = x1 / (jnp.linalg.norm(x1, axis=dim, keepdims=True) + eps)
    x2_norm = x2 / (jnp.linalg.norm(x2, axis=dim, keepdims=True) + eps)
    return jnp.sum(x1_norm * x2_norm, axis=dim)

class DEN_loss(nn.Module):
    diversity_loss_epsilon: float = 0.3
    diversity_loss_sigma: float = 1
    diversity_loss_intermediate_repr_epsilon: float = 0.3
    entropy_loss_m_bits: float = 1.8
    all_motifs: np.ndarray = None
    all_motif_coefs: np.ndarray = None
    use_intermediate_repr: bool = True
    eps: float = 1e-8

    @nn.compact
    def __call__(self, fitness_loss_coef, diversity_loss_coef, entropy_loss_coef, motif_loss_coef, \
                 seq1_pwm, seq2_pwm, \
                 seq1_samples, seq2_samples, \
                 predicted_fitness_seq1_pwm, predicted_fitness_seq2_pwm, \
                 predicted_fitness_seq1_samples, predicted_fitness_seq2_samples, \
                 intermediate_repr_seq1_samples=None, intermediate_repr_seq2_samples=None):
        # seq1_pwm and seq2_pwm are the two generated sequences' PWMs - shape (batch_size, seq_length, alphabet_size)
        # seq1_samples and seq2_samples are the two generated sequences - shape (batch_size, num_samples, seq_length, alphabet_size)
        # predicted_fitness_seq1_pwm and predicted_fitness_seq2_pwm are the predicted fitnesses of the two generated sequences' PWMs - shape (batch_size)
        # predicted_fitness_seq1_samples and predicted_fitness_seq2_samples are the predicted fitnesses of the two generated sequences - shape (batch_size, num_samples)
        # intermediate_repr_seq1_samples and intermediate_repr_seq2_samples are the intermediate representations of the two generated sequences - shape (batch_size, num_samples, intermediate_repr_size)

        assert self.use_intermediate_repr == (intermediate_repr_seq1_samples is not None and intermediate_repr_seq2_samples is not None)

        # fitness loss, computed as the negative of the average predicted fitnesses of the two generated sequences in both PWM and samples modes
        fitness_loss = jnp.mean(predicted_fitness_seq1_pwm) + jnp.mean(predicted_fitness_seq2_pwm) + \
                          jnp.mean(predicted_fitness_seq1_samples) + jnp.mean(predicted_fitness_seq2_samples)
        fitness_loss = -fitness_loss

        std = jnp.exp(fitness_loss_coef)**(1/2)
        coeff = 1 / (2*(std**2))
        fitness_loss = coeff * fitness_loss + jnp.log(std)

        # diversity-based loss function that computes sequence-level cosine similarity between the two generated sequences
        # optionally, an additional loss term can be added to penalize the similarity between the intermediate representations
        diversity_loss = nn.relu(-self.diversity_loss_epsilon + jnp.mean(cosine_similarity(seq1_samples, seq2_samples, dim=2), dim=1))
        diversity_loss = jax.lax.fori_loop(0, self.diversity_loss_sigma, \
                                           lambda x, y: jnp.max(nn.relu(-self.diversity_loss_epsilon + jnp.mean(cosine_similarity(seq1_samples[:, x:], seq2_samples[:, :-x], dim=2), dim=1)), y), \
                                           diversity_loss)
        diversity_loss = jax.lax.fori_loop(0, self.diversity_loss_sigma, \
                                           lambda x, y: jnp.max(nn.relu(-self.diversity_loss_epsilon + jnp.mean(cosine_similarity(seq1_samples[:, :-x], seq2_samples[:, x:], dim=2), dim=1)), y), \
                                           diversity_loss)
        diversity_loss = jnp.mean(diversity_loss)
        
        intermediate_repr_loss = jax.lax.switch(self.use_intermediate_repr, \
                                                jnp.mean(nn.relu(-self.diversity_loss_intermediate_repr_epsilon + cosine_similarity(intermediate_repr_seq1_samples, intermediate_repr_seq2_samples, dim=1))), \
                                                0)
        
        total_diversity_loss = diversity_loss + intermediate_repr_loss

        std = jnp.exp(diversity_loss_coef)**(1/2)
        coeff = 1 / (2*(std**2))
        total_diversity_loss = coeff * total_diversity_loss + jnp.log(std)

        # entropy-based loss function that computes the entropy of the generated sequence and enforces a minimum average conservation
        entropy_loss = nn.relu(self.entropy_loss_m_bits - jnp.mean(jnp.log2(seq1_pwm.shape[2]) - \
                                                                   jnp.sum(-seq1_pwm * jnp.log2(seq1_pwm + self.eps), dim=2), dim=1))
        entropy_loss = jnp.mean(entropy_loss)

        std = jnp.exp(entropy_loss_coef)**(1/2)
        coeff = 1 / (2*(std**2))
        entropy_loss = coeff * entropy_loss + jnp.log(std)

        # total loss
        total_loss = fitness_loss + total_diversity_loss + entropy_loss
        
        return total_loss, fitness_loss, total_diversity_loss, entropy_loss, diversity_loss, intermediate_repr_loss