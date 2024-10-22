from functools import partial
import numpy as np
import mlxu
from tqdm import tqdm, trange
from pprint import pprint, pformat

import jax
import jax.numpy as jnp
from flax.jax_utils import replicate
import einops
import mlxu.jax_utils as jax_utils

from promoter_design.promoter_modelling.data import FinetuneDataset
from promoter_design.promoter_modelling.model import FinetuneNetwork
from promoter_design.COMs.seq_opt import SequenceOptimizer, ExpressionObjective


FLAGS, FLAGS_DEF = mlxu.define_flags_with_default(
    seed=42,
    load_checkpoint='',
    output_file='',
    expression_objective=ExpressionObjective.get_default_config(),
    sequence_optimizer=SequenceOptimizer.get_default_config(),
    finetune_network=FinetuneNetwork.get_default_config(),
    data=FinetuneDataset.get_default_config(),
)


def main(argv):
    assert FLAGS.output_file != ''
    jax_utils.set_random_seed(FLAGS.seed)
    jax_device_count = jax.device_count()

    FLAGS.data.sequential_sample = True  # Ensure we iterate over the whole dataset
    dataset = FinetuneDataset(FLAGS.data)

    model = FinetuneNetwork(FLAGS.finetune_network)
    if FLAGS.load_checkpoint != '':
        params = jax.device_put(mlxu.load_pickle(FLAGS.load_checkpoint))
    else:
        params = model.init(
            inputs=jnp.zeros((1, 1000, 4)),
            deterministic=False,
            rngs=jax_utils.next_rng(model.rng_keys()),
        )

    sequence_optimizer = SequenceOptimizer(FLAGS.sequence_optimizer)
    expression_objective = ExpressionObjective(FLAGS.expression_objective)

    @partial(jax.pmap, axis_name='dp')
    def optimization_step(params, rng, batch):
        rng_generator = jax_utils.JaxRNG(rng)
        starting_seq = jax.nn.one_hot(batch['sequences'], 5, dtype=jnp.float32)[:, :, :4]

        def objectve_funtion(seq, rng, params, target='all'):
            rng_generator = jax_utils.JaxRNG(rng)
            thp1_pred, jurkat_pred, k562_pred = model.apply(
                params,
                inputs=seq,
                deterministic=True,
                rngs=rng_generator(model.rng_keys()),
            )
            thp1_diff, jurkat_diff, k562_diff = expression_objective(
                thp1_pred, jurkat_pred, k562_pred
            )

            if target == 'thp1':
                return thp1_diff
            elif target == 'jurkat':
                return jurkat_diff
            elif target == 'k562':
                return k562_diff
            elif target == 'all':
                return dict(
                    thp1_pred=thp1_pred,
                    jurkat_pred=jurkat_pred,
                    k562_pred=k562_pred,
                    thp1_diff=thp1_diff,
                    jurkat_diff=jurkat_diff,
                    k562_diff=k562_diff,
                )
            else:
                raise ValueError(f'Unknown target {target}')

        def count_mutations(start, end):
            return jnp.sum(
                jnp.argmax(start, axis=-1) != jnp.argmax(end, axis=-1),
                axis=-1,
            ).astype(jnp.float32)

        starting_seq_metrics = objectve_funtion(
            starting_seq, rng_generator(),
            params=params, target='all'
        )
        ds_thp1_pred, ds_jurkat_pred, ds_k562_pred = (
            starting_seq_metrics['thp1_pred'],
            starting_seq_metrics['jurkat_pred'],
            starting_seq_metrics['k562_pred'],
        )
        ds_thp1_diff, ds_jurkat_diff, ds_k562_diff = (
            starting_seq_metrics['thp1_diff'],
            starting_seq_metrics['jurkat_diff'],
            starting_seq_metrics['k562_diff'],
        )

        thp1_optimized_seq = sequence_optimizer(
            objectve_funtion,
            starting_seq,
            rng_generator(),
            params=params,
            target='thp1'
        )
        thp1_n_mutations = count_mutations(starting_seq, thp1_optimized_seq)
        opt_thp1_metrics = objectve_funtion(
            thp1_optimized_seq, rng_generator(),
            params=params, target='all'
        )
        opt_thp1_pred, opt_thp1_diff = (
            opt_thp1_metrics['thp1_pred'],
            opt_thp1_metrics['thp1_diff'],
        )
        opt_thp1_metrics = {
            f'thp1_opt_seq_{key}': val for key, val in opt_thp1_metrics.items()
        }

        jurkat_optimized_seq = sequence_optimizer(
            objectve_funtion,
            starting_seq,
            rng_generator(),
            params=params,
            target='jurkat'
        )

        jurkat_n_mutations = count_mutations(starting_seq, jurkat_optimized_seq)
        opt_jurkat_metrics = objectve_funtion(
            jurkat_optimized_seq, rng_generator(),
            params=params, target='all'
        )
        opt_jurkat_pred, opt_jurkat_diff = (
            opt_jurkat_metrics['jurkat_pred'],
            opt_jurkat_metrics['jurkat_diff'],
        )
        opt_jurkat_metrics = {
            f'jurkat_opt_seq_{key}': val for key, val in opt_jurkat_metrics.items()
        }

        k562_optimized_seq = sequence_optimizer(
            objectve_funtion,
            starting_seq,
            rng_generator(),
            params=params,
            target='k562'
        )

        k562_n_mutations = count_mutations(starting_seq, k562_optimized_seq)
        opt_k562_metrics = objectve_funtion(
            k562_optimized_seq, rng_generator(),
            params=params, target='all'
        )
        opt_k562_pred, opt_k562_diff = (
            opt_k562_metrics['k562_pred'],
            opt_k562_metrics['k562_diff'],
        )
        opt_k562_metrics = {
            f'k562_opt_seq_{key}': val for key, val in opt_k562_metrics.items()
        }

        thp1_gap = opt_thp1_diff - ds_thp1_diff
        jurkat_gap = opt_jurkat_diff - ds_jurkat_diff
        k562_gap = opt_k562_diff - ds_k562_diff

        results = dict(
            original_seq=batch['sequences'],
            thp1_output=batch['thp1_output'],
            jurkat_output=batch['jurkat_output'],
            k562_output=batch['k562_output'],

            ds_thp1_pred=ds_thp1_pred,
            ds_jurkat_pred=ds_jurkat_pred,
            ds_k562_pred=ds_k562_pred,
            ds_thp1_diff=ds_thp1_diff,
            ds_jurkat_diff=ds_jurkat_diff,
            ds_k562_diff=ds_k562_diff,

            thp1_optimized_seq=jnp.argmax(thp1_optimized_seq, axis=-1),
            jurkat_optimized_seq=jnp.argmax(jurkat_optimized_seq, axis=-1),
            k562_optimized_seq=jnp.argmax(k562_optimized_seq, axis=-1),

            opt_thp1_pred=opt_thp1_pred,
            opt_jurkat_pred=opt_jurkat_pred,
            opt_k562_pred=opt_k562_pred,
            opt_thp1_diff=opt_thp1_diff,
            opt_jurkat_diff=opt_jurkat_diff,
            opt_k562_diff=opt_k562_diff,
            thp1_gap=thp1_gap,
            jurkat_gap=jurkat_gap,
            k562_gap=k562_gap,
            thp1_n_mutations=thp1_n_mutations,
            jurkat_n_mutations=jurkat_n_mutations,
            k562_n_mutations=k562_n_mutations,

            **opt_thp1_metrics,
            **opt_jurkat_metrics,
            **opt_k562_metrics,
        )
        return rng_generator(), results

    data_iterator = dataset.batch_iterator(pmap_axis_dim=jax_device_count)
    steps = len(dataset) // FLAGS.data.batch_size

    rng = jax.device_put_sharded(
        list(jax_utils.next_rng(jax.device_count())),
        jax.devices(),
    )
    params = replicate(params)

    results = []

    for _, batch in zip(trange(steps, ncols=0), data_iterator):
        rng, r = optimization_step(params, rng, batch)
        results.append({
            key: einops.rearrange(val, 'd b ... -> (d b) ...')
            for key, val in jax.device_get(r).items()
        })

    results = {
        key: np.concatenate([r[key] for r in results], axis=0)
        for key in results[0].keys()
    }

    mlxu.save_pickle(results, FLAGS.output_file)


if __name__ == '__main__':
    mlxu.run(main)