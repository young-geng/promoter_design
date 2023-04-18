from functools import partial
import numpy as np
import scipy.stats as stats
from sklearn.metrics import r2_score
import mlxu
from tqdm import tqdm, trange
from pprint import pprint, pformat
import os
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import flax
from flax.jax_utils import replicate, unreplicate
from flax.training.train_state import TrainState
import optax
import einops
import mlxu.jax_utils as jax_utils
from ml_collections import ConfigDict

import pdb

from data import FinetuneDataset
from model import FinetuneNetwork
from DEN_model import DEN
from utils import average_metrics, global_norm, get_weight_decay_mask, get_generic_mask


FLAGS, FLAGS_DEF = mlxu.define_flags_with_default(
    seed=42,
    total_steps=3000,
    log_freq=20,
    eval_freq=125,
    save_model=False,
    remat=True,
    accumulate_gradient_steps=1,
    lr=0.001,
    lr_warmup_steps=100,
    weight_decay=1e-4,
    clip_gradient=10.0,
    batch_size=8,
    pretrained_predictor_path="./saved_models/finetune_vanilla/best_params.pkl",
    generator_config_updates=ConfigDict({}),
    predictor_config_updates=ConfigDict({"return_intermediate": True}),
    loss_config_updates=ConfigDict({"diff_exp_cell_ind": 1}),
    oracle_train_data=FinetuneDataset.get_default_config({"split": "train", "path": "./data/finetune_data.pkl", "batch_size": 96}),
    oracle_val_data=FinetuneDataset.get_default_config({"split": "val", "path": "./data/finetune_data.pkl", "sequential_sample": True, "batch_size": 96}),
    oracle_test_data=FinetuneDataset.get_default_config({"split": "test", "path": "./data/finetune_data.pkl", "sequential_sample": True, "batch_size": 96}),
    logger=mlxu.WandBLogger.get_default_config({"output_dir": "./saved_models", "project": "promoter_design_jax", "wandb_dir": "./wandb", "online": True, \
                                                "experiment_id": "DENs_trial_Jurkat_1"}),
)

def reshape_batch_for_pmap(batch, pmap_axis_dim):
    return einops.rearrange(batch, '(p b) ... -> p b ...', p=pmap_axis_dim)

# batch iterator that generates random data
def batch_iterator(batch_size, latent_size, pmap_axis_dim=None):
    while True:
        batch1 = np.random.uniform(-1, 1, (batch_size, latent_size))
        batch2 = np.random.uniform(-1, 1, (batch_size, latent_size))
        if pmap_axis_dim is not None:
            batch1 = reshape_batch_for_pmap(batch1, pmap_axis_dim)
            batch2 = reshape_batch_for_pmap(batch2, pmap_axis_dim)
        yield batch1, batch2

# holds a static set of random data for evaluation
class DENValidationDataset:
    def __init__(self, batch_size, latent_size, num_samples=1024):
        self.batch_size = batch_size
        self.latent_size = latent_size
        self.num_samples = num_samples
        
        np.random.seed(97)
        self.random_data = np.random.uniform(-1, 1, (num_samples, latent_size))

    def __len__(self):
        return self.num_samples

    def batch_iterator(self, pmap_axis_dim=None):
        for i in range(0, self.num_samples, self.batch_size):
            batch = self.random_data[i:min(i+self.batch_size, self.num_samples)]
            if pmap_axis_dim is not None:
                batch = reshape_batch_for_pmap(batch, pmap_axis_dim)
            yield batch
        yield None

def main(argv):
    jax_utils.set_random_seed(FLAGS.seed)
    jax_device_count = jax.device_count()

    # create DEN
    model = DEN(FLAGS.generator_config_updates, \
                FLAGS.predictor_config_updates, \
                FLAGS.loss_config_updates)
    
    # create pure predictor
    predictor = FinetuneNetwork(FLAGS.predictor_config_updates)

    # init models
    params = model.init(
        inputs1=jnp.zeros((5, 100)),
        inputs2=jnp.zeros((5, 100)),
        deterministic=False,
        rngs=jax_utils.next_rng(model.rng_keys()),
    )

    predictor_params = predictor.init(
        inputs=jnp.zeros((1, 1000, 4)),
        deterministic=False,
        rngs=jax_utils.next_rng(predictor.rng_keys()),
    )

    # load pretrained predictor
    # in DEN
    params = flax.core.unfreeze(params)
    params['params']['predictor'] = jax.device_put(
        mlxu.load_pickle(FLAGS.pretrained_predictor_path)['params']
    )
    params = flax.core.freeze(params)
    # in predictor
    predictor_params = flax.core.unfreeze(predictor_params)
    predictor_params['params'] = jax.device_put(
        mlxu.load_pickle(FLAGS.pretrained_predictor_path)['params']
    )
    predictor_params = flax.core.freeze(predictor_params)

    # create optimizer
    learning_rate_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=FLAGS.lr,
        warmup_steps=FLAGS.lr_warmup_steps,
        decay_steps=FLAGS.total_steps,
        end_value=0.0,
    )

    # mask to only update the generator during training
    optimizer_mask = get_generic_mask(["predictor"], true_set="no_grad", false_set="adamw")(params)

    optimizer = optax.multi_transform(
        {"adamw": optax.chain(
                                optax.clip_by_global_norm(FLAGS.clip_gradient),
                                optax.adamw(
                                                learning_rate=learning_rate_schedule,
                                                weight_decay=FLAGS.weight_decay,
                                                mask=get_weight_decay_mask(['bias', 'loss']),
                                            )
                            ),
        "no_grad": optax.set_to_zero()},
        optimizer_mask
    )
    
    if FLAGS.accumulate_gradient_steps > 1:
        optimizer = optax.MultiSteps(optimizer, FLAGS.accumulate_gradient_steps)

    # create train state
    # for DEN
    train_state = TrainState.create(
        params=params,
        tx=optimizer,
        apply_fn=None
    )
    # for predictor
    predictor_train_state = TrainState.create(
        params=predictor_params,
        tx=optax.set_to_zero(),
        apply_fn=None
    )
    
    # create RNGs
    rng = jax.device_put_sharded(
        list(jax_utils.next_rng(jax.device_count())),
        jax.devices(),
    )

    # replicate train state across devices
    train_state = replicate(train_state)
    predictor_train_state = replicate(predictor_train_state)

    # function to get predictions of the pretrained predictor
    @partial(jax.pmap, axis_name='dp', donate_argnums=1)
    def eval_predictor_step(predictor_train_state, rng, batch):
        rng_generator = jax_utils.JaxRNG(rng)

        _, thp1_output, jurkat_output, k562_output = predictor.apply(
            predictor_train_state.params,
            inputs=jax.nn.one_hot(batch['sequences'], 5, dtype=jnp.float32)[:, :, :4],
            deterministic=True,
            rngs=rng_generator(predictor.rng_keys()),
        )

        return batch['thp1_output'], batch['jurkat_output'], batch['k562_output'], \
               thp1_output, jurkat_output, k562_output, \
               rng_generator()

    # first get predictions of the pretrained predictor on the oracle data
    oracle_test_data = FinetuneDataset(FLAGS.oracle_test_data)
    oracle_test_iterator = oracle_test_data.batch_iterator(pmap_axis_dim=jax_device_count)
    batch = next(oracle_test_iterator)
    all_y = {'THP1': [], 'Jurkat': [], 'K562': []}
    all_yhat = {'THP1': [], 'Jurkat': [], 'K562': []}
    while batch is not None:
        thp1_y, jurkat_y, k562_y, \
        thp1_output, jurkat_output, k562_output, \
        rng = eval_predictor_step(
            predictor_train_state, rng, batch
        )
        all_y['THP1'].append(jax.device_get(thp1_y))
        all_y['Jurkat'].append(jax.device_get(jurkat_y))
        all_y['K562'].append(jax.device_get(k562_y))

        all_yhat['THP1'].append(jax.device_get(thp1_output))
        all_yhat['Jurkat'].append(jax.device_get(jurkat_output))
        all_yhat['K562'].append(jax.device_get(k562_output))

        batch = next(oracle_test_iterator)

    all_y = {k: np.hstack(v).reshape(-1) for k, v in all_y.items()}
    all_yhat = {k: np.hstack(v).reshape(-1) for k, v in all_yhat.items()}
    print("y shape: {}".format(all_y["THP1"].shape))
    print("yhat shape: {}".format(all_yhat["THP1"].shape))
    
    test_metrics = {}
    for k in all_y:
        # Compute Pearson correlation
        test_metrics[f'test/{k}_PearsonR'] = stats.pearsonr(
            all_y[k], all_yhat[k]
        )[0]
        # Compute Spearman correlation
        test_metrics[f'test/{k}_SpearmanR'] = stats.spearmanr(
            all_y[k], all_yhat[k]
        )[0]
        # Compute R2
        test_metrics[f'test/{k}_R2'] = r2_score(
            all_y[k], all_yhat[k]
        )

    # print test metrics
    print('Oracle dataset test metrics:')
    print(pformat(test_metrics))

    # create plots of predictions vs. ground truth
    fig, axes = plt.subplots(1, 3, figsize=(15, 7))
    for i, k in enumerate(all_y):
        axes[i].scatter(all_y[k], all_yhat[k], s=1)

        # draw x=y line
        axes[i].plot(
            [all_y[k].min(), all_y[k].max()],
            [all_y[k].min(), all_y[k].max()],
            'k--',
            lw=1
        )

        axes[i].set_xlabel('ground truth')
        axes[i].set_ylabel('prediction')
        axes[i].set_title(k + '\nSpearmanR: {:.3f}'.format(test_metrics[f'test/{k}_SpearmanR']) + '\nPearsonR: {:.3f}'.format(test_metrics[f'test/{k}_PearsonR']))
    plt.savefig(os.path.join('oracle_test_predictions.png'))
    plt.show()
    
    # create pairwise plots of predictions and ground truth
    fig, axes = plt.subplots(1, 3, figsize=(15, 7))
    count = 0
    for i, k in enumerate(all_y):
        for j, k2 in enumerate(all_y):
            if i >= j:
                continue
            axes[count].scatter(all_y[k], all_y[k2], s=1, label="ground truth")
            axes[count].scatter(all_yhat[k], all_yhat[k2], s=1, label="prediction")

            # draw x=y line
            axes[count].plot(
                [all_y[k].min(), all_y[k].max()],
                [all_y[k].min(), all_y[k].max()],
                'k--',
                lw=1
            )

            axes[count].set_xlabel(k)
            axes[count].set_ylabel(k2)
            axes[count].legend()
            axes[count].set_title(k + ' vs. ' + k2)
            count += 1
    plt.savefig(os.path.join('oracle_test_predictions_pairwise.png'))
    plt.show()

    @partial(jax.pmap, axis_name='dp', donate_argnums=(0, 1))
    def train_step(train_state, rng, batch1, batch2):
        rng_generator = jax_utils.JaxRNG(rng)

        def loss_fn(params, rng, batch1, batch2):
            rng_generator = jax_utils.JaxRNG(rng)
            total_loss, \
            fitness_loss, \
            total_diversity_loss, \
            entropy_loss, \
            diversity_loss, \
            intermediate_repr_loss, \
            samples_predictions1, \
            samples_predictions2 = model.apply(
                params,
                inputs1=batch1,
                inputs2=batch2,
                deterministic=False,
                rngs=rng_generator(model.rng_keys()),
            )

            average_thp1 = jnp.mean(samples_predictions1[:, :, 0])
            average_jurkat = jnp.mean(samples_predictions1[:, :, 1])
            average_k562 = jnp.mean(samples_predictions1[:, :, 2])
            
            aux_values = {
                "fitness_loss": fitness_loss,
                "total_diversity_loss": total_diversity_loss,
                "entropy_loss": entropy_loss,
                "diversity_loss": diversity_loss,
                "intermediate_repr_loss": intermediate_repr_loss,
                "loss": total_loss,
                "average_thp1": average_thp1,
                "average_jurkat": average_jurkat,
                "average_k562": average_k562,
            }

            return total_loss, aux_values

        if FLAGS.remat:
            loss_fn = jax.checkpoint(
                loss_fn, policy=jax.checkpoint_policies.checkpoint_dots
            )

        (_, aux_values), grads = jax.value_and_grad(
            loss_fn, has_aux=True
        )(train_state.params, rng_generator(), batch1, batch2)
        grads = jax.lax.pmean(grads, axis_name='dp')

        aux_values['learning_rate'] = learning_rate_schedule(train_state.step)
        aux_values['grad_norm'] = global_norm(grads)
        aux_values['param_norm'] = global_norm(train_state.params)

        metrics = jax_utils.collect_metrics(
            aux_values,
            ['fitness_loss', 'total_diversity_loss', 'entropy_loss', 'diversity_loss', 'intermediate_repr_loss', 'loss',
             "average_thp1", "average_jurkat", "average_k562",
             'learning_rate', 'grad_norm', 'param_norm'],
            prefix='train',
        )
        metrics = jax.lax.pmean(metrics, axis_name='dp')

        train_state = train_state.apply_gradients(grads=grads)
        return train_state, rng_generator(), metrics

    @partial(jax.pmap, axis_name='dp', donate_argnums=2)
    def eval_step(train_state, predictor_train_state, rng, batch):
        rng_generator = jax_utils.JaxRNG(rng)

        total_loss, \
        fitness_loss, \
        total_diversity_loss, \
        entropy_loss, \
        diversity_loss, \
        intermediate_repr_loss, \
        samples_predictions1, \
        samples_predictions2, \
        pwm1, \
        pwm2, \
        samples1, \
        samples2 = model.apply(
            train_state.params,
            inputs1=batch,
            inputs2=batch,
            deterministic=True,
            return_samples=True,
            rngs=rng_generator(model.rng_keys()),
        )

        # _, thp1_output, jurkat_output, k562_output = predictor.apply(
        #     predictor_train_state.params,
        #     inputs=samples1.reshape(-1, samples1.shape[2], samples1.shape[3]),
        #     deterministic=True,
        #     rngs=rng_generator(predictor.rng_keys()),
        # )

        aux_values = {
                "fitness_loss": fitness_loss,
                "average_thp1": jnp.mean(samples_predictions1[:, :, 0]),
                "average_jurkat": jnp.mean(samples_predictions1[:, :, 1]),
                "average_k562": jnp.mean(samples_predictions1[:, :, 2]),
                # "average_thp1_from_predictor": jnp.mean(thp1_output),
                # "average_jurkat_from_predictor": jnp.mean(jurkat_output),
                # "average_k562_from_predictor": jnp.mean(k562_output),
        }

        metrics = jax_utils.collect_metrics(
            aux_values,
            ['fitness_loss', 
             "average_thp1", "average_jurkat", "average_k562", 
            #  "average_thp1_from_predictor", "average_jurkat_from_predictor", "average_k562_from_predictor"
            ],
            prefix='eval',
        )

        metrics = jax.lax.pmean(metrics, axis_name='dp')

        return metrics, samples1, samples_predictions1, rng_generator()

    # define logger
    logger = mlxu.WandBLogger(
        config=FLAGS.logger,
        variant=mlxu.get_user_flags(FLAGS, FLAGS_DEF),
    )

    train_iterator = batch_iterator(FLAGS.batch_size, 100, pmap_axis_dim=jax_device_count)
    val_dataset = DENValidationDataset(FLAGS.batch_size, 100)

    best_val_fitness = np.inf

    for step in trange(FLAGS.total_steps, ncols=0):
        batch1, batch2 = next(train_iterator)
        train_state, rng, train_metrics = train_step(
            train_state, rng, batch1, batch2
        )
        if step % FLAGS.log_freq == 0:
            train_metrics = jax.device_get(unreplicate(train_metrics))
            train_metrics['step'] = step
            logger.log(train_metrics)
            tqdm.write(pformat(train_metrics))

        if step % FLAGS.eval_freq == 0:
            eval_metrics = []
            all_predicted_exps = {'THP1': [], 'Jurkat': [], 'K562': []}

            val_iterator = val_dataset.batch_iterator(pmap_axis_dim=jax_device_count)
            batch = next(val_iterator)
            while batch is not None:
                metrics, samples, samples_predictions, rng = eval_step(
                    train_state, predictor_train_state, rng, batch
                )
                eval_metrics.append(unreplicate(metrics))

                all_predicted_exps['THP1'].append(samples_predictions[:, :, 0].reshape(-1))
                all_predicted_exps['Jurkat'].append(samples_predictions[:, :, 1].reshape(-1))
                all_predicted_exps['K562'].append(samples_predictions[:, :, 2].reshape(-1))

                batch = next(val_iterator)

            eval_metrics = average_metrics(jax.device_get(eval_metrics))

            if eval_metrics['eval/fitness_loss'] < best_val_fitness:
                best_val_fitness = eval_metrics['eval/fitness_loss']
                if FLAGS.save_model:
                    logger.save_pickle(
                        jax.device_get(unreplicate(train_state).params),
                        'best_params.pkl',
                    )

            eval_metrics['eval/fitness_loss'] = best_val_fitness
            eval_metrics['step'] = step
            logger.log(eval_metrics)
            tqdm.write(pformat(eval_metrics))
    
    # # load best params
    # if FLAGS.save_model:
    #     train_state = train_state.replace(
    #         params=mlxu.utils.load_pickle(os.path.join(logger.output_dir, 'best_params.pkl'))
    #     )
    #     train_state = replicate(train_state)

    # # best val metrics
    # val_iterator = val_dataset.batch_iterator(pmap_axis_dim=jax_device_count)
    # batch = next(val_iterator)
    # all_y = {'THP1': [], 'Jurkat': [], 'K562': []}
    # all_yhat = {'THP1': [], 'Jurkat': [], 'K562': []}
    # while batch is not None:
    #     metrics, \
    #         thp1_y, jurkat_y, k562_y, \
    #             thp1_output, jurkat_output, k562_output, rng = eval_step(
    #         train_state, rng, batch
    #     )
    #     all_y['THP1'].append(jax.device_get(thp1_y))
    #     all_y['Jurkat'].append(jax.device_get(jurkat_y))
    #     all_y['K562'].append(jax.device_get(k562_y))

    #     all_yhat['THP1'].append(jax.device_get(thp1_output))
    #     all_yhat['Jurkat'].append(jax.device_get(jurkat_output))
    #     all_yhat['K562'].append(jax.device_get(k562_output))

    #     batch = next(val_iterator)
    
    # all_y = {k: np.hstack(v).reshape(-1) for k, v in all_y.items()}
    # all_yhat = {k: np.hstack(v).reshape(-1) for k, v in all_yhat.items()}
    # print("y shape: {}".format(all_y["THP1"].shape))
    # print("yhat shape: {}".format(all_yhat["THP1"].shape))

    # val_metrics = {}
    # for k in all_y:
    #     # Compute Pearson correlation
    #     val_metrics[f'val/{k}_PearsonR'] = stats.pearsonr(
    #         all_y[k], all_yhat[k]
    #     )[0]
    #     # Compute Spearman correlation
    #     val_metrics[f'val/{k}_SpearmanR'] = stats.spearmanr(
    #         all_y[k], all_yhat[k]
    #     )[0]
    #     # Compute R2
    #     val_metrics[f'val/{k}_R2'] = r2_score(
    #         all_y[k], all_yhat[k]
    #     )
    
    # # print best val metrics
    # print('Best val metrics:')
    # print(pformat(val_metrics))

    # # test metrics
    # test_iterator = test_dataset.batch_iterator(pmap_axis_dim=jax_device_count)
    # batch = next(test_iterator)
    # all_y = {'THP1': [], 'Jurkat': [], 'K562': []}
    # all_yhat = {'THP1': [], 'Jurkat': [], 'K562': []}
    # while batch is not None:
    #     metrics, \
    #         thp1_y, jurkat_y, k562_y, \
    #             thp1_output, jurkat_output, k562_output, rng = eval_step(
    #         train_state, rng, batch
    #     )
    #     all_y['THP1'].append(jax.device_get(thp1_y))
    #     all_y['Jurkat'].append(jax.device_get(jurkat_y))
    #     all_y['K562'].append(jax.device_get(k562_y))

    #     all_yhat['THP1'].append(jax.device_get(thp1_output))
    #     all_yhat['Jurkat'].append(jax.device_get(jurkat_output))
    #     all_yhat['K562'].append(jax.device_get(k562_output))

    #     batch = next(test_iterator)

    # all_y = {k: np.hstack(v).reshape(-1) for k, v in all_y.items()}
    # all_yhat = {k: np.hstack(v).reshape(-1) for k, v in all_yhat.items()}
    # print("y shape: {}".format(all_y["THP1"].shape))
    # print("yhat shape: {}".format(all_yhat["THP1"].shape))
    
    # test_metrics = {}
    # for k in all_y:
    #     # Compute Pearson correlation
    #     test_metrics[f'test/{k}_PearsonR'] = stats.pearsonr(
    #         all_y[k], all_yhat[k]
    #     )[0]
    #     # Compute Spearman correlation
    #     test_metrics[f'test/{k}_SpearmanR'] = stats.spearmanr(
    #         all_y[k], all_yhat[k]
    #     )[0]
    #     # Compute R2
    #     test_metrics[f'test/{k}_R2'] = r2_score(
    #         all_y[k], all_yhat[k]
    #     )

    # # print test metrics
    # logger.log(test_metrics)
    # print('Test metrics:')
    # print(pformat(test_metrics))


if __name__ == '__main__':
    mlxu.run(main)