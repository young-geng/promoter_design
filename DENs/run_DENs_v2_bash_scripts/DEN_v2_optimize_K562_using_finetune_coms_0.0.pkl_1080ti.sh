#!/bin/bash
export scripts_dir=/global/home/users/aniketh/promoter/
export pretrained_predictor_path=/global/scratch/users/aniketh/promoter_modelling/jax_data/trained_predictors/finetune_coms_0.0.pkl
export diff_exp_cell_ind=2
export oracle_test_data_path=/global/scratch/users/aniketh/promoter_modelling/jax_data/finetune_data.pkl
export saved_models_dir=/global/scratch/users/aniketh/promoter_modelling/jax_data/saved_DEN_models_v2
export wandb_dir=/global/scratch/users/aniketh/promoter_modelling/jax_data/wandb_v2_imp
export experiment_id=DEN_v2_optimize_K562_using_finetune_coms_0.0.pkl_1080ti
bash /global/home/users/aniketh/promoter/promoter/run_DENs_v2.sh
