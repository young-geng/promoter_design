#!/bin/bash
export scripts_dir=/global/home/users/aniketh/promoter/
export pretrained_predictor_path=/global/scratch/users/aniketh/promoter_modelling/jax_data/trained_predictors/coms_0.03_thp1_1.0.pkl
export diff_exp_cell_ind=0
export oracle_test_data_path=/global/scratch/users/aniketh/promoter_modelling/jax_data/finetune_data.pkl
export saved_models_dir=/global/scratch/users/aniketh/promoter_modelling/jax_data/saved_DEN_models
export wandb_dir=/global/scratch/users/aniketh/promoter_modelling/jax_data/wandb_v2
export experiment_id=DEN_optimize_THP1_using_coms_0.03_thp1_1.0.pkl_V100
bash /global/home/users/aniketh/promoter/promoter/run_DENs.sh
