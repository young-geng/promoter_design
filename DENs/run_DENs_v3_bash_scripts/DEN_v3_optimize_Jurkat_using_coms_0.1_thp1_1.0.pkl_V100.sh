#!/bin/bash
export scripts_dir=/global/home/users/aniketh/promoter/
export pretrained_predictor_path=/global/scratch/users/aniketh/promoter_modelling/jax_data/trained_predictors/coms_0.1_thp1_1.0.pkl
export diff_exp_cell_ind=1
export oracle_test_data_path=/global/scratch/users/aniketh/promoter_modelling/jax_data/finetune_data.pkl
export saved_models_dir=/global/scratch/users/aniketh/promoter_modelling/jax_data/saved_DEN_models_v3
export wandb_dir=/global/scratch/users/aniketh/promoter_modelling/jax_data/wandb_v3_imp
export experiment_id=DEN_v3_optimize_Jurkat_using_coms_0.1_thp1_1.0.pkl_V100
bash /global/home/users/aniketh/promoter/promoter/run_DENs_v3.sh
