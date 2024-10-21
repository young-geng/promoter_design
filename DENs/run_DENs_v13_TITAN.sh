#!/bin/bash
# Job name:
#SBATCH --job-name=train_DEN
#
# Account:
#SBATCH --account=co_rail
#
# Partition:
#SBATCH --partition=savio3_gpu
#
# QoS:
#SBATCH --qos=rail_gpu3_normal
#
# Number of tasks (one for each GPU desired for use case):
#SBATCH --ntasks=1
#
# Processors per task (please always specify the total number of processors twice the number of GPUs):
#SBATCH --cpus-per-task=4
#
# Request one GPU:
#SBATCH --gres=gpu:TITAN:1
#
# Wall clock limit:
#SBATCH --time=12:00:00
#
#save output and error messages
#SBATCH --output=/global/scratch/users/aniketh/slurm_logs/slurm_job_%j.out
#SBATCH --error=/global/scratch/users/aniketh/slurm_logs/slurm_job_%j.err
#
# send email when job begins
#SBATCH --mail-type=begin  
# send email when job ends      
#SBATCH --mail-type=end  
# send email if job fails        
#SBATCH --mail-type=fail         
#SBATCH --mail-user=aniketh@berkeley.edu
#
## Command(s) to run:
#
source /global/home/users/aniketh/.bashrc
cd $scripts_dir
mamba activate /global/scratch/users/aniketh/mamba_envs/jax+pt_apr132023/

python -m promoter.DEN_main_v13_TITAN \
    --pretrained_predictor_path=$pretrained_predictor_path \
    --loss_config_updates.diff_exp_cell_ind=$diff_exp_cell_ind \
    --loss_config_updates.diversity_loss_coef=$diversity_loss_coef \
    --loss_config_updates.entropy_loss_coef=$entropy_loss_coef \
    --loss_config_updates.base_entropy_loss_coef=$base_entropy_loss_coef \
    --oracle_test_data.path=$oracle_test_data_path \
    --logger.output_dir=$saved_models_dir \
    --logger.wandb_dir=$wandb_dir \
    --logger.online=True \
    --logger.experiment_id=$experiment_id \
    --logger.project="promoter_design_jax_final_v13_TITAN"