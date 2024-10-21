#!/bin/bash
# Job name:
#SBATCH --job-name=get_oracle_preds
#
# Account:
#SBATCH --account=co_rail
#
# Partition:
#SBATCH --partition=savio3_gpu
#
# QoS:
#SBATCH --qos=savio_lowprio
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

python -m promoter.get_finetune_data_predictions \
    --pretrained_predictor_path=$pretrained_predictor_path \
    --oracle_test_data.path=$oracle_test_data_path \
    --predictions_save_dir=$predictions_save_dir