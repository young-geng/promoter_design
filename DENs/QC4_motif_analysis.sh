#!/bin/bash
# Job name:
#SBATCH --job-name=Splicing
#
# Account:
#SBATCH --account=co_nilah
#
# Partition:
#SBATCH --partition=savio2
#
# Request one node:
#SBATCH --nodes=1
#
# Wall clock limit:
#SBATCH --time=72:00:00
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
source $bashrc_path
cd $scripts_dir
conda activate $conda_env_path

NEW_CACHE=$TMPDIR/cache_$fasta_path
mkdir -p $NEW_CACHE
if [ -z $XDG_CACHE_HOME ]; then
    XDG_CACHE_HOME=$HOME/.cache
fi
cp -r $XDG_CACHE_HOME/gimmemotifs $NEW_CACHE/
export XDG_CACHE_HOME=$NEW_CACHE
echo $XDG_CACHE_HOME

# start main script
python QC4_motif_analysis.py $fasta_path $denovo_output_dir $known_analysis_output_path $pfm_file_path