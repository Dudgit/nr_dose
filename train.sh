#!/bin/bash
#SBATCH --job-name=Dose_train
#SBATCH --partition=ai
#SBATCH --gres=gpu:4
#SBATCH --mem=250G 
#SBATCH --cpus-per-task=32            
#SBATCH --time=8:00:00
#SBATCH --output=logs/vanilla.log

module load singularity

export SCRATCH="$(pwd)/nr_dose"
export REAL_SCRATCH=$(readlink -f $SCRATCH)

export HDD="/home/nr_fldb/nr_floraai/data/ct_rate_subset/dataset/train_fixed"
export REAL_HDD=$(readlink -f $HDD)

export SINGULARITYENV_WANDB_API_KEY="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"
singularity exec --nv --pwd $REAL_SCRATCH -B $REAL_SCRATCH:$REAL_SCRATCH -B $REAL_HDD:/mnt/ct_data FLORA/flora.sif python main_stage2.py