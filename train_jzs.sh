#!/bin/bash
#SBATCH --job-name=Energy_improvement
#SBATCH --partition=ai
#SBATCH --gres=gpu:4
#SBATCH --mem=300G 
#SBATCH --cpus-per-task=4            
#SBATCH --time=4:00:00
#SBATCH --output=logs/new_ener.log

export SINGULARITYENV_WANDB_API_KEY="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"
export SCRATCH="/home/nr_dodb/nr_dose_scratch"
export REAL_SCRATCH=$(readlink -f $SCRATCH)

export HDD="/home/nr_dodb/nr_dose"
export REAL_HDD=$(readlink -f $HDD)

module load singularity
singularity exec --nv -B $REAL_SCRATCH:$REAL_SCRATCH -B $REAL_HDD:$REAL_HDD dose.sif python $REAL_HDD/train.py --hw komondor --config jzs
