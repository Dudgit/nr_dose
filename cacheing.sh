#!/bin/bash
#SBATCH --job-name=dose_cache
#SBATCH --partition=cpu          # or the appropriate CPU partition
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --output=logs/cache.log

module load singularity

export SCRATCH="/home/nr_dodb/nr_dose_scratch"
export REAL_SCRATCH=$(readlink -f $SCRATCH)

export HDD="/home/nr_dodb/nr_dose"
export REAL_HDD=$(readlink -f $HDD)

singularity exec \
    -B $REAL_SCRATCH:$REAL_SCRATCH \
    -B $REAL_HDD:$REAL_HDD \
    dose.sif \
    python create_cache.py