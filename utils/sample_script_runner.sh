#!/bin/bash
module load singularity

export PROJECT_DIR="/home/nr_dodb/nr_dose"
export REAL_PROJECT_DIR=$(readlink -f $PROJECT_DIR)

singularity exec --nv --bind $REAL_PROJECT_DIR utils/dose.sif python scripts/metrics.py