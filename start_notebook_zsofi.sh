#!/bin/bash
module load singularity
export PROJECT_DIR= $(pwd)
#"/home/nr_dodb/nr_dose"
export REAL_PROJECT_DIR=$(readlink -f $PROJECT_DIR)
export DATA_DIR="/home/nr_dojz/nr_dose_scratch"
export REAL_DATA_DIR=$(readlink -f $DATA_DIR)

echo "Server runs at http://10.150.1.134:18888/?token=flora"
export SINGULARITYENV_JUPYTER_PORT=18888
export SINGULARITYENV_PYTHONPATH=""  # prevent host path injection

singularity exec --nv \
    --bind $REAL_PROJECT_DIR:/project/nr_dose \
    --bind $REAL_DATA_DIR:/project/nr_dose_scratch \
    utils/dose.sif \
    jupyter lab --no-browser \
    --NotebookApp.token=flora \
    --ServerApp.root_dir=/project/nr_dose