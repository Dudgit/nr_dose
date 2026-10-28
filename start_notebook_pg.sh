#!/bin/bash
module load singularity
export PROJECT_DIR= "$(pwd)"
#"/home/nr_dodb/nr_dose"
export REAL_PROJECT_DIR=$(readlink -f $PROJECT_DIR)
export DATA_DIR="/home/nr_dopg/nr_dose_scratch"
#export REAL_DATA_DIR=$(readlink -f $DATA_DIR)
export REAL_SCRATCH=$(readlink -f $DATA_DIR)

echo "Server runs at http://10.150.1.134:18889/?token=flora"
export SINGULARITYENV_JUPYTER_PORT=18889
export SINGULARITYENV_PYTHONPATH=""  # prevent host path injection

singularity exec --nv \
    --bind $REAL_PROJECT_DIR:/project/nr_dose \
    --bind $REAL_SCRATCH:$REAL_SCRATCH \
    dose.sif \
    jupyter lab --no-browser \
    --NotebookApp.token=flora \
    --ServerApp.root_dir=/project/nr_dose

    #--bind $REAL_SCRATCH:/project/nr_dose_scratch \
