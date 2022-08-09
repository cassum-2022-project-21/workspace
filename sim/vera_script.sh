#!/usr/bin/env bash

#SBATCH -A C3SE508-18-3 -p vera
#SBATCH -J <<<JOB_NAME>>>
#SBATCH -N 1 --exclusive
#SBATCH -t 0-24:00:00

# flat_modules
# module load foss/2021b SciPy-bundle/2021.10-foss-2021b matplotlib/3.4.3-foss-2021b
# source /priv/c3-astro-scratch1/iostelea/bert/bert-python/bin/activate

source ~/bert-vera.sh

DIRS=($(ls -d */))

# Fetch one directory from the array based on the task ID (index starts from 0)
CURRENT_DIR=${DIRS[$SLURM_ARRAY_TASK_ID]}

echo "Running simulation $CURRENT_DIR"

# Go to folder
cd $CURRENT_DIR

/usr/bin/env bash exec.sh
