#!/usr/bin/env bash

#SBATCH -A C3SE508-18-3 -p astro1
#SBATCH -J <<<JOB_NAME>>>
#SBATCH -N 1 --exclusive
#SBATCH -t 1-12:00:00

# flat_modules
# module load foss/2021b SciPy-bundle/2021.10-foss-2021b matplotlib/3.4.3-foss-2021b
# source /priv/c3-astro-scratch1/iostelea/bert/bert-python/bin/activate

source ~/bert-vera.sh

DIRS=($(ls -d */))

# Fetch one directory from the array based on the task ID (index starts from 0)

for CURRENT_DIR in $DIRS; do
    echo "Running simulation $CURRENT_DIR"
    cd $CURRENT_DIR
    srun -n 1 exec.sh &
    cd ..
done
wait
