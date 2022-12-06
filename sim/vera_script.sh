#!/usr/bin/env bash

#SBATCH -A C3SE508-18-3 -p astro1
#SBATCH -J <<<JOB_NAME>>>
#SBATCH --ntasks=1

#SBATCH -t 1-12:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu 1024MB

# flat_modules
# module load foss/2021b SciPy-bundle/2021.10-foss-2021b matplotlib/3.4.3-foss-2021b
# source /priv/c3-astro-scratch1/iostelea/bert/bert-python/bin/activate

source ~/bert-vera.sh

# Fetch one directory from the array based on the task ID (index starts from 0)
EXECS=($(find . -name "exec.sh"))
CURRENT_DIR=$(dirname ${EXECS[$SLURM_ARRAY_TASK_ID]})

(
    cd $CURRENT_DIR
    pwd
    chmod u+x exec.sh && ./exec.sh
)
