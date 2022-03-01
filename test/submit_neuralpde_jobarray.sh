#!/bin/bash

# Slurm sbatch options
#SBATCH -o neuralpde_jobarray.log-%A-%a
#SBATCH -a 1-16
#SBATCH -c 1

# Initialize julia path
source $HOME/.julia_profile

echo "args:"
echo $SLURM_ARRAY_TASK_ID
echo $SLURM_ARRAY_TASK_COUNT

# Call your script as you would from the command line
julia neuralpde_jobarray.jl $SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_COUNT
