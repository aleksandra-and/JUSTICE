#!/bin/sh
#
#SBATCH --job-name="momarl-array"
#SBATCH --partition=compute
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem-per-cpu=3G
#SBATCH --account=Education-EEMCS-MSc-CS
#SBATCH --array=2-19%3
# 20 array jobs, each training 5 weights (100 total)
# %3 limits to 3 concurrent jobs (3 × 20 = 60 CPUs)

cd /scratch/aandrasz/JUSTICE

module load 2025
module load python

source venv/bin/activate

export WANDB_MODE=offline

# Each array task handles 5 weights
# Task 0: weights 0-4, Task 1: weights 5-9, etc.
WEIGHTS_PER_JOB=5
START_WEIGHT=$((SLURM_ARRAY_TASK_ID * WEIGHTS_PER_JOB))
END_WEIGHT=$((START_WEIGHT + WEIGHTS_PER_JOB))

echo "Array task $SLURM_ARRAY_TASK_ID: training weights $START_WEIGHT to $((END_WEIGHT - 1))"

srun python thesis_rl/train_momarl.py --load_config 'thesis_rl/configs/economy_welfare.yaml' \
    --wandb_project momarl_welfare \
    --weights_generation uniform \
    --num_weights 100 \
    --start_uniform_weight $START_WEIGHT \
    --end_uniform_weight $END_WEIGHT > run_momarl_happo_b${SLURM_ARRAY_TASK_ID}.log