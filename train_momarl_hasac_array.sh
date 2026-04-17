#!/bin/sh
#
#SBATCH --job-name="momarl-mappo"
#SBATCH --partition=compute
#SBATCH --time=13:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem-per-cpu=3G
#SBATCH --account=research-eemcs-insy
#SBATCH --array=0-9%10
# 10 array jobs, first 9 train 10 weights each, last trains 10 (100 total)
# %10 = all concurrent (10 × 20 = 200 CPUs, well under 2400 research limit)

cd /scratch/aandrasz/JUSTICE

module load 2025
module load python

source venv/bin/activate

# Each array task handles 10 weights (~1h each = ~10h)
WEIGHTS_PER_JOB=10
START_WEIGHT=$((SLURM_ARRAY_TASK_ID * WEIGHTS_PER_JOB))
END_WEIGHT=$((START_WEIGHT + WEIGHTS_PER_JOB))

export WANDB_MODE=offline

# Last job picks up remaining weights
TOTAL_WEIGHTS=100
if [ $END_WEIGHT -gt $TOTAL_WEIGHTS ]; then
    END_WEIGHT=$TOTAL_WEIGHTS
fi

echo "Array task $SLURM_ARRAY_TASK_ID: training weights $START_WEIGHT to $((END_WEIGHT - 1))"

srun python thesis_rl/train_momarl.py --load_config 'thesis_rl/configs/hasac_temp_econ.yaml' \
    --wandb_project momarl_algo_comparison \
    --weights_generation uniform \
    --num_weights $TOTAL_WEIGHTS \
    --start_uniform_weight $START_WEIGHT \
    --end_uniform_weight $END_WEIGHT > run_momarl_hasac_${SLURM_ARRAY_TASK_ID}.log
