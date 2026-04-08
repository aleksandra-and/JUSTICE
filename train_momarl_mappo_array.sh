#!/bin/sh
#
#SBATCH --job-name="momarl-mappo"
#SBATCH --partition=compute
#SBATCH --time=23:50:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem-per-cpu=3G
#SBATCH --account=research-eemcs-insy
#SBATCH --array=0-10%11
# 11 array jobs, first 10 train 9 weights each, last trains 10 (100 total)
# %11 = all concurrent (11 × 20 = 220 CPUs, well under 2400 research limit)

cd /scratch/aandrasz/JUSTICE

module load 2025
module load python

source venv/bin/activate

# Each array task handles 9 weights (~1h each = ~9h, well under 16h limit)
WEIGHTS_PER_JOB=9
START_WEIGHT=$((SLURM_ARRAY_TASK_ID * WEIGHTS_PER_JOB))
END_WEIGHT=$((START_WEIGHT + WEIGHTS_PER_JOB))

# Last job picks up remaining weights
TOTAL_WEIGHTS=100
if [ $END_WEIGHT -gt $TOTAL_WEIGHTS ]; then
    END_WEIGHT=$TOTAL_WEIGHTS
fi

echo "Array task $SLURM_ARRAY_TASK_ID: training weights $START_WEIGHT to $((END_WEIGHT - 1))"

srun python thesis_rl/train_momarl.py --load_config 'thesis_rl/configs/mappo_temp_econ.yaml' \
    --wandb_project momarl_happo_mappo_2 \
    --weights_generation uniform \
    --num_weights $TOTAL_WEIGHTS \
    --start_uniform_weight $START_WEIGHT \
    --end_uniform_weight $END_WEIGHT > run_momarl_mappo_${SLURM_ARRAY_TASK_ID}.log
