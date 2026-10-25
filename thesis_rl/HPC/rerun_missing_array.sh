#!/bin/sh
#
#SBATCH --job-name="momarl-rerun-missing"
#SBATCH --partition=compute
#SBATCH --time=10:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem-per-cpu=3G
#SBATCH --account=Education-EEMCS-MSc-CS
#SBATCH --array=0-23%24
# 24 missing runs (1 task per weight), all concurrent
# HAPPO seed=2 : 1  run  (indices 0)
# HAPPO seed=3 : 9  runs (indices 1-9)
# MAPPO seed=3 : 14 runs (indices 10-23)

cd /scratch/aandrasz/JUSTICE

module load 2025
module load python

source venv/bin/activate

export WANDB_MODE=offline

# --- Lookup tables ---
# Format: "ALGO SEED WEIGHT_IDX"
TASKS=(
    "HAPPO 2 3"
    "HAPPO 3 9"
    "HAPPO 3 18"
    "HAPPO 3 19"
    "HAPPO 3 35"
    "HAPPO 3 36"
    "HAPPO 3 37"
    "HAPPO 3 38"
    "HAPPO 3 39"
    "HAPPO 3 89"
    "MAPPO 3 8"
    "MAPPO 3 9"
    "MAPPO 3 18"
    "MAPPO 3 19"
    "MAPPO 3 47"
    "MAPPO 3 48"
    "MAPPO 3 49"
    "MAPPO 3 59"
    "MAPPO 3 69"
    "MAPPO 3 95"
    "MAPPO 3 96"
    "MAPPO 3 97"
    "MAPPO 3 98"
    "MAPPO 3 99"
)

TASK="${TASKS[$SLURM_ARRAY_TASK_ID]}"
ALGO=$(echo $TASK | awk '{print $1}')
SEED=$(echo $TASK | awk '{print $2}')
W_IDX=$(echo $TASK | awk '{print $3}')
W_END=$((W_IDX + 1))

if [ "$ALGO" = "HAPPO" ]; then
    CONFIG="thesis_rl/configs/happo_vs_mappo/happo_temp_econ_continous_v2.yaml"
else
    CONFIG="thesis_rl/configs/happo_vs_mappo/mappo_temp_econ_continous_v2.yaml"
fi

echo "Task $SLURM_ARRAY_TASK_ID: $ALGO seed=$SEED weight_idx=$W_IDX"

srun python thesis_rl/train_momarl.py \
    --load_config "$CONFIG" \
    --wandb_project momarl_algorithm_comparison \
    --seed "$SEED" \
    --weights_generation uniform \
    --num_weights 100 \
    --start_uniform_weight "$W_IDX" \
    --end_uniform_weight "$W_END" \
    > "rerun_missing_${SLURM_ARRAY_TASK_ID}_${ALGO}_s${SEED}_w${W_IDX}.log"
