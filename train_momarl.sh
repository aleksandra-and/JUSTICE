#!/bin/sh
#
#SBATCH --job-name="thesis-rl-basic-ppo"
#SBATCH --partition=compute
#SBATCH --time=20:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem-per-cpu=3G
#SBATCH --account=Education-EEMCS-MSc-CS

# Change to JUSTICE root directory
cd /scratch/aandrasz/JUSTICE

module load 2025
module load python

# Activate virtual environment and set Python path
source venv/bin/activate

export WANDB_MODE=offline

# Run training script from JUSTICE root
# reward can be: 'stepwise_marl_reward', 'consumption_per_capita', 'regional_temperature', 'global_temperature'
srun  python thesis_rl/train_momarl.py --algo mappo --env harl_justice_momarl \
    --weights_generation OLS  \
    --num_weights 100 --start_uniform_weight 10 --end_uniform_weight 20 \
    --timesteps_per_weight 2508000 > run_momarl_mappo_uniform.log
# srun wandb sync /scratch/aandrasz/JUSTICE/wandb/offline-*
