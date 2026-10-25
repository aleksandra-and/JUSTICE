#!/bin/sh
#
#SBATCH --job-name="thesis-rl-basic-ppo"
#SBATCH --partition=compute
#SBATCH --time=10:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=3G
#SBATCH --account=Education-EEMCS-MSc-CS

# Change to JUSTICE root directory
cd /scratch/aandrasz/JUSTICE

module load 2025
module load python

# Activate virtual environment and set Python path
source venv/bin/activate

# export WANDB_API_KEY=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
export WANDB_MODE=offline

# Run training script from JUSTICE root
# reward can be: 'stepwise_marl_reward', 'consumption_per_capita', 'regional_temperature', 'global_temperature'
srun  python ./thesis_rl/train.py --env harl_justice --algo happo > run2.log

# srun wandb sync /scratch/aandrasz/JUSTICE/wandb/offline-*
