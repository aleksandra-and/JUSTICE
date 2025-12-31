#!/bin/sh
#
#SBATCH --job-name="thesis-rl-basic-ppo"
#SBATCH --partition=compute
#SBATCH --time=10:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=3G
#SBATCH --account=Education-EEMCS-MSc-CS

module load 2025
module load python
module load py-numpy
module load py-torch

# Change to JUSTICE root directory
cd /home/aandrasz/JUSTICE

# Activate virtual environment and set Python path
source /home/aandrasz/JUSTICE/env/bin/activate
export PYTHONPATH=/home/aandrasz/JUSTICE:$PYTHONPATH

# Run training script from JUSTICE root
# reward can be: 'stepwise_marl_reward', 'consumption_per_capita', 'regional_temperature', 'global_temperature'
srun python thesis_rl/train.py --total_episodes 20000 --backup_interval 1000 --reward 'stepwise_marl_reward' > run.log