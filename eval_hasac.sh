#!/bin/sh
#
#SBATCH --job-name="eval-hasac"
#SBATCH --partition=compute
#SBATCH --time=23:50:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=3G
#SBATCH --account=Education-EEMCS-MSc-CS

cd /scratch/aandrasz/JUSTICE

module load 2025
module load python

source venv/bin/activate


srun python evaluate_hasac.py > eval_hasac.log
