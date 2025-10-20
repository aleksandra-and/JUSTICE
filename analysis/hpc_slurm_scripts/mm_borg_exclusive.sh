#!/bin/bash
#SBATCH --job-name=ExBorg
#SBATCH --partition=compute-p2
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=61          # 5 islands, 11 workers each -> 61 ranks
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --account=research-tpm-mas
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

module load 2025
module load openmpi
module load miniconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate justice311

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

export BORG_ISLANDS=5

nfe=2000
myswf=0
seed=100
scenario_index=2

mpirun -np "$SLURM_NTASKS" python hpc_run.py "$nfe" "$myswf" "$seed" "$scenario_index"