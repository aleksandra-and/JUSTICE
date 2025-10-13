#!/bin/bash
#SBATCH --job-name="mm10kBorg"
#SBATCH --partition=memory
#SBATCH --time=20:00:00
#SBATCH --ntasks=45
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16G
#SBATCH --account=research-tpm-mas
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err    # <- add an explicit error log

module load 2025
module load openmpi
module load miniconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate justice311

# Prevent threaded BLAS from oversubscribing
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs                 # <- ensure logs dir exists

# MS: 1 master + 1 worker (BORG_ISLANDS env not used by MSBorgMOEA, but harmless)
export BORG_ISLANDS=4

# Per-run args
nfe=10000
myswf=0
seed=69

mpirun -np "$SLURM_NTASKS" python hpc_run.py "$nfe" "$myswf" "$seed"