#!/bin/bash
#SBATCH --job-name="200kMM"
#SBATCH --partition=memory
#SBATCH --time=30:00:00
#SBATCH --ntasks=45
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16G
#SBATCH --account=research-tpm-mas
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err    # <- add an explicit error log

module load 2025
module load openmpi


source /scratch/$USER/.conda/etc/profile.d/conda.sh
conda activate justice311

# Prevent threaded BLAS from oversubscribing
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs                 # <- ensure logs dir exists

# MS: 1 master + 1 worker (BORG_ISLANDS env not used by MSBorgMOEA, but harmless) N islands each with K workers, request N*(K+1) + 1 MPI nodes when submitting the job
export BORG_ISLANDS=4

# Per-run args
nfe=200000
myswf=0
seed=555
scenario_index=2

mpirun -np "$SLURM_NTASKS" python hpc_run.py "$nfe" "$myswf" "$seed" "$scenario_index"