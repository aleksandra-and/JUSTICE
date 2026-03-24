#!/bin/bash
#SBATCH --job-name="SingleAgent_MOMA"
#SBATCH --partition=memory
#SBATCH --time=00:10:00
#SBATCH --ntasks=45
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16G
#SBATCH --account=research-tpm-mas
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --array=0-4    # one task per macro agent (0..4)

module load 2025
module load openmpi

source /scratch/$USER/.conda/etc/profile.d/conda.sh
conda activate justice311

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

export BORG_ISLANDS=4

# Base run parameters
nfe=200
myswf=0
seed=4444
scenario_index=2
policy_index=0    # <-- row index in pareto_nash_profiles.csv (was MOMA_reference_set.csv)
pop_size=100

# Each array task gets its macro index from SLURM_ARRAY_TASK_ID
macro_index=${SLURM_ARRAY_TASK_ID}

echo "Starting single-agent Nash verification run:"
echo "  nfe               = $nfe"
echo "  seed              = $seed"
echo "  scenario_index    = $scenario_index"
echo "  policy_index      = $policy_index  (row in pareto_nash_profiles.csv)"
echo "  variable_macro_idx= $macro_index"
echo "  population_size   = $pop_size"
echo "  MPI tasks         = $SLURM_NTASKS"

mpirun -np "$SLURM_NTASKS" python hpc_run.py \
    "$nfe" "$myswf" "$seed" "$scenario_index" \
    "$policy_index" "$macro_index" "$pop_size"
