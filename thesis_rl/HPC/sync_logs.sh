#!/bin/sh
#
#SBATCH --job-name="sync-wandb"
#SBATCH --partition=compute
#SBATCH --time=3:00:00
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


# Fail fast if this node cannot reach W&B. Compute nodes on some clusters are egress-restricted.
if ! curl -sSf --connect-timeout 10 https://api.wandb.ai >/dev/null; then
	echo "Cannot reach https://api.wandb.ai from this node."
	echo "Run sync from a node/partition with outbound internet access."
	exit 2
fi


if [ ${#offline_runs[@]} -eq 0 ]; then
	echo "No offline W&B runs found in /scratch/aandrasz/JUSTICE/wandb"
	exit 0
fi

wandb sync /scratch/aandrasz/JUSTICE/wandb/offline-run-202605* > syncing.log
