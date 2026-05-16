#!/bin/bash
# Array-job sweep over random seeds on the Lancaster Hex cluster.
# Submit with: sbatch hex/run_sweep.slurm
#
# --array=0-9 runs 10 independent seeds in parallel. Bump the range for more.
# Each task writes its own run_N{N}_seed{i}.npz, so outputs don't collide.

#SBATCH -J lif_sweep
#SBATCH -t 04:00:00
#SBATCH --partition=cpu-6h
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --array=0-9
#SBATCH -o logs/%x_%A_%a.out
#SBATCH -e logs/%x_%A_%a.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs results

source .venv/bin/activate

export PYTHONUNBUFFERED=1

echo "[task $SLURM_ARRAY_TASK_ID] host=$(hostname) started=$(date)"

python random_neurons.py \
    --N 2000 \
    --warmup 5000 \
    --steps 20000 \
    --seed "$SLURM_ARRAY_TASK_ID" \
    --outdir results

echo "[task $SLURM_ARRAY_TASK_ID] finished=$(date)"
