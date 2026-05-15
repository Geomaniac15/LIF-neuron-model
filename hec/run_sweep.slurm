#!/bin/bash
# Array-job sweep over random seeds, so you get error bars on emergent dynamics.
# Submit with: sbatch hec/run_sweep.slurm
#
# --array=0-9 runs 10 independent seeds in parallel. Bump the range for more.
# Each task writes its own run_N{N}_seed{i}.npz, so outputs don't collide.

#SBATCH -p serial
#SBATCH -J lif_sweep
#SBATCH -t 04:00:00
#SBATCH --mem=8G
#SBATCH --array=0-9
#SBATCH -o logs/%x_%A_%a.out
#SBATCH -e logs/%x_%A_%a.err

source /etc/profile
module add miniforge
source activate "$global_storage/lif_env"

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs results

echo "[task $SLURM_ARRAY_TASK_ID] host=$(hostname) started=$(date)"

python random_neurons.py \
    --N 2000 \
    --warmup 5000 \
    --steps 20000 \
    --seed "$SLURM_ARRAY_TASK_ID" \
    --outdir results

echo "[task $SLURM_ARRAY_TASK_ID] finished=$(date)"
