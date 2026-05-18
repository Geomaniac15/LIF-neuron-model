#!/bin/bash
# Array-job sweep over random seeds for random_neurons_fixed.py.
# Submit with: sbatch run_sweep_fixed.slurm
#
# --array=0-9 runs 10 independent seeds in parallel. Each task writes its
# own results_fixed/run_N2000_seed{i}.npz so outputs don't collide.

#SBATCH -p cpu-6h
#SBATCH -J lif_sweep_fixed
#SBATCH -t 02:00:00
#SBATCH --mem=8G
#SBATCH --array=0-9
#SBATCH -o logs/%x_%A_%a.out
#SBATCH -e logs/%x_%A_%a.err

source /etc/profile
module add miniforge
source activate "$global_storage/lif_env"

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs results_fixed

echo "[task $SLURM_ARRAY_TASK_ID] host=$(hostname) started=$(date)"

python random_neurons_fixed.py \
    --N 2000 \
    --warmup 5000 \
    --steps 20000 \
    --seed "$SLURM_ARRAY_TASK_ID" \
    --outdir results_fixed

echo "[task $SLURM_ARRAY_TASK_ID] finished=$(date)"
