#!/bin/bash
# Single-run submission for random_neurons.py on the Lancaster Hex cluster.
# Submit with: sbatch hex/run_random_neurons.slurm
#
# Tune --time and --mem to your run size. Rough guide (same as HEC version):
#   N=500,  steps=10000  -> ~10 min,    1G  RAM
#   N=2000, steps=20000  -> ~2 hours,   4G  RAM
#   N=5000, steps=20000  -> ~12+ hours, 16G RAM

#SBATCH -J lif_ei
#SBATCH -t 04:00:00
#SBATCH --partition=cpu-6h
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs results

# Activate the project venv created by hex/setup_env.sh
source .venv/bin/activate

export PYTHONUNBUFFERED=1

echo "[job] host=$(hostname) jobid=$SLURM_JOB_ID started=$(date)"

python random_neurons.py \
    --N 2000 \
    --warmup 5000 \
    --steps 20000 \
    --seed 0 \
    --outdir results

echo "[job] finished=$(date)"
