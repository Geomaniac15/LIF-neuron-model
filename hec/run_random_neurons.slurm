#!/bin/bash
# Single-run submission for random_neurons.py on the Lancaster HEC.
# Submit with: sbatch hec/run_random_neurons.slurm
#
# Tune --time and --mem to your run size. Rough guide:
#   N=500,  steps=10000  -> ~10 min,    1G  RAM
#   N=2000, steps=20000  -> ~2 hours,   4G  RAM
#   N=5000, steps=20000  -> ~12+ hours, 16G RAM (consider --no-spikes)

#SBATCH -p serial
#SBATCH -J lif_ei
#SBATCH -t 04:00:00
#SBATCH --mem=8G
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

source /etc/profile
module add miniforge
source activate "$global_storage/lif_env"

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs results

echo "[job] host=$(hostname) jobid=$SLURM_JOB_ID started=$(date)"

python random_neurons.py \
    --N 2000 \
    --warmup 5000 \
    --steps 20000 \
    --seed 0 \
    --outdir results

echo "[job] finished=$(date)"
