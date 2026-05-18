#!/bin/bash
# Single-run submission for random_neurons_fixed.py on the Lancaster HEC.
# Submit with: sbatch run_random_neurons_fixed.slurm
#
# Wall time bumped vs the original because the sign-fixed simulator fires
# ~40x more spikes per second, so the STDP inner loop does more work.

#SBATCH -p cpu-6h
#SBATCH -J lif_ei_fixed
#SBATCH -t 02:00:00
#SBATCH --mem=8G
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

source /etc/profile
module add miniforge
source activate "$global_storage/lif_env"

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs results_fixed

echo "[job] host=$(hostname) jobid=$SLURM_JOB_ID started=$(date)"

python random_neurons_fixed.py \
    --N 2000 \
    --warmup 5000 \
    --steps 20000 \
    --seed 0 \
    --outdir results_fixed

echo "[job] finished=$(date)"
