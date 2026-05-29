#!/bin/bash
# Final confirmation run for random_neurons_fixed.py on the Lancaster Hex cluster.
# Submit with: sbatch hex/confirm_ai.slurm
#
# Locked asynchronous-irregular parameters (found by the I-mean/sigma sweep +
# the w_scale coupling-strength probes):
#     I_mean=1.4  sigma=3.0  g_inh=5.0  connectivity=0.1
#     w_total=0.00424  (=> w_scale 0.0003 at N=2000; size-invariant scaling)
#     a_plus=0.002  a_minus=0.0024
#
# Purpose of THIS run (two things at once):
#   1. Seed robustness: run seeds 0, 1, 2 so the result isn't a seed-0 fluke.
#   2. Full-length + plots: 20000 main steps with spikes + figures ON, to get
#      the raster and weight histogram that are the actual deliverable.
#
# Expected regime (from the short probes): pop_CV ~0.5-0.6, ISI_CV ~0.5, ~12 Hz,
# no silent neurons, depression-dominated stable STDP. The [stats] line self-
# reports pop_CV / ISI_CV so you can confirm all three seeds land together.
# NB: this is "asynchronous-irregular-looking", NOT textbook Poisson (ISI_CV ~1);
# the 2 ms refractory floor caps ISI_CV at this firing rate.
#
# Output folders are separate from earlier runs:
#     logs    -> logs_ai/
#     results -> results_ai/ai_confirm_seed{N}/

#SBATCH -J lif_ai_confirm
#SBATCH -t 04:00:00
#SBATCH --partition=cpu-6h
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --array=0-2
#SBATCH -o logs_ai/%x_%A_%a.out
#SBATCH -e logs_ai/%x_%A_%a.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
# IMPORTANT: slurm resolves the -o/-e paths above BEFORE this script body runs,
# so logs_ai/ must exist at SUBMIT time or the job has nowhere to write its log.
# The mkdir here covers results_ai/ and re-creates logs_ai/ for safety, but if
# this is the first time you use logs_ai/, run once by hand before submitting:
#     mkdir -p logs_ai results_ai
mkdir -p logs_ai results_ai

# Relax -u around venv activation (some activate scripts touch unset vars).
set +u
source .venv/bin/activate
set -u

export PYTHONUNBUFFERED=1

# One task per seed.
SEEDS=(0 1 2)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

# Per-seed outdir keeps each raster / weights / npz separate.
OUTDIR="results_ai/ai_confirm_seed${SEED}"
mkdir -p "$OUTDIR"

echo "[job] host=$(hostname) jobid=$SLURM_JOB_ID arrayid=$SLURM_ARRAY_TASK_ID started=$(date)"
echo "[job] CONFIRM seed=$SEED  locked AI params  outdir=$OUTDIR"

# Spikes ON (needed for raster + regime stats), plots ON (this is the deliverable).
python random_neurons_fixed.py \
    --N 2000 \
    --warmup 5000 \
    --steps 20000 \
    --seed "$SEED" \
    --I-mean 1.4 \
    --sigma 3.0 \
    --g-inh 5.0 \
    --connectivity 0.1 \
    --w-total 0.00424 \
    --a-plus 0.002 \
    --a-minus 0.0024 \
    --outdir "$OUTDIR"

echo "[job] finished=$(date)"
