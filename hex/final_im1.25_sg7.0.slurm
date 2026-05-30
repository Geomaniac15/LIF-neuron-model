#!/bin/bash
# Final long run of the winning AI cell from the tune_softbound sweep.
# Cell: I_mean=1.25, sigma=7.0 (lowest pop_CV in the grid: 0.040, ISI_CV 0.60).
# random_neurons_fixed.py on the Lancaster Hex cluster.
# Submit with: sbatch hex/final_im1.25_sg7.0.slurm
#
# Differences vs hex/tune_softbound.slurm:
#   * Single cell, not an array. No --array line.
#   * Long run: steps=20000 (was 4000) and warmup=5000 (was 3000) so the weight
#     distribution and regime metrics settle for a publishable figure.
#   * Plots ON (no --no-plots) -> raster + weight histogram saved to OUTDIR.
#   * Stale-code guard: aborts if the w_ceiling inhibition fix is not present in
#     the checkout, so a forgotten `git pull` fails loudly instead of silently
#     regenerating the old (collapsed) result.
#
# Locked params identical to the sweep:
#   g_inh=5.0 conn=0.1 w_total=0.00424 a_plus=0.002 a_minus=0.0024 w_max_mult=2.0

#SBATCH -J lif_final_im1.25_sg7.0
#SBATCH -t 04:00:00
#SBATCH --partition=cpu-6h
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH -o logs_final/%x_%j.out
#SBATCH -e logs_final/%x_%j.err

set -euo pipefail

# NOTE: slurm resolves -o/-e ABOVE before this body runs, so logs_final/ must
# exist at SUBMIT time. Create it once by hand before the first sbatch:
#     mkdir -p logs_final results_final
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs_final results_final

# ---- stale-code guard ----
# The inhibition-collapse fix introduces a per-population ceiling named
# 'w_ceiling'. If it is missing, this checkout predates the fix: abort.
echo "[guard] HEAD=$(git rev-parse --short HEAD 2>/dev/null || echo 'no-git')"
if ! grep -q 'w_ceiling' random_neurons_fixed.py; then
    echo "[guard] FATAL: w_ceiling fix not found in random_neurons_fixed.py."
    echo "[guard] Run 'git pull origin main' in this directory before submitting."
    exit 1
fi
echo "[guard] w_ceiling fix present, proceeding."

# Relax -u around venv activation (some activate scripts touch unset vars).
set +u
source .venv/bin/activate
set -u

export PYTHONUNBUFFERED=1

I_MEAN=1.25
SIGMA=7.0
OUTDIR="results_final/final_im${I_MEAN}_sg${SIGMA}"
mkdir -p "$OUTDIR"

echo "[job] host=$(hostname) jobid=$SLURM_JOB_ID started=$(date)"
echo "[job] I_mean=$I_MEAN sigma=$SIGMA  (locked: g_inh=5.0 conn=0.1 w_total=0.00424 a_plus=0.002 a_minus=0.0024 w_max_mult=2.0)  outdir=$OUTDIR"

python random_neurons_fixed.py \
    --N 2000 \
    --warmup 5000 \
    --steps 20000 \
    --seed 0 \
    --I-mean "$I_MEAN" \
    --sigma "$SIGMA" \
    --g-inh 5.0 \
    --connectivity 0.1 \
    --w-total 0.00424 \
    --a-plus 0.002 \
    --a-minus 0.0024 \
    --w-max-mult 2.0 \
    --outdir "$OUTDIR"

echo "[job] finished=$(date)"
