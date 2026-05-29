#!/bin/bash
# I-mean x sigma sweep for the asynchronous-irregular (AI) search.
# random_neurons_fixed.py (noisy-drive / inhibition-dominated version).
# Submit with: sbatch hex/ai_sweep.slurm
#
# Grid: 4 I-mean values x 4 sigma values = 16 tasks (--array=0-15).
# Fixed:  g_inh=5.0, connectivity=0.1, a-plus=0.002, a-minus=0.0024.
# Goal:   find the cell with LOW pop_CV and HIGH ISI_CV (the model self-reports
#         both on its [stats] line). Lower firing rate + larger sigma is the
#         expected direction, so the grid leans that way.
#
# Stats-only triage: --no-spikes --no-plots keeps each task cheap. The regime
# metrics (pop_CV, ISI_CV) need spikes, so they are DISABLED here too -- wait,
# they don't: see note below. Read the winner from the logs, then re-run it
# with plots (drop both flags) to get the raster.
#
# NOTE: regime stats (pop_CV / ISI_CV) are computed from the spike record, so
# they only print when spikes are recorded. For this sweep we KEEP spikes on
# (no --no-spikes) precisely because the whole point is to read ISI_CV. We drop
# only --no-plots. At N=2000 that costs some memory for the spike list; mem is
# set to 12G to cover it.

#SBATCH -J lif_ai_sweep
#SBATCH -t 04:00:00
#SBATCH --partition=cpu-6h
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --array=0-15
#SBATCH -o logs/%x_%A_%a.out
#SBATCH -e logs/%x_%A_%a.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs results

# Relax -u around venv activation (some activate scripts touch unset vars).
set +u
source .venv/bin/activate
set -u

export PYTHONUNBUFFERED=1

# ----- grid -----
I_MEANS=(1.2 1.3 1.4 1.5)
SIGMAS=(2.0 3.0 4.0 5.0)

N_SIGMA=${#SIGMAS[@]}                              # 4
IM_IDX=$(( SLURM_ARRAY_TASK_ID / N_SIGMA ))
SG_IDX=$(( SLURM_ARRAY_TASK_ID % N_SIGMA ))

I_MEAN=${I_MEANS[$IM_IDX]}
SIGMA=${SIGMAS[$SG_IDX]}

# Each cell to its own outdir so the .npz / figures never collide
# (filenames key on N+seed only, which is constant across the grid).
OUTDIR="results/ai_im${I_MEAN}_sg${SIGMA}"
mkdir -p "$OUTDIR"

echo "[job] host=$(hostname) jobid=$SLURM_JOB_ID arrayid=$SLURM_ARRAY_TASK_ID started=$(date)"
echo "[job] I_mean=$I_MEAN sigma=$SIGMA g_inh=5.0 conn=0.1 outdir=$OUTDIR"

# Spikes ON (needed for pop_CV / ISI_CV). Plots OFF for the bulk sweep.
python random_neurons_fixed.py \
    --N 2000 \
    --warmup 5000 \
    --steps 20000 \
    --seed 0 \
    --I-mean "$I_MEAN" \
    --sigma "$SIGMA" \
    --g-inh 5.0 \
    --connectivity 0.1 \
    --a-plus 0.002 \
    --a-minus 0.0024 \
    --outdir "$OUTDIR" \
    --no-plots

echo "[job] finished=$(date)"
