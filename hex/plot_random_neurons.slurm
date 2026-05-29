#!/bin/bash
# Plotting re-run for random_neurons_fixed.py on the Lancaster Hex cluster.
# Submit with: sbatch hex/plot_random_neurons.slurm
#
# This is the WINNER re-run: spikes + plots ON (the sweep ran with both OFF for
# speed). Point it at the config(s) you picked from the sweep stats. Each task
# writes to its OWN outdir so .npz / raster / weights files never collide.
#
# Default below plots two candidate configs across the three seeds (6 tasks).
# Trim CONFIGS / SEEDS and the --array range to plot only your actual winner.
#
# Plotting runs are heavier than the stats-only sweep: spike recording uses more
# memory (every spike stored) and figure rendering adds time. Mem bumped to 12G.

#SBATCH -J lif_ei_plot
#SBATCH -t 04:00:00
#SBATCH --partition=cpu-6h
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --array=0-5
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

# ----- configs to plot -----
# Edit these down to your winning config(s) once you've read the sweep stats.
# Each CONFIG entry is "<a-minus> <I-lo> <I-hi>"; a-plus stays 0.002.
SEEDS=(3 4 5)

CONFIGS=(
    '0.0022 1.1 1.3'
    '0.0024 1.1 1.3'
)

N_CONFIGS=${#CONFIGS[@]}
SEED_IDX=$(( SLURM_ARRAY_TASK_ID / N_CONFIGS ))
CONFIG_IDX=$(( SLURM_ARRAY_TASK_ID % N_CONFIGS ))

SEED=${SEEDS[$SEED_IDX]}
read -r A_MINUS I_LO I_HI <<< "${CONFIGS[$CONFIG_IDX]}"

# Per-task outdir so plots + .npz don't overwrite across configs sharing a seed.
OUTDIR="results/plot_cfg${CONFIG_IDX}_seed${SEED}"
mkdir -p "$OUTDIR"

echo "[job] host=$(hostname) jobid=$SLURM_JOB_ID arrayid=$SLURM_ARRAY_TASK_ID started=$(date)"
echo "[job] seed=$SEED a_minus=$A_MINUS I=[$I_LO,$I_HI] outdir=$OUTDIR"

# Note: NO --no-spikes / --no-plots here. Spikes are needed for the raster,
# and plots are the whole point of this run.
python random_neurons_fixed.py \
    --N 2000 \
    --warmup 5000 \
    --steps 20000 \
    --seed "$SEED" \
    --a-plus 0.002 \
    --a-minus "$A_MINUS" \
    --I-lo "$I_LO" \
    --I-hi "$I_HI" \
    --outdir "$OUTDIR"

echo "[job] finished=$(date)"
