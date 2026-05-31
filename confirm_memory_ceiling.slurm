#!/bin/bash
# Pin down the exact memory ceiling at the chosen adaptation setting.
# reservoir_capacity.py intrinsic benchmark, single cell, long run, deep delay.
# Submit with: sbatch confirm_memory_ceiling.slurm
#
# Why this run (vs sweep_memory_adapt.slurm):
#   The sweep found adapt_b=4 / tau_w=400 ms as the robust sweet spot, but each
#   sweep cell only saw ~24k steps (~160 reservoir-steps) and probed delays out to
#   25 steps. Running LONGER does not add memory capacity (MC is a property of the
#   dynamics, not the runtime) but it sharpens the ESTIMATE: the readout gets far
#   more training data, so the horizon stops bouncing and the per-seed sd shrinks.
#   We also probe delays out to ~525 ms so the true total MC is not truncated.
#
# Fixed cell: adapt_b=4, tau_w=400 ms, input_scale=8, inh_scale=1.0, tau_syn=5 ms.
#   hold=150 (15 ms / reservoir-step); max_delay=35 -> probes up to 525 ms.
#   sweep_steps=90000 -> ~600 reservoir-steps (~285 train / 285 test after washout).
# Array: one task per trained seed, --array=0-9 -> run_N2000_seed0..9.npz.
#   Aggregate the 10 CSVs afterwards with: python aggregate_memory_sweep.py \
#       'sweep_csv/mem_confirm_seed*.csv'
#
# WEIGHTS_DIR defaults to results_fixed/ here; override at submit time:
#   WEIGHTS_DIR=/abs/path sbatch confirm_memory_ceiling.slurm

#SBATCH --partition=cpu-6h
#SBATCH -J lif_mem_confirm
#SBATCH -t 5:00:00
#SBATCH --mem=8G
#SBATCH --array=0-9
#SBATCH -o logs/%x_%A_%a.out
#SBATCH -e logs/%x_%A_%a.err

source /etc/profile
module add miniforge
source activate "$global_storage/lif_env"

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs sweep_csv plots

# ----- trained reservoir weights (one seed per array task) -----
WEIGHTS_DIR="${WEIGHTS_DIR:-results_fixed}"
SEED=$SLURM_ARRAY_TASK_ID
WEIGHTS="$WEIGHTS_DIR/run_N2000_seed${SEED}.npz"
if [ ! -f "$WEIGHTS" ]; then
    echo "[fatal] weights file not found: $WEIGHTS"
    echo "[fatal] set WEIGHTS_DIR=/abs/path before sbatch, or check the seed range."
    exit 1
fi

CSV="sweep_csv/mem_confirm_seed${SEED}.csv"
PLOT="plots/mem_confirm_seed${SEED}.png"

echo "[seed $SEED] host=$(hostname) started=$(date)"
echo "[seed $SEED] weights=$WEIGHTS -> $CSV  (adapt_b=4 tau_w=400, long run)"

python reservoir_capacity.py \
    --weights "$WEIGHTS" \
    --sweep \
    --sweep-adapt-bs '4' \
    --sweep-tau-ws '400' \
    --sweep-input-scales '8' \
    --sweep-inh-scales '1.0' \
    --sweep-max-delay 35 \
    --sweep-steps 90000 \
    --sweep-sep-steps 12000 \
    --sweep-streams 8 \
    --hold 150 \
    --tau-syn 5.0 \
    --n-state 600 \
    --sweep-csv "$CSV" \
    --plot "$PLOT"

echo "[seed $SEED] finished=$(date)"
