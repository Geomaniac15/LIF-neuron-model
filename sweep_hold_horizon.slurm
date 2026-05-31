#!/bin/bash
# Test whether a slower reservoir step (bigger --hold) extends the recall horizon
# in ms, at the chosen adaptation setting. Submit with: sbatch sweep_hold_horizon.slurm
#
# Why this sweep:
#   At adapt_b=4 / tau_w=400 the recall horizon plateaus at ~30 ms (~2 reservoir-
#   steps of 15 ms each) no matter how deep we probe. The remaining lever is the
#   reservoir-step duration itself: --hold sets ms-per-step (hold * dt). A bigger
#   hold makes each remembered input occupy more wall-clock time, so the SAME ~2
#   step memory could span more ms, BUT it also coarsens resolution and holds each
#   input long enough to blur into the network's own dynamics. A quick local probe
#   already hinted bigger hold may not help (hold 400 dropped the horizon to 0 ms);
#   this run settles it properly across all 10 seeds with constant statistics.
#
#   reservoir_capacity.py keeps the reservoir-STEP COUNT constant across holds
#   (scaling total sim steps), so MC estimates stay comparable; only ms-per-step
#   changes. max_delay=20 steps covers up to 20 * hold * dt ms (800 ms at hold=400).
#
# Fixed cell: adapt_b=4, tau_w=400 ms, input_scale=8, inh_scale=1.0, tau_syn=5 ms.
# Hold grid: 150 (15 ms/step), 250 (25 ms/step), 400 (40 ms/step).
# Array: one task per trained seed, --array=0-9 -> run_N2000_seed0..9.npz.
#   Aggregate afterwards with:
#       python aggregate_memory_sweep.py 'sweep_csv/mem_hold_seed*.csv'
#   (it auto-detects that 'hold' is the varying axis and groups by it.)
#
# WEIGHTS_DIR defaults to results_fixed/ here; override at submit time:
#   WEIGHTS_DIR=/abs/path sbatch sweep_hold_horizon.slurm

#SBATCH --partition=cpu-6h
#SBATCH -J lif_hold_horizon
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

CSV="sweep_csv/mem_hold_seed${SEED}.csv"
PLOT="plots/mem_hold_seed${SEED}.png"

echo "[seed $SEED] host=$(hostname) started=$(date)"
echo "[seed $SEED] weights=$WEIGHTS -> $CSV  (adapt_b=4 tau_w=400, hold sweep)"

# --sweep-steps is the reservoir-step budget for the FIRST hold (24000/150 = 160
# reservoir-steps); the script scales the others so every hold gets ~160 steps.
python reservoir_capacity.py \
    --weights "$WEIGHTS" \
    --sweep \
    --sweep-adapt-bs '4' \
    --sweep-tau-ws '400' \
    --sweep-input-scales '8' \
    --sweep-inh-scales '1.0' \
    --sweep-holds '150,250,400' \
    --sweep-max-delay 20 \
    --sweep-steps 24000 \
    --sweep-sep-steps 6000 \
    --sweep-streams 6 \
    --tau-syn 5.0 \
    --n-state 600 \
    --sweep-csv "$CSV" \
    --plot "$PLOT"

echo "[seed $SEED] finished=$(date)"
