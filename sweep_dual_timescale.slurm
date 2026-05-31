#!/bin/bash
# Confirm (across all 10 seeds) whether a SECOND slow timescale breaks the ~30 ms
# memory ceiling, or whether the ceiling is structural (set by the frozen weights).
# Submit with: sbatch sweep_dual_timescale.slurm
#
# Context:
#   Single-channel adaptation lifted recall from ~10 ms to a solid ~30 ms, then
#   plateaued. Local (seed0) tests of three downstream levers all FAILED to push
#   the deepest recall ('reach') past 30 ms:
#     - slow adaptation channel (adapt_b2, tau_w2 ~1-1.5 s): no gain
#     - slow readout trace (large tau_state): worse (blurs per-step inputs)
#     - dual fast+slow readout (tau_state2): worse
#   Each raised separation but lost delay-specific recall. The likely reason: slow
#   variables carry aggregate/common-mode history, not the identity of a specific
#   past input, and the input-specific signal is gone from the recurrent state
#   after ~2 reservoir-steps because the trained weights are strongly contracting.
#
#   This run is the honest cross-seed check before concluding. It turns on the dual
#   readout (tau_state2=400 ms, fast trace still 20 ms) AND sweeps the slow
#   adaptation channel, giving the slow timescales their best combined shot. The
#   metric to watch is REACH (deepest delay still recalled, non-contiguous), not
#   just the contiguous horizon. max_delay=60 probes up to 900 ms.
#
# Fixed: adapt_b=4, tau_w=400 ms (the proven fast channel), input_scale=8,
#        inh_scale=1.0, tau_syn=5 ms, hold=150 (15 ms/step), tau_state2=400 ms.
# Swept per task: adapt_b2 in {0,1,2,3} x tau_w2 in {800,1500,3000} ms (12 cells).
# Array: one task per trained seed, --array=0-9 -> run_N2000_seed0..9.npz.
#   Aggregate afterwards with:
#       python aggregate_memory_sweep.py 'sweep_csv/mem_dual_seed*.csv'
#   (auto-detects adapt_b2 / tau_w2 as the varying axes). If no cell beats ~30 ms
#   reach on any seed, the ceiling is structural -> the next move is retraining the
#   weights for memory or adding explicit delay/feedback, NOT more time constants.
#
# WEIGHTS_DIR defaults to results_fixed/ here; override at submit time:
#   WEIGHTS_DIR=/abs/path sbatch sweep_dual_timescale.slurm

#SBATCH --partition=cpu-6h
#SBATCH -J lif_dual_ts
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

CSV="sweep_csv/mem_dual_seed${SEED}.csv"
PLOT="plots/mem_dual_seed${SEED}.png"

echo "[seed $SEED] host=$(hostname) started=$(date)"
echo "[seed $SEED] weights=$WEIGHTS -> $CSV  (dual readout + slow-channel sweep)"

python reservoir_capacity.py \
    --weights "$WEIGHTS" \
    --sweep \
    --sweep-adapt-bs '4' \
    --sweep-tau-ws '400' \
    --sweep-adapt-b2s '0,1,2,3' \
    --sweep-tau-w2s '800,1500,3000' \
    --sweep-input-scales '8' \
    --sweep-inh-scales '1.0' \
    --tau-state 20.0 \
    --tau-state2 400.0 \
    --sweep-max-delay 60 \
    --sweep-steps 36000 \
    --sweep-sep-steps 8000 \
    --sweep-streams 6 \
    --hold 150 \
    --tau-syn 5.0 \
    --n-state 600 \
    --sweep-csv "$CSV" \
    --plot "$PLOT"

echo "[seed $SEED] finished=$(date)"
