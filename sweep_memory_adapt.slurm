#!/bin/bash
# Map the long-memory landscape of the trained LIF reservoir on the hex cluster.
# reservoir_capacity.py intrinsic benchmark (synthetic input, no recorded data).
# Submit with: sbatch sweep_memory_adapt.slurm
#
# Why this sweep:
#   The trained N=2000 brain is a healthy *short*-memory reservoir: its recall
#   horizon is bounded by its fastest relaxation (tau_syn=5 ms, membrane ~20 ms),
#   so the linear readout can only reconstruct input from ~1-2 reservoir-steps
#   back (~15-30 ms). Raising tau_syn does NOT help: it blurs successive inputs
#   and trades memory for separation (verified locally: tau_syn 5->80 ms dropped
#   total MC 0.86->0.31 and the horizon collapsed to 0 ms).
#
#   The real long-memory lever is SPIKE-TRIGGERED ADAPTATION: a slow per-neuron
#   current w_adapt that each spike bumps by --adapt-b and that decays with
#   --tau-w (100-400 ms). It adds a long echo ORTHOGONAL to the fast synapse, so
#   it lengthens memory without smearing the input. Local smoke test on N2000
#   seed0 (input_scale=8): adapt_b 0->4 lifted total MC 0.35->0.89 and the recall
#   horizon 0->30 ms, moving the net from washed-out into FADING MEMORY.
#   This grid pushes adapt_b/tau_w further and checks it holds across all 10 seeds.
#
# Grid (per task): adapt_b x tau_w, run inside reservoir_capacity.py --sweep.
#   adapt_b: 0 (control) .. 8. tau_w: 100, 200, 400 ms.
#   input_scale fixed at 8 (where input actually registers), inh_scale=1.0.
# Array: one task per trained seed, --array=0-9 -> run_N2000_seed0..9.npz.
#
# Read the result: each task prints a table and writes a CSV to
#   sweep_csv/mem_adapt_seed<seed>.csv. Want the largest horizon_ms whose regime
#   stays FADING MEMORY / CRITICAL-ish (not CHAOTIC), consistent across seeds.
#
# WEIGHTS_DIR defaults to results_fixed/ here; override at submit time:
#   WEIGHTS_DIR=/abs/path sbatch sweep_memory_adapt.slurm

#SBATCH --partition=cpu-6h
#SBATCH -J lif_mem_adapt
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

CSV="sweep_csv/mem_adapt_seed${SEED}.csv"
PLOT="plots/mem_adapt_seed${SEED}.png"

echo "[seed $SEED] host=$(hostname) started=$(date)"
echo "[seed $SEED] weights=$WEIGHTS -> $CSV"

# Full adaptation grid runs inside one --sweep call.
#   --sweep-max-delay 25 so a long horizon (up to 25*15ms = 375ms) isn't truncated.
#   --steps via --sweep-steps kept modest; raise if MC looks noisy.
python reservoir_capacity.py \
    --weights "$WEIGHTS" \
    --sweep \
    --sweep-adapt-bs '0,2,4,6,8' \
    --sweep-tau-ws '100,200,400' \
    --sweep-input-scales '8' \
    --sweep-inh-scales '1.0' \
    --sweep-max-delay 25 \
    --sweep-steps 24000 \
    --sweep-sep-steps 8000 \
    --sweep-streams 6 \
    --hold 150 \
    --tau-syn 5.0 \
    --n-state 600 \
    --sweep-csv "$CSV" \
    --plot "$PLOT"

echo "[seed $SEED] finished=$(date)"
