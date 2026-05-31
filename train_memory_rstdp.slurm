#!/bin/bash
# Train the E-I LIF reservoir for long memory with R-STDP, one seed per array
# task, then benchmark each trained reservoir. Submit with:
#   sbatch train_memory_rstdp.slurm
#
# What this tests:
#   The ~30 ms recall ceiling is structural: it is set by the FROZEN random
#   contracting recurrent weights. R-STDP trains those weights, so this is the
#   honest test of whether learning the recurrence pushes the ceiling past the
#   passive ~30 ms (2 reservoir-steps) limit.
#
#   Each task: (1) train_memory_rstdp.py builds + trains run_N2000_seed${SEED}.npz
#   with adaptation ON (adapt_b=4, tau_w=400 -> baseline memory for R-STDP to
#   shape) and an RLS/FORCE readout (so the error teaching signal d(t) is real).
#   (2) reservoir_capacity.py scores the trained weights into a per-seed CSV.
#
#   target_delay=2 (30 ms) sits right at the known ceiling: the place to see if
#   training extends it. Bump --target-delay later to chase deeper recall.
#
# Aggregate the 10 benchmark CSVs afterwards with:
#   python aggregate_memory_sweep.py 'sweep_csv/mem_rstdp_seed*.csv'
#   Compare reach/horizon against the random-weight baseline. If trained reach
#   beats ~30 ms, R-STDP moved the ceiling. If not, the next lever is surrogate-
#   gradient BPTT or output-feedback (FORCE) training of the recurrence.
#
# Runtime note: the dense (N,N) eligibility update runs every sim step, so N=2000
# x 10000 reservoir-steps is the expensive part (~1-2 h/seed). 5 h is ample.

#SBATCH --partition=cpu-6h
#SBATCH -J lif_rstdp_train
#SBATCH -t 5:00:00
#SBATCH --mem=16G
#SBATCH --array=0-9
#SBATCH -o logs/%x_%A_%a.out
#SBATCH -e logs/%x_%A_%a.err

source /etc/profile
module add miniforge
source activate "$global_storage/lif_env"

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs results_mem sweep_csv plots

SEED=$SLURM_ARRAY_TASK_ID
WEIGHTS="results_mem/run_N2000_seed${SEED}.npz"
CSV="sweep_csv/mem_rstdp_seed${SEED}.csv"
PLOT="plots/mem_rstdp_seed${SEED}.png"

echo "[seed $SEED] host=$(hostname) started=$(date)"

# ----- 1. train the reservoir with R-STDP -----
echo "[seed $SEED] training -> $WEIGHTS"
python train_memory_rstdp.py \
    --N 2000 \
    --steps 10000 \
    --seed "$SEED" \
    --target-delay 2 \
    --adapt-b 4 \
    --tau-w 400 \
    --tau-syn 20 \
    --hold 150 \
    --input-scale 8 \
    --n-readout 600 \
    --outdir results_mem \
    --no-plots

if [ ! -f "$WEIGHTS" ]; then
    echo "[fatal] training did not produce $WEIGHTS"
    exit 1
fi

# ----- 2. benchmark the trained reservoir (adaptation matched to training) -----
echo "[seed $SEED] benchmarking -> $CSV"
python reservoir_capacity.py \
    --weights "$WEIGHTS" \
    --tau-syn 20 \
    --hold 150 \
    --tau-state 20 \
    --n-state 600 \
    --sweep \
    --sweep-adapt-bs '4' \
    --sweep-tau-ws '400' \
    --sweep-input-scales '8' \
    --sweep-inh-scales '1.0' \
    --sweep-max-delay 60 \
    --sweep-steps 36000 \
    --sweep-sep-steps 8000 \
    --sweep-streams 6 \
    --sweep-csv "$CSV" \
    --plot "$PLOT"

echo "[seed $SEED] finished=$(date)"
