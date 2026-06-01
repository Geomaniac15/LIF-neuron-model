#!/bin/bash
# Train the E-I LIF reservoir for delayed-recall memory with surrogate-gradient
# BPTT, CURRENT-BASED synapse variant. One seed per array task.
#   sbatch train_memory_bptt_current.slurm
#
# WHY THIS SCRIPT EXISTS
#   The conductance-synapse BPTT trainer was mathematically correct but its
#   through-time adjoint diverged exponentially (the per-step adjoint norm just
#   filled whatever clip ceiling it was given), so the gradient reaching W was
#   clipped noise and recall MSE never left chance (1/3). See
#   diagnostics/bptt_rootcause.png.
#
#   train_memory_bptt_current.py fixes that with two changes:
#     1. current-based synapses  Isyn = g_e*ge - g_i*gi   (no V dependence), and
#     2. detach-reset in the backward pass (on by default),
#   which together bound the adjoint. Verified: the adjoint norm stays flat
#   (~hundreds) regardless of the clip ceiling, and recall MSE now drops below
#   chance, which the old trainer never did.
#
# ENV NOTE (important)
#   Hex has NO module system and NO $global_storage (see hex/setup_env.sh). We
#   activate the project-local .venv instead. The old `module add miniforge`
#   block in the other slurm scripts is what produced the
#   "module: command not found" failures and killed the rstdp run.
#   One-time setup on the login node:   bash hex/setup_env.sh
#
# BENCHMARK NOTE (important)
#   This network uses CURRENT-BASED synapses, so reservoir_capacity.py (which is
#   conductance-based) will NOT measure the saved W correctly. The benchmark
#   step is intentionally left out below. Port the benchmark's simulate() to
#   current-based synapses first, then score results_mem_current/*.npz.
#
# FIRING-RATE CALIBRATION NOTE
#   The trainer now AUTO-CALIBRATES the background drive before training: it
#   bisects the mean I_ext on a frozen-W forward pass so the settled (adapted)
#   steady-state population rate lands in [--calib-rate-lo, --calib-rate-hi]
#   (default 10-20 Hz). This replaces the old by-hand --background-mean scan and
#   handles the N-dependence automatically (verified: N=300 -> mean 3.75, N=2000
#   -> mean 4.45, both ~15 Hz on the settled tail). --background-mean is now only
#   the initial guess; only its per-neuron SPREAD survives calibration.
#   The tail-only measurement (--calib-warm-frac) is essential: averaging over
#   the adaptation transient overestimates the rate and the network goes silent
#   a few hundred ms into training. Pass --no-calibrate to opt out.

#SBATCH --partition=cpu-6h
#SBATCH -J lif_bptt_current
#SBATCH -t 05:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-9
#SBATCH -o logs/%x_%A_%a.out
#SBATCH -e logs/%x_%A_%a.err

set -eo pipefail
source /etc/profile 2>/dev/null || true

cd "$SLURM_SUBMIT_DIR"

# --- activate the project venv (NOT the module system) ---
if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
else
    echo "[fatal] .venv not found. Run 'bash hex/setup_env.sh' on the login node first." >&2
    exit 1
fi

# let numpy's BLAS use the cores we asked for (the dense W@adjoint matmul)
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

mkdir -p logs results_mem_current

SEED=$SLURM_ARRAY_TASK_ID
WEIGHTS="results_mem_current/run_N2000_seed${SEED}.npz"

echo "[seed $SEED] host=$(hostname) started=$(date)"
echo "[seed $SEED] python=$(which python)  numpy=$(python -c 'import numpy;print(numpy.__version__)')"
echo "[seed $SEED] training -> $WEIGHTS"

# --- train ---
# Start with delay=2 (30 ms) and confirm it holds clearly below chance, THEN
# raise --target-delay to chase deeper recall. --chunk must exceed target_delay
# so the recall target lies inside the truncated-BPTT window.
python train_memory_bptt_current.py \
    --N             2000   \
    --seed          "$SEED" \
    --steps         8000   \
    --target-delay  2      \
    --chunk         4      \
    --hold          150    \
    --dt            0.1    \
    --tau-syn       5.0    \
    --syn-gain-exc  60.0   \
    --syn-gain-inh  15.0   \
    --adapt-b       4.0    \
    --tau-w         400.0  \
    --background-mean 3.0  \
    --input-scale   8.0    \
    --n-input       100    \
    --n-readout     600    \
    --lr-w          1e-4   \
    --lr-out        1e-3   \
    --clip          1.0    \
    --detach-reset         \
    --washout       20     \
    --report-every  200    \
    --outdir        results_mem_current

if [ ! -f "$WEIGHTS" ]; then
    echo "[fatal] training did not produce $WEIGHTS" >&2
    exit 1
fi

echo "[seed $SEED] training done=$(date)"
echo "[seed $SEED] NOTE: benchmark skipped - port reservoir_capacity.py to current-based synapses before scoring $WEIGHTS"
