#!/bin/bash
# Long-memory ALIF reservoir, store-and-hold cue task (target 5 s recall).
# One seed per array task.
#   sbatch train_memory_alif_hold.slurm
#
# WHY THIS IS CHEAP (vs the BPTT trainer)
#   No backprop-through-time: the recurrent weights are FIXED and only the linear
#   readout is fit, by closed-form ridge regression. So each seed is a forward
#   simulation plus one linear solve, not a multi-hour gradient run. Memory comes
#   from slow per-neuron adaptation variables (tau_a up to 8 s), not from trained
#   recurrence. Verified locally: N=2000 holds the cue to 5 s at R2~0.997 with a
#   low-activity delay, and R2~0.85 even under strong spiking interference.
#
# ENV NOTE
#   Hex has NO module system. We activate the project-local .venv (numpy + scipy).
#   scipy is only needed if --rec-gain > 0 (sparse recurrence); the default run
#   below uses no recurrence, since the memory does not depend on it.
#   One-time setup on the login node:  bash hex/setup_env.sh  (ensure scipy is in
#   the venv if you plan to enable recurrence).

#SBATCH --partition=cpu-6h
#SBATCH -J alif_hold
#SBATCH -t 01:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-9
#SBATCH -o logs/%x_%A_%a.out
#SBATCH -e logs/%x_%A_%a.err

set -eo pipefail
source /etc/profile 2>/dev/null || true
cd "$SLURM_SUBMIT_DIR"

if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
else
    echo "[fatal] .venv not found. Run 'bash hex/setup_env.sh' on the login node first." >&2
    exit 1
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUNBUFFERED=1

mkdir -p logs results_alif_hold_bg10
SEED=$SLURM_ARRAY_TASK_ID

echo "[seed $SEED] host=$(hostname) started=$(date)"
echo "[seed $SEED] python=$(which python)"

# --background is the retention-vs-activity knob: lower = quieter delay = cleaner
#   hold; ~1.0 = strong ongoing spiking interference (harder, more realistic).
# --tau-a-max is the memory horizon: keep well above the longest delay you score.
python train_memory_alif_hold.py \
    --N             2000 \
    --seed          "$SEED" \
    --dt            1.0  \
    --cue-dur       200  \
    --delays        250,500,1000,2000,3000,5000 \
    --n-train       1500 \
    --n-test        400  \
    --tau-a-min     200  \
    --tau-a-max     8000 \
    --beta-a        1.5  \
    --input-scale   2.0  \
    --background    1.0  \
    --background-spread 0.1 \
    --rec-gain      0.0  \
    --ridge         10.0 \
    --outdir        results_alif_hold_bg10

echo "[seed $SEED] done=$(date)"
