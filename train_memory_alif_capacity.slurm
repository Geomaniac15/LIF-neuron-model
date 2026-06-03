#!/bin/bash
# Working-memory CAPACITY sweep for the ALIF reservoir: how many items can it hold
# simultaneously? One seed per array task; each runs the full K sweep at N=2000
# under interference (bg=1.0). Closed-form readout, so this is cheap (forward sim
# + ridge solve), not a gradient run.
#   sbatch train_memory_alif_capacity.slurm
#
# n-train is set ABOVE the feature dimension (2N+1 = 4001) so the per-item readouts
# are well-posed; with too few trials the high-K points overfit and go negative.

#SBATCH --partition=cpu-6h
#SBATCH -J alif_cap
#SBATCH -t 05:55:00
#SBATCH --mem=24G
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
    echo "[fatal] .venv not found. Run 'bash hex/setup_env.sh' first." >&2
    exit 1
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUNBUFFERED=1

mkdir -p logs results_alif_capacity
SEED=$SLURM_ARRAY_TASK_ID
echo "[seed $SEED] host=$(hostname) started=$(date)"

# dt=2.0 (validated; halves sim-steps). Slow-only features -> dim N+1=2001, so
# n_train=3000 keeps every per-item readout well-posed. K capped at 64 (capacity
# is already exceeded well before then at N=2000).
python train_memory_alif_capacity.py \
    --N             2000 \
    --seed          "$SEED" \
    --dt            2.0  \
    --cue-dur       200  \
    --delays        500,2000,5000 \
    --K-list        1,2,4,8,16,32,64 \
    --n-train       3000 \
    --n-test        800  \
    --tau-a-max     8000 \
    --beta-a        1.5  \
    --input-scale   2.0  \
    --background    1.0  \
    --background-spread 0.1 \
    --rec-gain      0.0  \
    --ridge         50.0 \
    --outdir        results_alif_capacity

echo "[seed $SEED] done=$(date)"
