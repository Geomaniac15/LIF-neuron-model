#!/bin/bash
# Job-array parameter sweep for random_neurons_fixed.py on the Lancaster Hex cluster.
# Submit with: sbatch hex/sweep_random_neurons.slurm
#
# Grid: 3 seeds x 4 STDP/drive configs = 12 tasks (--array=0-11).
# Each task is one independent run; results land in results/ keyed by N + seed.
# Bulk sweep runs with --no-spikes --no-plots to stay cheap; re-run winners with
# plots afterwards (see note at the bottom).
#
# Walltime guide (single-threaded): N=2000, steps=20000 -> ~2h, so 04:00:00 is
# comfortable headroom on the cpu-6h partition.

#SBATCH -J lif_ei_sweep
#SBATCH -t 04:00:00
#SBATCH --partition=cpu-6h
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --array=0-11
#SBATCH -o logs/%x_%A_%a.out
#SBATCH -e logs/%x_%A_%a.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs results

# Activate the project venv created by hex/setup_env.sh.
# Temporarily relax -u: some activate scripts reference unset vars.
set +u
source .venv/bin/activate
set -u

export PYTHONUNBUFFERED=1

# ----- parameter grid -----
# Two axes:
#   seed       in {3, 4, 5}
#   config     in {4 combinations of a-minus and the I drive band}
# a-plus is fixed at 0.002; a-minus >= a-plus biases toward depression (the fix).
SEEDS=(3 4 5)

# Each CONFIG entry is "<a-minus> <I-lo> <I-hi>"
CONFIGS=(
    '0.0021 1.1 1.3'
    '0.0022 1.1 1.3'
    '0.0024 1.1 1.3'
    '0.0022 1.2 1.4'
)

N_CONFIGS=${#CONFIGS[@]}      # 4
SEED_IDX=$(( SLURM_ARRAY_TASK_ID / N_CONFIGS ))
CONFIG_IDX=$(( SLURM_ARRAY_TASK_ID % N_CONFIGS ))

SEED=${SEEDS[$SEED_IDX]}
read -r A_MINUS I_LO I_HI <<< "${CONFIGS[$CONFIG_IDX]}"

echo "[job] host=$(hostname) jobid=$SLURM_JOB_ID arrayid=$SLURM_ARRAY_TASK_ID started=$(date)"
echo "[job] seed=$SEED a_minus=$A_MINUS I=[$I_LO,$I_HI]"

python random_neurons_fixed.py \
    --N 2000 \
    --warmup 5000 \
    --steps 20000 \
    --seed "$SEED" \
    --a-plus 0.002 \
    --a-minus "$A_MINUS" \
    --I-lo "$I_LO" \
    --I-hi "$I_HI" \
    --outdir results \

echo "[job] finished=$(date)"
