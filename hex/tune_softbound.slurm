#!/bin/bash
# Soft-bound STDP: drive sweep for the asynchronous-IRREGULAR regime.
# random_neurons_fixed.py on the Lancaster Hex cluster.
# Submit with: sbatch hex/tune_softbound.slurm
#
# Context / why these axes:
#   Soft-bound STDP fixed the runaway streaks and self-stabilises the firing
#   rate near ~25 Hz almost regardless of the weight knobs (w_max_mult, a_minus
#   barely move rate -- verified). What DOES move the regime is the DRIVE:
#   lower mean drive + larger noise pushes the network into the fluctuation-
#   driven, irregular regime (ISI_CV up). A short probe at I_mean=1.15,sigma=5
#   already hit pop_CV~0.05, ISI_CV~0.52 (AI-ish, clean max/mean~2).
#   So this sweep is over I_mean x sigma, NOT the weight knobs.
#
# Grid: 3 I_mean x 3 sigma = 9 tasks (--array=0-8).
# Fixed: g_inh=5.0 conn=0.1 w_total=0.00424 a_plus=0.002 a_minus=0.0024 w_max_mult=2.0
#
# Spikes ON (regime stats + runaway guard need them), plots OFF (triage).
# Short run (steps=4000) is trustworthy now: soft bounds removed the
# length-dependent instability that previously forced long runs.
#
# Target cell: ISI_CV >= 0.5 with pop_CV low and max/mean ~2 (no RUNAWAY).
# Read winner from logs_tune/, then re-run it with plots (drop --no-plots) for
# the raster. NOTE: rate will likely stay ~25 Hz across the grid -- that's the
# self-stabilised operating point, expected, not a failure.
#
# Output folders (separate from all earlier runs):
#     logs    -> logs_tune/
#     results -> results_tune/tune_im{...}_sg{...}/

#SBATCH -J lif_tune_drive
#SBATCH -t 02:00:00
#SBATCH --partition=cpu-6h
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --array=0-8
#SBATCH -o logs_tune/%x_%A_%a.out
#SBATCH -e logs_tune/%x_%A_%a.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
# slurm resolves -o/-e BEFORE this body runs, so logs_tune/ must exist at SUBMIT
# time. First use of logs_tune/: run once by hand before sbatch:
#     mkdir -p logs_tune results_tune
mkdir -p logs_tune results_tune

# Relax -u around venv activation (some activate scripts touch unset vars).
set +u
source .venv/bin/activate
set -u

export PYTHONUNBUFFERED=1

# ----- grid: I_mean x sigma -----
I_MEANS=(1.1 1.25 1.4)
SIGMAS=(3.0 5.0 7.0)

N_SG=${#SIGMAS[@]}                           # 3
IM_IDX=$(( SLURM_ARRAY_TASK_ID / N_SG ))
SG_IDX=$(( SLURM_ARRAY_TASK_ID % N_SG ))

I_MEAN=${I_MEANS[$IM_IDX]}
SIGMA=${SIGMAS[$SG_IDX]}

OUTDIR="results_tune/tune_im${I_MEAN}_sg${SIGMA}"
mkdir -p "$OUTDIR"

echo "[job] host=$(hostname) jobid=$SLURM_JOB_ID arrayid=$SLURM_ARRAY_TASK_ID started=$(date)"
echo "[job] I_mean=$I_MEAN sigma=$SIGMA  (locked: g_inh=5.0 conn=0.1 w_total=0.00424 a_plus=0.002 a_minus=0.0024 w_max_mult=2.0)  outdir=$OUTDIR"

python random_neurons_fixed.py \
    --N 2000 \
    --warmup 3000 \
    --steps 4000 \
    --seed 0 \
    --I-mean "$I_MEAN" \
    --sigma "$SIGMA" \
    --g-inh 5.0 \
    --connectivity 0.1 \
    --w-total 0.00424 \
    --a-plus 0.002 \
    --a-minus 0.0024 \
    --w-max-mult 2.0 \
    --outdir "$OUTDIR"

echo "[job] finished=$(date)"
