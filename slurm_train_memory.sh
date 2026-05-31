#!/bin/bash
# Train E-I LIF network with R-STDP for long memory. A5000 partition, 10 seeds.
#
# Usage:  sbatch slurm_train_memory.sh
#
# After all tasks complete, aggregate with:
#   python aggregate_memory_sweep.py 'sweep_csv/mem_rstdp_seed*.csv'
#
# Local sanity check before submitting (N=200, ~5 min on CPU):
#   python train_memory_rstdp.py --N 200 --steps 2000 --seed 0 \
#       --target-delay 5 --outdir results_mem_test
#   Expected: error reduction >10%, calibrated I_mean printed,
#             approx_rate in 5-25 Hz range throughout.
#
# Parameter notes:
#   --steps 10000   = 10000 reservoir-steps = 10000*150*0.1ms = 150s of simulated
#                     biology per seed. Enough for R-STDP to shape W at 495ms delay.
#   --target-delay 33  = 33 * 150 * 0.1ms = 495ms
#   --tau-elig 500  must be >= target_delay*hold*dt (495ms) so eligibility
#                   traces survive until d(t) modulates them.
#   --tau-syn 20    NMDA-like; matches what reservoir_capacity.py will use.
#   --tau-m-max 200 wide heterogeneous membrane time constants = free memory.
#
#SBATCH --partition=a5000-48h
#SBATCH -J lif_rstdp_mem
#SBATCH -t 12:00:00
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --array=0-9
#SBATCH -o logs/%x_%A_%a.out
#SBATCH -e logs/%x_%A_%a.err

source /etc/profile
module add miniforge
source activate "$global_storage/lif_env"
cd "$SLURM_SUBMIT_DIR"

mkdir -p logs results_mem sweep_csv plots

SEED=$SLURM_ARRAY_TASK_ID
echo "[seed $SEED] host=$(hostname) started=$(date)" >&2

python train_memory_rstdp.py \
    --N            2000      \
    --seed         $SEED     \
    --warmup       5000      \
    --steps        10000     \
    --dt           0.1       \
    --connectivity 0.1       \
    --w-total      0.042     \
    --w-max-mult   4.0       \
    --I-mean       2.0       \
    --sigma        5.0       \
    --g-inh        5.0       \
    --tau-m-min    20.0      \
    --tau-m-max    200.0     \
    --tau-syn      20.0      \
    --tau-stdp     20.0      \
    --tau-elig     500.0     \
    --eta-rstdp    5e-5      \
    --eta-readout  1e-3      \
    --target-delay 33        \
    --hold         150       \
    --n-input      100       \
    --input-scale  8.0       \
    --n-readout    600       \
    --report-every 500       \
    --outdir       results_mem

echo "[seed $SEED] training done=$(date)" >&2

# -- benchmark immediately after training --
python reservoir_capacity.py \
    --weights      "results_mem/run_N2000_seed${SEED}.npz" \
    --sweep                        \
    --sweep-adapt-bs    '0'        \
    --sweep-tau-ws      '400'      \
    --sweep-adapt-b2s   '0'        \
    --sweep-tau-w2s     '1000'     \
    --sweep-input-scales '8'       \
    --sweep-inh-scales  '1.0'      \
    --tau-state         20.0       \
    --tau-syn           20.0       \
    --hold              150        \
    --sweep-max-delay   60         \
    --sweep-steps       36000      \
    --sweep-sep-steps   8000       \
    --sweep-streams     6          \
    --n-state           600        \
    --sweep-csv "sweep_csv/mem_rstdp_seed${SEED}.csv" \
    --plot      "plots/mem_rstdp_seed${SEED}.png"

echo "[seed $SEED] benchmark done=$(date)" >&2