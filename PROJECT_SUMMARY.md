# Project Summary: Computational Neuroscience from Scratch

A from-scratch (NumPy only, no ML frameworks) exploration of spiking neural
networks, associative memory, and the question of how a biological-style network
can actually hold information over time. The work splits into two arcs: a
**foundations arc** (single neuron up to Modern Hopfield / MNIST, documented in
`README.md`) and a more recent **spiking-memory research arc** (May to June 2026)
that is the real frontier of the project and is not yet written up anywhere else.

This document explains every component and outlines all results.

---

## Part 1: Foundations (single neuron to Hopfield memory)

The build follows one continuous arc:

```
Single neuron -> Synaptic circuits -> E-I networks -> Hodgkin-Huxley
   -> STDP learning -> Hopfield memory -> Modern Hopfield -> MNIST retrieval
```

### Spiking neuron models

`neuron_class.py` is a single leaky integrate-and-fire (LIF) neuron implementing
`tau_m dV/dt = -(V - V_rest) + R*I(t)`, with refractory period and exponentially
decaying synaptic current. It produces regular spiking at ~36 Hz (28 ms ISI) and
a verified critical current threshold of 1.5 nA below which it never fires.

`two_neurons.py` connects two LIF neurons with a one-way synapse. Neuron B fires
~10 ms after each A spike, confirming causal synaptic transmission through an
exponential-decay current kernel.

`random_neurons.py` is a 100-neuron E-I network (80 excitatory / 20 inhibitory,
5% sparse connectivity, conductance-based synapses, heterogeneous time constants,
STDP). It shows emergent synchrony, asynchronous irregular firing, and ~15 Hz E-I
oscillations depending on parameters, mirroring the transition between epileptic
and healthy cortical activity.

`hodgkin_huxley_model.py` is a full Hodgkin-Huxley neuron with voltage-gated Na+
and K+ channels via gating variables m, h, n. There is no manual spike reset; the
channel physics generates the action potential shape: sharp upstroke to +40 mV,
rapid downstroke, hyperpolarisation undershoot to -75 mV, and a natural refractory
period. Spikes peak at +40 mV with ~3 ms width.

`hh_model_with_stdp.py` is a feedforward HH network with STDP-trained associative
memory: 20 input neurons, 10 output neurons with winner-take-all lateral
inhibition. After 100 training trials it correctly identifies noisy versions of
pattern A (20% bit flips) and maps distinct patterns to distinct output neurons.

### Hopfield networks

`hopfield_network.py` / `hopfield_with_softmax.py` implement both the classic
binary Hopfield network (Hebbian weights, sign update, energy minima) and the
continuous / Modern Hopfield network (softmax update). Key results:

- Binary network: perfect retrieval up to 30% corruption; capacity limit verified
  empirically at ~0.14N (perfect to 10 patterns at N=100, collapses by 20).
- Energy `E = -1/2 s^T W s` decreases monotonically during retrieval.
- Continuous network: capacity jumps from 14 to 500+ patterns, degrading
  gracefully; optimal temperature beta ~2 to 3 for random +/-1 patterns.
- The continuous update is shown to be mathematically identical to transformer
  attention `softmax(QK^T / sqrt(d)) V`, which motivates the 1/sqrt(d) scaling.

`mnist_hopfield.py` applies the Modern Hopfield network to MNIST digit completion
(top-k attention, prototype memories, momentum update). With 50 prototype memories
(5 per digit) at 30% noise: **57.2% digit accuracy, 89.6% pixel overlap**; digits
0/1/7 most reliable, 2/3/8 most confused. Full write-up with confusion matrices,
reconstructions and PCA retrieval trajectories is in
`MNIST_Hopfield Tests/hopfield_mnist_report.html`.

### Foundations results table

| Model | Key result |
|---|---|
| Single LIF neuron | 36 Hz spiking, 1.5 nA critical current |
| Two neurons | Causal transmission, B fires 10 ms after A |
| 100-neuron E-I network | Emergent 15 Hz oscillations, synchrony transition |
| Hodgkin-Huxley | Realistic action potentials from channel kinetics |
| STDP network | Patterns A/B separated after 100 trials |
| Binary Hopfield | 0.14N capacity verified |
| Continuous Hopfield | 500+ patterns vs 14 for binary |
| MNIST Hopfield | 57% accuracy, 90% pixel overlap at 30% noise |

---

## Part 2: The spiking-memory research arc (the recent work)

The driving question: **can a biologically plausible spiking network hold a
stimulus in memory over time, and where does that memory actually come from?**
This arc moved through three distinct attempts and landed on a clear answer.

### 2a. A realistic, tunable reservoir + a way to measure its memory

`random_neurons_fixed.py` (the cluster-ready refactor of `random_neurons.py`)
fixes a sign bug in the conductance synaptic current (excitatory synapses were
silently hyperpolarising), rescales weights accordingly, and adds a noisy
fluctuation-driven external drive to reach the asynchronous-irregular (AI) regime
of Brunel 2000. It self-reports `pop_CV` (population synchrony) and `ISI_CV`
(firing irregularity). Realistic outcome: low pop_CV with ISI_CV ~0.5 to 0.6, a
solid AI baseline (true Poisson ISI_CV ~1.0 is capped by the 2 ms refractory
floor without structural changes).

`reservoir_capacity.py` benchmarks any trained network as a reservoir using three
standard measures: **memory capacity** (Jaeger 2001: R^2 of a linear readout
reconstructing the input delayed by k steps; total MC = sum over k), **perturbation
divergence** (edge-of-chaos / fading-memory test), and **separation** (Legenstein
& Maass). Parameter tuning was run as SLURM sweeps on the Lancaster HEC cluster
(`hec/`, `hex/`, `results_tune/`, `results_ai/`), sweeping input-mean and noise
sigma to find non-chaotic, high-separation operating points.

**Finding (the problem statement for everything after):** the frozen random
reservoir holds almost no linearly decodable memory past ~30 ms (about 2
reservoir-steps). This is a structural ceiling set by the random contracting
recurrent weights. The memory-adaptation sweep across spike-frequency adaptation
(`adapt_b`) and adaptation timescale (`tau_w`) confirms it:

| adapt_b | tau_w | recall horizon (ms) | total MC | chaotic seeds |
|---|---|---|---|---|
| 0 | any | 10.5 +/- 13.5 | 0.50 | 0/10 |
| 4 | 400 | 33.0 +/- 6.0 | 0.86 | 0/10 |
| 6 | 100 | 33.0 +/- 9.0 | 0.88 | 0/10 |
| 8 | 100 | 30.0 +/- 6.7 | 0.90 | 0/10 |

Even the best adaptation setting tops out around a **33 ms** recall horizon.
Adaptation helps but does not buy real long memory.

### 2b. Trying to train memory into the weights

**Reward-modulated STDP** (`train_memory_rstdp.py`): a biologically grounded
attempt where a per-synapse eligibility trace is gated by a global neuromodulatory
error signal `d(t)` (the readout's reconstruction error acting as a dopamine-like
teaching signal). It could not push past the ceiling, because with no decodable
memory there is no teaching signal to bootstrap from.

**Surrogate-gradient BPTT** (`train_memory_bptt.py`): computes a real gradient of
a delayed-recall loss through time, using a fast-sigmoid surrogate derivative for
the spike (SuperSpike / Zenke & Ganguli 2018), while keeping the forward pass
byte-identical to the benchmark so trained weights transfer. The conductance
version is mathematically correct (finite-difference gradient check passes at
~1e-10) but its through-time adjoint is **exponentially divergent**: the
voltage-dependent synaptic current introduces a self-amplifying `(V - E)` factor
(~65 to 80 mV) every step, so the gradient reaching W is clipped noise and recall
MSE never leaves chance (1/3). Root cause documented in
`diagnostics/bptt_rootcause.png`.

**The fix** (`train_memory_bptt_current.py`): switch the trained model to
**current-based synapses** (`Isyn = g_e*ge - g_i*gi`, no voltage dependence),
which removes the self-feedback term and keeps the backward recursion bounded.
This works numerically (best run `results_mem_bptt_v7`, 50 ms synapses, reached
final MSE ~0.32 versus the 0.33 chance floor at target delay 2), but it remained a
short-memory, hard-to-stabilise result and did not scale to long horizons. The
through-time approach keeps drifting into unstable recurrent reverberation
(saturation / silence bursting).

### 2c. The answer: long memory lives in slow state variables, not reverberation

The key insight that resolved the arc: long memory should not come from recurrent
reverberation (the unstable, short-lived kind BPTT kept finding). It should come
from **slow state variables whose time constant matches the lag you want to hold.**

`train_memory_alif_hold.py` implements this with an adaptive-threshold LIF
(ALIF / LSNN-style) reservoir. Every neuron has an adaptive firing threshold that
decays with its own time constant `tau_a`, drawn log-uniformly up to several
seconds (default max 8000 ms). A brief cue loads these slow variables; during the
quiet delay they decay as `exp(-delay/tau_a)`, so neurons with `tau_a >> delay`
still carry the cue at recall. The recurrent weights are fixed; only a linear
readout is fit by closed-form ridge regression (so it scales to 5 s, which full
BPTT cannot tape).

**Store-and-hold result (N=2000, 10 seeds):** near-perfect recall of a scalar cue
all the way out to **5 seconds**.

| delay (ms) | 250 | 500 | 1000 | 2000 | 3000 | 5000 |
|---|---|---|---|---|---|---|
| mean R^2 | 0.9985 | 0.9983 | 0.9981 | 0.9978 | 0.9977 | 0.9975 |
| min R^2 | 0.9975 | 0.9973 | 0.9971 | 0.9969 | 0.9967 | 0.9965 |

This is the headline result of the project: a roughly **150x improvement** in
memory horizon over the best frozen/adaptation reservoir (33 ms to 5000 ms), from
a principled change in mechanism rather than more tuning.

`train_memory_alif_capacity.py` then asks how many items the same network can hold
at once. K scalar cues are presented simultaneously through random input
projections (per-item drive normalised by 1/sqrt(K) so the firing regime stays
fixed), and one readout per item decodes from the shared population state.

**Working-memory capacity (N=2000, 10 seeds), mean R^2:**

| K items | 500 ms | 2000 ms | 5000 ms |
|---|---|---|---|
| 1 | 0.991 | 0.967 | 0.863 |
| 2 | 0.990 | 0.965 | 0.869 |
| 4 | 0.985 | 0.971 | 0.878 |
| 8 | 0.976 | 0.953 | 0.840 |
| 16 | 0.945 | 0.909 | 0.739 |
| 32 | 0.894 | 0.814 | 0.397 |
| 64 | 0.799 | 0.634 | -0.058 |

The network holds roughly 8 to 16 items well at short-to-medium delays and
degrades gracefully with both item count and delay as the random projections start
to interfere. This is the same qualitative shape as the human 'about four items'
working-memory limit, and it is interference-driven (not the network being driven
harder), because per-item drive is held constant.

---

## Supporting infrastructure and side pieces

- **`diagnostics/`**: the debugging trail for the BPTT failure and the ALIF
  results, e.g. `bptt_rootcause.png`, `bptt_diagnosis.png`,
  `alif_capacity_final.png`, `alif_capacity_vs_N.png`, plus standalone diagnosis
  scripts (`diag_bptt.py`, `diag_current.py`, `diag_adjclip.py`, `diag_drive.py`).
- **Cluster scripts** (`hec/`, `hex/`, `*.slurm`, `slurm_train_memory.sh`):
  SLURM job definitions for the Lancaster HEC, covering the AI-regime tuning
  sweeps, memory-adaptation sweeps, dual-timescale and hold-horizon sweeps, and
  the ALIF training runs.
- **`aggregate_memory_sweep.py`**: aggregates per-seed sweep CSVs (in
  `sweep_csv/`) into the recall-horizon / capacity tables shown above, auto-
  detecting which parameters vary and flagging chaotic cells.
- **`neuro-portfolio/`** and **`neuro-portfolio.jsx`**: a React + Vite portfolio
  site presenting the work.
- **`dirichlet_vs_nuemann.py`**: a small standalone PDE boundary-condition
  comparison (Dirichlet vs Neumann), separate from the main arc.
- **Output folders**: `graphs/`, `plots/`, `results*/`, `logs*/`, and the various
  `capacity_*.png` / `sweep_*.png` figures hold the generated rasters, weight
  matrices, diagnostics, and capacity curves.

---

## The story in one paragraph

Starting from a single LIF neuron, the project builds up through biophysics
(Hodgkin-Huxley), learning (STDP), and associative memory (Hopfield, Modern
Hopfield, MNIST, and the link to transformer attention). It then tackles a harder
question: how a spiking network holds information over time. A frozen random
reservoir is shown to remember for only ~30 ms; reward-modulated STDP cannot fix
this; surrogate-gradient BPTT is derailed by an exponentially divergent adjoint
from voltage-dependent synapses (fixed with current-based synapses, but still
short-memory). The resolution is mechanistic rather than computational: putting
memory into slow per-neuron adaptation variables (ALIF) yields near-perfect recall
out to 5 seconds and a human-like working-memory capacity curve over 1 to 64
simultaneous items.

Core references: Dayan & Abbott (Theoretical Neuroscience); Hodgkin & Huxley
(1952); Brunel (2000); Jaeger (2001); Legenstein & Maass; Zenke & Ganguli (2018,
SuperSpike); Ramsauer et al. (2020, Hopfield Networks is All You Need).
