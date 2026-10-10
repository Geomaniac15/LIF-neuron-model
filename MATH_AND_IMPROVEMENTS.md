# The Mathematics, and Everything Improved or Changed

A deep technical companion to `PROJECT_SUMMARY.md`. This document states the
equations as actually implemented in the code, and explains every bug fix,
mechanism change, and design decision, with the reasoning behind each. Notation
follows the source files. Time is discretised with forward Euler at step `dt`
(0.1 ms for the spiking models unless noted).

---

## 1. The base neuron: leaky integrate-and-fire

Every spiking model starts from the LIF membrane equation

```
tau_m dV/dt = -(V - V_rest) + R * I(t)
```

discretised as

```
V <- V + (dt / tau_m) * ( -(V - V_rest) + R * I )
```

A spike is the threshold crossing `V >= V_th`, after which `V` is clamped to
`V_reset` for a refractory window. `I` is the sum of an external drive and the
synaptic current `I_syn`. Everything downstream is a question of what `I_syn`,
`V_th`, and the weights `W` are made to do.

In the biophysical models the units are real (mV, nA): `V_rest = -70`,
`E_exc = 0`, `E_inh = -80`, `V_th ~ -55`. In the ALIF memory model the units are
normalised: `V_rest = 0`, `V_th = 1`, `V_reset = 0`, `R = 1`, which removes mV
bookkeeping and makes the adaptation algebra clean.

---

## 2. Hodgkin-Huxley: spikes from channel physics, no manual reset

The HH neuron replaces the hard threshold-and-reset with conductances that are
themselves dynamical:

```
C_m dV/dt = -g_Na * m^3 * h * (V - E_Na)
            -g_K  * n^4 * (V - E_K)
            -g_L  * (V - E_L) + I
```

Each gating variable `x in {m, h, n}` follows first-order kinetics with
voltage-dependent rates:

```
dx/dt = alpha_x(V) * (1 - x) - beta_x(V) * x
```

The important conceptual point implemented here: there is **no manual spike reset**.
The upstroke (Na+ activation via `m`, positive feedback), the downstroke (Na+
inactivation via `h`, K+ activation via `n`), the hyperpolarising undershoot (K+
overshoot), and the refractory period all emerge from the channel kinetics. The
physics generates the +40 mV peak and ~3 ms width on its own.

---

## 3. The conductance-synapse sign fix (a real bug, corrected)

This is the first substantive correction in the project. Conductance-based
synaptic current should drive `V` toward a reversal potential. The original code
had

```
I_syn = g_exc * (V - E_exc) + g_inh * (V - E_inh)        # WRONG sign
```

With `E_exc = 0` and `V` near `-65`, the term `g_exc * (V - E_exc)` is negative,
so an excitatory spike was **hyperpolarising** its target. The fix flips the sign:

```
I_syn = -g_exc * (V - E_exc) - g_inh_v * (V - E_inh)     # FIXED
```

Now an excitatory event (raising `g_exc`) pushes `V` up toward `E_exc = 0`
(depolarising) and an inhibitory event pushes `V` toward `E_inh = -80`
(hyperpolarising), which is the correct biophysics.

Two consequences had to be cleaned up after fixing the sign:

- **Weight rescaling.** The original peak weight (~0.5) had been silently
  calibrated against the wrong-sign equation. With the correct sign it saturates
  the network, so the default was rescaled by 1/100 to ~0.005.
- **STDP learning-rate reduction.** Once the network actually fires at 15 to 25
  Hz, the old STDP rates (0.01) push weights to the rails, so they were dropped to
  ~0.002.

Synaptic conductances themselves are exponential traces driven by presynaptic
spikes, decaying with `tau_syn`:

```
g <- g - (g / tau_syn) * dt        # between spikes
g <- g + (incoming weights)        # on a presynaptic spike
```

---

## 4. The asynchronous-irregular (AI) regime: three changes

A constant-current LIF neuron fires like a clock (inter-spike-interval
coefficient of variation `ISI_CV -> 0`). Real cortex fires irregularly and
asynchronously (Brunel 2000). Reaching that regime required three coordinated
changes.

### 4.1 Fluctuating white-noise drive with dt-invariant scaling

The external current became a baseline plus white noise:

```
I_ext = I_base + sigma * N(0,1) / sqrt(dt)
```

with `I_base` drawn per neuron uniformly in `[I_mean - I_spread, I_mean + I_spread]`
and the noise redrawn every step. The `1 / sqrt(dt)` factor is the correct
white-noise scaling: it keeps the integrated noise variance over a fixed time
window independent of the step size, so `dt` stays interchangeable and the noise
amplitude is a physical quantity rather than a step-size artefact. Setting
`sigma = 0` recovers the old constant-drive behaviour. Irregular firing
**requires** this fluctuating input; you cannot get `ISI_CV ~ 1` from constant
drive.

### 4.2 Inhibition-dominated balance

The AI state is inhibition-dominated: per synapse, inhibition must be several
times stronger than excitation to track and cancel it, otherwise the population
synchronises into waves. Inhibitory outgoing weights are scaled by `-g_inh`
(with `g_inh ~ 4 to 6`) rather than just `-1`:

```
W[inhibitory rows, :] *= -g_inh
```

### 4.3 Size-invariant (balanced-network) weight scaling

To compare networks of different size `N` at the same operating point, peak weight
scales with the in-degree `K`:

```
w_scale = w_total / sqrt(K)
```

This is the standard balanced-network scaling: total recurrent input variance per
neuron stays O(1) as connectivity grows, so the firing regime is preserved across
`N`. The soft-bound ceiling on `|W|` is scaled by `g_inh` for inhibitory rows so a
single mechanism bounds both excitatory and (larger-magnitude) inhibitory weights.

### 4.4 Self-diagnosing regime statistics

The run reports two scalars so it can grade itself without eyeballing a raster:

- `pop_CV`: coefficient of variation of the binned population spike count. High
  means a global rhythm / synchronous waves.
- `ISI_CV`: mean per-neuron CV of inter-spike intervals. ~1.0 is Poisson-like
  (irregular), ~0 is clock-like.

Target AI regime is **low `pop_CV` and high `ISI_CV`**. Realistically the runs
reach `ISI_CV ~ 0.5 to 0.6`; true Poisson `ISI_CV ~ 1.0` is capped by the 2 ms
refractory floor at these rates and would need structural changes (synaptic
delays, lower rates, tighter E/I balance). This is documented honestly as a good
baseline rather than a final calibration.

---

## 5. STDP and the ordering fix

Spike-timing-dependent plasticity changes a synapse based on the time difference
`delta_t = t_post - t_pre` between post- and pre-synaptic spikes:

- pre before post (`delta_t > 0`): potentiation (LTP), weight grows.
- post before pre (`delta_t < 0`): depression (LTD), weight shrinks.

The change decays exponentially with the timing gap. In the fixed implementation
the update is **multiplicative** (proportional to current weight magnitude) and
masked to existing synapses, with potentiation on the incoming column and
depression on the outgoing row:

```
W[:, i] += A_plus  * decay * |W[:, i]| * sign * (W[:, i] != 0)   # LTP into i
W[i, :] -= A_minus * decay * |W[i, :]| * sign * (W[i, :] != 0)   # LTD out of i
```

Two correctness details were tightened:

- **Update ordering.** `last_spike[i] = t_now` is set **after** the weight update,
  so `delta_t` is computed against the previous spike time and the self-loop term
  is unambiguous.
- **Soft-bound instead of hard clip.** The multiplicative `|W|` factor plus the
  `g_inh`-scaled ceiling keeps weights bounded smoothly, rather than relying on a
  hard `np.clip` to catch a runaway additive rule.

In the HH+STDP associative-memory network, this rule trained a 20->10 feedforward
map with winner-take-all lateral inhibition to separate two patterns after 100
trials, with state reset between trials to prevent interference.

---

## 6. Hopfield networks and the attention identity

### 6.1 Binary Hopfield

Hebbian outer-product weights and a sign update:

```
W_ij = (1/N) * sum_mu  x_i^mu * x_j^mu
s_i  <- sign( sum_j W_ij s_j )
```

Stored patterns are minima of the energy

```
E = -1/2 * s^T W s
```

and each asynchronous update is non-increasing in `E`, so corrupted inputs roll
downhill to the nearest memory. Empirical capacity matches the theoretical
`~0.14 N`.

### 6.2 Modern (continuous) Hopfield and the softmax update

Replacing the sign with a softmax over stored patterns `X` dramatically raises
capacity:

```
s_new = X^T * softmax( beta * X * s )
```

`beta` is an inverse temperature. As `beta -> infinity` this recovers the binary
network. The optimal `beta` depends on the scale of the dot products `X s`, which
is exactly the reasoning behind the `1 / sqrt(d)` scaling in attention.

### 6.3 The identity with transformer attention

Writing the query as the noisy input, the keys as stored patterns, the values as
retrieved content:

```
Attention(Q, K, V) = softmax( Q K^T / sqrt(d) ) V
```

is the same computation as one continuous-Hopfield update. `softmax(Q K^T)`
produces similarity weights; the weighted sum of values is the retrieved memory.
This is the project's conceptual bridge from neuroscience to modern deep learning.

### 6.4 MNIST retrieval refinements

Applying this to MNIST digit completion added three practical mechanisms:

- **Top-k attention:** attend only to the `k` most similar stored patterns per
  step (sparse softmax).
- **Prototype memories:** average several examples per digit class instead of
  storing raw images.
- **Momentum update:** `state <- 0.6 * state + 0.4 * new_state` for stable
  convergence.

The regime insight: with a large pattern library the softmax concentrates on the
single nearest neighbour, making retrieval effectively exact nearest-neighbour
search; with a sparse library the network genuinely blends patterns. Result with
50 prototypes at 30% noise: 57.2% accuracy, 89.6% pixel overlap.

---

## 7. Measuring memory: the reservoir benchmark

Before trying to improve memory, the project defined how to measure it
(`reservoir_capacity.py`), using three standard reservoir-computing quantities.

### 7.1 Memory capacity (Jaeger 2001)

Inject an i.i.d. random scalar stream `u(t)`. For each delay `k`, fit a linear
(Ridge) readout to reconstruct `u(t - k)` from the network state, on held-out
data. The per-delay capacity is the squared correlation between prediction and
truth:

```
MC_k = max(0, corr(prediction, u(t-k))^2)
total MC = sum_k MC_k
```

`total MC` is, loosely, how many past inputs the state still linearly carries. A
**recall horizon** is reported as the largest delay with `MC_k` above a threshold
(0.1).

### 7.2 Perturbation divergence (edge of chaos)

Run the reservoir twice on streams identical except for one perturbed value at
`t = 0`, and track the normalised Euclidean distance between the two state
trajectories:

```
dist(t) = || state_base(t) - state_pert(t) || / sqrt(state_dim)
```

A classifier on the end/peak ratio labels the regime: decays to ~0 is
fading-memory / echo-state (ordered, usable), stays flat is critical-ish (often
the sweet spot), grows is chaotic (small differences blow up, readouts cannot
generalise).

### 7.3 Separation (Legenstein and Maass)

Feed many distinct random streams; measure the average pairwise distance of the
resulting states (same `1/sqrt(dim)` normalisation). The goal is **high
separation with low divergence**: tell inputs apart without being chaotic.

**What the benchmark established:** the frozen random reservoir has a recall
horizon of only ~30 ms (about 2 reservoir-steps), and spike-frequency adaptation
sweeps lift this to at best ~33 ms. This structural ceiling is the problem the
rest of the work attacks.

---

## 8. Surrogate-gradient BPTT, and why the first version failed

### 8.1 The surrogate gradient

The spike is a Heaviside step `s = H(V - V_th)`, whose derivative is zero almost
everywhere and undefined at the threshold, so true gradients cannot flow. The fix
(SuperSpike, Zenke and Ganguli 2018) keeps the **hard** spike in the forward pass
(so the forward dynamics stay byte-identical to the benchmark and the trained
weights transfer) but substitutes a smooth **surrogate derivative** in the
backward pass:

```
forward:   s = H(V - V_th)
backward:  ds/dV  ->  phi'(V - V_th) = 1 / (1 + beta * |V - V_th|)^2
```

This fast-sigmoid bump is large near threshold and small far from it, so
gradient flows through neurons that are close to spiking. A `tanh`-style smooth
forward mode is also provided purely so a finite-difference gradient check is
meaningful; that check passes at ~1e-10, confirming the backward pass is
mathematically correct.

### 8.2 The objective

After a washout, at every reservoir-step a linear readout `W_out` reconstructs the
input delayed by `target_delay` from a leaky spike trace `tr`:

```
tr  <- tr * exp(-dt / tau_state) + s
pred = W_out . [tr ; 1]
loss = mean( (pred - u(t - target_delay))^2 )   (+ a firing-rate homeostasis term)
```

Only the recurrent matrix `W` (with Dale's law, sparsity, and ceiling preserved)
and `W_out` are trained; BPTT is truncated to short windows.

### 8.3 Why the conductance version diverges (the root cause)

Even though the gradient is correct, the **through-time adjoint is exponentially
divergent** with conductance synapses. The synaptic current

```
I_syn = -g_e * (V - E_exc) - g_i * (V - E_inh)
```

is voltage-dependent. Two things follow in the backward recursion:

1. A direct self-feedback term `d(I_syn)/dV`, because `I_syn` depends on `V` and
   `V` depends on `I_syn`. This is a self-amplifying loop.
2. A multiplicative factor of order `|V - E| ~ 65 to 80 mV` picked up every single
   step when propagating through the conductance.

So the per-step adjoint norm is multiplied by a large factor each step and grows
without bound. Per-step adjoint-norm clipping (`a <- a * clip / ||a||` when over a
ceiling) does not save it: the norm simply fills whatever ceiling you set
(`1e3 -> 13k`, `1e7 -> 1e8`). The gradient reaching `W` is then clipped noise with
no usable direction, and recall MSE never leaves chance (1/3 for a 3-level target).
This is captured in `diagnostics/bptt_rootcause.png`.

### 8.4 The fix: current-based synapses

The cure is to make the synaptic current **independent of voltage**:

```
I_syn = g_e * ge - g_i * gi          # current-based, no (V - E) factor
```

`ge`, `gi` remain per-step decaying traces driven by presynaptic spikes through
`W`, but the injected current no longer scales with `(V - E)`. This kills the
`d(I_syn)/dV` self-feedback term entirely and removes the `(V - E)` amplifier, so
the backward recursion stays bounded. The constant gains `g_e, g_i` are chosen so
the resting recurrent drive roughly matches the old conductance model
(`|V - E_exc| ~ 65`, `|V - E_inh| ~ 15`), i.e. `g_e ~ 60`, `g_i ~ 15`.

Trade-off, stated honestly: this changes the forward dynamics, so the benchmark
must be run with the same current-based model for the weights to transfer. The
best run (`results_mem_bptt_v7`, 50 ms synapses) reached final MSE ~0.32 against
the 0.33 chance floor at delay 2, but it stayed short-memory and prone to drifting
into unstable recurrent reverberation (saturation / silence bursting). The
trainer is numerically fixed; the *mechanism* of holding memory in recurrent
reverberation is the real limitation.

---

## 9. The resolution: memory in slow state variables (ALIF / LSNN)

The decisive realisation: long memory should not come from recurrent
reverberation (the unstable, short-lived kind BPTT kept finding). It should come
from **slow per-neuron state variables whose time constant matches the lag you
want to hold.**

### 9.1 Adaptive-threshold dynamics

Each neuron carries an adaptation variable `b` that raises its own firing
threshold and decays with its **own** time constant `tau_a`:

```
V   <- V + (dt / tau_m) * ( -V + R * I )
thr  = 1 + beta_a * b
s    = 1 if V >= thr else 0
V   <- V * (1 - s)                 # reset to 0 on spike
b   <- b * rho_a + s,   rho_a = exp(-dt / tau_a)
```

The crucial design choice is the **distribution of `tau_a`**: drawn log-uniformly
up to `tau_a_max` (default 8000 ms), so the population tiles timescales from short
up to and beyond the target delay. The feedforward cue weights `W_in` are
mixed-sign random, so different neurons load their slow variable differently as a
function of the cue, which makes the cue linearly decodable from the population.

### 9.2 Why this holds memory for seconds

A brief cue perturbs spiking, which loads the slow `b` variables. During the quiet
delay there is no further cue, so each `b` simply decays:

```
b(delay) ~ b(0) * exp(-delay / tau_a)
```

Neurons with `tau_a >> delay` still carry the cue at recall time. The memory is not
a reverberating activity loop (which is unstable); it is the residual value of
slow variables, which is intrinsically stable.

### 9.3 Closed-form readout, no BPTT

Because the recurrent weights `W` are **fixed** (set to a controlled, stable gain)
and only a **linear readout** is fit, training is closed-form ridge regression:

```
w = (X^T X + lambda I)^-1 X^T y
```

where each feature row is the population state `[b | fast trace | bias]` read at
the recall delay. This scales to 5 s effortlessly, where full BPTT would have to
tape ~50k simulation steps (many GB). The score is `R^2 = 1 - MSE / Var(cue)`,
where `Var(cue)` is the chance MSE of predicting the mean.

### 9.4 Store-and-hold result

N=2000, 10 seeds, scalar cue recalled after a quiet delay:

| delay (ms) | 250 | 500 | 1000 | 2000 | 3000 | 5000 |
|---|---|---|---|---|---|---|
| mean R^2 | 0.9985 | 0.9983 | 0.9981 | 0.9978 | 0.9977 | 0.9975 |
| min R^2 | 0.9975 | 0.9973 | 0.9971 | 0.9969 | 0.9967 | 0.9965 |

Near-perfect recall out to 5 seconds, roughly a 150x improvement over the 33 ms
adaptation-reservoir ceiling, from changing the mechanism rather than tuning
harder.

---

## 10. Working-memory capacity: superposition and interference

`train_memory_alif_capacity.py` asks how many items the same network holds at
once. `K` scalar cues are presented **simultaneously**, each through its own random
projection `W_in_k`, so the slow variables hold a linear superposition:

```
cue current = (1 / sqrt(K)) * sum_{k=1..K} cue_k * W_in_k
```

### 10.1 Why 1/sqrt(K) and why it works

The `1 / sqrt(K)` normalisation keeps the total cue drive (and hence the firing
regime, the network's operating point) fixed as `K` grows. This is deliberate: it
means any capacity drop measures **representational interference**, not the network
simply being driven harder.

In an `N`-dimensional space, random projections are near-orthogonal, so `K` items
written as a weighted sum stay linearly separable until `K` grows large enough that
the projections overlap. One ridge readout per item then decodes its value from the
shared state. Capacity therefore scales with `N` and degrades with both `K` and
delay, the same qualitative shape as the human 'about four items' working-memory
limit.

### 10.2 Capacity result

N=2000, 10 seeds, mean R^2:

| K items | 500 ms | 2000 ms | 5000 ms |
|---|---|---|---|
| 1 | 0.991 | 0.967 | 0.863 |
| 2 | 0.990 | 0.965 | 0.869 |
| 4 | 0.985 | 0.971 | 0.878 |
| 8 | 0.976 | 0.953 | 0.840 |
| 16 | 0.945 | 0.909 | 0.739 |
| 32 | 0.894 | 0.814 | 0.397 |
| 64 | 0.799 | 0.634 | -0.058 |

Roughly 8 to 16 items held well at short-to-medium delays, degrading gracefully as
interference grows with `K` and as the slow variables decay with delay. The
negative R^2 at K=64, 5 s means the readout is worse than predicting the mean:
total interference.

---

## 11. Summary of every change, in one place

| Area | Before / problem | Change | Why it matters |
|---|---|---|---|
| Conductance synapse | `+g(V-E)` sign hyperpolarised on excitation | flip to `-g(V-E)` | correct biophysics; spikes now depolarise |
| Weights after sign fix | peak ~0.5 saturates | rescale to ~0.005 | network fires at a sane 15-25 Hz |
| STDP rate after fix | 0.01 hits the rails | drop to ~0.002 | stable learning |
| STDP bookkeeping | `last_spike` updated before delta_t | update after; soft-bound not hard clip | unambiguous self term, smooth bounding |
| Drive | constant -> clock-like firing | `I_mean + sigma N(0,1)/sqrt(dt)` | irregular AI firing; dt-invariant noise |
| E/I balance | `-1` inhibition synchronises | scale inhibition by `-g_inh` (4-6) | inhibition-dominated AI regime |
| Weight scaling | size-dependent operating point | `w_total / sqrt(K)` | comparable across N |
| Hopfield | binary sign update, 0.14N capacity | softmax update (Modern Hopfield) | 500+ patterns; identical to attention |
| MNIST | dense softmax, raw images | top-k + prototypes + momentum | stable, interpretable retrieval |
| Memory training (1) | R-STDP, no signal to bootstrap | abandoned | no decodable memory to reward |
| Memory training (2) | conductance BPTT adjoint diverges | current-based synapses | bounded backward pass |
| Memory mechanism | recurrent reverberation, unstable | slow adaptation variables (ALIF) | stable recall to 5 s, R^2 ~ 0.998 |
| Capacity probe | single item | K-item superposition, 1/sqrt(K) | human-like working-memory curve |

The throughline: most of the gains came from getting the **mechanism** right
(correct synapse sign, fluctuating drive, voltage-independent gradients, slow
state variables) rather than from more parameter tuning.

References: Hodgkin and Huxley (1952); Brunel (2000); Jaeger (2001); Legenstein
and Maass (separation); Zenke and Ganguli (2018, SuperSpike surrogate gradients);
Bellec et al. (LSNN, adaptive spiking memory); Ramsauer et al. (2020, Hopfield
Networks is All You Need); Dayan and Abbott (Theoretical Neuroscience).
