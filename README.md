# Computational Neuroscience from Scratch

A from-scratch implementation of spiking neural network models and associative 
memory systems in NumPy, built to understand the computational basis of 
biological neural circuits and their connection to modern AI.

Everything is implemented from first principles using only NumPy, Matplotlib, 
and scikit-learn. No ML frameworks.

## What this covers

This project builds a complete arc from single neuron biophysics to modern 
Hopfield networks and transformer attention, implementing each step by hand:

```
Single neuron → Synaptic circuits → E-I networks → Hodgkin-Huxley → STDP learning → Hopfield memory → Modern Hopfield → MNIST retrieval
```

---

## Spiking Neural Networks

### neuron_class.py
Single leaky integrate-and-fire (LIF) neuron. Implements the core 
differential equation:

```
τ_m dV/dt = -(V - V_rest) + R·I(t)
```

Voltage integrates input current, leaks back to rest, and fires when 
it crosses threshold. Includes refractory period and exponentially 
decaying synaptic currents.

**Results:** Regular spiking at ~36Hz with ISI of 28ms. Verified critical 
current threshold at 1.5 nA below which the neuron never fires.

### two_neurons.py
Two connected LIF neurons with a unidirectional synapse. Demonstrates 
causal firing: neuron A drives neuron B through a synaptic weight. 
Synaptic current uses an exponential decay kernel to model 
neurotransmitter clearance.

**Results:** Neuron B fires ~10ms after each A spike, confirming causal 
synaptic transmission.

### random_neurons.py
100-neuron E-I network with:
- 80 excitatory / 20 inhibitory neurons
- Random sparse connectivity (5%)
- Conductance-based synapses with biologically realistic reversal 
  potentials (E_exc = 0mV, E_inh = -80mV)
- Heterogeneous membrane time constants and thresholds
- Enforced refractory periods
- STDP learning rule with exponential decay kernel

**Results:** Exhibits emergent synchrony, asynchronous irregular firing, 
and rhythmic E-I oscillations at ~15Hz depending on connectivity parameters. 
The transition between synchronous and asynchronous regimes mirrors the 
difference between epileptic and healthy cortical activity.

### hodgkin_huxley_model.py
Full Hodgkin-Huxley neuron with voltage-gated ion channels. Models 
sodium (Na+) and potassium (K+) conductances via gating variables m, h, n:

```
C_m dV/dt = -g_Na·m³h·(V-E_Na) - g_K·n⁴·(V-E_K) - g_L·(V-E_L) + I
```

Each gating variable follows:
```
dx/dt = α_x(V)·(1-x) - β_x(V)·x
```

Produces realistic action potential waveforms including:
- Sharp upstroke to +40mV (Na+ channel opening, positive feedback)
- Rapid downstroke (Na+ inactivation via h gate, K+ activation via n gate)
- Hyperpolarisation undershoot to -75mV (K+ overshoot)
- Natural refractory period emerging from channel kinetics

No manual spike reset. The physics generates the spike shape.

**Results:** Action potentials peaking at +40mV with realistic ~3ms spike 
width. Gating variable plots show the exact phase relationship between 
m, h, and n that produces each spike component.

### hh_model_with_stdp.py
Feedforward network with associative memory using conductance-based synapses 
and spike-timing-dependent plasticity (STDP). Architecture:
- 20 input neurons encoding binary patterns
- 10 output neurons with lateral inhibition (winner-take-all)
- W_in: 20x10 feedforward weight matrix trained via STDP
- W_lat: 10x10 fixed inhibitory matrix

STDP rule: synapses strengthen when pre-synaptic neuron fires just before 
post-synaptic, weaken when it fires after. Weight change decays exponentially 
with spike time difference.

Trained on two non-overlapping patterns over 100 trials with state reset 
between trials to prevent interference.

**Results:** Correctly identifies noisy/corrupted versions of pattern A 
(20% bit flips) and maps distinct patterns to distinct output neurons 
(pattern A → neuron 2, pattern B → neuron 8).

---

## Hopfield Networks

### hopfield_network.py
Binary and continuous Hopfield networks demonstrating associative memory 
and the connection to transformer attention.

**Binary Hopfield network:**
- Hebbian weight learning: W = (1/N) Σ_μ x_i^μ x_j^μ
- Sign update rule: s_i = sign(Σ_j W_ij s_j)
- Theoretical capacity: ~0.14N patterns

**Energy landscape:**
Each retrieval step decreases network energy E = -½ s^T W s. Memories 
are energy minima. Corrupted inputs roll downhill to the nearest memory.

**Results:**
- Noise tolerance: perfect retrieval up to 30% corruption, degrades at 40%, 
  collapses at 50%
- Capacity: perfect up to 10 patterns, degrades at 15, collapses at 20 
  (theoretical limit for N=100 is 14 patterns)
- Energy plots confirm monotonic decrease during retrieval, with more 
  iterations required near capacity limit

**Continuous (Modern) Hopfield network:**
Replaces sign update with softmax, dramatically increasing capacity:

```
s_new = X^T · softmax(β · X · s)
```

Temperature parameter β controls retrieval sharpness. At β → ∞ recovers 
binary network. Optimal β depends on dot product scale, motivating the 
1/√d scaling in transformer attention.

**Connection to transformers:**
The continuous Hopfield update is mathematically identical to attention:

```
Attention(Q, K, V) = softmax(QK^T / √d) · V
```

Q is the query (noisy input), K is the keys (stored patterns), V is the 
values (retrieved content), softmax(QK^T) computes similarity weights.

**Results:**
- Capacity increases from 14 to 500+ patterns with near-perfect retrieval
- Optimal β empirically identified around 2.0-3.0 for random ±1 patterns
- Performance degrades gracefully rather than catastrophically

### mnist_hopfield.py
Modern Hopfield network applied to MNIST digit completion. Given a corrupted 
or partial digit image, the network retrieves the most similar stored memory.

**Architecture:**
- Top-k attention: only attends to k most similar stored patterns per step
- Prototype memories: averages multiple examples per digit class
- Momentum update: state = 0.6·state + 0.4·new_state for stable convergence
- Noise model: random pixel flips at configurable noise level

**Key insight:** With large pattern libraries the softmax concentrates weight 
on the single nearest neighbour, making retrieval effectively exact nearest 
neighbour search. With sparse libraries the network genuinely blends stored 
patterns, producing reconstructions that differ from any individual memory. 
The regime depends on pattern density relative to query space.

**Results with 50 prototype memories (5 per digit, 30% noise):**
- Digit classification accuracy: 57.2%
- Pixel overlap: 89.6%
- Digits 0, 1, 7 retrieved most reliably
- Digits 2, 3, 8 most confused (visually similar stroke patterns)

**Visualisations:**
- PCA trajectory plot: shows retrieval path through 2D projected memory 
  space, starting from corrupted input (red) rolling toward nearest memory 
  cluster (green)
- Animated GIF of retrieval dynamics
- Beta comparison: effect of temperature on retrieval sharpness

---

## Results Summary

| Model | Key result |
|---|---|
| Single LIF neuron | 36Hz spiking, verified 1.5nA critical current |
| Two neurons | Causal synaptic transmission, B fires 10ms after A |
| 100-neuron E-I network | Emergent 15Hz oscillations, synchrony/asynchrony transition |
| Hodgkin-Huxley | Realistic action potentials, ion channel dynamics |
| STDP network | Pattern A/B separated after 100 training trials |
| Binary Hopfield | Capacity limit at 0.14N verified empirically |
| Continuous Hopfield | 500+ patterns stored vs 14 for binary version |
| MNIST Hopfield | 57% digit accuracy, 90% pixel overlap at 30% noise |

---

## Background

Built while learning computational neuroscience independently, implementing 
every model from first principles before reading the formal theory.

Core references:
- Dayan & Abbott, Theoretical Neuroscience (MIT Press)
- Hodgkin & Huxley (1952), A quantitative description of membrane current 
  and its application to conduction and excitation in nerve
- Ramsauer et al. (2020), Hopfield Networks is All You Need

## Author

George :)
