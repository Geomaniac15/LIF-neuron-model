# LIF Neuron Model

A from-scratch implementation of spiking neural network models in NumPy, 
built to understand the computational basis of biological neural circuits.

## What this is

This project implements progressively more complex models of neural dynamics,
starting from a single neuron and building up to a network capable of 
associative memory and pattern recognition.

Everything is implemented from first principles using only NumPy and 
Matplotlib. No ML frameworks.

## Files

### neuron_class.py
Single leaky integrate-and-fire (LIF) neuron. Implements the core 
differential equation:

τ_m dV/dt = -(V - V_rest) + R·I(t)

Voltage integrates input current, leaks back to rest, and fires when 
it crosses threshold. Includes refractory period and exponentially 
decaying synaptic currents.

### two_neurons.py
Two connected LIF neurons with a unidirectional synapse. Demonstrates 
causal firing: neuron A drives neuron B through a synaptic weight. 
Synaptic current uses an exponential decay kernel to model 
neurotransmitter clearance.

### random_neurons.py
100-neuron E-I network with:
- 80 excitatory / 20 inhibitory neurons
- Random sparse connectivity (5%)
- Conductance-based synapses with biologically realistic reversal 
  potentials (E_exc = 0mV, E_inh = -80mV)
- Heterogeneous membrane time constants and thresholds
- Enforced refractory periods
- STDP learning rule

Exhibits emergent synchrony, asynchronous irregular firing, and 
rhythmic E-I oscillations depending on parameters.

### hodgkin_huxley_model.py
Full Hodgkin-Huxley neuron with voltage-gated ion channels. Models 
sodium (Na+) and potassium (K+) conductances via gating variables 
m, h, n. Produces realistic action potential waveforms including:
- Sharp upstroke to +40mV (Na+ channel opening)
- Rapid downstroke (Na+ inactivation, K+ activation)
- Hyperpolarisation undershoot (K+ overshoot)
- Natural refractory period from channel kinetics

No manual spike reset needed. The physics generates the spike shape.

### hh_model_with_stdp.py
Feedforward network with associative memory. Architecture:
- 20 input neurons encoding binary patterns
- 10 output neurons with lateral inhibition (winner-take-all)
- W_in: 20x10 feedforward weight matrix, trained via STDP
- W_lat: 10x10 fixed inhibitory matrix

Trained on two non-overlapping patterns (A and B) over 100 trials.
After training, correctly identifies noisy/corrupted versions of 
pattern A and maps distinct patterns to distinct output neurons.

This is the same computational principle underlying hippocampal 
pattern completion in biological memory.

## Results

- Single LIF neuron: regular spiking at ~36Hz with ISI of 28ms
- Two neurons: causal firing, B responds ~10ms after A
- 100 neuron network: emergent E-I oscillations at ~15Hz
- HH neuron: action potentials peaking at +40mV with realistic 
  3ms spike width
- Pattern network: correctly recognises pattern A even with 20% 
  corrupted bits, separates pattern A (neuron 2) from pattern B 
  (neuron 8)

## Background

Built while learning computational neuroscience independently. 
Core references:
- Dayan & Abbott, Theoretical Neuroscience (MIT Press)
- Hodgkin & Huxley (1952), A quantitative description of membrane 
  current and its application to conduction and excitation in nerve

## Author

George :)
