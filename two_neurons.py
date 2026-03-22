'''
1. Start at V = V_{rest}
2. Each timestep, compute dV/dt using the equation
3. Update V += dV ⋅ dt
4. If V ≥ V_{threshold}, record a spike and reset V = V_{reset}
5. Repeat
'''

import matplotlib.pyplot as plt
import numpy as np
from neuron_class import Neuron

dt = 0.1 # ms
tau_syn = 5 # ms
I_syn_B = 0 # starts at zero

neuronA = Neuron(-70, -80, -55, 20, 10, 2, w=2)
neuronB = Neuron(-70, -80, -55, 20, 10, 1.4)

neuronA_spike_times = []
neuronB_spike_times = []

neuronA_voltage_history = []
neuronB_voltage_history = []

V_A = neuronA.V_rest
V_B = neuronB.V_rest

# print(V_A)

for x in range(1_000):
    dV_A = neuronA.calculate(V_A)

    V_A = V_A + dV_A * dt
    neuronA_voltage_history.append(V_A)

    a_spiked = V_A >= neuronA.V_threshold
    b_spiked = V_B >= neuronB.V_threshold

    synaptic_input = neuronA.weight if a_spiked else 0
    dV_B = neuronB.calculate(V_B, I_syn_B)

    V_B = V_B + dV_B * dt
    neuronB_voltage_history.append(V_B)

    if a_spiked:
        I_syn_B += neuronA.weight
        print(f'neuron A spike at step {x}')
        neuronA_spike_times.append(x * dt)

        V_A = neuronA.V_reset
    
    I_syn_B -= (I_syn_B / tau_syn) * dt # decay every step

    if b_spiked:
        print(f'neuron B spike at step {x}')
        neuronB_spike_times.append(x * dt)

        V_B = neuronB.V_reset
    
    # print(f'V_A: {V_A}')
    # print(f'V_B: {V_B}')

# print(f'total spikes: {len(spike_times)}')
# print(f'average ISI: {(spike_times[-1] - spike_times[0]) / (len(spike_times) - 1):.2f} ms')

A_steps = np.arange(len(neuronA_voltage_history)) * dt
B_steps = np.arange(len(neuronB_voltage_history)) * dt

plt.figure(figsize=(12, 4))

plt.plot(A_steps, neuronA_voltage_history, color='teal')
plt.plot(B_steps, neuronB_voltage_history, color='red')

plt.axhline(neuronA.V_threshold, color='orange', linestyle='--', label='A threshold')
plt.axhline(neuronA.V_rest, color='steelblue', linestyle='--', label='A rest')

plt.axhline(neuronB.V_threshold, color='purple', linestyle='--', label='B threshold')
plt.axhline(neuronB.V_rest, color='gray', linestyle='--', label='B rest')

plt.xlabel('time (ms)')
plt.ylabel('membrane voltage (mV)')

plt.legend()
plt.show()