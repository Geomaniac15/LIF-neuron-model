'''
1. Start at V = V_{rest}
2. Each timestep, compute dV/dt using the equation
3. Update V += dV ⋅ dt
4. If V ≥ V_{threshold}, record a spike and reset V = V_{reset}
5. Repeat
'''

import matplotlib.pyplot as plt
import numpy as np

V_rest = -70 # mV
V_reset = -80 # mV
V_threshold = -55 # mV
tau_m = 20 # ms
R = 10 # M Ohms
I = 2 # nA
dt = 0.1 # ms

spike_times = []
voltage_history = []

def calculate(V, V_rest, R, I):
    dV = (1 / 20) * (-(V - V_rest) + R * I)
    return dV

V = V_rest

print(V)

for x in range(1_000):
    dV = calculate(V, V_rest, R, I)

    V = V + dV * dt
    voltage_history.append(V)

    if V >= V_threshold:
        print(f'spike at step {x}')
        spike_times.append(x * dt)
        V = V_reset
    
    print(V)

print(f'total spikes: {len(spike_times)}')
print(f'average ISI: {(spike_times[-1] - spike_times[0]) / (len(spike_times) - 1):.2f} ms')

steps = np.arange(len(voltage_history)) * dt

plt.figure(figsize=(12, 4))
plt.plot(steps, voltage_history, color='teal')
plt.axhline(V_threshold, color='orange', linestyle='--', label='threshold')
plt.axhline(V_rest, color='steelblue', linestyle='--', label='rest')
plt.xlabel('time (ms)')
plt.ylabel('membrane voltage (mV)')
plt.legend()
plt.show()