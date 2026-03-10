import numpy as np
import matplotlib as plt

N = 10
V = np.full(N, -70.0)
I_syn = np.zeros(N)
I_ext = np.full(N, 2.0)

tau_m = 20
tau_syn = 5
R = 10
V_rest = -70.0
V_threshold = -55.0
V_reset = -80.0
dt = 0.1

W = np.random.rand(N, N)
mask = np.random.rand(N, N) > 0.8
W = W * mask
np.fill_diagonal(W, 0)

spike_record = []

for step in range(1000):
    dV = (1 / tau_m) * (-(V - V_rest) + R * (I_ext + I_syn))
    V = V + dV * dt

    spikes = V >= V_threshold
    V[spikes] = V_reset

    for i in np.where(spikes)[0]:
        I_syn += W[i, :]
        spike_record.append((step * dt, i))
    
    I_syn -= (I_syn / tau_syn) * dt

# print(V)
# print(spike_record[20:])