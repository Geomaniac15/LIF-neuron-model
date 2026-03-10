import numpy as np
import matplotlib.pyplot as plt

N = 100
V = np.full(N, -70.0)
I_syn = np.zeros(N)
I_ext = np.random.uniform(1.4, 1.6, N)

tau_m = 20
tau_syn = 5
R = 10
V_rest = -70.0
V_threshold = -55.0
V_reset = -80.0
dt = 0.1

W = np.random.rand(N, N) * 0.5
mask = np.random.rand(N, N) > 0.95 # connectivity is 5%
W = W * mask
np.fill_diagonal(W, 0)

n_inhibitory = 20
W[80:, :] *= -1

W_initial = W.copy()

tau_stdp = 20  # ms
A_plus = 0.01  # strengthening rate
A_minus = 0.01  # weakening rate
last_spike = np.full(N, -np.inf)  # when each neuron last fired

# warmup
for step in range(2000):
    dV = (1 / tau_m) * (-(V - V_rest) + R * (I_ext + I_syn))
    V = V + dV * dt
    spikes = V >= V_threshold
    V[spikes] = V_reset
    for i in np.where(spikes)[0]:
        I_syn += W[i, :]
    I_syn -= (I_syn / tau_syn) * dt

spike_record = []

print(W.round(2))

for step in range(3000):
    dV = (1 / tau_m) * (-(V - V_rest) + R * (I_ext + I_syn))
    V = V + dV * dt

    spikes = V >= V_threshold
    V[spikes] = V_reset

    for i in np.where(spikes)[0]:
        I_syn += W[i, :]
        spike_record.append((step * dt, i))

        last_spike[i] = step * dt
        t_now = step * dt
        delta_t = t_now - last_spike

        W[:, i] += A_plus * np.exp(-delta_t / tau_stdp) * (W[:, i] != 0)
        W[i, :] -= A_minus * np.exp(-delta_t / tau_stdp) * (W[i, :] != 0)

        np.clip(W, -1.0, 1.0, out=W)
    
    I_syn -= (I_syn / tau_syn) * dt

print(W.round(2))

# print(V)
# print(spike_record[20:])

times = [t for t, i in spike_record]
neurons = [i for t, i in spike_record]

# plt.figure(figsize=(12, 5))
# plt.scatter(times, neurons, s=2, color='teal')
# plt.xlabel('time (ms)')
# plt.ylabel('neuron index')
# plt.title('spike raster')
# plt.yticks(range(0, N, 10))
# plt.show()

plt.figure(figsize=(12, 4))
plt.hist(W_initial[W_initial != 0].flatten(), bins=50, alpha=0.5, label='before', color='steelblue')
plt.hist(W[W != 0].flatten(), bins=50, alpha=0.5, label='after', color='teal')
plt.xlabel('synaptic weight')
plt.ylabel('count')
plt.title('weight distribution before vs after STDP')
plt.legend()
plt.show()