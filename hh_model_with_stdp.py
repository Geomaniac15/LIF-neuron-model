import numpy as np
import matplotlib.pyplot as plt

N_in = 20   # input neurons
N_out = 10  # output neurons

dt = 0.1
tau_m = 20
tau_syn = 5
R = 10
V_rest = -70.0
V_threshold = -55.0
V_reset = -80.0
tau_stdp = 20
A_plus = 0.01
A_minus = 0.01

pattern_A = np.zeros(N_in)
pattern_A[:10] = 1

pattern_B = np.zeros(N_in)
pattern_B[10:] = 1

# # feedforward weights, input to output, random small positive values
W_in = np.random.rand(N_in, N_out) * 0.5

# lateral inhibition, output to output, fixed negative weights
W_lat = np.full((N_out, N_out), 2.0)
np.fill_diagonal(W_lat, 0)  # no self connections

V_out = np.full(N_out, V_rest)
g_exc = np.zeros(N_out)
g_inh = np.zeros(N_out)
refractory = np.zeros(N_out)
last_spike = np.full(N_out, -np.inf)

spike_record = []

# print(W_in)

for trial in range(100):
    # alternate patterns
    pattern = pattern_A if trial % 2 == 0 else pattern_B
    I_ext_in = pattern * 3.0  # active neurons get 3 nA

    for step in range(500):  # 50ms per pattern
        t = (trial * 500 + step) * dt

        # input layer: which input neurons are spiking this step?
        # simple threshold: active neurons fire regularly
        input_spikes = (pattern == 1) & (step % 10 == 0)

        # compute synaptic input to output layer
        # from input layer
        g_exc += W_in[input_spikes, :].sum(axis=0) * 0.3

        # compute current and update output voltages
        I_syn = g_exc * (0.0 - V_out) + g_inh * (-80.0 - V_out)
        dV = (1 / tau_m) * (-(V_out - V_rest) + R * (2.0 + I_syn))
        V_out = V_out + dV * dt
        V_out = np.clip(V_out, -90.0, 50.0)

        # spike detection
        refractory -= dt
        spikes_out = (V_out >= V_threshold) & (refractory <= 0)
        V_out[spikes_out] = V_reset
        refractory[spikes_out] = 2.0

        # lateral inhibition from output spikes
        for i in np.where(spikes_out)[0]:
            g_inh += W_lat[i, :]
            spike_record.append((t, i))

            # STDP on W_in
            delta_t = t - last_spike[i]
            W_in[:, i] += A_plus * np.exp(-delta_t / tau_stdp) * input_spikes
            last_spike[i] = t

        g_exc -= (g_exc / tau_syn) * dt
        g_inh -= (g_inh / tau_syn) * dt

# print(W_in.round(2))
# print(W_lat)

# create noisy pattern A
noise_mask = np.random.rand(N_in) < 0.2  # flip 20% of bits
pattern_A_noisy = pattern_A.copy()
pattern_A_noisy[noise_mask] = 1 - pattern_A_noisy[noise_mask]

# reset output layer state
V_out = np.full(N_out, V_rest)
g_exc = np.zeros(N_out)
g_inh = np.zeros(N_out)
refractory = np.zeros(N_out)

# run for 50ms and count spikes per output neuron
test_spikes = np.zeros(N_out)

for step in range(500):
    input_spikes = (pattern_A_noisy == 1) & (step % 10 == 0)
    g_exc += W_in[input_spikes, :].sum(axis=0) * 0.3

    I_syn = g_exc * (0.0 - V_out) + g_inh * (-80.0 - V_out)
    dV = (1 / tau_m) * (-(V_out - V_rest) + R * (2.0 + I_syn))
    V_out = V_out + dV * dt
    V_out = np.clip(V_out, -90.0, 50.0)

    refractory -= dt
    spikes_out = (V_out >= V_threshold) & (refractory <= 0)
    V_out[spikes_out] = V_reset
    refractory[spikes_out] = 2.0

    for i in np.where(spikes_out)[0]:
        g_inh += W_lat[i, :]
        test_spikes[i] += 1

    g_exc -= (g_exc / tau_syn) * dt
    g_inh -= (g_inh / tau_syn) * dt

print('output neuron spike counts for noisy pattern A:')
print(test_spikes.astype(int))
print(f'most active neuron: {np.argmax(test_spikes)}')

# test with clean pattern A
V_out = np.full(N_out, V_rest)
g_exc = np.zeros(N_out)
g_inh = np.zeros(N_out)
refractory = np.zeros(N_out)
test_spikes_clean = np.zeros(N_out)

for step in range(500):
    input_spikes = (pattern_A == 1) & (step % 10 == 0)
    g_exc += W_in[input_spikes, :].sum(axis=0) * 0.3
    I_syn = g_exc * (0.0 - V_out) + g_inh * (-80.0 - V_out)
    dV = (1 / tau_m) * (-(V_out - V_rest) + R * (2.0 + I_syn))
    V_out = V_out + dV * dt
    V_out = np.clip(V_out, -90.0, 50.0)
    refractory -= dt
    spikes_out = (V_out >= V_threshold) & (refractory <= 0)
    V_out[spikes_out] = V_reset
    refractory[spikes_out] = 2.0
    for i in np.where(spikes_out)[0]:
        g_inh += W_lat[i, :]
        test_spikes_clean[i] += 1
    g_exc -= (g_exc / tau_syn) * dt
    g_inh -= (g_inh / tau_syn) * dt

np.save('W_in_trained.npy', W_in)
np.savetxt('W_in_trained.txt', W_in.round(3), fmt='%.3f')

# test with pattern B
V_out = np.full(N_out, V_rest)
g_exc = np.zeros(N_out)
g_inh = np.zeros(N_out)
refractory = np.zeros(N_out)
test_spikes_B = np.zeros(N_out)

for step in range(500):
    input_spikes = (pattern_B == 1) & (step % 10 == 0)
    g_exc += W_in[input_spikes, :].sum(axis=0) * 0.3
    I_syn = g_exc * (0.0 - V_out) + g_inh * (-80.0 - V_out)
    dV = (1 / tau_m) * (-(V_out - V_rest) + R * (2.0 + I_syn))
    V_out = V_out + dV * dt
    V_out = np.clip(V_out, -90.0, 50.0)
    refractory -= dt
    spikes_out = (V_out >= V_threshold) & (refractory <= 0)
    V_out[spikes_out] = V_reset
    refractory[spikes_out] = 2.0
    for i in np.where(spikes_out)[0]:
        g_inh += W_lat[i, :]
        test_spikes_B[i] += 1
    g_exc -= (g_exc / tau_syn) * dt
    g_inh -= (g_inh / tau_syn) * dt

print('pattern A:', pattern_A.astype(int))
print('pattern B:', pattern_B.astype(int))
overlap = np.sum(pattern_A * pattern_B)
print(f'overlap: {overlap} shared active neurons')

print(f'clean A winner: neuron {np.argmax(test_spikes_clean)}')
print(f'noisy A winner: neuron {np.argmax(test_spikes)}')
print(f'pattern B winner: neuron {np.argmax(test_spikes_B)}')