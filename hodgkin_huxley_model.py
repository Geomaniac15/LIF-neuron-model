import numpy as np
import matplotlib.pyplot as plt

# HH parameters
C_m = 1.0      # membrane capacitance, uF/cm^2
g_Na = 120.0   # max sodium conductance
g_K = 36.0     # max potassium conductance
g_L = 0.3      # leak conductance
E_Na = 50.0    # sodium reversal potential
E_K = -77.0    # potassium reversal potential
E_L = -54.4    # leak reversal potential

dt = 0.01      # smaller timestep needed, ms
I_ext = 10.0   # external current injection

# initial conditions
V = -65.0
m = 0.05
h = 0.6
n = 0.32

voltage_history = []
m_history = []
h_history = []
n_history = []


for _ in range(5000):
    # for m (sodium activation)
    alpha_m = 0.1 * (V + 40) / (1 - np.exp(-(V + 40) / 10))
    beta_m  = 4.0 * np.exp(-(V + 65) / 18)

    # for h (sodium inactivation)
    alpha_h = 0.07 * np.exp(-(V + 65) / 20)
    beta_h  = 1.0 / (1 + np.exp(-(V + 35) / 10))

    # for n (potassium activation)
    alpha_n = 0.01 * (V + 55) / (1 - np.exp(-(V + 55) / 10))
    beta_n  = 0.125 * np.exp(-(V + 65) / 80)

    # gating updates
    dm = (alpha_m * (1 - m) - beta_m * m) * dt
    m = m + dm

    dh = (alpha_h * (1 - h) - beta_h * h) * dt
    h = h + dh

    dn = (alpha_n * (1 - n) - beta_n * n) * dt
    n = n + dn

    # voltage update
    dV = ((-g_Na * m**3 * h * (V - E_Na))
    + (-g_K  * n**4      * (V - E_K))
    + (-g_L             * (V - E_L))
    + I_ext) / C_m

    V = V + dV * dt

    voltage_history.append(V)
    m_history.append(m)
    h_history.append(h)
    n_history.append(n)

times = np.arange(len(voltage_history)) * dt

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

ax1.plot(times, voltage_history, color='teal')
ax1.set_ylabel('voltage (mV)')

ax2.plot(times, m_history, label='m (Na activation)')
ax2.plot(times, h_history, label='h (Na inactivation)')
ax2.plot(times, n_history, label='n (K activation)')
ax2.set_ylabel('gating variable')
ax2.set_xlabel('time (ms)')
ax2.legend()

plt.show()