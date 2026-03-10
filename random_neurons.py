import numpy as np

N = 10
V = np.full(N, -70.0)
I_syn = np.zeros(N)
I_ext = np.full(N, 1.4)

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

for step in range(1000):
    dV = (1 / tau_m) * (-(V - V_rest) + R * (I_ext + I_syn))
    V = V + dV * dt

print(V)