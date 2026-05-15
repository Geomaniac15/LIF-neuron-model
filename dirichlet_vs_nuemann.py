import numpy as np
import matplotlib.pyplot as plt

# parameters
L = 1.0
nx = 50
dx = L / (nx - 1)
D = 1.0
dt = 0.1 * dx**2 / D

lambda_ = D * dt / dx**2

# initial condition
u = np.zeros(nx)
u[int(nx/2)] = 100  # spike in the middle

# time evolution
for _ in range(500):

    u_new = u.copy()

    for i in range(1, nx-1):
        u_new[i] = u[i] + lambda_ * (u[i+1] - 2*u[i] + u[i-1])

    # Dirichlet
    u_new[0] = 0
    u_new[-1] = 0

    # Neumann (zero flux)
    # u_new[0] = u_new[1]
    # u_new[-1] = u_new[-2]

    u = u_new

plt.plot(u)
plt.show()