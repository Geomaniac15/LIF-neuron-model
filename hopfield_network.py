import numpy as np

N = 10  # neurons

pattern_A = np.array([1, -1, 1, 1, -1, 1, -1, 1, -1, 1])
pattern_B = np.array([-1, 1, -1, 1, 1, -1, 1, -1, 1, 1])

# W = (1/N) * (outer product of A with itself + outer product of B with itself)
W = (1/N) * (np.outer(pattern_A, pattern_A) + np.outer(pattern_B, pattern_B))
np.fill_diagonal(W, 0)  # no self connections

# print(W)
np.savetxt('W.txt', W.round(3), fmt='%.3f')

state = pattern_A.copy()
noisy_A = np.random.rand(state) < 0.2 # flip 20% bits

# then update repeatedly until convergence
for iteration in range(20):
    new_state = np.sign(W @ state)
    if np.array_equal(new_state, state):
        print(f'converged at iteration {iteration}')
        break
    state = new_state

print('retrieved:', state)
print('pattern A:', pattern_A)
print('match:', np.array_equal(state, pattern_A))