import numpy as np
import matplotlib.pyplot as plt

def energy(state, W):
    return -0.5 * state @ W @ state

def softmax(x, beta=1.0):
    e = np.exp(beta * (x - np.max(x)))  # subtract max for numerical stability
    return e / e.sum()

def hopfield_update(state, patterns, beta=1.0):
    similarities = patterns @ state
    weights = softmax(similarities, beta)
    return patterns.T @ weights

N = 100
n_patterns = 14
patterns = np.array([np.random.choice([-1, 1], size=N) for _ in range(n_patterns)])

# print(patterns.shape)  # should be (14, 100)

# start with noisy pattern 0
state = patterns[0].astype(float).copy()
noise_mask = np.random.rand(N) < 0.3
state[noise_mask] *= -1

for iteration in range(20):
    new_state = hopfield_update(state, patterns)
    if np.allclose(new_state, state, atol=1e-6):
        print(f'converged at iteration {iteration}')
        break
    state = new_state

# check if it retrieved pattern 0
# continuous state needs to be thresholded back to binary for comparison
retrieved = np.sign(state)
match = np.array_equal(retrieved, patterns[0])
print(f'match: {match}')
print(f'fraction correct: {np.mean(retrieved == patterns[0]):.2f}')