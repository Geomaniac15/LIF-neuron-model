import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml

mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X = mnist.data
y = mnist.target.astype(int)

X_binary = np.where(X > 127, 1.0, -1.0)
print(X_binary)

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

n_trials = 100

# for n_patterns in [14, 50, 100, 200, 500, 1000, 2500, 5000, 7500, 10000, 12500, 15000]:
#     patterns = np.array([np.random.choice([-1, 1], size=N) for _ in range(n_patterns)])
    
#     successes = 0
#     for trial in range(n_trials):
#         state = patterns[0].astype(float).copy()
#         noise_mask = np.random.rand(N) < 0.3
#         state[noise_mask] *= -1

#         for iteration in range(20):
#             new_state = hopfield_update(state, patterns)
#             if np.allclose(new_state, state, atol=1e-6):
#                 break
#             state = new_state

#         retrieved = np.sign(state)
#         if np.array_equal(retrieved, patterns[0]):
#             successes += 1

#     print(f'{n_patterns} patterns: {successes}/{n_trials} correct')

# patterns = np.array([np.random.choice([-1, 1], size=N) for _ in range(100)])

# for d in [4, 8, 16, 32, 64, 100]:
#     beta = 1 / np.sqrt(d)
#     successes = 0
#     for trial in range(20):
#         state = patterns[0].astype(float).copy()
#         noise_mask = np.random.rand(N) < 0.3
#         state[noise_mask] *= -1

#         for iteration in range(20):
#             new_state = hopfield_update(state, patterns, beta=beta)
#             if np.allclose(new_state, state, atol=1e-6):
#                 break
#             state = new_state

#         retrieved = np.sign(state)
#         if np.array_equal(retrieved, patterns[0]):
#             successes += 1

#     print(f'd={d}, beta={beta:.3f}: {successes}/20 correct')