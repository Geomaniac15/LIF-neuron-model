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

N = 100  # neurons
n_patterns = 14
patterns = [np.random.choice([-1, 1], size=N) for _ in range(n_patterns)]
W = sum(np.outer(p, p) for p in patterns) / N
np.fill_diagonal(W, 0)

pattern_A = patterns[0]
# pattern_B = np.random.choice([-1, 1], size=N)

# W = (1/N) * (outer product of A with itself + outer product of B with itself)
# W = (1/N) * (np.outer(pattern_A, pattern_A) + np.outer(pattern_B, pattern_B))
# np.fill_diagonal(W, 0)  # no self connections

# print(W)
np.savetxt('W.txt', W.round(3), fmt='%.3f')

energies = []

# then update repeatedly until convergence
# for iteration in range(20):
#     energies.append(energy(state, W))
#     new_state = np.sign(W @ state)
#     if np.array_equal(new_state, state):
#         break
#     state = new_state

# print('retrieved:', state)
# print('pattern A:', pattern_A)
# print('match:', np.array_equal(state, pattern_A))

# for noise_level in [0.1, 0.2, 0.3, 0.4, 0.5]:
#     successes = 0
#     for trial in range(20):
#         state = pattern_A.copy()
#         noise_mask = np.random.rand(N) < noise_level
#         state[noise_mask] *= -1

#         for iteration in range(20):
#             new_state = np.sign(W @ state)
#             if np.array_equal(new_state, state):
#                 break
#             state = new_state

#         if np.array_equal(state, pattern_A):
#             successes += 1

#     print(f'noise {int(noise_level*100)}%: {successes}/20 correct')

# for n_patterns in [2, 5, 10, 15, 20]:
#     patterns = [np.random.choice([-1, 1], size=N) for _ in range(n_patterns)]
#     W = sum(np.outer(p, p) for p in patterns) / N
#     np.fill_diagonal(W, 0)

#     successes = 0
#     for trial in range(20):
#         state = patterns[0].copy()
#         noise_mask = np.random.rand(N) < 0.2
#         state[noise_mask] *= -1

#         for iteration in range(20):
#             new_state = np.sign(W @ state)
#             if np.array_equal(new_state, state):
#                 break
#             state = new_state

#         if np.array_equal(state, patterns[0]):
#             successes += 1

#     print(f'{n_patterns} patterns: {successes}/20 correct')

plt.figure(figsize=(10, 5))
for trial in range(5):
    state = pattern_A.copy()
    noise_mask = np.random.rand(N) < 0.3
    state[noise_mask] *= -1

    energies = []
    for iteration in range(20):
        energies.append(energy(state, W))
        new_state = np.sign(W @ state)
        if np.array_equal(new_state, state):
            break
        state = new_state

    plt.plot(energies, marker='o')

plt.xlabel('iteration')
plt.ylabel('energy')
plt.title('Hopfield network energy during retrieval')
plt.show()