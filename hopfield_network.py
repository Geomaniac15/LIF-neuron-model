import numpy as np

N = 100  # neurons

pattern_A = np.random.choice([-1, 1], size=N)
pattern_B = np.random.choice([-1, 1], size=N)

# W = (1/N) * (outer product of A with itself + outer product of B with itself)
W = (1/N) * (np.outer(pattern_A, pattern_A) + np.outer(pattern_B, pattern_B))
np.fill_diagonal(W, 0)  # no self connections

# print(W)
np.savetxt('W.txt', W.round(3), fmt='%.3f')

state = pattern_A.copy()
noise_mask = np.random.rand(N) < 0.2 # flip 20% bits
state[noise_mask] *= -1

# print('noisy input:', state)
# print('pattern A: ', pattern_A)

# then update repeatedly until convergence
# for iteration in range(20):
#     new_state = np.sign(W @ state)
#     if np.array_equal(new_state, state):
#         print(f'converged at iteration {iteration}')
#         break
#     state = new_state

# print('retrieved:', state)
# print('pattern A:', pattern_A)
# print('match:', np.array_equal(state, pattern_A))

for noise_level in [0.1, 0.2, 0.3, 0.4, 0.5]:
    successes = 0
    for trial in range(20):
        state = pattern_A.copy()
        noise_mask = np.random.rand(N) < noise_level
        state[noise_mask] *= -1

        for iteration in range(20):
            new_state = np.sign(W @ state)
            if np.array_equal(new_state, state):
                break
            state = new_state

        if np.array_equal(state, pattern_A):
            successes += 1

    print(f'noise {int(noise_level*100)}%: {successes}/20 correct')

for n_patterns in [2, 5, 10, 15, 20]:
    patterns = [np.random.choice([-1, 1], size=N) for _ in range(n_patterns)]
    W = sum(np.outer(p, p) for p in patterns) / N
    np.fill_diagonal(W, 0)

    successes = 0
    for trial in range(20):
        state = patterns[0].copy()
        noise_mask = np.random.rand(N) < 0.2
        state[noise_mask] *= -1

        for iteration in range(20):
            new_state = np.sign(W @ state)
            if np.array_equal(new_state, state):
                break
            state = new_state

        if np.array_equal(state, patterns[0]):
            successes += 1

    print(f'{n_patterns} patterns: {successes}/20 correct')