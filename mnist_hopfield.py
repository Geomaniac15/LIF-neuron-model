import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from hopfield_with_softmax import hopfield_update

mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X = mnist.data
y = mnist.target.astype(int)

X_binary = np.where(X > 127, 1.0, -1.0)
# print(X_binary)

# get indices of digit 0
digit_indices = np.where(y == 0)[0]

# take first 5
patterns = X_binary[digit_indices[:5]]

# fig, axes = plt.subplots(1, 5, figsize=(12, 3))
# for i, ax in enumerate(axes):
#     ax.imshow(patterns[i].reshape(28, 28), cmap='gray')
#     ax.axis('off')
#     ax.set_title(f'pattern {i}')
# plt.suptitle('stored patterns (digit 0)')
# plt.show()

# store 10 examples of each digit
patterns = np.vstack([X_binary[y == d][:10] for d in range(10)])
print(f'stored {len(patterns)} patterns total')

N = 784
beta = 1 / np.sqrt(N)

# corrupt a 7 and retrieve
seven_idx = np.where(y == 7)[0][0]

fig, axes = plt.subplots(1, 4, figsize=(14, 3))
for idx, beta in enumerate([1.0, 2.0, 3.0, 5.0]):
    corrupted = X_binary[seven_idx].copy()
    corrupted[784//2:] = -1
    state = corrupted.copy()
    for iteration in range(20):
        new_state = hopfield_update(state, patterns, beta=beta)
        if np.allclose(new_state, state, atol=1e-6):
            break
        state = new_state
    axes[idx].imshow(state.reshape(28, 28), cmap='gray')
    axes[idx].set_title(f'beta={beta}')
    axes[idx].axis('off')
plt.suptitle('effect of beta on retrieval sharpness')
plt.show()