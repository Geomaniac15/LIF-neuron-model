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
corrupted = X_binary[seven_idx].copy()
corrupted[784//2:] = -1

state = corrupted.copy()
for iteration in range(20):
    new_state = hopfield_update(state, patterns, beta=beta)
    if np.allclose(new_state, state, atol=1e-6):
        break
    state = new_state

fig, axes = plt.subplots(1, 3, figsize=(9, 3))
axes[0].imshow(X_binary[seven_idx].reshape(28, 28), cmap='gray')
axes[0].set_title('original (7)')
axes[0].axis('off')
axes[1].imshow(corrupted.reshape(28, 28), cmap='gray')
axes[1].set_title('corrupted')
axes[1].axis('off')
axes[2].imshow(state.reshape(28, 28), cmap='gray')
axes[2].set_title('retrieved')
axes[2].axis('off')
plt.suptitle('Hopfield retrieval: 100 patterns, 10 per digit')
plt.show()