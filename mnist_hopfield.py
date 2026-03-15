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

# get a single zero to corrupt
zero_patterns = X_binary[y == 0]
original = zero_patterns[0].copy()

# corrupt bottom half
corrupted = original.copy()
corrupted[784//2:] = -1

# retrieve from full MNIST
state = corrupted.copy()
beta = 2.0
for iteration in range(20):
    new_state = hopfield_update(state, X_binary, beta=beta)
    if np.allclose(new_state, state, atol=1e-6):
        print(f'converged at iteration {iteration}')
        break
    state = new_state

# find which stored zero is most similar to the retrieved state
similarities = X_binary @ state
most_similar_idx = np.argmax(similarities)
most_similar = X_binary[most_similar_idx]

fig, axes = plt.subplots(1, 4, figsize=(12, 3))
axes[0].imshow(original.reshape(28, 28), cmap='gray')
axes[0].set_title('original')
axes[0].axis('off')
axes[1].imshow(corrupted.reshape(28, 28), cmap='gray')
axes[1].set_title('corrupted')
axes[1].axis('off')
axes[2].imshow(state.reshape(28, 28), cmap='gray')
axes[2].set_title('retrieved')
axes[2].axis('off')
axes[3].imshow(most_similar.reshape(28, 28), cmap='gray')
axes[3].set_title(f'nearest neighbour\n(digit {y[most_similar_idx]})')
axes[3].axis('off')
plt.show()