import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.metrics import confusion_matrix
import sys

# redirect stdout to file
console_output_file = 'mnist_console_output.txt'
sys.stdout = open(console_output_file, 'w')
print(f'writing to {console_output_file}')

# hopfield update

def softmax(x, beta=1.0):
    e = np.exp(beta * (x - np.max(x)))
    return e / e.sum()

def hopfield_update(state, patterns, beta=1.0):
    similarities = (patterns @ state) / np.sqrt(state.shape[0])
    weights = softmax(similarities, beta)
    return patterns.T @ weights


def retrieve(corrupted, patterns, beta=2.0, max_iters=20, tol=1e-6):
    state = corrupted.astype(float).copy()

    for _ in range(max_iters):
        new_state = hopfield_update(state, patterns, beta)

        # print('similarity to closest memory:', 
        #       np.max(patterns @ new_state))

        if np.linalg.norm(new_state - state) < tol:
            break

        state = new_state

    return state


# noise model

def corrupt_pattern(x, noise_level=0.3):
    corrupted = x.copy()
    mask = np.random.rand(x.shape[0]) < noise_level
    corrupted[mask] *= -1
    return corrupted


# classification via nearest stored pattern

def nearest_label(x, patterns, labels):
    similarities = (patterns @ x) / np.sqrt(x.shape[0])
    idx = np.argmax(similarities)
    return labels[idx]


# evaluation loop

def evaluate(X_test, y_test, patterns, pattern_labels, beta=2.0, noise_level=0.3):

    preds = []
    pixel_overlaps = []

    for x, true_label in zip(X_test, y_test):

        corrupted = corrupt_pattern(x, noise_level)

        state = retrieve(corrupted, patterns, beta)

        retrieved = np.sign(state)

        pred = nearest_label(retrieved, patterns, pattern_labels)

        preds.append(pred)

        overlap = np.mean(retrieved == x)
        pixel_overlaps.append(overlap)

    preds = np.array(preds)

    digit_accuracy = np.mean(preds == y_test)
    pixel_accuracy = np.mean(pixel_overlaps)

    cm = confusion_matrix(y_test, preds)

    return digit_accuracy, pixel_accuracy, cm


# visualisation

def show_examples(X_test, patterns, pattern_labels, beta=2.0, noise_level=0.3, n=5):

    fig, axes = plt.subplots(n,3, figsize=(6,2*n))

    for i in range(n):

        x = X_test[i]

        corrupted = corrupt_pattern(x, noise_level)

        retrieved = retrieve(corrupted, patterns, beta)

        axes[i,0].imshow(x.reshape(28,28), cmap='gray')
        axes[i,0].set_title('original')

        axes[i,1].imshow(corrupted.reshape(28,28), cmap='gray')
        axes[i,1].set_title('corrupted')

        axes[i,2].imshow(retrieved.reshape(28,28), cmap='gray')
        axes[i,2].set_title('retrieved')

        for ax in axes[i]:
            ax.axis('off')

    plt.tight_layout()
    plt.show()


# load MNIST

print('Loading MNIST...')

mnist = fetch_openml('mnist_784', version=1, as_frame=False)

X = mnist.data
y = mnist.target.astype(int)

# binarise
X_binary = np.where(X > 127, 1.0, -1.0)


# memory setup

memory_sizes = [1, 2, 5, 10, 20, 50]

memory_sizes = [1,2,5,10,20,50]

results = []

for stored_per_digit in memory_sizes:

    patterns = np.vstack([
        X_binary[y == d][:stored_per_digit]
        for d in range(10)
    ])

    pattern_labels = np.hstack([
        [d]*stored_per_digit
        for d in range(10)
    ])

    X_test = np.vstack([
        X_binary[y == d][stored_per_digit:stored_per_digit+50]
        for d in range(10)
    ])

    y_test = np.hstack([
        [d]*50
        for d in range(10)
    ])

    acc, pix, _ = evaluate(
        X_test,
        y_test,
        patterns,
        pattern_labels,
        beta=4.0,
        noise_level=0.3
    )

    print(
        f"stored_per_digit={stored_per_digit} "
        f"total_memories={len(patterns)} "
        f"accuracy={acc:.3f}"
    )

    results.append((len(patterns), acc))


# hyperparameter sweep

# betas = [0.5, 1.0, 2.0, 4.0]
# noise_levels = [0.1, 0.3, 0.5, 0.7]

# for beta in betas:

#     for noise in noise_levels:

#         acc, pix, _ = evaluate(X_test, y_test, patterns, pattern_labels,
#                                beta=beta, noise_level=noise)

#         print(
#             f'beta={beta:.1f}  noise={noise:.1f}  digit_acc={acc:.3f}  pixel_overlap={pix:.3f}'
#         )


# confusion matrix

acc, pix, cm = evaluate(X_test, y_test, patterns, pattern_labels,
                        beta=2.0, noise_level=0.3)

print('\nFinal accuracy:', acc)
print('\nConfusion matrix:\n', cm)


# example reconstructions

show_examples(X_test, patterns, pattern_labels, beta=2.0, noise_level=0.3)
sys.stdout.close()

memories = [r[0] for r in results]
accuracies = [r[1] for r in results]

plt.figure()
plt.plot(memories, accuracies, marker='o')
plt.xlabel("Number of stored patterns")
plt.ylabel("Digit classification accuracy")
plt.title("Hopfield memory capacity experiment")
plt.show()