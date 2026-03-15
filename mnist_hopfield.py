import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.metrics import confusion_matrix
import sys
from sklearn.decomposition import PCA


console_output_file = 'mnist_console_output.txt'
sys.stdout = open(console_output_file, 'w')


# softmax

def softmax(x, beta=1.0):
    e = np.exp(beta * (x - np.max(x)))
    return e / e.sum()


# hopfield update (top-k attention)

def hopfield_update(state, patterns, beta=1.0, k=20):

    similarities = (patterns @ state) / np.sqrt(state.shape[0])

    top_idx = np.argsort(similarities)[-k:]
    top_patterns = patterns[top_idx]
    top_sims = similarities[top_idx]

    weights = softmax(top_sims, beta)

    return top_patterns.T @ weights


# retrieval dynamics

def retrieve_with_trajectory(corrupted, patterns, beta=4.0, max_iters=10, tol=1e-6, k=20):

    state = corrupted.astype(float).copy()
    trajectory = [state.copy()]

    for _ in range(max_iters):

        new_state = hopfield_update(state, patterns, beta=beta, k=k)

        trajectory.append(new_state.copy())

        if np.linalg.norm(new_state - state) < tol:
            break

        state = new_state

    return state, trajectory


# noise model

def corrupt_pattern(x, noise_level=0.3):

    corrupted = x.copy()

    mask = np.random.rand(x.shape[0]) < noise_level

    corrupted[mask] *= -1

    return corrupted


# nearest stored label

def nearest_label(x, patterns, labels):

    similarities = (patterns @ x) / np.sqrt(x.shape[0])

    idx = np.argmax(similarities)

    return labels[idx]


# prototype creation

def make_prototypes(X_binary, y, digit, n_prototypes=5, samples_per_proto=10):

    digit_imgs = X_binary[y == digit][:n_prototypes * samples_per_proto]

    groups = digit_imgs.reshape(n_prototypes, samples_per_proto, -1)

    prototypes = np.sign(groups.mean(axis=1))

    prototypes[prototypes == 0] = 1

    return prototypes


# evaluation

def evaluate(X_test, y_test, patterns, pattern_labels, beta=4.0, noise_level=0.3, k=20):

    preds = []
    pixel_overlaps = []

    for x, true_label in zip(X_test, y_test):

        corrupted = corrupt_pattern(x, noise_level)

        state, _ = retrieve_with_trajectory(corrupted, patterns, beta=beta, k=k)

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

def show_examples(X_test, patterns, pattern_labels, beta=4.0, noise_level=0.3, k=20, n=5):

    fig, axes = plt.subplots(n,3, figsize=(6,2*n))

    for i in range(n):

        x = X_test[i]

        corrupted = corrupt_pattern(x, noise_level)

        state, _ = retrieve_with_trajectory(corrupted, patterns, beta=beta, k=k)

        retrieved = np.sign(state)

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


# visualisation

def show_trajectory(x, patterns, beta=4.0, noise_level=0.3, k=20):

    corrupted = corrupt_pattern(x, noise_level)

    final_state, trajectory = retrieve_with_trajectory(
        corrupted, patterns, beta=beta, k=k
    )

    steps = len(trajectory)

    fig, axes = plt.subplots(1, steps+1, figsize=(2*(steps+1),2))

    axes[0].imshow(x.reshape(28,28), cmap='gray')
    axes[0].set_title("original")
    axes[0].axis("off")

    for i, state in enumerate(trajectory):

        axes[i+1].imshow(np.sign(state).reshape(28,28), cmap='gray')
        axes[i+1].set_title(f"step {i}")
        axes[i+1].axis("off")

    plt.show()

def plot_pca_trajectory(x, patterns, pattern_labels, beta=4.0, noise_level=0.3, k=20):

    corrupted = corrupt_pattern(x, noise_level)

    final_state, trajectory = retrieve_with_trajectory(
        corrupted, patterns, beta=beta, k=k
    )

    # convert trajectory list → array
    traj = np.array(trajectory)

    # build PCA using stored memories
    pca = PCA(n_components=2)
    pca.fit(patterns)

    patterns_2d = pca.transform(patterns)
    traj_2d = pca.transform(traj)

    plt.figure(figsize=(6,6))

    # plot memory points
    for d in range(10):
        mask = pattern_labels == d
        plt.scatter(
            patterns_2d[mask,0],
            patterns_2d[mask,1],
            label=str(d),
            alpha=0.6
        )

    # plot trajectory
    plt.plot(traj_2d[:,0], traj_2d[:,1], 'k-o', label="trajectory")

    # start/end markers
    plt.scatter(traj_2d[0,0], traj_2d[0,1], c='red', s=100, label="start")
    plt.scatter(traj_2d[-1,0], traj_2d[-1,1], c='green', s=100, label="final")

    plt.legend()
    plt.title("Hopfield retrieval trajectory (PCA space)")
    plt.show()

# load MNIST

print("Loading MNIST...")

mnist = fetch_openml('mnist_784', version=1, as_frame=False)

X = mnist.data
y = mnist.target.astype(int)

X_binary = np.where(X > 127, 1.0, -1.0)


# build prototype memories

n_prototypes = 5
samples_per_proto = 10

patterns = np.vstack([
    make_prototypes(X_binary, y, d, n_prototypes, samples_per_proto)
    for d in range(10)
])

pattern_labels = np.hstack([
    [d]*n_prototypes
    for d in range(10)
])

print("Stored memories:", len(patterns))


# test set

test_per_digit = 50

X_test = np.vstack([
    X_binary[y == d][n_prototypes*samples_per_proto:
                    n_prototypes*samples_per_proto + test_per_digit]
    for d in range(10)
])

y_test = np.hstack([
    [d]*test_per_digit
    for d in range(10)
])


# evaluate

acc, pix, cm = evaluate(
    X_test,
    y_test,
    patterns,
    pattern_labels,
    beta=4.0,
    noise_level=0.3,
    k=20
)

print("\nDigit accuracy:", acc)
print("Pixel overlap:", pix)

print("\nConfusion matrix:\n", cm)


# example reconstructions

show_examples(
    X_test,
    patterns,
    pattern_labels,
    beta=4.0,
    noise_level=0.3,
    k=20
)

show_trajectory(X_test[0], patterns, beta=4.0, noise_level=0.3)

patterns = patterns / np.linalg.norm(patterns, axis=1, keepdims=True)
plot_pca_trajectory(X_test[0], patterns, pattern_labels)

sys.stdout.close()