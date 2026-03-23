import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.metrics import confusion_matrix
import sys
from sklearn.decomposition import PCA
from matplotlib.animation import FuncAnimation


console_output_file = 'mnist_console_output.txt'
sys.stdout = open(console_output_file, 'w')


# CORE FUNCTIONS

def softmax(x, beta=1.0):
    e = np.exp(beta * (x - np.max(x)))
    return e / e.sum()


def hopfield_update(state, patterns, beta=1.0, k=20):
    similarities = (patterns @ state) / np.sqrt(state.shape[0])

    top_idx = np.argsort(similarities)[-k:]
    top_patterns = patterns[top_idx]
    top_sims = similarities[top_idx]

    weights = softmax(top_sims, beta)
    return top_patterns.T @ weights


def retrieve_with_trajectory(corrupted, patterns, beta=4.0, max_iters=40, tol=1e-6, k=20):
    state = corrupted.astype(float).copy()
    trajectory = [state.copy()]

    for _ in range(max_iters):
        new_state = hopfield_update(state, patterns, beta=beta, k=k)
        trajectory.append(new_state.copy())

        if np.linalg.norm(new_state - state) < tol:
            break

        state = 0.6 * state + 0.4 * new_state

    return state, trajectory


# CORRUPTION MODELS

def corrupt_pattern(x, noise_level=0.3):
    corrupted = x.copy()
    mask = np.random.rand(x.shape[0]) < noise_level
    corrupted[mask] *= -1
    return corrupted


def corrupt_block(x, block_size=10):
    img = x.reshape(28, 28).copy()
    i = np.random.randint(0, 28 - block_size)
    j = np.random.randint(0, 28 - block_size)
    img[i:i+block_size, j:j+block_size] = -1
    return img.flatten()


# ANALYSIS FUNCTIONS

def analyse_trajectory(trajectory, true_pattern, patterns):
    sims_to_true = []
    sims_to_all = []

    for state in trajectory:
        state_bin = np.sign(state)

        sim_true = np.mean(state_bin == true_pattern)
        sims_to_true.append(sim_true)

        sims = (patterns @ state_bin) / np.sqrt(state.shape[0])
        sims_to_all.append(np.max(sims))

    return sims_to_true, sims_to_all


def plot_convergence(sims_to_true, sims_to_all):
    plt.plot(sims_to_true, label='to true')
    plt.plot(sims_to_all, label='to memory')
    plt.xlabel('iteration')
    plt.ylabel('similarity')
    plt.legend()
    plt.title('Convergence')
    plt.show()


def convergence_steps(trajectory, tol=1e-3):
    for i in range(1, len(trajectory)):
        if np.linalg.norm(trajectory[i] - trajectory[i-1]) < tol:
            return i
    return len(trajectory)


# LABELING

def nearest_label(x, patterns, labels):
    similarities = (patterns @ x) / np.sqrt(x.shape[0])
    return labels[np.argmax(similarities)]


# PROTOTYPES

def make_prototypes(X_binary, y, digit, n_prototypes=5, samples_per_proto=10):
    digit_imgs = X_binary[y == digit][:n_prototypes * samples_per_proto]
    groups = digit_imgs.reshape(n_prototypes, samples_per_proto, -1)
    prototypes = np.sign(groups.mean(axis=1))
    prototypes[prototypes == 0] = 1
    return prototypes


# EVALUATION

def evaluate(X_test, y_test, patterns, pattern_labels, beta=4.0, noise_level=0.3, k=20):
    preds = []
    pixel_overlaps = []

    failure_cases = []
    correct_cases = []

    for i, (x, true_label) in enumerate(zip(X_test, y_test)):
        corrupted = corrupt_pattern(x, noise_level)
        state, trajectory = retrieve_with_trajectory(corrupted, patterns, beta=beta, k=k)

        retrieved = np.sign(state)
        pred = nearest_label(retrieved, patterns, pattern_labels)

        preds.append(pred)

        overlap = np.mean(retrieved == x)
        pixel_overlaps.append(overlap)

        # DEBUG CASES
        if pred == true_label:
            correct_cases.append((int(i), int(true_label), int(pred)))
        if pred != true_label:
            failure_cases.append((int(i), int(true_label), int(pred)))
            print(f'FAIL: idx={i}, true={true_label}, pred={pred}')
            print('Convergence steps:', convergence_steps(trajectory))

    preds = np.array(preds)

    digit_accuracy = np.mean(preds == y_test)
    pixel_accuracy = np.mean(pixel_overlaps)
    cm = confusion_matrix(y_test, preds)

    return digit_accuracy, pixel_accuracy, cm, failure_cases, correct_cases


def noise_sweep(X_test, y_test, patterns, pattern_labels):
    noise_levels = [0.1, 0.3, 0.5, 0.7]
    accs = []

    for nl in noise_levels:
        acc, _, _ = evaluate(X_test, y_test, patterns, pattern_labels, noise_level=nl)
        accs.append(acc)
        print(f'Noise {nl}: {acc}')

    plt.plot(noise_levels, accs, marker='o')
    plt.xlabel('noise')
    plt.ylabel('accuracy')
    plt.title('Robustness')
    plt.show()


# VISUALS

def show_trajectory(x, patterns, beta=4.0, noise_level=0.3, k=20):
    corrupted = corrupt_pattern(x, noise_level)
    final_state, trajectory = retrieve_with_trajectory(corrupted, patterns, beta=beta, k=k)

    sims_true, sims_all = analyse_trajectory(trajectory, x, patterns)
    plot_convergence(sims_true, sims_all)

    steps = len(trajectory)
    fig, axes = plt.subplots(1, steps+1, figsize=(2*(steps+1),2))

    axes[0].imshow(x.reshape(28,28), cmap='gray')
    axes[0].set_title('original')
    axes[0].axis('off')

    for i, state in enumerate(trajectory):
        axes[i+1].imshow(np.sign(state).reshape(28,28), cmap='gray')
        axes[i+1].set_title(f'step {i}')
        axes[i+1].axis('off')
    
    pred = nearest_label(np.sign(final_state), patterns, pattern_labels)
    print('Predicted:', pred)

    plt.show()


# MAIN

print('Loading MNIST...')

mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X = mnist.data
y = mnist.target.astype(int)

X_binary = np.where(X > 127, 1.0, -1.0)

n_prototypes = 5
samples_per_proto = 10

patterns = np.vstack([
    make_prototypes(X_binary, y, d, n_prototypes, samples_per_proto)
    for d in range(10)
])

pattern_labels = np.hstack([
    [d]*n_prototypes for d in range(10)
])

print('Stored memories:', len(patterns))

test_per_digit = 50

X_test = np.vstack([
    X_binary[y == d][n_prototypes*samples_per_proto:
                    n_prototypes*samples_per_proto + test_per_digit]
    for d in range(10)
])

y_test = np.hstack([
    [d]*test_per_digit for d in range(10)
])


# THESE ARE THE THREE FUNCTIONS
# FOR EXPERIMENTS

MODE = 'trajectory' # or 'evaluate', 'noise', 'trajectory'

if MODE == 'evaluate':
    # How good is it?
    # accuracy, confusion matrix, overall behaviour

    acc, pix, cm, failure_cases, correct_cases = evaluate(
        X_test,
        y_test,
        patterns,
        pattern_labels,
        noise_level=0.3
    )

    print('First 10 correct cases:', correct_cases[:10])
    print('First 10 failure cases:', failure_cases[:10])

    print('\nDigit accuracy:', acc)
    print('Pixel overlap:', pix)
    print('\nConfusion matrix:\n', cm)

elif MODE == 'noise':
    # How does it break?
    # perfomance vs noise, robustness curve

    noise_sweep(X_test, y_test, patterns, pattern_labels)

elif MODE == 'trajectory':
    # What is it doing internally?
    # convergence behaviour, attractor dynamics, failure analysis

    show_trajectory(X_test[165], patterns)

sys.stdout.close()