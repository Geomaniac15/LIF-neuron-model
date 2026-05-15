import numpy as np
import matplotlib.pyplot as plt


class SpikingAssociativeMemory:
    def __init__(
        self,
        n_neurons: int = 100,
        dt: float = 1.0,
        tau_mem: float = 20.0,
        tau_trace: float = 20.0,
        threshold: float = 1.0,
        reset_value: float = 0.0,
        a_plus: float = 0.015,
        a_minus: float = 0.01,
        weight_clip: float = 1.0,
        input_gain: float = 1.2,
        recurrent_gain: float = 0.3,
        leak_gain: float = 1.0,
        seed: int = 42,
    ) -> None:
        self.n = n_neurons
        self.dt = dt
        self.tau_mem = tau_mem
        self.tau_trace = tau_trace
        self.threshold = threshold
        self.reset_value = reset_value
        self.a_plus = a_plus
        self.a_minus = a_minus
        self.weight_clip = weight_clip
        self.input_gain = input_gain
        self.recurrent_gain = recurrent_gain
        self.leak_gain = leak_gain

        self.rng = np.random.default_rng(seed)

        self.W = 0.01 * self.rng.standard_normal((self.n, self.n))
        np.fill_diagonal(self.W, 0.0)

        self.V = np.zeros(self.n, dtype=float)
        self.pre_trace = np.zeros(self.n, dtype=float)
        self.post_trace = np.zeros(self.n, dtype=float)
        # FIX 2: store previous spikes for recurrent input
        self.last_spikes = np.zeros(self.n, dtype=float)

    def reset_state(self) -> None:
        self.V.fill(0.0)
        self.pre_trace.fill(0.0)
        self.post_trace.fill(0.0)
        self.last_spikes.fill(0.0)

    def pattern_to_drive(self, pattern: np.ndarray) -> np.ndarray:
        '''
        Convert {-1, +1} pattern into non-negative external drive.
        Active pixels get positive current, inactive get zero.
        '''
        return np.where(pattern > 0, 1.0, 0.0)

    def step(self, external_drive: np.ndarray) -> np.ndarray:
        '''
        One simulation step of a LIF-like recurrent spiking network.
        '''
        # FIX 2: use spikes from the previous timestep, not current V
        recurrent_input = self.W @ self.last_spikes

        drive = external_drive + 0.1

        dV = (
            -self.leak_gain * self.V
            + self.input_gain * drive
            + self.recurrent_gain * recurrent_input
        ) * (self.dt / self.tau_mem)

        self.V += dV

        spikes = (self.V >= self.threshold).astype(float)
        self.V[spikes > 0] = self.reset_value

        # store for next step
        self.last_spikes = spikes.copy()

        return spikes

    def update_traces(self, spikes: np.ndarray) -> None:
        # FIX 3: both traces updated with current spikes so weight updates
        # are symmetric, giving Hopfield-like associative memory
        decay = np.exp(-self.dt / self.tau_trace)
        self.pre_trace = self.pre_trace * decay + spikes
        self.post_trace = self.post_trace * decay + spikes

    def apply_stdp(self, spikes: np.ndarray) -> None:
        '''
        Pair-based STDP using traces.
        W[i,j] is the synapse from pre=j to post=i
        (because recurrent_input = W @ spikes).
        '''
        # FIX 1: correct outer product orientation
        # LTP: post i fires (spikes[i]=1), pre j had recent activity (pre_trace[j])
        # dW[i,j] += a_plus * spikes[i] * pre_trace[j]  =>  outer(spikes, pre_trace)
        dW_plus = self.a_plus * np.outer(spikes, self.pre_trace)

        # LTD: pre j fires (spikes[j]=1), post i had recent activity (post_trace[i])
        # dW[i,j] -= a_minus * post_trace[i] * spikes[j]  =>  outer(post_trace, spikes)
        dW_minus = self.a_minus * np.outer(self.post_trace, spikes)

        self.W += dW_plus
        self.W -= dW_minus

        # remove self-connections
        np.fill_diagonal(self.W, 0.0)

        # enforce symmetry for Hopfield-like retrieval
        self.W = 0.5 * (self.W + self.W.T)

        # clip for stability
        self.W = np.clip(self.W, -self.weight_clip, self.weight_clip)

    def train_pattern(
        self,
        pattern: np.ndarray,
        n_steps: int = 60,
        learn: bool = True,
    ) -> np.ndarray:
        '''
        Present a pattern for a number of steps.
        Returns spike counts during presentation.
        '''
        self.reset_state()
        drive = self.pattern_to_drive(pattern)
        spike_counts = np.zeros(self.n, dtype=float)

        for _ in range(n_steps):
            spikes = self.step(drive)
            self.update_traces(spikes)

            if learn:
                self.apply_stdp(spikes)

            spike_counts += spikes

        return spike_counts

    def retrieve(
        self,
        corrupted_pattern: np.ndarray,
        n_steps: int = 60,
        cue_steps: int = 15,
    ) -> tuple[np.ndarray, np.ndarray]:
        '''
        Retrieval:
        - first give a partial cue
        - then remove external input and let recurrence dominate
        Returns:
        - final retrieved {-1, +1} pattern
        - spike counts over the whole retrieval
        '''
        self.reset_state()
        cue_drive = self.pattern_to_drive(corrupted_pattern)
        zero_drive = np.zeros(self.n, dtype=float)

        spike_counts = np.zeros(self.n, dtype=float)

        for t in range(n_steps):
            drive = cue_drive if t < cue_steps else zero_drive
            spikes = self.step(drive)
            self.update_traces(spikes)
            spike_counts += spikes

        retrieved = np.where(spike_counts > 0, 1, -1)
        return retrieved, spike_counts


def make_random_patterns(
    n_patterns: int,
    n_neurons: int,
    sparsity: float = 0.3,
    seed: int = 123,
) -> np.ndarray:
    '''
    Create sparse binary patterns in {-1, +1}.
    sparsity = fraction of active (+1) neurons.
    '''
    rng = np.random.default_rng(seed)
    patterns = []
    for _ in range(n_patterns):
        active = rng.random(n_neurons) < sparsity
        pattern = np.where(active, 1, -1)
        patterns.append(pattern)
    return np.array(patterns)


def corrupt_pattern(
    pattern: np.ndarray,
    noise_level: float = 0.3,
    seed: int | None = None,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    corrupted = pattern.copy()
    mask = rng.random(pattern.shape[0]) < noise_level
    corrupted[mask] *= -1
    return corrupted


def similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(a == b))


def plot_pattern(vec: np.ndarray, title: str, side: int = 10) -> None:
    plt.imshow(vec.reshape(side, side), cmap='gray')
    plt.title(title)
    plt.axis('off')


def main() -> None:
    n_neurons = 100
    side = 10
    n_patterns = 3
    noise_level = 0.3

    net = SpikingAssociativeMemory(
        n_neurons=n_neurons,
        threshold=1.0,
        a_plus=0.01,
        a_minus=0.012,
        weight_clip=2.0,
        input_gain=1.4,
        recurrent_gain=0.6,
        seed=42,
    )

    patterns = make_random_patterns(
        n_patterns=n_patterns,
        n_neurons=n_neurons,
        sparsity=0.3,
        seed=7,
    )

    print('Training on patterns...')
    n_epochs = 20
    for epoch in range(n_epochs):
        for p_idx, pattern in enumerate(patterns):
            spike_counts = net.train_pattern(pattern, n_steps=60, learn=True)
        if epoch % 5 == 0 or epoch == n_epochs - 1:
            avg_abs_w = np.mean(np.abs(net.W))
            print(f'Epoch {epoch + 1}/{n_epochs}, mean |W| = {avg_abs_w:.4f}')

    # Test retrieval on each stored pattern
    print('\nTesting retrieval...')
    fig, axes = plt.subplots(n_patterns, 3, figsize=(8, 3 * n_patterns))

    if n_patterns == 1:
        axes = np.array([axes])

    for i, pattern in enumerate(patterns):
        corrupted = corrupt_pattern(pattern, noise_level=noise_level, seed=100 + i)
        retrieved, spike_counts = net.retrieve(corrupted, n_steps=60, cue_steps=15)

        sim_corrupted = similarity(pattern, corrupted)
        sim_retrieved = similarity(pattern, retrieved)

        print(
            f'Pattern {i}: '
            f'sim(original, corrupted)={sim_corrupted:.3f}, '
            f'sim(original, retrieved)={sim_retrieved:.3f}'
        )

        plt.sca(axes[i, 0])
        plot_pattern(pattern, f'Original {i}', side=side)

        plt.sca(axes[i, 1])
        plot_pattern(corrupted, f'Corrupted {i}', side=side)

        plt.sca(axes[i, 2])
        plot_pattern(retrieved, f'Retrieved {i}', side=side)

    plt.tight_layout()
    plt.show()

    # Visualise learned weights
    plt.figure(figsize=(6, 5))
    plt.imshow(net.W, cmap='coolwarm', aspect='auto')
    plt.colorbar(label='Weight')
    plt.title('Learned symmetric recurrent weights')
    plt.xlabel('Post neuron')
    plt.ylabel('Pre neuron')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()