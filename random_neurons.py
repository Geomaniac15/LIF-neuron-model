'''E-I LIF network with STDP. Refactored for headless / batch runs on the Lancaster HEC.

Local quick run:
    python random_neurons.py --N 100 --warmup 2000 --steps 3000 --seed 0 --outdir results

Big run on the cluster (see hec/run_random_neurons.slurm):
    python random_neurons.py --N 2000 --warmup 5000 --steps 20000 --seed 0 --outdir results
'''
import argparse
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # headless: no display needed on compute nodes
import matplotlib.pyplot as plt


def run(N=100, warmup=2000, steps=3000, dt=0.1, seed=0,
        exc_frac=0.8, connectivity=0.05, save_spikes=True):
    '''Run the E-I network. Returns (W_initial, W_final, spikes).

    spikes is an (M, 2) array of (time_ms, neuron_index). Pass save_spikes=False
    for very large runs where the spike list itself becomes the memory hog.
    '''
    rng = np.random.default_rng(seed)
    n_exc = int(round(exc_frac * N))

    V = np.full(N, -70.0)
    I_ext = rng.uniform(1.4, 1.6, N)
    g_exc = np.zeros(N)
    g_inh = np.zeros(N)

    E_exc, E_inh = 0.0, -80.0
    tau_m = rng.uniform(15, 25, N)
    tau_syn = 5.0
    R = 10.0
    V_rest = -70.0
    V_threshold = rng.uniform(-57, -53, N)
    V_reset = -80.0

    W = rng.random((N, N)) * 0.5
    mask = rng.random((N, N)) > (1.0 - connectivity)
    W = W * mask
    np.fill_diagonal(W, 0)
    W[n_exc:, :] *= -1  # inhibitory neurons have negative outgoing weights
    refractory = np.zeros(N)

    tau_stdp = 40.0
    A_plus = 0.01
    A_minus = 0.01
    last_spike = np.full(N, -np.inf)

    # ---- warmup (no STDP) ----
    for step in range(warmup):
        I_syn = g_exc * (V - E_exc) + g_inh * (V - E_inh)
        dV = (1.0 / tau_m) * (-(V - V_rest) + R * (I_ext + I_syn))
        V += dV * dt
        refractory -= dt
        spikes = (V >= V_threshold) & (refractory <= 0)
        V[spikes] = V_reset
        refractory[spikes] = 2.0

        # vectorised synaptic update (order within a step doesn't matter here)
        exc_fired = spikes.copy()
        exc_fired[n_exc:] = False
        inh_fired = spikes.copy()
        inh_fired[:n_exc] = False
        if exc_fired.any():
            g_exc += W[exc_fired].sum(axis=0)
        if inh_fired.any():
            g_inh += W[inh_fired].sum(axis=0)

        g_exc -= (g_exc / tau_syn) * dt
        g_inh -= (g_inh / tau_syn) * dt

    W_initial = W.copy()
    spike_record = [] if save_spikes else None

    # ---- main run with STDP ----
    # NOTE: kept as an explicit loop because STDP updates within a step are
    # order-dependent (last_spike is mutated as we iterate).
    for step in range(steps):
        I_syn = g_exc * (V - E_exc) + g_inh * (V - E_inh)
        dV = (1.0 / tau_m) * (-(V - V_rest) + R * (I_ext + I_syn))
        V += dV * dt
        refractory -= dt
        spikes = (V >= V_threshold) & (refractory <= 0)
        V[spikes] = V_reset
        refractory[spikes] = 2.0

        t_now = step * dt
        for i in np.where(spikes)[0]:
            if i < n_exc:
                g_exc += W[i, :]
            else:
                g_inh += W[i, :]

            if save_spikes:
                spike_record.append((t_now, i))

            last_spike[i] = t_now
            delta_t = t_now - last_spike
            W[:, i] += A_plus * np.exp(-delta_t / tau_stdp) * (W[:, i] != 0)
            W[i, :] -= A_minus * np.exp(-delta_t / tau_stdp) * (W[i, :] != 0)
            np.clip(W, -1.0, 1.0, out=W)

        g_exc -= (g_exc / tau_syn) * dt
        g_inh -= (g_inh / tau_syn) * dt

    spikes_arr = np.array(spike_record) if save_spikes else np.empty((0, 2))
    return W_initial, W, spikes_arr


def main():
    p = argparse.ArgumentParser(description='E-I LIF network with STDP.')
    p.add_argument('--N', type=int, default=100, help='number of neurons')
    p.add_argument('--warmup', type=int, default=2000, help='warmup steps (no STDP)')
    p.add_argument('--steps', type=int, default=3000, help='main run steps (with STDP)')
    p.add_argument('--dt', type=float, default=0.1, help='step size (ms)')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--exc-frac', type=float, default=0.8)
    p.add_argument('--connectivity', type=float, default=0.05)
    p.add_argument('--outdir', type=str, default='results')
    p.add_argument('--no-spikes', action='store_true',
                   help='skip spike recording (saves memory for huge runs)')
    p.add_argument('--no-plots', action='store_true',
                   help='skip rendering figures, only save .npz')
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    tag = f'N{args.N}_seed{args.seed}'
    print(f'[run] N={args.N} warmup={args.warmup} steps={args.steps} '
          f'dt={args.dt} seed={args.seed} outdir={args.outdir}', flush=True)

    W_initial, W, spikes = run(
        N=args.N, warmup=args.warmup, steps=args.steps, dt=args.dt,
        seed=args.seed, exc_frac=args.exc_frac, connectivity=args.connectivity,
        save_spikes=not args.no_spikes,
    )

    npz_path = os.path.join(args.outdir, f'run_{tag}.npz')
    np.savez_compressed(npz_path, W_initial=W_initial, W_final=W, spikes=spikes,
                        N=args.N, seed=args.seed, dt=args.dt,
                        warmup=args.warmup, steps=args.steps)
    print(f'[save] {npz_path} ({spikes.shape[0]} spikes)', flush=True)

    if args.no_plots:
        return

    if spikes.size:
        plt.figure(figsize=(12, 5))
        plt.scatter(spikes[:, 0], spikes[:, 1], s=2, color='teal')
        plt.xlabel('time (ms)')
        plt.ylabel('neuron index')
        plt.title(f'spike raster (N={args.N}, seed={args.seed})')
        plt.tight_layout()
        plt.savefig(os.path.join(args.outdir, f'raster_{tag}.png'), dpi=150)
        plt.close()

    plt.figure(figsize=(12, 4))
    plt.hist(W_initial[W_initial != 0].flatten(), bins=50, alpha=0.5,
             label='before', color='steelblue')
    plt.hist(W[W != 0].flatten(), bins=50, alpha=0.5,
             label='after', color='teal')
    plt.xlabel('synaptic weight')
    plt.ylabel('count')
    plt.title(f'weight distribution before vs after STDP (N={args.N})')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, f'weights_{tag}.png'), dpi=150)
    plt.close()

    print(f'[done] figures saved to {args.outdir}/', flush=True)


if __name__ == '__main__':
    main()
