'''Intrinsic reservoir benchmark for a trained LIF 'brain' (no recorded data needed).

Drives the trained network with synthetic input streams and measures the two
properties that decide whether it is a useful reservoir at all:

  1. MEMORY CAPACITY (Jaeger 2001). Inject an i.i.d. random scalar stream, then
     ask a linear readout to reconstruct that input delayed by k reservoir-steps,
     for k = 1..max_delay. MC_k = R^2 of the best linear reconstruction on held-out
     data; total MC = sum_k MC_k. This is how many past inputs the network still
     holds in its state. Generated on the fly, so the cat/cap data scarcity that
     sank the speech task is irrelevant here.

  2. PERTURBATION DIVERGENCE (edge of chaos / fading memory). Run the reservoir
     twice on input streams that are identical except for a single perturbed value
     at t=0, and watch the Euclidean distance between the two state trajectories.
       - decays back to ~0  -> fading-memory / echo-state regime (ordered, usable)
       - stays flat          -> critical-ish (often the sweet spot)
       - grows               -> chaotic; small input differences blow up, readouts
                                can't generalise.

  3. SEPARATION (Legenstein & Maass). Feed many distinct random streams and measure
     the average pairwise distance of the resulting states. High separation + low
     divergence is the goal: tell different inputs apart without being chaotic.

The LIF dynamics here are copied verbatim from speech_reservoir_fixed.py
(conductance synapses, fixed I_syn sign, frozen background drive, optional
inh-scale) so the brain behaves identically to the speech pipeline.

Example (tuned N=2000 network):
    python reservoir_capacity.py \
        --weights results_fixed/run_N2000_seed0.npz \
        --input-scale 4 --plot capacity_N2000.png
(defaults: hold=150 steps=40000 max-delay=15, a few minutes on N=2000.)
'''
import argparse
import numpy as np
from sklearn.linear_model import Ridge

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--weights', default='results_fixed/run_N500_seed0.npz',
                   help='path to a sign-fixed run_*.npz (W_final + N)')
    p.add_argument('--input-scale', type=float, default=4.0,
                   help='how hard the input stream drives the input neuron block. '
                        'The network self-stabilises via strong inhibition, so weak '
                        'input is washed out; use the same value that worked elsewhere.')
    p.add_argument('--background-mean', type=float, default=1.25)
    p.add_argument('--background-spread', type=float, default=0.1)
    p.add_argument('--inh-scale', type=float, default=1.0,
                   help='extra multiplier on inhibitory rows (1.0 = as trained).')
    p.add_argument('--seed', type=int, default=0,
                   help='seed for frozen neuron params, background drive, and input.')
    p.add_argument('--n-input', type=int, default=100,
                   help='number of input neurons the stream is injected into.')
    p.add_argument('--hold', type=int, default=150,
                   help='simulation steps each input value is held for. One '
                        '"reservoir step" = hold * dt ms; memory is measured in '
                        'these units. Hold must be >= the membrane time constant '
                        '(~20 ms = 200 steps) or each input is smeared away before '
                        'the network can register it. 150 steps @ dt=0.1ms = 15 ms.')
    p.add_argument('--steps', type=int, default=40000,
                   help='total simulation steps for the memory-capacity run. With '
                        'hold=150 this is ~266 reservoir steps. (A few minutes on '
                        'N=2000; drop it for a faster first look.)')
    p.add_argument('--washout', type=int, default=30,
                   help='reservoir steps discarded at the start (transient).')
    p.add_argument('--max-delay', type=int, default=15,
                   help='largest delay k tested in the memory-capacity sum.')
    p.add_argument('--tau-state', type=float, default=20.0,
                   help='time constant (ms) of the leaky spike trace used as the '
                        'continuous reservoir state for the linear readout.')
    p.add_argument('--n-state', type=int, default=600,
                   help='number of readout neurons subsampled into the state vector '
                        '(keeps the ridge solve fast on big networks).')
    p.add_argument('--ridge-alpha', type=float, default=1.0)
    p.add_argument('--sep-streams', type=int, default=12,
                   help='number of distinct random streams for the separation metric.')
    p.add_argument('--sep-steps', type=int, default=12000,
                   help='simulation steps per stream for divergence/separation.')
    p.add_argument('--plot', default='reservoir_capacity.png',
                   help='output PNG path for the MC and divergence curves.')
    p.add_argument('--dt', type=float, default=0.1)
    # --- separation sweep mode ---
    p.add_argument('--sweep', action='store_true',
                   help='instead of one full report, sweep input-scale x inh-scale '
                        'and report MC_1, separation and regime per cell. Finds where '
                        'separation climbs into a usable range. Synthetic data, so '
                        'fast and noise-free.')
    p.add_argument('--sweep-input-scales', default='4,8,16,24',
                   help='comma-separated input-scale grid for --sweep.')
    p.add_argument('--sweep-inh-scales', default='1.0,0.6',
                   help='comma-separated inh-scale grid for --sweep.')
    p.add_argument('--sweep-steps', type=int, default=18000,
                   help='MC-run length per sweep cell (shorter than the full run).')
    p.add_argument('--sweep-sep-steps', type=int, default=6000,
                   help='separation/divergence run length per sweep cell.')
    p.add_argument('--sweep-streams', type=int, default=6,
                   help='number of streams for separation per sweep cell.')
    return p.parse_args()


def classify_regime(dist):
    '''Given a perturbation-distance trajectory, return (end/peak ratio, label).'''
    peak_i = int(np.argmax(dist))
    peak_d = dist[peak_i]
    end_d = dist[-5:].mean()
    if peak_d < 1e-6:
        return 0.0, 'INPUT BARELY PERTURBS STATE (too weak / washed out)'
    ratio = end_d / peak_d
    tail = dist[peak_i:]
    slope = np.polyfit(np.arange(len(tail)), tail, 1)[0] if len(tail) > 3 else 0.0
    if slope > 1e-4 and end_d >= peak_d:
        return ratio, 'CHAOTIC (perturbation grows)'
    if ratio < 0.3:
        return ratio, 'FADING MEMORY (ordered, usable)'
    return ratio, 'CRITICAL-ish (perturbation persists)'


def pick_readout(params, args):
    '''Deterministically subsample readout neurons into the state vector.'''
    rng = np.random.default_rng(args.seed + 3)
    pool = np.arange(args.n_input, params['N'])
    n_state = min(args.n_state, len(pool))
    return np.sort(rng.choice(pool, size=n_state, replace=False))


def build_network(args):
    '''Load weights and set up the frozen LIF parameters, mirroring
    speech_reservoir_fixed.py exactly.'''
    data = np.load(args.weights)
    W = data['W_final'].copy()
    N = int(data['N'])
    n_exc = int(0.8 * N)
    if args.inh_scale != 1.0:
        W[n_exc:, :] *= args.inh_scale
        print(f'  applied inh_scale={args.inh_scale} to inhibitory rows (>= {n_exc})')

    rng = np.random.default_rng(args.seed)
    params = dict(
        N=N, n_exc=n_exc, W=W,
        tau_m=rng.uniform(15, 25, N),
        tau_syn=5.0, R=10.0,
        V_rest=-70.0,
        V_threshold=rng.uniform(-57, -53, N),
        V_reset=-80.0,
        E_exc=0.0, E_inh=-80.0,
        I_ext=rng.uniform(args.background_mean - args.background_spread,
                          args.background_mean + args.background_spread, N),
    )
    return params


def simulate(params, drive, args, record_every, readout_idx):
    '''Run the conductance-based LIF network.

    drive: (steps, N) array of external current added each step (already includes
           the input injection; background is added internally).
    Returns a (n_records, len(readout_idx)) array: a leaky spike trace ('state')
    sampled every record_every steps. The trace is the same low-pass a downstream
    synapse would see, which gives the linear readout something continuous to fit.'''
    N = params['N']; n_exc = params['n_exc']; W = params['W']
    tau_m = params['tau_m']; tau_syn = params['tau_syn']; R = params['R']
    V_rest = params['V_rest']; V_th = params['V_threshold']; V_reset = params['V_reset']
    E_exc = params['E_exc']; E_inh = params['E_inh']; I_ext = params['I_ext']
    dt = args.dt
    steps = drive.shape[0]

    V = np.full(N, -70.0)
    g_exc = np.zeros(N)
    g_inh = np.zeros(N)
    refractory = np.zeros(N)
    trace = np.zeros(N)                 # leaky spike trace = continuous state
    trace_decay = np.exp(-dt / args.tau_state)

    n_records = steps // record_every
    states = np.zeros((n_records, len(readout_idx)))
    rec = 0

    for step in range(steps):
        current_I = I_ext + drive[step]

        I_syn = -g_exc * (V - E_exc) - g_inh * (V - E_inh)
        dV = (1.0 / tau_m) * (-(V - V_rest) + R * (current_I + I_syn))
        V += dV * dt
        refractory -= dt

        spikes = (V >= V_th) & (refractory <= 0)
        V[spikes] = V_reset
        refractory[spikes] = 2.0

        trace *= trace_decay
        trace[spikes] += 1.0

        exc_fired = spikes.copy(); exc_fired[n_exc:] = False
        inh_fired = spikes.copy(); inh_fired[:n_exc] = False
        if exc_fired.any():
            g_exc += W[exc_fired].sum(axis=0)
        if inh_fired.any():
            g_inh += W[inh_fired].sum(axis=0)
        g_exc -= (g_exc / tau_syn) * dt
        g_inh -= (g_inh / tau_syn) * dt

        if (step + 1) % record_every == 0 and rec < n_records:
            states[rec] = trace[readout_idx]
            rec += 1

    return states[:rec]


def make_drive(u_stream, hold, N, n_input, input_scale):
    '''Expand a per-reservoir-step input stream into a (steps, N) current array by
    holding each value for `hold` sim steps and injecting it into the input block.'''
    steps = len(u_stream) * hold
    drive = np.zeros((steps, N))
    inj = np.repeat(u_stream, hold) * input_scale       # (steps,)
    drive[:, :n_input] = inj[:, None]
    return drive


def memory_capacity(params, args, readout_idx):
    '''Inject i.i.d. uniform input; fit linear readouts to reconstruct delayed
    input. Returns (per_delay_mc, total_mc).'''
    N = params['N']
    n_rsteps = args.steps // args.hold
    rng = np.random.default_rng(args.seed + 7)
    u = rng.uniform(-1.0, 1.0, n_rsteps)

    drive = make_drive(u, args.hold, N, args.n_input, args.input_scale)
    states = simulate(params, drive, args, record_every=args.hold,
                      readout_idx=readout_idx)
    m = min(len(states), len(u))
    states = states[:m]; u = u[:m]

    # drop washout, then split train/test in time (no shuffling: respect causality)
    w = args.washout
    X = states[w:]
    n = len(X)
    split = n // 2

    per_delay = []
    for k in range(1, args.max_delay + 1):
        target = u[w - k: w - k + n] if w - k >= 0 else None
        if target is None or len(target) != n:
            # build target aligned to X[t] = u[(w+t) - k]
            idx = np.arange(w, w + n) - k
            if idx[0] < 0:
                per_delay.append(0.0)
                continue
            target = u[idx]
        Xtr, Xte = X[:split], X[split:]
        ytr, yte = target[:split], target[split:]
        reg = Ridge(alpha=args.ridge_alpha).fit(Xtr, ytr)
        pred = reg.predict(Xte)
        # MC_k = squared correlation between prediction and true delayed input
        if np.std(pred) < 1e-9 or np.std(yte) < 1e-9:
            mc_k = 0.0
        else:
            r = np.corrcoef(pred, yte)[0, 1]
            mc_k = float(max(0.0, r * r))
        per_delay.append(mc_k)

    return np.array(per_delay), float(np.sum(per_delay))


def divergence(params, args, readout_idx):
    '''Run on a base stream and a single-value-perturbed copy; return the L2
    distance between state trajectories over time (normalised per state dim).'''
    N = params['N']
    n_rsteps = args.sep_steps // args.hold
    rng = np.random.default_rng(args.seed + 11)
    u = rng.uniform(-1.0, 1.0, n_rsteps)
    u_pert = u.copy()
    u_pert[0] += 0.5            # one perturbed input value at the very start

    d0 = simulate(params, make_drive(u, args.hold, N, args.n_input, args.input_scale),
                  args, args.hold, readout_idx)
    d1 = simulate(params, make_drive(u_pert, args.hold, N, args.n_input, args.input_scale),
                  args, args.hold, readout_idx)
    m = min(len(d0), len(d1))
    dist = np.linalg.norm(d0[:m] - d1[:m], axis=1) / np.sqrt(d0.shape[1])
    return dist


def separation(params, args, readout_idx):
    '''Mean pairwise distance of final states across distinct random streams.'''
    N = params['N']
    n_rsteps = args.sep_steps // args.hold
    finals = []
    for s in range(args.sep_streams):
        rng = np.random.default_rng(args.seed + 100 + s)
        u = rng.uniform(-1.0, 1.0, n_rsteps)
        st = simulate(params, make_drive(u, args.hold, N, args.n_input, args.input_scale),
                      args, args.hold, readout_idx)
        # cap washout so short runs still leave states to average
        wo = min(args.washout, max(1, len(st) // 3))
        finals.append(st[wo:].mean(axis=0))   # mean state after washout
    finals = np.array(finals)
    dists = []
    for i in range(len(finals)):
        for j in range(i + 1, len(finals)):
            dists.append(np.linalg.norm(finals[i] - finals[j]))
    dists = np.array(dists) / np.sqrt(finals.shape[1])
    return float(dists.mean()), float(dists.std())


def run_sweep(args):
    '''Sweep input-scale x inh-scale; report MC_1, separation and regime per cell.'''
    input_scales = [float(x) for x in args.sweep_input_scales.split(',') if x.strip()]
    inh_scales = [float(x) for x in args.sweep_inh_scales.split(',') if x.strip()]
    print(f'Sweeping input_scale={input_scales} x inh_scale={inh_scales}  '
          f'({len(input_scales) * len(inh_scales)} cells)')
    print(f'weights = {args.weights}\n')
    header = (f'{"input":>6} {"inh":>5} {"MC_1":>6} {"MC_2":>6} {"separation":>11} '
              f'{"end/peak":>9}  regime')
    print(header)
    print('-' * (len(header) + 10))

    rows = []
    for inh in inh_scales:
        args.inh_scale = inh
        params = build_network(args)
        readout_idx = pick_readout(params, args)
        for isc in input_scales:
            args.input_scale = isc
            # light MC (k=1,2 only)
            s_steps, s_md = args.steps, args.max_delay
            args.steps, args.max_delay = args.sweep_steps, 2
            per_delay, _ = memory_capacity(params, args, readout_idx)
            args.steps, args.max_delay = s_steps, s_md
            # light separation + divergence
            s_ss, s_ns = args.sep_steps, args.sep_streams
            args.sep_steps, args.sep_streams = args.sweep_sep_steps, args.sweep_streams
            sep_mean, sep_std = separation(params, args, readout_idx)
            dist = divergence(params, args, readout_idx)
            args.sep_steps, args.sep_streams = s_ss, s_ns
            ratio, regime = classify_regime(dist)
            print(f'{isc:>6.0f} {inh:>5.1f} {per_delay[0]:>6.3f} {per_delay[1]:>6.3f} '
                  f'{sep_mean:>11.4f} {ratio:>9.2f}  {regime.split(" (")[0]}')
            rows.append(dict(input_scale=isc, inh_scale=inh, mc1=per_delay[0],
                             mc2=per_delay[1], sep=sep_mean, ratio=ratio, regime=regime))

    usable = [r for r in rows if 'CHAOTIC' not in r['regime']]
    if usable:
        best = max(usable, key=lambda r: r['sep'])
        print(f'\nBest separation among non-chaotic cells: input_scale={best["input_scale"]:.0f} '
              f'inh_scale={best["inh_scale"]:.1f}  separation={best["sep"]:.3f} '
              f'MC_1={best["mc1"]:.3f} ({best["regime"].split(" (")[0]})')

    # plot separation vs input-scale, one line per inh-scale
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for inh in inh_scales:
        cells = [r for r in rows if r['inh_scale'] == inh]
        xs = [r['input_scale'] for r in cells]
        ax[0].plot(xs, [r['sep'] for r in cells], 'o-', label=f'inh={inh:.1f}')
        ax[1].plot(xs, [r['mc1'] for r in cells], 'o-', label=f'inh={inh:.1f}')
    ax[0].set_xlabel('input-scale'); ax[0].set_ylabel('separation (per dim)')
    ax[0].set_title('Separation vs drive'); ax[0].legend()
    ax[1].set_xlabel('input-scale'); ax[1].set_ylabel('MC_1')
    ax[1].set_title('Short-term memory vs drive'); ax[1].legend()
    fig.suptitle(f'{args.weights}  separation/MC sweep')
    fig.tight_layout(); fig.savefig(args.plot, dpi=120)
    print(f'\nPlot saved to {args.plot}')
    print('\nReading it: you want separation to climb (different inputs -> different')
    print('states) while the regime stays out of CHAOTIC. That cell is where this')
    print('brain becomes usable as a classifier substrate. If separation never rises,')
    print('the input block (100/2000 neurons) is too small a foothold: raise --n-input.')


def main():
    args = parse_args()
    if args.sweep:
        run_sweep(args)
        return
    print(f'Loading {args.weights} ...')
    params = build_network(args)
    N = params['N']
    print(f'  N={N}, input neurons={args.n_input}, input_scale={args.input_scale}, '
          f'hold={args.hold} ({args.hold * args.dt:.1f} ms/reservoir-step)')

    readout_idx = pick_readout(params, args)
    print(f'  state vector = {len(readout_idx)} readout neurons, '
          f'tau_state={args.tau_state} ms\n')

    print('Measuring memory capacity...')
    per_delay, total_mc = memory_capacity(params, args, readout_idx)
    print(f'  total memory capacity = {total_mc:.2f} reservoir-steps '
          f'({total_mc * args.hold * args.dt:.1f} ms of usable history)')
    print(f'  MC at k=1: {per_delay[0]:.3f}   MC falls below 0.1 by k='
          f'{next((i+1 for i, v in enumerate(per_delay) if v < 0.1), ">max")}')

    print('\nMeasuring perturbation divergence (edge of chaos)...')
    dist = divergence(params, args, readout_idx)
    ratio, regime = classify_regime(dist)
    print(f'  peak distance={dist.max():.4f}, end={dist[-5:].mean():.4f}, '
          f'end/peak={ratio:.2f}')
    print(f'  regime: {regime}')

    print('\nMeasuring separation...')
    sep_mean, sep_std = separation(params, args, readout_idx)
    print(f'  mean pairwise state distance = {sep_mean:.3f} +/- {sep_std:.3f}')

    # ---- plot ----
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ks = np.arange(1, len(per_delay) + 1)
    ax[0].bar(ks, per_delay, color='teal')
    ax[0].set_xlabel('delay k (reservoir-steps)')
    ax[0].set_ylabel('MC_k  (R^2 of delayed-input reconstruction)')
    ax[0].set_title(f'Memory capacity = {total_mc:.2f}')
    ax[0].set_ylim(0, 1.05)

    ax[1].plot(dist, color='darkorange')
    ax[1].set_xlabel('reservoir-step after perturbation')
    ax[1].set_ylabel('state distance (per dim)')
    ax[1].set_title(f'Perturbation divergence ({regime.split()[0]})')
    fig.suptitle(f'{args.weights}  |  input_scale={args.input_scale} '
                 f'inh_scale={args.inh_scale}')
    fig.tight_layout()
    fig.savefig(args.plot, dpi=120)
    print(f'\nPlot saved to {args.plot}')

    print('\nHow to read this:')
    print('  - Spiking reservoirs are lossy: MC_1 of a few tenths and a total MC of')
    print('    one or more reservoir-steps means the network genuinely holds recent')
    print('    input, the precondition for any temporal task. MC ~0 means it does not.')
    print('  - Divergence that DECAYS = fading memory (good). Growth = chaotic.')
    print('  - High separation with non-chaotic divergence is the usable sweet spot.')
    print('  - If MC is ~0 here, the speech failure was the network/injection, not')
    print('    the data. If MC is healthy, the speech failure was just data scarcity.')


if __name__ == '__main__':
    main()
