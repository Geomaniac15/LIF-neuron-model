'''Intrinsic reservoir benchmark for a trained LIF 'brain' (no recorded data needed).

Drives the trained network with synthetic input streams and measures the two
properties that decide whether it is a useful reservoir at all:

  1. MEMORY CAPACITY (Jaeger 2001). Inject an i.i.d. random scalar stream, then
     ask a linear readout to reconstruct that input delayed by k reservoir-steps,
     for k = 1..max_delay. MC_k = R^2 of the best linear reconstruction on held-out
     data; total MC = sum_k MC_k. This is how many past inputs the network still
     holds in its state. 

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
    p.add_argument('--tau-state2', type=float, default=0.0,
                   help='if > 0, add a SECOND, slower readout trace with this time '
                        'constant (ms) and concatenate it to the fast one, so the '
                        'linear readout sees fast detail AND slow context as separate '
                        'features. A single trace cannot hold both; this is the lever '
                        'that lets a longer reach show up in the readout. Try 200-800. '
                        '0 = off (single fast trace, original behaviour).')
    # --- intrinsic time constants (the real memory-horizon knobs) ---
    p.add_argument('--tau-syn', type=float, default=5.0,
                   help='synaptic decay (ms). The single biggest memory lever: longer '
                        'tau_syn makes each input spike linger, so echoes persist and '
                        'the recall horizon stretches. Trained default was 5 ms. NOT '
                        'baked into W, so changing it needs no retraining.')
    p.add_argument('--tau-m-min', type=float, default=15.0,
                   help='lower end of the per-neuron membrane time constant (ms).')
    p.add_argument('--tau-m-max', type=float, default=25.0,
                   help='upper end of the per-neuron membrane time constant (ms). '
                        'Raise both ends (e.g. 50-100) to integrate over longer '
                        'windows and extend memory.')
    # --- spike-triggered adaptation: the real long-memory channel ---
    p.add_argument('--adapt-b', type=float, default=0.0,
                   help='spike-triggered adaptation increment. Each spike bumps a '
                        'slow per-neuron current w_adapt by this amount; w_adapt then '
                        'decays with tau_w and is subtracted from the membrane drive. '
                        'This adds a long memory trace ORTHOGONAL to the fast synapse, '
                        'so unlike raising tau_syn it does not blur successive inputs. '
                        '0 = off (matches the trained network). Try 0.5-3.')
    p.add_argument('--tau-w', type=float, default=150.0,
                   help='adaptation time constant (ms). This is the memory-horizon '
                        'knob: 100-300 ms gives a slow echo that outlasts the 5 ms '
                        'synapse and ~20 ms membrane. Only matters when --adapt-b > 0.')
    # --- SECOND, much slower adaptation channel: the lever past the ~30 ms ceiling ---
    p.add_argument('--adapt-b2', type=float, default=0.0,
                   help='spike-triggered increment for a SECOND adaptation current '
                        'with its own (much longer) time constant --tau-w2. Runs '
                        'alongside the fast channel: the fast one keeps per-step '
                        'detail, this slow one carries a long context trace. A single '
                        'channel caps the recall horizon at ~30 ms; this is the lever '
                        'meant to push past it. 0 = off. Try 0.5-3.')
    p.add_argument('--tau-w2', type=float, default=1000.0,
                   help='time constant (ms) of the second adaptation channel. Make it '
                        'far slower than --tau-w (e.g. 1000-2000 ms) so it adds memory '
                        'at long delays instead of duplicating the fast channel. Only '
                        'matters when --adapt-b2 > 0.')
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
    p.add_argument('--sweep-n-inputs', default='',
                   help='optional comma-separated grid of input-neuron counts for '
                        '--sweep (e.g. "100,300,500"). Tests whether widening the '
                        'input footprint lifts separation. Empty = just use --n-input.')
    p.add_argument('--sweep-tau-syns', default='',
                   help='optional comma-separated grid of synaptic time constants (ms) '
                        'for --sweep (e.g. "5,20,50,100"). Empty = just use --tau-syn. '
                        'NOTE: raising tau_syn alone trades memory for separation and '
                        'blurs inputs; use --sweep-adapt-bs for genuine long memory.')
    p.add_argument('--sweep-adapt-bs', default='',
                   help='optional comma-separated grid of adaptation increments for '
                        '--sweep (e.g. "0,0.5,1,2"). The recommended long-memory lever. '
                        'Empty = just use --adapt-b.')
    p.add_argument('--sweep-tau-ws', default='',
                   help='optional comma-separated grid of adaptation time constants (ms) '
                        'for --sweep (e.g. "100,200,400"). Empty = just use --tau-w.')
    p.add_argument('--sweep-adapt-b2s', default='',
                   help='optional comma-separated grid of SECOND-channel increments for '
                        '--sweep (e.g. "0,0.5,1,2"). The lever past the ~30 ms ceiling. '
                        'Empty = just use --adapt-b2.')
    p.add_argument('--sweep-tau-w2s', default='',
                   help='optional comma-separated grid of SECOND-channel time constants '
                        '(ms) for --sweep (e.g. "800,1500,3000"). Empty = use --tau-w2.')
    p.add_argument('--sweep-holds', default='',
                   help='optional comma-separated grid of hold values (sim steps per '
                        'reservoir-step) for --sweep (e.g. "150,250,400"). Bigger hold '
                        '= slower reservoir step = each remembered input spans more ms, '
                        'the lever for a longer horizon in ms (at coarser resolution). '
                        'Reservoir-step COUNT is held constant across holds for a fair '
                        'MC comparison; only ms-per-step changes. Empty = use --hold.')
    p.add_argument('--sweep-max-delay', type=int, default=12,
                   help='largest delay k probed for memory per sweep cell. Raise it '
                        'when chasing long memory so the horizon is not truncated.')
    p.add_argument('--sweep-steps', type=int, default=18000,
                   help='MC-run length per sweep cell (shorter than the full run).')
    p.add_argument('--sweep-sep-steps', type=int, default=6000,
                   help='separation/divergence run length per sweep cell.')
    p.add_argument('--sweep-streams', type=int, default=6,
                   help='number of streams for separation per sweep cell.')
    p.add_argument('--sweep-csv', default='',
                   help='optional path to write per-cell sweep results as CSV '
                        '(recommended on the cluster so array tasks save clean data).')
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
        tau_m=rng.uniform(args.tau_m_min, args.tau_m_max, N),
        tau_syn=args.tau_syn, R=10.0,
        V_rest=-70.0,
        V_threshold=rng.uniform(-57, -53, N),
        V_reset=-80.0,
        E_exc=0.0, E_inh=-80.0,
        I_ext=rng.uniform(args.background_mean - args.background_spread,
                          args.background_mean + args.background_spread, N),
        adapt_b=args.adapt_b, tau_w=args.tau_w,
        adapt_b2=args.adapt_b2, tau_w2=args.tau_w2,
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
    adapt_b = params['adapt_b']; tau_w = params['tau_w']
    adapt_b2 = params['adapt_b2']; tau_w2 = params['tau_w2']
    dt = args.dt
    steps = drive.shape[0]

    V = np.full(N, -70.0)
    g_exc = np.zeros(N)
    g_inh = np.zeros(N)
    refractory = np.zeros(N)
    trace = np.zeros(N)                 # leaky spike trace = continuous state
    trace_decay = np.exp(-dt / args.tau_state)
    trace2_on = getattr(args, 'tau_state2', 0.0) > 0.0
    trace2 = np.zeros(N)                # optional second, slower readout trace
    trace2_decay = np.exp(-dt / args.tau_state2) if trace2_on else 0.0
    w_adapt = np.zeros(N)               # fast slow-adaptation current (tau_w)
    w_adapt2 = np.zeros(N)              # second, much slower channel (tau_w2)
    adapt_on = (adapt_b > 0.0) or (adapt_b2 > 0.0)
    adapt_decay = np.exp(-dt / tau_w)
    adapt_decay2 = np.exp(-dt / tau_w2)

    n_records = steps // record_every
    width = len(readout_idx) * (2 if trace2_on else 1)
    states = np.zeros((n_records, width))
    rec = 0

    for step in range(steps):
        current_I = I_ext + drive[step]

        I_syn = -g_exc * (V - E_exc) - g_inh * (V - E_inh)
        dV = (1.0 / tau_m) * (-(V - V_rest) + R * (current_I + I_syn)
                              - w_adapt - w_adapt2)
        V += dV * dt
        if adapt_on:
            # physiological clamp, ONLY on the adaptation path so the adapt-off network
            # stays byte-identical to speech_reservoir_fixed.py. Guards the explicit-
            # Euler step from a transient runaway when a large w_adapt hyperpolarises V.
            np.clip(V, E_inh - 10.0, E_exc + 10.0, out=V)
        refractory -= dt
        if adapt_on:
            w_adapt *= adapt_decay      # fast adaptation decay between spikes
            w_adapt2 *= adapt_decay2    # slow second-channel decay

        spikes = (V >= V_th) & (refractory <= 0)
        V[spikes] = V_reset
        refractory[spikes] = 2.0
        if adapt_on:
            w_adapt[spikes] += adapt_b   # each spike loads the fast slow-current
            w_adapt2[spikes] += adapt_b2  # ... and the slower second channel

        trace *= trace_decay
        trace[spikes] += 1.0
        if trace2_on:
            trace2 *= trace2_decay
            trace2[spikes] += 1.0

        exc_fired = spikes.copy(); exc_fired[n_exc:] = False
        inh_fired = spikes.copy(); inh_fired[:n_exc] = False
        if exc_fired.any():
            g_exc += W[exc_fired].sum(axis=0)
        if inh_fired.any():
            g_inh += W[inh_fired].sum(axis=0)
        g_exc -= (g_exc / tau_syn) * dt
        g_inh -= (g_inh / tau_syn) * dt

        if (step + 1) % record_every == 0 and rec < n_records:
            if trace2_on:
                states[rec] = np.concatenate([trace[readout_idx],
                                              trace2[readout_idx]])
            else:
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


def memory_horizon(per_delay, args, thresh=0.1):
    '''Last delay k (1-indexed) whose MC stays >= thresh = the recall horizon.
    CONTIGUOUS: stops at the first delay that drops below threshold.
    Returns (horizon_k, horizon_ms).'''
    horizon_k = 0
    for i, v in enumerate(per_delay):
        if v >= thresh:
            horizon_k = i + 1
        else:
            break
    return horizon_k, horizon_k * args.hold * args.dt


def memory_reach(per_delay, args, thresh=0.1):
    '''Deepest delay k (1-indexed) with MC >= thresh, NOT requiring contiguity.
    A slow channel can add memory at long delays with a dip in between; the
    contiguous horizon misses that, but reach catches it.
    Returns (reach_k, reach_ms).'''
    hits = [i + 1 for i, v in enumerate(per_delay) if v >= thresh]
    reach_k = hits[-1] if hits else 0
    return reach_k, reach_k * args.hold * args.dt


def run_sweep(args):
    '''Sweep input-scale x inh-scale x optional n-input x optional tau-syn.
    Reports total memory capacity, recall horizon (ms), separation and regime.'''
    input_scales = [float(x) for x in args.sweep_input_scales.split(',') if x.strip()]
    inh_scales = [float(x) for x in args.sweep_inh_scales.split(',') if x.strip()]
    n_inputs = ([int(x) for x in args.sweep_n_inputs.split(',') if x.strip()]
                or [args.n_input])
    tau_syns = ([float(x) for x in args.sweep_tau_syns.split(',') if x.strip()]
                or [args.tau_syn])
    adapt_bs = ([float(x) for x in args.sweep_adapt_bs.split(',') if x.strip()]
                or [args.adapt_b])
    tau_ws = ([float(x) for x in args.sweep_tau_ws.split(',') if x.strip()]
              or [args.tau_w])
    adapt_b2s = ([float(x) for x in args.sweep_adapt_b2s.split(',') if x.strip()]
                 or [args.adapt_b2])
    tau_w2s = ([float(x) for x in args.sweep_tau_w2s.split(',') if x.strip()]
               or [args.tau_w2])
    holds = ([int(x) for x in args.sweep_holds.split(',') if x.strip()]
             or [args.hold])
    # Hold the reservoir-step COUNT constant across hold values so MC estimates are
    # comparable; only ms-per-step changes. Budgets are derived from the first hold.
    mc_rsteps = max(1, args.sweep_steps // holds[0])
    sep_rsteps = max(1, args.sweep_sep_steps // holds[0])
    n_cells = (len(input_scales) * len(inh_scales) * len(n_inputs) * len(tau_syns)
               * len(adapt_bs) * len(tau_ws) * len(adapt_b2s) * len(tau_w2s)
               * len(holds))
    print(f'Sweeping input_scale={input_scales} x inh_scale={inh_scales} '
          f'x n_input={n_inputs} x tau_syn={tau_syns} x adapt_b={adapt_bs} '
          f'x tau_w={tau_ws} x adapt_b2={adapt_b2s} x tau_w2={tau_w2s} '
          f'x hold={holds}  ({n_cells} cells)')
    print(f'weights = {args.weights}  ({mc_rsteps} reservoir-steps/cell)\n')
    header = (f'{"hold":>5} {"ms/st":>6} {"adb":>5} {"tau_w":>6} {"adb2":>5} '
              f'{"tau_w2":>7} {"input":>6} {"inh":>5} {"MC_1":>6} {"totMC":>6} '
              f'{"horiz":>6} {"reach":>6} {"separ":>8} {"end/pk":>7}  regime')
    print(header)
    print('-' * (len(header) + 10))

    rows = []
    for inh in inh_scales:
        args.inh_scale = inh
        params = build_network(args)        # inh is baked into W here
        for tsyn in tau_syns:
            params['tau_syn'] = tsyn         # cheap to swap; no rebuild needed
            for adb in adapt_bs:
                params['adapt_b'] = adb      # adaptation lives in params, not W
                for tw in tau_ws:
                    params['tau_w'] = tw
                    for adb2 in adapt_b2s:
                        params['adapt_b2'] = adb2   # second, slower channel
                        for tw2 in tau_w2s:
                            params['tau_w2'] = tw2
                            for n_in in n_inputs:
                                args.n_input = n_in  # make_drive + pick_readout
                                readout_idx = pick_readout(params, args)
                                for isc in input_scales:
                                    args.input_scale = isc
                                    for hld in holds:
                                        args.hold = hld  # ms/step = hold * dt
                                        # scale sim steps so rstep count is constant
                                        eff_steps = mc_rsteps * hld
                                        eff_sep = sep_rsteps * hld
                                        s_steps, s_md = args.steps, args.max_delay
                                        args.steps, args.max_delay = (
                                            eff_steps, args.sweep_max_delay)
                                        per_delay, total_mc = memory_capacity(
                                            params, args, readout_idx)
                                        args.steps, args.max_delay = s_steps, s_md
                                        hk, hms = memory_horizon(per_delay, args)
                                        rk, rms = memory_reach(per_delay, args)
                                        # light separation + divergence
                                        s_ss, s_ns = args.sep_steps, args.sep_streams
                                        args.sep_steps, args.sep_streams = (
                                            eff_sep, args.sweep_streams)
                                        sep_mean, sep_std = separation(
                                            params, args, readout_idx)
                                        dist = divergence(params, args, readout_idx)
                                        args.sep_steps, args.sep_streams = s_ss, s_ns
                                        ratio, regime = classify_regime(dist)
                                        print(f'{hld:>5} {hld * args.dt:>6.1f} '
                                              f'{adb:>5.1f} {tw:>6.0f} {adb2:>5.1f} '
                                              f'{tw2:>7.0f} {isc:>6.0f} {inh:>5.1f} '
                                              f'{per_delay[0]:>6.3f} {total_mc:>6.2f} '
                                              f'{hms:>6.0f} {rms:>6.0f} '
                                              f'{sep_mean:>8.4f} {ratio:>7.2f}  '
                                              f'{regime.split(" (")[0]}')
                                        rows.append(dict(
                                            hold=hld, tau_syn=tsyn, adapt_b=adb,
                                            tau_w=tw, adapt_b2=adb2, tau_w2=tw2,
                                            n_input=n_in, input_scale=isc,
                                            inh_scale=inh, mc1=per_delay[0],
                                            total_mc=total_mc, horizon_ms=hms,
                                            reach_ms=rms, sep=sep_mean, ratio=ratio,
                                            regime=regime))

    usable = [r for r in rows if 'CHAOTIC' not in r['regime']]
    if usable:
        # rank by reach (deepest recall) since the slow channel adds far-delay memory
        best = max(usable, key=lambda r: (r['reach_ms'], r['total_mc']))
        print(f'\nDeepest recall among non-chaotic cells: hold={best["hold"]} '
              f'({best["hold"] * args.dt:.0f}ms/step) '
              f'adapt_b={best["adapt_b"]:.1f}/tau_w={best["tau_w"]:.0f} '
              f'adapt_b2={best["adapt_b2"]:.1f}/tau_w2={best["tau_w2"]:.0f} '
              f'input_scale={best["input_scale"]:.0f}  '
              f'reach={best["reach_ms"]:.0f}ms horizon={best["horizon_ms"]:.0f}ms '
              f'totalMC={best["total_mc"]:.2f} ({best["regime"].split(" (")[0]})')

    if args.sweep_csv:
        import csv as _csv
        with open(args.sweep_csv, 'w', newline='') as f:
            w = _csv.DictWriter(f, fieldnames=['hold', 'tau_syn', 'adapt_b', 'tau_w',
                                               'adapt_b2', 'tau_w2', 'n_input',
                                               'input_scale', 'inh_scale', 'mc1',
                                               'total_mc', 'horizon_ms', 'reach_ms',
                                               'sep', 'ratio', 'regime'])
            w.writeheader()
            w.writerows(rows)
        print(f'CSV saved to {args.sweep_csv}')

    # plot vs the primary swept lever. Preference: slow channel (adapt_b2 > tau_w2)
    # is the lever past the ceiling, then fast adaptation, tau_w, hold, tau_syn, input.
    if len(adapt_b2s) > 1:
        xkey, xlabel = 'adapt_b2', 'slow-channel increment b2'
    elif len(tau_w2s) > 1:
        xkey, xlabel = 'tau_w2', 'slow-channel tau_w2 (ms)'
    elif len(holds) > 1:
        xkey, xlabel = 'hold', 'hold (sim steps/reservoir-step)'
    elif len(adapt_bs) > 1:
        xkey, xlabel = 'adapt_b', 'adaptation increment b'
    elif len(tau_ws) > 1:
        xkey, xlabel = 'tau_w', 'tau_w (ms)'
    elif len(tau_syns) > 1:
        xkey, xlabel = 'tau_syn', 'tau_syn (ms)'
    else:
        xkey, xlabel = 'input_scale', 'input-scale'
    group_keys = [k for k in ('inh_scale', 'n_input', 'tau_syn', 'adapt_b', 'tau_w',
                              'adapt_b2', 'tau_w2', 'hold', 'input_scale')
                  if k != xkey]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    seen = set()
    for r in rows:
        combo = tuple(r[k] for k in group_keys)
        if combo in seen:
            continue
        seen.add(combo)
        cells = sorted([c for c in rows
                        if all(c[k] == v for k, v in zip(group_keys, combo))],
                       key=lambda c: c[xkey])
        lab = ', '.join(f'{k.split("_")[0]}={v:g}' for k, v in zip(group_keys, combo))
        xs = [c[xkey] for c in cells]
        ax[0].plot(xs, [c['reach_ms'] for c in cells], 'o-', label=lab)
        ax[0].plot(xs, [c['horizon_ms'] for c in cells], 'o--', alpha=0.4)
        ax[1].plot(xs, [c['total_mc'] for c in cells], 'o-', label=lab)
    ax[0].set_xlabel(xlabel); ax[0].set_ylabel('recall depth (ms)')
    ax[0].set_title('Memory reach (solid) & contiguous horizon (dashed)')
    ax[0].legend(fontsize=6)
    ax[1].set_xlabel(xlabel); ax[1].set_ylabel('total memory capacity')
    ax[1].set_title('Total MC'); ax[1].legend(fontsize=6)
    fig.suptitle(f'{args.weights}  memory sweep')
    fig.tight_layout(); fig.savefig(args.plot, dpi=120)
    print(f'\nPlot saved to {args.plot}')
    print('\nReading it: the slow second channel (adapt_b2 with tau_w2 ~1-2 s) should')
    print('push REACH (deepest recall, solid) and total MC past the ~30 ms single-')
    print('channel ceiling. The contiguous horizon (dashed) may lag if memory dips')
    print('then recovers. Keep cells whose regime stays FADING / CRITICAL (not CHAOTIC).')


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
