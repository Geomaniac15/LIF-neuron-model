'''Long-memory spiking model: adaptive-threshold LIF (ALIF / LSNN-style) reservoir
with HETEROGENEOUS adaptation time constants spanning several seconds, plus a
trained linear readout. Store-and-hold task: a scalar cue is injected briefly,
then the network must report it after a long delay (target 5 s).

WHY THIS DESIGN (vs the BPTT trainer)
  Long memory does NOT come from recurrent reverberation - that is the unstable,
  short-lived kind the BPTT trainer kept drifting into (saturate/silence bursting).
  It comes from SLOW STATE VARIABLES whose time constant matches the lag you want
  to hold. Here every neuron has an adaptive firing threshold b that decays with
  its OWN tau_a, drawn log-uniformly up to tau_a_max. A cue perturbs spiking, which
  loads the slow b variables; during the quiet delay b decays as exp(-delay/tau_a),
  so neurons with tau_a >> delay still carry the cue at recall. A linear readout
  decodes the cue from the population b-state.

  No backprop-through-time: recurrent W is FIXED (controlled gain) and only the
  readout is fit, by closed-form ridge regression. This scales to 5 s, which full
  BPTT cannot (it would tape ~50k sim-steps, many GB).

THE LEVER THAT SETS HOW LONG IT REMEMBERS
  --tau-a-max is the longest adaptation time constant. To hold ~5 s cleanly you
  want a good fraction of neurons with tau_a comfortably above 5000 ms (default
  max 8000 ms). --beta-a sets how strongly the slow variable gates the threshold
  (how much memory it can store).

Normalised LIF units: V_rest=0, V_th=1, V_reset=0, R=1 (no mV bookkeeping).
'''
import argparse
import os
import time
import numpy as np

# scipy only needed for sparse recurrence; gracefully skip recurrence without it
try:
    from scipy import sparse as sp
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False


def build_network(args, rng):
    N = args.N
    n_exc = int(round(args.exc_frac * N))
    # fast channel: heterogeneous membrane time constants
    tau_m = rng.uniform(args.tau_m_min, args.tau_m_max, N)
    # SLOW memory channel: log-uniform adaptation time constants so the population
    # tiles timescales from short up to (and past) the target delay
    tau_a = np.exp(rng.uniform(np.log(args.tau_a_min), np.log(args.tau_a_max), N))
    # mixed-sign feedforward cue weights: different neurons load their slow var
    # differently as a function of the cue, making the cue linearly decodable
    W_in = rng.standard_normal(N) * args.input_scale

    W = None
    if args.rec_gain > 0.0 and HAVE_SCIPY:
        mask = (rng.random((N, N)) < args.connectivity)
        np.fill_diagonal(mask, False)
        Wd = rng.random((N, N)) * mask
        # Dale's law on the PRE neuron (row): inhibitory rows negative
        Wd[n_exc:, :] *= -args.g_inh
        # scale so the mean total recurrent drive per neuron ~ rec_gain (stable)
        mean_in = np.abs(Wd).sum(axis=0).mean()
        Wd *= args.rec_gain / max(1e-9, mean_in)
        W = sp.csr_matrix(Wd)             # recurrent input = spikes @ W
    elif args.rec_gain > 0.0 and not HAVE_SCIPY:
        print('[warn] scipy missing - running with NO recurrence (rec_gain ignored)')

    return dict(N=N, n_exc=n_exc, tau_m=tau_m, tau_a=tau_a, W_in=W_in, W=W)


def simulate(args, net, cues, rng):
    '''Vectorised over trials (rows). Inject each trial's scalar cue for cue_dur,
    then run quiet for max(delay). Return {delay_ms: feature_matrix (B, 2N+1)},
    the population [adaptation b | fast spike trace | bias] read at each delay.'''
    B = len(cues)
    N = net['N']; dt = args.dt
    inv_tau_m = dt / net['tau_m']
    rho_a = np.exp(-dt / net['tau_a'])             # slow-var decay per step
    syn_decay = np.exp(-dt / args.tau_syn)
    trace_decay = np.exp(-dt / args.tau_trace)
    W_in = net['W_in']; W = net['W']

    V = np.zeros((B, N))
    g = np.zeros((B, N))
    b = np.zeros((B, N))                           # adaptation = slow memory
    rtr = np.zeros((B, N))                         # fast trace = recent activity
    bg = args.background + args.background_spread * rng.standard_normal(N)
    cue_vec = cues[:, None] * W_in[None, :]        # (B, N)

    cue_steps = int(round(args.cue_dur / dt))
    delays = args.delay_list
    checkpoints = {int(round(d / dt)): d for d in delays}
    total_steps = cue_steps + max(checkpoints)
    feats = {}

    for t in range(total_steps):
        I = bg[None, :]
        if t < cue_steps:
            I = I + cue_vec
        if W is not None:
            I = I + args.syn_gain * g
        V = V + inv_tau_m * (-V + args.R * I)
        thr = 1.0 + args.beta_a * b
        s = (V >= thr).astype(np.float64)
        V = V * (1.0 - s)                          # reset to 0 on spike
        b = b * rho_a + s
        rtr = rtr * trace_decay + s
        if W is not None:
            g = g * syn_decay + s @ W

        after_cue = t - cue_steps + 1              # 1-based steps since cue ended
        if after_cue in checkpoints:
            feats[checkpoints[after_cue]] = np.concatenate(
                [b.copy(), rtr.copy(), np.ones((B, 1))], axis=1)
    return feats


def ridge_fit(X, y, lam):
    # closed-form ridge: w = (X^T X + lam I)^-1 X^T y  (bias column already in X)
    d = X.shape[1]
    A = X.T @ X + lam * np.eye(d)
    return np.linalg.solve(A, X.T @ y)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--N', type=int, default=2000)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--dt', type=float, default=1.0, help='ms per sim-step.')
    # task
    p.add_argument('--cue-dur', type=float, default=200.0, help='cue window (ms).')
    p.add_argument('--delays', type=str, default='250,500,1000,2000,3000,5000',
                   help='comma-sep recall delays (ms) to read out / score.')
    p.add_argument('--n-train', type=int, default=800)
    p.add_argument('--n-test', type=int, default=200)
    # neuron dynamics
    p.add_argument('--tau-m-min', type=float, default=15.0)
    p.add_argument('--tau-m-max', type=float, default=30.0)
    p.add_argument('--tau-a-min', type=float, default=200.0,
                   help='shortest adaptation time constant (ms).')
    p.add_argument('--tau-a-max', type=float, default=8000.0,
                   help='LONGEST adaptation time constant (ms): the memory-horizon '
                        'lever. Keep well above the target delay.')
    p.add_argument('--beta-a', type=float, default=1.5,
                   help='threshold-adaptation strength (how much the slow var can '
                        'store).')
    p.add_argument('--tau-syn', type=float, default=10.0)
    p.add_argument('--tau-trace', type=float, default=20.0)
    p.add_argument('--R', type=float, default=1.0)
    # drive
    p.add_argument('--input-scale', type=float, default=2.0)
    p.add_argument('--background', type=float, default=0.0,
                   help='constant background drive (0 = quiet delay = cleanest hold).')
    p.add_argument('--background-spread', type=float, default=0.0)
    # recurrence (fixed; memory does not depend on it, kept modest for stability)
    p.add_argument('--rec-gain', type=float, default=0.0,
                   help='recurrent input scale (0 disables recurrence).')
    p.add_argument('--syn-gain', type=float, default=1.0,
                   help='gain on the recurrent synaptic current.')
    p.add_argument('--connectivity', type=float, default=0.05)
    p.add_argument('--g-inh', type=float, default=1.0)
    p.add_argument('--exc-frac', type=float, default=0.8)
    # readout
    p.add_argument('--ridge', type=float, default=1.0)
    p.add_argument('--outdir', type=str, default='results_alif_hold')
    args = p.parse_args()

    args.delay_list = [float(x) for x in args.delays.split(',')]
    rng = np.random.default_rng(args.seed)
    t0 = time.time()

    net = build_network(args, rng)
    frac_long = float(np.mean(net['tau_a'] >= max(args.delay_list)))
    print(f'[init] N={args.N} dt={args.dt}ms  tau_a in [{args.tau_a_min:.0f},'
          f'{args.tau_a_max:.0f}]ms  frac(tau_a>={max(args.delay_list):.0f}ms)='
          f'{frac_long:.2f}  recurrence={"on" if net["W"] is not None else "off"}',
          flush=True)

    cue_tr = rng.uniform(-1.0, 1.0, args.n_train)
    cue_te = rng.uniform(-1.0, 1.0, args.n_test)
    feats_tr = simulate(args, net, cue_tr, rng)
    feats_te = simulate(args, net, cue_te, rng)
    print(f'[sim] {args.n_train}+{args.n_test} trials simulated  '
          f'({time.time()-t0:.0f}s)', flush=True)

    chance = 1.0 / 3.0                              # E[c^2] for c ~ U(-1,1)
    results = {}
    for d in args.delay_list:
        # drop the bias column from features (re-added explicitly below)
        Xtr = feats_tr[d][:, :-1]; Xte = feats_te[d][:, :-1]
        mu = Xtr.mean(axis=0)
        # GLOBAL scale (single scalar), not per-feature z-score: per-feature
        # scaling amplifies near-constant features into noise and wrecks the readout
        scale = (Xtr - mu).std() + 1e-12
        Xtr_n = np.hstack([(Xtr - mu) / scale, np.ones((len(Xtr), 1))])
        Xte_n = np.hstack([(Xte - mu) / scale, np.ones((len(Xte), 1))])
        w = ridge_fit(Xtr_n, cue_tr, args.ridge)
        pred = Xte_n @ w
        mse = float(np.mean((pred - cue_te) ** 2))
        r2 = 1.0 - mse / chance
        results[d] = (mse, r2)
        print(f'[delay {d:6.0f}ms] test MSE={mse:.4f}  R2={r2:+.3f}  '
              f'(chance MSE={chance:.4f})', flush=True)

    os.makedirs(args.outdir, exist_ok=True)
    out = os.path.join(args.outdir, f'alif_hold_N{args.N}_seed{args.seed}.npz')
    np.savez(out, delays=np.array(args.delay_list),
             mse=np.array([results[d][0] for d in args.delay_list]),
             r2=np.array([results[d][1] for d in args.delay_list]),
             tau_a=net['tau_a'], seed=args.seed, N=args.N)
    print(f'[saved] {out}  total {time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
