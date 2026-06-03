'''Working-memory CAPACITY for the ALIF long-memory reservoir: how many items can
it hold at once? K scalar cues are presented SIMULTANEOUSLY, each through its own
random input projection, so the slow adaptation variables hold a superposition of
K patterns. After a delay, one linear readout per item decodes its value from the
shared population state. Sweeping K gives a capacity curve (recall vs item count).

WHY THIS WORKS (and what sets the limit)
  Random projections in an N-dim space are near-orthogonal, so K items written as a
  weighted sum stay linearly separable until K grows large enough that the
  projections overlap and items interfere. Capacity therefore scales with N and
  degrades with K and with delay - the same qualitative shape as the human
  'about four items' working-memory limit.

  Per-item cue drive is normalised by 1/sqrt(K) so the network's operating point
  (firing regime) stays fixed as K grows; the capacity drop then reflects
  representational interference, not the network simply being driven harder.

Same fixed-reservoir + ridge-readout design as train_memory_alif_hold.py: no BPTT.
'''
import argparse
import os
import time
import numpy as np

try:
    from scipy import sparse as sp
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False


def build_network(args, rng):
    N = args.N
    n_exc = int(round(args.exc_frac * N))
    tau_m = rng.uniform(args.tau_m_min, args.tau_m_max, N)
    tau_a = np.exp(rng.uniform(np.log(args.tau_a_min), np.log(args.tau_a_max), N))
    # one random projection per item, up to the largest K we will test; smaller K
    # use the first K rows so networks are comparable across the sweep
    K_max = max(args.K_list)
    W_in = rng.standard_normal((K_max, N)) * args.input_scale

    W = None
    if args.rec_gain > 0.0 and HAVE_SCIPY:
        mask = (rng.random((N, N)) < args.connectivity)
        np.fill_diagonal(mask, False)
        Wd = rng.random((N, N)) * mask
        Wd[n_exc:, :] *= -args.g_inh
        Wd *= args.rec_gain / max(1e-9, np.abs(Wd).sum(axis=0).mean())
        W = sp.csr_matrix(Wd)
    return dict(N=N, n_exc=n_exc, tau_m=tau_m, tau_a=tau_a, W_in=W_in, W=W)


def simulate(args, net, cues, rng):
    '''cues: (B, K) item values. Inject the superposition sum_k cue_k * W_in_k
    (normalised by 1/sqrt(K)) for cue_dur, then run to max delay. Return
    {delay_ms: (B, 2N+1)} population state at each delay.'''
    B, K = cues.shape
    N = net['N']; dt = args.dt
    inv_tau_m = dt / net['tau_m']
    rho_a = np.exp(-dt / net['tau_a'])
    syn_decay = np.exp(-dt / args.tau_syn)
    trace_decay = np.exp(-dt / args.tau_trace)
    W_in = net['W_in'][:K]; W = net['W']

    V = np.zeros((B, N)); g = np.zeros((B, N))
    b = np.zeros((B, N)); rtr = np.zeros((B, N))
    bg = args.background + args.background_spread * rng.standard_normal(N)
    cue_vec = (cues @ W_in) / np.sqrt(K)           # (B, N) superposition

    cue_steps = int(round(args.cue_dur / dt))
    checkpoints = {int(round(d / dt)): d for d in args.delay_list}
    total_steps = cue_steps + max(checkpoints)
    feats = {}
    for t in range(total_steps):
        I = bg[None, :]
        if t < cue_steps:
            I = I + cue_vec
        if W is not None:
            I = I + args.syn_gain * g
        V = V + inv_tau_m * (-V + args.R * I)
        s = (V >= (1.0 + args.beta_a * b)).astype(np.float64)
        V = V * (1.0 - s)
        b = b * rho_a + s
        rtr = rtr * trace_decay + s
        if W is not None:
            g = g * syn_decay + s @ W
        after = t - cue_steps + 1
        if after in checkpoints:
            # slow adaptation variable b is where multi-second memory lives; the
            # fast trace is ~0 by these delays, so we read b only. This halves the
            # feature dimension (N+1), keeping the per-item readouts well-posed.
            feats[checkpoints[after]] = np.concatenate(
                [b.copy(), np.ones((B, 1))], axis=1)
    return feats


def ridge_fit(X, Y, lam):
    # Y can be (n,) or (n, K); solves all item readouts at once
    A = X.T @ X + lam * np.eye(X.shape[1])
    return np.linalg.solve(A, X.T @ Y)


def decode(feats_tr, feats_te, cue_tr, cue_te, delay_list, ridge):
    chance = 1.0 / 3.0
    out = {}
    for d in delay_list:
        Xtr = feats_tr[d][:, :-1]; Xte = feats_te[d][:, :-1]
        mu = Xtr.mean(axis=0); sc = (Xtr - mu).std() + 1e-12
        Xtr_n = np.hstack([(Xtr - mu) / sc, np.ones((len(Xtr), 1))])
        Xte_n = np.hstack([(Xte - mu) / sc, np.ones((len(Xte), 1))])
        W_out = ridge_fit(Xtr_n, cue_tr, ridge)     # (2N+1, K)
        pred = Xte_n @ W_out                         # (n_te, K)
        mse_per_item = np.mean((pred - cue_te) ** 2, axis=0)   # (K,)
        r2_per_item = 1.0 - mse_per_item / chance
        out[d] = r2_per_item
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--N', type=int, default=2000)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--dt', type=float, default=1.0)
    p.add_argument('--cue-dur', type=float, default=200.0)
    p.add_argument('--delays', type=str, default='500,2000,5000')
    p.add_argument('--K-list', type=str, default='1,2,4,8,16,32',
                   help='comma-sep item counts to sweep (simultaneous items).')
    p.add_argument('--n-train', type=int, default=1500)
    p.add_argument('--n-test', type=int, default=400)
    p.add_argument('--tau-m-min', type=float, default=15.0)
    p.add_argument('--tau-m-max', type=float, default=30.0)
    p.add_argument('--tau-a-min', type=float, default=200.0)
    p.add_argument('--tau-a-max', type=float, default=8000.0)
    p.add_argument('--beta-a', type=float, default=1.5)
    p.add_argument('--tau-syn', type=float, default=10.0)
    p.add_argument('--tau-trace', type=float, default=20.0)
    p.add_argument('--R', type=float, default=1.0)
    p.add_argument('--input-scale', type=float, default=2.0)
    p.add_argument('--background', type=float, default=1.0)
    p.add_argument('--background-spread', type=float, default=0.1)
    p.add_argument('--rec-gain', type=float, default=0.0)
    p.add_argument('--syn-gain', type=float, default=1.0)
    p.add_argument('--connectivity', type=float, default=0.05)
    p.add_argument('--g-inh', type=float, default=1.0)
    p.add_argument('--exc-frac', type=float, default=0.8)
    p.add_argument('--ridge', type=float, default=10.0)
    p.add_argument('--outdir', type=str, default='results_alif_capacity')
    args = p.parse_args()

    args.delay_list = [float(x) for x in args.delays.split(',')]
    args.K_list = [int(x) for x in args.K_list.split(',')]
    rng = np.random.default_rng(args.seed)
    t0 = time.time()
    net = build_network(args, rng)
    print(f'[init] N={args.N} dt={args.dt}ms background={args.background} '
          f'tau_a_max={args.tau_a_max:.0f}ms  K sweep={args.K_list}', flush=True)

    chance = 1.0 / 3.0
    grid = np.full((len(args.K_list), len(args.delay_list)), np.nan)  # mean R2
    for ki, K in enumerate(args.K_list):
        cue_tr = rng.uniform(-1.0, 1.0, (args.n_train, K))
        cue_te = rng.uniform(-1.0, 1.0, (args.n_test, K))
        ftr = simulate(args, net, cue_tr, rng)
        fte = simulate(args, net, cue_te, rng)
        r2 = decode(ftr, fte, cue_tr, cue_te, args.delay_list, args.ridge)
        row = []
        for di, d in enumerate(args.delay_list):
            mean_r2 = float(np.mean(r2[d]))
            grid[ki, di] = mean_r2
            row.append(f'{d/1000:.1f}s:{mean_r2:+.3f}')
        print(f'[K={K:3d}] mean recall R2 per delay  ' + '  '.join(row) +
              f'   ({time.time()-t0:.0f}s)', flush=True)

    os.makedirs(args.outdir, exist_ok=True)
    out = os.path.join(args.outdir, f'capacity_N{args.N}_seed{args.seed}.npz')
    np.savez(out, K_list=np.array(args.K_list),
             delays=np.array(args.delay_list), r2_grid=grid,
             seed=args.seed, N=args.N, background=args.background)
    print(f'[saved] {out}  total {time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
