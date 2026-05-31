'''Surrogate-gradient BPTT trainer for a delay-k memory LIF reservoir.

WHY THIS EXISTS
---------------
The frozen random reservoir holds almost no linearly-decodable memory past
~30 ms (2 reservoir-steps): that ceiling is structural, set by the random
contracting recurrent weights. Reward-modulated STDP could not push past it,
because with no decodable memory there is no real teaching signal to bootstrap
from. This trainer instead computes a REAL gradient of a delayed-recall loss
through time and uses it to CREATE delay-k memory directly in the recurrent
weights W.

HOW
---
The forward pass is numerically identical to reservoir_capacity.simulate
(conductance synapses, hard spike threshold + reset, optional spike-frequency
adaptation, leaky readout trace). The ONLY change is in the backward pass: the
discontinuous spike s = H(V - V_th) is given a smooth surrogate derivative
phi'(V - V_th) (fast sigmoid, SuperSpike / Zenke & Ganguli 2018 style) so that
gradients flow through spikes. Because the forward dynamics match the benchmark
exactly, the trained W transfers: reservoir_capacity.py and
aggregate_memory_sweep.py run on the saved .npz unchanged.

WHAT IS TRAINED
---------------
Only the recurrent weight matrix W (Dale's law + sparsity pattern + ceiling
preserved) and an internal linear readout W_out that supplies the teaching
gradient. The input injection is the SAME fixed current-into-input-block scheme
the benchmark uses (no input matrix to learn), and the benchmark refits its own
Ridge readout, so only W is saved and only W matters downstream.

OBJECTIVE
---------
At every reservoir-step r (after washout) a linear readout reconstructs the
input delayed by target_delay reservoir-steps, u[r - target_delay], from the
leaky spike trace. Loss = mean squared reconstruction error. BPTT (truncated to
short windows) backpropagates this loss into W.

OUTPUT
------
results_*/run_N<N>_seed<seed>.npz with keys W_initial, W_final, N, seed, dt
(plus a few training extras), matching random_neurons_fixed.py so the existing
benchmark pipeline works without modification.

Example:
    python train_memory_bptt.py --N 2000 --seed 0 --steps 4000 \
        --target-delay 2 --adapt-b 4 --tau-w 400 --outdir results_mem_bptt
'''
import argparse
import os
import time
import numpy as np


# ----------------------------------------------------------------------------
# Surrogate spike nonlinearity
# ----------------------------------------------------------------------------
def spike_forward(x, smooth, beta):
    '''Forward spike value. Hard Heaviside in training (so the forward pass is
    byte-identical to the benchmark); smooth fast-sigmoid only for the
    finite-difference gradient check.'''
    if smooth:
        return 1.0 / (1.0 + np.exp(-beta * x))
    return (x >= 0.0).astype(np.float64)


def spike_surrogate_grad(x, smooth, beta):
    '''ds/dx used in the backward pass. In smooth mode this is the exact
    derivative of the sigmoid (so the gradient check is meaningful). In hard
    mode it is the SuperSpike fast-sigmoid surrogate 1 / (1 + beta|x|)^2, which
    replaces the (zero / undefined) true Heaviside derivative.'''
    if smooth:
        s = 1.0 / (1.0 + np.exp(-beta * x))
        return beta * s * (1.0 - s)
    return 1.0 / (1.0 + beta * np.abs(x)) ** 2


# ----------------------------------------------------------------------------
# Fresh network construction (mirrors random_neurons_fixed.py recipe)
# ----------------------------------------------------------------------------
def build_fresh_network(args):
    '''Build a fresh sparse E-I weight matrix and frozen neuron params from
    scratch, using the validated AI-regime recipe so firing stays sane.'''
    rng = np.random.default_rng(args.seed)
    N = args.N
    n_exc = int(round(0.8 * N))

    W = rng.random((N, N)) * args.w_scale
    mask = rng.random((N, N)) > (1.0 - args.connectivity)
    W *= mask
    np.fill_diagonal(W, 0.0)
    W[n_exc:, :] *= -args.g_inh          # Dale: inhibitory rows negative

    conn_mask = W != 0.0                  # frozen sparsity pattern (signed)

    # per-row soft ceiling on |W| (exc vs inh have different scales)
    w_ceiling = np.full((N, 1), args.w_max_mult * args.w_scale)
    w_ceiling[n_exc:] = args.w_max_mult * args.g_inh * args.w_scale

    params = dict(
        N=N, n_exc=n_exc,
        tau_m=rng.uniform(args.tau_m_min, args.tau_m_max, N),
        V_th=rng.uniform(-57.0, -53.0, N),
        I_ext=rng.uniform(args.background_mean - args.background_spread,
                          args.background_mean + args.background_spread, N),
        R=10.0, V_rest=-70.0, V_reset=-80.0, E_exc=0.0, E_inh=-80.0,
        tau_syn=args.tau_syn, dt=args.dt,
        adapt_b=args.adapt_b, tau_w=args.tau_w,
    )
    return W, conn_mask, w_ceiling, n_exc, params


# ----------------------------------------------------------------------------
# Forward over one TBPTT window, taping everything backward needs
# ----------------------------------------------------------------------------
def forward_window(W, W_out, carry, drive_win, targets, readout_idx,
                   params, args, smooth):
    '''Run `chunk` reservoir-steps (chunk*hold sim-steps) forward.

    carry      : dict of state arrays (V, ge, gi, wad, tr, refr) carried in
    drive_win  : (n_sim, N) external input current for this window
    targets    : list of (boundary_sim_index, target_value) for readout steps
    Returns (carry_out, loss, tape, readout_grads).
    '''
    N = params['N']; n_exc = params['n_exc']
    tau_m = params['tau_m']; inv_tau_m = 1.0 / tau_m
    R = params['R']; V_rest = params['V_rest']; V_reset = params['V_reset']
    V_th = params['V_th']; E_exc = params['E_exc']; E_inh = params['E_inh']
    tau_syn = params['tau_syn']; dt = params['dt']
    adapt_on = params['adapt_b'] > 0.0
    adapt_b = params['adapt_b']
    adapt_decay = np.exp(-dt / params['tau_w']) if adapt_on else 0.0
    syn_decay = 1.0 - dt / tau_syn
    trace_decay = np.exp(-dt / args.tau_state)
    beta = args.beta
    use_refr = args.refractory > 0.0

    exc_mask = np.zeros(N); exc_mask[:n_exc] = 1.0
    inh_mask = np.zeros(N); inh_mask[n_exc:] = 1.0

    V = carry['V'].copy(); ge = carry['ge'].copy(); gi = carry['gi'].copy()
    wad = carry['wad'].copy(); tr = carry['tr'].copy(); refr = carry['refr'].copy()

    n_sim = drive_win.shape[0]
    # tape arrays
    tp_V = np.empty((n_sim, N)); tp_ge = np.empty((n_sim, N)); tp_gi = np.empty((n_sim, N))
    tp_wad = np.empty((n_sim, N)); tp_tr = np.empty((n_sim, N))
    tp_Vh = np.empty((n_sim, N)); tp_s = np.empty((n_sim, N)); tp_gate = np.empty((n_sim, N))

    target_map = dict(targets)
    loss = 0.0
    readout_grads = {}            # sim_index -> grad on tr (added in backward)
    gW_out = np.zeros_like(W_out)
    spike_sum = 0.0

    for t in range(n_sim):
        tp_V[t] = V; tp_ge[t] = ge; tp_gi[t] = gi; tp_wad[t] = wad; tp_tr[t] = tr
        gate = (refr <= 0.0).astype(np.float64) if use_refr else np.ones(N)
        tp_gate[t] = gate

        Isyn = -ge * (V - E_exc) - gi * (V - E_inh)
        dV = inv_tau_m * (-(V - V_rest) + R * (params['I_ext'] + drive_win[t] + Isyn) - wad)
        Vh = V + dV * dt
        tp_Vh[t] = Vh

        s = spike_forward(Vh - V_th, smooth, beta) * gate
        tp_s[t] = s
        spike_sum += s.sum()

        Vnew = Vh - (Vh - V_reset) * s
        if adapt_on:
            wad = wad * adapt_decay + adapt_b * s
        tr = tr * trace_decay + s

        x_exc = s * exc_mask
        x_inh = s * inh_mask
        ge = (ge + W.T @ x_exc) * syn_decay
        gi = (gi - W.T @ x_inh) * syn_decay

        if use_refr:
            refr = refr - dt
            # hard refractory set on (forward) spikes; gate is stop-grad anyway
            fired = tp_s[t] >= (0.5 if smooth else 0.5)
            refr[fired] = args.refractory
        V = Vnew

        # readout at reservoir-step boundary
        if t in target_map:
            r_vec = np.empty(len(readout_idx) + 1)
            r_vec[:-1] = tr[readout_idx]; r_vec[-1] = 1.0
            pred = float(W_out @ r_vec)
            err = pred - target_map[t]
            loss += err * err
            gtr = np.zeros(N)
            gtr[readout_idx] = 2.0 * err * W_out[:-1]
            readout_grads[t] = gtr
            gW_out += 2.0 * err * r_vec

    # rate-homeostasis penalty: pull EACH neuron's mean spike-prob toward the
    # target so the network can't drift its excitability to game the readout.
    p_target = args.target_rate * dt / 1000.0
    r_i = tp_s.sum(axis=0) / n_sim                 # per-neuron mean spike prob
    rate_loss = 0.5 * args.rate_reg * float(np.sum((r_i - p_target) ** 2))
    g_rate = args.rate_reg * (r_i - p_target) / n_sim   # d(rate_loss)/d s_{i,t}
    loss += rate_loss

    carry_out = dict(V=V, ge=ge, gi=gi, wad=wad, tr=tr, refr=refr)
    tape = dict(V=tp_V, ge=tp_ge, gi=tp_gi, wad=tp_wad, tr=tp_tr,
                Vh=tp_Vh, s=tp_s, gate=tp_gate, drive=drive_win)
    return carry_out, loss, tape, readout_grads, gW_out, g_rate


def backward_window(W, tape, readout_grads, g_rate, params, args, smooth):
    '''Reverse-mode through the taped window. Returns gW (gradient wrt W).'''
    N = params['N']; n_exc = params['n_exc']
    tau_m = params['tau_m']; inv_tau_m = 1.0 / tau_m
    R = params['R']; V_rest = params['V_rest']; V_reset = params['V_reset']
    V_th = params['V_th']; E_exc = params['E_exc']; E_inh = params['E_inh']
    tau_syn = params['tau_syn']; dt = params['dt']
    adapt_on = params['adapt_b'] > 0.0
    adapt_b = params['adapt_b']
    adapt_decay = np.exp(-dt / params['tau_w']) if adapt_on else 0.0
    syn_decay = 1.0 - dt / tau_syn
    trace_decay = np.exp(-dt / args.tau_state)
    beta = args.beta

    exc_mask = np.zeros(N); exc_mask[:n_exc] = 1.0
    inh_mask = np.zeros(N); inh_mask[n_exc:] = 1.0

    n_sim = tape['V'].shape[0]
    gV = np.zeros(N); gge = np.zeros(N); ggi = np.zeros(N)
    gwad = np.zeros(N); gtr = np.zeros(N)
    gW = np.zeros_like(W)
    adj_clip = args.adj_clip

    def _clip(a):
        # per-step adjoint-norm clipping: keeps the through-time recursion from
        # blowing up (the conductance adjoint is amplified by ~|V - E| each step).
        n = np.linalg.norm(a)
        return a * (adj_clip / n) if n > adj_clip else a

    for t in range(n_sim - 1, -1, -1):
        V = tape['V'][t]; ge = tape['ge'][t]; gi = tape['gi'][t]
        wad = tape['wad'][t]; Vh = tape['Vh'][t]; s = tape['s'][t]
        gate = tape['gate'][t]

        if t in readout_grads:
            gtr = gtr + readout_grads[t]

        # --- synapse update (step 8): ge_new=(ge+W.T@x_exc)*syn_decay ---
        gge_mid = gge * syn_decay
        ggi_mid = ggi * syn_decay
        # grad to s through W.T@x : gs += mask*(W @ g_mid)
        gs = exc_mask * (W @ gge_mid) - inh_mask * (W @ ggi_mid)
        # grad to W : outer(x, g_mid), x sparse -> index fired rows
        x_exc = s * exc_mask
        x_inh = s * inh_mask
        nz_e = np.nonzero(x_exc)[0]
        if nz_e.size:
            gW[nz_e] += np.outer(x_exc[nz_e], gge_mid)
        nz_i = np.nonzero(x_inh)[0]
        if nz_i.size:
            gW[nz_i] -= np.outer(x_inh[nz_i], ggi_mid)
        # carry-in grads (ge, gi passthrough)
        gge_in = gge_mid
        ggi_in = ggi_mid

        # --- trace (step 7): tr_new = tr*trace_decay + s ---
        gs = gs + gtr
        gtr_in = gtr * trace_decay

        # --- rate-homeostasis term: per-neuron grad applied every step ---
        if args.rate_reg != 0.0:
            gs = gs + g_rate

        # --- adaptation (step 6): wad_new = wad*adapt_decay + adapt_b*s ---
        if adapt_on:
            gs = gs + adapt_b * gwad
            gwad_in = gwad * adapt_decay
        else:
            gwad_in = np.zeros(N)

        # --- reset (step 5): Vnew = Vh - (Vh - V_reset)*s ---
        gVh = gV * (1.0 - s)
        gs = gs + gV * (-(Vh - V_reset))

        # --- spike (step 4): s = phi(Vh - V_th)*gate ---
        gVh = gVh + gs * spike_surrogate_grad(Vh - V_th, smooth, beta) * gate

        # --- integrate (steps 3,2): Vh = V + dt*inv_tau_m*(...) ---
        gdV = gVh * dt
        gV_pre = gVh.copy()
        gV_pre += gdV * inv_tau_m * (-1.0)        # leak -(V-V_rest)
        gIsyn = gdV * inv_tau_m * R
        gwad_in += gdV * (-inv_tau_m)             # -wad term uses carry-in wad

        # --- I_syn (step 1): Isyn = -ge*(V-E_exc) - gi*(V-E_inh) ---
        gV_pre += gIsyn * (-ge - gi)
        gge_in += gIsyn * (-(V - E_exc))
        ggi_in += gIsyn * (-(V - E_inh))

        # roll to previous step (clip each adjoint to keep the recursion stable)
        gV = _clip(gV_pre); gge = _clip(gge_in); ggi = _clip(ggi_in)
        gwad = _clip(gwad_in); gtr = _clip(gtr_in)

    return gW


# ----------------------------------------------------------------------------
# Gradient check (tiny smooth net vs finite differences)
# ----------------------------------------------------------------------------
def gradient_check():
    rng = np.random.default_rng(0)
    N = 12; n_exc = 9
    hold = 5; chunk = 3; n_sim = hold * chunk
    n_ro = 6

    class A:
        pass
    args = A()
    args.tau_state = 20.0; args.beta = 1.0; args.refractory = 0.0; args.dt = 0.1
    args.adj_clip = 1e12   # effectively off, so the check tests the true gradient
    args.rate_reg = 0.7; args.target_rate = 15.0   # exercise the rate term too

    params = dict(
        N=N, n_exc=n_exc, tau_m=rng.uniform(15, 25, N), V_th=rng.uniform(-57, -53, N),
        I_ext=rng.uniform(1.4, 1.6, N), R=10.0, V_rest=-70.0, V_reset=-80.0,
        E_exc=0.0, E_inh=-80.0, tau_syn=5.0, dt=0.1, adapt_b=4.0, tau_w=400.0,
    )
    W = rng.normal(0, 0.01, (N, N)); np.fill_diagonal(W, 0.0)
    W[n_exc:] = -np.abs(W[n_exc:])
    readout_idx = np.arange(n_ro)
    W_out = rng.normal(0, 0.1, n_ro + 1)

    drive = rng.normal(0, 0.5, (n_sim, N))
    targets = [(hold * (k + 1) - 1, rng.uniform(-1, 1)) for k in range(chunk)]
    carry = dict(V=np.full(N, -65.0), ge=np.zeros(N), gi=np.zeros(N),
                 wad=np.zeros(N), tr=np.zeros(N), refr=np.zeros(N))

    def loss_of(Wmat):
        out = forward_window(Wmat, W_out, carry, drive, targets,
                             readout_idx, params, args, smooth=True)
        return out[1]

    _, _, tape, rg, _, g_rate = forward_window(W, W_out, carry, drive, targets,
                                               readout_idx, params, args, smooth=True)
    gW = backward_window(W, tape, rg, g_rate, params, args, smooth=True)

    eps = 1e-6
    max_rel = 0.0
    rngc = np.random.default_rng(1)
    for _ in range(40):
        i = rngc.integers(N); j = rngc.integers(N)
        Wp = W.copy(); Wp[i, j] += eps
        Wm = W.copy(); Wm[i, j] -= eps
        num = (loss_of(Wp) - loss_of(Wm)) / (2 * eps)
        ana = gW[i, j]
        denom = max(1.0, abs(num), abs(ana))
        rel = abs(num - ana) / denom
        max_rel = max(max_rel, rel)
    print(f'[gradcheck] max relative error over 40 entries: {max_rel:.2e}')
    return max_rel


# ----------------------------------------------------------------------------
# Adam
# ----------------------------------------------------------------------------
class Adam:
    def __init__(self, shape, lr, b1=0.9, b2=0.999, eps=1e-8):
        self.m = np.zeros(shape); self.v = np.zeros(shape)
        self.lr = lr; self.b1 = b1; self.b2 = b2; self.eps = eps; self.t = 0

    def step(self, g):
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * g
        self.v = self.b2 * self.v + (1 - self.b2) * (g * g)
        mhat = self.m / (1 - self.b1 ** self.t)
        vhat = self.v / (1 - self.b2 ** self.t)
        return self.lr * mhat / (np.sqrt(vhat) + self.eps)


# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------
def train(args):
    t0 = time.time()
    W, conn_mask, w_ceiling, n_exc, params = build_fresh_network(args)
    N = params['N']; dt = params['dt']; hold = args.hold
    W_initial = W.copy()

    rng = np.random.default_rng(args.seed + 3)
    readout_idx = np.sort(rng.choice(N, size=min(args.n_readout, N), replace=False))
    W_out = np.zeros(len(readout_idx) + 1)

    # input stream (held per reservoir-step), injected into input block
    n_rsteps = args.steps
    u = rng.uniform(-1.0, 1.0, n_rsteps + args.target_delay)
    exc_rows = np.arange(N) < n_exc

    opt_W = Adam(W.shape, args.lr_w)
    opt_o = Adam(W_out.shape, args.lr_out)

    chunk = args.chunk
    win_sim = chunk * hold
    carry = dict(V=np.full(N, -65.0), ge=np.zeros(N), gi=np.zeros(N),
                 wad=np.zeros(N), tr=np.zeros(N), refr=np.zeros(N))

    # warmup (no learning) to wash out the transient
    warm_rsteps = args.washout
    print(f'[init] N={N} n_exc={n_exc} readout={len(readout_idx)} '
          f'delay={args.target_delay} chunk={chunk} adapt_b={args.adapt_b}')

    rstep = 0
    report_every = max(1, args.report_every)
    recent = []
    while rstep < n_rsteps:
        this_chunk = min(chunk, n_rsteps - rstep)
        win_sim = this_chunk * hold
        drive_win = np.zeros((win_sim, N))
        targets = []
        for c in range(this_chunk):
            r = rstep + c
            val = u[r] * args.input_scale
            drive_win[c * hold:(c + 1) * hold, :args.n_input] = val
            # target at the END of this reservoir-step: reconstruct u[r-delay]
            tr_idx = r - args.target_delay
            if rstep + c >= warm_rsteps and tr_idx >= 0:
                targets.append((c * hold + hold - 1, u[tr_idx]))

        smooth = False
        carry_out, loss, tape, rg, gWout, g_rate = forward_window(
            W, W_out, carry, drive_win, targets, readout_idx, params, args, smooth)

        if targets:
            gW = backward_window(W, tape, rg, g_rate, params, args, smooth)
            n_t = len(targets)
            gW /= n_t; gWout = gWout / n_t
            # skip poisoned steps rather than corrupting the weights
            if not (np.all(np.isfinite(gW)) and np.all(np.isfinite(gWout))):
                carry = {k: v.copy() for k, v in carry_out.items()}
                rstep += this_chunk
                continue
            # mask gradient to the frozen sparsity pattern
            gW *= conn_mask
            # global gradient-norm clipping for stability
            gn = np.linalg.norm(gW)
            if gn > args.clip:
                gW *= args.clip / gn
            W -= opt_W.step(gW)
            W_out -= opt_o.step(gWout)
            # enforce Dale's law sign + sparsity + ceiling
            W *= conn_mask
            W[exc_rows] = np.clip(W[exc_rows], 0.0, None)
            W[~exc_rows] = np.clip(W[~exc_rows], None, 0.0)
            np.clip(W, -w_ceiling, w_ceiling, out=W)
            recent.append(loss / max(1, len(targets)))

        # detach carry across windows (truncated BPTT)
        carry = {k: v.copy() for k, v in carry_out.items()}
        rstep += this_chunk

        if recent and rstep % report_every < chunk:
            mse = float(np.mean(recent[-50:]))
            # mean firing rate over the last window (sanity)
            rate = tape['s'].mean() / dt * 1000.0
            print(f'[rstep {rstep:5d}/{n_rsteps}] recall MSE={mse:.4f} '
                  f'rate={rate:5.1f}Hz  ({time.time()-t0:.0f}s)')

    final_mse = float(np.mean(recent[-50:])) if recent else float('nan')
    print(f'[done] final recall MSE={final_mse:.4f} '
          f'(chance for U(-1,1) = {1/3:.4f})  total {time.time()-t0:.0f}s')

    os.makedirs(args.outdir, exist_ok=True)
    out = os.path.join(args.outdir, f'run_N{N}_seed{args.seed}.npz')
    np.savez(out,
             W_initial=W_initial, W_final=W, N=N, seed=args.seed, dt=dt,
             target_delay=args.target_delay, adapt_b=args.adapt_b,
             tau_w=args.tau_w, tau_syn=params['tau_syn'],
             final_mse=final_mse)
    print(f'[saved] {out}')
    return out


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--N', type=int, default=2000)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--steps', type=int, default=4000,
                   help='number of reservoir-steps of input to train on.')
    p.add_argument('--target-delay', type=int, default=2,
                   help='reconstruct u(t - this many reservoir-steps).')
    p.add_argument('--hold', type=int, default=150,
                   help='sim-steps per reservoir-step (1 rstep = hold*dt ms).')
    p.add_argument('--chunk', type=int, default=4,
                   help='reservoir-steps per truncated-BPTT window. Must exceed '
                        'target_delay so the recall target lies inside the window.')
    p.add_argument('--dt', type=float, default=0.1)
    # network recipe (validated AI regime)
    p.add_argument('--connectivity', type=float, default=0.05)
    p.add_argument('--w-scale', type=float, default=0.005)
    p.add_argument('--w-max-mult', type=float, default=4.0)
    p.add_argument('--g-inh', type=float, default=1.0)
    p.add_argument('--tau-syn', type=float, default=5.0)
    p.add_argument('--tau-m-min', type=float, default=15.0)
    p.add_argument('--tau-m-max', type=float, default=25.0)
    p.add_argument('--background-mean', type=float, default=1.5)
    p.add_argument('--background-spread', type=float, default=0.1)
    p.add_argument('--refractory', type=float, default=2.0,
                   help='refractory period (ms). Set 0 to disable.')
    # adaptation (the slow memory channel BPTT shapes)
    p.add_argument('--adapt-b', type=float, default=4.0)
    p.add_argument('--tau-w', type=float, default=400.0)
    # readout / loss
    p.add_argument('--n-readout', type=int, default=600)
    p.add_argument('--tau-state', type=float, default=20.0)
    p.add_argument('--input-scale', type=float, default=8.0)
    p.add_argument('--n-input', type=int, default=100)
    p.add_argument('--washout', type=int, default=20,
                   help='reservoir-steps to run before learning starts.')
    # surrogate / optimisation
    p.add_argument('--beta', type=float, default=1.0,
                   help='steepness of the fast-sigmoid surrogate gradient.')
    p.add_argument('--lr-w', type=float, default=5e-5)
    p.add_argument('--lr-out', type=float, default=1e-2)
    p.add_argument('--clip', type=float, default=1.0,
                   help='max L2 norm of the W gradient per step.')
    p.add_argument('--adj-clip', type=float, default=1e3,
                   help='max L2 norm of each per-step adjoint in BPTT (stops the '
                        'through-time recursion from overflowing).')
    p.add_argument('--rate-reg', type=float, default=0.5,
                   help='weight of the rate-homeostasis penalty (0 = off).')
    p.add_argument('--target-rate', type=float, default=15.0,
                   help='target mean firing rate (Hz) for the homeostasis term.')
    p.add_argument('--report-every', type=int, default=100)
    p.add_argument('--outdir', default='results_mem_bptt')
    p.add_argument('--gradcheck', action='store_true',
                   help='run the finite-difference gradient check and exit.')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if args.gradcheck:
        gradient_check()
    else:
        train(args)
