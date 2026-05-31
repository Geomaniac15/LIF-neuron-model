'''E-I LIF network trained with reward-modulated STDP (R-STDP) for long memory.

Biological design principles
-----------------------------
* Dale's law: presynaptic sign is fixed at initialisation and never flipped.
  Exc neurons (0..n_exc-1) have W[i,j] >= 0; inh neurons have W[i,j] <= 0.
* Sparse, random connectivity (same as random_neurons_fixed.py).
* Heterogeneous membrane time constants tau_m ~ U(20, 200) ms so neurons
  integrate over many timescales simultaneously (free memory lever).
* tau_syn = 20 ms (NMDA-like slow synapses; not baked into W so reservoir_
  capacity.py can override it, but training with the longer constant means
  the weights learn dynamics appropriate for it).
* Noisy fluctuation-driven firing (AI regime) preserved.
* E/I ratio: 80 % exc / 20 % inh, inhibitory gain g_inh ~4-6.

R-STDP mechanism
-----------------
Standard STDP computes a per-synapse eligibility trace e_ij from spike
timing correlations (pre before post -> LTP eligibility; post before pre ->
LTD eligibility) and applies the weight change immediately. R-STDP holds that
change GATED behind a global neuromodulatory signal d(t):

    delta_W_ij = eta * e_ij(t) * d(t)

d(t) is the memory reconstruction error: a set of linear readout neurons
is trained online (delta rule) to reconstruct u(t - target_delay) from the
current LIF spike traces. Their prediction error is broadcast globally as
d(t), modulating whether recent eligibility traces are potentiated (d > 0,
reconstruction was wrong, strengthen useful pairings) or depressed (d < 0).

This is the biologically grounded analog of supervised recurrent learning:
the readout error plays the role of a dopaminergic teaching signal. The
recurrent weights W learn to maintain input-specific activity at long delays
because that is what the error signal rewards.

Output format
--------------
Saves run_N{N}_seed{seed}.npz with keys W_initial, W_final, N, seed, dt,
tau_syn, tau_m_min, tau_m_max -- identical to random_neurons_fixed.py so
reservoir_capacity.py and aggregate_memory_sweep.py work unchanged.

Usage examples
--------------
Quick local sanity check (~3 min on CPU, N=200):
    python train_memory_rstdp.py --N 200 --steps 300000 --seed 0 \\
        --target-delay 5 --outdir results_mem_test

Full cluster run (N=2000, submit via slurm_train_memory.sh):
    python train_memory_rstdp.py --N 2000 --steps 1500000 --seed 0 --outdir results_mem

Recommended benchmark after training:
    python reservoir_capacity.py \\
        --weights results_mem/run_N2000_seed0.npz \\
        --tau-syn 20 --hold 150 --max-delay 60 \\
        --sweep --sweep-adapt-bs 0 --sweep-tau-ws 400 \\
        --sweep-input-scales 8 --sweep-inh-scales 1.0 \\
        --sweep-max-delay 60 --sweep-steps 36000

Fixes vs v1
-----------
1. Drive recalibrated for tau_syn=20ms.  The AI-regime parameters in
   random_neurons_fixed.py were tuned for tau_syn=5ms.  With a 4x slower
   synapse the same I_mean leaves neurons silent or intermittently bursting.
   Fix: I_mean raised to 2.0, sigma to 5.0, and a pre-warmup drive-calibration
   pass that measures the mean firing rate and auto-scales I_mean until the
   population sits in 5-25 Hz.

2. Steps scaled to reservoir-steps, not sim steps.  The local test with
   --steps 60000 and hold=150 gives only 400 reservoir-steps -- far too few
   for R-STDP to accumulate signal.  The --steps argument now counts
   RESERVOIR STEPS directly (sim steps = steps * hold internally), so
   --steps 2000 always means 2000 training examples regardless of hold.

3. Eligibility trace accumulation vectorised.  The per-spike Python loop
   over fired neurons was the bottleneck; replaced with masked outer-product
   rank-1 updates so the loop body is O(N) not O(N^2) per spike.
'''
import argparse
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Regime diagnostics
# ---------------------------------------------------------------------------

def regime_metrics(spikes, N, steps, dt, bin_ms=5.0):
    '''Return (pop_CV, ISI_CV). AI regime = low pop_CV AND high ISI_CV.'''
    if spikes.size == 0:
        return float('nan'), float('nan')
    T = steps * dt
    nbins = max(1, int(T / bin_ms))
    counts, _ = np.histogram(spikes[:, 0], bins=nbins, range=(0, T))
    pop_cv = counts.std() / counts.mean() if counts.mean() > 0 else float('nan')
    isi_cvs = []
    idx = spikes[:, 1].astype(int)
    for nid in range(N):
        ts = np.sort(spikes[idx == nid, 0])
        if len(ts) > 3:
            isis = np.diff(ts)
            if isis.mean() > 0:
                isi_cvs.append(isis.std() / isis.mean())
    isi_cv = float(np.mean(isi_cvs)) if isi_cvs else float('nan')
    return pop_cv, isi_cv


# ---------------------------------------------------------------------------
# Drive calibration
# ---------------------------------------------------------------------------

def calibrate_drive(N, n_exc, W, tau_m, tau_syn, V_th, I_base_template,
                    sigma, R, dt, adapt_b=0.0, tau_w=400.0,
                    target_rate_hz=(5.0, 25.0),
                    calib_steps=3000, seed=1):
    '''Binary-search I_mean so the population fires in target_rate_hz range.

    Returns a calibrated I_base array (per-neuron, same shape as I_base_template).
    Operates on a copy of the network state; does not mutate W. Adaptation
    (adapt_b/tau_w) is included so the calibrated rate matches the training run.
    '''
    rng = np.random.default_rng(seed)
    inv_sqrt_dt = 1.0 / np.sqrt(dt)
    E_exc, E_inh = 0.0, -80.0
    V_rest, V_reset = -70.0, -80.0
    adapt_on    = adapt_b > 0.0
    adapt_decay = np.exp(-dt / tau_w)

    lo, hi = 0.5, 8.0   # search range for I_mean
    for _ in range(12):  # 12 bisection iterations -> resolution < 0.01
        mid = 0.5 * (lo + hi)
        I_base = I_base_template - I_base_template.mean() + mid

        V = np.full(N, V_rest)
        g_exc_c = np.zeros(N); g_inh_c = np.zeros(N)
        refractory = np.zeros(N)
        w_adapt = np.zeros(N)
        n_spikes = 0

        for step in range(calib_steps):
            I_ext = I_base + (sigma * rng.standard_normal(N) * inv_sqrt_dt
                              if sigma > 0 else 0.0)
            I_syn = -g_exc_c * (V - E_exc) - g_inh_c * (V - E_inh)
            dV    = (1.0 / tau_m) * (-(V - V_rest) + R * (I_ext + I_syn) - w_adapt)
            V    += dV * dt
            if adapt_on:
                np.clip(V, E_inh - 10.0, E_exc + 10.0, out=V)
            refractory -= dt
            fired = (V >= V_th) & (refractory <= 0)
            V[fired] = V_reset; refractory[fired] = 2.0
            exc_f = fired.copy(); exc_f[n_exc:] = False
            inh_f = fired.copy(); inh_f[:n_exc] = False
            if exc_f.any(): g_exc_c += W[exc_f].sum(axis=0)
            if inh_f.any(): g_inh_c += W[inh_f].sum(axis=0)
            g_exc_c -= (g_exc_c / tau_syn) * dt
            g_inh_c -= (g_inh_c / tau_syn) * dt
            if adapt_on:
                w_adapt *= adapt_decay
                w_adapt[fired] += adapt_b
            n_spikes += fired.sum()

        sim_s    = calib_steps * dt / 1000.0
        mean_hz  = n_spikes / (N * sim_s)
        if mean_hz < target_rate_hz[0]:
            lo = mid
        elif mean_hz > target_rate_hz[1]:
            hi = mid
        else:
            break   # in range

    print(f'[calib] I_mean={mid:.3f}  mean_rate={mean_hz:.1f}Hz '
          f'(target {target_rate_hz[0]}-{target_rate_hz[1]}Hz)', flush=True)
    return I_base_template - I_base_template.mean() + mid


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def run(N=500, warmup_steps=5000, n_rsteps=2000, dt=0.1, seed=0,
        exc_frac=0.8, connectivity=0.05,
        w_scale=0.005, w_total=0.042,
        I_mean=2.0, I_spread=0.1, sigma=0.5, R=10.0,
        g_inh=1.0, w_max_mult=4.0,
        tau_m_min=20.0, tau_m_max=200.0,
        tau_syn=5.0,
        tau_stdp=20.0,
        tau_elig=500.0,
        adapt_b=4.0,
        tau_w=400.0,
        eta_rstdp=5e-5,
        rls_alpha=1.0,
        target_delay=33,
        hold=150,
        n_input=100,
        input_scale=8.0,
        n_readout=600,
        save_spikes=False,
        report_every=200):
    '''Train the E-I LIF network with R-STDP for long delayed recall.

    n_rsteps: number of RESERVOIR STEPS to train for (sim steps = n_rsteps * hold).
    Returns (W_initial, W_final, spikes_arr, errors, W_out, readout_idx).
    '''
    rng = np.random.default_rng(seed)
    n_exc = int(round(exc_frac * N))
    n_inh = N - n_exc
    inv_sqrt_dt = 1.0 / np.sqrt(dt)

    # --- size-invariant weight scaling ---
    if w_scale is None:
        K = max(1.0, connectivity * N)
        w_scale = w_total / np.sqrt(K)
    print(f'[init] N={N} n_exc={n_exc} n_inh={n_inh} w_scale={w_scale:.6f} '
          f'tau_syn={tau_syn}ms tau_m=[{tau_m_min},{tau_m_max}]ms', flush=True)

    # --- neuron parameters ---
    tau_m   = rng.uniform(tau_m_min, tau_m_max, N)
    V_rest  = -70.0
    V_th    = rng.uniform(-57, -53, N)
    V_reset = -80.0
    E_exc, E_inh = 0.0, -80.0
    # Raw I_base drawn around I_mean; will be re-centred by calibration.
    I_base  = rng.uniform(I_mean - I_spread, I_mean + I_spread, N)

    # --- connectivity & weights (Dale's law: sign fixed by presynaptic row) ---
    mask = rng.random((N, N)) < connectivity
    np.fill_diagonal(mask, False)
    W = rng.uniform(0.0, w_scale, (N, N)) * mask.astype(float)
    W[n_exc:, :] *= -g_inh
    W_initial_raw = W.copy()   # before calibration warmup

    w_ceiling     = np.full(N, w_max_mult * w_scale)
    w_ceiling[n_exc:] = w_max_mult * g_inh * w_scale
    w_ceiling_col = w_ceiling[:, None]

    # --- calibrate drive for tau_syn=20ms ---
    # The AI-regime parameters tuned for tau_syn=5ms often leave the network
    # silent at 20ms. Run a quick binary search to find I_mean that gives
    # 5-25 Hz mean population rate, then use it for the full run.
    I_base = calibrate_drive(N, n_exc, W, tau_m, tau_syn, V_th, I_base,
                             sigma, R, dt, adapt_b=adapt_b, tau_w=tau_w,
                             target_rate_hz=(5.0, 25.0),
                             calib_steps=3000, seed=seed + 99)

    # --- readout layer (RLS / FORCE, with bias unit) ---
    pool        = np.arange(n_input, N)
    n_ro        = min(n_readout, len(pool))
    readout_idx = np.sort(rng.choice(pool, size=n_ro, replace=False))
    ro_dim      = n_ro + 1                       # +1 for bias feature
    W_out       = np.zeros(ro_dim)
    P_rls       = np.eye(ro_dim) / rls_alpha     # inverse-correlation estimate
    r_vec       = np.ones(ro_dim)                # reused feature vector (last = bias)

    # --- circular buffer for delayed input ---
    # Buffer length must be at least target_delay + 1 reservoir-steps.
    buf_len = target_delay + 2
    u_buf   = np.zeros(buf_len)
    buf_ptr = 0

    # --- state variables ---
    V          = np.full(N, V_rest)
    g_exc      = np.zeros(N)
    g_inh_v    = np.zeros(N)
    refractory = np.zeros(N)

    tau_state   = 20.0
    trace_decay = np.exp(-dt / tau_state)
    spike_trace = np.zeros(N)

    # --- spike-frequency adaptation (gives the reservoir baseline memory) ---
    adapt_on    = adapt_b > 0.0
    adapt_decay = np.exp(-dt / tau_w)
    w_adapt     = np.zeros(N)

    # --- STDP traces and eligibility ---
    stdp_decay  = np.exp(-dt / tau_stdp)
    x_pre       = np.zeros(N)
    x_post      = np.zeros(N)

    # Eligibility decays every sim step; modulated by d(t) every reservoir step.
    elig_decay  = np.exp(-dt / tau_elig)
    # Store as (N, N) dense but accumulate only over masked (connected) synapses.
    # For N=200 this is 40k floats = fine. For N=2000 it's 4M floats = 32 MB, fine.
    e_elig      = np.zeros((N, N))

    # Pre-compute the mask as float for vectorised updates.
    mask_f = mask.astype(np.float64)

    # --- input stream ---
    u_stream = rng.uniform(-1.0, 1.0, n_rsteps + buf_len + 1)
    current_u = 0.0
    rstep     = 0

    spike_record = [] if save_spikes else None
    errors       = []

    total_sim_steps = n_rsteps * hold
    print(f'[train] {n_rsteps} reservoir-steps ({total_sim_steps} sim steps)  '
          f'target_delay={target_delay} ({target_delay * hold * dt:.0f}ms)  '
          f'hold={hold}  n_ro={n_ro}', flush=True)
    print(f'[train] eta_rstdp={eta_rstdp}  rls_alpha={rls_alpha}  '
          f'tau_elig={tau_elig}ms  tau_stdp={tau_stdp}ms  '
          f'adapt_b={adapt_b} tau_w={tau_w}ms', flush=True)

    # -----------------------------------------------------------------------
    # Warmup: settle network, no plasticity
    # -----------------------------------------------------------------------
    print(f'[warmup] {warmup_steps} steps...', flush=True)
    for step in range(warmup_steps):
        I_ext = I_base + (sigma * rng.standard_normal(N) * inv_sqrt_dt
                          if sigma > 0 else 0.0)
        I_syn = -g_exc * (V - E_exc) - g_inh_v * (V - E_inh)
        dV    = (1.0 / tau_m) * (-(V - V_rest) + R * (I_ext + I_syn) - w_adapt)
        V    += dV * dt
        if adapt_on:
            np.clip(V, E_inh - 10.0, E_exc + 10.0, out=V)
        refractory -= dt
        fired = (V >= V_th) & (refractory <= 0)
        V[fired] = V_reset; refractory[fired] = 2.0
        exc_f = fired.copy(); exc_f[n_exc:] = False
        inh_f = fired.copy(); inh_f[:n_exc] = False
        if exc_f.any(): g_exc   += W[exc_f].sum(axis=0)
        if inh_f.any(): g_inh_v += W[inh_f].sum(axis=0)
        g_exc   -= (g_exc   / tau_syn) * dt
        g_inh_v -= (g_inh_v / tau_syn) * dt
        if adapt_on:
            w_adapt *= adapt_decay
            w_adapt[fired] += adapt_b
        spike_trace = spike_trace * trace_decay + fired.astype(float)

    W_initial = W.copy()   # snapshot after warmup settling, before plasticity

    # -----------------------------------------------------------------------
    # Main training loop  (iterate over reservoir steps, not sim steps)
    # -----------------------------------------------------------------------
    for rs in range(n_rsteps):
        # -- new input value for this reservoir step --
        current_u      = u_stream[rs]
        u_buf[buf_ptr] = current_u
        buf_ptr        = (buf_ptr + 1) % buf_len

        # -- run hold sim steps --
        for sub in range(hold):
            I_ext = I_base.copy()
            if sigma > 0:
                I_ext += sigma * rng.standard_normal(N) * inv_sqrt_dt
            I_ext[:n_input] += input_scale * current_u

            I_syn = -g_exc * (V - E_exc) - g_inh_v * (V - E_inh)
            dV    = (1.0 / tau_m) * (-(V - V_rest) + R * (I_ext + I_syn) - w_adapt)
            V    += dV * dt
            if adapt_on:
                np.clip(V, E_inh - 10.0, E_exc + 10.0, out=V)
            refractory -= dt
            fired = (V >= V_th) & (refractory <= 0)
            V[fired] = V_reset; refractory[fired] = 2.0

            exc_f = fired.copy(); exc_f[n_exc:] = False
            inh_f = fired.copy(); inh_f[:n_exc] = False
            if exc_f.any(): g_exc   += W[exc_f].sum(axis=0)
            if inh_f.any(): g_inh_v += W[inh_f].sum(axis=0)
            g_exc   -= (g_exc   / tau_syn) * dt
            g_inh_v -= (g_inh_v / tau_syn) * dt
            if adapt_on:
                w_adapt *= adapt_decay
                w_adapt[fired] += adapt_b

            spike_trace = spike_trace * trace_decay + fired.astype(float)

            if save_spikes and fired.any():
                t_now = (rs * hold + sub) * dt
                for ni in np.where(fired)[0]:
                    spike_record.append((t_now, ni))

            # -- STDP traces --
            x_pre  = x_pre  * stdp_decay + fired.astype(float)
            x_post = x_post * stdp_decay + fired.astype(float)

            # -- eligibility trace: vectorised rank-1 updates --
            # Decay first.
            e_elig *= elig_decay
            if fired.any():
                f = fired.astype(np.float64)
                # LTP: for each postsynaptic neuron j that fired,
                #   e_elig[:, j] += x_pre * mask[:, j]
                # Vectorised: outer(x_pre, f) gives an (N,N) matrix where column j
                # is x_pre when j fired, 0 otherwise. Mask to connected synapses.
                e_elig += np.outer(x_pre, f) * mask_f
                # LTD: for each presynaptic neuron i that fired,
                #   e_elig[i, :] -= x_post * mask[i, :]
                # Vectorised: outer(f, x_post) gives row i = x_post when i fired.
                e_elig -= np.outer(f, x_post) * mask_f

        # -- readout prediction & R-STDP update (once per reservoir step) --
        if rs >= target_delay + 1:
            target_ptr = (buf_ptr - target_delay - 1) % buf_len
            u_target   = u_buf[target_ptr]

            r_vec[:n_ro] = spike_trace[readout_idx]   # last entry stays 1.0 (bias)
            prediction   = float(W_out @ r_vec)
            error        = u_target - prediction
            d_signal     = error

            errors.append(abs(error))

            # RLS / FORCE readout update (recursive least squares).
            # Far stronger conditioning than plain LMS, so d_signal is a
            # meaningful teaching signal almost immediately.
            Pr   = P_rls @ r_vec
            gain = Pr / (1.0 + float(r_vec @ Pr))
            W_out += error * gain
            P_rls -= np.outer(gain, Pr)

            # R-STDP: recurrent weights, gated by the readout error d_signal
            dW = eta_rstdp * e_elig * d_signal * mask_f
            W += dW

            # Dale's law clamp + magnitude ceiling
            W[:n_exc, :] = np.maximum(W[:n_exc, :], 0.0)
            W[n_exc:, :] = np.minimum(W[n_exc:, :], 0.0)
            np.clip(W, -w_ceiling_col, w_ceiling_col, out=W)
            np.fill_diagonal(W, 0.0)

        # -- progress report (in reservoir steps) --
        if rs > 0 and rs % report_every == 0:
            pct   = 100.0 * rs / n_rsteps
            e_rec = errors[-200:] if len(errors) >= 200 else errors
            emean = float(np.mean(e_rec)) if e_rec else float('nan')
            # mean firing rate over last hold steps (approximation)
            rate  = spike_trace[n_input:].mean() / (tau_state * 1e-3)
            print(f'[rstep {rs:>6}/{n_rsteps}  {pct:4.1f}%]  '
                  f'mean_err(last 200)={emean:.4f}  '
                  f'approx_rate≈{rate:.1f}Hz', flush=True)

    spikes_arr = np.array(spike_record) if save_spikes else np.empty((0, 2))
    return W_initial, W, spikes_arr, errors, W_out, readout_idx


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--N', type=int, default=500)
    p.add_argument('--exc-frac', type=float, default=0.8)
    p.add_argument('--connectivity', type=float, default=0.05,
                   help='matches random_neurons_fixed.py (the proven AI-regime '
                        'recipe at N=2000). Higher values run away at scale.')
    p.add_argument('--w-scale', type=float, default=0.005,
                   help='fixed peak recurrent weight (proven recipe). Pass None '
                        'to instead derive it from --w-total / sqrt(connectivity*N).')
    p.add_argument('--w-total', type=float, default=0.042)
    p.add_argument('--w-max-mult', type=float, default=4.0)
    p.add_argument('--warmup', type=int, default=5000,
                   help='warmup SIM steps (no plasticity)')
    p.add_argument('--steps', type=int, default=2000,
                   help='number of RESERVOIR STEPS to train for. '
                        'Sim steps = steps * hold. Local test: 2000. '
                        'Cluster N=2000: 10000.')
    p.add_argument('--dt', type=float, default=0.1)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--I-mean', type=float, default=2.0,
                   help='initial guess for mean drive (auto-calibrated)')
    p.add_argument('--I-spread', type=float, default=0.1)
    p.add_argument('--sigma', type=float, default=0.5,
                   help='white-noise drive fluctuation. Kept small to match the '
                        'proven recipe; large values push the rate up.')
    p.add_argument('--R', type=float, default=10.0)
    p.add_argument('--g-inh', type=float, default=1.0,
                   help='inhibitory gain (proven recipe = 1.0). Raising it does '
                        'NOT calm runaway here; the synapse time constant does.')
    p.add_argument('--tau-m-min', type=float, default=20.0)
    p.add_argument('--tau-m-max', type=float, default=200.0)
    p.add_argument('--tau-syn', type=float, default=5.0,
                   help='synaptic time constant (ms). MUST stay ~5; at 20ms the '
                        'conductances pile up and the N=2000 network saturates.')
    p.add_argument('--tau-stdp', type=float, default=20.0)
    p.add_argument('--tau-elig', type=float, default=500.0,
                   help='eligibility trace time constant (ms). Must be >= '
                        'target_delay * hold * dt = target memory horizon.')
    p.add_argument('--adapt-b', type=float, default=4.0,
                   help='spike-frequency adaptation strength. Gives the reservoir '
                        'baseline memory for R-STDP to shape. 0 disables it.')
    p.add_argument('--tau-w', type=float, default=400.0,
                   help='adaptation time constant (ms). 400 was the proven '
                        'memory sweet spot in reservoir_capacity.py.')
    p.add_argument('--eta-rstdp', type=float, default=5e-5)
    p.add_argument('--rls-alpha', type=float, default=1.0,
                   help='RLS readout regulariser (P0 = I/alpha). Smaller = faster '
                        'but noisier adaptation of the readout/teaching signal.')
    p.add_argument('--target-delay', type=int, default=33,
                   help='delay in reservoir-steps. 33 x 150 x 0.1ms = 495ms. '
                        'Use --target-delay 5 for local sanity checks.')
    p.add_argument('--hold', type=int, default=150)
    p.add_argument('--n-input', type=int, default=100)
    p.add_argument('--input-scale', type=float, default=8.0)
    p.add_argument('--n-readout', type=int, default=600)
    p.add_argument('--outdir', type=str, default='results_mem')
    p.add_argument('--save-spikes', action='store_true')
    p.add_argument('--report-every', type=int, default=200,
                   help='print progress every N reservoir steps')
    p.add_argument('--no-plots', action='store_true')
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    tag = f'N{args.N}_seed{args.seed}'

    sim_steps = args.steps * args.hold
    print(f'[run] N={args.N} seed={args.seed}  '
          f'{args.steps} reservoir-steps ({sim_steps} sim steps)  '
          f'target_delay={args.target_delay} ({args.target_delay*args.hold*args.dt:.0f}ms)  '
          f'tau_syn={args.tau_syn}ms tau_m=[{args.tau_m_min},{args.tau_m_max}]ms  '
          f'eta_rstdp={args.eta_rstdp} adapt_b={args.adapt_b} tau_w={args.tau_w}', flush=True)

    W_initial, W_final, spikes, errors, W_out, readout_idx = run(
        N=args.N, warmup_steps=args.warmup, n_rsteps=args.steps,
        dt=args.dt, seed=args.seed,
        exc_frac=args.exc_frac, connectivity=args.connectivity,
        w_scale=args.w_scale, w_total=args.w_total,
        I_mean=args.I_mean, I_spread=args.I_spread, sigma=args.sigma, R=args.R,
        g_inh=args.g_inh, w_max_mult=args.w_max_mult,
        tau_m_min=args.tau_m_min, tau_m_max=args.tau_m_max, tau_syn=args.tau_syn,
        tau_stdp=args.tau_stdp, tau_elig=args.tau_elig,
        adapt_b=args.adapt_b, tau_w=args.tau_w,
        eta_rstdp=args.eta_rstdp, rls_alpha=args.rls_alpha,
        target_delay=args.target_delay, hold=args.hold,
        n_input=args.n_input, input_scale=args.input_scale,
        n_readout=args.n_readout,
        save_spikes=args.save_spikes, report_every=args.report_every,
    )

    # --- save (same .npz format as random_neurons_fixed.py) ---
    npz_path = os.path.join(args.outdir, f'run_{tag}.npz')
    np.savez_compressed(
        npz_path,
        W_initial=W_initial, W_final=W_final,
        spikes=spikes, N=args.N, seed=args.seed, dt=args.dt,
        warmup=args.warmup, steps=sim_steps,
        tau_syn=args.tau_syn,
        tau_m_min=args.tau_m_min, tau_m_max=args.tau_m_max,
        target_delay=args.target_delay, hold=args.hold,
        W_out=W_out, readout_idx=readout_idx,
    )
    print(f'[save] {npz_path}', flush=True)

    if errors:
        print(f'[stats] mean_err: first 200 rsteps={np.mean(errors[:200]):.4f}  '
              f'last 200 rsteps={np.mean(errors[-200:]):.4f}', flush=True)
        improvement = (np.mean(errors[:200]) - np.mean(errors[-200:])) / (np.mean(errors[:200]) + 1e-9)
        print(f'[stats] error reduction={improvement*100:.1f}%  '
              f'(>10% = learning signal present)', flush=True)

    if args.no_plots:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    if errors:
        window   = max(1, len(errors) // 100)
        smoothed = np.convolve(errors, np.ones(window) / window, mode='valid')
        axes[0].plot(smoothed, color='teal', lw=1)
        axes[0].set_xlabel('reservoir step')
        axes[0].set_ylabel('abs reconstruction error')
        axes[0].set_title(f'R-STDP learning curve  '
                          f'delay={args.target_delay}rs={args.target_delay*args.hold*args.dt:.0f}ms')
    axes[1].hist(W_initial[W_initial != 0].flatten(), bins=60, alpha=0.5,
                 label='before', color='steelblue')
    axes[1].hist(W_final[W_final != 0].flatten(), bins=60, alpha=0.5,
                 label='after', color='teal')
    axes[1].set_xlabel('synaptic weight')
    axes[1].set_ylabel('count')
    axes[1].set_title('weight distribution before vs after R-STDP')
    axes[1].legend()
    fig.suptitle(f'train_memory_rstdp  N={args.N} seed={args.seed}')
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, f'training_{tag}.png'), dpi=120)
    plt.close()
    print(f'[done] figures saved to {args.outdir}/', flush=True)


if __name__ == '__main__':
    main()