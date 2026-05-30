'''E-I LIF network with STDP. Refactored for headless / batch runs on the Lancaster HEC.

Changes vs random_neurons.py:
  * Fixed sign in the conductance-based synaptic current. The original had
    I_syn = g_exc*(V-E_exc) + g_inh*(V-E_inh) which silently made excitatory
    synapses HYPERPOLARIZE their targets. With the correct sign, every spike
    drives V toward the reversal potential E_exc=0 (depolarizing) or E_inh=-80
    (hyperpolarizing).
  * Rescaled default weight magnitude from 0.5 to 0.005 (1/100), because the
    original weights were silently calibrated against the wrong-sign equation
    and would saturate the network once the sign is fixed.
  * Reduced default STDP learning rates from 0.01 to 0.002 to avoid pushing
    weights to the rails now that the network actually fires at 15-25 Hz.
  * Moved last_spike[i] = t_now to AFTER the W update so the self-loop term
    is unambiguous (was harmless before, just confusing).
  * Exposed w-scale, a-plus, a-minus, I-lo, I-hi on the CLI for sweeps.

Changes for the asynchronous-irregular (AI) regime (Brunel 2000-style):
  * NOISY DRIVE. The external current is now I_mean + sigma * N(0,1)/sqrt(dt),
    redrawn every step, instead of a constant per-neuron value. Constant drive
    makes a LIF neuron fire like a clock (ISI_CV -> 0); irregular firing REQUIRES
    a fluctuating input. The /sqrt(dt) keeps the noise amplitude independent of
    the time step (proper white-noise scaling), so dt stays interchangeable.
    --I-mean sets the mean drive; --sigma sets the fluctuation size. With
    --sigma 0 the model reduces to the old constant-drive behaviour, except the
    mean is now drawn in a small band around I_mean (see I_base below).
  * INHIBITORY GAIN. Inhibitory outgoing weights are scaled by -g_inh instead
    of just -1. The AI state is inhibition-dominated: inhibition has to be
    several times stronger than excitation (per synapse) to track and cancel it,
    otherwise the population synchronises into waves. --g-inh sets this; ~4-6 is
    a sensible starting range.
  * SELF-REPORTING REGIME STATS. The [stats] line now also prints pop_CV
    (population synchrony: CV of the binned population spike count; high => global
    rhythm/waves) and ISI_CV (mean per-neuron coefficient of variation of
    inter-spike intervals; ~1.0 => Poisson-like/irregular, ~0 => clock-like).
    Target AI regime: pop_CV low AND ISI_CV high. Lets a run self-diagnose
    without eyeballing the raster.

NOTE on reaching a *textbook* AI state: parameter tuning gets you a network that
looks asynchronous and irregular (low pop_CV, ISI_CV ~0.5-0.6), which is a solid
realistic baseline. Quantitatively Poisson firing (ISI_CV ~1.0) is capped here by
the 2 ms refractory floor at these rates and would need structural changes
(synaptic delays, lower operating rates, tighter E/I balance). Treat the AI
parameters below as a good starting regime, not a final calibration.

Local quick run:
    python random_neurons_fixed.py --N 100 --warmup 2000 --steps 3000 --seed 0 --outdir results

Local AI-regime sanity check (WITH spikes so you see the rate):
    python random_neurons_fixed.py --N 200 --warmup 2000 --steps 4000 --seed 0 \
        --I-mean 1.4 --sigma 3.0 --g-inh 5.0 --connectivity 0.1 --outdir results

Big run on the cluster:
    python random_neurons_fixed.py --N 2000 --warmup 5000 --steps 20000 --seed 0 \
        --I-mean 1.4 --sigma 3.0 --g-inh 5.0 --connectivity 0.1 --outdir results
'''
import argparse
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # headless: no display needed on compute nodes
import matplotlib.pyplot as plt


def run(N=100, warmup=2000, steps=3000, dt=0.1, seed=0,
        exc_frac=0.8, connectivity=0.05, save_spikes=True,
        w_scale=0.005, w_total=None, I_mean=1.5, I_spread=0.1, sigma=0.0, R=10.0,
        A_plus=0.002, A_minus=0.002, g_inh=1.0, w_max_mult=4.0):
    '''Run the E-I network. Returns (W_initial, W_final, spikes).

    spikes is an (M, 2) array of (time_ms, neuron_index). Pass save_spikes=False
    for very large runs where the spike list itself becomes the memory hog.

    Drive model: each neuron has a fixed baseline I_base drawn uniformly in
    [I_mean - I_spread, I_mean + I_spread] (heterogeneity across the population),
    plus, every step, a white-noise fluctuation sigma * N(0,1) / sqrt(dt) (within-
    neuron temporal noise). sigma=0 recovers constant per-neuron drive.

    g_inh scales the magnitude of inhibitory outgoing weights (inhibition-dominance).

    Weight scaling: the network's dynamical regime depends on the per-neuron
    TOTAL input, which is roughly (peak weight) x (number of inputs K), where
    K = connectivity * N. If you hold the peak weight fixed and grow N, total
    input grows with N and the network tips from asynchronous into synchronised
    bursting. To keep the regime invariant across sizes, pass w_total instead of
    w_scale: the peak weight is then set to w_total / sqrt(K), the balanced-network
    (1/sqrt(K)) scaling that keeps input FLUCTUATIONS constant as N changes (the
    relevant quantity for the asynchronous-irregular state). If w_total is None,
    the explicit w_scale is used as-is (manual mode / backward compatible).
    '''
    rng = np.random.default_rng(seed)
    n_exc = int(round(exc_frac * N))

    # Size-invariant weight scaling (overrides w_scale when w_total is given).
    if w_total is not None:
        K = max(1.0, connectivity * N)        # expected inputs per neuron
        w_scale = w_total / np.sqrt(K)

    V = np.full(N, -70.0)
    # Baseline drive: small static heterogeneity across neurons.
    I_base = rng.uniform(I_mean - I_spread, I_mean + I_spread, N)
    g_exc = np.zeros(N)
    g_inh_v = np.zeros(N)  # inhibitory CONDUCTANCE (renamed from g_inh to free the
                           # name for the inhibitory GAIN parameter g_inh)

    E_exc, E_inh = 0.0, -80.0
    tau_m = rng.uniform(15, 25, N)
    tau_syn = 5.0
    V_rest = -70.0
    V_threshold = rng.uniform(-57, -53, N)
    V_reset = -80.0

    W = rng.random((N, N)) * w_scale
    mask = rng.random((N, N)) > (1.0 - connectivity)
    W = W * mask
    np.fill_diagonal(W, 0)
    W[n_exc:, :] *= -g_inh  # inhibitory neurons: negative AND scaled by the gain
    refractory = np.zeros(N)

    tau_stdp = 40.0
    last_spike = np.full(N, -np.inf)

    # Soft-bound ceiling for |W|. Must exceed the initial peak weight (w_scale)
    # or synapses start at the ceiling with no room to potentiate. w_max_mult is
    # how many times the initial peak the ceiling sits at.
    #
    # PER-POPULATION ceiling. Inhibitory weights are initialised at magnitudes up
    # to g_inh*w_scale (see W[n_exc:, :] *= -g_inh above), which is g_inh times
    # the excitatory peak. A single scalar ceiling of w_max_mult*w_scale sits
    # BELOW the inhibitory starting magnitudes, so the clip/soft-bound slams every
    # inhibitory synapse onto the excitatory rail and the whole inhibitory
    # population collapses onto one value (a delta spike in the weight histogram).
    # The ceiling therefore has to be scaled by g_inh for inhibitory rows.
    # w_ceiling is indexed by PRESYNAPTIC neuron (the row of W), since the row
    # determines whether a synapse is excitatory or inhibitory.
    w_max = w_max_mult * w_scale            # excitatory ceiling (kept for logs)
    w_ceiling = np.full(N, w_max)
    w_ceiling[n_exc:] = w_max_mult * g_inh * w_scale
    w_ceiling_col = w_ceiling[:, None]      # (N,1) for per-row clipping

    inv_sqrt_dt = 1.0 / np.sqrt(dt)

    # ---- warmup (no STDP) ----
    for step in range(warmup):
        # Noisy external drive: baseline + white-noise fluctuation (if sigma>0).
        I_ext = I_base
        if sigma > 0.0:
            I_ext = I_base + sigma * rng.standard_normal(N) * inv_sqrt_dt
        # Conductance-based synaptic current: I = -g*(V-E) (sign-fixed).
        I_syn = -g_exc * (V - E_exc) - g_inh_v * (V - E_inh)
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
            g_inh_v += W[inh_fired].sum(axis=0)

        g_exc -= (g_exc / tau_syn) * dt
        g_inh_v -= (g_inh_v / tau_syn) * dt

    W_initial = W.copy()
    spike_record = [] if save_spikes else None

    # ---- main run with STDP ----
    # NOTE: kept as an explicit loop because STDP updates within a step are
    # order-dependent (last_spike is mutated as we iterate).
    for step in range(steps):
        I_ext = I_base
        if sigma > 0.0:
            I_ext = I_base + sigma * rng.standard_normal(N) * inv_sqrt_dt
        I_syn = -g_exc * (V - E_exc) - g_inh_v * (V - E_inh)  # FIX: see comment above
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
                g_inh_v += W[i, :]

            if save_spikes:
                spike_record.append((t_now, i))

            # delta_t computed BEFORE updating last_spike[i] so the self term
            # uses the previous spike of neuron i (harmless either way because
            # W[i, i] = 0, but this reads more naturally).
            delta_t = t_now - last_spike
            decay = np.exp(-delta_t / tau_stdp)

            # SOFT-BOUND (multiplicative) STDP. Additive STDP (the previous
            # `W += A_plus*decay`) is unstable: it relies on np.clip to catch
            # runaway, so over a long run individual synapses saturate at the
            # rail and their targets fire continuously (the "streak" neurons in
            # the raster). Soft bounds remove that: the potentiation step shrinks
            # to zero as a synapse approaches its ceiling, and depression shrinks
            # as it approaches zero, so weights asymptote smoothly instead of
            # slamming the bound. This gives a stable weight distribution.
            #
            # The network has BOTH signs of weight (excitatory >=0, inhibitory
            # <=0), so we bound on MAGNITUDE and preserve sign: potentiation
            # pushes |W| toward w_max, depression pushes |W| toward 0.
            #   headroom = (w_max - |W|)  -> ~0 near the ceiling, kills potentiation
            #   |W|                       -> ~0 near zero,        kills depression
            # W[:, i] is the column of synapses INTO neuron i, indexed by the
            # presynaptic neuron (row j). w_ceiling is also indexed by row j, so
            # each synapse potentiates toward its own population's ceiling.
            sign_col = np.sign(W[:, i])
            head_col = (w_ceiling - np.abs(W[:, i]))
            W[:, i] += A_plus * decay * head_col * sign_col * (W[:, i] != 0)

            sign_row = np.sign(W[i, :])
            mag_row = np.abs(W[i, :])
            W[i, :] -= A_minus * decay * mag_row * sign_row * (W[i, :] != 0)

            last_spike[i] = t_now
            # Safety clip retained as a backstop; with soft bounds it should
            # rarely if ever bind. Per-row bounds so inhibitory synapses are
            # clipped to their (larger) ceiling, not the excitatory one.
            np.clip(W, -w_ceiling_col, w_ceiling_col, out=W)

        g_exc -= (g_exc / tau_syn) * dt
        g_inh_v -= (g_inh_v / tau_syn) * dt

    spikes_arr = np.array(spike_record) if save_spikes else np.empty((0, 2))
    return W_initial, W, spikes_arr


def regime_metrics(spikes, N, steps, dt, bin_ms=5.0):
    '''Return (pop_CV, ISI_CV) describing where the run sits in the dynamical map.

    pop_CV: CV of the population spike count in bin_ms windows. High => the whole
            population fires in synchronised waves (global rhythm). Low => spikes
            are spread evenly in time (asynchronous).
    ISI_CV: mean over neurons of the per-neuron inter-spike-interval CV. ~0 =>
            clock-like/regular firing; ~1.0 => Poisson-like/irregular.
    AI regime = low pop_CV AND high ISI_CV.
    '''
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


def main():
    p = argparse.ArgumentParser(description='E-I LIF network with STDP (sign-fixed, AI-capable).')
    p.add_argument('--N', type=int, default=100, help='number of neurons')
    p.add_argument('--warmup', type=int, default=2000, help='warmup steps (no STDP)')
    p.add_argument('--steps', type=int, default=3000, help='main run steps (with STDP)')
    p.add_argument('--dt', type=float, default=0.1, help='step size (ms)')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--exc-frac', type=float, default=0.8)
    p.add_argument('--connectivity', type=float, default=0.05,
                   help='connection probability (denser helps inhibition track excitation; ~0.1 for AI)')
    p.add_argument('--w-scale', type=float, default=0.005,
                   help='peak initial weight magnitude (manual mode; ignored if --w-total set)')
    p.add_argument('--w-total', type=float, default=None,
                   help='size-invariant weight target: peak weight = w_total/sqrt(K), '
                        'K=connectivity*N. Use this instead of --w-scale to keep the '
                        'dynamical regime fixed across N. (e.g. ~0.0042 reproduces '
                        'w_scale=0.0003 at N=2000, conn=0.1)')
    # --- drive model ---
    p.add_argument('--I-mean', type=float, default=1.5,
                   help='mean external drive (rheobase ~1.5 for these params; '
                        'near/below threshold + noise gives fluctuation-driven firing)')
    p.add_argument('--I-spread', type=float, default=0.1,
                   help='half-width of static per-neuron drive heterogeneity around I_mean')
    p.add_argument('--sigma', type=float, default=0.0,
                   help='white-noise drive amplitude (0 = constant drive = clock-like firing; '
                        '~3 gives irregular firing). Scaled internally by 1/sqrt(dt).')
    p.add_argument('--R', type=float, default=10.0, help='membrane resistance')
    # --- inhibition ---
    p.add_argument('--g-inh', type=float, default=1.0,
                   help='inhibitory weight gain (1 = matched to excitation; ~4-6 = '
                        'inhibition-dominated, needed for the asynchronous state)')
    # --- STDP ---
    p.add_argument('--a-plus', type=float, default=0.002, help='STDP potentiation rate')
    p.add_argument('--a-minus', type=float, default=0.002, help='STDP depression rate')
    p.add_argument('--w-max-mult', type=float, default=4.0,
                   help='soft-bound ceiling for |W| as a multiple of the initial peak '
                        'weight (w_max = w_max_mult * w_scale). Must be > 1 to leave '
                        'headroom for potentiation; ~3-5 is sensible.')
    p.add_argument('--outdir', type=str, default='results')
    p.add_argument('--no-spikes', action='store_true',
                   help='skip spike recording (saves memory; also disables regime stats)')
    p.add_argument('--no-plots', action='store_true',
                   help='skip rendering figures, only save .npz')
    p.add_argument('--raster-window-ms', type=float, default=500.0,
                   help='plot only the last N ms of spikes in the raster so long '
                        'runs do not saturate into a solid block. <=0 = whole run.')
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    tag = f'N{args.N}_seed{args.seed}'
    # Mirror the in-run scaling so the log reports the weight actually used.
    if args.w_total is not None:
        K = max(1.0, args.connectivity * args.N)
        eff_w_scale = args.w_total / np.sqrt(K)
        w_desc = f'w_scale={eff_w_scale:.6f} (from w_total={args.w_total}, K={K:.0f})'
    else:
        w_desc = f'w_scale={args.w_scale}'
    print(f'[run] N={args.N} warmup={args.warmup} steps={args.steps} '
          f'dt={args.dt} seed={args.seed} {w_desc} '
          f'I_mean={args.I_mean} I_spread={args.I_spread} sigma={args.sigma} '
          f'g_inh={args.g_inh} conn={args.connectivity} R={args.R} '
          f'A+={args.a_plus} A-={args.a_minus} outdir={args.outdir}', flush=True)

    W_initial, W, spikes = run(
        N=args.N, warmup=args.warmup, steps=args.steps, dt=args.dt,
        seed=args.seed, exc_frac=args.exc_frac, connectivity=args.connectivity,
        save_spikes=not args.no_spikes,
        w_scale=args.w_scale, w_total=args.w_total, I_mean=args.I_mean,
        I_spread=args.I_spread,
        sigma=args.sigma, R=args.R,
        A_plus=args.a_plus, A_minus=args.a_minus, g_inh=args.g_inh,
        w_max_mult=args.w_max_mult,
    )

    npz_path = os.path.join(args.outdir, f'run_{tag}.npz')
    np.savez_compressed(npz_path, W_initial=W_initial, W_final=W, spikes=spikes,
                        N=args.N, seed=args.seed, dt=args.dt,
                        warmup=args.warmup, steps=args.steps)
    print(f'[save] {npz_path} ({spikes.shape[0]} spikes)', flush=True)

    # quick sanity summary so the slurm log shows whether the run actually fired
    if spikes.size:
        sim_s = (args.steps * args.dt) / 1000.0
        counts = np.bincount(spikes[:, 1].astype(int), minlength=args.N)
        rates = counts / sim_s
        pop_cv, isi_cv = regime_metrics(spikes, args.N, args.steps, args.dt)
        # Runaway guard: a few neurons pinned near the refractory ceiling (max
        # rate >> mean rate) inflate ISI_CV via burst-like interval clumping and
        # masquerade as "irregular". A genuine AI state has max rate only a small
        # multiple of the mean. Flag when that ratio is large so the verdict
        # can't be fooled by streak/runaway neurons.
        rate_ratio = rates.max() / rates.mean() if rates.mean() > 0 else float('inf')
        runaway = rate_ratio > 10.0
        ai_ish = (pop_cv < 0.5) and (isi_cv > 0.5) and not runaway
        print(f'[stats] mean_rate={rates.mean():.2f} Hz  max={rates.max():.2f} Hz  '
              f'silent={int((rates == 0).sum())}/{args.N}', flush=True)
        print(f'[stats] pop_CV={pop_cv:.3f} (low=async)  ISI_CV={isi_cv:.2f} (high=irregular)  '
              f'max/mean={rate_ratio:.1f}'
              f'{" RUNAWAY" if runaway else ""}  '
              f'-> {"AI-ish" if ai_ish else "not AI"}', flush=True)
    else:
        print('[stats] NO SPIKES recorded (network silent, or --no-spikes set)', flush=True)
    mask = W_initial != 0
    if mask.any():
        dw = (W - W_initial)[mask]
        print(f'[stats] synapses={mask.sum()}  potentiated={(dw > 0).mean()*100:.1f}%  '
              f'depressed={(dw < 0).mean()*100:.1f}%  '
              f'unchanged={(dw == 0).mean()*100:.1f}%  '
              f'mean_dW={dw.mean():+.5f}', flush=True)

    if args.no_plots:
        return

    if spikes.size:
        # Overplotting guard: a long run (e.g. 20000 steps) packs >100k spikes
        # across the full 2000 ms into the scatter, saturating it into a solid
        # block with no visible structure. Plot only a window (default the last
        # 500 ms = the settled state) with small markers so the asynchronous-
        # irregular pattern is actually legible. --raster-window-ms <= 0 restores
        # the old whole-run behaviour.
        T = args.steps * args.dt
        if args.raster_window_ms and args.raster_window_ms > 0:
            t_lo = max(0.0, T - args.raster_window_ms)
            sel = spikes[:, 0] >= t_lo
            rs = spikes[sel]
            win_desc = f'  [last {T - t_lo:.0f} ms]'
        else:
            rs = spikes
            win_desc = ''
        plt.figure(figsize=(12, 5))
        plt.scatter(rs[:, 0], rs[:, 1], s=1.0, color='teal',
                    linewidths=0, rasterized=True)
        plt.xlabel('time (ms)')
        plt.ylabel('neuron index')
        plt.title(f'spike raster (N={args.N}, seed={args.seed}){win_desc}')
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