'''Picture of how the ALIF reservoir remembers (train_memory_alif_hold.py).
Top: one slow neuron's adaptation variable b over a trial, for 5 cue values,
in a quiet network vs one with background drive. Separated lines = the cue is
still readable; merged lines = the memory has been overwritten.
Bottom: readout R^2 vs delay for the same two settings.
Run from the repo root: python diagnostics/viz_alif_memory.py'''
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import train_memory_alif_hold as H

N, N_TRAIN, N_TEST, SEED = 500, 400, 200, 0
CUES = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
SETTINGS = [('quiet delay (background 0)', 0.0, 0.0, '#1baf7a'),
            ('background drive 1.0', 1.0, 0.1, '#eb6834')]
CUE_CMAP = LinearSegmentedColormap.from_list('cue', ['#b7d3f6', '#104281'])


def trace_b(args, net, cues, idx, rng):
    # same update as H.simulate, but records b of neuron idx at every step
    # shortcut: ignores recurrence (off by default), add g if you turn rec_gain on
    inv_tau_m = args.dt / net['tau_m']
    rho_a = np.exp(-args.dt / net['tau_a'])
    V = np.zeros((len(cues), N)); b = np.zeros((len(cues), N))
    bg = args.background + args.background_spread * rng.standard_normal(N)
    cue_vec = cues[:, None] * net['W_in'][None, :]
    cue_steps = int(round(args.cue_dur / args.dt))
    steps = cue_steps + int(round(max(args.delay_list) / args.dt))
    out = np.empty((steps, len(cues)))
    for t in range(steps):
        I = bg[None, :] + (cue_vec if t < cue_steps else 0.0)
        V = V + inv_tau_m * (-V + args.R * I)
        s = (V >= 1.0 + args.beta_a * b).astype(np.float64)
        V = V * (1.0 - s)
        b = b * rho_a + s
        out[t] = b[:, idx]
    return out


def r2_curve(args, net, rng):
    # mirrors the scoring loop in H.main
    cue_tr = rng.uniform(-1, 1, N_TRAIN); cue_te = rng.uniform(-1, 1, N_TEST)
    ftr = H.simulate(args, net, cue_tr, rng); fte = H.simulate(args, net, cue_te, rng)
    r2 = []
    for d in args.delay_list:
        Xtr = ftr[d][:, :-1]; Xte = fte[d][:, :-1]
        mu = Xtr.mean(axis=0); sc = (Xtr - mu).std() + 1e-12
        Xtr_n = np.hstack([(Xtr - mu) / sc, np.ones((len(Xtr), 1))])
        Xte_n = np.hstack([(Xte - mu) / sc, np.ones((len(Xte), 1))])
        w = H.ridge_fit(Xtr_n, cue_tr, args.ridge)
        r2.append(1 - np.mean((Xte_n @ w - cue_te) ** 2) / (1 / 3))
    return np.array(r2)


fig = plt.figure(figsize=(12, 8), constrained_layout=True)
gs = fig.add_gridspec(2, 2)
ax_r2 = fig.add_subplot(gs[1, :])
ax_top = None

for col, (label, bg, spread, color) in enumerate(SETTINGS):
    args = H.parse_args(['--N', str(N), '--seed', str(SEED),
                         '--background', str(bg), '--background-spread', str(spread)])
    net = H.build_network(args, np.random.default_rng(SEED))
    # a slow neuron (tau_a > longest delay) that the cue drives strongly
    slow = np.where(net['tau_a'] > max(args.delay_list))[0]
    idx = slow[np.argmax(np.abs(net['W_in'][slow]))]

    tr = trace_b(args, net, CUES, idx, np.random.default_rng(SEED + 1))
    t_s = (np.arange(len(tr)) - args.cue_dur / args.dt) * args.dt / 1000
    ax = fig.add_subplot(gs[0, col], sharey=ax_top)
    ax_top = ax_top or ax
    ax.axvspan(t_s[0], 0, color='0.9', lw=0)
    ax.text(t_s[0], 1.0, ' cue on', transform=ax.get_xaxis_transform(),
            va='top', fontsize=9, color='0.35')
    for j, c in enumerate(CUES):
        ax.plot(t_s, tr[:, j], lw=2, color=CUE_CMAP(j / (len(CUES) - 1)))
    # direct labels at the line ends; cues that ended on (nearly) the same b share one label
    ends = np.round(tr[-1], 1)
    for e in np.unique(ends):
        ax.annotate('cue ' + ', '.join(f'{c:+.1f}' for c in CUES[ends == e]), (t_s[-1], e),
                    xytext=(6, 0), textcoords='offset points', va='center', fontsize=9)
    ax.set_title(f'{label}\nneuron {idx}: tau_a = {net["tau_a"][idx]/1000:.1f} s',
                 fontsize=11, loc='left')
    ax.set_xlabel('time since cue ended (s)')
    ax.set_ylabel('adaptation b (memory)')
    ax.set_xlim(t_s[0], t_s[-1] * 1.35)

    r2 = r2_curve(args, net, np.random.default_rng(SEED + 2))
    d_s = np.array(args.delay_list) / 1000
    ax_r2.plot(d_s, r2, lw=2, marker='o', ms=8, color=color, label=label,
               markeredgecolor='white', markeredgewidth=2)
    ax_r2.annotate(f'{r2[-1]:.2f}', (d_s[-1], r2[-1]), xytext=(8, 0),
                   textcoords='offset points', va='center', fontsize=9)
    print(f'[{label}] neuron {idx}  R2 per delay: ' + ' '.join(f'{x:.3f}' for x in r2))

ax_r2.axhline(0, color='0.6', lw=1, ls='--')
ax_r2.text(0.02, 0.06, 'R² = 0: no better than guessing', transform=ax_r2.transAxes,
           fontsize=9, color='0.4')
ax_r2.set_ylim(-0.05, 1.05)
ax_r2.set_xlabel('delay before recall (s)')
ax_r2.set_ylabel('recall R² (1 = perfect)')
ax_r2.set_title(f'How well a linear readout of all {N} neurons recovers the cue',
                fontsize=11, loc='left')
ax_r2.legend(frameon=False, loc='lower left', bbox_to_anchor=(0, 0.1))
for ax in fig.axes:
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', color='0.92')

out = os.path.join(os.path.dirname(__file__), 'alif_memory_explained.png')
fig.savefig(out, dpi=130)
print(f'[saved] {out}')
