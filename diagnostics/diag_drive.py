'''Run several BPTT configs, capture loss-curve shape, and test whether the
instability (learns-then-diverges) is caused by the exploding through-time
adjoint. Saves a plot.'''
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import diag_bptt as D

def summarize(curve, label):
    c=np.array(curve)
    # rolling-min trajectory to see how low it gets and whether it stays
    k=20
    roll=np.array([c[max(0,i-k):i+1].mean() for i in range(len(c))])
    best_i=int(np.argmin(roll))
    print(f'  [{label}] best rolling-MSE={roll.min():.4f} @ window {best_i}/{len(c)}  | final rolling={roll[-1]:.4f}  | chance=0.3333')
    return roll

configs = [
    dict(label='A: baseline lr5e-5 gclip1.0', kw={}),
    dict(label='B: lower lr_w=1e-5',          kw=dict(lr_w=1e-5)),
    dict(label='E: looser grad_clip=10',      kw=dict(grad_clip=10.0)),
]

plt.figure(figsize=(11,6))
N=300; steps=900; delay=2
results={}
for cfg in configs:
    lab=cfg['label']
    chunk=3 if lab.startswith('C') else 4
    # monkeypatch chunk by temporarily setting in make_args via attribute
    orig_make=D.make_args
    def make_args2(N,steps,seed=0,delay=2,_chunk=chunk):
        a=orig_make(N,steps,seed,delay); a.chunk=_chunk; return a
    D.make_args=make_args2
    r=D.run(N,steps,delay,label=lab,**cfg['kw'])
    D.make_args=orig_make
    roll=summarize(r['curve'], lab)
    results[lab]=roll
    plt.plot(roll, label=lab, lw=1.5)

plt.axhline(1/3, color='k', ls='--', lw=1, label='chance (0.333)')
plt.axhline(0.05, color='gray', ls=':', lw=1)
plt.xlabel('training window'); plt.ylabel('recall MSE (rolling mean, k=20)')
plt.title('BPTT delay-2 memory: loss trajectory across configs (N=400)')
plt.legend(fontsize=8); plt.ylim(0,1.0); plt.grid(alpha=0.3)
plt.tight_layout(); plt.savefig('bptt_diagnosis.png', dpi=110)
print('\nsaved bptt_diagnosis.png')
