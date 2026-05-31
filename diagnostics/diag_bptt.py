'''Diagnostic harness around train_memory_bptt.

Runs a SHORT BPTT training while logging:
  - full per-window recall MSE curve (vs chance = 1/3)
  - how often the per-step adjoint clip fires, and the adjoint norms
  - how often the global gradient-norm clip fires
  - gradient magnitude that actually reaches W
  - firing rate sanity

It monkeypatches backward_window to record adjoint-clip statistics without
changing the maths.
'''
import sys, types, time
import numpy as np

sys.path.insert(0, '/sessions/jolly-elegant-noether/mnt/LIF-neuron-model')
import train_memory_bptt as T

# ---- instrument the per-step adjoint clip --------------------------------
clip_events = {'n_calls': 0, 'n_clipped': 0, 'norms': []}
_orig_norm = np.linalg.norm

def make_counting_norm(adj_clip):
    '''Returns a drop-in for np.linalg.norm that records each scalar norm and
    whether it would trip the adjoint clip, then returns the true norm.'''
    def _norm(a, *args, **kw):
        n = _orig_norm(a, *args, **kw)
        if np.ndim(n) == 0:
            clip_events['n_calls'] += 1
            clip_events['norms'].append(float(n))
            if float(n) > adj_clip:
                clip_events['n_clipped'] += 1
        return n
    return _norm

# We wrap backward_window so it uses our instrumented _clip but otherwise runs
# the identical code. Easiest: re-implement by copying logic is risky; instead
# patch via a closure that swaps the local _clip. Simplest robust approach:
# record clip stats by intercepting np.linalg.norm inside backward only.
orig_backward = T.backward_window
def backward_instrumented(W, tape, readout_grads, g_rate, params, args, smooth):
    gW = orig_backward(W, tape, readout_grads, g_rate, params, args, smooth)
    return gW

# Track global grad clip + gW magnitude by wrapping the optimizer step path:
# we instead patch train loop indirectly. Simpler: run train with a custom
# args object and add hooks by editing module-level counters in forward.

# ---- minimal args -------------------------------------------------------
class A: pass

def make_args(N, steps, seed=0, delay=2):
    a = A()
    a.N=N; a.seed=seed; a.steps=steps; a.target_delay=delay; a.hold=150; a.chunk=4
    a.dt=0.1; a.connectivity=0.05; a.w_scale=0.005; a.w_max_mult=4.0; a.g_inh=1.0
    a.tau_syn=5.0; a.tau_m_min=15.0; a.tau_m_max=25.0
    a.background_mean=1.5; a.background_spread=0.1; a.refractory=2.0
    a.adapt_b=4.0; a.tau_w=400.0
    a.n_readout=600 if N>=600 else N; a.tau_state=20.0; a.input_scale=8.0
    a.n_input=min(100,N//4); a.washout=20
    a.beta=1.0; a.lr_w=5e-5; a.lr_out=1e-2; a.clip=1.0; a.adj_clip=1e3
    a.rate_reg=0.5; a.target_rate=15.0; a.report_every=100; a.outdir='diag_out'
    a.gradcheck=False
    return a

# ---- custom train loop with rich logging (mirrors T.train) --------------
def run(N, steps, delay=2, seed=0, adj_clip=1e3, grad_clip=1.0, lr_w=5e-5,
        label=''):
    args = make_args(N, steps, seed, delay)
    args.adj_clip=adj_clip; args.clip=grad_clip; args.lr_w=lr_w
    clip_events['n_calls']=0; clip_events['n_clipped']=0; clip_events['norms']=[]

    W, conn_mask, w_ceiling, n_exc, params = T.build_fresh_network(args)
    dt=params['dt']; hold=args.hold
    rng = np.random.default_rng(args.seed+3)
    readout_idx = np.sort(rng.choice(N, size=min(args.n_readout,N), replace=False))
    W_out = np.zeros(len(readout_idx)+1)
    n_rsteps=args.steps
    u = rng.uniform(-1,1,n_rsteps+args.target_delay)
    exc_rows = np.arange(N) < n_exc
    opt_W=T.Adam(W.shape,args.lr_w); opt_o=T.Adam(W_out.shape,args.lr_out)
    chunk=args.chunk; warm=args.washout
    carry=dict(V=np.full(N,-65.0),ge=np.zeros(N),gi=np.zeros(N),
               wad=np.zeros(N),tr=np.zeros(N),refr=np.zeros(N))
    # patch norm only during backward to count clips
    rstep=0; recent=[]; curve=[]; gnorms=[]; gclip_fires=0; gW_after=[]
    t0=time.time()
    while rstep<n_rsteps:
        this=min(chunk,n_rsteps-rstep); win=this*hold
        drive=np.zeros((win,N)); targets=[]
        for c in range(this):
            r=rstep+c; val=u[r]*args.input_scale
            drive[c*hold:(c+1)*hold,:args.n_input]=val
            ti=r-args.target_delay
            if rstep+c>=warm and ti>=0: targets.append((c*hold+hold-1,u[ti]))
        co,loss,tape,rg,gWout,g_rate=T.forward_window(W,W_out,carry,drive,targets,
                                                      readout_idx,params,args,False)
        if targets:
            np.linalg.norm = make_counting_norm(args.adj_clip)  # instrument
            gW=T.backward_window(W,tape,rg,g_rate,params,args,False)
            np.linalg.norm = _orig_norm                # restore
            nt=len(targets); gW/=nt; gWout=gWout/nt
            if np.all(np.isfinite(gW)) and np.all(np.isfinite(gWout)):
                gW*=conn_mask
                gn=_orig_norm(gW); gnorms.append(gn)
                if gn>args.clip: gW*=args.clip/gn; gclip_fires+=1
                gW_after.append(_orig_norm(gW))
                W-=opt_W.step(gW); W_out-=opt_o.step(gWout)
                W*=conn_mask
                W[exc_rows]=np.clip(W[exc_rows],0.0,None)
                W[~exc_rows]=np.clip(W[~exc_rows],None,0.0)
                np.clip(W,-w_ceiling,w_ceiling,out=W)
                recent.append(loss/max(1,len(targets)))
                curve.append(loss/max(1,len(targets)))
        carry={k:v.copy() for k,v in co.items()}
        rstep+=this
    np.linalg.norm=_orig_norm
    mse=float(np.mean(recent[-50:])) if recent else float('nan')
    first=float(np.mean(curve[:20])) if len(curve)>=20 else float('nan')
    best=float(min(curve)) if curve else float('nan')
    norms=np.array(clip_events['norms'])
    print(f'\n=== {label}  N={N} delay={delay} steps={steps} adj_clip={adj_clip} grad_clip={grad_clip} lr_w={lr_w} ===')
    print(f'  windows trained={len(curve)}  time={time.time()-t0:.0f}s')
    print(f'  recall MSE: first20={first:.4f}  best={best:.4f}  final50={mse:.4f}   (chance=0.3333)')
    print(f'  global grad-clip fired {gclip_fires}/{len(gnorms)} windows ({100*gclip_fires/max(1,len(gnorms)):.0f}%)  raw gW norm: median={np.median(gnorms):.3f} max={np.max(gnorms):.3f}')
    print(f'  per-step ADJOINT clip fired {clip_events["n_clipped"]}/{clip_events["n_calls"]} ({100*clip_events["n_clipped"]/max(1,clip_events["n_calls"]):.1f}%)  adjoint norm: median={np.median(norms):.2f} p99={np.percentile(norms,99):.1f} max={norms.max():.1f}')
    return dict(curve=curve, mse=mse, best=best, gnorms=gnorms)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument('--N',type=int,default=400)
    p.add_argument('--steps',type=int,default=1500)
    p.add_argument('--delay',type=int,default=2)
    p.add_argument('--adj-clip',type=float,default=1e3)
    p.add_argument('--grad-clip',type=float,default=1.0)
    p.add_argument('--lr-w',type=float,default=5e-5)
    p.add_argument('--label',default='run')
    a=p.parse_args()
    run(a.N,a.steps,a.delay,adj_clip=a.adj_clip,grad_clip=a.grad_clip,lr_w=a.lr_w,label=a.label)
