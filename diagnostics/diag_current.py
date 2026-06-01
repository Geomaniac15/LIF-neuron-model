'''Diagnostic for the CURRENT-BASED synapse trainer. Same instrumentation as
diag_bptt but imports train_memory_bptt_current and adds the synaptic gains.

Tests:
  1. firing-rate sanity over a window
  2. does the per-step adjoint norm still track the clip ceiling? (it should NOT
     anymore if the fix works)
  3. does recall MSE settle clearly below chance (1/3)?
'''
import importlib.util, sys, time
import numpy as np

spec = importlib.util.spec_from_file_location(
    'tc', '/sessions/jolly-elegant-noether/mnt/LIF-neuron-model/train_memory_bptt_current.py')
T = importlib.util.module_from_spec(spec); spec.loader.exec_module(T)

_orig_norm = np.linalg.norm
clip_events = {'n_calls': 0, 'n_clipped': 0, 'norms': []}
def make_counting_norm(adj_clip):
    def _norm(a, *aa, **kw):
        n = _orig_norm(a, *aa, **kw)
        if np.ndim(n) == 0:
            clip_events['n_calls'] += 1; clip_events['norms'].append(float(n))
            if float(n) > adj_clip: clip_events['n_clipped'] += 1
        return n
    return _norm

class A: pass
def make_args(N, steps, seed=0, delay=2, ge=60.0, gi=15.0):
    a=A()
    a.N=N; a.seed=seed; a.steps=steps; a.target_delay=delay; a.hold=150; a.chunk=4
    a.dt=0.1; a.connectivity=0.05; a.w_scale=0.005; a.w_max_mult=4.0; a.g_inh=1.0
    a.tau_syn=5.0; a.tau_m_min=15.0; a.tau_m_max=25.0
    a.background_mean=1.5; a.background_spread=0.1; a.refractory=2.0
    a.adapt_b=4.0; a.tau_w=400.0
    a.n_readout=600 if N>=600 else N; a.tau_state=20.0; a.input_scale=8.0
    a.n_input=min(100,N//4); a.washout=20
    a.beta=1.0; a.lr_w=5e-5; a.lr_out=1e-2; a.clip=1.0; a.adj_clip=1e3
    a.rate_reg=0.5; a.target_rate=15.0; a.report_every=100; a.outdir='diag_out'
    a.gradcheck=False; a.syn_gain_exc=ge; a.syn_gain_inh=gi
    a.detach_reset=True; a.surr_scale=1.0
    return a

def run(N, steps, delay=2, seed=0, adj_clip=1e3, grad_clip=1.0, lr_w=5e-5,
        ge=60.0, gi=15.0, label=''):
    args=make_args(N,steps,seed,delay,ge,gi)
    args.adj_clip=adj_clip; args.clip=grad_clip; args.lr_w=lr_w
    clip_events['n_calls']=0; clip_events['n_clipped']=0; clip_events['norms']=[]
    W,conn_mask,w_ceiling,n_exc,params=T.build_fresh_network(args)
    dt=params['dt']; hold=args.hold
    rng=np.random.default_rng(args.seed+3)
    readout_idx=np.sort(rng.choice(N,size=min(args.n_readout,N),replace=False))
    W_out=np.zeros(len(readout_idx)+1)
    n_rsteps=args.steps; u=rng.uniform(-1,1,n_rsteps+args.target_delay)
    exc_rows=np.arange(N)<n_exc
    opt_W=T.Adam(W.shape,args.lr_w); opt_o=T.Adam(W_out.shape,args.lr_out)
    chunk=args.chunk; warm=args.washout
    carry=dict(V=np.full(N,-65.0),ge=np.zeros(N),gi=np.zeros(N),
               wad=np.zeros(N),tr=np.zeros(N),refr=np.zeros(N))
    rstep=0; recent=[]; curve=[]; gnorms=[]; gclip=0; rates=[]; t0=time.time()
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
        rates.append(tape['s'].mean()/dt*1000.0)
        if targets:
            np.linalg.norm=make_counting_norm(args.adj_clip)
            gW=T.backward_window(W,tape,rg,g_rate,params,args,False)
            np.linalg.norm=_orig_norm
            nt=len(targets); gW/=nt; gWout=gWout/nt
            if np.all(np.isfinite(gW)) and np.all(np.isfinite(gWout)):
                gW*=conn_mask; gn=_orig_norm(gW); gnorms.append(gn)
                if gn>args.clip: gW*=args.clip/gn; gclip+=1
                W-=opt_W.step(gW); W_out-=opt_o.step(gWout)
                W*=conn_mask
                W[exc_rows]=np.clip(W[exc_rows],0.0,None)
                W[~exc_rows]=np.clip(W[~exc_rows],None,0.0)
                np.clip(W,-w_ceiling,w_ceiling,out=W)
                recent.append(loss/max(1,len(targets))); curve.append(loss/max(1,len(targets)))
        carry={k:v.copy() for k,v in co.items()}; rstep+=this
    np.linalg.norm=_orig_norm
    c=np.array(curve); k=20
    roll=np.array([c[max(0,i-k):i+1].mean() for i in range(len(c))]) if len(c) else np.array([np.nan])
    norms=np.array(clip_events['norms'])
    print(f'=== {label} | ge={ge} gi={gi} adj_clip={adj_clip:.0e} lr_w={lr_w:.0e} ===')
    print(f'  mean rate={np.mean(rates):5.1f}Hz   time={time.time()-t0:.0f}s   windows={len(curve)}')
    print(f'  recall MSE: first20={np.mean(c[:20]):.4f}  best_roll={roll.min():.4f}  final_roll={roll[-1]:.4f}   (chance=0.3333)')
    print(f'  adjoint clip fired {100*clip_events["n_clipped"]/max(1,clip_events["n_calls"]):.1f}%   adjoint norm median={np.median(norms):.1f} (clip ceiling={adj_clip:.0e})')
    return dict(roll=roll, rate=np.mean(rates), adjmed=np.median(norms), curve=curve)
