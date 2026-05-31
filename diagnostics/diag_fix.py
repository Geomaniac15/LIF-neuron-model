'''Confirmatory test: the through-time adjoint explodes in proportion to the
number of sim-steps in a TBPTT window (hold * chunk). If that explosion is what
corrupts the update, shrinking the window (smaller hold) should let delay-2
memory train STABLY below chance. We keep delay=2 and vary hold.'''
import numpy as np
import diag_bptt as D

def run_hold(N, steps, hold, delay=2, label=''):
    orig=D.make_args
    def mk(N,steps,seed=0,delay=2,_h=hold):
        a=orig(N,steps,seed,delay); a.hold=_h; return a
    D.make_args=mk
    r=D.run(N,steps,delay,label=label)
    D.make_args=orig
    c=np.array(r['curve']); k=20
    roll=np.array([c[max(0,i-k):i+1].mean() for i in range(len(c))])
    print(f'  --> [{label}] final rolling-MSE={roll[-1]:.4f}  best rolling={roll.min():.4f}  (chance=0.3333)\n')
    return roll[-1], roll.min()

if __name__=='__main__':
    N=300; steps=900
    print('hold=150 is the value used in all of George\'s v3-v7 runs (15ms/rstep).')
    run_hold(N, steps, hold=150, delay=2, label='hold=150 (window=600 sim-steps)')
    run_hold(N, steps, hold=50,  delay=2, label='hold=50  (window=200 sim-steps)')
    run_hold(N, steps, hold=20,  delay=2, label='hold=20  (window=80 sim-steps)')
