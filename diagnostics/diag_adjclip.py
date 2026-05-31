import numpy as np, diag_bptt as D
N=300; steps=900
for ac in [1e3, 1e5, 1e7]:
    r=D.run(N,steps,2,adj_clip=ac,label=f'adj_clip={ac:.0e}')
    c=np.array(r['curve']); k=20
    roll=np.array([c[max(0,i-k):i+1].mean() for i in range(len(c))])
    print(f'  --> adj_clip={ac:.0e}: final_rolling={roll[-1]:.4f} best_rolling={roll.min():.4f} (chance=0.333)\n')
