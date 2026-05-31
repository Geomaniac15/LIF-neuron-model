'''Aggregate per-seed memory-sweep CSVs from the reservoir_capacity.py slurms.

Reads the per-seed CSVs, auto-detects which sweep parameters actually vary across
cells (hold, adapt_b, tau_w, input_scale, n_input, inh_scale, tau_syn), groups by
that varying combo, and reports the recall horizon (mean +/- sd across seeds),
total memory capacity, MC_1 and how many seeds went chaotic per cell. Prints the
best non-chaotic setting.

Usage:
    python aggregate_memory_sweep.py                            # default glob
    python aggregate_memory_sweep.py 'sweep_csv/mem_hold_*.csv' # custom glob
'''
import csv
import glob
import sys
import collections
import statistics as st

# candidate sweep parameters, in display order
PARAM_COLS = ('hold', 'tau_syn', 'adapt_b', 'tau_w', 'n_input', 'input_scale',
              'inh_scale')
METRIC_COLS = ('total_mc', 'horizon_ms', 'mc1', 'sep', 'ratio')


def main():
    pattern = sys.argv[1] if len(sys.argv) > 1 else 'sweep_csv/mem_adapt_seed*.csv'
    files = sorted(glob.glob(pattern))
    if not files:
        sys.exit(f'no CSVs matched: {pattern}')

    rows = []
    for f in files:
        for r in csv.DictReader(open(f)):
            for k in METRIC_COLS:
                r[k] = float(r[k])
            for k in PARAM_COLS:
                if k in r:
                    r[k] = float(r[k])
            rows.append(r)
    print(f'{len(files)} seed files, {len(rows)} rows')

    present = [k for k in PARAM_COLS if k in rows[0]]
    varying = [k for k in present if len({r[k] for r in rows}) > 1] or present[:1]
    print(f'grouping by: {", ".join(varying)}\n')

    groups = collections.defaultdict(list)
    for r in rows:
        groups[tuple(r[k] for k in varying)].append(r)

    keyhdr = ' '.join(f'{k:>8}' for k in varying)
    header = (f'{keyhdr} {"horizon_ms (mean+/-sd)":>22} {"totMC":>7} {"MC_1":>7} '
              f'{"chaotic":>8} {"n":>3}')
    print(header)
    print('-' * len(header))

    agg = []
    for combo, rs in sorted(groups.items()):
        h = [r['horizon_ms'] for r in rs]
        hm, hsd = st.mean(h), (st.pstdev(h) if len(h) > 1 else 0.0)
        mc = st.mean(r['total_mc'] for r in rs)
        m1 = st.mean(r['mc1'] for r in rs)
        nchaos = sum('CHAOTIC' in r['regime'] for r in rs)
        agg.append(dict(combo=combo, hm=hm, hsd=hsd, mc=mc, m1=m1,
                        nchaos=nchaos, n=len(rs)))
        keystr = ' '.join(f'{v:>8g}' for v in combo)
        print(f'{keystr} {hm:>12.1f} +/- {hsd:>5.1f} {mc:>7.2f} {m1:>7.3f} '
              f'{nchaos:>8} {len(rs):>3}')

    ok = [a for a in agg if a['nchaos'] == 0]
    if ok:
        def label(a):
            return ', '.join(f'{k}={v:g}' for k, v in zip(varying, a['combo']))
        # longest mean horizon, tie-break on total MC, then lowest variance
        best = max(ok, key=lambda a: (a['hm'], a['mc'], -a['hsd']))
        print(f'\nBest (no chaotic seeds): {label(best)} -> horizon '
              f'{best["hm"]:.1f}+/-{best["hsd"]:.1f}ms, totalMC {best["mc"]:.2f}, '
              f'MC_1 {best["m1"]:.3f}')
        robust = min(ok, key=lambda a: a['hsd'])
        print(f'Most consistent across seeds: {label(robust)} (sd {robust["hsd"]:.1f}ms, '
              f'horizon {robust["hm"]:.1f}ms, totalMC {robust["mc"]:.2f})')


if __name__ == '__main__':
    main()
