'''Aggregate per-seed memory-sweep CSVs from sweep_memory_adapt.slurm.

Reads sweep_csv/mem_adapt_seed*.csv, groups by (adapt_b, tau_w), and reports the
recall horizon (mean +/- sd across seeds), total memory capacity, MC_1 and how
many seeds went chaotic per cell. Prints the best non-chaotic setting.

Usage:
    python aggregate_memory_sweep.py                       # default glob
    python aggregate_memory_sweep.py 'sweep_csv/*.csv'     # custom glob
'''
import csv
import glob
import sys
import collections
import statistics as st

FLOAT_COLS = ('adapt_b', 'tau_w', 'total_mc', 'horizon_ms', 'mc1', 'sep', 'ratio')


def main():
    pattern = sys.argv[1] if len(sys.argv) > 1 else 'sweep_csv/mem_adapt_seed*.csv'
    files = sorted(glob.glob(pattern))
    if not files:
        sys.exit(f'no CSVs matched: {pattern}')

    rows = []
    for f in files:
        for r in csv.DictReader(open(f)):
            for k in FLOAT_COLS:
                r[k] = float(r[k])
            rows.append(r)
    print(f'{len(files)} seed files, {len(rows)} rows\n')

    groups = collections.defaultdict(list)
    for r in rows:
        groups[(r['adapt_b'], r['tau_w'])].append(r)

    header = (f'{"adapt_b":>7} {"tau_w":>6} {"horizon_ms (mean+/-sd)":>22} '
              f'{"totMC":>7} {"MC_1":>7} {"chaotic":>8} {"n":>3}')
    print(header)
    print('-' * len(header))

    agg = []
    for (b, tw), rs in sorted(groups.items()):
        h = [r['horizon_ms'] for r in rs]
        hm, hsd = st.mean(h), (st.pstdev(h) if len(h) > 1 else 0.0)
        mc = st.mean(r['total_mc'] for r in rs)
        m1 = st.mean(r['mc1'] for r in rs)
        nchaos = sum('CHAOTIC' in r['regime'] for r in rs)
        agg.append((b, tw, hm, hsd, mc, m1, nchaos, len(rs)))
        print(f'{b:>7.0f} {tw:>6.0f} {hm:>12.1f} +/- {hsd:>5.1f} '
              f'{mc:>7.2f} {m1:>7.3f} {nchaos:>8} {len(rs):>3}')

    ok = [a for a in agg if a[6] == 0]
    if ok:
        # longest mean horizon, tie-break on total MC, then lowest variance
        best = max(ok, key=lambda a: (a[2], a[4], -a[3]))
        print(f'\nBest (no chaotic seeds): adapt_b={best[0]:.0f} tau_w={best[1]:.0f}ms '
              f'-> horizon {best[2]:.1f}+/-{best[3]:.1f}ms, totalMC {best[4]:.2f}, '
              f'MC_1 {best[5]:.3f}')
        most_robust = min(ok, key=lambda a: a[3])
        print(f'Most consistent across seeds: adapt_b={most_robust[0]:.0f} '
              f'tau_w={most_robust[1]:.0f}ms (sd {most_robust[3]:.1f}ms, '
              f'horizon {most_robust[2]:.1f}ms, totalMC {most_robust[4]:.2f})')


if __name__ == '__main__':
    main()
