"""
scan_transmission.py -- analysis of a whole Tully momentum scan: for every scan point (nucP_*/) of one or
more method dirs, run analyze_transmission.py in-process, write the point's analyze_transmission.png into the
point dir exactly as that script does, and collect the FINAL values (at the point's last time) of the
four channel populations C_Rg, C_Tg, C_Re, C_Te with their block standard errors.

Per method dir it prints a table and copy-pasteable arrays, and saves both to <method_dir>/scan_transmission.txt:

    nucP_init = [5, 6, ...]
    CRg = [...]    CRg_err = [...]
    CTg = [...]    CTg_err = [...]
    CRe = [...]    CRe_err = [...]
    CTe = [...]    CTe_err = [...]
    MRg_maxabs = [...]    MRg_maxabs_err = [...]      (max over t of |M_Rg(t)|, SE at that t)
    ... likewise MTg, MRe, MTe

The per-unit partials are cached as usual (<shards>_partials/), so a re-run only does new shards.

Usage (PYTHONPATH=<RP_MASH root>):
    python scan_transmission.py Momentum-Scan/*/ --ncores 10
    cd Momentum-Scan/MASH-Classical && python ../../scan_transmission.py --ncores 10
"""

import os
import re
import sys
import glob
import argparse
import traceback

os.environ.setdefault('MPLBACKEND', 'Agg')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import analyze_transmission


_POINT_RE = re.compile(r'^nucP_(.+)$')
_KEYS = ('CRg', 'CTg', 'CRe', 'CTe')
_MKEYS = ('MRg', 'MTg', 'MRe', 'MTe')


def scan_points(method_dir):
    """[(nucP value, point dir)] sorted by value."""
    pts = []
    for d in glob.glob(os.path.join(method_dir, 'nucP_*/')):
        m = _POINT_RE.match(os.path.basename(os.path.normpath(d)))
        try:
            pts.append((float(m.group(1)), os.path.normpath(d)))
        except (AttributeError, ValueError):
            continue
    return sorted(pts)


def analyze_point(point_dir, a):
    path = os.path.join(point_dir, 'shards')
    if not os.path.isdir(path):
        path = os.path.join(point_dir, 'data.hdf')
        if not os.path.isfile(path):
            raise RuntimeError(f'no shards/ or data.hdf in {point_dir}')
    r = analyze_transmission.compute(path, nblocks=a.nblocks, ncores=a.ncores, refresh=a.refresh)
    analyze_transmission._plot(os.path.join(point_dir, 'analyze_transmission.png'), r, tmax=a.tmax)
    return r


def _fmt_list(xs, fmt):
    return '[' + ', '.join('nan' if not np.isfinite(x) else format(x, fmt) for x in xs) + ']'


def run_method(method_dir, a):
    method_dir = os.path.normpath(method_dir)
    label = os.path.basename(os.path.abspath(method_dir))
    pts = scan_points(method_dir)
    if not pts:
        print(f'[scan] {method_dir}: no nucP_*/ scan points, skipping')
        return
    print(f'\n[scan] ===== {label}: {len(pts)} point(s) =====', flush=True)

    rows = []          # (nucP, n_traj, T, {key: (value, se)})
    for v, d in pts:
        print(f'[scan] --- {os.path.basename(d)}', flush=True)
        try:
            r = analyze_point(d, a)
            fin = {k: (float(r[k][-1]), float(r[k + '_SE'][-1])) for k in _KEYS}
            for k in _MKEYS:                       # max over time of |M_c(t)|, SE at that time
                i = int(np.nanargmax(np.abs(r[k])))
                fin[k] = (float(abs(r[k][i])), float(r[k + '_SE'][i]))
            rows.append((v, int(r['n_traj']), float(r['time'][-1]) if 'time' in r else np.nan, fin))
            print('[scan]     ' + '  '.join(f'{k}={fin[k][0]:.4f}+/-{fin[k][1]:.4f}' for k in _KEYS)
                  + f'  (n_traj={r["n_traj"]})', flush=True)
            print('[scan]     ' + '  '.join(f'max|{k}|={fin[k][0]:.3e}+/-{fin[k][1]:.1e}' for k in _MKEYS),
                  flush=True)
        except Exception as e:
            print(f'[scan]     FAILED: {e}', flush=True)
            if a.verbose:
                traceback.print_exc()
            rows.append((v, 0, np.nan, {k: (np.nan, np.nan) for k in _KEYS + _MKEYS}))

    head = f'# {"nucP_init":>9s} {"n_traj":>9s} {"T":>8s}' + ''.join(
        f' {k:>11s} {k + "_err":>11s}' for k in _KEYS) + ''.join(
        f' {"max|" + k + "|":>11s} {"err":>9s}' for k in _MKEYS)
    lines = [f'# {label}  (final values C(T); max over t of |M(t)|, err = SE at that t; '
             f'nblocks={a.nblocks})', head]
    for v, n, T, fin in rows:
        lines.append(f'  {v:9g} {n:9d} {T:8g}' + ''.join(
            f' {fin[k][0]:11.6f} {fin[k][1]:11.6f}' for k in _KEYS) + ''.join(
            f' {fin[k][0]:11.3e} {fin[k][1]:9.1e}' for k in _MKEYS))
    lines += ['', 'nucP_init = ' + _fmt_list([r[0] for r in rows], 'g')]
    for k in _KEYS:
        lines.append(f'{k} = ' + _fmt_list([r[3][k][0] for r in rows], '.6f'))
        lines.append(f'{k}_err = ' + _fmt_list([r[3][k][1] for r in rows], '.6f'))
    for k in _MKEYS:
        lines.append(f'{k}_maxabs = ' + _fmt_list([r[3][k][0] for r in rows], '.6e'))
        lines.append(f'{k}_maxabs_err = ' + _fmt_list([r[3][k][1] for r in rows], '.6e'))
    text = '\n'.join(lines) + '\n'
    print('\n' + text, flush=True)
    out = os.path.join(method_dir, a.out)
    with open(out, 'w') as f:
        f.write(text)
    print(f'[scan] wrote {out}')


def _main():
    ap = argparse.ArgumentParser(description='Run analyze_transmission over every nucP_* point of the given '
                                             'method dirs and collect the final channel populations.')
    ap.add_argument('dirs', nargs='*', default=['.'], help='method dirs (default: .)')
    ap.add_argument('--ncores',  type=int, default=None, help='worker processes (default: SLURM CPUs, else 1)')
    ap.add_argument('--nblocks', type=int, default=100, help='trajectory blocks (default 100)')
    ap.add_argument('--tmax',    type=float, default=None, help='truncate the PLOTS to t <= tmax')
    ap.add_argument('--refresh', action='store_true', help='recompute every cached partial')
    ap.add_argument('--out',     default='scan_transmission.txt', help='summary file name in each method dir')
    ap.add_argument('-v', '--verbose', action='store_true', help='print tracebacks of failed points')
    a = ap.parse_args()
    for d in a.dirs:
        if os.path.basename(os.path.normpath(d)).startswith('__'):     # __pycache__ from a glob
            continue
        run_method(d, a)


if __name__ == '__main__':
    _main()
