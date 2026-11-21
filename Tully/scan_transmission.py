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

--- running the scan in batches (shards deleted between batches) ---------------------------------

Every analysed point also saves its result to <point dir>/transmission.npz: the full C_c(t), M_c(t)
time series with their SEs, n_traj, nbds, nblocks (~1 MB). A point whose shards/ (or data.hdf) is
gone is read back from that file instead, so scan_transmission.txt always covers EVERY point analysed
so far, whichever batch it came from -- there is no separate accumulate step. Per batch:

    1. config.py: leave only this batch's momenta in nucP_init_scan (comment out the rest)
    2. run it (workflow.runner / workflow.slurm + sbatch) and wait until it has finished
    3. python ../../../scan_transmission.py --ncores 10 --delete-shards
       -> analyses the batch, writes transmission.npz + analyze_transmission.png in each point dir,
          rewrites scan_transmission.txt over all points, then deletes shards/ and shards_partials/
          of every point that was just analysed from COMPLETE shards (n_traj reached the config's
          n_traj). An incomplete point keeps its shards (with a warning) and is redone next time.
    4. next batch

Without --delete-shards nothing is deleted (delete by hand once transmission.npz exists). A point
that still has shards is always re-analysed from them and its transmission.npz rewritten;
--saved-only skips that and just rebuilds the summary from the saved files.

Points can be added at any time (e.g. lower / higher momenta): the summary is built from every nucP_*/
present, shards or transmission.npz, sorted by momentum. A point read from transmission.npz gets its
analyze_transmission.png redrawn only if the png is missing, or for all of them with --replot (e.g.
--replot --tmax 3000). NOTE: keep finished momenta out of nucP_init_scan -- the runner sees no shards
in their dirs and would run them again from scratch.

Usage (PYTHONPATH=<RP_MASH root>):
    python scan_transmission.py Momentum-Scan/*/ --ncores 10
    cd Momentum-Scan/MASH-Classical && python ../../scan_transmission.py --ncores 10
    python ../../../scan_transmission.py --ncores 10 --delete-shards     # batch mode, see above
"""

import os
import re
import sys
import glob
import json
import shutil
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


SAVED = 'transmission.npz'                 # per-point result, survives deleting the shards
_SERIES = analyze_transmission._NAMES
_SCALARS = ('n_traj', 'nbds', 'nblocks', 'n_jumps', 'jump_label')


def _save_point(path, r, n_traj_cfg):
    tmp = path + '.tmp.npz'                  # written whole, then renamed: never a half file
    np.savez(tmp, time=r['time'], n_traj_cfg=np.int64(n_traj_cfg),
             **{k: r[k] for k in _SERIES}, **{k + '_SE': r[k + '_SE'] for k in _SERIES},
             **{k: np.asarray(r[k]) for k in _SCALARS})
    os.replace(tmp, path)


def _load_point(path):
    with np.load(path) as z:
        r = {k: z[k] for k in z.files}
    for k in _SCALARS + ('n_traj_cfg',):
        r[k] = r[k].item()
    return r


def _source(point_dir):
    """shards/ or data.hdf of a point, or None if the raw data is gone."""
    for name, ok in (('shards', os.path.isdir), ('data.hdf', os.path.isfile)):
        p = os.path.join(point_dir, name)
        if ok(p):
            return p
    return None


def _config_n_traj(path):
    """n_traj the run was configured for (config_json of the first shard / data.hdf)."""
    f0 = path if os.path.isfile(path) else sorted(glob.glob(os.path.join(path, '*.hdf')))[0]
    with analyze_transmission.open_readonly(f0) as f:
        return int(json.loads(f.attrs['config_json'])['n_traj'])


def analyze_point(point_dir, a):
    """(result dict, 'shards' | 'saved'): from the raw data if it is still there (and the result is
    saved to transmission.npz), else from transmission.npz."""
    path, saved = _source(point_dir), os.path.join(point_dir, SAVED)
    if path is not None and not a.saved_only:
        r = analyze_transmission.compute(path, nblocks=a.nblocks, ncores=a.ncores, refresh=a.refresh)
        analyze_transmission._plot(os.path.join(point_dir, 'analyze_transmission.png'), r, tmax=a.tmax)
        r['n_traj_cfg'] = _config_n_traj(path)
        _save_point(saved, r, r['n_traj_cfg'])
        return r, 'shards'
    if os.path.isfile(saved):
        r = _load_point(saved)
        png = os.path.join(point_dir, 'analyze_transmission.png')
        if a.replot or not os.path.isfile(png):           # the saved series are all the plot needs
            analyze_transmission._plot(png, r, tmax=a.tmax)
        return r, 'saved'
    raise RuntimeError(f'no shards/, data.hdf or {SAVED} in {point_dir}'
                       + (' (--saved-only)' if path is not None else ''))


def delete_shards(point_dir, r):
    """After a successful analysis from COMPLETE shards: remove shards/ and shards_partials/."""
    if r['n_traj'] < r['n_traj_cfg']:
        print(f"[scan]     NOT deleting shards: only {r['n_traj']} of {r['n_traj_cfg']} trajectories "
              f"(still running, or failed trajectories?)", flush=True)
        return
    if not os.path.isfile(os.path.join(point_dir, SAVED)):    # belt and braces
        raise RuntimeError(f'{SAVED} missing in {point_dir}: refusing to delete the shards')
    gone = []
    for name in ('shards', 'shards_partials'):
        p = os.path.join(point_dir, name)
        if os.path.isdir(p):
            shutil.rmtree(p)
            gone.append(name + '/')
    if gone:
        print(f"[scan]     deleted {' and '.join(gone)} (result kept in {SAVED})", flush=True)


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
            r, src = analyze_point(d, a)
            if src == 'saved':
                print(f'[scan]     from {SAVED} (no shards)', flush=True)
            fin = {k: (float(r[k][-1]), float(r[k + '_SE'][-1])) for k in _KEYS}
            for k in _MKEYS:                       # max over time of |M_c(t)|, SE at that time
                i = int(np.nanargmax(np.abs(r[k])))
                fin[k] = (float(abs(r[k][i])), float(r[k + '_SE'][i]))
            rows.append((v, int(r['n_traj']), float(r['time'][-1]) if 'time' in r else np.nan, fin))
            print('[scan]     ' + '  '.join(f'{k}={fin[k][0]:.4f}+/-{fin[k][1]:.4f}' for k in _KEYS)
                  + f'  (n_traj={r["n_traj"]})', flush=True)
            print('[scan]     ' + '  '.join(f'max|{k}|={fin[k][0]:.3e}+/-{fin[k][1]:.1e}' for k in _MKEYS),
                  flush=True)
            if a.delete_shards and src == 'shards':
                delete_shards(d, r)
        except Exception as e:
            print(f'[scan]     FAILED: {e}', flush=True)
            if a.verbose:
                traceback.print_exc()
            rows.append((v, 0, np.nan, {k: (np.nan, np.nan) for k in _KEYS + _MKEYS}))

    head = f'# {"nucP_init":>9s} {"n_traj":>9s} {"T":>8s}' + ''.join(
        f' {k:>11s} {k + "_err":>11s}' for k in _KEYS) + ''.join(
        f' {"max|" + k + "|":>11s} {"err":>9s}' for k in _MKEYS)
    lines = [f'# {label}  (final values C(T); max over t of |M(t)|, err = SE at that t; '
             f'nblocks as saved per point, {a.nblocks} for the points analysed in this run)', head]
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
    ap.add_argument('--delete-shards', action='store_true',
                    help='after analysing a point from COMPLETE shards (and saving transmission.npz), '
                         'delete its shards/ and shards_partials/ (batch mode)')
    ap.add_argument('--saved-only', action='store_true',
                    help='do not analyse any shards: rebuild the summary from transmission.npz only')
    ap.add_argument('--replot', action='store_true',
                    help='redraw analyze_transmission.png of points read from transmission.npz (e.g. with a '
                         'new --tmax); without it only a missing png is redrawn')
    ap.add_argument('-v', '--verbose', action='store_true', help='print tracebacks of failed points')
    a = ap.parse_args()
    for d in a.dirs:
        if os.path.basename(os.path.normpath(d)).startswith('__'):     # __pycache__ from a glob
            continue
        run_method(d, a)


if __name__ == '__main__':
    _main()
