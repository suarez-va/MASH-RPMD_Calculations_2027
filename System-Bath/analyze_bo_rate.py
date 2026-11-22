"""
analyze_bo_rate.py -- the acceptor population and rate of a Born-Oppenheimer run (dynamics='bo',
e.g. Low-Friction/Delta-Scan/BO-Classical): the nuclei never leave the lower adiabat, so there are no
weights and no spin -- a trajectory is on the acceptor side when its centroid is past the crossing.
Output is a 1x2 figure (analyze_bo_rate.png): P_2(t) on the left, K_12(t) with its block analysis
of the rate on the right, built exactly as in analyze_rate.py so the two rates compare.

--- the observables --------------------------------------------------------------------------------

    P_2(t)  = < h(Rbar(t) - R*) >,       R* = -epsil / sqrt(2*lbd*kvec)

h is the Heaviside step (h(0) = 1/2), Rbar the bead-average (centroid) nucR, and <...> the average
over trajectories.  R* is the diabatic crossing, kappa(R*) = sqrt(lbd*kvec/2) R* + epsil/2 = 0 (0 for
epsil = 0), which is the top of the barrier on the lower adiabat Vbar - Vz.  Then, as for MASH,

    P2eq    = 1 / (1 + exp(-beta*epsil))
    K_12(t) = -P2eq ln| 1 - P_2(t)/P2eq |

and the rate is the slope of a least-squares line fit of K_12 over [fitmin, fitmax].

--- error bars -------------------------------------------------------------------------------------

The blocks are `nblocks` contiguous groups of whole units (np.array_split over the units, in
trajectory order), exactly as in analyze_rate.py:

    P_2   SE = std of the block means / sqrt(nblocks)
    K_12  SE = SE(P_2) / |u|,  u = 1 - P_2/P2eq
    rate  m_bar = mean of the per-block slopes,  sigma_m = std(slopes, ddof=1) / sqrt(nblocks)

--- how it runs ------------------------------------------------------------------------------------

The same cached, parallel map over units of <= --unit-rows trajectories as analyze_rate.py
(workflow.plan_units / map_cached): each unit is reduced to its count
of trajectories past R* at every time, cached in <file>_partials/<version>/ (the version holds
_VERSION and R*, so --epsil gets its own partials) and reused by every later run.  A shard still
being written is read without a lock, never cached, and skipped with a warning if caught mid-append.

--- the saved result: bo_rate.npz (so the shards can be deleted) -----------------------------------

Every analysis of raw data (shards/ or data.hdf) also writes <point dir>/bo_rate.npz: the per-unit
partials themselves (unit_n: trajectories per unit, unit_past: count past R* at every time, in
trajectory order) + the time grid, R*, the run's config_json and its configured n_traj.  That is
everything this analysis is built from, so from bo_rate.npz alone P_2, K_12, the fit and the block
analysis are recomputed exactly -- with ANY --nblocks (blocks are groups of whole units), --fitmin /
--fitmax, --beta and --tmax.  Only --epsil (it moves R*) needs the raw data again.  A few hundred kB
for 10^6 trajectories.

    --delete-shards   after bo_rate.npz is written, delete shards/ and shards_partials/ -- only when
                      the run is complete (n_traj reached the config's n_traj, no shard still being
                      written); otherwise they are kept, with a warning
    --saved-only      analyse bo_rate.npz even if the raw data is still there

With no shards/ or data.hdf in the point dir, bo_rate.npz is read automatically.

Usage (from a BO scan-point dir, e.g. Low-Friction/Delta-Scan/BO-Classical/delta_1e-1.10,
PYTHONPATH=<RP_MASH root>):
    python ../../../../analyze_bo_rate.py                     # shards/, else data.hdf, else bo_rate.npz
    python ../../../../analyze_bo_rate.py --file shards --ncores 16 --nblocks 100
    python ../../../../analyze_bo_rate.py --ncores 16 --delete-shards     # analyse, save, free the disk
    python ../../../../analyze_bo_rate.py --nblocks 50 --fitmin 3         # later: from bo_rate.npz
"""

import os
import json
import shutil
import argparse
import functools

import numpy as np

from workflow import plan_units, map_cached, ncores_source, open_readonly   # cached map-reduce
from analyze_rate import (block_sem_from_means, k12_from_p2, line_fit, block_rate,
                          default_cache_dir, k12_panel, _band)


# Bump whenever the observable changes: every cached partial is then recomputed.
_VERSION = 'bo_v1'
SAVED = 'bo_rate.npz'          # per-point saved result (the per-unit partials), survives deleting the shards


def crossing(kvec, lbd, epsil):
    """R* where the diabats cross, kappa(R*) = 0: the barrier top of the lower adiabat."""
    return -float(epsil) / np.sqrt(2.0 * float(lbd) * float(kvec)) + 0.0     # + 0.0: no -0 for epsil = 0


def _unit_partial(path, lo, hi, time, Rs):
    """
    MAP step, run in a worker: the number of trajectories of rows lo..hi of one file with their
    centroid past R* at every time (h(0) = 1/2).
    """
    with open_readonly(path) as f:                     # no lock: never blocks a running writer
        if not np.array_equal(f['time'][:], time):
            raise ValueError(f'{path}: time grid differs from the rest of the run')
        Rbar = np.asarray(f['nucR'][lo:hi, :, :, 0], dtype=np.float64).mean(axis=2)   # (s, T)
    past = (Rbar > Rs).sum(axis=0) + 0.5 * (Rbar == Rs).sum(axis=0)
    return dict(n=np.int64(hi - lo), past=past.astype(np.float64))


def reduce(path, epsil_ovr=None, ncores=None, unit_rows=1000, cache=True, refresh=False, cache_dir=None):
    """
    REDUCE step: every unit of `path` (a shard directory or a data.hdf) reduced to its partial count
    by _unit_partial on `ncores` workers, or read back from the cache.  Returns the per-unit partials
    (unit_n, unit_past, in trajectory order) and what the analysis needs besides them -- exactly
    what bo_rate.npz stores.
    """
    units = plan_units(path, unit_rows)
    with open_readonly(units[0].file) as f:
        time = f['time'][:]
        cfg_json = f.attrs['config_json']
        nbds = int(f.attrs['nbds'])
    cfg = json.loads(cfg_json)
    if cfg.get('dynamics', 'mash') != 'bo':
        raise RuntimeError(f"{units[0].file} is a {cfg.get('dynamics', 'mash')!r} run, not dynamics='bo': "
                           f"analyse MASH runs with analyze_rate.py")
    epsil = float(cfg['epsil']) if epsil_ovr is None else float(epsil_ovr)
    Rs = crossing(cfg['kvec'], cfg['lbd'], epsil)

    if ncores is None:
        ncores, why = ncores_source()
    else:
        why = 'given --ncores'
    version = f'{_VERSION}_{Rs:.12g}'
    cache_dir = cache_dir or default_cache_dir(path, version)
    fn = functools.partial(_unit_partial, time=time, Rs=Rs)
    parts, st = map_cached(fn, units, cache_dir, version, ncores=ncores, refresh=refresh,
                           use_cache=cache, tag='[bo]')
    n_inprog = sum(not u.complete for u in units)
    print(f'[bo] {len(units)} unit(s) of <= {unit_rows} trajectories: {st["cached"]} from cache, '
          f'{st["computed"] + st["in_progress"]} computed on {ncores} worker(s) ({why})'
          + (f'; {n_inprog} unit(s) in shards still being written (never cached)' if n_inprog else '')
          + ('' if cache else '; cache off'), flush=True)
    for note in st['notes']:
        print(f'[bo] WARNING: {note}', flush=True)

    used = [p for p in parts if p is not None]
    if not used:
        raise RuntimeError('no unit could be read: every file is still being written. Rerun later.')
    return dict(time=np.asarray(time, dtype=np.float64),
                unit_n=np.array([int(p['n']) for p in used], dtype=np.int64),
                unit_past=np.stack([p['past'] for p in used]).astype(np.float64),
                Rs=float(Rs), epsil=float(epsil), nbds=nbds, config_json=str(cfg_json),
                n_traj_cfg=int(cfg.get('n_traj', 0)), n_files=len({u.file for u in units}),
                complete=bool(n_inprog == 0 and len(used) == len(units)), unit_rows=int(unit_rows),
                cache_stats=st)


def save_reduced(path, red):
    """bo_rate.npz: the per-unit partials + metadata (written whole, then renamed: never a half file)."""
    tmp = path + '.tmp.npz'
    np.savez(tmp, version=_VERSION, time=red['time'], unit_n=red['unit_n'], unit_past=red['unit_past'],
             Rs=red['Rs'], epsil=red['epsil'], nbds=red['nbds'], config_json=red['config_json'],
             n_traj_cfg=red['n_traj_cfg'], n_files=red['n_files'], complete=red['complete'],
             unit_rows=red['unit_rows'])
    os.replace(tmp, path)


def load_reduced(path, epsil_ovr=None):
    """The reduce() dict back from a bo_rate.npz."""
    with np.load(path, allow_pickle=False) as z:
        if str(z['version']) != _VERSION:
            raise RuntimeError(f'{path} was written by analyze_bo_rate {z["version"]}, this is {_VERSION}: '
                               f'it needs the raw data again')
        red = {k: z[k] for k in z.files}
    for k in ('Rs', 'epsil'):
        red[k] = float(red[k])
    for k in ('nbds', 'n_traj_cfg', 'n_files', 'unit_rows'):
        red[k] = int(red[k])
    red['complete'] = bool(red['complete'])
    red['config_json'] = str(red['config_json'])
    red['cache_stats'] = None
    if epsil_ovr is not None and not np.isclose(float(epsil_ovr), red['epsil']):
        raise RuntimeError(f'--epsil {epsil_ovr} moves R*, but {path} holds the counts past R* = '
                           f'{red["Rs"]:.6g} (epsil {red["epsil"]:g}): that needs the raw data')
    return red


def finish(red, nblocks=None, fitmin=5.0, fitmax=15.0, beta_ovr=None):
    """P_2, K_12, the full fit and the block analysis from the per-unit partials (reduce / bo_rate.npz)."""
    time, used_n, used_past = red['time'], red['unit_n'], red['unit_past']
    cfg = json.loads(red['config_json'])
    kvec, lbd, delta = float(cfg['kvec']), float(cfg['lbd']), float(cfg['delta'])
    epsil, Rs, nbds = red['epsil'], red['Rs'], red['nbds']
    beta = float(cfg['beta']) if beta_ovr is None else float(beta_ovr)
    n_units = int(used_n.size)
    n_traj = int(used_n.sum())

    nb = n_units if nblocks is None else max(1, min(int(nblocks), n_units))
    if nblocks is not None and int(nblocks) > n_units:
        print(f'[bo] NOTE: only {n_units} units, so {nb} blocks instead of {nblocks}', flush=True)
    tot = np.zeros(time.size)
    blk = np.zeros((nb, time.size))
    block_sizes = np.zeros(nb)
    for b, grp in enumerate(np.array_split(np.arange(n_units), nb)):
        for i in grp:
            tot += used_past[i]
            blk[b] += used_past[i]
            block_sizes[b] += int(used_n[i])

    P2 = tot / n_traj
    blk_P2 = blk / block_sizes[:, None]
    P2_SE = block_sem_from_means(blk_P2, nb)
    P2eq = 1.0 / (1.0 + np.exp(-beta * epsil))
    K12, K12_SE, _ = k12_from_p2(P2, P2eq, P2_SE)
    m_full, b_full, r_full, n_full = line_fit(time, K12, fitmin, fitmax)
    br = block_rate(time, blk_P2, P2eq, fitmin, fitmax)

    return dict(time=time, P2=P2, P2_SE=P2_SE, K12=K12, K12_SE=K12_SE,
                m_full=m_full, b_full=b_full, r_full=r_full, n_full=n_full, **br,
                P2eq=P2eq, beta=beta, Rs=Rs, n_traj=n_traj, nbds=nbds, nblocks=nb,
                n_files=red['n_files'], n_units=n_units, cache_stats=red['cache_stats'],
                n_traj_cfg=red['n_traj_cfg'], complete=red['complete'],
                fitmin=fitmin, fitmax=fitmax, params=dict(kvec=kvec, epsil=epsil, lbd=lbd, delta=delta))


def compute(path, nblocks=None, fitmin=5.0, fitmax=15.0, beta_ovr=None, epsil_ovr=None,
            ncores=None, unit_rows=1000, cache=True, refresh=False, cache_dir=None, save_to=None):
    """
    reduce() + finish(): the analysis of a shard directory or a data.hdf -- or of a bo_rate.npz, if
    `path` is one.  save_to: also write the per-unit partials there (bo_rate.npz).
    """
    if str(path).endswith('.npz'):
        red = load_reduced(path, epsil_ovr)
    else:
        red = reduce(path, epsil_ovr=epsil_ovr, ncores=ncores, unit_rows=unit_rows, cache=cache,
                     refresh=refresh, cache_dir=cache_dir)
        if save_to and epsil_ovr is None:
            save_reduced(save_to, red)
        elif save_to:                        # never overwrite the run's own counts with a moved R*
            print(f'[bo] --epsil given: {save_to} not written (it holds the counts past the run\'s own R*)',
                  flush=True)
    return finish(red, nblocks=nblocks, fitmin=fitmin, fitmax=fitmax, beta_ovr=beta_ovr)


def raw_source(point_dir):
    """shards/ or data.hdf of a point dir, or None if the raw data is gone."""
    for name, ok in (('shards', os.path.isdir), ('data.hdf', os.path.isfile)):
        p = os.path.join(point_dir, name)
        if ok(p):
            return p
    return None


def analyze_point(point_dir, saved_only=False, **kw):
    """
    (result, 'raw' | 'saved') of one point dir: from its shards/ (or data.hdf) if present -- saving
    bo_rate.npz next to them -- else (or with saved_only) from its bo_rate.npz.
    """
    raw, saved = raw_source(point_dir), os.path.join(point_dir, SAVED)
    if raw is not None and not saved_only:
        return compute(raw, save_to=saved, **kw), 'raw'
    if os.path.isfile(saved):
        kw = {k: v for k, v in kw.items() if k in ('nblocks', 'fitmin', 'fitmax', 'beta_ovr', 'epsil_ovr')}
        return compute(saved, **kw), 'saved'
    raise RuntimeError(f'no shards/, data.hdf or {SAVED} in {point_dir or "."}'
                       + (' (--saved-only)' if raw is not None else ''))


def delete_shards(point_dir, r, tag='[bo]'):
    """
    After an analysis of COMPLETE raw data whose bo_rate.npz exists: remove shards/ and
    shards_partials/.  A data.hdf is left alone (delete it by hand).  Returns True if deleted.
    """
    if not r.get('complete', False) or r['n_traj'] < r['n_traj_cfg']:
        print(f"{tag} NOT deleting the shards: {r['n_traj']} of {r['n_traj_cfg']} trajectories read"
              + ('' if r.get('complete', False) else ', some shards still being written')
              + ' (still running, or failed trajectories?)', flush=True)
        return False
    if not os.path.isfile(os.path.join(point_dir, SAVED)):           # belt and braces
        raise RuntimeError(f'{SAVED} missing in {point_dir or "."}: refusing to delete the shards')
    gone = []
    for name in ('shards', 'shards_partials'):
        p = os.path.join(point_dir, name)
        if os.path.isdir(p):
            shutil.rmtree(p)
            gone.append(name + '/')
    if gone:
        print(f"{tag} deleted {' and '.join(gone)} (result kept in {SAVED})", flush=True)
    elif os.path.isfile(os.path.join(point_dir, 'data.hdf')):
        print(f"{tag} raw data is a data.hdf: not deleted automatically", flush=True)
    return bool(gone)


def _plot(out, r, tmax=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    t = r['time']
    K12_blks = r['K12_blks']
    if tmax is not None:                         # truncate for plotting only (fits already done)
        msk = t <= tmax
        t = t[msk]
        K12_blks = [k[msk] for k in K12_blks]
        r = {k: (v[msk] if isinstance(v, np.ndarray) and v.shape == msk.shape else v)
             for k, v in r.items()}

    fig, ax = plt.subplots(1, 2, figsize=(13.0, 5.0), sharex=True)
    _band(ax[0], t, r['P2'], r['P2_SE'], 'C0', r'$P_2(t)=\langle h(\bar R(t)-R^*)\rangle$')
    ax[0].axhline(r['P2eq'], color='k', lw=1.0, ls=':', label=r'$P_2^{\mathrm{eq}}$')
    ax[0].legend(loc='best', frameon=False, fontsize=9)
    ax[0].set_ylabel(r'$P_2(t)$')
    k12_panel(plt, ax[1], t, r, K12_blks)

    p = r['params']
    fig.suptitle(f"BO (lower adiabat): n_traj={r['n_traj']}, nbds={r['nbds']}, nblocks={r['nblocks']}   "
                 f"delta={p['delta']:.4g} (log10 {np.log10(p['delta']):.4g}), R*={r['Rs']:.4g}\n"
                 f"P2eq={r['P2eq']:.6f} (beta={r['beta']:g}, epsil={p['epsil']:g})   "
                 f"fit window [{r['fitmin']:g}, {r['fitmax']:g}]", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out, dpi=200)
    plt.close(fig)


def _main():
    ap = argparse.ArgumentParser(
        description='Acceptor population P2(t) = <h(Rbar - R*)> and K12(t) with a block analysis of '
                    'the rate, from the shards (or a data.hdf) of a dynamics=\'bo\' run.')
    ap.add_argument('--file',    default=None,
                    help='a shard directory, a data.hdf or a bo_rate.npz (default: shards/ if it exists, '
                         'else data.hdf, else bo_rate.npz)')
    ap.add_argument('--delete-shards', action='store_true',
                    help='after saving bo_rate.npz, delete shards/ and shards_partials/ (complete runs only)')
    ap.add_argument('--saved-only', action='store_true',
                    help='analyse bo_rate.npz even if shards/ or data.hdf is still there')
    ap.add_argument('--nblocks', type=int, default=100, help='trajectory blocks (default 100)')
    ap.add_argument('--fitmin',  type=float, default=5.0, help='fit window start (default 5)')
    ap.add_argument('--fitmax',  type=float, default=15.0, help='fit window end (default 15)')
    ap.add_argument('--tmax',    type=float, default=None, help='truncate the PLOT to t <= tmax')
    ap.add_argument('--beta',    type=float, default=None, help='override full inverse temperature')
    ap.add_argument('--epsil',   type=float, default=None,
                    help='override driving force epsil (moves R* and P2eq)')
    ap.add_argument('--unit-rows', type=int, default=1000,
                    help='trajectories per unit of work / cached partial (default 1000); the blocks '
                         'are groups of whole units')
    ap.add_argument('--no-cache', action='store_true',
                    help='neither read nor write the per-unit partials')
    ap.add_argument('--refresh', action='store_true',
                    help='recompute (and re-cache) every unit, ignoring existing partials')
    ap.add_argument('--cache-dir', default=None, help='where the partials live')
    ap.add_argument('--ncores',  type=int, default=None,
                    help='worker processes (default: the CPUs of the SLURM allocation, else 1)')
    ap.add_argument('--out',     default='analyze_bo_rate.png', help='output png')
    a = ap.parse_args()
    kw = dict(nblocks=a.nblocks, fitmin=a.fitmin, fitmax=a.fitmax, beta_ovr=a.beta, epsil_ovr=a.epsil,
              ncores=a.ncores, unit_rows=a.unit_rows, cache=not a.no_cache, refresh=a.refresh,
              cache_dir=a.cache_dir)
    try:
        if a.file is None:                   # this point dir: raw data if present, else bo_rate.npz
            r, src = analyze_point('', saved_only=a.saved_only, **kw)
            path = (raw_source('') if src == 'raw' else SAVED)
        else:
            path = a.file
            point = os.path.dirname(os.path.normpath(path))
            save_to = None if path.endswith('.npz') else os.path.join(point, SAVED)
            r, src = compute(path, save_to=save_to, **kw), ('saved' if path.endswith('.npz') else 'raw')
    except RuntimeError as e:
        raise SystemExit(f'[bo] {e}')
    _plot(a.out, r, tmax=a.tmax)
    if src == 'raw' and a.epsil is None:
        print(f"[bo] saved the per-unit partials to {os.path.join(os.path.dirname(os.path.normpath(path)), SAVED)}")
    else:
        print(f"[bo] analysed from {path} (no raw data read)")

    p = r['params']
    print(f"[bo] {path}: n_traj={r['n_traj']} from {r['n_files']} file(s) / {r['n_units']} unit(s), "
          f"nbds={r['nbds']}, nblocks={r['nblocks']}")
    print(f"[bo] params: kvec={p['kvec']} epsil={p['epsil']} lbd={p['lbd']} delta={p['delta']:.6e} "
          f"(log10 {np.log10(p['delta']):.4g})")
    print(f"[bo] beta={r['beta']:g}  P2eq={r['P2eq']:.6f}  crossing R*={r['Rs']:.6g}")
    print(f"[bo] P2(0)={r['P2'][0]:.6e} +/- {r['P2_SE'][0]:.3e}   "
          f"P2(T)={r['P2'][-1]:.6e} +/- {r['P2_SE'][-1]:.3e}")
    print(f"[bo] FULL fit of K12 over t=[{r['fitmin']:g},{r['fitmax']:g}] "
          f"({r['n_full']} pts):  m={r['m_full']:.6g}  b={r['b_full']:.6g}  r={r['r_full']:.6f}")
    print(f"[bo] BLOCK analysis ({r['slopes'].size} usable blocks):  "
          f"m = {r['m_bar']:.6g} +/- {r['sigma_m']:.6g}")
    if r['n_nonphys']:
        print(f"[bo] WARNING: {r['n_nonphys']} of {r['nblocks']} blocks have u = 1 - P2/P2eq <= 0 "
              f"inside the fit window; those block slopes are unphysical. Use fewer, larger blocks.")
    if r['n_skipped']:
        print(f"[bo] WARNING: {r['n_skipped']} block(s) had <2 finite points in the window "
              f"and were skipped.")
    print(f"[bo] wrote {a.out}")
    if a.delete_shards:
        if src == 'raw' and os.path.basename(os.path.normpath(path)) == 'shards':
            delete_shards(os.path.dirname(os.path.normpath(path)), r)
        elif src == 'raw':
            print('[bo] --delete-shards: the raw data is a data.hdf, not deleted automatically')
        else:
            print('[bo] --delete-shards: nothing to delete (analysed from bo_rate.npz)')


if __name__ == '__main__':
    _main()
