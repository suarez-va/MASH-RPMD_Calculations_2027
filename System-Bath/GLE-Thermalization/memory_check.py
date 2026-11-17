"""
memory_check.py -- how long must the GLE memory kernel be?  Pure numerics on memK, no trajectories.

memK is generated exactly as the integrator does (utils.friction_kernel on the config's delt grid),
and integrated with numpy as a function of the cutoff Tmem:

    R_k(Tmem) = gamma + delt * ( K_k(0)/2 + sum_{l=1}^{Nmem-1} K_k(l*delt) ),   Nmem = ceil(Tmem/delt)

i.e. the instantaneous friction (gamma + delt*K(0)/2) plus the memory sum the code applies.  For
every internal mode the exact kernel satisfies gamma + int_0^inf K_k dt = 0 (Khat_k(0) = 0: the long
memory cancels the delta friction), so R_k -> 0 and |R_k|/gamma is the Markovian friction left over
by cutting the kernel at Tmem.  The centroid has K_0 = 0, so its R_0 = gamma for any Tmem (plain
Langevin, Tmem irrelevant) and it is not shown.

Usage (PYTHONPATH=<RP_MASH root>):
    python memory_check.py --config ../Low-Friction/Delta-Scan/MASH-Quantum/config.py
    python memory_check.py --config <cfg> --tmax 100 --tol 0.01
"""

import os
import argparse
import importlib.util

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import utils, normal_mode                         # RP_MASH

here = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument('--config', default=os.path.join(here, 'config.py'))
ap.add_argument('--tmax', type=float, default=50.0, help='longest Tmem to evaluate (a.u.)')
ap.add_argument('--tol', type=float, default=0.01, help='threshold on |R_k|/gamma for the verdict')
ap.add_argument('--out', default='memory_check.png')
a = ap.parse_args()

spec = importlib.util.spec_from_file_location('_mc_cfg', os.path.abspath(a.config))
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
cfg = mod.CONFIG
nbds, beta = int(cfg['nbds']), float(cfg['beta'])
gamma, delt = float(cfg['langevin_params']['gamma']), float(cfg['delt'])
Tmem_cfg = float(cfg['langevin_params']['Tmem'])

w_all = normal_mode.calc_normal_mode_freq(beta / nbds, nbds)
ks = list(range(1, nbds // 2 + 1))                # distinct internal modes (k, nbds-k degenerate)
if not ks:
    raise SystemExit('nbds = 1: only the centroid, whose kernel is zero -- Tmem does not matter')
wk = w_all[ks]

n = int(np.ceil(a.tmax / delt))
t = np.arange(n) * delt
K = utils._friction_kernel(t, gamma, wk)          # (modes, n), the continuous part as in memK
cum = np.cumsum(K, axis=1)                        # cum[:, m] = sum_{l=0}^{m} K_l
# Nmem = m+1 lags kept -> R = gamma + delt*(K0/2 + sum_{l=1}^{m} K_l) = gamma + delt*(cum[m] - K0/2)
R = gamma + delt * (cum - 0.5 * K[:, :1])         # (modes, n), R[:, m] for Tmem = (m+1)*delt
Tm = (np.arange(n) + 1) * delt
rel = np.abs(R) / gamma

print(f'{os.path.abspath(a.config)}\n  nbds={nbds} beta={beta:g} gamma={gamma:g} delt={delt:g} '
      f'Tmem(config)={Tmem_cfg:g}  ({n} lags up to {a.tmax:g})')
marks = [x for x in (1, 2, 3, 5, 7.5, 10, 15, 20, 25, 35, 50, 75, 100) if x <= a.tmax]
if Tmem_cfg <= a.tmax and Tmem_cfg not in marks:
    marks = sorted(marks + [Tmem_cfg])
print(f"\n{'k':>3} {'w_k':>6} " + ''.join(f'{x:>9g}' for x in marks) + '   <- Tmem;  |R_k|/gamma')
for i, k in enumerate(ks):
    vals = [rel[i, int(np.ceil(x / delt)) - 1] for x in marks]
    print(f'{k:3d} {wk[i]:6.2f} ' + ''.join(f'{v:9.4f}' for v in vals))
# smallest Tmem after which every mode stays within tol (the residual oscillates, so "stays")
worst = rel.max(axis=0)
bad = np.nonzero(worst > a.tol)[0]
stay = Tm[bad[-1] + 1] if bad.size and bad[-1] + 1 < n else (Tm[0] if not bad.size else None)
print(f'\nworst mode at config Tmem={Tmem_cfg:g}: '
      + (f'{worst[int(np.ceil(Tmem_cfg / delt)) - 1]:.4f}' if Tmem_cfg <= a.tmax else 'beyond --tmax'))
print(f'every mode stays within |R|/gamma <= {a.tol:g} for Tmem >= '
      + (f'{stay:.3g}' if stay is not None else f'(not reached by {a.tmax:g})'))

step = max(1, n // 20000)                         # thin the curve for plotting only
cols = plt.cm.viridis(np.linspace(0, 0.9, len(ks)))
fig, ax = plt.subplots(figsize=(8, 5))
for i, k in enumerate(ks):
    ax.plot(Tm[::step], np.maximum(rel[i, ::step], 1e-8), color=cols[i], lw=1.2,
            label=f'k={k} (w_k={wk[i]:.2f})')
ax.axhline(a.tol, color='#e34948', ls='--', lw=1, label=f'tol {a.tol:g}')
ax.axvline(Tmem_cfg, color='gray', ls=':', lw=1, label=f'config Tmem {Tmem_cfg:g}')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('Tmem (a.u.)')
ax.set_ylabel(r'$|\gamma + \delta t\,(K_0/2 + \sum_{l\geq1} K_l)|\ /\ \gamma$')
ax.set_title(f'GLE memory truncation   nbds={nbds}, beta={beta:g}, gamma={gamma:g}, delt={delt:g}',
             loc='left')
ax.grid(True, alpha=0.2)
ax.legend(fontsize=8, frameon=False)
fig.tight_layout()
out = a.out if os.path.isabs(a.out) else os.path.join(here, a.out)
fig.savefig(out, dpi=150)
print(f'wrote {out}')
