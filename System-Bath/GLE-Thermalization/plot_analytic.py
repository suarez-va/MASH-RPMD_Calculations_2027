"""
plot_analytic.py -- quick look at the ANALYTIC position distributions of analyze_gle.py's
histogram.png, no simulation data: Classical, Quantum (no friction), and the GLE-RPMD analytic bead
distribution for a scan of nbds, all on one figure.  Same formulas as analyze_gle.analytic():

    classical          var = 1/(beta kvec)
    quantum harmonic   var = 1/(2 m omega tanh(beta omega/2))         (no friction)
    GLE-RPMD analytic  var = (1/nbds) sum_k 1/(beta_p (kvec + m w_k^2 + m gamma w_k))

nbds = 1 gives the classical width.  With gamma > 0 the GLE curve does not approach the frictionless
quantum width as nbds grows: the m gamma w_k term narrows every internal mode.

Usage (from GLE-Thermalization/, PYTHONPATH=<RP_MASH root>):
    python plot_analytic.py                                   # parameters from config.py
    python plot_analytic.py --nbds 1 4 16 64 256 --gamma 0.25 --kvec 0.0625
"""

import os
import argparse
import importlib.util

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

here = os.path.dirname(os.path.abspath(__file__))


def _load(name):
    spec = importlib.util.spec_from_file_location(f'_pa_{name}', os.path.join(here, f'{name}.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ag = _load('analyze_gle')                      # analytic(), gauss()
cfg = dict(_load('config').CONFIG)

ap = argparse.ArgumentParser()
ap.add_argument('--nbds',  type=int, nargs='+', default=[1, 2, 4, 8, 16, 32, 64])
ap.add_argument('--kvec',  type=float, default=None)
ap.add_argument('--gamma', type=float, default=None)
ap.add_argument('--beta',  type=float, default=None)
ap.add_argument('--out',   default='analytic_scan.png')
a = ap.parse_args()
if a.kvec is not None:
    cfg['kvec'] = a.kvec
beta  = float(cfg['beta'])  if a.beta  is None else a.beta
gamma = float(cfg['langevin_params']['gamma']) if a.gamma is None else a.gamma

A = {n: ag.analytic(cfg, n, beta, gamma) for n in a.nbds}
A0 = A[a.nbds[0]]
R0, var_cl, var_qm = A0['R0'], A0['var_cl'], A0['var_qm']

print(f"kvec={cfg['kvec']:g} lbd={cfg['lbd']:g} mass={A0['mass']:g} beta={beta:g} gamma={gamma:g}   "
      f"R0={R0:.4f}")
print(f"classical var = {var_cl:.5f}   quantum (no friction) var = {var_qm:.5f}")
print(f"{'nbds':>6} {'var_gle':>10} {'/classical':>11} {'/quantum':>9}")
for n in a.nbds:
    v = A[n]['var_gle']
    print(f"{n:6d} {v:10.5f} {v / var_cl:11.4f} {v / var_qm:9.4f}")

sd = np.sqrt(max([var_cl, var_qm] + [A[n]['var_gle'] for n in a.nbds]))
x = np.linspace(R0 - 6 * sd, R0 + 6 * sd, 800)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(x, ag.gauss(x, R0, var_cl), color='#e34948', lw=2.2, label=f'Classical (var {var_cl:.3g})')
ax.plot(x, ag.gauss(x, R0, var_qm), color='#1baf7a', lw=2.2,
        label=f'Quantum, no friction (var {var_qm:.3g})')
cols = plt.cm.viridis(np.linspace(0, 0.9, len(a.nbds)))
for c, n in zip(cols, a.nbds):
    ax.plot(x, ag.gauss(x, R0, A[n]['var_gle']), color=c, lw=1.6, ls='--',
            label=f"GLE-RPMD nbds={n} (var {A[n]['var_gle']:.3g})")
ax.set_xlabel('position (a.u.)')
ax.set_ylabel('probability density')
ax.set_title(f"Analytic bead distributions   kvec={cfg['kvec']:g}, beta={beta:g}, gamma={gamma:g}",
             loc='left')
ax.legend(frameon=False, fontsize=8)
ag._style(ax)
fig.tight_layout()
out = a.out if os.path.isabs(a.out) else os.path.join(here, a.out)
fig.savefig(out, dpi=200)
print(f'wrote {out}')
