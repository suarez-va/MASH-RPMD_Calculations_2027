"""
plot_free_rp.py -- sampled bead distribution of a FREE ring polymer with its centroid fixed at 0,
for a scan of nbds, against the analytic result.  The sampling is System-Bath/therm_scan.py's
_sample_nvt with kvec = 0 and gamma = 0: every internal normal mode k >= 1 is drawn as

    Q_k ~ N(0, dR_k),   dR_k = sqrt(nbds / (beta * mass * w_k^2)),   w_k = (2 nbds/beta) sin(k pi/nbds)

(w_k from normal_mode.calc_normal_mode_freq(beta/nbds, nbds)), the centroid mode Q_0 is set to 0,
and the beads come from normal_mode.normal_mode_to_real -- the same calls _sample_nvt makes.

Analytic reference (orthonormal normal-mode transform, so a bead's variance is the mean over modes):

    <R_b^2> = (1/nbds) sum_{k>=1} dR_k^2 = beta (nbds^2 - 1) / (12 mass nbds^2)
            -> beta hbar^2 / (12 mass)    as nbds -> inf   (free-particle radius of gyration^2)

from sum_{k=1}^{n-1} csc^2(k pi/n) = (n^2 - 1)/3.

Usage (PYTHONPATH=<RP_MASH root>):
    python plot_free_rp.py                                        # beta 1, mass 2000 (Tully)
    python plot_free_rp.py --nbds 2 6 12 24 48 --beta 1 --mass 2000 --nsamples 50000
"""

import os
import argparse

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import normal_mode                                # RP_MASH

here = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument('--nbds',     type=int, nargs='+', default=[2, 4, 8, 16, 32, 64])
ap.add_argument('--beta',     type=float, default=1.0)
ap.add_argument('--mass',     type=float, default=2000.0)
ap.add_argument('--nsamples', type=int, default=20000, help='ring polymers drawn per nbds')
ap.add_argument('--seed',     type=int, default=None)
ap.add_argument('--bins',     type=int, default=120)
ap.add_argument('--out',      default='free_rp.png')
a = ap.parse_args()
rng = np.random.default_rng(a.seed)


def sample(nbds, beta, mass, n, rng):
    """(n, nbds) bead positions: _sample_nvt with kvec = gamma = 0 and the centroid mode at 0."""
    omega_k = normal_mode.calc_normal_mode_freq(beta / nbds, nbds)
    dR_k = np.zeros(nbds)
    dR_k[1:] = np.sqrt(nbds / (beta * mass * omega_k[1:]**2))     # k = 0 would be 1/0: centroid fixed
    nucR_nm = rng.normal(0.0, dR_k, (n, nbds))
    nucR_nm[:, 0] = 0.0
    return np.array([normal_mode.normal_mode_to_real(q) for q in nucR_nm])


def var_exact(nbds, beta, mass):
    return beta * (nbds**2 - 1) / (12.0 * mass * nbds**2)


nbds_list = [n for n in a.nbds if n > 1]
if len(nbds_list) < len(a.nbds):
    print('note: nbds = 1 skipped (only the centroid, which is fixed at 0)')
var_inf = a.beta / (12.0 * a.mass)
print(f'beta={a.beta:g} mass={a.mass:g} nsamples={a.nsamples}   nbds -> inf: var = {var_inf:.6e}')
print(f"{'nbds':>6} {'sampled var':>13} {'analytic':>13} {'ratio':>8} {'max|centroid|':>14}")
data = {}
for n in nbds_list:
    R = sample(n, a.beta, a.mass, a.nsamples, rng)
    v, ve = R.var(), var_exact(n, a.beta, a.mass)
    data[n] = R
    print(f'{n:6d} {v:13.6e} {ve:13.6e} {v / ve:8.4f} {np.abs(R.mean(axis=1)).max():14.2e}')

sd = np.sqrt(max([var_inf] + [var_exact(n, a.beta, a.mass) for n in nbds_list]))
edges = np.linspace(-5 * sd, 5 * sd, a.bins + 1)
x = np.linspace(edges[0], edges[-1], 800)
gauss = lambda x, v: np.exp(-0.5 * x**2 / v) / np.sqrt(2 * np.pi * v)

fig, ax = plt.subplots(figsize=(8, 5))
cols = plt.cm.viridis(np.linspace(0, 0.9, len(nbds_list)))
for c, n in zip(cols, nbds_list):
    ve = var_exact(n, a.beta, a.mass)
    ax.hist(data[n].ravel(), bins=edges, density=True, histtype='step', color=c, lw=1.3)
    ax.plot(x, gauss(x, ve), color=c, lw=1.6, ls='--', label=f'nbds={n} (var {ve:.3g})')
ax.plot(x, gauss(x, var_inf), color='k', lw=1.8, label=rf'nbds$\to\infty$: $\beta/12m$ ({var_inf:.3g})')
ax.set_xlabel('bead position relative to the centroid (a.u.)')
ax.set_ylabel('probability density')
ax.set_title(f'Free ring polymer, centroid at 0   beta={a.beta:g}, mass={a.mass:g}   '
             f'(steps: sampled, dashed: analytic)', loc='left', fontsize=10)
ax.legend(frameon=False, fontsize=8)
for s in ('top', 'right'):
    ax.spines[s].set_visible(False)
ax.grid(True, color='k', alpha=0.07, lw=0.7)
fig.tight_layout()
out = a.out if os.path.isabs(a.out) else os.path.join(here, a.out)
fig.savefig(out, dpi=200)
print(f'wrote {out}')
