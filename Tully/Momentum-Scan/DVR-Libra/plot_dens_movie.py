"""
plot_dens_movie.py -- animate the exact ground / excited ADIABATIC nuclear densities of one scan point

    rho_g(x,t) = |psi_g(x,t)|^2      lower adiabat, RED
    rho_e(x,t) = |psi_e(x,t)|^2      upper adiabat, BLUE

into an .mp4.  This is the exact counterpart of ../../plot_rho_movie.py (MASH: rho_-(R,t) on the
lower adiabat, rho_+(R,t) on the upper), drawn the same way:

    integral rho_g dx = c_-(t) (= Cm in transmission.npz),   integral rho_e dx = c_+(t) (= Cp)

It reads the raw densities.npz every run writes.  2_collect.py --delete-raw removes it; to make a
movie afterwards, rerun that one point (deterministic, same result):

    python3 plot_dens_movie.py --p0 5.0                  # -> p0_grid/p0_5.0/dens_movie.mp4
    python3 1_run_p0_scan_local.py --only 5.0 --force    # (if its densities.npz was deleted)
    python3 plot_dens_movie.py --p0 5.0 --tmax 3000 --rmin -40 --rmax 40

--p0 picks the finished point (a p0_grid/p0_*/ with densities.npz) nearest the value given.  By
default the movie covers the whole run; with the same run_time_factor the DVR and MASH runs end at
the same time (both use t_final = run_time_factor * m |x0| / p0), so the movies are comparable
(--tmax T stops earlier).  The x window is fixed (--rmin/--rmax, default [-15, 15] as
plot_rho_movie.py); weight outside it is reported, so clipping is never silent.
"""

import os
import argparse

import numpy as np
import yaml

import glob

from run_dvr_point import point_name, grid_path

script_dir = os.path.dirname(os.path.abspath(__file__))

ap = argparse.ArgumentParser(description="Animate the exact adiabatic nuclear densities of one p0.")
ap.add_argument("--p0", type=float, required=True,
                help="incident momentum: the nearest finished point with densities.npz is used")
ap.add_argument("--config", default=None)
ap.add_argument("--rmin", type=float, default=-15.0)
ap.add_argument("--rmax", type=float, default=15.0)
ap.add_argument("--tmax", type=float, default=None,
                help="last time shown in a.u. (default / 0: the whole run)")
ap.add_argument("--frames", type=int, default=300, help="target movie frames")
ap.add_argument("--fps", type=int, default=30)
ap.add_argument("--out", default=None, help="output .mp4 (default: <point dir>/dens_movie.mp4)")
a = ap.parse_args()

with open(a.config or os.path.join(script_dir, "config.yaml")) as f:
    cfg = yaml.safe_load(f)
have = sorted(glob.glob(os.path.join(grid_path(cfg), "p0_*", "densities.npz")))
if not have:
    raise SystemExit(f"[dens] no p0_*/densities.npz under {grid_path(cfg)} (deleted by 2_collect.py "
                     f"--delete-raw?)\n[dens] rerun the point:  python3 run_dvr_point.py --p0 {a.p0!r}")
vals = np.array([float(os.path.basename(os.path.dirname(h))[3:]) for h in have])
k = int(np.argmin(np.abs(vals - a.p0)))
path = have[k]
point = os.path.dirname(path)
if point_name(vals[k]) != point_name(a.p0):
    print(f"[dens] no densities for p0 = {a.p0:g}; using the nearest, p0 = {vals[k]:g}   (make p0 = "
          f"{a.p0:g} with: python3 run_dvr_point.py --p0 {a.p0!r})")

d = np.load(path)
t, x = d["t"], d["x"]
rho_g, rho_e = d["rho_g"].astype(np.float64), d["rho_e"].astype(np.float64)
dx = x[1] - x[0]
c_m, c_p = rho_g.sum(axis=1) * dx, rho_e.sum(axis=1) * dx

tmax = t[-1] if a.tmax is None or a.tmax <= 0 else a.tmax
nt = int(np.searchsorted(t, tmax, side="right"))
fidx = np.unique(np.linspace(0, nt - 1, min(a.frames, nt)).astype(int))
win = (x >= a.rmin) & (x <= a.rmax)
xw = x[win]
G, E = rho_g[fidx][:, win], rho_e[fidx][:, win]
lost = 1.0 - (G.sum(axis=1) + E.sum(axis=1)) * dx / (c_m + c_p)[fidx]

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

plt.rcParams.update({
    'figure.dpi': 150, 'axes.linewidth': 1.5,
    'axes.labelsize': 15, 'axes.titlesize': 13,
    'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.top': True, 'ytick.right': True, 'legend.frameon': False,
})
fig, ax = plt.subplots(figsize=(7.4, 5.0))
ax.set_xlim(a.rmin, a.rmax)
ymax = max(G.max(), E.max())
ax.set_ylim(0.0, ymax * 1.08 if ymax > 0 else 1.0)
ax.set_xlabel(r'$R$  (a.u.)')
ax.set_ylabel(r'$|\psi_{g/e}(R,t)|^2$  (a.u.)')
ax.grid(True, color='k', alpha=0.08, lw=0.7)

# ground drawn first so the (smaller) excited density stays visible on top, as in plot_rho_movie.py
fill = {}
line_g, = ax.plot(xw, G[0], color='#d62728', lw=1.0)
line_e, = ax.plot(xw, E[0], color='#2a78d6', lw=1.0)
ax.fill_between([], [], color='#d62728', alpha=0.65, lw=0, label=r'$|\psi_g|^2$  (lower adiabat)')
ax.fill_between([], [], color='#2a78d6', alpha=0.65, lw=0, label=r'$|\psi_e|^2$  (upper adiabat)')
ax.legend(loc='upper left')
title = ax.set_title("")


def update(fi):
    for key in ('g', 'e'):
        if key in fill:
            fill[key].remove()
    fill['g'] = ax.fill_between(xw, G[fi], color='#d62728', alpha=0.65, lw=0)
    fill['e'] = ax.fill_between(xw, E[fi], color='#2a78d6', alpha=0.65, lw=0)
    line_g.set_ydata(G[fi]); line_e.set_ydata(E[fi])
    f = fidx[fi]
    title.set_text(f"t = {t[f]:.1f} a.u.    $c_+$ = {c_p[f]:.3f},  $c_-$ = {c_m[f]:.3f}    "
                   f"(exact, $p_0$ = {float(d['p0']):.3g})")
    return [line_g, line_e, title, fill['g'], fill['e']]


out = a.out or os.path.join(point, "dens_movie.mp4")
ani = FuncAnimation(fig, update, frames=fidx.size, blit=False)
ani.save(out, writer=FFMpegWriter(fps=a.fps))
plt.close(fig)

print(f"[dens] {path}: p0={float(d['p0']):g} sigma={float(d['sigma']):g}  {fidx.size} frames over "
      f"t = 0..{t[fidx[-1]]:.1f} a.u. (run ends at {t[-1]:.1f})  window [{a.rmin:g}, {a.rmax:g}] "
      f"(dx={dx:.4f})")
print(f"[dens] c_+ : {c_p[0]:.4f} -> {c_p[fidx[-1]]:.4f}    c_- : {c_m[0]:.4f} -> {c_m[fidx[-1]]:.4f}")
if "norm" in d.files:
    print(f"[dens] check vs the run's norm: max|c_- + c_+ - norm| = {np.abs(c_m + c_p - d['norm']).max():.2e}")
if lost.max() > 1e-4:
    w = int(np.argmax(lost))
    print(f"[dens] WARNING: the [{a.rmin:g}, {a.rmax:g}] window clips weight; worst frame "
          f"t={t[fidx[w]]:.1f} loses {100*lost[w]:.2f}% ({100*lost.mean():.2f}% on average). "
          f"Widen with --rmin/--rmax to capture it all.")
print(f"[dens] wrote {out}")
