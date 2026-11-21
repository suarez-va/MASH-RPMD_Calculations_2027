"""
plot_dens_movie.py -- animate the exact ground / excited ADIABATIC nuclear densities of one scan point

    rho_g(x,t) = |psi_g(x,t)|^2      lower adiabat, RED
    rho_e(x,t) = |psi_e(x,t)|^2      upper adiabat, BLUE

into an .mp4.  This is the exact counterpart of ../../../plot_rho_movie.py (MASH: rho_-(R,t) on the
lower adiabat, rho_+(R,t) on the upper), drawn the same way:

    integral rho_g dx = c_-(t) (= Cm in dynamics.npz),   integral rho_e dx = c_+(t) (= Cp)

The scan does not keep the wavefunction densities (dynamics.npz is time series only), so make them
for the point first -- it reruns that one p0, deterministic, same result.json / dynamics.npz:

    python3 run_dvr_point.py --idx 120 --save-dens       # -> p0_grid/val120/densities.npz
    python3 plot_dens_movie.py --idx 120                 # -> p0_grid/val120/dens_movie.mp4
    python3 plot_dens_movie.py --p0 5.0                  # the grid point nearest p0 = 5
    python3 plot_dens_movie.py --idx 120 --tmax 0 --rmin -40 --rmax 40   # the whole DVR run

The DVR runs 4 crossing times, the MASH runs (transmission_scan.n_steps) 2: by default the movie stops
at half the DVR run, so it covers the same time span as the MASH movie (--tmax T sets it, 0 = whole
run).  The x window is fixed (--rmin/--rmax, default [-15, 15] as plot_rho_movie.py); weight outside
it is reported, so clipping is never silent.
"""

import os
import argparse

import numpy as np
import yaml

from run_dvr_point import p0_grid

script_dir = os.path.dirname(os.path.abspath(__file__))

ap = argparse.ArgumentParser(description="Animate the exact adiabatic nuclear densities of one p0.")
g = ap.add_mutually_exclusive_group(required=True)
g.add_argument("--idx", type=int, help="index into the p0 scan grid")
g.add_argument("--p0", type=float, help="incident momentum: the nearest grid point is used")
ap.add_argument("--config", default=None)
ap.add_argument("--rmin", type=float, default=-15.0)
ap.add_argument("--rmax", type=float, default=15.0)
ap.add_argument("--tmax", type=float, default=None,
                help="last time shown in a.u. (default: half the run = the MASH run length; 0 = all)")
ap.add_argument("--frames", type=int, default=300, help="target movie frames")
ap.add_argument("--fps", type=int, default=30)
ap.add_argument("--out", default=None, help="output .mp4 (default: <point dir>/dens_movie.mp4)")
a = ap.parse_args()

with open(a.config or os.path.join(script_dir, "config.yaml")) as f:
    cfg = yaml.safe_load(f)
grid = p0_grid(cfg)
idx = a.idx if a.idx is not None else int(np.argmin(np.abs(grid - a.p0)))
if not 0 <= idx < grid.size:
    raise SystemExit(f"--idx {idx} is outside the grid (0..{grid.size - 1})")
point = os.path.join(script_dir, cfg["runtime"].get("grid_dir", "p0_grid"), f"val{idx}")
path = os.path.join(point, "densities.npz")
if not os.path.isfile(path):
    raise SystemExit(f"[dens] no {path}\n[dens] make it first:  python3 run_dvr_point.py --idx {idx} "
                     f"--save-dens   (p0 = {grid[idx]:g})")

d = np.load(path)
t, x = d["t"], d["x"]
rho_g, rho_e = d["rho_g"].astype(np.float64), d["rho_e"].astype(np.float64)
dx = x[1] - x[0]
c_m, c_p = rho_g.sum(axis=1) * dx, rho_e.sum(axis=1) * dx

tmax = 0.5 * t[-1] if a.tmax is None else (t[-1] if a.tmax <= 0 else a.tmax)
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
if "Cm" in d.files:
    print(f"[dens] check vs dynamics: max|c_- - Cm| = {np.abs(c_m - d['Cm']).max():.2e}  "
          f"max|c_+ - Cp| = {np.abs(c_p - d['Cp']).max():.2e}")
if lost.max() > 1e-4:
    w = int(np.argmax(lost))
    print(f"[dens] WARNING: the [{a.rmin:g}, {a.rmax:g}] window clips weight; worst frame "
          f"t={t[fidx[w]]:.1f} loses {100*lost[w]:.2f}% ({100*lost.mean():.2f}% on average). "
          f"Widen with --rmin/--rmax to capture it all.")
print(f"[dens] wrote {out}")
