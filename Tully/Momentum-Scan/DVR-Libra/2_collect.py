"""
Collect the per-point result.json files into one table and plot the four channel-resolved
populations at the end of each run, C_Rg, C_Tg, C_Re, C_Te, as a function of the incident
momentum p0 (same observables as MASH-RPMD_Tully/analyze_tully.py).

    python3 2_collect.py                            # config.yaml
    python3 2_collect.py --config config_smoke.yaml  # results of a test scan

Writes p0_grid/results.csv and p0_grid/scan.png. Points that have not finished yet are reported
as missing rather than silently omitted, so a partial scan is obvious.
"""

import os
import csv
import argparse
import json

import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from run_dvr_point import p0_grid

script_dir = os.path.dirname(os.path.abspath(__file__))

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="config.yaml", help="same config the scan was run with")
args = ap.parse_args()
cfg_path = args.config if os.path.isabs(args.config) else os.path.join(script_dir, args.config)
with open(cfg_path) as f:
    cfg = yaml.safe_load(f)
grid_dir = os.path.join(script_dir, cfg.get("runtime", {}).get("grid_dir", "p0_grid"))

p0_ar = p0_grid(cfg)

rows, missing = [], []
for i in range(len(p0_ar)):
    p = os.path.join(grid_dir, f"val{i}", "result.json")
    if not os.path.exists(p):
        missing.append(i)
        continue
    with open(p) as f:
        rows.append(json.load(f))

if missing:
    print(f"WARNING: {len(missing)} point(s) have no result.json yet: {missing}")
    print("         re-run: python3 1_run_p0_scan_local.py")
if not rows:
    raise SystemExit("no results found - run 1_run_p0_scan_local.py first")

rows.sort(key=lambda r: r["p0"])

csv_path = os.path.join(grid_dir, "results.csv")
fields = ["idx", "p0", "E_total", "CRg_final", "CTg_final", "CRe_final", "CTe_final",
          "norm_final", "max_norm_drift", "max_energy_drift", "max_edge_population",
          "high_k_population_final", "t_cross", "t_cross_free_estimate", "t_final", "nsteps",
          "dt", "npts", "t_dyn_s"]
with open(csv_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
    w.writeheader()
    w.writerows(rows)
print(f"wrote {csv_path}  ({len(rows)} points)")

p0 = np.array([r["p0"] for r in rows])

plt.rcParams.update({
    'figure.figsize': (6.0, 4.0),
    'figure.dpi': 150,
    'figure.facecolor': 'white',
    'figure.edgecolor': 'white',
    'lines.linewidth': 2,
    'axes.linewidth': 3,
    'axes.labelsize': 15,
    'axes.titlesize': 15,
    'xtick.direction': 'in',
    'xtick.top': True,
    'ytick.direction': 'in',
    'ytick.right': True,
    'xtick.major.width': 1.5,
    'ytick.major.width': 1.5,
    'xtick.major.size': 4,
    'ytick.major.size': 4,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'legend.frameon': False,
})

fig, (ax1) = plt.subplots(1, 1, figsize=(6, 4))
ax1.set_ylabel('$C(t_\\mathrm{final})$')
ax1.set_xlabel('$p_0$ (a.u.)')
# same colours as MASH-RPMD_Tully/analyze_tully.py
for key, col, lab in (("CRg_final", "C0", "$C_{Rg}$"), ("CTg_final", "C2", "$C_{Tg}$"),
                      ("CRe_final", "C1", "$C_{Re}$"), ("CTe_final", "C3", "$C_{Te}$")):
    ax1.plot(p0, [r[key] for r in rows], "o-", markersize=4, lw=1.5, color=col, label=lab)
ax1.set_ylim(-0.03, 1.03)
ax1.legend(fontsize=10, loc='best', handletextpad=0.4)
fig.tight_layout()
png = os.path.join(grid_dir, "scan.png")
fig.savefig(png)
print(f"wrote {png}")

print(f"\n{'p0':>7} {'C_Rg':>9} {'C_Tg':>9} {'C_Re':>9} {'C_Te':>9} {'norm':>9} "
      f"{'edge pop':>9} {'t_final':>9} {'t_dyn':>7}")
for r in rows:
    print(f"{r['p0']:>7.3f} {r['CRg_final']:>9.5f} {r['CTg_final']:>9.5f} {r['CRe_final']:>9.5f} "
          f"{r['CTe_final']:>9.5f} {r['norm_final']:>9.5f} {r['max_edge_population']:>9.1e} "
          f"{r['t_final']:>9.0f} {r['t_dyn_s']:>6.1f}s"
          + ("   [t_cross: free-particle estimate]" if r.get("t_cross_free_estimate") else ""))

edge = [r["idx"] for r in rows if r["max_edge_population"] > 1e-4]
if edge:
    print(f"\nWARNING: wavepacket reaches the grid boundary at point(s) {edge} - enlarge the grid")
