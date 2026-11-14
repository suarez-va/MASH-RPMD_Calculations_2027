"""
Collect the HEOM scan, the way ../../../scan_rates.py does for the RP_MASH method dirs: for every
grid point delta_grid/val<idx>/ plot the raw dynamics and compute the rate, then save the rates of
the whole scan to scan_rates.txt in this directory.

    python3 2_collect.py                            # config.yaml
    python3 2_collect.py --config config_smoke.yaml  # results of a test scan

Writes
    delta_grid/val<idx>/dynamics.png   the raw data of each point (P2, coherence, k(t), K2 + fit)
    delta_grid/results.csv             every result.json field, one row per point
    scan_rates.txt                     log10_delta / rate / rate_err arrays, read by ../figure.py

Points that have not finished yet are reported as missing rather than silently omitted, so a
partial scan is obvious. The comparison figure (HEOM vs BOMD / MASH and the reference rate theories)
is no longer made here: run  python3 ../figure.py  afterwards.

Two HEOM rates per point: k_fit, the slope of a line fit of K2(t) = -P2eq ln|1 - P2/P2eq| over
[fitmin, fitmax] (config `rate:` block) - this is `rate` in scan_rates.txt - and k_HEOM, the last
point of k(t) (`rate_last`).
    python3 2_collect.py --fitmin 5 --fitmax 15   # refit from the saved trajectories, no HEOM rerun
    python3 2_collect.py --no-plots               # only the numbers, skip the per-point PNGs
    python3 2_collect.py --plot-fits              # also write fits.png: K2(t) + fit for every point
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

from run_heom_point import log10_delta_grid, fit_window, refit_rows, plot_fits, k2_from_p2

script_dir = os.path.dirname(os.path.abspath(__file__))

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="config.yaml", help="same config the scan was run with")
ap.add_argument("--fitmin", type=float, default=None, help="K2 fit window start, t/(beta*hbar) (default: config rate.fitmin)")
ap.add_argument("--fitmax", type=float, default=None, help="K2 fit window end, t/(beta*hbar) (default: config rate.fitmax)")
ap.add_argument("--no-plots", action="store_true", help="do not write the per-point dynamics.png")
ap.add_argument("--plot-fits", action="store_true", help="also write fits.png (K2(t) and its fit, per point)")
ap.add_argument("--out", default="scan_rates.txt", help="summary file name in this directory")
args = ap.parse_args()
cfg_path = args.config if os.path.isabs(args.config) else os.path.join(script_dir, args.config)
with open(cfg_path) as f:
    cfg = yaml.safe_load(f)
grid_dir = os.path.join(script_dir, cfg.get("runtime", {}).get("grid_dir", "delta_grid"))

log10_delta_ar = log10_delta_grid(cfg)

rows, missing = [], []
for i in range(len(log10_delta_ar)):
    p = os.path.join(grid_dir, f"val{i}", "result.json")
    if not os.path.exists(p):
        missing.append(i)
        continue
    with open(p) as f:
        rows.append(json.load(f))

if missing:
    print(f"WARNING: {len(missing)} point(s) have no result.json yet: {missing}")
    print("         re-run: python3 1_run_delta_scan_local.py")
if not rows:
    raise SystemExit("no results found - run 1_run_delta_scan_local.py first")

rows.sort(key=lambda r: r["log10_Delta"])

# K2 line-fit rate: reuse result.json's fit if it used this window, else refit from the trajectory
fitmin, fitmax = fit_window(cfg)
fitmin = args.fitmin if args.fitmin is not None else fitmin
fitmax = args.fitmax if args.fitmax is not None else fitmax
nofit = refit_rows(rows, grid_dir, fitmin, fitmax)
if nofit:
    print(f"WARNING: no saved trajectory for point(s) {nofit} - no K2 fit for them")


# ---------------- per-point plot of the raw data ----------------
# Same figure as HEOM_Calculations_2026/Single-Point/2_plot.py. The scan's dynamics.npz only holds
# the trajectory, so the point's parameters and its fit come from the result.json row.
def plot_point(valdir, row, fitmin, fitmax):
    """Write <valdir>/dynamics.png; returns its path, or None if the point has no dynamics.npz."""
    npz_path = os.path.join(valdir, "dynamics.npz")
    if not os.path.exists(npz_path):
        return None
    d = np.load(npz_path)
    beta = float(d["beta"])
    t = d["t_beta"]
    rho = d["rho"]
    kt_b = d["kt"] * beta                          # beta*hbar*k(t)
    k_final_b = float(row["k_HEOM"]) * beta        # last point of k(t)
    P2eq = float(d["P2eq"])
    P2 = rho[:, 1, 1].real
    K2 = k2_from_p2(P2, P2eq)
    m, b, r = row.get("fit_slope"), row.get("fit_intercept"), row.get("fit_r")

    fig, ((ax1, ax2), (ax4, ax3)) = plt.subplots(2, 2, figsize=(12, 8.5), sharex=True)
    fig.suptitle(f"$\\beta\\epsilon$ = {float(row['beta_epsilon']):g},   "
                 f"$\\log_{{10}}(\\beta\\Delta)$ = {float(row['log10_Delta']):g},   "
                 f"KK = {int(row['KK'])}, LL = {int(row['LL'])}", fontsize=12)

    ax1.plot(t, P2, color="tab:red", label="$P_2(t)$")
    ax1.set_ylabel("population")
    ax1.legend(loc="best")

    ax2.plot(t, rho[:, 0, 1].real, color="tab:purple", label="Re $\\rho_{12}(t)$")
    ax2.plot(t, rho[:, 0, 1].imag, color="tab:green", label="Im $\\rho_{12}(t)$")
    ax2.set_ylabel("coherence")
    ax2.legend(loc="best")

    # bottom-left: k(t) and its last point
    ax4.plot(t, kt_b, color="k", lw=1.5, label="$k(t)$")
    ax4.axhline(k_final_b, color="gray", ls="--", lw=1, label=f"$k$ final = {k_final_b:.3e}")
    ax4.axhline(0.0, color="k", ls="--", lw=1)
    if k_final_b > 0:
        ax4.set_ylim(0.0 - 0.1*k_final_b, 5.1*k_final_b)
    else:
        # the limits above assume k_final > 0; otherwise scale to the back half of k(t)
        tail = kt_b[len(kt_b) // 2:]
        tail = tail[np.isfinite(tail)]
        if tail.size:
            lo, hi = np.percentile(tail, [2.0, 98.0])
            pad = 0.2 * max(hi - lo, 1e-12)
            ax4.set_ylim(lo - pad, hi + pad)
    ax4.set_ylabel("$\\beta\\hbar\\, k(t)$")
    ax4.set_xlabel("$t\\, /\\, \\beta\\hbar$")
    ax4.legend(loc="best")

    # bottom-right: K2(t) and its line fit
    ax3.axvspan(fitmin, fitmax, color="0.92", zorder=0)
    ax3.axhline(0.0, color="0.7", lw=0.8, ls="--")
    ax3.plot(t, K2, color="C0", lw=1.6, zorder=4,
             label=r"$K_2(t)=-P_2^{\mathrm{eq}}\ln|1-P_2/P_2^{\mathrm{eq}}|$")
    if m is not None:
        tf = np.array([fitmin, min(fitmax, t[-1])])
        ax3.plot(tf, m * tf + b, color="k", lw=2.0, ls="--", zorder=6,
                 label=f"fit [{fitmin:g},{fitmax:g}]:  m={m:.4g}, b={b:.4g}, r={r:.4f}")

        def _log(x):
            return f"{np.log10(x):.4f}" if x > 0 else "n/a"
        ax3.text(0.97, 0.05,
                 f"$\\beta\\hbar k$ (fit) = {m:.4e}   (log$_{{10}}$ {_log(m)})\n"
                 f"$\\beta\\hbar k_\\mathrm{{HEOM}}$ (last pt) = {k_final_b:.4e}   "
                 f"(log$_{{10}}$ {_log(k_final_b)})",
                 transform=ax3.transAxes, fontsize=8, va="bottom", ha="right")
    ax3.set_ylabel("$K_2(t)$")
    ax3.set_xlabel("$t\\, /\\, \\beta\\hbar$")
    ax3.legend(loc="upper left", fontsize=8)

    fig.tight_layout()
    png = os.path.join(valdir, "dynamics.png")
    fig.savefig(png)
    plt.close(fig)
    return png


if not args.no_plots:
    plt.rcParams.update({
        'figure.dpi': 150,
        'figure.facecolor': 'white',
        'figure.edgecolor': 'white',
        'lines.linewidth': 2,
        'axes.linewidth': 3,
        'axes.labelsize': 15,
        'axes.titlesize': 13,
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
    noplot = []
    for r in rows:
        if plot_point(os.path.join(grid_dir, f"val{r['idx']}"), r, fitmin, fitmax) is None:
            noplot.append(r["idx"])
    print(f"wrote {len(rows) - len(noplot)} x {os.path.join(grid_dir, 'val<idx>', 'dynamics.png')}")
    if noplot:
        print(f"WARNING: no dynamics.npz for point(s) {noplot} - no dynamics.png for them")

# ---------------- results.csv ----------------
csv_path = os.path.join(grid_dir, "results.csv")
fields = ["idx", "log10_Delta", "beta_Delta", "beta_epsilon", "k_HEOM", "log10_beta_k_HEOM",
          "k_fit", "log10_beta_k_fit", "fit_r", "fit_npts", "fitmin", "fitmax",
          "P2eq", "P2t_final", "nn_tot", "KK", "LL", "t_therm_s", "t_dyn_s"]
with open(csv_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
    w.writeheader()
    w.writerows(rows)
print(f"wrote {csv_path}  ({len(rows)} points)")

print(f"\nK2 fit window: t/(beta*hbar) = [{fitmin:g}, {fitmax:g}]")
print(f"\n{'log10(bD)':>9} {'log10(bk) fit':>14} {'fit r':>7} {'log10(bk) last':>15} "
      f"{'P2t_final':>11} {'P2eq':>9} {'t_dyn':>8}")
for r in rows:
    vf, v = r["log10_beta_k_fit"], r["log10_beta_k_HEOM"]
    print(f"{r['log10_Delta']:>9.2f} {('%.5f' % vf) if vf is not None else 'n/a':>14} "
          f"{('%.4f' % r['fit_r']) if r.get('fit_r') is not None else 'n/a':>7} "
          f"{('%.5f' % v) if v is not None else 'k<=0':>15} "
          f"{r['P2t_final']:>11.6f} {r['P2eq']:>9.6f} {r['t_dyn_s']:>7.0f}s")


# ---------------- scan_rates.txt (same layout as ../../../scan_rates.py) ----------------
def _num(x):
    return float("nan") if x is None else float(x)


def _fmt_list(xs, fmt):
    return "[" + ", ".join("nan" if not np.isfinite(x) else format(x, fmt) for x in xs) + "]"


label = os.path.basename(script_dir)
table = []
for r in rows:
    kf, kl = _num(r.get("k_fit")), _num(r.get("k_HEOM"))
    warn = []
    if not np.isfinite(kf):
        warn.append("nofit")
    elif kf <= 0:
        warn.append("k_fit<=0")
    if not kl > 0:
        warn.append("k_last<=0")
    table.append((float(r["log10_Delta"]), kf, _num(r.get("fit_r")), kl, ",".join(warn)))

betas = sorted({float(r["beta"]) for r in rows})
if len(betas) > 1:
    print(f"WARNING: the points do not share one beta: {betas}")
lines = [f"# {label}  (HEOM, K2 fit [{fitmin:g},{fitmax:g}], KK={rows[0]['KK']}, LL={rows[0]['LL']})",
         f"# {'log10_delta':>11s} {'k_fit':>14s} {'fit_r':>14s} {'k_last':>14s}  warn"]
for ld, kf, fr, kl, w in table:
    lines.append(f"{ld:13.10g} {kf:14.6e} {fr:14.6f} {kl:14.6e}  {w}")
lines += ["",
          f"beta = {betas[0]:.10g}",
          "log10_delta = " + _fmt_list([t[0] for t in table], ".10g"),
          "rate = " + _fmt_list([t[1] for t in table], ".6e"),
          "rate_err = " + _fmt_list([0.0 for _ in table], "g"),
          "rate_last = " + _fmt_list([t[3] for t in table], ".6e")]
out = os.path.join(script_dir, args.out)
with open(out, "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"\nwrote {out}")

if args.plot_fits:
    print(f"wrote {plot_fits(rows, grid_dir, fitmin, fitmax, 'log10_Delta', 'log10(beta*Delta) = {:.2f}')}")

print("\nnext: python3 ../figure.py")
