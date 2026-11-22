"""
Analysis of the DVR momentum scan, the counterpart of ../../scan_transmission.py for the MASH dirs:
for every point p0_grid/p0_<p0>/ it

  1. computes the channel-resolved populations C_Rg, C_Tg, C_Re, C_Te(t) (and Cm = C_Rg + C_Tg,
     Cp = C_Re + C_Te, the norm) from the raw densities.npz the run wrote,
         C_Rg(t) = int rho_g(x,t) h(x_split - x) dx,  C_Tg(t) = int rho_g h(x - x_split) dx,  ...
     h = 1 / 0 / 1/2 exactly at x_split (a grid point of the scan's box),
  2. saves them to transmission.npz (time series only, ~100 kB) and draws analyze_transmission.png,
  3. with --delete-raw, deletes densities.npz once transmission.npz is safely written,

and then writes scan_transmission.txt in this directory over ALL points analysed so far (final values
C_c(t_final); *_err = 0, exact dynamics has no M) -- the layout ../figure.py reads.  A point whose
densities.npz is gone is read back from its transmission.npz, so batches accumulate with no separate
merge step:

    python3 2_collect.py                      # analyse every point (raw data where present)
    python3 2_collect.py --delete-raw         # ... and delete each densities.npz afterwards
    python3 2_collect.py --saved-only         # only rebuild scan_transmission.txt from transmission.npz
    python3 2_collect.py --replot --tmax 5000 # redraw every point's png (from transmission.npz too)
    python3 2_collect.py --config config_smoke.yaml

Per batch (as the MASH Momentum-Scan dirs): 1_run_p0_scan_local.py, then 2_collect.py --delete-raw,
then the next batch.  Keep densities.npz of a point you want a movie of (plot_dens_movie.py), or
make the movie before deleting.  Points of the current config batch without a result are reported.
"""

import os
import glob
import json
import argparse

import numpy as np
import yaml

from run_dvr_point import p0_grid, point_name, grid_path

script_dir = os.path.dirname(os.path.abspath(__file__))

RAW, SAVED, PNG = "densities.npz", "transmission.npz", "analyze_transmission.png"
SERIES = ("CRg", "CTg", "CRe", "CTe", "Cm", "Cp", "norm")
META = ("p0", "mass", "x0", "sigma", "x_split", "dt", "nsteps", "run_time_factor")

ap = argparse.ArgumentParser(description="Per-point C_c(t) of the DVR scan -> transmission.npz, png, "
                                         "scan_transmission.txt")
ap.add_argument("--config", default="config.yaml", help="config of the scan (for grid_dir and the batch)")
ap.add_argument("--out", default="scan_transmission.txt", help="summary file name in this directory")
ap.add_argument("--delete-raw", action="store_true",
                help="delete each point's densities.npz after its transmission.npz is written")
ap.add_argument("--saved-only", action="store_true",
                help="do not touch densities.npz: summarise from transmission.npz only")
ap.add_argument("--replot", action="store_true",
                help="redraw the png of points read from transmission.npz (else only a missing png)")
ap.add_argument("--tmax", type=float, default=None, help="truncate the PLOTS to t <= tmax")
args = ap.parse_args()
cfg_path = args.config if os.path.isabs(args.config) else os.path.join(script_dir, args.config)
with open(cfg_path) as f:
    cfg = yaml.safe_load(f)
grid_dir = grid_path(cfg)


# ------------------------------------------------------------------ per point
def compute(raw_path):
    """C_c(t) etc. from a densities.npz."""
    with np.load(raw_path) as z:
        x, xs = z["x"].astype(np.float64), float(z["x_split"])
        dx = x[1] - x[0]
        tol = 1e-3 * dx                      # libra's float32 grid: the x_split point is ~1e-6 off
        hT = np.where(x > xs + tol, 1.0, np.where(x < xs - tol, 0.0, 0.5))
        hR = 1.0 - hT
        r = {"time": z["t"].astype(np.float64)}
        for lab, key in (("g", "rho_g"), ("e", "rho_e")):
            rho = z[key].astype(np.float64)                     # [nt, npts]
            r[f"CR{lab}"] = rho @ hR * dx
            r[f"CT{lab}"] = rho @ hT * dx
        r["Cm"], r["Cp"] = r["CRg"] + r["CTg"], r["CRe"] + r["CTe"]
        r["norm"] = r["Cm"] + r["Cp"]
        for k in META:
            r[k] = float(z[k]) if k in z.files else np.nan
        r["npts"] = int(x.size)
        r["dx"] = float(dx)
        r["x_min"], r["x_max"] = float(x[0]), float(x[-1])
        r["on_grid"] = bool(np.abs(x - xs).min() <= tol)
    return r


def save(path, r):
    tmp = path + ".tmp.npz"                  # written whole, then renamed: never a half file
    np.savez(tmp, time=r["time"], **{k: r[k] for k in SERIES},
             **{k: np.asarray(r[k]) for k in META + ("npts", "dx", "x_min", "x_max", "on_grid")})
    os.replace(tmp, path)


def load(path):
    with np.load(path) as z:
        r = {k: z[k] for k in z.files}
    for k in META + ("npts", "dx", "x_min", "x_max", "on_grid"):
        if k in r:
            r[k] = r[k].item()
    return r


def plot(path, r, diag):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = r["time"]
    m = t <= args.tmax if args.tmax is not None else np.ones_like(t, bool)
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    for key, col, lab in (("CRg", "C0", r"$C_{Rg}(t)$"), ("CTg", "C2", r"$C_{Tg}(t)$"),
                          ("CRe", "C1", r"$C_{Re}(t)$"), ("CTe", "C3", r"$C_{Te}(t)$")):
        ax.plot(t[m], r[key][m], color=col, lw=2, label=lab)
    ax.plot(t[m], r["norm"][m], color="0.5", lw=1, ls="--", label="norm")
    ax.set_xlabel("t (a.u.)")
    ax.set_ylabel(r"$C(t)$")
    ax.set_ylim(-0.03, 1.05)
    ax.legend(frameon=False, ncol=3, fontsize=10)
    ax.set_title(f"exact (DVR), $p_0$ = {r['p0']:g}:  run_time_factor {r['run_time_factor']:g}, "
                 f"{r['npts']} points, dx = {r['dx']:.3g}" + (f"\n{diag}" if diag else ""), fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def diagnostics(point):
    p = os.path.join(point, "result.json")
    if not os.path.isfile(p):
        return {}, ""
    with open(p) as f:
        res = json.load(f)
    return res, (f"norm drift {res['max_norm_drift']:.1e}, edge pop {res['max_edge_population']:.1e}, "
                 f"high-k pop {res['high_k_population_final']:.1e}")


rows, failed = [], []
points = sorted(glob.glob(os.path.join(grid_dir, "p0_*/")),
                key=lambda d: float(os.path.basename(os.path.normpath(d))[3:]))
for d in points:
    point = os.path.normpath(d)
    name = os.path.basename(point)
    raw, saved, png = (os.path.join(point, f) for f in (RAW, SAVED, PNG))
    res, diag = diagnostics(point)
    try:
        if os.path.isfile(raw) and not args.saved_only:
            if not res:
                print(f"[collect] {name}: densities.npz but no result.json (run still going or crashed) "
                      f"- skipped")
                continue
            r = compute(raw)
            save(saved, r)
            plot(png, r, diag)
            src = "raw"
            if args.delete_raw:
                os.remove(raw)
                src = "raw, deleted"
        elif os.path.isfile(saved):
            r = load(saved)
            if args.replot or not os.path.isfile(png):
                plot(png, r, diag)
            src = "saved"
        else:
            if res or os.listdir(point):
                print(f"[collect] {name}: no {RAW} or {SAVED} - not finished / not run")
            continue
    except Exception as e:                     # one broken point must not stop the summary
        print(f"[collect] {name}: FAILED ({e})")
        failed.append(name)
        continue
    rows.append((r, res, src))
    print(f"[collect] {name:12s} ({src:12s})  " + "  ".join(f"{k}={r[k][-1]:.5f}" for k in ("CRg", "CTg", "CRe", "CTe"))
          + f"  norm={r['norm'][-1]:.5f}  t_final={r['time'][-1]:g}  factor={r['run_time_factor']:g}"
          + ("" if r.get("on_grid", True) else "  [x_split not a grid point]"), flush=True)

done = {point_name(r["p0"]) for r, _, _ in rows}
missing = [float(p) for p in p0_grid(cfg) if point_name(p) not in done]
if missing:
    print(f"\nWARNING: {len(missing)} point(s) of the current batch have no result yet: {missing}")
    print("         run: python3 1_run_p0_scan_local.py")
if not rows:
    raise SystemExit("no analysed points - run 1_run_p0_scan_local.py first")

edge = [r["p0"] for r, res, _ in rows if res.get("max_edge_population", 0.0) > 1e-4]
if edge:
    print(f"\nWARNING: wavepacket reaches the grid boundary at p0 = {edge} - raise grid.n_sigma / "
          f"grid.edge_frac and rerun those points")


# ---------------- scan_transmission.txt (same layout as ../../scan_transmission.py) ----------------
def _fmt_list(xs, fmt):
    return "[" + ", ".join("nan" if not np.isfinite(x) else format(x, fmt) for x in xs) + "]"


label = os.path.basename(script_dir)
lines = [f"# {label}  (exact quantum dynamics: final C_c(t_final); no M -- exact dynamics has no "
         f"time-dependent weights)",
         f"# {'nucP_init':>9s} {'t_final':>9s} {'factor':>6s}" + "".join(f" {k:>11s}" for k in ("CRg", "CTg", "CRe", "CTe"))
         + f" {'norm':>9s}"]
for r, _, _ in rows:
    lines.append(f"  {r['p0']:9g} {r['time'][-1]:9g} {r['run_time_factor']:6g}"
                 + "".join(f" {r[k][-1]:11.6f}" for k in ("CRg", "CTg", "CRe", "CTe")) + f" {r['norm'][-1]:9.6f}")
lines += ["", "nucP_init = " + _fmt_list([r["p0"] for r, _, _ in rows], "g")]
for k in ("CRg", "CTg", "CRe", "CTe"):
    lines.append(f"{k} = " + _fmt_list([r[k][-1] for r, _, _ in rows], ".6f"))
    lines.append(f"{k}_err = " + _fmt_list([0.0 for _ in rows], "g"))
out = args.out if os.path.isabs(args.out) else os.path.join(script_dir, args.out)
with open(out, "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"\nwrote {out}  ({len(rows)} points)" + (f";  FAILED: {failed}" if failed else ""))
print("next: python ../figure.py   (exact + MASH-Classical + MASH-Quantum)")
