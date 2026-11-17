"""
The final figure of the epsilon (driving-force) scan: log10(beta*hbar*k) vs epsilon/lambda at fixed
Delta -- the Marcus parabola, HEOM, MASH and MASH-RPMD -- in the style of
HEOM_Calculations_2026/Low-Friction/Epsilon-Scan/2_collect.py.

    python3 figure.py                                   # -> figure.png
    python3 figure.py --xlim -0.05 2.55 --ylim -5.25 -2.75
    python3 figure.py --root /some/copy/of/Epsilon-Scan --out test.png

It only reads each method's scan_rates.txt (the `epsil`, `rate`, `rate_err` arrays), so make those
first:

    HEOM-Libra/                     python3 2_collect.py
    MASH-Classical/ MASH-Quantum/   python ../../../scan_rates.py */ --ncores 16

A method directory without a scan_rates.txt is skipped with a warning. The theory curves (Marcus;
Zusman and the quantum nonadiabatic rate are computed but commented out in the plot) take Omega,
Gamma, Lambda, log10_Delta from HEOM-Libra/config.yaml.
"""

import os
import re
import argparse
import importlib.util

import numpy as np
from scipy.special import iv
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))

HEOM_DIR = "HEOM-Libra"
# (directory, legend label, colour) of the trajectory methods, drawn as points with error bars
METHODS = [("MASH-Classical", "MASH",      "r"),
           ("MASH-Quantum",   "MASH-RPMD", "b")]

ap = argparse.ArgumentParser()
ap.add_argument("--root", default=script_dir, help="directory holding the method dirs (default: here)")
ap.add_argument("--rates", default="scan_rates.txt", help="summary file name in each method dir")
ap.add_argument("--xlim", type=float, nargs=2, default=None, metavar=("LO", "HI"),
                help="x range in epsilon/lambda (default: spans all the data)")
ap.add_argument("--ylim", type=float, nargs=2, default=None, metavar=("LO", "HI"))
ap.add_argument("--out", default="figure.png", help="output png (relative to --root)")
args = ap.parse_args()


def read_scan(path):
    """The `name = [...]` / `name = value` lines of a scan_rates.txt -> {name: array or float}."""
    out = {}
    with open(path) as f:
        for line in f:
            m = re.match(r"^(\w+)\s*=\s*(.+)$", line.strip())
            if not m:
                continue
            name, val = m.group(1), m.group(2).strip()
            if val.startswith("["):
                out[name] = np.array([float(x) for x in val.strip("[]").split(",") if x.strip()])
            else:
                out[name] = float(val)
    return out


def load_scan(method):
    """scan_rates.txt of one method dir, or None (with a warning) if it is not there / unusable."""
    path = os.path.join(args.root, method, args.rates)
    if not os.path.isfile(path):
        print(f"WARNING: {os.path.join(method, args.rates)} not found - skipping {method}")
        return None
    s = read_scan(path)
    if "epsil" not in s or "rate" not in s:
        print(f"WARNING: no epsil / rate arrays in {path} (not an epsilon scan?) - skipping {method}")
        return None
    if "rate_err" not in s:
        s["rate_err"] = np.zeros_like(s["rate"])
    return s


def mash_config(method):
    """CONFIG of an RP_MASH method dir (its beta and lbd)."""
    spec = importlib.util.spec_from_file_location("_figure_cfg_" + method.replace("-", "_"),
                                                  os.path.join(args.root, method, "config.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.CONFIG


# --- HEOM + the theory parameters ---
with open(os.path.join(args.root, HEOM_DIR, "config.yaml")) as f:
    cfg = yaml.safe_load(f)
heom = load_scan(HEOM_DIR)
beta = heom.get("beta", 1.0) if heom is not None else 1.0
Omega = cfg["system"]["Omega"] / beta
Gamma = cfg["system"]["Gamma"] / beta
Lambda = cfg["system"]["Lambda"] / beta
Delta = 10.0 ** cfg["system"]["log10_Delta"] / beta

if heom is not None:
    x_heom = heom["epsil"] / (beta * Lambda)       # epsil is beta*epsilon
    with np.errstate(divide="ignore", invalid="ignore"):
        lkf = np.log10(heom["rate"] * beta)        # K2 line fit
        #lk = np.log10(heom["rate_last"] * beta)   # last point of k(t)

# --- MASH rates ---
points = []                                        # (label, colour, x, y, yerr)
for method, label, color in METHODS:
    s = load_scan(method)
    if s is None:
        continue
    c = mash_config(method)
    b, lbd = float(c["beta"]), float(c["lbd"])
    mk, sk = s["rate"], s["rate_err"]
    ok = np.isfinite(mk) & (mk > 0)
    if not ok.all():
        print(f"WARNING: {method}: dropping {int((~ok).sum())} point(s) with no positive rate")
    if not ok.any():
        print(f"WARNING: {method}: no positive rate at all - skipping {method}")
        continue
    points.append((label, color, s["epsil"][ok] / lbd, np.log10(mk[ok] * b),
                   sk[ok] / (mk[ok] * np.log(10))))

xs = ([x_heom] if heom is not None else []) + [p[2] for p in points]
if not xs:
    raise SystemExit(f"no {args.rates} in any method directory under {args.root} - nothing to plot")

# --- reference rate theories vs epsilon (Delta fixed), on a fine grid for smooth curves ---
xmin = min(x.min() for x in xs)
xmax = max(x.max() for x in xs)
if args.xlim is not None:
    xmin, xmax = min(xmin, args.xlim[0]), max(xmax, args.xlim[1])
Dx = max(xmax - xmin, 0.1)
eps_fine = np.linspace(xmin - 0.1*Dx, xmax + 0.1*Dx, 400) * Lambda
kMT = 2 * np.pi * Delta**2 * np.sqrt(beta / (4 * np.pi * Lambda)) \
      * np.exp(-beta * (Lambda - eps_fine)**2 / (4 * Lambda))
kZUS = kMT / (1 + 4 * np.pi * Delta**2 * Gamma / (Lambda * Omega**2))
kNAD = 2 * np.pi / Omega * Delta**2 * np.exp(0.5 * beta * eps_fine) \
       * np.exp(-Lambda / (Omega * np.tanh(0.5 * beta * Omega))) \
       * iv(eps_fine / Omega, 2 * Lambda * np.exp(-0.5 * beta * Omega) \
       / (Omega * (1 - np.exp(-beta * Omega))))

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
ax1.set_ylabel('$log(k\\beta\\hbar)$')
ax1.set_xlabel('$\\epsilon/\\lambda$')
ax1.set_xlim(args.xlim if args.xlim is not None else [xmin - 0.05*Dx, xmax + 0.05*Dx])
if args.ylim is not None:
    ax1.set_ylim(args.ylim)
ax1.plot(eps_fine / Lambda, np.log10(kMT * beta), color="darkred", lw=1.5, label="Marcus")
#ax1.plot(eps_fine / Lambda, np.log10(kZUS * beta), color="darkorange", lw=1.5, label="Zusman")
#ax1.plot(eps_fine / Lambda, np.log10(kNAD * beta), color="darkgreen", lw=1.5, label="Quantum")
if heom is not None:
    ax1.plot(x_heom, lkf, color="k", lw=1.5, label="HEOM")
    #ax1.plot(x_heom, lk, color="k", lw=1.5, ls="--", label="HEOM (last $k(t)$)")
for label, color, x, y, yerr in points:
    ax1.errorbar(x, y, yerr, fmt='o', markersize=5, linewidth=1.5, color=color, label=label)
ax1.legend(fontsize=9, loc='best', handletextpad=0.2)
fig.tight_layout()
png = args.out if os.path.isabs(args.out) else os.path.join(args.root, args.out)
fig.savefig(png)
print(f"wrote {png}")
