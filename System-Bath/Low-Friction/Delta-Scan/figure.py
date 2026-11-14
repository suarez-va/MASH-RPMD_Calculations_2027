"""
The final figure of the Delta scan: log10(beta*hbar*k) vs log10(beta*Delta) at fixed epsilon, for
every method directory next to this script, alongside the reference rate theories.

    python3 figure.py                     # -> figure.png
    python3 figure.py --xlim -0.575 0.575 # fixed x window
    python3 figure.py --out scan.png

It only reads each method's scan_rates.txt (the `log10_delta`, `rate`, `rate_err` arrays), so make
those first:

    HEOM-Libra/                                    python3 2_collect.py
    BOMD-Classical/ MASH-Classical/ MASH-Quantum/  python ../../scan_rates.py */ --ncores 16

A method directory without a scan_rates.txt is skipped with a warning. The theory curves (Marcus;
Zusman and the quantum nonadiabatic rate are computed but commented out in the plot) take
Omega, Gamma, Lambda, epsilon from HEOM-Libra/config.yaml.
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
METHODS = [("BOMD-Classical", "BOMD",      "g"),
           ("MASH-Classical", "MASH",      "r"),
           ("MASH-Quantum",   "MASH-RPMD", "b")]

ap = argparse.ArgumentParser()
ap.add_argument("--xlim", type=float, nargs=2, default=None, metavar=("LO", "HI"),
                help="x range in log10(beta*Delta) (default: spans all the data)")
ap.add_argument("--rates", default="scan_rates.txt", help="summary file name in each method dir")
ap.add_argument("--out", default="figure.png", help="output png (relative to this directory)")
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
    path = os.path.join(script_dir, method, args.rates)
    if not os.path.isfile(path):
        print(f"WARNING: {os.path.join(method, args.rates)} not found - skipping {method}")
        return None
    s = read_scan(path)
    if "log10_delta" not in s or "rate" not in s:
        print(f"WARNING: no log10_delta / rate arrays in {path} - skipping {method}")
        return None
    if "rate_err" not in s:
        s["rate_err"] = np.zeros_like(s["rate"])
    return s


def config_beta(method):
    """beta of an RP_MASH method dir, from its config.py."""
    spec = importlib.util.spec_from_file_location("_figure_cfg_" + method.replace("-", "_"),
                                                  os.path.join(script_dir, method, "config.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return float(mod.CONFIG["beta"])


# --- HEOM ---
heom = load_scan(HEOM_DIR)
beta = heom.get("beta", 1.0) if heom is not None else 1.0
if heom is not None:
    ld = heom["log10_delta"]                       # already log10(beta*Delta)
    with np.errstate(divide="ignore", invalid="ignore"):
        lkf = np.log10(heom["rate"] * beta)        # K2 line fit
        #lk = np.log10(heom["rate_last"] * beta)   # last point of k(t)

# --- BOMD / MASH rates ---
points = []                                        # (label, colour, x, y, yerr)
for method, label, color in METHODS:
    s = load_scan(method)
    if s is None:
        continue
    b = config_beta(method)
    mk, sk = s["rate"], s["rate_err"]
    ok = np.isfinite(mk) & (mk > 0)
    if not ok.all():
        print(f"WARNING: {method}: dropping {int((~ok).sum())} point(s) with no positive rate")
    points.append((label, color, s["log10_delta"][ok] + np.log10(b), np.log10(mk[ok] * b),
                   sk[ok] / (mk[ok] * np.log(10))))

xs = ([ld] if heom is not None else []) + [p[2] for p in points]
if not xs:
    raise SystemExit("no scan_rates.txt in any method directory - nothing to plot")

# --- reference rate theories vs Delta (epsilon fixed), on a fine grid for smooth curves ---
with open(os.path.join(script_dir, HEOM_DIR, "config.yaml")) as f:
    cfg = yaml.safe_load(f)
Omega = cfg["system"]["Omega"] / beta
Gamma = cfg["system"]["Gamma"] / beta
Lambda = cfg["system"]["Lambda"] / beta
epsilon = cfg["system"]["epsilon"] / beta

ldmin = min(x.min() for x in xs)
ldmax = max(x.max() for x in xs)
if args.xlim is not None:
    ldmin, ldmax = min(ldmin, args.xlim[0]), max(ldmax, args.xlim[1])
Dld = max(ldmax - ldmin, 0.1)
ld_fine = np.linspace(ldmin - 0.1*Dld, ldmax + 0.1*Dld, 400)
Delta_fine = 10.0 ** ld_fine / beta
kMT = 2 * np.pi * Delta_fine**2 * np.sqrt(beta / (4 * np.pi * Lambda)) \
      * np.exp(-beta * (Lambda - epsilon)**2 / (4 * Lambda))
kZUS = kMT / (1 + 4 * np.pi * Delta_fine**2 * Gamma / (Lambda * Omega**2))
kNAD = 2 * np.pi / Omega * Delta_fine**2 * np.exp(0.5 * beta * epsilon) \
       * np.exp(-Lambda / (Omega * np.tanh(0.5 * beta * Omega))) \
       * iv(epsilon / Omega, 2 * Lambda * np.exp(-0.5 * beta * Omega) \
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
ax1.set_xlabel('$\\log_{10}(\\beta\\Delta)$')
ax1.set_xlim(args.xlim if args.xlim is not None else [ldmin - 0.05*Dld, ldmax + 0.05*Dld])
ax1.plot(ld_fine, np.log10(kMT * beta), color="darkred", lw=1.5, label="Marcus")
#ax1.plot(ld_fine, np.log10(kZUS * beta), color="darkorange", lw=1.5, label="Zusman")
#ax1.plot(ld_fine, np.log10(kNAD * beta), color="darkgreen", lw=1.5, label="Quantum")
if heom is not None:
    ax1.plot(ld, lkf, color="k", lw=1.5, label="HEOM")
    #ax1.plot(ld, lk, color="k", lw=1.5, ls="--", label="HEOM (last $k(t)$)")
for label, color, x, y, yerr in points:
    ax1.errorbar(x, y, yerr, fmt='o', markersize=5, linewidth=1.5, color=color, label=label)
ax1.legend(fontsize=9, loc='best', handletextpad=0.2)
fig.tight_layout()
png = args.out if os.path.isabs(args.out) else os.path.join(script_dir, args.out)
fig.savefig(png)
print(f"wrote {png}")
