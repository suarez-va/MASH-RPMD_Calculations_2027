"""
The final figure of the Tully momentum scan: exact quantum dynamics vs MASH vs MASH-RPMD, as a
function of the incident momentum nucP_init.  Two panels sharing the x axis:

    top     final channel populations C_Rg, C_Tg, C_Re, C_Te
            (exact: lines; MASH methods: points with block standard errors)
    bottom  max over t of |M_c(t)| for the MASH methods -- the largest correction from the
            time-dependent weights in each channel (with the SE at that time)

    python3 figure.py                       # -> figure.png
    python3 figure.py --logm --xlim 2 31    # log y on the M panel, fixed x window
    python3 figure.py --root /some/copy/of/Momentum-Scan --out test.png

It only reads each method's scan_tully.txt, so make those first:

    DVR-Libra/                         python3 2_collect.py
    MASH-Classical/ MASH-Quantum/      python ../../scan_tully.py --ncores 10   (in each dir)

A method directory without a scan_tully.txt is skipped with a warning.
"""

import os
import re
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

script_dir = os.path.dirname(os.path.abspath(__file__))

EXACT = ("DVR-Libra", "Exact")
# (directory, legend label, marker, filled) of the trajectory methods
METHODS = [("MASH-Classical", "MASH",      "o", False),
           ("MASH-Quantum",   "MASH-RPMD", "s", True)]
# channels, colours as analyze_tully.py
CHANNELS = [("Rg", "C0"), ("Tg", "C2"), ("Re", "C1"), ("Te", "C3")]

ap = argparse.ArgumentParser()
ap.add_argument("--root", default=script_dir, help="directory holding the method dirs (default: here)")
ap.add_argument("--rates", default="scan_tully.txt", help="summary file name in each method dir")
ap.add_argument("--xlim", type=float, nargs=2, default=None, metavar=("LO", "HI"))
ap.add_argument("--logm", action="store_true", help="log y axis on the max|M| panel")
ap.add_argument("--out", default="figure.png", help="output png (relative to --root)")
args = ap.parse_args()


def read_scan(path):
    """The `name = [...]` / `name = value` lines of a scan_tully.txt -> {name: array or float}."""
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


def load(method):
    path = os.path.join(args.root, method, args.rates)
    if not os.path.isfile(path):
        print(f"WARNING: {os.path.join(method, args.rates)} not found - skipping {method}")
        return None
    s = read_scan(path)
    if "nucP_init" not in s:
        print(f"WARNING: no nucP_init array in {path} - skipping {method}")
        return None
    order = np.argsort(s["nucP_init"])
    return {k: (v[order] if isinstance(v, np.ndarray) and v.shape == order.shape else v)
            for k, v in s.items()}


exact = load(EXACT[0])
mash = [(lab, mk, fill, s) for d, lab, mk, fill in METHODS if (s := load(d)) is not None]
if exact is None and not mash:
    raise SystemExit(f"no {args.rates} in any method directory under {args.root} - nothing to plot")

plt.rcParams.update({
    'figure.dpi': 150,
    'figure.facecolor': 'white',
    'figure.edgecolor': 'white',
    'lines.linewidth': 2,
    'axes.linewidth': 3,
    'axes.labelsize': 18,
    'axes.titlesize': 18,
    'xtick.direction': 'in',
    'xtick.top': True,
    'ytick.direction': 'in',
    'ytick.right': True,
    'xtick.major.width': 1.5,
    'ytick.major.width': 1.5,
    'xtick.major.size': 5,
    'ytick.major.size': 5,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 13,
    'legend.frameon': False,
})

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True,
                               gridspec_kw={'height_ratios': [3, 2]})

# --- top: final channel populations
for ch, col in CHANNELS:
    if exact is not None and f"C{ch}" in exact:
        ax1.plot(exact["nucP_init"], exact[f"C{ch}"], "-", color=col, lw=2.2)
    for lab, mk, fill, s in mash:
        if f"C{ch}" not in s:
            continue
        ax1.errorbar(s["nucP_init"], s[f"C{ch}"], s.get(f"C{ch}_err"), fmt=mk, ms=8, lw=1.5,
                     capsize=3, color=col, mfc=col if fill else "white", mew=1.8)
ax1.set_ylabel(r"$C_c(t_\mathrm{final})$")
ax1.set_ylim(-0.03, 1.03)

# --- bottom: max over t of |M_c(t)|
any_m = False
for ch, col in CHANNELS:
    for lab, mk, fill, s in mash:
        key = f"M{ch}_maxabs"
        if key not in s:
            continue
        any_m = True
        y, e = s[key], s.get(f"{key}_err")
        if args.logm:                      # keep error bars inside the positive axis
            e = None if e is None else np.vstack([np.minimum(e, 0.9 * y), e])
        ax2.errorbar(s["nucP_init"], y, e, fmt=mk + "-", ms=8, lw=1.2, capsize=3, color=col,
                     mfc=col if fill else "white", mew=1.8)
if not any_m:
    ax2.text(0.5, 0.5, "no M data (no MASH scan_tully.txt)", ha="center", va="center",
             transform=ax2.transAxes, fontsize=14, color="0.4")
ax2.axhline(0.0, color="0.7", lw=0.8, ls="--")
if args.logm and any_m:
    ax2.set_yscale("log")
ax2.set_ylabel(r"$\max_t\,|M_c(t)|$")
ax2.set_xlabel(r"$P_\mathrm{init}$ (a.u.)")
if args.xlim is not None:
    ax2.set_xlim(args.xlim)

# --- legends: channels (colours) and methods (styles)
ch_handles = [Line2D([], [], color=col, lw=6, label=f"$C_{{{ch}}}$") for ch, col in CHANNELS]
m_handles = []
if exact is not None:
    m_handles.append(Line2D([], [], color="k", lw=2.2, label=EXACT[1]))
for lab, mk, fill, _ in mash:
    m_handles.append(Line2D([], [], color="k", marker=mk, ls="none", ms=8, mew=1.8,
                            mfc="k" if fill else "white", label=lab))
# both legends sit just above the top panel, clear of the data
leg = ax1.legend(handles=ch_handles, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=4,
                 handlelength=1.2, borderaxespad=0.2)
ax1.add_artist(leg)
ax1.legend(handles=m_handles, loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=len(m_handles),
           borderaxespad=0.2)

fig.tight_layout()
fig.subplots_adjust(hspace=0.06)        # panels nearly touching, shared x axis
png = args.out if os.path.isabs(args.out) else os.path.join(args.root, args.out)
fig.savefig(png)
print(f"wrote {png}")
