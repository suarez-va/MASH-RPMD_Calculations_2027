"""
The final figure of the Tully momentum scan: exact quantum dynamics vs MASH vs MASH-RPMD, as a
function of the incident momentum nucP_init.  Two panels sharing the x axis:

    top     final channel populations C_Rg, C_Tg, C_Re, C_Te
            exact: dashed lines; MASH: solid lines with the block SE as a shaded band;
            MASH-RPMD: filled circles with error bars (the System-Bath figures' style)
    bottom  max over t of |M_c(t)| for MASH only -- the largest correction from the
            time-dependent weights in each channel (shaded: the SE at that time)

    python3 figure.py                       # -> figure.png
    python3 figure.py --logm --xlim 2 31    # log y on the M panel, fixed x window
    python3 figure.py --root /some/copy/of/Momentum-Scan --out test.png

It only reads each method's scan_transmission.txt, so make those first:

    ../../Wigner-Init/Momentum-Scan/DVR-Libra/   python3 2_collect.py   (shared exact result)
    MASH-Classical/ MASH-Quantum-beta*/   python ../../../scan_transmission.py --ncores 10   (in each dir)

A method directory without a scan_transmission.txt is skipped with a warning.  --exact DIR takes the exact
results from elsewhere (relative to --root).
"""

import os
import re
import glob
import argparse
import importlib.util

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

script_dir = os.path.dirname(os.path.abspath(__file__))

# exact quantum dynamics: directory (relative to --root, override with --exact) and legend label.
# Deterministic-Init/'s copy of this script points at Wigner-Init/'s DVR-Libra: there is one exact
# calculation for both trees (a Gaussian wavepacket -- strictly the exact counterpart of Wigner-Init's
# Wigner-sampled centroid).
EXACT = (os.path.join("..", "..", "Wigner-Init", "Momentum-Scan", "DVR-Libra"), "Exact")
# (directory, legend label, draw style) of the trajectory methods: "line" = solid line + SE band
# (also drawn in the max|M| panel), "points" = markers with error bars (top panel only). The
# MASH-RPMD dirs are every MASH-Quantum-beta*/ next to this script, ordered by the beta in their
# config.py, each with its own marker (RPMD_MARKERS, in that order).
METHODS = [("MASH-Classical", "MASH", "line")]
RPMD_GLOB = "MASH-Quantum-beta*"
RPMD_MARKERS = ["o", "s", "^", "D", "v", "P"]
# channels, colours as analyze_transmission.py
CHANNELS = [("Rg", "C0"), ("Tg", "C2"), ("Re", "C1"), ("Te", "C3")]

ap = argparse.ArgumentParser()
ap.add_argument("--root", default=script_dir, help="directory holding the method dirs (default: here)")
ap.add_argument("--rates", default="scan_transmission.txt", help="summary file name in each method dir")
ap.add_argument("--exact", default=EXACT[0], help="exact (DVR) results dir, relative to --root")
ap.add_argument("--xlim", type=float, nargs=2, default=None, metavar=("LO", "HI"))
ap.add_argument("--logm", action="store_true", help="log y axis on the max|M| panel")
ap.add_argument("--out", default="figure.png", help="output png (relative to --root)")
args = ap.parse_args()


def read_scan(path):
    """The `name = [...]` / `name = value` lines of a scan_transmission.txt -> {name: array or float}."""
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


def config_beta(method):
    """beta of an RP_MASH method dir, from its config.py."""
    spec = importlib.util.spec_from_file_location("_figure_cfg_" + re.sub(r"\W", "_", method),
                                                  os.path.join(args.root, method, "config.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return float(mod.CONFIG["beta"])


def rpmd_methods():
    """[(dir, label, 'points', marker)] of the MASH-Quantum-beta*/ dirs, by increasing beta."""
    dirs = [os.path.basename(os.path.normpath(d))
            for d in glob.glob(os.path.join(args.root, RPMD_GLOB + "/"))]
    found = sorted((config_beta(d), d) for d in dirs if os.path.isfile(os.path.join(args.root, d, "config.py")))
    return [(d, rf"MASH-RPMD $\beta$={b:g}", "points", RPMD_MARKERS[i % len(RPMD_MARKERS)])
            for i, (b, d) in enumerate(found)]


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


exact = load(args.exact)
mash = [(lab, style, mk, s) for d, lab, style, mk in [m + (None,) for m in METHODS] + rpmd_methods()
        if (s := load(d)) is not None]
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
        ax1.plot(exact["nucP_init"], exact[f"C{ch}"], "--", color=col, lw=2.2)
    for lab, style, mk, s in mash:
        if f"C{ch}" not in s:
            continue
        x, y, e = s["nucP_init"], s[f"C{ch}"], s.get(f"C{ch}_err")
        if style == "line":
            ax1.plot(x, y, "-", color=col, lw=2.2)
            if e is not None:
                ax1.fill_between(x, y - e, y + e, color=col, alpha=0.2, lw=0)
        else:
            ax1.errorbar(x, y, e, fmt=mk, markersize=5, linewidth=1.2, color=col)
ax1.set_ylabel(r"$C_c(t_\mathrm{final})$")
ax1.set_ylim(-0.03, 1.03)

# --- bottom: max over t of |M_c(t)|, MASH ("line" methods) only
any_m = False
for ch, col in CHANNELS:
    for lab, style, mk, s in mash:
        key = f"M{ch}_maxabs"
        if style != "line" or key not in s:
            continue
        any_m = True
        x, y, e = s["nucP_init"], s[key], s.get(f"{key}_err")
        ax2.plot(x, y, "-", color=col, lw=2.2)
        if e is not None:
            lo = np.maximum(y - e, 0.1 * y) if args.logm else y - e   # stay positive on a log axis
            ax2.fill_between(x, lo, y + e, color=col, alpha=0.2, lw=0)
if not any_m:
    ax2.text(0.5, 0.5, "no M data (no MASH-Classical scan_transmission.txt)", ha="center", va="center",
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
    m_handles.append(Line2D([], [], color="k", lw=2.2, ls="--", label=EXACT[1]))
for lab, style, mk, _ in mash:
    if style == "line":
        m_handles.append(Line2D([], [], color="k", lw=2.2, label=lab))
    else:
        m_handles.append(ax1.errorbar([], [], [], fmt=mk, markersize=5, linewidth=1.2, color="k",
                                      label=lab))
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
