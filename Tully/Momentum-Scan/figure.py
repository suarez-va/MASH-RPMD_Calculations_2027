"""
The final figure of the Tully momentum scan: exact quantum dynamics vs MASH vs MASH-RPMD as a function of
the incident momentum nucP_init, for both initial conditions.  Three panels sharing the x axis:

    top     Deterministic-Init: final channel populations C_Rg, C_Tg, C_Re, C_Te
    middle  Wigner-Init:        the same
            exact (DVR-Libra, the same curves in both): dashed lines; MASH (MASH-Classical): solid lines
            with the block SE as a shaded band; MASH-RPMD (every MASH-Quantum-beta*/, ordered by the
            beta in its config.py): markers with error bars, one marker per beta in both panels
    bottom  max over t of |M_c(t)| of Wigner-Init MASH-Classical -- the largest correction from the
            time-dependent weights in each channel (shaded: the SE at that time)

    python3 figure.py                       # -> figure.png
    python3 figure.py --logm --xlim 2 31    # log y on the M panel, fixed x window
    python3 figure.py --root /some/copy/of/Momentum-Scan --out test.png

It only reads each method's scan_transmission.txt, so make those first:

    DVR-Libra/                                          python3 2_collect.py
    <Init>/MASH-Classical/, <Init>/MASH-Quantum-beta*/  python ../../../scan_transmission.py --ncores 10

A method directory without a scan_transmission.txt is skipped with a warning.  A scan_transmission.txt
whose first line starts with "# PLACEHOLDER" (a stand-in copy of another method's result, so the figure
can be laid out before the data exists) is plotted, but flagged in the terminal and in the figure title;
the real scan_transmission.py / 2_collect.py overwrite it.
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

# exact quantum dynamics (one calculation, drawn on both C panels): directory and legend label
EXACT = ("DVR-Libra", "Exact")
# the two initial-condition trees, top to bottom: (directory, panel label)
INITS = [("Deterministic-Init", "Deterministic-Init"), ("Wigner-Init", "Wigner-Init")]
# MASH (solid line + SE band); MASH-RPMD = every RPMD_GLOB dir, markers by increasing beta
MASH = ("MASH-Classical", "MASH")
RPMD_GLOB = "MASH-Quantum-beta*"
RPMD_MARKERS = ["o", "s", "^", "D", "v", "P"]
# the max|M| panel: (init tree, method dir)
M_FROM = ("Wigner-Init", "MASH-Classical")
# channels, colours as analyze_transmission.py
CHANNELS = [("Rg", "C0"), ("Tg", "C2"), ("Re", "C1"), ("Te", "C3")]
PLACEHOLDER = "# PLACEHOLDER"

ap = argparse.ArgumentParser()
ap.add_argument("--root", default=script_dir, help="the Momentum-Scan directory (default: here)")
ap.add_argument("--rates", default="scan_transmission.txt", help="summary file name in each method dir")
ap.add_argument("--xlim", type=float, nargs=2, default=None, metavar=("LO", "HI"))
ap.add_argument("--logm", action="store_true", help="log y axis on the max|M| panel")
ap.add_argument("--out", default="figure.png", help="output png (relative to --root)")
args = ap.parse_args()

placeholders = []


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


def load(rel):
    """scan_transmission.txt of method dir `rel` (relative to --root), sorted by nucP_init; None if absent."""
    path = os.path.join(args.root, rel, args.rates)
    if not os.path.isfile(path):
        print(f"WARNING: {os.path.join(rel, args.rates)} not found - skipping {rel}")
        return None
    with open(path) as f:
        if f.readline().startswith(PLACEHOLDER) and rel not in placeholders:
            placeholders.append(rel)
            print(f"WARNING: {os.path.join(rel, args.rates)} is a PLACEHOLDER, not this method's result")
    s = read_scan(path)
    if "nucP_init" not in s:
        print(f"WARNING: no nucP_init array in {path} - skipping {rel}")
        return None
    order = np.argsort(s["nucP_init"])
    return {k: (v[order] if isinstance(v, np.ndarray) and v.shape == order.shape else v)
            for k, v in s.items()}


def config_beta(rel):
    """beta of an RP_MASH method dir, from its config.py."""
    spec = importlib.util.spec_from_file_location("_figure_cfg_" + re.sub(r"\W", "_", rel),
                                                  os.path.join(args.root, rel, "config.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return float(mod.CONFIG["beta"])


def rpmd_dirs(init):
    """{beta: dir relative to --root} of the MASH-Quantum-beta*/ dirs of one init tree."""
    out = {}
    for d in glob.glob(os.path.join(args.root, init, RPMD_GLOB + "/")):
        rel = os.path.join(init, os.path.basename(os.path.normpath(d)))
        if os.path.isfile(os.path.join(args.root, rel, "config.py")):
            out[config_beta(rel)] = rel
    return out


# ---------------------------------------------------------------- data
exact = load(EXACT[0])
rpmd = {init: rpmd_dirs(init) for init, _ in INITS}
betas = sorted(set(b for d in rpmd.values() for b in d))           # one marker per beta, all panels
marker = {b: RPMD_MARKERS[i % len(RPMD_MARKERS)] for i, b in enumerate(betas)}
panels = []                       # per init: (label, mash data or None, [(beta, data)])
for init, label in INITS:
    m = load(os.path.join(init, MASH[0]))
    r = [(b, s) for b in sorted(rpmd[init]) if (s := load(rpmd[init][b])) is not None]
    panels.append((label, m, r))
mdata = load(os.path.join(*M_FROM))
if exact is None and mdata is None and not any(m is not None or r for _, m, r in panels):
    raise SystemExit(f"no {args.rates} in any method directory under {args.root} - nothing to plot")

# ---------------------------------------------------------------- figure
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

fig, axs = plt.subplots(3, 1, figsize=(16, 14), sharex=True, gridspec_kw={'height_ratios': [3, 3, 2]})


def no_data(ax, text):
    ax.text(0.5, 0.5, text, ha="center", va="center", transform=ax.transAxes, fontsize=14, color="0.4")


# --- C panels: one per init tree
for ax, (label, m, r) in zip(axs[:2], panels):
    for ch, col in CHANNELS:
        if exact is not None and f"C{ch}" in exact:
            ax.plot(exact["nucP_init"], exact[f"C{ch}"], "--", color=col, lw=2.2)
        if m is not None and f"C{ch}" in m:
            x, y, e = m["nucP_init"], m[f"C{ch}"], m.get(f"C{ch}_err")
            ax.plot(x, y, "-", color=col, lw=2.2)
            if e is not None:
                ax.fill_between(x, y - e, y + e, color=col, alpha=0.2, lw=0)
        for b, s in r:
            if f"C{ch}" in s:
                ax.errorbar(s["nucP_init"], s[f"C{ch}"], s.get(f"C{ch}_err"), fmt=marker[b], markersize=5,
                            linewidth=1.2, color=col)
    if m is None and not r:
        no_data(ax, f"no {label} MASH / MASH-RPMD data")
    ax.text(0.99, 0.5, label, transform=ax.transAxes, ha="right", va="center", fontsize=16,
            bbox=dict(facecolor="white", edgecolor="0.6", boxstyle="round,pad=0.3"))
    ax.set_ylabel(r"$C_c(t_\mathrm{final})$")
    ax.set_ylim(-0.03, 1.03)

# --- bottom: max over t of |M_c(t)|, Wigner-Init MASH-Classical
ax = axs[2]
if mdata is not None and any(f"M{ch}_maxabs" in mdata for ch, _ in CHANNELS):
    for ch, col in CHANNELS:
        key = f"M{ch}_maxabs"
        if key not in mdata:
            continue
        x, y, e = mdata["nucP_init"], mdata[key], mdata.get(f"{key}_err")
        ax.plot(x, y, "-", color=col, lw=2.2)
        if e is not None:
            lo = np.maximum(y - e, 0.1 * y) if args.logm else y - e   # stay positive on a log axis
            ax.fill_between(x, lo, y + e, color=col, alpha=0.2, lw=0)
    if args.logm:
        ax.set_yscale("log")
else:
    no_data(ax, f"no M data ({os.path.join(*M_FROM)}/{args.rates})")
ax.axhline(0.0, color="0.7", lw=0.8, ls="--")
ax.text(0.99, 0.9, f"{M_FROM[0]}, {MASH[1]}", transform=ax.transAxes, ha="right", va="top", fontsize=16,
        bbox=dict(facecolor="white", edgecolor="0.6", boxstyle="round,pad=0.3"))
ax.set_ylabel(r"$\max_t\,|M_c(t)|$")
ax.set_xlabel(r"$P_\mathrm{init}$ (a.u.)")
if args.xlim is not None:
    ax.set_xlim(args.xlim)

# --- legends above the top panel: channels (colours) and methods (styles, union of both trees)
ch_handles = [Line2D([], [], color=col, lw=6, label=f"$C_{{{ch}}}$") for ch, col in CHANNELS]
m_handles = []
if exact is not None:
    m_handles.append(Line2D([], [], color="k", lw=2.2, ls="--", label=EXACT[1]))
if any(m is not None for _, m, _ in panels):
    m_handles.append(Line2D([], [], color="k", lw=2.2, label=MASH[1]))
for b in sorted(set(b for _, _, r in panels for b, _ in r)):
    m_handles.append(axs[0].errorbar([], [], [], fmt=marker[b], markersize=5, linewidth=1.2, color="k",
                                     label=rf"MASH-RPMD $\beta$={b:g}"))
leg = axs[0].legend(handles=ch_handles, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=4,
                    handlelength=1.2, borderaxespad=0.2)
axs[0].add_artist(leg)
axs[0].legend(handles=m_handles, loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=len(m_handles),
              borderaxespad=0.2)
if placeholders:
    fig.suptitle(f"PLACEHOLDER data in {len(placeholders)} method dir(s) (listed in the terminal)",
                 color="red", fontsize=13, y=0.995)

fig.tight_layout()
fig.subplots_adjust(hspace=0.06)        # panels nearly touching, shared x axis
png = args.out if os.path.isabs(args.out) else os.path.join(args.root, args.out)
fig.savefig(png)
print(f"wrote {png}")
if placeholders:
    print(f"NOTE: {len(placeholders)} placeholder file(s) plotted: " + ", ".join(placeholders))
