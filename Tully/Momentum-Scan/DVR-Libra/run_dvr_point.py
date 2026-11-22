"""
Run ONE point of the momentum scan: a Gaussian wavepacket with incident momentum p0 on the Tully 1
model (simple avoided crossing; config `model: form:` picks the tanh form RP_MASH uses, coded below,
or libra's original exponential form), propagated on a grid with Libra's split-operator Fourier
transform (SOFT) solver, and save the channel-resolved populations C_Rg, C_Tg, C_Re, C_Te.

This is the single-point worker invoked by 1_run_p0_scan_local.py. It is also runnable by hand
for debugging a single value:

    python3 run_dvr_point.py --p0 6.0
    python3 run_dvr_point.py --p0 6.0 --save-dens   # also keep the densities for plot_dens_movie.py
    python3 run_dvr_point.py --idx 3                # the 4th p0 of the config's scan.p0 list

Everything physical comes from config.yaml. Per point:

  run length  t_final = scan.run_time_factor * t0,  t0 = m |x0 - x_split| / p0, the free-particle
              crossing time (the same definition as the MASH runs' n_steps, so one factor means the
              same final time in both); nsteps = t_final / dt rounded up to a multiple of save_every
  grid        fixed spacing grid.dx; the box is sized from p0 and t_final (box(): the fastest
              transmitted / reflected fronts plus a buffer), with x = x_split = 0 on the grid

Each point writes into its own p0_grid/p0_<p0>/ directory (named by value, like the MASH nucP_<v>/,
so batches with different p0 lists and run_time_factors never collide).
"""

import os
import io
import json
import time
import argparse
import contextlib

import yaml

script_dir = os.path.dirname(os.path.abspath(__file__))

# Only when run as a script: the driver/collector also import this module for its grid function.
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--p0", type=float, help="incident momentum of this point")
    g.add_argument("--idx", type=int, help="index into the config's scan.p0 list")
    ap.add_argument("--config", default=None)
    ap.add_argument("--num-threads", type=int, default=None,
                    help="cores for this point (default: dyn.num_threads from the config)")
    ap.add_argument("--save-dens", action="store_true",
                    help="also write densities.npz: the ground/excited adiabatic nuclear densities "
                         "|psi_g(x,t)|^2, |psi_e(x,t)|^2 at every snapshot (for plot_dens_movie.py; "
                         "~30 MB, so off in the scan)")
    args = ap.parse_args()

    cfg_path = args.config or os.path.join(script_dir, "config.yaml")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    # Thread count. Must precede numpy/torch import.
    num_threads = int(args.num_threads if args.num_threads is not None else cfg["dyn"]["num_threads"])
    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[_v] = str(num_threads)

import numpy as np
import torch

import libra_py.dynamics.exact_torch.compute as compute
from libra_py.models import Tully_for_exact_pytorch as tully


# =================================================================================================
# Tully 1 (simple avoided crossing), tanh form -- the potential RP_MASH runs (potential.tully_1ac):
#
#     V11 = A*tanh(B*x),   V22 = -V11,   V12 = V21 = C*exp(-D*x^2)
#
# Libra's own Tully 1 (libra_py.models.Tully_for_exact_pytorch / Tully.Tully1_py) is Tully's
# original V11 = A*(1 - exp(-B*x)) for x > 0, -A*(1 - exp(B*x)) for x < 0: same asymptotes +-A, a
# different shape near the crossing.  config.yaml `model: form:` picks "tanh" (this one, matches the
# MASH runs) or "exp" (libra's).
#
# Two functions, as libra splits a model:
#   tully_model1_tanh       torch, the exact_torch solver's potential_fn: Q [ndof, *grid] -> the
#                           DIABATIC matrix [*grid, 2, 2] (diagonal + electronic coupling). This is
#                           all the solver takes: it builds the adiabats itself (torch.linalg.eigh).
#   tully1_matrices         numpy companion with everything else Tully.Tully1_py sets, at points x:
#                           ham_dia, d1ham_dia (dH_dia/dx), dc1_dia (= 0, diabats do not depend on
#                           x), plus the adiabatic energies E_adi and nonadiabatic coupling dc1_adi.
#                           Used for the crossing-time estimate and to check the solver's adiabats.
# =================================================================================================

def tully_model1_tanh(Q, params):
    """
    Q: tensor [ndof, N_1, ..., N_ndof] (ndof = 1).  Returns the diabatic potential matrix
    [N_1, ..., N_ndof, 2, 2] (complex), same format as libra's tully_model1_simple_avoided_crossing.
    """
    x = Q[0]
    A = params.get("A", 0.01)
    B = params.get("B", 1.6)
    C = params.get("C", 0.005)
    D = params.get("D", 1.0)

    V11 = A * torch.tanh(B * x)            # diabatic state 1
    V22 = -V11                             # diabatic state 2 (mirror)
    V12 = C * torch.exp(-D * x**2)         # electronic coupling

    Vmat = torch.zeros((*x.shape, 2, 2), dtype=torch.cfloat, device=x.device)
    Vmat[..., 0, 0] = V11
    Vmat[..., 1, 1] = V22
    Vmat[..., 0, 1] = V12
    Vmat[..., 1, 0] = torch.conj(V12)
    return Vmat


def tully1_matrices(x, params, form="tanh"):
    """
    Everything libra's Tully1_py sets, vectorized over the points x (1d array), for either form:

        ham_dia   [n,2,2]  diabatic Hamiltonian          d1ham_dia [n,2,2]  d ham_dia / dx
        dc1_dia   [n,2,2]  <dia_i|d/dx|dia_j> = 0         E_adi     [n,2]    adiabatic energies (lower, upper)
        dc1_adi   [n,2,2]  <adi_i|d/dx|adi_j>, antisymmetric, [0,1] = d12

    With V22 = -V11 the adiabats are E = -+sqrt(V11^2 + V12^2), mixing angle tan(2 theta) = V12/V11,
    and d12 = d theta/dx = (V11 dV12 - V12 dV11) / (2 (V11^2 + V12^2)) -- its overall sign is the
    (arbitrary) phase convention of the adiabatic eigenvectors.
    """
    x = np.asarray(x, dtype=np.float64)
    A, B, C, D = (float(params[k]) for k in ("A", "B", "C", "D"))
    if form == "tanh":
        V11 = A * np.tanh(B * x)
        dV11 = A * B / np.cosh(B * x) ** 2
    elif form == "exp":
        e = np.exp(-B * np.abs(x))
        V11 = np.sign(x) * A * (1.0 - e)
        dV11 = A * B * e
    else:
        raise ValueError(f"model form must be 'tanh' or 'exp', got {form!r}")
    V12 = C * np.exp(-D * x ** 2)
    dV12 = -2.0 * D * x * V12

    n = x.size
    ham = np.zeros((n, 2, 2)); dham = np.zeros((n, 2, 2))
    ham[:, 0, 0], ham[:, 1, 1], ham[:, 0, 1], ham[:, 1, 0] = V11, -V11, V12, V12
    dham[:, 0, 0], dham[:, 1, 1], dham[:, 0, 1], dham[:, 1, 0] = dV11, -dV11, dV12, dV12
    gap2 = V11 ** 2 + V12 ** 2
    Eup = np.sqrt(gap2)
    d12 = 0.5 * (V11 * dV12 - V12 * dV11) / gap2
    nac = np.zeros((n, 2, 2)); nac[:, 0, 1], nac[:, 1, 0] = d12, -d12
    return {"ham_dia": ham, "d1ham_dia": dham, "dc1_dia": np.zeros((n, 2, 2)),
            "E_adi": np.stack([-Eup, Eup], axis=1), "dc1_adi": nac}


def model_form(cfg):
    return str(cfg["model"].get("form", "exp")).lower()   # no key: libra's original (old configs)


def potential_fn(cfg):
    """The exact_torch potential_fn for the configured form."""
    return tully_model1_tanh if model_form(cfg) == "tanh" else tully.tully_model1_simple_avoided_crossing


def p0_grid(cfg):
    """The incident momenta of the current batch: the config's scan.p0 list."""
    if "grids" in cfg and "scan" not in cfg:
        raise KeyError("old config layout (grids: p0: start/stop/num): use scan: p0: [...] and "
                       "scan: run_time_factor:")
    return np.asarray(cfg["scan"]["p0"], dtype=np.float64)


def point_name(p0):
    """Directory name of one scan point, by value: p0_2.0, p0_4.473, ..."""
    return f"p0_{float(p0)!r}"


def grid_path(cfg):
    """The scan's output directory (runtime.grid_dir, relative to this folder unless absolute)."""
    return os.path.join(script_dir, cfg["runtime"].get("grid_dir", "p0_grid"))


def run_length(cfg, p0):
    """(t0, t_final, nsteps, save_every) of one point: t_final = run_time_factor * t0 rounded up to
    whole save_every blocks, t0 = m |x0 - x_split| / p0 (free-particle crossing time)."""
    w, d = cfg["wavepacket"], cfg["dyn"]
    mass, x0, dt = float(w["mass"]), float(w["x0"]), float(d["dt"])
    x_split = float(cfg["analysis"]["x_split"])
    f = float(cfg["scan"]["run_time_factor"])
    t0 = mass * abs(x_split - x0) / abs(float(p0))
    # long runs: widen the snapshot interval so at most ~max_snapshots wavefunction snapshots are
    # held in memory (the solver stores every one of them in 4 representations)
    save_every = max(int(d["save_every"]), int(np.ceil(f * t0 / dt / int(d.get("max_snapshots", 500)))))
    nsteps = int(np.ceil(f * t0 / dt / save_every)) * save_every
    return t0, nsteps * dt, nsteps, save_every


def box(cfg, p0, t_final):
    """
    (x_min, x_max, npts) of the FFT grid of one point: spacing exactly grid.dx (libra builds the grid
    with linspace(x_min, x_max, npts), both ends included), npts EVEN (libra's momentum grid,
    fftshift(arange(-n//2, n//2)) dk, is the standard FFT ordering only for even n) and
    x_min = -npts/2 dx, x_max = (npts/2 - 1) dx, so x = 0 (= x_split, the R/T divide) is a grid
    point.  The half-width covers, by
    t_final, with momenta up to p_hi = p0 + n_sigma * sigma_p (sigma_p = 1/(2 sigma)):
        the transmitted front          x0 + p_hi t / m
        the front reflected at x_split  2 x_split - x0 - p_hi t / m
        backward-moving components     x0 - max(0, -(p0 - n_sigma sigma_p)) t / m
        the initial packet             |x0| + n_sigma sigma
    plus n_sigma * sigma, then enlarged so all of it stays out of the outer edge_frac on each side
    (where the edge-population check looks).  Nothing in this model accelerates the packet beyond
    its initial momenta (both adiabats have equal asymptotes, the excited channel costs energy), so
    p_hi bounds every component.
    """
    w, g = cfg["wavepacket"], cfg["grid"]
    mass, x0, sigma = float(w["mass"]), float(w["x0"]), float(w["sigma"])
    x_split = float(cfg["analysis"]["x_split"])
    dx, ns, edge = float(g["dx"]), float(g.get("n_sigma", 6)), float(g.get("edge_frac", 0.05))
    sp = 1.0 / (2.0 * sigma)
    p_hi, p_lo = float(p0) + ns * sp, float(p0) - ns * sp
    fronts = [x0 + p_hi * t_final / mass,
              2.0 * x_split - x0 - p_hi * t_final / mass,
              x0 - max(0.0, -p_lo) * t_final / mass,
              abs(x0) + ns * sigma]
    half = (max(abs(f) for f in fronts) + ns * sigma) / (1.0 - 2.0 * edge)
    n_half = int(np.ceil(half / dx)) + 1
    return -n_half * dx, (n_half - 1) * dx, 2 * n_half


def observables(psi_dia, U, V, x, k, mass, x_split):
    """
    Double-precision observables from diabatic wavefunctions.

    psi_dia [nt, npts, 2], U [npts, 2, 2] (C_adi = U C_dia), V [npts, 2, 2] diabatic potential,
    x [npts] grid, k [npts] momentum grid in FFT order.
    """
    npts = x.size
    dx = x[1] - x[0]
    psi_adi = np.einsum("xij,txj->txi", U, psi_dia)
    psi_k = np.fft.fft(psi_dia, axis=1)

    dens_dia = np.abs(psi_dia) ** 2                 # [nt, npts, 2]
    dens_adi = np.abs(psi_adi) ** 2
    dens_k = np.abs(psi_k) ** 2 * dx / npts         # sums to the norm

    # Heaviside step in position with h(0) = 1/2, as in MASH-RPMD_Tully/analyze_transmission.py
    hR = np.where(x > x_split, 1.0, np.where(x < x_split, 0.0, 0.5))[None, :]
    rho_g, rho_e = dens_adi[:, :, 0], dens_adi[:, :, 1]     # ground / excited adiabat densities
    norm = dens_dia.sum(axis=(1, 2)) * dx
    out = {
        "norm": norm,
        "pop_dia": dens_dia.sum(axis=1) * dx,
        "pop_adi": dens_adi.sum(axis=1) * dx,
        # channel-resolved populations  C_c(t) = < h(+-R) P_{g/e} >  (R: reflected, x < x_split;
        # T: transmitted, x > x_split; g/e: ground/excited adiabat).  Cm = CRg+CTg, Cp = CRe+CTe.
        "CRg": (rho_g * (1.0 - hR)).sum(axis=1) * dx,
        "CTg": (rho_g * hR).sum(axis=1) * dx,
        "CRe": (rho_e * (1.0 - hR)).sum(axis=1) * dx,
        "CTe": (rho_e * hR).sum(axis=1) * dx,
        "Cm": rho_g.sum(axis=1) * dx,
        "Cp": rho_e.sum(axis=1) * dx,
        "rho_dia": np.einsum("txi,txj->tij", psi_dia, psi_dia.conj()) * dx,
        "rho_adi": np.einsum("txi,txj->tij", psi_adi, psi_adi.conj()) * dx,
        "x_avg": (dens_dia.sum(axis=2) * x).sum(axis=1) * dx / norm,
        "p_avg": (dens_k.sum(axis=2) * k).sum(axis=1) / norm,
        "Ekin": (dens_k.sum(axis=2) * (0.5 * k ** 2 / mass)).sum(axis=1),
        "Epot": np.einsum("txi,xij,txj->t", psi_dia.conj(), V, psi_dia).real * dx,
        "dens_dia": dens_dia.astype(np.float32),
        "dens_adi": dens_adi.astype(np.float32),
    }
    out["Etot"] = out["Ekin"] + out["Epot"]
    return out


def crossing_time(model, mass, x0, p0, x_split, representation, state, form="tanh"):
    """
    Classical time for the wavepacket centre to travel from x0 to x_split with initial momentum p0
    on the adiabatic surface the wavepacket starts on:  t = int_x0^x_split  m dx / p(x),
    p(x) = sqrt(p0^2 - 2 m (E(x) - E(x0))).   Returns (t_cross, free), where free=True means the
    surface has a classical turning point before x_split (p0 too low to get there), in which case
    the free-particle time m |x_split - x0| / p0 is returned instead.
    """
    xs = np.linspace(x0, x_split, 20001)
    mats = tully1_matrices(xs, model, form)
    V11 = mats["ham_dia"][:, 0, 0]
    E_up = mats["E_adi"][:, 1]
    if representation == "adiabatic":
        upper = int(state) == 1
    else:                                   # the adiabat the initial diabatic state sits on at x0
        upper = (V11[0] if int(state) == 0 else -V11[0]) > 0
    E = E_up if upper else -E_up
    p2 = p0 ** 2 - 2.0 * mass * (E - E[0])
    t_free = mass * abs(x_split - x0) / abs(p0)
    if np.any(p2 <= 0.0):
        return t_free, True
    f = mass / np.sqrt(p2)
    return float(abs(np.sum(0.5 * (f[1:] + f[:-1]) * np.diff(xs)))), False


def main():
    m, w, d = cfg["model"], cfg["wavepacket"], cfg["dyn"]
    p0 = float(args.p0) if args.p0 is not None else float(p0_grid(cfg)[args.idx])
    mass, x0, sigma = float(w["mass"]), float(w["x0"]), float(w["sigma"])
    dt = float(d["dt"])
    x_split = float(cfg["analysis"]["x_split"])
    rtf = float(cfg["scan"]["run_time_factor"])

    # run length (free-particle crossing time x run_time_factor) and the box it needs
    form = model_form(cfg)
    t0, t_final, nsteps, save_every = run_length(cfg, p0)
    t_cross, free = crossing_time(m, mass, x0, p0, x_split, w["representation"], w["state"], form)
    x_lo, x_hi, npts = box(cfg, p0, t_final)
    g = {"x_min": x_lo, "x_max": x_hi, "npts": npts}
    print(f"run_time_factor = {rtf:g} x t0 = {t0:.1f} a.u. -> nsteps = {nsteps} (t_final = {t_final:g});  "
          f"classical arrival at x_split on the adiabat: "
          + ("none (p0 below the barrier)" if free else f"{t_cross:.1f} a.u."), flush=True)

    grid_dir = grid_path(cfg)
    outdir = os.path.join(grid_dir, point_name(p0))
    os.makedirs(outdir, exist_ok=True)
    os.chdir(outdir)
    if os.path.exists("result.json"):      # a failed re-run must not leave a stale result behind
        os.remove("result.json")

    torch.set_num_threads(num_threads)

    model_params = {k: float(m[k]) for k in ("A", "B", "C", "D")}
    # libra's gaussian_wavepacket: sigma_x = 1/sqrt(2*mass*omega)
    omega = 1.0 / (2.0 * mass * sigma ** 2)

    print(f"Tully 1 ({form} form): {model_params}", flush=True)
    print(f"wavepacket: x0={x0} p0={p0} sigma={sigma} mass={mass} "
          f"({w['representation']} state {w['state']})", flush=True)
    print(f"grid: [{g['x_min']:g}, {g['x_max']:g}] with {g['npts']} points (dx = {float(cfg['grid']['dx']):g});  "
          f"dt={dt}, nsteps={nsteps}", flush=True)
    print(f"output -> {outdir}", flush=True)

    solver = compute.exact_tdse_solver_multistate({
        "prefix": "exact",
        "grid_size": [int(g["npts"])],
        "q_min": [float(g["x_min"])], "q_max": [float(g["x_max"])],
        "mass": [mass],
        "dt": dt, "nsteps": nsteps, "save_every_n_steps": save_every,
        "Nstates": 2,
        "potential_fn": potential_fn(cfg),
        "potential_fn_params": model_params,
        "psi0_fn": compute.gaussian_wavepacket,
        "psi0_fn_params": {"mass": [mass], "omega": [omega], "q0": [x0], "p0": [p0]},
        "representation": w["representation"],
        "initial_state_index": int(w["state"]),
        "method": d["method"],
        "device": torch.device("cpu"),
    })

    # The solver prints one line per snapshot; keep that in a log file instead of the terminal.
    # solve() is not used because its save() dumps every snapshot in 4 representations to a .pt.
    t0 = time.time()
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        solver.initialize_grids()
        solver.initialize_operators()
        solver.propagate()
    t_dyn = time.time() - t0
    with open("solver.log", "w") as f:
        f.write(log.getvalue())

    # ---------------- collect: snapshots (taken before steps 0, n, 2n, ...) + the final state ----
    psi = np.concatenate([solver.psi_r_dia_all.numpy(), solver.psi_r_dia.numpy()[None]], axis=0)
    psi = psi.astype(np.complex128)
    t_ar = np.arange(psi.shape[0]) * save_every * dt

    x = solver.Q[0].numpy().astype(np.float64)
    k = solver.K[0].numpy().astype(np.float64)
    U = solver.eigvecs.numpy().astype(np.complex128)
    V = solver.V.numpy().astype(np.complex128)
    E_adi = solver.eigvals.numpy().astype(np.float64)
    # the solver's adiabats (eigh of the potential_fn matrix) vs the analytic model
    adiabat_mismatch = float(np.abs(E_adi - tully1_matrices(x, model_params, form)["E_adi"]).max())

    obs = observables(psi, U, V, x, k, mass, x_split)

    # time series only (no wavepacket densities) to keep the scan directory small
    np.savez("dynamics.npz", t=t_ar,
             mass=mass, x0=x0, p0=p0, sigma=sigma, dt=dt, nsteps=nsteps, x_split=x_split,
             **model_params, **{key: val for key, val in obs.items() if not key.startswith("dens_")})

    if args.save_dens:
        np.savez("densities.npz", t=t_ar, x=x, rho_g=obs["dens_adi"][:, :, 0],
                 rho_e=obs["dens_adi"][:, :, 1], mass=mass, x0=x0, p0=p0, sigma=sigma,
                 x_split=x_split, Cm=obs["Cm"], Cp=obs["Cp"], norm=obs["norm"])
        print(f"densities -> {os.path.join(outdir, 'densities.npz')}", flush=True)

    # Is anything close to the (periodic) grid boundary? Check the outer 5% on each side.
    nedge = max(1, x.size // 20)
    dx = x[1] - x[0]
    tot = obs["dens_dia"].sum(axis=2).astype(np.float64)
    edge_pop = float(((tot[:, :nedge].sum(axis=1) + tot[:, -nedge:].sum(axis=1)) * dx).max())
    # Momentum-grid check: population in the outer 5% of |k|
    kmax = np.abs(k).max()
    psi_k = np.fft.fft(psi[-1], axis=0)
    hi_k = float((np.abs(psi_k[np.abs(k) > 0.9 * kmax]) ** 2).sum() * dx / x.size)

    result = {
        "point": point_name(p0), "run_time_factor": rtf, "t0": t0,
        "x_min": float(x_lo), "x_max": float(x_hi),
        "model": model_params, "model_form": form, "max_adiabat_mismatch": adiabat_mismatch,
        "mass": mass, "x0": x0, "p0": p0, "sigma": sigma,
        "representation": w["representation"], "state": int(w["state"]),
        "E_total": float(obs["Etot"][0]),
        "t_final": float(t_ar[-1]),
        "pop_adi_final": [float(v) for v in obs["pop_adi"][-1]],
        "pop_dia_final": [float(v) for v in obs["pop_dia"][-1]],
        "CRg_final": float(obs["CRg"][-1]), "CTg_final": float(obs["CTg"][-1]),
        "CRe_final": float(obs["CRe"][-1]), "CTe_final": float(obs["CTe"][-1]),
        "x_avg_final": float(obs["x_avg"][-1]), "p_avg_final": float(obs["p_avg"][-1]),
        "norm_final": float(obs["norm"][-1]),
        "max_norm_drift": float(np.abs(obs["norm"] - obs["norm"][0]).max()),
        "max_energy_drift": float(np.abs(obs["Etot"] - obs["Etot"][0]).max()),
        "max_edge_population": edge_pop,
        "high_k_population_final": hi_k,
        "npts": int(x.size), "dx": float(dx), "k_max": float(kmax),
        "dt": dt, "nsteps": nsteps, "t_cross": t_cross, "t_cross_free_estimate": bool(free), "method": d["method"],
        "t_dyn_s": t_dyn,
    }
    with open("result.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"E_total = {result['E_total']:.6f}   t_final = {result['t_final']:g}", flush=True)
    print("  ".join(f"C_{c}(t_final) = {result[f'C{c}_final']:.6f}" for c in ("Rg", "Tg", "Re", "Te")),
          flush=True)
    print(f"norm drift {result['max_norm_drift']:.2e}   energy drift {result['max_energy_drift']:.2e}   "
          f"edge population {edge_pop:.2e}   high-k population {hi_k:.2e}", flush=True)
    if edge_pop > 1e-4:
        print("WARNING: the wavepacket reaches the grid boundary (periodic wrap-around) - "
              "raise grid.n_sigma / grid.edge_frac or shorten the run", flush=True)
    print(f"model check: max |E_adi(solver) - E_adi(analytic)| = {adiabat_mismatch:.2e}", flush=True)
    print(f"timing: dyn {t_dyn:.1f}s", flush=True)


if __name__ == "__main__":
    main()
