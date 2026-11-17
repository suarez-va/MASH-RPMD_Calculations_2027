"""
transmission_scan.py -- the driver of the Tully single-avoided-crossing (Tully 1) momentum scan: every
trajectory runs once at EVERY scan point of `nucP_init_scan`.

Per trajectory idx (one call of run_trajectory), for every scan point (name, overrides) from
scan_points(cfg): build the ring polymer's initial state (below: thermal internal modes, centroid at
or Wigner-sampled around (nucR_init, nucP_init of that point)), sample the spin in the lower hemisphere
(init_map_spin(0): the particle enters on the lower adiabat), and propagate microcanonically (no
thermostat), writing the point's .dat files into the subdirectory <name>/ of the trajectory's work
dir.  The run length is per point (see n_steps): twice the free-particle time to reach R = 0,
ceil(2 |nucR_init| mass / (nucP_init delt)) steps, unless the config sets `traj_time`, which then
applies to every point.

Analysis of the whole scan: scan_transmission.py next to this file (analyze_transmission.py on every point).

workflow.shards sees scan_points() and stores point <name> of trajectory idx as a row of
<run_dir>/<name>/shards/shard_<s>.hdf, with the point's own nucP_init in the shard's config_json.
So <run_dir>/nucP_10.0/shards/ is an ordinary shard directory and the Tully analysis runs inside it
unchanged:

    cd nucP_10.0 && python ../../../analyze_transmission.py --file shards --ncores 10

--- the scan --------------------------------------------------------------------------------------

`nucP_init_scan` is always a list (a one-element list is a one-point scan); point v runs in
nucP_<v>/ (e.g. nucP_5.0, nucP_12.0, nucP_30.0).  Everything else (nucR_init, the Tully parameters
A..D, nbds, delt, traj_time, the jumps) is shared by every point.

--- the initial ring polymer ---------------------------------------------------------------------

Internal normal modes (always, every run): drawn thermally as System-Bath/therm_scan._sample_nvt
with kvec = gamma = 0 (a free ring polymer; no thermalization run, there is no GLE memory to fill):

    Q_k ~ N(0, sqrt(nbds / (beta m w_k^2))),   P_k ~ N(0, sqrt(m nbds / beta)),   k >= 1,

w_k = calc_normal_mode_freq(beta/nbds, nbds), `beta` the sampling temperature.  nbds = 1 has none.

Centroid (`centroid_init`):

    'deterministic' (default)  exactly (nucR_init, nucP_init)                     -> Deterministic-Init/
    'wigner'                   the Wigner distribution of the Gaussian wavepacket
                               psi ~ exp(-(R - nucR_init)^2 / (4 sigma_init^2)) exp(i nucP_init R)
                               (DVR-Libra's initial state, sigma_init = its `sigma`):
                                   R_c ~ N(nucR_init, sigma_init),  P_c ~ N(nucP_init, hbar/(2 sigma_init))
                                                                                  -> Wigner-Init/

Every bead is the centroid plus the internal-mode displacement.  All draws come from the
trajectory's initial-condition seed (see below), re-seeded at every scan point, so trajectory k has
the SAME standard-normal draws at every nucP_init (its Wigner momentum is just shifted with the scan).

--- quantum jumps ---------------------------------------------------------------------------------

The same rule at every point, from ONE of two triggers (as in MASH-RPMD_System-Bath/therm_scan.py):
jump_index (production step indices k, t = k*delt; [] = none) or jump_gap (Egap: jump once on
leaving |V1 - V0| < Egap at the centroid; None = off, needs jump_index = []).  Note the Tully
diabatic gap |2 A tanh(B R)| is at most 2A = 0.02, so Egap must be on that scale.

--- random numbers ---------------------------------------------------------------------------------

The trajectory seed is split exactly as in MASH-RPMD_Tully-Delta/traj.py (initial-condition half:
the ring-polymer start above; dynamics half: the spin samples),
so a given base_seed gives trajectory k the same
spin samples as Tully-Delta.  The dynamics seed is the SAME at every scan point: correlated
sampling across the scan, so differences between momenta are less noisy.

Usage (from a MASH-Classical / MASH-Quantum dir, PYTHONPATH=<RP_MASH root>):
    python -m workflow.runner --config config.py                             # local
    python -m workflow.slurm  --config config.py && sbatch submit_grid.sh    # HPC
"""

import os

import numpy as np
import mash_rpmd   # via PYTHONPATH (= RP_MASH root)
import normal_mode


# ------------------------------------------------------------------ the scan
def _point_name(value):
    return f'nucP_{float(value)!r}'


def scan_points(cfg):
    """
    workflow.shards hook: [(directory name, config overrides)] for every scan point, in list order.
    """
    if 'nucP_init' in cfg:
        raise KeyError('this is a momentum scan: give the momenta as the list `nucP_init_scan`, '
                       'not a scalar `nucP_init`')
    if 'nucP_init_scan' not in cfg:
        raise KeyError('the config needs a `nucP_init_scan` list')
    _centroid_init(cfg)                            # a bad centroid_init / sigma_init fails at startup
    values = list(cfg['nucP_init_scan'])
    if not values:
        raise ValueError('nucP_init_scan is empty')
    if cfg.get('jump_gap') is not None and list(cfg.get('jump_index', [])):
        raise ValueError(f"jump_gap={cfg['jump_gap']} and jump_index={list(cfg['jump_index'])} are two "
                         f"different jump triggers: set jump_index = [] to use jump_gap")
    points = [(_point_name(v), {'nucP_init': float(v)}) for v in values]
    names = [n for n, _ in points]
    if len(set(names)) != len(names):
        raise ValueError(f'nucP_init_scan has repeated values: {values}')
    return points


# ------------------------------------------------------------------ one trajectory
def _seeds(seed):
    """(initial conditions, dynamics) seeds, exactly as MASH-RPMD_Tully-Delta/traj.py.
    seed=None -> one fresh dynamics seed, still shared by every scan point."""
    if seed is None:
        seed = int(np.random.SeedSequence().generate_state(1, dtype=np.uint32)[0])
    ic, dyn = np.random.SeedSequence(int(seed)).generate_state(2, dtype=np.uint32)
    return int(ic), int(dyn)


HBAR = 1.0                                         # atomic units


def _centroid_init(cfg):
    """Validated centroid mode ('deterministic' / 'wigner'); refuses the old rp_init key."""
    if 'rp_init' in cfg:
        raise KeyError("rp_init is no longer used: the internal ring-polymer modes are always sampled; "
                       "choose the centroid start with centroid_init = 'deterministic' or 'wigner'")
    mode = str(cfg.get('centroid_init', 'deterministic')).lower()
    if mode not in ('deterministic', 'wigner'):
        raise ValueError(f"centroid_init must be 'deterministic' or 'wigner', got {cfg.get('centroid_init')!r}")
    if mode == 'wigner':
        if cfg.get('sigma_init') is None or float(cfg['sigma_init']) <= 0.0:
            raise ValueError("centroid_init = 'wigner' needs a positive sigma_init (the wavepacket width)")
    elif cfg.get('sigma_init') is not None:
        raise ValueError("sigma_init is only used with centroid_init = 'wigner' (this config is "
                         "'deterministic'): remove it, or set centroid_init = 'wigner'")
    return mode


def initial_state(cfg, nucP_init, ic_seed):
    """
    (nucR, nucP), each (nbds, nnuc): the centroid at (nucR_init, nucP_init) exactly ('deterministic')
    or Wigner-sampled around it ('wigner'), plus thermally sampled internal normal modes (module
    docstring).  Fresh rng on ic_seed: the centroid draws first, then the internal modes.
    """
    nbds, nnuc = int(cfg['nbds']), int(cfg['nnuc'])
    mass = np.broadcast_to(np.asarray(cfg['mass'], float), (nnuc,))
    beta = float(cfg['beta'])
    rng = np.random.default_rng(ic_seed)

    Rc = np.full(nnuc, float(cfg['nucR_init']))
    Pc = np.full(nnuc, float(nucP_init))
    if _centroid_init(cfg) == 'wigner':
        sigma = float(cfg['sigma_init'])
        Rc = Rc + sigma * rng.standard_normal(nnuc)
        Pc = Pc + HBAR / (2.0 * sigma) * rng.standard_normal(nnuc)

    nucR = np.repeat(Rc[None, :], nbds, axis=0)
    nucP = np.repeat(Pc[None, :], nbds, axis=0)
    if nbds == 1:
        return nucR, nucP
    omega_k = normal_mode.calc_normal_mode_freq(beta / nbds, nbds)
    dR_k = np.zeros((nbds, nnuc))
    dR_k[1:] = np.sqrt(nbds / (beta * mass[None, :] * omega_k[1:, None]**2))
    dP_k = np.zeros((nbds, nnuc))
    dP_k[1:] = np.sqrt(mass[None, :] * nbds / beta)
    nucR_nm = rng.normal(0.0, dR_k)               # centroid rows have width 0 -> exactly 0
    nucP_nm = rng.normal(0.0, dP_k)
    for i in range(nnuc):
        nucR[:, i] += normal_mode.normal_mode_to_real(nucR_nm[:, i])
        nucP[:, i] += normal_mode.normal_mode_to_real(nucP_nm[:, i])
    return nucR, nucP


def _build(cfg, nucR, nucP, dyn_seed):
    nbds, nnuc = int(cfg['nbds']), int(cfg['nnuc'])
    # tully_1ac wants each parameter as its own (nnuc,) array -- see potential.tully_1ac.error_check
    potparams = [np.array([cfg['A']]), np.array([cfg['B']]),
                 np.array([cfg['C']]), np.array([cfg['D']])]
    return mash_rpmd.mash_rpmd(
        nstates=cfg['nstates'], nnuc=nnuc, nbds=nbds,
        beta=cfg['beta'], mass=np.asarray(cfg['mass'], float),
        potype='tully_1ac', potparams=potparams, nucR=nucR, nucP=nucP,
        spinmap_bool=True, centroid_bool=True, bead_bool=False,
        jump_index=cfg.get('jump_index', []), jump_gap=cfg.get('jump_gap'),
        full_jump=cfg.get('full_jump', True), seed=dyn_seed, verbose=False)


def n_steps(cfg, nucP_init):
    """
    Production steps of one scan point.  With `traj_time` in the config: round(traj_time/delt), the
    same for every point.  Without it: TWICE the free-particle time to reach the crossing at R = 0,

        t_0 = |nucR_init| / v = |nucR_init| * mass / nucP_init,   Nprod = ceil(2 t_0 / delt),

    so slow points run longer and fast ones are not wasted (e.g. nucR_init = -10, mass = 2000,
    delt = 5: P = 5 -> 1600 steps (8000 a.u.), P = 30 -> 267 steps).
    """
    delt = float(cfg['delt'])
    if cfg.get('traj_time') is not None:
        return int(round(float(cfg['traj_time']) / delt))
    mass = float(np.atleast_1d(cfg['mass'])[0])
    if float(nucP_init) <= 0.0:
        raise ValueError(f'automatic run length needs nucP_init > 0 (got {nucP_init}); set traj_time')
    return int(np.ceil(2.0 * abs(float(cfg['nucR_init'])) * mass / (float(nucP_init) * delt)))


def run_trajectory(cfg, seed, idx=None, run_dir=None):
    points = scan_points(cfg)
    ic_seed, dyn_seed = _seeds(seed)
    home = os.getcwd()
    for name, over in points:
        Nprod = n_steps(cfg, over['nucP_init'])
        os.makedirs(name)
        os.chdir(name)
        try:
            nucR, nucP = initial_state(cfg, over['nucP_init'], ic_seed)
            m = _build(cfg, nucR, nucP, dyn_seed)
            m.init_map_spin(0)          # lower hemisphere: enter on the lower adiabat
            m.run_dynamics(Nprod, cfg['Nprint'], cfg['delt'], cfg['intype'],
                           init_time=0.0, small_dt_ratio=1)
        finally:
            os.chdir(home)
