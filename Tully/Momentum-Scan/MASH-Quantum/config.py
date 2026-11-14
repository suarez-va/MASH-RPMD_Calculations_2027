"""
Tully 1 (single avoided crossing) momentum scan, classical (1 bead) MASH.  Delta-function initial
state (every bead at nucR_init, nucP_init) as in MASH-RPMD_Tully-Delta; one trajectory runs every
nucP_init in `nucP_init_scan`, point v landing in nucP_<v>/shards/ (see ../../tully_scan.py).

    python -m workflow.runner --config config.py                             # Local
    python -m workflow.slurm  --config config.py && sbatch submit_grid.sh    # HPC
"""

CONFIG = {
    # ----------------------------------------------------------------- orchestration (workflow)
    'n_traj'      : 10000,                    # number of independent trajectories to run
    'base_seed'   : None,                       # int -> reproducible; None -> fresh OS entropy each trajectory
    'ncores'      : 10,                         # local cores for `python -m workflow.runner`
    'shard_size'  : 10,                       # trajectories per shards/shard_<s>.hdf (None -> one dir per trajectory)
    'shard_dir'   : 'shards',
    'system_file' : '../../tully_scan.py',
    'grid_dir'    : 'traj_grid',

    'slurm': {
        'max_tasks'     : 100,
        'job_name'      : 'MASH',
        'time'          : '29:00:00',
        'cpus_per_task' : 10,
        'max_concurrent': None,
        'partition'     : None,
        'account'       : 'gts-jkretchmer3-chemx',
        'mem_per_cpu'   : '8GB',
        'python'        : 'python',
        'extra_directives': [],
        'extra_commands': [
            'eval "$(/storage/home/hcoda1/8/vsuarez6/r-jkretchmer3-0/MiniConda/bin/conda shell.bash hook)"',
            'conda activate map-rpmd'],
    },

    # ------------------------------------------------------------------ dynamics
    'nbds'              : 6,                     # number of ring-polymer beads
    'jump_index'        : [],                    # no quantum jumps (as Tully-Delta)
    'jump_gap'          : None,                  # Egap trigger off (Tully gap |2A tanh(BR)| <= 0.02)
    'full_jump'         : True,                  # unused without jumps

    # ----------------------------------------------------------------- Tully single avoided crossing (a.u.)
    # V11 = A*tanh(B*R),  V22 = -V11,  V12 = V21 = C*exp(-D*R^2)
    'nnuc'              : 1,
    'nstates'           : 2,
    'mass'              : [2000.0],              # mass
    'beta'              : 1.0,                   # inverse temperature (ring-polymer springs only)
    'A'                 : 0.01,
    'B'                 : 1.6,
    'C'                 : 0.005,
    'D'                 : 1.0,

    # ------------------------------------------------------------------ initial state + scan
    'nucR_init'         : -10.0,                 # start left of the crossing at R = 0
    'nucP_init_scan'    : [3.0, 4.0, 5.0, 6.0, 7.0],   # incident momenta

    # ------------------------------------------------------------------ integrator
    'delt'              : 5.0,
    'intype'            : 'spin_magnus_adaptive',
    'Nprint'            : 5,
}
