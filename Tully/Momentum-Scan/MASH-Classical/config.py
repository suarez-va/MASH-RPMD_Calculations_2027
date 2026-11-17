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
    'nbds'              : 1,                     # number of ring-polymer beads
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
    'nucP_init_scan'    : [1.8, 2.0, 2.2, 2.4, 2.6, 2.8,
                           3.0, 3.2, 3.4, 3.6, 3.8,
                           4.0, 4.2, 4.4, 4.6, 4.8,
                           5.0, 5.2, 5.4, 5.6, 5.8,
                           6.0, 6.2, 6.4, 6.6, 6.8,
                           7.0, 7.2, 7.4, 7.6, 7.8,
                           8.0, 8.2, 8.4, 8.6, 8.8,
                           9.0, 9.2, 9.4, 9.6, 9.8,
                           10.0, 10.2, 10.4, 10.6, 10.8,
                           11.0, 11.2, 11.4, 11.6, 11.8,
                           12.0, 12.2, 12.4, 12.6, 12.8,
                           13.0, 13.2, 13.4, 13.6, 13.8,
                           14.0, 14.2, 14.4, 14.6, 14.8,
                           15.0, 15.2, 15.4, 15.6, 15.8,
                           16.0, 16.2, 16.4, 16.6, 16.8,
                           17.0, 17.2, 17.4, 17.6, 17.8,
                           18.0, 18.2, 18.4, 18.6, 18.8,
                           19.0, 19.2, 19.4, 19.6, 19.8,
                           20.0, 20.2, 20.4, 20.6, 20.8,
                           21.0, 21.2, 21.4, 21.6, 21.8,
                           22.0, 22.2, 22.4, 22.6, 22.8,
                           23.0, 23.2, 23.4, 23.6, 23.8,
                           24.0, 24.2, 24.4, 24.6, 24.8,
                           25.0, 25.2, 25.4, 25.6, 25.8,
                           26.0, 26.2, 26.4, 26.6, 26.8,
                           27.0, 27.2, 27.4, 27.6, 27.8,
                           28.0, 28.2, 28.4, 28.6, 28.8,
                           29.0, 29.2, 29.4, 29.6, 29.8, 30.0, 30.2],   # incident momenta

    # ------------------------------------------------------------------ integrator
    'delt'              : 5.0,
    'intype'            : 'spin_magnus_adaptive',
    'Nprint'            : 5,
}
