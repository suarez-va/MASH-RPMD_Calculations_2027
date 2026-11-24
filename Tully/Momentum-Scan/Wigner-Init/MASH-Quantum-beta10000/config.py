"""
Tully 1 (single avoided crossing) momentum scan, classical (1 bead) MASH.  Centroid starts exactly at
(nucR_init, nucP_init) (centroid_init = 'deterministic'); internal ring-polymer modes thermal at beta. One trajectory runs every
nucP_init in `nucP_init_scan`, point v landing in nucP_<v>/shards/ (see ../../../transmission_scan.py).

    python -m workflow.runner --config config.py                             # Local
    python -m workflow.slurm  --config config.py && sbatch submit_grid.sh    # HPC
"""

CONFIG = {
    # ----------------------------------------------------------------- orchestration (workflow)
    'n_traj'      : 100000,                      # number of independent trajectories to run
    'base_seed'   : None,                        # int -> reproducible; None -> fresh OS entropy each trajectory
    'ncores'      : 10,                          # local cores for `python -m workflow.runner`
    'shard_size'  : 100,                         # trajectories per shards/shard_<s>.hdf (None -> one dir per trajectory)
    'shard_dir'   : 'shards',
    'system_file' : '../../../transmission_scan.py',
    'grid_dir'    : 'traj_grid',

    'slurm': {
        'max_tasks'     : 100,
        'job_name'      : 'MASHRPMD',
        'time'          : '39:00:00',
        'cpus_per_task' : 10,
        'max_concurrent': None,
        'partition'     : None,
        'account'       : 'gts-jkretchmer3-chemx',
        'mem_per_cpu'   : '16GB',
        'python'        : 'python',
        'extra_directives': [],
        'extra_commands': [
            'eval "$(/storage/home/hcoda1/8/vsuarez6/r-jkretchmer3-0/MiniConda/bin/conda shell.bash hook)"',
            'conda activate map-rpmd'],
    },

    # ------------------------------------------------------------------ dynamics
    'nbds'              : 16,                    # number of ring-polymer beads
    'jump_index'        : [],                    # no quantum jumps (as Tully-Delta)
    'jump_gap'          : None,                  # Egap trigger off (Tully gap |2A tanh(BR)| <= 0.02)
    'full_jump'         : True,                  # unused without jumps

    # ----------------------------------------------------------------- Tully single avoided crossing (a.u.)
    # V11 = A*tanh(B*R),  V22 = -V11,  V12 = V21 = C*exp(-D*R^2)
    'nnuc'              : 1,
    'nstates'           : 2,
    'mass'              : [2000.0],              # mass
    'beta'              : 10000.0,                 # inverse temperature (ring-polymer springs only)
    'A'                 : 0.01,
    'B'                 : 1.6,
    'C'                 : 0.005,
    'D'                 : 1.0,

    # ------------------------------------------------------------------ initial state + scan
    'centroid_init'     : 'wigner',              # centroid ~ Wigner of the Gaussian wavepacket
    'sigma_init'        : 1.0,                   # wavepacket width (= DVR-Libra sigma): R ~ N(nucR_init, sigma), P ~ N(nucP_init, 1/(2 sigma))
    'nucR_init'         : -10.0,                 # start left of the crossing at R = 0

    # Batch 1
    'nucP_init_scan'    : [2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9,
                           3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9,
                           4.0, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9,
                           5.0, 5.5, 6.0, 6.5],   # incident momenta
    'run_time_factor'   : 2.0,

    # Batch 2
    #'nucP_init_scan'    : [7.0, 7.5, 8.0, 8.5, 9.0, 9.5],   # incident momenta
    #'run_time_factor'   : 10.0,

    # Batch 3
    #'nucP_init_scan'    : [10.0, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0, 13.5, 14.0, 14.5,
    #                       15.0, 15.5, 16.0, 16.5, 17.0, 17.5, 18.0, 18.5, 19.0, 19.5,
    #                       20.0, 20.5, 21.0, 21.5, 22.0, 22.5, 23.0, 23.5, 24.0, 24.5,
    #                       25.0, 25.5, 26.0, 26.5, 27.0, 27.5, 28.0, 28.5, 29.0, 29.5, 30.0],   # incident momenta
    #'run_time_factor'   : 2.0,

    # ------------------------------------------------------------------ integrator
    'delt'              : 1.0,
    'intype'            : 'spin_magnus_adaptive',
    'Nprint'            : 5,
}
