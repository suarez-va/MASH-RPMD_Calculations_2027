"""
Driver: run the current batch of the incident-momentum (p0) scan locally -- the config's scan.p0
list at its scan.run_time_factor. Each point gets dyn.num_threads cores (pinned with taskset), and
runtime.ncores // dyn.num_threads points run at once.

    python3 1_run_p0_scan_local.py                # run whatever of this batch is not yet done
    python3 1_run_p0_scan_local.py --force        # re-run this batch from scratch
    python3 1_run_p0_scan_local.py --ncores 12    # total cores to use (override config.yaml)
    python3 1_run_p0_scan_local.py --threads 2    # cores per point (override dyn.num_threads)
    python3 1_run_p0_scan_local.py --only 4.5 8.0 # just these p0 values (of this batch)
    python3 1_run_p0_scan_local.py --config config_smoke.yaml   # cheap test, own grid_dir

Batches (as the MASH Momentum-Scan configs): leave one batch's scan.p0 / scan.run_time_factor
uncommented in config.yaml, run this, run 2_collect.py, switch to the next batch. Points are named
by value (p0_grid/p0_<p0>/), so batches never collide and 2_collect.py gathers all of them.

Already-completed points (those with a result.json) are skipped unless --force, so the scan is
restartable - if it is interrupted, or a few points fail, just run it again. A finished point is
NOT rerun when its batch's run_time_factor changes: use --force (or --only ... --force) for that.
"""

import os
import sys
import argparse

import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_grid import run_grid_local, prepare_grid
from run_dvr_point import p0_grid, point_name, grid_path

script_dir = os.path.dirname(os.path.abspath(__file__))

ap = argparse.ArgumentParser()
ap.add_argument("--ncores", type=int, default=None, help="total cores (override runtime.ncores)")
ap.add_argument("--threads", type=int, default=None,
                help="cores per point (override dyn.num_threads)")
ap.add_argument("--force", action="store_true", help="re-run points that already have results")
ap.add_argument("--only", type=float, nargs="+", default=None, help="run only these p0 values")
ap.add_argument("--config", default="config.yaml",
                help="config file to use (e.g. config_smoke.yaml for a fast dry run)")
args = ap.parse_args()

cfg_path = args.config if os.path.isabs(args.config) else os.path.join(script_dir, args.config)
with open(cfg_path) as f:
    cfg = yaml.safe_load(f)

p0_ar = p0_grid(cfg)
ncores = args.ncores if args.ncores is not None else cfg["runtime"]["ncores"]
threads = args.threads if args.threads is not None else cfg["dyn"]["num_threads"]

grid_dir = grid_path(cfg)
names = [point_name(p) for p in p0_ar]
only = None
if args.only is not None:
    bad = [p for p in args.only if point_name(p) not in names]
    if bad:
        raise SystemExit(f"--only {bad}: not in this batch's scan.p0 {list(p0_ar)}")
    only = [names.index(point_name(p)) for p in args.only]
todo = prepare_grid(grid_dir, names, only=only, force=args.force)
if not todo:
    print("nothing to do - every requested point already has a result.json "
          "(use --force to recompute)")
    sys.exit(0)

npoints = len(p0_ar)
args_list = [["--p0", repr(float(p)), "--config", cfg_path, "--num-threads", str(threads)]
             for p in p0_ar]
cwd_list = [script_dir] * npoints                  # the point script chdir's itself
log_list = [os.path.join(grid_dir, n, "run.log") for n in names]

print(f"batch: {npoints} p0 values at run_time_factor {cfg['scan']['run_time_factor']}: {[float(p) for p in p0_ar]}")
print(f"output -> {grid_dir}")
print(f"running: {[float(p0_ar[i]) for i in todo]}")

codes = run_grid_local(os.path.join(script_dir, "run_dvr_point.py"),
                       args_list, ncores=ncores, threads_per_job=threads,
                       cwd_list=cwd_list, log_list=log_list, indices=todo)

print("\nnext: python3 2_collect.py" + ("" if args.config == "config.yaml" else f" --config {args.config}"))
sys.exit(1 if any(rc != 0 for rc in codes.values()) else 0)
