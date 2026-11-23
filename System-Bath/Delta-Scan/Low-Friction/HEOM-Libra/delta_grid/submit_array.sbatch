#!/bin/bash
#SBATCH --job-name=HEOM
#SBATCH --array=0-99
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=/storage/project/r-jkretchmer3-0/vsuarez6/repos/GET_BETTER_WITH_HPC/MASH-RPMD_Calculations_2027/System-Bath/Delta-Scan/Low-Friction/HEOM-Libra/delta_grid/val%a/slurm-%A_%a.log
#SBATCH --time=29:00:00
#SBATCH --mem=100G
#SBATCH --account=gts-jkretchmer3-chemx

set -eo pipefail

# ---- environment setup (slurm.setup in the config) ----
# export PYTHONPATH=/storage/home/hcoda1/8/vsuarez6/r-jkretchmer3-0/repos/libra-code/_build/src:$PYTHONPATH
# eval "$(/storage/home/hcoda1/8/vsuarez6/r-jkretchmer3-0/MiniConda/bin/conda shell.bash hook)"
# conda activate libra

# one point per array task; it gets exactly the cores SLURM allocated
NT=${SLURM_CPUS_PER_TASK:-4}
for v in OMP_NUM_THREADS OPENBLAS_NUM_THREADS MKL_NUM_THREADS NUMEXPR_NUM_THREADS VECLIB_MAXIMUM_THREADS NUMBA_NUM_THREADS; do export $v=$NT; done

echo "job ${SLURM_ARRAY_JOB_ID:-?}_${SLURM_ARRAY_TASK_ID} on $(hostname), $NT thread(s), started $(date)"
cd /storage/project/r-jkretchmer3-0/vsuarez6/repos/GET_BETTER_WITH_HPC/MASH-RPMD_Calculations_2027/System-Bath/Delta-Scan/Low-Friction/HEOM-Libra
python3 /storage/project/r-jkretchmer3-0/vsuarez6/repos/GET_BETTER_WITH_HPC/MASH-RPMD_Calculations_2027/System-Bath/Delta-Scan/Low-Friction/HEOM-Libra/run_heom_point.py --idx "$SLURM_ARRAY_TASK_ID" --config /storage/project/r-jkretchmer3-0/vsuarez6/repos/GET_BETTER_WITH_HPC/MASH-RPMD_Calculations_2027/System-Bath/Delta-Scan/Low-Friction/HEOM-Libra/config.yaml --num-threads "$NT"
echo "finished $(date)"
