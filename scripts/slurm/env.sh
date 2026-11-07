# Environment for the cluster jobs. Created once by scripts/slurm/setup_fasrc.sh.
# Adjust the module name if your cluster provides a different Python (see `module avail python`).
module load python 2>/dev/null || true
source "${HOME}/venvs/rh/bin/activate"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p logs results/json/cluster
