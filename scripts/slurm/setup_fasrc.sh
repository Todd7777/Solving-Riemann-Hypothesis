#!/bin/bash
# One-time setup on the cluster (run on a login node or in an interactive job):
#   bash scripts/slurm/setup_fasrc.sh
set -euo pipefail
module load python 2>/dev/null || true
python3 -m venv "${HOME}/venvs/rh"
source "${HOME}/venvs/rh/bin/activate"
pip install --upgrade pip
pip install numpy scipy mpmath sympy matplotlib pytest python-flint
python3 -c "import flint, mpmath, numpy, scipy; print('python-flint', flint.__version__, 'ok')"
# quick correctness check of the core modules (a few minutes)
python3 -m pytest tests/test_weil_form.py tests/test_package.py -q
