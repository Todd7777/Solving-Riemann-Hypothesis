#!/bin/bash
# Weil-form convergence / zero-capture study: one (a, J) case per array task.
# Submit from the repository root:   sbatch --array=0-13 scripts/slurm/weil_form_array.sbatch
# Each task is single-threaded (python-flint + mpmath); big cases take hours, so they get
# a generous time limit. Edit the partition/time/memory to suit your allocation.
#SBATCH --job-name=weil_form
#SBATCH --partition=shared
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source scripts/slurm/env.sh

# pick the line for this task (skip comments)
LINE=$(grep -v '^#' scripts/slurm/weil_form_cases.txt | sed -n "$((SLURM_ARRAY_TASK_ID + 1))p")
read -r A J EXTRA <<<"${LINE}"
OUT="results/json/cluster/weil_a${A}_J${J}.json"
echo "task ${SLURM_ARRAY_TASK_ID}: a=${A} J=${J} extra='${EXTRA:-}' -> ${OUT}"
python3 scripts/run_weil_form_case.py --a "${A}" --J "${J}" ${EXTRA:-} --out "${OUT}"
