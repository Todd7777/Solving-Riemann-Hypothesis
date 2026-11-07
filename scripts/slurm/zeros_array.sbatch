#!/bin/bash
# Rigorous zeta zeros 1,000,001 .. 10,000,000 (Arb), in 9 blocks of 10^6 zeros.
# Submit from the repository root:   sbatch --array=0-8 scripts/slurm/zeros_array.sbatch
# Each block uses all cores of its allocation; ~10^6 zeros take ~10 minutes on 16 cores.
#SBATCH --job-name=zeta_zeros
#SBATCH --partition=shared
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=0-06:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source scripts/slurm/env.sh

START=$((1000001 + SLURM_ARRAY_TASK_ID * 1000000))
END=$((START + 999999))
python3 scripts/compute_zeros.py --start "${START}" --count 1000000 \
    --workers "${SLURM_CPUS_PER_TASK}" --chunk 10000 \
    --out "data/zeros/zeros_${START}_${END}"
