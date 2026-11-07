# Running the heavy computations on a Slurm cluster (e.g. FASRC Cannon)

These jobs reproduce and extend computations that are too slow for the 4-core container this
repository is developed in. They are plain Slurm batch scripts; nothing here needs network access
once the Python packages are installed.

## 1. One-time setup

```bash
ssh <your-username>@login.rc.fas.harvard.edu
git clone https://github.com/Todd7777/Solving-Riemann-Hypothesis.git
cd Solving-Riemann-Hypothesis
git checkout claude/gallant-albattani-z0p9mf
bash scripts/slurm/setup_fasrc.sh
```

`setup_fasrc.sh` creates a virtual environment in `~/venvs/rh` and runs a few tests. If your cluster
needs a specific module (`module avail python`), edit `scripts/slurm/env.sh` and `setup_fasrc.sh`.

## 2. Jobs

| job | what it computes | submit | cost |
|---|---|---|---|
| `weil_form_array.sbatch` | smallest eigenvalues of the restricted Weil form and the "zeros from primes" capture, for the 14 `(a, J)` cases in `weil_form_cases.txt` (up to a = 2, J = 384, 500+ digits) | `sbatch --array=0-13 scripts/slurm/weil_form_array.sbatch` | 1 core per task; minutes for small cases, about 1–2 days for the largest |
| `zeros_array.sbatch` | rigorous (Arb) zeta zeros 1,000,001 … 10,000,000 in 9 blocks | `sbatch --array=0-8 scripts/slurm/zeros_array.sbatch` | 16 cores per task, about 10–30 min each |

Partition names, time limits and memory are set for FASRC's `shared` partition; for many short
tasks `serial_requeue` is usually faster to schedule. Logs go to `logs/`.

## 3. Returning the results

```bash
git add results/json/cluster/*.json data/zeros/zeros_*_*.npy data/zeros/zeros_*_*.json
git commit -m "Cluster results: Weil-form cases and zeros to 1e7"
git push origin claude/gallant-albattani-z0p9mf
```

The zero blocks are 8 MB each (72 MB in total). If you prefer not to put them in git, push only
the `results/json/cluster/*.json` files and keep the zero blocks on the cluster.

Each result file records its own parameters, working precision, timings and Slurm job id, and each
case checks itself: if the smallest eigenvalue approaches the precision floor, the case is rerun at
1.5x the precision automatically.
