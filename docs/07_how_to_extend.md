# 7. How to extend the platform

## Conventions

* **One criterion = one module** in `src/rh/`, with a module docstring that states the theorem (with reference), the formulas used, and what is rigorous versus floating point.
* **Every module has an independent check** in `tests/`: a closed-form value, a high-precision recomputation by a different method, or a published number. Tests must run in well under a minute; heavy computations go to `scripts/run_<module>.py`.
* **Scripts write** `results/json/<module>_summary.json` (all headline numbers, with the parameters that produced them) and `results/figures/<module>_*.png`. Add the script to `scripts/run_all.py` with a `--quick` variant.
* **Zeros** come from `rh.data.load_zeros(n)` (Arb-certified ordinates, accuracy $\le 6\times10^{-11}$). Never hard-code zeros; never use zeros beyond the dataset's height without computing them with `rh.data.compute_zeros_arb`.
* **Rigorous arithmetic** where feasible: `python-flint` (`arb`, `acb`, `arb_mat`, `arb_series` — note `acb_series.zeta(deflate=True)` for expansions at $s=1$ and `arb.zeta_nzeros(arb(T))` for the rigorous count $N(T)$) or exact integers/fractions. Floating point results must carry an error estimate in the JSON summary.
* **Report conventions**: the experiment log (`docs/06_experiment_log.md`) records date, command, method, numbers with errors, and a one-paragraph interpretation. Say explicitly what the result does *not* show.

## Adding a new equivalent criterion

1. State it precisely in `docs/02_equivalent_formulations.md` with the reference.
2. Derive (or look up) a numerically benign formula. Many criteria are stated as alternating binomial sums or contour integrals that are hopeless in floating point; look for a Möbius/Dirichlet-series form (as for $c_k$ in `sequential_criteria.py`) or a Mellin–Barnes/explicit-formula form that exposes the dependence on the zeros.
3. Implement, test against a closed form, and — most importantly — **compute the sensitivity**: how large would the effect of a hypothetical off-line zero $\tfrac12+\delta+i\gamma$ be on the quantity you compute, at the sizes you can reach? (See `li.detection_horizon` and §4.4 of the barriers document.) If the answer is "invisible", say so in the log.
4. If a control is available (the Davenport–Heilbronn function, or $\zeta$ with a fake zero inserted), run it.

## Useful building blocks

| Need | Where |
|---|---|
| $Z(t)$, $\theta(t)$, Gram points, zero finding, $N(T)$ certificate | `rh.zeta`, `rh.verify` |
| rigorous zeros, counts | `rh.data`, `flint.acb.zeta_zeros`, `flint.arb.zeta_nzeros` |
| Vasyunin sums / Gram matrix of $\{1/(kx)\}$ | `rh.nyman_beurling` |
| tanh-sinh rule for log-singular integrands | `rh.integral_criteria.tanh_sinh_rule` |
| Möbius sieve, $\sigma(n)$ sieve, colossally abundant numbers | `rh.sequential_criteria.mobius_array`, `rh.arithmetic` |
| GUE spacing density / pair correlation | `rh.statistics` |
| explicit formula with Gaussian test functions | `rh.explicit_formula` |
| zero dynamics under the heat flow | `rh.debruijn_newman` |

## Ideas not yet implemented

* Weil positivity extremal test functions (optimise over a finite-dimensional family; the arithmetic side as a quadratic form; its smallest eigenvalue as a function of support length).
* Nyman–Beurling with the Mellin-side weight modified (Báez-Duarte's and Burnol's generalisations with $|s|^{-2}$ replaced by other weights; the constant in the lower bound changes to $\sum_\rho w(\rho)$).
* Li coefficients and $c_k$ for Dirichlet $L$-functions (python-flint has `acb.dirichlet_l` and `dirichlet_char`); GRH versions of every experiment.
* Higher moments of $\zeta$ on the line against the CFKRS conjecture, using the zero dataset for the comparison of $S(T)$ statistics.
