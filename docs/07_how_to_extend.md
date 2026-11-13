# 7. How to extend the platform

## Conventions

* **One criterion = one module** in `src/rh/`, with a module docstring that states the theorem (with reference), the formulas used, and what is rigorous versus floating point.
* **Every module has an independent check** in `tests/`: a closed-form value, a high-precision recomputation by a different method, or a published number. Tests must run in well under a minute; heavy computations go to `scripts/run_<module>.py`.
* **Scripts write** `results/json/<module>_summary.json` (all headline numbers, with the parameters that produced them) and `results/figures/<module>_*.png`. Add the script to `scripts/run_all.py` with a `--quick` variant.
* **Zeros** come from `rh.data.load_zeros(n)` (Arb-certified ordinates, ball radius $\le1.5\times10^{-11}$ for the first $10^5$ and $\le1.2\times10^{-10}$ for zeros up to $10^6$). Never hard-code zeros; never use zeros beyond the dataset's height without computing them with `rh.data.compute_zeros_arb`.
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
| Weil quadratic form on $[-a,a]$ in ball arithmetic, minimiser zeros | `rh.weil_form` |
| Weil form for ζ, $L(s,(n/5))$ and Davenport–Heilbronn; wave-packet bases; zero side with off-line quadruples; hypothetical zeros | `rh.weil_form_general` |
| certified enclosure of Weil's form at one explicit test function (every error bounded) | `rh.weil_certificate` |
| certified positivity of Weil's form on a whole window $[-L,L]$ (envelope reduction; rigorous spherical Bessel functions to high order; residual-certified blocked $LDL^T$; optional comb certificate) | `rh.weil_envelope` |
| non-pointwise comb certificates $A'-P_L+H\ge0$ with $H$ of spectrum outside $(-2L,2L)$: separable (per prime, Fejér–Egerváry–Szász / Toeplitz) and joint (torus sums of squares over smooth numbers); exact dyadic coefficients | `rh.weil_comb` |
| trajectory check of $\Omega-P_L+H\ge\beta$ on $[T,T_2]$ (IEEE double with a-priori error bounds, phases from Arb, second-derivative bound; used by `weil_envelope.beta_star`) | `rh.weil_comb.trajectory_check` |
| wave packets $e(s)\cos(ws)$, $e=(\sin(bs/m)/(bs/m))^m$, $w-b\ge2L$: localised parts of $H$; LP fit, Arb and certified double evaluation, ellipse and curvature bounds | `rh.weil_local` |
| gapped comb certificates: the terms of $H$ with frequency in $[2L,2L+g)$ moved into $A'$ (they resonate with the frequency-$2L$ part of $\lvert F\rvert^2$ on $[T,\infty)$, E16) | `rh.weil_comb.gap_certificate` |
| per-dip wave packets: one min-$\sum\lvert a\rvert$ linear program per group of dips of $\Omega-\beta-P_L+H$, carriers $\ge2L+b+$gap | `rh.weil_local.fix_dips` |
| explicit formula for real periodic Dirichlet series (no Euler product needed) | `rh.explicit_formula_general` |
| Nyman–Beurling distances for periodic-coefficient Dirichlet series | `rh.nyman_beurling_periodic` |
| zero dynamics under the heat flow | `rh.debruijn_newman` |

## Ideas not yet implemented

* Weil form beyond $L=1.8$ (E15–E18). Torus certificates are limited by the operator norm $A^*$; trajectory certificates are not, and wave packets bring the comb's requirement down to the local operator norm. What limits $T$ now is the envelope floor: below it the envelope form is indefinite even with a valid $\beta$. It sits near $2\pi e^{1+2L}$ for $L\le1.2$, and well above it for larger $L$ and large $H$ (above $T=2460$ at $L=1.7$). E16 found that the part that grows with $H$ comes from a few terms of $H$ just above frequency $2L$: build certificates without them (`gap_certificate`) and with $\beta$ as large as the trajectory allows; that certifies $[-1.7,1.7]$ at $T=2460$, and with packets of carriers up to $2L+b+8$ (`fix_dips(W=8, core_pad=..., far_step=..., buffer=...)`) from $T=1000$ (E17). The same steps certify $[-1.8,1.8]$ from $T=2000$ (E18; `fix_dips(core_per_dip=True)` when one group holds dips far apart). The floor there lies between $2.4$ and $3.2\,t_0$, against below $1.95\,t_0$ at $L=1.7$, so the mode count grows faster than $e^{2L}$. Choose the mode count so that the tail bound $\approx(eLT/4N)^{2N}$ stays far below the expected margin. The certification keeps $R_A$ in blocks (`storage="blocks"`, the default of `certify_window(s)` since E19; `envelope_block(s)` take the same option for probes): about a third of the dense path's memory at $N\approx1000$ and less at larger $N$. Pass `threads=3` (FLINT's products). The chunk (nodes per flush of the buffers) matters while $R$ is small: at $N\approx1000$ chunks of 512 cut the peak by a third, while for $N$ in the thousands the factorisation sets the peak and chunks of 1024 make the flushes about 10% faster. Directions: a reduction that keeps part of the spectral side above $T$ (for instance on the near-null directions only); certificates whose $H$ is small where the near-minimisers live; a prolate basis (about $2LT/\pi$ modes instead of $\approx1.5LT$), which would make larger $T$ affordable. Cheaper steps: larger joint certificates (bigger $\Lambda$, or the separable cross of high prime powers added to the smooth-number monomials) to close more of the gap between the separable constant and $A^*$; a prolate (band-limited) basis instead of Legendre polynomials, which needs about $2LT/\pi$ modes instead of $\approx1.7LT$.
* The same certificates for $L(s,(n/5))$ (comb $\chi(n)\Lambda(n)$; the per-prime certificates take signed coefficients unchanged) to quantify how the margins depend on the conductor.
* Nyman–Beurling with the Mellin-side weight modified (Báez-Duarte's and Burnol's generalisations with $|s|^{-2}$ replaced by other weights; the constant in the lower bound changes to $\sum_\rho w(\rho)$).
* Li coefficients and $c_k$ for Dirichlet $L$-functions (python-flint has `acb.dirichlet_l` and `dirichlet_char`); GRH versions of every experiment.
* Higher moments of $\zeta$ on the line against the CFKRS conjecture, using the zero dataset for the comparison of $S(T)$ statistics.
