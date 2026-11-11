# 5. The research program of this repository

## 5.1 Honest framing

The Riemann Hypothesis will not be settled by a software project, and this repository does not claim to settle it. What a well-built computational platform *can* do:

1. **Remove doubt about the easy part.** Reproduce, from scratch and with rigorous arithmetic, the facts every approach relies on (zeros to height $\sim6\times10^5$ and a sample at height $\sim3.3\times10^{9}$; zero statistics; the elementary criteria to large bounds).
2. **Make the equivalent criteria quantitative.** Each "RH $\iff$ $X$" turns RH into the asymptotics of an explicit sequence ($\lambda_n$, $d_N$, $c_k$, Robin ratios of colossally abundant numbers, …). Computing these sequences far enough to see their *second-order* behaviour produces conjectures sharper than RH itself, and the sharper conjectures are often where structure (and hence a possible proof) becomes visible.
3. **Find structure in positive-definite finite-dimensional objects.** Two of the criteria (Weil positivity and Nyman–Beurling) reduce RH to positivity/asymptotics of explicit Gram-type matrices. We can compute these matrices exactly (Vasyunin sums; explicit formula terms) and study their spectra, optimal vectors and extremal test functions.
4. **Provide a trustworthy, tested code base** so that any future idea can be tested in minutes rather than weeks.

## 5.2 Work packages and status

| # | Package | Module(s) | Status |
|---|---|---|---|
| A | Rigorous zeros dataset ($10^6$ + high samples) | `scripts/compute_zeros.py`, `data.py` | done (Arb, ball radius $\le 3\times10^{-11}$) |
| B | Independent Riemann–Siegel + Turing verification | `zeta.py`, `verify.py` | done, cross-checked against Arb |
| C | GUE statistics, height dependence | `statistics.py` | done |
| D | Li coefficients to $n=10^4$; high-precision cross-check | `li.py` | done |
| E | Nyman–Beurling $d_N$ to $N=10^4$ (rigorous to 800) and coefficient structure | `nyman_beurling.py` | done |
| F | Robin / Lagarias / Nicolas / CA numbers; Speiser; Davenport–Heilbronn | `arithmetic.py`, `speiser.py`, `davenport_heilbronn.py` | done |
| G | de Bruijn–Newman dynamics and Lehmer pairs | `debruijn_newman.py` | done |
| H | Explicit formula and Weil positivity numerics | `explicit_formula.py` | done |
| I | Integral and sequential criteria (Balazard–Saias–Yor, Volchkov, Báez-Duarte's $c_k$) | `integral_criteria.py`, `sequential_criteria.py` | done |
| J | Spectral analysis of Weil's quadratic form on bounded supports | `weil_form.py` | done (E10) |
| K | Sensitivity of Nyman–Beurling to off-line zeros; Davenport–Heilbronn and $L(s,(\cdot/5))$ controls | `nyman_beurling_periodic.py` | done (E11) |
| L | Explicit formula without an Euler product; Weil positivity for $L(s,(\cdot/5))$ and Davenport–Heilbronn; detection horizon of Weil's criterion; certified values of the Weil form | `explicit_formula_general.py`, `weil_form_general.py`, `weil_certificate.py` | done (E12) |
| M | Certified Weil positivity on whole windows $[-L,L]$ (envelope reduction in ball arithmetic) and the reach of the method | `weil_envelope.py` | done (E13) |
| N | Non-pointwise bounds for the prime comb (separable and joint sum-of-squares certificates $A'-P_L+H\ge0$, $H$ invisible to the window); certified windows past the pointwise barrier; the operator-norm barrier | `weil_comb.py` | done (E14) |
| O | Trajectory certificates (the comb bound checked only along $t\ge T$), local operator norms, wave packets; the envelope floor that then blocks $[-1.7,1.7]$ | `weil_comb.py` (`trajectory_check`), `weil_local.py` | done (E15) |
| P | The part of the envelope floor that grows with $H$: a resonance of the terms just above frequency $2L$; thresholds $\beta^*(T)$; gapped certificates and per-dip wave packets; $[-1.7,1.7]$ certified | `weil_comb.py` (`gap_certificate`), `weil_local.py` (`fix_dips`) | done (E16) |

The experiment log ([06_experiment_log.md](06_experiment_log.md)) records every number produced, with the command that produced it.

## 5.3 Specific questions we are pursuing

**Q1 (Nyman–Beurling second-order behaviour).** Burnol: $\liminf d_N^2\log N\ge C=2+\gamma_E-\log4\pi$. Conjecture: $d_N^2\log N\to C$. Our data show $d_N^2\log N$ *oscillating around $C$* with excursions of a few percent (last above $C$ at $N=2679$, then $1$–$4\%$ *below* $C$ up to $N=10^4$ — allowed by a liminf statement, which however forces an eventual rise). Questions: what drives the oscillation (the sharp drops of $d_N^2$ occur at primes only for $N\lesssim50$; beyond that no arithmetic class of $N$ is privileged), is the amplitude decaying, and is there a second-order term $d_N^2=\frac C{\log N}+\frac{c_2+\text{osc}(N)}{\log^2N}+\dots$? A formula for the second-order term would be a new conjecture in the spirit of Bettin–Conrey–Farmer.

**Q2 (optimal coefficients vs Möbius).** The optimiser's coefficients $c_k$ have sign $-\mu(k)$ for *every* squarefree $k$ (100 % agreement at $N=2000$) and are small on non-squarefree $k$. What is the empirical taper $f(\log k/\log N)$ in $c_k\approx-\mu(k)f(\log k/\log N)$? Bettin–Conrey–Farmer use $f(x)=1-x$; the data suggest something steeper at $x\to1$. Knowing the optimal taper precisely gives a better *unconditional upper bound* for $d_N^2$ in terms of explicit quantities — potentially provable without RH for a specific family of mollifiers.

**Q3 (Li coefficients fluctuation).** Under RH, $\lambda_n=\frac n2\log n+c_1n+O(\sqrt n\log n)$ with explicit $c_1$. The fluctuating part is a trigonometric sum over zeros, $-2\sum_\gamma\cos(n\theta_\gamma)$ with $\theta_\gamma=2\arctan(1/2\gamma)$; its size for $n\le10^4$ tests the GUE-based prediction of the variance (Lagarias; Coffey). A failure of RH would appear as *exponential* growth of a single mode; we quantify how far from that the data are.

**Q4 (Weil positivity structure).** *Partly answered by E10:* on polynomial test functions supported in $[-a,a]$, the smallest eigenvalue of the arithmetic side is positive for every $a\le2.5$ tried, collapses super-exponentially (below $10^{-61}$ at $a=2$), and its eigenvector's Fourier transform vanishes at the low zeros of ζ ($\gamma_1$ to 44 digits from the prime powers $n\le54$). The original question follows. For the Gaussian family $g_\sigma$, which $\sigma$ minimises the ratio (arithmetic side)/(archimedean side) of $W(g*\tilde g)$? How much of the prime term is cancelled by the archimedean term and how much by the pole term? The extremal test functions of the Weil form are those for which positivity is *tight*; the literature (Bombieri 2000; Connes–Consani 2021) suggests they are related to the small-support regime. Numerics can map this landscape.

**Q5 (GUE deviations).** The lower-order corrections to GUE (Bogomolny–Keating; Conrey–Snaith) are arithmetic: they contain the primes. We measure the spacing-distribution deviation in five windows up to height $6\times10^5$ and at $\sim3.3\times10^9$ and compare with the $1/\log T$ scaling.

**Q6 (Davenport–Heilbronn as a control).** *Answered for Nyman–Beurling (E11):* the Davenport–Heilbronn distances are indistinguishable from those of $L(s,(n/5))$ for $N\le2000$; the violation shows only near $N\approx10^{302}$. *Answered for Weil positivity (E12):* the Davenport–Heilbronn form turns negative at support $a\approx1.88$, where only the coefficients $n\le42$ enter. This is certified rigorously for explicit test functions, while $L(s,(n/5))$ stays positive on the same spaces. For ζ with a hypothetical off-line zero at height $\gamma$, the threshold grows like $\tfrac12\log(\gamma/2\pi)$: primes up to $\approx\gamma$ suffice. Weil's criterion is by far the most sensitive of those computed here. The original question follows. Every criterion we compute for $\zeta$ can also be computed for the DH function $f$, which violates RH. Which criteria *detect* the violation at computationally accessible scale (e.g. Li-type coefficients for $f$ grow exponentially; the analogue of $d_N$ stalls)? This calibrates how sensitive each criterion is — a necessary sanity check before reading anything into numerics for $\zeta$.

## 5.4 What would constitute progress

* A **proved** inequality of the form $d_N^2\le \Phi(N)\to0$ for an explicit family of coefficients (this is RH, so we do not expect it); or a proved second-order lower bound improving Burnol's constant conditionally on the pair-correlation conjecture (a realistic research-paper-level goal).
* A **conjecture with numerical support** for the oscillatory term in $d_N^2\log N$ or in $\lambda_n$, with an explanation in terms of the zeros or the primes.
* A **negative result**: showing that some proposed "proof strategy" (e.g. a specific Hilbert–Pólya candidate operator) is numerically inconsistent with the zeros. The repository makes such checks easy.

## 5.5 Non-goals

* No claim of proof. No "proof sketches" that ignore §4.
* No reliance on floating-point results where exact or ball arithmetic is feasible; any float result is reported with an error estimate.
