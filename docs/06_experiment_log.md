# 6. Experiment log

All runs on a 4-core cloud container, Python 3.13, numpy 2.5, mpmath 1.3, python-flint 0.9 (Arb). Every number below can be regenerated with the command shown; JSON summaries are in `results/json/`, figures in `results/figures/`. "Rigorous" means ball/interval or exact arithmetic; everything else is float64 with the stated error estimate.

---

## E1. Rigorous zeros dataset (2026-11-05)

`python3 scripts/compute_zeros.py --start 1 --count 100000 --workers 4 --out data/zeros/zeros_1_100000`
`python3 scripts/compute_zeros.py --start 100001 --count 900000 --workers 2 --out data/zeros/zeros_100001_1000000`

Arb's `acb_dirichlet_zeta_zeros` isolates the n-th zero rigorously (Turing's method internally) and returns a ball $\tfrac12+i[\gamma_n\pm r]$. We store midpoints as float64 and the maximal radius in the manifest.

| dataset | zeros | height | max radius | wall time |
|---|---|---|---|---|
| `zeros_1_100000` | $1\ldots10^5$ | $\gamma_{10^5}=74920.827\,499$ | $1.5\times10^{-11}$ | 218 s (4 workers) |
| `zeros_100001_1000000` | $10^5+1\ldots10^6$ | $\gamma_{10^6}=600269.677\,012$ | $\le 6\times10^{-11}$ | ~50 min (2 workers) |
| `sample_1e10` | $10^{10}\ldots10^{10}+5999$ | $\approx3.29\times10^{9}$ | see manifest | (statistics worker) |

Throughput: ~150 zeros/s/core independent of height for $n\le10^6$; ~2 s/zero at $n=10^{12}$.

---

## E2. Nyman–Beurling–Báez-Duarte distances (2026-11-05)

`python3 scripts/run_nyman_beurling.py --N 2000 --arb 400 --tag _N2000` (105 s) and `--N 10000 --arb 800 --tag _N10000`.

**Method.** $d_N^2=1-b^TG_N^{-1}b$ with $G_{jk}=\int_0^\infty\{u/j\}\{u/k\}u^{-2}du$ from the Vasyunin closed form and $b_k=(1-\gamma_E+\log k)/k$; one Cholesky factorisation gives all $d_n^2$, $n\le N$. The closed form was *re-verified* against a direct evaluation of the integral (exact integration over the first $K$ periods + moment/Hurwitz-zeta tail) to $10^{-37}$ for many coprime pairs (the formula with $c_0(h/k)$-type cotangent sums in some references is **not** the right one; the correct one uses $V(h,k)=\sum_{m=1}^{k-1}\{mh/k\}\cot(\pi m/k)$, see `nyman_beurling.py`).

**Conditioning.** $\lambda_{\min}(G_N)\approx4.8\times10^{-5}$ at $N=160$, $\mathrm{cond}(G_{2000})=1.5\times10^{7}$: only polynomial growth ($\sim N^{2.2}$), so float64 suffices; rigorous Arb enclosures (50 digits) at $N=50,100,200,400$ agree with float64 to $\le 8\times10^{-15}$.

**Results ($N\le2000$).**

| $N$ | 10 | 20 | 50 | 100 | 200 | 500 | 1000 | 2000 |
|---|---|---|---|---|---|---|---|---|
| $d_N^2\log N$ | 0.05474 | 0.04944 | 0.04643 | 0.04692 | 0.04726 | 0.04611 | 0.04529 | 0.04647 |

Burnol's constant $C=2+\gamma_E-\log4\pi=0.046191$. Over $200\le N\le2000$: $\min d_N^2\log N=0.04419$ (at $N\approx700$), $\max=0.04690$. A one-parameter fit $d_N^2\log N=C_0+a/\log N$ on $N\ge200$ gives $C_0=0.04610$, $a=-0.003$ — i.e. the data are consistent with the conjecture $d_N^2\sim C/\log N$ to $0.2\%$ in the constant, but with *oscillations of $\pm5\%$ around it that dip below $C$* (permitted: Burnol's bound is a liminf statement).

**Structure of the drops.** $d_N^2$ is non-increasing; relative drops $d_{N-1}^2-d_N^2$ are largest when $N$ is **prime** (mean relative drop $8.4\times10^{-3}$ for primes vs $7\times10^{-4}$ for $N=pq$ and $9\times10^{-4}$ for non-squarefree $N$): each new prime lets the Dirichlet polynomial absorb a new Euler factor of $1/\zeta$. The sawtooth of $d_N^2\log N$ is the competition between these drops and the growth of $\log N$.

**Optimal coefficients.** At $N=2000$, $\operatorname{sign}(c_k)=-\mu(k)$ for **100 %** of squarefree $k$; mean $|c_k|$ is $0.286$ on squarefree and $0.066$ on non-squarefree $k$. Writing $c_k\approx-\mu(k)f(x)$, $x=\log k/\log N$, the empirical taper is *shallower* than the Bettin–Conrey–Farmer choice $1-x$: the binned medians of $-c_k/\mu(k)$ are $0.98,0.86,0.64,0.43,0.18$ at $x=0.03,0.25,0.5,0.75,0.98$, well fitted by $(1-x)^{0.45}$. Restricting to coefficients of the form $-\mu(k)(1-x)^\alpha$ (an explicit mollifier, hence an *upper bound* for $d_N^2$), the best exponent is $\alpha\approx0.50$ giving $U_N^2=0.0241$ at $N=2000$ versus $\alpha=1$: $0.0674$, and the true optimum $d_{2000}^2=0.00611$. So at this size the optimiser gains a factor 4 over the best Möbius power taper, largely through the non-squarefree coefficients and $k$-by-$k$ fluctuations (`scripts/analyze_nb_structure.py`; figures `nyman_beurling_coefficients_N2000.png`, `nyman_beurling_structure_N2000.png`).

Figures: `nyman_beurling_dN_logN_N2000.png`, `nyman_beurling_dN_N2000.png`, `nyman_beurling_coefficients_N2000.png`. *(N = 10 000 results: see E2b below when complete.)*

---

## E3. Báez-Duarte's sequential criterion and its explicit formula (2026-11-05)

`python3 scripts/run_sequential_criteria.py` (28 s)

$c_k=\sum_{j=0}^k(-1)^j\binom kj/\zeta(2j+2)=\sum_n\mu(n)n^{-2}(1-n^{-2})^k$; RH $\iff c_k=O(k^{-3/4+\varepsilon})$.

We derived (Mellin–Barnes with the Beta function) and verified the explicit formula
$$c_k=\sum_{\rho}\frac{B(1-\rho/2,\,k+1)}{2\zeta'(\rho)}+\sum_{m\ge1}\frac{B(1+m,\,k+1)}{2\zeta'(-2m)}+\ldots$$
(nontrivial and trivial zeros; $B$ = Beta function). With 8 nontrivial and 12 trivial zeros it reproduces the exact $c_k$ for $10\le k\le10^6$ (382 values) with **max absolute error $1.7\times10^{-9}$** (relative $1.1\times10^{-5}$ for $k\ge10^3$; the residual is the $k^{-3/2}$-type remainder and the omitted zeros).

Two facts worth recording:
* The **trivial-zero term** $-1/(2\zeta'(-2)(k+1)(k+2))\approx16.4/k^2$ dominates $c_k$ until $k\approx19\,758$, even though it is formally $O(k^{-2})\ll k^{-3/4}$: the nontrivial-zero amplitude carries the factor $|\Gamma(\tfrac34-i\gamma_1/2)|\approx1.2\times10^{-4}$. Any numerical test of the $k^{-3/4}$ law must subtract it.
* After subtracting it, $k^{3/4}c_k$ is a clean sinusoid in $\log k$ of frequency $\gamma_1/2$ with amplitude $7.78\times10^{-5}=|\Gamma(1-\rho_1/2)|/|\zeta'(\rho_1)|$ (max of $|k^{3/4}c_k|$ for $k\ge10^5$ is $1.05\times$ that, the excess being $\rho_2$'s contribution). **One sees the first zero of $\zeta$ directly in the sequence $c_k$.** A zero $\beta+i\gamma$ with $\beta<\tfrac12$ would add $k^{-(1+\beta)/2}e^{-\pi\gamma/4}$-sized oscillations; for $\gamma\gtrsim 60$ these are below $10^{-20}$ relative and invisible — the criterion, like all of them, is numerically blind at height.

Figure: `sequential_criteria.png`.

---

## E4. Li coefficients (2026-11-05)

`python3 scripts/run_li.py` (67 s). Module `li.py` (worker report verified by an independent check of its non-standard claim, see below).

* $\lambda_n$ for $n\le10^4$ from $10^5$ zeros ($\lambda_n=4\sum_{\gamma>0}\sin^2(n\arctan\frac1{2\gamma})$) plus an integral tail started at the point $T_{\rm eff}$ where the smooth count $N_{\rm smooth}(T_{\rm eff})=10^5$; tail error bounded via Abel summation and Trudgian's explicit bound on $S(T)$.
* Independent **rigorous** values for $n\le5000$ (Arb, 1810 digits, deflated zeta power series at $s=1$, series composition $z\mapsto z/(1-z)$): max relative discrepancy with the zero-sum method $3.6\times10^{-12}$ (at $n=5000$), $\lambda_1$ to $2\times10^{-15}$ (closed form $1+\gamma_E/2-\tfrac12\log4\pi$ reproduced to 60 digits).
* **All $\lambda_n>0$ for $n\le10^4$**, rigorously (ball lower bounds) for $n\le5000$; minimum is $\lambda_1=0.0230957089661210338\ldots$. First values: $\lambda_2=0.0923457352280466704$, $\lambda_3=0.2076389205543248038$, $\lambda_4=0.3687904794922416386$, $\lambda_5=0.5755427144611774524$; $\lambda_{10^4}=34736.58$.
* Asymptotics: fit of $\lambda_n-\tfrac n2\log n$ to $c_1n+c_0$ gives $c_1=-1.130358$ vs theory $\tfrac12(\gamma_E-1-\log2\pi)=-1.130331$.
* Archimedean part: the worker found, and we confirmed independently with the Bombieri–Lagarias formula in mpmath (80 digits), that $S_\infty(n)=\tfrac n2(H_n-1-\log2\pi)+\tfrac12$ up to exponentially small terms ($6\times10^{-14}$ at $n=150$), i.e. $\tfrac n2(\log n+\gamma_E-1-\log2\pi)+\tfrac34-\tfrac1{24n}+O(n^{-3})$. The constant term in the $\log n$ form is $3/4$.
* Oscillating part $S_2(n)=\lambda_n-S_\infty(n)$: mean $+0.96$, rms $5.6$, $\max|S_2|=19.3$ at $n=9947$, $|S_2|/\sqrt n<0.29$ for $n\ge100$ (consistent with $O(\sqrt n\log n)$ under RH).
* **Sensitivity calibration** (`li.detection_horizon`, `li.zero_contribution`). The mirror $\tfrac12-\delta+i\gamma$ of an off-line zero contributes a term of modulus $|1-1/\rho'|^n\approx\exp(n\delta/\gamma^2)$ to $\lambda_n$, so positivity can only fail once $n\delta/\gamma^2\gtrsim\log(\tfrac n2\log n)$. Exact evaluation for the Davenport–Heilbronn off-line zero $0.8085+85.699i$ (as if it were a zero of $\zeta$): its quadruple contributes $+7.9$ at $n=10^4$, $+38$ at $n=10^5$, $-4.2\times10^5$ at $n=3\times10^5$ and $-3\times10^{18}$ at $n=10^6$; the predicted horizon is $n^*\approx3.5\times10^5$. Horizons for a hypothetical zero with $\delta=0.1$: $n^*\approx1.6\times10^6$ at $\gamma=100$, $2\times10^8$ at $\gamma=10^3$, $2.7\times10^{10}$ at $\gamma=10^4$, $6.5\times10^{26}$ at $\gamma=10^{12}$. **Positivity of $\lambda_n$ for $n\le10^4$ is therefore insensitive to any off-line zero above height $\approx30$**; it is a consistency check of the first few zeros and the growth of $S(T)$, not independent evidence for RH.

Figures: `li_coefficients.png`, `li_oscillation.png`.

---

## E5. Integral criteria (Balazard–Saias–Yor, Volchkov) — running

`python3 scripts/run_integral_criteria.py`

*(to be filled in)*

---

## E6–E9. Zeta/Turing verification, arithmetic criteria, statistics, de Bruijn–Newman, explicit formula — workers running

*(to be filled in)*
