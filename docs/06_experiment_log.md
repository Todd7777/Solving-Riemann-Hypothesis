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

## E5. Integral criteria: Balazard–Saias–Yor and Volchkov (2026-11-05)

`python3 scripts/run_integral_criteria.py --nzeros 20000` (BSY 644 s with Arb's $\zeta$ at 0.3 ms per evaluation; Volchkov 140 s). Module `integral_criteria.py` (tanh-sinh rule with 81–127 nodes per panel between consecutive zeros, robust to the logarithmic endpoint singularities).

**Balazard–Saias–Yor.** $I(T)=\frac1\pi\int_0^T\frac{\log|\zeta(1/2+it)|}{1/4+t^2}dt$, which equals $\sum_{\Re\rho>1/2}\log|\rho/(1-\rho)|$ in the limit, hence $0$ under RH:

| $T$ (at zero no.) | 49.8 (10) | 236.5 (100) | 1419 (1000) | 9878 (10 000) | 18 046 (20 000) |
|---|---|---|---|---|---|
| $I(T)$ | $-5.5\times10^{-5}$ | $6.4\times10^{-6}$ | $-3.6\times10^{-8}$ | $-8.2\times10^{-11}$ | $-7.2\times10^{-10}$ |

The identity holds to ten decimals; the truncation error decays much faster than the naive $O(1/T)$ because $\log|\zeta|$ has mean zero on the line (Selberg), so the tail cancels. Figure `integral_criteria_bsy.png`.

**Volchkov.** $\int_0^\infty\frac{1-12t^2}{(1+4t^2)^3}\int_{1/2}^\infty\log|\zeta(\sigma+it)|\,d\sigma\,dt=\pi\frac{3-\gamma_E}{32}=0.237\,856\,295\,887$. Two numerical traps had to be removed to see the identity: the pole at $s=1$ is an interior logarithmic singularity of the inner integrand for $t\to0$ (splitting the $\sigma$-range at $\sigma=1$ removed a systematic $-2.7\times10^{-4}$), and near each zero the inner integrand peaks sharply at $\sigma\to\tfrac12$ (subtracting $\log|s-\rho_j|$ for the nearest zero analytically). Result at $T=200$, $\sigma_{\max}=30$, (32, 64) nodes: $0.237\,856\,381$, error $8.5\times10^{-8}$, independent of $T\in[60,300]$ (quadrature-order limited: $3.5\times10^{-6}$ with (24, 48) nodes).

**Interpretation.** Both identities are confirmed to the accuracy of the quadrature. As calibrated in §4.4 of the barriers document, a zero $\tfrac12+\delta+i\gamma$ would change $I$ by $\approx\delta/(\tfrac14+\gamma^2)$: at $\gamma=10^4$ and $\delta=0.1$ that is $10^{-9}$ — at the level of our truncation error, and beyond that height invisible. The integral criteria are exact reformulations but numerically the least sensitive of all.

---

## E6. Independent Riemann–Siegel implementation and Turing verification (2026-11-05)

`python3 scripts/run_verify.py` (8.6 s). Modules `zeta.py` (839 lines), `verify.py`; 68 tests.

* **Riemann–Siegel** $Z(t)$ with correction terms $C_0..C_4$ (coefficient series generated at import from exact power-series division of $\Psi(p)$, reproducing Haselgrove's tables to 20 digits) and Gabcke's remainder bounds; Euler–Maclaurin with explicit remainder below $t=500$. $\theta(t)$ from the Bernoulli-number expansion through $t^{-13}$ (mpmath below $t=12$). Gram points by Newton; zero search on a grid uniform in the Gram coordinate $\theta/\pi$ with Rosser-rule block refinement (this catches the Lehmer pair $7005.0629/7005.1006$).
* **Turing's method** with Lehman's constants ($|S_1(t_2)-S_1(t_1)|\le2.30+0.128\log(t_2/2\pi)$ for $168\pi<t_1<t_2$; Trudgian's smaller constants are available but flagged as quoted from memory): certify $N(g_m)=m+1$ from $m+1$ sign changes below $g_m$ and the integrated inequality over $k$ Gram intervals above.
* **Result.** $N(74920.9)=100\,000$ **certified** (7 Gram intervals, quotient $(B-F)/L=0.929<1$): all 100 000 zeros below $g_{99999}=74921.2257$ are simple and on the critical line. 100 082 zeros found on $[0,74976]$; 34 Gram blocks needed refinement, none unresolved; all brackets certified without mpmath fallback. Cross-check: Arb's rigorous count agrees, and $\max|\gamma_{\rm found}-\gamma_{\rm Arb}|=5.8\times10^{-10}$ over all $10^5$ zeros (float64 error; the Arb radius is $1.5\times10^{-11}$).
* **Rigour boundary** (stated in the JSON): the theorem and the truncation bounds are rigorous; float64 rounding in $Z$, $\theta$, Gram points and the quadrature is estimated (10–100× conservatively) but not bounded — a fully rigorous version is exactly what Arb does, which we keep as the independent reference.

---

## E7. Arithmetic criteria, Speiser's criterion, Davenport–Heilbronn control (2026-11-05)

`python3 scripts/run_arithmetic.py --workers 3` (259 s). Modules `arithmetic.py`, `speiser.py`, `davenport_heilbronn.py`; 23 tests.

**Robin** ($N=10^7$, exact integer sieve): the violators of $\sigma(n)<e^{\gamma_E}n\log\log n$ for $n\ge2$ are exactly the 27 known values $\{2,3,4,\dots,2520,5040\}$. Beyond 5040 the maximum of $\sigma(n)/(e^{\gamma_E}n\log\log n)$ is $0.98582$ at $n=10080$ (then 55440: 0.98325; 27720: 0.97836).

**Lagarias** ($N=10^7$): no violators; equality only at $n=1$; minimum slack $0.3172$ at $n=2$; maximum of $\sigma(n)/(H_n+e^{H_n}\log H_n)$ is $0.98864$ at $n=12$.

**Nicolas** ($K=10^4$ primorials, 50 digits): $r_k=\frac{N_k/\varphi(N_k)}{e^{\gamma_E}\log\log N_k}$ is $>1$ and **strictly decreasing for all $k\ge2$**, from $2.888$ to $1.000\,564$ at $k=10^4$ ($p_k=104\,729$); empirically $(r_k-1)\sqrt{p_k}\log p_k\approx2.1$ (stable from $k=2000$ to $10^4$), i.e. $r_k-1\approx2.1/(\sqrt{p_k}\log p_k)$ — the rate predicted by the prime number theorem with square-root error.

**Colossally abundant numbers** (first 200, exact exponent vectors via the critical-$\varepsilon$ chain, each re-derived from the closed-form exponents; the first 21 match OEIS A004490; brute force agrees for the first 10): the 200th has 433 digits and 165 prime factors. Robin's ratio is $<1$ for every CA number $>5040$ and increases (with 12 small dips among 191 steps) from $0.98325$ (55440) to $0.995794$ (#200); **$(1-\text{ratio})\sqrt{\log n}\approx0.13$ is stable for $150\le\log n\le995$**, i.e. $1-\text{ratio}\approx0.13/\sqrt{\log n}$ (Robin's inequality is approached but with margin $\asymp(\log n)^{-1/2}$, as expected from $\sigma(n)/n=e^{\gamma_E}\log\log n-c/\sqrt{\log n}+\ldots$ for CA numbers under RH). Lagarias holds for all 200 with log-slack decreasing to $0.0043$. Gronwall: $\sigma(n)/(n\log\log n)=1.77358$ at #200 vs $e^{\gamma_E}=1.78107$.

**Speiser** ($T=2000$, rectangle $-1\le\sigma\le4$, $0.25\le t\le2000$; the double pole at $s=1$ gives winding $-2$ on a small box): **1298 zeros of $\zeta'$, all certified by Krawczyk's test with rigorous bounds, all with $\Re s>\tfrac12$**; minimum $\Re s=0.503428$ at $t=1977.2224$ (zeros of $\zeta'$ approach the line from the right exactly where $\zeta$'s zeros are closest — cf. Levinson–Montgomery); maximum $\Re s=2.658$. Counts agree with the independent whole-rectangle winding number and with Berndt's main term $N(T)-\frac T{2\pi}\log2$ ($1296.4$ vs $1298$). First zero $2.46316187+23.29832049i$ (Spira).

**Davenport–Heilbronn control** ($0<t<300$): $\kappa=0.2840790438404123$ (closed form $\tan(\arg\varepsilon/2)$ with the root number $\varepsilon=\tau(\chi)/(i\sqrt5)$, verified to $10^{-40}$); functional equation residual $<2\times10^{-37}$. **204 zeros on the line and 5 certified off-line zeros** (10 with their mirrors), total winding $214=204+2\cdot5$: $0.808517182457+85.699348485378i$, $0.650830080610+114.163342730757i$, $0.574356050451+166.479305913168i$, $0.724257694627+176.702461242856i$, $0.869530579641+240.404672351441i$ (the first two agree with Spira 1994 / Balanzario–Sánchez-Ortiz 2007 to their printed precision). Minimum distance of an off-line zero from the line: $0.0744$. Figure `davenport_heilbronn_zeros.png`. This is the control for every criterion in this repository: a Dirichlet series with real coefficients, entire continuation and a $\zeta$-type functional equation for which RH is false. Its only missing property is the Euler product.

---

## E8. Zero statistics versus GUE (2026-11-05)

`python3 scripts/run_statistics.py` (11 s on $10^5$ zeros; re-run on the full $10^6$ set below). Module `statistics.py` (~900 lines); 14 tests. High sample: 6000 consecutive zeros from index $10^{10}$ ($T\approx3.29\times10^9$; Arb, radius $4.9\times10^{-7}$; 556 s on 3 cores).

**Reference laws.** Exact GUE spacing density $p_2(s)=\partial_s^2\det(I-K_s)$ (sine kernel on $[0,s]$) by Bornemann's Gauss–Legendre Nyström method (40–90 nodes agree to $10^{-15}$), Chebyshev-interpolated; checks: $\int p_2=1+3\times10^{-10}$, mean $1+10^{-11}$, variance $0.17999$ (Wigner surmise $0.17810$), small-$s$ series $\pi^2s^2/3-2\pi^4s^4/45+\dots$ matched to $1.6\times10^{-10}$, Jacobi-formula derivatives agree to $10^{-11}$. $\max|p_2-\text{Wigner}|=0.0052$ at $s\approx1.1$.

**Spacings.** Unfolded by $\theta(\gamma)/\pi$ (long double; needed at $x\sim10^{10}$). Zeros $1..10^5$: mean spacing $1.00000$, variance $0.1607$ (GUE $0.1800$); sample at $10^{10}$: variance $0.1779$. Kolmogorov–Smirnov distance to exact GUE (noise floor $0.87/\sqrt n$ in parentheses): zeros $1..10^4$: $0.0289$ $(0.0087)$ at $\log T=5.9$; $10^4..3\times10^4$: $0.0241$ $(0.0061)$; $3\times10^4..10^5$: $0.0192$ $(0.0033)$; aggregate $1..10^5$: $0.0207$ at $s=0.646$ with the empirical CDF *below* GUE (fewer small spacings: stronger repulsion at finite height); sample at $\log T=21.9$: $0.0056$ $(0.0112)$ — indistinguishable from GUE. The three low windows plus the sample are consistent with both $c/\log T$ ($c=0.19$) and $c/(\log T)^2$ ($c=1.2$) and cannot discriminate (Bogomolny–Bohigas–Leboeuf–Monastra 2006 predict the latter via an effective CUE dimension $N_{\rm eff}\propto\log(T/2\pi)$).

**Pair correlation.** First $10^5$ zeros: on $\alpha\in[0.9,1.1]$, $\max|R_2-R_2^{\rm GUE}|=0.034$; over all bins the rms deviation $0.024$ exceeds the Poisson error $0.013$ ($\chi^2/\text{bin}=5.8$): real finite-height structure — excess at $\alpha\approx0.4$, suppression below $0.2$, dip at $\alpha\approx1.3$–$1.5$, the Bogomolny–Keating pattern. The $10^{10}$ sample is consistent with GUE ($\chi^2/\text{bin}=0.59$). Montgomery's form factor $F(\alpha)$: plateau $0.999$ (first $10^5$) and $1.005$ (sample) for $\alpha>1.2$ with the diagonal normalisation; $F(0.5)=0.650$ vs the finite-height GUE expectation $0.661$.

**Number variance.** With a jittered start grid (a regular grid aliases with the near-unit lattice of unfolded zeros and biases counts by 5 %): GUE-like for $L\lesssim1$, then **saturation** at $\Sigma^2_\infty=0.3592\pm0.0028$ (zeros $1..10^5$; Berry's semiclassical prediction $0.3621$) and $0.4414\pm0.0106$ (sample; Berry $0.4459$), with the prime-driven oscillations of $\Sigma^2(L)$ in phase with Berry's diagonal-approximation curve ($L=10,20,50$: $0.495/0.452/0.437$ observed vs $0.492/0.450/0.434$). GUE would give $0.813$ at $L=100$.

**Lehmer pairs.** Smallest normalised spacing among the first $10^5$ zeros: $0.02186$ at $n=95248$ ($\gamma=71732.901208/71732.915909$, gap $0.0147$); Lehmer's classical pair $n=6709$ ($7005.062866/7005.100565$, normalised $0.0421$) ranks 6th; Csordas–Smith–Varga statistic $\Delta=0.0069\ll4/5$ for it, and all 10 closest pairs have $\Delta<0.014$. In the sample the smallest normalised spacing is $0.01137$ (gap $0.00356\gg$ radius).

Figures: `statistics_spacing.png`, `statistics_pair_correlation.png`, `statistics_number_variance.png`, `statistics_height_dependence.png`.

**Interpretation.** Everything is as the GUE model with arithmetic (prime) corrections predicts, at four heights spanning $10^2$ to $3\times10^9$ — consistent with the Hilbert–Pólya picture and with the Bogomolny–Keating/Berry semiclassics, and, as §3.3 of the survey says, silent on the *location* of the zeros.

## E9. de Bruijn–Newman flow and explicit formula / Weil positivity — worker running

*(to be filled in)*
