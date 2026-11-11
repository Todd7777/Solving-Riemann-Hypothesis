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
| `zeros_100001_1000000` | $10^5+1\ldots10^6$ | $\gamma_{10^6}=600269.677\,012$ | $1.2\times10^{-10}$ | 59 min (2 workers) |
| `sample_1e10` | $10^{10}\ldots10^{10}+5999$ | $\approx3.29\times10^{9}$ | $4.9\times10^{-7}$ | 556 s (3 workers) |

Throughput: 115–127 zeros/s/core for $n\le10^6$ (nearly independent of height); ~2 s/zero at $n=10^{12}$.

---

## E2. Nyman–Beurling–Báez-Duarte distances (2026-11-05)

`python3 scripts/run_nyman_beurling.py --N 2000 --arb 400 --tag _N2000` (105 s) and `--N 10000 --arb 800 --tag _N10000`.

**Method.** $d_N^2=1-b^TG_N^{-1}b$ with $G_{jk}=\int_0^\infty\{u/j\}\{u/k\}u^{-2}du$ from the Vasyunin closed form and $b_k=(1-\gamma_E+\log k)/k$; one Cholesky factorisation gives all $d_n^2$, $n\le N$. The closed form was *re-verified* against a direct evaluation of the integral (exact integration over the first $K$ periods + moment/Hurwitz-zeta tail) to $10^{-37}$ for many coprime pairs (the formula with $c_0(h/k)$-type cotangent sums in some references is **not** the right one; the correct one uses $V(h,k)=\sum_{m=1}^{k-1}\{mh/k\}\cot(\pi m/k)$, see `nyman_beurling.py`).

**Conditioning.** $\lambda_{\min}(G_N)\approx4.8\times10^{-5}$ at $N=160$; $\mathrm{cond}(G_N)=3.7\times10^4,\ 6.2\times10^5,\ 2.4\times10^6,\ 9.7\times10^6,\ 1.5\times10^7$ at $N=100,400,800,1600,2000$: growth $\sim N^{2.0}$, so float64 suffices; rigorous Arb enclosures (50 digits) at $N=50,100,200,400,800$ agree with float64 to $\le 8\times10^{-15}$.

**Results ($N\le2000$).**

| $N$ | 10 | 20 | 50 | 100 | 200 | 500 | 1000 | 2000 |
|---|---|---|---|---|---|---|---|---|
| $d_N^2\log N$ | 0.05474 | 0.04944 | 0.04643 | 0.04692 | 0.04726 | 0.04611 | 0.04529 | 0.04647 |

Burnol's constant $C=2+\gamma_E-\log4\pi=0.046191$. Over $200\le N\le2000$: $\min d_N^2\log N=0.04419$ (at $N\approx700$), $\max=0.04726$ (at $N=200$; $0.04690$ for $N>200$). A one-parameter fit $d_N^2\log N=C_0+a/\log N$ on $N\ge200$ gives $C_0=0.04610$, $a=-0.003$ — i.e. the data are consistent with the conjecture $d_N^2\sim C/\log N$ to $0.2\%$ in the constant, but with *oscillations of $\pm5\%$ around it that dip below $C$* (permitted: Burnol's bound is a liminf statement).

**Structure of the drops.** $d_N^2$ is non-increasing. For $N<50$ the drops are dominated by the primes ($N=2,3,5,7,11$ alone account for most of the decrease from $N=1$; prime share of the decrease over $2\le N<50$: 96 %), as each new small prime lets the Dirichlet polynomial absorb an Euler factor of $1/\zeta$. **Beyond $N\approx50$ nothing distinguishes primes**: over $50\le N<1000$ primes contribute $14.5\%$ of the decrease while being $16.1\%$ of the integers, and over $10^3\le N<10^4$ $8.2\%$ versus $11.8\%$ (median relative drop $8\times10^{-6}$ at primes vs $1.3\times10^{-5}$ at non-primes). The sawtooth of $d_N^2\log N$ is the competition between a slowly varying decrease of $d_N^2$ and the growth of $\log N$, not an arithmetic effect of individual $N$. (An earlier version of this log claimed a prime effect at all $N$; that was an artefact of class means dominated by $N\le11$.)

**Optimal coefficients.** At $N=2000$, $\operatorname{sign}(c_k)=-\mu(k)$ for **100 %** of squarefree $k$; mean $|c_k|$ is $0.286$ on squarefree and $0.066$ on non-squarefree $k$. Writing $c_k\approx-\mu(k)f(x)$, $x=\log k/\log N$, the empirical taper is *shallower* than the Bettin–Conrey–Farmer choice $1-x$: the binned medians of $-c_k/\mu(k)$ are $0.98,0.86,0.64,0.43,0.18$ at $x=0.03,0.25,0.5,0.75,0.98$, well fitted by $(1-x)^{0.45}$. Restricting to coefficients of the form $-\mu(k)(1-x)^\alpha$ (an explicit mollifier, hence an *upper bound* for $d_N^2$), the best exponent is $\alpha\approx0.30$ giving $U_N^2=0.0180$ at $N=2000$ versus $\alpha=1$: $0.0674$, and the true optimum $d_{2000}^2=0.00611$. So at this size the optimiser gains a factor 3 over the best Möbius power taper, largely through the non-squarefree coefficients and $k$-by-$k$ fluctuations (`scripts/analyze_nb_structure.py`; figures `nyman_beurling_coefficients_N2000.png`, `nyman_beurling_structure_N2000.png`).

Figures: `nyman_beurling_dN_logN_N2000.png`, `nyman_beurling_dN_N2000.png`, `nyman_beurling_coefficients_N2000.png`.

### E2b. $N=10\,000$ (2026-11-05)

`python3 scripts/run_nyman_beurling.py --N 10000 --arb 800 --tag _N10000` (3865 s: Vasyunin table $\approx3400$ s for the $3.04\times10^7$ coprime pairs, Gram assembly, one Cholesky of the $10^4\times10^4$ matrix, and rigorous Arb solves at $N=50,100,200,400,800$).

| $N$ | 500 | 1000 | 2000 | 3000 | 5000 | 8000 | 10 000 |
|---|---|---|---|---|---|---|---|
| $d_N^2\log N$ | 0.04611 | 0.04529 | 0.04647 | 0.04494 | 0.04517 | 0.04556 | **0.04512** |

* Rigorous enclosures (50 digits; radius $\le5\times10^{-41}$) at $N=50,100,200,400,800$ agree with the float64 Cholesky values to $\le8\times10^{-15}$ ($d_{400}^2=0.0077077737536834$, $d_{800}^2$ enclosed at $N=800$ in 228 s); the float64 results for larger $N$ are backed by the measured condition number ($1.5\times10^7$ at $N=2000$, $\sim N^{2.2}$), i.e. $\lesssim10^{-9}$ absolute error at $N=10^4$.
* **The sequence sits below Burnol's constant.** Over $1000\le N\le10^4$, $d_N^2\log N\in[0.04450,0.04667]$ (minimum at $N=3470$, maximum at $N=2605$); it is last above $C=0.046191$ at $N=2679$ and *below* $C$ for every $N$ from $2680$ to $10^4$, ending at $0.04512$ — $2.3\%$ below $C$. Burnol's theorem is a $\liminf$ statement, so finite-$N$ values below $C$ are allowed, but it forces the sequence to rise again: the convergence $d_N^2\log N\to C$ (if the conjecture holds) is slow and non-monotone, with a relative deficit of $\approx2\%$ still present at $\log N=9.2$. A naive one-parameter fit $C_0+a/\log N$ over $N\ge1000$ now returns $C_0=0.0436<C$ — a fit that Burnol's theorem says cannot be the truth; the lesson is that the fluctuations (whose amplitude is not visibly decaying between $N=10^3$ and $10^4$) dominate any extrapolation at these sizes. Landreau–Richard (2002) reached similar conclusions at smaller $N$.
* The Möbius mollifier $a_k=\mu(k)(1-\log k/\log N)$ gives $0.04688$ at $N=10^4$, versus $d_{10^4}^2=0.004899$: a factor $9.6$; the sign agreement of the optimal $c_k$ with $-\mu(k)$ on squarefree $k$ is $99.92\%$; mean $|c_k|$: $0.255$ (squarefree), $0.065$ (non-squarefree).

Figures: `nyman_beurling_dN_logN_N10000.png`, `nyman_beurling_dN_N10000.png`, `nyman_beurling_coefficients_N10000.png`.

### E2c. Second-order structure (2026-11-05)

`python3 scripts/analyze_nb_structure.py --tag _N2000` (2 min, rebuilds $G_{2000}$) and `--tag _N10000 --rebuild-gram` (≈1 h).

* **Where the decrease comes from.** Over $10\le N\le10^4$ the $1229$ primes account for $39.5\%$ of the total decrease of $d_N^2$ — but this is entirely the effect of $N\le50$ (prime share 96 % there). By range: $14.5\%$ for $50\le N<10^3$ and $8.2\%$ for $10^3\le N<10^4$, i.e. *less* than the primes' share of the integers (16 % and 12 %); medians of the relative drop are $8\times10^{-6}$ (primes) vs $1.3\times10^{-5}$ (non-primes) in the last range. Class means are misleading because they are dominated by $N=2,3,5,7,11$.
* **Spectrum of the oscillation.** The power spectrum of $d_N^2\log N-C$ in $u=\log N$ ($50\le N\le10^4$, Hann window) peaks at $\omega\approx1.2$ (period $5.3$ in $\log N$, i.e. the slow swing visible in the figure), then $4.7$, $7.1$, $11.9$, $16.6$. The peak at $7.1$ coincides with $\gamma_1/2=7.07$, but the frequency resolution ($2\pi/5.3\approx1.2$) is too coarse to claim that the zeros are visible in $d_N^2$; the question needs $N\gg10^4$ and is left open.
* **Taper of the optimal coefficients.** With $x=\log k/\log N$, the binned medians of $-c_k/\mu(k)$ over squarefree $k$ follow $(1-x)^{0.48}$ at $N=10^4$ ($(1-x)^{0.45}$ at $N=2000$): shallower than $1-x$, with the coefficients at $k\approx N$ still $\approx0.2$ rather than $0$.
* **Explicit mollifiers as upper bounds.** For $c_k=-\mu(k)(1-x)^\alpha$ the norm $U_N^2=1-2b\cdot c+c^TGc$ is an explicit upper bound for $d_N^2$. At $N=2000$ the best exponent is $\alpha\approx0.30$, giving $U^2=0.0180=2.95\,d_{2000}^2$ (the Bettin–Conrey–Farmer $\alpha=1$ gives $0.0674=11\,d_{2000}^2$); at $N=10^4$ the best exponent is $\alpha\approx0.36$, giving $U^2=0.01441=2.94\,d_{10^4}^2$ versus $0.0469=9.6\,d_{10^4}^2$ for $\alpha=1$. **The ratio $U^2_{\rm best}/d_N^2\approx2.94$ is the same at $N=2000$ and $N=10^4$**: the optimiser beats the best *pure* Möbius power taper by a stable factor $\approx3$, through the non-squarefree coefficients (mean $|c_k|=0.065$) and $k$-by-$k$ fluctuations around the taper. A mollifier family that captures this factor would be a concrete, provable improvement over $1-x$ in the Nyman–Beurling bound.

Figures: `nyman_beurling_structure_N2000.png`, `nyman_beurling_structure_N10000.png`.

---

## E3. Báez-Duarte's sequential criterion and its explicit formula (2026-11-05)

`python3 scripts/run_sequential_criteria.py` (28 s)

$c_k=\sum_{j=0}^k(-1)^j\binom kj/\zeta(2j+2)=\sum_n\mu(n)n^{-2}(1-n^{-2})^k$; RH $\iff c_k=O(k^{-3/4+\varepsilon})$.

We derived (Mellin–Barnes with the Beta function) and verified the explicit formula
$$c_k=\sum_{\rho}\frac{B(1-\rho/2,\,k+1)}{2\zeta'(\rho)}+\sum_{m\ge1}\frac{B(1+m,\,k+1)}{2\zeta'(-2m)}+\ldots$$
(nontrivial and trivial zeros; $B$ = Beta function). With 8 nontrivial and 12 trivial zeros it reproduces the exact $c_k$ for $10\le k\le10^6$ (382 values) with **max absolute error $1.7\times10^{-9}$** (relative $1.1\times10^{-5}$ for $k\ge10^3$; the residual is the $k^{-3/2}$-type remainder and the omitted zeros).

Two facts worth recording:
* The **trivial-zero term** $1/(2\zeta'(-2)(k+1)(k+2))\approx-16.4/k^2$ (with $\zeta'(-2)=-\zeta(3)/4\pi^2$) dominates $c_k$ until $k\approx19\,758$, even though it is formally $O(k^{-2})\ll k^{-3/4}$: the nontrivial-zero amplitude carries the factor $|\Gamma(\tfrac34-i\gamma_1/2)|\approx6.2\times10^{-5}$ (combined weight of the conjugate pair $1.2\times10^{-4}$). Any numerical test of the $k^{-3/4}$ law must subtract it.
* After subtracting it, $k^{3/4}c_k$ is a clean sinusoid in $\log k$ of frequency $\gamma_1/2$ with amplitude $7.78\times10^{-5}=|\Gamma(1-\rho_1/2)|/|\zeta'(\rho_1)|$ (the maximum of $|k^{3/4}(c_k-\text{trivial})|$ for $k\ge10^5$ is $1.003\times$ that amplitude, the excess being $\rho_2$'s contribution; without subtracting the trivial term the ratio is $1.05$). **One sees the first zero of $\zeta$ directly in the sequence $c_k$.** A zero $\beta+i\gamma$ with $\beta<\tfrac12$ would add $k^{-(1+\beta)/2}e^{-\pi\gamma/4}$-sized oscillations; for $\gamma\gtrsim 60$ these are below $10^{-20}$ relative and invisible — the criterion, like all of them, is numerically blind at height.

Figure: `sequential_criteria.png`.

---

## E4. Li coefficients (2026-11-05)

`python3 scripts/run_li.py` (67 s). Module `li.py` (worker report verified by an independent check of its non-standard claim, see below).

* $\lambda_n$ for $n\le10^4$ from $10^5$ zeros ($\lambda_n=4\sum_{\gamma>0}\sin^2(n\arctan\frac1{2\gamma})$) plus an integral tail started at the point $T_{\rm eff}$ where the smooth count $N_{\rm smooth}(T_{\rm eff})=10^5$; tail error bounded via Abel summation and Trudgian's explicit bound on $S(T)$.
* Independent **rigorous** values for $n\le5000$ (Arb, 1810 digits, deflated zeta power series at $s=1$, series composition $z\mapsto z/(1-z)$): max relative discrepancy with the zero-sum method $3.6\times10^{-12}$ (at $n=5000$), $\lambda_1$ to $2\times10^{-15}$ (closed form $1+\gamma_E/2-\tfrac12\log4\pi$ reproduced to 60 digits).
* **All $\lambda_n>0$ for $n\le10^4$**, rigorously (ball lower bounds) for $n\le5000$; minimum is $\lambda_1=0.0230957089661210338\ldots$. First values: $\lambda_2=0.0923457352280466704$, $\lambda_3=0.2076389205543248038$, $\lambda_4=0.3687904794922416386$, $\lambda_5=0.5755427144611774524$; $\lambda_{10^4}=34736.58$.
* Asymptotics: fit of $\lambda_n-\tfrac n2\log n$ to $c_1n+c_0$ gives $c_1=-1.130358$ vs theory $\tfrac12(\gamma_E-1-\log2\pi)=-1.130331$.
* Archimedean part: the worker found, and we confirmed independently with the Bombieri–Lagarias formula in mpmath (80 digits), that $S_\infty(n)=\tfrac n2(H_n-1-\log2\pi)+\tfrac12$ up to exponentially small terms ($6\times10^{-14}$ at $n=150$), i.e. $\tfrac n2(\log n+\gamma_E-1-\log2\pi)+\tfrac34-\tfrac1{24n}+O(n^{-3})$. The constant term in the $\log n$ form is $3/4$.
* **Re-run with $10^6$ zeros** (`python3 scripts/run_li.py`, 128 s; $T_{\rm eff}=600269.91$): the tail is now $0.95\%$ of $\lambda_{10^4}$ ($331$ of $34737$), the zero-sum values agree with the rigorous Arb enclosures to **max relative error $9.9\times10^{-16}$** for $n\le5000$ (absolute $1.5\times10^{-11}$), $\lambda_1$ to $3.5\times10^{-17}$; positivity and the fits are unchanged ($c_1=-1.130358$, free slope $0.49991$).
* **Extension to $n\le10^5$ (2026-11-07).** `python3 scripts/run_li.py --nmax 100000 --hp-max 5000 --json results/json/li_summary_n1e5.json` took 959 s; the zero sums over $10^6$ zeros dominate. Results:
  * All $\lambda_n>0$ for $n\le10^5$, still positive after subtracting the rigorous tail-error bound.
  * Rigorous Arb values for $n\le5000$ agree with the zero sums to $9.9\times10^{-16}$ relative.
  * The fit of $\lambda_n-\tfrac n2\log n$ gives $c_1=-1.130334$, against RH's $(\gamma-1-\log2\pi)/2=-1.130331$; the free slope is $0.49999$.
  * **Growth of the oscillating part.** Over log-spaced windows from $n=200$ to $10^5$, the rms of $S_2(n)$ grows like $C\,n^{\beta}$ with $\beta=0.32\pm0.05$ and $C\approx0.38$. The windowed rms of $S_2/n^{1/3}$ is $0.35, 0.34, 0.336, 0.333$ for $n\in[10^2,10^3),[10^3,10^4),[10^4,3\times10^4),[3\times10^4,10^5)$, while that of $S_2/\sqrt n$ falls from $0.125$ to $0.053$. The data cannot distinguish $n^{1/3}$ from $n^{1/4}(\log n)^{1/2}$; both have about 15% scatter.
  * A heuristic favours the second. Integrating by parts against $S(t)$ gives $S_2(n)\approx2\int_0^\infty S(n/x)\sin x\,dx$. This is dominated by heights $t\lesssim\sqrt{n/\log n}$, where $S$ changes sign on the scale of the oscillation of $\sin x$, and behaves like a random walk of size $\approx n^{1/4}$ up to logarithms.
  * Either way the oscillating part is far below the $O(\sqrt n\log n)$ that RH implies (Lagarias 2007). This is an empirical sharpening, not a theorem.

* Oscillating part $S_2(n)=\lambda_n-S_\infty(n)$: mean $+0.96$, rms $5.6$, $\max|S_2|=19.3$ at $n=9947$, $|S_2|/\sqrt n<0.29$ for $n\ge100$ (consistent with $O(\sqrt n\log n)$ under RH).
* **Sensitivity calibration** (`li.detection_horizon`, `li.zero_contribution`). The mirror $\tfrac12-\delta+i\gamma$ of an off-line zero contributes a term of modulus $|1-1/\rho'|^n\approx\exp(n\delta/\gamma^2)$ to $\lambda_n$, so positivity can only fail once $n\delta/\gamma^2\gtrsim\log(\tfrac n2\log n)$. Exact evaluation for the Davenport–Heilbronn off-line zero $0.8085+85.699i$ (as if it were a zero of $\zeta$): its quadruple contributes $+7.9$ at $n=10^4$, $+38$ at $n=10^5$, $-4.2\times10^5$ at $n=3\times10^5$ and $-3\times10^{18}$ at $n=10^6$; the predicted horizon is $n^*\approx3.3\times10^5$ (threshold $2e^{nr}\approx\tfrac n2\log n$, corrected after review). Horizons for a hypothetical zero with $\delta=0.1$: $n^*\approx1.6\times10^6$ at $\gamma=100$, $2.1\times10^8$ at $\gamma=10^3$, $2.6\times10^{10}$ at $\gamma=10^4$, $6.4\times10^{26}$ at $\gamma=10^{12}$. **Positivity of $\lambda_n$ for $n\le10^4$ is therefore insensitive to any off-line zero above height $\approx30$**; it is a consistency check of the first few zeros and the growth of $S(T)$, not independent evidence for RH.

Figures: `li_coefficients.png`, `li_oscillation.png`.

---

## E5. Integral criteria: Balazard–Saias–Yor and Volchkov (2026-11-05)

`python3 scripts/run_integral_criteria.py --nzeros 20000` (BSY 644 s with Arb's $\zeta$ at 0.3 ms per evaluation; Volchkov 133 s). Module `integral_criteria.py` (tanh-sinh rule with 81–127 nodes per panel between consecutive zeros, robust to the logarithmic endpoint singularities).

**Balazard–Saias–Yor.** $I(T)=\frac1\pi\int_0^T\frac{\log|\zeta(1/2+it)|}{1/4+t^2}dt$, which equals $\sum_{\Re\rho>1/2}\log|\rho/(1-\rho)|$ in the limit, hence $0$ under RH:

| $T$ (at zero no.) | 49.8 (10) | 236.5 (100) | 1419 (1000) | 9878 (10 000) | 18 046 (20 000) |
|---|---|---|---|---|---|
| $I(T)$ | $-5.5\times10^{-5}$ | $6.4\times10^{-6}$ | $-3.6\times10^{-8}$ | $-8.2\times10^{-11}$ | $-7.2\times10^{-10}$ |

The identity holds to nine–ten decimals (the last two checkpoints are $-8\times10^{-11}$ and $-7\times10^{-10}$; the truncation error oscillates in sign); the truncation error decays much faster than the naive $O(1/T)$ because $\log|\zeta|$ has mean zero on the line (Selberg), so the tail cancels. Figure `integral_criteria_bsy.png`.

**Volchkov.** $\int_0^\infty\frac{1-12t^2}{(1+4t^2)^3}\int_{1/2}^\infty\log|\zeta(\sigma+it)|\,d\sigma\,dt=\pi\frac{3-\gamma_E}{32}=0.237\,856\,295\,887$. Two numerical traps had to be removed to see the identity: the pole at $s=1$ is an interior logarithmic singularity of the inner integrand for $t\to0$ (splitting the $\sigma$-range at $\sigma=1$ removed a systematic $-2.7\times10^{-4}$), and near each zero the inner integrand peaks sharply at $\sigma\to\tfrac12$ (subtracting $\log|s-\rho_j|$ for the nearest zero analytically). Result at $T=200$, $\sigma_{\max}=30$, (32, 64) nodes: $0.237\,856\,381$, error $8.5\times10^{-8}$, independent of $T\in[60,300]$ (quadrature-order limited: $3.5\times10^{-6}$ with (24, 48) nodes).

**Interpretation.** Both identities are confirmed to the accuracy of the quadrature. As calibrated in §4.4 of the barriers document, a zero $\tfrac12+\delta+i\gamma$ together with its conjugate would change $I$ by $\approx2\delta/(\tfrac14+\gamma^2)$: at $\gamma=10^4$ and $\delta=0.1$ that is $2\times10^{-9}$ — at the level of our truncation error, and beyond that height invisible. The integral criteria are exact reformulations but numerically the least sensitive of all.

---

## E6. Independent Riemann–Siegel implementation and Turing verification (2026-11-05)

`python3 scripts/run_verify.py` (8.6 s). Modules `zeta.py` (839 lines), `verify.py`; 68 tests.

* **Riemann–Siegel** $Z(t)$ with correction terms $C_0..C_4$ (coefficient series generated at import from exact power-series division of $\Psi(p)$, reproducing Haselgrove's tables to 20 digits) and Gabcke's remainder bounds; Euler–Maclaurin with explicit remainder below $t=500$. $\theta(t)$ from the Bernoulli-number expansion through $t^{-13}$ (mpmath below $t=12$). Gram points by Newton; zero search on a grid uniform in the Gram coordinate $\theta/\pi$ with Rosser-rule block refinement (this catches the Lehmer pair $7005.0629/7005.1006$).
* **Turing's method** with Lehman's constants ($|S_1(t_2)-S_1(t_1)|\le2.30+0.128\log(t_2/2\pi)$ for $168\pi<t_1<t_2$; Trudgian's smaller constants are available but flagged as quoted from memory): certify $N(g_m)=m+1$ from $m+1$ sign changes below $g_m$ and the integrated inequality over $k$ Gram intervals above.
* **Result.** $N(74920.9)=100\,000$ **certified** (7 Gram intervals, quotient $(B-F)/L=0.929<1$): all 100 000 zeros below $g_{99999}=74921.2257$ are simple and on the critical line. 100 082 zeros found on $[0,74976]$; 34 Gram blocks needed refinement, none unresolved; all brackets certified without mpmath fallback. Cross-check: Arb's rigorous count agrees, and $\max|\gamma_{\rm found}-\gamma_{\rm Arb}|=5.8\times10^{-10}$ over all $10^5$ zeros (float64 error; the Arb radius is $1.5\times10^{-11}$).
* **Million-zero run.** `python3 scripts/run_verify.py --T 600270 --time-budget 2400 --out results/json/verify_summary_1e6.json` (the default output file `verify_summary.json` now holds a 600-s-budget re-run that certified $N(544610.25)=898\,841$ with $k=8$, quotient $0.8605$, 133 s; the first-$10^5$ run quoted above was overwritten by it) (152 s: zero search 125 s on a grid of $8.0\times10^6$ points, Turing 27 s): **$N(600270)=1\,000\,000$ certified** (quotient $0.880<1$ with $k=7$ Gram intervals above $g_{10^6}=600270.4598$); 1 000 084 brackets certified, 0 failed, 1 mpmath fallback; 421 of 869 966 Gram blocks needed refinement, none unresolved. All $10^6$ zeros below $600270.46$ are simple and on the critical line (modulo the float64 caveat below). Arb's rigorous count agrees and $\max|\gamma_{\rm found}-\gamma_{\rm Arb}|=6.8\times10^{-9}$ over all $10^6$ zeros.
* **Rigour boundary** (stated in the JSON): the theorem and the truncation bounds are rigorous; float64 rounding in $Z$, $\theta$, Gram points and the quadrature is estimated (10–100× conservatively) but not bounded — a fully rigorous version is exactly what Arb does, which we keep as the independent reference.

---

## E7. Arithmetic criteria, Speiser's criterion, Davenport–Heilbronn control (2026-11-05)

`python3 scripts/run_arithmetic.py --workers 3` (259 s). Modules `arithmetic.py`, `speiser.py`, `davenport_heilbronn.py`; 23 tests.

**Robin** ($N=10^7$, exact integer sieve): the violators of $\sigma(n)<e^{\gamma_E}n\log\log n$ for $n\ge2$ are exactly the 27 known values $\{2,3,4,\dots,2520,5040\}$. Beyond 5040 the maximum of $\sigma(n)/(e^{\gamma_E}n\log\log n)$ is $0.98582$ at $n=10080$ (then 55440: 0.98325; 27720: 0.97836).

**Lagarias** ($N=10^7$): no violators; equality only at $n=1$; minimum slack $0.3172$ at $n=2$; maximum of $\sigma(n)/(H_n+e^{H_n}\log H_n)$ is $0.98864$ at $n=12$.

**Nicolas** ($K=10^4$ primorials, 50 digits): $r_k=\frac{N_k/\varphi(N_k)}{e^{\gamma_E}\log\log N_k}$ is $>1$ and **strictly decreasing for all $k\ge2$**, from $2.888$ to $1.000\,564$ at $k=10^4$ ($p_k=104\,729$); empirically $(r_k-1)\sqrt{p_k}\log p_k\approx2.1$ (stable from $k=2000$ to $10^4$), i.e. $r_k-1\approx2.1/(\sqrt{p_k}\log p_k)$ — the rate predicted by the prime number theorem with square-root error.

**Colossally abundant numbers** (first 200, exact exponent vectors via the critical-$\varepsilon$ chain, each re-derived from the closed-form exponents; the first 22 match OEIS A004490; brute-force maximisation over all $n\le10^8$ agrees for the first 13, and over all Hardy–Ramanujan integers $\le10^{16}$ for the first 21): the 200th has 433 digits and 165 prime factors. Robin's ratio is $<1$ for every CA number $>5040$ and increases (with 12 small dips among 191 steps) from $0.98325$ (55440) to $0.995794$ (#200); $(1-\text{ratio})\sqrt{\log n}$ drifts slowly from $0.16$ at $\log n\approx150$ to $0.133$ at $\log n\approx995$ (while $e^{\gamma_E}\log\log n\,(1-\text{ratio})\sqrt{\log n}$ rises from $1.41$ to $1.63$): the margin in Robin's inequality on CA numbers decays like $(\log n)^{-1/2}$ up to a slowly varying factor, as expected from $\sigma(n)/n=e^{\gamma_E}\log\log n-c\,(\log\log n)/\sqrt{\log n}+\ldots$ for CA numbers under RH. Lagarias holds for all 200 with log-slack decreasing to $0.0043$. Gronwall: $\sigma(n)/(n\log\log n)=1.77358$ at #200 vs $e^{\gamma_E}=1.78107$.

**Speiser** ($T=2000$, rectangle $-1\le\sigma\le4$, $0.25\le t\le2000$; the double pole at $s=1$ gives winding $-2$ on a small box): **1298 zeros of $\zeta'$, all certified by Krawczyk's test with rigorous, outward-rounded bounds and pairwise disjoint boxes (so they are distinct), all with $\Re s>\tfrac12$**; minimum $\Re s=0.503428$ at $t=1977.2224$ (zeros of $\zeta'$ approach the line from the right exactly where $\zeta$'s zeros are closest — cf. Levinson–Montgomery); maximum $\Re s=2.658$. Counts agree with the independent whole-rectangle winding number and with Berndt's main term $N(T)-\frac T{2\pi}\log2$ ($1296.4$ vs $1298$). First zero $2.46316187+23.29832049i$ (Spira).

**Davenport–Heilbronn control** ($0<t<300$): $\kappa=0.2840790438404123$ (closed form $\tan(\arg\varepsilon/2)$ with the root number $\varepsilon=\tau(\chi)/(i\sqrt5)$, verified to $10^{-40}$); functional equation residual $<2\times10^{-37}$. **204 zeros on the line and 5 certified off-line zeros** (10 with their mirrors), total winding $214=204+2\cdot5$: $0.808517182457+85.699348485378i$, $0.650830080610+114.163342730757i$, $0.574356050451+166.479305913168i$, $0.724257694627+176.702461242856i$, $0.869530579641+240.404672351441i$ (the first two agree with Spira 1994 / Balanzario–Sánchez-Ortiz 2007 to their printed precision). Minimum distance of an off-line zero from the line: $0.0744$. Figure `davenport_heilbronn_zeros.png`. This is the control for every criterion in this repository: a Dirichlet series with real coefficients, entire continuation and a $\zeta$-type functional equation for which RH is false. Its only missing property is the Euler product.

---

## E8. Zero statistics versus GUE (2026-11-05)

`python3 scripts/run_statistics.py` (11 s on $10^5$ zeros; re-run on the full $10^6$ set below). Module `statistics.py` (~900 lines); 14 tests. High sample: 6000 consecutive zeros from index $10^{10}$ ($T\approx3.29\times10^9$; Arb, radius $4.9\times10^{-7}$; 556 s on 3 cores).

**Reference laws.** Exact GUE spacing density $p_2(s)=\partial_s^2\det(I-K_s)$ (sine kernel on $[0,s]$) by Bornemann's Gauss–Legendre Nyström method (40–90 nodes agree to $10^{-15}$), Chebyshev-interpolated; checks: $\int p_2=1+3\times10^{-10}$, mean $1+10^{-11}$, variance $0.17999$ (Wigner surmise $0.17810$), small-$s$ series $\pi^2s^2/3-2\pi^4s^4/45+\dots$ matched to $1.6\times10^{-10}$, Jacobi-formula derivatives agree to $10^{-11}$. $\max|p_2-\text{Wigner}|=0.0052$ at $s\approx1.1$.

**Spacings.** Unfolded by $\theta(\gamma)/\pi$ (long double; needed at $x\sim10^{10}$). Zeros $1..10^5$: mean spacing $1.00000$, variance $0.1607$ (GUE $0.1800$); sample at $10^{10}$: variance $0.1779$. Kolmogorov–Smirnov distance to exact GUE. The noise floor $0.87/\sqrt n$ is given in parentheses; it is the mean of the null Kolmogorov law, whose median is $0.83/\sqrt n$. The quoted $\log T$ is the mean of $\log t$ over the window's spacings (corrected after review; an earlier version used the geometric mean of the window endpoints, which sits far below where the spacings lie). Zeros $1..10^4$: $0.0289$ $(0.0087)$ at $\log T=8.35$; $10^4..3\times10^4$: $0.0241$ $(0.0061)$; $3\times10^4..10^5$: $0.0192$ $(0.0033)$; aggregate $1..10^5$: $0.0207$ at $s=0.646$ with the empirical CDF *below* GUE (fewer small spacings: stronger repulsion at finite height); sample at $\log T=21.9$: $0.0056$ $(0.0112)$ — indistinguishable from GUE. With the corrected heights, the five low windows of the $10^6$-zero run fit $c/\log T$ with $c=0.22$ (rms $2.1\times10^{-3}$) and $c/(\log T)^2$ with $c=2.2$ (rms $1.4\times10^{-3}$); the latter fits slightly better, as Bogomolny–Bohigas–Leboeuf–Monastra (2006) predict via an effective CUE dimension $N_{\rm eff}\propto\log(T/2\pi)$. The $10^{10}$ sample ($D=0.0056$, below its noise floor) cannot discriminate.

**Pair correlation.** First $10^5$ zeros (numbers from the first run, superseded in the JSON by the $10^6$ re-run, which gives $\max|R_2-R_2^{\rm GUE}|=0.023$ on $[0.9,1.1]$, rms deviation $0.0148$ vs Poisson error $0.0042$, $\chi^2/\text{bin}=24$): on $\alpha\in[0.9,1.1]$, $\max|R_2-R_2^{\rm GUE}|=0.034$; over all bins the rms deviation $0.024$ exceeds the Poisson error $0.013$ ($\chi^2/\text{bin}=5.8$): real finite-height structure — excess at $\alpha\approx0.4$, suppression below $0.2$, dip at $\alpha\approx1.3$–$1.5$, the Bogomolny–Keating pattern. The $10^{10}$ sample is consistent with GUE ($\chi^2/\text{bin}=0.59$). Montgomery's form factor $F(\alpha)$: plateau $0.999$ (first $10^5$) and $1.005$ (sample) for $\alpha>1.2$ with the diagonal normalisation; $F(0.5)=0.650$ vs the finite-height GUE expectation $0.661$.

**Number variance.** With a jittered start grid (a regular grid aliases with the near-unit lattice of unfolded zeros and biases counts by 5 %): GUE-like for $L\lesssim1$, then **saturation** at $\Sigma^2_\infty=0.3592\pm0.0028$ (zeros $1..10^5$; Berry's semiclassical prediction $0.3621$) and $0.4414\pm0.0106$ (sample; Berry $0.4459$), with the prime-driven oscillations of $\Sigma^2(L)$ in phase with Berry's diagonal-approximation curve (sample at $10^{10}$, $L=10,20,50$: $0.495/0.452/0.437$ observed vs $0.492/0.450/0.434$; low set with $10^6$ zeros: $0.391/0.373/0.355$ vs $0.393/0.373/0.357$). GUE would give $0.813$ at $L=100$.

**Re-run with $10^6$ zeros** (`python3 scripts/run_statistics.py`, 11 s). KS distance to exact GUE by window (noise floor in parentheses): zeros $1..10^4$: $0.0289$ $(0.0087)$; $10^4..3\times10^4$: $0.0241$ $(0.0061)$; $3\times10^4..10^5$: $0.0192$ $(0.0033)$; $10^5..3\times10^5$: $0.0169$ $(0.0019)$; $3\times10^5..10^6$: $0.0143$ $(0.0010)$; the $10^{10}$ sample: $0.0056$ $(0.0112)$ — a monotone approach to GUE over five windows, every one of them significant (the deviation is always of the same sign: fewer small spacings than GUE). Aggregate over $10^6$ zeros: $D=0.0153$ at $s=0.67$. Number-variance plateau over $10^6$ zeros: $0.3813\pm0.0038$ vs Berry $0.3834$. Smallest normalised spacing among $10^6$ zeros: $0.00970$ at $n=420\,891$ ($\gamma=273193.663138/273193.668841$, gap $0.00570$; Csordas–Smith–Varga $\Delta=4\times10^{-4}$), then $0.01225$ at $n=839\,026$ and $0.01466$ at $n=415\,587$.

**Lehmer pairs.** Smallest normalised spacing among the first $10^5$ zeros: $0.02186$ at $n=95248$ ($\gamma=71732.901208/71732.915909$, gap $0.0147$); Lehmer's classical pair $n=6709$ ($7005.062866/7005.100565$, normalised $0.0421$) ranks 6th; Csordas–Smith–Varga statistic $\Delta=0.0069\ll4/5$ for it, and all 10 closest pairs have $\Delta<0.014$. In the sample the smallest normalised spacing is $0.01137$ (gap $0.00356\gg$ radius).

Figures: `statistics_spacing.png`, `statistics_pair_correlation.png`, `statistics_number_variance.png`, `statistics_height_dependence.png`.

**Interpretation.** Everything is as the GUE model with arithmetic (prime) corrections predicts, at four heights spanning $10^2$ to $3\times10^9$ — consistent with the Hilbert–Pólya picture and with the Bogomolny–Keating/Berry semiclassics, and, as §3.3 of the survey says, silent on the *location* of the zeros.

## E9. de Bruijn–Newman heat flow and the explicit formula / Weil positivity (2026-11-05)

`python3 scripts/run_debruijn_newman.py` (40 s), `python3 scripts/run_explicit_formula.py` (30–95 s). Modules `debruijn_newman.py`, `explicit_formula.py`; 19 tests.

**de Bruijn–Newman.** $H_0(z)=\xi(\tfrac12+\tfrac{iz}2)/8$ verified by direct quadrature of $\int_0^\infty\Phi(u)\cos(zu)\,du$ at $z=0,10,30,57,90$ to $10^{-34}$ relative to $H_0(0)$ (i.e. absolute $\sim10^{-35}$; $H_0$ itself decays like $e^{-\pi z/8}$, so the agreement relative to the local value is $\sim10^{-23}$ at $z=90$) ($H_0(0)=0.0621400972735392637$; $H_0(2\gamma_1)/H_0(0)=2.6\times10^{-18}$). The zero dynamics $\dot x_j=\sum_{k\ne j}2/(x_j-x_k)$ is integrated with explicit zeros plus an analytic tail field for the zeros beyond the dataset (closed form in terms of Legendre's $\chi_2$, verified against quadrature to $10^{-12}$; its analytic Jacobian verified by finite differences to $10^{-9}$). The common drift of the zeros under the flow (first half of the 20 000 explicit zeros): median $-0.7926$, mean $-0.7846$, matching the derived value $-\pi/4=-0.7854$ (a separate check over $10^5$ zeros in the module docstring gives $-0.7857$).
* *Lehmer's pair* $n=6709/6710$ ($\gamma=7005.06286617/7005.10056467$): $\Delta x=0.075397$, Csordas–Smith–Varga $\Delta^2g_k=0.00693<4/5$. Backward-flow **collision time $t^*=-7.11205\times10^{-4}$** from the ODE, in agreement with the two-body estimate $-\Delta^2/8=-7.10588\times10^{-4}$ and with a frozen-background closed form $\log(1-\Delta^2S/2)/(4S)=-7.11205\times10^{-4}$ (agreement to $10^{-11}$).
* *Closest pair in the first $10^5$ zeros*, $n=95248/95249$ ($\gamma=71732.90121/71732.91591$, $\Delta x=0.029403$, $\Delta^2g=0.00201$): $t^*=-1.08094\times10^{-4}$. Since real simple zeros for $t>\Lambda$ imply $\Lambda\ge t^*$ under the (heuristic, float64) model, the dataset gives the **lower-bound estimate $\Lambda\gtrsim-1.1\times10^{-4}$**; the published rigorous bounds ($-4.4\times10^{-6}$ CSV 1994, $-2.7\times10^{-9}$ Odlyzko 2000, $-1.15\times10^{-11}$ Saouter–Gourdon–Demichel 2011) come from much closer pairs at far greater heights, and Rodgers–Tao's $\Lambda\ge0$ supersedes all of them. The 12 closest pairs are tabulated in the JSON; all are Lehmer pairs in the CSV sense.
* *Forward flow to $t=0.2$* (the Platt–Trudgian upper bound) for the first 2000 zeros: the spacing distribution rigidifies — standard deviation of the normalised spacings drops to $0.738$ of its $t=0$ value, the minimum normalised spacing rises from $0.089$ to $0.468$ — the "crystallisation" that Rodgers–Tao's argument exploits in the other direction.

Figures: `debruijn_newman_flow.png`, `debruijn_newman_spacings.png`.

**Explicit formula.** For $g$ even with $\hat g=h$: $\sum_\rho h(\gamma_\rho)=h(\tfrac i2)+h(-\tfrac i2)-2\sum_n\frac{\Lambda(n)}{\sqrt n}g(\log n)+\frac1{2\pi}\int h(r)\bigl(\Re\psi(\tfrac14+\tfrac{ir}2)-\log\pi\bigr)dr$, with the archimedean term also evaluated in $u$-space (agreement $10^{-16}$), $10^5$ zeros plus a Riemann–von Mangoldt density tail, primes to $10^6$ ($10^7$ for wide $g$).
* Gaussians $g=e^{-u^2/2\sigma^2}$, $\sigma\in\{1.5,1,0.5,\dots,0.0005\}$: **residual $\le2.7\times10^{-15}$**; $\sigma=2$: $2.9\times10^{-8}$ with primes to $10^6$ (prime truncation, as predicted) and $1.4\times10^{-11}$ with primes to $10^7$. (For a plain Gaussian either the primes or the zeros contribute, not both — the uncertainty principle: the zeros start at $14.13$.)
* Modulated Gaussians $e^{-u^2/2\sigma^2}\cos(\omega u)$, where both sides are $O(1)$: residuals $4\times10^{-15}$ ($\omega=50$), $7.6\times10^{-14}$ ($500$), $5\times10^{-12}$ ($2\times10^4$), $1.6\times10^{-11}$ ($7\times10^4$, limited by the zero accuracy times $h'$).
* Convergence: with the density tail, $\sigma=0.003$ gives residual $2.6\times10^{-7}$ with 1000 zeros and $1.8\times10^{-15}$ with 3000; in primes ($\omega=100$): $0.1$ ($n\le10$), $4.5\times10^{-5}$ ($10^2$), $1.2\times10^{-11}$ ($10^3$), $2\times10^{-15}$ ($10^4$).

**Weil positivity** (arithmetic side $W(f*\tilde f)$ computed *without* the zeros; must be $\ge0$ under RH):
* Gaussian bumps, $\sigma\in[0.02,0.4]$: $W\in[2.6\times10^{-14},0.0466]$ (maximum at $\sigma\approx0.034$, then decreasing to $2.6\times10^{-14}$ at $\sigma=0.4$), all $\ge0$; zero-side cross-check residual $\le4.6\times10^{-16}$.
* Two bumps at $\pm a$ ($\sigma=0.2$, $a\in[0.05,5]$): $W$ oscillates in $[1.76\times10^{-8},6.8\times10^{-4}]$ like $1+\cos(2a\gamma_1)$ — **the prime side reproduces the first zero** — minimum $1.763\times10^{-8}$ at $a=1$, all positive; residual $\le5\times10^{-14}$.
* Box $f=\mathbf 1_{[-a,a]}$, $a\in[0.1,7]$: $W\in[0.0585,0.134]$, minimum at $a=6$, maximum at $a=0.55$ (residual vs zero side $\le2.5\times10^{-7}$ because of the slow $1/r^2$ decay).

Figures: `explicit_formula_residual.png`, `explicit_formula_weil_positivity.png`.

**Interpretation.** The explicit formula is verified to machine precision with freshly computed zeros and primes — a strong end-to-end check of the whole pipeline (zeros, $\theta$, $\Lambda(n)$ sieve, archimedean term). Weil's form is positive on every family tried, as RH requires; the near-zero values ($10^{-8}$ for the two-bump family) show how *tight* positivity is for test functions that resonate with the first zero. As §3.2 of the survey explains, a proof would have to establish this positivity for *all* $f$ without knowing the zeros; the numerics locate where it is tightest: for two bumps at $\pm a$, $W$ is maximal ($6.8\times10^{-4}$) at $a=\pi k/\gamma_1$ and nearly vanishes ($10^{-8}$–$10^{-7}$) near $a=\pi(k+\tfrac12)/\gamma_1$, where the first zero's contribution $|\hat f(\gamma_1)|^2\propto1+\cos2a\gamma_1$ is destroyed by interference — positivity is then carried by the higher zeros alone. The small-support regime ($f*\tilde f$ supported where no prime power lives) is the one the Connes–Consani theorems address.

---

## E10. Spectral analysis of Weil's quadratic form on test functions of bounded support (2026-11-07)

`python3 scripts/run_weil_form.py` (224 s). Module `weil_form.py`; 5 tests.

**Setup.** For a real test function $f$ supported in $[-a,a]$ (additive variable $u=\log x$) put $g=f*\tilde f$, so $\hat g=|\hat f|^2$. The arithmetic side of the explicit formula,
$$Q_a(f)=2F(\tfrac12)F(-\tfrac12)-2\sum_{n\le e^{2a}}\frac{\Lambda(n)}{\sqrt n}\,g(\log n)-(\gamma_E+\log\pi)\,g(0)+2\int_0^\infty\frac{g(0)e^{-2u}-g(u)e^{-u/2}}{1-e^{-2u}}\,du,$$
($F(s)=\int f e^{su}du$) uses only the prime powers up to $e^{2a}$ and the Gamma factor; under RH it equals $2\sum_{\gamma>0}|\hat f(\gamma)|^2$, and Weil's criterion says RH $\iff Q_a\ge0$ for every $a$. We restrict $Q_a$ to polynomials of degree $<J$ on $[-a,a]$ (Legendre basis), which splits into even and odd sectors, and compute the smallest generalised eigenvalue $\lambda_{\min}(a,J)=\min Q_a(f)/\|f\|_2^2$ on that subspace. This is a Rayleigh–Ritz *upper* bound for the infimum over all $f$ supported in $[-a,a]$. All matrices are assembled in Arb ball arithmetic at 60–150 digits, because the pole, prime and archimedean terms cancel to dozens of digits; float64 would give nonsense here, as Zhu (2026) also warns. The correlations $\int\phi_i(v)\phi_j(v-u)dv$ are integrated exactly by Gauss–Legendre quadrature. The archimedean integral uses a Gauss–Legendre rule whose node count is scaled to the precision, and its error is estimated by comparing two node counts.

**Validation.** For random test functions the arithmetic side (no zeros used) equals the zero side computed from our $10^6$ zeros plus an asymptotic tail to **$3.1\times10^{-9}$ relative**.

**Positivity.** All 100 smallest eigenvalues in the scan ($J=32,48$; $a=0.1,0.2,\dots,2.5$; both sectors) are positive, and each exceeds a Weyl-type perturbation bound (Frobenius norm of the ball radii plus the estimated quadrature error). This is a numerical certificate for these finite-dimensional subspaces, not a rigorous one: the quadrature error is estimated. For $a<(\log 2)/2$ no prime enters, and positivity is a theorem there (Yoshida 1992; Connes–Consani 2021).

| $a$ | prime powers used | $\lambda_{\min}$ even ($J=48$) | $\lambda_{\min}$ odd ($J=48$) |
|---|---|---|---|
| 0.1 | none | 0.457 | 1.33 |
| 0.5 | 2 | $9.4\times10^{-7}$ | $1.9\times10^{-4}$ |
| 1.0 | 2, 3, 4, 5, 7 | $2.0\times10^{-25}$ | $4.6\times10^{-23}$ |
| 1.5 | $n\le20$ (12) | $1.2\times10^{-34}$ | $1.1\times10^{-31}$ |
| 2.0 | $n\le54$ (24) | $9.3\times10^{-42}$ | $1.5\times10^{-39}$ |
| 2.5 | $n\le148$ (47) | $1.2\times10^{-46}$ | $3.5\times10^{-43}$ |

**Basis size.** At $a=1$ the even-sector value falls with $J=16,32,48,64,80$ as $9.8\times10^{-13}$, $1.6\times10^{-20}$, $2.0\times10^{-25}$, $2.5\times10^{-28}$, $7.8\times10^{-30}$, and is visibly slowing. At $a=2$ it falls as $9.0\times10^{-17}$, $2.5\times10^{-31}$, $9.3\times10^{-42}$, $1.7\times10^{-52}$, $1.2\times10^{-61}$, with no sign of saturation. So the true infimum at $a=2$ is below $1.2\times10^{-61}$. That is consistent with the certified upper bound $\approx3\times10^{-283}$ reported by Zhu (arXiv:2608.24827, read via a search summary only).

**Zeros from primes.** Let $r_k$ be the real zero of the minimiser's Fourier transform $\hat f_{\min}$ nearest to $\gamma_k$, with $\gamma_k$ computed independently by mpmath to 80 digits.

| $a$ (prime powers $n\le e^{2a}$) | $J$ | sector | $\lvert r_1-\gamma_1\rvert$ | $\lvert r_2-\gamma_2\rvert$ | $\lvert r_3-\gamma_3\rvert$ | $\lvert r_5-\gamma_5\rvert$ | $\lvert r_8-\gamma_8\rvert$ | zeros matched to $10^{-3}$ |
|---|---|---|---|---|---|---|---|---|
| 0.5 ($n=2$ only) | 48 | even | $5.6\times10^{-4}$ | 0.033 | 0.19 | – | – | 1 |
| 1.0 ($n\le7$) | 48 | even | $1.3\times10^{-21}$ | $2.4\times10^{-18}$ | $2.7\times10^{-16}$ | $2.5\times10^{-11}$ | $1.1\times10^{-5}$ | 8 |
| 1.0 ($n\le7$) | 48 | odd | $1.7\times10^{-20}$ | $9.9\times10^{-18}$ | $5.6\times10^{-16}$ | $9.6\times10^{-12}$ | $1.1\times10^{-6}$ | 9 |
| 1.5 ($n\le20$) | 48 | even | $1.1\times10^{-28}$ | $2.7\times10^{-22}$ | $7.6\times10^{-18}$ | $1.9\times10^{-11}$ | $5.3\times10^{-6}$ | 9 |
| 2.0 ($n\le54$) | 64 | even | $\mathbf{4.1\times10^{-45}}$ | $2.6\times10^{-35}$ | $5.2\times10^{-30}$ | $1.3\times10^{-21}$ | $1.7\times10^{-12}$ | **14** |
| 2.0 ($n\le54$) | 64 | odd | $4.1\times10^{-43}$ | $3.4\times10^{-34}$ | $3.0\times10^{-29}$ | $6.0\times10^{-23}$ | $4.9\times10^{-12}$ | 12 |

Both the eigenvalue and the distances are unchanged to all printed digits when the working precision goes from 130 to 200 digits (case $a=2$, $J=64$). Figures: `weil_form_lambda_min.png`, `weil_form_zero_match.png`, `weil_form_minimiser_fourier.png`. The last one shows $|\hat f_{\min}|$ for $a=1$ dipping at every zero of ζ up to $r=60$.

**Interpretation.**
* *The primes pin down the low zeros.* A $32\times32$ eigenproblem built from the 24 prime powers $n\le54$ and the Gamma factor produces a function whose Fourier transform vanishes within $4\times10^{-45}$ of $\gamma_1=14.1347\ldots$. Logically this is the explicit formula plus RH for the low zeros: a test function with tiny $Q_a$ must have $\hat f$ nearly vanishing at every $\gamma_n$. It is nonetheless a striking quantitative picture of how rigidly a few primes determine the bottom of the spectrum. It independently reproduces observations of Connes–Consani (arXiv:2106.01715), Connes–Consani–Moscovici (arXiv:2310.18423, 2511.22755) and arXiv:2610.32704; we read those only through search summaries, since arxiv.org is not reachable from this environment.
* *Weil positivity is extremely tight.* The positivity margin of the arithmetic side collapses super-exponentially in the support length: below $10^{-61}$ at $a=2$ (and $\approx10^{-283}$ per Zhu). Any proof of RH through positivity of the arithmetic side for supports of length $a$ must control quantities of that size. That is why the positivity theorems (Yoshida; Connes–Consani; Zhu's certified $a=0.8$) reach only small supports. It also quantifies barrier §4.2.
* *Are the minimisers Laguerre–Pólya functions? No (addendum, 2026-11-07).* Suppose the transforms $\hat f_{\min}$ had only real zeros and converged to ξ as $a$ grows. Then Hurwitz's theorem would give a route to RH. We counted zeros by the argument principle at 50–60 digits in rectangles $0.5\le\Re z\le120.5$, $|\Im z|\le8$, and located the non-real ones with Newton's method.
  * $a=1$, $J=48$, even sector: all 35 zeros are real.
  * $a=2$, even sector: there are non-real conjugate pairs: 4 pairs at $J=48$, 1 pair at $J=64$ (at $48.70\pm4.09i$, with a further near-real candidate at $78.24\pm0.13i$ that the two counting methods do not settle), and 6 pairs at $J=80$ (e.g. $30.62\pm2.80i$ and $51.50\pm0.45i$).
  * Most pairs sit near the edge of the range where $\hat f_{\min}$ tracks ζ's zeros; at least one lies inside it.

  So these finite-basis approximants are not in the Laguerre–Pólya class, and real-rootedness cannot be expected to pass to a limit along this family. This is the familiar obstacle that real-rootedness of approximants is as hard as RH itself.
* *A heuristic "capture law", tested and refuted (addendum, 2026-11-07).* Before testing we predicted, from a density argument, that the minimiser's Fourier transform could track ζ's zeros only up to about $T^*=2\pi e^{2a}$. The argument: a function of exponential type $a$ has real-zero density $a/\pi$, while ζ's zeros have density $\log(t/2\pi)/2\pi$. That would mean "primes up to $X$ see zeros up to $2\pi X$". `python3 scripts/run_weil_form_capture_law.py` (results in `weil_form_capture_law.json`) counts the leading zeros matched to $10^{-3}$ as the basis grows:

  | $a$ | $T^*$ | predicted cap $N(T^*)$ | $J=48$ | $J=64$ | $J=96$ | $J=128$ |
  |---|---|---|---|---|---|---|
  | 1.0 | 46.4 | 9 | 8 (even) / 9 (odd) | 13 / 13 | 18 / 17 | – |
  | 1.25 | 76.5 | 20 | 9 / 9 | 14 / 14 | 21 / 21 | **29 / 31** |

  The count passes the predicted cap and keeps growing with the basis, while the smallest eigenvalue nearly converges (at $a=1$: $2.5\times10^{-28}$, then $6.3\times10^{-30}$). The precision also improves at every zero: at $a=1.25$, $J=128$, $\gamma_1$ is matched to 46 digits, $\gamma_{10}$ to 27 and $\gamma_{29}$ to 3. **The heuristic is wrong.** Real-zero density $a/\pi$ holds only on average; locally an entire function of exponential type can vanish at many more points, paying in norm. The minimiser appears to track every zero, with precision degrading smoothly with height. The same flawed argument underlies the pre-registered E12 prediction $a^*\approx\tfrac12\log(\gamma_0/2\pi)$, which should therefore be read as an upper estimate of the detection threshold, not a sharp one.
* *Converged positivity margins (addendum, 2026-11-07).* `python3 scripts/run_weil_form_convergence.py` (results in `weil_form_convergence.json`) pushes the basis to $J=128$ at $a=0.8$ and to $J=192$ at $a=1$, at up to 297 digits. The Rayleigh–Ritz values decrease monotonically toward the true infimum:

  | $a$ | even sector, $J=48,64,96,128$ (and 160, 192) | extrapolated $\lambda^*(a)$ | odd sector $\lambda^*(a)$ |
  |---|---|---|---|
  | 0.8 | $1.746,\ 1.703,\ 1.656,\ 1.648\ (\times10^{-17})$ | $\approx1.64\times10^{-17}$ | $\approx1.55\times10^{-14}$ |
  | 1.0 | $J=96..192$: $6.28,\ 5.90,\ 5.83,\ 5.80\ (\times10^{-30})$ | $\approx5.7\times10^{-30}$ | $\approx1.44\times10^{-26}$ |

  The last digits converge algebraically, roughly like $J^{-2}$. That points to an endpoint singularity of the true minimiser, so a basis with built-in endpoint behaviour would converge faster. Practical meaning: Zhu's certified window $[-0.8,0.8]$ had to resolve a margin of about $1.6\times10^{-17}$; extending certification to $[-1,1]$ means resolving about $6\times10^{-30}$, 13 orders smaller. The even sector always carries the smallest eigenvalue, 3–4 orders of magnitude below the odd sector.
* *What this does not show.* Positivity for all $f$ supported in $[-a,a]$: only polynomial subspaces are examined. Anything rigorous: the quadrature error is estimated, not bounded. Anything about zeros above the heights the minimiser resolves.

---

## E11. Sensitivity of the Nyman–Beurling criterion, with a real RH violation as control (2026-11-07)

`python3 scripts/run_nyman_beurling_control.py --N 2000` (729 s). Module `nyman_beurling_periodic.py` (plus `nyman_beurling.offline_floor_zeta`, `detection_horizon_log_N`); 19 tests.

**Setup.** Let $F(s)=\sum c_nn^{-s}$ have real coefficients of period $q$ summing to $0$ over a period. Then the functions $\psi_k(x)=A(1/(kx))$, with $A(y)=c_1+\dots+c_{\lfloor y\rfloor}$, are bounded, supported in $(0,1]$, and have Mellin transforms $k^{-s}F(s)/s$. So $d_N^2=\min\|\mathbf 1_{(0,1]}-\sum_{k\le N}c_k\psi_k\|^2=\min_{A_N}\frac1{2\pi}\int|1-FA_N|^2|s|^{-2}dt$, exactly as for ζ. We use two controls with $q=5$:
* $L(s,(n/5))$, whose zeros lie on the critical line (GRH);
* the Davenport–Heilbronn function, which has zeros off the line (the first at $0.8085+85.699i$; E7).

The Gram matrix $\langle\psi_j,\psi_k\rangle$ has a closed form: summing the jumps of the step-function integrand and applying digamma asymptotics to the resulting periodic series gives an $O(q(h+k))$ formula. It agrees with an independent exact period-sum formula to $2.2\times10^{-16}$.

**Constants.**
* By the Hadamard product, $\sum_\rho\frac1{\rho(1-\rho)}=2\Lambda'/\Lambda(1)=\log\frac q\pi+\psi(\tfrac{1+e}2)+2F'(1)/F(1)$. For ζ this formula gives Burnol's $2+\gamma-\log4\pi$.
* $L(s,(n/5))$: $C=0.156557$. Here $F(1)=2\log\phi/\sqrt5$ is reproduced, and $F'(1)$ is checked against mpmath.
* Davenport–Heilbronn: the sum over all zeros is $0.195272$. Removing its 5 off-line quadruples below $t=300$ (which contribute $0.001193$) leaves the on-line constant $C_{\rm on}=0.194079$. The direct sum over its 204 computed on-line zeros plus a density tail gives $0.194083$.
* Off-line floor: the Mellin transform maps $L^2(0,1)$ onto the Hardy space $H^2(\Re s>\tfrac12)$, so each off-line zero forces a value at a point. The minimal-norm interpolation bound gives **$d_N^2\ge2.793\times10^{-4}$ for every $N$** (rigorous given the zeros).

**Results** ($d_N^2\log N$):

| $N$ | 10 | 100 | 300 | 1000 | 2000 | predicted limit |
|---|---|---|---|---|---|---|
| $L(s,(n/5))$ (GRH) | 0.320 | 0.153 | 0.145 | 0.136 | 0.136 | $C=0.157$ |
| Davenport–Heilbronn (RH false) | 0.122 | 0.155 | 0.163 | 0.167 | 0.175 | $C_{\rm on}=0.194$ (+ floor·log N = 0.002 at $N=2000$) |

Both sequences are within 13–16% of their on-line constants at $N=2000$, approaching from opposite sides. Davenport–Heilbronn's $d_{2000}^2=0.0231$ is 80 times its floor. The off-line zeros add only floor·$\log N\approx0.002$ to $d_N^2\log N$ at $N=2000$, about 1% of $C_{\rm on}$. That is far below the 15% finite-$N$ deviations both functions show. Under the model $d_N^2\approx C_{\rm on}/\log N+$floor, the floor dominates only for $\log N>C_{\rm on}/{\rm floor}\approx695$, i.e. **$N\approx10^{302}$**. Figure `nyman_beurling_control.png`.

**ζ with a hypothetical off-line zero.** Báez-Duarte's functions are not supported in $(0,1)$; their tail is $-B/x$ on $[1,\infty)$, whose Mellin transform is $B/(s-1)$. With that correction the floor of a pair $\tfrac12+\delta\pm i\gamma$ is computed exactly by `offline_floor_zeta`, and is $\approx4\delta/\gamma^2$. The detection horizon is $\log N^*=C\gamma^2/(4\delta)$ with $C=0.0462$, compared below with Li's for the same zeros ($\delta=0.1$):

| $\gamma$ | 14.13 | 100 | $10^3$ | $3\times10^{12}$ |
|---|---|---|---|---|
| Nyman–Beurling $N^*$ | $10^{10}$ | $10^{502}$ | $10^{50\,000}$ | $10^{4.5\times10^{23}}$ |
| Li $n^*$ | $2.2\times10^4$ | $1.6\times10^6$ | $2.1\times10^8$ | $6\times10^{27}$ |

**Interpretation.** RH is equivalent to $d_N\to0$. But on-line zeros make $d_N^2$ decay only like $C/\log N$, while a violation adds a floor proportional to $\delta/\gamma^2$. The criterion is therefore numerically blind: even the gross Davenport–Heilbronn violation ($\delta\approx0.31$ at height 86) is invisible for every computable $N$. This is a property of the $\log N$ scale of the criterion, not of our numerics. Any use of Nyman–Beurling toward RH must be a genuinely asymptotic argument. The finite-$N$ behaviour also warns against reading convergence off the data: neither $d_N^2\log N$ curve is within 10% of its limit at $N=2000$ (ζ's is 2% below at $N=10^4$, E2b).

---

## E12. Weil positivity against a real RH violation, and its detection horizon (2026-11-07 – 2026-11-09)

### Stage 1: the explicit formula without an Euler product (2026-11-07)

Modules `explicit_formula_general.py`, `scripts/compute_lchi5_zeros.py`; tests `test_explicit_formula_general.py`, `test_lchi5_zeros.py`.

* **Explicit formula for real, self-dual, periodic Dirichlet series** (ζ, $L(s,(n/5))$, Davenport–Heilbronn). The derivation uses the Dirichlet series of $-F'/F$ on one line $\Re s=c$ of absolute convergence only, so no Euler product is needed. The "prime" side becomes $-2\sum_n a_nn^{-1/2}g(\log n)$ over the coefficients $a_n$ of $-F'/F$, which for Davenport–Heilbronn are not supported on prime powers ($a_6=(1+\kappa^2)\log6$). It holds to relative residuals $10^{-16}$–$10^{-11}$ for all three series. For Davenport–Heilbronn, dropping the off-line quadruples breaks it by exactly their contribution (0.753 and 0.200 for two test functions), and dropping the $n$ that are not prime powers breaks it by 0.084.
* **Zeros of $L(s,(n/5))$ to height 1000:** 904 zeros, all simple and on the line. Ordinates are certified to $6\times10^{-14}$ by ball-arithmetic sign changes of $Z_\chi$, and the counts by winding numbers at $T=100,300,600,1000$. An independent mpmath check agrees to $5\times10^{-14}$.

### Stage 2: does Weil positivity see an off-line zero, and from how far? (2026-11-09)

`python3 scripts/run_weil_detection.py` (27 min; `--parts A23` re-runs only the arithmetic-side checks and the certificates, about 8 min). Results in `weil_detection_summary.json`. Modules `weil_form_general.py`, `weil_certificate.py`; 33 tests.

**Setup.** As in E10, $Q_a(f)$ is the arithmetic side of Weil's form for real $f$ supported in $[-a,a]$. Only the coefficients $a_n$ with $n\le e^{2a}$ and the Gamma factor enter, now for all three series. By the explicit formula $Q_a(f)=\sum_\rho\hat f(\tau_\rho)\hat f(-\tau_\rho)$ with $\tau_\rho=(\rho-\tfrac12)/i$. An on-line zero contributes $2|\hat f(\gamma)|^2\ge0$; an off-line quadruple $\tfrac12\pm\delta\pm i\gamma$ contributes $4\Re\hat f(\gamma-i\delta)^2$, an indefinite rank-2 form. To probe the height of an off-line zero we use *wave packets* $\phi(u)=(1-u^2/a^2)^mP_j(u/a)\{\cos,\sin\}(r_0u)$, $j<J$, whose transforms concentrate near $\pm r_0$ and decay like $|r\mp r_0|^{-m-1}$. Matrices are assembled in Arb at 40 digits, as in E10. Validation:
* the arithmetic side equals the zero side to $10^{-16}$ for ζ and $L(s,(n/5))$, and to $10^{-13}$ for Davenport–Heilbronn, whose off-line zeros contribute $O(0.1)$ there;
* ζ in the Legendre basis reproduces the E10 matrices to $10^{-50}$.

A negative eigenvalue on any subspace means $Q_a$ takes negative values. So every threshold below is an **upper bound** on the true support at which Weil positivity fails.

**A. Davenport–Heilbronn: the violation is detected.** For wave packets centred at each certified off-line zero, the smallest support with a negative eigenvalue (zero side, float64; grid step 0.02; best of five bases, $(J,m)\in\{(20,2),(36,1),(36,2),(20,4),(28,4)\}$). The last column splits the minimiser's zero side into its off-line and on-line parts.

| off-line zero | $\delta$ | first negative $a$ | coefficients used | basis $(J,m)$ | $\lambda_{\min}$ DH | $\lambda_{\min}$ $L(s,(n/5))$, same space | $\tfrac12\log(5\gamma/2\pi)$ | off-line / on-line |
|---|---|---|---|---|---|---|---|---|
| $0.8085+85.699i$ | 0.309 | 1.88 | $n\le42$ | (36, 1) | $-0.073$ | 0.65 | 2.11 | $-0.77$ / $+0.69$ |
| $0.6508+114.163i$ | 0.151 | 2.16 | $n\le75$ | (36, 1) | $-0.017$ | 0.75 | 2.26 | $-0.24$ / $+0.22$ |
| $0.5744+166.479i$ | 0.074 | 2.26 | $n\le91$ | (36, 1) | $-0.100$ | 0.86 | 2.44 | $-0.50$ / $+0.40$ |
| $0.7243+176.702i$ | 0.224 | 2.24 | $n\le88$ | (36, 1) | $-0.005$ | 0.53 | 2.47 | $-0.47$ / $+0.46$ |
| $0.8695+240.405i$ | 0.370 | 2.16 | $n\le75$ | (36, 1) | $-0.051$ | 1.13 | 2.63 | $-1.71$ / $+1.65$ |

* Every certified off-line zero below height 300 is detected, at a support 0.2–0.5 below the zero-density estimate. Each time, the minimiser's negative value comes from the off-line quadruple it is centred on.
* *Basis dependence of the first threshold* (`python3 scripts/run_weil_detection_checks.py`, 12 min, results in `weil_detection_checks.json`). All 120 bases with $J\in\{12,20,28,36\}$, $m\in\{0,1,2,3,4,6\}$, $r_0\in\{82,84,85.7,87.5,90\}$ detect the violation, at $a=1.86$–$2.18$.
  * By window order: 1.86–1.90 for $m\le1$, rising to 1.96–2.18 for $m=6$. Smoother test functions detect later.
  * By basis size: larger bases detect earlier ($J=36$: 1.86–1.96).
  * The most negative eigenvalue over $J=20$–$36$, $m=0$–$4$, $r_0=84$–$87.5$ is $+0.21$ at $a=1.8$, $-0.23$ at $a=1.9$ and $-0.67$ at $a=2.0$, with $L(s,(n/5))$ at $+0.87$, $+0.64$ and $+0.51$ on the same spaces.
* **Arithmetic side alone** (Arb, 40 digits, no zeros used; error bound from ball radii and quadrature estimates):

  | $a$ | basis | $\lambda_{\min}$ DH | $\lambda_{\min}$ $L(s,(n/5))$ |
  |---|---|---|---|
  | 1.86 | $J=36$, $m=1$, $r_0=85.699$ | $+0.003877$ | – |
  | 1.88 | same | $-0.0733917238922\pm4\times10^{-23}$ | $+0.64552$ |
  | 1.92 | same | $-0.273806898213\pm4\times10^{-23}$ | $+0.59945$ |
  | 2.0 | $J=20$, $m=4$, $r_0=85.7$ | $-0.290030152626\pm1\times10^{-26}$ | $+0.61062$ |

  Both sides agree with the zero-side values.

**A3. Machine-checked: Weil positivity fails for Davenport–Heilbronn on $[-1.88,1.88]$.** `rh.weil_certificate` encloses $Q_a(f)$ for an explicit $f$: the minimiser rounded to rationals, with its exact coefficients stored in the JSON. Every error is bounded:
* $g=f*\tilde f$ in closed form (polynomial times exponential, acb_poly algebra with guard digits);
* $a_n$ as Arb balls;
* the archimedean integral by `acb.integral` (rigorous), plus a mean-value bound on $[0,10^{-12}]$, where the integrand has a removable singularity. The bound includes the jump terms of test functions that do not vanish at $\pm a$; for the windowed functions used here those terms are zero.

| support | coefficients used | enclosure of $Q_a(f)$ | $\lVert f\rVert_2^2$ |
|---|---|---|---|
| $[-1.88,1.88]$ | $n\le42$ | $[-0.0733917240,\,-0.0733917238]$ | $1\pm2\times10^{-16}$ |
| $[-1.92,1.92]$ | $n\le46$ | $[-0.2738068983,\,-0.2738068981]$ | $1\pm4\times10^{-16}$ |
| $[-2,2]$ | $n\le54$ | $[-0.2900301527,\,-0.2900301525]$ | $1\pm5\times10^{-16}$ |

By the explicit formula, a theorem for this function (derivation in `explicit_formula_general`), $Q_a(f)=\sum_\rho\hat f(\tau_\rho)\hat f(-\tau_\rho)$, and zeros on the critical line contribute $|\hat f(\gamma)|^2\ge0$. So **the Davenport–Heilbronn function has a zero off the critical line**, proved from its first 42 Dirichlet coefficients, its functional equation and a finite computation. Such zeros have been known since Davenport–Heilbronn (1936); the point is how little data Weil's criterion needs. For comparison, Nyman–Beurling shows this violation only near $N\approx10^{302}$ (E11). Li's criterion, by `li.detection_horizon` (ζ-normalised), needs $n\approx3\times10^5$.

**B. ζ: how far up can it see?** Thought experiment: replace two adjacent zeros $\gamma_k,\gamma_{k+1}$ by the quadruple $\tfrac12\pm\delta+i\gamma_0$, $\gamma_0=(\gamma_k+\gamma_{k+1})/2$. The zero count is unchanged, as when a close pair collides and leaves the line. By the explicit formula, the arithmetic side of such a function equals its zero side, computed from the $10^6$ certified zeros within $\pm2000$ of $\gamma_0$ plus the density tail; the packets decay like $|r-\gamma_0|^{-10}$. Below, $a^*$ is the smallest support (grid step 0.05) at which $J=16$ packets at $r_0=\gamma_0$ ($m=4$) give a negative eigenvalue: median [min, max] over 8 consecutive pairs per height.

| $\gamma_0$ | $\tfrac12\log\frac{\gamma_0}{2\pi}$ | $\delta=0.4$ | 0.2 | 0.1 | 0.05 | 0.02 | 0.01 |
|---|---|---|---|---|---|---|---|
| 110 | 1.43 | 1.35 [1.30, 1.45] | 1.50 [1.45, 1.55] | 1.55 [1.50, 1.60] | 1.60 [1.55, 1.60] | 1.67 [1.60, 1.70] | 1.70 [1.70, 1.75] |
| 403 | 2.08 | 1.80 [1.60, 2.05] | 2.05 [1.90, 2.15] | 2.17 [2.15, 2.20] | 2.25 [2.20, 2.30] | 2.38 [2.30, 2.40] | 2.42 [2.35, 2.50] |
| 1 005 | 2.54 | 2.20 [1.85, 2.35] | 2.50 [2.20, 2.60] | 2.65 [2.50, 2.70] | 2.75 [2.65, 2.80] | 2.85 [2.80, 2.90] | 2.92 [2.80, 3.05] |
| 3 137 | 3.11 | 2.42 [2.10, 2.50] | 2.85 [2.50, 3.00] | 3.15 [2.95, 3.20] | 3.30 [3.20, 3.35] | 3.45 [3.35, 3.50] | 3.55 [3.45, 3.60] |
| 10 002 | 3.69 | 2.67 [2.40, 2.95] | 3.48 [3.00, 3.55] | 3.73 [3.30, 3.85] | 3.95 [3.85, 4.00] | 4.10 [4.05, 4.15] | 4.25 [4.20, 4.30] |
| 33 193 | 4.29 | 2.90 [2.55, 3.15] | 3.75 [3.15, 4.20] | 4.28 [3.70, 4.45] | 4.50 [4.00, 4.60] | 4.65 [4.50, 4.80] | 4.78 [4.65, 4.95] |
| 100 003 | 4.84 | 2.92 [2.80, 3.25] | 3.97 [3.55, 4.30] | 4.38 [4.20, 4.85] | 4.75 [4.65, 5.15] | 5.15 [4.95, 5.45] | 5.30 [5.20, 5.60] |
| 260 879 | 5.32 | 3.25 [3.15, 3.40] | 4.53 [4.00, 4.95] | 5.30 [4.70, 5.45] | 5.57 [5.25, 5.70] | 5.93 [5.65, 6.10] | 6.18 [5.90, 6.25] |
| 489 740 | 5.63 | 3.25 [2.85, 3.45] | 4.53 [3.80, 5.15] | 5.12 [4.45, 5.65] | 5.67 [4.95, 5.90] | 6.03 [5.65, 6.15] | 6.18 [5.95, 6.30] |

* *Law.* Least-squares slope of the medians against $\log(\gamma_0/2\pi)$:

  | $\delta$ | 0.01 | 0.02 | 0.05 | 0.1 | 0.2 | 0.4 |
  |---|---|---|---|---|---|---|
  | slope | $0.544\pm0.014$ | $0.524\pm0.010$ | $0.488\pm0.014$ | $0.439\pm0.021$ | $0.363\pm0.015$ | $0.216\pm0.017$ |

  For small $\delta$ the threshold follows the pre-registered zero-density prediction $\tfrac12\log(\gamma_0/2\pi)$ with an $O(1)$ offset. The mean offset is $+0.49$ at $\delta=0.01$, $+0.16$ at $0.05$ and $-0.07$ at $0.1$. Gross violations are seen earlier, and the more so the higher they are.
* *Robustness.* With bases $(J,m)=(24,4),(16,2),(32,3)$ at $\gamma_0\approx10^3,10^4,10^5$ (4 pairs each), the thresholds move by at most 0.35, and larger bases give lower ones, as an upper bound should. With $J=32$ the small-$\delta$ slope over these heights is 0.49 (e.g. $\delta=0.01$: 2.70, 3.90, 4.97). The float64 eigenvalues agree with 50-digit solves of the same matrices to 8 digits, and perturbing every zero by its accuracy ($10^{-10}$) changes nothing visible (`run_weil_detection_checks.py`). At the largest height, $\delta=0.01$, $\lambda_{\min}$ goes from $+2.9\times10^{-4}$ ($a=6.2$) to $-2.4\times10^{-3}$ ($a=6.25$).
* *Arithmetic-side check.* At $\gamma_0=102.52$ (zeros 30, 31), $\delta=0.1$, ζ's Arb arithmetic side minus the two on-line pairs plus the quadruple gives $\lambda_{\min}=+0.0256435598$ at $a=1.45$ and $-0.0122667835$ at $a=1.5$. The zero side agrees to $6\times10^{-10}$.
* *Additive model* (the quadruple added, no zeros removed, so it sits between two on-line zeros): thresholds are 1–2 units higher, e.g. 3.60, 5.05, 6.38 at $\gamma_0\approx10^3,10^4,10^5$ for $\delta=0.1$. An off-line zero crowded by on-line neighbours is harder to see.

**C. Comparison of detection horizons** ($\delta=0.1$; Li's $n^*$ from `li.detection_horizon`, Nyman–Beurling's $N^*$ from `nyman_beurling.detection_horizon_log_N`):

| $\gamma_0$ | 110 | 1 005 | 10 002 | 100 003 | 489 740 |
|---|---|---|---|---|---|
| Weil: largest prime power used, $e^{2a^*}$ | 22 | 200 | $1.7\times10^3$ | $6.3\times10^3$ | $2.8\times10^4$ |
| Li: $n^*$ | $1.9\times10^6$ | $2.1\times10^8$ | $2.6\times10^{10}$ | $3.1\times10^{12}$ | $8.2\times10^{13}$ |
| Nyman–Beurling: $\log_{10}N^*$ | 607 | $5.1\times10^4$ | $5.0\times10^6$ | $5.0\times10^8$ | $1.2\times10^{10}$ |

For every $\delta\ge0.01$ tried, $e^{2a^*}\le0.9\gamma_0$: primes up to the height of the zero suffice. Figures: `weil_detection_dh.png`, `weil_detection_zeta.png`, `weil_detection_horizons.png`.

**Interpretation.**
* *Weil positivity is the sensitive criterion.* Seeing an off-line zero at height $\gamma$ needs primes up to $\approx\gamma$, against $n^*\propto\gamma^2/\delta$ Li coefficients and $\log N^*\propto\gamma^2/\delta$ for Nyman–Beurling. For a real violation, Davenport–Heilbronn, 42 coefficients suffice, and the failure of positivity is certified rigorously.
* *Why the threshold sits at $\tfrac12\log(\gamma/2\pi)$* (**Heuristic**). $\hat f$ has exponential type $a$, so its real zeros have average density at most $a/\pi$. To expose the off-line term, the on-line contributions near $\gamma$ must be suppressed, i.e. $\hat f$ must nearly vanish at the zeros there, whose density is $\log(\gamma/2\pi)/2\pi$. This is possible once $a\gtrsim\tfrac12\log(\gamma/2\pi)$. Approximate vanishing slightly beyond that density costs norm, as E10's refuted capture law showed. This explains the $O(1)$ offsets, and why large-$\delta$ violations, whose off-line term grows like $e^{2a\delta}$, are seen earlier.
* *The reach is local in height.* By the same token, positivity at a fixed support constrains only the zeros below height $\sim2\pi e^{2a}$ (heuristic). Weil positivity on $[-a,a]$ is therefore a finite statement about low zeros, much like zero verification. Its efficiency is no shortcut to RH, which needs every $a$, while for ζ the positivity margin collapses super-exponentially in $a$ (E10).
* *What this does not show.* The thresholds are upper bounds from wave-packet subspaces: other test functions might detect earlier. The ζ part is a thought experiment that keeps all other zeros fixed. Its numbers are float64 on the zero side, cross-checked against the Arb arithmetic side at one height. Nothing here concerns the actual zeros of ζ, which are all on the line as far as computed.

---

## E13. Certified Weil positivity on whole windows, past the previous record (2026-11-09 – 2026-11-10)

`python3 scripts/run_weil_positivity.py` (31 min; `--parts upper` recomputes the upper bounds;
`--only 1.0 --merge` adds the window $L=1$, about an hour per sector). Results in `weil_positivity_summary.json`. Module `weil_envelope.py`; 12 tests.

**Statement (Machine-checked).** For each window $L$ in the table below and every real $f\in L^2$ supported in $[-L,L]$, the arithmetic side of Weil's quadratic form satisfies $Q(f)\ge c\,\lVert f\rVert^2$. Here $c=\min(c_{\rm even},c_{\rm odd})>0$ is certified in ball arithmetic. The form is
$$Q(f)=2F(\tfrac i2)F(-\tfrac i2)+\frac1{2\pi}\int_{\mathbb R}|F(t)|^2\,\Psi_L(t)\,dt,$$
with $F=\hat f$ and $\Psi_L(t)=\Re\psi(\tfrac14+\tfrac{it}2)-\log\pi-\sum_{\log n<2L}\frac{2\Lambda(n)}{\sqrt n}\cos(t\log n)$. It may take the value $+\infty$. Only the prime powers $n<e^{2L}$ and the Gamma factor enter; no zero of ζ is used. Weil's criterion says RH holds if and only if this positivity holds for every $L$. Positivity on $[-L,L]$ implies it on every smaller window. **The largest window certified here is $[-1,1]$**, with $c=5.1369\times10^{-30}$ there; only $n\in\{2,3,4,5,7\}$ and the Gamma factor enter. The previous record we could find was $[-0.8,0.8]$ (Zhu 2026, arXiv:2608.24827v2).

**Method: Zhu's envelope reduction, rebuilt in interval arithmetic.** We read Zhu's paper through WebFetch summaries of its arXiv abstract and HTML full text, not the PDF.
* *Envelope.* $\Omega(t)=\Re\psi(\tfrac14+\tfrac{it}2)-\log\pi$ increases on $t>0$, because $\Im\psi'<0$ in the upper half-plane. With $|\cos|\le1$ this gives $\Psi_L(t)\ge\beta:=\Omega(T)-A_L$ for $t\ge T$, where $A_L=\sum_{\log n<2L}2\Lambda(n)/\sqrt n$ is the comb constant.
* *Lower form.* Plancherel then gives $Q\ge R_T:=\beta I+\text{pole}+C_T$, where $C_T$ is the part of the form with $t\le T$.
* *Matrix.* In an orthonormal Legendre basis of each parity sector, the entries of $C_T$ decay super-exponentially beyond order $\approx eLT/2$.
* *Finite certificate.* A finite block $A$ is certified positive definite. The rest is controlled by explicit bounds: $\epsilon_D$ for the tail and $\epsilon_B$ for the coupling. Then
$$\lambda_{\min}(R_T)\ \ge\ \min(\lambda_{\min}(R_A),\beta-\epsilon_D)-\epsilon_B.$$

Every quantity is rigorous:
* *Quadrature.* The $t$-integrals use composite Gauss–Legendre with an a-priori Bernstein-ellipse bound (Trefethen 2008, Thm 4.5). The symbol is bounded on each ellipse by Arb on adaptively bisected boxes; Arb's complex digamma returns NaN on wide boxes. The poles of $\Omega$ at $\pm i(\tfrac12+2k)$ are excluded.
* *Bessel values.* The values $j_n(Lt)$, of order up to 1430, come from power-series starting values and three-term recurrences. These are run with guard digits sized to the "wrapping effect" of ball arithmetic, up to about 450 digits, and verified against 60-digit mpmath.
* *Tail and coupling bounds.* These follow from $|j_n(x)|\le x^n/(2n+1)!!$ and $i_n(c)\le c^ne^c/(2n+1)!!$.
* *Positive definiteness.* An approximate $LDL^T$ of the midpoint matrix is followed by a residual computed in ball arithmetic. If $A-\lambda_0I=\tilde L\tilde D\tilde L^T+E$ with $\tilde D\ge0$, then $\lambda_{\min}(A)\ge\lambda_0-\lVert E\rVert_F$. A ball Cholesky fails at this size because of the wrapping effect.

Zhu's own lower bound used a-priori error bounds and a verified floating-point Cholesky rather than interval arithmetic.

**Validation.**
* The envelope matrix equals the independently tested $u$-space Legendre matrices of `rh.weil_form`, minus the frequency tail above $T$, to $10^{-7}$. That figure is the accuracy of the float tail used in the check.
* $Q_A-R_A$ is positive semidefinite, as it must be.
* At $L=0.8$ Zhu's certificate is reproduced and sharpened: $8.9\times10^{-18}\to1.0104\times10^{-17}$. Our $\beta$ uses $\Omega(T)$ itself rather than Zhu's bound $\log(T/2\pi)-1/T$.

**Results.**

| $L$ | prime powers $n<e^{2L}$ | $A_L$ | $T$ (even/odd) | orders | even: certified $c$ | even: upper | odd: certified $c$ | odd: upper |
|---|---|---|---|---|---|---|---|---|
| 0.5 | 2 | 0.980 | 40/40 | ≤ 73 | $5.6725\times10^{-07}$ | $9.3521\times10^{-07}$ | $1.0642\times10^{-04}$ | $1.9410\times10^{-04}$ |
| 0.8 | 2, 3, 4 | 2.942 | 200/200 | ≤ 311 | $1.0104\times10^{-17}$ | $1.7026\times10^{-17}$ | $1.1194\times10^{-14}$ | $1.6197\times10^{-14}$ |
| 0.8047 | 2, 3, 4 | 2.942 | 200/280 | ≤ 423 | $5.2156\times10^{-18}$ | $9.5164\times10^{-18}$ | $6.2984\times10^{-15}$ | $1.0088\times10^{-14}$ |
| 0.81 | 2, 3, 4, 5 | 4.381 | 600/600 | ≤ 865 | $3.9121\times10^{-18}$ | $5.1081\times10^{-18}$ | $4.3057\times10^{-15}$ | $5.6149\times10^{-15}$ |
| 0.85 | 2, 3, 4, 5 | 4.381 | 600/600 | ≤ 907 | $3.3152\times10^{-20}$ | $4.3544\times10^{-20}$ | $3.7815\times10^{-17}$ | $5.2107\times10^{-17}$ |
| 0.9 | 2, 3, 4, 5 | 4.381 | 600/600 | ≤ 957 | $3.1171\times10^{-23}$ | $4.2322\times10^{-23}$ | $5.2544\times10^{-20}$ | $7.3724\times10^{-20}$ |
| 0.95 | 2, 3, 4, 5 | 4.381 | 600/600 | ≤ 1009 | $1.7700\times10^{-26}$ | $2.6438\times10^{-26}$ | $3.8265\times10^{-23}$ | $5.5097\times10^{-23}$ |
| 0.97 | 2, 3, 4, 5 | 4.381 | 600/600 | ≤ 1029 | $7.7733\times10^{-28}$ | $1.1684\times10^{-27}$ | $1.7681\times10^{-24}$ | $2.4735\times10^{-24}$ |
| 0.9729 | 2, 3, 4, 5 | 4.381 | 600/840 | ≤ 1429 | $4.8865\times10^{-28}$ | $7.1173\times10^{-28}$ | $1.1323\times10^{-24}$ | $1.6137\times10^{-24}$ |
| 1.0 | 2, 3, 4, 5, 7 | 5.852 | 2600/2600 | ≤ 3817 | $5.1369\times10^{-30}$ | $6.2828\times10^{-30}$ | $1.2954\times10^{-26}$ | $1.6787\times10^{-26}$ |

The upper bounds are rigorous enclosures (`rh.weil_certificate`) of $Q(f)/\lVert f\rVert^2$ for an explicit polynomial $f$: the Legendre minimiser of degree $<64$ for $L<0.9$, $<96$ beyond. So each window's $\inf Q/\lVert f\rVert^2$ is enclosed two-sidedly. At $L=0.8$ the even sector gives $[1.0104,\ 1.7026]\times10^{-17}$, against Zhu's $[0.89,\ 2.27]\times10^{-17}$. Each certificate needs $T$ above $T_1=2\pi e^{A_L}$. The envelope's loss must stay below the margin, and it does at $T=600$ for every $L\le0.97$. The one exception is the odd sector at $L=0.8047$ and $L=0.9729$: there the loss exceeded the odd margin at the first $T$. The positive-definiteness check caught it, the run was retried at $1.4T$, and the retry succeeded. The residuals ($10^{-33}$–$10^{-42}$) and the coupling bounds ($\le5\times10^{-33}$) lie far below the margins. One window takes 5–180 s per sector.

**The reach of the envelope reduction (barrier).** The comb constant is $A_L\sim4e^L$ (prime number theorem), so the cutoff $T_1=2\pi e^{A_L}$ grows doubly exponentially in $L$. The Legendre order needed, $\approx eLT_1/2$, grows with it:

| $L$ | 0.8 | 0.9729 | 1.0 | 1.1 | 1.2 | 1.5 | 2.0 |
|---|---|---|---|---|---|---|---|
| $A_L$ | 2.94 | 4.38 | 5.85 | 7.08 | 8.52 | 13.0 | 24.4 |
| order $\approx eLT_1/2$ | 130 | 660 | 3000 | $1.1\times10^4$ | $5\times10^4$ | $6\times10^6$ | $7\times10^{11}$ |

Zhu shows that $A_L$ is the best pointwise bound for the prime comb (Weyl equidistribution). So no implementation of this reduction gets past $L\approx1.1$–$1.2$; larger windows need an argument that is not pointwise in $t$. (E14 supplies one and certifies up to $L=1.6$.) The margins themselves also collapse, from $\approx10^{-17}$ at $L=0.8$ to $\approx10^{-27}$ at $0.97$ (even sector).

**Interpretation.**
* This is a theorem-level, zero-free certificate for a finite piece of Weil's criterion. On $[-1,1]$ it uses only $n\in\{2,3,4,5,7\}$ and the Gamma factor. It extends the largest certified window from 0.8 to 1, a quarter longer, as far as we could find. We could not survey the literature exhaustively.
* *What the certificate says about zeros (heuristic).* By E12's detection law, test functions of support $L$ resolve only zeros below height $\sim2\pi e^{2L}$, which is about 44 at $L=0.97$. Those zeros are long known to lie on the line. The certificate carries no new information about zeros; its content is structural. The positivity margins of the arithmetic side collapse super-exponentially. The certification cost grows doubly exponentially in $L$, i.e. like $\exp(c\sqrt{\text{height}})$ in the height of the zeros the window "sees". That is far more than zero verification costs for the same height.
* *What this does not show.* RH, or positivity on any larger window. The labels are: Machine-checked for the lower bounds and upper bounds; Heuristic for the remarks about heights.

---

## E14. A non-pointwise bound for the prime comb: certified windows past the pointwise barrier (2026-11-10)

`OPENBLAS_NUM_THREADS=1 python3 scripts/run_weil_comb.py --no-upper` reproduces the certified lower bounds in about an hour on one core. Without `--no-upper` it also runs the rigorous upper-bound enclosures, which at degree $J\ge160$ take hours each; that is why the table's upper bounds for $L\ge1.1$ are numerical. `--parts constants` recomputes the comb constants. A single-threaded BLAS matters: the joint solver ran ten times slower with OpenBLAS threads competing for the cores. Results are in `weil_comb_summary.json`. The certificates themselves are in `weil_comb_certificates.json`: $A'$ and the exact coefficients of $H$, so $(*)$ can be re-checked without this code. The new module is `weil_comb.py`; `weil_envelope.py` gains comb certificates and a blocked $LDL^T$. There are 14 tests in `tests/test_weil_comb.py`.

**Statement (Machine-checked).** For each window $L$ in the table below and every real $f\in L^2$ supported in $[-L,L]$, the arithmetic side of Weil's form (as in E13) satisfies $Q(f)\ge c\,\lVert f\rVert^2$, with $c=\min(c_{\rm even},c_{\rm odd})>0$. The certified constants are:

| window | $c$ | certificate |
|---|---|---|
| $[-1.1,1.1]$ | $1.1235\times10^{-38}$ | separable |
| $[-1.2,1.2]$ | $4.5724\times10^{-49}$ | separable |
| $[-1.3,1.3]$ | $4.6852\times10^{-62}$ | joint |
| $[-1.4,1.4]$ | $6.5434\times10^{-78}$ | joint |
| $[-1.5,1.5]$ | $2.6695\times10^{-97}$ | joint (ADMM) |
| $[-1.6,1.6]$ | $3.5422\times10^{-121}$ | joint (ADMM) |

**The largest window certified is now $[-1.6,1.6]$.** Only $n\in\{2,3,4,5,7,8,9,11,13,16,17,19,23\}$ and the Gamma factor enter. Earlier records were $[-1,1]$ (E13) and $[-0.8,0.8]$ (Zhu 2026). The pointwise envelope of E13 would need $T>T_1=7427$ at $L=1.1$, $3.2\times10^4$ at $1.2$, $1.3\times10^5$ at $1.3$, $1.9\times10^5$ at $1.4$, $2.8\times10^6$ at $1.5$ and $10^7$ at $1.6$.

**The idea: the comb only has to be bounded against test functions, not pointwise.** E13 bounded the prime comb $P_L(t)=\sum_{\log n<2L}c_n\cos(t\log n)$ by its maximum $A_L=\sum c_n$ for $t\ge T$. That forced $T>2\pi e^{A_L}$, which is doubly exponential in $L$. Suppose instead we have a constant $A'$ and a function $H$ whose frequencies all have modulus at least $2L$, with
$$A'-P_L(t)+H(t)\ \ge\ 0\qquad\text{for all real }t.\tag{$*$}$$
For real $f$ supported in $[-L,L]$, $|F|^2$ is the Fourier transform of the autocorrelation of $f$, which vanishes outside $(-2L,2L)$. So $\int_0^\infty H|F|^2=0$: **$H$ is invisible to every admissible test function.** On $[T,\infty)$ we then have $\Psi_L+H\ge\Omega(T)-A'=:\beta''$, and
$$Q(f)\ \ge\ \beta''\lVert f\rVert^2+\text{pole}+\frac1\pi\int_0^T\bigl(\Psi_L+H-\beta''\bigr)|F|^2\,dt.$$
This has the structure of E13, with $A_L$ replaced by $A'$. The rest of E13 carries over unchanged: the Legendre block, the tail and coupling bounds, and the residual-certified $LDL^T$. By Kronecker, $(*)$ is the same as a nonnegativity statement for a trigonometric polynomial on the torus of prime phases $\theta_p=t\log p$.

**Certificates for $(*)$: sums of squares.**
* *Separable (used for $L\le1.2$).* Write $P_L=\sum_pP_p(t\log p)$. For each prime, take a nonnegative trigonometric polynomial $S_p(\theta)=\sum_i|q_i(e^{i\theta})|^2$ of degree $M_p$ whose low coefficients match $-c_{p^k}/2$. Then $H_p$ is its part of frequency $>K_p$, and $A_p=s_0+\sum_k|2s_k+c_{p^k}|$ absorbs any mismatch.
  * The best $A_p$ for degree $M$ solves a small semidefinite program, which we solve with a log-barrier Newton method. For one cosine it reproduces the Fejér–Egerváry–Szász value $c/(2\cos\frac{\pi}{M+2})$. As $M\to\infty$ it tends to the Toeplitz eigenvalue $\lambda_{\max}[c_{p^{|j-l|}}/2]$, the best separable constant.
  * We take $M_p=\lfloor24/\log p\rfloor$, so the frequencies of $H$ are at most 24. This gives $A'\approx0.53A_L$.
* *Joint (used for $L=1.3$–$1.6$).* These are sums of squares on the torus whose monomials are the integers $n\le e^\Lambda$ built from the primes $p<e^{2L}$. $H$ may use every ratio $n/n'$ with $|\log(n/n')|\ge2L$.
  * The separable bound ignores the constraints coming from composite frequencies, such as "no mass at $\log12$" when $12>e^{2L}$; the joint certificate exploits them.
  * The dual is a moment problem on these monomials, solved by the same barrier method.
  * Its $H$ has small amplitude and frequencies at most $\Lambda$, which keeps the quadrature cheap.
  * At $L=1.3$ with $\Lambda=5.5$, $A'=4.578$, against the separable optimum 5.20 and $A_L=9.94$.
  * At $L=1.4$ with $\Lambda=6.5$, $A'=4.606$, against 5.47 and 10.29.
  * At $L=1.5$ the barrier method no longer fits: $\Lambda=8$ gives 619 monomials and 32181 free moments, and the barrier's Newton systems have one unknown per free moment. Instead, ADMM on the primal (`joint_gram_admm`) is cheap per step: each step is one eigendecomposition of the $619\times619$ Gram matrix, and the projection onto the constraints is a uniform shift per monomial, because different monomials constrain disjoint entries. It is over-relaxed ($\alpha=1.6$). After 25000 steps (28 min) it gives $A'=5.329$, against the separable optimum 6.83 and $A_L=13.02$; that brings $T_1''=1295$ down from 7041. The remaining mismatch, 0.11, is part of this $A'$, which is certified as it stands. H then has 8008 terms, evaluated at each node through the integer powers $n^{it}$.
  * At $L=1.6$ (nine primes) the same method with $\Lambda=9$ handles 1239 monomials and 118196 free moments. After 40000 steps (148 min, 3 BLAS threads, checkpointed) it gives $A'=5.933$, against the separable optimum 7.49, $A^*=4.89$ and $A_L=14.32$. So $T_1''=2370$, down from 11210 for the separable certificate and $10^7$ pointwise. The remaining mismatch, 0.32, is part of this $A'$; ADMM converges slowly at this size, and the mismatch costs about a factor $e^{0.32}$ in $T$. H has 27518 terms.

**Rigour.**
* The coefficients of $H$ are exact dyadic numbers, from integer Gram matrices or 4000-bit sums, checked to have radius zero. Its frequencies $\log(n/n')$ are recomputed at the working precision, and each is certified $\ge2L$ in Arb.
* $|H|\le\sum|h|\cosh(\omega b_e)$ bounds $H$ on every Bernstein ellipse analytically.
* The tail and coupling bounds use $\sup|\Psi_L+H-\beta''|\le\max(\Omega(T)-\Omega(0),\,A'+A_L+\sup|H|)$.
* Everything else is as in E13. The $LDL^T$ behind the residual certificate is now blocked and right-looking, with the Schur updates done as FLINT matrix products. This makes $N\approx900$ a matter of minutes.
* Two lessons from the process:
  * An $H$ with 53-bit coefficients gave matrix entries of radius $10^{-12}$. The midpoint matrix was then inconsistent at the $10^{-21}$ level, and its approximate spectrum showed spurious negative eigenvalues. The ball certificate would have refused, not lied.
  * `mpmath.quadosc` gets the slowly oscillating tails $\int_T^\infty\cos(\omega t)G_0^2$ wrong. The sine-integral formula settles them.

**Validation.**
* $Q_A-R''_A$ is positive semidefinite against the independent $u$-space matrices of `rh.weil_form` (at $L=0.5$ and $L=1$).
* The band-limit identity $\int_0^\infty\cos(\omega t)G_0(t)^2dt=0$ for $\omega\ge2L$ holds to $10^{-29}$.
* The certificate inequality is checked on $6\times10^5$-point grids and in Arb at random points. The separable certificates match the closed form; the joint ones match their barrier duals to $2\times10^{-3}$.
* At $L=0.5$ the comb certificate beats the pointwise one ($6.8\times10^{-7}$ vs $5.7\times10^{-7}$ at $T=40$).
* At $L=1$ it certifies at $T=300$ with $N=275$ modes, in 25 s per sector: $c=4.09\times10^{-30}$ (even) and $1.04\times10^{-26}$ (odd). E13 needed $T=2600$, $N=1909$ and about an hour per sector for $5.14\times10^{-30}$ and $1.30\times10^{-26}$.

**Results.**

| $L$ | prime powers $n<e^{2L}$ | $A_L$ | comb certificate | $A'$ | $T_1$: pointwise / comb | $T$ | $N$ | even: certified $c$ | even: upper | odd: certified $c$ | odd: upper |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1.0 | 2, 3, 4, 5, 7 | 5.852 | separable, $X=24$ | 3.022 | 2187 / 129 | 300 | 275 | $4.0904\times10^{-30}$ | $6.2828\times10^{-30}$ | $1.0422\times10^{-26}$ | $1.6787\times10^{-26}$ |
| 1.1 | 2, 3, 4, 5, 7, 8, 9 | 7.075 | separable, $X=24$ | 3.825 | 7427 / 288 | 400 | 394 | $1.1235\times10^{-38}$ | $1.9983\times10^{-38}$ | $4.5640\times10^{-35}$ | $7.6246\times10^{-35}$ |
| 1.2 | 2, 3, 4, 5, 7, 8, 9, 11 | 8.521 | separable, $X=24$ | 4.574 | 31535 / 609 | 800 | 836 | $4.5724\times10^{-49}$ | $7.2377\times10^{-49}$ | $2.9165\times10^{-45}$ | ($4.35\times10^{-45}$) |
| 1.3 | 2, 3, 4, 5, 7, 8, 9, 11, 13 | 9.944 | joint, $\Lambda=5.5$ | 4.578 | $1.3\times10^5$ / 612 | 800 | 904 | $4.6852\times10^{-62}$ | ($6.6\times10^{-59}$) | $4.3485\times10^{-58}$ | ($1.4\times10^{-55}$) |
| 1.4 | 2, 3, 4, 5, 7, 8, 9, 11, 13, 16 | 10.290 | joint, $\Lambda=6.5$ | 4.606 | $1.9\times10^5$ / 629 | 850 | 1031 | $6.5434\times10^{-78}$ | ($1.3\times10^{-67}$) | $1.0931\times10^{-73}$ | ($4.8\times10^{-64}$) |
| 1.5 | 2, 3, 4, 5, 7, 8, 9, 11, 13, 16, 17, 19 | 13.016 | joint (ADMM), $\Lambda=8$ | 5.329 | $2.8\times10^6$ / 1295 | 1684 | 2166 | $2.6695\times10^{-97}$ | ($5.8\times10^{-87}$) | $5.8556\times10^{-93}$ | ($1.0\times10^{-83}$) |
| 1.6 | 2, 3, 4, 5, 7, 8, 9, 11, 13, 16, 17, 19, 23 | 14.323 | joint (ADMM), $\Lambda=9$ | 5.933 | $10^7$ / 2370 | 2845 | 3423 | $3.5422\times10^{-121}$ | ($9.8\times10^{-97}$) | $1.2939\times10^{-116}$ | ($8.5\times10^{-93}$) |

At $L=1$, $L=1.1$ and $L=1.2$ (even) the upper bounds are rigorous enclosures (`rh.weil_certificate`) of $Q(f)/\lVert f\rVert^2$ for the Legendre minimiser of degree $<96$ (E13), $<160$ and $<192$. These are enclosed two-sidedly; the enclosure at $L=1.2$ took 4.3 hours. At $L=1.1$ this needed the minimiser's coefficients rounded exactly at about 100 digits. Rounding them through floats moves $Q(f)$ by about $10^{-33}$, which gave a valid but useless $4.7\times10^{-34}$. The $L=1.1$ enclosures took about two hours each. The bracketed values are numerical: the smallest eigenvalue $\lambda_J$ of the form on Legendre polynomials of degree $<J$ (`rh.weil_form`; $J=160$, $224$ at $L=1.2$). They estimate the infimum from above but are not rigorous; at this degree the rigorous enclosures of `rh.weil_certificate` take hours. The $\lambda_J$ still fall with $J$: at $L=1.2$ they are $7.94$, $7.24$ and $7.04\times10^{-49}$ at $J=160$, $192$ and $224$, at $L=1.4$ they are $2.6\times10^{-61}$ and $1.3\times10^{-67}$ at $J=128$ and $160$, at $L=1.5$ they are $5.9\times10^{-67}$, $6.8\times10^{-74}$, $2.3\times10^{-81}$ and $5.8\times10^{-87}$ at $J=128$–$224$, and at $L=1.6$ the value at $J=224$ is $9.8\times10^{-97}$. So for $L\ge1.3$ the true margins sit near the certified lower bounds. The residuals ($5\times10^{-48}$ at $L=1$ down to $10^{-92}$ at $L=1.4$, at 55–100 digits) and the coupling bounds ($\le2\times10^{-71}$) lie at least 14 orders of magnitude below the certified margins. One sector takes 25 s at $L=1$, about 10 min at $L=1.2$–$1.4$ about 40 min at $L=1.5$ (130 digits), and at $L=1.6$ (160 digits, $N=3423$) both sectors together take 5.4 hours on one core: they share one assembly (nodes, symbol with H, Bessel values) and are then certified in turn. At $L=1.6$ the number of Legendre modes is $1.1\,eLT/4$ per sector instead of $1.25\,eLT/4$; the coupling bound is still below $10^{-400}$, so the smaller safety factor costs nothing. The blocked $LDL^T$ keeps its working matrix as column blocks and the residual check runs in column blocks, so that both $N\approx3400$ blocks at 160 digits fit in the container's memory. At $L=1.5$ the block is assembled with each node's Bessel orders truncated where $|G_n(t)|<10^{-(\mathrm{dps}+40)}$, and the dropped part is covered by a rigorous radius.

**Comb constants and the next barrier.**

| $L$ | 0.8 | 0.9 | 1.0 | 1.1 | 1.2 | 1.3 | 1.4 | 1.5 | 1.6 | 1.7 |
|---|---|---|---|---|---|---|---|---|---|---|
| $A_L$ (pointwise) | 2.94 | 4.38 | 5.85 | 7.08 | 8.52 | 9.94 | 10.29 | 13.02 | 14.32 | 16.64 |
| $A'$ separable, $X=24$ (certified) | 1.53 | 2.27 | 3.02 | 3.83 | 4.57 | 5.32 | 5.59 | 7.02 | 7.72 | 9.23 |
| separable optimum | 1.52 | 2.24 | 2.98 | 3.77 | 4.49 | 5.20 | 5.47 | 6.83 | 7.49 | 8.93 |
| $A'$ joint, $\Lambda=6$ (certified) | 1.36 | 1.99 | 2.47 | 2.99 | 3.65 | 4.38 | 4.78 | 6.14 | 7.15 | 8.57 |
| $A^*$ (operator norm) | 1.22 | 1.55 | 1.93 | 2.29 | 2.70 | 3.20 | 3.70 | 4.29 | 4.89 | 5.56 |
| Fejér lower bound for $A^*$ | 1.05 | 1.41 | 1.75 | 2.15 | 2.56 | 3.04 | 3.53 | 4.09 | 4.67 | 5.30 |
| $T_1$ pointwise | 119 | 502 | 2187 | 7427 | $3.2\times10^{4}$ | $1.3\times10^{5}$ | $1.9\times10^{5}$ | $2.8\times10^{6}$ | $1.0\times10^{7}$ | $1.1\times10^{8}$ |
| $T_1$ separable / joint / $A^*$ | 29 / 24 / 21 | 61 / 46 / 30 | 129 / 74 / 43 | 288 / 124 / 62 | 609 / 242 / 93 | 1278 / 499 / 154 | 1680 / 746 / 255 | 7041 / 2918 / 457 | 14119 / 8025 / 839 | 64056 / 33015 / 1627 |

* Any $(*)$ implies $A'\ge A^*$, the norm of the comb operator on $L^2[-L,L]$: integrate $(*)$ against $|F|^2$.
* A Fejér window centred where all phases align (Kronecker) gives $A^*\ge\sum_{n<e^{2L}}c_n(1-\frac{\log n}{2L})\approx0.3A_L$. By the prime number theorem this is $(4+o(1))e^L/L$ (**Theorem** for the inequality; **Numerical** for the values).
* Consequently **every** certificate of the form $(*)$ leaves the cutoff $T_1\ge2\pi e^{A^*}$ doubly exponential in $L$; only the exponent shrinks, by about a factor 3 for $L\le2$. The Fejér bound alone gives $T_1\ge1.8\times10^3$ at $L=1.75$ and $1.2\times10^4$ at $L=2$.
* The separable certificate runs out near $L\approx1.3$. Joint certificates of feasible size close much of the gap to $A^*$; at $\Lambda=6$ they sit about 0.5–1.1 above $A^*$. At $L=1.5$ the joint certificate with $\Lambda=6$ would need $T_1\approx2900$; with $\Lambda=8$ (ADMM) it needs 1295, and the window is certified at $T=1684$. At $L=1.6$, $\Lambda=9$ gives $T_1''=2370$ and the window is certified at $T=2845$. The envelope approach with any comb certificate ends near $L\approx1.7$–$2$.
* The margins keep collapsing (even sector): $\lambda^*\in[1.1,2.0]\times10^{-38}$ at $L=1.1$ and $[4.6,7.2]\times10^{-49}$ at $1.2$ (both rigorous). At $1.3$, $1.4$, $1.5$ and $1.6$ they are about $10^{-61}$, $10^{-77}$, $2.7\times10^{-97}$ and $3.5\times10^{-121}$ (certified lower bounds; upper ends numerical).

**Where it stops on this machine: $L=1.7$.** (E15 tries a different comb certificate.)
* At $L=1.7$ ten primes enter (up to 29). The comb constants are $A_L=16.64$, separable optimum 8.93, joint with $\Lambda=6$ 8.57, and $A^*=5.56$.
* With $\Lambda=9$ (1408 monomials), ADMM reaches a certified $A'=7.45$ after 10000 steps. Its trace (6.32, still rising, as at $L=1.6$) points to an optimum near $A'\approx6.7$ for this monomial set (Numerical). Even that gives $T_1''\approx5104$, and certifying at $T\approx1.2\,T_1''$ would need about 7805 Legendre modes per sector at about 175 digits. One such ball matrix takes about 8 GB, and the blocked $LDL^T$ holds two or three at once: far beyond the roughly 11 GB available here. The margins also keep falling: $\lambda_J$ is $1.2\times10^{-86}$ and $9.8\times10^{-97}$ at $J=160$ and $192$.
* A better solver does not close the gap. The low-rank augmented Lagrangian (`joint_gram_alm`) matches the barrier optimum on small cases. At $L=1.6$ (1239 monomials) it ends at 5.948 against ADMM's 5.933, with its trace rising towards about 5.88 as the mismatch falls. So the $\Lambda=9$ optimum at $L=1.6$ is about 5.85–5.9, and the remaining gap to $A^*$ (about 1) is set by the size of the monomial set, which grows like $e^\Lambda$.
* So $L\approx1.6$ is the practical limit of the joint-certificate envelope on this hardware. $L=1.7$ would need $\Lambda\approx10$–11 (2400–4000 monomials) and a machine with roughly 32–64 GB of memory, or a comb certificate that is not a sum of squares over smooth numbers. (Numerical: the $\Lambda=9$ value and the resource estimates.)

**Interpretation.**
* The barrier of E13 came from bounding the comb pointwise. Bounding it only against band-limited test functions moves it, and the certified window grows from $[-1,1]$ to $[-1.6,1.6]$.
* The new limit is the operator norm of the comb, which still grows exponentially in $L$. Certificates of this envelope type cannot go far beyond $L\approx2$. Going further would need a treatment of the frequencies above $T$ that does not bound the comb by a constant there. (E15 gives one: the same certificates, checked along the trajectory $t\ge T$, are not bound by $A^*$. The envelope reduction then meets its own floor, and $[-1.7,1.7]$ is not certified.)
* The window $[-1.6,1.6]$ "sees" (heuristically, E12) zeros up to height $\sim2\pi e^{3.2}\approx154$. The certificate is a zero-free proof of a finite piece of Weil's criterion, not new information about zeros.
* *What this does not show.* RH, or positivity on any larger window.
* Labels:
  * **Machine-checked:** the certified constants, and the upper bounds at $L=1$.
  * **Theorem:** $A'\ge A^*\ge\sum c_n(1-\log n/2L)$.
  * **Numerical:** $A^*$, the joint dual values and the numerical $\lambda_J$.
  * **Heuristic:** the remarks about heights.

## E15. Trajectory certificates for the prime comb, and the envelope floor (2026-11-11)

`OPENBLAS_NUM_THREADS=1 python3 scripts/run_weil_trajectory.py` reproduces the trajectory checks (with and without wave packets), the local operator norms and the scans of the envelope floor in about 40 minutes. With `--quick` it takes about a minute and is part of `run_all.py --quick`. `--certify` repeats the attempt on $[-1.7,1.7]$ described below: 5.8 hours on one core, with a peak of about 9.7 GB; run nothing else alongside it. Results are in `weil_trajectory_summary.json`. The Gram factor of the $L=1.7$ certificate is in `weil_trajectory_gram_L1.7.npz`; the exact certificate is rebuilt from it in 10 s. New code: `rh.weil_comb.trajectory_check` (`rh.weil_envelope.beta_star` uses its constant) and `rh.weil_local` (wave packets). There are nine new tests: three in `test_weil_comb.py`, five in `test_weil_local.py` and one in `test_weil_envelope.py`.

**Outcome.**
* A new kind of comb certificate. The torus bound of E14 is needed only along the trajectory $t\ge T$, and there it can be checked directly (Machine-checked). At $L=1.5$, 1.6 and 1.7 this lowers the comb's cutoff by factors 1.5, 2.0 and 4.4. With wave packets (localised parts of $H$), the $L=1.7$ cutoff drops from $T_1''=10853$ to $T=1000$.
* **The window $[-1.7,1.7]$ is not certified.** At $T=2460$ the envelope form is indefinite: the even sector has a negative pivot $-0.141$, and the odd sector has $\lambda_{\min}\approx-2.8\times10^{-146}$. At $T=1000$, with the packets, both sectors are clearly indefinite.
* The comb is no longer what limits the method: the envelope reduction's own floor is. Below a cutoff that grows with $L$ (and, at $\beta=0.05$, with a few terms of $H$ just above frequency $2L$; E16), the near-minimisers of $Q$ keep more spectral mass above $T$ than their margin. The largest certified window remains $[-1.6,1.6]$ (E14); E16 certifies $[-1.7,1.7]$ with a gapped certificate.

**The idea: the comb has to be bounded only along the trajectory $t\ge T$, not on the whole torus.** The envelope reduction needs $\Psi_L+H\ge\beta$ only for $t\ge T$. E14 got this from the torus bound $(*)$, valid for every real $t$, which forces $T\ge2\pi e^{A'+\beta}$ with $A'\ge A^*$. But on a finite range $[T,T_2]$, the phases $(t\log p)_p$ trace a curve on the torus. That curve stays away from the points where $P_L-H$ is near its maximum: by Kronecker it gets close to every point eventually, but only after a very long time. So:
* for $t\ge T_2$, where $\Omega(T_2)-A'\ge\beta$, the torus bound gives $\Omega-P_L+H\ge\beta$;
* for $T\le t\le T_2$, check $\Omega(t)-P_L(t)+H(t)\ge\beta$ directly. This is a one-dimensional problem.

At $L=1.7$, the joint certificate ($\Lambda=9$, ADMM after 10000 steps; $A'=7.454$) has $\max(P_L-H)=6.166$ on $[2460,11410]$, against $A'=7.454$. The check certifies $\beta=0.05$ from $T=2460$ (and from 2450), a factor 4.4 below $T_1''=10853$. That would mean 3147 modes per sector instead of about 13900.

**Rigour of the trajectory check.**
* $G=P_L-H$ is evaluated in IEEE double on a grid of step $\delta=0.02$.
* Every transcendental value comes from Arb: the prime phases $p^{it}$ at the start of each block of 2048 points, and $p^{i\delta}$. The phases $m^{it}$ of the 1408 integers $m$ that occur are products of these. Each complex product has relative error at most $\sqrt5u$ (Brent, Percival and Zimmermann 2007; we use $3u$), and a sum of $k$ terms has error at most $\gamma_k\sum|\cdot|$. The coefficients are rounded to double, and the rounding errors are summed in Arb. At $L=1.7$ the resulting bound is $1.4\times10^{-9}$; the computed values agree with Arb to $10^{-13}$.
* Between grid points, $|G''|\le M_2=\sum c_n\log^2n+\sum|h|\omega^2$ (Arb; 1782 at $L=1.7$). So on a cell, $G\le\max(G_k,G_{k+1})+M_2\delta^2/8$.
* $\Omega$ is increasing (E13); on each group of 25 cells it is bounded below by its value at the start of the group, computed in Arb.
* Cells that fail are split into four, up to four times. At $L=1.7$, 447519 cells pass and 31 need one split. The check takes 30 s.
* Tests: agreement with Arb at random points; certification below $T_1''$ at $L=1.2$, confirmed on a dense float grid; refusal below the last violation; at $T<T_1''$, an envelope form that stays below the true form and certifies.

**Results.**

| $L$ | certificate | $A'$ | $T_1''$ (torus) | $T$ (trajectory) | $T_2$ | max of $P_L-H$ on $[T,T_2]$ | $2\pi e^{1+2L}$ | modes: $1.2\,T_1''$ / $T$ |
|---|---|---|---|---|---|---|---|---|
| 1.5 | joint, $\Lambda=8$ (E14) | 5.329 | 1295 | 853 | 1363 | 4.954 | 343 | 1763 / 977 |
| 1.6 | joint, $\Lambda=9$ (E14) | 5.933 | 2370 | 1180 | 2493 | 5.471 | 419 | 3423 / 1432 |
| 1.7 | joint, $\Lambda=9$ (10000 ADMM steps) | 7.454 | 10853 | 2450 | 11410 | 6.166 | 512 | 16571 / 3134 |
| 1.7 | the same, plus 893 wave packets | 7.454 | 10853 | 1000 | 11410 | – | 512 | 16571 / 1291 |

The trajectory constant $\beta=0.05$ is certified from $T$ on (Machine-checked). The last column shows the modes per sector, $\lceil1.1eLT/4\rceil+20$, at $1.2\,T_1''$ (E14's choice) and at the trajectory $T$. At $L=1.5$ and $1.6$ the stored E14 certificates are used. Their windows were certified in E14 with the torus bound; with the trajectory check the same windows would need about half the modes.

**The attempt at $[-1.7,1.7]$.** The run used $T=2460$, 3147 modes per sector, 195 digits, and 49173 nodes in 324 panels (quadrature error unit $5.5\times10^{-193}$). Assembly took 3.7 hours and the whole run 5.8 hours.

| sector | midpoint $LDL^T$ | $\lambda_{\min}(R_A)$ | residual | $\varepsilon_B$ | certified? |
|---|---|---|---|---|---|
| even | negative pivot $-0.141$ (index 233) | indefinite (the eigenvalue closest to 0 is $1.85\times10^{-150}$) | – | $2.8\times10^{-278}$ | no |
| odd | all pivots positive (smallest $3.9\times10^{-13}$) | $\approx-2.75\times10^{-146}$ | $1.2\times10^{-185}$ | $9.3\times10^{-279}$ | no |

The odd sector's shifted factorisation $R_A+2.75\times10^{-146}I=LDL^T+E$, with $\lVert E\rVert_F\le1.2\times10^{-185}$, bounds $\lambda_{\min}$ from below by $-2.75\times10^{-146}$. The Rayleigh quotient puts it at about the same negative value, so this sector sits right at its floor. With the 893 packets at $T=1000$ (1291 modes) the pivots are $-0.21$ (even, index 82) and $-0.012$ (odd, index 93).

**What limits $T$ now.**
* *The local operator norm* (**Theorem** + **Numerical**). Integrate $\Omega-P_L+H-\beta\ge0$ on $[T,\infty)$ against $|K(t-t_0)|^2$, with $K\in PW_L$ concentrated near $t_0\gg T$. This kills $H$, since $|K|^2$ has spectrum in $[-2L,2L]$, and gives $\lambda(t_0)\le\Omega(t_0)-\beta+o(1)$. Here $\lambda(t_0)=\max_K\int P_L(t_0+s)|K(s)|^2ds/\int|K|^2$ is the local operator norm of the comb at height $t_0$.
  * At $t_0=0$ it is $A^*$; it then drops quickly as the phases decorrelate.
  * At $L=1.7$: $\lambda(0)=5.556$, while the maximum of $\lambda(t_0)$ is 3.37 on $[100,300]$, 4.00 on $[300,1000]$, 4.01 on $[1000,2000]$ and 4.44 on $[2000,3000]$. At $L=1.6$ the corresponding values are 4.894, 3.24, 3.80, 3.68 and 4.16 (piecewise-constant $k$ on cells of width 0.025, step 0.05 in $t_0$).
  * So a certificate that only needs to hold on $[T,\infty)$ could, as far as the comb is concerned, go down to $T\approx154$ at $L=1.7$.
  * The joint certificate stops short of this because of isolated bumps where $P_L-H$ comes close to its torus maximum. At $L=1.7$ the last is at $t=2447.4$.
  * Wave packets can remove such bumps. These are localised parts of $H$: band-limited envelopes $(\sin(bs/m)/(bs/m))^m$ with carriers $\ge2L+b$ (`rh.weil_local`). A receding-horizon linear program removes every violation on $[1000,2500]$ at $L=1.7$ with 893 packets. With them, the trajectory check certifies $\beta=0.05$ from $T=1000$ (Machine-checked, 40 s).
  * The window cannot be certified at $T=1000$ all the same: the envelope form there is indefinite in both sectors (see the attempt above). This is the second limit below.
* *The envelope floor* (**Numerical**). A valid $\beta$ does not by itself make the envelope form positive. The reduction drops $\frac1\pi\int_T^\infty(\Psi_L+H-\beta)|F|^2$, and this has to be negligible for the near-minimisers of $Q$, whose margins are tiny ($10^{-30}$ at $L=1$, $10^{-78}$ at $L=1.4$). Scans of the envelope form locate the cutoff below which it is indefinite. They use the even sector, with $\beta=0.05$ imposed where the trajectory check does not certify it, so they are probes, not certificates.

| $L$ | $t_0=2\pi e^{1+2L}$ | $H=0$ | the joint certificate's $H$ |
|---|---|---|---|
| 1.0 | 126 | indefinite at 120 ($0.95\,t_0$), positive at 150 | $\Lambda=5$, $\sum\lvert h\rvert=9.6$: indefinite at 100, positive at 110 ($0.87\,t_0$) |
| 1.2 | 188 | indefinite at 170, positive at 190 ($1.01\,t_0$) | $\Lambda=5.5$, $\sum\lvert h\rvert=14.3$: indefinite at 190, positive at 210 ($1.12\,t_0$) |
| 1.4 | 281 | indefinite at 400 ($1.42\,t_0$), positive at 500 ($1.78\,t_0$) | $\Lambda=6$, $\sum\lvert h\rvert=18.6$: indefinite up to 750 ($2.67\,t_0$); E14 ($\Lambda=6.5$) certified at 850 |
| 1.6 | 419 | – | $\Lambda=9$, $\sum\lvert h\rvert=61.7$: indefinite at 1180 ($2.8\,t_0$); E14 certified at 2845 ($6.8\,t_0$) |
| 1.7 | 512 | – | $\Lambda=9$ plus 893 packets: indefinite at 1000 ($1.95\,t_0$); $\Lambda=9$, $\sum\lvert h\rvert=60.7$: indefinite at 2460 ($4.8\,t_0$): even pivot $-0.14$, odd $\lambda_{\min}\approx-2.8\times10^{-146}$ |

  * Heuristic for the scale. Under RH $Q(f)=\sum_\gamma|F(\gamma)|^2$. An $F$ of exponential type $L$ has about $Lt/\pi$ zeros in $[0,t]$, while $\zeta$ has $\frac t{2\pi}\log\frac t{2\pi e}$. So $F$ can vanish at every zero only up to $t_0=2\pi e^{1+2L}$: 126 at $L=1$, 281 at $L=1.4$, 512 at $L=1.7$. Beyond $t_0$ a near-minimiser must have small spectral mass, so the reduction needs $T\gtrsim t_0$.
  * Without $H$ the threshold is about $t_0$ at $L\le1.2$ and 1.4–1.8 $t_0$ at $L=1.4$. A large $H$ raises it: at $L=1.4$ the joint certificate ($\sum|h|=18.6$) leaves the form indefinite up to at least $2.7\,t_0$. Its contribution on $[T,\infty)$ is dropped with the rest. It is not a simple boundary term: putting the cutoff at a zero of the antiderivative of $H$ changes nothing. E16 traces it to a few terms of $H$ within a few $10^{-3}$ of frequency $2L$, against $\beta=0.05$; the size of $H$ does not matter.
  * So, as far as the comb is concerned, the trajectory removes the torus barrier, but the envelope's own floor now sets $T$. It grows faster than $t_0$ in the range measured: about $1.2\,t_0$ at $L=1$, above $4.8\,t_0$ at $L=1.7$ (even sector, with $\sum|h|=60.7$; E16: this includes the resonance of the term $30/1$). Whether it stays singly exponential is open.

**Interpretation.**
* E14's barrier, the operator norm of the comb, applies to certificates that hold for every real $t$. The envelope needs one only for $t\ge T$. Checked along the trajectory, the same sum-of-squares certificate lowers the comb's cutoff 4.4-fold at $L=1.7$, and wave packets lower it 10.9-fold.
* That is not enough for the window: the envelope reduction discards the spectral side above $T$, and there the near-minimisers of $Q$ still carry more mass than their margin. At $L=1.7$ the floor, with this $H$, lies above $T=2460$ in the even sector and right at it in the odd sector. Wave packets bring the comb's requirement down to the local operator norm, but not below the floor, and they raise it if they are large near $T$.
* Going further needs one of three things: a reduction that keeps part of the spectral side above $T$; certificates whose $H$ is small where the near-minimisers live; or a larger $T$. The last would mean about 4000–4500 modes per sector at $L=1.7$, roughly 10–13 GB per sector at 195 digits. This is a resource estimate, not a result. (E16 adds a fourth, which certifies $[-1.7,1.7]$ at $T=2460$: remove the frequencies of $H$ just above $2L$, and raise $\beta$.)
* Nothing here is information about the zeros, and nothing here bears on RH beyond the finite pieces of Weil's criterion certified in E13 and E14.
* Labels:
  * **Machine-checked:** the trajectory constants in the table (with and without packets), and the odd sector's lower bound $\lambda_{\min}\ge-2.75\times10^{-146}$.
  * **Theorem:** the local necessary condition $\lambda(t_0)\le\Omega(t_0)-\beta+o(1)$.
  * **Numerical:** the values of $\lambda(t_0)$, the LP for the packets, the indefinite pivots, and the scans of the envelope floor.
  * **Heuristic:** the scale $2\pi e^{1+2L}$.

## E16. Frequencies of $H$ just above $2L$; $[-1.7,1.7]$ certified with a gapped certificate (2026-11-11)

`OPENBLAS_NUM_THREADS=1 python3 scripts/run_weil_gap.py` reproduces everything below except the certification run, in about six and a half hours (3.7 of them for the two leading-block probes and one for the probe at $T=1000$). `--quick` (about 3 minutes, part of `run_all.py --quick`) reproduces the table of near-$2L$ terms and the two trajectory checks of the gapped $L=1.7$ certificate. `--certify` repeats the certification of $[-1.7,1.7]$ (below; about 6.2 hours and 10 GB; run nothing else alongside). Results are in `weil_gap_summary.json`; the wave packets are in `weil_gap_packets_L1.7.json` ($\beta=0.2$) and `weil_gap_packets_L1.7_beta0.05.json`. New code: `rh.weil_comb.gap_certificate`, `rh.weil_local.fix_dips` and the option `gap` of `rh.weil_local.fit_local_correction`; three new tests.

**Outcome.**
* The part of the envelope floor that E15 attributed to a large $H$ is a resonance. A few terms of $H$ within a few $10^{-3}$ of frequency $2L$ beat against the component of $|F|^2$ at frequency $2L$, on $[T,\infty)$, where the envelope reduction drops them. At $L=1.4$, $T=650$, $\beta=0.05$, one term out of 557 makes the form indefinite. The other 556, with $\sum|h|=18.25$, do not matter (Numerical).
* Such a term can take up to $|h|/2$ per unit of spectral mass above $T$ (an exact identity, and a heuristic for the size). The form then needs a larger $\beta$, or the term removed. At $L=1.4$ the term $33/2$ ($|h|/2=0.185$) raises the threshold $\beta^*$ by 0.10–0.15 at $T=400$, 500 and 650. Without the near-$2L$ terms the threshold is at most that of $H=0$, which falls steeply with $T$: 0.27, 0.01 and 0.00 at $1.42$, $1.78$ and $2.31\,t_0$ (Numerical).
* E14 certified its windows with $\beta=0.18$–$0.30$. E15 used $\beta=0.05$, below the $|h|/2=0.19$ of the term $30/1$ in its $L=1.7$ certificate. Moving the 35 terms within 0.002 of $2L$ into $A'$ makes the leading 400-mode block of the $L=1.7$ form at $T=2460$ positive definite. With the E15 certificate that block carries the negative pivot of the full E15 run (Numerical).
* The gapped $L=1.7$ certificate, with 14 wave packets, gives $\beta=0.2$ from $T=2460$ (Machine-checked). With it $[-1.7,1.7]$ is certified: $Q(f)\ge1.79\times10^{-150}\lVert f\rVert^2$ for every real $f$ supported there (Machine-checked; 3147 modes per sector, 6.2 hours). E15 could not certify this window.
* At $L=1.7$ the floor without the near-$2L$ terms is below $T=1000$ ($1.95\,t_0$; E15 placed it above $4.8\,t_0$). A certificate from $T=1000$ would need the comb bound there, where 47 dips of $\Omega-0.2-P_L+H$ remain on $[1000,2460]$ (Numerical).

**The finding: one term of $H$ makes the envelope form indefinite** (**Numerical**). $L=1.4$, $T=650$ ($2.31\,t_0$), even sector, 896 modes, 100 digits, $\beta=0.05$ imposed on $[T,\infty)$ (probes, as in E15):

| $H$ | terms | $\sum\lvert h\rvert$ | envelope form |
|---|---|---|---|
| joint certificate, $\Lambda=6$ | 557 | 18.62 | indefinite (1 negative eigenvalue) |
| without $n/d=33/2$ (gap $0.0034$ above $2L$, $h=+0.370$) | 556 | 18.25 | positive, $\lambda_{\min}\approx3.42\times10^{-78}$ |
| without the terms of gap $<0.01$ | 555 | 18.14 | positive, $3.81\times10^{-78}$ |
| without the terms of gap $<0.03$ | 547 | 17.68 | positive, $3.954\times10^{-78}$ |
| without the terms of gap $<0.1$ | 521 | 16.00 | positive, $3.954\times10^{-78}$ |
| $H=0$ | 0 | 0 | positive, $3.954\times10^{-78}$ |
| the full $H$, with $\beta=0.1$ | 557 | 18.62 | indefinite (1 negative eigenvalue) |
| the full $H$, with $\beta=0.2$ | 557 | 18.62 | positive, $5.28\times10^{-78}$ |

* Without the ten terms of gap $<0.03$, $\lambda_{\min}$ is that of $H=0$ to two parts in $10^4$. The other 547 terms ($\sum|h|=17.7$) do not matter.
* The floor is then where it is for $H=0$: with gap $0.03$ the form is indefinite at $T=450$ and positive at 500 ($1.78\,t_0$), against "indefinite up to 750" with the full $H$ (E15).
* So the size of $H$ does not matter; its frequencies within a few $10^{-3}$ of $2L$ do, at this $\beta$.

**Anatomy of the negative direction** (**Numerical**; an exact decomposition from the matrices, no truncation). With the full $H$ and $\beta=0.05$ the envelope form has one negative eigenvalue, $\lambda=-1.549\times10^{-55}$; E14 certified $Q\ge6.5\times10^{-78}\lVert f\rVert^2$ on this window. The form at $\beta$ is $R(\beta)=R(0)+\beta K$, where $K$ is the Gram matrix of the mass above $T$, $\frac1\pi\int_T^\infty|F|^2$. Differences of forms with and without parts of $H$ give the rest. For the unit eigenvector $f$:

| part of $R(0.05)$ at $f$ | value | |
|---|---|---|
| pole $+\frac1\pi\int_0^T\Psi_L\lvert F\rvert^2$ | $+4.32\times10^{-55}$ | |
| $\beta\times$ mass above $T$ | $+1.01\times10^{-54}$ | mass $2.02\times10^{-53}$ |
| the term $33/2$ on $[0,T]$ | $-1.70\times10^{-54}$ | 0.228 of $\lvert h\rvert\times$ mass |
| the other 556 terms on $[0,T]$ | $+1.06\times10^{-55}$ | $-3\times10^{-4}$ of $\sum\lvert h\rvert\times$ mass |
| total | $-1.55\times10^{-55}$ | |

* One term takes back 1.7 times what $\beta$ credits; the other 556 terms, with $\sum|h|=18.25$, contribute nothing that matters.
* The mass above $T$ is the endpoint oscillation of $F$: on $[690,1050]$ the part of $|F|^2$ at frequency $2L$ has the amplitude of its local mean to 0.1%. $|F|^2$ is $1.3\times10^{-58}$ at $T$ and $1.6\times10^{-55}$ at $t=690$.

**Thresholds** (**Numerical**). The form at $\beta$ is $R(0.05)+(\beta-0.05)K$. The smallest $\beta$ for which it is positive definite, $\beta^*$, follows by bisection on the inertia. At $T=650$ (two independent bisections, to $\pm0.0016$, intersected):

| $H$ | $\beta^*$ at $T=650$ |
|---|---|
| joint certificate, 557 terms | 0.103 |
| without $33/2$ | 0.011 |
| without the terms of gap $<0.03$ | between $-0.002$ and 0.001 |
| $H=0$ | between 0 and 0.001 |

The single term $33/2$ moves $\beta^*$ by 0.09, half of its $|h|/2=0.185$ (see below). This agrees with the probes: $\beta=0.1$ fails, $\beta=0.2$ holds.

**The floor in the $(T,\beta)$ plane** (**Numerical**). The same threshold at three cutoffs ($L=1.4$, even sector, 100 digits, $\lceil1.4eLT/4\rceil+30$ modes):

| $T$ | $T/t_0$ | $H=0$ | without the terms of gap $<0.03$ | full $H$ (557 terms) |
|---|---|---|---|---|
| 400 | 1.42 | 0.266 | 0.237 | 0.351 |
| 500 | 1.78 | 0.011 | 0.008 | 0.161 |
| 650 | 2.31 | 0.000 | 0.000 | 0.103 |

* Without the near-$2L$ terms the threshold is that of $H=0$, or slightly lower. The other 547 terms do not raise it.
* The near-$2L$ terms add 0.10–0.15 at every $T$ measured. That is of the order of $|h|/2=0.185$ for $33/2$, and it does not fall with $T$ in this range. With $\beta=0.05$ (E15) they alone keep the form indefinite.
* Without them the threshold falls steeply with $T$: from 0.27 at $1.42\,t_0$ to 0.01 at $1.78\,t_0$. So $\beta$ buys little $T$: with $\beta=0.27$ instead of 0.05 the floor of $H=0$ moves from $(400,500]$ to 400. The floor is set by $T$, and by the near-$2L$ terms.

**Why** (**Theorem** for the identity, **Heuristic** for the size). For a polynomial $f$ supported in $[-L,L]$, integrating by parts to the end gives $F(t)=\int f(x)e^{ixt}dx=e^{iLt}\varphi_+(t)-e^{-iLt}\varphi_-(t)$ exactly, with $\varphi_\pm(t)=\sum_k(-1)^kf^{(k)}(\pm L)/(it)^{k+1}$. For even real $f$, $\varphi_-=-\overline{\varphi_+}$, so $F=2\,\mathrm{Re}(e^{iLt}\varphi_+)$ and
$$|F|^2=2|\varphi_+|^2+2\,\mathrm{Re}\big(\varphi_+^2e^{2iLt}\big).$$
The part at frequency $2L$ has exactly the amplitude of the mean; for odd $f$ only its sign changes. A term $h\cos(\omega t)$ with $\omega=2L+g$ turns it into
$$h\cos(\omega t)|F|^2=h\,\mathrm{Re}\big(\varphi_+^2e^{-igt}\big)+\big(\text{terms at frequencies }\approx2L\text{ and }4L\big).$$
* On $(-\infty,\infty)$ the integral is exactly 0. The envelope keeps only $[0,T]$, so it carries $-\frac1\pi\int_T^\infty h\cos(\omega t)|F|^2$.
* Where $\varphi_+$ varies slowly on the scale $1/L$, the fast terms average out, and the slow term is at most $|h|/2$ times the mass density $2|\varphi_+|^2$. So a near-$2L$ term can take up to $|h|/2$ per unit of mass above $T$, and takes a fixed fraction of it when $g$ is small against the inverse width of that mass. For $g$ large the slow term oscillates and cancels.
* The envelope form credits the mass above $T$ with $\beta$ per unit. So it survives such a term only if $\beta$ exceeds about the coherent part of $|h|/2$. At $L=1.4$: $|h|/2=0.185$ for $33/2$; $\beta=0.1$ fails, $\beta=0.2$ holds.
* This fits E14 and E15. E14 certified $[-1.4,1.4]$, $[-1.5,1.5]$ and $[-1.6,1.6]$ with $\beta=\Omega(T)-A'=0.30$, $0.26$ and $0.18$, above $|h|/2$ for every near-$2L$ term of those certificates (at most 0.229, 0.109 and 0.108). E15 used $\beta=0.05$ throughout: in its floor scans with $H$, and in the $L=1.7$ attempt, whose certificate has the term $30/1$ with $|h|/2=0.194$ at gap $0.0012$.

The near-$2L$ terms appear only from $L\approx1.3$ on, which is where E15 saw the floor grow faster than $t_0$:

| $L$ | certificate | $\sum\lvert h\rvert$ | $\sum\lvert h\rvert$ within 0.005 / 0.01 / 0.03 of $2L$ | largest near-$2L$ terms ($n/d$: gap, $h$) |
|---|---|---|---|---|
| 1.0, 1.1, 1.2 | E14 | 38.3, 39.4, 44.4 | 0 / 0 / 0 | – |
| 1.3 | E14 | 14.1 | 0.006 / 0.014 / 0.132 | $55/4$: 0.0210, $+0.084$ |
| 1.4 | E14 ($\Lambda=6.5$) | 23.6 | 0.463 / 0.627 / 1.192 | $33/2$: 0.0034, $+0.458$; $117/7$: 0.0163, $+0.174$; $182/11$: 0.0061, $+0.158$ |
| 1.5 | E14 | 41.7 | 0.367 / 0.628 / 1.513 | $221/11$: 0.0003, $+0.217$; $143/7$: 0.0169, $+0.168$ |
| 1.6 | E14 | 61.7 | 0.518 / 1.007 / 2.421 | $25/1$: 0.0189, $+0.216$; $323/13$: 0.0127, $+0.206$; $221/9$: 0.0009, $+0.125$ |
| 1.7 | E15 | 60.7 | 0.562 / 0.817 / 1.846 | $30/1$: 0.0012, $+0.387$; $92/3$: 0.0232, $+0.147$; $91/3$: 0.0122, $+0.116$ |

The sum of squares has no reason to avoid these frequencies: $\log(n/d)$ for $n/d$ just above $e^{2L}$ is as cheap as any other frequency of $H$. At $L\le1.2$ the stored certificates happen to have none within 0.03 of $2L$.

**A gapped certificate at $L=1.7$** (**Machine-checked**). `gap_certificate` moves every term with $\log(n/d)<2L+g$ into $A'$; since $h\cos\ge-|h|$, the torus bound $(*)$ still holds.
* With $g=0.002$ it moves the term $30/1$ (gap 0.0012, $h=+0.387$) and 34 small ones, $\sum|h|=0.411$: $A'=7.865$ instead of 7.454.
* The trajectory check then fails at a few isolated dips above 2460. `fix_dips` removes them with wave packets whose carriers are $\ge2L+b+0.05$, so their spectrum stays 0.05 above $2L$: a linear program per group of dips that minimises $\sum|a|$ subject to a margin $\ge0.01$ wherever the packets reach.
* $\beta=0.05$ from $T=2460$: 3 packets ($\sum|a|=0.494$, at the dips $t=2819.1$, 3100.0 and 3689.1); minimum margin 0.0011, evaluation error $4\times10^{-9}$, $T_2=17205$; about a minute.
* $\beta=0.2$ from $T=2460$: 14 packets at 8 dips ($\sum|a|=1.548$); minimum margin 0.0024 (at $t=3517.35$), evaluation error $7.4\times10^{-9}$, $T_2=19989$; 876455 cells, 840 of them refined once; about a minute.

**Leading-block probes** (**Numerical**). The leading $N'\times N'$ block of the envelope form is the form on the first $N'$ modes, so a negative pivot there is one of the full form. At $T=2460$, even sector, 400 modes, 195 digits, $\beta=0.05$:
* the E15 certificate: one negative pivot, at index 233, value $-0.14126$. This is exactly the pivot of the full E15 run, found in 1.9 hours instead of 5.8;
* the gapped certificate with its 3 packets: positive definite (smallest pivot $1.03\times10^{-13}$).

**The floor at $L=1.7$ without the near-$2L$ terms** (**Numerical**). The envelope form at $T=1000$ ($1.95\,t_0$), both sectors, 1291 modes each, 195 digits, with the gapped $H$ ($g=0.002$, no packets) and $\beta=0.2$ imposed on $[T,\infty)$, is positive definite in both sectors: $\lambda_{\min}\approx1.1\times10^{-150}$ (even) and $6.5\times10^{-146}$ (odd). E15's attempt at $T=1000$, with the full $H$, 893 packets and $\beta=0.05$, had negative pivots $-0.21$ and $-0.012$. So at $L=1.7$, too, the floor that E15 placed above $4.8\,t_0$ was the resonance; without it the floor is below $2\,t_0$. A certificate at $T=1000$ needs $\beta=0.2$ to hold from $T=1000$ on: 54 dips of $\Omega-0.2-P_L+H$ go below 0 on $[1000,3700]$ (the deepest $-1.06$, at $t=1126.8$), and none above.

**Certification of $[-1.7,1.7]$** (**Machine-checked**). `run_weil_gap.py --certify`: the gapped certificate with its 14 packets, $T=2460$, $\beta=0.2$, 3147 modes per sector ($\lceil1.1\,eLT/4\rceil+20$), 195 digits, one thread.
* Both sectors were assembled together (49173 nodes in 324 panels, quadrature error unit $5.5\times10^{-193}$) in 4.7 hours. The $LDL^T$ certificate then took 46 minutes for the even sector and 45 for the odd one; 6.2 hours in all, peak memory 9.9 GB.
* Even sector: $\lambda_{\min}(R_A)\approx1.833\times10^{-150}$, certified $c=1.796\times10^{-150}$.
* Odd sector: $\lambda_{\min}(R_A)\approx9.746\times10^{-146}$, certified $c=9.550\times10^{-146}$.
* The tails of the basis do not matter at this precision ($\varepsilon_B\le3\times10^{-278}$, $\varepsilon_D\le10^{-300}$).
* So $Q(f)\ge1.79\times10^{-150}\lVert f\rVert^2$ for every real $f$ supported in $[-1.7,1.7]$. E15's run at the same $T$, with the ungapped certificate and $\beta=0.05$, was indefinite in both sectors.
* The certified constants fall by 19, 24 and 29 orders of magnitude from $L=1.4$ to 1.5, 1.6 and 1.7 ($6.54\times10^{-78}$, $2.67\times10^{-97}$, $3.54\times10^{-121}$, $1.79\times10^{-150}$). They are lower bounds, not the true margins.

**Labels.**
* **Machine-checked:** the trajectory constants of the gapped $L=1.7$ certificate, and the bound $Q\ge1.79\times10^{-150}\lVert f\rVert^2$ on $[-1.7,1.7]$.
* **Theorem:** the identity $|F|^2=2|\varphi_+|^2+2\,\mathrm{Re}(\varphi_+^2e^{2iLt})$ for even polynomial $f$, and the bound $|h|/2$ on the slow term.
* **Numerical:** the probes at $L=1.4$ (including the thresholds and the anatomy), and at $L=1.7$ the leading-block probes and the probe at $T=1000$.
* **Heuristic:** that the fast terms average out on $[T,\infty)$, and so the rule "$\beta$ above the coherent part of $|h|/2$".
* Nothing here bears on RH beyond the finite pieces of Weil's criterion that are certified.

---

## Summary of the state of the evidence after all experiments

Every criterion computed behaves exactly as RH predicts, to the accuracy and height reached (including Weil positivity of the arithmetic side on polynomial test-function spaces, E10). The controls behave as they must: Weil positivity fails, rigorously, for the Davenport–Heilbronn function, whose RH violation it detects from 42 coefficients (E12). For ζ, Weil positivity is now certified on whole windows up to $[-1,1]$ with the pointwise envelope (E13), up to $[-1.6,1.6]$ with a non-pointwise bound for the prime comb (E14), and on $[-1.7,1.7]$ with a gapped trajectory certificate (E16). These are zero-free proofs of finite pieces of Weil's criterion. Torus certificates of this envelope type are limited by the operator norm of the comb, which still grows exponentially in $L$. Trajectory certificates are not (E15). What limits the envelope then is its own floor: the near-minimisers of $Q$ keep spectral mass above $T$. At $L=1.7$, with E15's certificate, that floor lies above the cutoff the trajectory allows ($T=2460$). E16 traced the part of the floor that grew with the certificate's $H$ to a resonance of a few terms of $H$ just above frequency $2L$; a gapped certificate and a larger $\beta$ remove it, and with them $[-1.7,1.7]$ is certified ($Q\ge1.79\times10^{-150}\lVert f\rVert^2$). None of this is new evidence beyond the known verifications; the value is (i) all of it is reproduced from scratch with rigorous zeros and tested code, (ii) several second-order phenomena are quantified (the oscillation of $d_N^2\log N$ around Burnol's constant, the prime-driven structure of the Nyman–Beurling optimiser, the explicit-formula representation of the sequential criterion, the trivial-zero dominance in $c_k$, the constant $3/4$ in the Li archimedean part, the $1/\sqrt{\log n}$ margin of Robin's inequality on colossally abundant numbers), and (iii) the sensitivity of each criterion to a hypothetical off-line zero is calibrated. For Li, Nyman–Beurling, Báez-Duarte and the integral criteria it is essentially nil at the heights that matter. Weil positivity is the exception: it needs primes only up to about the height of the zero (E12). But this reach is local in height, so it is no shortcut either.
