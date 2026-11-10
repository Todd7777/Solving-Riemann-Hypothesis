# 8. Findings and outlook

This page is the synthesis: what the experiments in [06_experiment_log.md](06_experiment_log.md) actually established, ranked by how much they tell us, and what kind of result would constitute real progress. It is written to be read on its own.

**The headline is unchanged: the Riemann Hypothesis is open, and nothing here proves it.** Every computation behaves exactly as RH predicts. The main thing we learned is *how little* most of that agreement means, and where the numbers do carry information.

## 8.1 What the computations established

**1. Most equivalent criteria are numerically blind to a counterexample (E4, E5, E11).** We asked of each criterion: if ζ had a zero pair $\tfrac12+\delta\pm i\gamma$ with $\delta=0.1$, how large would the computation have to be to notice?

| criterion | $\gamma=100$ | $\gamma=10^3$ | $\gamma=3\times10^{12}$ (beyond verification) |
|---|---|---|---|
| Li coefficients $\lambda_n$ | $n\approx1.6\times10^6$ | $2.1\times10^8$ | $6\times10^{27}$ |
| Nyman–Beurling distance $d_N$ | $N\approx10^{502}$ | $10^{50\,000}$ | $10^{4.5\times10^{23}}$ |
| Báez-Duarte $c_k$ | effect $10^{-33}$ | $10^{-340}$ | $10^{-10^{12}}$ |
| Balazard–Saias–Yor integral | shift $2\times10^{-5}$ | $2\times10^{-7}$ | $2\times10^{-26}$ |

The Davenport–Heilbronn function, a genuine RH violation at height 86, confirmed this experimentally. Its Nyman–Beurling distances are indistinguishable from those of the GRH-true $L(s,(n/5))$ out to $N=2000$; the violation would show near $N\approx10^{302}$. Positivity of the first $10^4$ Li coefficients says nothing about zeros above height $\approx30$. **Consequence:** numerical confirmation of these criteria is not independent evidence for RH beyond the verified zeros. Their value is structural, not evidential.

**1b. Weil positivity is the exception: it sees off-line zeros from few primes, but only locally in height (E12).** On test functions supported in $[-a,a]$, Weil's form uses only the prime powers (for Davenport–Heilbronn, the Dirichlet coefficients) $n\le e^{2a}$.
* *Davenport–Heilbronn.* The form turns negative from $a\approx1.86$–$1.90$ on the least smooth wave-packet bases, and by $a=2.18$ on all 120 bases tried; only the coefficients $n\le42$ enter at $a=1.88$. The control $L(s,(n/5))$ stays positive on the same spaces. This is **machine-checked** at $a=1.88$: a rigorous enclosure gives $Q(f)\in[-0.0733917240,-0.0733917238]$ for an explicit $f$ with $\lVert f\rVert=1$. By the explicit formula the function therefore has a zero off the line, derived from 42 coefficients and its functional equation. All five of its known off-line zeros below height 300 are detected, each at a support below the zero-density estimate. Nyman–Beurling would need $N\approx10^{302}$, Li's criterion $n\approx3\times10^5$.
* *ζ, thought experiment.* Move a pair of adjacent zeros off the line to $\tfrac12\pm\delta+i\gamma$. The form detects it once $a\gtrsim\tfrac12\log(\gamma/2\pi)+c(\delta)$, measured for $10^2\le\gamma\le5\times10^5$ and $0.01\le\delta\le0.4$. For $\delta\le0.05$ the slope in $\log\gamma$ is ≈ ½, the zero-density value ($0.49$–$0.54$); gross violations are seen earlier. Primes up to $X^*=e^{2a^*}\le0.9\gamma$ sufficed in every case, against $n^*\propto\gamma^2/\delta$ for Li and $\log N^*\propto\gamma^2/\delta$ for Nyman–Beurling.
* *The catch* (**Heuristic**). The threshold is a resolution limit. A test function of support $[-a,a]$ has a transform of exponential type $a$, which can isolate one zero from neighbours spaced $2\pi/\log(\gamma/2\pi)$ apart only when $a\gtrsim\tfrac12\log(\gamma/2\pi)$. So positivity at a fixed support constrains (heuristically) only the zeros below height $\sim2\pi e^{2a}$. The criterion's finite-support content is local in height, like zero verification, and gives no shortcut to RH. Together with item 2, certifying positivity at support $a$ means resolving margins below $10^{-61}$ at $a=2$ while "seeing" only up to height $\sim2\pi e^4\approx340$.

**1c. Weil positivity is now certified on the window $[-1,1]$ (E13).** Zhu's envelope reduction, rebuilt here entirely in ball arithmetic, certifies $Q(f)\ge c\lVert f\rVert^2>0$ for every real $f$ supported in $[-L,L]$ (**Machine-checked**).
* *Reach.* Windows $L\le0.9729$ are certified in both parity sectors. $L=1$ is certified in the even sector, where $c=5.137\times10^{-30}$, and in the odd sector if the E13 entry says so. The previous largest certified window we could find was $L=0.8$ (Zhu 2026).
* *Inputs.* The certificates use only the prime powers $n<e^{2L}$ (at most 2, 3, 4, 5, 7) and the Gamma factor. No zeros are used.
* *Sharpness.* Our two-sided enclosure at $L=0.8$ is tighter than Zhu's: $[1.0104,1.7026]\times10^{-17}$. At $L=1$ the certified lower bound is within about 10% of the true margin.
* *The barrier.* The envelope's cutoff is $T_1=2\pi e^{A_L}$, with comb constant $A_L\sim4e^L$, so the cost grows doubly exponentially in $L$: Legendre orders ~3000 at $L=1$, $5\times10^4$ at 1.2, $6\times10^6$ at 1.5. The reduction cannot pass $L\approx1.1$–$1.2$ on any computer. Heuristically (item 1b), a window of half-width $L$ "sees" zeros up to height $\sim2\pi e^{2L}$, about 46 at $L=1$. So the certified statements concern a region where the zeros are long known to be on the line. They are zero-free proofs of finite pieces of Weil's criterion, not new information about zeros.

**2. Weil positivity is extraordinarily tight (E10).** The arithmetic side of Weil's quadratic form uses only the primes up to $e^{2a}$ and the Gamma factor. On test functions supported in $[-a,a]$ it stays positive in every case computed, but its smallest eigenvalue collapses super-exponentially: below $10^{-61}$ at $a=2$, with no sign of saturation in the basis size (a 2026 preprint reports $\approx10^{-283}$). Any proof of RH through positivity of the arithmetic side must control quantities of this size. That explains why the proved positivity results (Yoshida; Connes–Consani; Zhu's certified window $a=0.8$) stop at small supports.

**3. The primes pin down the low zeros with astonishing precision (E10).** The minimising test function of the restricted Weil form, built from the 24 prime powers $n\le54$, has a Fourier transform vanishing within $4\times10^{-45}$ of $\gamma_1=14.1347\ldots$ and within $10^{-3}$ of the first 14 zeros. This is the explicit formula at work, observed independently in the Connes–Consani program. It is the best "spectral realisation from primes" in this repository. The number of zeros tracked is not capped at height $2\pi e^{2a}$ as a density heuristic predicted. At $a=1.25$ it grows from 9 to 31 as the basis grows from 48 to 128 functions, with precision degrading smoothly with height (46 digits at $\gamma_1$, 3 at $\gamma_{29}$). But the minimisers are not Laguerre–Pólya functions: at $a=2$ they have non-real zeros. So the naive route "real-rooted approximants converging to ξ" is closed along this family.

**4. The second-order structure of the Nyman–Beurling problem (E2, E2b, E2c).**
* $d_N^2\log N$ oscillates around Burnol's constant $C=2+\gamma-\log4\pi$ and sits about 2% below it at $N=10^4$. That is allowed by Burnol's liminf theorem, but it means the conjectured limit is approached slowly and non-monotonically.
* The optimal coefficients follow $-\mu(k)(1-\log k/\log N)^{\alpha}$ with $\alpha\approx0.36$–$0.48$, much shallower than the $\alpha=1$ Möbius mollifier. The best pure power taper is a stable factor $\approx2.9$ worse than optimal at both $N=2000$ and $N=10^4$.
* These are new, quantitative observations in the spirit of Bettin–Conrey–Farmer. A mollifier family capturing the factor 2.9 would give a better explicit upper bound for $d_N$.

**5. Explicit-formula structure in the Báez-Duarte sequence (E3).** $c_k=\sum_j(-1)^j\binom kj/\zeta(2j+2)$ is reproduced to $10^{-9}$ for $k\le10^6$ by an explicit formula over 8 nontrivial and 12 trivial zeros. The trivial-zero term dominates until $k\approx2\times10^4$, which any numerical test of the $k^{-3/4}$ law must account for.

**6. An empirical growth law for the Li coefficients' oscillation (E4).** For $n\le10^5$ every $\lambda_n$ is positive, the linear constant matches RH's prediction to $3\times10^{-6}$, and the oscillating part grows like $n^{0.32\pm0.05}$. That is consistent with a heuristic $n^{1/4}(\log n)^{1/2}$ and far below the proven RH bound $O(\sqrt n\log n)$.

**7. Everything else is consistent with RH and with the random-matrix picture.** Zeros to $10^6$ certified by an independent Turing-method implementation; Robin, Lagarias and Nicolas inequalities to $10^7$ and $10^4$ primorials; colossally abundant numbers with Robin margin $\approx0.13/\sqrt{\log n}$; 1298 zeros of $\zeta'$ right of the line; GUE statistics approaching the limit like $c/(\log T)^2$; the explicit formula to $10^{-15}$.

## 8.2 What would constitute progress

Real progress on RH has always come from new structure, not from more verification. Measured against that, the honest options for a computational program are:

* **A new theorem-level computer-assisted result inside a live program.** *Done in E13:* Weil positivity is certified on windows past the previous record ($a=0.8$, Zhu 2026), up to $L=1$ (item 1c). The next step would need a new idea, not more computation: a bound for the prime comb that is not pointwise in $t$. The original framing follows. The prime candidate was *certified* Weil positivity on a window larger than the then record ($a=0.8$, Zhu 2026). This is a real (small) step in the Connes–Consani program. We measured the true margins: $\lambda^*(0.8)\approx1.64\times10^{-17}$ and $\lambda^*(1)\approx5.7\times10^{-30}$ (E10 addendum). Going from $0.8$ to $1$ therefore needs a finite reduction with error control 13 orders of magnitude finer, and a basis that handles the minimiser's endpoint singularity. E12 built part of the error control: `rh.weil_certificate` encloses the form rigorously at one explicit test function (closed-form correlations, rigorous integration). Certified positivity needs the same for a whole matrix, followed by a ball Cholesky factorisation and a bound on the part of the space the basis misses.
* **A sharpened conjecture with strong numerical support**, for example the second-order term of $d_N^2\log N$, or the law governing $\lambda_{\min}(a)$ and the number of zeta zeros the Weil minimiser captures. Such statements can guide theory.
* **Negative results that close routes**, like the Laguerre–Pólya finding in E10 and the blindness results in E11. These save effort and are publishable when sharp.

What cannot constitute progress: further verification of zeros at feasible heights, or further positivity checks of criteria whose detection horizon is astronomically beyond reach (§8.1, item 1).

## 8.3 Status of open threads

* **E13** (done, item 1c): certified Weil positivity on whole windows up to $L=1$ by the envelope reduction in ball arithmetic. Open: a non-pointwise bound for the prime comb, which is needed beyond $L\approx1.1$; the analogous certificates for $L(s,(n/5))$.
* **E12** (done, item 1b): Weil positivity detects the Davenport–Heilbronn violation at support $a\approx1.88$, where only $n\le42$ enters, certified rigorously. For ζ the detection threshold for an off-line zero at height $\gamma$ is $a^*\approx\tfrac12\log(\gamma/2\pi)+c(\delta)$, as pre-registered. Weil's criterion is therefore the right one to push computationally, but its reach is local in height. Open: whether test functions other than wave packets detect earlier (our thresholds are upper bounds), a theory for $c(\delta)$, and certified *positivity* on whole subspaces (the converse direction, needed for windows beyond Zhu's $a=0.8$).
* **Literature sweep** (running): a verified update of the 2024–2026 frontier for [01_state_of_the_art.md](01_state_of_the_art.md).
* **Next experiments** (being selected by a critique-and-rank process): candidates will be logged in [05_research_program.md](05_research_program.md).
