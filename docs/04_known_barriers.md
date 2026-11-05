# 4. Known barriers: what any proof must get past

This page collects the structural reasons why the standard approaches stall. It is the most important page for anyone proposing a new attack, because every item below has killed at least one published or circulated "proof".

## 4.1 The Euler product is indispensable (Davenport–Heilbronn)

Let $\chi$ be the Dirichlet character mod 5 with $\chi(2)=i$, and $\kappa=\frac{\sqrt{10-2\sqrt5}-2}{\sqrt5-1}\approx0.2840$. The function
$$f(s)=\tfrac{1-i\kappa}{2}L(s,\chi)+\tfrac{1+i\kappa}{2}L(s,\bar\chi)$$
is a Dirichlet series with real coefficients, converges for $\Re s>1$, continues to an entire function and satisfies **exactly** a Riemann-type functional equation $f(s)=\left(\tfrac{5}{\pi}\right)^{1/2-s}\frac{\Gamma((2-s)/2)}{\Gamma((1+s)/2)}f(1-s)$. It has infinitely many zeros on the line, and *also* infinitely many zeros with $\Re s>\tfrac12$ (Davenport–Heilbronn 1936; the first is near $s\approx0.8085+85.699i$; `src/rh/davenport_heilbronn.py` finds them). It lacks only the Euler product.

**Consequence.** Any argument that uses only (a) analytic continuation, (b) the functional equation, (c) growth conditions, (d) the real-coefficient Dirichlet series structure — but not multiplicativity — cannot prove RH. This rules out, for instance, "proofs" via general properties of entire functions of order 1 (Laguerre–Pólya class arguments *alone*), via the functional equation symmetry $s\leftrightarrow1-s$ alone, or via generic Hilbert-space constructions that would apply verbatim to $f$. A huge fraction of amateur proofs fail precisely here.

Conversely, Euler product *without* a functional equation also fails: there are Euler products (e.g. $\zeta(s)\zeta(s-\tfrac12)$-type or partial Euler products, and Beurling generalised primes systems — Diamond–Montgomery–Vorhauer 2006) with zeros off the line. One needs both, interacting.

## 4.2 Positivity is the content, and positivity is global

Weil's criterion says RH $\iff W(g*\tilde g)\ge0$. Writing $W=W_\infty+W_{\mathrm{pole}}-W_{\mathrm{fin}}$ with $W_{\mathrm{fin}}(g)=\sum_n\Lambda(n)n^{-1/2}(g(\log n)+g(-\log n))$: the archimedean and pole terms are positive on $g*\tilde g$ for a reason that can be proved locally (this is essentially what Connes–Consani prove for small supports), but $W_{\mathrm{fin}}$ is *subtracted* and its positivity-destroying effect must be dominated by the other two for *every* test function. For a test function of support $[-\log X,\log X]$ the prime term involves all primes up to $X$ and the cancellation needed is equivalent to square-root cancellation in $\psi(x)-x$ *uniformly* — i.e. to RH. No local-to-global principle is known. The de Branges approach (positivity in a Hilbert space of entire functions) failed exactly because the required positivity condition does not hold for $\zeta$ (Conrey–Li 2000).

## 4.3 A Hilbert–Pólya operator must be arithmetic and cannot be "found" by matching statistics

* GUE statistics are shared by *all* $L$-functions in the unitary symmetry class and by random matrices with no arithmetic content; they cannot pin down an operator.
* The explicit formula fixes what the trace formula of a hypothetical $H$ must look like: periodic orbits of lengths $\log p^k$, amplitudes $p^{-k/2}$, *no* time-reversal symmetry, and a "Maslov"-type phase structure coming from $\log\pi$ and $\Gamma$-factors. Berry–Keating's $xp$ reproduces the counting function but no known quantisation produces the primes.
* Bombieri–Garrett (2011): in the natural automorphic settings (Friedrichs extensions of restrictions of the Laplacian on $\Gamma\backslash\mathfrak H$ designed so that zeros of $\zeta$ would appear as eigenvalues), one can prove that at most a *sparse* set of zeros can occur (because the regularity of the spectrum would contradict known gap statistics / pair correlation of zeros). So "zeros as eigenvalues of a nice differential operator on a nice space" is constrained severely.
* Self-adjointness must be *proved*, not assumed: Bender–Brody–Müller's $\hat H$ (2017) is formally correct but its symmetric realisation is not shown self-adjoint; without that the eigenvalue reality is not established.

## 4.4 Numerics cannot prove RH, and can mislead

* **Height.** RH has been verified for $t\le3\times10^{12}$. The relevant scale for a possible counterexample is not $t$ but $\log\log t$-type quantities (e.g. the first sign change of $\pi(x)-\mathrm{li}(x)$ is near $x\approx1.4\times10^{316}$ — Skewes/Bays–Hudson/Saouter–Demichel), so a counterexample at height $10^{10^{10}}$ is not excluded by anything we can compute.
* **$S(T)$ grows.** The maximum of $|S(T)|$ grows (like $\sqrt{\log T\log\log T}$-ish conjecturally; proven unboundedly); regions where $S(T)$ is large are where Gram's law fails badly and the zeros are "disordered". The GUE model predicts $S(T)$'s variance; a putative violation would be a *rare* event in that model, not an impossible one.
* **Lehmer phenomenon.** Pairs of zeros come arbitrarily close (relative to the mean spacing), and $Z(t)$ has local extrema arbitrarily close to zero of the "wrong" sign-type; if such an extremum ever failed to cross zero, RH would be false. Numerically these pairs are exactly what gives the lower bounds for the de Bruijn–Newman constant (and Rodgers–Tao show they *must* exist if RH holds!).
* **Mertens' conjecture** ($|M(x)|<\sqrt x$) held for all $x$ anyone could compute, had strong heuristic support, and is false (Odlyzko–te Riele 1985, via lattice reduction on the zeros; first violation expected near $10^{10^{23}}$—it has never been exhibited explicitly). Equivalent criteria that are "almost" RH at every finite scale can still be false.
* **Statistical equivalences are one-sided.** All the elementary criteria (Robin, Lagarias, Nicolas, Li, Nyman–Beurling, …) have been checked as far as possible and behave as RH predicts; they have the same evidential status as the zero verifications: strong but not decisive. In particular the Li coefficients $\lambda_n$ for $n\le10^4$ being positive is implied by the first $N$ zeros being on the line and the growth of $S(T)$; it is not independent evidence.

## 4.5 Analytic limits

* Zero-free regions: with exponential-sum methods the best imaginable is $\sigma>1-c/(\log t)^{2/3}$ shape; the "$2/3$" has stood since 1958. A region $\sigma>1-\delta$ would already be a major theorem (and gives $\psi(x)-x=O(x^{1-\delta})$).
* Density: Montgomery's large-values conjecture $\Rightarrow$ density hypothesis; even the density hypothesis gives nothing about *individual* zeros.
* Mollifiers: Levinson's method is bounded by mean-value theorems for long Dirichlet polynomials; even assuming optimal ("$\theta=\infty$") mollifiers the method does not reach 100 % (Farmer; Bui–Conrey–Young discuss what the method can and cannot do).
* Moments: the 6th and 8th moments are open; the Lindelöf hypothesis (weaker than RH) is open.

## 4.6 Logical barriers and "almost" statements

* Conrey: there is no known statement that is a *weak* form of RH and is provable—no "RH up to $\varepsilon$". Every known deformation (zero-free region, density, Lindelöf exponent) is of a different nature.
* RH is $\Pi_1$ (equivalent to a statement checkable by a Turing machine that halts iff RH is false—e.g. via Robin's inequality or Lagarias'), so it is *not* independent of PA in the way that would make it "true but unprovable" in the naive sense: if RH is false there is a finite counterexample; if it is true it might still be unprovable in a given system, but no evidence suggests this.

## 4.7 How these barriers shape this repository's program

* We do not attempt "proofs" that ignore the Euler product (4.1) — e.g. every criterion we test is one where the primes enter explicitly.
* We treat all numerics as **evidence and structure-finding**, not proof (4.4); every result is reported with its truncation or rounding error, and rigorous (ball) arithmetic is used where the statement is exact (zeros, Gram enclosures, high-precision Li coefficients).
* The two directions we think are most worth *computational* exploration are the structure of Weil positivity (which test functions are extremal; how the prime term is dominated) and the structure of the Nyman–Beurling optimal coefficients (how the optimiser "discovers" $\mu(k)$ and what taper it chooses), because both turn RH into a question about explicit finite-dimensional positive-definite objects whose asymptotics are unknown. See [05_research_program.md](05_research_program.md).
