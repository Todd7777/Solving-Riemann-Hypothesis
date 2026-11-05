# 3. Approaches to a proof: ideas, status, obstructions

For each approach: the idea, what it has achieved, and precisely where it is stuck. This is the map we use to decide what is worth exploring computationally (see [05_research_program.md](05_research_program.md)).

## 3.1 The function-field model (the only place RH has been proved)

**Setting.** For a smooth projective curve $C/\mathbb F_q$ of genus $g$, $Z(C,T)=\frac{P(T)}{(1-T)(1-qT)}$ with $P\in\mathbb Z[T]$ of degree $2g$, and "RH" says all roots of $P$ have $|T|=q^{-1/2}$.

**Why it works.** (i) The zeros are eigenvalues of Frobenius on $H^1(C)$ (Weil/Grothendieck–Lefschetz); (ii) the number of zeros is finite ($2g$); (iii) positivity: the Castelnuovo–Severi inequality / Hodge index theorem on the surface $C\times C$ bounds the trace of Frobenius, $|\mathrm{tr}(\mathrm{Fr}^n)|\le 2g\,q^{n/2}$, which forces $|\alpha_i|=q^{1/2}$. Deligne's general proof replaces (iii) by the Rankin–Selberg/tensor-power trick and a weight argument.

**Obstruction over $\mathbb Z$.** There is no known cohomology theory in which $\xi(s)$ is a characteristic polynomial of a Frobenius, no analogue of the surface $C\times C$ ("$\overline{\mathrm{Spec}\,\mathbb Z}\times\overline{\mathrm{Spec}\,\mathbb Z}$" over the mythical $\mathbb F_1$), and infinitely many zeros. Programs trying to build these objects: Deninger (dynamical systems / foliated spaces, "Riemann–Roch for $\overline{\mathrm{Spec}\,\mathbb Z}$"), Connes–Consani (the scaling site, the arithmetic site, $\mathbb F_1$-geometry, "ζ-cycles"), Haran, Durov, Borger ($\Lambda$-rings), Lorscheid (blueprints). All reproduce *formal* features (e.g. a trace formula whose terms match the explicit formula); none produces the positivity that would prove RH.

**What is nevertheless transferable.** The *shape* of the argument: RH $\Leftarrow$ positivity of a quadratic form (Weil's explicit formula functional $W(g*\tilde g)\ge0$). This is exactly Weil's positivity criterion (§2, item 12) and is the common heart of the Connes and Deninger programs.

## 3.2 Weil positivity and the explicit formula

**Idea.** RH $\iff W(g*\tilde g)\ge0$ for all test functions $g$, where $W$ is the explicit-formula distribution (archimedean term + prime terms + pole terms). A proof would exhibit $W$ as a positive quadratic form by *construction* (e.g. as a trace of $A^*A$ on a Hilbert space) rather than by knowledge of the zeros.

**Status.** Yoshida (1992), Bombieri (2000), Burnol, Lagarias ("Hilbert spaces of entire functions and Dirichlet L-functions", de Branges) analysed the functional; Bombieri showed positivity holds for a certain class of test functions and proved a quantitative version of Weil's criterion. Connes–Consani (2021, *Weil positivity and trace formula, the archimedean place*) prove positivity for test functions supported in a small interval around 1 (the regime where only the archimedean place and finitely many small primes contribute), and in later work (2022–2024: "ζ-cycles", "Spectral triples and ζ-cycles", "the scaling Hamiltonian") construct spectral realisations of low-lying zeros in model Hilbert spaces.

**Obstruction.** Positivity for all test functions requires controlling the interaction of *all* primes (the "finite part" of $W$ is $-\sum_n\Lambda(n)n^{-1/2}(g(\log n)+g(-\log n))$, which has the wrong sign to be obviously positive). Equivalently one needs an inequality of large-sieve type that is sharper than anything the large sieve gives. The *de Branges* approach (Hilbert spaces of entire functions; de Branges' 1992 claimed proof was shown by Conrey–Li 2000 to rest on a positivity condition that is false for $\zeta$) is a well-documented failure mode here.

**In this repository.** `explicit_formula.py` evaluates both sides of the explicit formula for Gaussian families to $10^{-8}$ and computes the arithmetic side of the Weil functional (positive, as it must be under RH). This lets one *see* the sizes of the archimedean, prime and pole contributions and how they conspire to be positive — useful for forming conjectures about the *structure* of the positivity, not a route to a proof by itself.

## 3.3 Hilbert–Pólya: a self-adjoint operator with spectrum $\{\gamma_n\}$

**Idea.** Find $H=H^*$ on a Hilbert space with $\mathrm{spec}(H)=\{\gamma_n\}$; then the $\gamma_n$ are real.

**Evidence.** The GUE statistics of zeros (Montgomery 1973, Odlyzko 1987) are those of eigenvalues of large random Hermitian matrices without time-reversal symmetry; the explicit formula has the form of a Gutzwiller/Selberg trace formula with "periodic orbits" of "lengths" $\log p^k$ and "stability" $p^{-k/2}$; Berry–Keating (1999) identified the formal classical Hamiltonian $xp$ and derived the counting function $N(T)$ semiclassically, including the $7/8$; Sierra, Srednicki, and Bender–Brody–Müller (2017; the operator $\hat H=\frac{1}{1-e^{-i\hat p}}(\hat x\hat p+\hat p\hat x)(1-e^{-i\hat p})$ reproduces the zeros formally but its self-adjointness and the completeness of its eigenfunctions are not established — see critiques by Bellissard 2017 and others).

**Obstruction.** Three structural facts: (a) the operator must be *arithmetic* (its trace formula must produce the primes), so random-matrix models cannot be the answer; (b) Bombieri–Garrett (2011) show that for natural spectral realisations in automorphic settings (Friedrichs extensions of Laplacians on $\Gamma\backslash\mathfrak H$ restricted to subspaces), at most a sparse subset of zeros can appear as eigenvalues; (c) the Selberg trace formula analogy (zeros of Selberg zeta functions *are* eigenvalues, and the "Riemann hypothesis" for them is a theorem) succeeds because there the geometry comes first and the zeta function is derived; for $\zeta$ we have the function and no geometry.

**In this repository.** The GUE comparisons (`statistics.py`) quantify how close, at various heights, the zeros are to GUE and how the deviations scale ($\propto 1/\log T$, Bogomolny–Keating). This is reproduction of known phenomena, done rigorously from freshly computed zeros.

## 3.4 The de Bruijn–Newman heat flow

**Idea.** Deform $\xi$ by the backward heat equation: $H_t$ with $H_0\propto\xi$. $H_t$ has only real zeros for $t\ge\Lambda$; RH $\iff\Lambda\le0$. The zeros obey $\dot x_j=\sum_{k\ne j}\frac{2}{x_j-x_k}$ (repulsive Dyson-type dynamics) while they are real.

**Status.** Rodgers–Tao (2018) proved $\Lambda\ge0$ by showing that if $\Lambda<0$ the zeros at time $0$ would have to be too regularly spaced (approximately an arithmetic progression at scale $\log T$), contradicting the known small gaps (Lehmer pairs / pair correlation inputs of Montgomery). Platt–Trudgian (2021) combined numerics with Polymath 15's analysis to get $\Lambda\le0.2$.

**Obstruction.** The heat flow is a *relaxation* dynamics; it tells us the zeros at $t=0$ are "critical", not why. Proving $\Lambda\le0$ is exactly RH; the dynamics does not produce new inequalities for $t<0$ because the zeros are not real there.

**In this repository.** `debruijn_newman.py` integrates the dynamics, locates Lehmer pairs (closest pairs of zeros, which govern the lower-bound arguments), and reproduces the collision-time heuristic for a lower bound on $\Lambda$.

## 3.5 Nyman–Beurling–Báez-Duarte: an approximation problem in $L^2$

**Idea.** RH $\iff$ the indicator function $\chi_{(0,1]}$ can be approximated in $L^2$ by combinations of $\{1/(kx)\}$, equivalently $1/\zeta$ by Dirichlet polynomials in a weighted $L^2$ on the critical line. This turns RH into a question in approximation theory / linear algebra: does $d_N\to0$?

**Status.** The theorem is exact; Burnol's lower bound $d_N^2\gtrsim C/\log N$ is sharp conjecturally; Bettin–Conrey–Farmer showed under RH plus a hypothesis that the Möbius mollifier achieves it. Numerics (Landreau–Richard 2002; this repository up to $N=10^4$ with rigorous enclosures to several hundred) show $d_N^2\log N$ decreasing towards $C\approx0.0462$.

**Obstruction.** Showing $d_N\to0$ amounts to constructing good Dirichlet-polynomial approximations to $1/\zeta$ on the critical line *with control near the zeros*; the natural candidates (Möbius mollifiers) are controlled only under RH. The Gram matrix is explicit (Vasyunin sums), but there is no structural reason known for $b^TG_N^{-1}b\to1$.

**In this repository.** `nyman_beurling.py` — see the experiment log for results and the observed structure of the optimal coefficients.

## 3.6 Analytic methods (mollifiers, zero density, large values)

**Idea.** Show no zeros off the line by analytic estimates.

**Status.** Zero-free region of Vinogradov–Korobov shape; zero densities (Ingham, Huxley, Guth–Maynard 2024); Levinson–Conrey proportions (>41.7 %); Selberg's and Bombieri's large-sieve inequalities; the Bourgain decoupling bound for $\zeta$ on the line.

**Obstruction.** Each method has a known limit short of the line: Levinson's method is limited by the achievable mollifier length; density methods are limited by large-value estimates for Dirichlet polynomials (Montgomery's conjecture on large values would give the density hypothesis but not RH); zero-free regions are limited by exponential-sum bounds (even optimal Vinogradov mean-value theorems cannot give $\sigma>1-\delta$). Conrey's remark: "we do not have an *approximate* RH" — there is no known inequality that is "almost" RH.

## 3.7 Probabilistic / random-matrix / random-multiplicative models

The random matrix model predicts the statistics of the zeros, the moments of $\zeta$, and the distribution of $\log\zeta$ (Keating–Snaith); random multiplicative functions model $\mu$ and $M(x)$ (Harper 2020 showed better-than-squareroot cancellation in a related model). These are **models**, consistent with RH, that explain *why* RH should be true (zeros repel; $M(x)$ is of size $\sqrt x$ up to slowly varying factors), but cannot prove it: they do not model the specific arithmetic of $\mathbb Z$.

## 3.8 Summary table

| Approach | Reformulates RH as | Proved so far | What is missing |
|---|---|---|---|
| function fields | positivity on $C\times C$ | RH for all varieties over $\mathbb F_q$ | the geometry over $\mathbb Z$ |
| Weil positivity / NCG | $W(g*\tilde g)\ge0$ | small-support test functions | all primes together |
| Hilbert–Pólya | spectrum of self-adjoint $H$ | statistics match; formal models | an arithmetic operator |
| de Bruijn–Newman | $\Lambda\le 0$ | $0\le\Lambda\le0.2$ | $\Lambda\le0$ is RH itself |
| Nyman–Beurling | $d_N\to0$ | $d_N^2\gtrsim C/\log N$; numerics to $10^4$ | upper bound $d_N\to0$ |
| analytic | no zeros off line | zero-free regions, densities, 41.7 % | an "approximate RH" |
| probabilistic | typical behaviour | heuristics for everything | arithmetic specificity |
