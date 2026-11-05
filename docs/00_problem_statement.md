# 0. The problem, stated precisely

## 0.1 The zeta function

For $\Re s > 1$ the Riemann zeta function is defined by the absolutely convergent series and Euler product

$$\zeta(s)=\sum_{n\ge 1} n^{-s}=\prod_{p\ \mathrm{prime}}\bigl(1-p^{-s}\bigr)^{-1}.$$

The Euler product is the whole reason the function matters: it encodes the multiplicative structure of the integers, and its logarithmic derivative is the generating Dirichlet series of the von Mangoldt function,

$$-\frac{\zeta'}{\zeta}(s)=\sum_{n\ge1}\Lambda(n)\,n^{-s},\qquad \Lambda(p^k)=\log p,\ \Lambda(n)=0\ \text{otherwise}.$$

Riemann (1859) showed that $\zeta$ continues meromorphically to $\mathbb C$ with a single simple pole at $s=1$ (residue 1), and satisfies the functional equation. In its symmetric form, with

$$\xi(s):=\tfrac12\,s(s-1)\,\pi^{-s/2}\,\Gamma\!\left(\tfrac s2\right)\zeta(s),$$

one has $\xi$ entire of order 1 and

$$\xi(s)=\xi(1-s).$$

The factor $\Gamma(s/2)$ forces $\zeta$ to vanish at the negative even integers $-2,-4,\dots$ (the *trivial zeros*). All other zeros of $\zeta$ are zeros of $\xi$ and lie in the *critical strip* $0<\Re s<1$ (Hadamard, de la Vallée Poussin 1896: none on $\Re s=1$, hence by the functional equation none on $\Re s=0$). They are symmetric with respect to both the real axis and the *critical line* $\Re s=\tfrac12$.

## 0.2 The hypothesis

> **Riemann Hypothesis (RH).** Every zero of $\xi(s)$ — equivalently every nontrivial zero of $\zeta(s)$ — has real part $\tfrac12$.

Equivalently: writing nontrivial zeros as $\rho=\beta+i\gamma$, RH says $\beta=\tfrac12$ for all $\rho$. The zeros are then $\rho_n=\tfrac12\pm i\gamma_n$ with
$0<\gamma_1<\gamma_2<\cdots$ and $\gamma_1=14.1347251417\ldots$, $\gamma_2=21.0220396387\ldots$, $\gamma_3=25.0108575801\ldots$

A conjecture that usually travels with RH, and that most approaches would also prove, is that all zeros are **simple**.

### Hardy's function

On the critical line it is convenient to use the real-valued function

$$Z(t)=e^{i\theta(t)}\,\zeta\!\left(\tfrac12+it\right),\qquad \theta(t)=\arg\Gamma\!\left(\tfrac14+\tfrac{it}{2}\right)-\tfrac t2\log\pi,$$

(the Riemann–Siegel theta function; $\theta$ is chosen so that $Z$ is real). Zeros of $\zeta$ *on* the line are exactly the real zeros of $Z$; a sign change of $Z$ certifies a zero on the line. The Riemann–Siegel formula gives $Z(t)$ to high accuracy in $O(t^{1/2})$ operations and is the classical basis of zero verifications; the recent rigorous large-scale ones (Platt 2017, Platt–Trudgian 2021) use windowed FFT-based algorithms of Booker–Platt type, and Odlyzko–Schönhage-type multi-evaluation is used for the statistical computations at great height.

### Counting zeros

Let $N(T)$ be the number of zeros with $0<\gamma\le T$ (multiplicity counted). The Riemann–von Mangoldt formula is

$$N(T)=\frac{\theta(T)}{\pi}+1+S(T),\qquad \frac{\theta(T)}{\pi}+1=\frac{T}{2\pi}\log\frac{T}{2\pi e}+\frac78+O\!\left(\frac1T\right),$$

where $S(T)=\pi^{-1}\arg\zeta(\tfrac12+iT)$ (continuous variation along the horizontal path from $+\infty$) satisfies $S(T)=O(\log T)$ unconditionally and $S(T)=O(\log T/\log\log T)$ under RH. The mean spacing of zeros at height $T$ is therefore $2\pi/\log(T/2\pi)$: zeros get denser, slowly.

## 0.3 Why it matters: what RH is equivalent to and what it implies

* **Primes.** With $\psi(x)=\sum_{n\le x}\Lambda(n)$, the explicit formula $\psi(x)=x-\sum_\rho x^\rho/\rho-\log 2\pi-\tfrac12\log(1-x^{-2})$ shows that the error term in the prime number theorem is governed by $\sup\Re\rho$. RH is **equivalent** to
  $\psi(x)=x+O(\sqrt x\,\log^2 x)$ (von Koch 1901), and to $\pi(x)=\mathrm{li}(x)+O(\sqrt x\log x)$. Schoenfeld (1976) made this explicit: under RH, $|\pi(x)-\mathrm{li}(x)|<\frac{1}{8\pi}\sqrt x\log x$ for $x\ge 2657$.
* **Möbius / Mertens.** RH is equivalent to $M(x)=\sum_{n\le x}\mu(n)=O(x^{1/2+\varepsilon})$ for every $\varepsilon>0$ (Littlewood). (The stronger Mertens conjecture $|M(x)|<\sqrt x$ is *false*: Odlyzko–te Riele 1985 — a cautionary tale for numerical evidence.)
* **Lindelöf hypothesis.** RH implies $\zeta(\tfrac12+it)=O(t^{\varepsilon})$ (Lindelöf), which is itself open; the best unconditional exponent is $13/84$ (Bourgain 2017).
* **Dozens of elementary equivalents**: Robin's inequality for $\sigma(n)$, Lagarias' inequality with harmonic numbers, Nicolas' inequality for $\varphi$ at primorials, Li's criterion, the Nyman–Beurling–Báez-Duarte Hilbert-space criterion, Speiser's criterion for $\zeta'$, Weil's positivity, the de Bruijn–Newman constant $\Lambda\le 0$, … (see [02_equivalent_formulations.md](02_equivalent_formulations.md)). Many of them are implemented and tested numerically in this repository.

## 0.4 The generalised and grand hypotheses

The same statement is conjectured for every $L$-function with an Euler product and a functional equation: Dirichlet $L(s,\chi)$ (GRH), Dedekind zeta functions of number fields, Hecke/automorphic $L$-functions, and in general all functions in the **Selberg class** $\mathcal S$ (Grand RH). Several approaches — in particular the spectral and noncommutative-geometric ones, and the Rodgers–Tao / Dobner arguments for the de Bruijn–Newman constant — are naturally formulated at this level of generality.

The analogue of RH for curves over finite fields was proved by Hasse (genus 1, 1933) and Weil (1948), and for all smooth projective varieties over finite fields by Deligne (1974). Those proofs (together with the Selberg zeta functions of compact hyperbolic surfaces, whose "Riemann hypothesis" follows from the spectral theory of the Laplacian) are the only places where a "Riemann hypothesis" has ever been established, and they use a structure (the Frobenius endomorphism acting on cohomology, and positivity of an intersection form) that has no known counterpart over $\mathbb Z$. See [03_approaches_survey.md](03_approaches_survey.md).

## 0.5 What a proof has to deliver

Any argument that proves RH must

1. use the **Euler product** (or an equivalent arithmetic input). There exist Dirichlet series with a functional equation of the same type as $\zeta$'s (with their own gamma factor and conductor), the same growth and a similar zero density, that have zeros off the line (Davenport–Heilbronn 1936; see `src/rh/davenport_heilbronn.py`, which finds such zeros numerically). Functional equation plus analytic continuation is therefore not enough.
2. control **all** zeros, not almost all: the zero-density results (Ingham, Huxley, Guth–Maynard 2024) show that zeros with $\Re\rho\ge\sigma$ are rare, but "rare" is not "nonexistent", and the known positive proportions of zeros on the line (Selberg, Levinson, Conrey, …, currently just over 5/12 ≈ 41.7 %) are far from 100 %.
3. do something that is **not** a finite computation: RH has been verified for the first $1.2\times10^{13}$ zeros (height $3\times10^{12}$; Platt–Trudgian 2021), and statistically to far greater heights (Odlyzko, Gourdon), but a hypothetical first counterexample could lie at heights beyond any computation (compare Skewes' number and the failure of the Mertens conjecture).

Nothing in this repository claims to do any of these three things. The repository is a platform for exploring the problem honestly — see [05_research_program.md](05_research_program.md).
