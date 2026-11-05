# 1. State of the art (as of late 2026)

This page records what is *proved*, with the sharpest known constants we are aware of, and what is only conjectured. Where we quote a specific number from memory of the literature we say so; the bibliography lists the primary sources.

## 1.1 Zero-free regions (how far from the line $\Re s=1$ zeros are known not to exist)

| Result | Zero-free region $\sigma > 1 - \ldots$ | Reference |
|---|---|---|
| de la Vallée Poussin (1899) | $c/\log t$ | classical |
| explicit constant | $1/(5.558691\,\log t)$ for $t \ge 2$ | Mossinghoff–Trudgian–Yang 2024 (improving 5.573412 of Mossinghoff–Trudgian 2015) |
| Vinogradov–Korobov (1958) | $c/\bigl((\log t)^{2/3}(\log\log t)^{1/3}\bigr)$ | sharpest *shape* known; explicit $c = 1/57.54$ (Ford 2002), improved to about $1/54$ (Bellotti 2024) |

No zero-free region of the form $\sigma > 1-\delta$ with a *fixed* $\delta>0$ is known ("the quasi-Riemann hypothesis" is open). The Vinogradov–Korobov shape has not been improved in 65 years; it comes from bounds for exponential sums (Vinogradov's mean value theorem, now sharp via Bourgain–Demeter–Guth 2016 / Wooley) and is believed to be close to the limit of that method.

## 1.2 Proportion of zeros on the line / simple

| Statement | Proportion | Reference |
|---|---|---|
| positive proportion on the line | $>0$ | Selberg 1942 |
| Levinson's method | $\ge 1/3$ | Levinson 1974 |
| | $\ge 2/5$ (40.77 %) | Conrey 1989 |
| | 41.05 % / 41.28 % | Bui–Conrey–Young 2011 / Feng 2012 |
| | $> 5/12$ (≈ 41.7 %) | Pratt–Robles–Zaharescu–Zeindler 2020 |
| simple zeros, unconditional | $\ge 40.7\%$ | via Levinson's method (Bui–Conrey–Young) |
| simple zeros assuming RH | $\ge 19/27 \approx 70.4\%$ | Bui–Heath-Brown 2013 |

Levinson's method is a mollifier method; it is understood (Farmer's "$\theta=\infty$" heuristic, and the work of Conrey and of Bui–Conrey–Young) that it cannot reach 100 % without a mollifier of length beyond what any known mean-value theorem controls. The "100 %" statement (all zeros on the line) is RH itself.

## 1.3 Zero-density estimates (how rare zeros off the line would be)

Let $N(\sigma,T)=\#\{\rho: \Re\rho\ge\sigma,\ 0<\Im\rho\le T\}$. Estimates of the form $N(\sigma,T)\ll T^{A(1-\sigma)+\varepsilon}$:

* Ingham (1940): $A=3/(2-\sigma)$ (so $A\le 3$), $A=2$ at $\sigma=1/2$;
* Huxley (1972): $A=12/5$ for $\sigma\ge 3/4$;
* **Guth–Maynard (2024, *Annals of Math.* 2026)**: $A=30/13\approx 2.31$ for all $\sigma$ — the first improvement on Ingham's exponent in the range $\sigma$ near $3/4$ in 80 years, via a new large-value estimate for Dirichlet polynomials. Consequence: primes in all short intervals $[x, x+x^{17/30+\varepsilon}]$.
* The **density hypothesis** $A=2$ is open; RH is the statement $N(\sigma,T)=0$ for $\sigma>1/2$.

Follow-up work (Tao–Trudgian–Yang 2025 and the collaborative "Analytic Number Theory Exponent Database") records the current frontier of such exponents.

## 1.4 Growth on the line (Lindelöf hypothesis)

$\zeta(\tfrac12+it)\ll t^{\mu+\varepsilon}$ is known with $\mu=13/84\approx0.1548$ (Bourgain 2017), via decoupling. The Lindelöf hypothesis $\mu=0$ follows from RH and is itself open. Lindelöf would follow from the density hypothesis-type statements for all moments; the 6th and 8th moments of $\zeta$ on the line are open (Hardy–Littlewood 1918 and Ingham 1926 proved the 2nd and 4th; Keating–Snaith 2000 and Conrey–Farmer–Keating–Rubinstein–Snaith 2005 conjecture all moments; Soundararajan 2009 and Harper 2013 give sharp upper bounds under RH; Heap–Radziwiłł–Soundararajan 2019 give sharp lower bounds unconditionally).

## 1.5 Numerical verification

| Height / count | Rigorous? | Reference |
|---|---|---|
| first $1.5\times10^9$ zeros | yes (Riemann–Siegel + Turing) | van de Lune–te Riele–Winter 1986 |
| $10^{13}$ zeros (height $2.4\times 10^{12}$) | floating point, no interval arithmetic; unpublished | Gourdon–Demichel 2004 |
| first $3.06\times10^{10}$ zeros | yes (interval arithmetic) | Platt 2017 |
| **height $3.0001753\times10^{12}$, $12\,363\,153\,437\,138$ zeros** | **yes** (interval arithmetic, Turing's method) | **Platt–Trudgian 2021** (Bull. LMS) |
| batches of $10^{8}$+ zeros near zero number $10^{20}$, $10^{21}$, $10^{22}$, $10^{23}$ | statistical (not a contiguous verification) | Odlyzko 1989–2001; Hiary; Bober–Hiary 2018 (near $10^{36}$) |

All verified zeros are simple and on the line. This repository recomputes zeros from scratch with Arb (rigorous ball arithmetic) — one million zeros plus samples at $n\approx10^{10}$ and $10^{12}$ — and independently re-implements the Riemann–Siegel/Turing machinery (`src/rh/zeta.py`, `src/rh/verify.py`).

## 1.6 The de Bruijn–Newman constant $\Lambda$

$H_t(z)=\int_0^\infty e^{tu^2}\Phi(u)\cos(zu)\,du$, with $H_0(z)=\tfrac18\xi(\tfrac12+\tfrac{iz}{2})$. $\Lambda$ is the smallest $t$ such that $H_t$ has only real zeros for all $t'\ge t$. RH $\iff \Lambda\le0$.

| Bound | Reference |
|---|---|
| $\Lambda\le 1/2$ | de Bruijn 1950 |
| $\Lambda<1/2$ | Ki–Kim–Lee 2009 |
| $\Lambda>-50$, …, $>-4.379\times10^{-6}$ (Lehmer pairs) | Csordas–Norfolk–Varga 1988; Csordas–Odlyzko–Smith–Varga 1993 |
| $\Lambda>-2.7\times10^{-9}$; $>-1.15\times10^{-11}$ | Odlyzko 2000; Saouter–Gourdon–Demichel 2011 |
| **$\Lambda\ge0$** (Newman's conjecture) | **Rodgers–Tao 2018** (Acta Math. 2020); Dobner 2021 for the Selberg class |
| $\Lambda\le0.22$ | Polymath 15 (2019) |
| **$\Lambda\le0.2$** | **Platt–Trudgian 2021** |

So RH is now **equivalent to $\Lambda=0$**: the zeros of $\xi$ are "at the edge" — the heat flow backwards by any amount creates non-real zeros. The Rodgers–Tao proof is a beautiful argument by contradiction using the dynamics of the zeros (see `src/rh/debruijn_newman.py` for the ODE and numerics); it gives no information about $\Lambda\le0$.

## 1.7 Statistics of zeros

* Montgomery (1973): pair correlation conjecture $R_2(\alpha)=1-\bigl(\frac{\sin\pi\alpha}{\pi\alpha}\bigr)^2$; proved for test functions with Fourier transform supported in $(-1,1)$ (under RH). Observed by Dyson to be the GUE kernel.
* Odlyzko (1987, 2001): spectacular numerical agreement of spacings and pair correlation with GUE at heights $10^{20}$–$10^{22}$, with lower-order deviations explained by Bogomolny–Keating (1996), Berry–Keating, and Conrey–Snaith (2008) ("ratios conjecture").
* Hejhal (1994, triple correlation), Rudnick–Sarnak (1996, $n$-level correlations for restricted test functions), Katz–Sarnak (1999, symmetry types in function fields).
* Everything beyond restricted test functions is conjectural. The statistics say *how* zeros are distributed along the line; they do not say *that* they are on it.

## 1.8 Equivalent criteria and their status

Li's criterion, the Nyman–Beurling–Báez-Duarte criterion, Robin/Lagarias/Nicolas inequalities, Weil positivity, Speiser's criterion, the Balazard–Saias–Yor and Volchkov integral criteria, the Riesz and Báez-Duarte sequential criteria: all are exact reformulations (see [02_equivalent_formulations.md](02_equivalent_formulations.md)). Numerically, every one of them behaves exactly as RH predicts as far as anyone has computed. None has led to a proof: in every case the reformulation moves the difficulty rather than removing it (see [04_known_barriers.md](04_known_barriers.md)).

## 1.9 Structural approaches

* **Function fields**: Weil 1948, Deligne 1974 — proved. The strategy (Frobenius, Lefschetz trace formula, Hodge index theorem / Castelnuovo–Severi inequality) is understood in detail; the obstruction over $\mathbb Z$ is the absence of a geometric object playing the role of $C\times_{\mathbb F_q} C$ ("$\overline{\mathrm{Spec}\,\mathbb Z}\times_{\mathbb F_1}\overline{\mathrm{Spec}\,\mathbb Z}$").
* **Spectral / Hilbert–Pólya**: the conjecture that the $\gamma_n$ are eigenvalues of a self-adjoint operator. Supported by the GUE statistics and by Selberg's trace formula analogy. Candidates: Berry–Keating $xp$ (1999), Connes' adele class space (1999) with the explicit formula as a trace formula, Sierra's and Bender–Brody–Müller's (2017) models (the latter's self-adjointness is not established; see critiques by Bellissard and others). No candidate operator has been proved to exist with the right spectrum.
* **Noncommutative geometry (Connes, Connes–Consani)**: RH $\iff$ positivity of Weil's functional $\iff$ a trace-formula positivity on the adele class space; recent papers (2019–2024) prove positivity for test functions of small support and construct "ζ-cycles" and a "scaling site", but RH remains open in this framework.
* **Deninger's program**: a conjectural cohomology theory for arithmetic schemes in which $\xi(s)$ is a regularised determinant $\det_\infty\!\bigl(\tfrac1{2\pi}(s-\Theta)\mid H^1\bigr)$ and RH follows from a Hodge-type positivity. Deninger (2000s–2020s) has built dynamical-systems models with the right formal properties; the cohomology itself has not been constructed.
* **Analytic**: Levinson–Conrey mollifiers (proportions), large sieve / density (Guth–Maynard), positivity of explicit-formula functionals, the de Bruijn–Newman heat flow, Rodgers–Tao. None reaches the line.

## 1.10 Claimed proofs

Every year several claimed proofs appear (including, in 2018, a sketch by Sir Michael Atiyah). None has survived scrutiny. The Clay Mathematics Institute lists RH as open. **This repository does not contain a proof of RH and does not claim one.**
