# Solving-Riemann-Hypothesis — a research platform

> **This repository does not contain a proof of the Riemann Hypothesis, and does not claim one.**
> It is an honest, tested, reproducible platform for *doing research on* RH: rigorous computation of zeros, implementations of the classical equivalent criteria, numerical experiments that probe their second-order behaviour, and a careful survey of what is known and why the standard approaches stall.

The Riemann Hypothesis (RH): every nontrivial zero of $\zeta(s)$ has real part $1/2$. Open since 1859; verified for the first $1.2\times10^{13}$ zeros (Platt–Trudgian 2021); equivalent to dozens of statements in analysis, arithmetic and spectral theory; none of them proved.

## What is here

| Directory | Contents |
|---|---|
| `docs/` | The written research: precise problem statement, state of the art, equivalent formulations, survey of approaches, **known barriers**, the research program, the experiment log, bibliography |
| `src/rh/` | Python package: rigorous zeros (Arb), Riemann–Siegel/Turing verification, Li coefficients, Nyman–Beurling distances, Robin/Lagarias/Nicolas, Speiser, Davenport–Heilbronn, GUE statistics, de Bruijn–Newman flow, explicit formula / Weil positivity, Báez-Duarte sequential criterion |
| `scripts/` | Reproducible experiments (`run_*.py`), zero computation, `run_all.py` |
| `data/zeros/` | Precomputed rigorous zero ordinates ($10^6$ zeros, ball radius $\le1.2\cdot10^{-10}$; $1.5\cdot10^{-11}$ for the first $10^5$) and a sample of 6000 zeros at index $10^{10}$ |
| `results/` | JSON summaries and figures produced by the scripts |
| `tests/` | pytest suite (every module is tested against independent references) |

Start with [`docs/00_problem_statement.md`](docs/00_problem_statement.md) and [`docs/06_experiment_log.md`](docs/06_experiment_log.md).

## Highlights of the experiments

All numbers come with method and error estimates in [`docs/06_experiment_log.md`](docs/06_experiment_log.md). In brief:

* **Zeros.** One million zeros recomputed with Arb's rigorous isolation (ball radius $\le1.2\times10^{-10}$), plus 6000 zeros at height $3.3\times10^9$. An independent float64 Riemann–Siegel implementation with Turing's method certifies $N(600270)=1\,000\,000$ — all of the first $10^6$ zeros simple and on the line — in 152 s, agreeing with Arb's ordinates to $6.8\times10^{-9}$ (the Turing inequality and truncation bounds are rigorous; float64 rounding is estimated, not bounded).
* **Nyman–Beurling.** With the Vasyunin closed form for the Gram matrix of $\{1/(kx)\}$ (re-verified to 37 digits against direct integration), the Báez-Duarte distances $d_N^2$ are computed to $N=10^4$ (rigorous ball enclosures to $N=800$, agreeing with float64 to $10^{-14}$). $d_N^2\log N$ oscillates by a few percent around Burnol's constant $C=2+\gamma-\log4\pi=0.04619$ and at $N=10^4$ sits $2.3\%$ *below* it ($0.04512$) — allowed by Burnol's liminf theorem, which then forces the sequence to rise again: the conjectured limit is approached slowly and non-monotonically, and no extrapolation from $N\le10^4$ is trustworthy. For $N\lesssim50$ the drops of $d_N^2$ happen at the primes, but beyond that no arithmetic class of $N$ is privileged; the optimal coefficients have sign $-\mu(k)$ for $99.9\%$ of squarefree $k$ with a taper close to $(1-\log k/\log N)^{1/2}$, shallower than the $1-\log k/\log N$ mollifier (which is $9.6\times$ worse than optimal at $N=10^4$; the best pure Möbius power taper is $2.9\times$ worse).
* **Li coefficients.** $\lambda_n>0$ for $n\le10^4$, rigorously (Arb enclosures from the power series of $\log\xi$) for $n\le5000$; zero-sum (with $10^6$ zeros) and series methods agree to $10^{-15}$ relative; the asymptotic constant $\tfrac12(\gamma-1-\log2\pi)$ is recovered to 5 digits and the constant term of the archimedean part is shown to be $3/4$. A *detection-horizon* calculation shows that $\lambda_n$ to $n=10^4$ is blind to any off-line zero above height $\approx30$.
* **Báez-Duarte sequential criterion.** $c_k=\sum_j(-1)^j\binom kj/\zeta(2j+2)$ to $k=10^6$ is reproduced to $10^{-9}$ by an explicit formula over 8 nontrivial and 12 trivial zeros; after removing the (dominant!) trivial-zero term, $k^{3/4}c_k$ is a pure sinusoid in $\log k$ with the frequency $\gamma_1/2$ of the first zero.
* **Arithmetic criteria.** Robin's inequality to $10^7$ (exactly the 27 known exceptions), Lagarias' to $10^7$, Nicolas' for $10^4$ primorials (ratio strictly decreasing, $r_k-1\approx2.1/(\sqrt{p_k}\log p_k)$), 200 colossally abundant numbers (Robin's ratio $<1$ with $1-\text{ratio}\approx0.13/\sqrt{\log n}$).
* **Speiser.** 1298 zeros of $\zeta'$ up to height 2000, each certified by Krawczyk's test to lie right of the critical line (minimum $\Re s=0.5034$).
* **Statistics.** Spacings, pair correlation, Montgomery's form factor and the number variance versus GUE at heights $10^2$–$3\times10^9$: KS distance to the exact GUE law falls monotonically from $0.029$ to $0.006$ over six windows; number variance saturates at Berry's semiclassical value ($0.381$ vs $0.383$ for $10^6$ zeros; $0.441$ vs $0.446$ at height $3.3\times10^9$).
* **de Bruijn–Newman flow.** $H_0=\xi/8$ verified to $10^{-34}$; the repulsive zero dynamics is integrated with an analytic tail field; Lehmer's pair collides at $t^*=-7.11\times10^{-4}$ under the backward flow (ODE and a frozen-background closed form agree to $10^{-11}$; the two-body estimate is within $0.1\%$), the closest pair among $10^5$ zeros at $-1.1\times10^{-4}$; forward flow to $t=0.2$ rigidifies the spacings.
* **Explicit formula and Weil positivity.** Weil's explicit formula is verified to $10^{-15}$ for Gaussian and modulated test functions with $10^5$ zeros and primes to $10^6$; the arithmetic side of Weil's quadratic form is positive on three families, and the two-bump family shows the first zero ($1+\cos2a\gamma_1$) emerging from the primes alone.
* **Integral criteria.** The Balazard–Saias–Yor integral is $-7\times10^{-10}$ at $T=18046$ (RH predicts $0$) and Volchkov's identity holds to $8.5\times10^{-8}$ — and both are shown to be numerically blind to off-line zeros at height.
* **Control.** The Davenport–Heilbronn function (functional equation, no Euler product) has 5 certified zeros off the line below height 300 — the reason every proof must use multiplicativity.

## Reproduce

```bash
pip install -e ".[dev]"            # numpy scipy mpmath sympy python-flint matplotlib pytest
python3 -m pytest                  # ~ a few minutes
python3 scripts/run_all.py --quick # all experiments at reduced size (~10 min)
python3 scripts/run_all.py         # full experiments (hours; zeros to 10^6 already in data/)
```

`python3 scripts/compute_zeros.py --start 1 --count 100000 --out data/zeros/zeros_1_100000` regenerates the zero dataset with Arb; `python3 scripts/verify_vasyunin_formula.py` re-checks the Gram-matrix closed form to 37 digits.

## Honesty statement

All numerical statements come with a method and an error estimate. "Rigorous" is used only for results obtained with ball/interval arithmetic or exact arithmetic (Arb zeros, Arb enclosures of $d_N^2$, high-precision Li coefficients, integer checks of Robin/Lagarias). Floating-point results are labelled as such. Nothing here is evidence *for* RH beyond what was already known; the value is in the structure the computations reveal and in the platform itself.

## License

MIT.
