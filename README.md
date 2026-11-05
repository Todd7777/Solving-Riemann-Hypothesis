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
| `data/zeros/` | Precomputed rigorous zero ordinates ($10^6$ zeros, radius $\le 3\cdot10^{-11}$) and high-height samples |
| `results/` | JSON summaries and figures produced by the scripts |
| `tests/` | pytest suite (every module is tested against independent references) |

Start with [`docs/00_problem_statement.md`](docs/00_problem_statement.md) and [`docs/06_experiment_log.md`](docs/06_experiment_log.md).

## Highlights of the experiments

See `docs/06_experiment_log.md` for all numbers with error estimates. In brief:

* **Zeros.** One million zeros recomputed with ball arithmetic; an independent Riemann–Siegel + Turing implementation certifies $N(T)$ and the location of all zeros on the line up to the height of the dataset; agreement with Arb to $\sim10^{-9}$.
* **Nyman–Beurling.** Using the Vasyunin closed form (re-verified to 37 digits) for the Gram matrix of $\{1/(kx)\}$, the Báez-Duarte distances are computed to $N=10^4$ (rigorous enclosures to $N=800$). $d_N^2\log N$ oscillates around Burnol's constant $C=2+\gamma-\log4\pi=0.04619\ldots$ (the conjectured limit), within a few percent, dipping below it; the optimal coefficients have the sign of $-\mu(k)$ for every squarefree $k$.
* **Li coefficients.** $\lambda_n$ to $n=10^4$ from the zeros, cross-checked against an independent high-precision power-series computation; all positive; the oscillatory part is far below the scale that any off-line zero would produce.
* **Báez-Duarte sequential criterion.** $c_k=\sum_j(-1)^j\binom kj/\zeta(2j+2)$ is computed to $k=10^6$ and reproduced to $10^{-16}$ by an explicit formula over the first few zeros, exhibiting the $k^{-3/4}$ decay (and a $k^{-2}$ trivial-zero term that dominates below $k\approx10^5$).
* **Arithmetic criteria.** Robin, Lagarias, Nicolas verified to large bounds; colossally abundant numbers' Robin ratios approach $1$ from below.
* **Statistics.** Spacings and pair correlation versus GUE at four heights, with the $1/\log T$ approach to GUE.
* **Control.** The Davenport–Heilbronn function (functional equation, no Euler product) is shown to have zeros off the line — the reason every proof must use multiplicativity.

## Reproduce

```bash
pip install -e ".[dev]"            # numpy scipy mpmath sympy python-flint matplotlib pytest
python3 -m pytest                  # ~ a few minutes
python3 scripts/run_all.py --quick # all experiments at reduced size (~10 min)
python3 scripts/run_all.py         # full experiments (hours; zeros to 10^6 already in data/)
```

`python3 scripts/compute_zeros.py --start 1 --count 100000 --out data/zeros/zeros_1_100000` regenerates the zero dataset with Arb.

## Honesty statement

All numerical statements come with a method and an error estimate. "Rigorous" is used only for results obtained with ball/interval arithmetic or exact arithmetic (Arb zeros, Arb enclosures of $d_N^2$, high-precision Li coefficients, integer checks of Robin/Lagarias). Floating-point results are labelled as such. Nothing here is evidence *for* RH beyond what was already known; the value is in the structure the computations reveal and in the platform itself.

## License

MIT.
