# CLAUDE.md

Guidance for Claude Code sessions in this repository, local or cloud.

## What this is

A research platform for the Riemann Hypothesis: rigorous computation of zeros (Arb), the classical
equivalent criteria implemented and tested, experiments on their second-order behaviour, and a survey of what
is known and why the standard approaches stall. RH is open, and this repository does not prove it.

## Ground rules

- Do not claim or imply that RH is proved. Label every new claim as **Theorem** (complete proof here),
  **Machine-checked**, **Cited** (only after reading the source), **Numerical** (with method and error
  estimate), or **Conjecture/Heuristic**.
- Numerics see only a short way: `docs/06_experiment_log.md` shows how blind several criteria are to off-line
  zeros at height. Never read agreement at small N as support for RH.
- Run the checks before and after every change. A new failure is a regression until shown otherwise.
- Commit verified work in small steps and push before the session ends. Work left only inside a cloud
  session can be lost with it.

## Setup and checks

```bash
pip install -e ".[dev]"
python3 -m pytest                   # 2026-11-11: 302 pass, about 6 minutes
python3 scripts/run_all.py --quick  # every experiment at reduced size, about 10 minutes
```

`data/zeros/` already holds 10^6 rigorous zero ordinates. Full experiment runs take hours.

## State on 2026-11-11

- E12 is complete (`docs/06_experiment_log.md`, E12; synthesis in `docs/08_findings_and_outlook.md`, item 1b).
  `src/rh/weil_form_general.py` was reviewed and is covered by `tests/test_weil_form_general.py`; a float64
  overflow in its far-field tables was fixed. `src/rh/weil_certificate.py` gives rigorous enclosures of the
  Weil form at one explicit test function. `scripts/run_weil_detection.py` (in `run_all.py`) reproduces the
  E12 numbers; the full run takes about 30 minutes.
- Headline: the Davenport–Heilbronn Weil form is negative at support a = 1.88, where only the coefficients
  n <= 42 enter (Machine-checked: rigorous enclosures at a = 1.88, 1.92, 2). For zeta with an off-line pair moved off the line at height gamma, the detection threshold
  is about (1/2) log(gamma / 2 pi) + c(delta) (Numerical, wave-packet bases, upper bounds).

- E13 is complete: `src/rh/weil_envelope.py` certifies Weil positivity on whole windows [-L, L] (envelope
  reduction in ball arithmetic). Q(f) >= 5.137e-30 ||f||^2 for every real f supported in [-1, 1]
  (Machine-checked), past the previous certified window L = 0.8 (Zhu 2026). `scripts/run_weil_positivity.py`
  reproduces L <= 0.9729 in about 30 minutes; L = 1 (`--only 1.0 --merge`) takes about an hour per sector.
  The method's cost grows doubly exponentially in L, so it ends near L = 1.1 (docs/06, E13).

- E14 is complete: `src/rh/weil_comb.py` gives non-pointwise certificates A' - P_L + H >= 0 for the prime comb
  (H has all frequencies >= 2L, so it is invisible to the window). They come in two kinds: separable (per prime)
  and joint (torus sums of squares over the integers n <= e^Lambda). `weil_envelope` accepts them (`comb_cert=`).
  Certified (Machine-checked): Q >= 1.12e-38 ||f||^2 on [-1.1, 1.1], 4.57e-49 on [-1.2, 1.2], 6.54e-78 on
  [-1.4, 1.4], 2.67e-97 on [-1.5, 1.5] and 3.54e-121 on [-1.6, 1.6] (joint certificates by ADMM, 619 and 1239 monomials). `scripts/run_weil_comb.py` reproduces them. Run numpy-heavy jobs with OPENBLAS_NUM_THREADS=1;
  multithreaded BLAS made the joint solver 10x slower here. The next barrier is the operator norm of the comb
  (A* >= sum c_n (1 - log n / 2L), about 0.3 A_L), so the envelope approach ends near L = 1.7-2 (docs/06, E14).

- E15 is complete: `trajectory_check` in `src/rh/weil_comb.py` certifies Omega - P_L + H >= beta along t in
  [T, T2] (IEEE double with a-priori error bounds, phases from Arb), so the comb no longer needs the torus cutoff
  2 pi e^{A'}: T = 853 / 1180 / 2460 at L = 1.5 / 1.6 / 1.7 instead of 1295 / 2370 / 10853, and T = 1000 at L = 1.7
  with the wave packets of `src/rh/weil_local.py` (Machine-checked). But [-1.7, 1.7] is NOT certified: the
  envelope form is indefinite at T = 2460 (even pivot -0.14; odd lambda_min ~ -2.8e-146). The limit is now the
  envelope floor (near-minimisers keep spectral mass above T): ~2 pi e^{1+2L} for L <= 1.2, higher for larger L
  and large H (docs/06, E15). `scripts/run_weil_trajectory.py` reproduces the analysis (~40 min); `--certify` the
  failed attempt (~5.8 h, ~9.7 GB; the effective memory limit here is ~13 GB, run nothing else).

- E16 (analysis complete): the part of the envelope floor that E15 attributed to a large H is a resonance of a few
  terms of H with frequency just above 2L. For even f, |F|^2 has a 2L component as large as its mean, and such a
  term can take up to |h|/2 per unit of mass above T, against the credit beta. At L = 1.4 one term (33/2) of 557
  makes the form indefinite at beta = 0.05; the thresholds beta*(T) are in docs/06, E16 (Numerical). E14 had beta
  0.18-0.30; E15 had 0.05, below the |h|/2 = 0.19 of the term 30/1 at L = 1.7. `rh.weil_comb.gap_certificate`
  moves such terms into A'; `rh.weil_local.fix_dips` lifts the remaining dips with wave packets. Gapped L = 1.7
  certificate: beta = 0.2 from T = 2460 with 14 packets (Machine-checked). `scripts/run_weil_gap.py` reproduces
  the analysis (`--quick` in run_all; `--certify` runs [-1.7, 1.7], ~6 h, ~10 GB).

## Where the next steps are

Start with `docs/08_findings_and_outlook.md`, then `docs/05_research_program.md` and `docs/07_how_to_extend.md`.
The open threads are listed in §8.3:
- the certification of [-1.7, 1.7] with the gapped certificate (`scripts/run_weil_gap.py --certify`, ~6 h,
  ~10 GB; run nothing else alongside);
- the envelope floor without H (E15, E16): how it grows with L, and a reduction that keeps part of the spectral
  side above T;
- sum-of-squares certificates with a spectral gap above 2L built in (E16), and a prolate basis to make larger T
  affordable (E15);
- the same certificates for L(s,(n/5));
- whether bases other than wave packets detect off-line zeros earlier;
- a theory for c(delta).
