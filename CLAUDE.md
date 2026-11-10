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
python3 -m pytest                   # 2026-11-10: 273 pass, about 2 minutes
python3 scripts/run_all.py --quick  # every experiment at reduced size, about 10 minutes
```

`data/zeros/` already holds 10^6 rigorous zero ordinates. Full experiment runs take hours.

## State on 2026-11-09

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

## Where the next steps are

Start with `docs/08_findings_and_outlook.md`, then `docs/05_research_program.md` and `docs/07_how_to_extend.md`.
The open threads are listed in §8.3: a bound for the prime comb that is not pointwise in t (needed for windows
beyond L ~ 1.1), the same certificates for L(s,(n/5)), whether bases other than wave packets detect off-line
zeros earlier, and a theory for c(delta).
