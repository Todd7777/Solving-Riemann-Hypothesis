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
python3 -m pytest                   # 2026-11-08: 228 pass
python3 scripts/run_all.py --quick  # every experiment at reduced size, about 10 minutes
```

`data/zeros/` already holds 10^6 rigorous zero ordinates. Full experiment runs take hours.

## State on 2026-11-08

- The last commit is a work-in-progress snapshot ("E12 stage 2, agent still working") that adds
  `src/rh/weil_form_general.py` (1,166 lines). No test covers it yet, so review it and add tests before
  relying on it.

## Where the next steps are

Start with `docs/08_findings_and_outlook.md`, then `docs/05_research_program.md` and `docs/07_how_to_extend.md`.
