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
python3 -m pytest                   # 2026-11-14: 315 pass, about 7 minutes
python3 scripts/run_all.py --quick  # every experiment at reduced size, about 16 minutes
```

`data/zeros/` already holds 10^6 rigorous zero ordinates. Full experiment runs take hours.

## State on 2026-11-14

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
  with the wave packets of `src/rh/weil_local.py` (Machine-checked). But with this certificate [-1.7, 1.7] is NOT
  certified (E16 certifies it): the envelope form is indefinite at T = 2460 (even pivot -0.14; odd lambda_min
  ~ -2.8e-146). The limit is now the envelope floor (near-minimisers keep spectral mass above T): ~2 pi e^{1+2L}
  for L <= 1.2, higher for larger L and large H (docs/06, E15). `scripts/run_weil_trajectory.py` reproduces the
  analysis (~40 min); `--certify` the failed attempt (~5.8 h, ~9.7 GB; the effective memory limit here is
  ~13 GB, run nothing else).

- E16 is complete: the part of the envelope floor that E15 attributed to a large H is a resonance of a few
  terms of H with frequency just above 2L. For even f, |F|^2 has a 2L component as large as its mean, and such a
  term can take up to |h|/2 per unit of mass above T, against the credit beta. At L = 1.4 one term (33/2) of 557
  makes the form indefinite at beta = 0.05; the thresholds beta*(T) are in docs/06, E16 (Numerical). E14 had beta
  0.18-0.30; E15 had 0.05, below the |h|/2 = 0.19 of the term 30/1 at L = 1.7. `rh.weil_comb.gap_certificate`
  moves such terms into A'; `rh.weil_local.fix_dips` lifts the remaining dips with wave packets. Gapped L = 1.7
  certificate: beta = 0.2 from T = 2460 with 14 packets. Certified (Machine-checked): Q >= 1.79e-150 ||f||^2 on
  [-1.7, 1.7] (3147 modes per sector, 195 digits). Without the near-2L terms the L = 1.7 floor is below
  T = 1000 (Numerical); a certificate from there needs 47 dips of the comb bound lifted on [1000, 2460].
  `scripts/run_weil_gap.py` reproduces the analysis (`--quick` in run_all; `--certify` repeats the
  certification, ~6.2 h, ~10 GB; run nothing else alongside).

- E17 is complete: every dip of the L = 1.7 comb bound above T = 1000 passes the local condition for lifting
  (E15's local operator norm at bandwidth L + gap/2: >= 1.10); E16's packets failed only for lack of high carriers
  (W = 8, carriers to 12.45: sum|a| 19 instead of 520 on the dense cluster). `fix_dips` has `core_pad`, `far_step`,
  `buffer` for many dips. 245 packets give beta = 0.2 from T = 1000, and [-1.7, 1.7] is certified there:
  Q >= 1.05e-150 ||f||^2 with 1361 modes per sector (Machine-checked, 1.0 h). 1291 modes were not enough:
  the tail bound eps_B ~ (eLT/4N)^{2N} must stay below the margin. `scripts/run_weil_lift.py` reproduces it
  (~20 min; `--quick` in run_all; `--certify` ~1 h).

- E18 is complete: [-1.8, 1.8] certified, Q >= 2.53e-186 ||f||^2 (Machine-checked). Joint certificate
  Lambda = 9 by ADMM (results/json/weil_gram_L1.8.npz, 20000 iterations, ~4 h): A' = 8.169 (gapped 8.490);
  230 packets (fix_dips, W = 8, core_per_dip) give beta = 0.2 from T = 2000; 2712 modes per sector, 235
  digits, one sector at a time (~3.6 h and ~7 GB each). The floor (beta = 0.2) is between 2.4 and 3.2 t0.
  `scripts/run_weil_l18.py` reproduces it (`--quick` in run_all; `--parts floor` the probes; `--certify`).

- E19 is complete: the certification keeps R_A as its lower block triangle (`SymBlocks`, blocks of 512), forms it
  in place and factorises it by blocks on one copy; `certify_window(s)` use it by default (`storage="dense"` is the
  old path). Same certificates (E17 re-certified: identical bounds). Peak at N = 962, 235 digits: 0.76 GB
  (chunk 512) vs 2.55 GB dense; a synthetic L = 1.9 sector (N = 4300, 280 digits) needs 4.5 GB (Numerical).
  Pass threads=3. chunk=512 cuts the peak at N ~ 1000; for N in the thousands the factorisation sets the peak
  and chunk=1024 flushes ~10% faster. `scripts/run_cert_memory.py` reproduces it (~1 h; `--quick` in
  run_all; `--parts recert17` adds E17 again, ~55 min).

- E20 is complete: [-1.9, 1.9] certified, Q >= 3.73e-230 ||f||^2 (Machine-checked). Joint certificate Lambda = 8.5
  by ADMM (results/json/weil_gram_L1.9.npz, best iterate after 9000 iterations): A' = 10.018 (gapped 10.132),
  T2 = 192942. The comb bound has 295 dips below 0 on [2300, 192942] (deepest -2.63; 45 and -0.85 at L = 1.8):
  1601 packets from fix_dips with the new options (W = 12, amax = 32, dtau = 2, dw = 1, max_chain = 3,
  merge_on_fail = 6, cut = 50, refine = 0.005; results/json/weil_l19_packets_T2300.json, ~1 h); trajectory
  check beta = 0.2 from T = 2300 (11 min). Certification: 3287 modes per sector, 280 digits, both sectors from
  one block assembly of 68808 nodes, ~9 h, checkpointed (RH_CHECKPOINT_DIR). The H = 0 probes are
  positive down to T = 1221 (1.60 t0) and indefinite at T = 916 (1.20 t0) (Numerical): the floor lies between
  1.20 and 1.60 t0, lower in units of t0 than at L = 1.8 (above 2.4 t0).
  `scripts/run_weil_l19.py` reproduces it (`--quick` in run_all; `--parts floor --floor-T ...`; `--certify`).

## Where the next steps are

Start with `docs/08_findings_and_outlook.md`, then `docs/05_research_program.md` and `docs/07_how_to_extend.md`.
The open threads are listed in §8.3:
- L = 2.0: the same steps (run_weil_l19.py as the template; RH_CHECKPOINT_DIR for every long run; state= for
  the ADMM). The hard part is now the comb bound: its dips grow in number and depth with L (295 and -2.63 at
  L = 1.9) and fix_dips with them. A certificate H shaped for the trajectory (not only for the torus maximum)
  would leave fewer dips. The H = 0 floor need not grow faster than t0 (E20); probe it, then size N with the
  tail bound (eLT/4N)^{2N}. The panel setup at 280 digits costs ~25 min per run (ellipse bounds at full
  precision); 128-bit bounds would remove it;
- the envelope floor without H (E15, E16): how it grows with L, and a reduction that keeps part of the spectral
  side above T;
- sum-of-squares certificates with a spectral gap above 2L built in (E16), and a prolate basis to make larger T
  affordable (E15);
- the same certificates for L(s,(n/5));
- whether bases other than wave packets detect off-line zeros earlier;
- a theory for c(delta).
