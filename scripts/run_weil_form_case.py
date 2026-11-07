#!/usr/bin/env python3
"""One case of the restricted-Weil-form study, for batch / cluster use.

For a support half-width a and a Legendre basis of size J, computes in both parity
sectors (even, odd):
  * the two smallest eigenvalues of the arithmetic side of Weil's quadratic form
    (Rayleigh–Ritz upper bounds for the true infimum lambda*(a));
  * the minimising coefficient vector (to 40 digits);
  * the distance from each of the first K zeta zeros gamma_k to the nearest real zero
    of the minimiser's Fourier transform ("zeros from primes").

The working precision is chosen automatically and the run is repeated at 1.5x the
precision whenever the smallest eigenvalue comes within 10^(30-dps) of the precision
floor, so the result is safe to trust without supervision.

Usage:
  python3 scripts/run_weil_form_case.py --a 1.25 --J 192 --out results/json/cluster/weil_a1.25_J192.json
Options: --dps N (default auto), --zeros K (default 60), --no-zero-match, --no-quad-check
"""
import argparse
import json
import math
import os
import platform
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import mpmath as mp  # noqa: E402

from rh import weil_form as wf  # noqa: E402


def auto_dps(a: float, J: int) -> int:
    # empirical: -log10(lambda_min) grows roughly like 0.75*J*min(a,2)/2 + 15 for the
    # bases used here; keep >= 40 digits of headroom
    return int(max(80, 1.4 * (0.8 * J + 30) + 40, 0.45 * J * max(a, 1.0) + 80))


def run(a: float, J: int, dps: int, K: int, zero_match: bool, quad_check: bool = True):
    t0 = time.time()
    W = wf.build_weil_form(a, J=J, dps=dps, estimate_quad_error=quad_check)
    t_build = time.time() - t0
    out = {"a": a, "J": J, "dps": dps, "n_quad": W.n_quad, "n_prime_powers": W.n_prime_powers,
           "quad_error_estimate": W.quad_error_estimate, "build_seconds": t_build, "sectors": {}}
    with mp.workdps(max(60, min(dps, 160))):
        gam = [mp.zetazero(k).imag for k in range(1, K + 1)] if zero_match else []
    floor_hit = False
    for parity in ("even", "odd"):
        t1 = time.time()
        Qs, Gs, idx = W.sector(parity)
        vals, vecs = wf.smallest_eigenpairs(Qs, Gs, dps=dps, k=2)
        bound, positive = wf.weyl_certificate(W, parity, vals[0], dps)
        if vals[0] < mp.mpf(10) ** (30 - dps):
            floor_hit = True
        rec = {"lambda1": mp.nstr(vals[0], 15), "lambda2": mp.nstr(vals[1], 10),
               "weyl_bound": mp.nstr(bound, 4), "numerically_certified_positive": positive,
               "coeffs": [mp.nstr(c, 40) for c in vecs[0]], "idx": idx}
        if zero_match:
            with mp.workdps(min(dps, 160)):
                d = wf.zero_match(vecs[0], idx, a, gam, dps=min(dps, 160), window=0.5)
            rec["abs_diff"] = [mp.nstr(x, 4) if x is not None else None for x in d]
            n_lead = 0
            for x in d:
                if x is not None and x < 1e-3:
                    n_lead += 1
                else:
                    break
            rec["leading_zeros_matched_1e-3"] = n_lead
        rec["seconds"] = time.time() - t1
        out["sectors"][parity] = rec
    out["precision_floor_hit"] = floor_hit
    out["total_seconds"] = time.time() - t0
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", type=float, required=True)
    ap.add_argument("--J", type=int, required=True)
    ap.add_argument("--dps", type=int, default=0)
    ap.add_argument("--zeros", type=int, default=60)
    ap.add_argument("--no-zero-match", action="store_true")
    ap.add_argument("--no-quad-check", action="store_true",
                    help="skip the second archimedean quadrature (halves the cost; the node count is already scaled to the precision)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    dps = args.dps or auto_dps(args.a, args.J)
    for attempt in range(4):
        res = run(args.a, args.J, dps, args.zeros, not args.no_zero_match, not args.no_quad_check)
        if not res["precision_floor_hit"]:
            break
        print(f"precision floor reached at dps={dps}; retrying at {int(dps * 1.5)}", flush=True)
        dps = int(dps * 1.5)
    res["environment"] = {"python": platform.python_version(), "host": platform.node(),
                          "slurm_job": os.environ.get("SLURM_JOB_ID"), "slurm_task": os.environ.get("SLURM_ARRAY_TASK_ID")}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)
    for p in ("even", "odd"):
        s = res["sectors"][p]
        print(f"a={args.a} J={args.J} dps={res['dps']} {p}: lambda1={s['lambda1']} "
              f"matched={s.get('leading_zeros_matched_1e-3')} [{s['seconds']:.0f}s]", flush=True)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
