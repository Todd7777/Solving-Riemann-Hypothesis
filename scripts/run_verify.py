#!/usr/bin/env python3
"""Verify RH up to the height of the 100000-th zero with Turing's method
(independent float64 Riemann–Siegel implementation, rh.zeta + rh.verify),
cross-check against Arb, and write results/json/verify_summary.json.

Usage:
    python3 scripts/run_verify.py [--T 74920.9] [--time-budget 600]
                                  [--refine illinois|brent] [--constants lehman|trudgian]
                                  [--pts 8] [--out results/json/verify_summary.json]

If the projected run time (calibrated on a short run) exceeds the budget, T
is lowered so that the run fits.
"""
import argparse
import json
import math
import os
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

import rh.verify as rv  # noqa: E402
import rh.zeta as zt  # noqa: E402

GAMMA_100000 = 74920.827498994  # Arb dataset: height of the 100000-th zero
DEFAULT_T = 74920.9             # N(T) = 100000


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--T", type=float, default=DEFAULT_T)
    ap.add_argument("--time-budget", type=float, default=600.0, help="seconds (default 600)")
    ap.add_argument("--refine", default="illinois", choices=["illinois", "brent"])
    ap.add_argument("--constants", default="lehman", choices=sorted(rv.TURING_CONSTANTS))
    ap.add_argument("--pts", type=int, default=8, help="grid points per Gram interval")
    ap.add_argument("--no-arb", action="store_true", help="skip the Arb cross-checks")
    ap.add_argument("--out", default=str(ROOT / "results" / "json" / "verify_summary.json"))
    a = ap.parse_args()

    # --- calibrate: cost per zero on [0, T_CAL], then project to T ---------
    T_CAL = 10000.0
    t0 = time.time()
    cal = rv.verify_rh_up_to(T_CAL, refine=a.refine, pts_per_spacing=a.pts, cross_check=False)
    cal_time = time.time() - t0
    per_zero = cal_time / cal.n_zeros_total
    T = a.T


    def projected(T):
        # cost per zero ~ number of Riemann–Siegel terms ~ sqrt(t); 30% headroom
        return 1.3 * per_zero * zt.N_smooth(T) * math.sqrt(T / T_CAL)

    proj = projected(T)
    print(f"calibration: {cal.n_zeros_total} zeros in {cal_time:.1f}s -> projected {proj:.0f}s for T = {T:g}")
    if proj > a.time_budget:
        lo_, hi_ = T_CAL, T                       # bisect for the largest T within budget
        for _ in range(60):
            mid = 0.5 * (lo_ + hi_)
            lo_, hi_ = (mid, hi_) if projected(mid) <= a.time_budget else (lo_, mid)
        T = lo_
        print(f"exceeds budget of {a.time_budget:.0f}s; lowering T to {T:.1f}")

    # --- run -----------------------------------------------------------------
    t0 = time.time()
    res = rv.verify_rh_up_to(T, constants=a.constants, pts_per_spacing=a.pts, refine=a.refine,
                             cross_check=not a.no_arb, verbose=True)
    total = time.time() - t0
    res.timings["total"] = total
    print(res.summary())
    c = res.certificate
    print("Turing certificate:", json.dumps({k: v for k, v in c.to_dict().items() if k != "tried"},
                                            indent=None, default=str))
    if res.max_abs_diff_vs_arb is not None:
        print(f"max |gamma_found - gamma_arb| = {res.max_abs_diff_vs_arb:.3e} over {res.n_compared_vs_arb} zeros")

    # --- summary file ----------------------------------------------------
    d = res.to_dict()
    d["script"] = "scripts/run_verify.py"
    d["arguments"] = vars(a)
    d["calibration"] = {"T": T_CAL, "seconds": cal_time, "zeros": cal.n_zeros_total,
                        "projected_seconds": proj}
    d["environment"] = {"python": platform.python_version(), "numpy": np.__version__,
                        "platform": platform.platform()}
    try:
        import flint, mpmath, scipy
        d["environment"].update(flint=flint.__version__, mpmath=mpmath.__version__, scipy=scipy.__version__)
    except Exception:  # pragma: no cover
        pass
    d["method"] = ("Riemann–Siegel formula (C_0..C_4, Gabcke remainder bound) / Euler–Maclaurin below t=500, "
                   "float64; zeros by sign changes on a Gram-coordinate grid with Gram-block refinement; "
                   f"Turing's method with {a.constants} constants; cross-checked against Arb")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(d, f, indent=2, default=float)
    print("wrote", a.out)
    return 0 if res.certified else 1


if __name__ == "__main__":
    sys.exit(main())
