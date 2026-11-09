#!/usr/bin/env python3
"""E12 stage 2, side checks for docs/06_experiment_log.md (E12):

1. Basis robustness of the Davenport–Heilbronn threshold: for wave packets near its first off-line
   zero (0.8085 + 85.699i), the smallest support a in [1.50, 2.20] (step 0.02) with a negative
   eigenvalue of the zero-side form, over J in {12, 20, 28, 36}, m in {0, 1, 2, 3, 4, 6},
   r0 in {82, 84, 85.7, 87.5, 90}; and the most negative eigenvalue over J in {20, 28, 36},
   m in {0, ..., 4}, r0 in {84, 85.7, 87.5} at a = 1.7, 1.8, 1.9, 2.0, with L(s,(n/5)) on the minimising basis.
2. Precision of the float64 thresholds of the zeta thought experiment: at the largest height
   (zeros 800000/800001 replaced, delta = 0.01 and 0.1), the smallest eigenvalue just below and at
   the threshold from the float64 solver, from a 50-digit solve of the same matrices, and with every
   zero perturbed by 1e-10 (the accuracy of the zero data).

Usage: python3 scripts/run_weil_detection_checks.py [--quick]   (full: about 15 min)
Writes results/json/weil_detection_checks.json.
"""
import argparse
import itertools
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mpmath as mp  # noqa: E402
import numpy as np  # noqa: E402

from rh import weil_form_general as wg  # noqa: E402
from rh.data import load_zeros  # noqa: E402
from run_weil_detection import ZetaThought, dh_zero_side, lam_min_float  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "json", "weil_detection_checks.json")


def dh_basis_scan(zd, Js, ms, r0s, a_grid):
    rows = []
    for J, m, r0 in itertools.product(Js, ms, r0s):
        first = None
        for a in a_grid:
            B = wg.Basis.localized(float(a), J, r0, m)
            lam = lam_min_float(B, "even", dh_zero_side(B, "even", zd, "dh"))[0]
            if lam < 0:
                first = {"a": float(a), "lambda": lam}
                break
        rows.append({"J": J, "m": m, "r0": r0, "first_negative": first})
    found = [r["first_negative"]["a"] for r in rows if r["first_negative"]]
    return {"rows": rows, "min_a": min(found) if found else None, "max_a": max(found) if found else None,
            "n_configs": len(rows), "n_detected": len(found)}


def dh_most_negative(zd, a_values, Js, ms, r0s):
    out = []
    for a in a_values:
        best = None
        for J, m, r0 in itertools.product(Js, ms, r0s):
            B = wg.Basis.localized(a, J, r0, m)
            lam = lam_min_float(B, "even", dh_zero_side(B, "even", zd, "dh"))[0]
            if best is None or lam < best[0]:
                best = (lam, J, m, r0)
        B = wg.Basis.localized(a, best[1], best[3], best[2])
        lam5 = lam_min_float(B, "even", dh_zero_side(B, "even", zd, "chi5"))[0]
        out.append({"a": a, "lambda_dh_min": best[0], "J": best[1], "m": best[2], "r0": best[3], "lambda_chi5": lam5})
        print(f"  a={a}: most negative DH eigenvalue {best[0]:.4f} at (J,m,r0)=({best[1]},{best[2]},{best[3]}); "
              f"L(s,(n/5)) on that space {lam5:.4f}", flush=True)
    return out


def zeta_precision(zt, k, deltas, J=16, m=4):
    out = []
    Z = zt.Z
    for d in deltas:
        r = zt.thresholds(k, [d], J, m, np.round(np.arange(4.0, 7.0, 0.05), 3))
        a_star = r["a_star"][str(d)]
        g0 = r["gamma0"]
        lo, hi = int(np.searchsorted(Z, g0 - zt.W)), int(np.searchsorted(Z, g0 + zt.W))
        idx = np.arange(lo, hi)
        idx = idx[(idx != k - 1) & (idx != k)]
        for a in (round(a_star - 0.05, 3), a_star):
            B = wg.Basis.localized(a, J, g0, m)
            v = wg.fourier_matrix(B, "even", np.array([g0 - 1j * d]))[0]
            M4 = 4 * np.real(np.outer(v, v))
            Zon = wg.zero_side_matrix(B, "even", on_line=Z[idx], series="zeta", tail=True, T=float(Z[hi - 1]))["Z"]
            G = wg.gram_matrix_float(B, "even")
            lf = float(wg.smallest_eigenpairs_float(Zon + M4, G)[0][0])
            with mp.workdps(50):
                Mm, Gm = mp.matrix((Zon + M4).tolist()), mp.matrix(G.tolist())
                Li = mp.inverse(mp.cholesky(Gm))
                lm = float(min(mp.eigsy(Li * Mm * Li.T)[0]))
            pert = Z[idx] + 1e-10 * np.random.default_rng(0).standard_normal(idx.size)
            Zp = wg.zero_side_matrix(B, "even", on_line=pert, series="zeta", tail=True, T=float(Z[hi - 1]))["Z"]
            lp = float(wg.smallest_eigenpairs_float(Zp + M4, G)[0][0])
            row = {"k": k, "gamma0": g0, "delta": d, "a": a, "a_star": a_star, "lambda_float": lf, "lambda_mp50": lm,
                   "lambda_zeros_perturbed_1e-10": lp, "cond_G": float(np.linalg.cond(G))}
            out.append(row)
            print(f"  k={k} delta={d} a={a}: float {lf:.6e}, 50 digits {lm:.6e}, perturbed zeros {lp:.6e}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    zd = {s: wg.load_zero_data(s) for s in ("chi5", "dh")}
    res = {"quick": args.quick}
    print("1. DH threshold over bases", flush=True)
    if args.quick:
        Js, ms, r0s = [12, 28], [0, 2], [85.7]
    else:
        Js, ms, r0s = [12, 20, 28, 36], [0, 1, 2, 3, 4, 6], [82.0, 84.0, 85.7, 87.5, 90.0]
    res["dh_basis_scan"] = dh_basis_scan(zd, Js, ms, r0s, np.round(np.arange(1.50, 2.21, 0.02), 3))
    s = res["dh_basis_scan"]
    print(f"  first negative a in [{s['min_a']}, {s['max_a']}] over {s['n_detected']}/{s['n_configs']} bases", flush=True)
    res["dh_most_negative"] = dh_most_negative(
        zd, [1.7, 1.8, 1.9, 2.0], [20, 28, 36] if not args.quick else [20], [0, 1, 2, 3, 4] if not args.quick else [0, 2],
        [84.0, 85.7, 87.5] if not args.quick else [85.7])
    print("2. precision of the zeta thresholds", flush=True)
    res["zeta_precision"] = zeta_precision(ZetaThought(load_zeros(10 ** 6)), 800000, [0.01, 0.1])
    res["seconds"] = time.time() - t0
    with open(OUT, "w") as f:
        json.dump(res, f, indent=1)
    print(f"wrote {OUT} [{res['seconds']:.0f}s]")


if __name__ == "__main__":
    main()
