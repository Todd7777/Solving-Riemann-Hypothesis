#!/usr/bin/env python3
"""Weil explicit formula and Weil positivity numerics.  Writes
results/json/explicit_formula_summary.json, results/figures/explicit_formula_residual.png
and results/figures/explicit_formula_weil_positivity.png.

    python3 scripts/run_explicit_formula.py [--quick]
"""
import argparse, json, math, sys, time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rh.data import load_zeros                 # noqa: E402
import rh.explicit_formula as ef               # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    T0 = time.time()
    z = load_zeros(100000)
    out = {"n_zeros": int(z.size), "zero_accuracy": "rigorous Arb isolation, |error| < 1.5e-11",
           "formula": "sum_rho h(r_rho) = h(i/2)+h(-i/2) - 2 sum Lambda(n) n^-1/2 g(log n) + (1/2pi) int h(r)[Re psi(1/4+ir/2) - log pi] dr"}

    # 1. explicit formula for Gaussian and modulated Gaussian test functions
    t0 = time.time()
    checks = []
    for s in (2.0, 1.5, 1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.003, 0.001, 0.0005):
        nmax = 10 ** 7 if s >= 1.5 else 10 ** 6
        checks.append(ef.explicit_formula_check(ef.gaussian(s), zeros=z, nmax=nmax))
    for s, w in ((1.0, 50.0), (1.0, 100.0), (1.0, 500.0), (1.0, 1000.0), (1.0, 5000.0), (1.0, 20000.0),
                 (1.0, 70000.0), (0.5, 5000.0), (0.25, 10000.0), (2.0, 1000.0)):
        checks.append(ef.explicit_formula_check(ef.modulated_gaussian(s, w), zeros=z, nmax=10 ** 7 if s >= 1.5 else 10 ** 6))
    out["explicit_formula_checks"] = checks
    out["max_abs_residual_gaussian_sigma_le_1"] = max(abs(c["residual"]) for c in checks if c["test_function"].startswith("gaussian") and c["params"]["sigma"] <= 1.0)
    out["max_abs_residual_modulated"] = max(abs(c["residual"]) for c in checks if c["test_function"].startswith("modulated"))
    for c in checks:
        print(f"{c['test_function']:45s} zero side {c['zero_side']:+.12f}  arithmetic {c['arithmetic_side']:+.12f}  "
              f"primes {c['prime_sum']:+.6f}  residual {c['residual']:+.3e}")
    out["checks_seconds"] = time.time() - t0

    # 2. convergence in zeros and in primes
    t0 = time.time()
    n_list = [10, 30, 100, 300, 1000, 3000, 10000, 30000, 100000]
    conv_z = [ef.convergence_in_zeros(ef.gaussian(s), z, n_list) for s in (0.01, 0.003, 0.001)]
    nmax_list = [10, 100, 1000, 10 ** 4, 10 ** 5, 10 ** 6] + ([10 ** 7] if not a.quick else [])
    conv_p = [ef.convergence_in_primes(tf, z, nmax_list) for tf in
              (ef.modulated_gaussian(1.0, 100.0), ef.modulated_gaussian(1.0, 1000.0), ef.gaussian(2.0))]
    out["convergence_in_zeros"] = conv_z
    out["convergence_in_primes"] = conv_p
    for c in conv_z:
        print(c["test_function"], [(r["n_zeros"], f"{r['residual']:.1e}", f"{r['residual_no_tail']:.1e}") for r in c["rows"]])
    for c in conv_p:
        print(c["test_function"], [(r["nmax"], f"{r['residual']:.1e}") for r in c["rows"]])
    fig1 = ROOT / "results" / "figures" / "explicit_formula_residual.png"
    ef.plot_residual_convergence(conv_z, conv_p, fig1)
    print("wrote", fig1)
    out["convergence_seconds"] = time.time() - t0

    # 3. Weil positivity scans (arithmetic side only; zero side for comparison)
    t0 = time.time()
    scans = [
        ef.weil_positivity_scan("gaussian_bump", np.geomspace(0.02, 0.4, 40 if not a.quick else 10), nmax=10 ** 6, zeros=z),
        ef.weil_positivity_scan("two_bumps", np.arange(0.05, 5.0001, 0.005 if not a.quick else 0.05), nmax=10 ** 6, zeros=z, sigma=0.2),
        ef.weil_positivity_scan("two_bumps", np.arange(0.05, 5.0001, 0.005 if not a.quick else 0.05), nmax=10 ** 6, zeros=z, sigma=0.1),
        ef.weil_positivity_scan("box", np.arange(0.1, 7.0001, 0.05 if not a.quick else 0.5), nmax=1_300_000, zeros=z),
    ]
    out["weil_positivity"] = []
    for sc in scans:
        res = max(abs(r["residual"]) for r in sc["rows"])
        summary = {"family": sc["family"], "kw": sc["kw"], "n_points": len(sc["rows"]),
                   "min_arithmetic_side": sc["min_arithmetic_side"], "argmin_parameter": sc["argmin_parameter"],
                   "all_nonnegative": sc["all_nonnegative"], "max_abs_residual_vs_zero_side": res,
                   "n_negative_beyond_rounding": sc["n_negative_beyond_rounding"],
                   "min_arithmetic_side_in_units_of_rounding_tolerance": sc["min_arithmetic_side_in_units_of_tolerance"],
                   "rows": [{k: r[k] for k in ("parameter", "h_poles", "prime_sum", "archimedean", "arithmetic_side", "zero_side")}
                            for r in sc["rows"]]}
        out["weil_positivity"].append(summary)
        print(f"Weil scan {sc['family']} {sc['kw']}: n={len(sc['rows'])} min arithmetic side {sc['min_arithmetic_side']:.6e} "
              f"at {sc['argmin_parameter']:.4g}; all >= 0: {sc['all_nonnegative']}; max |resid| vs zeros {res:.2e}")
    fig2 = ROOT / "results" / "figures" / "explicit_formula_weil_positivity.png"
    ef.plot_weil_positivity([scans[0], scans[1], scans[3]], fig2)
    print("wrote", fig2)
    out["weil_seconds"] = time.time() - t0

    # 4. a sampled test function (no closed form needed): f = bump * cos, compared with the zeros
    u = np.linspace(-8, 8, 16001)
    f = np.exp(-u ** 2 / (2 * 0.4 ** 2)) * (1 + 0.5 * np.cos(3 * u))
    rs = ef.weil_functional_from_samples(u, f, 10 ** 6, z)
    out["weil_sampled_example"] = rs
    print("sampled f = e^{-u^2/0.32}(1+cos(3u)/2):", rs)

    out["total_seconds"] = time.time() - T0
    jpath = ROOT / "results" / "json" / "explicit_formula_summary.json"
    jpath.write_text(json.dumps(out, indent=2, default=float))
    print("wrote", jpath, f"({out['total_seconds']:.1f}s)")


if __name__ == "__main__":
    main()
