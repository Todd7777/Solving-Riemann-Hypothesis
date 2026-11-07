#!/usr/bin/env python3
"""Experiment E10: spectral analysis of Weil's quadratic form restricted to test
functions supported in [-a, a] (Legendre basis of dimension J), arithmetic side only.

  1. validation: arithmetic side vs zero side (10^6 zeros) for random test functions
  2. lambda_min(a) for both parity sectors over a grid of a (two basis sizes),
     with a numerical Weyl-bound positivity certificate
  3. basis-size convergence of lambda_min at fixed a
  4. "zeros from primes": distance between the real zeros of the minimiser's
     Fourier transform and the zeros of zeta (computed independently by mpmath)

Usage: python3 scripts/run_weil_form.py [--quick]
Writes results/json/weil_form_summary.json and results/figures/weil_form_*.png
"""
import argparse, json, math, os, sys, time
import numpy as np
import mpmath as mp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from rh import weil_form as wf
from rh.data import load_zeros

ap = argparse.ArgumentParser()
ap.add_argument("--quick", action="store_true")
ap.add_argument("--figures-only", action="store_true", help="redraw figures from results/json/weil_form_summary.json")
args = ap.parse_args()
os.makedirs("results/json", exist_ok=True)
os.makedirs("results/figures", exist_ok=True)
T0 = time.time()
summary = {"description": "Weil quadratic form on polynomials of degree < J on [-a,a]; arithmetic side only (poles, primes <= e^{2a}, Gamma factor).",
           "basis": "Legendre P_j(u/a), u = log x"}


def dps_for(expected_log10):
    return int(max(60, 1.4 * expected_log10 + 40))


if not args.figures_only:
    # --------------------------------------------------------------------------- 1
    print("[1] validation against the zero side", flush=True)
    zeros = load_zeros(None if not args.quick else 100000)
    rng = np.random.default_rng(12345)
    val = []
    for a in ([0.3, 0.8, 1.5, 2.5] if not args.quick else [0.3, 1.5]):
        W = wf.build_weil_form(a, J=16, dps=40, estimate_quad_error=False)
        for parity in ("even", "odd"):
            Qs, Gs, idx = W.sector(parity)
            for trial in range(3):
                c = rng.standard_normal(len(idx))
                qa = sum(c[i] * c[j] * float(Qs[i, j].mid()) for i in range(len(idx)) for j in range(len(idx)))
                qz = wf.zero_side_value(c, idx, a, zeros)
                val.append({"a": a, "parity": parity, "arithmetic_side": qa, "zero_side": qz, "rel_diff": abs(qa - qz) / abs(qz)})
    summary["validation"] = {"n_zeros": int(zeros.size), "max_rel_diff": max(v["rel_diff"] for v in val), "rows": val}
    print("   max rel diff arithmetic vs zero side:", summary["validation"]["max_rel_diff"], flush=True)

    # --------------------------------------------------------------------------- 2
    print("[2] lambda_min(a) scan", flush=True)
    a_grid = [round(x, 2) for x in np.arange(0.1, 2.51, 0.1)] if not args.quick else [0.25, 0.5, 1.0]
    scan = []
    for J in ([32, 48] if not args.quick else [24]):
        expected = 0.0
        for a in a_grid:
            dps = dps_for(max(expected, 10))
            t0 = time.time()
            W = wf.build_weil_form(a, J=J, dps=dps, estimate_quad_error=True)
            row = {"a": a, "J": J, "dps": dps, "n_quad": W.n_quad, "n_prime_powers": W.n_prime_powers,
                   "quad_error_estimate": W.quad_error_estimate}
            for parity in ("even", "odd"):
                Qs, Gs, idx = W.sector(parity)
                vals, vecs = wf.smallest_eigenpairs(Qs, Gs, dps=dps, k=2)
                bound, positive = wf.weyl_certificate(W, parity, vals[0], dps)
                row[parity] = {"lambda1": mp.nstr(vals[0], 12), "lambda2": mp.nstr(vals[1], 12),
                               "log10_lambda1": float(mp.log10(abs(vals[0]))) if vals[0] != 0 else None,
                               "weyl_bound": mp.nstr(bound, 4), "numerically_certified_positive": positive}
            row["seconds"] = time.time() - t0
            expected = max(-(row["even"]["log10_lambda1"] or 0), -(row["odd"]["log10_lambda1"] or 0)) + 10
            scan.append(row)
            print(f"   J={J} a={a:4.2f} dps={dps} even {mp.nstr(mp.mpf(row['even']['lambda1']), 4):>11} odd {mp.nstr(mp.mpf(row['odd']['lambda1']), 4):>11}"
                  f"  certified={row['even']['numerically_certified_positive'] and row['odd']['numerically_certified_positive']}"
                  f"  [{row['seconds']:.0f}s]", flush=True)
    summary["lambda_min_scan"] = scan
    summary["all_eigenvalues_positive_in_scan"] = all(float(mp.mpf(r[p]["lambda1"])) > 0 for r in scan for p in ("even", "odd"))
    summary["all_numerically_certified"] = all(r[p]["numerically_certified_positive"] for r in scan for p in ("even", "odd"))

    # --------------------------------------------------------------------------- 3
    print("[3] basis-size convergence", flush=True)
    conv = []
    for a in ([1.0, 2.0] if not args.quick else [1.0]):
        for J in ([16, 24, 32, 48, 64, 80] if not args.quick else [16, 32]):
            dps = dps_for(J * 0.7 + 20)
            W = wf.build_weil_form(a, J=J, dps=dps, estimate_quad_error=False)
            row = {"a": a, "J": J, "dps": dps}
            for parity in ("even", "odd"):
                Qs, Gs, idx = W.sector(parity)
                vals, _ = wf.smallest_eigenpairs(Qs, Gs, dps=dps, k=1)
                row[parity] = mp.nstr(vals[0], 8)
            conv.append(row)
            print(f"   a={a} J={J} dps={dps}: even {row['even']}  odd {row['odd']}", flush=True)
    summary["basis_convergence"] = conv

    # --------------------------------------------------------------------------- 4
    print("[4] zeros of the minimiser's Fourier transform vs zeros of zeta", flush=True)
    K = 30
    with mp.workdps(80):
        gam = [mp.zetazero(k).imag for k in range(1, K + 1)]
    match = []
    cases = [(0.5, 48), (1.0, 48), (1.5, 48), (2.0, 64)] if not args.quick else [(1.0, 32)]
    for a, J in cases:
        dps = dps_for(J * 0.8 + 30)
        W = wf.build_weil_form(a, J=J, dps=dps, estimate_quad_error=False)
        for parity in ("even", "odd"):
            Qs, Gs, idx = W.sector(parity)
            vals, vecs = wf.smallest_eigenpairs(Qs, Gs, dps=dps, k=1)
            with mp.workdps(dps):
                d = wf.zero_match(vecs[0], idx, a, gam, dps=min(dps, 120), window=0.5)
            digits = [(-float(mp.log10(x)) if (x is not None and x > 0) else None) for x in d]
            n_good = 0
            for x in d:
                if x is not None and x < 1e-3:
                    n_good += 1
                else:
                    break
            rec = {"a": a, "J": J, "parity": parity, "dps": dps, "lambda1": mp.nstr(vals[0], 8),
                   "primes_used_up_to": math.exp(2 * a),
                   "abs_diff": [mp.nstr(x, 3) if x is not None else None for x in d],
                   "digits": digits, "leading_zeros_matched_to_1e-3": n_good,
                   "coeffs": [mp.nstr(c, 30) for c in vecs[0]], "idx": idx}
            match.append(rec)
            print(f"   a={a} J={J} {parity}: lambda1={rec['lambda1']}  first zeros matched to 1e-3: {n_good}; "
                  f"digits of gamma_1..gamma_8: {[round(x,1) if x else None for x in digits[:8]]}", flush=True)
    summary["zero_match"] = match
    summary["gamma_reference"] = [mp.nstr(g, 40) for g in gam]
    summary["wall_seconds"] = time.time() - T0
    with open("results/json/weil_form_summary.json", "w") as f:
        json.dump(summary, f, indent=1)


else:
    summary = json.load(open("results/json/weil_form_summary.json"))
    scan, match = summary["lambda_min_scan"], summary["zero_match"]
    with mp.workdps(40):
        gam = [mp.mpf(g) for g in summary["gamma_reference"]]

# --------------------------------------------------------------------------- figures
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 5))
for J, ls in zip(sorted(set(r["J"] for r in scan)), ["-", "--"]):
    rows = [r for r in scan if r["J"] == J]
    for parity, col in (("even", "C0"), ("odd", "C3")):
        ax.plot([r["a"] for r in rows], [r[parity]["log10_lambda1"] for r in rows], ls, color=col, marker="o", ms=3,
                label=f"{parity} sector, J={J}")
ax.axvline(math.log(2) / 2, color="gray", lw=0.8, ls=":")
ax.text(math.log(2) / 2 - 0.03, -38, "no prime enters\nfor a < (log 2)/2", fontsize=8, va="top", ha="right")
ax.set_xlabel("support half-width a   (f supported in [-a, a], primes up to $e^{2a}$ enter)")
ax.set_ylabel(r"$\log_{10}\lambda_{\min}$  (min of $Q_a(f)/\|f\|^2$ on the subspace)")
ax.set_title("Weil quadratic form (arithmetic side only): smallest eigenvalues stay positive\nand collapse as the support grows", fontsize=10)
ax.grid(alpha=.3); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig("results/figures/weil_form_lambda_min.png", dpi=150)

fig, ax = plt.subplots(figsize=(8, 5))
for rec in match:
    ks = [k + 1 for k, x in enumerate(rec["digits"]) if x is not None]
    ys = [x for x in rec["digits"] if x is not None]
    ax.plot(ks, ys, marker="o", ms=3, lw=1, ls="-" if rec["parity"] == "even" else "--",
            label=f"a={rec['a']} (prime powers n ≤ {math.floor(rec['primes_used_up_to'])}), J={rec['J']}, {rec['parity']}")
ax.set_xlabel("k  (index of the zeta zero $\\gamma_k$)")
ax.set_ylabel(r"$-\log_{10}|r_k - \gamma_k|$   ($r_k$: nearest real zero of $\hat f_{\min}$)")
ax.set_title("Zeros of the Weil-form minimiser's Fourier transform vs the zeros of zeta", fontsize=10)
ax.axhline(3, color="gray", lw=0.6, ls=":")
ax.grid(alpha=.3); ax.legend(fontsize=7)
fig.tight_layout(); fig.savefig("results/figures/weil_form_zero_match.png", dpi=150)

rec = next(r for r in match if r["a"] == 1.0 and r["parity"] == "even")
coeffs = [float(mp.mpf(c)) for c in rec["coeffs"]]
rs = np.linspace(0.01, 60, 6000)
fh = wf.fourier_transform_float(coeffs, rec["idx"], rec["a"], rs)
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.semilogy(rs, np.abs(fh), lw=0.8, label=r"$|\hat f_{\min}(r)|$, a = 1, even sector")
for g in gam:
    if float(g) < 60:
        ax.axvline(float(g), color="C3", lw=0.6, ls=":")
ax.set_xlabel("r"); ax.set_ylabel(r"$|\hat f_{\min}(r)|$")
ax.set_title(r"Minimiser of the Weil form on $[-1,1]$ (prime powers 2, 3, 4, 5, 7 only): dips at the zeros of zeta (red)", fontsize=10)
ax.legend(fontsize=8); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig("results/figures/weil_form_minimiser_fourier.png", dpi=150)
print("done", time.time() - T0)
