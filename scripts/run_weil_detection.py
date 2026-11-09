#!/usr/bin/env python3
"""E12 stage 2: how sensitive is Weil's positivity criterion to zeros off the critical line?

Part A. Davenport–Heilbronn (DH), a genuine RH violation, with L(s,(n/5)) (GRH) as control.
  A1  For wave packets centred at each certified off-line zero of DH, the smallest support a at which
      the restricted Weil form (zero side: 204 on-line zeros, 5 off-line quadruples, density tail;
      float64) has a negative eigenvalue, and the control's smallest eigenvalue on the same space.
  A2  The same sign from the ARITHMETIC side alone (Arb; only a_n, n <= e^{2a}, and the Gamma factor).
  A3  Rigorous certificates (rh.weil_certificate): Q_DH(f) < 0 for explicit f supported in [-a, a];
      by the explicit formula DH then has a zero off the critical line.
Part B. Zeta, thought experiment.  Replace the adjacent zeros gamma_k, gamma_{k+1} by the quadruple
  1/2 +- delta + i gamma0 (gamma0 their midpoint; the zero count is unchanged, as when a close pair
  collides and leaves the line).  By the explicit formula the arithmetic side of this hypothetical
  function equals its zero side, computed here from the 10^6 Arb-certified zeros.  For each
  (gamma0, delta): the smallest support a*(gamma0, delta) at which Weil's form, restricted to J wave
  packets at height gamma0, has a negative eigenvalue (an UPPER bound for the true detection
  threshold: a negative value on a subspace is negative on the whole space).  Several pairs per
  height, several bases, and the additive model (extra quadruple, no zeros removed).  Arb
  cross-check of the zero-side eigenvalues at gamma0 ~ 102.5 from zeta's arithmetic side.
Part C. Comparison with the detection horizons of Li's and the Nyman–Beurling criterion (E4, E11).

Usage:  python3 scripts/run_weil_detection.py [--quick] [--skip-certificates]
Writes results/json/weil_detection_summary.json and results/figures/weil_detection_*.png.
"""
import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import mpmath as mp  # noqa: E402
import numpy as np  # noqa: E402

from rh import weil_certificate as wc  # noqa: E402
from rh import weil_form_general as wg  # noqa: E402
from rh.data import load_zeros  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT_JSON = os.path.join(ROOT, "results", "json", "weil_detection_summary.json")
FIG = os.path.join(ROOT, "results", "figures")


def lam_min_float(B, par, Z):
    ev, vec, rank = wg.smallest_eigenpairs_float(Z, wg.gram_matrix_float(B, par))
    return float(ev[0]), vec[:, 0], rank


# ----------------------------------------------------------------------------
# Part A
# ----------------------------------------------------------------------------
def dh_zero_side(B, par, zd, s):
    z = zd[s]
    return wg.zero_side_matrix(B, par, on_line=z["on_line"], off_line=z["off_line"], series=s, tail=True,
                               T=z["T"], s_correction=True)["Z"]


def part_a1(zd, configs, a_grid):
    out = []
    for rho in zd["dh"]["off_line"]:
        r0 = float(rho.imag)
        best = None
        for (J, m) in configs:
            for a in a_grid:
                B = wg.Basis.localized(float(a), J, r0, m)
                lam, vec, rank = lam_min_float(B, "even", dh_zero_side(B, "even", zd, "dh"))
                if lam < 0:
                    if best is None or a < best["a"] - 1e-9 or (abs(a - best["a"]) < 1e-9 and lam < best["lambda_dh"]):
                        z = zd["dh"]
                        dec = wg.zero_side(B, "even", vec, on_line=z["on_line"], off_line=z["off_line"], series="dh",
                                           tail=True, T=z["T"], s_correction=True)
                        lam5, _, _ = lam_min_float(B, "even", dh_zero_side(B, "even", zd, "chi5"))
                        best = {"a": float(a), "J": J, "m": m, "r0": r0, "lambda_dh": lam, "lambda_chi5": lam5,
                                "n_max": int(math.floor(math.exp(2 * a))),
                                "zero_side_on_line": dec["on_line"], "zero_side_off_line": dec["off_line"],
                                "off_line_each": [(e["rho"], e["contribution"]) for e in dec["off_line_each"]]}
                    break
        row = {"rho": [float(rho.real), float(rho.imag)], "delta": float(rho.real) - 0.5,
               "density_prediction": 0.5 * math.log(5 * r0 / (2 * math.pi)), "threshold": best}
        out.append(row)
        print(f"  A1 rho={rho.real:.4f}+{rho.imag:.3f}i: first negative at "
              f"a={best['a'] if best else None} (J={best['J'] if best else '-'}, m={best['m'] if best else '-'}), "
              f"lambda_dh={best['lambda_dh'] if best else float('nan'):.4f}, chi5 {best['lambda_chi5'] if best else float('nan'):.4f}; "
              f"density prediction {row['density_prediction']:.3f}", flush=True)
    return out


def part_a_curve(zd, J, m, r0, a_grid):
    rows = []
    for a in a_grid:
        B = wg.Basis.localized(float(a), J, r0, m)
        rows.append([float(a)] + [lam_min_float(B, "even", dh_zero_side(B, "even", zd, s))[0] for s in ("dh", "chi5")])
    return rows


def part_a2(cases, dps):
    out = []
    for (a, J, m, r0) in cases:
        B = wg.Basis.localized(a, J, r0, m)
        rec = {"a": a, "J": J, "m": m, "r0": r0, "dps": dps}
        for s in ("dh", "chi5"):
            t = time.time()
            S = wg.build_sector(s, B, "even", dps=dps)
            vals, vecs = wg.smallest_eigenpairs(S, k=1)
            eb = wg.eigenvalue_error_bound(S, vals[0])
            rec[s] = {"lambda": mp.nstr(vals[0], 15), "error_bound": mp.nstr(eb["bound"], 3),
                      "sign_certified_numerically": eb["sign_certified"], "n_terms": len(S.terms),
                      "seconds": time.time() - t, "coeffs": [mp.nstr(c, 30) for c in vecs[0]]}
        out.append(rec)
        print(f"  A2 a={a} J={J} m={m} r0={r0}: arithmetic side DH lambda={rec['dh']['lambda']} "
              f"(+-{rec['dh']['error_bound']}), chi5 lambda={rec['chi5']['lambda']}", flush=True)
    return out


def part_a3(a2_rows, dps):
    out = []
    for rec in a2_rows:
        B = wg.Basis.localized(rec["a"], rec["J"], rec["r0"], rec["m"])
        coeffs = wc.rationalize([mp.mpf(c) for c in rec["dh"]["coeffs"]])
        res = wc.certify_weil_value("dh", B, "even", coeffs, dps=dps)
        row = {"a": rec["a"], "J": rec["J"], "m": rec["m"], "r0": rec["r0"], "n_max": int(math.floor(math.exp(2 * rec["a"]))),
               "Q_lower": res["lower"], "Q_upper": res["upper"], "norm2": res["norm2"].str(15),
               "ratio": res["ratio"].str(15), "negative": res["negative"], "terms_used": res["terms"],
               "seconds": res["seconds"], "coeffs_exact": [f"{c.numerator}/{c.denominator}" for c in coeffs]}
        out.append(row)
        print(f"  A3 certificate a={rec['a']} (n <= {row['n_max']}): Q(f) in [{res['lower']:.12g}, {res['upper']:.12g}], "
              f"||f||^2 = {row['norm2']}, negative={res['negative']} [{res['seconds']:.0f}s]", flush=True)
    return out


# ----------------------------------------------------------------------------
# Part B
# ----------------------------------------------------------------------------
class ZetaThought:
    def __init__(self, zeros, window=2000.0):
        self.Z = zeros
        self.W = window

    def thresholds(self, k, deltas, J, m, a_grid, model="replace", par="even"):
        """Smallest a on a_grid with a negative eigenvalue, per delta, for the pair (gamma_k, gamma_{k+1})
        (k 1-based) replaced by (model 'replace') or supplemented with (model 'add') the quadruple."""
        Z = self.Z
        g1, g2 = float(Z[k - 1]), float(Z[k])
        g0 = 0.5 * (g1 + g2)
        lo, hi = int(np.searchsorted(Z, g0 - self.W)), int(np.searchsorted(Z, g0 + self.W))
        idx = np.arange(lo, hi)
        if model == "replace":
            idx = idx[(idx != k - 1) & (idx != k)]
        on = Z[idx]
        T = float(Z[hi - 1])
        sg = 1 if par == "even" else -1
        found = {d: None for d in deltas}
        curve = []
        for a in a_grid:
            B = wg.Basis.localized(float(a), J, g0, m)
            G = wg.gram_matrix_float(B, par)
            Zon = wg.zero_side_matrix(B, par, on_line=on, series="zeta", tail=True, T=T)["Z"]
            v = wg.fourier_matrix(B, par, np.array([g0 - 1j * d for d in deltas]))
            row = []
            for i, d in enumerate(deltas):
                M = 4 * sg * np.real(np.outer(v[i], v[i]))
                lam = float(wg.smallest_eigenpairs_float(Zon + M, G)[0][0])
                row.append(lam)
                if lam < 0 and found[d] is None:
                    found[d] = float(a)
            curve.append([float(a)] + row)
            if all(found[d] is not None for d in deltas):
                break
        return {"k": k, "gamma_k": g1, "gamma_k1": g2, "gamma0": g0, "gap": g2 - g1,
                "mean_gap": 2 * math.pi / math.log(g0 / (2 * math.pi)), "a_star": {str(d): found[d] for d in deltas},
                "curve": curve}


def part_b(zt, ks_by_height, n_pairs, deltas, configs, a_grid, model="replace"):
    out = []
    for k0 in ks_by_height:
        for (J, m) in configs:
            t = time.time()
            recs = [zt.thresholds(k0 + i, deltas, J, m, a_grid, model=model) for i in range(n_pairs)]
            g0 = float(np.mean([r["gamma0"] for r in recs]))
            summ = {}
            for d in deltas:
                vals = [r["a_star"][str(d)] for r in recs if r["a_star"][str(d)] is not None]
                summ[str(d)] = {"median": float(np.median(vals)) if vals else None,
                                "min": float(min(vals)) if vals else None, "max": float(max(vals)) if vals else None,
                                "n_found": len(vals)}
            row = {"k0": k0, "gamma0_mean": g0, "J": J, "m": m, "model": model, "n_pairs": n_pairs,
                   "density_prediction": 0.5 * math.log(g0 / (2 * math.pi)), "summary": summ,
                   "pairs": [{kk: v for kk, v in r.items() if kk != "curve"} for r in recs]}
            out.append(row)
            print(f"  B {model:7s} gamma0~{g0:10.1f} J={J} m={m}: pred {row['density_prediction']:.3f} | " +
                  "  ".join(f"d={d}: {summ[str(d)]['median']} [{summ[str(d)]['min']},{summ[str(d)]['max']}]" for d in deltas) +
                  f" [{time.time() - t:.0f}s]", flush=True)
    return out


def part_b_arb_check(zt, k, delta, J, m, a_values, dps):
    """Zeta's ARITHMETIC side (Arb) minus the two removed on-line pairs plus the quadruple, against the
    zero-side (float64) eigenvalue."""
    Z = zt.Z
    g1, g2 = float(Z[k - 1]), float(Z[k])
    g0 = 0.5 * (g1 + g2)
    out = []
    for a in a_values:
        B = wg.Basis.localized(a, J, g0, m)
        S = wg.build_sector("zeta", B, "even", dps=dps)
        Q = wg.add_zero_quadruple(S, g1, 0.0, weight=-1)
        Q = wg.add_zero_quadruple(Q, g2, 0.0, weight=-1, basis=B, parity="even", dps=dps)
        Q = wg.add_zero_quadruple(Q, g0, delta, weight=1, basis=B, parity="even", dps=dps)
        vals, _ = wg.smallest_eigenpairs(Q, S.G, dps=dps, k=1)
        zr = zt.thresholds(k, [delta], J, m, [a])["curve"][0][1]
        out.append({"a": a, "lambda_arith": mp.nstr(vals[0], 12), "lambda_zero_side": zr,
                    "difference": float(abs(vals[0] - zr))})
        print(f"  B-check k={k} gamma0={g0:.3f} delta={delta} a={a}: arithmetic side {mp.nstr(vals[0], 12)}, "
              f"zero side {zr:.12f}", flush=True)
    return out


# ----------------------------------------------------------------------------
# Part C
# ----------------------------------------------------------------------------
def part_c(b_rows, deltas):
    from rh.li import detection_horizon
    from rh.nyman_beurling import detection_horizon_log_N
    out = []
    for row in b_rows:
        if row["model"] != "replace" or (row["J"], row["m"]) != (16, 4):
            continue
        g0 = row["gamma0_mean"]
        for d in deltas:
            a = row["summary"][str(d)]["median"]
            out.append({"gamma0": g0, "delta": d, "weil_a_star": a,
                        "weil_X_star": math.exp(2 * a) if a is not None else None,
                        "li_n_star": detection_horizon(g0, d), "nb_log10_N_star": detection_horizon_log_N(g0, d) / math.log(10)})
    return out


# ----------------------------------------------------------------------------
# figures
# ----------------------------------------------------------------------------
def figures(res, deltas):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(FIG, exist_ok=True)
    # DH vs chi5 curve
    c = np.array(res["A_curve"]["rows"])
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(c[:, 0], c[:, 1], "o-", ms=3, color="#c0392b", label="Davenport–Heilbronn (RH false)")
    ax.plot(c[:, 0], c[:, 2], "s-", ms=3, color="#2e86c1", label="L(s,(n/5)) (GRH)")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("support half-width a   (coefficients n ≤ e^{2a} enter)")
    ax.set_ylabel("smallest eigenvalue of Weil's form")
    p = res["A_curve"]
    ax.set_title(f"Wave packets at height {p['r0']} (J={p['J']}, m={p['m']})", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "weil_detection_dh.png"), dpi=150)
    plt.close(fig)
    # zeta thresholds
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    rows = [r for r in res["B"] if r["model"] == "replace" and (r["J"], r["m"]) == (16, 4)]
    g = np.array([r["gamma0_mean"] for r in rows])
    cmap = plt.get_cmap("viridis")
    for i, d in enumerate(deltas):
        med = np.array([r["summary"][str(d)]["median"] or np.nan for r in rows])
        lo = np.array([r["summary"][str(d)]["min"] or np.nan for r in rows])
        hi = np.array([r["summary"][str(d)]["max"] or np.nan for r in rows])
        col = cmap(i / max(1, len(deltas) - 1))
        ax.errorbar(g, med, yerr=[med - lo, hi - med], fmt="o-", ms=3, capsize=2, color=col, label=f"δ = {d}")
    gg = np.geomspace(g.min() * 0.8, g.max() * 1.25, 100)
    ax.plot(gg, 0.5 * np.log(gg / (2 * np.pi)), "k--", lw=1, label="½ log(γ/2π) (zero-density prediction)")
    ax.set_xscale("log")
    ax.set_xlabel("height γ0 of the hypothetical off-line zero")
    ax.set_ylabel("detection threshold a*  (primes ≤ e^{2a*})")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "weil_detection_zeta.png"), dpi=150)
    plt.close(fig)
    # horizons
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    cc = [r for r in res["C"] if abs(r["delta"] - 0.1) < 1e-12]
    if cc:
        g = np.array([r["gamma0"] for r in cc])
        ax.loglog(g, [r["weil_X_star"] for r in cc], "o-", label="Weil: largest prime power used, e^{2a*}")
        ax.loglog(g, [r["li_n_star"] for r in cc], "s-", label="Li: index n* of the first negative λ_n")
        ax.set_xlabel("height γ0 of the off-line zero (δ = 0.1)")
        ax.set_ylabel("size of the computation")
        ax2 = ax.twinx()
        ax2.semilogx(g, [r["nb_log10_N_star"] for r in cc], "^-", color="#7d3c98", label="Nyman–Beurling: log10 N*")
        ax2.set_yscale("log")
        ax2.set_ylabel("log10 N* (Nyman–Beurling)")
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper left")
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, "weil_detection_horizons.png"), dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--skip-certificates", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    zd = {s: wg.load_zero_data(s) for s in ("chi5", "dh")}
    res = {"description": "E12 stage 2: sensitivity of Weil positivity to off-line zeros", "quick": args.quick}
    deltas = [0.4, 0.2, 0.1, 0.05, 0.02, 0.01]
    print("Part A: Davenport–Heilbronn vs L(s,(n/5))", flush=True)
    a_grid_a = np.round(np.arange(1.0, 3.51, 0.05 if args.quick else 0.02), 3)
    cfg_a = [(20, 2)] if args.quick else [(20, 2), (36, 1), (36, 2), (20, 4), (28, 4)]
    res["A1"] = part_a1(zd, cfg_a, a_grid_a)
    res["A_curve"] = {"J": 20, "m": 4, "r0": 85.7,
                      "rows": part_a_curve(zd, 20, 4, 85.7, np.round(np.arange(0.4, 3.01, 0.1 if args.quick else 0.05), 3))}
    t1 = res["A1"][0]["threshold"]
    a2_cases = [(2.0, 20, 4, 85.7)] if args.quick else [(2.0, 20, 4, 85.7), (round(t1["a"] + 0.04, 2), t1["J"], t1["m"], round(t1["r0"], 3))]
    res["A2"] = part_a2(a2_cases, dps=40)
    if not args.skip_certificates:
        res["A3"] = part_a3(res["A2"], dps=15)
    print("Part B: zeta with a pair moved off the line", flush=True)
    zeros = load_zeros(10 ** 6)
    zt = ZetaThought(zeros)
    a_grid_b = np.round(np.arange(0.8, 8.0, 0.05), 3)
    ks = [30, 649, 10142, 138069, 800000] if args.quick else [30, 200, 649, 2600, 10142, 40000, 138069, 400000, 800000]
    n_pairs = 2 if args.quick else 8
    res["B"] = part_b(zt, ks, n_pairs, deltas, [(16, 4)], a_grid_b)
    if not args.quick:
        res["B"] += part_b(zt, [649, 10142, 138069], 4, deltas, [(24, 4), (16, 2), (32, 3)], a_grid_b)
        res["B"] += part_b(zt, [649, 10142, 138069], 4, deltas, [(16, 4)], a_grid_b, model="add")
    res["B_arb_check"] = part_b_arb_check(zt, 30, 0.1, 16, 4, [1.45, 1.5], dps=40)
    res["C"] = part_c(res["B"], deltas)
    res["seconds"] = time.time() - t0
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(res, f, indent=1, default=str)
    figures(res, deltas)
    print(f"wrote {OUT_JSON} [{res['seconds']:.0f}s]")


if __name__ == "__main__":
    main()
