#!/usr/bin/env python3
"""E13: certified positivity of Weil's quadratic form on whole windows [-L, L] (zeta).

For each window [-L, L] and each parity sector (even / odd real f):
  * lower bound: the envelope reduction of `rh.weil_envelope` in ball arithmetic gives a certified c
    with Q(f) >= c ||f||^2 for EVERY real f of the sector supported in [-L, L];
  * upper bound: `rh.weil_certificate` encloses Q(f)/||f||^2 rigorously for an explicit polynomial f
    (the minimiser of `rh.weil_form` on Legendre polynomials of degree < J), so
    lambda_min(L) := inf Q(f)/||f||^2 lies in [c, upper].
Even and odd sectors decouple, so min(c_even, c_odd) bounds Q(f)/||f||^2 for all real f.

Usage: python3 scripts/run_weil_positivity.py [--quick] [--only L1,L2,...]
Writes results/json/weil_positivity_summary.json and results/figures/weil_positivity.png.
"""
import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import mpmath as mp  # noqa: E402

from rh import weil_certificate as wc  # noqa: E402
from rh import weil_envelope as we  # noqa: E402
from rh import weil_form as wf  # noqa: E402
from rh import weil_form_general as wg  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT_JSON = os.path.join(ROOT, "results", "json", "weil_positivity_summary.json")
FIG = os.path.join(ROOT, "results", "figures", "weil_positivity.png")

# (L, T, dps): T > T_1 = 2 pi e^{A_L} is needed for beta > 0:
# A_L = 2.942 (n = 2, 3, 4) for L < log(5)/2 = 0.80472, 4.381 (n <= 5) for L < log(7)/2 = 0.97296.
# The number of Legendre modes per sector follows from T (`modes_for`).  If the envelope form is not
# positive (its loss exceeds the margin), T is raised by a factor 1.4 and the sector retried.
WINDOWS = [
    (0.5, 40.0, 40),
    (0.8, 200.0, 40),           # Zhu's window (arXiv:2608.24827: c = 8.9e-18, even)
    (0.8047, 200.0, 40),        # the largest window whose comb is {2, 3, 4}
    (0.81, 600.0, 40),
    (0.85, 600.0, 40),
    (0.9, 600.0, 50),
    (0.95, 600.0, 50),
    (0.97, 600.0, 50),
    (0.9729, 600.0, 50),        # the largest window whose comb is {2, 3, 4, 5} (log(7)/2 = 0.972955)
]
QUICK = [(0.5, 40.0, 40), (0.8, 200.0, 40)]
# beyond log(7)/2 the comb adds n = 7 (A_L = 5.852, T_1 ~ 2187): about 1900 modes per sector,
# ~1 hour per sector here; run with --only 1.0 --merge
EXTENDED = [(1.0, 2600.0, 55)]


def upper_bound(L, parity, J=None, dps=40):
    """Rigorous upper bound for lambda_min(L): Q(f)/||f||^2 for the degree < J Legendre minimiser
    (J = 64 for L < 0.9, 96 beyond, where the Legendre minimisers converge slowly)."""
    J = J or (64 if L < 0.9 else 96)
    W = wf.build_weil_form(L, J, dps=max(80, dps + 30), estimate_quad_error=False)
    Qs, Gs, idx = W.sector(parity)
    v, vec = wf.smallest_eigenpairs(Qs, Gs, max(80, dps + 30), k=1)
    c = wc.rationalize(vec[0])
    r = wc.certify_weil_value("zeta", wg.Basis.legendre(L, J), parity, c, dps=dps, eps=10.0 ** (-(dps + 8)))
    return {"J": J, "lambda_J_numerical": mp.nstr(v[0], 10), "upper": r["ratio_upper"],
            "lower_of_ratio": r["ratio_lower"], "seconds": r["seconds"]}


def modes_for(L, T):
    """Legendre modes per sector: orders up to ~1.25 e L T / 2, where |j_n(LT)| <= (LT)^n/(2n+1)!! is
    already ~e^{-0.2 n} below 1 and the coupling bounds are negligible."""
    return int(math.ceil(1.25 * math.e * L * T / 4)) + 20


def figure(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for parity, col, mk in (("even", "#c0392b", "o"), ("odd", "#2e86c1", "s")):
        r = [x for x in rows if x["parity"] == parity and x["positive"]]
        Ls = [x["L"] for x in r]
        ax.semilogy(Ls, [x["certified_lower_bound"] for x in r], mk + "-", color=col, ms=4,
                    label=f"certified lower bound ({parity})")
        ax.semilogy(Ls, [x["upper"]["upper"] for x in r], mk + "--", color=col, ms=3, mfc="none",
                    label=f"rigorous upper bound ({parity})")
    ax.errorbar([0.8], [math.sqrt(8.9e-18 * 2.27e-17)], yerr=[[math.sqrt(8.9e-18 * 2.27e-17) - 8.9e-18],
                [2.27e-17 - math.sqrt(8.9e-18 * 2.27e-17)]], fmt="none", ecolor="k", capsize=4,
                label="Zhu 2026, even: [8.9e-18, 2.27e-17]")
    for x0, lab in ((math.log(5) / 2, "n = 5 enters"),):
        ax.axvline(x0, color="0.6", lw=0.8, ls=":")
        ax.text(x0, ax.get_ylim()[0] * 10, " " + lab, fontsize=7, color="0.4")
    ax.set_xlabel("window half-width L   (test functions supported in [-L, L])")
    ax.set_ylabel("inf Q(f) / ||f||^2")
    ax.set_title("Weil positivity certified on whole windows (zeta)", fontsize=10)
    ax.legend(fontsize=7)
    fig.tight_layout()
    os.makedirs(os.path.dirname(FIG), exist_ok=True)
    fig.savefig(FIG, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", default="")
    ap.add_argument("--merge", action="store_true",
                    help="add/replace the computed windows in the stored summary instead of overwriting it")
    ap.add_argument("--parts", choices=["all", "upper"], default="all",
                    help="'upper': recompute only the rigorous upper bounds of the stored summary")
    args = ap.parse_args()
    if args.parts == "upper":
        with open(OUT_JSON) as f:
            out = json.load(f)
        t0 = time.time()
        for r in out["windows"]:
            r["upper"] = upper_bound(r["L"], r["parity"], dps=r["dps"])
            r["certified_interval"] = [r["certified_lower_bound"], r["upper"]["upper"]]
            print(f"  L={r['L']} {r['parity']}: upper {r['upper']['upper']:.6e} "
                  f"(J={r['upper']['J']}, {r['upper']['seconds']:.0f}s)", flush=True)
        out["upper_seconds"] = time.time() - t0
        with open(OUT_JSON, "w") as f:
            json.dump(out, f, indent=1, default=str)
        figure(out["windows"])
        print(f"updated upper bounds in {OUT_JSON}")
        return
    windows = QUICK if args.quick else WINDOWS
    if args.only:
        keep = {float(x) for x in args.only.split(",")}
        windows = [w for w in WINDOWS + EXTENDED if w[0] in keep]
    t0 = time.time()
    rows = []
    for (L, T0, dps) in windows:
        for parity in ("even", "odd"):
            T, attempts = T0, []
            for _ in range(4):
                r = we.certify_window(L, parity, T=T, N=modes_for(L, T), dps=dps, verbose=True)
                attempts.append({"T": T, "N": r["N"], "lambda_approx": r["lambda_approx"],
                                 "certified_lower_bound": r["certified_lower_bound"]})
                if r["positive"]:
                    break
                T = round(T * 1.4)
            r["attempts"] = attempts
            r["upper"] = upper_bound(L, parity, dps=dps)
            r["certified_interval"] = [r["certified_lower_bound"], r["upper"]["upper"]]
            rows.append(r)
            lo = f"{r['certified_lower_bound']:.4e}" if r["positive"] else "NOT CERTIFIED"
            print(f"  L={L} {parity} (T={r['T']}): lambda_min in [{lo}, {r['upper']['upper']:.4e}]", flush=True)
    if args.merge and os.path.exists(OUT_JSON):
        with open(OUT_JSON) as f:
            old = json.load(f)
        new_keys = {(r["L"], r["parity"]) for r in rows}
        rows = sorted([r for r in old["windows"] if (r["L"], r["parity"]) not in new_keys] + rows,
                      key=lambda r: (r["L"], r["parity"]))
    out = {"description": "E13: certified Weil positivity on whole windows [-L, L] (zeta), envelope reduction",
           "quick": args.quick, "windows": rows, "seconds": time.time() - t0}
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=1, default=str)
    figure(rows)
    print(f"wrote {OUT_JSON} [{out['seconds']:.0f}s]")


if __name__ == "__main__":
    main()
