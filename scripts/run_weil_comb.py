#!/usr/bin/env python3
"""E14: Weil positivity on whole windows past the pointwise barrier, via the non-pointwise comb bound.

The envelope reduction of E13 (`rh.weil_envelope`) needs T > T_1 = 2 pi e^{A_L}, where A_L is the
pointwise comb constant. Since A_L ~ 4 e^L, T_1 is doubly exponential in L. `rh.weil_comb` replaces
A_L by a certified constant A' (separable sum-of-squares certificates A' - P_L + H >= 0, where H has
all frequencies above 2L), so the cutoff becomes T_1'' = 2 pi e^{A'}. This script:

  * tabulates the comb constants: A_L (pointwise), A' (certified, frequency budget X), the separable
    optimum, and the operator-norm lower limit A*, with their cutoffs T_1;
  * certifies Q(f) >= c ||f||^2 for every real f of each parity supported in [-L, L], for the windows
    below (`certify_window` with the comb certificate, all in ball arithmetic);
  * encloses inf Q / ||f||^2 from above rigorously with an explicit polynomial (`rh.weil_certificate`).

Usage: python3 scripts/run_weil_comb.py [--quick] [--only L1,L2] [--merge] [--parts upper|constants]
Writes results/json/weil_comb_summary.json and results/figures/weil_comb.png.
"""
import argparse
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import mpmath as mp  # noqa: E402

from rh import weil_certificate as wc  # noqa: E402
from rh import weil_comb as wcb  # noqa: E402
from rh import weil_envelope as we  # noqa: E402
from rh import weil_form as wf  # noqa: E402
from rh import weil_form_general as wg  # noqa: E402

ROOT = os.path.join(HERE, "..")
OUT_JSON = os.path.join(ROOT, "results", "json", "weil_comb_summary.json")
E13_JSON = os.path.join(ROOT, "results", "json", "weil_positivity_summary.json")
FIG = os.path.join(ROOT, "results", "figures", "weil_comb.png")
CERT_JSON = os.path.join(ROOT, "results", "json", "weil_comb_certificates.json")

X_BUDGET = 24.0          # frequencies of H at most 24 (separable certificates)
# (L, T, dps, J_upper, certificate): T a little above T_1'' = 2 pi e^{A'}; dps from the expected margins;
# certificate ("sep", X) = separable SOS with frequency budget X, ("joint", Lambda) = joint torus SOS
WINDOWS = [
    (1.0, 300.0, 55, 96, ("sep", X_BUDGET)),
    (1.1, 400.0, 62, 160, ("sep", X_BUDGET)),
    (1.2, 800.0, 72, 192, ("sep", X_BUDGET)),
    (1.3, 800.0, 85, 160, ("joint", 5.5)),
    (1.4, 850.0, 100, 160, ("joint", 6.5)),
    (1.5, None, 130, 160, ("joint-admm", 8.0, 25000)),   # T = 1.3 T_1''; ADMM for the 619-monomial SOS
    # L = 1.6: 1239-monomial SOS (ADMM, ~2.5 h with 3 BLAS threads); T = 1.2 T_1'', 1.1 e L T / 4 modes per
    # sector, both sectors from one shared assembly (~5.4 h at 160 digits, ~9-10 GB peak memory)
    (1.6, None, 160, 160, ("joint-admm", 9.0, 40000), {"ratio": 1.2, "modes_factor": 1.1, "paired": True}),
]
DEFAULT_OPTS = {"ratio": 1.3, "modes_factor": 1.25, "paired": False}
QUICK = [(1.0, 170.0, 50, 64, ("sep", X_BUDGET))]
CONSTANT_LS = [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6]
JOINT_LAMBDA = 6.0       # joint certificates in the constants table (monomials n <= e^6)


def modes_for(L, T, factor=1.25):
    """Legendre modes per sector (as in E13): orders up to ~factor e L T / 2 (the coupling bounds stay far below
    the margins down to factor ~1.1 at these sizes)."""
    return int(math.ceil(factor * math.e * L * T / 4)) + 20


def constants(Ls, X=X_BUDGET, J=100, Lam=JOINT_LAMBDA):
    """Comb constants: pointwise A_L, separable A' (budget X), separable optimum, joint A' (monomials
    n <= e^Lam; needs a single-threaded BLAS to be fast), the Fejer lower bound and the operator norm A*."""
    rows = []
    for L in Ls:
        d = wcb.comb_constants(L, X=X, J=J)
        d.pop("per_prime", None)
        jc = wcb.joint_comb_certificate(L, Lam).describe()
        d.update({"A_joint": jc["A_prime"], "joint_Lambda": Lam, "T1_joint": jc["T1_certificate"],
                  "joint_dual": jc["dual"]})
        comb = wcb.prime_comb(L)
        d["A_fejer_lower"] = sum(float(c.mid()) * (1 - math.log(n) / (2 * L)) for t in comb.values() for n, c in t)
        rows.append(d)
        print(f"  L={L}: A_L={d['A_L']:.3f} separable A'={d['A_prime']:.3f} (optimum {d['separable_optimum']:.3f}) "
              f"joint A'={d['A_joint']:.3f} A*={d['A_star']:.3f} (Fejer >= {d['A_fejer_lower']:.3f}); "
              f"T_1 = {d['T1_pointwise']:.4g} / {d['T1_certificate']:.4g} / {d['T1_joint']:.4g} / "
              f"{d['T1_operator']:.4g}", flush=True)
    return rows


def upper_bound(L, parity, J, dps):
    """Rigorous upper bound for inf Q/||f||^2: Q(f)/||f||^2 for the degree < J Legendre minimiser."""
    dps_eig = int(1.5 * J) + 40
    W = wf.build_weil_form(L, J, dps=dps_eig, estimate_quad_error=False)
    Qs, Gs, idx = W.sector(parity)
    v, vec = wf.smallest_eigenpairs(Qs, Gs, dps_eig, k=1)
    # round the minimiser exactly at ~dps + 30 digits (rounding through float would move Q(f) by ~1e-33)
    c = wc.rationalize(vec[0], bits=int(3.33 * (dps + 30)) + 64)
    r = wc.certify_weil_value("zeta", wg.Basis.legendre(L, J), parity, c, dps=dps, eps=10.0 ** (-(dps + 8)))
    return {"J": J, "lambda_J_numerical": mp.nstr(v[0], 10), "upper": r["ratio_upper"],
            "lower_of_ratio": r["ratio_lower"], "seconds": r["seconds"]}


_CERTS = {}


def comb_cert(L, spec):
    """The comb certificate for window L ('sep', X) or ('joint', Lambda), built once per run."""
    key = (L, tuple(spec))
    if key not in _CERTS:
        kind, par = spec[0], spec[1]
        if kind == "sep":
            _CERTS[key] = wcb.comb_certificate(L, par)
        elif kind == "joint-admm":
            _CERTS[key] = wcb.joint_comb_certificate(L, par, solver="admm", admm_iters=spec[2])
        else:
            _CERTS[key] = wcb.joint_comb_certificate(L, par)
    return _CERTS[key]


def certificate_record(cert, spec):
    """Everything needed to re-check (*) independently: A' and the exact coefficients of H."""
    from flint import ctx
    old = ctx.prec
    ctx.prec = 4000
    try:
        terms = [[n, d, h.mid().str(1300, radius=False)] for n, d, h in cert.terms]
    finally:
        ctx.prec = old
    return {"L": cert.L, "kind": spec[0], "parameter": spec[1], "A_prime_upper": float(cert.A_prime.upper()),
            "A_L": float(cert.A_L.mid()), "H_terms_n_d_h": terms,
            "note": "A' - sum_{log n < 2L} (2 Lambda(n)/sqrt(n)) cos(t log n) + sum h cos(t log(n/d)) >= 0 for all real t; "
                    "every log(n/d) >= 2L"}


def _cert_summary(cert, spec):
    d = cert.describe()
    out = {k: d[k] for k in ("A_L", "A_prime", "separable_optimum", "T1_pointwise", "T1_certificate", "H_sup",
                             "n_terms", "omega_min", "X")}
    out["per_prime"] = d["per_prime"]
    out["kind"] = spec[0]
    out.update({k: d[k] for k in ("dual", "nB", "n_free", "n_dropped") if k in d})
    return out


def certify_pair(L, T0, dps, spec, opts):
    """Both sectors from one shared-node assembly (`certify_windows`); no automatic retry."""
    cert = comb_cert(L, spec)
    if T0 is None:
        T0 = float(round(opts["ratio"] * 2 * math.pi * math.exp(float(cert.A_prime.upper()))))
    N = modes_for(L, T0, opts["modes_factor"])
    res = we.certify_windows(L, T0, {"even": N, "odd": N}, dps=dps, verbose=True, comb_cert=cert, wmax=8.0,
                             chunk=1000)
    for r in res.values():
        r["attempts"] = [{"T": T0, "N": N, "lambda_approx": r["lambda_approx"],
                          "certified_lower_bound": r["certified_lower_bound"]}]
        r["comb_certificate"] = _cert_summary(cert, spec)
        r["modes_factor"] = opts["modes_factor"]
    return res


def certify(L, T0, dps, parity, spec=("sep", X_BUDGET), opts=None):
    opts = dict(DEFAULT_OPTS, **(opts or {}))
    cert = comb_cert(L, spec)
    if T0 is None:                                   # a fixed multiple of the cutoff 2 pi e^{A'}
        T0 = float(round(opts["ratio"] * 2 * math.pi * math.exp(float(cert.A_prime.upper()))))
    T, attempts = T0, []
    for _ in range(4):
        r = we.certify_window(L, parity, T=T, N=modes_for(L, T, opts["modes_factor"]), dps=dps, verbose=True,
                              comb_cert=cert, wmax=8.0, chunk=2000)
        attempts.append({"T": T, "N": r["N"], "lambda_approx": r["lambda_approx"],
                         "certified_lower_bound": r["certified_lower_bound"]})
        if r["positive"]:
            break
        T = round(T * 1.3)
    r["attempts"] = attempts
    d = cert.describe()
    r["comb_certificate"] = {k: d[k] for k in ("A_L", "A_prime", "separable_optimum", "T1_pointwise",
                                               "T1_certificate", "H_sup", "n_terms", "omega_min", "X")}
    r["comb_certificate"]["per_prime"] = d["per_prime"]
    r["comb_certificate"]["kind"] = spec[0]
    r["comb_certificate"].update({k: d[k] for k in ("dual", "nB", "n_free", "n_dropped") if k in d})
    return r


def figure(rows, e13_rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.6, 4.3))
    for parity, col, mk in (("even", "#c0392b", "o"), ("odd", "#2e86c1", "s")):
        e = [x for x in e13_rows if x["parity"] == parity and x["positive"]]
        ax.semilogy([x["L"] for x in e], [x["certified_lower_bound"] for x in e], mk + ":", color=col, ms=3,
                    alpha=0.6, label=f"E13 pointwise envelope ({parity})")
        r = [x for x in rows if x["parity"] == parity and x["positive"]]
        ax.semilogy([x["L"] for x in r], [x["certified_lower_bound"] for x in r], mk + "-", color=col, ms=5,
                    label=f"E14 comb certificate ({parity})")
        u = [x for x in rows if x["parity"] == parity and x.get("upper")]
        ax.semilogy([x["L"] for x in u], [x["upper"]["upper"] for x in u], mk + "--", color=col, ms=4, mfc="none",
                    label=f"rigorous upper bound ({parity})")
    ax.axvline(1.0, color="0.6", lw=0.8, ls=":")
    ax.set_xlabel("window half-width L   (test functions supported in [-L, L])")
    ax.set_ylabel("inf Q(f) / ||f||^2")
    ax.set_title("Weil positivity certified on whole windows (zeta)", fontsize=10)
    ax.legend(fontsize=6.5)
    fig.tight_layout()
    os.makedirs(os.path.dirname(FIG), exist_ok=True)
    fig.savefig(FIG, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", default="")
    ap.add_argument("--merge", action="store_true", help="add/replace windows in the stored summary")
    ap.add_argument("--parts", choices=["all", "upper", "constants"], default="all")
    ap.add_argument("--no-upper", action="store_true", help="skip the rigorous upper bounds")
    args = ap.parse_args()
    old = None
    if os.path.exists(OUT_JSON):
        with open(OUT_JSON) as f:
            old = json.load(f)
    t0 = time.time()
    if args.parts == "constants":
        out = old or {"windows": []}
        out["constants"] = constants(CONSTANT_LS)
    elif args.parts == "upper":
        out = old
        for r in out["windows"]:
            J = next((w[3] for w in WINDOWS + QUICK if w[0] == r["L"]), 96)  # noqa: E501
            r["upper"] = upper_bound(r["L"], r["parity"], J, r["dps"])
            r["certified_interval"] = [r["certified_lower_bound"], r["upper"]["upper"]]
            print(f"  L={r['L']} {r['parity']}: upper {r['upper']['upper']:.6e} (J={J})", flush=True)
    else:
        windows = QUICK if args.quick else WINDOWS
        if args.only:
            keep = {float(x) for x in args.only.split(",")}
            windows = [w for w in WINDOWS + QUICK if w[0] in keep]
        rows = []
        records = {}
        for w in windows:
            L, T, dps, J, spec = w[:5]
            opts = dict(DEFAULT_OPTS, **(w[5] if len(w) > 5 else {}))
            pair = certify_pair(L, T, dps, spec, opts) if opts["paired"] else None
            for parity in ("even", "odd"):
                r = pair[parity] if pair else certify(L, T, dps, parity, spec, opts)
                records[str(L)] = certificate_record(comb_cert(L, spec), spec)
                if not args.no_upper:
                    r["upper"] = upper_bound(L, parity, J, dps)
                    r["certified_interval"] = [r["certified_lower_bound"], r["upper"]["upper"]]
                rows.append(r)
                lo = f"{r['certified_lower_bound']:.4e}" if r["positive"] else "NOT CERTIFIED"
                up = f"{r['upper']['upper']:.4e}" if r.get("upper") else "-"
                print(f"  L={L} {parity} (T={r['T']}, N={r['N']}): lambda_min in [{lo}, {up}]", flush=True)
        allrec = {}
        if os.path.exists(CERT_JSON) and args.merge:
            with open(CERT_JSON) as f:
                allrec = json.load(f)
        allrec.update(records)
        os.makedirs(os.path.dirname(CERT_JSON), exist_ok=True)
        with open(CERT_JSON, "w") as f:
            json.dump(allrec, f, indent=0)
        if args.merge and old:
            new_keys = {(r["L"], r["parity"]) for r in rows}
            rows = sorted([r for r in old["windows"] if (r["L"], r["parity"]) not in new_keys] + rows,
                          key=lambda r: (r["L"], r["parity"]))
        out = {"description": "E14: certified Weil positivity on whole windows (zeta), envelope reduction with "
                              "the non-pointwise comb certificate of rh.weil_comb",
               "quick": args.quick, "X_budget": X_BUDGET, "windows": rows,
               "constants": (old or {}).get("constants") or constants(CONSTANT_LS[:3] if args.quick else CONSTANT_LS)}
    out["seconds_last_run"] = time.time() - t0
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=1, default=str)
    e13 = []
    if os.path.exists(E13_JSON):
        with open(E13_JSON) as f:
            e13 = json.load(f)["windows"]
    if out.get("windows"):
        figure(out["windows"], e13)
    print(f"wrote {OUT_JSON} [{out['seconds_last_run']:.0f}s]")


if __name__ == "__main__":
    main()
