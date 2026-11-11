#!/usr/bin/env python3
"""E15: trajectory certificates for the prime comb, and Weil positivity on [-1.7, 1.7].

The envelope reduction (E13, E14) needs Omega(t) - P_L(t) + H(t) >= beta only for t >= T. A torus
certificate A' - P_L + H >= 0 (E14) gives this from T_1'' = 2 pi e^{A' + beta} on. Along a finite range
[T, T2], however, the phases (t log p)_p visit only part of the torus, and P_L - H stays well below A'.
`rh.weil_comb.trajectory_check` certifies the inequality on [T, T2] directly; beyond T2 = 2 pi e^{A' + beta}
the torus bound applies. This script:

  * trajectory: the smallest certified T (beta = 0.05) for the stored joint certificates of E14 (L = 1.5,
    1.6) and for the L = 1.7 certificate (Lambda = 9, ADMM after 10000 steps; Gram factor stored in
    results/json/weil_trajectory_gram_L1.7.npz), against T_1'' (Machine-checked);
  * packets: the L = 1.7 certificate plus 893 stored wave packets (rh.weil_local), certified from T = 1000
    (Machine-checked);
  * local: the local comb operator norm lambda(t0) = max over K in PW_L of int P_L(t0 + s)|K(s)|^2 / int |K|^2,
    which bounds from below what any certificate can reach near height t0 (Numerical: piecewise-constant
    k on [-L, L], a lower bound that converges as the cells shrink);
  * loss: lambda_min of the envelope form against T at L = 1.0 (Numerical). Below about
    t0 = 2 pi e^{1 + 2L} the envelope form is indefinite even though beta > 0;
  * certify (hours): Q(f) >= c ||f||^2 for every real f supported in [-1.7, 1.7], both parities, from the
    stored Gram factor, with the trajectory constant (Machine-checked; ~6 h at 195 digits, ~10 GB).

Usage: python3 scripts/run_weil_trajectory.py [--quick] [--parts trajectory,packets,local,loss] [--certify]
Writes results/json/weil_trajectory_summary.json and results/figures/weil_trajectory.png.
"""
import argparse
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import numpy as np  # noqa: E402

from rh import weil_comb as wcb  # noqa: E402
from rh import weil_envelope as we  # noqa: E402

ROOT = os.path.join(HERE, "..")
OUT_JSON = os.path.join(ROOT, "results", "json", "weil_trajectory_summary.json")
FIG = os.path.join(ROOT, "results", "figures", "weil_trajectory.png")
E14_CERTS = os.path.join(ROOT, "results", "json", "weil_comb_certificates.json")
GRAM_17 = os.path.join(ROOT, "results", "json", "weil_trajectory_gram_L1.7.npz")
BETA = 0.05
# the L = 1.7 run: T from the trajectory check, 1.1 e L T / 4 modes per sector, 195 digits
CERTIFY_17 = {"L": 1.7, "T": 2460.0, "beta": BETA, "dps": 195, "modes_factor": 1.1}


def omega_float(t):
    from scipy.special import digamma
    return np.real(digamma(0.25 + 0.5j * np.asarray(t, float))) - math.log(math.pi)


def gram_17():
    z = np.load(GRAM_17)
    meta = json.loads(str(z["meta"]))
    G = {"B": z["B"].astype(np.int64), "R": z["R"], "primes": [int(p) for p in z["primes"]]}
    G.update(meta)
    return G


def cert_17():
    """The L = 1.7 joint certificate, rebuilt exactly from the stored Gram factor (the factor is rounded to
    the grid 2^-80 and X = R^T R is formed in integers, so (*) holds by construction)."""
    return wcb.joint_comb_certificate(1.7, 9.0, gram=gram_17())


def cert_e14(L):
    """A joint certificate of E14 from its stored record (A' and the exact coefficients of H)."""
    from flint import arb, ctx
    with open(E14_CERTS) as f:
        rec = json.load(f)[str(L)]
    cert = wcb.CombCertificate(L=L, X=rec["parameter"])
    old = ctx.prec
    ctx.prec = 4000
    try:
        cert.terms = [(int(n), int(d), arb(h)) for n, d, h in rec["H_terms_n_d_h"]]
        assert all(h.rad() == 0 for _, _, h in cert.terms[:200])
    finally:
        ctx.prec = old
    cert.A_prime = arb(rec["A_prime_upper"])
    cert.A_L = sum((c for t in wcb.prime_comb(L).values() for _, c in t), arb(0))
    cert.H_sup = sum(float(abs(h).upper()) for _, _, h in cert.terms) * (1 + 1e-12)
    cert.omega_min = min(math.log(n / d) for n, d, _ in cert.terms)
    cert.meta = dict(rec.get("meta", {}))
    return cert


def last_violation(cert, beta, t_lo, step=0.01):
    """Float scan: the last t in [t_lo, T2] with Omega - P_L + H < beta (None if there is none)."""
    comb = wcb.prime_comb(cert.L)
    ns = np.array([n for t in comb.values() for n, _ in t], float)
    cs = np.array([float(c.mid()) for t in comb.values() for _, c in t])
    w = np.array([math.log(n / d) for n, d, _ in cert.terms])
    h = np.array([float(x.mid()) for _, _, x in cert.terms])
    T2 = 2 * math.pi * math.exp(float(cert.A_prime.upper()) + beta) + 1
    last, prof = None, []
    for a in np.arange(t_lo, T2, 250.0):
        ts = np.arange(a, min(a + 250.0, T2), step)
        g = np.cos(np.outer(ts, np.log(ns))) @ cs
        for i in range(0, len(w), 1000):                 # (25000 x 1000 temporaries: ~200 MB)
            g -= np.cos(np.outer(ts, w[i:i + 1000])) @ h[i:i + 1000]
        bad = np.nonzero(g > omega_float(ts) - beta)[0]
        if len(bad):
            last = float(ts[bad[-1]])
        prof.append((float(a), float(g.max())))
    return last, T2, prof


def smallest_T(cert, beta, t_lo, verbose=True):
    """The smallest T (to 0.1%) from which the trajectory check certifies beta, starting from the float scan."""
    last, T2, prof = last_violation(cert, beta, t_lo)
    T = float(math.ceil((last if last is not None else t_lo) + 1.0))
    for _ in range(12):
        cert.trajectory = None
        r = wcb.trajectory_check(cert, T, beta)
        if r["certified"]:
            break
        T = float(math.ceil(T * 1.01))
    if verbose:
        print(f"  L={cert.L}: A'={float(cert.A_prime.upper()):.4f}, T1''={2 * math.pi * math.exp(float(cert.A_prime.upper())):.0f}; "
              f"float scan: last violation {last}; certified from T={T} (max G {r['max_G']:.3f}, margin "
              f"{r['min_margin']:.2e}, {r['seconds']:.0f}s)", flush=True)
    return r, prof


def part_trajectory(quick):
    rows = []
    if quick:
        cases = [(1.2, lambda: wcb.joint_comb_certificate(1.2, 5.5), 100.0)]
    else:
        cases = [(1.5, lambda: cert_e14(1.5), 300.0), (1.6, lambda: cert_e14(1.6), 500.0), (1.7, cert_17, 800.0)]
    for L, make, t_lo in cases:
        cert = make()
        r, prof = smallest_T(cert, BETA, t_lo)
        A = float(cert.A_prime.upper())
        T1 = 2 * math.pi * math.exp(A)
        t0 = 2 * math.pi * math.exp(1 + 2 * L)
        rows.append({"L": L, "A_prime": A, "T1_torus": T1, "T_trajectory": r["T"], "beta": BETA, "T2": r["T2"],
                     "max_G_on_T_T2": r["max_G"], "max_G_t": r["max_G_t"], "min_margin": r["min_margin"],
                     "cells_per_level": r["cells_per_level"], "eval_error": r["eval_error"], "M2": r["M2"],
                     "seconds": r["seconds"], "t0_zero_matching": t0,
                     "modes_torus_1.2T1": int(math.ceil(1.1 * math.e * L * 1.2 * T1 / 4)) + 20,
                     "modes_trajectory": int(math.ceil(1.1 * math.e * L * r["T"] / 4)) + 20,
                     "profile_max_G_per_250": prof, "n_terms": len(cert.terms), "kind": cert.meta.get("kind", "joint"),
                     "Lambda": cert.meta.get("Lambda", cert.X)})
    return rows


def local_norm(L, t_max, step, h=0.025):
    """lambda(t0) on a grid of t0 (piecewise-constant k on cells of width h: a lower bound)."""
    comb = wcb.prime_comb(L)
    lam = np.array([math.log(n) for t in comb.values() for n, _ in t])
    cs = np.array([float(c.mid()) for t in comb.values() for _, c in t])
    M = int(round(2 * L / h))
    x = -L + h * (np.arange(M) + 0.5)
    D = x[:, None] - x[None, :]
    Tn = [np.maximum(0, h - np.abs(D + l)) / h for l in lam]
    ts = np.arange(0.0, t_max, step)
    out = np.empty_like(ts)
    for i, t0 in enumerate(ts):
        A = np.zeros((M, M), complex)
        for c, l, Tm in zip(cs, lam, Tn):
            e = np.exp(1j * t0 * l)
            A += (c / 2) * (e * Tm + np.conj(e) * Tm.T)
        out[i] = np.linalg.eigvalsh(A)[-1]
    return ts, out


PACKETS_17 = os.path.join(ROOT, "results", "json", "weil_trajectory_packets_L1.7.json")


def part_packets():
    """The L = 1.7 certificate plus the stored wave packets (LP on [1000, 2500]): trajectory check from T = 1000."""
    from rh import weil_local as wl
    with open(PACKETS_17) as f:
        P = json.load(f)
    loc = wl.LocalCorrection(L=P["L"], b=P["b"], m=P["m"], atoms=[tuple(x) for x in P["atoms_tau_w_kind_a"]])
    cert = wcb.attach_local(cert_17(), loc)
    r = wcb.trajectory_check(cert, P["T"], P["beta"])
    print(f"  L=1.7 with {len(loc.atoms)} wave packets: certified={r['certified']} from T={r['T']} (margin "
          f"{r['min_margin']}, error bound {r['eval_error']:.1e}, {r['seconds']:.0f}s)", flush=True)
    return {"L": 1.7, "T": r["T"], "beta": r["beta"], "certified": r["certified"], "n_packets": len(loc.atoms),
            "sum_abs_a": loc.sum_abs(), "m": loc.m, "b": loc.b, "T2": r["T2"], "min_margin": r["min_margin"],
            "eval_error": r["eval_error"], "cells_per_level": r["cells_per_level"], "seconds": r["seconds"],
            "modes": int(math.ceil(1.1 * math.e * 1.7 * r["T"] / 4)) + 20}


def part_local(quick):
    rows = []
    for L, tmax, step in ([(1.2, 600.0, 0.25)] if quick else [(1.6, 3000.0, 0.05), (1.7, 3000.0, 0.05)]):
        t = time.time()
        ts, lam = local_norm(L, tmax, step)
        om = omega_float(np.maximum(ts, 1e-9)) - BETA
        bad = np.nonzero((lam > om) & (ts > 1))[0]
        bins = [(a, b) for a, b in ((1, 100), (100, 300), (300, 1000), (1000, 2000), (2000, 3000)) if a < tmax]
        rows.append({"L": L, "lambda_0": float(lam[0]), "h": 0.025, "step": step,
                     "max_lambda": [{"range": [a, b], "max": float(lam[(ts >= a) & (ts < b)].max()),
                                     "at": float(ts[(ts >= a) & (ts < b)][np.argmax(lam[(ts >= a) & (ts < b)])])}
                                    for a, b in bins],
                     "T_local": float(ts[bad[-1]]) if len(bad) else None,
                     "series_every_10": [[float(a), float(b)] for a, b in zip(ts[::10], lam[::10])]})
        print(f"  local norm L={L}: lambda(0)={lam[0]:.4f}; local limit (lambda <= Omega - beta beyond) "
              f"T_loc={rows[-1]['T_local']}; " + ", ".join(f"[{m['range'][0]},{m['range'][1]}): {m['max']:.3f}"
                                                          for m in rows[-1]["max_lambda"]) + f" [{time.time() - t:.0f}s]",
              flush=True)
    return rows


def part_loss(quick):
    """lambda_min of the envelope form (even sector) against T at L = 1.0, joint certificate Lambda = 5, with the
    trajectory constant beta = 0.05 where it is certified (mpmath eigenvalues of the midpoint matrix)."""
    import mpmath as mp
    from flint import ctx
    L, dps = 1.0, 45
    cert = wcb.joint_comb_certificate(L, 5.0)
    rows = []
    for T in ([84.0, 120.0, 200.0] if quick else [84.0, 100.0, 110.0, 120.0, 150.0, 200.0, 300.0]):
        cert.trajectory = None
        tr = wcb.trajectory_check(cert, T, BETA)
        ctx.dps = dps
        N = int(math.ceil(1.4 * math.e * L * T / 4)) + 30
        blk = we.envelope_block(L, T, N, "even", dps=dps, comb_cert=cert, wmax=8.0)
        mp.mp.dps = dps
        M = mp.matrix([[mp.mpf(blk["R"][i, j].mid().str(dps, radius=False)) for j in range(N)] for i in range(N)])
        ev = sorted(mp.eigsy(M, eigvals_only=True))
        rows.append({"L": L, "T": T, "trajectory_certified": tr["certified"], "beta": float(blk["beta"].mid()),
                     "N": N, "lambda_min": float(ev[0]), "lambda_2": float(ev[1])})
        print(f"  envelope L={L} T={T}: beta={rows[-1]['beta']:.4f} lambda_min={float(ev[0]):.4e}", flush=True)
        ctx.dps = 15
    return {"L": L, "Lambda": 5.0, "A_prime": float(cert.A_prime.upper()),
            "T1_torus": 2 * math.pi * math.exp(float(cert.A_prime.upper())), "t0_zero_matching": 2 * math.pi * math.exp(1 + 2 * L),
            "rows": rows}


def part_certify(parities=("even", "odd")):
    """Q >= c ||f||^2 on [-1.7, 1.7] from the stored Gram factor, with the trajectory constant."""
    p = CERTIFY_17
    cert = cert_17()
    tr = wcb.trajectory_check(cert, p["T"], p["beta"], verbose=True)
    if not tr["certified"]:
        raise SystemExit("trajectory check failed")
    N = int(math.ceil(p["modes_factor"] * math.e * p["L"] * p["T"] / 4)) + 20
    res = we.certify_windows(p["L"], p["T"], {par: N for par in parities}, dps=p["dps"], verbose=True,
                             comb_cert=cert, wmax=8.0, chunk=1000)
    for r in res.values():
        r["comb_certificate"] = {k: v for k, v in cert.describe().items() if k != "per_prime"}
        r["modes_factor"] = p["modes_factor"]
    return list(res.values())


def figure(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 3, figsize=(13.5, 4.0))
    ax = axs[0]
    for row, col in zip(out.get("trajectory", []), ("#7f8c8d", "#2e86c1", "#c0392b")):
        a = np.array(row["profile_max_G_per_250"])
        ax.plot(a[:, 0] + 125, a[:, 1], "-", color=col, lw=1, label=f"L={row['L']}: max of P-H per 250")
        ax.axhline(row["A_prime"], color=col, ls=":", lw=0.8)
        ax.axvline(row["T_trajectory"], color=col, ls="--", lw=0.8)
    ts = np.linspace(200, 15000, 400)
    ax.plot(ts, omega_float(ts) - BETA, "k-", lw=1, label="Omega(t) - beta")
    ax.set_xscale("log")
    ax.set_xlabel("t")
    ax.set_title("trajectory max of P_L - H vs torus bound A' (dotted)", fontsize=9)
    ax.legend(fontsize=6)
    ax = axs[1]
    for row, col in zip(out.get("local", []), ("#2e86c1", "#c0392b")):
        a = np.array(row["series_every_10"])
        ax.plot(a[:, 0], a[:, 1], "-", color=col, lw=0.6, label=f"lambda(t0), L={row['L']}")
    ts = np.linspace(20, 3000, 300)
    ax.plot(ts, omega_float(ts) - BETA, "k-", lw=1, label="Omega(t0) - beta")
    ax.set_ylim(0, 7)
    ax.set_xlabel("t0")
    ax.set_title("local comb operator norm (Numerical)", fontsize=9)
    ax.legend(fontsize=6)
    ax = axs[2]
    loss = out.get("loss")
    if loss:
        r = loss["rows"]
        pos = [x for x in r if x["lambda_min"] > 0]
        neg = [x for x in r if x["lambda_min"] <= 0]
        ax.semilogy([x["T"] for x in pos], [x["lambda_min"] for x in pos], "o-", color="#27ae60", label="lambda_min > 0")
        ax.semilogy([x["T"] for x in neg], [-x["lambda_min"] for x in neg], "x", color="#c0392b",
                    label="-lambda_min (indefinite)")
        ax.axvline(loss["T1_torus"], color="0.5", ls=":", lw=0.8, label="T1'' (torus)")
        ax.axvline(loss["t0_zero_matching"], color="k", ls="--", lw=0.8, label="2 pi e^(1+2L)")
        ax.set_xlabel("T")
        ax.set_title(f"envelope form at L={loss['L']}: lambda_min vs T", fontsize=9)
        ax.legend(fontsize=6)
    fig.tight_layout()
    os.makedirs(os.path.dirname(FIG), exist_ok=True)
    fig.savefig(FIG, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--parts", default="trajectory,packets,local,loss")
    ap.add_argument("--certify", action="store_true", help="also run the L = 1.7 window certification (hours)")
    args = ap.parse_args()
    t0 = time.time()
    out = {}
    if os.path.exists(OUT_JSON) and not args.quick:
        with open(OUT_JSON) as f:
            out = json.load(f)
    parts = [p for p in args.parts.split(",") if p]
    if "trajectory" in parts:
        print("trajectory checks (beta = 0.05):", flush=True)
        out["trajectory"] = part_trajectory(args.quick)
    if "packets" in parts and not args.quick:
        print("wave packets at L = 1.7:", flush=True)
        out["packets"] = part_packets()
    if "local" in parts:
        print("local comb operator norm:", flush=True)
        out["local"] = part_local(args.quick)
    if "loss" in parts:
        print("envelope form against T:", flush=True)
        out["loss"] = part_loss(args.quick)
    if args.certify:
        out["windows"] = part_certify()
    out["description"] = ("E15: trajectory certificates (rh.weil_comb.trajectory_check) and Weil positivity on "
                          "[-1.7, 1.7]")
    out["quick"] = args.quick
    out["seconds_last_run"] = time.time() - t0
    if not args.quick:
        os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
        with open(OUT_JSON, "w") as f:
            json.dump(out, f, indent=1, default=str)
        figure(out)
    print(f"done [{out['seconds_last_run']:.0f}s]" + ("" if args.quick else f"; wrote {OUT_JSON}"))


if __name__ == "__main__":
    main()
