#!/usr/bin/env python3
"""E16: the envelope floor comes from the frequencies of H just above 2L; gapped certificates.

The envelope reduction (E13-E15) drops (1/pi) int_T^inf (Psi_L + H - beta) |F|^2 >= 0. E15 found that it is
indefinite below a cutoff that grows faster than t0 = 2 pi e^{1+2L} and is raised by the certificate's H.
This script shows that the H-induced part comes from a few terms h cos(t log(n/d)) with log(n/d) just above
2L (|F|^2 has spectrum in [-2L, 2L], so such a term beats slowly against it and its part on [T, inf) does not
average out), and that a certificate without them (`rh.weil_comb.gap_certificate`) restores the floor of H = 0.

Parts:
  * terms: the near-2L terms of the stored certificates (E14 for L = 1.0-1.6, E15 for L = 1.7);
  * resonance: L = 1.4, joint Lambda = 6, envelope form (even sector, 100 digits, beta = 0.05 imposed on
    [T, inf)) at T = 650 with the full H, with the terms of gap < 0.005, 0.01, 0.03, 0.1 removed, and with
    H = 0; and the floor with gap 0.03 (T = 450, 500). Numerical (probes), about 25 minutes;
  * anatomy: the negative direction at L = 1.4, T = 650 with the full H: its eigenvalue, where its spectral
    mass sits, and the parts of the dropped tail (Numerical, about 15 minutes);
  * gap17: the L = 1.7 certificate of E15 with the terms of gap < 0.002 moved into A' (the term 30/1 at
    gap 0.0012, h = 0.387, and 34 small ones), plus 14 wave packets with carriers >= 2L + 1.05 that lift the
    eight dips of Omega - G below 0.2 on [2460, 19989]; trajectory constant beta = 0.2 from T = 2460
    (Machine-checked, about 2 minutes). Packets in results/json/weil_gap_packets_L1.7.json;
  * blocks: the leading 400 x 400 block (even sector, 195 digits) of the L = 1.7 envelope form at T = 2460 for
    the E15 certificate (first negative pivot at index 233, as in the full E15 run) and for the gapped one
    (Numerical, about 30 minutes each);
  * --certify (hours): the certification of [-1.7, 1.7] with the gapped certificate at T = 2460, beta = 0.2
    (3147 modes per sector, 195 digits; ~6 h and ~10 GB; run nothing else alongside).

Usage: python3 scripts/run_weil_gap.py [--quick] [--parts terms,resonance,anatomy,gap17,blocks] [--certify]
--quick runs `terms` and `gap17` only (about 3 minutes). Writes results/json/weil_gap_summary.json.
"""
import argparse
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402

from rh import weil_comb as wcb  # noqa: E402
from rh import weil_envelope as we  # noqa: E402
from rh import weil_local as wl  # noqa: E402
import run_weil_trajectory as rwt  # noqa: E402

ROOT = os.path.join(HERE, "..")
OUT_JSON = os.path.join(ROOT, "results", "json", "weil_gap_summary.json")
PACKETS = os.path.join(ROOT, "results", "json", "weil_gap_packets_L1.7.json")
BETA = 0.05
# the gapped L = 1.7 certificate: terms with log(n/d) < 2L + g moved into A'; packets (fix_dips) at the dips of
# Omega - beta - G on [T, T2] with carriers >= 2L + b + packet_gap; trajectory constant beta from T
GAP_17 = {"L": 1.7, "g": 0.002, "packet_gap": 0.05, "T": 2460.0, "beta": 0.2, "target": 0.01, "lp_step": 0.04,
          "dips": [2599.2, 2630.98, 2819.14, 2931.62, 2974.12, 3099.96, 3517.36, 3689.14],
          "dps": 195, "modes_factor": 1.1}


def omega_float(t):
    from scipy.special import digamma
    return np.real(digamma(0.25 + 0.5j * np.asarray(t, float))) - math.log(math.pi)


def near_terms(cert, gaps=(0.005, 0.01, 0.03)):
    L = cert.L
    rows = sorted(((math.log(n / d) - 2 * L, n, d, float(h.mid())) for n, d, h in cert.terms))
    big = sorted([r for r in rows if r[0] < 0.03], key=lambda r: -abs(r[3]))[:3]
    return {"L": L, "A_prime": float(cert.A_prime.upper()), "sum_abs_h": sum(abs(r[3]) for r in rows),
            "sum_abs_h_gap_below": {str(g): sum(abs(r[3]) for r in rows if r[0] < g) for g in gaps},
            "largest": [{"n_d": f"{n}/{d}", "gap": g, "h": h} for g, n, d, h in big]}


def part_terms():
    rows = []
    for L in (1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6):
        rows.append(dict(near_terms(rwt.cert_e14(L)), source="E14"))
    rows.append(dict(near_terms(rwt.cert_17()), source="E15 (Lambda = 9, ADMM)"))
    for r in rows:
        print(f"  L={r['L']} ({r['source']}): sum|h| = {r['sum_abs_h']:.1f}; within 0.005 / 0.01 / 0.03 of 2L: "
              + " / ".join(f"{v:.3f}" for v in r["sum_abs_h_gap_below"].values())
              + ("; largest " + ", ".join(f"{x['n_d']} (gap {x['gap']:.4f}, h {x['h']:+.3f})" for x in r["largest"])
                 if r["largest"] else ""), flush=True)
    return rows


def probe(L, cert, T, dps):
    """Even-sector envelope form with beta = 0.05 imposed on [T, inf) (rwt.floor_probe)."""
    t = time.time()
    r = rwt.floor_probe(L, cert, T, dps)
    r["seconds"] = time.time() - t
    return r


def part_resonance():
    L, Lam, dps = 1.4, 6.0, 100
    full = wcb.joint_comb_certificate(L, Lam)
    t0 = 2 * math.pi * math.exp(1 + 2 * L)
    rows = []
    cases = [("full H", None, 650.0)] + [(f"gap {g}", g, 650.0) for g in (0.005, 0.01, 0.03, 0.1)] + \
            [("H = 0", 99.0, 650.0), ("gap 0.03", 0.03, 450.0), ("gap 0.03", 0.03, 500.0)]
    for name, g, T in cases:
        cert = full if g is None else wcb.gap_certificate(full, g)
        r = probe(L, cert, T, dps)
        r.update({"case": name, "L": L, "Lambda": Lam, "gap": g, "n_terms": len(cert.terms), "sum_abs_h": cert.H_sup,
                  "T_over_t0": T / t0})
        rows.append(r)
        print(f"  L=1.4 {name:9s} T={T:.0f} ({T / t0:.2f} t0), {len(cert.terms)} terms, sum|h|={cert.H_sup:.2f}: "
              + (f"lambda_min ~ {r['lambda_min']:.4e}" if r["positive"] else
                 f"indefinite ({r['negative_eigenvalues']} negative)") + f" [{r['seconds']:.0f}s]", flush=True)
    return rows


def part_anatomy():
    """Negative direction of the envelope form at L = 1.4, T = 650 (full H): eigenvalue by bisection on the
    inertia and inverse iteration, profile |F(t)|^2, and the dropped tail split into Psi - beta, the term 33/2
    and the rest of H."""
    from flint import arb, arb_mat, ctx
    from scipy.special import digamma
    L, T, dps, N = 1.4, 650.0, 100, 896
    full = wcb.joint_comb_certificate(L, 6.0)
    full.trajectory = {"certified": True, "T": T, "beta": BETA}
    t = time.time()
    ctx.dps = dps
    blk = we.envelope_blocks(L, T, {"even": N}, dps=dps, comb_cert=full, wmax=8.0)["even"]
    R = blk["R"]
    est = we.lambda_min_estimate(R, lo_exp=-(dps - 5), hi_exp=0, digits=0.1)
    f = we.ldl_mid_blocked(R, arb(1.3 * est["lambda_min"]))
    Lm, d = f["L"], f["d"]
    del f
    n = R.nrows()
    x = [arb(1) / (1 + i) for i in range(n)]
    for _ in range(8):                         # inverse iteration with the shift below lambda_min
        z = [None] * n
        for i in range(n):
            acc = x[i]
            for j in range(i):
                acc -= Lm[i, j] * z[j]
            z[i] = acc.mid()
        w = [(z[i] / d[i]).mid() for i in range(n)]
        y = [None] * n
        for i in range(n - 1, -1, -1):
            acc = w[i]
            for j in range(i + 1, n):
                acc -= Lm[j, i] * y[j]
            y[i] = acc.mid()
        nrm = sum((v * v for v in y), arb(0)).sqrt()
        x = [(v / nrm).mid() for v in y]
    Rx = R * arb_mat([[v] for v in x])
    lam = float(sum((x[i] * Rx[i, 0] for i in range(n)), arb(0)).mid())
    orders = blk["orders"]
    LL = arb(L)
    scale = [((2 * LL * (2 * o + 1)).sqrt()) * (1 if (o // 2) % 2 == 0 else -1) for o in orders]
    ts = np.concatenate([np.arange(0.05, T - 60, 0.5), np.arange(T - 60, T + 400, 0.05)])
    F2 = np.empty(len(ts))
    for k, tt in enumerate(ts):
        js = we.sph_bessel_j_all(LL * arb(float(tt)), orders[-1], check=False)
        F = sum((x[i] * scale[i] * js[orders[i]] for i in range(n)), arb(0))
        F2[k] = float((F * F).mid())
    ctx.dps = 15
    Om = np.real(digamma(0.25 + 0.5j * ts)) - math.log(math.pi)
    comb = wcb.prime_comb(L)
    P = sum(float(c.mid()) * np.cos(ts * math.log(nn)) for tt in comb.values() for nn, c in tt)
    Hf = sum(float(h.mid()) * np.cos(ts * math.log(a / b)) for a, b, h in full.terms)
    Hres = sum(float(h.mid()) * np.cos(ts * math.log(a / b)) for a, b, h in full.terms if math.log(a / b) < 2 * L + 0.005)
    dt = np.diff(ts, append=ts[-1])
    above = ts >= T
    out = {"L": L, "T": T, "N": N, "dps": dps, "lambda": lam, "lambda_bisection": est["lambda_min"],
           "mass_below_T": float(np.sum((F2 * dt)[~above])) / math.pi,
           "mass_above_T": float(np.sum((F2 * dt)[above])) / math.pi,
           "tail_Psi_minus_beta": float(np.sum(((Om - P - BETA) * F2 * dt)[above])) / math.pi,
           "tail_H_33_2": float(np.sum((Hres * F2 * dt)[above])) / math.pi,
           "tail_H_rest": float(np.sum(((Hf - Hres) * F2 * dt)[above])) / math.pi,
           "F2_peak_t": float(ts[np.argmax(F2)]), "seconds": time.time() - t,
           "profile_every_20": [[float(a), float(b)] for a, b in zip(ts[::20], F2[::20])]}
    print(f"  anatomy L=1.4 T=650: lambda = {lam:.3e}; mass below/above T = {out['mass_below_T']:.3e} / "
          f"{out['mass_above_T']:.3e}; dropped tail: Psi - beta {out['tail_Psi_minus_beta']:+.3e}, 33/2 "
          f"{out['tail_H_33_2']:+.3e}, rest of H {out['tail_H_rest']:+.3e} [{out['seconds']:.0f}s]", flush=True)
    return out


def gapped_17():
    """The gapped L = 1.7 certificate with its 3 wave packets (from PACKETS if present, else fitted)."""
    p = GAP_17
    cert = wcb.gap_certificate(rwt.cert_17(), p["g"])
    if os.path.exists(PACKETS):
        with open(PACKETS) as f:
            P = json.load(f)
        loc = wl.LocalCorrection(L=P["L"], b=P["b"], m=P["m"], atoms=[tuple(a) for a in P["atoms_tau_w_kind_a"]])
    else:
        comb = wcb.prime_comb(p["L"])
        ns = np.array([n for t in comb.values() for n, _ in t], float)
        cs = np.array([float(c.mid()) for t in comb.values() for _, c in t])
        w = np.array([math.log(n / d) for n, d, _ in cert.terms])
        h = np.array([float(x.mid()) for _, _, x in cert.terms])

        def G(ts):
            out = np.empty(len(ts))
            for i in range(0, len(ts), 1500):
                t = ts[i:i + 1500]
                out[i:i + 1500] = np.cos(np.outer(t, np.log(ns))) @ cs - np.cos(np.outer(t, w)) @ h
            return out
        loc = wl.fix_dips(G, omega_float, p["L"], p["T"], p["dips"], p["beta"], gap=p["packet_gap"],
                          target=p["target"], step=p["lp_step"])
        with open(PACKETS, "w") as f:
            json.dump({"L": loc.L, "b": loc.b, "m": loc.m, "g": p["g"], "packet_gap": p["packet_gap"], "beta": p["beta"],
                       "T": p["T"], "dips": p["dips"], "atoms_tau_w_kind_a": [list(a) for a in loc.atoms]}, f, indent=1)
    return wcb.attach_local(cert, loc)


def part_gap17():
    p = GAP_17
    cert = gapped_17()
    r = wcb.trajectory_check(cert, p["T"], p["beta"])
    print(f"  L=1.7 gapped (g = {p['g']}: {cert.meta['n_moved']} terms, sum|h| = {cert.meta['moved_to_A_prime']:.3f} "
          f"moved; A' = {float(cert.A_prime.upper()):.4f}) + {len(cert.local.atoms)} packets (sum|a| = "
          f"{cert.local.sum_abs():.3f}): beta = {r['beta']} certified={r['certified']} from T={r['T']} (margin {r['min_margin']}, "
          f"T2 = {r['T2']:.0f}, {r['seconds']:.0f}s)", flush=True)
    return {"L": 1.7, "g": p["g"], "n_moved": cert.meta["n_moved"], "moved_to_A_prime": cert.meta["moved_to_A_prime"],
            "A_prime": float(cert.A_prime.upper()), "omega_min_minus_2L": cert.omega_min - 3.4,
            "packets": [list(a) for a in cert.local.atoms], "sum_abs_a": cert.local.sum_abs(),
            "trajectory": {k: r.get(k) for k in ("certified", "T", "beta", "T2", "min_margin", "worst_t", "eval_error",
                                                 "cells_per_level", "seconds")}}


def leading_block(cert, T, Np, parity, dps):
    from flint import ctx
    t = time.time()
    ctx.dps = dps
    blk = we.envelope_block(1.7, T, Np, parity, dps=dps, comb_cert=cert, wmax=8.0, chunk=1000)
    f = we.ldl_mid_blocked(blk["R"], None)
    ctx.dps = 15
    return {"T": T, "N_prime": Np, "parity": parity, "dps": dps, "n_negative": f["n_negative"],
            "first_negative": f["first_negative"], "min_pivot": float(f["min_pivot"].mid()),
            "n_nodes": blk["n_nodes"], "seconds": time.time() - t}


def part_blocks():
    rows = []
    for name, cert, beta in (("E15 certificate", rwt.cert_17(), BETA), ("gapped + packets", gapped_17(), GAP_17["beta"])):
        tr = wcb.trajectory_check(cert, GAP_17["T"], beta)
        assert tr["certified"]
        r = leading_block(cert, GAP_17["T"], 400, "even", GAP_17["dps"])
        r["certificate"], r["beta"] = name, beta
        rows.append(r)
        print(f"  L=1.7 T=2460 leading 400 x 400 block (even, {GAP_17['dps']} digits), {name}: "
              f"{r['n_negative']} negative pivots (first at {r['first_negative']}), min pivot {r['min_pivot']:.3e} "
              f"[{r['seconds']:.0f}s]", flush=True)
    return rows


def part_certify(parities=("even", "odd")):
    p = GAP_17
    cert = gapped_17()
    tr = wcb.trajectory_check(cert, p["T"], p["beta"], verbose=True)
    if not tr["certified"]:
        raise SystemExit("trajectory check failed")
    N = int(math.ceil(p["modes_factor"] * math.e * p["L"] * p["T"] / 4)) + 20
    res = we.certify_windows(p["L"], p["T"], {par: N for par in parities}, dps=p["dps"], verbose=True,
                             comb_cert=cert, wmax=8.0, chunk=1000)
    for r in res.values():
        r["comb_certificate"] = {k: v for k, v in cert.describe().items() if k != "per_prime"}
        r["modes_factor"] = p["modes_factor"]
        r["setup"] = (f"E15 certificate, gap {p['g']} moved into A', {len(cert.local.atoms)} packets, T = {p['T']}, "
                      f"beta = {p['beta']}")
    return list(res.values())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--parts", default="terms,resonance,anatomy,gap17,blocks")
    ap.add_argument("--certify", action="store_true", help="also run the L = 1.7 window certification (hours)")
    args = ap.parse_args()
    t0 = time.time()
    out = {}
    if os.path.exists(OUT_JSON):
        with open(OUT_JSON) as f:
            out = json.load(f)
    parts = ["terms", "gap17"] if args.quick else args.parts.split(",")
    fns = {"terms": part_terms, "resonance": part_resonance, "anatomy": part_anatomy, "gap17": part_gap17,
           "blocks": part_blocks}
    for name in parts:
        print(f"[{name}]", flush=True)
        res = fns[name]()
        if not args.quick:
            out[name] = res
    if args.certify:
        print("[certify]", flush=True)
        out["certify_L17"] = part_certify()
    out["description"] = __doc__.split("\n")[0]
    if not args.quick:
        out["seconds_last_run"] = time.time() - t0
        with open(OUT_JSON, "w") as f:
            json.dump(out, f, indent=1, default=str)
    print(f"done [{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()
