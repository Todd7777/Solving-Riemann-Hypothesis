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
    H = 0; the floor with gap 0.03 (T = 450, 500); the full H with beta = 0.1 and 0.2. Numerical (probes), about
    30 minutes;
  * anatomy: the negative direction at L = 1.4, T = 650 with the full H, decomposed exactly with the matrices
    (mass above T, the term 33/2, the rest of H), the threshold beta* for the full H, without 33/2 and for
    H = 0, and the profile of |F|^2 above T (Numerical, about 30 minutes);
  * floor_beta: the threshold beta*(T) at L = 1.4 for T = 400, 500, 650 (full H, gap 0.03, H = 0; Numerical,
    about 45 minutes);
  * gap17: the L = 1.7 certificate of E15 with the terms of gap < 0.002 moved into A' (the term 30/1 at
    gap 0.0012, h = 0.387, and 34 small ones), plus wave packets with carriers >= 2L + 1.05 at the dips of
    Omega - beta - G on [2460, T2]: 3 packets for beta = 0.05 and 14 for beta = 0.2; trajectory constants from
    T = 2460 (Machine-checked, about 2 minutes). Packets in results/json/weil_gap_packets_L1.7*.json;
  * blocks: the leading 400 x 400 block (even sector, 195 digits, beta = 0.05) of the L = 1.7 envelope form at
    T = 2460 for the E15 certificate (first negative pivot at index 233, as in the full E15 run) and for the
    gapped one with its 3 packets (Numerical, about 2 hours each);
  * --certify (hours): the certification of [-1.7, 1.7] with the gapped certificate and its 14 packets at
    T = 2460, beta = 0.2 (3147 modes per sector, 195 digits; ~6 h and ~10 GB; run nothing else alongside).

Usage: python3 scripts/run_weil_gap.py [--quick] [--parts terms,resonance,anatomy,floor_beta,gap17,blocks] [--certify]
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
PACKETS = {0.2: os.path.join(ROOT, "results", "json", "weil_gap_packets_L1.7.json"),
           0.05: os.path.join(ROOT, "results", "json", "weil_gap_packets_L1.7_beta0.05.json")}
BETA = 0.05
# the gapped L = 1.7 certificate: terms with log(n/d) < 2L + g moved into A'; packets (fix_dips) at the dips of
# Omega - beta - G on [T, T2] with carriers >= 2L + b + packet_gap; trajectory constant beta from T. Two packet
# sets: beta = 0.2 (14 packets, used for the certification) and beta = 0.05 (3 packets, the leading-block probe)
GAP_17 = {"L": 1.7, "g": 0.002, "packet_gap": 0.05, "T": 2460.0, "beta": 0.2, "target": 0.01, "lp_step": 0.04,
          "dips": {0.2: [2599.2, 2630.98, 2819.14, 2931.62, 2974.12, 3099.96, 3517.36, 3689.14],
                   0.05: [2819.14, 3099.96, 3689.14]},
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


def probe(L, cert, T, dps, beta=BETA):
    """Even-sector envelope form with beta imposed on [T, inf), as rwt.floor_probe (which uses beta = 0.05)."""
    from flint import ctx
    t = time.time()
    cert.trajectory = {"certified": True, "T": T, "beta": beta}
    ctx.dps = dps
    N = int(math.ceil(1.4 * math.e * L * T / 4)) + 30
    blk = we.envelope_block(L, T, N, "even", dps=dps, comb_cert=cert, wmax=8.0)
    est = we.approx_lambda_min_ldl(blk["R"])
    r = {"T": T, "N": N, "beta": beta, "positive": est is not None,
         "lambda_min": float(est[0].mid()) if est is not None else None,
         "negative_eigenvalues": 0 if est is not None else we.inertia_count(blk["R"], None)}
    ctx.dps = 15
    cert.trajectory = None
    r["seconds"] = time.time() - t
    return r


def part_resonance():
    L, Lam, dps = 1.4, 6.0, 100
    full = wcb.joint_comb_certificate(L, Lam)
    t0 = 2 * math.pi * math.exp(1 + 2 * L)
    rows = []
    cases = [("full H", None, 650.0, BETA)] + [(f"gap {g}", g, 650.0, BETA) for g in (0.005, 0.01, 0.03, 0.1)] + \
            [("H = 0", 99.0, 650.0, BETA), ("gap 0.03", 0.03, 450.0, BETA), ("gap 0.03", 0.03, 500.0, BETA),
             ("full H", None, 650.0, 0.1), ("full H", None, 650.0, 0.2)]
    for name, g, T, beta in cases:
        cert = full if g is None else wcb.gap_certificate(full, g)
        r = probe(L, cert, T, dps, beta)
        r.update({"case": name, "L": L, "Lambda": Lam, "gap": g, "n_terms": len(cert.terms), "sum_abs_h": cert.H_sup,
                  "T_over_t0": T / t0})
        rows.append(r)
        print(f"  L=1.4 {name:9s} T={T:.0f} ({T / t0:.2f} t0) beta={beta}, {len(cert.terms)} terms, sum|h|={cert.H_sup:.2f}: "
              + (f"lambda_min ~ {r['lambda_min']:.4e}" if r["positive"] else
                 f"indefinite ({r['negative_eigenvalues']} negative)") + f" [{r['seconds']:.0f}s]", flush=True)
    return rows


def _even_form(L, T, N, dps, cert, beta):
    """Even-sector envelope matrix R(beta) with beta imposed on [T, inf) (a probe: beta is not checked)."""
    from flint import ctx
    cert.trajectory = {"certified": True, "T": T, "beta": beta}
    ctx.dps = dps
    blk = we.envelope_block(L, T, N, "even", dps=dps, comb_cert=cert, wmax=8.0)
    cert.trajectory = None
    assert abs(float(blk["beta"].mid()) - beta) < 1e-12, blk["beta"]
    return blk


def _quad(A, x):
    """x^T A x (arb) for an arb_mat A and a list x."""
    from flint import arb, arb_mat
    Ax = A * arb_mat([[v] for v in x])
    return sum((x[i] * Ax[i, 0] for i in range(len(x))), arb(0))


def _threshold(R, K, beta0, lo=-0.3, hi=0.5, tol=0.005):
    """Smallest beta (to tol) for which R + (beta - beta0) K is positive definite: bisection on the inertia of the
    midpoint LDL^T (Numerical).  R + (beta - beta0) K is the envelope form with beta in place of beta0."""
    from flint import arb

    def pos(b):
        return we.ldl_mid_blocked(R + K * arb(b - beta0), None)["n_negative"] == 0
    if pos(lo):
        return {"beta_star": None, "below": lo}
    if not pos(hi):
        return {"beta_star": None, "above": hi}
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        lo, hi = (lo, mid) if pos(mid) else (mid, hi)
    return {"beta_star": 0.5 * (lo + hi), "bracket": [lo, hi]}


def part_anatomy():
    """L = 1.4, T = 650, joint Lambda = 6, even sector, 896 modes, 100 digits (Numerical).
    * Exact decomposition of the negative direction x of the envelope form R(0.05) with the full H. Since
      R(beta) = R(0) + beta K with K = (1/pi) int_T^inf G_m G_n (the mass above T), K = (R(0.15) - R(0.05)) / 0.1,
      and the part of H on [0, T] is R_full - R_{H=0} (the term 33/2: R_full - R_{without 33/2}); all from the
      matrices, no truncation.
    * The threshold beta* above which the envelope form is positive definite, for the full H, without 33/2 and
      for H = 0.
    * The profile |F(t)|^2 of x on [T - 60, T + 400]: the local mean and the amplitude of its frequency-2L part
      (least squares on windows of width 20)."""
    from flint import arb, ctx
    L, T, dps, N = 1.4, 650.0, 100, 896
    t = time.time()
    full = wcb.joint_comb_certificate(L, 6.0)
    no33 = wcb.gap_certificate(full, 0.005)            # removes 33/2 only (gap 0.0034)
    zero = wcb.gap_certificate(full, 99.0)             # H = 0
    h33 = [float(h.mid()) for n, d, h in full.terms if (n, d) == (33, 2)][0]
    blk = _even_form(L, T, N, dps, full, BETA)
    R = {"full": blk["R"], "no33": _even_form(L, T, N, dps, no33, BETA)["R"],
         "zero": _even_form(L, T, N, dps, zero, BETA)["R"]}
    K = (_even_form(L, T, N, dps, zero, BETA + 0.1)["R"] - R["zero"]) * arb(10)
    print(f"  anatomy: 4 forms assembled [{time.time() - t:.0f}s]", flush=True)
    ctx.dps = dps
    est = we.lambda_min_estimate(R["full"], lo_exp=-(dps - 5), hi_exp=0, digits=0.1)
    f = we.ldl_mid_blocked(R["full"], arb(1.3 * est["lambda_min"]))
    Lm, d = f["L"], f["d"]
    del f
    n = N
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
    del Lm, d
    lam = float(_quad(R["full"], x).mid())
    mass = float(_quad(K, x).mid())
    H_part = float(_quad(R["full"] - R["zero"], x).mid())      # (1/pi) int_0^T H |F|^2 = -(the tail of H)
    p33 = float(_quad(R["full"] - R["no33"], x).mid())
    Qt0 = float(_quad(R["zero"], x).mid()) - BETA * mass     # pole + (1/pi) int_0^T Psi |F|^2
    out = {"L": L, "T": T, "N": N, "dps": dps, "beta": BETA, "lambda": lam, "lambda_bisection": est["lambda_min"],
           "mass_above_T": mass, "beta_credit": BETA * mass, "H_on_0_T": H_part, "term_33_2_on_0_T": p33,
           "rest_of_H_on_0_T": H_part - p33, "pole_plus_Psi_on_0_T": Qt0, "h_33_2": h33,
           "term_33_2_share": -p33 / (abs(h33) * mass), "rest_share": -(H_part - p33) / ((full.H_sup - abs(h33)) * mass)}
    print(f"  anatomy L=1.4 T=650 (full H, beta={BETA}): lambda = {lam:.4e}; mass above T = {mass:.4e}, beta credit "
          f"{BETA * mass:.3e}; on [0, T]: pole + Psi {Qt0:+.3e}, 33/2 {p33:+.3e} ({out['term_33_2_share']:.3f} of |h| x mass), "
          f"rest of H {H_part - p33:+.3e} ({out['rest_share']:.2e} of its sum|h| x mass) [{time.time() - t:.0f}s]", flush=True)
    out["thresholds"] = {}
    for name in ("full", "no33", "zero"):
        out["thresholds"][name] = _threshold(R[name], K, BETA)
        print(f"  threshold beta* ({name}): {out['thresholds'][name]} [{time.time() - t:.0f}s]", flush=True)
    # profile of x above T
    orders = blk["orders"]
    LL = arb(L)
    ctx.dps = dps
    scale = [((2 * LL * (2 * o + 1)).sqrt()) * (1 if (o // 2) % 2 == 0 else -1) for o in orders]
    ts = np.arange(T - 60, T + 400, 0.05)
    F2 = np.empty(len(ts))
    for k, tt in enumerate(ts):
        js = we.sph_bessel_j_all(LL * arb(float(tt)), orders[-1], check=False)
        F = sum((x[i] * scale[i] * js[orders[i]] for i in range(n)), arb(0))
        F2[k] = float((F * F).mid())
    ctx.dps = 15
    win = []
    for a in np.arange(T, T + 400, 40.0):
        s = (ts >= a) & (ts < a + 20)
        M = np.stack([np.ones(s.sum()), np.cos(2 * L * ts[s]), np.sin(2 * L * ts[s])], 1)
        c = np.linalg.lstsq(M, F2[s], rcond=None)[0]
        win.append({"t": float(a), "mean": float(c[0]), "amp_2L": float(math.hypot(c[1], c[2]))})
    out["profile_windows"] = win
    out["F2_at_T"] = float(F2[np.argmin(abs(ts - T))])
    out["seconds"] = time.time() - t
    print("  profile above T (mean, amplitude of the 2L part): " + ", ".join(
        f"{w['t']:.0f}: {w['mean']:.2e}/{w['amp_2L']:.2e}" for w in win) + f" [{out['seconds']:.0f}s]", flush=True)
    return out


def part_floor_beta():
    """L = 1.4 (joint Lambda = 6), even sector, 100 digits: the threshold beta*(T) above which the envelope form is
    positive definite, for the full H, without the terms of gap < 0.03, and for H = 0 (Numerical). With K the mass
    above T (as in `part_anatomy`), the form at beta is R(0.05) + (beta - 0.05) K."""
    from flint import arb
    L, dps = 1.4, 100
    full = wcb.joint_comb_certificate(L, 6.0)
    certs = {"full": full, "gap 0.03": wcb.gap_certificate(full, 0.03), "zero": wcb.gap_certificate(full, 99.0)}
    t0 = 2 * math.pi * math.exp(1 + 2 * L)
    rows = []
    for T in (400.0, 500.0, 650.0):
        t = time.time()
        N = int(math.ceil(1.4 * math.e * L * T / 4)) + 30
        Rz = _even_form(L, T, N, dps, certs["zero"], BETA)["R"]
        K = (_even_form(L, T, N, dps, certs["zero"], BETA + 0.1)["R"] - Rz) * arb(10)
        row = {"L": L, "T": T, "T_over_t0": T / t0, "N": N, "dps": dps}
        for name, cert in certs.items():
            R = Rz if name == "zero" else _even_form(L, T, N, dps, cert, BETA)["R"]
            row[name] = _threshold(R, K, BETA, lo=-0.5, hi=1.0)
        row["seconds"] = time.time() - t
        rows.append(row)
        print(f"  L=1.4 T={T:.0f} ({T / t0:.2f} t0), N={N}: beta* full {row['full']}, gap 0.03 {row['gap 0.03']}, "
              f"H = 0 {row['zero']} [{row['seconds']:.0f}s]", flush=True)
    return rows


def gapped_17(beta=None):
    """The gapped L = 1.7 certificate with the wave packets for trajectory constant beta (default GAP_17["beta"];
    from PACKETS[beta] if present, else fitted with fix_dips; the fit is deterministic up to the LP solver)."""
    p = GAP_17
    beta = p["beta"] if beta is None else beta
    cert = wcb.gap_certificate(rwt.cert_17(), p["g"])
    if os.path.exists(PACKETS[beta]):
        with open(PACKETS[beta]) as f:
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
        loc = wl.fix_dips(G, omega_float, p["L"], p["T"], p["dips"][beta], beta, gap=p["packet_gap"],
                          target=p["target"], step=p["lp_step"])
        with open(PACKETS[beta], "w") as f:
            json.dump({"L": loc.L, "b": loc.b, "m": loc.m, "g": p["g"], "packet_gap": p["packet_gap"], "beta": beta,
                       "T": p["T"], "dips": p["dips"][beta], "atoms_tau_w_kind_a": [list(a) for a in loc.atoms]}, f,
                      indent=1)
    return wcb.attach_local(cert, loc)


def part_gap17():
    p = GAP_17
    rows = []
    for beta in (0.05, 0.2):
        cert = gapped_17(beta)
        r = wcb.trajectory_check(cert, p["T"], beta)
        print(f"  L=1.7 gapped (g = {p['g']}: {cert.meta['n_moved']} terms, sum|h| = {cert.meta['moved_to_A_prime']:.3f} "
              f"moved; A' = {float(cert.A_prime.upper()):.4f}) + {len(cert.local.atoms)} packets (sum|a| = "
              f"{cert.local.sum_abs():.3f}): beta = {r['beta']} certified={r['certified']} from T={r['T']} (margin "
              f"{r['min_margin']:.4f}, T2 = {r['T2']:.0f}, {r['seconds']:.0f}s)", flush=True)
        rows.append({"L": 1.7, "g": p["g"], "n_moved": cert.meta["n_moved"],
                     "moved_to_A_prime": cert.meta["moved_to_A_prime"], "A_prime": float(cert.A_prime.upper()),
                     "omega_min_minus_2L": cert.omega_min - 3.4, "packets": [list(a) for a in cert.local.atoms],
                     "sum_abs_a": cert.local.sum_abs(),
                     "trajectory": {k: r.get(k) for k in ("certified", "T", "beta", "T2", "min_margin", "worst_t",
                                                          "eval_error", "cells_per_level", "seconds")}})
    return rows


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
    """Leading 400 x 400 block of the even sector at T = 2460, beta = 0.05: the E15 certificate against the gapped
    one with its 3 packets (about 2 hours each)."""
    rows = []
    for name, cert in (("E15 certificate", rwt.cert_17()), ("gapped, 3 packets", gapped_17(0.05))):
        tr = wcb.trajectory_check(cert, GAP_17["T"], BETA)
        assert tr["certified"]
        r = leading_block(cert, GAP_17["T"], 400, "even", GAP_17["dps"])
        r["certificate"], r["beta"] = name, BETA
        rows.append(r)
        print(f"  L=1.7 T=2460 beta={BETA} leading 400 x 400 block (even, {GAP_17['dps']} digits), {name}: "
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
    ap.add_argument("--parts", default="terms,resonance,anatomy,floor_beta,gap17,blocks")
    ap.add_argument("--certify", action="store_true", help="also run the L = 1.7 window certification (hours)")
    args = ap.parse_args()
    t0 = time.time()
    out = {}
    if os.path.exists(OUT_JSON):
        with open(OUT_JSON) as f:
            out = json.load(f)
    parts = ["terms", "gap17"] if args.quick else args.parts.split(",")
    fns = {"terms": part_terms, "resonance": part_resonance, "anatomy": part_anatomy, "floor_beta": part_floor_beta,
           "gap17": part_gap17, "blocks": part_blocks}
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
