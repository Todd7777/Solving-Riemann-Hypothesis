#!/usr/bin/env python3
"""E17: lifting the comb along the trajectory from T = 1000 at L = 1.7.

E16 certified [-1.7, 1.7] at T = 2460 with a gapped certificate and found the envelope form already positive at
T = 1000 (Numerical): what a cheaper certificate lacks is the comb bound Omega - beta - P_L + H >= 0 on
[1000, 2460], where the margin has 54 dips below 0. This script shows that every dip can be lifted by wave
packets in principle (a necessary condition from the band limit, checked at each dip) and in practice (once the
packets' carriers reach well above 2L), fits the packets and checks the trajectory constant rigorously.

Parts:
  * scan: the local minima of the margin r = Omega - 0.2 - P_L + H below 0.03 on [1000, T2] for the gapped L = 1.7
    certificate of E16 (no packets; values from the certified evaluator's midpoints), about 2 minutes;
  * liftable: for each dip d, lambda(d) = min of int r |p|^2 / int |p|^2 over p in the span of the 25 shifted sincs
    sin(Omega (t - t_k)) / (Omega (t - t_k)), |t_k - d| <= 12 pi / Omega, Omega = (2L + gap) / 2 (Numerical,
    quadrature on a grid of 0.02 over d +- 120). Packets with spectrum in |xi| >= 2L + gap are orthogonal to
    every such |p|^2, so lambda(d) < target means that no packets can lift the dip (docs/06, E17). About 15
    minutes;
  * cluster: the packet program for the nine dips on [1058, 1144] alone (its own points only), with carriers up to
    2L + b + gap + W for W = 4, 8, 12, amplitude caps 4 and 16, and narrower envelopes (b = 2). About 5 minutes;
  * fit: `rh.weil_local.fix_dips` on all dips (W = 8, centres within 8 of each dip, grid 0.03 within 25 of the
    centres and 0.15 beyond, a buffer of 0.06 on each program's own points), then `trajectory_check` from
    T = 1000 with beta = 0.2 (Machine-checked). About 5 minutes. The stored packets
    (results/json/weil_lift_packets_L1.7_T1000.json) are those of the certification run; the part refits them
    and compares (the LP solver reproduces them to about 1e-12), then checks the stored ones;
  * --certify (about an hour, ~5 GB): the certification of [-1.7, 1.7] at T = 1000 with these packets
    (1361 modes per sector, 195 digits; with 1291 the tail bound eps_B ~ (eLT/4N)^{2N} is ~1e-124, too large).

Usage: python3 scripts/run_weil_lift.py [--quick] [--parts scan,liftable,cluster,fit] [--certify]
--quick checks the trajectory constant of the stored packets only (about a minute; part of run_all.py --quick). Writes
results/json/weil_lift_summary.json.
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
OUT_JSON = os.path.join(ROOT, "results", "json", "weil_lift_summary.json")
PACKETS = os.path.join(ROOT, "results", "json", "weil_lift_packets_L1.7_T1000.json")
P = {"L": 1.7, "g": 0.002, "beta": 0.2, "T": 1000.0, "gap": 0.05, "thr": 0.03,
     "fit": {"W": 8.0, "amax": 16.0, "half_width": 8.0, "step": 0.03, "far_step": 0.15, "core_pad": 25.0,
             "target": 0.02, "buffer": 0.06, "max_group": 2, "dw": 0.5, "dtau": 1.0},
     "dps": 195, "modes_factor": 1.16}      # 1.1 leaves eps_B ~ 1e-124 at T = 1000 (above lambda_min); 1.16: ~1e-192
CLUSTER = (1058.0, 1145.0)


def omega_float(t):
    from scipy.special import digamma
    return np.real(digamma(0.25 + 0.5j * np.asarray(t, float))) - math.log(math.pi)


def gapped():
    return wcb.gap_certificate(rwt.cert_17(), P["g"])


def G_function(cert):
    """P_L - H in floats (H without packets), vectorised in chunks."""
    comb = wcb.prime_comb(P["L"])
    ns = np.array([n for t in comb.values() for n, _ in t], float)
    cs = np.array([float(c.mid()) for t in comb.values() for _, c in t])
    w = np.array([math.log(n / d) for n, d, _ in cert.terms])
    h = np.array([float(x.mid()) for _, _, x in cert.terms])

    def G(ts):
        ts = np.asarray(ts, float)
        out = np.empty(len(ts))
        for i in range(0, len(ts), 1000):
            t = ts[i:i + 1000]
            out[i:i + 1000] = np.cos(np.outer(t, np.log(ns))) @ cs - np.cos(np.outer(t, w)) @ h
        return out
    return G


def part_scan(cert=None):
    """Local minima of Omega - beta - G below thr on [T, T2] (as scan in E16), merged within 3."""
    from flint import arb, ctx
    cert = gapped() if cert is None else cert
    T, beta, thr = P["T"], P["beta"], P["thr"]
    T2 = 2 * math.pi * math.exp(float(cert.A_prime.upper()) + beta) + 1
    ev = wcb._TrajectoryEvaluator(cert)
    t0 = time.time()
    old = ctx.prec
    ctx.prec = 128
    step, K, k0, s = 0.02, 2048, 0, T
    mins = []
    try:
        while s < T2:
            starts = [arb(T) + arb(step) * (k0 + K * b) for b in range(16)]
            G, _, _ = ev.evaluate(starts, step, K)
            sf = np.array([float(x.mid()) for x in starts])
            t = (sf[:, None] + step * np.arange(K)[None, :]).ravel()
            m = omega_float(t) - beta - np.asarray(G).ravel()
            keep = t < T2
            idx = np.nonzero((m[1:-1] <= m[:-2]) & (m[1:-1] <= m[2:]) & (m[1:-1] < thr) & keep[1:-1])[0] + 1
            mins += list(zip(t[idx].tolist(), m[idx].tolist()))
            k0 += 16 * K
            s = float(sf[-1]) + step * K
    finally:
        ctx.prec = old
    dips = []
    for t_, m_ in sorted(mins):
        if dips and t_ - dips[-1][0] < 3.0:
            if m_ < dips[-1][1]:
                dips[-1] = (t_, m_)
        else:
            dips.append((t_, m_))
    neg = [d for d in dips if d[1] < 0]
    print(f"  scan: {len(dips)} local minima below {thr} on [{T:.0f}, {T2:.0f}], {len(neg)} below 0 (deepest "
          f"{min(d[1] for d in dips):+.3f} at t = {min(dips, key=lambda d: d[1])[0]:.2f}); none below 0 after "
          f"t = {max(d[0] for d in neg):.1f} [{time.time() - t0:.0f}s]", flush=True)
    return {"T": T, "beta": beta, "thr": thr, "T2": T2, "dips": dips, "n_negative": len(neg),
            "n_negative_below_2460": sum(1 for d in neg if d[0] < 2460), "seconds": time.time() - t0}


def liftable_lambda(G, d, K=12, span=120.0, h=0.02):
    """lambda(d): the smallest generalised eigenvalue of int r s_j s_k against int s_j s_k (s_k shifted sincs of
    bandwidth Omega = L + gap/2 at Nyquist spacing around d), r = Omega - beta - G on d +- span."""
    from scipy.linalg import eigh
    Om = P["L"] + P["gap"] / 2
    ts = np.arange(d - span, d + span, h)
    r = omega_float(ts) - P["beta"] - G(ts)
    tk = d + (math.pi / Om) * np.arange(-K, K + 1)
    B = np.sinc(Om * (ts[None, :] - tk[:, None]) / math.pi)
    lam, V = eigh((B * r) @ B.T * h, B @ B.T * h)
    p2 = (V[:, 0] @ B) ** 2
    p2 /= p2.sum()
    centre = float((ts * p2).sum())
    return float(lam[0]), centre, float(np.sqrt(((ts - centre) ** 2 * p2).sum()))


def part_liftable(scan):
    cert = gapped()
    G = G_function(cert)
    t0 = time.time()
    rows = []
    for d, m in scan["dips"]:
        if m >= 0:
            continue
        lam, c, s = liftable_lambda(G, d)
        rows.append({"t": d, "depth": m, "lambda": lam, "minimiser_centre": c, "minimiser_spread": s})
    worst = min(rows, key=lambda x: x["lambda"])
    print(f"  liftable: {len(rows)} dips, lambda(d) in [{worst['lambda']:.3f}, {max(x['lambda'] for x in rows):.3f}] "
          f"(smallest at t = {worst['t']:.2f}, depth {worst['depth']:+.3f}) [{time.time() - t0:.0f}s]", flush=True)
    return {"Omega": P["L"] + P["gap"] / 2, "K": 12, "rows": rows, "seconds": time.time() - t0}


def cluster_program(G, dips, lo, hi, W, amax, b=1.0, m=6, half_width=12.0, dtau=1.0, step=0.04, target=0.01,
                    max_group=2):
    """The packet program of fix_dips for the group of dips in [lo, hi], with only its own points constrained
    (ownership by nearest dip among all dips): is it feasible, and at what sum |a|?"""
    import scipy.sparse as sp
    from scipy.optimize import linprog
    L, beta, T, gap = P["L"], P["beta"], P["T"], P["gap"]
    mb = m / b
    reach = mb * 10 ** (8.0 / m)
    groups = []
    for d in sorted(dips):
        if groups and d - groups[-1][-1] < 2 * (reach + half_width) and (
                len(groups[-1]) < max_group or d - groups[-1][-1] < 2.5 * half_width):
            groups[-1].append(d)
        else:
            groups.append([d])
    k = [i for i, grp in enumerate(groups) if any(lo <= d <= hi for d in grp)][0]
    grp = groups[k]
    all_dips = np.array([d for gg in groups for d in gg])
    dip_group = np.array([i for i, gg in enumerate(groups) for _ in gg])
    omegas = 2 * L + b + gap + 0.5 * np.arange(int(round(W / 0.5)) + 1)

    def env(s):
        x = s / mb
        with np.errstate(invalid="ignore", divide="ignore"):
            u = np.where(np.abs(x) < 1e-8, 1.0, np.sin(x) / x)
        return u ** m
    taus = np.unique(np.concatenate([np.arange(d - half_width, d + half_width + 1e-9, dtau) for d in grp]))
    ts = np.arange(max(T, grp[0] - half_width - reach), grp[-1] + half_width + reach, step)
    j = np.searchsorted(all_dips, ts)
    jl, jr = np.clip(j - 1, 0, len(all_dips) - 1), np.clip(j, 0, len(all_dips) - 1)
    nearest = np.where(np.abs(ts - all_dips[jl]) <= np.abs(ts - all_dips[jr]), jl, jr)
    ts = ts[dip_group[nearest] == k]
    r = omega_float(ts) - beta - G(ts)
    C = np.empty((len(ts), 2 * len(taus) * len(omegas)))
    c = 0
    for tau in taus:
        e1, e2 = env(ts - tau), env(ts + tau)
        for om in omegas:
            C[:, c] = e1 * np.cos(om * (ts - tau)) + e2 * np.cos(om * (ts + tau))
            C[:, c + 1] = e1 * np.sin(om * (ts - tau)) - e2 * np.sin(om * (ts + tau))
            c += 2
    n = C.shape[1]
    t0 = time.time()
    res = linprog(np.ones(2 * n), A_ub=sp.csr_matrix(np.hstack([-C, C])), b_ub=r - target,
                  bounds=[(0, amax)] * (2 * n), method="highs")
    out = {"dips": grp, "W": W, "amax": amax, "b": b, "half_width": half_width, "points": int(len(ts)),
           "columns": int(n), "feasible": res.status == 0, "seconds": time.time() - t0}
    if res.status == 0:
        a = res.x[:n] - res.x[n:]
        out.update({"sum_abs": float(np.abs(a).sum()), "max_abs": float(np.abs(a).max()),
                    "packets": int(np.count_nonzero(np.abs(a) > 1e-9))})
    return out


def part_cluster(scan):
    G = G_function(gapped())
    dips = [d for d, m in scan["dips"] if m < 0.03]
    rows = []
    for W, amax, b in ((4.0, 4.0, 1.0), (4.0, 16.0, 1.0), (8.0, 16.0, 1.0), (12.0, 16.0, 1.0), (4.0, 16.0, 2.0)):
        # b = 2: envelopes half as wide, centres twice as dense; the same nine-dip group (half_width 12)
        row = cluster_program(G, dips, *CLUSTER, W=W, amax=amax, b=b, half_width=12.0, dtau=1.0 / b)
        rows.append(row)
        print(f"  cluster W={W:g} amax={amax:g} b={b:g}: {len(row['dips'])} dips, "
              + (f"sum|a| {row['sum_abs']:.2f}, max|a| {row['max_abs']:.2f}, {row['packets']} packets"
                 if row["feasible"] else "infeasible") + f" [{row['seconds']:.0f}s]", flush=True)
    return rows


def fit_packets(scan=None):
    """Packets for every dip of the scan with fix_dips (deterministic up to the LP solver's rounding)."""
    cert = gapped()
    scan = part_scan(cert) if scan is None else scan
    F = P["fit"]
    dips = [d for d, m in scan["dips"] if m < F["target"] + 0.02]
    t0 = time.time()
    loc = wl.fix_dips(G_function(cert), omega_float, P["L"], P["T"], dips, P["beta"], gap=P["gap"],
                      target=F["target"], step=F["step"], W=F["W"], dw=F["dw"], dtau=F["dtau"],
                      half_width=F["half_width"], amax=F["amax"], max_group=F["max_group"],
                      core_pad=F["core_pad"], far_step=F["far_step"], buffer=F["buffer"])
    return loc, dips, time.time() - t0


def lifted():
    """The gapped certificate with the stored packets from T = 1000 (fitted and stored if absent)."""
    cert = gapped()
    if not os.path.exists(PACKETS):
        loc, dips, sec = fit_packets()
        with open(PACKETS, "w") as f:
            json.dump({"L": loc.L, "b": loc.b, "m": loc.m, "g": P["g"], "packet_gap": P["gap"], "beta": P["beta"],
                       "T": P["T"], "fit": P["fit"], "dips": dips, "fit_seconds": sec,
                       "atoms_tau_w_kind_a": [list(a) for a in loc.atoms]}, f, indent=1)
    with open(PACKETS) as f:
        D = json.load(f)
    loc = wl.LocalCorrection(L=D["L"], b=D["b"], m=D["m"], atoms=[tuple(a) for a in D["atoms_tau_w_kind_a"]])
    return wcb.attach_local(cert, loc), D


def part_fit(scan=None, refit=True):
    """Refit the packets and compare with the stored ones (those of the certification run), then check the
    trajectory constant of the stored packets."""
    cert, D = lifted()
    loc = cert.local
    row = {"packets": len(loc.atoms), "sum_abs": loc.sum_abs(), "max_abs": max(abs(a[3]) for a in loc.atoms),
           "carriers": [min(a[1] for a in loc.atoms), max(a[1] for a in loc.atoms)]}
    if refit:
        new, _, sec = fit_packets(scan)
        same = len(new.atoms) == len(loc.atoms) and all(a[:3] == b[:3] for a, b in zip(new.atoms, loc.atoms))
        row["refit"] = {"seconds": sec, "packets": len(new.atoms), "sum_abs": new.sum_abs(),
                        "same_centres_and_carriers": same,
                        "max_amplitude_difference": max(abs(a[3] - b[3]) for a, b in zip(new.atoms, loc.atoms))
                        if same else None}
        print(f"  refit: {len(new.atoms)} packets, sum|a| = {new.sum_abs():.6f}, same centres and carriers: {same}, "
              f"largest amplitude difference {row['refit']['max_amplitude_difference']} [{sec:.0f}s]", flush=True)
    tr = wcb.trajectory_check(cert, P["T"], P["beta"], verbose=True)
    row["trajectory"] = {k: tr.get(k) for k in ("certified", "T", "beta", "T2", "min_margin", "worst_t",
                                                "eval_error", "cells_per_level", "seconds")}
    print(f"  stored packets: {row['packets']}, sum|a| = {row['sum_abs']:.3f}, max|a| = {row['max_abs']:.3f}; "
          f"trajectory constant beta = {P['beta']} from T = {P['T']:.0f}: certified = {tr['certified']}, "
          f"min margin {tr.get('min_margin')}", flush=True)
    return row


def part_certify(parities=("even", "odd")):
    cert, _ = lifted()
    tr = wcb.trajectory_check(cert, P["T"], P["beta"], verbose=True)
    if not tr["certified"]:
        raise SystemExit("trajectory check failed")
    N = int(math.ceil(P["modes_factor"] * math.e * P["L"] * P["T"] / 4)) + 20
    res = we.certify_windows(P["L"], P["T"], {par: N for par in parities}, dps=P["dps"], verbose=True,
                             comb_cert=cert, wmax=8.0, chunk=1000)
    for r in res.values():
        r["comb_certificate"] = {k: v for k, v in cert.describe().items() if k != "per_prime"}
        r["setup"] = (f"E15 certificate, gap {P['g']} moved into A', {len(cert.local.atoms)} packets, "
                      f"T = {P['T']}, beta = {P['beta']}")
    return list(res.values())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--parts", default="scan,liftable,cluster,fit")
    ap.add_argument("--certify", action="store_true", help="also certify [-1.7, 1.7] at T = 1000 (about an hour)")
    args = ap.parse_args()
    t0 = time.time()
    out = {}
    if os.path.exists(OUT_JSON):
        with open(OUT_JSON) as f:
            out = json.load(f)
    if args.quick:
        print("[fit, stored packets]", flush=True)
        part_fit(refit=False)
    else:
        parts = args.parts.split(",")
        scan = None
        if "scan" in parts or "liftable" in parts or "cluster" in parts or "fit" in parts:
            print("[scan]", flush=True)
            scan = part_scan()
            out["scan"] = scan
        if "liftable" in parts:
            print("[liftable]", flush=True)
            out["liftable"] = part_liftable(scan)
        if "cluster" in parts:
            print("[cluster]", flush=True)
            out["cluster"] = part_cluster(scan)
        if "fit" in parts:
            print("[fit]", flush=True)
            out["fit"] = part_fit(scan)
    if args.certify:
        print("[certify]", flush=True)
        out["certify_L17_T1000"] = part_certify()
    out["description"] = __doc__.split("\n")[0]
    if not args.quick or args.certify:
        out["seconds_last_run"] = time.time() - t0
        with open(OUT_JSON, "w") as f:
            json.dump(out, f, indent=1, default=str)
    print(f"done [{time.time() - t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
