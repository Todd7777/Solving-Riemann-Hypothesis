#!/usr/bin/env python3
"""E20: Weil positivity on [-1.9, 1.9]: the steps of E18 at L = 1.9, with the block storage of E19.

The steps: a joint torus certificate (by ADMM), the terms of H within 0.002 of 2L moved into A' (E16), the floor
of the envelope form located with H = 0, the dips of the comb bound lifted with wave packets (E17), the trajectory
constant checked rigorously (E15), and the envelope form certified at a cutoff above its floor.

Parts:
  * floor: the envelope form with H = 0 and beta = 0.2 imposed on [T, inf), both sectors from one assembly
    (block storage), at the cutoffs `--floor-T` (default P['floor_T']); each probe is merged into the summary as
    soon as it ends (Numerical; hours per probe, ~4-6 GB). Not in the default parts;
  * cert: the joint certificate rebuilt from the stored Gram factor (results/json/weil_gram_L1.9.npz), its
    near-2L terms, and the gapped certificate (seconds);
  * local: the comb's local operator norm against Omega near the cutoffs (a minute);
  * scan: the local minima of Omega - 0.2 - P_L + H below 0.03 on [T, T2];
  * fit: `rh.weil_local.fix_dips` with the options of E18 (packets stored in results/json/weil_l19_packets_T*.json),
    then `trajectory_check` of the stored packets from T with beta = 0.2 (Machine-checked);
  * --certify: the certification of [-1.9, 1.9] at T with these packets, both sectors from one assembly (block
    storage; run nothing else alongside). With RH_CHECKPOINT_DIR set, the assembly is checkpointed there every
    16 panels (~2 GB) and resumes after a restart.

Usage: python3 scripts/run_weil_l19.py [--quick] [--parts cert,local,scan,fit] [--floor-T 2300,2700] [--certify]
--quick evaluates the margin with the stored packets on [T, T + 700] in floats (a smoke test, about a minute; the
rigorous trajectory check is --parts fit). Writes results/json/weil_l19_summary.json (not with --quick).
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

ROOT = os.path.join(HERE, "..")
OUT_JSON = os.path.join(ROOT, "results", "json", "weil_l19_summary.json")
GRAM = os.path.join(ROOT, "results", "json", "weil_gram_L1.9.npz")
P = {"L": 1.9, "Lambda": 8.5, "g": 0.002, "beta": 0.2, "T": 2300.0, "gap": 0.05, "thr": 0.03,
     "fit": {"W": 12.0, "amax": 32.0, "half_width": 8.0, "step": 0.03, "far_step": 0.15, "core_pad": 25.0,
             "target": 0.02, "buffer": 0.06, "max_group": 2, "dw": 1.0, "dtau": 2.0, "core_per_dip": True,
             "max_chain": 3, "merge_on_fail": 6, "cut": 50.0, "refine": 0.005},
     "dps": 280, "modes_factor": 1.1, "floor_T": (2300.0,), "floor_dps": 280, "threads": 2, "chunk": 1024}
T0 = 2 * math.pi * math.exp(1 + 2 * P["L"])           # t0 = 2 pi e^{1+2L} = 763.4


def packets_path(T):
    return os.path.join(ROOT, "results", "json", f"weil_l19_packets_T{T:.0f}.json")


def n_modes(T):
    return int(math.ceil(P["modes_factor"] * math.e * P["L"] * T / 4)) + 20


def load_summary():
    if os.path.exists(OUT_JSON):
        with open(OUT_JSON) as f:
            return json.load(f)
    return {}


def save_summary(out, keys=None):
    """Write the summary; with `keys`, only those keys of `out` are merged into the file as it is now (other
    runs, e.g. a floor probe, may have written it meanwhile)."""
    if keys is not None:
        fresh = load_summary()
        fresh.update({k: out[k] for k in keys if k in out})
        out = fresh
    out["description"] = __doc__.split("\n")[0]
    tmp = OUT_JSON + ".tmp"
    with open(tmp, "w") as f:
        json.dump(out, f, indent=1, default=str)
    os.replace(tmp, OUT_JSON)


def omega_float(t):
    from scipy.special import digamma
    return np.real(digamma(0.25 + 0.5j * np.asarray(t, float))) - math.log(math.pi)


# ----------------------------------------------------------------------------------------------------------------
# floor
# ----------------------------------------------------------------------------------------------------------------
def h0_certificate(T):
    """H = 0 with beta imposed on [T, inf): the envelope form without the comb's high-pass part (a probe of the
    floor, not a certificate)."""
    from flint import arb
    L = P["L"]
    comb = wcb.prime_comb(L)
    A_L = sum((c for t in comb.values() for _, c in t), arb(0))
    cert = wcb.CombCertificate(L=L, X=2 * L)
    cert.A_L, cert.A_prime, cert.terms, cert.H_sup, cert.omega_min = A_L, A_L, [], 0.0, 2 * L + 1.0
    cert.trajectory = {"certified": True, "T": T, "beta": P["beta"], "imposed": True}
    return cert


def floor_probe(T, dps=None, threads=None, chunk=None):
    """Both sectors from one assembly (block storage): the inertia of mid(R_A), and lambda_min if positive."""
    dps = dps or P["floor_dps"]
    N = n_modes(T)
    t0 = time.time()
    rows = []
    ck = os.environ.get("RH_CHECKPOINT_DIR")          # the assembly resumes from there after a restart
    ck = os.path.join(ck, f"l19_floor_T{T:.0f}_d{dps}.pkl") if ck else None
    with we._threads(threads or P["threads"]):
        blks = we.envelope_blocks(P["L"], T, {"even": N, "odd": N}, dps=dps, comb_cert=h0_certificate(T),
                                  wmax=8.0, chunk=chunk or P["chunk"], storage="blocks", verbose=True,
                                  checkpoint=ck, checkpoint_every=16)
        t_asm = time.time() - t0
        for par in ("even", "odd"):
            blk = blks.pop(par)
            t1 = time.time()
            with we._ctx(dps):
                f = we.ldl_mid_blocked(blk["R"], None)
                lam = None
                if f["n_negative"] == 0:
                    est = we.approx_lambda_min_ldl(blk["R"])
                    lam = float(est[0].mid()) if est is not None else None
            row = {"T": T, "T_over_t0": T / T0, "parity": par, "N": N, "dps": dps, "n_nodes": blk["n_nodes"],
                   "n_negative": f["n_negative"], "first_negative": f["first_negative"],
                   "min_pivot": float(f["min_pivot"]), "lambda_approx": lam, "assembly_seconds": t_asm,
                   "seconds": time.time() - t1}
            del blk, f
            rows.append(row)
            print(f"  floor T = {T:.0f} ({T / T0:.2f} t0) {par}: {row['n_negative']} negative pivots (first at "
                  f"{row['first_negative']}, min pivot {row['min_pivot']:.3g}), lambda_min ~ {lam} "
                  f"[assembly {t_asm:.0f}s, {row['seconds']:.0f}s]", flush=True)
    return rows


def part_floor(Ts):
    """Each probe is merged into the summary on disk as soon as it ends."""
    for T in Ts:
        rows = floor_probe(T)
        out = load_summary()
        out["floor"] = sorted([r for r in out.get("floor", []) if r["T"] != T] + rows,
                              key=lambda r: (r["T"], r["parity"]))
        save_summary(out)
    return load_summary()


# ----------------------------------------------------------------------------------------------------------------
# comb certificate
# ----------------------------------------------------------------------------------------------------------------
def gram():
    z = np.load(GRAM)
    G = {"B": z["B"].astype(np.int64), "R": z["R"], "primes": [int(p) for p in z["primes"]]}
    G.update(json.loads(str(z["meta"])))
    return G


def joint():
    return wcb.joint_comb_certificate(P["L"], P["Lambda"], gram=gram())


def gapped():
    return wcb.gap_certificate(joint(), P["g"])


def G_function(cert):
    comb = wcb.prime_comb(P["L"])
    ns = np.array([n for t in comb.values() for n, _ in t], float)
    cs = np.array([float(c.mid()) for t in comb.values() for _, c in t])
    w = np.array([math.log(n / d) for n, d, _ in cert.terms])
    h = np.array([float(x.mid()) for _, _, x in cert.terms])

    def G(ts):
        ts = np.asarray(ts, float)
        res = np.empty(len(ts))
        for i in range(0, len(ts), 1000):
            t = ts[i:i + 1000]
            res[i:i + 1000] = np.cos(np.outer(t, np.log(ns))) @ cs - np.cos(np.outer(t, w)) @ h
        return res
    return G


def part_cert():
    t0 = time.time()
    c = joint()
    near = sorted((math.log(n / d) - 2 * P["L"], n, d, float(h.mid())) for n, d, h in c.terms
                  if math.log(n / d) < 2 * P["L"] + 0.03)
    cg = wcb.gap_certificate(c, P["g"])
    A, Ag = float(c.A_prime.upper()), float(cg.A_prime.upper())
    row = {"Lambda": P["Lambda"], "A_L": float(c.A_L.mid()), "A_prime": A, "terms": len(c.terms),
           "sum_abs_h": c.H_sup,
           "near_2L_sum_abs": {str(gg): sum(abs(x[3]) for x in near if x[0] < gg) for gg in (0.002, 0.005, 0.01, 0.03)},
           "largest_near_2L": [{"n/d": f"{x[1]}/{x[2]}", "gap": x[0], "h": x[3]}
                               for x in sorted(near, key=lambda x: -abs(x[3]))[:5]],
           "gapped_A_prime": Ag, "gapped_terms": len(cg.terms),
           "T2": 2 * math.pi * math.exp(Ag + P["beta"]) + 1, "seconds": time.time() - t0}
    print(f"  joint certificate (Lambda = {P['Lambda']}): A' = {A:.4f} (A_L = {row['A_L']:.3f}), {row['terms']} terms, "
          f"sum|h| = {row['sum_abs_h']:.2f}; gapped (g = {P['g']}): A' = {Ag:.4f}, T2 = {row['T2']:.0f}", flush=True)
    return row


def local_norm_range(L, t_lo, t_hi, step, h=0.025):
    """lambda(t0) of E15 (piecewise-constant kernels on cells of width h; a lower bound) on [t_lo, t_hi)."""
    comb = wcb.prime_comb(L)
    lam = np.array([math.log(n) for t in comb.values() for n, _ in t])
    cs = np.array([float(c.mid()) for t in comb.values() for _, c in t])
    M = int(round(2 * L / h))
    x = -L + h * (np.arange(M) + 0.5)
    D = x[:, None] - x[None, :]
    Tn = [np.maximum(0, h - np.abs(D + l)) / h for l in lam]
    ts = np.arange(t_lo, t_hi, step)
    res = np.empty_like(ts)
    for i, t0 in enumerate(ts):
        A = np.zeros((M, M), complex)
        for c, l, Tm in zip(cs, lam, Tn):
            e = np.exp(1j * t0 * l)
            A += (c / 2) * (e * Tm + np.conj(e) * Tm.T)
        res[i] = np.linalg.eigvalsh(A)[-1]
    return ts, res


def part_local(ranges=((1.9, 1700.0, 2000.0), (1.9, 2200.0, 2400.0), (1.9, 2600.0, 2800.0))):
    rows = []
    for L, lo, hi in ranges:
        ts, lam = local_norm_range(L, lo, hi, 0.05)
        gap = omega_float(ts) - lam
        i = int(np.argmin(gap))
        rows.append({"L": L, "range": [lo, hi], "max_lambda": float(lam.max()), "min_Omega_minus_lambda": float(gap[i]),
                     "at": float(ts[i])})
        print(f"  L = {L} on [{lo:.0f}, {hi:.0f}]: max lambda {lam.max():.3f}, min Omega - lambda {gap[i]:.3f} "
              f"at {ts[i]:.2f}", flush=True)
    return rows


# ----------------------------------------------------------------------------------------------------------------
# comb bound along the trajectory: scan, packets, trajectory check
# ----------------------------------------------------------------------------------------------------------------
def part_scan(cert=None):
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
          f"{min(d[1] for d in dips):+.3f} at t = {min(dips, key=lambda d: d[1])[0]:.2f}); last below 0 at "
          f"t = {max(d[0] for d in neg) if neg else float('nan'):.1f} [{time.time() - t0:.0f}s]", flush=True)
    return {"T": T, "beta": beta, "thr": thr, "T2": T2, "dips": dips, "n_negative": len(neg), "seconds": time.time() - t0}


def fit_packets(scan=None):
    cert = gapped()
    scan = part_scan(cert) if scan is None else scan
    F = P["fit"]
    dips = [d for d, m in scan["dips"] if m < F["target"] + 0.02]
    t0 = time.time()
    loc = wl.fix_dips(G_function(cert), omega_float, P["L"], P["T"], dips, P["beta"], gap=P["gap"],
                      target=F["target"], step=F["step"], W=F["W"], dw=F["dw"], dtau=F["dtau"],
                      half_width=F["half_width"], amax=F["amax"], max_group=F["max_group"],
                      core_pad=F["core_pad"], far_step=F["far_step"], buffer=F["buffer"],
                      core_per_dip=F["core_per_dip"], max_chain=F["max_chain"], merge_on_fail=F["merge_on_fail"],
                      cut=F["cut"], refine=F["refine"])
    return loc, dips, time.time() - t0


def lifted(scan=None):
    cert = gapped()
    path = packets_path(P["T"])
    if not os.path.exists(path):
        loc, dips, sec = fit_packets(scan)
        with open(path, "w") as f:
            json.dump({"L": loc.L, "b": loc.b, "m": loc.m, "g": P["g"], "packet_gap": P["gap"], "beta": P["beta"],
                       "T": P["T"], "Lambda": P["Lambda"], "fit": P["fit"], "dips": dips, "fit_seconds": sec,
                       "atoms_tau_w_kind_a": [list(a) for a in loc.atoms]}, f, indent=1)
    with open(path) as f:
        D = json.load(f)
    loc = wl.LocalCorrection(L=D["L"], b=D["b"], m=D["m"], atoms=[tuple(a) for a in D["atoms_tau_w_kind_a"]])
    return wcb.attach_local(cert, loc), D


def part_fit(scan=None):
    cert, D = lifted(scan)
    loc = cert.local
    row = {"T": P["T"], "packets": len(loc.atoms), "sum_abs": loc.sum_abs(),
           "max_abs": max(abs(a[3]) for a in loc.atoms) if loc.atoms else 0.0,
           "carriers": [min(a[1] for a in loc.atoms), max(a[1] for a in loc.atoms)] if loc.atoms else None,
           "fit_seconds": D.get("fit_seconds")}
    tr = wcb.trajectory_check(cert, P["T"], P["beta"], verbose=True)
    row["trajectory"] = {k: tr.get(k) for k in ("certified", "T", "beta", "T2", "min_margin", "worst_t",
                                                "eval_error", "cells_per_level", "seconds")}
    print(f"  packets: {row['packets']}, sum|a| = {row['sum_abs']:.3f}, max|a| = {row['max_abs']:.3f}; "
          f"trajectory constant beta = {P['beta']} from T = {P['T']:.0f}: certified = {tr['certified']}, "
          f"min margin {tr.get('min_margin')}", flush=True)
    return row


def part_quick(span=700.0, step=0.005):
    """A float smoke test of the stored packets: the margin Omega - beta - P_L + H + packets on [T, T + span] at
    step `step` (the rigorous check over [T, T2] is `--parts fit`, about 11 minutes)."""
    from flint import arb, ctx
    cert, D = lifted()
    ts = np.arange(P["T"], P["T"] + span, step)
    old = ctx.prec
    ctx.prec = 128
    try:
        vals, errs = cert.local.eval_float([arb(P["T"])], step, len(ts))
    finally:
        ctx.prec = old
    m = omega_float(ts) - P["beta"] - G_function(cert)(ts) + np.asarray(vals[0])
    i = int(np.argmin(m))
    row = {"packets": len(cert.local.atoms), "span": [P["T"], P["T"] + span], "step": step,
           "min_margin_float": float(m[i]), "at": float(ts[i])}
    print(f"  {row['packets']} packets; float margin on [{P['T']:.0f}, {P['T'] + span:.0f}] at step {step}: min "
          f"{m[i]:.4f} at t = {ts[i]:.3f}", flush=True)
    if not m[i] > 0:
        raise SystemExit("the stored packets leave a negative margin")
    return row


def part_certify():
    cert, _ = lifted()
    tr = wcb.trajectory_check(cert, P["T"], P["beta"], verbose=True)
    if not tr["certified"]:
        raise SystemExit("trajectory check failed")
    N = n_modes(P["T"])
    ck = os.environ.get("RH_CHECKPOINT_DIR")          # the assembly resumes from there after a restart (~2 GB)
    ck = os.path.join(ck, f"l19_assembly_T{P['T']:.0f}.pkl") if ck else None
    res = we.certify_windows(P["L"], P["T"], {"even": N, "odd": N}, dps=P["dps"], verbose=True, comb_cert=cert,
                             wmax=8.0, chunk=P["chunk"], storage="blocks", threads=P["threads"], checkpoint=ck,
                             checkpoint_every=16)
    out = []
    for par in ("even", "odd"):
        r = res[par]
        r["comb_certificate"] = {k: v for k, v in cert.describe().items() if k != "per_prime"}
        r["setup"] = (f"joint Lambda = {P['Lambda']} (ADMM), gap {P['g']} moved into A', {len(cert.local.atoms)} "
                      f"packets, T = {P['T']}, beta = {P['beta']}")
        out.append(r)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--parts", default="cert,local,scan,fit")
    ap.add_argument("--floor-T", default=None, help="comma-separated cutoffs for the floor probes")
    ap.add_argument("--certify", action="store_true", help="also certify [-1.9, 1.9] at T (hours)")
    args = ap.parse_args()
    t0 = time.time()
    out = load_summary()
    if args.quick:
        print("[quick: the stored packets near T, in floats]", flush=True)
        part_quick()
        print(f"done [{time.time() - t0:.0f}s]", flush=True)
        return
    parts = args.parts.split(",")
    if "floor" in parts:
        Ts = [float(x) for x in args.floor_T.split(",")] if args.floor_T else list(P["floor_T"])
        print(f"[floor] T = {Ts}", flush=True)
        out = part_floor(Ts)
    for name, fn in (("cert", part_cert), ("local", part_local)):
        if name in parts:
            print(f"[{name}]", flush=True)
            out[name] = fn()
    scan = None
    stored = out.get("scan")
    if "fit" in parts and "scan" not in parts and stored and stored.get("T") == P["T"] \
            and stored.get("beta") == P["beta"] and abs(stored.get("T2", 0) - (2 * math.pi * math.exp(
                float(gapped().A_prime.upper()) + P["beta"]) + 1)) < 1e-6:
        print("[scan] the stored scan of this certificate", flush=True)
        scan = stored
    elif "scan" in parts or "fit" in parts:
        print("[scan]", flush=True)
        scan = part_scan()
        out["scan"] = scan
    if "fit" in parts:
        print("[fit]", flush=True)
        out["fit"] = part_fit(scan)
    if args.certify:
        print("[certify]", flush=True)
        tc = time.time()
        res = part_certify()
        ok = all(r["positive"] for r in res)
        out["certify_L19"] = res
        out["certify_L19_status"] = "certified in both sectors" if ok else "NOT certified"
        out["certify_L19_summary"] = {"L": P["L"], "T": P["T"], "beta": P["beta"], "N": res[0]["N"],
                                      "dps": res[0]["dps"], "seconds": time.time() - tc,
                                      "certified_lower_bound": min(r["certified_lower_bound"] for r in res)
                                      if ok else None}
    out["seconds_last_run"] = time.time() - t0
    touched = [k for k in ("cert", "local", "fit") if k in parts]
    if "scan" in parts or "fit" in parts:
        touched.append("scan")
    if args.certify:
        touched += ["certify_L19", "certify_L19_status", "certify_L19_summary"]
    save_summary(out, keys=touched + ["seconds_last_run"])
    print(f"done [{time.time() - t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
