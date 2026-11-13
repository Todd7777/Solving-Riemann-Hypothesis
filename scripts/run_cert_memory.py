#!/usr/bin/env python3
"""E19: memory of the certification (`rh.weil_envelope.certify_window`), the dense arb_mat path against the block
path (`SymBlocks`, the default since E19).

Parts (each measurement in a fresh process; the peak RSS of each phase is VmHWM, reset between phases):
  * profile: one even sector (L = 0.9, T = 1400, N = 962, 235 digits) certified with storage 'dense' (chunks of
    1000 nodes) and 'blocks' (chunks of 1000 and of 512): assembly, inverse iteration, residual certificate.  The
    three runs go in parallel processes (about 25 minutes, 4.5 GB in all); they must give the same estimate and
    the same certified bound;
  * bench: a synthetic SymBlocks matrix of the size of an L = 1.9 sector (N = 4300, 280 digits): memory and time
    of one full-width assembly flush, of the blocked LDL^T with inverse iteration and of the residual certificate
    (about 25 minutes with 3 threads, 4.5 GB; the matrix is synthetic, only memory and time are meaningful);
  * inner: one block product (512 x k)(k x 512) at 280 digits, as in an assembly flush, for k = 256, 512 and 1024
    (the chunk): time per multiply-add (under a minute);
  * recert17 (not in the default parts): [-1.7, 1.7] from T = 1000 (E17) certified again with the block path
    and compared with the stored dense results (about an hour, both sectors).

Usage: python3 scripts/run_cert_memory.py [--quick] [--parts profile,bench,inner,recert17] [--threads 3]
--quick: the profile at L = 0.5, T = 40, N = 37 (40 digits, blocks of 8), the benchmark at N = 300 and the
products with blocks of 64 (about a minute); it prints and writes nothing.  Otherwise writes (and merges into)
results/json/cert_memory_summary.json.
"""
import argparse
import json
import math
import os
import random
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
OUT_JSON = os.path.join(ROOT, "results", "json", "cert_memory_summary.json")

PROFILE = {"L": 0.9, "T": 1400.0, "N": 962, "dps": 235, "block": 512}
PROFILE_RUNS = [("dense", 1000), ("blocks", 1000), ("blocks", 512)]
PROFILE_QUICK = {"L": 0.5, "T": 40.0, "N": 37, "dps": 40, "block": 8}
PROFILE_QUICK_RUNS = [("dense", 50), ("blocks", 50)]
BENCH = {"n": 4300, "dps": 280, "chunk": 1024, "block": 512}
BENCH_QUICK = {"n": 300, "dps": 60, "chunk": 64, "block": 64}
INNER = {"dps": 280, "block": 512, "inner": [256, 512, 1024]}
INNER_QUICK = {"dps": 60, "block": 64, "inner": [32, 64]}


def _status() -> dict:
    d = {}
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith(("VmRSS", "VmHWM")):
                k, v = line.split(":")
                d["peak_GB" if k == "VmHWM" else "rss_GB"] = round(int(v.split()[0]) / 1e6, 3)
    return d


def _reset_peak() -> None:
    with open("/proc/self/clear_refs", "w") as f:
        f.write("5")


def child_profile(mode: str, p: dict) -> dict:
    from flint import arb
    from rh import weil_envelope as we
    out = {"mode": mode, **p}
    _reset_peak()
    t0 = time.time()
    blk = we.envelope_block(p["L"], p["T"], p["N"], "even", dps=p["dps"], chunk=p["chunk"], wmax=8.0,
                            storage=mode, block=p["block"])
    out["assembly"] = dict(_status(), seconds=round(time.time() - t0, 1), n_nodes=blk["n_nodes"])
    _reset_peak()
    with we._ctx(p["dps"]):
        t1 = time.time()
        est = we.approx_lambda_min_ldl(blk["R"])
        out["estimate"] = dict(_status(), seconds=round(time.time() - t1, 1))
        lam = est[0]
        _reset_peak()
        t1 = time.time()
        cert = we.residual_certify(blk["R"], arb(float(lam.mid()) * 0.98))
        out["residual"] = dict(_status(), seconds=round(time.time() - t1, 1))
    out.update(lambda_approx=float(lam.mid()), certified=cert["certified"],
               lower_bound=float(cert["lower_bound"].lower()) if cert["certified"] else None,
               residual_F=float(cert["residual_F"].upper()) if cert["residual_F"] is not None else None,
               seconds=round(time.time() - t0, 1))
    return out


def child_bench(p: dict, threads: int) -> dict:
    """Synthetic: a diagonally dominant SymBlocks matrix with full-precision entries."""
    from flint import arb, ctx
    from rh import weil_envelope as we
    random.seed(1)
    ctx.dps, ctx.threads = p["dps"], threads
    n, chunk, block = p["n"], p["chunk"], p["block"]
    out = {**p, "threads": threads}
    t0 = time.time()
    S = we.SymBlocks(n, block)
    s2 = arb(2).sqrt()
    for (I, J), B in S.blocks.items():
        for a in range(S.sizes[I]):
            for c in range(S.sizes[J] if I > J else a + 1):
                v = (arb(random.random() - 0.5) * s2) if (I, a) != (J, c) else arb(n) * s2
                B[a, c] = v
                if I == J:
                    B[c, a] = v
    out["matrix"] = dict(_status(), seconds=round(time.time() - t0, 1),
                         entries=sum(B.nrows() * B.ncols() for B in S.blocks.values()))
    _reset_peak()
    t1 = time.time()
    sec = {"S": S, "GT": [None] * S.nb, "wts": [], "orders": list(range(n))}
    js = [arb(random.random()) * s2 for _ in range(n)]
    for _ in range(chunk):
        we._add_node_blocks(sec, js, n, arb(random.random()) * s2, chunk)
    out["fill"] = dict(_status(), seconds=round(time.time() - t1, 1))
    t1 = time.time()
    we._flush_blocks(sec)
    out["flush"] = dict(_status(), seconds=round(time.time() - t1, 1))
    del js, sec
    _reset_peak()
    t1 = time.time()
    est = we.approx_lambda_min_ldl(S)
    out["estimate"] = dict(_status(), seconds=round(time.time() - t1, 1))
    _reset_peak()
    t1 = time.time()
    cert = we.residual_certify(S, arb(float(est[0].mid()) * 0.5))   # no isolated small eigenvalue: a safe shift
    out["residual"] = dict(_status(), seconds=round(time.time() - t1, 1), certified=cert["certified"])
    out["seconds"] = round(time.time() - t0, 1)
    return out


def child_inner(p: dict, threads: int) -> dict:
    """A block product (block x k)(k x block), as in `_flush_blocks`: seconds per multiply-add against the inner
    dimension k (the chunk); random full-precision entries, best of two."""
    from flint import arb, arb_mat, ctx
    random.seed(2)
    ctx.dps, ctx.threads = p["dps"], threads
    s2 = arb(2).sqrt()
    b = p["block"]

    def rnd(m, n):
        M = arb_mat(m, n)
        for i in range(m):
            for j in range(n):
                M[i, j] = arb(random.random() - 0.5) * s2
        return M

    out = {**p, "threads": threads, "ns_per_multiply_add": {}}
    for k in p["inner"]:
        A, B = rnd(b, k), rnd(k, b)
        best = math.inf
        for _ in range(2):
            t = time.time()
            A * B
            best = min(best, time.time() - t)
        out["ns_per_multiply_add"][str(k)] = round(best / (b * b * k) * 1e9, 2)
    return out


def child_recert17(threads: int) -> dict:
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import run_weil_lift as rl
    from rh import weil_comb as wcb
    from rh import weil_envelope as we
    P = rl.P
    cert, _ = rl.lifted()
    tr = wcb.trajectory_check(cert, P["T"], P["beta"], verbose=True)
    if not tr["certified"]:
        raise SystemExit("trajectory check failed")
    N = int(math.ceil(P["modes_factor"] * math.e * P["L"] * P["T"] / 4)) + 20
    _reset_peak()
    t0 = time.time()
    res = we.certify_windows(P["L"], P["T"], {"even": N, "odd": N}, dps=P["dps"], verbose=True, comb_cert=cert,
                             wmax=8.0, chunk=1000, storage="blocks", threads=threads)
    with open(rl.OUT_JSON) as f:
        stored = {r["parity"]: r for r in json.load(f)["certify_L17_T1000"]}
    out = {"N": N, "dps": P["dps"], "threads": threads, "seconds": round(time.time() - t0, 1), **_status(),
           "stored_dense": {"seconds": 3582.0, "peak_GB": 4.41}, "sectors": {}}
    for par, r in res.items():
        s = stored[par]
        out["sectors"][par] = {"certified_lower_bound": r["certified_lower_bound"],
                               "stored_certified_lower_bound": s["certified_lower_bound"],
                               "lambda_approx": r["lambda_approx"], "stored_lambda_approx": s["lambda_approx"],
                               "residual_F": r["residual_F"], "stored_residual_F": s["residual_F"],
                               "same_bound": r["certified_lower_bound"] == s["certified_lower_bound"]}
    return out


def _start_child(*args):
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1")
    return subprocess.Popen([sys.executable, os.path.abspath(__file__), "--child", json.dumps(list(args))],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)


def _finish_child(proc, name) -> dict:
    out, err = proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"child {name} failed:\n{err[-3000:]}")
    return json.loads(out.strip().splitlines()[-1])


def run_child(*args) -> dict:
    """Run one measurement in a fresh process; its last stdout line is the JSON result."""
    return _finish_child(_start_child(*args), args[0])


def part_profile(p: dict, runs) -> dict:
    """The runs in parallel processes (the peak RSS is per process)."""
    procs = [((mode, chunk), _start_child("profile", mode, dict(p, chunk=chunk))) for mode, chunk in runs]
    rows = [dict(_finish_child(proc, "profile"), chunk=key[1]) for key, proc in procs]
    for name in ("assembly", "estimate", "residual"):
        print(f"  {name:9s} " + "   ".join(f"{r['mode']}/{r['chunk']} {r[name]['peak_GB']:.2f} GB "
                                         f"{r[name]['seconds']:7.1f} s" for r in rows), flush=True)
    ref = rows[0]
    same = all(r["certified"] and abs(r["lambda_approx"] / ref["lambda_approx"] - 1) < 1e-12
               and abs(r["lower_bound"] / ref["lower_bound"] - 1) < 1e-12 for r in rows)
    print("  lambda " + ", ".join(f"{r['lambda_approx']:.10e}" for r in rows) + "; certified bound "
          + ", ".join(f"{r['lower_bound']:.10e}" for r in rows) + (": same" if same else ": DIFFERENT"), flush=True)
    if not same:
        raise SystemExit("the storage modes disagree")
    return {"settings": p, "runs": rows, "same_certificate": same}


def part_bench(p: dict, threads: int) -> dict:
    r = run_child("bench", p, threads)
    for name in ("matrix", "fill", "flush", "estimate", "residual"):
        print(f"  {name:9s} {r[name]['peak_GB']:.2f} GB {r[name]['seconds']:7.1f} s", flush=True)
    if not r["residual"]["certified"]:
        raise SystemExit("benchmark residual not certified")
    return r


def part_inner(p: dict, threads: int) -> dict:
    r = run_child("inner", p, threads)
    print("  " + ", ".join(f"inner {k}: {v} ns" for k, v in r["ns_per_multiply_add"].items())
          + f" per multiply-add ({r['dps']} digits, blocks of {r['block']}, {r['threads']} threads)", flush=True)
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--parts", default="profile,bench,inner")
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--child", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.child is not None:
        kind, *rest = json.loads(args.child)
        fn = {"profile": child_profile, "bench": child_bench, "inner": child_inner,
              "recert17": child_recert17}[kind]
        print(json.dumps(fn(*rest), default=str), flush=True)
        return
    t0 = time.time()
    if args.quick:
        print("[profile, quick]", flush=True)
        part_profile(PROFILE_QUICK, PROFILE_QUICK_RUNS)
        print("[bench, quick]", flush=True)
        part_bench(BENCH_QUICK, 1)
        print("[inner, quick]", flush=True)
        part_inner(INNER_QUICK, 1)
        print(f"done [{time.time() - t0:.0f}s]", flush=True)
        return
    out = {}
    if os.path.exists(OUT_JSON):
        with open(OUT_JSON) as f:
            out = json.load(f)
    parts = args.parts.split(",")
    if "profile" in parts:
        print("[profile]", flush=True)
        out["profile"] = part_profile(PROFILE, PROFILE_RUNS)
    if "bench" in parts:
        print("[bench]", flush=True)
        out["bench"] = part_bench(BENCH, args.threads)
    if "inner" in parts:
        print("[inner]", flush=True)
        out["inner"] = part_inner(INNER, args.threads)
    if "recert17" in parts:
        print("[recert17]", flush=True)
        out["recert17"] = run_child("recert17", args.threads)
        print(json.dumps(out["recert17"]["sectors"], indent=1), flush=True)
    out["description"] = __doc__.split("\n")[0]
    out["seconds_last_run"] = round(time.time() - t0, 1)
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=1, default=str)
    print(f"wrote {OUT_JSON} [{out['seconds_last_run']:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
