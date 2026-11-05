#!/usr/bin/env python3
"""Compute nontrivial zeros of the Riemann zeta function with rigorous error
bounds using Arb (python-flint), in parallel, and store them as float64 arrays.

Each zero is returned by Arb as a complex ball 1/2 + i*[gamma +/- r]; Arb
certifies (via Turing's method internally) that the n-th zero is the n-th zero
and that it lies on the critical line within the ball.  We store the midpoint
of gamma and record the largest radius in a manifest so downstream code knows
the accuracy.

Usage:
    python3 scripts/compute_zeros.py --start 1 --count 100000 --workers 4 \
        --chunk 5000 --out data/zeros/first_1e5
"""
import argparse, json, os, sys, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np


def _compute_chunk(args):
    start, count, dps = args
    from flint import acb, ctx
    ctx.dps = dps
    t0 = time.time()
    zs = acb.zeta_zeros(start, count)
    gam = np.empty(count, dtype=np.float64)
    maxrad = 0.0
    for i, z in enumerate(zs):
        im = z.imag
        gam[i] = float(im.mid())
        r = float(im.rad())
        if r > maxrad:
            maxrad = r
    return start, gam, maxrad, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=1, help="index of first zero (1-based)")
    ap.add_argument("--count", type=int, default=100000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--chunk", type=int, default=5000)
    ap.add_argument("--dps", type=int, default=15)
    ap.add_argument("--out", required=True, help="output prefix, e.g. data/zeros/first_1e5")
    a = ap.parse_args()

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    tasks = []
    s = a.start
    end = a.start + a.count
    while s < end:
        c = min(a.chunk, end - s)
        tasks.append((s, c, a.dps))
        s += c

    gam = np.empty(a.count, dtype=np.float64)
    maxrad = 0.0
    t0 = time.time()
    done = 0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for start, g, rad, dt in ex.map(_compute_chunk, tasks):
            gam[start - a.start:start - a.start + len(g)] = g
            maxrad = max(maxrad, rad)
            done += len(g)
            print(f"[{time.time()-t0:8.1f}s] zeros {start}..{start+len(g)-1} done "
                  f"({done}/{a.count}), chunk time {dt:.1f}s, max radius {rad:.2e}",
                  flush=True)

    # sanity: strictly increasing
    assert np.all(np.diff(gam) > 0), "zeros not strictly increasing!"
    np.save(a.out + ".npy", gam)
    manifest = {
        "first_index": a.start,
        "count": a.count,
        "dps": a.dps,
        "max_radius": maxrad,
        "first_gamma": float(gam[0]),
        "last_gamma": float(gam[-1]),
        "method": "Arb acb_dirichlet_zeta_zeros via python-flint (rigorous isolation, Turing method)",
        "wall_seconds": time.time() - t0,
        "workers": a.workers,
    }
    with open(a.out + ".json", "w") as f:
        json.dump(manifest, f, indent=2)
    print("saved", a.out + ".npy", json.dumps(manifest))


if __name__ == "__main__":
    main()
