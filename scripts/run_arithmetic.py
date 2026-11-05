#!/usr/bin/env python3
"""Run the heavier versions of the elementary RH criteria, Speiser's criterion and
the Davenport-Heilbronn study, and write JSON results and a figure.

    python3 scripts/run_arithmetic.py                 # all parts, defaults below
    python3 scripts/run_arithmetic.py --parts speiser --speiser-T 2000 --workers 4

Outputs (results/json, results/figures):
    arithmetic_robin.json, arithmetic_lagarias.json, arithmetic_nicolas.json,
    arithmetic_ca.json, speiser_zeros.json, davenport_heilbronn_zeros.json,
    davenport_heilbronn_zeros.png
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rh import arithmetic as A                      # noqa: E402
from rh import davenport_heilbronn as D             # noqa: E402
from rh import speiser as S                         # noqa: E402


def _dump(obj, name, out_json: Path, t0: float, **meta):
    obj = dict(obj)
    obj["_meta"] = {"generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "wall_seconds": round(time.time() - t0, 2), **meta}
    path = out_json / name
    path.write_text(json.dumps(obj, indent=1, default=float))
    print(f"  wrote {path.relative_to(ROOT)} ({path.stat().st_size // 1024} kB)")


def run_robin_lagarias(N: int, out_json: Path):
    print(f"[robin/lagarias] sieving sigma(n) to N = {N:,} ...", flush=True)
    t0 = time.time()
    sig = A.sigma_sieve(N)
    r = A.robin_check(N, sigma=sig)
    print(f"  Robin violators (n>=2): {len(r.violators)} -> known 27: {r.violators_match_known}")
    print(f"  max sigma(n)/(n log log n) for 5040 < n <= N: {r.max_ratio_beyond_5040:.10f} at n = {r.argmax_beyond_5040}"
          f"  (/e^gamma = {r.max_robin_ratio_beyond_5040:.10f})")
    _dump(r.to_dict(), "arithmetic_robin.json", out_json, t0, N=N)
    t0 = time.time()
    l = A.lagarias_check(N, sigma=sig)
    print(f"  Lagarias violators: {l.violators}; min slack {l.min_slack:.12f} at n = {l.argmin_slack};"
          f" max sigma/RHS {l.max_ratio:.12f} at n = {l.argmax_ratio}")
    _dump(l.to_dict(), "arithmetic_lagarias.json", out_json, t0, N=N)


def run_nicolas(K: int, out_json: Path):
    print(f"[nicolas] first {K} primorials at 50 digits ...", flush=True)
    t0 = time.time()
    r = A.nicolas_check(K, dps=50)
    print(f"  violations: {r.violations}; decreasing for k>=2: {r.decreasing_from_2};"
          f" ratio at k={K}: {r.ratio[-1]:.12f}; (ratio-1) sqrt(p) log p -> {r.scaled_tail[-1][1]:.6f}")
    _dump(r.to_dict(), "arithmetic_nicolas.json", out_json, t0, K=K)


def run_ca(K: int, out_json: Path):
    print(f"[colossally abundant] first {K} CA numbers at 60 digits ...", flush=True)
    t0 = time.time()
    cas = A.ca_numbers(K, dps=60, verify_formula=True)
    summ = A.ca_summary(cas)
    print(f"  first 14 match A004490: {summ['matches_A004490']}; largest has {summ['largest_digits']} digits,"
          f" omega = {summ['largest_omega']}")
    print(f"  Robin holds for all CA n > 5040: {summ['robin_holds_for_all_n_gt_5040']};"
          f" max ratio beyond 5040 = {summ['max_robin_ratio_beyond_5040']:.10f} (CA #{summ['argmax_robin_ratio_beyond_5040_index']})")
    print(f"  last: ratio = {summ['last_robin_ratio']:.12f}, 1-ratio = {summ['last_one_minus_robin']:.3e},"
          f" sqrt(log n)(e^g loglog n - sigma/n) = {summ['last_robin_scaled']:.6f},"
          f" sigma/(n loglog n) = {summ['last_gronwall_ratio']:.10f} (e^gamma = {A.E_GAMMA:.10f})")
    print(f"  Lagarias holds for all: {summ['lagarias_holds_for_all']}; min log-slack {summ['min_lagarias_log_slack']:.6f};"
          f" min critical-epsilon gap {summ['min_gap_to_runner_up']:.3e}")
    _dump({"summary": summ, "ca_numbers": [c.to_dict() for c in cas]}, "arithmetic_ca.json", out_json, t0, K=K)


def run_speiser(T: float, workers: int, out_json: Path):
    print(f"[speiser] zeros of zeta' in [-1,4] x [0.25,{T}] with {workers} workers ...", flush=True)
    t0 = time.time()
    r = S.speiser_scan(T, workers=workers)
    print(f"  zeros: {r.count} (certified {r.count_certified}); independent winding {r.winding_total_independent};"
          f" Berndt main term {r.berndt_main_term:.2f}; N(T) - (T/2pi)log2 = {r.N_T_minus_log2_term:.2f} (N(T) = {r.N_T})")
    print(f"  zeros with Re s < 1/2: {r.count_re_lt_half} (rigorous: {r.count_re_lt_half_rigorous});"
          f" min Re s = {r.min_re:.10f} at s = {r.zero_min_re[0]:.10f} + {r.zero_min_re[1]:.10f} i")
    print(f"  pole check (should be -2): {r.pole_check_winding}; Spira's first zeros reproduced: {r.first_zeros_match_spira};"
          f" {r.evaluations} evaluations, {r.wall_seconds:.1f} s")
    _dump(r.to_dict(), "speiser_zeros.json", out_json, t0, T=T, workers=workers)


def run_dh(T: float, out_json: Path, out_fig: Path):
    print(f"[davenport-heilbronn] zeros with 0 < t < {T} ...", flush=True)
    t0 = time.time()
    r = D.dh_scan(T=T, t0=0.0)
    print(f"  kappa = {r.kappa:.16f}; kappa vs root number: {r.kappa_check['difference_upper']:.1e};"
          f" sigma_0 = {r.sigma0:.10f}")
    print(f"  functional equation max |f(s) - G(s) f(1-s)|: {max(x['abs_residual'] for x in r.functional_equation):.2e}")
    print(f"  zeros on the line: {r.on_line_count}; off the line with Re s > 1/2: {r.off_line_count_right}"
          f" (certified {r.off_line_certified}); total by winding number: {r.total_count_winding}; consistent: {r.consistent}")
    for z in r.off_line_zeros:
        print(f"    rho = {z['re']:.12f} + {z['im']:.12f} i   |f(rho)| = {z['residual']:.1e}"
              f"   rigorous Re in [{z.get('re_lower', float('nan')):.12f}, {z.get('re_upper', float('nan')):.12f}]")
    for k in r.known_zeros_found:
        print(f"  known {k['known']} found at distance {k['distance']:.2e}")
    _dump(r.to_dict(), "davenport_heilbronn_zeros.json", out_json, t0, T=T)
    fig = out_fig / "davenport_heilbronn_zeros.png"
    D.plot_zeros(r, str(fig))
    print(f"  wrote {fig.relative_to(ROOT)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parts", default="robin,nicolas,ca,speiser,dh",
                    help="comma-separated subset of robin,nicolas,ca,speiser,dh")
    ap.add_argument("--robin-N", type=int, default=10 ** 7)
    ap.add_argument("--nicolas-K", type=int, default=10 ** 4)
    ap.add_argument("--ca-K", type=int, default=200)
    ap.add_argument("--speiser-T", type=float, default=2000.0)
    ap.add_argument("--dh-T", type=float, default=300.0)
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1))
    ap.add_argument("--out", default=str(ROOT / "results"))
    a = ap.parse_args()
    out_json = Path(a.out) / "json"
    out_fig = Path(a.out) / "figures"
    out_json.mkdir(parents=True, exist_ok=True)
    out_fig.mkdir(parents=True, exist_ok=True)
    parts = {p.strip() for p in a.parts.split(",")}
    t0 = time.time()
    if "robin" in parts:
        run_robin_lagarias(a.robin_N, out_json)
    if "nicolas" in parts:
        run_nicolas(a.nicolas_K, out_json)
    if "ca" in parts:
        run_ca(a.ca_K, out_json)
    if "speiser" in parts:
        run_speiser(a.speiser_T, a.workers, out_json)
    if "dh" in parts:
        run_dh(a.dh_T, out_json, out_fig)
    print(f"done in {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
