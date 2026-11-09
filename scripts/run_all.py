#!/usr/bin/env python3
"""Run every experiment script in sequence and collect the summaries.

    python3 scripts/run_all.py --quick     # reduced sizes (~10-20 min)
    python3 scripts/run_all.py             # full sizes (hours)

Each entry: (script, full-size args, quick args).  Scripts write their own
results/json/*_summary.json and results/figures/*.png; this driver only
records wall times and exit codes in results/json/run_all_summary.json.
"""
import argparse, json, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

EXPERIMENTS = [
    ("verify_vasyunin_formula.py", [], []),
    ("run_verify.py", [], ["--T", "5000"]),
    ("run_arithmetic.py", [], ["--robin-N", "1000000", "--nicolas-K", "1000", "--ca-K", "50", "--speiser-T", "300", "--dh-T", "150"]),
    ("run_statistics.py", [], ["--n-windows", "20000"]),
    ("run_li.py", [], ["--nmax", "2000", "--hp-max", "300"]),
    ("run_nyman_beurling.py", ["--N", "10000", "--arb", "800", "--tag", "_N10000"], ["--N", "1000", "--arb", "200", "--tag", "_quick"]),
    ("analyze_nb_structure.py", ["--tag", "_N10000"], ["--tag", "_quick"]),
    ("run_sequential_criteria.py", [], []),
    ("run_integral_criteria.py", ["--nzeros", "20000"], ["--nzeros", "2000", "--volchkov_T", "60"]),
    ("run_debruijn_newman.py", [], ["--quick"]),
    ("run_weil_form.py", [], ["--quick"]),
    ("run_explicit_formula.py", [], ["--quick"]),
    ("run_weil_detection.py", [], ["--quick"]),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", nargs="*", default=None, help="subset of script names")
    a = ap.parse_args()
    os.chdir(ROOT)
    os.makedirs("results/json", exist_ok=True)
    log = {}
    for script, full, quick in EXPERIMENTS:
        if a.only and script not in a.only:
            continue
        path = os.path.join(HERE, script)
        if not os.path.exists(path):
            log[script] = {"status": "missing"}
            print(f"[skip] {script} (missing)")
            continue
        args = quick if a.quick else full
        t0 = time.time()
        print(f"[run ] {script} {' '.join(args)}", flush=True)
        r = subprocess.run([sys.executable, path] + args, capture_output=True, text=True)
        dt = time.time() - t0
        log[script] = {"status": "ok" if r.returncode == 0 else f"exit {r.returncode}", "seconds": dt,
                       "stderr_tail": r.stderr[-2000:] if r.returncode else ""}
        print(f"[done] {script}: {log[script]['status']} in {dt:.0f}s", flush=True)
        if r.returncode:
            print(r.stderr[-2000:])
    json.dump(log, open("results/json/run_all_summary.json", "w"), indent=2)


if __name__ == "__main__":
    main()
