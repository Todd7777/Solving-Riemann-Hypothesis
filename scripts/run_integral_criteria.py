#!/usr/bin/env python3
"""Experiment: Balazard–Saias–Yor and Volchkov integral criteria.
I(T) = (1/pi) int_0^T log|zeta(1/2+it)|/(1/4+t^2) dt  -> 0 (RH);
Volchkov double integral -> pi(3-gamma)/32."""
import json, os, sys, time, math
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from rh import integral_criteria as ic
from rh.data import load_zeros
from rh.constants import EULER_GAMMA
import argparse
ap = argparse.ArgumentParser(); ap.add_argument("--nzeros", type=int, default=100000); ap.add_argument("--volchkov_T", type=float, default=200.0)
a = ap.parse_args()
os.makedirs("results/json", exist_ok=True); os.makedirs("results/figures", exist_ok=True)
t0 = time.time()
z = load_zeros(a.nzeros)
T, I, partial = ic.bsy_integral(z, progress=2000)
checkpoints = {}
for n in [10, 100, 1000, 10000, 100000, 1000000]:
    if n <= len(z):
        checkpoints[str(n)] = {"T": float(z[n - 1]), "I_T": float(partial[n - 1])}
t1 = time.time()
V = ic.volchkov_integral(z, T=a.volchkov_T, sigma_max=30.0, n_nodes_t=24, n_nodes_sigma=48)
target = math.pi * (3 - EULER_GAMMA) / 32
summary = {
    "bsy": {"T_max": T, "I_T": I, "checkpoints": checkpoints, "expected_under_RH": 0.0,
            "note": "truncation error is O(1/T) with fluctuating sign; sensitivity to a zero 1/2+delta+i*gamma is ~delta/(1/4+gamma^2)",
            "seconds": t1 - t0},
    "volchkov": {"T": a.volchkov_T, "value": V, "target_pi_3_minus_gamma_over_32": target, "abs_error": abs(V - target),
                 "seconds": time.time() - t1},
}
json.dump(summary, open("results/json/integral_criteria_summary.json", "w"), indent=2)
print(json.dumps(summary, indent=2))
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.semilogx(z, partial, lw=.8)
ax.axhline(0, color="C3", ls="--", lw=.8)
ax.set_xlabel("T (upper limit, at a zero)"); ax.set_ylabel(r"$I(T)=\frac{1}{\pi}\int_0^T \frac{\log|\zeta(1/2+it)|}{1/4+t^2}dt$")
ax.set_title("Balazard–Saias–Yor criterion: RH $\\Leftrightarrow I(\\infty)=0$")
ax.grid(alpha=.3); fig.tight_layout(); fig.savefig("results/figures/integral_criteria_bsy.png", dpi=150)
print("done", time.time() - t0)
