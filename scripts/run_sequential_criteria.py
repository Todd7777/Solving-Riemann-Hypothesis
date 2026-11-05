#!/usr/bin/env python3
"""Experiment: Báez-Duarte's sequential criterion c_k = O(k^{-3/4+eps}) <=> RH,
and the explicit formula c_k = sum_rho B(1-rho/2,k+1)/(2 zeta'(rho)) + trivial terms.
Writes results/json/sequential_criteria_summary.json and figures."""
import json, os, sys, time, math
import numpy as np
import mpmath as mp
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from rh import sequential_criteria as sc

os.makedirs("results/json", exist_ok=True); os.makedirs("results/figures", exist_ok=True)
mp.mp.dps = 30
t0 = time.time()
NZ = 8
rho = [mp.mpc(0.5, mp.zetazero(j).imag) for j in range(1, NZ + 1)]
zp = [mp.zeta(r, derivative=1) for r in rho]
ks = np.unique(np.round(np.logspace(1, 6, 400)).astype(int))
mu = sc.mobius_array(10 ** 6 * 2 + 10)
rows = []
for k in ks:
    exact = sc.c_k_float(int(k), mu=mu)
    tot, nt, tr = sc.c_k_explicit(int(k), rho, zp, return_parts=True)
    # first-zero-only prediction
    one = sc.c_k_explicit(int(k), rho[:1], zp[:1], include_trivial=False)
    rows.append((int(k), exact, float(tot), float(nt), float(tr), float(one)))
rows = np.array(rows, dtype=float)
k = rows[:, 0]; exact = rows[:, 1]; pred = rows[:, 2]; nontriv = rows[:, 3]; triv = rows[:, 4]; one = rows[:, 5]
np.save("results/json/sequential_criteria_rows.npy", rows)

scaled = exact * k ** 0.75
# amplitude predicted from the first zero alone: 2|Gamma(1-rho1/2)|/(2|zeta'(rho1)|) asymptotically
amp1 = float(2 * abs(mp.gamma(1 - rho[0] / 2)) / (2 * abs(zp[0])))
summary = {
    "k_range": [int(k[0]), int(k[-1])], "n_points": len(k),
    "max_abs_c_k": float(np.abs(exact).max()),
    "max_abs_error_explicit_formula": float(np.abs(exact - pred).max()),
    "max_rel_error_explicit_formula_k_ge_1e3": float(np.max(np.abs(exact - pred)[k >= 1e3] / np.abs(exact)[k >= 1e3])),
    "max_abs_c_k_times_k34_for_k_ge_1e4": float(np.abs(scaled[k >= 1e4]).max()),
    "predicted_asymptotic_amplitude_first_zero": amp1,
    "ratio_max_scaled_to_amp1": float(np.abs(scaled[k >= 1e5]).max() / amp1),
    "k_where_trivial_term_equals_nontrivial": None,
    "zeros_used": NZ, "wall_seconds": time.time() - t0,
    "statement": "RH <=> c_k = O(k^{-3/4+eps}); a zero with Re rho = beta < 1/2 would add a term of size k^{-(1+beta)/2} >> k^{-3/4}.",
}
cross = np.where(np.abs(triv) < np.abs(nontriv))[0]
if len(cross):
    summary["k_where_trivial_term_equals_nontrivial"] = int(k[cross[0]])
with open("results/json/sequential_criteria_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print(json.dumps(summary, indent=2))

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, axs = plt.subplots(2, 1, figsize=(9, 8))
ax = axs[0]
ax.semilogx(k, scaled, ".-", ms=3, lw=.7, label=r"$c_k\,k^{3/4}$ (exact, Möbius sum)")
ax.semilogx(k, (pred) * k ** 0.75, "--", lw=.8, label="explicit formula: 8 nontrivial + 12 trivial zeros")
ax.semilogx(k, triv * k ** 0.75, ":", lw=1, label="trivial-zero part (dominant for $k\\lesssim 10^5$)")
ax.axhline(amp1, color="C3", lw=.6); ax.axhline(-amp1, color="C3", lw=.6, label=r"$\pm$ amplitude from $\rho_1$ alone")
ax.set_xlabel("k"); ax.set_ylabel(r"$c_k\,k^{3/4}$"); ax.legend(fontsize=8); ax.grid(alpha=.3)
ax.set_title("Báez-Duarte sequential criterion: $c_k=\\sum_j(-1)^j\\binom{k}{j}/\\zeta(2j+2)$,  RH $\\Leftrightarrow c_k=O(k^{-3/4+\\epsilon})$")
ax = axs[1]
ax.semilogx(k, (exact - triv) * k ** 0.75, ".", ms=3, label=r"$(c_k - \mathrm{trivial})\,k^{3/4}$")
ax.semilogx(k, one * k ** 0.75, "-", lw=.8, label=r"first zero $\rho_1$ only: $2\,\mathrm{Re}\,B(1-\rho_1/2,k+1)/(2\zeta'(\rho_1))\,k^{3/4}$")
ax.set_xlabel("k"); ax.set_ylabel(r"$k^{3/4}\times$(nontrivial part)"); ax.legend(fontsize=8); ax.grid(alpha=.3)
ax.set_title(r"The nontrivial part is a sinusoid in $\log k$ with frequency $\gamma_1/2$ (period $e^{4\pi/\gamma_1}\approx 2.43$ in $k$)")
fig.tight_layout(); fig.savefig("results/figures/sequential_criteria.png", dpi=150)
print("done")
