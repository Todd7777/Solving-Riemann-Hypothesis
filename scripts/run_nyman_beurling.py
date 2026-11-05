#!/usr/bin/env python3
"""Experiment: Báez-Duarte distances d_N^2 versus Burnol's constant C/log N.

Usage: python3 scripts/run_nyman_beurling.py --N 3000 [--arb 400]
Writes results/json/nyman_beurling_summary.json, results/json/nyman_beurling_dN.npy
and figures results/figures/nyman_beurling_*.png
"""
import argparse, json, math, os, sys, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from rh import nyman_beurling as nb
from rh.constants import BAEZ_DUARTE_C

ap = argparse.ArgumentParser()
ap.add_argument("--N", type=int, default=3000)
ap.add_argument("--arb", type=int, default=0, help="largest N for the rigorous Arb solve (0 = skip)")
ap.add_argument("--tag", default="")
a = ap.parse_args()
N = a.N
os.makedirs("results/json", exist_ok=True); os.makedirs("results/figures", exist_ok=True)

t0 = time.time()
tab = nb.vasyunin_table(N, progress=lambda q: print(f"  V table q={q} [{time.time()-t0:.0f}s]", flush=True))
print(f"Vasyunin table N={N}: {time.time()-t0:.1f}s, {len(tab)} coprime pairs", flush=True)
t1 = time.time()
G = nb.gram_matrix(N, tab)
print(f"Gram matrix: {time.time()-t1:.1f}s", flush=True)
t1 = time.time()
d2, c = nb.baez_duarte_distances(N, G, return_coeffs=True)
print(f"Cholesky + distances: {time.time()-t1:.1f}s", flush=True)
ns = np.arange(1, N + 1)
prod = d2 * np.log(ns)
np.save(f"results/json/nyman_beurling_dN{a.tag}.npy", d2)
np.save(f"results/json/nyman_beurling_coeffs{a.tag}.npy", c)

# condition number estimate
ev_min = float(np.linalg.eigvalsh(G[:min(N, 2000), :min(N, 2000)])[0])
ev_max = float(np.linalg.eigvalsh(G[:min(N, 2000), :min(N, 2000)])[-1])

# Möbius mollifier upper bounds and a tapered variant
mob = {}
for n in [10, 30, 100, 300, 1000, 3000, 10000]:
    if n <= N:
        mob[n] = nb.mobius_mollifier_bound(n, G)

# empirical taper: compare optimal c_k with -mu(k) f(log k / log N)
from sympy import mobius
mu = np.array([int(mobius(int(k))) for k in ns], dtype=float)
sq_free = mu != 0
ratio = np.full(N, np.nan)
ratio[sq_free] = -c[sq_free] / mu[sq_free]
x = np.log(ns) / math.log(N)

checkpoints = [n for n in [10, 20, 50, 100, 200, 500, 1000, 2000, 3000, 5000, 8000, 10000, 15000, 20000] if n <= N]
summary = {
    "N": N,
    "C_burnol": BAEZ_DUARTE_C,
    "d2_at": {str(n): float(d2[n - 1]) for n in checkpoints},
    "d2_logN_at": {str(n): float(prod[n - 1]) for n in checkpoints},
    "min_d2_logN_over_range": float(prod[9:].min()),
    "argmin_d2_logN": int(ns[9:][prod[9:].argmin()]),
    "max_d2_logN_last_decade": float(prod[N // 10:].max()),
    "min_d2_logN_last_decade": float(prod[N // 10:].min()),
    "mobius_mollifier_bound": {str(k): v for k, v in mob.items()},
    "eig_min_block2000": ev_min, "eig_max_block2000": ev_max,
    "cond_block2000": ev_max / ev_min,
    "mean_abs_c_nonsquarefree": float(np.mean(np.abs(c[~sq_free]))) if (~sq_free).any() else None,
    "mean_abs_c_squarefree": float(np.mean(np.abs(c[sq_free]))),
    "sign_agreement_with_minus_mobius": float(np.mean(np.sign(c[sq_free]) == -mu[sq_free])),
    "wall_seconds": time.time() - t0,
}

# Fit d2 * log N = C + a/log N + b/log^2 N  on the last 90% of the range
sel = ns >= max(10, N // 10)
X = np.vstack([np.ones(sel.sum()), 1 / np.log(ns[sel]), 1 / np.log(ns[sel]) ** 2]).T
coef, *_ = np.linalg.lstsq(X, prod[sel], rcond=None)
summary["fit_d2logN_eq_C0_plus_a_over_logN_plus_b_over_log2N"] = {"C0": float(coef[0]), "a": float(coef[1]), "b": float(coef[2])}
X1 = np.vstack([np.ones(sel.sum()), 1 / np.log(ns[sel])]).T
coef1, *_ = np.linalg.lstsq(X1, prod[sel], rcond=None)
summary["fit_d2logN_eq_C0_plus_a_over_logN"] = {"C0": float(coef1[0]), "a": float(coef1[1])}

if a.arb:
    arbres = {}
    for n in [50, 100, 200, 400, 800, 1600]:
        if n <= a.arb:
            t1 = time.time()
            d2a, _ = nb.baez_duarte_distance_arb(n, dps=50)
            arbres[str(n)] = {"lower": float(d2a.lower()), "upper": float(d2a.upper()),
                              "float64": float(d2[n - 1]),
                              "abs_diff_float64_vs_rigorous": abs(float(d2[n - 1]) - float(d2a.mid())),
                              "enclosure_radius": float(d2a.rad()),
                              "seconds": time.time() - t1}
            print("arb", n, arbres[str(n)], flush=True)
    summary["rigorous_arb"] = arbres

with open(f"results/json/nyman_beurling_summary{a.tag}.json", "w") as f:
    json.dump(summary, f, indent=2)
print(json.dumps(summary, indent=2))

# ---- figures
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(8, 5))
ax.semilogx(ns[9:], prod[9:], lw=1, label=r"$d_N^2 \log N$ (computed)")
ax.axhline(BAEZ_DUARTE_C, color="C3", ls="--", label=r"Burnol constant $C=\sum_\rho |\rho|^{-2} = 2+\gamma-\log 4\pi$")
ax.set_xlabel("N"); ax.set_ylabel(r"$d_N^2\,\log N$")
ax.set_title("Báez-Duarte distances: $d_N^2 \\log N$ vs Burnol's lower-bound constant")
ax.set_ylim(0.9 * BAEZ_DUARTE_C, min(prod[9:].max() * 1.05, 3 * BAEZ_DUARTE_C))
ax.legend(); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig(f"results/figures/nyman_beurling_dN_logN{a.tag}.png", dpi=150)

fig, ax = plt.subplots(figsize=(8, 5))
ax.loglog(ns, d2, lw=1, label=r"$d_N^2$")
ax.loglog(ns[1:], BAEZ_DUARTE_C / np.log(ns[1:]), "--", label=r"$C/\log N$")
ax.set_xlabel("N"); ax.set_ylabel(r"$d_N^2$"); ax.legend(); ax.grid(alpha=.3)
ax.set_title("Báez-Duarte distances $d_N^2$ (RH $\\Leftrightarrow d_N \\to 0$)")
fig.tight_layout(); fig.savefig(f"results/figures/nyman_beurling_dN{a.tag}.png", dpi=150)

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(x[sq_free], ratio[sq_free], s=4, alpha=.5, label=r"$-c_k/\mu(k)$ (squarefree $k$)")
ax.plot(np.linspace(0, 1, 100), 1 - np.linspace(0, 1, 100), "C3--", label=r"Möbius mollifier taper $1 - \log k/\log N$")
ax.plot(np.linspace(0, 1, 100), (1 - np.linspace(0, 1, 100)) ** 2, "C2--", label=r"$(1-\log k/\log N)^2$")
ax.set_xlabel(r"$\log k / \log N$"); ax.set_ylabel("coefficient / Möbius sign")
ax.set_title(f"Optimal Nyman–Beurling coefficients vs the Möbius pattern (N={N})")
ax.set_ylim(-0.5, 1.5); ax.legend(); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig(f"results/figures/nyman_beurling_coefficients{a.tag}.png", dpi=150)
print("done", time.time() - t0)
