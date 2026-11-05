#!/usr/bin/env python3
"""Second-order structure of the Báez-Duarte distances:
(1) where d_N^2 drops (arithmetic of N); (2) spectrum of the oscillation of
d_N^2 log N - C in u = log N (are the zeros visible?); (3) the empirical taper of
the optimal coefficients c_k ~ -mu(k) f(log k / log N) and the quality of the
best mollifier of that shape (an explicit upper bound for d_N^2).
Usage: python3 scripts/analyze_nb_structure.py --tag _N2000
"""
import argparse, json, math, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from rh.constants import BAEZ_DUARTE_C
from rh import nyman_beurling as nb
from rh.data import load_zeros
import sympy

ap = argparse.ArgumentParser(); ap.add_argument("--tag", default="_N2000"); ap.add_argument("--rebuild-gram", action="store_true")
a = ap.parse_args()
d2 = np.load(f"results/json/nyman_beurling_dN{a.tag}.npy"); c = np.load(f"results/json/nyman_beurling_coeffs{a.tag}.npy")
N = len(d2); ns = np.arange(1, N + 1); logn = np.log(ns)
prod = d2 * logn
out = {"N": N, "C": BAEZ_DUARTE_C}

# (1) drops by arithmetic class
rel = -np.diff(d2) / d2[:-1]
mu = np.array([int(sympy.mobius(int(n))) for n in ns], dtype=int)
omega = np.array([len(sympy.factorint(int(n))) for n in ns], dtype=int)
isp = np.array([sympy.isprime(int(n)) for n in ns])
cls = {}
m = np.arange(2, N + 1)
for name, sel in [("prime", isp[1:]), ("squarefree_omega2", (mu[1:] != 0) & (omega[1:] == 2)),
                  ("squarefree_omega3", (mu[1:] != 0) & (omega[1:] == 3)), ("nonsquarefree", mu[1:] == 0)]:
    cls[name] = {"count": int(sel.sum()), "mean_rel_drop": float(rel[sel].mean()), "median_rel_drop": float(np.median(rel[sel]))}
out["drops_by_class"] = cls
# share of the total decrease of d2 from N=10 to N contributed by primes
tot = d2[9] - d2[-1]
prime_share = float((-np.diff(d2))[9:][isp[10:]].sum() / tot)
out["share_of_decrease_from_primes_N_ge_10"] = prime_share

# (2) spectrum of osc(u) = d_N^2 log N - C, u = log N, N >= 50, on a uniform u grid
sel = ns >= 50
u = logn[sel]; osc = prod[sel] - BAEZ_DUARTE_C
ug = np.linspace(u[0], u[-1], 4096); og = np.interp(ug, u, osc)
og = og - og.mean()
win = np.hanning(len(og)); F = np.fft.rfft(og * win); freqs = 2 * np.pi * np.fft.rfftfreq(len(ug), d=ug[1] - ug[0])
P = np.abs(F) ** 2
peaks = []
for i in range(1, len(P) - 1):
    if P[i] > P[i - 1] and P[i] > P[i + 1] and freqs[i] > 1.0:
        peaks.append((float(freqs[i]), float(P[i])))
peaks.sort(key=lambda x: -x[1])
gam = load_zeros(20)
out["osc_spectrum_top_peaks_omega"] = peaks[:8]
out["first_zeros_gamma"] = [float(g) for g in gam[:5]]
out["note_spectrum"] = "angular frequencies omega in u=log N; compare with gamma_n (N^{i gamma}) and gamma_n/2"

# (3) empirical taper: c_k = -mu(k) f(x), x = log k / log N; bin by x
x = logn / math.log(N); sq = mu != 0
r = -c[sq] / mu[sq]; xs = x[sq]
bins = np.linspace(0, 1, 21); centers = 0.5 * (bins[1:] + bins[:-1])
med = [float(np.median(r[(xs >= bins[i]) & (xs < bins[i + 1])])) if ((xs >= bins[i]) & (xs < bins[i + 1])).any() else None for i in range(20)]
out["taper_median_by_x_bin"] = dict(zip([round(float(cc), 3) for cc in centers], med))
# fit f(x) = (1-x)^alpha on x in [0.3, 0.98] via least squares in log
msk = (xs > 0.3) & (xs < 0.98) & (r > 0)
alpha = float(np.polyfit(np.log(1 - xs[msk]), np.log(r[msk]), 1)[0])
out["taper_powerlaw_alpha_fit"] = alpha

# quality of Möbius mollifiers with tapers (1-x)^alpha: U^2 = 1 - 2 b.c + c G c
G = nb.gram_matrix(N) if (N <= 3000 or a.rebuild_gram) else None
if G is not None:
    b = nb.rhs_vector(N)
    res = {}
    for al in [1, 2, 3, alpha, 4, 6]:
        cc = -mu * np.clip(1 - x, 0, None) ** al
        res[f"alpha={al:.3f}"] = float(1 - 2 * b @ cc + cc @ G @ cc)
    # best alpha by golden search
    def U(al):
        cc = -mu * np.clip(1 - x, 0, None) ** al
        return float(1 - 2 * b @ cc + cc @ G @ cc)
    lo, hi = 0.5, 12.0
    for _ in range(40):
        m1 = lo + (hi - lo) * 0.382; m2 = lo + (hi - lo) * 0.618
        if U(m1) < U(m2): hi = m2
        else: lo = m1
    out["mollifier_bounds_U2"] = res
    out["best_taper_alpha"] = float(0.5 * (lo + hi)); out["best_taper_U2"] = U(0.5 * (lo + hi))
    out["d2_N"] = float(d2[-1]); out["ratio_bestU2_over_d2"] = U(0.5 * (lo + hi)) / float(d2[-1])
    out["best_taper_U2_logN"] = U(0.5 * (lo + hi)) * math.log(N)

json.dump(out, open(f"results/json/nyman_beurling_structure{a.tag}.json", "w"), indent=2)
print(json.dumps(out, indent=2))

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
axs[0].plot(freqs[freqs < 40], P[freqs < 40] / P[freqs < 40].max(), lw=.8)
for g in gam[:4]:
    axs[0].axvline(g, color="C3", ls=":", lw=.8); axs[0].axvline(g / 2, color="C2", ls=":", lw=.8)
axs[0].set_xlabel(r"angular frequency $\omega$ in $u=\log N$"); axs[0].set_ylabel("normalised power")
axs[0].set_title(r"Spectrum of $d_N^2\log N - C$; dotted: $\gamma_n$ (red), $\gamma_n/2$ (green)")
axs[1].scatter(xs, r, s=3, alpha=.4, label=r"$-c_k/\mu(k)$")
xx = np.linspace(0, 1, 200); axs[1].plot(xx, (1 - xx) ** alpha, "C3", label=rf"$(1-x)^{{{alpha:.2f}}}$ (fit)")
axs[1].plot(xx, 1 - xx, "C2--", label=r"$1-x$ (Bettin–Conrey–Farmer)")
axs[1].set_xlabel(r"$x=\log k/\log N$"); axs[1].set_ylim(-0.3, 1.3); axs[1].legend(); axs[1].set_title(f"Empirical taper of the optimal coefficients, N={N}")
fig.tight_layout(); fig.savefig(f"results/figures/nyman_beurling_structure{a.tag}.png", dpi=150)
