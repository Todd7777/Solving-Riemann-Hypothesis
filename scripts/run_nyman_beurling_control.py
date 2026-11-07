#!/usr/bin/env python3
"""Experiment E11: how sensitive is the Nyman–Beurling criterion to zeros off the line?

Controls: two Dirichlet series with period-5 coefficients and the same conductor,
  * L(s, (n/5))           — zeros on the critical line (GRH, verified numerically),
  * Davenport–Heilbronn f — zeros off the line (first: 0.8085 + 85.70i).
For both we compute the Báez-Duarte-type distances d_N^2 (closed-form Gram
matrix, see rh.nyman_beurling_periodic) and compare with the model
d_N^2 ~ C_on/log N + floor, where C_on = sum over on-line zeros of 1/|rho|^2 and
floor = the exact lower bound forced by the off-line zeros.  For zeta itself we
tabulate the floor and the "detection horizon" a hypothetical off-line zero
would produce, next to Li's criterion's horizon.

Usage: python3 scripts/run_nyman_beurling_control.py [--N 2000]
"""
import argparse, json, math, os, sys, time
import numpy as np
import mpmath as mp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from rh.nyman_beurling_periodic import CHI5, DAVENPORT_HEILBRONN as DH, offline_floor, gram_entry_direct_mp
from rh import nyman_beurling as nb, li

ap = argparse.ArgumentParser()
ap.add_argument("--N", type=int, default=2000)
a = ap.parse_args()
N = a.N
os.makedirs("results/json", exist_ok=True); os.makedirs("results/figures", exist_ok=True)
T0 = time.time()
mp.mp.dps = 30
out = {"N": N}

# ---- validation of the closed-form Gram entries
worst = 0.0
for S in (CHI5, DH):
    for h, k in [(1, 1), (1, 2), (2, 3), (3, 5), (4, 7), (5, 6), (7, 11), (3, 13)]:
        worst = max(worst, abs(S.gram_coprime(h, k) - float(gram_entry_direct_mp(S, h, k))))
out["gram_closed_form_vs_direct_max_abs_diff"] = worst
print("Gram closed form vs direct: max diff", worst, flush=True)

# ---- constants
dh_json = json.load(open("results/json/davenport_heilbronn_zeros.json"))
off = [mp.mpc(z["re"], z["im"]) for z in dh_json["off_line_zeros"]]
on = [float(g) for g in dh_json["on_line_zeros"]]
C_chi5 = CHI5.zero_sum_constant()
S_all_dh = DH.zero_sum_constant()
S_off_dh = sum(4 * mp.re(1 / (r * (1 - r))) for r in off)
C_on_dh = S_all_dh - S_off_dh
direct_on = sum(2 / (0.25 + g * g) for g in on) + float(2 * mp.quad(lambda t: mp.log(5 * t / (2 * mp.pi)) / (2 * mp.pi) / (0.25 + t * t), [dh_json["T"], mp.inf]))
floor_dh = offline_floor(off + [mp.conj(z) for z in off])
out["constants"] = {
    "C_chi5 = 2 Lambda'/Lambda(1)": float(C_chi5),
    "L(1,chi5)": float(CHI5.F1()), "L(1,chi5) closed form 2 log(golden)/sqrt5": float(2 * mp.log((1 + mp.sqrt(5)) / 2) / mp.sqrt(5)),
    "DH sum over all zeros of 1/(rho(1-rho))": float(S_all_dh),
    "DH off-line quadruples below t=300": float(S_off_dh),
    "DH C_on (Hadamard formula minus off-line)": float(C_on_dh),
    "DH C_on (direct: 204 on-line zeros + density tail)": direct_on,
    "DH off-line zeros used": [mp.nstr(z, 12) for z in off],
    "DH floor (exact interpolation bound, all N)": floor_dh,
    "log N at which DH floor equals C_on/log N": float(C_on_dh) / floor_dh,
}
print(json.dumps(out["constants"], indent=1), flush=True)

# ---- distances
res = {}
for S in (CHI5, DH):
    t0 = time.time()
    G = S.gram_matrix(N, progress=lambda j: print(f"   {S.name}: row {j}", flush=True) if j % 500 == 0 else None)
    tg = time.time() - t0
    ev = np.linalg.eigvalsh(G[:min(N, 1500), :min(N, 1500)])
    d2, c = S.distances(N, G, return_coeffs=True)
    np.save(f"results/json/nyman_beurling_control_{'chi5' if S is CHI5 else 'dh'}_dN.npy", d2)
    ns = np.arange(1, N + 1)
    prod = d2 * np.log(np.maximum(ns, 2))
    checkpoints = [n for n in [10, 30, 100, 300, 1000, 2000, 3000, 5000] if n <= N]
    res[S.name] = {"gram_seconds": tg, "cond_block1500": float(ev[-1] / ev[0]),
                   "d2": {str(n): float(d2[n - 1]) for n in checkpoints},
                   "d2_logN": {str(n): float(prod[n - 1]) for n in checkpoints},
                   "d2_logN_range_last_half": [float(prod[N // 2:].min()), float(prod[N // 2:].max())]}
    print(S.name, json.dumps(res[S.name]), flush=True)
out["distances"] = res

# ---- zeta: floors and detection horizons of hypothetical off-line zeros, vs Li
tab = []
for g, d in [(14.134725, 0.1), (100.0, 0.1), (100.0, 0.4), (1e3, 0.1), (1e4, 0.1), (3e12, 0.1)]:
    fl = nb.offline_floor_zeta([complex(0.5 + d, g), complex(0.5 + d, -g)])
    lnN = nb.detection_horizon_log_N(g, d)
    tab.append({"gamma": g, "delta": d, "floor": fl, "NB_log_N_star": lnN, "NB_log10_N_star": lnN / math.log(10),
                "Li_n_star": li.detection_horizon(g, d)})
out["zeta_hypothetical_zero_horizons"] = tab
out["wall_seconds"] = time.time() - T0
json.dump(out, open("results/json/nyman_beurling_control_summary.json", "w"), indent=1)

# ---- figures
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, axs = plt.subplots(1, 2, figsize=(13, 5))
ns = np.arange(1, N + 1)
for S, col, Cline in ((CHI5, "C0", float(C_chi5)), (DH, "C3", float(C_on_dh))):
    d2 = np.load(f"results/json/nyman_beurling_control_{'chi5' if S is CHI5 else 'dh'}_dN.npy")
    axs[0].semilogx(ns[9:], d2[9:] * np.log(ns[9:]), color=col, lw=1, label=f"{S.name}: $d_N^2\\log N$")
    axs[0].axhline(Cline, color=col, ls="--", lw=0.8)
axs[0].plot(ns[9:], float(C_on_dh) + floor_dh * np.log(ns[9:]), color="C3", ls=":", lw=1,
            label="DH model $C_{on}$ + floor$\\cdot\\log N$")
axs[0].set_xlabel("N"); axs[0].set_ylabel(r"$d_N^2 \log N$")
axs[0].set_title("Nyman–Beurling distances: GRH holds (blue) vs RH fails (red)", fontsize=10)
axs[0].legend(fontsize=8); axs[0].grid(alpha=.3)
L = np.logspace(0.5, 3.2, 400)          # log N from ~3 to ~1600
axs[1].loglog(L, float(C_on_dh) / L, "C3--", lw=1, label=r"DH on-line part $C_{on}/\log N$")
axs[1].loglog(L, float(C_on_dh) / L + floor_dh, "C3", lw=1.5, label="DH model with off-line floor")
axs[1].axhline(floor_dh, color="k", lw=0.8, ls=":", label=f"rigorous floor from 5 off-line zero pairs = {floor_dh:.2e}")
for S, col in ((CHI5, "C0"), (DH, "C3")):
    d2 = np.load(f"results/json/nyman_beurling_control_{'chi5' if S is CHI5 else 'dh'}_dN.npy")
    axs[1].loglog(np.log(ns[9:]), d2[9:], color=col, lw=2, alpha=.6)
axs[1].axvspan(np.log(10), np.log(N), color="gray", alpha=0.12, label="computed range")
axs[1].set_xlabel(r"$\log N$"); axs[1].set_ylabel(r"$d_N^2$")
axs[1].set_title(r"The off-line zeros only matter once $\log N \gtrsim C_{on}/\mathrm{floor} \approx 700$, i.e. $N \approx 10^{300}$", fontsize=10)
axs[1].legend(fontsize=8); axs[1].grid(alpha=.3)
fig.tight_layout(); fig.savefig("results/figures/nyman_beurling_control.png", dpi=150)
print("done", time.time() - T0)
