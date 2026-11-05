#!/usr/bin/env python3
"""Run the de Bruijn–Newman heat-flow experiments and write
results/json/debruijn_newman_summary.json, results/figures/debruijn_newman_flow.png,
results/figures/debruijn_newman_spacings.png.

    python3 scripts/run_debruijn_newman.py [--n-zeros 100000] [--quick]
"""
import argparse, json, math, sys, time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rh.data import load_zeros                     # noqa: E402
import rh.debruijn_newman as dn                    # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-zeros", type=int, default=100000)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    T0 = time.time()
    out = {"normalisation": "H_t(z) = int_0^inf e^{t u^2} Phi(u) cos(zu) du, zeros x = 2 gamma",
           "n_zeros_loaded": a.n_zeros}
    g = load_zeros(a.n_zeros)
    x = 2.0 * g

    # 1. H_0 vs xi
    t0 = time.time()
    out["H0_checks"] = dn.check_H0_vs_xi(zs=(0.0, 10.0, 30.0, 2 * g[0], 57.0, 90.0), dps=30)
    out["H0_checks_seconds"] = time.time() - t0
    print("H_0 checks:", json.dumps(out["H0_checks"], indent=1))

    # 2. velocity field: mean drift
    t0 = time.time()
    n_v = min(20000, a.n_zeros) if not a.quick else 3000
    v = dn.velocity_field(x[:n_v], 0.0, None, dn.tail_start(n_v))
    out["velocity_field"] = {"n_explicit": n_v, "median_first_half": float(np.median(v[:n_v // 2])),
                             "mean_first_half": float(np.mean(v[:n_v // 2])), "minus_pi_over_4": -math.pi / 4,
                             "v_first_five": v[:5].tolist(), "seconds": time.time() - t0}
    print("velocity field:", out["velocity_field"])

    # 3. Lehmer pairs among the loaded zeros
    t0 = time.time()
    pairs = dn.find_lehmer_pairs(x, n_best=12)
    out["lehmer_pairs_closest"] = pairs
    out["lehmer_pair_7005"] = dn.lehmer_pair_criterion(x, 6708)
    print("Lehmer pair 7005:", out["lehmer_pair_7005"])
    print("closest pairs:", [(p["k"], round(p["Delta"], 5), round(p["Delta2_g"], 5), p["t_two_body"]) for p in pairs])

    # 4. collision times: Lehmer pair at 7005 and the closest pair in the dataset
    cols = {}
    for label, k in (("lehmer_7005", 6708), ("closest_pair", int(np.argmin(np.diff(x))))):
        c = dn.estimate_collision_time(x, k, n_window=100 if not a.quick else 30)
        cols[label] = {"k": c.k, "gamma_pair": list(c.gamma_pair), "Delta0": c.Delta0,
                       "t_star_ode": c.t_star, "t_two_body": c.t_two_body,
                       "t_frozen_background": c.t_frozen_background, "S0": c.S0, "g_k": c.g_k,
                       "Delta2_g": c.Delta2_g, "is_lehmer_pair": c.is_lehmer_pair,
                       "window": list(c.window), "n_rhs": c.n_rhs, "status": c.status}
        print(f"collision {label}: t* = {c.t_star}  (two-body {c.t_two_body:.6e}, frozen {c.t_frozen_background:.6e})")
        if label == "lehmer_7005":
            col_lehmer = c
    out["collisions"] = cols
    out["lambda_lower_bound_estimate_from_dataset"] = max(v["t_star_ode"] for v in cols.values() if v["t_star_ode"] is not None)
    out["literature_lower_bounds_approx"] = {
        "Csordas-Smith-Varga_1994": -4.379e-6, "Csordas-Odlyzko-Smith-Varga_1993": -5.895e-9,
        "Odlyzko_2000": -2.7e-9, "Saouter-Gourdon-Demichel_2011": -1.15e-11,
        "Rodgers-Tao_2018": 0.0, "note": "values quoted from memory; Rodgers–Tao proved Lambda >= 0"}
    out["lehmer_seconds"] = time.time() - t0

    # 5. trajectories of zeros 1..30 for t in [-0.5, 0.5]
    t0 = time.time()
    # active set for the backward branch: the largest K <= 300 whose pairs all survive to t = -0.5
    d = np.diff(x)
    K = 30
    while K < 300 and np.min(d[:K + 1]) ** 2 / 8.0 > 0.6:
        K += 1
    n_frozen = 5000 if not a.quick else 1000
    xf = x[K:K + n_frozen]
    fwd = dn.zero_dynamics(x[:K], (0.0, 0.5), t_eval=np.linspace(0, 0.5, 101), x_frozen=xf, rtol=1e-9, atol=1e-11)
    bwd = dn.zero_dynamics(x[:K], (0.0, -0.5), t_eval=np.linspace(0, -0.5, 101), x_frozen=xf, rtol=1e-9, atol=1e-11)
    tt = np.concatenate([bwd.t[::-1], fwd.t[1:]])
    xx = np.concatenate([bwd.x[::-1], fwd.x[1:]])
    flow30 = dn.FlowResult(t=tt, x=xx, X_tail=fwd.X_tail, status=min(fwd.status, bwd.status),
                           message=f"fwd: {fwd.message} | bwd: {bwd.message}")
    out["flow_first_zeros"] = {"n_active": K, "n_frozen": n_frozen, "t_min": float(tt[0]), "t_max": float(tt[-1]),
                               "bwd_status": bwd.status, "fwd_status": fwd.status,
                               "min_spacing_t_minus_0.5": float(np.min(np.diff(xx[0]))),
                               "min_spacing_t_0": float(np.min(np.diff(x[:K]))),
                               "min_spacing_t_plus_0.5": float(np.min(np.diff(xx[-1]))),
                               "mean_displacement_t_plus_0.5": float(np.mean(xx[-1] - x[:K])),
                               "seconds": time.time() - t0}
    print("flow first zeros:", out["flow_first_zeros"])
    fig1 = ROOT / "results" / "figures" / "debruijn_newman_flow.png"
    dn.plot_flow(flow30, col_lehmer, fig1)
    print("wrote", fig1)

    # 6. spacing rigidity at t = 0.2 (Platt–Trudgian upper bound) for the first 2000 zeros
    t0 = time.time()
    rig = dn.flow_spacing_rigidity(g, n_active=2500 if not a.quick else 600, n_frozen=2500 if not a.quick else 600,
                                   n_stat=2000 if not a.quick else 500, t_final=0.2, t_eval=(0.0, 0.05, 0.1, 0.2))
    out["spacing_rigidity_t0.2"] = {k: v for k, v in rig.items() if k != "flow"}
    out["spacing_rigidity_t0.2"]["n_rhs"] = rig["flow"].n_rhs
    out["spacing_rigidity_t0.2"]["seconds"] = time.time() - t0
    print("rigidity: std ratio", rig["ratio_std"], "normalised std ratio", rig["ratio_norm_std"],
          "min spacing", rig["min_spacing"])
    fig2 = ROOT / "results" / "figures" / "debruijn_newman_spacings.png"
    dn.plot_spacing_histograms(rig, fig2, n_stat=2000 if not a.quick else 500)
    print("wrote", fig2)

    out["total_seconds"] = time.time() - T0
    jpath = ROOT / "results" / "json" / "debruijn_newman_summary.json"
    jpath.write_text(json.dumps(out, indent=2, default=float))
    print("wrote", jpath, f"({out['total_seconds']:.1f}s)")


if __name__ == "__main__":
    main()
