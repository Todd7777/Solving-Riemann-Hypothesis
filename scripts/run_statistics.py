#!/usr/bin/env python3
"""Spectral statistics of the zeta zeros versus GUE, on every zero dataset available.

Computes, for all stored low-height zeros (data/zeros/zeros_*.npy, loaded with
rh.data.load_zeros(None)) and for the high-height Arb sample
(data/zeros/sample_1e10.npy, zeros from index 10^10 at height ~3.3e9):

  * nearest-neighbour spacing histograms vs Wigner surmise / exact GUE / Poisson,
  * Kolmogorov–Smirnov distances to GUE in height windows (approach to GUE),
  * Montgomery pair correlation R_2(alpha) and form factor F(alpha),
  * number variance Sigma^2(L) with Mehta's GUE formula and Berry's saturation,
  * smallest normalized spacings (Lehmer phenomenon) and Lehmer-pair statistics.

Writes results/json/statistics_summary.json and four figures
results/figures/statistics_{spacing,pair_correlation,number_variance,height_dependence}.png.

Usage:  python3 scripts/run_statistics.py [--n-windows 200000] [--no-figures]
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from rh import statistics as st  # noqa: E402
from rh.data import available_datasets, load_zeros, manifest  # noqa: E402

JSON_DIR = REPO / "results" / "json"
FIG_DIR = REPO / "results" / "figures"

# Colour roles (reference data-viz palette, light surface).  Categorical slots in
# fixed order: slot 1 blue = low-height zeros, slot 2 orange = high-height sample,
# slot 3 aqua = Wigner surmise; exact GUE is drawn in primary ink and Poisson in
# the muted axis gray, since they are reference laws rather than data series.
C = {
    "surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
    "grid": "#e1e0d9", "axis": "#c3c2b7",
    "low": "#2a78d6", "high": "#eb6834", "wigner": "#1baf7a", "gue": "#0b0b0b", "poisson": "#898781",
}


# --------------------------------------------------------------------------- #
def jsonable(o):
    """Recursively convert numpy scalars/arrays to plain Python for json.dump."""
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return [jsonable(v) for v in o.tolist()]
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return f if math.isfinite(f) else None
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def semiclassical_height_averaged(L, gammas, n_heights=40):
    """Berry's Sigma^2_sc(L; T) averaged over the heights T of the dataset
    (sampled at quantiles of the zero index), for a dataset spanning many heights."""
    idx = np.linspace(0, len(gammas) - 1, n_heights + 2)[1:-1].astype(int)
    acc = np.zeros(len(L))
    for i in idx:
        acc += st.number_variance_semiclassical(L, float(gammas[i]))
    return acc / len(idx)


def analyse_dataset(name, g, start_index, L_grid, n_windows, bins, pc_bins, timings):
    """All statistics for one contiguous block of zeros; returns a dict."""
    out = {"name": name, "start_index": int(start_index), "n_zeros": int(g.size),
           "T_min": float(g[0]), "T_max": float(g[-1]),
           "logT_mid": math.log(math.sqrt(float(g[0]) * float(g[-1])))}
    T_ref = float(g[len(g) // 2])            # median height, used for Berry's scale
    out["T_ref"] = T_ref

    t0 = time.time()
    sp = st.spacing_distribution(g, bins=bins, s_max=3.5)
    out["spacing"] = sp
    d = st.normalized_spacings(g)
    out["spacing_density_method"] = {"mean": float(st.normalized_spacings(g, "density").mean())}
    out["ks_all"] = {
        "gue": st.ks_distance(d), "wigner": st.ks_distance(d, st.wigner_surmise_cdf),
        "poisson": st.ks_distance(d, st.poisson_cdf),
    }
    timings[f"{name}: spacings+KS"] = time.time() - t0

    t0 = time.time()
    pc = st.pair_correlation(g, alpha_max=5.0, bins=pc_bins)
    dev = pc["R2"] - pc["R2_gue"]
    sel = (pc["alpha"] > 0.9) & (pc["alpha"] < 1.1)
    pc_summary = {
        "max_abs_deviation": float(np.max(np.abs(dev))),
        "alpha_at_max_deviation": float(pc["alpha"][np.argmax(np.abs(dev))]),
        "rms_deviation": float(np.sqrt(np.mean(dev ** 2))),
        "rms_poisson_error": float(np.sqrt(np.mean(pc["se"] ** 2))),
        "chi2_per_bin": float(np.mean((dev / np.maximum(pc["se"], 1e-12)) ** 2)),
        "max_abs_deviation_alpha_0.9_1.1": float(np.max(np.abs(dev[sel]))),
        "R2_mean_alpha_gt_3": float(pc["R2"][pc["alpha"] > 3].mean()),
    }
    out["pair_correlation"] = pc
    out["pair_correlation_summary"] = pc_summary
    timings[f"{name}: pair correlation"] = time.time() - t0

    t0 = time.time()
    g_F = g[:100_000]                          # O(N * window * density) pairs; cap the cost
    alphas = np.linspace(0.0, 2.0, 41)
    T_F = float(g_F[-1])
    F_diag = st.montgomery_F(g_F, alphas, window=30.0, T=T_F, normalization="diagonal")
    ratio = g_F.size / ((T_F / (2 * np.pi)) * math.log(T_F))
    out["montgomery_F"] = {
        "alphas": alphas, "F_diagonal_normalized": F_diag,
        "F_montgomery": F_diag * ratio if start_index == 1 else None,   # Montgomery's normalisation needs all zeros up to T
        "prediction": st.montgomery_F_prediction(alphas, T_F),
        "prediction_gue_finite_height": st.montgomery_F_gue_finite(alphas, g_F, window=30.0, T=T_F),
        "n_zeros_used": int(g_F.size), "T": T_F, "window": 30.0,
        "N_over_T2pi_logT": ratio,
        "plateau_alpha_gt_1.2": float(F_diag[alphas > 1.2].mean()),
    }
    timings[f"{name}: Montgomery F"] = time.time() - t0

    t0 = time.time()
    nv = st.number_variance(g, L_grid, n_windows=n_windows)
    nv["berry_at_T_ref"] = st.number_variance_semiclassical(L_grid, T_ref)
    narrow = (float(g[-1]) - float(g[0])) / float(g[0]) < 1e-3       # a short window at one height
    nv["berry_height_averaged"] = nv["berry_at_T_ref"] if narrow else semiclassical_height_averaged(L_grid, g)
    sat = st.detect_saturation(nv["L"], nv["sigma2"], T=T_ref, se=nv["se"])
    out["number_variance"] = nv
    out["number_variance_saturation"] = sat
    timings[f"{name}: number variance"] = time.time() - t0

    t0 = time.time()
    out["smallest_spacings"] = st.lehmer_pairs(g, k=10, start_index=start_index)
    timings[f"{name}: Lehmer"] = time.time() - t0
    return out


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def _style(ax, title=None, xlabel=None, ylabel=None):
    ax.set_facecolor(C["surface"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(C["axis"])
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=C["muted"], labelcolor=C["ink2"], width=0.6, length=3, labelsize=8)
    ax.grid(True, color=C["grid"], linewidth=0.6, linestyle="-")
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, color=C["ink"], fontsize=10, loc="left", pad=8)
    if xlabel:
        ax.set_xlabel(xlabel, color=C["ink2"], fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=C["ink2"], fontsize=9)


def _legend(ax, **kw):
    leg = ax.legend(frameon=False, fontsize=8, labelcolor=C["ink2"], **kw)
    return leg


def _label(ds):
    n, lo, hi = ds["n_zeros"], ds["T_min"], ds["T_max"]
    if ds["start_index"] == 1:
        return f"zeros 1..{n:,}  (T ≤ {hi:.3g})"
    return f"zeros {ds['start_index']:.0e}.. (+{n:,}),  T ≈ {lo:.3g}"


def fig_spacing(datasets, path, plt):
    s = np.linspace(0, 3.5, 701)
    gue = st.gue_spacing_density(s)
    n = len(datasets)
    fig, axes = plt.subplots(1, n + 1, figsize=(5.2 * (n + 1), 4.2), facecolor=C["surface"])
    axes = np.atleast_1d(axes)
    for ax, ds in zip(axes[:n], datasets):
        sp = ds["spacing"]
        color = C[ds["role"]]
        ax.bar(sp["centers"], sp["density"], width=sp["edges"][1] - sp["edges"][0],
               color=color, alpha=0.30, linewidth=0, label=f"zeros, n = {sp['n']:,}")
        ax.step(sp["edges"], np.append(sp["density"], sp["density"][-1]), where="post",
                color=color, linewidth=1.0)
        ax.plot(s, gue, color=C["gue"], linewidth=1.3, label=r"exact GUE  $p_2(s) = E_2''(0;s)$")
        ax.plot(s, st.wigner_surmise(s), color=C["wigner"], linewidth=1.3, linestyle="--",
                label=r"Wigner surmise  $(32/\pi^2)\,s^2 e^{-4s^2/\pi}$")
        ax.plot(s, st.poisson_density(s), color=C["poisson"], linewidth=1.1, linestyle=":", label=r"Poisson  $e^{-s}$")
        ks = ds["ks_all"]["gue"]
        ax.text(0.98, 0.96, f"KS vs GUE: D = {ks['D']:.4f}\nvar(s) = {sp['var']:.4f}  (GUE {sp['var_gue']:.4f})",
                transform=ax.transAxes, ha="right", va="top", fontsize=8, color=C["ink2"])
        _style(ax, title=_label(ds), xlabel="normalized spacing s", ylabel="density")
        ax.set_xlim(0, 3.5)
        ax.set_ylim(0, 1.05)
        _legend(ax, loc="center right", bbox_to_anchor=(1.0, 0.55))
    ax = axes[-1]
    for ds in datasets:
        sp = ds["spacing"]
        width = sp["edges"][1] - sp["edges"][0]
        err = np.sqrt(np.maximum(sp["counts"], 1)) / (sp["n"] * width)
        ax.errorbar(sp["centers"], sp["density"] - sp["gue"], yerr=err, fmt="o", markersize=2.6,
                    color=C[ds["role"]], ecolor=C[ds["role"]], elinewidth=0.6, capsize=0,
                    label=_label(ds))
    ax.plot(s, st.wigner_surmise(s) - gue, color=C["wigner"], linewidth=1.3, linestyle="--", label="Wigner − GUE")
    ax.axhline(0, color=C["gue"], linewidth=1.0)
    _style(ax, title="deviation from exact GUE", xlabel="normalized spacing s", ylabel=r"density $-\ p_2(s)$")
    ax.set_xlim(0, 3.5)
    _legend(ax, loc="upper right")
    fig.suptitle("Nearest-neighbour spacings of zeta zeros: Montgomery–Odlyzko law",
                 color=C["ink"], fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=C["surface"])
    plt.close(fig)


def fig_pair_correlation(datasets, path, plt):
    a = np.linspace(0, 5, 1001)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=C["surface"])
    for ds in datasets:
        pc = ds["pair_correlation"]
        ax.errorbar(pc["alpha"], pc["R2"], yerr=pc["se"], fmt="o", markersize=2.6, color=C[ds["role"]],
                    ecolor=C[ds["role"]], elinewidth=0.6, capsize=0, label=_label(ds))
    ax.plot(a, st.pair_correlation_gue(a), color=C["gue"], linewidth=1.3,
            label=r"GUE / Montgomery  $1 - (\sin \pi\alpha / \pi\alpha)^2$")
    ax.axhline(1.0, color=C["poisson"], linewidth=1.0, linestyle=":", label="Poisson")
    _style(ax, title="pair correlation R₂(α) of unfolded zeros", xlabel="α (units of mean spacing)", ylabel="R₂(α)")
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 1.25)
    _legend(ax, loc="lower right")
    for ds in datasets:
        mf = ds["montgomery_F"]
        ax2.plot(mf["alphas"], mf["F_diagonal_normalized"], "o-", markersize=2.6, linewidth=1.0,
                 color=C[ds["role"]], label=f"{_label(ds)}, n = {mf['n_zeros_used']:,}")
        ax2.plot(mf["alphas"], mf["prediction_gue_finite_height"], linestyle="--", linewidth=1.0,
                 color=C[ds["role"]], alpha=0.85, label="GUE expectation at the data's height (same window)")
    a2 = np.linspace(0, 2, 401)
    ax2.plot(a2, st.montgomery_F_prediction(a2, datasets[0]["montgomery_F"]["T"]), color=C["gue"], linewidth=1.0,
             label=r"Montgomery asymptotics  $\alpha + T^{-2\alpha}\log T$ ($\alpha \leq 1$),  1 ($\alpha \geq 1$)")
    _style(ax2, title="Montgomery form factor F(α), diagonal-normalized", xlabel="α", ylabel=r"$F(\alpha)\,/\,F_{\rm diag}$")
    ax2.set_xlim(0, 2)
    ax2.set_ylim(0, 1.7)
    _legend(ax2, loc="upper right")
    fig.suptitle("Two-point statistics of zeta zeros vs GUE", color=C["ink"], fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=C["surface"])
    plt.close(fig)


def fig_number_variance(datasets, path, plt):
    fig, ax = plt.subplots(figsize=(7.5, 4.6), facecolor=C["surface"])
    L_dense = np.logspace(-1, 2, 400)
    ax.plot(L_dense, st.number_variance_gue(L_dense), color=C["gue"], linewidth=1.3, label="GUE (Mehta)")
    ax.plot(L_dense, st.number_variance_poisson(L_dense), color=C["poisson"], linewidth=1.0, linestyle=":", label="Poisson Σ² = L")
    for ds in datasets:
        nv, sat = ds["number_variance"], ds["number_variance_saturation"]
        col = C[ds["role"]]
        ok = np.isfinite(nv["sigma2"])
        ax.errorbar(nv["L"][ok], nv["sigma2"][ok], yerr=nv["se"][ok], fmt="o", markersize=2.8, color=col,
                    ecolor=col, elinewidth=0.6, capsize=0,
                    label=f"{_label(ds)}:  plateau {sat['plateau']:.3f} ± {sat['plateau_se']:.3f}")
        ax.plot(nv["L"][ok], nv["berry_height_averaged"][ok], linestyle="--", linewidth=1.0, color=col, alpha=0.85,
                label=f"Berry semiclassical prediction:  plateau {sat['plateau_berry']:.3f}")
        ax.axvline(sat["L_scale"], color=col, linewidth=0.8, linestyle=":", alpha=0.9)
        ax.annotate(f"L = log(T/2π)/2π = {sat['L_scale']:.2f}", xy=(sat["L_scale"], 0.985), xytext=(3, 0),
                    textcoords="offset points", rotation=90, fontsize=7, color=col, va="top")
    _style(ax, title="number variance Σ²(L): GUE at short range, Berry saturation beyond L ~ log(T/2π)/2π",
           xlabel="interval length L (mean spacings)", ylabel="Σ²(L)")
    ax.set_xscale("log")
    ax.set_xlim(0.1, 110)
    ax.set_ylim(0, 1.0)
    leg = _legend(ax, loc="lower right")
    leg.set_frame_on(True)
    leg.get_frame().set_facecolor(C["surface"])
    leg.get_frame().set_edgecolor("none")
    leg.get_frame().set_alpha(0.92)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=C["surface"])
    plt.close(fig)


def fig_height_dependence(ks_rows, path, plt):
    fig, ax = plt.subplots(figsize=(7.5, 4.4), facecolor=C["surface"])
    low = [r for r in ks_rows if r["role"] == "low"]
    high = [r for r in ks_rows if r["role"] == "high"]
    for rows, role, lab in ((low, "low", "windows of the low-height dataset"), (high, "high", "Arb sample at index 10¹⁰")):
        if not rows:
            continue
        x = np.array([r["logT_mid"] for r in rows])
        y = np.array([r["ks_gue"] for r in rows])
        nf = np.array([r["noise_floor"] for r in rows])
        ax.errorbar(x, y, yerr=nf, fmt="o", markersize=4.5, color=C[role], ecolor=C[role], elinewidth=0.8,
                    capsize=2, label=f"KS distance to exact GUE, {lab}")
        ax.plot(x, nf, marker="_", markersize=9, linestyle="none", color=C["muted"],
                label="sampling noise floor 0.87/√n" if role == "low" else None)
    if low:
        x = np.array([r["logT_mid"] for r in low])
        y = np.array([r["ks_gue"] for r in low])
        xx = np.linspace(min(x) - 0.5, max([r["logT_mid"] for r in ks_rows]) + 0.5, 200)
        c1 = np.sum(y / x) / np.sum(1 / x ** 2)          # least squares  D = c1 / log T
        c2 = np.sum(y / x ** 2) / np.sum(1 / x ** 4)     # least squares  D = c2 / (log T)^2
        ax.plot(xx, c1 / xx, color=C["ink2"], linewidth=0.9, linestyle="--", label=f"{c1:.2f} / log T")
        ax.plot(xx, c2 / xx ** 2, color=C["ink2"], linewidth=0.9, linestyle="-.", label=f"{c2:.1f} / (log T)²")
    for r in ks_rows:
        ax.annotate(r["label"], xy=(r["logT_mid"], r["ks_gue"]), xytext=(8, 7), textcoords="offset points",
                    fontsize=7, color=C["ink2"])
    _style(ax, title="approach to GUE with height: KS distance of the spacing law", xlabel="log T", ylabel="KS distance D")
    ax.set_ylim(0, max(0.04, 1.3 * max(r["ks_gue"] for r in ks_rows)))
    _legend(ax, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=C["surface"])
    plt.close(fig)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-windows", type=int, default=200_000, help="start positions for the number variance")
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--sample", default="sample_1e10", help="name of the high-height sample in data/zeros")
    args = ap.parse_args()
    JSON_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    timings: dict[str, float] = {}
    t_start = time.time()

    # ---- data ------------------------------------------------------------- #
    sets = available_datasets()
    g_low = load_zeros(None)
    print(f"low-height zeros: {g_low.size:,} (gamma_1 = {g_low[0]:.6f} .. gamma_N = {g_low[-1]:.3f}) from "
          + ", ".join(p.name for _, _, p in sets))
    g_high, man_high = st.load_high_sample(args.sample)
    if g_high is not None:
        print(f"high sample: {g_high.size:,} zeros from index {man_high.get('first_index')} at T ≈ {g_high[0]:.6e}, "
              f"max radius {man_high.get('max_radius')}")
    else:
        print("high sample not found; skipping")

    L_grid = np.concatenate([np.arange(0.1, 1.0, 0.1), np.arange(1.0, 10.0, 0.5), np.arange(10.0, 100.1, 2.5)])

    datasets = []
    ds_low = analyse_dataset("low", g_low, 1, L_grid, args.n_windows, bins=70, pc_bins=100, timings=timings)
    ds_low["role"] = "low"
    ds_low["sources"] = [{"file": p.name, **manifest(p)} for _, _, p in sets]
    datasets.append(ds_low)
    if g_high is not None:
        ds_high = analyse_dataset("high", g_high, int(man_high.get("first_index", 10 ** 10)), L_grid,
                                  args.n_windows, bins=35, pc_bins=50, timings=timings)
        ds_high["role"] = "high"
        ds_high["manifest"] = man_high
        datasets.append(ds_high)

    # ---- KS by height ----------------------------------------------------- #
    t0 = time.time()
    ks_rows = []
    for r in st.ks_by_height(g_low):
        r["role"] = "low"
        ks_rows.append(r)
    if g_high is not None:
        fi = int(man_high.get("first_index", 10 ** 10))
        for r in st.ks_by_height(g_high, windows=[(fi, fi + g_high.size - 1)], start_index=fi,
                                 label=f"zeros {fi:.0e}..+{g_high.size}"):
            r["role"] = "high"
            ks_rows.append(r)
    agg_windows = [(1, min(10 ** 5, g_low.size))] + ([(1, g_low.size)] if g_low.size > 10 ** 5 else [])
    ks_aggregate = st.ks_by_height(g_low, windows=agg_windows)
    timings["KS by height"] = time.time() - t0

    # ---- summary ---------------------------------------------------------- #
    sp_low = ds_low["spacing"]
    summary = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "python": platform.python_version(), "numpy": np.__version__,
        "datasets": {ds["name"]: {k: v for k, v in ds.items() if k not in
                                  ("spacing", "pair_correlation", "number_variance", "montgomery_F")}
                     for ds in datasets},
        "spacing_moments": {ds["name"]: {"n": ds["spacing"]["n"], "mean": ds["spacing"]["mean"], "var": ds["spacing"]["var"],
                                         "var_gue": ds["spacing"]["var_gue"], "var_wigner": ds["spacing"]["var_wigner"],
                                         "min": ds["spacing"]["min"], "max": ds["spacing"]["max"]} for ds in datasets},
        "ks_by_height": ks_rows,
        "ks_aggregate": ks_aggregate,
        "gue_reference": {"E2_at_1": float(st.gue_gap_probability(1.0)),
                          "p2_max": float(np.max(st.gue_spacing_density(np.linspace(0, 3, 3001)))),
                          "p2_argmax": float(np.linspace(0, 3, 3001)[np.argmax(st.gue_spacing_density(np.linspace(0, 3, 3001)))]),
                          "variance": float(st.gue_table().variance),
                          "max_abs_dev_from_wigner": float(np.max(np.abs(st.gue_spacing_density(np.linspace(0, 4, 4001))
                                                                        - st.wigner_surmise(np.linspace(0, 4, 4001)))))},
        "timings_seconds": timings,
        "notes": [
            "Unfolding: x_n = theta(gamma_n)/pi + 1 with the Riemann–Siegel theta asymptotic expansion (long double).",
            "Exact GUE spacing law from the sine-kernel Fredholm determinant (Bornemann Gauss–Legendre Nyström, m=60) "
            "with Chebyshev-interpolated derivatives; max |p_GUE - Wigner| ≈ 0.005.",
            "KS p-values assume independent spacings; consecutive zeta spacings are weakly correlated, so they are indicative only.",
            "Montgomery F(alpha) is truncated to pairs with |gamma - gamma'| <= 30 and computed on at most the first 1e5 zeros; "
            "the 'diagonal' normalization divides by N instead of (T/2pi) log T.",
            "Berry's semiclassical Sigma^2 uses the diagonal approximation with exact prime powers up to 1e6 and the PNT integral "
            "beyond; for the low-height dataset it is averaged over the heights spanned by the data.",
            "Lehmer pairs: Csordas–Smith–Varga statistic Delta < 4/5 evaluated with the available zeros (and their conjugates).",
        ],
    }
    # full arrays for the figures / further analysis
    summary["spacing_histograms"] = {ds["name"]: ds["spacing"] for ds in datasets}
    summary["pair_correlation"] = {ds["name"]: ds["pair_correlation"] for ds in datasets}
    summary["montgomery_F"] = {ds["name"]: ds["montgomery_F"] for ds in datasets}
    summary["number_variance"] = {ds["name"]: ds["number_variance"] for ds in datasets}

    # ---- figures ---------------------------------------------------------- #
    if not args.no_figures:
        t0 = time.time()
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams.update({"font.family": "sans-serif", "text.color": C["ink"], "axes.labelcolor": C["ink2"]})
        fig_spacing(datasets, FIG_DIR / "statistics_spacing.png", plt)
        fig_pair_correlation(datasets, FIG_DIR / "statistics_pair_correlation.png", plt)
        fig_number_variance(datasets, FIG_DIR / "statistics_number_variance.png", plt)
        fig_height_dependence(ks_rows, FIG_DIR / "statistics_height_dependence.png", plt)
        timings["figures"] = time.time() - t0
        summary["figures"] = [str((FIG_DIR / f"statistics_{n}.png").relative_to(REPO))
                              for n in ("spacing", "pair_correlation", "number_variance", "height_dependence")]

    timings["total"] = time.time() - t_start
    out = JSON_DIR / "statistics_summary.json"
    out.write_text(json.dumps(jsonable(summary), indent=1))

    # ---- report ----------------------------------------------------------- #
    print("\n=== spacing distribution ===")
    for ds in datasets:
        sp, ks = ds["spacing"], ds["ks_all"]
        print(f"{_label(ds)}: n={sp['n']:,} mean={sp['mean']:.5f} var={sp['var']:.4f} (GUE {sp['var_gue']:.4f}); "
              f"KS vs GUE {ks['gue']['D']:.4f} at s={ks['gue']['s_at']:.2f} (noise floor {ks['gue']['noise_floor']:.4f}), "
              f"vs Wigner {ks['wigner']['D']:.4f}, vs Poisson {ks['poisson']['D']:.3f}")
    print("\n=== KS distance to GUE by height window ===")
    for r in ks_rows:
        print(f"{r['label']:>28}: n={r['n_spacings']:>7,} log T={r['logT_mid']:6.2f}  D={r['ks_gue']:.4f} "
              f"(floor {r['noise_floor']:.4f}, p={r['ks_gue_pvalue']:.2g}) at s={r['ks_gue_s_at']:.2f} sign {r['ks_gue_sign']:+d}; "
              f"Wigner {r['ks_wigner']:.4f}; var={r['var_spacing']:.4f}")
    print("\n=== pair correlation ===")
    for ds in datasets:
        s = ds["pair_correlation_summary"]
        print(f"{_label(ds)}: max|R2-GUE|={s['max_abs_deviation']:.4f} at alpha={s['alpha_at_max_deviation']:.2f}; "
              f"rms dev {s['rms_deviation']:.4f} vs Poisson error {s['rms_poisson_error']:.4f} (chi2/bin {s['chi2_per_bin']:.2f}); "
              f"max dev on [0.9,1.1] {s['max_abs_deviation_alpha_0.9_1.1']:.4f}; R2(alpha>3) mean {s['R2_mean_alpha_gt_3']:.4f}")
        mf = ds["montgomery_F"]
        print(f"   Montgomery F (diag-normalized, n={mf['n_zeros_used']:,}): plateau alpha>1.2 = {mf['plateau_alpha_gt_1.2']:.3f}; "
              f"F(0.5)={mf['F_diagonal_normalized'][10]:.3f} (asymptotic {mf['prediction'][10]:.3f}, "
              f"finite-height GUE {mf['prediction_gue_finite_height'][10]:.3f}); N/((T/2pi)logT)={mf['N_over_T2pi_logT']:.3g}")
    print("\n=== number variance ===")
    for ds in datasets:
        sat = ds["number_variance_saturation"]
        print(f"{_label(ds)}: plateau {sat['plateau']:.4f} ± {sat['plateau_se']:.4f} (Berry {sat['plateau_berry']:.4f}); "
              f"GUE at L={sat['L_max_tested']:.0f}: {sat['gue_at_L_max']:.3f}; L_cross={sat['L_cross']:.2f}, "
              f"L_depart(3σ)={sat['L_depart']}, L_scale=log(T/2π)/2π={sat['L_scale']:.2f}; saturated={sat['saturated']}")
    print("\n=== smallest normalized spacings (Lehmer phenomenon) ===")
    for ds in datasets:
        for r in ds["smallest_spacings"][:5]:
            print(f"  {ds['name']}: n={r['index']}: gamma={r['gamma_n']:.6f}, {r['gamma_n1']:.6f}  gap={r['gap']:.6f} "
                  f"normalized={r['normalized_spacing']:.5f}  CSV Delta={r['csv_delta']:.4f} Lehmer={r['is_lehmer_pair']}")
    print(f"\nwrote {out} ; total {timings['total']:.1f}s")


if __name__ == "__main__":
    main()
