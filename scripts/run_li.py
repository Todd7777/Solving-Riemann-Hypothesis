#!/usr/bin/env python3
"""Compute Li/Keiper coefficients lambda_n two independent ways and summarise.

1. From the rigorously computed zeros in data/zeros (all that are available,
   via rh.data.load_zeros(None)) plus the Riemann-von Mangoldt integral tail,
   for n <= --nmax (default 10000; reduced automatically to fit --time-budget).
2. Rigorous enclosures with Arb (python-flint) from the Taylor series of log xi
   at s = 1, for n <= --hp-max (default 5000).

Then: cross-check (1) against (2), check positivity (Li's criterion), fit the
asymptotic law lambda_n ~ (n/2) log n + c1 n + c0 and compare with the predicted
c1 = (gamma - 1 - log 2 pi)/2, c0 = 3/4, and write

    results/json/li_summary.json
    results/figures/li_coefficients.png
    results/figures/li_oscillation.png

Usage: python3 scripts/run_li.py [--nmax 10000] [--hp-max 5000] [--time-budget 600]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from rh import li                                  # noqa: E402
from rh.constants import LI_LAMBDA_1, EULER_GAMMA, LOG_2PI  # noqa: E402
from rh.data import load_zeros, available_datasets, manifest  # noqa: E402

# ----------------------------------------------------------------------------- style
# Reference palette of the dataviz skill (light mode): categorical slots 1-2,
# chart chrome in text tokens, hairline solid grid.
C = {
    "surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
    "grid": "#e1e0d9", "axis": "#c3c2b7", "s1": "#2a78d6", "s2": "#eb6834", "s3": "#1baf7a",
}


def _style_axes(ax):
    ax.set_facecolor(C["surface"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(C["axis"])
        ax.spines[side].set_linewidth(0.8)
    ax.grid(True, color=C["grid"], linewidth=0.8, linestyle="-")
    ax.set_axisbelow(True)
    ax.tick_params(colors=C["muted"], labelcolor=C["ink2"], length=3, width=0.8)
    ax.xaxis.label.set_color(C["ink2"])
    ax.yaxis.label.set_color(C["ink2"])
    ax.title.set_color(C["ink"])


def _figure(nrows, ncols, figsize):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Segoe UI", "Helvetica", "Arial", "sans-serif"],
        "font.size": 9.5, "axes.titlesize": 10.5, "axes.labelsize": 9.5,
        "legend.fontsize": 9, "legend.frameon": False,
        "figure.facecolor": C["surface"], "savefig.facecolor": C["surface"],
        "axes.titlelocation": "left", "axes.titleweight": "bold",
    })
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    return plt, fig, axes


# ----------------------------------------------------------------------------- figures
def plot_coefficients(ns, lam, smooth, hp_ns, hp_lam, path):
    """Left: lambda_n and its smooth part (log-log).  Right: running estimate of c1."""
    plt, fig, (ax1, ax2) = _figure(1, 2, (11, 4.3))
    for ax in (ax1, ax2):
        _style_axes(ax)
    # left panel: two series -> legend + selective direct labels at the right end
    ax1.plot(ns, lam, color=C["s1"], lw=2, solid_joinstyle="round", solid_capstyle="round",
             label=r"$\lambda_n$ (sum over zeros + tail)")
    m = ns >= 10                      # the smooth part is negative for n <= 8 (log axis)
    ax1.plot(ns[m], smooth[m], color=C["s2"], lw=2, alpha=0.9,
             label=r"smooth part $\frac{n}{2}(H_n - 1 - \log 2\pi) + \frac{1}{2}$  (n ≥ 10)")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("n")
    ax1.set_ylabel(r"$\lambda_n$")
    ax1.set_title(f"Li coefficients, n ≤ {int(ns[-1]):,}: all positive, min = λ₁ = {lam[0]:.6f}")
    ax1.legend(loc="upper left", labelcolor=C["ink2"])
    ax1.annotate(f"λ₁ = {lam[0]:.4f}", (ns[0], lam[0]), xytext=(8, -4), textcoords="offset points",
                 color=C["ink2"], fontsize=8.5, va="top")
    ax1.annotate(f"λ_{int(ns[-1])} = {lam[-1]:,.1f}", (ns[-1], lam[-1]), xytext=(-6, 6),
                 textcoords="offset points", color=C["ink2"], fontsize=8.5, ha="right", va="bottom")
    # right panel: c(n) = (lambda_n - (n/2) log n - 3/4) / n -> c1
    c_run = (lam - 0.5 * ns * np.log(ns) - li.LI_C0) / ns
    ax2.plot(ns, c_run, color=C["s1"], lw=2, label=r"$(\lambda_n - \frac{n}{2}\log n - \frac{3}{4})/n$")
    ax2.axhline(li.LI_C1, color=C["s2"], lw=2, label=r"$\frac{1}{2}(\gamma - 1 - \log 2\pi)$ = %.5f" % li.LI_C1)
    ax2.set_xscale("log")
    ax2.set_xlabel("n")
    ax2.set_ylabel("running estimate of c₁")
    ax2.set_title("Linear coefficient: c₁(n) → (γ − 1 − log 2π)/2")
    lo = max(li.LI_C1 - 0.6, np.min(c_run[ns >= 5]) - 0.05)
    hi = min(li.LI_C1 + 0.6, np.max(c_run[ns >= 5]) + 0.05)
    ax2.set_ylim(lo, hi)
    ax2.legend(loc="upper right", labelcolor=C["ink2"])
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_oscillation(ns, osc, path, hp_ns=None, hp_osc=None):
    """Top: lambda_n - smooth part.  Bottom: the same divided by sqrt(n)."""
    plt, fig, (ax1, ax2) = _figure(2, 1, (11, 6.2))
    for ax in (ax1, ax2):
        _style_axes(ax)
    ax1.axhline(0, color=C["axis"], lw=0.8)
    ax1.plot(ns, osc, color=C["s1"], lw=1.2)
    rms = float(np.sqrt(np.mean(osc ** 2)))
    i = int(np.argmax(np.abs(osc)))
    ax1.set_ylabel(r"$\lambda_n - \lambda_n^{\mathrm{smooth}}$")
    ax1.set_title(f"Oscillating (arithmetic) part S₂(n) = λₙ − S₁(n), n ≤ {int(ns[-1]):,}: "
                  f"mean {osc.mean():+.2f}, rms {rms:.2f}, max |S₂| = {abs(osc[i]):.2f} at n = {int(ns[i]):,}")
    ax2.axhline(0, color=C["axis"], lw=0.8)
    ax2.plot(ns, osc / np.sqrt(ns), color=C["s1"], lw=1.2)
    ax2.set_xlabel("n")
    ax2.set_ylabel(r"$(\lambda_n - \lambda_n^{\mathrm{smooth}})/\sqrt{n}$")
    r = osc / np.sqrt(ns)
    r100 = np.max(np.abs(r[ns >= 100])) if np.any(ns >= 100) else np.max(np.abs(r))
    ax2.set_title(f"Scaled by √n: within [{r.min():+.2f}, {r.max():+.2f}], and |·| < {r100:.2f} for n ≥ 100 "
                  f"(RH implies O(√n log n): Lagarias 2007, Thm 1.1)")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _mp_str(x, digits):
    import mpmath as mp
    with mp.workdps(digits + 5):
        return mp.nstr(x, digits)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nmax", type=int, default=10000, help="largest n for the zero-sum method")
    ap.add_argument("--hp-max", type=int, default=5000, help="largest n for the Arb high-precision method")
    ap.add_argument("--nzeros", type=int, default=None, help="use only the first N zeros (default: all available)")
    ap.add_argument("--time-budget", type=float, default=600.0, help="seconds; nmax/hp-max are reduced to fit")
    ap.add_argument("--json", default=str(REPO / "results" / "json" / "li_summary.json"))
    ap.add_argument("--figdir", default=str(REPO / "results" / "figures"))
    a = ap.parse_args()
    t_start = time.time()

    # ---- zeros -----------------------------------------------------------------
    zeros = load_zeros(a.nzeros)
    N = len(zeros)
    sets = available_datasets()
    man = manifest(max(sets, key=lambda s: s[1])[2]) if sets else {}
    print(f"[{time.time()-t_start:6.1f}s] {N} zeros loaded, gamma_N = {zeros[-1]:.6f}, "
          f"max radius {man.get('max_radius', 'n/a')}", flush=True)

    # ---- method 1: zeros + tail, with a time estimate ---------------------------
    nmax = a.nmax
    probe = np.arange(max(1, nmax - 63), nmax + 1, dtype=float)
    t0 = time.time()
    li.li_partial_sum(probe, zeros)
    per_n = (time.time() - t0) / len(probe)
    est = per_n * nmax
    budget_zero_sum = 0.5 * a.time_budget
    if est > budget_zero_sum:
        nmax = max(1000, int(budget_zero_sum / per_n))
        print(f"  estimated {est:.0f}s for nmax={a.nmax}; reducing nmax to {nmax}", flush=True)
    ns = np.arange(1, nmax + 1)
    lam, info = li.li_from_zeros(nmax, zeros, details=True)
    print(f"[{time.time()-t_start:6.1f}s] zero-sum lambda_n for n <= {nmax} "
          f"({info['time_partial_sum']:.1f}s sums, {info['time_tail']:.2f}s tail); "
          f"T_eff = {info['T_eff']:.6f} (gamma_N + {info['T_eff']-zeros[-1]:.4f}, "
          f"mean spacing {info['mean_spacing_at_cutoff']:.4f})", flush=True)
    print(f"  lambda_1 = {lam[0]!r}, closed form {LI_LAMBDA_1!r}, diff {lam[0]-LI_LAMBDA_1:.3e}, "
          f"tail bound {info['error_bound'][0]:.3e}", flush=True)

    # ---- method 2: Arb -----------------------------------------------------------
    hp_max = min(a.hp_max, nmax)
    t0 = time.time()
    li.li_high_precision(min(1000, hp_max))
    t_1000 = time.time() - t0
    est_hp = t_1000 * (hp_max / min(1000, hp_max)) ** 2.5
    remaining = a.time_budget - (time.time() - t_start) - 30.0
    if est_hp > remaining:
        hp_max = max(500, int(min(1000, hp_max) * (max(remaining, 1.0) / t_1000) ** 0.4))
        print(f"  estimated {est_hp:.0f}s for hp-max={a.hp_max}; reducing to {hp_max}", flush=True)
    t0 = time.time()
    hp_balls = li.li_high_precision(hp_max)
    t_hp = time.time() - t0
    hp_dps = li.default_dps(hp_max)
    hp = li.balls_to_float(hp_balls)
    hp_rad = li.balls_max_radius(hp_balls)
    hp_ns = np.arange(1, hp_max + 1)
    print(f"[{time.time()-t_start:6.1f}s] Arb lambda_n for n <= {hp_max} at {hp_dps} digits in {t_hp:.1f}s; "
          f"max enclosure radius {hp_rad:.2e}", flush=True)
    t0 = time.time()
    arch_balls = li.li_high_precision(hp_max, part="archimedean")
    arch = li.balls_to_float(arch_balls)
    s2_exact = hp - arch                                  # arithmetic part, n <= hp_max
    print(f"[{time.time()-t_start:6.1f}s] archimedean part S_1(n) for n <= {hp_max} in {time.time()-t0:.1f}s", flush=True)

    # ---- cross-check -----------------------------------------------------------------
    err = lam[:hp_max] - hp
    rel = np.abs(err) / hp
    i_abs, i_rel = int(np.argmax(np.abs(err))), int(np.argmax(rel))
    ratio = np.abs(err) / info["error_bound"][:hp_max]
    cross = {
        "nmax": int(hp_max),
        "max_abs_error": float(np.abs(err[i_abs])), "argmax_abs_n": int(hp_ns[i_abs]),
        "max_rel_error": float(rel[i_rel]), "argmax_rel_n": int(hp_ns[i_rel]),
        "lambda_1_error": float(err[0]),
        "max_error_over_tail_bound": float(ratio.max()),
        "bound_holds_everywhere": bool(np.all(np.abs(err) <= info["error_bound"][:hp_max])),
        "rel_error_n_le_100": float(rel[:100].max()),
        "rel_error_n_le_1000": float(rel[:min(1000, hp_max)].max()),
    }
    print(f"  cross-check: max |zero-sum - Arb| = {cross['max_abs_error']:.3e} at n={cross['argmax_abs_n']}, "
          f"max rel = {cross['max_rel_error']:.3e} at n={cross['argmax_rel_n']}; "
          f"error/bound <= {cross['max_error_over_tail_bound']:.2e}", flush=True)

    # ---- positivity --------------------------------------------------------------------
    pos = li.positivity_report(ns, lam)
    lower_zero_sum = lam - info["error_bound"]
    pos.update({
        "all_positive_minus_tail_bound": bool(np.all(lower_zero_sum > 0)),
        "min_lambda_minus_tail_bound": float(lower_zero_sum.min()),
        "rigorous_all_positive_arb_n_le": int(hp_max),
        "rigorous_arb_positive": bool(all(float(b.mid()) - float(b.rad()) > 0 for b in hp_balls)),
        "min_lambda_arb": li.balls_to_str([hp_balls[int(np.argmin(hp))]], 30)[0],
        "argmin_arb_n": int(hp_ns[int(np.argmin(hp))]),
    })
    print(f"  positivity: all lambda_n > 0 for n <= {nmax}: {pos['all_positive']}; "
          f"min = {pos['min_lambda']:.12f} at n = {pos['argmin_n']}; rigorous (Arb) for n <= {hp_max}: "
          f"{pos['rigorous_arb_positive']}", flush=True)

    # ---- asymptotics ----------------------------------------------------------------------
    smooth_h = li.li_smooth(ns, "harmonic")
    osc = lam - smooth_h
    fits = {
        "zero_sum_all": li.fit_asymptotic(ns, lam),
        "zero_sum_n_ge_100": li.fit_asymptotic(ns, lam, nmin=100),
        "zero_sum_n_ge_1000": li.fit_asymptotic(ns, lam, nmin=1000) if nmax > 2000 else None,
        "arb_all": li.fit_asymptotic(hp_ns, hp),
    }
    sm_check = {
        "max_abs_S1_minus_harmonic_form_n_ge_50": float(np.max(np.abs(arch[49:] - li.li_smooth(hp_ns[49:], "harmonic")))),
        "max_abs_S1_minus_log3_form_n_ge_100": float(np.max(np.abs(arch[99:] - li.li_smooth(hp_ns[99:], "log3")))),
        "S1_minus_log_form_times_24n_n_ge_100_range": [
            float(np.min((arch[99:] - li.li_smooth(hp_ns[99:], "log")) * 24 * hp_ns[99:])),
            float(np.max((arch[99:] - li.li_smooth(hp_ns[99:], "log")) * 24 * hp_ns[99:]))],
    }
    sqrt_scaled = osc / np.sqrt(ns)
    j = int(np.argmax(np.abs(osc)))
    asym = {
        "theory": {"c1": float(li.LI_C1), "c0": float(li.LI_C0),
                   "c1_formula": "(gamma - 1 - log 2pi)/2", "c0_note":
                   "S_1(n) = (n/2)(H_n - 1 - log 2pi) + 1/2 + exp. small = (n/2)(log n + gamma - 1 - log 2pi) + 3/4 - 1/(24n) + O(n^-3)"},
        "fits": fits,
        "smooth_part_checks_archimedean_arb_vs_formula": sm_check,
        "oscillating_part": {
            "definition": "lambda_n - [(n/2)(H_n - 1 - log 2pi) + 1/2]  (= S_2(n) up to < 1e-9 for n >= 50)",
            "nmax": int(nmax),
            "mean": float(osc.mean()), "rms": float(np.sqrt(np.mean(osc ** 2))),
            "max_abs": float(np.abs(osc[j])), "argmax_abs_n": int(ns[j]),
            "min": float(osc.min()), "max": float(osc.max()),
            "max_abs_over_sqrt_n": float(np.max(np.abs(sqrt_scaled))),
            "max_abs_over_sqrt_n_log_n": float(np.max(np.abs(osc[1:]) / (np.sqrt(ns[1:]) * np.log(ns[1:])))),
            "S2_arb_n_le_hp_max": {"mean": float(s2_exact.mean()), "rms": float(np.sqrt(np.mean(s2_exact ** 2))),
                                   "max_abs": float(np.max(np.abs(s2_exact))),
                                   "n_with_S2_negative": int(np.sum(s2_exact < 0))},
        },
    }
    f = fits["zero_sum_all"]
    print(f"  fit (all n): c1 = {f['fit_c1']:.6f} (theory {li.LI_C1:.6f}), c0 = {f['fit_c0']:.3f} (theory 0.75, "
          f"biased by the oscillation mean {osc.mean():+.3f}); free slope a = {f['fit_free_a']:.5f} (theory 0.5)", flush=True)
    print(f"  oscillating part: rms {asym['oscillating_part']['rms']:.3f}, max |.| {asym['oscillating_part']['max_abs']:.3f} "
          f"at n={asym['oscillating_part']['argmax_abs_n']}, max |.|/sqrt(n) {asym['oscillating_part']['max_abs_over_sqrt_n']:.3f}", flush=True)

    # ---- JSON ------------------------------------------------------------------------------
    sel = np.concatenate([np.arange(1, 101), np.arange(200, nmax + 1, 100)])
    sel = sel[sel <= nmax]
    hp_str = li.balls_to_str(hp_balls, 25)
    values = []
    for n in sel:
        k = int(n) - 1
        rec = {"n": int(n), "lambda": float(lam[k]), "partial_sum": float(info["partial_sum"][k]),
               "tail": float(info["tail"][k]), "tail_error_bound": float(info["error_bound"][k]),
               "smooth": float(smooth_h[k]), "oscillation": float(osc[k])}
        if n <= hp_max:
            rec["lambda_arb"] = hp_str[k]
            rec["lambda_arb_radius"] = float(hp_balls[k].rad())
            rec["S1_archimedean"] = float(arch[k])
            rec["S2_arithmetic"] = float(s2_exact[k])
        values.append(rec)
    summary = {
        "title": "Li/Keiper coefficients lambda_n and Li's criterion",
        "generated": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "methods": {
            "zero_sum": "lambda_n = 4 sum_{gamma<=gamma_N} sin^2(n arctan(1/(2 gamma))) + integral tail from T_eff "
                        "(N_smooth(T_eff) = N) against (1/2pi) log(u/2pi) du; error bound sup|S| * TV(f) with "
                        "Trudgian's |S(T)| <= 0.112 log T + 0.278 log log T + 2.510",
            "arb": "lambda_n = n [z^n] log xi(1/(1-z)) via arb_series (deflated zeta, lgamma, log, composition); "
                   "rigorous ball enclosures",
        },
        "zeros": {"count": int(N), "first_gamma": float(zeros[0]), "last_gamma": float(zeros[-1]),
                  "max_radius": man.get("max_radius"), "T_eff": info["T_eff"],
                  "T_eff_minus_last_gamma": info["T_eff"] - float(zeros[-1]),
                  "mean_spacing_at_cutoff": info["mean_spacing_at_cutoff"],
                  "S_bound_used": info["s_bound"]},
        "nmax_zero_sum": int(nmax),
        "high_precision": {"nmax": int(hp_max), "dps": int(hp_dps), "max_radius": hp_rad, "seconds": t_hp},
        "lambda_1": {"closed_form": "1 + gamma/2 - log(4 pi)/2",
                     "closed_form_value": _mp_str(li.lambda_1_closed_form(45), 40),
                     "zero_sum": float(lam[0]), "zero_sum_minus_closed_form": float(lam[0] - LI_LAMBDA_1),
                     "arb": hp_balls[0].str(40, radius=True)},
        "first_values_arb_30_digits": {str(n): li.balls_to_str([hp_balls[n - 1]], 30)[0] for n in range(1, 11)},
        "positivity": pos,
        "cross_check": cross,
        "asymptotics": asym,
        "values": values,
        "timing_seconds": {"partial_sums": info["time_partial_sum"], "tail": info["time_tail"],
                           "arb": t_hp, "total": time.time() - t_start},
    }
    os.makedirs(os.path.dirname(a.json), exist_ok=True)
    with open(a.json, "w") as fh:
        json.dump(summary, fh, indent=1)
    print(f"[{time.time()-t_start:6.1f}s] wrote {a.json}", flush=True)

    # ---- figures ----------------------------------------------------------------------------
    os.makedirs(a.figdir, exist_ok=True)
    p1 = os.path.join(a.figdir, "li_coefficients.png")
    p2 = os.path.join(a.figdir, "li_oscillation.png")
    plot_coefficients(ns.astype(float), lam, smooth_h, hp_ns, hp, p1)
    plot_oscillation(ns.astype(float), osc, p2)
    print(f"[{time.time()-t_start:6.1f}s] wrote {p1} and {p2}", flush=True)


if __name__ == "__main__":
    main()
