r"""Integral criteria equivalent to RH: Balazard–Saias–Yor and Volchkov.

Balazard–Saias–Yor (Adv. Math. 143 (1999) 284–287):

    int_{Re s = 1/2} log|zeta(s)| |ds| / |s|^2  =  2 pi  sum_{Re rho > 1/2} log| rho / (1 - rho) |,

so RH  <==>  the integral vanishes.  With s = 1/2 + it, |s|^2 = 1/4 + t^2 and the
integrand is even in t:

    I := (1/pi) int_0^inf log|zeta(1/2+it)| / (1/4 + t^2) dt  =  sum_{beta>1/2} log|rho/(1-rho)|  (= 0 under RH).

Volchkov (Ukrainian Math. J. 47 (1995) 491–493):

    int_0^inf (1 - 12 t^2) / (1 + 4 t^2)^3  int_{1/2}^inf log|zeta(sigma + i t)| d sigma  dt  =  pi (3 - gamma_E) / 32
    <==>  RH.

Numerical remarks (important).  Both are *exact identities* under RH, so they
can be checked numerically only up to truncation (the BSY integrand has log
singularities at every zero and a 1/t^2 tail; the truncation at height T
leaves an error of order 1/T).  A hypothetical zero rho = 1/2 + delta + i gamma
(together with its conjugate, which also has real part > 1/2) would change I by
about 2 delta/(1/4+gamma^2): the criteria are extremely insensitive at large
height.  We implement them as calibration of that
insensitivity and as a check of the identities.

Implementation.  The integrand log|zeta(1/2+it)| = log|Z(t)| is integrated on
the intervals between consecutive zeros with a fixed tanh-sinh rule (robust to
the endpoint log singularities), evaluating zeta with Arb (python-flint) or any
user-supplied vectorised function.
"""
from __future__ import annotations

import math
import numpy as np

__all__ = ["tanh_sinh_rule", "bsy_integral", "volchkov_integral", "logabs_zeta_line_arb"]


def tanh_sinh_rule(n: int = 40, h: float | None = None):
    """Nodes x_k in (-1,1) and weights w_k of the tanh-sinh rule with 2n+1 points.
    Converges exponentially for integrands with endpoint log singularities."""
    if h is None:
        h = 3.2 / n  # covers k h up to ~3.2 where weights are ~1e-15
    k = np.arange(-n, n + 1)
    u = k * h
    x = np.tanh(0.5 * np.pi * np.sinh(u))
    w = h * 0.5 * np.pi * np.cosh(u) / np.cosh(0.5 * np.pi * np.sinh(u)) ** 2
    # drop the extreme nodes (weights < 1e-15) whose abscissae are within a few
    # ulps of +-1: they contribute nothing and can round onto interval endpoints
    keep = np.abs(x) < 1 - 1e-14
    return x[keep], w[keep]


def logabs_zeta_line_arb(ts, dps=15):
    """log|zeta(1/2 + i t)| for an array of t using Arb (rigorous to ~1e-14)."""
    from flint import acb, ctx
    old = ctx.dps
    ctx.dps = dps
    try:
        out = np.empty(len(ts))
        for i, t in enumerate(ts):
            out[i] = math.log(float(abs(acb(0.5, float(t)).zeta())))
        return out
    finally:
        ctx.dps = old


def bsy_integral(zeros, T=None, logabs_zeta=None, n_nodes=40, progress=None):
    """I(T) = (1/pi) int_0^T log|zeta(1/2+it)| / (1/4+t^2) dt, integrated on
    [0, gamma_1], [gamma_1, gamma_2], ..., up to the last zero <= T.
    Returns (T_used, I_T, partial_values) where partial_values[j] = I(gamma_j)."""
    if logabs_zeta is None:
        logabs_zeta = logabs_zeta_line_arb
    zeros = np.asarray(zeros, dtype=float)
    if T is not None:
        zeros = zeros[zeros <= T]
    x, w = tanh_sinh_rule(n_nodes)
    edges = np.concatenate([[0.0], zeros])
    partial = np.empty(len(zeros))
    total = 0.0
    for j in range(len(zeros)):
        a, b = edges[j], edges[j + 1]
        mid, half = 0.5 * (a + b), 0.5 * (b - a)
        t = mid + half * x
        f = logabs_zeta(t) / (0.25 + t * t)
        total += half * float(np.dot(w, f))
        partial[j] = total / math.pi
        if progress and (j + 1) % progress == 0:
            print(f"  BSY: {j+1} intervals, T={b:.1f}, I={partial[j]:.6e}", flush=True)
    return float(edges[-1]), total / math.pi, partial


def volchkov_integral(zeros, T=120.0, sigma_max=30.0, n_nodes_t=24, n_nodes_sigma=48, dps=15,
                      subtract_nearest_zero=True):
    """Volchkov's double integral truncated at height T and sigma_max, using Arb
    for zeta.  The t-range is split at the zeros.  Expected value under RH:
    pi(3-gamma)/32 = 0.23785629588680...

    Inner integral: for t near a zero rho_j = 1/2 + i gamma_j the integrand
    log|zeta(sigma+it)| has a sharp peak at sigma -> 1/2 (a log singularity when
    t = gamma_j).  With `subtract_nearest_zero` we write
        log|zeta(s)| = log|s - rho_j| + log|zeta(s)/(s - rho_j)|
    for the nearest zero and integrate the first term in closed form over
    [1/2, sigma_max]:  with x = sigma - 1/2, e = t - gamma_j,
        int_0^X log sqrt(x^2 + e^2) dx = X log sqrt(X^2+e^2) - X + |e| arctan(X/|e|).
    The remaining integrand is smooth, so the tanh-sinh rule converges fast.
    The sigma-range is split at sigma = 1 so that the pole of zeta (a log
    singularity of the integrand for t -> 0) sits at an endpoint of a panel.
    Truncation: the t-tail beyond T is O(T^-3) (weight ~ 12 t^2/(64 t^6)), the
    sigma-tail beyond sigma_max is O(2^-sigma_max).
    """
    from flint import acb, ctx
    old = ctx.dps
    ctx.dps = dps
    try:
        zeros = np.asarray(zeros, dtype=float)
        zeros_T = zeros[zeros <= T]
        edges = np.concatenate([[0.0], zeros_T, [T]])
        xt, wt = tanh_sinh_rule(n_nodes_t)
        xs, ws = tanh_sinh_rule(n_nodes_sigma)
        X = sigma_max - 0.5

        def inner(t):
            # nearest zero (in t) for the subtraction
            j = int(np.searchsorted(zeros, t))
            cands = [k for k in (j - 1, j) if 0 <= k < len(zeros)]
            g = min(cands, key=lambda k: abs(zeros[k] - t)) if cands and subtract_nearest_zero else None
            e = (t - zeros[g]) if g is not None else None
            tot = 0.0
            # split at sigma = 1 so that the pole's log singularity (t -> 0) is an endpoint
            for (a, b) in ((0.5, 1.0), (1.0, 2.0), (2.0, sigma_max)):
                mid, half = 0.5 * (a + b), 0.5 * (b - a)
                sig = mid + half * xs
                vals = np.empty(len(sig))
                for i, si in enumerate(sig):
                    z = acb(float(si), float(t)).zeta()
                    if g is not None:
                        d = acb(float(si) - 0.5, float(e))
                        if abs(float(si) - 0.5) + abs(float(e)) < 1e-13:
                            # exact coincidence with the zero: the limit is |zeta'(rho)|
                            from flint import acb_series
                            ser = acb_series([acb(0.5, float(zeros[g])), 1]).zeta()
                            z = ser.coeffs()[1]
                        else:
                            z = z / d
                    v = float(abs(z))
                    vals[i] = math.log(v) if (v > 0 and math.isfinite(v)) else 0.0
                tot += half * float(np.dot(ws, vals))
            if g is not None:
                ae = abs(e)
                closed = X * math.log(math.hypot(X, ae)) - X + (ae * math.atan(X / ae) if ae > 0 else 0.0)
                tot += closed
            return tot

        total = 0.0
        for j in range(len(edges) - 1):
            a, b = edges[j], edges[j + 1]
            mid, half = 0.5 * (a + b), 0.5 * (b - a)
            t = mid + half * xt
            wts = (1 - 12 * t * t) / (1 + 4 * t * t) ** 3
            f = np.array([inner(ti) for ti in t])
            total += half * float(np.dot(wt, wts * f))
        return total
    finally:
        ctx.dps = old
