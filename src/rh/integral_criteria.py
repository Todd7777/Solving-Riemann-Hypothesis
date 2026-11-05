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
would change I by about delta/(1/4+gamma^2): the criteria are extremely
insensitive at large height.  We implement them as calibration of that
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
    # avoid nodes rounding to exactly +-1
    keep = np.abs(x) < 1 - 1e-16
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


def volchkov_integral(zeros, T=120.0, sigma_max=30.0, n_nodes_t=24, n_nodes_sigma=48, dps=15):
    """Volchkov's double integral truncated at height T and sigma_max, using Arb
    for zeta.  The t-range is split at the zeros (log singularities of the inner
    integral's integrand at sigma=1/2).  Expected value under RH: pi(3-gamma)/32."""
    from flint import acb, ctx
    old = ctx.dps
    ctx.dps = dps
    try:
        zeros = np.asarray(zeros, dtype=float)
        zeros = zeros[zeros <= T]
        edges = np.concatenate([[0.0], zeros, [T]])
        xt, wt = tanh_sinh_rule(n_nodes_t)
        xs, ws = tanh_sinh_rule(n_nodes_sigma)
        # inner integral over sigma in [1/2, sigma_max]; split [1/2, 2] and [2, sigma_max]
        def inner(t):
            tot = 0.0
            for (a, b) in ((0.5, 2.0), (2.0, sigma_max)):
                mid, half = 0.5 * (a + b), 0.5 * (b - a)
                s = mid + half * xs
                vals = np.array([math.log(float(abs(acb(float(si), float(t)).zeta()))) for si in s])
                tot += half * float(np.dot(ws, vals))
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
