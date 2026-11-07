#!/usr/bin/env python3
r"""Zeros of the Dirichlet L-function L(s, (n/5)) with 0 < Im s <= 1000, certified.

The character
-------------
chi = (n/5) is the Legendre symbol mod 5: chi(1..5) = 1, -1, -1, 1, 0.  It is
real, even (chi(-1) = chi(4) = 1) and primitive of conductor 5; in python-flint
it is ``dirichlet_char(5, 4)`` (Conrey label 5.4; the labels 2 and 3 are the
two complex characters with chi(2) = +-i used by ``rh.davenport_heilbronn``).

Functional equation, Z-function and the counting function
---------------------------------------------------------
Lambda(s) = (5/pi)^{s/2} Gamma(s/2) L(s, chi) is entire and satisfies
Lambda(s) = eps Lambda(1 - s) with root number eps = tau(chi)/sqrt 5, where
tau(chi) = sum_a chi(a) e^{2 pi i a/5} = 2cos(2pi/5) - 2cos(4pi/5) = sqrt 5, so
eps = 1.  (For a real primitive character tau(chi)^2 = chi(-1) q, so eps = +-1;
the ball enclosure of tau(chi)/sqrt 5 computed by ``root_number`` excludes -1,
which proves eps = 1.)  As chi is real,
Lambda(conj s) = conj Lambda(s); hence Lambda(1/2 + it) = Lambda(1/2 - it) =
conj Lambda(1/2 + it) is real.  Writing
(5/pi)^{s/2} Gamma(s/2) = (5/pi)^{1/4} |Gamma(1/4 + it/2)| e^{i theta(t)} at s = 1/2 + it,

    theta_chi(t) = (t/2) log(5/pi) + Im log Gamma(1/4 + it/2)        (log Gamma: the
                                                                     analytic branch, real on t = 0)
    Z_chi(t)     = e^{i theta_chi(t)} L(1/2 + it, chi)
                 = Lambda(1/2 + it) / ((5/pi)^{1/4} |Gamma(1/4 + it/2)|)   is real,

and its sign changes are zeros on the critical line.  This theta is the
Arb phase acb_dirichlet_hardy_theta for (q, delta, eps) = (5, 0, 1) (agreement
with ``dirichlet_char.hardy_z`` is checked).

Counting function.  Let N(T) be the number of zeros of L(s, chi) with
0 < Im s <= T (with multiplicity; they all have 0 < Re s < 1: L(s) != 0 for
Re s > 1 by the Euler product and, through the functional equation, the only
zeros with Re s < 0 are the trivial ones s = 0, -2, -4, ... -- simple zeros of
L, cancelling the poles of Gamma(s/2)).  Suppose L(sigma) != 0 for 0 < sigma < 1
(proved below) and L(1/2 + iT) != 0.  Apply the argument principle to Lambda on
[-5/2, 7/2] x [0, T]: Lambda is real and nonzero on the bottom edge, and the
reflection s -> 1 - conj(s) maps the left half of the contour onto the right
half reversed while Lambda(1 - conj s) = conj Lambda(s); so the left half
contributes as much as the right half C: 1/2 -> 7/2 -> 7/2 + iT -> 1/2 + iT.
On C the Gamma factor contributes theta_chi(T) (its logarithm is analytic in
Re s > 0) and L contributes nothing on the real segment (L > 0 there), so

    N(T) = theta_chi(T)/pi + S(T),
    S(T) = (1/pi) [ Arg L(7/2 + iT) + Delta_{7/2 + iT -> 1/2 + iT} arg L ],

where on Re s = 7/2 the continuous argument is the principal one because
|L - 1| <= zeta(7/2) - 1 = 0.127 < 1.  There is no "+1" as for zeta
(theta(T)/pi + 1): Lambda has no pole and no factor s(s - 1).  The smooth part is

    N_smooth(T) = theta_chi(T)/pi = (T/2pi) log(5T/(2 pi e)) - 1/8 + O(1/T).

Rigorous counts (two independent ways)
--------------------------------------
(A) Full rectangle, no functional equation in any evaluation (it enters only
    through the classical fact that the non-trivial zeros have 0 < Re s < 1,
    so that the rectangle contains all of them): the winding number of L itself
    (entire; no zeros or poles of Gamma(s/2) in Im s > 0) around
    [1/2 - 3, 1/2 + 3] x [1, T] counts the zeros with 1 < Im s < T.  Every
    value and Taylor coefficient of L used here comes from Arb's
    acb_dirichlet_l_jet, i.e. 5^{-s} sum_a chi(a) zeta(s, a/5) with Hurwitz zeta
    by Euler-Maclaurin summation (Arb's default acb_dirichlet_l switches to the
    functional equation for Re s < -1/2, which (A) must not use).  The
    rectangle [-5/2, 7/2] x [-1, 1] must have winding number 2 (the simple
    trivial zeros s = 0 and s = -2) -- this also proves that L has no zero
    in 0 < sigma < 1 and none with 0 < |Im s| <= 1.  Edges are shared between
    the heights T = 100, 300, 600, 1000.
(B) The formula N(T) = theta_chi(T)/pi + S(T) above (uses the functional
    equation and root number 1), evaluated only on the top edge right half.

Both compute arguments with the same certified segment test: on a horizontal
or vertical segment with midpoint m and half-length r, Arb gives balls for the
Taylor coefficients c_0..c_{K-1} of L at m and a ball for c_K valid on the whole
segment (power series over a box), so every value lies within
R = sum_{2<=k<K} |c_k| r^k + sup|c_K| r^K of the line segment
{c_0 + c_1 e u : |u| <= r}.  If the distance from 0 to that line segment
exceeds R, the values on the segment lie in a closed convex set avoiding 0,
hence in an open half-plane, so the continuous change of arg L along it is the
principal Arg(L(z1)/L(z0)) in (-pi, pi), enclosed by acb.arg.  Otherwise the
segment is bisected.  Every ingredient is a ball enclosure, so the counts are
rigorous; the sum of the arg changes / 2pi must contain exactly one integer.

Zeros on the line
-----------------
Z_chi is evaluated in ball arithmetic on a grid of step 1/16 (exact binary
points); a sign change between certified signs proves a zero of odd
multiplicity in between.  If, between consecutive heights where N(T) is
computed by (B) (every 50, and at 100, 300, 600, 1000), the number of sign
changes equals the increase of N, then every zero in that range is simple,
lies on the critical line and is the unique zero in its bracket; a mismatch
triggers a finer local grid.  Each zero is refined by the Illinois method at
30 digits and certified: with x the float64 nearest the refined ordinate and
eps = ulp(x)/2 + 1e-15, Z_chi(x - eps) and Z_chi(x + eps) have opposite
certified signs, so |gamma - x| < eps (rigorous, <= 5.8e-14 below t = 1024);
the same test at 30 digits certifies the printed 27-digit ordinate to 1e-20.
The ordinates are also recomputed independently with mpmath for a sample.

Usage
-----
    python3 scripts/compute_lchi5_zeros.py                # T = 1000, writes data/zeros/lchi5_zeros.{npy,json}
    python3 scripts/compute_lchi5_zeros.py --T 200 --out /tmp/x --heights 100,200   # smaller run

The output ``lchi5_zeros.npy`` is not named ``zeros_<a>_<b>.npy`` and so is
not picked up by ``rh.data.load_zeros`` (which serves zeta zeros).

References: H. Davenport, Multiplicative Number Theory, ch. 9 (functional
equation) and ch. 16 (the number N(T, chi)); R. Rumely, Numerical computations
concerning the ERH, Math. Comp. 61 (1993) (zeros of L(s, chi) for small
conductors, Turing-type counting); D. Platt, Numerical computations concerning
the GRH, Math. Comp. 85 (2016); F. Johansson, Arb: efficient arbitrary-precision
midpoint-radius interval arithmetic, IEEE Trans. Comput. 66 (2017).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUT = os.path.join(REPO, "data", "zeros", "lchi5_zeros")

MODULUS = 5
CONREY_INDEX = 4                          # dirichlet_char(5, 4) = (n/5)
CHI_VALUES = (0, 1, -1, -1, 1)            # chi(n) for n = 0, 1, 2, 3, 4 (mod 5)
SIGMA_LEFT, SIGMA_RIGHT = -2.5, 3.5       # 1/2 -+ 3
T_BOTTOM = 1.0                            # rectangle (A) starts at Im s = 1; [-1, 1] handled separately
TAYLOR_K = 6                              # order of the Taylor model in the segment test
SEG_DPS = 20                              # precision of the argument-principle computations
GRID_DPS = 15                             # precision of the grid evaluation of Z
REFINE_DPS = 30                           # precision of refinement and certification
MAX_DEPTH = 40                            # bisection depth limit in the segment test
DEFAULT_HEIGHTS = (100.0, 300.0, 600.0, 1000.0)


# ----------------------------------------------------------------------------
# The character, theta, Z
# ----------------------------------------------------------------------------
def character():
    """python-flint's dirichlet_char(5, 4), checked to be (n/5)."""
    from flint import dirichlet_char
    chi = dirichlet_char(MODULUS, CONREY_INDEX)
    vals = [complex(chi(n)) for n in range(MODULUS)]          # exact values 0, +-1
    if vals != [complex(w) for w in CHI_VALUES]:
        raise AssertionError(f"dirichlet_char(5, {CONREY_INDEX}) has values {vals}, expected {CHI_VALUES}")
    assert chi.is_real() and chi.is_primitive() and chi.parity() == 0 and chi.conductor() == 5
    return chi


_CHI = None


def _chi():
    global _CHI
    if _CHI is None:
        _CHI = character()
    return _CHI


def theta_chi(t):
    """theta_chi(t) = (t/2) log(5/pi) + Im log Gamma(1/4 + it/2) as an arb ball (t: float, str or arb)."""
    from flint import acb, arb
    t = t if isinstance(t, arb) else arb(t)
    return t / 2 * (arb(5) / arb.pi()).log() + acb(arb(1) / 4, t / 2).lgamma().imag


def n_smooth(T) -> "arb":
    """N_smooth(T) = theta_chi(T)/pi (arb ball)."""
    from flint import arb
    return theta_chi(T) / arb.pi()


def n_smooth_asymptotic(T: float) -> float:
    """(T/2pi) log(5T/(2 pi e)) - 1/8, the Stirling form of theta_chi(T)/pi (error O(1/T))."""
    return T / (2 * math.pi) * math.log(5 * T / (2 * math.pi * math.e)) - 0.125


def L_value(s):
    """L(s, chi) as an acb ball by Arb's default acb_dirichlet_l (s: acb or (x, y) pair of floats).

    For Re s < -1/2 Arb evaluates this through the functional equation."""
    from flint import acb
    if not isinstance(s, acb):
        s = acb(s[0], s[1])
    return acb.dirichlet_l(s, _chi())


def L_hurwitz(s):
    """L(s, chi) as an acb ball through acb_dirichlet_l_jet, i.e. 5^{-s} sum_a chi(a) zeta(s, a/5)
    with the Hurwitz zeta function by Euler-Maclaurin summation -- never the functional equation."""
    from flint import acb
    if not isinstance(s, acb):
        s = acb(s[0], s[1])
    return _taylor(s, 2)[0]


def Z_chi(t):
    """Z_chi(t) = e^{i theta_chi(t)} L(1/2 + it, chi) as an acb ball (imaginary part encloses 0)."""
    from flint import acb, arb
    t = t if isinstance(t, arb) else arb(t)
    return (acb(0, 1) * theta_chi(t)).exp() * acb.dirichlet_l(acb(arb(1) / 2, t), _chi())


def Lambda(s):
    """Completed L-function (5/pi)^{s/2} Gamma(s/2) L(s, chi) (acb ball; s away from 0, -2, ...),
    with L from the Hurwitz-zeta route (so that comparing Lambda(s), Lambda(1-s) tests the
    functional equation instead of using it)."""
    from flint import acb, arb
    s = s if isinstance(s, acb) else acb(s.real, s.imag)
    return (arb(5) / arb.pi()) ** (s / 2) * (s / 2).gamma() * L_hurwitz(s)


def root_number(dps: int = 40) -> dict:
    """tau(chi)/sqrt(5) in ball arithmetic (must enclose 1)."""
    from flint import acb, arb, ctx
    old = ctx.dps
    ctx.dps = dps
    try:
        chi = _chi()
        tau = sum((chi(a) * acb(0, 2 * arb.pi() * a / 5).exp() for a in range(1, 5)), acb(0))
        eps = tau / arb(5).sqrt()
        d = eps - 1
        return {"tau": tau.real.str(25), "root_number": eps.real.str(25),
                "abs_root_number_minus_1_upper": float(abs(d).upper()),
                "encloses_1": bool(d.real.contains(0) and d.imag.contains(0))}
    finally:
        ctx.dps = old


def functional_equation_check(points=((0.3, 10.0), (2.0, 50.0), (-1.5, 300.0), (-2.5, 777.0), (0.8, 999.5)),
                              dps: int = 40) -> list:
    """|Lambda(s) - Lambda(1 - s)| / |Lambda(s)| upper bounds at a few points (balls must overlap);
    both sides use the Hurwitz-zeta evaluation of L.  Also compares, for Re s < -1/2, Arb's default
    L (computed there by the functional equation, with Arb's root number) with the Hurwitz value."""
    from flint import acb, ctx
    old = ctx.dps
    ctx.dps = dps
    try:
        out = []
        for x, y in points:
            s = acb(x, y)
            a, b = Lambda(s), Lambda(1 - s)
            d = a - b
            rec = {"s": [x, y], "rel_diff_upper": float((abs(d) / abs(a)).upper()),
                   "overlap": bool(d.real.contains(0) and d.imag.contains(0))}
            if x < -0.5:
                u, v = L_value(s), L_hurwitz(s)
                rec["arb_default_vs_hurwitz_overlap"] = bool((u - v).real.contains(0) and (u - v).imag.contains(0))
                rec["arb_default_vs_hurwitz_rel_upper"] = float((abs(u - v) / abs(v)).upper())
                rec["overlap"] = rec["overlap"] and rec["arb_default_vs_hurwitz_overlap"]
            out.append(rec)
        return out
    finally:
        ctx.dps = old


def hardy_z_crosscheck(ts=(0.5, 6.6, 17.0, 99.9, 500.25, 999.0), dps: int = 30) -> dict:
    """Compare our Z_chi with Arb's acb_dirichlet_hardy_z (python-flint dirichlet_char.hardy_z)."""
    from flint import acb, arb, ctx
    old = ctx.dps
    ctx.dps = dps
    try:
        mx, ok = 0.0, True
        for t in ts:
            a = Z_chi(t)
            b = _chi().hardy_z(acb(t))
            d = a - b
            ok = ok and bool(d.real.contains(0) and d.imag.contains(0))
            mx = max(mx, float(abs(d).upper()))
        return {"points": list(ts), "max_abs_diff_upper": mx, "all_overlap": ok}
    finally:
        ctx.dps = old


def reality_check_high_precision(ts=(0.7, 6.648, 33.3, 123.456, 456.7, 777.7, 999.99), dps: int = 50) -> dict:
    """At dps digits: the Im Z_chi ball contains 0 and max |Im Z|/|Z| (upper bound) is ~10^-dps."""
    from flint import ctx
    old = ctx.dps
    ctx.dps = dps
    try:
        rel, ok = 0.0, True
        for t in ts:
            z = Z_chi(t)
            ok = ok and bool(z.imag.contains(0))
            rel = max(rel, float((abs(z.imag) / abs(z)).upper()))
        return {"dps": dps, "points": list(ts), "Im_contains_0": ok, "max_rel_Im_upper": rel}
    finally:
        ctx.dps = old


def certified_sign(z) -> int:
    """+1 / -1 if the real ball z excludes 0, else 0."""
    if z > 0:
        return 1
    if z < 0:
        return -1
    return 0


def Z_sign(t, dps_list=(15, 30, 60)):
    """Certified sign of Z_chi at t (float or arb ball), raising precision when needed.

    Returns (sign, Z ball at the last precision); sign 0 if undecided."""
    from flint import ctx
    old = ctx.dps
    try:
        z = None
        for d in dps_list:
            ctx.dps = d
            z = Z_chi(t)
            if not (z.imag.contains(0)):
                raise ArithmeticError(f"Z_chi({t}) is not real: {z}")
            s = certified_sign(z.real)
            if s:
                return s, z
        return 0, z
    finally:
        ctx.dps = old


# ----------------------------------------------------------------------------
# Certified argument changes along segments (argument principle)
# ----------------------------------------------------------------------------
class ArgumentPrincipleError(RuntimeError):
    """The segment test failed (a zero on or extremely close to the contour)."""


def _taylor(s, n):
    """Taylor coefficients c_0..c_{n-1} of L(., chi) at s (acb point or box)."""
    from flint import acb, acb_series
    c = acb_series([s, 1], n).dirichlet_l(_chi()).coeffs()
    return list(c) + [acb(0)] * (n - len(c))


def _dist_lower(A, B, r):
    """Rigorous lower bound (float-comparable arb) for min_{|u| <= r, u real} |A + B u|, A, B acb balls."""
    lbs = [(abs(A) - abs(B) * r).lower()]
    absB = abs(B)
    if absB.lower() > 0:
        AB = A * B.conjugate()
        lbs.append((abs(AB.imag) / absB).lower())            # distance to the whole line
        ustar = -AB.real / (absB * absB)                      # foot of the perpendicular
        if ustar.lower() > r:
            lbs.append(abs(A + B * r).lower())
        elif ustar.upper() < -r:
            lbs.append(abs(A - B * r).lower())
    best = lbs[0]
    for v in lbs[1:]:
        if v > best:
            best = v
    return best


def _segment_arg(z0, z1, f0, f1, K, depth, stats):
    """Certified change of arg L along the segment z0 -> z1 (horizontal or vertical, exact binary
    endpoints), given balls f0 = L(z0), f1 = L(z1).  Returns an arb ball."""
    from flint import acb, arb
    stats["segments"] += 1
    dx, dy = z1[0] - z0[0], z1[1] - z0[1]
    if (dx == 0) == (dy == 0):
        raise ValueError("segment must be horizontal or vertical and nondegenerate")
    m = ((z0[0] + z1[0]) / 2, (z0[1] + z1[1]) / 2)
    if m[0] - z0[0] != z1[0] - m[0] or m[1] - z0[1] != z1[1] - m[1]:
        raise ArgumentPrincipleError(f"midpoint of {z0}->{z1} not exact in binary")
    h = abs(dx) + abs(dy)
    r = arb(h / 2)                                            # exact
    e = acb(math.copysign(1.0, dx), 0) if dy == 0 else acb(0, math.copysign(1.0, dy))
    pc = _taylor(acb(m[0], m[1]), K)
    R = arb(0)
    rk = r * r
    for k in range(2, K):
        R += abs(pc[k]) * rk
        rk = rk * r
    lb = _dist_lower(pc[0], pc[1] * e, r)
    if lb > R.upper():
        box = acb(arb(z0[0]).union(arb(z1[0])), arb(z0[1]).union(arb(z1[1])))
        cK = _taylor(box, K + 1)[K]
        R += abs(cK) * rk                                     # rk = r^K here
        if lb > R.upper():
            a = (f1 / f0).arg()
            if a.rad() < 0.01 and abs(a.mid()) < 3.0:
                return a
    if depth >= MAX_DEPTH:
        raise ArgumentPrincipleError(f"segment test failed at depth {depth} near {m}")
    stats["splits"] += 1
    fm = pc[0]
    return (_segment_arg(z0, m, f0, fm, K, depth + 1, stats)
            + _segment_arg(m, z1, fm, f1, K, depth + 1, stats))


def edge_arg(z0, z1, h0: float, K: int = TAYLOR_K, stats=None):
    """Certified change of arg L along the straight edge z0 -> z1, cut into pieces of length <= h0
    (a power of 2 dividing the edge length so that all endpoints are exact)."""
    from flint import arb
    if stats is None:
        stats = {"segments": 0, "splits": 0}
    length = abs(z1[0] - z0[0]) + abs(z1[1] - z0[1])
    n = max(1, int(math.ceil(length / h0 - 1e-12)))
    pts = [(z0[0] + (z1[0] - z0[0]) * k / n, z0[1] + (z1[1] - z0[1]) * k / n) for k in range(n + 1)]
    pts[-1] = z1
    for p, q in zip(pts, pts[1:]):
        if (q[0] - p[0]) * n != (z1[0] - z0[0]) or (q[1] - p[1]) * n != (z1[1] - z0[1]):
            raise ArgumentPrincipleError(f"edge {z0}->{z1} cannot be cut exactly into {n} pieces")
    vals = [L_hurwitz(p) for p in pts]          # Hurwitz route, as the Taylor models: no functional equation
    for v in vals:
        if v.real.contains(0) and v.imag.contains(0):
            raise ArgumentPrincipleError("L vanishes (or nearly) at a contour vertex")
    tot = arb(0)
    for k in range(n):
        tot += _segment_arg(pts[k], pts[k + 1], vals[k], vals[k + 1], K, 0, stats)
    return tot, stats


def _edge_worker(args):
    """Process-pool wrapper: returns the arg change as a decimal ball string (exactly re-parsable,
    containing the computed ball) plus statistics."""
    z0, z1, h0, K, dps = args
    from flint import ctx
    ctx.dps = dps
    t0 = time.time()
    a, st = edge_arg(z0, z1, h0, K)
    return {"z0": z0, "z1": z1, "arg": a.str(0, radius=True, more=True), "segments": st["segments"],
            "splits": st["splits"], "seconds": time.time() - t0}


def _to_integer(x) -> int | None:
    """The unique integer in the ball x, or None."""
    lo, hi = math.ceil(float(x.lower()) - 1e-9), math.floor(float(x.upper()) + 1e-9)
    cands = [k for k in range(lo - 1, hi + 2) if x.contains(k)]
    return cands[0] if len(cands) == 1 and x.rad() < 0.25 else None


def winding(edges_args) -> dict:
    """Sum of certified arg changes over closed-contour pieces -> winding number (integer)."""
    from flint import arb
    tot = arb(0)
    for s in edges_args:
        tot += arb(s)
    w = tot / (2 * arb.pi())
    return {"ball": w.str(10, radius=True), "integer": _to_integer(w)}


def bottom_rectangle(dps: int = SEG_DPS) -> dict:
    """Winding number of L around [-5/2, 7/2] x [-1, 1]; = 2 iff the only zeros there are s = 0, -2."""
    from flint import ctx
    ctx.dps = dps
    sl, sr, t = SIGMA_LEFT, SIGMA_RIGHT, T_BOTTOM
    pieces = [((sl, -t), (sr, -t), 0.25), ((sr, -t), (sr, t), 0.5), ((sr, t), (sl, t), 0.25), ((sl, t), (sl, -t), 0.25)]
    res = [_edge_worker((a, b, h, TAYLOR_K, dps)) for a, b, h in pieces]
    w = winding([r["arg"] for r in res])
    w["segments"] = sum(r["segments"] for r in res)
    return w


def backlund_count(T: float, dps: int = SEG_DPS) -> dict:
    """Method (B): N(T) = theta_chi(T)/pi + S(T), S(T) = (Arg L(7/2+iT) + Delta_top arg L)/pi, rigorous."""
    from flint import arb, ctx
    old = ctx.dps
    ctx.dps = dps
    try:
        # |L(7/2 + it) - 1| <= zeta(7/2) - 1 < 1, so the continuous arg on Re s = 7/2 is the principal one
        bound = arb(SIGMA_RIGHT).zeta() - 1
        assert bound < 1
        aR = L_value((SIGMA_RIGHT, T)).arg()
        top, st = edge_arg((SIGMA_RIGHT, T), (0.5, T), 0.25)
        S = (aR + top) / arb.pi()
        th = n_smooth(T)
        N = th + S
        return {"T": T, "N": _to_integer(N), "N_ball": N.str(10, radius=True), "N_smooth": float(th.mid()),
                "N_smooth_ball": th.str(15, radius=True), "S": float(S.mid()), "S_ball": S.str(10, radius=True),
                "segments": st["segments"], "zeta_7_2_minus_1_upper": float(bound.upper())}
    finally:
        ctx.dps = old


def full_rectangle_counts(heights, workers: int = 1, dps: int = SEG_DPS, chunk: float = 50.0) -> dict:
    """Method (A): winding number of L around [-5/2, 7/2] x [1, T] for every T in heights, from
    shared, independently certified edges (left edge in chunks, in parallel)."""
    heights = sorted(float(T) for T in heights)
    sl, sr, tb = SIGMA_LEFT, SIGMA_RIGHT, T_BOTTOM
    # vertical edges cut at the heights and at multiples of `chunk`
    cuts = sorted(set([tb] + list(heights) + [float(c) for c in np.arange(chunk, heights[-1], chunk) if c > tb]))
    jobs = []
    for a, b in zip(cuts, cuts[1:]):
        jobs.append(("L", (sl, b), (sl, a), 0.25))           # left edge, downwards
        jobs.append(("R", (sr, a), (sr, b), 1.0))            # right edge, upwards
    jobs.append(("B", (sl, tb), (sr, tb), 0.25))              # bottom edge at Im s = 1, rightwards
    for T in heights:
        jobs.append(("T", (sr, T), (sl, T), 0.25))            # top edges, leftwards
    args = [(j[1], j[2], j[3], TAYLOR_K, dps) for j in jobs]
    t0 = time.time()
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            res = list(ex.map(_edge_worker, args))
    else:
        res = [_edge_worker(a) for a in args]
    out = {"rectangle": f"[{sl}, {sr}] x [{tb}, T]", "counts": [], "segments": sum(r["segments"] for r in res),
           "seconds": time.time() - t0}
    for T in heights:
        pieces = []
        for (kind, z0, z1, _), r in zip(jobs, res):
            if kind == "B":
                pieces.append(r["arg"])
            elif kind == "L" and z0[1] <= T + 1e-12:
                pieces.append(r["arg"])
            elif kind == "R" and z1[1] <= T + 1e-12:
                pieces.append(r["arg"])
            elif kind == "T" and z0[1] == T:
                pieces.append(r["arg"])
        w = winding(pieces)
        out["counts"].append({"T": T, "N_minus_N1": w["integer"], "ball": w["ball"]})
    return out


# ----------------------------------------------------------------------------
# Grid evaluation, sign changes, refinement
# ----------------------------------------------------------------------------
def _grid_worker(args):
    """Z_chi at t = k/den for k in [k0, k1]: returns lists (t, Z midpoint, certified sign, |Im Z| upper)."""
    k0, k1, den, dps = args
    from flint import ctx
    ctx.dps = dps
    ts, zs, sg, im = [], [], [], 0.0
    for k in range(k0, k1 + 1):
        t = k / den                                           # exact for den a power of 2
        z = Z_chi(t)
        if not z.imag.contains(0):
            raise ArithmeticError(f"Z_chi({t}) not real: {z}")
        s = certified_sign(z.real)
        if s == 0:
            s, z = Z_sign(t, (30, 60, 120))
        ts.append(t)
        zs.append(float(z.real.mid()))
        sg.append(s)
        im = max(im, float(abs(z.imag).upper()))
    return ts, zs, sg, im


def grid_signs(t0: float, t1: float, den: int, workers: int = 1, dps: int = GRID_DPS, block: int = 400):
    """Certified signs of Z_chi on the grid t = k/den, t0 <= t <= t1 (den a power of 2)."""
    k0, k1 = int(round(t0 * den)), int(round(t1 * den))
    assert k0 / den == t0 and k1 / den == t1, "grid endpoints must lie on the grid"
    tasks = [(a, min(a + block - 1, k1), den, dps) for a in range(k0, k1 + 1, block)]
    if workers > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            parts = list(ex.map(_grid_worker, tasks))
    else:
        parts = [_grid_worker(t) for t in tasks]
    ts = np.concatenate([p[0] for p in parts])
    zs = np.concatenate([p[1] for p in parts])
    sg = np.concatenate([p[2] for p in parts]).astype(int)
    im = max(p[3] for p in parts)
    return ts, zs, sg, im


def brackets_from_grid(ts, sg):
    """Consecutive grid points with certified opposite signs (undecided points are skipped over)."""
    out = []
    idx = [i for i in range(len(ts)) if sg[i] != 0]
    for i, j in zip(idx, idx[1:]):
        if sg[i] != sg[j]:
            out.append((float(ts[i]), float(ts[j]), int(sg[i])))
    return out


def _refine_worker(args):
    """Illinois iteration at REFINE_DPS digits inside a certified bracket, then a certificate.

    Returns (s, x, eps, certified, r_hp): s the refined ordinate as a 27-digit decimal string,
    x the float64 nearest to it, and certified means that Z_chi(x - eps), Z_chi(x + eps) have
    opposite certified signs with a < x - eps < x + eps < b, so the unique zero of the bracket
    satisfies |gamma - x| < eps; likewise |gamma - s| < r_hp (nan if that check failed)."""
    a, b, sa, dps = args
    from flint import arb, ctx
    ctx.dps = dps

    def f(x):
        return Z_chi(x).real.mid()

    # Illinois (modified regula falsi) on midpoints: the root stays between xa and xb
    xa, xb = arb(a), arb(b)
    fa, fb = f(xa), f(xb)
    tol = arb("1e-24")
    for _ in range(200):
        x = (xb - fb * (xb - xa) / (fb - fa)).mid()
        lo_, hi_ = (xa, xb) if xa < xb else (xb, xa)
        if not (lo_ < x < hi_):
            x = ((xa + xb) / 2).mid()
        fx = f(x)
        if fx.is_zero():
            xa = xb = x
            break
        step = abs(x - xb)
        if (fx > 0) != (fb > 0):          # root between xb and x
            xa, fa = xb, fb
        else:                             # root between xa and x: Illinois halving
            fa = fa / 2
        xb, fb = x, fx
        if abs(xb - xa) < tol or step < tol:
            break
    tstar = xb if abs(xb - xa) >= tol else ((xa + xb) / 2).mid()

    def brackets_zero(center, radius) -> bool:
        lo, hi = center - arb(radius), center + arb(radius)
        if not (lo > a and hi < b):
            return False
        s_lo, _ = Z_sign(lo, (dps, 2 * dps))
        s_hi, _ = Z_sign(hi, (dps, 2 * dps))
        return s_lo == sa and s_hi == -sa

    # high-precision certificate for the printed decimal ordinate s: |gamma - s| < 1e-20
    s_hp = tstar.str(27, radius=False)
    hp = next((r for r in ("1e-20", "1e-18", "1e-16") if brackets_zero(arb(s_hp), arb(r))), None)
    # float64 certificate: |gamma - x| < eps, eps = ulp(x)/2 + 1e-15 (x = nearest double to t*)
    x = float(tstar)
    for eps in (math.ulp(x) / 2 + 1e-15, 1e-13, 1e-12):
        if brackets_zero(arb(x), eps):
            return s_hp, x, eps, True, (float(hp) if hp else float("nan"))
    return s_hp, x, float("nan"), False, (float(hp) if hp else float("nan"))


def refine_all(brackets, workers: int = 1, dps: int = REFINE_DPS):
    args = [(a, b, s, dps) for a, b, s in brackets]
    if workers > 1 and len(args) > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            return list(ex.map(_refine_worker, args, chunksize=8))
    return [_refine_worker(a) for a in args]


# ----------------------------------------------------------------------------
# Independent check with mpmath (different library: Hurwitz-zeta based Dirichlet series)
# ----------------------------------------------------------------------------
def Z_mpmath(t, dps: int = 30):
    import mpmath as mp
    with mp.workdps(dps):
        t = mp.mpf(t)
        th = t / 2 * mp.log(5 / mp.pi) + mp.im(mp.loggamma(mp.mpf(1) / 4 + 0.5j * t))
        return mp.exp(1j * th) * mp.dirichlet(mp.mpf(1) / 2 + 1j * t, list(CHI_VALUES))


def mpmath_zero(guess: float, dps: int = 30):
    """Zero of Re Z_chi near `guess` by mpmath.findroot (secant), independent of Arb."""
    import mpmath as mp
    with mp.workdps(dps):
        g = mp.mpf(guess)
        return mp.findroot(lambda t: mp.re(Z_mpmath(t, dps)), (g - mp.mpf("1e-6"), g + mp.mpf("1e-6")),
                           tol=mp.mpf(10) ** (-dps + 5))


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------
def compute(T: float = 1000.0, heights=DEFAULT_HEIGHTS, den: int = 16, block_step: float = 50.0,
            workers: int = 3, log=print) -> dict:
    t_start = time.time()
    heights = sorted(set(float(h) for h in heights if h <= T) | {float(T)})
    from flint import ctx
    ctx.dps = SEG_DPS
    info = {"root_number": root_number(), "functional_equation": functional_equation_check(),
            "hardy_z_crosscheck": hardy_z_crosscheck()}
    log(f"root number {info['root_number']['root_number']}, FE ok "
        f"{all(d['overlap'] for d in info['functional_equation'])}, hardy_z ok {info['hardy_z_crosscheck']['all_overlap']}")
    # (A0) bottom rectangle
    b0 = bottom_rectangle()
    log(f"winding of L around [-2.5, 3.5] x [-1, 1] = {b0['integer']} ({b0['ball']})")
    if b0["integer"] != 2:
        raise RuntimeError("bottom rectangle: expected exactly the trivial zeros 0, -2")
    # grid of signs
    t0 = 1.0 / den
    ts, zs, sg, im_max = grid_signs(t0, T, den, workers)
    br = brackets_from_grid(ts, sg)
    log(f"[{time.time() - t_start:7.1f}s] grid step 1/{den}: {len(ts)} points, {len(br)} sign changes, "
        f"undecided signs {int(np.sum(sg == 0))}, max |Im Z| {im_max:.2e}")
    # (B) counts at block boundaries; compare and refine locally
    bounds = sorted(set([float(x) for x in np.arange(block_step, T, block_step)] + heights))
    bcount = {}
    for Tb in bounds:
        bcount[Tb] = backlund_count(Tb)
        if bcount[Tb]["N"] is None:
            raise RuntimeError(f"Backlund count at T={Tb} not an integer: {bcount[Tb]}")
    log(f"[{time.time() - t_start:7.1f}s] Backlund counts at {len(bounds)} heights: N({T:g}) = {bcount[float(T)]['N']}")
    refinements, blocks = [], []
    prev = 1.0
    final_br = []
    for Tb in bounds:
        want = bcount[Tb]["N"] - (bcount[prev]["N"] if prev in bcount else 0)
        cur = [x for x in br if prev < x[1] <= Tb and x[0] >= prev]
        d = den
        while len(cur) != want:
            d *= 8
            if d > den * 8 ** 3:
                raise RuntimeError(f"cannot match the count in ({prev}, {Tb}]: {len(cur)} sign changes vs N diff {want}")
            ts2, zs2, sg2, im2 = grid_signs(prev, Tb, d, workers)
            im_max = max(im_max, im2)
            cur = brackets_from_grid(ts2, sg2)
            refinements.append({"interval": [prev, Tb], "den": d, "sign_changes": len(cur), "N_diff": want})
            log(f"   refined ({prev:g}, {Tb:g}] with step 1/{d}: {len(cur)} sign changes, need {want}")
        blocks.append({"interval": [prev, Tb], "N_diff": int(want), "sign_changes": len(cur), "grid_den": d})
        final_br.extend(cur)
        prev = Tb
    final_br.sort()
    log(f"[{time.time() - t_start:7.1f}s] {len(final_br)} certified sign changes in (1, {T:g}]")
    # (A) full rectangle
    fr = full_rectangle_counts(heights, workers)
    log(f"[{time.time() - t_start:7.1f}s] full-rectangle windings: "
        + ", ".join(f"T={c['T']:g}: {c['N_minus_N1']}" for c in fr["counts"]))
    # refinement
    ref = refine_all(final_br, workers)
    gam = np.array([r[1] for r in ref], dtype=np.float64)
    eps = np.array([r[2] for r in ref], dtype=np.float64)
    hp_rad = np.array([r[4] for r in ref], dtype=np.float64)
    ok = all(r[3] for r in ref)
    log(f"[{time.time() - t_start:7.1f}s] refined {len(ref)} zeros, all certified: {ok}, max eps {np.nanmax(eps):.2e}, "
        f"high-precision radius max {np.max(hp_rad):.1e} (nan: {int(np.sum(np.isnan(hp_rad)))})")
    if not ok:
        raise RuntimeError("some zero failed its certificate")
    if not np.all(np.diff(gam) > 0):
        raise RuntimeError("ordinates not strictly increasing")
    # disjoint certificate intervals
    disjoint = bool(np.all(gam[1:] - eps[1:] > gam[:-1] + eps[:-1]))
    counts = []
    for Th in heights:
        A = next(c for c in fr["counts"] if c["T"] == Th)
        Bc = bcount[Th]
        nsc = int(np.sum(gam <= Th))
        counts.append({"T": Th, "sign_changes": nsc, "N_backlund": Bc["N"], "N_winding": A["N_minus_N1"],
                       "N_smooth": Bc["N_smooth"], "N_smooth_ball": Bc["N_smooth_ball"],
                       "N_smooth_asymptotic": n_smooth_asymptotic(Th), "S": Bc["S"], "S_ball": Bc["S_ball"],
                       "agree": bool(nsc == Bc["N"] == A["N_minus_N1"])})
    # independent mpmath check on a sample
    sample = sorted(set([0, 1, 2, 3, 4, len(gam) // 4, len(gam) // 2, 3 * len(gam) // 4, len(gam) - 2, len(gam) - 1]))
    import mpmath as mp
    mpc = []
    for i in sample:
        r = mpmath_zero(float(gam[i]))
        with mp.workdps(30):
            d_hp = abs(r - mp.mpf(ref[i][0]))
        mpc.append({"index": i + 1, "stored": float(gam[i]), "arb_hp": ref[i][0], "mpmath": mp.nstr(r, 27),
                    "abs_diff_stored": float(abs(r - gam[i])), "abs_diff_hp": float(d_hp)})
    log(f"[{time.time() - t_start:7.1f}s] mpmath check: max |mpmath - stored| "
        f"{max(d['abs_diff_stored'] for d in mpc):.2e}, max |mpmath - Arb high precision| "
        f"{max(d['abs_diff_hp'] for d in mpc):.2e}")
    reality_hp = reality_check_high_precision()
    gaps = np.diff(gam)
    res = {
        "gam": gam, "eps": eps, "tstar": [r[0] for r in ref], "hp_radius": hp_rad,
        "manifest": {
            "L_function": "L(s, chi), chi = (n/5) the Legendre symbol mod 5 (real, even, primitive; Conrey label 5.4)",
            "character_values_n_1_to_5": [1, -1, -1, 1, 0],
            "python_flint_character": f"dirichlet_char({MODULUS}, {CONREY_INDEX})",
            "functional_equation": "Lambda(s) = (5/pi)^{s/2} Gamma(s/2) L(s,chi) = Lambda(1-s), root number tau(chi)/sqrt5 = 1",
            "theta": "theta_chi(t) = (t/2) log(5/pi) + Im logGamma(1/4 + i t/2)",
            "Z": "Z_chi(t) = exp(i theta_chi(t)) L(1/2+it, chi), real",
            "N_smooth": "theta_chi(T)/pi = (T/2pi) log(5T/(2 pi e)) - 1/8 + O(1/T); N(T) = N_smooth(T) + S(T)",
            "T_max": float(T),
            "count": int(len(gam)),
            "first_ordinate": float(gam[0]),
            "last_ordinate": float(gam[-1]),
            "max_error": float(np.max(eps)),
            "error_semantics": ("rigorous: for each n, the n-th zero ordinate gamma_n satisfies |gamma_n - stored_n| < eps_n "
                                "<= max_error, certified by opposite ball-arithmetic signs of Z_chi at stored_n -+ eps_n "
                                "(eps_n = ulp/2 + 1e-15)"),
            "all_zeros_on_line_and_simple": bool(all(c["agree"] for c in counts) and disjoint
                                                 and all(b["N_diff"] == b["sign_changes"] for b in blocks)),
            "hp_radius": float(np.max(hp_rad)),
            "hp_semantics": ("rigorous: |gamma_n - ordinates_hp[n]| < hp_radius, the same sign-change certificate "
                             "at 30 digits around the printed decimal"),
            "certification": {
                "statement": (f"every zero of L(s,(n/5)) with 0 < Im s <= {T:g} is simple, has Re s = 1/2 and is one of the "
                              f"{len(gam)} stored ordinates (to within its eps); the counts below agree"),
                "bottom_rectangle_winding_[-2.5,3.5]x[-1,1]": b0,
                "counts": counts,
                "block_heights_backlund": {str(k): v["N"] for k, v in bcount.items()},
                "blocks": blocks,
                "facts_used": ["L(s,chi) != 0 for Re s > 1 (Euler product); hence (functional equation) the "
                               "non-trivial zeros have 0 <= Re s <= 1, inside the rectangles of (A)",
                               "Lambda(s) = eps Lambda(1-s) with eps = +-1 for real primitive chi; the ball "
                               "enclosure of tau(chi)/sqrt5 excludes -1, so eps = 1 (used by Z real and by (B) only)",
                               "zeros with Re s < 0 are the trivial ones 0, -2, -4, ... (from the functional equation)",
                               "Arb/python-flint enclosures are correct (acb_dirichlet_l on Re s >= 1/2, "
                               "acb_dirichlet_l_jet via Hurwitz zeta, lgamma, exp, arg)"],
                "grid_step": f"1/{den}",
                "local_refinements": refinements,
                "full_rectangle": {k: v for k, v in fr.items() if k != "counts"},
                "certificate_intervals_disjoint": disjoint,
                "method": ("sign changes of Z_chi in Arb ball arithmetic (python-flint acb.dirichlet_l) on a grid; "
                           "counts by the argument principle in ball arithmetic: (A) winding number of L around "
                           "[1/2-3, 1/2+3] x [1, T] plus [-2.5,3.5]x[-1,1] (= 2 trivial zeros), with every value and "
                           "Taylor coefficient of L from acb_dirichlet_l_jet (Hurwitz zeta, Euler-Maclaurin; no "
                           "functional equation); "
                           "(B) N(T) = theta_chi(T)/pi + S(T) with S(T) from the certified arg of L on "
                           "7/2+iT -> 1/2+iT (functional equation, root number 1); segment test: order-6 Taylor model "
                           "with box remainder, values in a convex set avoiding 0"),
            },
            "reality_check": {"max_abs_Im_Z_upper_on_grid": im_max,
                              "grid_dps": GRID_DPS,
                              "note": "the Im Z_chi ball contains 0 at every grid point (else the run aborts)",
                              "high_precision": reality_hp},
            "root_number": info["root_number"],
            "functional_equation_check": info["functional_equation"],
            "hardy_z_crosscheck": info["hardy_z_crosscheck"],
            "independent_mpmath_check": {"max_abs_diff_stored": max(d["abs_diff_stored"] for d in mpc),
                                         "max_abs_diff_hp": max(d["abs_diff_hp"] for d in mpc),
                                         "zeros": mpc,
                                         "method": "mpmath.findroot on Re[exp(i theta) mpmath.dirichlet(1/2+it, chi)] at 30 digits"},
            "min_gap": float(np.min(gaps)), "min_gap_at": float(gam[int(np.argmin(gaps))]),
            "dps": {"grid": GRID_DPS, "segments": SEG_DPS, "refine": REFINE_DPS},
            "wall_seconds": time.time() - t_start,
            "workers": workers,
            "script": "scripts/compute_lchi5_zeros.py",
        },
    }
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--T", type=float, default=1000.0)
    ap.add_argument("--heights", default=",".join(str(int(h)) for h in DEFAULT_HEIGHTS))
    ap.add_argument("--den", type=int, default=16, help="grid step 1/den (den a power of 2)")
    ap.add_argument("--block", type=float, default=50.0)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--out", default=DEFAULT_OUT)
    a = ap.parse_args()
    if a.den & (a.den - 1):
        ap.error("--den must be a power of 2")
    heights = [float(x) for x in a.heights.split(",") if x]
    res = compute(a.T, heights, a.den, a.block, a.workers)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    np.save(a.out + ".npy", res["gam"])
    man = res["manifest"]
    man["ordinates_hp"] = res["tstar"]
    man["eps"] = [float(e) for e in res["eps"]]
    with open(a.out + ".json", "w") as f:
        json.dump(man, f, indent=1)
    print("saved", a.out + ".npy", a.out + ".json")
    print(json.dumps({k: man[k] for k in ("count", "first_ordinate", "last_ordinate", "max_error",
                                          "all_zeros_on_line_and_simple")}))
    for c in man["certification"]["counts"]:
        print(c)


if __name__ == "__main__":
    main()
