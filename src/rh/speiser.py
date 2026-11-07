"""Speiser's criterion: zeros of zeta'(s) and the Riemann Hypothesis.

Speiser (1934) proved that RH is equivalent to the statement that zeta'(s)
has no zeros in the half-strip 0 < Re s < 1/2.  Levinson and Montgomery
(1974) gave a quantitative form: zeta and zeta' have, up to O(log T), the same
number of zeros in 0 < sigma < 1/2, 0 < t < T, and under RH every derivative
zeta^(k) has at most finitely many non-real zeros in sigma < 1/2.  They also
proved unconditionally that zeta' has only real zeros in the half-plane
sigma <= 0 (one between consecutive trivial zeros of zeta), and that zeta' can
vanish on the critical line only at a multiple zero of zeta.  Berndt (1970)
proved that the number N_1(T) of zeros of zeta' with 0 < t < T satisfies

    N_1(T) = (T/2pi) log(T/(4pi)) - T/(2pi) + O(log T)
           = N(T) - (T/2pi) log 2 + O(log T),

where N(T) counts the zeros of zeta.  The zeros of zeta' cluster to the right
of the critical line, roughly in 1/2 < sigma < 2.5; the zero with the smallest
positive ordinate is 2.46316... + 23.29832... i (Spira 1965).

Zero-free regions used to bound the search rectangle
-----------------------------------------------------
* sigma >= 4:  |zeta'(s) + 2^{-s} log 2| <= sum_{n>=3} (log n) n^{-4}
  = -zeta'(4) - (log 2)/16 = 0.02559 < (log 2)/16 = 0.04332, and the ratio of
  the two sides decreases with sigma, so zeta'(s) != 0 for sigma >= 4.
  (Titchmarsh, Thm 11.5(C), gives a sharper region.)
* Real axis: zeta is strictly decreasing on (-2.717.., 1) and on (1, oo), so
  zeta'(sigma) != 0 for -1 <= sigma <= 4 except for the double pole at s = 1.
  The real zeros of zeta' (between consecutive trivial zeros: -2.7173,
  -4.9368, ...) lie outside the rectangles scanned here.
* sigma <= 0: no non-real zeros at all (Levinson-Montgomery); the scan
  nevertheless starts at sigma = -1 as an independent check.

Method
------
1. The rectangle [sigma_min, sigma_max] x [t_min, T] (default [-1, 4] x
   [0.25, T]; t_min > 0 keeps the double pole of zeta' at s = 1 outside) is cut
   into cells.  For each cell the winding number of zeta' along the boundary is
   obtained from the argument principle with adaptive sampling of arg zeta'
   (a segment is bisected until the phase jump between consecutive samples is
   below a threshold).  zeta', zeta'' come from Arb (python-flint
   ``acb_series.zeta``) and are rigorous enclosures; the phase-sampling step is
   the only non-rigorous ingredient and is controlled by three independent
   checks (below).
2. Cells with winding number 1 are handed to a complex Newton iteration started
   at the cell centre; cells with winding number >= 2 (or a Newton failure) are
   bisected and re-counted.
3. Every zero is then *certified* with the Krawczyk operator in ball
   arithmetic: for a box X around the Newton point, c = zeta''(mid X) and
   K(X) = mid - zeta'(mid)/c + (1 - zeta''(X)/c)(X - mid); if K(X) lies in
   the interior of X then X contains exactly one zero of zeta', and it lies in
   K(X) (Brouwer + contraction).  This yields rigorous bounds on Re s for each
   zero, in particular a rigorous proof that Re s > 1/2 for every zero found.
4. Consistency checks: (a) the certified boxes are pairwise disjoint (each
   holds exactly one zero, so the certified zeros are distinct: two Newton
   runs ending on the same zero would be caught) and lie inside the
   rectangle, and their number equals the winding number of the whole
   rectangle computed with an independent, finer sampling (``complete``).
   Since the zeros returned are distinct genuine zeros in the rectangle and
   their number equals its winding number, none was missed.  (The sum of the
   cell winding numbers is not an independent check on its own: shared edges
   are cached and cancel, so it telescopes to the outer boundary sampled
   with the coarse step; an interior sampling error shifts a zero between
   neighbouring cells, which the bisection consistency w1 + w2 = w and the
   disjointness test detect.)  (b) comparison with Berndt's main term and with
   N(T) - (T/2pi) log 2, where N(T) is the rigorous zero count of zeta
   (``arb.zeta_nzeros``); (c) the rectangle [sigma_min, sigma_max] x
   [-t_min, t_min] has winding number -2 (the double pole, no zeros).

References
----------
* A. Speiser, Geometrisches zur Riemannschen Zetafunktion, Math. Ann. 110
  (1934) 514-521.
* B. C. Berndt, The number of zeros for zeta^(k)(s), J. London Math. Soc. (2)
  2 (1970) 577-580.
* N. Levinson, H. L. Montgomery, Zeros of the derivatives of the Riemann
  zeta-function, Acta Math. 133 (1974) 49-65.
* R. Spira, Zero-free regions of zeta^(k)(s), J. London Math. Soc. 40 (1965)
  677-682; Zeros of zeta'(s) and the Riemann hypothesis, Illinois J. Math. 17
  (1973) 147-152.
* R. E. Moore, Interval Analysis (1966); R. Krawczyk, Computing 4 (1969)
  187-201 (the Krawczyk operator).
* E. C. Titchmarsh, The Theory of the Riemann Zeta-Function, 2nd ed., §11.5.
"""
from __future__ import annotations

import cmath
import math
import multiprocessing as mproc
import time
from dataclasses import dataclass, asdict, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from flint import acb, arb, ctx, acb_series

TWO_PI = 2.0 * math.pi

# The first few zeros of zeta' with t > 0 (Spira 1965), used as a sanity check.
FIRST_ZETA_PRIME_ZEROS = (
    complex(2.4631618694, 23.2983204928),
    complex(1.2864968223, 31.7082500831),
    complex(2.3075700637, 38.4899831731),
    complex(1.3827636057, 42.2909645546),
    complex(0.9646856227, 48.8471599051),
)


class ArgumentPrincipleError(RuntimeError):
    """Raised when the sampled argument changes too fast (zero on/near the contour)."""


# ----------------------------------------------------------------------------
# zeta' and zeta'' via Arb power series
# ----------------------------------------------------------------------------
def _to_complex(z: acb) -> complex:
    return complex(float(z.real), float(z.imag))


def zeta_prime(s: complex, dps: int = 15) -> complex:
    """zeta'(s) as a Python complex (midpoint of the Arb enclosure)."""
    ctx.dps = dps
    c = acb_series([acb(s.real, s.imag), 1], 2).zeta().coeffs()
    return _to_complex(c[1])


def zeta_prime_jet(s: complex, dps: int = 15) -> Tuple[complex, complex]:
    """(zeta'(s), zeta''(s)) as Python complex numbers."""
    ctx.dps = dps
    c = acb_series([acb(s.real, s.imag), 1], 3).zeta().coeffs()
    return _to_complex(c[1]), 2.0 * _to_complex(c[2])


def zeta_prime_ball_jet(X: acb) -> Tuple[acb, acb]:
    """Rigorous enclosures of zeta' and zeta'' over the complex box X."""
    c = acb_series([X, 1], 3).zeta().coeffs()
    return c[1], 2 * c[2]


# ----------------------------------------------------------------------------
# Generic argument principle on rectangles
# ----------------------------------------------------------------------------
def arg_change(f: Callable[[complex], complex], z0: complex, z1: complex, n0: int = 4,
               threshold: float = 1.0, max_depth: int = 40) -> float:
    """Continuous change of arg f along the segment z0 -> z1 by adaptive sampling.

    Consecutive samples whose phase jump exceeds `threshold` (radians) are
    bisected, up to `max_depth` levels; exceeding the depth means a zero lies
    on or extremely close to the segment and ``ArgumentPrincipleError`` is
    raised (the caller then perturbs the contour).
    """
    pts = [z0 + (z1 - z0) * (k / n0) for k in range(n0 + 1)]
    vals = [f(p) for p in pts]
    if any(v == 0 for v in vals):
        raise ArgumentPrincipleError(f"f vanishes at a sample point on {z0}->{z1}")
    total = 0.0
    stack = [(pts[k], vals[k], pts[k + 1], vals[k + 1], 0) for k in range(n0)]
    while stack:
        a, fa, b, fb, d = stack.pop()
        dphi = cmath.phase(fb / fa)
        if abs(dphi) <= threshold:
            total += dphi
        elif d >= max_depth:
            raise ArgumentPrincipleError(f"argument of f changes too fast near {a}")
        else:
            m = 0.5 * (a + b)
            fm = f(m)
            if fm == 0:
                raise ArgumentPrincipleError(f"f vanishes at {m}")
            stack.append((m, fm, b, fb, d + 1))
            stack.append((a, fa, m, fm, d + 1))
    return total


def _key(z0: complex, z1: complex):
    return (round(z0.real, 10), round(z0.imag, 10), round(z1.real, 10), round(z1.imag, 10))


def edge_arg_change(f, z0: complex, z1: complex, step: float, threshold: float,
                    cache: Optional[Dict] = None, max_depth: int = 40) -> float:
    """arg change along z0->z1 with caching of shared (possibly reversed) edges."""
    if cache is not None:
        k = _key(z0, z1)
        if k in cache:
            return cache[k]
        kr = _key(z1, z0)
        if kr in cache:
            return -cache[kr]
    n0 = max(1, int(math.ceil(abs(z1 - z0) / step)))
    val = arg_change(f, z0, z1, n0, threshold, max_depth)
    if cache is not None:
        cache[_key(z0, z1)] = val
    return val


def winding_number(f, s0: float, s1: float, t0: float, t1: float, step_h: float = 0.25,
                   step_v: float = 0.1, threshold: float = 1.0, cache: Optional[Dict] = None,
                   max_depth: int = 40) -> int:
    """Number of zeros minus poles of f in the rectangle [s0,s1] x [t0,t1].

    Boundary traversed counter-clockwise; raises ``ArgumentPrincipleError`` if
    the total is not within 0.05 of an integer.
    """
    A, B, C, D = complex(s0, t0), complex(s1, t0), complex(s1, t1), complex(s0, t1)
    total = (edge_arg_change(f, A, B, step_h, threshold, cache, max_depth)
             + edge_arg_change(f, B, C, step_v, threshold, cache, max_depth)
             + edge_arg_change(f, C, D, step_h, threshold, cache, max_depth)
             + edge_arg_change(f, D, A, step_v, threshold, cache, max_depth))
    w = total / TWO_PI
    if abs(w - round(w)) > 0.05:
        raise ArgumentPrincipleError(f"non-integer winding number {w} on [{s0},{s1}]x[{t0},{t1}]")
    return int(round(w))


# ----------------------------------------------------------------------------
# Zero isolation: cells -> winding numbers -> Newton
# ----------------------------------------------------------------------------
@dataclass
class LocatedZero:
    re: float
    im: float
    residual: float            # |f(z)| at the Newton point
    newton_steps: int
    multiplicity: int = 1      # > 1 only if a tiny cell still has winding number > 1
    cell: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)

    @property
    def z(self) -> complex:
        return complex(self.re, self.im)


class _Counted:
    """Wrap f to count evaluations."""

    def __init__(self, f):
        self.f = f
        self.n = 0

    def __call__(self, z):
        self.n += 1
        return self.f(z)


def newton_refine(jet, z0: complex, tol: float = 1e-12, max_steps: int = 60,
                  box: Optional[Tuple[float, float, float, float]] = None,
                  escape: float = 3.0) -> Tuple[Optional[complex], int, float]:
    """Complex Newton iteration z <- z - f/f' using jet(z) = (f(z), f'(z)).

    Returns (zero or None, steps, |f(zero)|).  Fails if the iterate escapes
    far from `box` (more than `escape` box-diagonals from its centre).
    """
    z = z0
    if box is not None:
        s0, s1, t0, t1 = box
        centre = complex(0.5 * (s0 + s1), 0.5 * (t0 + t1))
        diag = math.hypot(s1 - s0, t1 - t0)
    for k in range(1, max_steps + 1):
        fz, dfz = jet(z)
        if dfz == 0:
            return None, k, abs(fz)
        dz = fz / dfz
        z = z - dz
        if box is not None and abs(z - centre) > escape * diag:
            return None, k, float("inf")
        if abs(dz) <= tol * max(1.0, abs(z)):
            fz, _ = jet(z)
            return z, k, abs(fz)
    return None, max_steps, float("inf")


def _inside(z: complex, box, slack: float = 1e-9) -> bool:
    s0, s1, t0, t1 = box
    return (s0 - slack <= z.real <= s1 + slack) and (t0 - slack <= z.imag <= t1 + slack)


def _resolve_cell(f, jet, box, w: int, cache, params, depth: int = 0) -> List[LocatedZero]:
    """Return the zeros in a cell with known winding number w (>= 0)."""
    s0, s1, t0, t1 = box
    if w == 0:
        return []
    if w < 0:
        raise ArgumentPrincipleError(f"negative winding number {w} in {box}: pole inside?")
    size = max(s1 - s0, t1 - t0)
    if w == 1:
        z, steps, res = newton_refine(jet, complex(0.5 * (s0 + s1), 0.5 * (t0 + t1)),
                                      params["newton_tol"], box=box)
        if z is not None and _inside(z, box):
            return [LocatedZero(z.real, z.imag, res, steps, 1, box)]
    if size < params["min_size"]:
        # cannot separate further: report the centre with multiplicity w
        return [LocatedZero(0.5 * (s0 + s1), 0.5 * (t0 + t1), float("nan"), 0, w, box)]
    # bisect along the longer side; perturb the cut if it hits a zero
    for frac in (0.5, 0.53, 0.47, 0.56, 0.44, 0.6, 0.4):
        try:
            if (s1 - s0) >= (t1 - t0):
                m = s0 + frac * (s1 - s0)
                b1, b2 = (s0, m, t0, t1), (m, s1, t0, t1)
            else:
                m = t0 + frac * (t1 - t0)
                b1, b2 = (s0, s1, t0, m), (s0, s1, m, t1)
            w1 = winding_number(f, *b1, params["step_h"], params["step_v"], params["threshold"], cache)
            w2 = winding_number(f, *b2, params["step_h"], params["step_v"], params["threshold"], cache)
            if w1 + w2 != w:
                raise ArgumentPrincipleError(f"winding numbers do not add up in {box}: {w1}+{w2}!={w}")
            return (_resolve_cell(f, jet, b1, w1, cache, params, depth + 1)
                    + _resolve_cell(f, jet, b2, w2, cache, params, depth + 1))
        except ArgumentPrincipleError:
            if frac == 0.4:
                raise
            continue
    raise ArgumentPrincipleError("unreachable")


def locate_zeros(f, jet, s0: float, s1: float, t0: float, t1: float, *,
                 sigma_breaks: Optional[Sequence[float]] = None, dt: float = 0.5,
                 step_h: float = 0.25, step_v: float = 0.1, threshold: float = 1.0,
                 newton_tol: float = 1e-12, min_size: float = 1e-4,
                 ) -> Tuple[List[LocatedZero], int, int]:
    """Locate all zeros of f in [s0,s1] x [t0,t1] (f analytic there, no poles).

    Returns (zeros, total winding number summed over cells, number of f
    evaluations).  Interior grid lines that happen to pass through a zero are
    detected (``ArgumentPrincipleError``) and shifted; a zero on the outer
    boundary raises.
    """
    fc = _Counted(f)
    params = dict(step_h=step_h, step_v=step_v, threshold=threshold,
                  newton_tol=newton_tol, min_size=min_size)
    breaks0 = list(sigma_breaks) if sigma_breaks is not None else list(np.arange(s0, s1, 1.0)) + [s1]
    breaks0[0], breaks0[-1] = s0, s1
    zeros: List[LocatedZero] = []
    total_w = 0
    cache: Dict = {}
    tb = t0
    while tb < t1 - 1e-12:
        tt = min(tb + dt, t1)
        for attempt in range(8):
            shift = 0.0137 * attempt
            br = [breaks0[0]] + [b + shift for b in breaks0[1:-1]] + [breaks0[-1]]
            top = tt if (tt >= t1 or attempt == 0) else min(t1, tt + 0.0173 * attempt * dt)
            try:
                strip_zeros: List[LocatedZero] = []
                strip_w = 0
                for j in range(len(br) - 1):
                    box = (br[j], br[j + 1], tb, top)
                    w = winding_number(fc, *box, step_h, step_v, threshold, cache)
                    strip_w += w
                    strip_zeros += _resolve_cell(fc, jet, box, w, cache, params)
                break
            except ArgumentPrincipleError:
                if attempt == 7:
                    raise
        zeros += strip_zeros
        total_w += strip_w
        tb = top
        # drop cache entries below the current strip to bound memory
        if len(cache) > 20000:
            cache = {k: v for k, v in cache.items() if min(k[1], k[3]) >= tb - 1e-9}
    return zeros, total_w, fc.n


# ----------------------------------------------------------------------------
# Rigorous certification (Krawczyk operator in ball arithmetic)
# ----------------------------------------------------------------------------
@dataclass
class Certificate:
    re: float
    im: float
    certified: bool
    radius: float                 # half-width of the box X proven to contain exactly one zero
    re_lower: float               # rigorous bounds for the zero (from K(X))
    re_upper: float
    im_lower: float
    im_upper: float
    dps: int

    def to_dict(self) -> dict:
        return asdict(self)


def _float_down(x: arb) -> float:
    """Largest float <= the exact (radius-0) arb x."""
    f = float(x)
    if arb(f) > x:
        f = math.nextafter(f, -math.inf)
    return f


def _float_up(x: arb) -> float:
    """Smallest float >= the exact (radius-0) arb x."""
    f = float(x)
    if arb(f) < x:
        f = math.nextafter(f, math.inf)
    return f


def certify_zero(ball_jet, z: complex, dps: int = 30,
                 radii: Sequence[float] = (1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3)) -> Certificate:
    """Prove that a box around z contains exactly one zero of f (Krawczyk test).

    ``ball_jet(X)`` must return rigorous enclosures (f(X), f'(X)) for a complex
    box X (python-flint ``acb``).  With c = f'(mid) and
    K = mid - f(mid)/c + (1 - f'(X)/c)(X - mid), K in interior(X) implies that
    g(z) = z - f(z)/c maps X into itself (Brouwer: a zero exists) and is a
    contraction there (uniqueness); every zero in X lies in K.  The float
    bounds of K returned in the certificate are rounded outward, so they
    enclose the zero rigorously.
    """
    old = ctx.dps
    ctx.dps = dps
    try:
        mid = acb(z.real, z.imag)
        fm, dfm = ball_jet(mid)
        for r in radii:
            X = acb(arb(z.real, r), arb(z.imag, r))
            _, dX = ball_jet(X)
            if dfm.contains(acb(0)):
                continue
            K = mid - fm / dfm + (1 - dX / dfm) * (X - mid)
            if X.contains_interior(K):
                return Certificate(z.real, z.imag, True, r,
                                   _float_down(K.real.lower()), _float_up(K.real.upper()),
                                   _float_down(K.imag.lower()), _float_up(K.imag.upper()), dps)
        return Certificate(z.real, z.imag, False, float("nan"), float("nan"), float("nan"),
                           float("nan"), float("nan"), dps)
    finally:
        ctx.dps = old


def overlapping_boxes(boxes: Sequence[Tuple[float, float, float, float]]) -> List[Tuple[int, int]]:
    """Index pairs of closed boxes (re_lo, re_hi, im_lo, im_hi) that intersect.

    Each certified box contains exactly one zero, so pairwise disjoint
    certified boxes prove that the certified zeros are *distinct* (two Newton
    runs converging to the same zero would show up here as an overlap).
    Sweep over the boxes sorted by im_lo.
    """
    order = sorted(range(len(boxes)), key=lambda k: boxes[k][2])
    out: List[Tuple[int, int]] = []
    for p, i in enumerate(order):
        r0, r1, _, i1 = boxes[i]
        for j in order[p + 1:]:
            s0, s1, j0, _ = boxes[j]
            if j0 > i1:
                break
            if s0 <= r1 and r0 <= s1:
                out.append((min(i, j), max(i, j)))
    return out


# ----------------------------------------------------------------------------
# Counting functions
# ----------------------------------------------------------------------------
def berndt_main_term(T: float) -> float:
    """(T/2pi) log(T/(4pi)) - T/(2pi): main term of N_1(T) (Berndt 1970)."""
    return T / TWO_PI * math.log(T / (4 * math.pi)) - T / TWO_PI


def zeta_zero_count(T: float, dps: int = 15) -> int:
    """Rigorous N(T) = number of zeros of zeta with 0 < Im s <= T (Arb, Turing's method).

    Arb returns an exact integer for a float T; an inexact ball (which it
    returns when it cannot separate T from a zero ordinate) raises instead of
    being rounded to a possibly wrong integer.
    """
    if not (math.isfinite(T) and T > 0):          # Arb crashes on a NaN argument
        raise ValueError(f"T must be finite and positive, got {T}")
    old = ctx.dps
    ctx.dps = dps
    try:
        v = arb(T).zeta_nzeros()
        if not v.is_exact():
            raise ArithmeticError(f"N({T}) not determined exactly by Arb: {v}")
        return int(float(v))                        # an exact integer < 2^53
    finally:
        ctx.dps = old


# ----------------------------------------------------------------------------
# The Speiser scan
# ----------------------------------------------------------------------------
DEFAULT_SIGMA_BREAKS = (-1.0, 0.5, 1.0, 1.5, 2.0, 2.5, 4.0)


def _zp(z: complex) -> complex:
    c = acb_series([acb(z.real, z.imag), 1], 2).zeta().coeffs()
    return _to_complex(c[1])


def _zp_jet(z: complex) -> Tuple[complex, complex]:
    c = acb_series([acb(z.real, z.imag), 1], 3).zeta().coeffs()
    return _to_complex(c[1]), 2.0 * _to_complex(c[2])


def _scan_chunk(args) -> Tuple[List[LocatedZero], int, int]:
    ta, tb, s0, s1, breaks, dt, dps, step_h, step_v, threshold = args
    ctx.dps = dps
    return locate_zeros(_zp, _zp_jet, s0, s1, ta, tb, sigma_breaks=breaks, dt=dt,
                        step_h=step_h, step_v=step_v, threshold=threshold)


def _edge_chunk(args) -> float:
    z0, z1, step, threshold, dps = args
    ctx.dps = dps
    return edge_arg_change(_zp, z0, z1, step, threshold, None)


@dataclass
class SpeiserResult:
    T: float
    t_min: float
    sigma_min: float
    sigma_max: float
    count: int                          # zeros of zeta' located in the rectangle
    count_certified: int
    count_re_lt_half: int               # zeros with Re s < 1/2 (RH predicts 0)
    count_re_lt_half_rigorous: int      # zeros whose certified box does not lie entirely in Re s > 1/2
    min_re: float
    zero_min_re: Tuple[float, float]
    max_re: float
    zero_max_re: Tuple[float, float]
    winding_total_cells: int            # sum of cell winding numbers
    winding_total_independent: int      # whole-rectangle winding number, finer sampling
    berndt_main_term: float
    N_T: int                            # rigorous N(T) for zeta
    N_T_minus_log2_term: float          # N(T) - (T/2pi) log 2
    pole_check_winding: Optional[int]   # should be -2
    first_zeros_match_spira: bool
    evaluations: int
    wall_seconds: float
    dps: int
    zeros: List[dict] = field(default_factory=list)
    # Completeness: the certified boxes (each holding exactly one zero) are pairwise disjoint,
    # so the certified zeros are distinct, and lie inside the scanned rectangle; together with
    # count_certified == winding_total_independent this shows no zero was missed or counted twice.
    count_certified_distinct: int = 0   # connected components of overlapping certified boxes
    certified_boxes_disjoint: bool = False
    certified_inside_rectangle: bool = False
    complete: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def certification_summary(zero_dicts: Sequence[dict], s0: float, s1: float, t0: float, t1: float) -> dict:
    """Distinctness and containment of certified zeros (dicts with re/im_lower/upper).

    Returns n_certified, n_distinct (connected components of the overlap graph of
    the certified boxes: a rigorous lower bound for the number of distinct zeros
    found), disjoint (no two certified boxes meet), inside (every certified box
    lies in the open rectangle (s0, s1) x (t0, t1)) and the overlapping pairs.
    """
    idx = [k for k, d in enumerate(zero_dicts) if d.get("certified")]
    boxes = [(zero_dicts[k]["re_lower"], zero_dicts[k]["re_upper"],
              zero_dicts[k]["im_lower"], zero_dicts[k]["im_upper"]) for k in idx]
    pairs = overlapping_boxes(boxes)
    parent = list(range(len(boxes)))

    def root(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in pairs:
        parent[root(a)] = root(b)
    n_distinct = len({root(a) for a in range(len(boxes))})
    inside = all(s0 < b[0] and b[1] < s1 and t0 < b[2] and b[3] < t1 for b in boxes)
    return {"n_certified": len(boxes), "n_distinct": n_distinct, "disjoint": not pairs,
            "inside": inside, "overlapping_pairs": [(idx[a], idx[b]) for a, b in pairs]}


def speiser_scan(T: float, t_min: float = 0.25, sigma_min: float = -1.0, sigma_max: float = 4.0,
                 *, sigma_breaks: Sequence[float] = DEFAULT_SIGMA_BREAKS, dt: float = 0.5,
                 step_h: float = 0.25, step_v: float = 0.1, threshold: float = 1.0,
                 workers: int = 1, chunk: float = 50.0, certify: bool = True, dps: int = 15,
                 certify_dps: int = 30, pole_check: bool = True) -> SpeiserResult:
    """Locate, certify and count all zeros of zeta' in [sigma_min, sigma_max] x [t_min, T].

    See the module docstring for the method.  `workers` > 1 distributes
    t-chunks of height `chunk` over a process pool.
    """
    t_start = time.time()
    if not (sigma_min < 1 < sigma_max) or t_min <= 0:
        raise ValueError("need sigma_min < 1 < sigma_max and t_min > 0 (pole of zeta' at s = 1)")
    breaks = [b for b in sigma_breaks if sigma_min < b < sigma_max]
    breaks = [sigma_min] + breaks + [sigma_max]
    # --- 1-2: cells, winding numbers, Newton -------------------------------------------
    edges = list(np.arange(t_min, T, chunk)) + [T]
    tasks = [(float(edges[i]), float(edges[i + 1]), sigma_min, sigma_max, breaks, dt, dps,
              step_h, step_v, threshold) for i in range(len(edges) - 1)]
    zeros: List[LocatedZero] = []
    w_cells = 0
    nev = 0
    if workers > 1 and len(tasks) > 1:
        with mproc.get_context("fork").Pool(workers) as pool:
            results = pool.map(_scan_chunk, tasks)
    else:
        results = [_scan_chunk(t) for t in tasks]
    for zs, w, n in results:
        zeros += zs
        w_cells += w
        nev += n
    zeros.sort(key=lambda z: z.im)
    # --- 4a: independent whole-rectangle winding number ---------------------------------
    A, B, C, D = complex(sigma_min, t_min), complex(sigma_max, t_min), complex(sigma_max, T), complex(sigma_min, T)
    fine_step, fine_thr = step_v / 2.0, 0.6
    etasks = [(A, B, step_h / 2.0, fine_thr, dps), (C, D, step_h / 2.0, fine_thr, dps)]
    for z0, z1 in ((B, C), (D, A)):
        n = max(1, int(math.ceil(abs(z1 - z0) / chunk)))
        for i in range(n):
            etasks.append((z0 + (z1 - z0) * (i / n), z0 + (z1 - z0) * ((i + 1) / n), fine_step, fine_thr, dps))
    if workers > 1 and len(etasks) > 1:
        with mproc.get_context("fork").Pool(workers) as pool:
            parts = pool.map(_edge_chunk, etasks)
    else:
        parts = [_edge_chunk(t) for t in etasks]
    w_total = sum(parts) / TWO_PI
    if abs(w_total - round(w_total)) > 0.05:
        raise ArgumentPrincipleError(f"non-integer global winding number {w_total}")
    w_total = int(round(w_total))
    # --- 3: certification -----------------------------------------------------------------
    zero_dicts = []
    n_cert = 0
    n_lt_half_rig = 0
    for z in zeros:
        d = {"re": z.re, "im": z.im, "residual": z.residual, "newton_steps": z.newton_steps,
             "multiplicity": z.multiplicity}
        if certify and z.multiplicity == 1:
            c = certify_zero(zeta_prime_ball_jet, z.z, dps=certify_dps)
            d.update({"certified": c.certified, "radius": c.radius, "re_lower": c.re_lower,
                      "re_upper": c.re_upper, "im_lower": c.im_lower, "im_upper": c.im_upper})
            if c.certified:
                n_cert += 1
                if not (c.re_lower > 0.5):
                    n_lt_half_rig += 1
            else:
                n_lt_half_rig += 1
        else:
            d.update({"certified": False})
            n_lt_half_rig += 1
        zero_dicts.append(d)
    cs = certification_summary(zero_dicts, sigma_min, sigma_max, t_min, T)
    count = sum(z.multiplicity for z in zeros)
    complete = bool(cs["disjoint"] and cs["inside"] and n_cert == count == w_cells == w_total)
    # --- 4b, 4c: counts and pole check -----------------------------------------------------
    NT = zeta_zero_count(T, dps)
    pc = None
    if pole_check:
        ctx.dps = dps
        pc = winding_number(_zp, sigma_min, sigma_max, -t_min, t_min, step_h / 2, step_v / 2, 0.6)
    re_vals = np.array([z.re for z in zeros]) if zeros else np.array([np.nan])
    i_min = int(np.argmin(re_vals)) if zeros else 0
    i_max = int(np.argmax(re_vals)) if zeros else 0
    match = True
    for z0 in FIRST_ZETA_PRIME_ZEROS:
        if z0.imag < T and t_min < z0.imag:
            match &= any(abs(z.z - z0) < 1e-8 for z in zeros)
    return SpeiserResult(
        T=T, t_min=t_min, sigma_min=sigma_min, sigma_max=sigma_max,
        count=count, count_certified=n_cert,
        count_re_lt_half=int(sum(1 for z in zeros if z.re < 0.5)),
        count_re_lt_half_rigorous=n_lt_half_rig,
        min_re=float(re_vals[i_min]), zero_min_re=(zeros[i_min].re, zeros[i_min].im) if zeros else (math.nan, math.nan),
        max_re=float(re_vals[i_max]), zero_max_re=(zeros[i_max].re, zeros[i_max].im) if zeros else (math.nan, math.nan),
        winding_total_cells=w_cells, winding_total_independent=w_total,
        berndt_main_term=berndt_main_term(T), N_T=NT,
        N_T_minus_log2_term=NT - T / TWO_PI * math.log(2.0),
        pole_check_winding=pc, first_zeros_match_spira=match, evaluations=nev,
        wall_seconds=time.time() - t_start, dps=dps, zeros=zero_dicts,
        count_certified_distinct=cs["n_distinct"], certified_boxes_disjoint=cs["disjoint"],
        certified_inside_rectangle=cs["inside"], complete=complete,
    )
