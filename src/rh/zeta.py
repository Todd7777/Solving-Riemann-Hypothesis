"""Riemann–Siegel machinery for the Hardy Z function on the critical line.

Everything in this module is implemented *independently of Arb*: the Riemann–
Siegel theta function, the Riemann–Siegel formula for Z(t) with the correction
terms C_0..C_4, an Euler–Maclaurin evaluation for small t, Gram points, and a
zero finder based on sign changes of Z.  Arb (python-flint) and mpmath appear
only in the reference functions ``Z_arb``, ``Z_mp`` and ``theta_mp`` that the
test-suite compares against.

Notation.  Z(t) = exp(i theta(t)) zeta(1/2 + i t) is real for real t and has
the same zeros as zeta on the critical line.  theta is the Riemann–Siegel theta
function, theta(t) = Im log Gamma(1/4 + i t/2) - (t/2) log pi.  The Gram points
g_n are the solutions of theta(g_n) = n pi (n >= -1).  N(T) is the number of
zeros rho = beta + i gamma of zeta with 0 < gamma <= T and N(T) = theta(T)/pi
+ 1 + S(T) (Riemann–von Mangoldt / Backlund).

What is rigorous and what is not
--------------------------------
* The truncation error of the Riemann–Siegel formula is bounded by Gabcke's
  theorem (``rs_remainder_bound``), valid for t >= 200, and the Euler–Maclaurin
  remainder is bounded by the classical estimate (``em_remainder_bound``).
* Float64 rounding is *not* tracked.  ``Z_rounding_estimate`` gives a heuristic
  (generous) estimate of it, which is used as part of an error *budget* when
  signs of Z are certified in :mod:`rh.verify`.  A fully rigorous version
  would use ball arithmetic (which is what Arb does).

References
----------
[Edwards]   H. M. Edwards, Riemann's Zeta Function, Academic Press 1974
            (Dover 2001), ch. 6 (Euler–Maclaurin), ch. 7 (Riemann–Siegel), ch. 8.
[Gabcke]    W. Gabcke, Neue Herleitung und explizite Restabschätzung der
            Riemann-Siegel-Formel, Dissertation, Göttingen 1979.
[Haselgrove] C. B. Haselgrove, J. C. P. Miller, Tables of the Riemann Zeta
            Function, Royal Soc. Math. Tables 6, Cambridge 1960 (the series
            for C_0..C_4 in z = 2p - 1 reproduced in Edwards §7.5).
[Odlyzko]   A. M. Odlyzko, A. Schönhage, Fast algorithms for multiple
            evaluations of the Riemann zeta function, Trans. AMS 309 (1988).
[Brent]     R. P. Brent, On the zeros of the Riemann zeta function in the
            critical strip, Math. Comp. 33 (1979) 1361–1372 (Gram blocks,
            Rosser's rule, practical zero finding).
[Lehman]    R. S. Lehman, On the distribution of zeros of the Riemann
            zeta-function, Proc. London Math. Soc. (3) 20 (1970) 303–320.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from fractions import Fraction

import mpmath as mp
import numpy as np
from scipy.optimize import brentq
from scipy.special import lambertw

PI = math.pi
TWO_PI = 2.0 * math.pi
LOG_PI = math.log(math.pi)
EPS = np.finfo(float).eps

# --------------------------------------------------------------------------
# theta
# --------------------------------------------------------------------------

#: below this height the asymptotic expansion of theta is replaced by mpmath.
#: The expansion's error is ~ exp(-pi t)/2 (the exponentially small terms of
#: Stirling's series for log Gamma(1/4 + i t/2)), i.e. 1e-14 at t = 10 and
#: 2e-17 (below double precision) at t = 12.
THETA_ASYMPTOTIC_TMIN = 12.0
THETA_FALLBACK_DPS = 25


def _theta_coefficients(nterms: int) -> list[float]:
    """Coefficients c_n of the asymptotic expansion

        theta(t) ~ (t/2) log(t/2pi) - t/2 - pi/8 + sum_{n>=1} c_n / t^(2n-1),

    c_n = (2^(2n-1) - 1) |B_{2n}| / (2n (2n-1) 2^(2n)):  1/48, 7/5760,
    31/80640, 127/430080, 511/1216512, ... (Edwards §6.5, eq. (5)).
    """
    out = []
    for n in range(1, nterms + 1):
        p, q = mp.bernfrac(2 * n)
        b = abs(Fraction(int(p), int(q)))
        c = (Fraction(2 ** (2 * n - 1) - 1) * b) / (2 * n * (2 * n - 1) * 2 ** (2 * n))
        out.append(float(c))
    return out


THETA_COEFFS = _theta_coefficients(7)  # terms through t^-13


def theta_asymptotic(t, nterms: int = 7):
    """Riemann–Siegel theta by the asymptotic expansion only (no fallback).

    Accurate to ~exp(-pi t)/2 absolute for t >= 10 (the series terms through
    t^-13 are far below that).  Vectorised; returns float64.
    """
    t = np.asarray(t, dtype=float)
    inv = 1.0 / t
    inv2 = inv * inv
    poly = np.zeros_like(t)
    for c in THETA_COEFFS[nterms - 1::-1]:
        poly = c + poly * inv2
    return 0.5 * t * np.log(t / TWO_PI) - 0.5 * t - PI / 8 + poly * inv


def theta_mp(t, dps: int = 30):
    """theta(t) = Im log Gamma(1/4 + i t/2) - (t/2) log pi with mpmath (mpf)."""
    with mp.workdps(dps):
        t = mp.mpf(t)
        return mp.im(mp.loggamma(mp.mpc(mp.mpf(1) / 4, t / 2))) - t / 2 * mp.log(mp.pi)


def theta(t):
    """Riemann–Siegel theta function theta(t), t >= 0 (vectorised, float64).

    Uses the asymptotic expansion for t >= THETA_ASYMPTOTIC_TMIN and mpmath
    (``theta_mp``) below.  theta is odd; negative arguments are handled by
    symmetry.
    """
    t = np.asarray(t, dtype=float)
    scalar = t.ndim == 0
    t = np.atleast_1d(t)
    sgn = np.sign(t)
    at = np.abs(t)
    out = np.empty_like(at)
    big = at >= THETA_ASYMPTOTIC_TMIN
    if big.any():
        out[big] = theta_asymptotic(at[big])
    small = ~big
    if small.any():
        out[small] = [float(theta_mp(x, THETA_FALLBACK_DPS)) for x in at[small]]
    out *= np.where(sgn == 0, 1.0, sgn)
    return float(out[0]) if scalar else out


def theta_prime(t):
    """theta'(t) = (1/2) log(t/2pi) - sum (2n-1) c_n / t^(2n)  (t >= ~10)."""
    t = np.asarray(t, dtype=float)
    inv2 = 1.0 / (t * t)
    poly = np.zeros_like(t)
    for n in range(len(THETA_COEFFS), 0, -1):
        poly = (2 * n - 1) * THETA_COEFFS[n - 1] + poly * inv2
    return 0.5 * np.log(t / TWO_PI) - poly * inv2


_THETA_INV_TMIN = 7.0


def theta_inverse(y, tol: float = 1e-13, maxiter: int = 30):
    """Solve theta(t) = y for t on the increasing branch t >= 2pi (vectorised).

    Initial guess from inverting the main term with the Lambert W function:
    x (log x - 1) = (y + pi/8)/pi, x = t/2pi  =>  x = exp(1 + W(w/e)).
    Then Newton's method with ``theta`` and ``theta_prime``.  Requires
    y >= theta(7) ~ -3.51 (theta is increasing on [2pi, inf) with minimum
    theta(2pi) ~ -3.53; the iterate is kept on t >= 7, where theta' > 0.05,
    so that Newton's method converges; g_{-1} = 9.667 is the smallest Gram
    point needed in practice).
    """
    y = np.asarray(y, dtype=float)
    scalar = y.ndim == 0
    y = np.atleast_1d(y)
    w = (y + PI / 8) / PI
    x = np.exp(1.0 + lambertw(np.maximum(w / math.e, -1.0 / math.e)).real)
    t = np.maximum(TWO_PI * x, _THETA_INV_TMIN)
    for _ in range(maxiter):
        dt = (theta(t) - y) / theta_prime(t)
        t = np.maximum(t - dt, _THETA_INV_TMIN)
        if np.all(np.abs(dt) <= tol * np.maximum(1.0, t)):
            break
    return float(t[0]) if scalar else t


# --------------------------------------------------------------------------
# Gram points
# --------------------------------------------------------------------------

def gram_point(n):
    """The n-th Gram point g_n, theta(g_n) = n pi, n >= -1 (vectorised).

    g_{-1} = 9.6669..., g_0 = 17.8455..., g_1 = 23.1702...  Gram's law says
    that gamma_{n+1}, the (n+1)-st zero, 'usually' lies in (g_{n-1}, g_n).
    Computed with ``theta_inverse`` (Lambert-W start + Newton); accurate to
    ~1e-10 absolute in float64 (the limit set by cancellation in theta).
    """
    n = np.asarray(n, dtype=float)
    return theta_inverse(n * PI)


def gram_index(t):
    """Largest n with g_n <= t, i.e. floor(theta(t)/pi), for t >= g_{-1}."""
    return np.floor(theta(t) / PI).astype(np.int64)


# --------------------------------------------------------------------------
# Riemann–Siegel coefficient functions C_k(p)
# --------------------------------------------------------------------------

RS_KMAX = 4


def _rs_coefficient_table(J: int = 64, dps: int = 60) -> np.ndarray:
    """Taylor coefficients (in z = 2p - 1) of the Riemann–Siegel functions
    C_0(p), ..., C_4(p); returns an array of shape (5, J+1).

    With Psi(p) = cos(2 pi (p^2 - p - 1/16)) / cos(2 pi p) (an entire function
    of p, the apparent poles at p = 1/4, 3/4 being removable) the coefficient
    functions are [Edwards §7.5 / Haselgrove]

        C_0 = Psi
        C_1 = -Psi^(3) / (96 pi^2)
        C_2 =  Psi^(2) / (64 pi^2) + Psi^(6) / (18432 pi^4)
        C_3 = -Psi^(1) / (64 pi^2) - Psi^(5) / (3840 pi^4) - Psi^(9) / (5308416 pi^6)
        C_4 =  Psi / (128 pi^2) + 19 Psi^(4) / (24576 pi^4)
               + 11 Psi^(8) / (5898240 pi^6) + Psi^(12) / (2038431744 pi^8).

    Substituting p = (1 + z)/2 gives Psi = -cos(pi z^2/2 - 5 pi/8) / cos(pi z),
    whose Taylor series we obtain by exact power-series division at high
    precision; derivatives in p are then d/dp = 2 d/dz term by term.  The
    result reproduces Haselgrove's tabulated series, e.g.
    C_0 = 0.38268343236508977173 + 0.43724046807752044936 z^2 + ...,
    C_1 = -0.02682510262837534703 z + 0.01378477342635185305 z^3 + ...,
    C_2 = 0.00518854283029316849 + ..., C_3 = -0.00133971609071945690 z + ...,
    C_4 = 0.00046483389361763382 - ...
    """
    with mp.workdps(dps):
        pi = mp.pi
        c58, s58 = mp.cos(5 * pi / 8), mp.sin(5 * pi / 8)
        u = [mp.mpf(0)] * (J + 1)   # numerator -cos(pi z^2/2 - 5pi/8)
        for k in range(0, J // 2 + 1):
            if 4 * k <= J:
                u[4 * k] += -c58 * (-1) ** k * (pi / 2) ** (2 * k) / mp.factorial(2 * k)
            if 4 * k + 2 <= J:
                u[4 * k + 2] += -s58 * (-1) ** k * (pi / 2) ** (2 * k + 1) / mp.factorial(2 * k + 1)
        v = [mp.mpf(0)] * (J + 1)   # denominator cos(pi z)
        for k in range(0, J // 2 + 1):
            v[2 * k] = (-1) ** k * pi ** (2 * k) / mp.factorial(2 * k)
        psi = [mp.mpf(0)] * (J + 1)  # power-series quotient
        for j in range(J + 1):
            s = u[j]
            for i in range(1, j + 1):
                s -= v[i] * psi[j - i]
            psi[j] = s

        def deriv(k):  # coefficients of Psi^(k)(p) as a series in z
            out = [mp.mpf(0)] * (J + 1)
            for j in range(k, J + 1):
                out[j - k] = psi[j] * mp.mpf(2) ** k * mp.factorial(j) / mp.factorial(j - k)
            return out

        def combine(terms):
            out = [mp.mpf(0)] * (J + 1)
            for c, k in terms:
                d = deriv(k)
                for j in range(J + 1):
                    out[j] += c * d[j]
            return out

        C = [
            combine([(mp.mpf(1), 0)]),
            combine([(-1 / (96 * pi ** 2), 3)]),
            combine([(1 / (64 * pi ** 2), 2), (1 / (18432 * pi ** 4), 6)]),
            combine([(-1 / (64 * pi ** 2), 1), (-1 / (3840 * pi ** 4), 5),
                     (-1 / (5308416 * pi ** 6), 9)]),
            combine([(1 / (128 * pi ** 2), 0), (mp.mpf(19) / (24576 * pi ** 4), 4),
                     (mp.mpf(11) / (5898240 * pi ** 6), 8),
                     (1 / (2038431744 * pi ** 8), 12)]),
        ]
        return np.array([[float(x) for x in row] for row in C], dtype=float)


#: RS_COEFFS[k, j] = coefficient of z^j in C_k, z = 2p - 1.  Computed once at
#: import (about 50 ms); coefficients beyond j ~ 52 are below 1e-22.
RS_COEFFS = _rs_coefficient_table()


def rs_C(k: int, p):
    """Riemann–Siegel coefficient function C_k(p), 0 <= p < 1, k = 0..4."""
    z = 2.0 * np.asarray(p, dtype=float) - 1.0
    return np.polynomial.polynomial.polyval(z, RS_COEFFS[k])


def psi_direct(p):
    """Psi(p) = cos(2 pi (p^2 - p - 1/16)) / cos(2 pi p) directly (= C_0)."""
    p = np.asarray(p, dtype=float)
    return np.cos(TWO_PI * (p * p - p - 1.0 / 16)) / np.cos(TWO_PI * p)


# --------------------------------------------------------------------------
# Riemann–Siegel formula
# --------------------------------------------------------------------------

#: Gabcke's constants: for t >= 200 and 0 <= K <= 4,
#:     |R_K(t)| <= GABCKE_C[K] * t^(-(2K+3)/4)
#: where R_K is the remainder of the Riemann–Siegel formula after the
#: correction terms C_0, ..., C_K  [Gabcke 1979; the constants as quoted by
#: Odlyzko–Schönhage (1988), Pugh (1998, M.Sc. thesis UBC) and the Arb
#: documentation of acb_dirichlet_zeta_rs].  The measured errors against
#: mpmath (tests/test_zeta.py) are below these bounds at every point tried.
GABCKE_C = (0.127, 0.053, 0.011, 0.031, 0.017)
GABCKE_TMIN = 200.0
RS_TMIN = TWO_PI  # the formula needs N = floor(sqrt(t/2pi)) >= 1


def rs_remainder_bound(t, K: int = 4):
    """Rigorous bound on the truncation error of ``Z_rs(t, K)`` (Gabcke 1979).

    Returns GABCKE_C[K] * t^(-(2K+3)/4) for t >= 200 and +inf below 200,
    where no proven bound is implemented.  (For K = 4 this is 0.017 t^(-11/4):
    about 8e-9 at t = 200, 1e-10 at t = 1000, 7e-16 at t = 75000.)
    """
    if not 0 <= K <= RS_KMAX:
        raise ValueError("K must be in 0..4")
    t = np.asarray(t, dtype=float)
    b = GABCKE_C[K] * np.power(t, -(2 * K + 3) / 4.0)
    return np.where(t >= GABCKE_TMIN, b, np.inf)


def _rs_chunk(t, K):
    a = np.sqrt(t / TWO_PI)
    N = np.floor(a).astype(np.int64)
    p = a - N
    th = theta(t)
    nmax = int(N.max())
    n = np.arange(1, nmax + 1, dtype=float)
    phase = th[:, None] - t[:, None] * np.log(n)[None, :]
    terms = np.cos(phase) * (1.0 / np.sqrt(n))[None, :]
    terms[n[None, :] > N[:, None]] = 0.0
    main = 2.0 * terms.sum(axis=1)
    z = 2.0 * p - 1.0
    ainv = 1.0 / a
    corr = np.zeros_like(t)
    for k in range(K, -1, -1):  # Horner in a^-1
        corr = corr * ainv + np.polynomial.polynomial.polyval(z, RS_COEFFS[k])
    sign = np.where(N & 1, 1.0, -1.0)  # (-1)^(N-1)
    return main + sign * corr / np.sqrt(a)


def Z_rs(t, K: int = 4, chunk: int = 8192):
    """Hardy's Z(t) by the Riemann–Siegel formula with correction terms
    C_0..C_K (0 <= K <= 4), vectorised in float64:

        Z(t) = 2 sum_{n=1}^{N} n^{-1/2} cos(theta(t) - t log n)
               + (-1)^{N-1} (t/2pi)^{-1/4} sum_{k=0}^{K} C_k(p) (t/2pi)^{-k/2}
               + R_K(t),

    with a = sqrt(t/2pi), N = floor(a), p = a - N  [Edwards §7.4–7.5,
    Odlyzko–Schönhage §2].  |R_K| is bounded by ``rs_remainder_bound`` for
    t >= 200.  Needs t >= 2pi (N >= 1); accuracy degrades quickly below
    t ~ 30 (use ``Z_em`` or ``Z`` there).
    """
    if not 0 <= K <= RS_KMAX:
        raise ValueError("K must be in 0..4")
    t = np.asarray(t, dtype=float)
    scalar = t.ndim == 0
    t = np.atleast_1d(t)
    if t.size and t.min() < RS_TMIN:
        raise ValueError("Z_rs requires t >= 2 pi")
    out = np.empty_like(t)
    for i in range(0, t.size, chunk):
        out[i:i + chunk] = _rs_chunk(t[i:i + chunk], K)
    return float(out[0]) if scalar else out


_RS_POWERS = np.arange(RS_COEFFS.shape[1], dtype=float)


def _Z_rs_scalar(t: float, K: int = 4) -> float:
    """Scalar fast path of ``Z_rs`` (same arithmetic, less numpy overhead)."""
    a = math.sqrt(t / TWO_PI)
    N = int(a)
    p = a - N
    th = float(theta_asymptotic(t)) if t >= THETA_ASYMPTOTIC_TMIN else theta(t)
    n = np.arange(1, N + 1, dtype=float)
    main = 2.0 * float(np.sum(np.cos(th - t * np.log(n)) / np.sqrt(n)))
    z = 2.0 * p - 1.0
    C = RS_COEFFS @ (z ** _RS_POWERS)   # C_0(p), ..., C_4(p)
    ainv = 1.0 / a
    corr = 0.0
    for k in range(K, -1, -1):
        corr = corr * ainv + float(C[k])
    return main + (1.0 if N & 1 else -1.0) * corr / math.sqrt(a)


# --------------------------------------------------------------------------
# Euler–Maclaurin evaluation (small t)
# --------------------------------------------------------------------------

EM_K = 10      # number of Bernoulli terms
EM_EXTRA = 10  # M = ceil(t) + EM_EXTRA
_BERN = [float(Fraction(int(p), int(q))) for p, q in (mp.bernfrac(2 * k) for k in range(1, EM_K + 2))]
_BERN_OVER_FACT = [_BERN[k - 1] / math.factorial(2 * k) for k in range(1, EM_K + 2)]


def zeta_em(t, M: int | None = None, K: int = EM_K):
    """zeta(1/2 + i t) by the Euler–Maclaurin summation formula, vectorised.

        zeta(s) = sum_{n<M} n^{-s} + M^{1-s}/(s-1) + M^{-s}/2
                  + sum_{k=1}^{K} B_{2k}/(2k)! s(s+1)...(s+2k-2) M^{-s-2k+1} + R,
        |R| <= |(s+2K+1)/(sigma+2K+1)| |B_{2K+2}/(2K+2)!| |s(s+1)...(s+2K)| M^{-sigma-2K-1}

    [Edwards §6.4].  Returns (values, remainder_bounds).  M defaults to
    ceil(max t) + EM_EXTRA, which makes the remainder < 1e-17 for t <= 500.
    Intended for 0 <= t <= a few hundred (cost is O(M) per point); measured
    accuracy against mpmath is ~1e-13 up to t = 800.
    """
    t = np.atleast_1d(np.asarray(t, dtype=float))
    s = 0.5 + 1j * t
    if M is None:
        M = int(math.ceil(float(t.max()))) + EM_EXTRA
    n = np.arange(1, M, dtype=float)
    terms = (1.0 / np.sqrt(n))[None, :] * np.exp(-1j * t[:, None] * np.log(n)[None, :])
    total = terms.sum(axis=1)
    Ms = np.exp(-s * math.log(M))  # M^{-s}
    total += M * Ms / (s - 1.0) + 0.5 * Ms
    prod = s.copy()                 # s(s+1)...(s+2k-2)
    Mpow = Ms / M                   # M^{-s-2k+1}
    for k in range(1, K + 1):
        if k > 1:
            prod = prod * (s + (2 * k - 3)) * (s + (2 * k - 2))
        total += _BERN_OVER_FACT[k - 1] * prod * Mpow
        Mpow = Mpow / (M * M)
    prod = prod * (s + (2 * K - 1)) * (s + 2 * K)   # s(s+1)...(s+2K)
    bound = (np.abs(s + (2 * K + 1)) / (0.5 + 2 * K + 1)) * abs(_BERN_OVER_FACT[K]) \
        * np.abs(prod) * M ** (-(0.5 + 2 * K + 1))
    return total, bound


def em_remainder_bound(t, K: int = EM_K):
    """Per-point Euler–Maclaurin truncation bound with M = ceil(t) + EM_EXTRA
    (a lower bound on the M actually used, hence conservative)."""
    t = np.atleast_1d(np.asarray(t, dtype=float))
    out = np.empty_like(t)
    for i, x in enumerate(t):
        out[i] = zeta_em(np.array([x]), K=K)[1][0]
    return out


def Z_em(t, K: int = EM_K):
    """Z(t) = Re(exp(i theta(t)) zeta(1/2 + i t)) with zeta by Euler–Maclaurin."""
    t = np.asarray(t, dtype=float)
    scalar = t.ndim == 0
    t = np.atleast_1d(t)
    out = np.empty_like(t)
    chunk = 2048
    for i in range(0, t.size, chunk):
        tt = t[i:i + chunk]
        z, _ = zeta_em(tt, K=K)
        out[i:i + chunk] = np.real(np.exp(1j * theta(tt)) * z)
    return float(out[0]) if scalar else out


# --------------------------------------------------------------------------
# Dispatcher and error estimates
# --------------------------------------------------------------------------

#: Z uses Euler–Maclaurin below this height and Riemann–Siegel at/above it.
#: Must be >= GABCKE_TMIN so that Gabcke's bound applies to every
#: Riemann–Siegel evaluation; 500 is chosen because Euler–Maclaurin is still
#: cheap there (M ~ 510 terms) and ~1e-13 accurate, whereas the K = 4
#: Riemann–Siegel formula is only ~1e-9 accurate at t ~ 200.
Z_EM_TMAX = 500.0
assert Z_EM_TMAX >= GABCKE_TMIN


def Z(t, K: int = RS_KMAX):
    """Hardy's Z function in float64, vectorised.

    Euler–Maclaurin (``Z_em``) for t < Z_EM_TMAX = 500, Riemann–Siegel with
    K+1 correction terms (``Z_rs``) above.  Truncation error bounded by
    ``Z_truncation_bound``; float64 rounding estimated by
    ``Z_rounding_estimate`` (heuristic).
    """
    t = np.asarray(t, dtype=float)
    scalar = t.ndim == 0
    t = np.atleast_1d(t)
    out = np.empty_like(t)
    small = t < Z_EM_TMAX
    if small.any():
        out[small] = Z_em(t[small])
    if (~small).any():
        out[~small] = Z_rs(t[~small], K)
    return float(out[0]) if scalar else out


def Z_scalar(t: float, K: int = RS_KMAX) -> float:
    """Scalar Z(t) (same dispatch as ``Z``; faster for one-point calls)."""
    t = float(t)
    if t < Z_EM_TMAX:
        return float(Z_em(np.array([t]))[0])
    return _Z_rs_scalar(t, K)


def Z_truncation_bound(t, K: int = RS_KMAX):
    """Rigorous bound on the truncation error of ``Z(t, K)`` (no rounding):
    Euler–Maclaurin remainder below Z_EM_TMAX, Gabcke's bound above."""
    t = np.atleast_1d(np.asarray(t, dtype=float))
    out = np.empty_like(t)
    small = t < Z_EM_TMAX
    if small.any():
        out[small] = em_remainder_bound(t[small])
    if (~small).any():
        out[~small] = rs_remainder_bound(t[~small], K)
    return out


def Z_rounding_estimate(t):
    """HEURISTIC estimate of the float64 rounding error of ``Z(t)``.

    The dominant rounding error comes from the phases theta(t) - t log n (or
    t log n in Euler–Maclaurin), each known to ~eps times its magnitude, so
        err ~ 4 eps (|theta(t)| + t log L) sum_{n<=L} n^{-1/2} (+ 4 eps),
    L = number of terms.  Measured errors against mpmath are 10–100x smaller
    (about 1e-10 at t = 75000); this is a deliberately generous budget, not a
    proof.
    """
    t = np.atleast_1d(np.asarray(t, dtype=float))
    L = np.where(t < Z_EM_TMAX, np.ceil(t) + EM_EXTRA, np.floor(np.sqrt(t / TWO_PI)))
    L = np.maximum(L, 1.0)
    th = np.abs(theta(t))
    return 4.0 * EPS * (th + t * np.log(L + 1.0)) * (2.0 * np.sqrt(L) + 1.0) + 4.0 * EPS


def Z_error_budget(t, K: int = RS_KMAX):
    """Truncation bound (rigorous) + rounding estimate (heuristic)."""
    return Z_truncation_bound(t, K) + Z_rounding_estimate(t)


# --------------------------------------------------------------------------
# High-precision / rigorous references
# --------------------------------------------------------------------------

def Z_mp(t, dps: int = 30):
    """Z(t) = Re(exp(i theta(t)) zeta(1/2 + i t)) with mpmath at `dps` digits.

    mpmath's zeta uses its own Riemann–Siegel implementation with heuristic
    error control for large t; this is a high-precision *reference*, not a
    rigorous one.  Returns an mpf.
    """
    with mp.workdps(dps):
        t = mp.mpf(t)
        th = mp.im(mp.loggamma(mp.mpc(mp.mpf(1) / 4, t / 2))) - t / 2 * mp.log(mp.pi)
        return mp.re(mp.expj(th) * mp.zeta(mp.mpc(mp.mpf(1) / 2, t)))


def Z_arb(t, dps: int = 30):
    """Rigorous ball for Z(t) with Arb (python-flint): Re(e^{i theta} zeta).

    theta is computed as Im lgamma(1/4 + i t/2) - (t/2) log pi in ball
    arithmetic; the float `t` is converted exactly.  Returns a flint ``arb``.
    """
    from flint import acb, arb, ctx
    old = ctx.dps
    ctx.dps = dps
    try:
        ta = arb(float(t))
        z = acb(arb(0.5), ta).zeta()
        th = acb(arb(0.25), ta / 2).lgamma().imag - (ta / 2) * arb.pi().log()
        return z.real * th.cos() - z.imag * th.sin()
    finally:
        ctx.dps = old


# --------------------------------------------------------------------------
# Riemann–von Mangoldt main term and S(T)
# --------------------------------------------------------------------------

def N_smooth(t):
    """Main term theta(t)/pi + 1 of N(t) = theta(t)/pi + 1 + S(t)."""
    return theta(t) / PI + 1.0


def S(t, N_t):
    """S(t) = N(t) - N_smooth(t) for a given (e.g. counted) value N(t)."""
    return np.asarray(N_t, dtype=float) - N_smooth(t)


def count_below(zeros, t):
    """Number of entries of the sorted array `zeros` that are <= t."""
    return int(np.searchsorted(np.asarray(zeros), t, side="right"))


# --------------------------------------------------------------------------
# Zero finding
# --------------------------------------------------------------------------

GRID_UNIFORM_TMAX = 10.0   # below: plain grid (theta is not monotone there)
GRID_UNIFORM_STEP = 0.5
_NO_INDEX = np.iinfo(np.int64).min


@dataclass
class ZeroSearch:
    """Result of ``find_zeros``.

    zeros          refined ordinates (sorted)
    lo, hi         grid brackets lo < zero < hi with Z(lo) Z(hi) < 0; disjoint
    Z_lo, Z_hi     the float64 values of Z at the bracket ends
    gram_blocks    list of (n_a, n_b, expected, found) for every complete Gram
                   block [g_{n_a}, g_{n_b}) inside the range (after refinement)
    refined_blocks number of Gram blocks that needed a finer grid
    unresolved_blocks  blocks still short of zeros after max_refine_level
    """
    t_min: float
    t_max: float
    zeros: np.ndarray
    lo: np.ndarray
    hi: np.ndarray
    Z_lo: np.ndarray
    Z_hi: np.ndarray
    n_grid: int
    pts_per_spacing: int
    refine: str
    gram_blocks: list = field(default_factory=list)
    refined_blocks: int = 0
    unresolved_blocks: list = field(default_factory=list)
    n_zero_grid_values: int = 0

    def __len__(self):
        return int(self.zeros.size)


def _gram_grid(t_min: float, t_max: float, pts: int):
    """Grid on [t_min, t_max]: uniform below GRID_UNIFORM_TMAX, then points
    t with theta(t)/pi = U/pts for consecutive integers U (so the grid has
    `pts` points per Gram interval and contains the Gram points, at U % pts
    == 0).  Returns (t_grid, U) with U = _NO_INDEX for non-Gram-coordinate
    points."""
    ts, Us = [], []
    if t_min < GRID_UNIFORM_TMAX:
        tu = np.arange(t_min, min(GRID_UNIFORM_TMAX, t_max), GRID_UNIFORM_STEP)
        ts.append(tu)
        Us.append(np.full(tu.size, _NO_INDEX, dtype=np.int64))
        t_lo = GRID_UNIFORM_TMAX
    else:
        t_lo = t_min
    if t_lo <= t_max:
        u_lo = theta(t_lo) / PI
        u_hi = theta(t_max) / PI
        U_lo = int(math.ceil(u_lo * pts - 1e-9))
        U_hi = int(math.floor(u_hi * pts + 1e-9))
        if U_hi >= U_lo:
            U = np.arange(U_lo, U_hi + 1, dtype=np.int64)
            tg = theta_inverse(U.astype(float) * (PI / pts))
            keep = (tg > t_lo) & (tg < t_max)
            ts.append(tg[keep])
            Us.append(U[keep])
        ts.append(np.array([t_lo, t_max]))
        Us.append(np.array([_NO_INDEX, _NO_INDEX], dtype=np.int64))
    t = np.concatenate(ts)
    U = np.concatenate(Us)
    order = np.argsort(t, kind="stable")
    t, U = t[order], U[order]
    keep = np.ones(t.size, dtype=bool)
    keep[1:] = np.diff(t) > 0
    return t[keep], U[keep]


def _sign_change_brackets(t, Zt):
    """Indices i with Z(t_i) Z(t_{i+1}) < 0 (exact zeros are counted as +)."""
    s = np.sign(Zt)
    nzero = int(np.sum(s == 0))
    s[s == 0] = 1.0
    idx = np.nonzero(s[:-1] * s[1:] < 0)[0]
    return idx, nzero


def _refine_brentq(lo, hi, K, xtol):
    f = lambda x: Z_scalar(x, K)
    out = np.empty(lo.size)
    for i in range(lo.size):
        out[i] = brentq(f, lo[i], hi[i], xtol=xtol, rtol=4 * EPS, maxiter=200)
    return out


def _refine_illinois(lo, hi, flo, fhi, K, xtol, maxiter=100):
    """Vectorised Illinois (modified regula falsi) with bisection safeguard.

    Stops when b - a <= xtol + 4 eps |b| (the same rule as scipy's brentq;
    a purely absolute tolerance could never be met once xtol is below one
    float64 ulp of t, e.g. 1.8e-12 for t in [8192, 16384)).  Stragglers (none
    in practice) are handed to brentq.
    """
    a, b, fa, fb = lo.copy(), hi.copy(), flo.copy(), fhi.copy()
    n = a.size
    side = np.zeros(n, dtype=np.int8)
    active = np.ones(n, dtype=bool)
    width_ref = b - a
    for it in range(maxiter):
        idx = np.nonzero(active)[0]
        if idx.size == 0:
            break
        aa, bb, faa, fbb = a[idx], b[idx], fa[idx], fb[idx]
        c = (aa * fbb - bb * faa) / (fbb - faa)
        # safeguard: every 4th iteration, bisect brackets that shrank by < 1/2
        if it % 4 == 3:
            slow = (bb - aa) > 0.5 * width_ref[idx]
            c = np.where(slow, 0.5 * (aa + bb), c)
            width_ref[idx] = bb - aa
        c = np.minimum(np.maximum(c, np.nextafter(aa, bb)), np.nextafter(bb, aa))
        fc = Z(c, K)
        exact = fc == 0.0
        same_b = (np.sign(fc) == np.sign(fbb)) & ~exact
        same_a = ~same_b & ~exact
        # replace b
        j = idx[same_b]
        b[j] = c[same_b]; fb[j] = fc[same_b]
        rep = side[j] == -1
        fa[j[rep]] *= 0.5
        side[j] = -1
        # replace a
        j = idx[same_a]
        a[j] = c[same_a]; fa[j] = fc[same_a]
        rep = side[j] == 1
        fb[j[rep]] *= 0.5
        side[j] = 1
        j = idx[exact]
        a[j] = c[exact]; b[j] = c[exact]
        active[idx] = (b[idx] - a[idx]) > xtol + 4.0 * EPS * np.abs(b[idx])
    if active.any():   # should not happen; fall back to Brent for stragglers
        j = np.nonzero(active)[0]
        a[j] = b[j] = _refine_brentq(lo[j], hi[j], K, xtol)
        fa[j] = fb[j] = 0.0
    # final secant estimate inside [a, b] (Illinois halving spoils fa, fb as
    # function values, so recompute them at the ends when needed)
    with np.errstate(divide="ignore", invalid="ignore"):
        c = np.where(fb != fa, (a * fb - b * fa) / (fb - fa), 0.5 * (a + b))
    return np.minimum(np.maximum(c, a), b)


def _blocks_from_gram_points(n_idx, good):
    """Gram blocks [n_a, n_b) between consecutive good Gram points."""
    goods = n_idx[good]
    return [(int(goods[i]), int(goods[i + 1])) for i in range(goods.size - 1)]


def find_zeros(t_min: float, t_max: float, pts_per_spacing: int = 8,
               refine: str = "brent", K: int = RS_KMAX, max_refine_level: int = 3,
               xtol: float = 1e-12, details: bool = False):
    """Locate all zeros of Z (= zeros of zeta on the critical line) in
    [t_min, t_max] by sign changes.

    1. Grid.  Points are placed uniformly in the 'Gram coordinate'
       u = theta(t)/pi, `pts_per_spacing` per unit, so the spacing is
       pi/(pts_per_spacing theta'(t)) ~ (2pi/log(t/2pi))/pts_per_spacing,
       i.e. a fixed fraction of the mean zero spacing, and every Gram point
       is a grid point.  (Below t = 10 a plain grid is used; Z < 0 there.)
    2. Brackets.  Each sign change of Z between consecutive grid points gives
       a bracket; brackets are disjoint grid intervals.
    3. Gram blocks (Brent 1979, §3).  A Gram point g_n is 'good' if
       (-1)^n Z(g_n) > 0.  Between consecutive good Gram points lies a Gram
       block [g_n, g_{n+k}); Rosser's rule (an empirical rule, violated for
       the first time near the 13,999,826th Gram point, far above anything
       done here) says it contains exactly k zeros.  Blocks with fewer than
       k sign changes are re-gridded 8x, 64x, ... finer (up to
       `max_refine_level` times), which catches close pairs of zeros (Lehmer
       pairs) missed by the coarse grid.  Blocks still short after that are
       reported in ``unresolved_blocks``.  This is a heuristic completeness
       check; the rigorous one is Turing's method in :mod:`rh.verify`.
    4. Refinement of each bracket to |error| <~ xtol + 4 eps t:
       `refine="brent"` uses scipy's brentq on ``Z_scalar``;
       `refine="illinois"` a vectorised Illinois iteration (same accuracy,
       much faster for >~ 10^4 zeros).

    Returns the sorted ordinates (float64 array), or a ``ZeroSearch`` when
    `details=True`.
    """
    if t_max <= t_min:
        raise ValueError("need t_min < t_max")
    if t_min < 0:
        raise ValueError("t_min must be >= 0")
    if refine not in ("brent", "illinois"):
        raise ValueError("refine must be 'brent' or 'illinois'")
    pts = int(pts_per_spacing)
    tg, U = _gram_grid(float(t_min), float(t_max), pts)
    Zg = Z(tg, K)
    idx, nzero = _sign_change_brackets(tg, Zg)
    lo, hi, Zlo, Zhi = tg[idx], tg[idx + 1], Zg[idx], Zg[idx + 1]

    # --- Gram blocks and local refinement -------------------------------
    is_gram = (U != _NO_INDEX) & (U % pts == 0)
    n_idx = U[is_gram] // pts
    g_pts = tg[is_gram]
    good = ((-1.0) ** (n_idx % 2)) * Zg[is_gram] > 0
    blocks = _blocks_from_gram_points(n_idx, good)
    g_of = dict(zip(n_idx.tolist(), g_pts.tolist()))
    # brackets are sorted and disjoint and the Gram points are grid points,
    # so the brackets inside [g_a, g_b] form the index range
    # [searchsorted(lo, g_a), searchsorted(hi, g_b, 'right'))  -- O(log n)
    # per block instead of O(n).
    ga_arr = np.array([g_of[na] for na, _ in blocks])
    gb_arr = np.array([g_of[nb] for _, nb in blocks])
    expected_arr = np.array([nb - na for na, nb in blocks], dtype=np.int64)
    found_arr = np.maximum(0, np.searchsorted(hi, gb_arr, side="right")
                           - np.searchsorted(lo, ga_arr, side="left"))
    unresolved, n_refined = [], 0
    for ib_, (na, nb) in enumerate(blocks):
        expected, found = int(expected_arr[ib_]), int(found_arr[ib_])
        if found >= expected:
            continue
        ga, gb = g_of[na], g_of[nb]
        level = 0
        while found < expected and level < max_refine_level:
            level += 1
            n_refined += (level == 1)
            Ub = np.arange(na * pts * 8 ** level, nb * pts * 8 ** level + 1)
            tb = theta_inverse(Ub.astype(float) * (PI / (pts * 8 ** level)))
            tb[0], tb[-1] = ga, gb
            Zb = Z(tb, K)
            ib, nz = _sign_change_brackets(tb, Zb)
            nzero += nz
            if ib.size >= expected or level == max_refine_level:
                inblock = (lo >= ga) & (hi <= gb)
                keep = ~inblock
                lo = np.concatenate([lo[keep], tb[ib]])
                hi = np.concatenate([hi[keep], tb[ib + 1]])
                Zlo = np.concatenate([Zlo[keep], Zb[ib]])
                Zhi = np.concatenate([Zhi[keep], Zb[ib + 1]])
                order = np.argsort(lo, kind="stable")
                lo, hi, Zlo, Zhi = lo[order], hi[order], Zlo[order], Zhi[order]
                found = int(np.sum((lo >= ga) & (hi <= gb)))
        found_arr[ib_] = found
        if found < expected:
            unresolved.append((na, nb, expected, found))
    block_info = [(na, nb, int(e), int(f)) for (na, nb), e, f in zip(blocks, expected_arr, found_arr)]

    # --- refine ------------------------------------------------------------
    if lo.size == 0:
        zeros = np.empty(0)
    elif refine == "brent":
        zeros = _refine_brentq(lo, hi, K, xtol)
    else:
        zeros = _refine_illinois(lo, hi, Zlo, Zhi, K, xtol)
    res = ZeroSearch(t_min=float(t_min), t_max=float(t_max), zeros=zeros, lo=lo, hi=hi,
                     Z_lo=Zlo, Z_hi=Zhi, n_grid=int(tg.size), pts_per_spacing=pts,
                     refine=refine, gram_blocks=block_info, refined_blocks=n_refined,
                     unresolved_blocks=unresolved, n_zero_grid_values=nzero)
    return res if details else zeros
