r"""Nyman–Beurling–Báez-Duarte criterion: the distances d_N.

Statement
---------
Nyman (1950) and Beurling (1955) proved that the Riemann Hypothesis is
equivalent to the density, in L^2(0,1), of the span of the functions
x -> {theta/x} - theta {1/x}, 0 < theta <= 1.  Báez-Duarte (2003, Atti Accad.
Naz. Lincei (9) 14, 5-11) strengthened this to a *sequence* of functions: with
rho_k(x) := {1/(kx)}  (fractional part; k = 1, 2, ...), considered in
L^2((0,infinity), dx),

    RH  <==>  chi_(0,1]  belongs to the closed linear span of {rho_k : k >= 1}.

Equivalently, defining the "Báez-Duarte distances"

    d_N^2 := inf_{c in R^N} || chi_(0,1] - sum_{k<=N} c_k rho_k ||^2
          = inf_{a}  (1/2pi) int_R |1 - zeta(1/2+it) A_N(1/2+it)|^2 dt/(1/4+t^2),

(A_N(s) = sum_{k<=N} a_k k^{-s}, a_k = -c_k; the second form follows from the
Mellin–Plancherel identity and  int_0^inf {1/x} x^{s-1} dx = -zeta(s)/s on
0 < Re s < 1),  RH holds if and only if d_N -> 0.

Known results
-------------
* Burnol (2002, Adv. Math. 170): liminf d_N^2 log N >= sum_rho m_rho^2/|rho|^2,
  the sum over the distinct zeros ON the critical line with multiplicities.  If
  RH holds and all zeros are simple this constant is
      C = sum_rho 1/|rho|^2 = 2 + gamma - log(4 pi) = 0.0461914179...
* Báez-Duarte, Balazard, Landreau, Saias (2000, Adv. Math. 149) conjectured
  d_N^2 ~ C / log N.  Numerical work: Landreau & Richard, Experiment. Math. 11
  (2002).  Bettin, Conrey & Farmer (2013) studied the "Möbius mollifier"
  a_k = mu(k)(1 - log k/log N) and showed (under RH + a hypothesis on the zeros)
  that it attains the Burnol constant asymptotically.

Computation
-----------
The Gram matrix G_{jk} = <rho_j, rho_k> = int_0^inf {u/j}{u/k} u^{-2} du has
the closed form (Vasyunin 1995; Báez-Duarte–Balazard–Landreau–Saias 2005), for
coprime h, k:

    G(h,k) = (log 2pi - gamma)/2 * (1/h + 1/k) + (k-h)/(2hk) * log(h/k)
             - pi/(2hk) * ( V(h,k) + V(k,h) ),
    V(h,k) = sum_{m=1}^{k-1} {m h / k} cot(pi m / k)        (Vasyunin sum),

and G(j,k) = G(j/g, k/g)/g for g = gcd(j,k).  THIS FORMULA WAS RE-VERIFIED in
this repository against a direct high-precision evaluation of the integral
(moment expansion + Hurwitz zeta tail): to 37 significant digits for many
coprime pairs by `scripts/verify_vasyunin_formula.py` (mpmath, 40 digits), and
to 1e-13 in the float64 test suite (tests/test_nyman_beurling.py).  The right-hand side vector is
b_k = <chi_(0,1], rho_k> = int_1^inf {u/k} u^{-2} du = (1 - gamma + log k)/k,
and ||chi||^2 = 1, so  d_N^2 = 1 - b^T G^{-1} b.

Because G_n is the leading n x n block of G_N, one Cholesky factorisation
G_N = L L^T yields d_n^2 = 1 - sum_{i<=n} y_i^2 (y = L^{-1} b) for ALL n <= N.
The system is only polynomially ill-conditioned (cond(G_N) ~ N^2.0, measured:
9.4e3, 3.7e4, 1.5e5, 6.2e5, 2.4e6, 9.7e6 at N = 50, 100, ..., 1600), so float64
is adequate for N up to ~10^4 (the float64 Vasyunin sums lose ~log10(k^2)
digits from the rounding of pi*m/k near m = k, i.e. ~1e-9 absolute at
k = 10^4, which enters the Gram entries multiplied by pi/(2hk)); a rigorous
ball-arithmetic (Arb) version is provided for moderate N and agrees with the
float64 values to ~1e-14 up to N = 800.
"""
from __future__ import annotations

import math
from math import gcd

import numpy as np

from .constants import EULER_GAMMA, LOG_2PI, BAEZ_DUARTE_C

__all__ = [
    "vasyunin_sum", "vasyunin_table", "gram_entry", "gram_matrix", "rhs_vector",
    "baez_duarte_distances", "mobius_mollifier_bound", "mobius_mollifier_coefficients",
    "gram_entry_arb", "baez_duarte_distance_arb", "vasyunin_sum_arb",
    "gram_entry_direct_mp", "BAEZ_DUARTE_C",
]

_L2PI_G = LOG_2PI - EULER_GAMMA  # log(2 pi) - gamma


# ----------------------------------------------------------------------------
# Vasyunin sums
# ----------------------------------------------------------------------------
def vasyunin_sum(h: int, k: int) -> float:
    """V(h,k) = sum_{m=1}^{k-1} {m h/k} cot(pi m/k)  in float64 (vectorised)."""
    if k == 1:
        return 0.0
    m = np.arange(1, k, dtype=np.int64)
    frac = ((m * h) % k) / k
    return float(np.dot(frac, 1.0 / np.tan(np.pi * m / k)))


def vasyunin_table(N: int, chunk: int = 2000, progress=None):
    """All Vasyunin sums V(a, q) for 1 <= a < q <= N, gcd(a,q)=1, as a dict
    {(a, q): float}.  Cost ~ 0.2 N^3 flops (vectorised per q)."""
    tab = {}
    for q in range(2, N + 1):
        m = np.arange(1, q, dtype=np.int64)
        cot = 1.0 / np.tan(np.pi * m / q)
        a_all = np.array([a for a in range(1, q) if gcd(a, q) == 1], dtype=np.int64)
        for s in range(0, len(a_all), chunk):
            a = a_all[s:s + chunk]
            frac = ((a[:, None] * m[None, :]) % q).astype(np.float64) / q
            vals = frac @ cot
            for ai, vi in zip(a.tolist(), vals.tolist()):
                tab[(ai, q)] = vi
        if progress and q % 500 == 0:
            progress(q)
    return tab


def _V_from_table(tab, h, k):
    """V(h,k) with h, k coprime, using the table of V(a,q) for a<q (V(h,k) for
    h>k reduces to V(h mod k, k))."""
    if k == 1:
        return 0.0
    return tab[(h % k, k)]


# ----------------------------------------------------------------------------
# Gram matrix and right-hand side
# ----------------------------------------------------------------------------
def gram_entry_coprime(h: int, k: int, Vhk: float, Vkh: float) -> float:
    return (_L2PI_G / 2 * (1.0 / h + 1.0 / k)
            + (k - h) / (2.0 * h * k) * math.log(h / k)
            - math.pi / (2.0 * h * k) * (Vhk + Vkh))


def gram_entry(j: int, k: int) -> float:
    """G_{jk} = int_0^inf {u/j}{u/k} u^{-2} du (float64, closed form)."""
    g = gcd(j, k)
    h, kk = j // g, k // g
    return gram_entry_coprime(h, kk, vasyunin_sum(h, kk), vasyunin_sum(kk, h)) / g


def gram_matrix(N: int, tab=None, progress=None) -> np.ndarray:
    """The N x N Gram matrix of rho_1..rho_N in float64."""
    if tab is None:
        tab = vasyunin_table(N, progress=progress)
    G = np.empty((N, N), dtype=np.float64)
    for j in range(1, N + 1):
        for k in range(j, N + 1):
            g = gcd(j, k)
            h, kk = j // g, k // g
            v = gram_entry_coprime(h, kk, _V_from_table(tab, h, kk), _V_from_table(tab, kk, h)) / g
            G[j - 1, k - 1] = v
            G[k - 1, j - 1] = v
    return G


def rhs_vector(N: int) -> np.ndarray:
    """b_k = <chi_(0,1], rho_k> = (1 - gamma + log k)/k."""
    k = np.arange(1, N + 1, dtype=np.float64)
    return (1.0 - EULER_GAMMA + np.log(k)) / k


# ----------------------------------------------------------------------------
# The distances
# ----------------------------------------------------------------------------
def baez_duarte_distances(N: int, G: np.ndarray | None = None, tab=None, return_coeffs=False):
    """Return the array d_n^2 for n = 1..N (float64) via one Cholesky factorisation.

    If return_coeffs, also return the optimal coefficients c (length N) for the
    full problem, i.e. the minimiser of ||chi - sum c_k rho_k||.
    """
    import scipy.linalg as sla
    if G is None:
        G = gram_matrix(N, tab=tab)
    b = rhs_vector(N)
    L = sla.cholesky(G, lower=True, check_finite=False)
    y = sla.solve_triangular(L, b, lower=True, check_finite=False)
    d2 = 1.0 - np.cumsum(y * y)
    if return_coeffs:
        c = sla.solve_triangular(L.T, y, lower=False, check_finite=False)
        return d2, c
    return d2


def mobius_mollifier_coefficients(N: int) -> np.ndarray:
    """c_k = -mu(k) (1 - log k / log N), k <= N  (so that A_N(s) = -sum c_k k^{-s}
    is the standard Möbius mollifier approximating 1/zeta(s))."""
    from sympy import mobius
    k = np.arange(1, N + 1)
    mu = np.array([int(mobius(int(x))) for x in k], dtype=np.float64)
    w = 1.0 - np.log(k) / math.log(N) if N > 1 else np.ones(1)
    return -mu * w


def mobius_mollifier_bound(N: int, G: np.ndarray, c: np.ndarray | None = None) -> float:
    """||chi - sum_{k<=N} c_k rho_k||^2 = 1 - 2 b.c + c^T G c for the given c
    (default: the Möbius mollifier).  Any c gives an UPPER bound for d_N^2."""
    if c is None:
        c = mobius_mollifier_coefficients(N)
    b = rhs_vector(N)
    return float(1.0 - 2.0 * b @ c + c @ G[:N, :N] @ c)


# ----------------------------------------------------------------------------
# Rigorous (ball arithmetic) versions with Arb via python-flint
# ----------------------------------------------------------------------------
def vasyunin_sum_arb(h: int, k: int):
    from flint import arb
    if k == 1:
        return arb(0)
    pi = arb.pi()
    s = arb(0)
    for m in range(1, k):
        s += (arb((m * h) % k) / k) * (pi * m / k).cot()
    return s


def gram_entry_arb(j: int, k: int, Vcache=None):
    """Rigorous ball enclosure of G_{jk}."""
    from flint import arb
    g = gcd(j, k)
    h, kk = j // g, k // g

    def V(a, q):
        if q == 1:
            return arb(0)
        key = (a % q, q)
        if Vcache is not None and key in Vcache:
            return Vcache[key]
        v = vasyunin_sum_arb(a % q, q)
        if Vcache is not None:
            Vcache[key] = v
        return v

    pi = arb.pi()
    A = ((2 * arb.pi()).log() - arb.const_euler()) / 2 * (arb(1) / h + arb(1) / kk)
    B = arb(kk - h) / (2 * h * kk) * (arb(h) / kk).log()
    Cc = -pi / (2 * h * kk) * (V(h, kk) + V(kk, h))
    return (A + B + Cc) / g


def baez_duarte_distance_arb(N: int, dps: int = 60, progress=None):
    """Rigorous enclosure (arb ball) of d_N^2 by solving the Gram system in ball
    arithmetic.  Also returns the leading-block values d_n^2 for n <= N? No — a
    single solve; call for the N you need.  Cost ~ 0.2 N^3 arb cot evaluations
    for the Vasyunin sums (dominant) + O(N^3) for the solve."""
    from flint import arb, arb_mat, ctx
    old = ctx.dps
    ctx.dps = dps
    try:
        Vcache = {}
        G = arb_mat(N, N)
        for j in range(1, N + 1):
            for k in range(j, N + 1):
                v = gram_entry_arb(j, k, Vcache)
                G[j - 1, k - 1] = v
                G[k - 1, j - 1] = v
            if progress and j % 50 == 0:
                progress(j)
        b = arb_mat(N, 1, [(1 - arb.const_euler() + arb(k).log()) / k for k in range(1, N + 1)])
        c = G.solve(b)
        bc = arb(0)
        for k in range(N):
            bc += b[k, 0] * c[k, 0]
        return 1 - bc, [c[k, 0] for k in range(N)]
    finally:
        ctx.dps = old


# ----------------------------------------------------------------------------
# Independent high-precision evaluation of the Gram integral (for validation)
# ----------------------------------------------------------------------------
def gram_entry_direct_mp(h: int, k: int, K: int = 200, dps: int = 40):
    """int_0^inf {u/h}{u/k} u^{-2} du for coprime integers h,k evaluated
    directly: exact integration over the first K periods (length L = hk) plus a
    tail expansion sum_m (-1)^m (m+1) L^{-(m+2)} mu_m zeta(m+2, K), where
    mu_m = int_0^L f(v) v^m dv are exact moments of the periodic function f.
    This does not use the Vasyunin formula and serves to validate it."""
    import mpmath as mp
    with mp.workdps(dps):
        L = h * k

        def I_period(kk):
            s = mp.mpf(0)
            for m in range(kk * L, (kk + 1) * L):
                a = m // h
                bb = m // k
                if m == 0:
                    s += mp.mpf(1) / (h * k)
                else:
                    s += (mp.mpf(1) / (h * k)
                          - (mp.mpf(a) / k + mp.mpf(bb) / h) * mp.log(mp.mpf(m + 1) / m)
                          + mp.mpf(a * bb) * (mp.mpf(1) / m - mp.mpf(1) / (m + 1)))
            return s

        head = sum(I_period(kk) for kk in range(K))

        def moment(mo):
            s = mp.mpf(0)
            for m in range(L):
                a = m // h
                bb = m // k
                c2 = mp.mpf(1) / (h * k)
                c1 = -(mp.mpf(a) / k + mp.mpf(bb) / h)
                c0 = mp.mpf(a * bb)
                s += (c2 * (mp.mpf(m + 1) ** (mo + 3) - mp.mpf(m) ** (mo + 3)) / (mo + 3)
                      + c1 * (mp.mpf(m + 1) ** (mo + 2) - mp.mpf(m) ** (mo + 2)) / (mo + 2)
                      + c0 * (mp.mpf(m + 1) ** (mo + 1) - mp.mpf(m) ** (mo + 1)) / (mo + 1))
            return s

        tail = mp.mpf(0)
        for mo in range(0, 80):
            term = (-1) ** mo * (mo + 1) * mp.mpf(L) ** (-(mo + 2)) * moment(mo) * mp.zeta(mo + 2, K)
            tail += term
            if mo > 5 and abs(term) < mp.mpf(10) ** (-dps):
                break
        return head + tail
