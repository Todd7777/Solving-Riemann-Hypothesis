"""Spectral statistics of the nontrivial zeros of zeta against random-matrix theory.

The Montgomery–Odlyzko law says that the (suitably rescaled) ordinates gamma_n
of the zeros of zeta behave statistically like the eigenvalues of large random
Hermitian matrices from the Gaussian Unitary Ensemble (GUE), i.e. like a
determinantal point process with the sine kernel K(x, y) = sin(pi(x-y))/(pi(x-y)).
This module implements the classical statistics used to test that law:

* unfolding (``unfold``, ``normalized_spacings``) with the Riemann–Siegel theta
  function, so that the unfolded zeros x_n = theta(gamma_n)/pi + 1 have unit mean
  spacing;
* the nearest-neighbour spacing distribution (``spacing_distribution``) compared
  with the Wigner surmise, the exact GUE law p_2(s) = E_2''(0; s) computed from the
  Fredholm determinant of the sine kernel (``GUESpacing``, ``gue_gap_probability``),
  and the Poisson law e^{-s};
* Montgomery's pair correlation R_2(alpha) (``pair_correlation``) and his form
  factor F(alpha) (``montgomery_F``);
* the number variance Sigma^2(L) (``number_variance``) with Mehta's GUE formula,
  and the Berry saturation predicted by the semiclassical (prime-sum) theory
  (``number_variance_semiclassical``, ``number_variance_saturation``,
  ``detect_saturation``);
* Kolmogorov–Smirnov distances between the empirical spacing law and GUE in
  height windows (``ks_distance``, ``ks_by_height``), which quantify the slow
  approach to the GUE limit as T grows;
* the Lehmer phenomenon: the smallest normalized spacings and the
  Csordas–Smith–Varga Lehmer-pair criterion (``smallest_spacings``,
  ``lehmer_pair_statistic``).

Nothing here proves anything about RH; the module is numerical evidence
machinery.  All zero ordinates come from Arb's rigorous isolation (see
``rh.data``); the statistics themselves are plain floating-point estimates whose
sampling errors are reported where meaningful.

References
----------
* H. L. Montgomery, "The pair correlation of zeros of the zeta function",
  Proc. Sympos. Pure Math. 24 (1973), 181–193.  (Pair correlation conjecture
  R_2(alpha) = 1 - (sin(pi alpha)/(pi alpha))^2; the form factor F(alpha).)
* A. M. Odlyzko, "On the distribution of spacings between zeros of the zeta
  function", Math. Comp. 48 (1987), 273–308.  (Numerical verification of the
  Montgomery–Odlyzko law; unfolding by the local density log(gamma/2pi)/2pi.)
* M. L. Mehta, "Random Matrices", 3rd ed., Elsevier (2004).  (Gap probability
  E_2(0; s) as a Fredholm determinant and p_2(s) = E_2''; number variance
  Sigma^2(L) of the unitary ensemble.)
* M. Gaudin, Nucl. Phys. 25 (1961), 447; F. J. Dyson, J. Math. Phys. 3 (1962), 166.
* F. Bornemann, "On the numerical evaluation of Fredholm determinants",
  Math. Comp. 79 (2010), 871–915.  (Gauss–Legendre Nyström discretisation.)
* M. V. Berry, "Semiclassical formula for the number variance of the Riemann
  zeros", Nonlinearity 1 (1988), 399–407.  (Saturation of Sigma^2(L) for
  L above ~ log(T/2pi)/(2pi).)
* E. B. Bogomolny and J. P. Keating, "Gutzwiller's trace formula and spectral
  statistics: beyond the diagonal approximation", Phys. Rev. Lett. 77 (1996),
  1472–1475; "Random matrix theory and the Riemann zeros I, II", Nonlinearity 8
  (1995) 1115 and 9 (1996) 911.  (Lower-order, height-dependent corrections to
  the GUE correlations, controlled by the primes.)
* E. Bogomolny, O. Bohigas, P. Leboeuf, A. G. Monastra, "On the spacing
  distribution of the Riemann zeros: corrections to the asymptotic result",
  J. Phys. A 39 (2006), 10743–10754.  (Finite-height corrections to p(s) behave
  like those of CUE matrices of effective size N_eff proportional to log(T/2pi).)
* G. Csordas, W. Smith, R. S. Varga, "Lehmer pairs of zeros, the de Bruijn–Newman
  constant Lambda, and the Riemann hypothesis", Constr. Approx. 10 (1994), 107–129.
* D. H. Lehmer, "On the roots of the Riemann zeta-function", Acta Arith. 2 (1956).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from numpy.polynomial.chebyshev import Chebyshev
from numpy.polynomial.legendre import leggauss
from scipy import integrate, special, stats
from scipy.linalg import lu_factor, lu_solve
from scipy.optimize import brentq

from .constants import EULER_GAMMA
from .data import ZEROS_DIR

__all__ = [
    "riemann_siegel_theta", "smooth_count", "mean_density",
    "unfold", "normalized_spacings",
    "wigner_surmise", "wigner_surmise_cdf", "poisson_density", "poisson_cdf",
    "gue_gap_probability", "gue_gap_derivatives_jacobi", "GUESpacing", "gue_table",
    "gue_spacing_density", "gue_spacing_cdf",
    "spacing_distribution",
    "pair_correlation", "pair_correlation_gue", "montgomery_F", "montgomery_F_prediction",
    "montgomery_F_gue_finite",
    "number_variance", "number_variance_gue", "number_variance_poisson",
    "number_variance_semiclassical", "number_variance_saturation", "detect_saturation",
    "ks_distance", "ks_by_height", "index_windows",
    "smallest_spacings", "lehmer_pair_statistic", "lehmer_pairs",
    "load_high_sample",
]

# --------------------------------------------------------------------------- #
# Extended precision helpers
# --------------------------------------------------------------------------- #
LD = np.longdouble
# pi to 50 digits, parsed at long-double precision (np.longdouble(np.pi) would
# only carry 53 bits).
PI_LD = LD("3.14159265358979323846264338327950288419716939937510")
TWO_PI_LD = 2 * PI_LD


# --------------------------------------------------------------------------- #
# 1. Unfolding
# --------------------------------------------------------------------------- #
def _theta_ld(t):
    """Riemann–Siegel theta by its asymptotic expansion, in long double.

    theta(t) = (t/2) log(t/2pi) - t/2 - pi/8 + 1/(48 t) + 7/(5760 t^3) + O(t^-5);
    the omitted term is 31/(80640 t^5) < 1e-9 for t >= gamma_1 = 14.13.
    """
    t = np.asarray(t, dtype=LD)
    return (0.5 * t * np.log(t / TWO_PI_LD) - 0.5 * t - PI_LD / 8
            + 1 / (48 * t) + 7 / (5760 * t ** 3))


def riemann_siegel_theta(t):
    """Riemann–Siegel theta function theta(t) for t >~ 10 (asymptotic expansion).

    theta(t) = arg Gamma(1/4 + it/2) - (t/2) log pi, the phase in
    zeta(1/2 + it) = e^{-i theta(t)} Z(t).  Evaluated through the expansion
    (t/2) log(t/2pi) - t/2 - pi/8 + 1/(48t) + 7/(5760 t^3), whose truncation
    error is below 1e-9 on the range of the zeros; returned as float64.
    """
    return np.asarray(_theta_ld(t), dtype=np.float64)


def smooth_count(t):
    """Smooth part of the zero counting function: N̄(t) = theta(t)/pi + 1.

    N(t) = N̄(t) + S(t) with S(t) = O(log t) (Riemann–von Mangoldt), so N̄(gamma_n)
    is the natural unfolding of the n-th zero.
    """
    return np.asarray(_theta_ld(t) / PI_LD + 1, dtype=np.float64)


def mean_density(t):
    """Mean density of zeros at height t: rho(t) = log(t/2pi)/(2pi) (= N̄'(t) + O(t^-2))."""
    t = np.asarray(t, dtype=np.float64)
    return np.log(t / (2 * np.pi)) / (2 * np.pi)


def _unfold_ld(gammas, method="theta"):
    """Unfolded positions in long double (see ``unfold`` for the two methods)."""
    g = np.asarray(gammas, dtype=LD)
    if g.ndim != 1:
        raise ValueError("gammas must be one-dimensional")
    if method == "theta":
        return _theta_ld(g) / PI_LD + 1
    if method == "density":
        # Odlyzko's normalisation: delta_n = (gamma_{n+1} - gamma_n) log(gamma_n/2pi)/(2pi),
        # accumulated into positions anchored at N̄(gamma_1).
        x = np.empty_like(g)
        x[0] = _theta_ld(g[:1])[0] / PI_LD + 1
        if len(g) > 1:
            rho = np.log(g[:-1] / TWO_PI_LD) / TWO_PI_LD
            x[1:] = x[0] + np.cumsum(rho * np.diff(g))
        return x
    raise ValueError("method must be 'theta' or 'density'")


def unfold(gammas, method="theta"):
    """Map zero ordinates gamma_n to unfolded positions x_n of unit mean spacing.

    method="theta"   : x_n = theta(gamma_n)/pi + 1 = N̄(gamma_n), the smooth
                       counting function (so x_n ≈ n - 1/2 - S(gamma_n)).
    method="density" : Odlyzko's local rescaling, x_{n+1} - x_n =
                       (gamma_{n+1} - gamma_n) log(gamma_n/2pi)/(2pi), anchored at
                       x_1 = N̄(gamma_1).  Differs from "theta" by O(Δγ^2/γ).

    Computed in long double (x_n ~ 1e10 for the high-height sample) and
    returned as float64; use ``normalized_spacings`` for accurate differences.
    """
    return np.asarray(_unfold_ld(gammas, method), dtype=np.float64)


def normalized_spacings(gammas, method="theta"):
    """Nearest-neighbour spacings delta_n = x_{n+1} - x_n of the unfolded zeros (mean ~ 1)."""
    x = _unfold_ld(gammas, method)
    return np.asarray(np.diff(x), dtype=np.float64)


# --------------------------------------------------------------------------- #
# 2. Reference spacing laws: Wigner surmise, Poisson, exact GUE
# --------------------------------------------------------------------------- #
def wigner_surmise(s):
    """GUE Wigner surmise p_W(s) = (32/pi^2) s^2 exp(-4 s^2/pi) (unit mean, normalised)."""
    s = np.asarray(s, dtype=np.float64)
    return (32 / np.pi ** 2) * s ** 2 * np.exp(-4 * s ** 2 / np.pi)


def wigner_surmise_cdf(s):
    """CDF of the GUE Wigner surmise: erf(2s/sqrt(pi)) - (4s/pi) exp(-4 s^2/pi)."""
    s = np.asarray(s, dtype=np.float64)
    s = np.maximum(s, 0.0)
    return special.erf(2 * s / np.sqrt(np.pi)) - (4 * s / np.pi) * np.exp(-4 * s ** 2 / np.pi)


def poisson_density(s):
    """Spacing density of a Poisson process of unit density, e^{-s}."""
    s = np.asarray(s, dtype=np.float64)
    return np.where(s >= 0, np.exp(-np.maximum(s, 0)), 0.0)


def poisson_cdf(s):
    s = np.asarray(s, dtype=np.float64)
    return np.where(s >= 0, 1 - np.exp(-np.maximum(s, 0)), 0.0)


_GL_CACHE: dict[int, tuple[np.ndarray, np.ndarray]] = {}


def _gauss_legendre_01(m):
    """Gauss–Legendre nodes and weights on [0, 1] (cached)."""
    if m not in _GL_CACHE:
        x, w = leggauss(m)
        _GL_CACHE[m] = (0.5 * (x + 1), 0.5 * w)
    return _GL_CACHE[m]


def _sine_kernel_matrices(s, m):
    """Nyström matrices for the sine kernel on an interval of length s.

    After the change of variables x = s u, u in [0, 1], the operator with kernel
    sin(pi(x-y))/(pi(x-y)) on L^2[0, s] is unitarily equivalent to the operator
    on L^2[0, 1] with kernel k_s(u, v) = sin(pi s (u-v))/(pi (u-v)).  Bornemann's
    discretisation is A_ij = sqrt(w_i) k_s(u_i, u_j) sqrt(w_j) with Gauss–Legendre
    nodes u_i and weights w_i; det(I - A) converges exponentially fast to
    E_2(0; s) as m grows (the kernel is entire).  Because the nodes do not depend
    on s, dA/ds = sqrt(w_i w_j) cos(pi s (u_i-u_j)) and
    d^2A/ds^2 = -pi (u_i-u_j) sqrt(w_i w_j) sin(pi s (u_i-u_j)) are explicit.
    Returns (A, dA, d2A).
    """
    u, w = _gauss_legendre_01(m)
    du = u[:, None] - u[None, :]
    sw = np.sqrt(w[:, None] * w[None, :])
    A = sw * s * np.sinc(s * du)              # np.sinc(x) = sin(pi x)/(pi x); diag = s
    dA = sw * np.cos(np.pi * s * du)
    d2A = -np.pi * du * sw * np.sin(np.pi * s * du)
    return A, dA, d2A


def gue_gap_probability(s, m=60):
    """E_2(0; s) = det(I - K_s): probability that an interval of length s (in
    units of the mean spacing) contains no GUE eigenvalue.

    Computed as the Fredholm determinant of the sine kernel on [0, s] with
    Bornemann's (2010) Gauss–Legendre Nyström method with m nodes; m = 60 gives
    about 1e-15 absolute accuracy for 0 <= s <= 8 (the error is checked in the
    tests by comparing m = 60 with m = 90).  Vectorised over s.
    """
    s_arr = np.atleast_1d(np.asarray(s, dtype=np.float64))
    out = np.empty(s_arr.shape, dtype=np.float64)
    eye = np.eye(m)
    for idx, si in np.ndenumerate(s_arr):
        if si <= 0:
            out[idx] = 1.0
            continue
        A, _, _ = _sine_kernel_matrices(float(si), m)
        sign, logdet = np.linalg.slogdet(eye - A)
        out[idx] = sign * np.exp(logdet)
    return out if np.ndim(s) else float(out[0])


def gue_gap_derivatives_jacobi(s, m=60):
    """(E, E', E'') for E(s) = det(I - A(s)) by Jacobi's formula.

    With B = I - A:  E' = E tr(B^{-1}B'),
    E'' = E [ (tr B^{-1}B')^2 + tr(B^{-1}B'') - tr(B^{-1}B' B^{-1}B') ].
    E'' is the GUE nearest-neighbour spacing density p_2(s) and 1 + E' its CDF.
    The formula is exact but needs B^{-1}, whose norm grows like 1/E; it is
    reliable (errors < 1e-9) for s <~ 2.5 and is used as an independent check
    of the Chebyshev route implemented by ``GUESpacing``.
    """
    s = float(s)
    if s <= 0:
        return 1.0, -1.0, 0.0
    A, dA, d2A = _sine_kernel_matrices(s, m)
    B = np.eye(m) - A
    sign, logdet = np.linalg.slogdet(B)
    E = sign * np.exp(logdet)
    lu = lu_factor(B)
    X = lu_solve(lu, -dA)       # B^{-1} B'   with B' = -A'
    Y = lu_solve(lu, -d2A)      # B^{-1} B''
    trX = np.trace(X)
    trX2 = np.sum(X * X.T)      # tr(X X)
    E1 = E * trX
    E2 = E * (trX ** 2 + np.trace(Y) - trX2)
    return float(E), float(E1), float(E2)


class GUESpacing:
    """The exact GUE nearest-neighbour spacing law from the sine-kernel determinant.

    E(s) = E_2(0; s) is sampled at Chebyshev points on [0, s_max] with the
    Nyström determinant (``gue_gap_probability``) and interpolated by a
    Chebyshev polynomial of the given degree; since E is entire in s the
    interpolant converges super-geometrically and reaches machine precision for
    degree ~ 60, so its first and second derivatives (the spacing CDF
    F(s) = 1 + E'(s) and density p_2(s) = E''(s), Mehta Ch. 6) are accurate to
    about 1e-13 and 1e-11 in the interior (1e-9 for p_2 at the endpoints),
    including the far tail where E is tiny.  Tiny negative round-off in p_2 is
    clipped to zero.
    Beyond s_max (default 6, where 1 - F < 1e-20) the law is extended by
    E = 0, F = 1, p = 0.
    """

    def __init__(self, s_max=6.0, degree=96, m=60):
        self.s_max = float(s_max)
        self.degree = int(degree)
        self.m = int(m)
        self._E = Chebyshev.interpolate(lambda t: gue_gap_probability(t, m=self.m),
                                        self.degree, domain=[0.0, self.s_max])
        self._dE = self._E.deriv(1)
        self._d2E = self._E.deriv(2)

    def _clip(self, s):
        s = np.asarray(s, dtype=np.float64)
        return s, np.clip(s, 0.0, self.s_max)

    def gap(self, s):
        """E_2(0; s)."""
        s, sc = self._clip(s)
        return np.where(s <= 0, 1.0, np.where(s >= self.s_max, 0.0, self._E(sc)))

    def cdf(self, s):
        """F(s) = P(spacing <= s) = 1 + E'(s)."""
        s, sc = self._clip(s)
        return np.where(s <= 0, 0.0, np.where(s >= self.s_max, 1.0,
                                              np.clip(1.0 + self._dE(sc), 0.0, 1.0)))

    def pdf(self, s):
        """p_2(s) = E''(s)."""
        s, sc = self._clip(s)
        return np.where((s <= 0) | (s >= self.s_max), 0.0, np.maximum(self._d2E(sc), 0.0))

    def moment(self, k, n_quad=200):
        """∫ s^k p_2(s) ds by Gauss–Legendre quadrature on [0, s_max] (an
        independent check of the normalisation and of the unit mean)."""
        u, w = _gauss_legendre_01(n_quad)          # cached nodes on [0, 1]
        t = self.s_max * u
        return float(np.sum(self.s_max * w * t ** k * self.pdf(t)))

    @property
    def mean(self):
        return self.moment(1)

    @property
    def variance(self):
        return self.moment(2) - self.moment(1) ** 2

    def chebyshev_tail(self):
        """Magnitude of the last five Chebyshev coefficients (convergence diagnostic)."""
        return np.abs(self._E.coef[-5:])


_GUE_TABLE: GUESpacing | None = None


def gue_table() -> GUESpacing:
    """Lazily built shared ``GUESpacing`` instance."""
    global _GUE_TABLE
    if _GUE_TABLE is None:
        _GUE_TABLE = GUESpacing()
    return _GUE_TABLE


def gue_spacing_density(s):
    """Exact GUE nearest-neighbour spacing density p_2(s) (unit mean spacing)."""
    return gue_table().pdf(s)


def gue_spacing_cdf(s):
    """Exact GUE nearest-neighbour spacing CDF."""
    return gue_table().cdf(s)


# --------------------------------------------------------------------------- #
# Spacing distribution
# --------------------------------------------------------------------------- #
def spacing_distribution(gammas, bins=70, s_max=3.5, method="theta"):
    """Histogram of normalized nearest-neighbour spacings with reference laws.

    Returns a dict with bin ``centers``/``edges``, the empirical ``density``
    (counts/(n*bin width), so it integrates to the fraction of spacings below
    s_max), ``counts``, the reference densities ``gue`` (exact), ``wigner`` and
    ``poisson`` at the centers, and the sample moments ``mean``, ``var``,
    ``min``, ``max`` together with the GUE values of the variance.
    """
    d = normalized_spacings(gammas, method)
    counts, edges = np.histogram(d, bins=bins, range=(0.0, s_max))
    width = edges[1] - edges[0]
    centers = 0.5 * (edges[1:] + edges[:-1])
    tab = gue_table()
    return {
        "n": int(d.size),
        "centers": centers,
        "edges": edges,
        "counts": counts,
        "density": counts / (d.size * width),
        "gue": tab.pdf(centers),
        "wigner": wigner_surmise(centers),
        "poisson": poisson_density(centers),
        "mean": float(d.mean()),
        "var": float(d.var()),
        "var_gue": float(tab.variance),
        "var_wigner": float(3 * np.pi / 8 - 1),
        "min": float(d.min()),
        "max": float(d.max()),
        "fraction_above_s_max": float(np.mean(d > s_max)),
    }


# --------------------------------------------------------------------------- #
# 3. Pair correlation (Montgomery) and the form factor F(alpha)
# --------------------------------------------------------------------------- #
def pair_correlation_gue(alpha):
    """Montgomery / GUE pair correlation 1 - (sin(pi a)/(pi a))^2."""
    alpha = np.asarray(alpha, dtype=np.float64)
    return 1.0 - np.sinc(alpha) ** 2


def pair_correlation(gammas, alpha_max=5.0, bins=100, method="theta"):
    """Empirical pair correlation R_2(alpha) of the unfolded zeros.

    Montgomery's conjecture (1973): for the unfolded zeros x_n, the number of
    pairs (n, m), n != m, with x_m - x_n in [a, a + da] is ~ N R_2(a) da with
    R_2(a) = 1 - (sin(pi a)/(pi a))^2.  Estimator: every zero x_n whose forward
    window [x_n, x_n + alpha_max] lies inside the data range contributes all
    differences x_m - x_n in (0, alpha_max); the histogram counts divided by
    (number of such reference zeros) x (bin width) estimate R_2 at the bin
    centres without edge bias.  Pairs are enumerated with sorted arrays and
    ``searchsorted`` (O(N * alpha_max) work).

    Returns dict(alpha=centers, R2=empirical, R2_gue=prediction, counts, n_ref,
    n_pairs, se=Poisson-type standard error sqrt(counts)/(n_ref*width)).
    """
    xl = _unfold_ld(gammas, method)
    x = np.asarray(xl - xl[0], dtype=np.float64)
    n = x.size
    n_ref = int(np.searchsorted(x, x[-1] - alpha_max, side="right"))
    if n_ref < 2:
        raise ValueError("not enough zeros for the requested alpha_max")
    hi = np.searchsorted(x, x[:n_ref] + alpha_max, side="left")   # first m with x_m >= x_n + alpha_max
    base = np.arange(n_ref)
    kmax = int((hi - base).max()) - 1
    diffs = []
    for k in range(1, kmax + 1):
        sel = np.nonzero(hi - base > k)[0]
        if sel.size == 0:
            break
        diffs.append(x[sel + k] - x[sel])
    diffs = np.concatenate(diffs) if diffs else np.empty(0)
    counts, edges = np.histogram(diffs, bins=bins, range=(0.0, alpha_max))
    width = edges[1] - edges[0]
    centers = 0.5 * (edges[1:] + edges[:-1])
    norm = n_ref * width
    return {
        "alpha": centers,
        "edges": edges,
        "R2": counts / norm,
        "R2_gue": pair_correlation_gue(centers),
        "counts": counts,
        "se": np.sqrt(counts) / norm,
        "n_ref": n_ref,
        "n_pairs": int(diffs.size),
        "n": int(n),
    }


def montgomery_F(gammas, alphas, window=30.0, T=None, normalization="montgomery"):
    """Montgomery's form factor F(alpha) = F(alpha, T).

    F(alpha) = ( (T/2pi) log T )^{-1} sum_{0 < gamma, gamma' <= T}
               T^{i alpha (gamma - gamma')} w(gamma - gamma'),   w(u) = 4/(4 + u^2).

    Montgomery proved F(alpha) = (1 + o(1)) T^{-2 alpha} log T + alpha + o(1) for
    0 <= alpha <= 1 (under RH) and conjectured F(alpha) = 1 + o(1) for alpha >= 1
    (``montgomery_F_prediction``).  Here T defaults to the largest ordinate and
    the sum is truncated to pairs with |gamma - gamma'| <= window, where w is
    already < 4/window^2; for alpha > 0 the omitted oscillatory tail is further
    suppressed, at alpha = 0 the truncation removes a fraction ~ 4/(pi window)
    of the off-diagonal mass.  Cost O(N * window * density * len(alphas)).

    normalization="montgomery" divides by (T/2pi) log T as above; "diagonal"
    divides by N instead, i.e. returns (1/N) sum T^{i alpha (gamma-gamma')} w(gamma-gamma'),
    which has the same GUE prediction (alpha + T^{-2 alpha} log T for alpha <= 1,
    1 for alpha >= 1) but is free of the slow O(1/log T) drift of the ratio
    N / ((T/2pi) log T) = 1 - (1 + log 2pi)/log T + ..., and also applies to a
    window of zeros at height T (then T^{-2 alpha} log T is negligible).
    """
    g = np.asarray(gammas, dtype=np.float64)
    alphas = np.atleast_1d(np.asarray(alphas, dtype=np.float64))
    if T is None:
        T = float(g[-1])
    logT = math.log(T)
    diffs = []
    k = 1
    while k < g.size:
        d = g[k:] - g[:-k]
        d = d[d <= window]
        if d.size == 0:
            break
        diffs.append(d)
        k += 1
    d = np.concatenate(diffs) if diffs else np.empty(0)
    wd = 4.0 / (4.0 + d * d)
    F = np.empty_like(alphas)
    for i, a in enumerate(alphas):
        F[i] = g.size + 2.0 * np.sum(np.cos(a * logT * d) * wd)   # diagonal + 2 Re(off-diagonal)
    if normalization == "montgomery":
        F /= (T / (2 * np.pi)) * logT
    elif normalization == "diagonal":
        F /= g.size
    else:
        raise ValueError("normalization must be 'montgomery' or 'diagonal'")
    return F


def montgomery_F_prediction(alphas, T):
    """T^{-2 alpha} log T + alpha for alpha <= 1, and 1 for alpha >= 1 (Montgomery 1973)."""
    a = np.asarray(alphas, dtype=np.float64)
    return np.where(a <= 1, T ** (-2 * a) * math.log(T) + a, 1.0)


def montgomery_F_gue_finite(alphas, gammas, window=30.0, T=None, n_heights=60, n_u=40001):
    """GUE expectation of the diagonal-normalised, window-truncated form factor
    at the finite heights of ``gammas``:

        F̃_GUE(alpha) = 1 + < rho(gamma) ∫_{-W}^{W} cos(alpha log T u) w(u) [1 - sinc^2(rho(gamma) u)] du >_gamma,

    the average running over the heights of the data (sampled at quantiles of
    the index) with rho the mean density log(gamma/2pi)/2pi.  As T -> infinity
    this tends to Montgomery's alpha + T^{-2 alpha} log T (alpha <= 1), 1
    (alpha >= 1); at finite height it differs by O(1/rho^2) because w(u) is not
    constant across the sine-kernel correlation hole |u| <~ 1/rho, and by the
    truncation to |gamma - gamma'| <= window.  Comparing data with this curve
    isolates the genuinely arithmetic (Bogomolny–Keating) corrections.
    """
    g = np.asarray(gammas, dtype=np.float64)
    alphas = np.atleast_1d(np.asarray(alphas, dtype=np.float64))
    if T is None:
        T = float(g[-1])
    logT = math.log(T)
    u = np.linspace(-window, window, n_u)
    wu = 4.0 / (4.0 + u * u)
    idx = np.linspace(0, g.size - 1, n_heights + 2)[1:-1].astype(int)
    cosu = np.cos(alphas[:, None] * logT * u[None, :])          # (n_alpha, n_u)
    acc = np.zeros_like(alphas)
    for i in idx:
        rho = math.log(g[i] / (2 * math.pi)) / (2 * math.pi)
        kern = rho * wu * (1.0 - np.sinc(rho * u) ** 2)
        acc += integrate.trapezoid(cosu * kern[None, :], u, axis=1)
    return 1.0 + acc / idx.size


# --------------------------------------------------------------------------- #
# 4. Number variance
# --------------------------------------------------------------------------- #
def number_variance_gue(L):
    """GUE number variance (Mehta, "Random Matrices", unitary ensemble):

    Sigma^2(L) = (1/pi^2) [ log(2 pi L) + gamma + 1 - cos(2 pi L) - Ci(2 pi L) ]
                 + L [ 1 - (2/pi) Si(2 pi L) ].
    Sigma^2(L) ~ L - L^2 for small L and ~ (1/pi^2)(log(2 pi L) + gamma + 1) for large L.
    """
    L = np.asarray(L, dtype=np.float64)
    z = 2 * np.pi * L
    with np.errstate(divide="ignore", invalid="ignore"):
        si, ci = special.sici(z)
        val = ((np.log(z) + EULER_GAMMA + 1 - np.cos(z) - ci) / np.pi ** 2
               + L * (1 - 2 / np.pi * si))
    return np.where(L > 0, val, 0.0)


def number_variance_poisson(L):
    """Poisson number variance Sigma^2(L) = L."""
    return np.asarray(L, dtype=np.float64)


def _primes_upto(n):
    """Primes <= n by a simple sieve."""
    n = int(n)
    if n < 2:
        return np.empty(0, dtype=np.int64)
    sieve = np.ones(n + 1, dtype=bool)
    sieve[:2] = False
    for p in range(2, int(n ** 0.5) + 1):
        if sieve[p]:
            sieve[p * p::p] = False
    return np.nonzero(sieve)[0].astype(np.int64)


def _offdiagonal_term(L):
    """∫_1^∞ sin^2(pi L tau)/tau^2 dtau = 1/2 - (1/2)[cos(2 pi L) - 2 pi L (pi/2 - Si(2 pi L))]."""
    z = 2 * np.pi * L
    si, _ = special.sici(z)
    return 0.5 - 0.5 * (np.cos(z) - z * (np.pi / 2 - si))


def number_variance_semiclassical(L, T, prime_cutoff=10 ** 6):
    """Berry's (1988) semiclassical number variance of the zeta zeros at height T.

    Writing Sigma^2(L) = (2/pi^2) ∫_0^∞ K(tau) sin^2(pi L tau)/tau^2 dtau with the
    form factor K (tau in units of the Heisenberg time 2 pi d, d = log(T/2pi)/2pi),
    the diagonal approximation of the explicit formula gives K as a sum of
    delta peaks at tau_{p,k} = k log p / log(T/2pi) with weights
    (log p)^2 / (4 pi^2 d^2 p^k), and the sum rule forces K = 1 for tau > 1.  Hence

        Sigma^2_sc(L) = (2/pi^2) [ sum_{p^k < T/2pi} sin^2(pi L k log p / log(T/2pi)) / (k^2 p^k)
                                   + ∫_1^∞ sin^2(pi L tau)/tau^2 dtau ].

    This reproduces GUE for L << L_max ~ log(T/2pi)/(2pi) and saturates for
    larger L because no prime power has log p^k < log 2 (no orbit shorter than
    log 2).  Prime powers with p <= prime_cutoff are summed exactly; larger
    primes are replaced by the prime-number-theorem integral.
    """
    L = np.atleast_1d(np.asarray(L, dtype=np.float64))
    Lam = math.log(T / (2 * np.pi))             # = 2 pi d = log(T/2pi)
    X = T / (2 * np.pi)                          # cutoff p^k < X
    cutoff = min(prime_cutoff, int(X))
    primes = _primes_upto(cutoff)
    S = np.zeros_like(L)
    k = 1
    while True:
        pk = primes.astype(np.float64) ** k
        sel = pk < X
        if not np.any(sel):
            break
        lp = np.log(primes[sel].astype(np.float64))
        pk = pk[sel]
        # sum over primes for each L (chunked to bound memory)
        for i0 in range(0, lp.size, 20000):
            lp_c = lp[i0:i0 + 20000]
            pk_c = pk[i0:i0 + 20000]
            S += np.sin(np.pi * L[:, None] * k * lp_c[None, :] / Lam) ** 2 @ (1.0 / (k * k * pk_c))
        k += 1
    if X > cutoff:
        # tail of the k = 1 sum via the PNT:  sum_{cutoff < p < X} f(log p)/p ≈ ∫ f(v)/v dv
        v0, v1 = math.log(cutoff), math.log(X)
        n_osc = 1 + int(np.ceil((v1 - v0) * L.max() / Lam))
        v = np.linspace(v0, v1, max(2001, 400 * n_osc))
        integrand = np.sin(np.pi * L[:, None] * v[None, :] / Lam) ** 2 / v[None, :]
        S += integrate.trapezoid(integrand, v, axis=1)
    return (2 / np.pi ** 2) * (S + _offdiagonal_term(L))


def number_variance_saturation(T, prime_cutoff=10 ** 6):
    """Berry's saturation value Sigma^2_inf(T) = (1/pi^2) [ 1 + sum_{p^k < T/2pi} 1/(k^2 p^k) ]
    (the L -> infinity limit of ``number_variance_semiclassical``; sin^2 -> 1/2),
    ~ (1/pi^2) [ log log(T/2pi) + 1.40 ] by Mertens' theorem."""
    X = T / (2 * np.pi)
    cutoff = min(prime_cutoff, int(X))
    primes = _primes_upto(cutoff).astype(np.float64)
    total = 0.0
    k = 1
    while True:
        pk = primes ** k
        sel = pk < X
        if not np.any(sel):
            break
        total += np.sum(1.0 / (k * k * pk[sel]))
        k += 1
    if X > cutoff:
        total += math.log(math.log(X)) - math.log(math.log(cutoff))   # ∫ dv/v = PNT tail of sum 1/p
    return (1 + total) / np.pi ** 2


def number_variance(gammas, L_values, method="theta", n_windows=200_000, seed=0):
    """Empirical number variance Sigma^2(L) = Var(#{unfolded zeros in [x, x+L)}).

    The start x runs over n_windows positions covering the data range on a
    jittered grid (one uniform random offset per cell, seeded): a regular grid
    must be avoided because the unfolded zeros x_n ≈ n - 1/2 - S(gamma_n) sit
    close to a unit lattice and alias with any commensurate step.  Counts are
    obtained with ``searchsorted``.  Returns a dict
    with ``L``, ``sigma2``, the mean count (which should be L), the number of
    independent windows span/L and the resulting standard error
    sigma2*sqrt(2/n_indep) (Gaussian approximation), plus the GUE and Poisson
    predictions.  L values larger than a quarter of the span give NaN.
    """
    xl = _unfold_ld(gammas, method)
    x = np.asarray(xl - xl[0], dtype=np.float64)
    span = float(x[-1])
    L_values = np.atleast_1d(np.asarray(L_values, dtype=np.float64))
    sigma2 = np.full(L_values.shape, np.nan)
    mean_count = np.full(L_values.shape, np.nan)
    rng = np.random.default_rng(seed)
    for i, L in enumerate(L_values):
        if L <= 0 or L > span / 4:
            continue
        starts = (span - L) * (np.arange(n_windows) + rng.random(n_windows)) / n_windows
        counts = (np.searchsorted(x, starts + L, side="left")
                  - np.searchsorted(x, starts, side="left"))
        sigma2[i] = counts.var()
        mean_count[i] = counts.mean()
    n_indep = span / L_values
    return {
        "L": L_values,
        "sigma2": sigma2,
        "mean_count": mean_count,
        "n_indep": n_indep,
        "se": sigma2 * np.sqrt(2.0 / np.maximum(n_indep - 1, 1)),
        "gue": number_variance_gue(L_values),
        "poisson": number_variance_poisson(L_values),
        "span": span,
        "n": int(x.size),
    }


def detect_saturation(L, sigma2, T, se=None):
    """Detect and quantify the Berry saturation of an empirical Sigma^2(L).

    Returns a dict with
      plateau        : mean of sigma2 over the upper half of the L range
                       (and its standard error from the scatter),
      L_cross        : L at which Mehta's GUE curve reaches the plateau,
      L_depart       : smallest L at which sigma2 < GUE - 3 se and the same holds
                       for at least 80% of the larger L in the grid,
      L_scale        : log(T/2pi)/(2pi), Berry's saturation scale,
      plateau_berry  : the semiclassical saturation value ``number_variance_saturation(T)``,
      saturated      : True if the plateau lies significantly below GUE at L_max.
    """
    L = np.asarray(L, dtype=np.float64)
    sigma2 = np.asarray(sigma2, dtype=np.float64)
    ok = np.isfinite(sigma2) & (L > 0)
    L, sigma2 = L[ok], sigma2[ok]
    if se is not None:
        se = np.asarray(se, dtype=np.float64)[ok]
    gue = number_variance_gue(L)
    upper = L >= 0.5 * L.max()
    plateau = float(np.mean(sigma2[upper]))
    plateau_se = float(np.std(sigma2[upper], ddof=1) / np.sqrt(max(upper.sum(), 2) - 1)) if upper.sum() > 1 else float("nan")
    gue_at_max = float(number_variance_gue(L.max()))
    L_cross = float("nan")
    if plateau > float(number_variance_gue(L.min())) and plateau < gue_at_max:
        L_cross = float(brentq(lambda t: float(number_variance_gue(t)) - plateau, L.min(), L.max()))
    L_depart = float("nan")
    if se is not None:
        below = sigma2 < gue - 3 * se
        for i in range(len(L)):
            if below[i] and np.mean(below[i:]) >= 0.8:
                L_depart = float(L[i])
                break
    L_scale = math.log(T / (2 * np.pi)) / (2 * np.pi)
    return {
        "plateau": plateau,
        "plateau_se": plateau_se,
        "L_cross": L_cross,
        "L_depart": L_depart,
        "L_scale": L_scale,
        "L_max_tested": float(L.max()),
        "gue_at_L_max": gue_at_max,
        "plateau_berry": float(number_variance_saturation(T)),
        "saturated": bool(plateau < gue_at_max - 3 * (plateau_se if np.isfinite(plateau_se) else 0.0)),
    }


# --------------------------------------------------------------------------- #
# 5. Kolmogorov–Smirnov distances by height
# --------------------------------------------------------------------------- #
def ks_distance(spacings, cdf=None):
    """Kolmogorov–Smirnov distance sup_s |F_emp(s) - F(s)| of the sample
    ``spacings`` from the law with CDF ``cdf`` (default: exact GUE).

    Returns dict(D, s_at, sign, n, pvalue, noise_floor); the p-value is the
    two-sided ``scipy.stats.kstwo`` value (exact for n <= 5000, the asymptotic
    Kolmogorov law ``kstwobign`` beyond), which assumes independent draws
    (consecutive zeta spacings are weakly correlated, so treat it as
    indicative).  noise_floor = 0.8687/sqrt(n) is the expected D of a sample
    that does follow the law exactly.
    """
    if cdf is None:
        cdf = gue_table().cdf
    s = np.sort(np.asarray(spacings, dtype=np.float64))
    n = s.size
    F = np.asarray(cdf(s), dtype=np.float64)
    hi = np.arange(1, n + 1) / n
    lo = np.arange(0, n) / n
    d_plus = hi - F
    d_minus = F - lo
    ip, im = int(np.argmax(d_plus)), int(np.argmax(d_minus))
    if d_plus[ip] >= d_minus[im]:
        D, s_at, sign = float(d_plus[ip]), float(s[ip]), +1
    else:
        D, s_at, sign = float(d_minus[im]), float(s[im]), -1
    return {
        "D": D,
        "s_at": s_at,
        "sign": sign,
        "n": int(n),
        "pvalue": float(stats.kstwo.sf(D, n)) if n <= 5000 else float(stats.kstwobign.sf(D * math.sqrt(n))),
        "noise_floor": float(0.8687 / math.sqrt(n)),
    }


def index_windows(n_available, start_index=1, edges=(1, 10 ** 4, 3 * 10 ** 4, 10 ** 5, 3 * 10 ** 5, 10 ** 6, 3 * 10 ** 6, 10 ** 7)):
    """Index ranges [lo, hi] (1-based, inclusive) cut at the given edges and
    restricted to the available zeros start_index .. start_index+n_available-1."""
    last = start_index + n_available - 1
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        lo, hi = max(a, start_index), min(b, last)
        if hi - lo >= 100:
            out.append((lo, hi))
    return out


def ks_by_height(gammas, windows=None, start_index=1, method="theta", label=None):
    """KS distances of the normalized spacing law in several height windows.

    ``windows`` is a list of (lo, hi) 1-based zero indices (default
    ``index_windows``).  For each window the result records n, the height range,
    log T at the geometric centre, the KS distances to GUE, to the Wigner
    surmise and to Poisson, the location of the GUE discrepancy and the noise
    floor 0.87/sqrt(n).  Bogomolny–Keating-type corrections make the
    discrepancy from GUE shrink slowly (like a power of 1/log T).
    """
    g = np.asarray(gammas, dtype=np.float64)
    if windows is None:
        windows = index_windows(g.size, start_index)
    tab = gue_table()
    out = []
    for lo, hi in windows:
        seg = g[lo - start_index: hi - start_index + 1]
        d = normalized_spacings(seg, method)
        k_gue = ks_distance(d, tab.cdf)
        k_wig = ks_distance(d, wigner_surmise_cdf)
        k_poi = ks_distance(d, poisson_cdf)
        T_mid = math.sqrt(float(seg[0]) * float(seg[-1]))
        out.append({
            "label": label or f"zeros {lo}..{hi}",
            "index_lo": int(lo), "index_hi": int(hi), "n_spacings": int(d.size),
            "T_lo": float(seg[0]), "T_hi": float(seg[-1]), "T_mid": T_mid,
            "logT_mid": math.log(T_mid),
            "mean_spacing": float(d.mean()), "var_spacing": float(d.var()),
            "ks_gue": k_gue["D"], "ks_gue_s_at": k_gue["s_at"], "ks_gue_sign": k_gue["sign"],
            "ks_gue_pvalue": k_gue["pvalue"],
            "ks_wigner": k_wig["D"], "ks_poisson": k_poi["D"],
            "noise_floor": k_gue["noise_floor"],
        })
    return out


# --------------------------------------------------------------------------- #
# 6. Lehmer phenomenon
# --------------------------------------------------------------------------- #
def smallest_spacings(gammas, k=10, method="theta", start_index=1):
    """The k smallest normalized spacings and where they occur.

    Returns a list of dicts (index n of the lower zero, 1-based; gamma_n,
    gamma_{n+1}; raw gap; normalized spacing), smallest first.  The Lehmer pair
    gamma_6709 = 7005.0629, gamma_6710 = 7005.1006 (Lehmer 1956) is the
    classical example of such a near-degeneracy; close pairs are where Z(t) only
    just crosses zero and are the obstruction to improving the de Bruijn–Newman
    bound by the Csordas–Smith–Varga argument.
    """
    g = np.asarray(gammas, dtype=np.float64)
    d = normalized_spacings(g, method)
    idx = np.argsort(d)[:k]
    return [{
        "index": int(start_index + i),
        "gamma_n": float(g[i]),
        "gamma_n1": float(g[i + 1]),
        "gap": float(g[i + 1] - g[i]),
        "normalized_spacing": float(d[i]),
    } for i in idx]


def lehmer_pair_statistic(gammas, i, include_conjugates=True):
    """Csordas–Smith–Varga Lehmer-pair quantity for the pair (gamma_i, gamma_{i+1})
    (0-based positions in ``gammas``):

        Delta = (gamma_{i+1} - gamma_i)^2 * sum_{j != i, i+1} [ 1/(gamma_j - gamma_i)^2 + 1/(gamma_j - gamma_{i+1})^2 ],

    the sum running over all other zeros (including the conjugate ordinates
    -gamma_j when include_conjugates).  The pair is a Lehmer pair when
    Delta < 4/5; such pairs give lower bounds for the de Bruijn–Newman constant
    (Csordas, Smith, Varga 1994).  The sum is truncated to the available zeros;
    the missing tail is O(density/gamma_max), negligible for the datasets here.
    """
    g = np.asarray(gammas, dtype=np.float64)
    a, b = g[i], g[i + 1]
    others = np.delete(g, [i, i + 1])
    if include_conjugates:
        others = np.concatenate([others, -g])
    tot = np.sum(1.0 / (others - a) ** 2 + 1.0 / (others - b) ** 2)
    return float((b - a) ** 2 * tot)


def lehmer_pairs(gammas, k=20, method="theta", start_index=1):
    """Evaluate the Csordas–Smith–Varga statistic for the k closest pairs and
    flag the Lehmer pairs (Delta < 4/5)."""
    g = np.asarray(gammas, dtype=np.float64)
    rows = smallest_spacings(g, k=k, method=method, start_index=start_index)
    for r in rows:
        r["csv_delta"] = lehmer_pair_statistic(g, r["index"] - start_index)
        r["is_lehmer_pair"] = bool(r["csv_delta"] < 0.8)
    return rows


# --------------------------------------------------------------------------- #
# Data helper for the high-height sample
# --------------------------------------------------------------------------- #
def load_high_sample(name="sample_1e10", directory=None):
    """Load data/zeros/<name>.npy (consecutive zeros starting at a high index,
    produced with Arb) and its manifest; returns (gammas, manifest) or (None, None)."""
    directory = Path(directory) if directory is not None else ZEROS_DIR
    p = directory / f"{name}.npy"
    if not p.exists():
        return None, None
    g = np.load(p)
    j = directory / f"{name}.json"
    man = json.loads(j.read_text()) if j.exists() else {}
    return g, man
