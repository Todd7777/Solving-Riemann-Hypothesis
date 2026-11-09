r"""Weil's quadratic form for zeta, L(s,(n/5)) and the Davenport–Heilbronn function on
test functions supported in [-a, a], in a Legendre and a frequency-localised basis.

Setting
-------
F(s) = sum c_n n^{-s} is one of the real, q-periodic, self-dual Dirichlet series of
`rh.explicit_formula_general` (Lambda(s) = (q/pi)^{s/2} Gamma((s+e)/2) F(s) = Lambda(1-s),
alpha = (1/2 + e)/2, a_n the Dirichlet coefficients of -F'/F):

* zeta:                q = 1, e = 0, a_n = Lambda(n), poles at s = 0, 1;
* L(s,(n/5)):          q = 5, e = 0, a_n = (n/5) Lambda(n);
* Davenport–Heilbronn: q = 5, e = 1, a_n NOT supported on prime powers (no Euler product).

For real f supported in [-a, a] let f~(u) = f(-u), g = f * f~ (even, supported in [-2a, 2a]),
f^(z) = int f(u) e^{izu} du (entire) and h(z) = g^(z) = f^(z) f^(-z) (= |f^(z)|^2 on R).  The
explicit formula of `rh.explicit_formula_general` (validated there to ~1e-14 for all three
series) applied to g gives

    Q_a(f) := sum_rho f^(tau_rho) f^(-tau_rho),          tau_rho = (rho - 1/2)/i        (zero side)
            = 2 F_L(1/2) F_L(-1/2)                       (zeta only; F_L(s) = int f(u) e^{su} du)
              - 2 sum_{2 <= n <= e^{2a}} a_n n^{-1/2} g(log n)
              + (log(q/pi) - gamma_E) g(0) + 2 int_0^inf [g(0) e^{-2u} - g(u) e^{-2 alpha u}]/(1 - e^{-2u}) du.

The right-hand side (the arithmetic side, what this module assembles) uses only the a_n with
n <= e^{2a} and the Gamma factor: no zero enters.  On the zero side an on-line pair
1/2 +- i gamma gives 2 |f^(gamma)|^2 >= 0, while an off-line quadruple rho, 1 - rho, conj rho,
1 - conj rho (rho = 1/2 + delta + i gamma, tau = gamma - i delta) gives
4 Re[f^(tau) f^(-tau)]; for even f this is 4 Re f^(tau)^2 = 4 [(Re f^(tau))^2 - (Im f^(tau))^2],
an INDEFINITE rank-2 form (for odd f the same with the opposite sign).  This is how an off-line
zero can make Q_a negative (Weil's criterion: for zeta, RH iff Q_a >= 0 for all a and f).

Bases (Q_a is invariant under f(u) -> f(-u), so even and odd sectors decouple)
-----------------------------------------------------------------------------
(i)  'legendre':  phi_j(u) = P_j(u/a), j < J (as in `rh.weil_form`); even sector j even, odd
     sector j odd.  Correlations are polynomials, integrated exactly by Gauss–Legendre.
(ii) 'localized': phi(u) = w(u) P_j(u/a) cos(r0 u) and w(u) P_j(u/a) sin(r0 u), j < J,
     w(u) = (1 - (u/a)^2)^m.  phi^ is concentrated near +-r0 and decays like |r -+ r0|^{-m-1}
     away from them.  Even sector: (j, cos) for j even and (j, sin) for j odd; odd sector the
     other two; each sector has J functions.  The Gram matrix is full and, for P_j against the
     weight w^2, ill-conditioned (condition ~1e12 for m = 8, J = 16): work at >= 40 digits when
     small eigenvalues matter.

Fourier transforms
------------------
int_{-a}^{a} P_k(u/a) e^{izu} du = 2a i^k j_k(az) (spherical Bessel), and w(u) P_j(u/a) is
expanded EXACTLY (rational arithmetic) as sum_k b_{jk} P_k(u/a), k <= j + 2m, so
phi^(z) for both bases is a finite combination of j_k(a z), j_k(a(z +- r0)) at any complex z:
`fourier_matrix` (float64/complex128, scipy), `fourier_vector_mp` (mpmath at any precision;
power series for small |w|, stable upward recurrence otherwise), `fourier_transform_arb`
(Gauss–Legendre in Arb, an independent check).  For large |z| the exact finite expansion
f^(z) = e^{iaz} A(z) + e^{-iaz} B(z), A, B rational (repeated integration by parts of a
polynomial; `far_field`), gives the density tail of the zero side in closed form.

Numerics and what is (not) rigorous
-----------------------------------
All matrices are assembled in Arb ball arithmetic (python-flint) at `dps` digits, with the a_n
as rigorous balls (`log_derivative_coefficients_arb`; a_n whose ball is ~0, e.g. non-prime
powers for zeta, are added as a rigorous Cauchy–Schwarz error radius instead of being
evaluated).  The ball radii cover rounding only.  Not covered: (1) the Gauss–Legendre error of
the archimedean integral (integrand analytic in |Im u| < pi; node count from the Bernstein
ellipse through the pole u = i pi, including the e^{r0 |Im u|} growth of the oscillation;
estimated by a second node count, as in `rh.weil_form`); (2) for the localised basis, the
Gauss–Legendre error of the correlations int phi_i(v) phi_j(v - u) dv and of the pole vectors
(entire integrands; estimated by doubling the nodes at sample points).  So the matrices are
numerical, not certified, and eigenvalue statements carry an ESTIMATED error
(`eigenvalue_error_bound`).  The zero side over a finite zero list plus a density tail is
float64 (`zero_side`, `zero_side_matrix`): an independent check of the matrices, not a proof.
"""
from __future__ import annotations

import json
import math
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from .explicit_formula_general import (CHI5, DAVENPORT_HEILBRONN, ZETA, SelfDualSeries, hardy_theta,
                                       load_dh_zeros, log_derivative_coefficients_arb)
from .weil_form import _gl_nodes

__all__ = [
    "SERIES", "get_series", "Basis", "SectorForm", "build_sector", "build_form", "smallest_eigenpairs",
    "eigenvalue_error_bound", "quadratic_value", "fourier_matrix", "fourier_vector_mp", "fourier_transform",
    "fourier_transform_arb", "far_field", "zero_density", "zero_side", "zero_side_matrix", "load_zero_data",
    "quadruple_matrix", "add_zero_quadruple", "boundary_vanishing_subspace", "basis_values",
]

REPO_ROOT = Path(__file__).resolve().parents[2]
LCHI5_ZEROS = REPO_ROOT / "data" / "zeros" / "lchi5_zeros.npy"
SERIES = {"zeta": ZETA, "chi5": CHI5, "dh": DAVENPORT_HEILBRONN}
_ALIASES = {"zeta": "zeta", "riemann": "zeta", "chi5": "chi5", "lchi5": "chi5", "(n/5)": "chi5",
            "l(s,(n/5))": "chi5", "dh": "dh", "davenport-heilbronn": "dh", "davenport_heilbronn": "dh"}


def get_series(series) -> SelfDualSeries:
    """A `SelfDualSeries` from itself or a name ('zeta', 'chi5', 'dh' and aliases)."""
    if isinstance(series, SelfDualSeries):
        return series
    key = _ALIASES.get(str(series).strip().lower())
    if key is None:
        raise ValueError(f"unknown series {series!r}; use one of {sorted(SERIES)} or a SelfDualSeries")
    return SERIES[key]


def _series_key(ser: SelfDualSeries) -> str:
    for k, v in SERIES.items():
        if v == ser:
            return k
    return ser.name


@contextmanager
def _flint_dps(dps):
    from flint import ctx
    old = ctx.dps
    ctx.dps = int(dps)
    try:
        yield
    finally:
        ctx.dps = old


def _check_parity(parity: str) -> str:
    if parity not in ("even", "odd"):
        raise ValueError("parity must be 'even' or 'odd'")
    return parity


# ----------------------------------------------------------------------------
# 1. Bases
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class Basis:
    """kind 'legendre': P_j(u/a), j < J.  kind 'localized': (1 - (u/a)^2)^m P_j(u/a) x {cos, sin}(r0 u),
    j < J (J functions per sector).  Construct with Basis.legendre(a, J) / Basis.localized(a, J, r0, m)."""
    kind: str
    a: float
    J: int
    r0: float = 0.0
    m: int = 0

    def __post_init__(self):
        object.__setattr__(self, "a", float(self.a))
        object.__setattr__(self, "J", int(self.J))
        object.__setattr__(self, "r0", float(self.r0))
        object.__setattr__(self, "m", int(self.m))
        if self.kind not in ("legendre", "localized"):
            raise ValueError("kind must be 'legendre' or 'localized'")
        if not (self.a > 0 and self.J >= 1):
            raise ValueError("need a > 0 and J >= 1")
        if self.kind == "legendre" and (self.r0 != 0.0 or self.m != 0):
            raise ValueError("the Legendre basis has no r0 / m")
        if self.kind == "localized" and (self.m < 0 or self.r0 < 0):
            raise ValueError("need m >= 0 and r0 >= 0")

    @classmethod
    def legendre(cls, a: float, J: int) -> "Basis":
        return cls("legendre", a, J)

    @classmethod
    def localized(cls, a: float, J: int, r0: float, m: int = 8) -> "Basis":
        return cls("localized", a, J, r0, m)

    def labels(self, parity: str) -> tuple:
        """((j, trig), ...) of the sector: trig None (Legendre), 'cos' or 'sin' (localised)."""
        ev = _check_parity(parity) == "even"
        if self.kind == "legendre":
            return tuple((j, None) for j in range(self.J) if (j % 2 == 0) == ev)
        return tuple((j, "cos" if (j % 2 == 0) == ev else "sin") for j in range(self.J))

    def size(self, parity: str) -> int:
        return len(self.labels(parity))

    @property
    def degree(self) -> int:
        """Largest polynomial degree of the (window x Legendre) factor: J - 1 + 2m."""
        return self.J - 1 + 2 * self.m

    def describe(self) -> dict:
        return {"kind": self.kind, "a": self.a, "J": self.J, "r0": self.r0, "m": self.m}


# exact rational helpers ------------------------------------------------------
def _times_x(c):
    """Legendre coefficients of x * sum_k c_k P_k (x P_k = ((k+1) P_{k+1} + k P_{k-1})/(2k+1))."""
    out = [Fraction(0)] * (len(c) + 1)
    for k, ck in enumerate(c):
        if ck:
            out[k + 1] += ck * Fraction(k + 1, 2 * k + 1)
            if k:
                out[k - 1] += ck * Fraction(k, 2 * k + 1)
    return out


@lru_cache(maxsize=None)
def _window_legendre(m: int, j: int) -> tuple:
    """Exact Legendre coefficients b_0..b_{j+2m} of (1 - x^2)^m P_j(x)."""
    c = [Fraction(0)] * j + [Fraction(1)]
    for _ in range(m):
        xxc = _times_x(_times_x(c))
        c = [(c[k] if k < len(c) else Fraction(0)) - xxc[k] for k in range(len(xxc))]
    c = c[: j + 2 * m + 1]
    return tuple(c)


@lru_cache(maxsize=None)
def _legendre_deriv_one(k: int, l: int) -> int:
    """P_k^{(l)}(1) * 2^l l! ... returned exactly as the integer (k+l)!/((k-l)!) // 1 with the
    2^l l! divisor applied by the caller: P_k^{(l)}(1) = (k+l)! / (2^l l! (k-l)!)  (0 for l > k)."""
    if l > k:
        return 0
    return math.factorial(k + l) // math.factorial(k - l)


def _Pkl_one(k: int, l: int) -> Fraction:
    return Fraction(_legendre_deriv_one(k, l), (2 ** l) * math.factorial(l))


@lru_cache(maxsize=None)
def _sector_tables(basis: Basis, parity: str):
    """Float tables of the sector: Legendre-expansion matrices (K x n) for phi^ in terms of
    j_k(a(z + r0)), j_k(a(z - r0)), and endpoint derivatives psi_i^{(l)}(+-a) (L x n) of the
    polynomial factor psi_i (= P_j(u/a) or w P_j(u/a))."""
    labels = basis.labels(parity)
    n = len(labels)
    if basis.kind == "legendre":
        K = basis.J
        bplus = np.zeros((K, n), dtype=complex)
        for i, (j, _) in enumerate(labels):
            bplus[j, i] = 0.5
        bminus = bplus.copy()
        polys = [tuple([Fraction(0)] * j + [Fraction(1)]) for j, _ in labels]
        kp = np.ones(n, dtype=complex)
        km = np.ones(n, dtype=complex)
    else:
        K = basis.degree + 1
        bplus = np.zeros((K, n), dtype=complex)
        bminus = np.zeros((K, n), dtype=complex)
        polys = []
        kp = np.zeros(n, dtype=complex)
        km = np.zeros(n, dtype=complex)
        for i, (j, trig) in enumerate(labels):
            b = _window_legendre(basis.m, j)
            polys.append(b)
            # cos = (e^{ir0u} + e^{-ir0u})/2 ; sin = (e^{ir0u} - e^{-ir0u})/(2i)
            kp[i], km[i] = (1.0, 1.0) if trig == "cos" else (-1j, 1j)
            for k, bk in enumerate(b):
                bplus[k, i] = 0.5 * kp[i] * float(bk)
                bminus[k, i] = 0.5 * km[i] * float(bk)
    # endpoint derivatives psi^{(l)}(+-a) = a^{-l} sum_k b_k P_k^{(l)}(+-1), stored SCALED as
    # psi^{(l)}(+-a) / S^l with S = K^2 / a, i.e. the exact rational sum_k b_k P_k^{(l)}(+-1) / K^{2l}:
    # P_k^{(l)}(1) <= (k(k+1))^l / (2^l l!) makes these <= sum |b_k| / (2^l l!), so nothing overflows
    # (unscaled, P_k^{(l)}(1) exceeds the float64 range for k, l ~ 150).
    L = K
    Ep = np.zeros((L, n))
    Em = np.zeros((L, n))
    for i, b in enumerate(polys):
        for l in range(L):
            sp = Fraction(0)
            sm = Fraction(0)
            for k, bk in enumerate(b):
                if bk and l <= k:
                    d = bk * _Pkl_one(k, l)
                    sp += d
                    sm += d if (k + l) % 2 == 0 else -d
            scale = Fraction(1, K ** (2 * l))
            Ep[l, i] = float(sp * scale)
            Em[l, i] = float(sm * scale)
    return {"labels": labels, "K": K, "bplus": bplus, "bminus": bminus, "Ep": Ep, "Em": Em,
            "deriv_scale": K * K / basis.a, "kp": kp, "km": km}


def basis_values(basis: Basis, parity: str, u) -> np.ndarray:
    """phi_i(u) (float64, shape u.shape + (n,)); zero outside [-a, a]."""
    from numpy.polynomial import legendre as npleg
    u = np.asarray(u, dtype=float)
    y = u / basis.a
    inside = np.abs(y) <= 1
    labels = basis.labels(parity)
    out = np.zeros(u.shape + (len(labels),))
    for i, (j, trig) in enumerate(labels):
        v = npleg.legval(y, np.eye(j + 1)[j])
        if basis.kind == "localized":
            v = v * (1 - y * y) ** basis.m * (np.cos(basis.r0 * u) if trig == "cos" else np.sin(basis.r0 * u))
        out[..., i] = np.where(inside, v, 0.0)
    return out


# ----------------------------------------------------------------------------
# 2. Assembly in Arb
# ----------------------------------------------------------------------------
def _value_rows(basis: Basis, labels, ys, scale=None):
    """Rows [phi_i(a y_k)]_i (lists of arb) at the arb points y_k, row k times scale[k] if given."""
    from flint import arb
    jmax = max(j for j, _ in labels)
    loc = basis.kind == "localized"
    if loc:
        ar = arb(basis.a) * arb(basis.r0)
    one = arb(1)
    rows = []
    for k, y in enumerate(ys):
        P = [one]
        if jmax >= 1:
            P.append(y)
            p0, p1 = one, y
            for n in range(1, jmax):
                p2 = ((2 * n + 1) * y * p1 - n * p0) / (n + 1)
                P.append(p2)
                p0, p1 = p1, p2
        sc = scale[k] if scale is not None else None
        if loc:
            w = (1 - y * y) ** basis.m
            s, c = (ar * y).sin_cos()
            wc, ws = w * c, w * s
            if sc is not None:
                wc, ws = wc * sc, ws * sc
            rows.append([P[j] * (wc if trig == "cos" else ws) for j, trig in labels])
        elif sc is not None:
            rows.append([P[j] * sc for j, _ in labels])
        else:
            rows.append([P[j] for j, _ in labels])
    return rows


def _correlation(basis: Basis, labels, t, n_nodes: int):
    """Symmetrised correlation s(u), u = a t, t in [0, 2] (arb):
    s_ij = (c_ij + c_ji)/2,  c_ij(u) = int phi_i(v) phi_j(v - u) dv = a int_{t-1}^{1} Phi_i(y) Phi_j(y - t) dy
    by Gauss–Legendre with n_nodes nodes (exact for the Legendre basis with n_nodes >= J)."""
    from flint import arb, arb_mat
    nodes, weights = _gl_nodes(n_nodes)
    half = (2 - t) / 2
    ys = [(t - 1) + half * (x + 1) for x in nodes]
    sc = half * arb(basis.a)
    B1 = arb_mat(_value_rows(basis, labels, ys))
    B2 = arb_mat(_value_rows(basis, labels, [y - t for y in ys], [w * sc for w in weights]))
    C = B1.transpose() * B2
    return (C + C.transpose()) * arb(0.5)


def _pole_vectors(basis: Basis, labels, n_nodes: int):
    """(int phi_i e^{u/2} du, int phi_i e^{-u/2} du) = (F_L(1/2), F_L(-1/2)) by Gauss–Legendre (arb)."""
    from flint import arb
    nodes, weights = _gl_nodes(n_nodes)
    A = arb(basis.a)
    rows = _value_rows(basis, labels, nodes)
    n = len(labels)
    vp = [arb(0)] * n
    vm = [arb(0)] * n
    for k, y in enumerate(nodes):
        ep = (A * y / 2).exp()
        wp, wm = weights[k] * ep, weights[k] / ep
        r = rows[k]
        for i in range(n):
            vp[i] += r[i] * wp
            vm[i] += r[i] * wm
    return [x * A for x in vp], [x * A for x in vm]


def _archimedean(basis: Basis, labels, alpha, s0, n_corr: int, n_quad: int):
    """2 int_0^{2a} [s(0) e^{-2u} - s(u) e^{-2 alpha u}]/(1 - e^{-2u}) du, Gauss–Legendre in t = u/a."""
    from flint import arb, arb_mat
    nodes, weights = _gl_nodes(n_quad)
    A = arb(basis.a)
    n = len(labels)
    total = arb_mat(n, n)
    for y, w in zip(nodes, weights):
        t = y + 1
        u = A * t
        su = _correlation(basis, labels, t, n_corr)
        e2 = (-2 * u).exp()
        ea = (-2 * alpha * u).exp()
        den = 1 - e2
        total += (s0 * (e2 / den) - su * (ea / den)) * (w * A)
    return total * 2


def _bernstein_through_pole(a: float):
    """Bernstein ellipse (rho, semi-minor axis b) through t = -1 + i pi/a, the image of the
    nearest pole u = i pi of 1/(1 - e^{-2u}) under u = a (t + 1)."""
    z = complex(-1.0, math.pi / a)
    A = (abs(z - 1) + abs(z + 1)) / 2
    b = math.sqrt(A * A - 1)
    return A + b, b


def default_nodes(basis: Basis, dps: int) -> dict:
    """Node counts: correlations (exact for Legendre: J + 2), archimedean, pole vectors.
    Localised correlations: integrand = polynomial of degree 2(J-1+2m) times e^{+-2i r0 v} on an
    interval of length <= 2a, i.e. bandwidth kappa + deg with kappa = 2 r0 a; Gauss–Legendre needs
    ~ (kappa + deg)/2 + ~kappa^{1/3} D^{2/3} nodes for D digits (measured: 1e-60 for e^{i 171 x}
    with 165 nodes).  Archimedean (localised): error ~ e^{r0 a b} rho^{-2n} on the Bernstein
    ellipse through the pole (measured to predict the required n within ~5%)."""
    D = dps + 10
    if basis.kind == "legendre":
        rho = math.pi / basis.a + math.sqrt(1 + (math.pi / basis.a) ** 2)
        return {"n_corr": basis.J + 2,
                "n_quad": basis.J + 20 + int(math.ceil(1.15 * dps / (2 * math.log10(rho)))),
                "n_pole": basis.J + 40}
    kappa = 2.0 * basis.r0 * basis.a
    deg = 2 * basis.degree
    n_corr = int(math.ceil((kappa + deg) / 2 + 1.1 * kappa ** (1 / 3) * D ** (2 / 3) + 0.3 * D + 10))
    rho, b = _bernstein_through_pole(basis.a)
    osc = basis.r0 * basis.a * b / math.log(10)
    n_quad = int(math.ceil((osc + 1.15 * dps + 10) / (2 * math.log10(rho)))) + basis.J + 2 * basis.m + 20
    kp = basis.r0 * basis.a
    n_pole = int(math.ceil((kp + basis.degree) / 2 + 1.1 * kp ** (1 / 3) * D ** (2 / 3) + 0.3 * D + 20))
    return {"n_corr": n_corr, "n_quad": n_quad, "n_pole": n_pole}


@dataclass
class SectorForm:
    """Arithmetic side of Q_a on one sector: Q = poles + archimedean + prime, G = Gram matrix."""
    series: str
    basis: Basis
    parity: str
    labels: tuple
    dps: int
    Q: object
    G: object
    poles: object
    archimedean: object
    prime: object
    n_corr: int
    n_quad: int
    n_pole: int
    quad_error_estimate: float = float("nan")
    corr_error_estimate: float = float("nan")
    error_estimate: float = float("nan")
    terms: tuple = ()
    n_zero_terms: int = 0
    info: dict = field(default_factory=dict)

    @property
    def size(self) -> int:
        return len(self.labels)

    def matrix_mp(self, which: str = "Q", dps: Optional[int] = None):
        return _arb_to_mp(getattr(self, which), dps or self.dps)

    def matrix_float(self, which: str = "Q") -> np.ndarray:
        M = getattr(self, which)
        n = M.nrows()
        return np.array([[float(M[i, j].mid()) for j in range(n)] for i in range(n)])

    def max_radius(self, which: str = "Q") -> float:
        M = getattr(self, which)
        n = M.nrows()
        return max(float(M[i, j].rad()) for i in range(n) for j in range(n))


def build_sector(series, basis: Basis, parity: str, dps: int = 60, n_quad: Optional[int] = None,
                 n_corr: Optional[int] = None, n_pole: Optional[int] = None, estimate_errors: bool = True,
                 error_samples: Sequence[float] = (0.0, 0.37, 0.81, 1.23, 1.71)) -> SectorForm:
    """Assemble the arithmetic side of Q_a on one sector of `basis` in Arb at `dps` digits.

    series   'zeta', 'chi5', 'dh' or a SelfDualSeries
    estimate_errors  archimedean quadrature with a second node count (1.5 n_quad + 4; the more
             accurate value is kept) and, for the localised basis, correlations / pole vectors with
             doubled nodes at the sample points t = u/a in `error_samples` and at log(n)/a."""
    from flint import arb, arb_mat, ctx
    ser = get_series(series)
    labels = basis.labels(parity)
    n = len(labels)
    t_start = time.time()
    nodes = default_nodes(basis, dps)
    nc = int(n_corr or nodes["n_corr"])
    nq = int(n_quad or nodes["n_quad"])
    npl = int(n_pole or nodes["n_pole"])
    with _flint_dps(dps):
        A = arb(basis.a)
        s0 = _correlation(basis, labels, arb(0), nc)
        if basis.kind == "legendre":
            G = arb_mat(n, n)
            for c, (j, _) in enumerate(labels):
                G[c, c] = 2 * A / (2 * j + 1)
        else:
            G = s0
        # ---- poles (zeta): 2 F_L(1/2) F_L(-1/2) as a bilinear form
        P = arb_mat(n, n)
        if ser.pole:
            vp, vm = _pole_vectors(basis, labels, npl)
            for i in range(n):
                for j in range(n):
                    P[i, j] = vp[i] * vm[j] + vm[i] * vp[j]
        # ---- archimedean
        alpha = arb(1 + 2 * ser.e) / 4
        const = -(1 - (-4 * A).exp()).log() + (arb(ser.q) / arb.pi()).log() - arb.const_euler()
        arch = _archimedean(basis, labels, alpha, s0, nc, nq)
        quad_err = float("nan")
        if estimate_errors:
            nq2 = int(1.5 * nq) + 4
            arch2 = _archimedean(basis, labels, alpha, s0, nc, nq2)
            d = arch2 - arch
            quad_err = max(abs(float(d[i, j].mid())) for i in range(n) for j in range(n))
            arch = arch2
            nq = nq2
        Ar = arch + s0 * const
        # ---- prime-type terms: -2 sum_n a_n n^{-1/2} s(log n)
        nmax = int(math.floor(math.exp(2 * basis.a) + 1e-9))
        Pr = arb_mat(n, n)
        terms, n_zero, R_zero = [], 0, 0.0
        weight_sum = 0.0
        if nmax >= 2:
            an = log_derivative_coefficients_arb(ser, nmax)
            tiny = 10.0 ** (-(dps - 10))
            for k in range(2, nmax + 1):
                x = an[k]
                if x.is_zero():
                    continue
                t = arb(k).log() / A
                if t > 2:
                    continue
                mag = float(abs(x).upper()) * (1 + 1e-9)
                weight_sum += 2 * mag / math.sqrt(k)
                if x.contains(0) and mag < tiny:
                    R_zero += 2 * mag / math.sqrt(k) * (1 + 1e-9)
                    n_zero += 1
                    continue
                Pr -= _correlation(basis, labels, t, nc) * (2 * x / arb(k).sqrt())
                terms.append(k)
            if R_zero > 0:          # |s_ij(u)| <= ||phi_i|| ||phi_j|| (Cauchy–Schwarz)
                g = [math.sqrt(float(G[i, i].upper()) * (1 + 1e-9)) for i in range(n)]
                for i in range(n):
                    for j in range(n):
                        Pr[i, j] += arb(0, R_zero * g[i] * g[j])
        Q = P + Ar + Pr
        # ---- quadrature error of the localised correlations / pole vectors (doubling)
        corr_err = 0.0 if basis.kind == "legendre" else float("nan")
        if estimate_errors and basis.kind == "localized":
            ts = [arb(float(x)) for x in error_samples] + [arb(k).log() / A for k in terms[:3]]
            diffs = []
            for t in ts:
                dd = _correlation(basis, labels, t, 2 * nc) - _correlation(basis, labels, t, nc)
                diffs.append(max(abs(float(dd[i, j].mid())) for i in range(n) for j in range(n)))
            if ser.pole:
                vp2, vm2 = _pole_vectors(basis, labels, 2 * npl)
                diffs.append(max(abs(float((vp2[i] - vp[i]).mid())) for i in range(n)))
                diffs.append(max(abs(float((vm2[i] - vm[i]).mid())) for i in range(n)))
            corr_err = max(diffs)
        # crude operator-norm weight of the arithmetic functional on a perturbation of s(u)
        kernel = abs(float(const.mid())) + 3.7 + 4 * basis.a + weight_sum
        err = (quad_err if quad_err == quad_err else 0.0) + (kernel * corr_err if corr_err == corr_err else 0.0)
        info = {"nmax": nmax, "seconds": time.time() - t_start, "prec_bits": ctx.prec,
                "series_name": ser.name, "q": ser.q, "e": ser.e}
    return SectorForm(series=_series_key(ser), basis=basis, parity=parity, labels=labels, dps=int(dps), Q=Q, G=G,
                      poles=P, archimedean=Ar, prime=Pr, n_corr=nc, n_quad=nq, n_pole=npl,
                      quad_error_estimate=quad_err, corr_error_estimate=corr_err,
                      error_estimate=err if estimate_errors else float("nan"), terms=tuple(terms),
                      n_zero_terms=n_zero, info=info)


def build_form(series, basis: Basis, dps: int = 60, **kw) -> dict:
    """Both sectors: {'even': SectorForm, 'odd': SectorForm}."""
    return {p: build_sector(series, basis, p, dps=dps, **kw) for p in ("even", "odd")}


# ----------------------------------------------------------------------------
# 3. Eigenproblem and values
# ----------------------------------------------------------------------------
def _arb_to_mp(M, dps: int):
    import mpmath as mp
    n, k = M.nrows(), M.ncols()
    with mp.workdps(dps):
        out = mp.matrix(n, k)
        for i in range(n):
            for j in range(k):
                out[i, j] = mp.mpf(M[i, j].mid().str(dps + 10, radius=False))
    return out


def smallest_eigenpairs(form, G=None, dps: Optional[int] = None, k: int = 3):
    """Smallest k generalised eigenpairs of (Q, G) at `dps` digits (mpmath; default the form's).

    `form` is a SectorForm (or an arb_mat Q, with G given).  G symmetric positive definite (full
    or diagonal): Cholesky G = L L^T, eigenpairs of L^{-1} Q L^{-T}.  Vectors are normalised
    x^T G x = 1, i.e. ||f||_2 = 1, so eigenvalues are values of Q(f)/||f||^2; the smallest is a
    Rayleigh–Ritz upper bound for the infimum over all f supported in [-a, a].  Sign convention:
    the largest-magnitude coefficient is positive."""
    import mpmath as mp
    if isinstance(form, SectorForm):
        Q, G, dps = form.Q, form.G, int(dps or form.dps)
    else:
        Q = form
        if G is None or dps is None:
            raise ValueError("pass G and dps with a bare matrix")
    n = Q.nrows()
    with mp.workdps(dps):
        Qm, Gm = _arb_to_mp(Q, dps), _arb_to_mp(G, dps)
        for M in (Qm, Gm):
            for i in range(n):
                for j in range(i + 1, n):
                    v = (M[i, j] + M[j, i]) / 2
                    M[i, j] = v
                    M[j, i] = v
        L = mp.cholesky(Gm)
        Li = mp.inverse(L)
        M = Li * Qm * Li.T
        for i in range(n):
            for j in range(i + 1, n):
                v = (M[i, j] + M[j, i]) / 2
                M[i, j] = v
                M[j, i] = v
        E, V = mp.eigsy(M)
        order = sorted(range(n), key=lambda i: E[i])
        vals, vecs = [], []
        for idx in order[:k]:
            y = mp.matrix([V[i, idx] for i in range(n)])
            x = Li.T * y
            vec = [x[i] for i in range(n)]
            big = max(range(n), key=lambda i: abs(vec[i]))
            s = 1 if vec[big] >= 0 else -1
            vals.append(E[idx])
            vecs.append([s * v for v in vec])
        return vals, vecs


def eigenvalue_error_bound(form: SectorForm, lam, dps: Optional[int] = None) -> dict:
    """Numerical (NOT rigorous) bound on |lam_computed - lam_exact| for a generalised eigenvalue of
    the restricted form: first-order perturbation theory for (Q + E_Q, G + E_G) gives
    |d lam| <= (||E_Q||_2 + |lam| ||E_G||_2) / lambda_min(G); ||E||_2 <= ||E||_F is taken as the
    Frobenius norm of the Arb radii plus n x the quadrature error estimate (`error_estimate` for
    Q, `corr_error_estimate` for G).  Returns {'bound', 'lambda_min_G', 'sign_certified'}."""
    import mpmath as mp
    dps = int(dps or form.dps)
    n = form.size
    with mp.workdps(dps):
        lam = mp.mpf(lam)
        Gm = _arb_to_mp(form.G, dps)
        gmin = min(mp.eigsy(Gm)[0])
        rq = mp.sqrt(mp.fsum(mp.mpf(form.Q[i, j].rad().str(10, radius=False)) ** 2
                             for i in range(n) for j in range(n)))
        rg = mp.sqrt(mp.fsum(mp.mpf(form.G[i, j].rad().str(10, radius=False)) ** 2
                             for i in range(n) for j in range(n)))
        eq = form.error_estimate if form.error_estimate == form.error_estimate else 0.0
        eg = form.corr_error_estimate if form.corr_error_estimate == form.corr_error_estimate else 0.0
        bound = (rq + n * mp.mpf(eq) + abs(lam) * (rg + n * mp.mpf(eg))) / gmin
        return {"bound": bound, "lambda_min_G": gmin, "sign_certified": bool(abs(lam) > bound)}


def quadratic_value(form: SectorForm, coeffs, dps: Optional[int] = None):
    """c^T Q c from the midpoints (float64, or mpmath at dps)."""
    if dps is None:
        c = np.asarray([float(x) for x in coeffs])
        return float(c @ form.matrix_float("Q") @ c)
    import mpmath as mp
    with mp.workdps(dps):
        Qm = form.matrix_mp("Q", dps)
        c = [mp.mpf(x) for x in coeffs]
        return mp.fsum(c[i] * Qm[i, j] * c[j] for i in range(len(c)) for j in range(len(c)))


# ----------------------------------------------------------------------------
# 4. Fourier transforms phi^(z) = int phi(u) e^{izu} du at complex z
# ----------------------------------------------------------------------------
def _ipow_vec(K: int) -> np.ndarray:
    return np.array([1, 1j, -1, -1j])[np.arange(K) % 4]


def _sph_jn_float(K: int, x) -> np.ndarray:
    """[j_0(x), ..., j_{K-1}(x)] stacked on the last axis (scipy; x real or complex)."""
    from scipy.special import spherical_jn
    x = np.asarray(x)
    return np.stack([spherical_jn(k, x) for k in range(K)], axis=-1)


def fourier_matrix(basis: Basis, parity: str, z, chunk: int = 20000) -> np.ndarray:
    """phi_i^(z) for the sector basis functions (complex128, shape (len(z), n)), z real or complex.
    Legendre: 2a i^j j_j(az).  Localised: sum_k b_{jk} 2a i^k [kp j_k(a(z + r0)) + km j_k(a(z - r0))]/2
    with (kp, km) = (1, 1) for cos and (-i, i) for sin."""
    tab = _sector_tables(basis, parity)
    z = np.atleast_1d(np.asarray(z))
    zc = z if np.iscomplexobj(z) else z.astype(float)
    a, K = basis.a, tab["K"]
    pre = 2 * a * _ipow_vec(K)
    out = np.empty((zc.size, len(tab["labels"])), dtype=complex)
    flat = zc.ravel()
    for s in range(0, flat.size, chunk):
        zz = flat[s:s + chunk]
        if basis.kind == "legendre" or basis.r0 == 0.0:
            Jz = _sph_jn_float(K, a * zz) * pre
            out[s:s + chunk] = Jz @ (tab["bplus"] + tab["bminus"])
        else:
            Jp = _sph_jn_float(K, a * (zz + basis.r0)) * pre
            Jm = _sph_jn_float(K, a * (zz - basis.r0)) * pre
            out[s:s + chunk] = Jp @ tab["bplus"] + Jm @ tab["bminus"]
    return out


def _sph_jn_all_mp(nmax: int, w, dps: int):
    """[j_0(w), ..., j_nmax(w)] for complex w at `dps` digits: power series (guard digits for its
    cancellation, ~|w|/ln 10) when |w| <= nmax + 1, upward recurrence (stable for n < |w|) otherwise."""
    import mpmath as mp
    w = mp.mpmathify(w)
    aw = float(abs(w))
    if aw == 0.0:
        return [mp.mpf(1)] + [mp.mpf(0)] * nmax
    if aw <= nmax + 1:
        guard = int(0.45 * aw) + 15
        with mp.workdps(dps + guard):
            w = mp.mpmathify(w)
            w2 = -w * w / 2
            eps = mp.mpf(10) ** (-(dps + guard))
            out, pref = [], mp.mpf(1)
            for n in range(nmax + 1):
                if n:
                    pref = pref * w / (2 * n + 1)
                term, s, k = mp.mpf(1), mp.mpf(1), 0
                while True:
                    k += 1
                    term = term * w2 / (k * (2 * n + 2 * k + 1))
                    s += term
                    if abs(term) <= eps * abs(s) and k > 2:
                        break
                out.append(pref * s)
        return [+x for x in out]
    with mp.workdps(dps + 20):
        w = mp.mpmathify(w)
        s, c = mp.sin(w), mp.cos(w)
        out = [s / w]
        if nmax >= 1:
            out.append(s / w ** 2 - c / w)
        for n in range(1, nmax):
            out.append((2 * n + 1) / w * out[n] - out[n - 1])
    return [+x for x in out]


def fourier_vector_mp(basis: Basis, parity: str, z, dps: int = 50) -> list:
    """[phi_i^(z)]_i at `dps` digits (mpmath mpc), z complex: the exact closed form of
    `fourier_matrix` with exact (rational) expansion coefficients.  For the localised basis the
    sum over k cancels like (a |z -+ r0|)^m far from +-r0 (the window makes psi^ decay like
    |a zeta|^{-m-1} while each term is ~|a zeta|^{-1}); guard digits cover that."""
    import mpmath as mp
    labels = basis.labels(parity)
    zc = complex(z)
    guard = 0 if basis.kind == "legendre" else \
        10 + int(math.ceil((basis.m + 1) * math.log10(1.0 + basis.a * (abs(zc) + basis.r0))))
    w = dps + guard
    with mp.workdps(w):
        z = mp.mpmathify(z)
        a = mp.mpf(basis.a)
        ipow = [mp.mpf(1), mp.j, mp.mpf(-1), -mp.j]
        if basis.kind == "legendre":
            jn = _sph_jn_all_mp(max(j for j, _ in labels), a * z, w)
            out = [2 * a * ipow[j % 4] * jn[j] for j, _ in labels]
        else:
            r0 = mp.mpf(basis.r0)
            K = basis.degree + 1
            jp = _sph_jn_all_mp(K - 1, a * (z + r0), w)
            jm = _sph_jn_all_mp(K - 1, a * (z - r0), w)
            out = []
            for j, trig in labels:
                b = _window_legendre(basis.m, j)
                sp = mp.mpf(0)
                sm = mp.mpf(0)
                for k, bk in enumerate(b):
                    if bk:
                        bb = mp.mpf(bk.numerator) / bk.denominator * ipow[k % 4]
                        sp += bb * jp[k]
                        sm += bb * jm[k]
                out.append(a * (sp + sm) if trig == "cos" else a * (sp - sm) / mp.j)
    with mp.workdps(dps):
        return [+x for x in out]


def fourier_transform(basis: Basis, parity: str, coeffs, z, dps: Optional[int] = None):
    """f^(z) for f = sum_i coeffs[i] phi_i: complex128 array (vectorised over z) when dps is None,
    else an mpmath mpc (z scalar) at dps digits."""
    if dps is None:
        return fourier_matrix(basis, parity, z) @ np.asarray(coeffs, dtype=float)
    import mpmath as mp
    with mp.workdps(dps):
        v = fourier_vector_mp(basis, parity, z, dps)
        return mp.fsum(mp.mpmathify(c) * x for c, x in zip(coeffs, v))


def fourier_transform_arb(basis: Basis, parity: str, z, dps: int = 50, n_nodes: Optional[int] = None) -> list:
    """[phi_i^(z)]_i by Gauss–Legendre on [-a, a] in Arb (acb balls; the quadrature error is not
    in the radius): an independent check of the closed form.  Default nodes from the bandwidth
    a(|z| + r0) + degree."""
    from flint import acb, arb
    labels = basis.labels(parity)
    zc = complex(z)
    with _flint_dps(dps):
        if n_nodes is None:
            kap = basis.a * (abs(zc.real) + basis.r0)
            D = dps + 10
            n_nodes = int(math.ceil((kap + basis.degree) / 2 + 1.1 * kap ** (1 / 3) * D ** (2 / 3)
                                    + basis.a * abs(zc.imag) + 0.3 * D + 20))
        nodes, weights = _gl_nodes(n_nodes)
        A = arb(basis.a)
        rows = _value_rows(basis, labels, nodes)
        iza = acb(0, 1) * acb(zc.real, zc.imag) * A
        out = [acb(0)] * len(labels)
        for k, y in enumerate(nodes):
            e = (iza * y).exp() * weights[k]
            for i, v in enumerate(rows[k]):
                out[i] += v * e
        return [x * A for x in out]


def far_field(basis: Basis, parity: str, z):
    """(A, B), each of shape (len(z), n), with phi_i^(z) = e^{iaz} A_i(z) + e^{-iaz} B_i(z) EXACTLY
    (z != 0, +-r0): repeated integration by parts of the polynomial factor psi_i,
    psi^(zeta) = sum_l (-1)^l [psi^{(l)}(a) e^{ia zeta} - psi^{(l)}(-a) e^{-ia zeta}] / (i zeta)^{l+1},
    and the shifts zeta = z +- r0 of the cos/sin factor.  Well conditioned for
    |z -+ r0| >> degree^2 / a (cancellation ~ e^{degree^2 / (2 a |z -+ r0|)})."""
    tab = _sector_tables(basis, parity)
    z = np.atleast_1d(np.asarray(z, dtype=complex)).ravel()
    L = tab["K"]
    S = tab["deriv_scale"]                 # the tables hold psi^{(l)}(+-a) / S^l
    sgn = (-1.0) ** np.arange(L)
    Ep, Em = sgn[:, None] * tab["Ep"], sgn[:, None] * tab["Em"]

    def AB(zeta):
        # psi^{(l)} (i zeta)^{-(l+1)} = [psi^{(l)} / S^l] (i zeta)^{-1} (S / (i zeta))^l, l = 0..L-1, by
        # repeated multiplication (|S / zeta| < 1 where the expansion is used; underflows gracefully)
        inv = 1.0 / (1j * zeta)
        ratio = S * inv
        pw = inv[:, None] * np.cumprod(np.concatenate([np.ones((zeta.size, 1), dtype=complex),
                                                       np.repeat(ratio[:, None], L - 1, axis=1)], axis=1), axis=1)
        return pw @ Ep, -(pw @ Em)

    if basis.kind == "legendre" or basis.r0 == 0.0:
        A0, B0 = AB(z)
        if basis.kind == "legendre":
            return A0, B0
        kp, km = tab["kp"], tab["km"]
        return 0.5 * A0 * (kp + km), 0.5 * B0 * (kp + km)
    ear = np.exp(1j * basis.a * basis.r0)
    Ap, Bp = AB(z + basis.r0)
    Am, Bm = AB(z - basis.r0)
    kp, km = tab["kp"], tab["km"]
    A = 0.5 * (ear * Ap * kp + Am * km / ear)
    B = 0.5 * (Bp * kp / ear + ear * Bm * km)
    return A, B


# ----------------------------------------------------------------------------
# 5. The zero side
# ----------------------------------------------------------------------------
def zero_density(series, r, kind: str = "smooth"):
    """Density of ordinates (one sign): 'smooth' theta'(r)/pi = (1/2pi)[log(q/pi) + Re psi(alpha + ir/2)],
    continued analytically to complex r as (1/2pi)[log(q/pi) + (psi(alpha + ir/2) + psi(alpha - ir/2))/2];
    'log' (1/2pi) log(q r/(2pi))."""
    from scipy.special import digamma
    ser = get_series(series)
    r = np.asarray(r)
    if kind == "log":
        return np.log(ser.q * r / (2 * np.pi) + 0j) / (2 * np.pi)
    al = ser.alpha
    return (math.log(ser.q / math.pi) + 0.5 * (digamma(al + 0.5j * r) + digamma(al - 0.5j * r))) / (2 * np.pi)


def _composite_gl(edges, n: int = 20):
    x, w = np.polynomial.legendre.leggauss(n)
    xs, ws = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        xs.append(0.5 * (hi - lo) * (x + 1) + lo)
        ws.append(0.5 * (hi - lo) * w)
    return np.concatenate(xs), np.concatenate(ws)


_TAIL_S_EDGES = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0)
_TAIL_CHUNK = 20000


def _tail_parts(basis: Basis, parity: str, series, T: float, density: str):
    """Pieces of 2 int_T^inf h_ij(r) d(r) dr (matrix form):
    [T, T1]: composite Gauss–Legendre on panels of width pi/(2a) with the closed-form phi^;
    [T1, inf): far field h = N + e^{2iar} M + e^{-2iar} M~, N part on r = T1 e^s, the
    oscillatory part on the rotated contour r = T1 + iy (integrand e^{-2ay}; analytic since
    T1 > r0)."""
    a, r0 = basis.a, basis.r0
    K = _sector_tables(basis, parity)["K"]
    T1 = max(float(T), r0 + K * K / a + 50.0)
    n = basis.size(parity)
    out = np.zeros((n, n))
    if T1 > T:
        width = math.pi / (2 * a)
        npan = int(math.ceil((T1 - T) / width))
        rr, ww = _composite_gl(np.linspace(T, T1, npan + 1), 16)
        for s0 in range(0, rr.size, _TAIL_CHUNK):          # bounded memory for large bases
            r_, w_ = rr[s0:s0 + _TAIL_CHUNK], ww[s0:s0 + _TAIL_CHUNK]
            V = fourier_matrix(basis, parity, r_)
            d = np.real(zero_density(series, r_, density))
            out += np.real((V * (w_ * d)[:, None]).T @ np.conj(V))
    # N part
    s, ws = _composite_gl(_TAIL_S_EDGES, 20)
    r = T1 * np.exp(s)
    Ap, Bp = far_field(basis, parity, r)
    Am, Bm = far_field(basis, parity, -r)
    d = np.real(zero_density(series, r, density))
    wN = ws * d * r
    N = (Ap * wN[:, None]).T @ Am + (Bp * wN[:, None]).T @ Bm
    out += np.real(N)
    # oscillatory part: int_T1^inf e^{2iar} M(r) d(r) dr = i e^{2iaT1} int_0^inf e^{-2ay} M(T1+iy) d dy
    x, wx = _composite_gl(_TAIL_S_EDGES, 20)
    y = x / (2 * a)
    zc = T1 + 1j * y
    Ap, Bp = far_field(basis, parity, zc)
    Am, Bm = far_field(basis, parity, -zc)
    dc = zero_density(series, zc, density)
    wM = wx * np.exp(-x) / (2 * a) * dc
    M = (Ap * wM[:, None]).T @ Bm                       # M_ij = A_i(r) B_j(-r)
    I = 1j * np.exp(2j * a * T1) * M
    out += np.real(I + I.T)                             # h_ij + h_ji over 2, times 2 Re
    out = 0.5 * (out + out.T)
    return 2.0 * out, T1


def _sector_sign(parity: str) -> int:
    return 1 if _check_parity(parity) == "even" else -1


def zero_side_matrix(basis: Basis, parity: str, on_line=(), off_line=(), series=None, tail: bool = False,
                     T: Optional[float] = None, n_below_T: Optional[int] = None, s_correction: bool = False,
                     density: str = "smooth", chunk: int = 20000) -> dict:
    """Zero side as a matrix: Z_ij = sum_tau (phi_i^(tau) phi_j^(-tau) + phi_j^(tau) phi_i^(-tau))/2 over
    tau = (rho - 1/2)/i for all given zeros (float64).

    on_line   ordinates gamma >= 0 (each gamma > 0 is the pair +-gamma: 2 sgn phi_i^ phi_j^; gamma = 0 once)
    off_line  zeros rho with Re rho > 1/2 (complex, (re, im) or {'re','im'}), completed to the
              quadruple rho, 1 - rho, conj rho, 1 - conj rho: 4 sgn Re(phi_i^(tau) phi_j^(tau))
              (a real rho to rho, 1 - rho)
    tail      + 2 int_T^inf h_ij(r) d(r) dr with the zero density d of `zero_density` (needs series;
              T defaults to the largest on-line ordinate)
    s_correction  + the boundary term -2 h(T) S(T) of int_T^inf h dS, S(T) = N(T) - theta(T)/pi (-1 for
              zeta), N(T) = n_below_T or, by default, the number of given zeros with 0 < Im rho <= T
              (assumes the list is complete up to T).
    Here sgn = +1 (even sector) or -1 (odd): phi^(-z) = sgn phi^(z)."""
    sg = _sector_sign(parity)
    n = basis.size(parity)
    on = np.asarray(on_line, dtype=float).ravel()
    if np.any(on < 0):
        raise ValueError("pass on-line ordinates gamma >= 0")
    Z_on = np.zeros((n, n))
    for s in range(0, on.size, chunk):
        g = on[s:s + chunk]
        V = fourier_matrix(basis, parity, g)
        wts = np.where(g > 0, 2.0, 1.0)
        Z_on += sg * np.real((V * wts[:, None]).T @ V)
    Z_off = np.zeros((n, n))
    each = []
    for zr in off_line:
        rho = _as_complex(zr)
        if not rho.real > 0.5:
            raise ValueError(f"off-line zeros must have Re rho > 1/2 (mirrors are added): {rho}")
        if rho.imag < 0:
            rho = rho.conjugate()
        tau = (rho - 0.5) / 1j
        v = fourier_matrix(basis, parity, np.array([tau]))[0]
        M = (2.0 if rho.imag == 0 else 4.0) * sg * np.real(np.outer(v, v))
        Z_off += M
        each.append({"rho": [rho.real, rho.imag], "matrix": M})
    Z_tail = np.zeros((n, n))
    T_used, S = None, None
    if tail or s_correction:
        if series is None:
            raise ValueError("the tail needs the series")
        ser = get_series(series)
        T_used = float(T if T is not None else on.max())
        if tail:
            Z_tail, _ = _tail_parts(basis, parity, ser, T_used, density)
        if s_correction:
            S = _S_of_T(ser, T_used, on, off_line, n_below_T)
            V = fourier_matrix(basis, parity, np.array([T_used]))[0]
            Z_tail = Z_tail - 2.0 * S * np.real(np.outer(V, np.conj(V)))
    Z = Z_on + Z_off + Z_tail
    return {"Z": 0.5 * (Z + Z.T), "on_line": Z_on, "off_line": Z_off, "off_line_each": each, "tail": Z_tail,
            "T": T_used, "S_T": S, "n_on_line": int(on.size), "n_off_line": len(each)}


def _as_complex(z) -> complex:
    if isinstance(z, dict):
        return complex(float(z["re"]), float(z["im"]))
    if isinstance(z, (tuple, list)):
        return complex(float(z[0]), float(z[1]))
    return complex(z)


def _S_of_T(ser: SelfDualSeries, T: float, on, off_line, n_below_T) -> float:
    """S(T) = N(T) - theta(T)/pi (- 1 for zeta), N(T) = #zeros with 0 < Im rho <= T."""
    if n_below_T is None:
        n_on = int(np.sum((on > 0) & (on <= T)))
        n_off = 0
        for zr in off_line:
            rho = _as_complex(zr)
            if 0 < abs(rho.imag) <= T:
                n_off += 2                 # rho and 1 - conj(rho) in the upper half plane
        n_below_T = n_on + n_off
    th = float(hardy_theta(ser, T, 30).mid())
    return float(n_below_T) - th / math.pi - (1.0 if ser.pole else 0.0)


def zero_side(basis: Basis, parity: str, coeffs, on_line=(), off_line=(), series=None, tail: bool = False,
              T: Optional[float] = None, n_below_T: Optional[int] = None, s_correction: bool = False,
              density: str = "smooth", chunk: int = 20000) -> dict:
    """Q_zero(f) = sum_rho f^(tau_rho) f^(-tau_rho) for f = sum_i coeffs[i] phi_i (float64), computed
    directly from f^ (not via the matrix): on-line pairs 2 |f^(gamma)|^2, off-line quadruples
    4 sgn Re f^(tau)^2, plus optionally the density tail and the S(T) boundary term (see
    `zero_side_matrix` for the conventions)."""
    sg = _sector_sign(parity)
    c = np.asarray(coeffs, dtype=float)
    on = np.asarray(on_line, dtype=float).ravel()
    if np.any(on < 0):
        raise ValueError("pass on-line ordinates gamma >= 0")
    on_sum = 0.0
    for s in range(0, on.size, chunk):
        g = on[s:s + chunk]
        fh = fourier_matrix(basis, parity, g) @ c
        on_sum += float(np.sum(np.where(g > 0, 2.0, 1.0) * np.abs(fh) ** 2))
    each = []
    for zr in off_line:
        rho = _as_complex(zr)
        if not rho.real > 0.5:
            raise ValueError(f"off-line zeros must have Re rho > 1/2 (mirrors are added): {rho}")
        if rho.imag < 0:
            rho = rho.conjugate()
        tau = (rho - 0.5) / 1j
        fv = complex((fourier_matrix(basis, parity, np.array([tau])) @ c)[0])
        each.append({"rho": [rho.real, rho.imag], "tau": [tau.real, tau.imag], "fhat": [fv.real, fv.imag],
                     "contribution": (2.0 if rho.imag == 0 else 4.0) * sg * (fv * fv).real})
    off_sum = math.fsum(e["contribution"] for e in each)
    tail_val, s_term, T_used, S = 0.0, 0.0, None, None
    if tail or s_correction:
        if series is None:
            raise ValueError("the tail needs the series")
        ser = get_series(series)
        T_used = float(T if T is not None else on.max())
        if tail:
            tail_val = _tail_scalar(basis, parity, c, ser, T_used, density)
        if s_correction:
            S = _S_of_T(ser, T_used, on, off_line, n_below_T)
            fT = complex((fourier_matrix(basis, parity, np.array([T_used])) @ c)[0])
            s_term = -2.0 * S * abs(fT) ** 2
    total = on_sum + off_sum + tail_val + s_term
    return {"total": total, "on_line": on_sum, "off_line": off_sum, "off_line_each": each, "tail": tail_val,
            "s_correction": s_term, "S_T": S, "T": T_used, "n_on_line": int(on.size), "n_off_line": len(each)}


def _tail_scalar(basis: Basis, parity: str, c, series, T: float, density: str) -> float:
    """2 int_T^inf |f^(r)|^2 d(r) dr for one coefficient vector (same scheme as `_tail_parts`)."""
    a, r0 = basis.a, basis.r0
    K = _sector_tables(basis, parity)["K"]
    T1 = max(float(T), r0 + K * K / a + 50.0)
    val = 0.0
    if T1 > T:
        width = math.pi / (2 * a)
        npan = int(math.ceil((T1 - T) / width))
        rr, ww = _composite_gl(np.linspace(T, T1, npan + 1), 16)
        for s0 in range(0, rr.size, _TAIL_CHUNK):
            r_, w_ = rr[s0:s0 + _TAIL_CHUNK], ww[s0:s0 + _TAIL_CHUNK]
            fh = fourier_matrix(basis, parity, r_) @ c
            val += float(np.sum(w_ * np.real(zero_density(series, r_, density)) * np.abs(fh) ** 2))
    s, ws = _composite_gl(_TAIL_S_EDGES, 20)
    r = T1 * np.exp(s)
    Ap, Bp = far_field(basis, parity, r)
    Am, Bm = far_field(basis, parity, -r)
    N = (Ap @ c) * (Am @ c) + (Bp @ c) * (Bm @ c)
    val += float(np.sum(ws * r * np.real(zero_density(series, r, density)) * np.real(N)))
    x, wx = _composite_gl(_TAIL_S_EDGES, 20)
    zc = T1 + 1j * x / (2 * a)
    Ap, _ = far_field(basis, parity, zc)
    _, Bm = far_field(basis, parity, -zc)
    M = (Ap @ c) * (Bm @ c)
    I = 1j * np.exp(2j * a * T1) * np.sum(wx * np.exp(-x) / (2 * a) * zero_density(series, zc, density) * M)
    val += 2.0 * float(np.real(I))
    return 2.0 * val


def load_zero_data(series, n_zeta: int = 10 ** 6) -> dict:
    """Zeros of the series from the repository data, as used by the zero side:
    {'on_line', 'off_line', 'T', 'complete_to_T', 'source'}.
    zeta: the first n_zeta ordinates of rh.data (Arb-certified; complete up to the last one);
    dh:   results/json/davenport_heilbronn_zeros.json (204 on-line + 5 off-line, complete below T = 300
          by two independent counts);
    chi5: data/zeros/lchi5_zeros.npy (904 zeros, all on the line, complete below T = 1000)."""
    ser = get_series(series)
    key = _series_key(ser)
    if key == "zeta":
        from .data import load_zeros
        z = load_zeros(int(n_zeta))
        return {"on_line": z, "off_line": [], "T": float(z[-1]), "complete_to_T": True, "source": "rh.data"}
    if key == "dh":
        d = load_dh_zeros()
        return {"on_line": d["on_line"], "off_line": list(d["off_line"]), "T": float(d["T"]),
                "complete_to_T": bool(d["consistent"]), "source": "davenport_heilbronn_zeros.json"}
    if key == "chi5":
        if not LCHI5_ZEROS.exists():
            raise FileNotFoundError(str(LCHI5_ZEROS))
        z = np.load(LCHI5_ZEROS)
        man = LCHI5_ZEROS.with_suffix(".json")
        T = float(json.loads(man.read_text()).get("T_max", z[-1])) if man.exists() else float(z[-1])
        return {"on_line": np.asarray(z, dtype=float), "off_line": [], "T": T, "complete_to_T": True,
                "source": "lchi5_zeros.npy"}
    raise ValueError(f"no zero data for {ser.name}")


# ----------------------------------------------------------------------------
# 6. Thought experiments: an extra zero quadruple
# ----------------------------------------------------------------------------
def quadruple_matrix(basis: Basis, parity: str, gamma0: float, delta: float, dps: int = 60, weight=1):
    """Exact quadratic form (arb_mat) of a hypothetical zero quadruple at
    rho0 = 1/2 + delta + i gamma0 (with 1 - rho0, conj rho0, 1 - conj rho0), tau0 = gamma0 - i delta:

        M_ij = weight * sum_{tau in {+-tau0, +-conj tau0}} phi_i^(tau) phi_j^(-tau)
             = weight * 4 sgn (Re v_i Re v_j - Im v_i Im v_j),      v = phi^(tau0),

    sgn = +1 (even) / -1 (odd).  Rank <= 2; for delta > 0 generically one positive and one negative
    eigenvalue.  delta = 0 means an extra on-line PAIR rho0, conj rho0 (factor 2 instead of 4,
    = 2 |f^(gamma0)|^2 >= 0).  v is the closed form at dps + 10 digits (`fourier_vector_mp`); its
    midpoints are exact to ~10^-dps but the returned balls have zero radius (not a certificate)."""
    import mpmath as mp
    from flint import arb, arb_mat
    if delta < 0:
        raise ValueError("delta >= 0 (the mirror 1 - rho0 is included automatically)")
    sg = _sector_sign(parity)
    fac = (4 if delta > 0 else 2) * sg
    with mp.workdps(dps + 10):
        tau0 = mp.mpc(mp.mpf(gamma0), -mp.mpf(delta))
        v = fourier_vector_mp(basis, parity, tau0, dps + 10)
        n = len(v)
        with _flint_dps(dps):
            M = arb_mat(n, n)
            w = mp.mpmathify(weight)
            for i in range(n):
                for j in range(n):
                    x = w * fac * (mp.re(v[i]) * mp.re(v[j]) - mp.im(v[i]) * mp.im(v[j]))
                    M[i, j] = arb(mp.nstr(x, dps + 5, strip_zeros=False))
    return M


def add_zero_quadruple(form, gamma0: float, delta: float, weight=1, basis: Optional[Basis] = None,
                       parity: Optional[str] = None, dps: Optional[int] = None):
    """Q + `quadruple_matrix` (arb_mat): the matrix of the form if the series had an extra zero
    quadruple at 1/2 + delta + i gamma0 (weight = -1 removes one; delta = 0: an on-line pair).
    `form` is a SectorForm, or an arb_mat with basis, parity, dps given."""
    if isinstance(form, SectorForm):
        Q, basis, parity, dps = form.Q, form.basis, form.parity, int(dps or form.dps)
    else:
        Q = form
        if basis is None or parity is None or dps is None:
            raise ValueError("pass basis, parity and dps with a bare matrix")
    with _flint_dps(dps):
        return Q + quadruple_matrix(basis, parity, gamma0, delta, dps=dps, weight=weight)


# ----------------------------------------------------------------------------
# 7. Smooth subspaces of the Legendre basis
# ----------------------------------------------------------------------------
def boundary_vanishing_subspace(basis: Basis, parity: str, order: int) -> np.ndarray:
    """Columns: an orthonormal (in coefficient space) basis of the coefficient vectors c of the
    sector for which f = sum c_i phi_i satisfies f^{(l)}(+-a) = 0 for l < order (by parity, +a
    suffices).  The constraint matrix is exact (rational) and its null space is computed exactly
    (sympy), then orthonormalised in float64.  Then f^(r) = O(r^{-order-1}), so the zero side
    converges fast and the density tail is small (useful for series with few known zeros)."""
    import sympy as sp
    if basis.kind == "localized":
        raise ValueError("the localised basis already vanishes to order m at +-a")
    tab_labels = basis.labels(parity)
    rows = []
    for l in range(order):
        row = [_Pkl_one(j, l) for j, _ in tab_labels]          # P_j^{(l)}(1) (0 for l > j)
        rows.append([sp.Rational(x.numerator, x.denominator) for x in row])
    Cm = sp.Matrix(rows)
    ns = Cm.nullspace()
    if not ns:
        return np.zeros((len(tab_labels), 0))
    N = np.array([[float(v[i]) for v in ns] for i in range(len(tab_labels))])
    q, _ = np.linalg.qr(N)
    return q
