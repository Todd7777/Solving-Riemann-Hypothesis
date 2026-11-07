r"""Weil's explicit formula for self-dual Dirichlet series with periodic coefficients.

Setting
-------
F(s) = sum_{n>=1} c_n n^{-s} with real coefficients, periodic of period q,
c_1 = 1 and sum_{n=1}^{q} c_n = 0 (so F = q^{-s} sum_a c_a zeta(s, a/q) is
entire), whose completed function

    Lambda(s) = (q/pi)^{s/2} Gamma((s+e)/2) F(s),        e in {0, 1},

satisfies Lambda(s) = Lambda(1-s) (a root number -1 changes nothing below).
e is the parity of the sequence: e = 0 if c_{q-n} = c_n, e = 1 if c_{q-n} = -c_n.
Covered here (functional equations checked in Arb by `functional_equation_check`):

* L(s, (n/5)):           q = 5, c = (1, -1, -1, 1, 0), e = 0;
* Davenport–Heilbronn f: q = 5, c = (1, kappa, -kappa, -1, 0), e = 1.  The
  equation f(s) = (5/pi)^{1/2-s} Gamma((2-s)/2)/Gamma((1+s)/2) f(1-s) of
  `rh.davenport_heilbronn` is exactly Lambda(s) = Lambda(1-s) with e = 1;
  with e = 0 the relative residual is O(1);
* zeta: q = 1, c = (1,), e = 0, the one case with poles (s = 0, 1).

The formula (validated numerically, see tests/test_explicit_formula_general.py)
-------------------------------------------------------------------------------
Let g be even and real, h(r) = int g(u) e^{iru} du, H(s) = h((s-1/2)/i) =
int g(u) e^{(s-1/2)u} du, alpha = (1/2 + e)/2, and let a_n be the Dirichlet
coefficients of -F'/F:

    sum_{d | n} a_d c_{n/d} = c_n log n     (a_1 = 0; for zeta a_n = Lambda(n),
                                             for a character a_n = chi(n) Lambda(n)).
Then

    sum_rho h(tau_rho) = [h(i/2) + h(-i/2)]   (zeta only)
                         - 2 sum_{n>=2} a_n n^{-1/2} g(log n)
                         + (1/2pi) int_R h(r) [log(q/pi) + Re psi(alpha + i r/2)] dr,      (EF)

    tau_rho = (rho - 1/2)/i,

where rho runs over ALL zeros of Lambda, with multiplicity (= the nontrivial
zeros of F: its trivial zeros -e, -e-2, ... are cancelled by the poles of the
Gamma factor).  The last term, in u-space (from psi(z) = -gamma_E + int_0^inf
(e^{-t} - e^{-zt})/(1 - e^{-t}) dt, t = 2u, and Fourier inversion), is

    (log(q/pi) - gamma_E) g(0) + 2 int_0^inf [g(0) e^{-2u} - g(u) e^{-2 alpha u}] / (1 - e^{-2u}) du.

Zero bookkeeping (F has real coefficients, so zeros come in conjugate pairs,
and Lambda(s) = Lambda(1-s) adds the mirror images): an on-line zero
1/2 + i gamma, gamma > 0, and its conjugate give 2 h(gamma); an off-line zero
rho = beta + i gamma with beta > 1/2 brings the quadruple rho, 1 - rho,
conj(rho), 1 - conj(rho) with tau = gamma - i (beta - 1/2), -tau, conj(tau),
-conj(tau), contributing h(tau) + h(-tau) + h(conj tau) + h(-conj tau)
= 4 Re h(tau) for h even and real on R (`zero_side` adds the mirrors and
conjugates).  This is the orchestrator's tentative formula; no sign or
normalisation had to be changed.

Derivation
----------
1. Zero-free regions.  B(sigma) := sum_{n>=2} |c_n| n^{-sigma} < 1 for
   sigma > sigma_0 (`zero_free_abscissa`: 1.39514 for DH, 1.72865 for zeta).
   In the Banach algebra of Dirichlet series absolutely convergent on
   Re s = sigma, F = 1 + G with ||G|| = B(sigma) < 1, so 1/F, hence -F'/F,
   converge absolutely for Re s > sigma_0, with
       sum_n |a_n| n^{-sigma} <= L(sigma)/(1 - B(sigma)),  L(sigma) = sum |c_n| log n n^{-sigma}
   (`coefficient_norm_bound`), and F != 0 there.  By the functional equation
   Lambda != 0 for Re s < 1 - sigma_0: all zeros of Lambda lie in
   1 - sigma_0 <= Re s <= sigma_0, a strip wider than the critical strip.
2. Fix c > sigma_0 and integrate (Lambda'/Lambda)(s) H(s) over the boundary of
   1 - c <= Re s <= c, |Im s| <= T_k, T_k -> inf avoiding zeros (there
   Lambda'/Lambda = O(log^2 T) and H(sigma + it) = O(t^-2) uniformly).  The
   residues give sum_rho H(rho) (minus H(0) + H(1) for the poles of zeta).
3. Lambda(s) = Lambda(1-s) gives Lambda'/Lambda(s) = -Lambda'/Lambda(1-s), and
   H(1-s) = H(s) for even g, so the line Re s = 1 - c equals the line Re s = c:
   sum_rho H(rho) = 2 (1/2 pi i) int_(c) (Lambda'/Lambda)(s) H(s) ds.
4. Lambda'/Lambda = (1/2) log(q/pi) + (1/2) psi((s+e)/2) + F'/F.  The Gamma
   part has no poles in Re s > -e; moved to Re s = 1/2 it gives the
   archimedean integral (Im psi is odd in r and drops out).  On Re s = c,
   F'/F = -sum a_n n^{-s} converges absolutely, so it may be integrated term
   by term (sum |a_n| n^{-c} * int |H(c+it)| dt < inf), and
   (1/2 pi i) int_(c) n^{-s} H(s) ds = n^{-1/2} g(log n) (shift the entire
   function n^{-s} H(s) to Re s = 1/2, Fourier inversion).

Why no Euler product is needed.  DH has none: (c_n) is not multiplicative,
a_n is not supported on prime powers (a_6 = (1 + kappa^2) log 6 != 0), and the
Dirichlet series of -F'/F is only known to converge absolutely for
Re s > 1.395 (it diverges somewhere in Re s > 1, because DH has zeros with
Re s > 1).  The proof uses that series on ONE line Re s = c > sigma_0 only,
where it converges absolutely; each term contributes n^{-1/2} g(log n), so for
g supported in [-U, U] the "prime" side is a FINITE sum over all n <= e^U.
Nothing is assumed about -F'/F inside the strip: instead the contour shift
from Re s = c to Re s = 1 - c must pick up EVERY zero of Lambda in
1 - c < Re s < c -- on-line, off-line, and (for DH, at larger heights) those
with 1 < Re s < sigma_0 together with their mirrors in Re s < 0.  Dropping
the off-line quadruples breaks (EF) by exactly their contribution (measured:
0.75 for g = e^{-u^2/(2 0.15^2)} cos(85.7 u)).

Conditions on g (sufficient): g even, real, C^2, and g, g', g'' times
e^{(c - 1/2)|u|} integrable for some c > sigma_0 (c > 1 for zeta).  Gaussians
e^{-u^2/(2 s^2)} (times cos(w u)) and C^2 functions of compact support
qualify; continuous piecewise-C^2 g of compact support (h = O(r^-2)) also work.

Zero density.  For entire Lambda, N(T) := #{rho : 0 < Im rho <= T} =
theta(T)/pi + S(T) with theta(T) = (T/2) log(q/pi) + Im log Gamma((1/2+e+iT)/2)
and S(T) = arg F(1/2 + iT)/pi (continuous variation from Re s = c)
(`zero_count`; zeta: + 1).  Hence dN/dT ~ (1/2pi)[log(q/pi) + Re psi(alpha + iT/2)]
= (1/2pi) log(qT/(2pi)) + O(T^-2): the archimedean integrand is the smooth
zero density.  `zero_side` can add the tail 2 int_T^inf h(r) dN(r).

Numerics and rigour
-------------------
* a_n: float64 sieve (`log_derivative_coefficients`), Arb balls
  (`log_derivative_coefficients_arb`, rigorous enclosures) and exact symbolic
  form a_n = sum_{p | n} b_{n,p} log p with b_{n,p} in Z[c_1, ..., c_q]
  (`log_derivative_coefficients_exact`; for DH b_{n,p} in Z[kappa], kappa a
  root of x^4 + 2x^3 - 6x^2 - 2x + 1).
* `arithmetic_side` (float64, scipy quad) evaluates (EF) for any even g given as
  an `rh.explicit_formula.TestFunction`; u-space and r-space archimedean terms
  agree to ~1e-14.
* `arithmetic_side_arb` evaluates (EF) for the Gaussian family
  g = e^{-u^2/(2 s^2)} cos(w u) in Arb ball arithmetic with every truncation
  bounded (prime-type tail via |a_n| <= n^sigma L/(1-B); r-integral by
  acb.integral; |psi(z)| <= gamma_E + 5|z - 1| for Re z >= 1/4 for the r-tail):
  a rigorous enclosure of the arithmetic side.
* `zero_side` / `zero_side_arb` evaluate the zero side over a GIVEN list of
  zeros.  For DH the list (results/json/davenport_heilbronn_zeros.json) has
  zeros certified individually (sign changes of Z, Krawczyk boxes), but its
  completeness below T = 300 rests on a numerical winding-number count, and
  the zeros above T are only estimated by the density; so the identity
  "arithmetic side = zero side" is a numerical check, not a proof.
"""
from __future__ import annotations

import math
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence

import numpy as np
from scipy.integrate import quad
from scipy.special import digamma

from .explicit_formula import EULER_GAMMA, TestFunction

__all__ = [
    "SelfDualSeries", "ZETA", "CHI5", "DAVENPORT_HEILBRONN", "coefficient_parity", "from_periodic_series",
    "series_value", "completed_value", "functional_equation_check", "hardy_theta", "hardy_Z",
    "on_line_zeros", "zero_count", "certify_on_line_zeros",
    "log_derivative_coefficients", "log_derivative_coefficients_arb", "log_derivative_coefficients_exact",
    "exact_coefficient", "zero_free_abscissa", "coefficient_norm_bound",
    "prime_type_sum", "archimedean_term_u", "archimedean_term_r", "arithmetic_side",
    "zero_density", "zero_side", "explicit_formula_check",
    "gaussian_family_arb", "arithmetic_side_arb", "zero_side_arb", "load_dh_zeros",
]

TWO_PI = 2.0 * math.pi
DH_ZEROS_JSON = Path(__file__).resolve().parents[2] / "results" / "json" / "davenport_heilbronn_zeros.json"


@contextmanager
def _flint_dps(dps: Optional[int]):
    """Temporarily set python-flint's working precision (restored afterwards)."""
    from flint import ctx
    old = ctx.dps
    if dps is not None:
        ctx.dps = int(dps)
    try:
        yield
    finally:
        ctx.dps = old


# ----------------------------------------------------------------------------
# 1. The Dirichlet series
# ----------------------------------------------------------------------------
def coefficient_parity(coeffs: Sequence[float], tol: float = 1e-14) -> Optional[int]:
    """0 if c_{q-n} = c_n for all n (even), 1 if c_{q-n} = -c_n (odd), None otherwise
    (indices mod q; c_0 := c_q)."""
    c = [float(x) for x in coeffs]
    q = len(c)
    at = lambda n: c[(n - 1) % q]
    if all(abs(at(q - n) - at(n)) <= tol for n in range(1, q + 1)):
        return 0
    if all(abs(at(q - n) + at(n)) <= tol for n in range(1, q + 1)):
        return 1
    return None


@dataclass(frozen=True)
class SelfDualSeries:
    """F(s) = sum c_n n^{-s}, c periodic with period q = len(coeffs), c_1 = 1,
    with Lambda(s) = (q/pi)^{s/2} Gamma((s+e)/2) F(s) = Lambda(1-s).

    coeffs        c_1, ..., c_q as floats (float64 code paths)
    e             parity of the Gamma factor (must match the parity of c)
    pole          True only for zeta (q = 1, c = (1,)): adds h(i/2) + h(-i/2)
    arb_coeffs    optional () -> list of arb: exact enclosures of c_1..c_q at the
                  current flint precision (default: the floats, exactly)
    sympy_coeffs  optional exact c_1..c_q (sympy; may contain symbols)
    sympy_subs    numerical meaning of the symbols in sympy_coeffs
    sympy_minpoly optional (symbol, polynomial) used to reduce exact coefficients
    """
    name: str
    coeffs: tuple
    e: int
    pole: bool = False
    arb_coeffs: Optional[Callable] = field(default=None, compare=False, hash=False, repr=False)
    sympy_coeffs: Optional[tuple] = field(default=None, compare=False, hash=False, repr=False)
    sympy_subs: Optional[dict] = field(default=None, compare=False, hash=False, repr=False)
    sympy_minpoly: Optional[tuple] = field(default=None, compare=False, hash=False, repr=False)

    def __post_init__(self):
        c = tuple(float(x) for x in self.coeffs)
        object.__setattr__(self, "coeffs", c)
        if len(c) < 1 or c[0] != 1.0:
            raise ValueError("need c_1 = 1")
        if self.e not in (0, 1):
            raise ValueError("e must be 0 or 1")
        if self.pole:
            if c != (1.0,) or self.e != 0:
                raise ValueError("pole=True is reserved for zeta: q = 1, c = (1,), e = 0")
        elif abs(math.fsum(c)) > 1e-12:
            raise ValueError("sum of the coefficients over a period must vanish (F entire)")
        par = coefficient_parity(c)
        if par != self.e:
            raise ValueError(f"parity of the coefficients is {par}, but e = {self.e}")

    @property
    def q(self) -> int:
        return len(self.coeffs)

    @property
    def alpha(self) -> float:
        """alpha = (1/2 + e)/2: the Gamma factor on the critical line is Gamma(alpha + i r/2)."""
        return (0.5 + self.e) / 2.0

    @property
    def log_conductor_term(self) -> float:
        """log(q/pi)."""
        return math.log(self.q / math.pi)

    def c_array(self, N: int) -> np.ndarray:
        """c_0..c_N as a float array (c_0 := 0 is never used)."""
        n = np.arange(N + 1)
        out = np.asarray(self.coeffs)[(n - 1) % self.q].copy()
        out[0] = 0.0
        return out

    def c_arb(self) -> list:
        from flint import arb
        if self.arb_coeffs is not None:
            return list(self.arb_coeffs())
        return [arb(x) for x in self.coeffs]


def _dh_kappa_arb():
    from flint import arb
    return ((10 - 2 * arb(5).sqrt()).sqrt() - 2) / (arb(5).sqrt() - 1)


def _dh_arb_coeffs():
    from flint import arb
    k = _dh_kappa_arb()
    return [arb(1), k, -k, arb(-1), arb(0)]


def _dh_sympy():
    import sympy as sp
    K = sp.Symbol("kappa")
    x = sp.Symbol("x")
    radical = (sp.sqrt(10 - 2 * sp.sqrt(5)) - 2) / (sp.sqrt(5) - 1)
    return K, radical, (K, K ** 4 + 2 * K ** 3 - 6 * K ** 2 - 2 * K + 1)


_K, _KAPPA_RADICAL, _KAPPA_MINPOLY = _dh_sympy()
_KAPPA = (math.sqrt(10 - 2 * math.sqrt(5)) - 2) / (math.sqrt(5) - 1)

ZETA = SelfDualSeries("zeta(s)", (1.0,), e=0, pole=True, sympy_coeffs=(1,))
CHI5 = SelfDualSeries("L(s,(n/5))", (1.0, -1.0, -1.0, 1.0, 0.0), e=0, sympy_coeffs=(1, -1, -1, 1, 0))
DAVENPORT_HEILBRONN = SelfDualSeries(
    "Davenport-Heilbronn", (1.0, _KAPPA, -_KAPPA, -1.0, 0.0), e=1, arb_coeffs=_dh_arb_coeffs,
    sympy_coeffs=(1, _K, -_K, -1, 0), sympy_subs={_K: _KAPPA_RADICAL}, sympy_minpoly=_KAPPA_MINPOLY)


def from_periodic_series(ps, name: Optional[str] = None) -> SelfDualSeries:
    """Wrap an `rh.nyman_beurling_periodic.PeriodicSeries` (parity taken from it).
    The functional equation is NOT implied by the parity: check it with
    `functional_equation_check`."""
    return SelfDualSeries(name or ps.name, tuple(ps.coeffs), e=int(ps.parity))


# ----------------------------------------------------------------------------
# 2. F, Lambda, the functional equation, Z(t), zeros on the line
# ----------------------------------------------------------------------------
def _acb(s):
    from flint import acb
    if isinstance(s, acb):
        return s
    z = complex(s)
    return acb(z.real, z.imag)


def series_value(series: SelfDualSeries, s, dps: Optional[int] = None):
    """F(s) = q^{-s} sum_a c_a zeta(s, a/q) as an Arb enclosure (s != 1)."""
    from flint import acb
    with _flint_dps(dps):
        s = _acb(s)
        c = series.c_arb()
        q = series.q
        tot = acb(0)
        for a in range(1, q + 1):
            if series.coeffs[a - 1] != 0.0:
                tot += c[a - 1] * s.zeta(acb(a) / q)
        return tot * acb(q) ** (-s)


def completed_value(series: SelfDualSeries, s, dps: Optional[int] = None, e: Optional[int] = None):
    """Lambda(s) = (q/pi)^{s/2} Gamma((s+e)/2) F(s) (Arb); `e` overrides the parity."""
    from flint import acb, arb
    with _flint_dps(dps):
        s = _acb(s)
        ee = series.e if e is None else int(e)
        return (arb(series.q) / arb.pi()) ** (s / 2) * ((s + ee) / 2).gamma() * series_value(series, s)


def functional_equation_check(series: SelfDualSeries, points: Iterable = (
        complex(0.3, 10.0), complex(0.8, 85.7), complex(1.7, 3.3), complex(-0.4, 50.2), complex(2.5, -7.1)),
        dps: int = 40, e: Optional[int] = None) -> list:
    """|Lambda(s) - Lambda(1-s)| (rigorous upper bound) and its relative size at each point."""
    out = []
    with _flint_dps(dps):
        for p in points:
            s = _acb(p)
            L1 = completed_value(series, s, e=e)
            L2 = completed_value(series, 1 - s, e=e)
            d = abs(L1 - L2)
            out.append({"s": [complex(p).real, complex(p).imag], "abs_residual": float(d.upper()),
                        "rel_residual": float((d / abs(L1)).upper()), "abs_Lambda": float(abs(L1).mid())})
    return out


def hardy_theta(series: SelfDualSeries, t, dps: Optional[int] = None):
    """theta(t) = (t/2) log(q/pi) + Im log Gamma((1/2 + e + i t)/2) (arb), so that
    Lambda(1/2 + it) = (q/pi)^{1/4} |Gamma((1/2+e+it)/2)| e^{i theta(t)} F(1/2 + it)."""
    from flint import acb, arb
    with _flint_dps(dps):
        tt = arb(t)
        return tt / 2 * (arb(series.q) / arb.pi()).log() + acb((1 + 2 * series.e) / arb(4), tt / 2).lgamma().imag


def hardy_Z(series: SelfDualSeries, t, dps: Optional[int] = None):
    """Z(t) = e^{i theta(t)} F(1/2 + it): real for self-dual F with root number +1.
    Returns the acb enclosure (its imaginary part contains 0)."""
    from flint import acb, arb
    with _flint_dps(dps):
        th = hardy_theta(series, t)
        return (acb(0, 1) * th).exp() * series_value(series, acb(arb(1) / 2, arb(t)))


def on_line_zeros(series: SelfDualSeries, t0: float, t1: float, step: float = 0.05, tol: float = 1e-13,
                  dps: int = 20) -> np.ndarray:
    """Sign changes of Z(t) on (t0, t1), refined by the Illinois method to `tol`.
    Zeros of even order or pairs closer than `step` are missed: compare the count
    with `zero_count`."""
    zf = lambda t: float(hardy_Z(series, float(t), dps).real.mid())
    grid = np.arange(t0, t1 + 0.5 * step, step)
    grid[-1] = min(grid[-1], t1)
    vals = np.array([zf(t) for t in grid])
    roots = []
    for i in np.nonzero(np.sign(vals[:-1]) * np.sign(vals[1:]) < 0)[0]:
        a, b, fa, fb = float(grid[i]), float(grid[i + 1]), float(vals[i]), float(vals[i + 1])
        side = 0
        for _ in range(200):
            m = (a * fb - b * fa) / (fb - fa)
            fm = zf(m)
            if fm == 0.0 or (b - a) < tol:
                a = b = m
                break
            if (fm < 0) == (fa < 0):
                a, fa = m, fm
                if side == -1:
                    fb *= 0.5
                side = -1
            else:
                b, fb = m, fm
                if side == 1:
                    fa *= 0.5
                side = 1
            if abs(b - a) < tol:
                break
        roots.append(0.5 * (a + b))
    return np.array(roots)


def zero_count(series: SelfDualSeries, T: float, sigma_right: float = 3.0, dps: int = 20,
               step: float = 0.05) -> dict:
    """N(T) = #{rho: 0 < Im rho <= T} = theta(T)/pi + S(T) (+1 for zeta), with
    S(T) = (1/pi) arg F(1/2 + iT), arg by continuous variation from s = sigma_right
    (where |F - 1| < 1, so arg F(sigma_right + iT) is the principal value) along
    the horizontal segment (adaptive sampling of `rh.speiser.edge_arg_change`;
    numerical, not rigorous).  Returns the real value and its nearest integer."""
    from .speiser import edge_arg_change
    B = _B_of_sigma(series, sigma_right)
    if not B < 1.0:
        raise ValueError(f"sigma_right = {sigma_right} is too small: B = {B} >= 1")
    fz = lambda z: _acb_to_complex(series_value(series, z, dps))
    z_right = complex(sigma_right, T)
    arg0 = math.atan2(fz(z_right).imag, fz(z_right).real)
    darg = edge_arg_change(fz, z_right, complex(0.5, T), step, 0.5)
    th = float(hardy_theta(series, T, dps).mid())
    S = (arg0 + darg) / math.pi
    val = th / math.pi + S + (1.0 if series.pole else 0.0)
    return {"T": float(T), "theta_over_pi": th / math.pi, "S": S, "N_real": val, "N": int(round(val)),
            "distance_to_integer": abs(val - round(val))}


def _acb_to_complex(z) -> complex:
    return complex(float(z.real.mid()), float(z.imag.mid()))


def certify_on_line_zeros(Z: Callable, ordinates: Sequence[float], delta: float = 1e-11,
                          dps: int = 30) -> dict:
    """Rigorous existence check: for each ordinate gamma, Z(gamma - delta) and
    Z(gamma + delta) (Arb enclosures of a continuous real function, Z(t) -> arb)
    have certified opposite signs, so [gamma - delta, gamma + delta] contains a zero.
    Also checks that the intervals are pairwise disjoint (distinct zeros)."""
    ok = []
    with _flint_dps(dps):
        for gma in ordinates:
            zm, zp = Z(float(gma) - delta), Z(float(gma) + delta)
            ok.append(bool((zm < 0 and zp > 0) or (zm > 0 and zp < 0)))
    o = np.sort(np.asarray(ordinates, dtype=float))
    disjoint = bool(np.all(np.diff(o) > 2 * delta)) if o.size > 1 else True
    return {"n": len(ok), "n_certified": int(sum(ok)), "certified": ok, "delta": delta, "disjoint": disjoint,
            "all_certified": bool(all(ok) and disjoint)}


# ----------------------------------------------------------------------------
# 3. The coefficients a_n of -F'/F
# ----------------------------------------------------------------------------
_COEFF_CACHE: dict = {}


def log_derivative_coefficients(series: SelfDualSeries, N: int) -> np.ndarray:
    """a_0..a_N (a_0 := 0, a_1 = 0) from sum_{d|n} a_d c_{n/d} = c_n log n, float64
    sieve in O(N log N).  Cached; the returned array is read-only."""
    N = int(N)
    key = (series.coeffs, series.pole)
    got = _COEFF_CACHE.get(key)
    if got is not None and got.size >= N + 1:
        return got[:N + 1]
    c = series.c_array(N)
    nz = c != 0.0
    a = np.zeros(N + 1)
    S = np.zeros(N + 1)
    logs = np.log(np.maximum(np.arange(N + 1), 1))
    for n in range(2, N + 1):
        an = c[n] * logs[n] - S[n]
        a[n] = an
        if an != 0.0 and 2 * n <= N:
            mmax = N // n
            m = np.arange(2, mmax + 1)
            sel = nz[2:mmax + 1]
            S[n * m[sel]] += an * c[m[sel]]
    a.setflags(write=False)
    _COEFF_CACHE[key] = a
    return a


def log_derivative_coefficients_arb(series: SelfDualSeries, N: int, dps: Optional[int] = None) -> list:
    """a_0..a_N as rigorous Arb enclosures (same recursion, ball arithmetic)."""
    from flint import arb
    with _flint_dps(dps):
        c = series.c_arb()
        q = series.q
        nz = [x != 0.0 for x in series.coeffs]
        zero = arb(0)
        a = [zero] * (N + 1)
        S = [zero] * (N + 1)
        for n in range(2, N + 1):
            r = (n - 1) % q
            an = (c[r] * arb(n).log() if nz[r] else zero) - S[n]
            a[n] = an
            for m in range(2, N // n + 1):
                rm = (m - 1) % q
                if nz[rm]:
                    S[n * m] = S[n * m] + an * c[rm]
        return a


def log_derivative_coefficients_exact(series: SelfDualSeries, N: int, reduce: bool = False) -> list:
    """Exact a_n = sum_{p | n} b_{n,p} log p with b_{n,p} in Z[c_1..c_q] (sympy).
    Returns a list (index n = 0..N) of dicts {p: b_{n,p}} (zero entries dropped).
    With reduce=True and a minimal polynomial on the series (DH: kappa), the
    b_{n,p} are reduced modulo it (canonical form of degree < 4 in kappa)."""
    import sympy as sp
    if series.sympy_coeffs is None:
        raise ValueError("series has no exact (sympy) coefficients")
    c = [sp.sympify(x) for x in series.sympy_coeffs]
    q = series.q
    mp_ = series.sympy_minpoly if reduce else None

    def red(x):
        x = sp.expand(x)
        if mp_ is not None and x != 0:
            x = sp.expand(sp.rem(sp.Poly(x, mp_[0]), sp.Poly(mp_[1], mp_[0])).as_expr())
        return x

    B = [dict() for _ in range(N + 1)]
    S = [defaultdict(lambda: sp.Integer(0)) for _ in range(N + 1)]
    for n in range(2, N + 1):
        cn = c[(n - 1) % q]
        fac = sp.factorint(n)
        bn = {}
        for p in set(fac) | set(S[n]):
            v = red(cn * fac.get(p, 0) - S[n].get(p, 0))
            if v != 0:
                bn[p] = v
        B[n] = bn
        for m in range(2, N // n + 1):
            cm = c[(m - 1) % q]
            if cm != 0:
                for p, v in bn.items():
                    S[n * m][p] = S[n * m][p] + v * cm
    return B


def exact_coefficient(series: SelfDualSeries, b_np: dict):
    """sympy expression sum_p b_{n,p} log p for one entry of `log_derivative_coefficients_exact`."""
    import sympy as sp
    return sp.Add(*[sp.factor(v) * sp.log(p) for p, v in sorted(b_np.items())]) if b_np else sp.Integer(0)


def _B_of_sigma(series: SelfDualSeries, sigma: float) -> float:
    """B(sigma) = sum_{n>=2} |c_n| n^{-sigma} (mpmath Hurwitz zeta)."""
    import mpmath as mp
    q = series.q
    tot = mp.mpf(0)
    for a in range(1, q + 1):
        ca = abs(series.coeffs[a - 1])
        if ca:
            tot += ca * mp.zeta(sigma, mp.mpf(a) / q)
    return float(tot * mp.mpf(q) ** (-sigma) - 1)


def zero_free_abscissa(series: SelfDualSeries) -> float:
    """sigma_0 with B(sigma_0) = sum_{n>=2} |c_n| n^{-sigma_0} = 1: F != 0 and the
    Dirichlet series of -F'/F converges absolutely for Re s > sigma_0."""
    import mpmath as mp
    return float(mp.findroot(lambda s: _B_of_sigma(series, float(s)) - 1, 1.5))


def coefficient_norm_bound(series: SelfDualSeries, sigma: float = 2.0, dps: Optional[int] = None):
    """Rigorous (Arb) A, B, L with sum_n |a_n| n^{-sigma} <= A = L/(1 - B), hence
    |a_n| <= A n^sigma for all n; B = sum_{n>=2} |c_n| n^{-sigma},
    L = sum_{n>=2} |c_n| log n n^{-sigma} via Hurwitz zeta power series."""
    from flint import acb, acb_series, arb
    with _flint_dps(dps):
        c = series.c_arb()
        q = series.q
        s = acb_series([acb(sigma), 1], 2)
        tot = acb_series([0], 2)
        qs = (s * (-arb(q).log())).exp()
        for a in range(1, q + 1):
            if series.coeffs[a - 1] != 0.0:
                tot += abs(c[a - 1]) * qs * acb_series.zeta(s, acb(a) / q)
        Z0, Z1 = tot.coeffs()
        B = Z0.real - 1
        L = -Z1.real
        if not B < 1:
            raise ValueError(f"B(sigma) >= 1 at sigma = {sigma}; take a larger sigma")
        return L / (1 - B), B, L


# ----------------------------------------------------------------------------
# 4. The arithmetic side
# ----------------------------------------------------------------------------
def _vec(g):
    def gv(x):
        x = np.asarray(x, dtype=float)
        try:
            return np.asarray(g(x), dtype=float)
        except Exception:
            return np.array([float(g(float(t))) for t in x.ravel()]).reshape(x.shape)
    return gv


def prime_type_sum(series: SelfDualSeries, g: Callable, N: int, a: Optional[np.ndarray] = None,
                   n_min: int = 2) -> float:
    """2 sum_{n_min <= n <= N} a_n n^{-1/2} g(log n)."""
    N = int(N)
    if N < n_min:
        return 0.0
    if a is None:
        a = log_derivative_coefficients(series, N)
    n = np.arange(n_min, N + 1)
    an = np.asarray(a[n_min:N + 1])
    sel = an != 0.0
    terms = an[sel] / np.sqrt(n[sel]) * _vec(g)(np.log(n[sel].astype(float)))
    return float(2.0 * math.fsum(terms.tolist()))


def archimedean_term_u(series: SelfDualSeries, g: Callable, u_cut: float, u_points: Sequence[float] = (),
                       epsabs: float = 1e-14, epsrel: float = 1e-13) -> float:
    """(log(q/pi) - gamma_E) g(0) + 2 int_0^inf [g(0) e^{-2u} - g(u) e^{-2 alpha u}]/(1 - e^{-2u}) du.
    g must vanish (or be negligible) beyond u_cut; there the integral is
    -(g(0)/2) log(1 - e^{-2 u_cut}) in closed form."""
    al = series.alpha
    g0 = float(g(0.0))

    def f(u):
        if u < 1e-9:      # removable singularity: limit (alpha - 1) g(0), next term O(u)
            return (al - 1.0) * g0
        return float((g0 * math.exp(-2 * u) - float(g(u)) * math.exp(-2 * al * u)) / (-math.expm1(-2 * u)))

    pts = sorted(set(float(p) for p in u_points if 0 < p < u_cut))
    edges = [0.0] + pts + [float(u_cut)]
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        total += quad(f, lo, hi, limit=400, epsabs=epsabs, epsrel=epsrel)[0]
    total += -0.5 * g0 * math.log1p(-math.exp(-2 * u_cut))
    return (series.log_conductor_term - EULER_GAMMA) * g0 + 2.0 * total


def archimedean_term_r(series: SelfDualSeries, h: Callable, r_windows=((0.0, 50.0),),
                       epsabs: float = 1e-14, epsrel: float = 1e-13) -> float:
    """(1/2pi) int_R h(r) [log(q/pi) + Re psi(alpha + i r/2)] dr = (1/pi) int_0^inf (h even),
    integrated over the windows of r >= 0 where h is not negligible."""
    al, L = series.alpha, series.log_conductor_term
    f = lambda r: float(np.real(h(r)) * (L + np.real(digamma(al + 0.5j * r))))
    total = 0.0
    for lo, hi in r_windows:
        n_sub = max(1, int(math.ceil((hi - lo) / 50.0)))
        edges = np.linspace(lo, hi, n_sub + 1)
        for a, b in zip(edges[:-1], edges[1:]):
            total += quad(f, a, b, limit=400, epsabs=epsabs, epsrel=epsrel)[0]
    return total / math.pi


def arithmetic_side(series: SelfDualSeries, tf: TestFunction, N: Optional[int] = None, arch: str = "u",
                    truncation_check: bool = True) -> dict:
    """Right-hand side of (EF) for an even test function (an `rh.explicit_formula.TestFunction`:
    g, complex-capable h, u_support or u_max, r_windows).

    N defaults to floor(e^{u_cut}) (exact for compact support).  arch = 'u', 'r' or
    'both' (u-space value used, r-space reported).  With truncation_check and
    non-compact g, the omitted terms N < n <= e^{1.25 u_cut} are summed and
    reported as `prime_sum_truncation_estimate`."""
    U = tf.u_cut()
    if N is None:
        N = max(1, int(math.floor(math.exp(U) * (1 + 1e-12))))
    a = log_derivative_coefficients(series, N)
    P = prime_type_sum(series, tf.g, N, a)
    out = {"series": series.name, "test_function": tf.name, "params": dict(tf.params), "N": int(N)}
    Wu = archimedean_term_u(series, tf.g, U, tf.u_points) if arch in ("u", "both") else None
    Wr = archimedean_term_r(series, tf.h, tf.r_windows) if arch in ("r", "both") else None
    W = Wu if Wu is not None else Wr
    poles = float(2.0 * np.real(tf.h(np.complex128(0.5j)))) if series.pole else 0.0
    out.update({"poles": poles, "prime_sum": P, "archimedean": W, "archimedean_u": Wu, "archimedean_r": Wr,
                "arithmetic_side": poles - P + W})
    if Wu is not None and Wr is not None:
        out["archimedean_u_minus_r"] = Wu - Wr
    if truncation_check and tf.u_support is None:
        N2 = int(math.floor(math.exp(1.25 * U)))
        if N2 > N:
            a2 = log_derivative_coefficients(series, N2)
            out["prime_sum_truncation_estimate"] = prime_type_sum(series, tf.g, N2, a2, n_min=N + 1)
            out["N_check"] = N2
    return out


# ----------------------------------------------------------------------------
# 5. The zero side
# ----------------------------------------------------------------------------
def zero_density(series: SelfDualSeries, r, kind: str = "log"):
    """Density of zero ordinates (one sign) at height r: (1/2pi) log(q r/(2pi)) ('log')
    or the smooth (1/2pi)[log(q/pi) + Re psi(alpha + i r/2)] ('smooth')."""
    r = np.asarray(r, dtype=float)
    if kind == "log":
        return np.log(series.q * r / TWO_PI) / TWO_PI
    return (series.log_conductor_term + np.real(digamma(series.alpha + 0.5j * r))) / TWO_PI


def _as_complex(z) -> complex:
    if isinstance(z, dict):
        return complex(float(z["re"]), float(z["im"]))
    if isinstance(z, (tuple, list)):
        return complex(float(z[0]), float(z[1]))
    return complex(z)


def zero_side(h: Callable, on_line: Sequence[float] = (), off_line: Sequence = (),
              series: Optional[SelfDualSeries] = None, tail: bool = False, T: Optional[float] = None,
              r_windows=None, density: str = "log") -> dict:
    """sum_rho h(tau_rho), tau = (rho - 1/2)/i, over the given zeros.

    on_line   ordinates gamma > 0 of zeros 1/2 + i gamma (each gives h(gamma) + h(-gamma);
              gamma = 0 counts once)
    off_line  zeros rho with Re rho > 1/2 (complex, (re, im) or {'re','im'}); each is
              completed to rho, 1 - rho, conj rho, 1 - conj rho (a real rho to rho, 1 - rho)
    tail      add 2 int_T^inf h(r) d(r) dr with the zero density d of `zero_density`
              (needs series; T defaults to the last on-line ordinate; integrated
              over r_windows intersected with [T, inf))."""
    on = np.asarray(on_line, dtype=float)
    if np.any(on < 0):
        raise ValueError("pass on-line ordinates gamma >= 0 (conjugates are added)")
    pos, zer = on[on > 0], on[on == 0]
    on_sum = float(2.0 * np.sum(np.real(h(pos)))) + float(np.sum(np.real(h(zer)))) if on.size else 0.0
    each, imag_max = [], 0.0
    for z in off_line:
        rho = _as_complex(z)
        if not rho.real > 0.5:
            raise ValueError(f"off-line zeros must have Re rho > 1/2 (mirrors are added): {rho}")
        if rho.imag < 0:
            rho = rho.conjugate()
        tau = (rho - 0.5) / 1j
        pts = [tau, -tau] if rho.imag == 0 else [tau, -tau, tau.conjugate(), -tau.conjugate()]
        v = complex(sum(complex(h(np.complex128(p))) for p in pts))
        each.append({"rho": [rho.real, rho.imag], "tau": [tau.real, tau.imag], "contribution": v.real,
                     "imag": v.imag})
        imag_max = max(imag_max, abs(v.imag))
    off_sum = math.fsum(d["contribution"] for d in each)
    tail_val = 0.0
    if tail:
        if series is None:
            raise ValueError("the density tail needs the series (conductor, parity)")
        Tt = float(T if T is not None else (on.max() if on.size else 0.0))
        wins = r_windows if r_windows is not None else ((Tt, Tt + 200.0),)
        for lo, hi in wins:
            a, b = max(lo, Tt), hi
            if b <= a:
                continue
            n_sub = max(1, int(math.ceil((b - a) / 50.0)))
            edges = np.linspace(a, b, n_sub + 1)
            for p, q_ in zip(edges[:-1], edges[1:]):
                tail_val += quad(lambda r: float(np.real(h(r)) * zero_density(series, r, density)), p, q_,
                                 limit=400, epsabs=1e-16, epsrel=1e-12)[0]
        tail_val *= 2.0
    return {"on_line": on_sum, "off_line": off_sum, "off_line_each": each, "off_line_imag_max": imag_max,
            "tail": tail_val, "total": on_sum + off_sum + tail_val, "n_on_line": int(on.size),
            "n_off_line": len(each)}


def explicit_formula_check(series: SelfDualSeries, tf: TestFunction, on_line: Sequence[float] = (),
                           off_line: Sequence = (), N: Optional[int] = None, tail: bool = False,
                           T: Optional[float] = None, arch: str = "u", density: str = "log") -> dict:
    """Both sides of (EF); residual = zero side - arithmetic side.  Also reports the
    residual when the off-line zeros are omitted (residual_without_off_line)."""
    A = arithmetic_side(series, tf, N=N, arch=arch)
    Z = zero_side(tf.h, on_line, off_line, series=series, tail=tail, T=T, r_windows=tf.r_windows,
                  density=density)
    res = Z["total"] - A["arithmetic_side"]
    scale = max(abs(Z["total"]), abs(A["arithmetic_side"]), 1e-300)
    return {"series": series.name, "test_function": tf.name, "params": dict(tf.params), "arithmetic": A,
            "zeros": Z, "arithmetic_side": A["arithmetic_side"], "zero_side": Z["total"], "residual": res,
            "relative_residual": abs(res) / scale, "off_line_contribution": Z["off_line"],
            "residual_without_off_line": res - Z["off_line"],
            "relative_residual_without_off_line": abs(res - Z["off_line"]) / scale}


# ----------------------------------------------------------------------------
# 6. Ball arithmetic for the Gaussian family g = e^{-u^2/(2 s^2)} cos(w u)
# ----------------------------------------------------------------------------
def gaussian_family_arb(s: float, w: float = 0.0):
    """(g, h) for g(u) = e^{-u^2/(2 s^2)} cos(w u), h(r) = (s sqrt(2pi)/2)[e^{-s^2(r-w)^2/2}
    + e^{-s^2(r+w)^2/2}], as functions of arb / acb balls (current precision).
    s and w are taken as the exact binary values of the given floats, so this is
    the same test function as rh.explicit_formula.modulated_gaussian(s, w)
    (w = 0: rh.explicit_formula.gaussian(s))."""
    from flint import arb

    def g(u):
        S, W = arb(float(s)), arb(float(w))
        return (-(u * u) / (2 * S * S)).exp() * (W * u).cos()

    def h(z):
        S, W = arb(float(s)), arb(float(w))
        c = S * (2 * arb.pi()).sqrt() / 2
        return c * ((-(S * S) * (z - W) ** 2 / 2).exp() + (-(S * S) * (z + W) ** 2 / 2).exp())

    return g, h


def arithmetic_side_arb(series: SelfDualSeries, s: float, w: float = 0.0, dps: int = 30,
                        tol: Optional[float] = None, sigma: float = 2.0, N_max: int = 100000,
                        R: Optional[float] = None) -> dict:
    """Rigorous Arb enclosure of the arithmetic side of (EF) for g = e^{-u^2/(2 s^2)} cos(w u).

    * prime-type sum: sum_{n<=N} in balls; the rest is bounded with |a_n| <= A n^sigma
      (`coefficient_norm_bound`) and, since t^{sigma-1/2} e^{-(log t)^2/(2s^2)} decreases
      for log t > (sigma - 1/2) s^2, by A e^{b^2 s^2/2} s sqrt(pi/2) erfc((log N - b s^2)/(s sqrt 2)),
      b = sigma + 1/2;
    * archimedean: log(q/pi) g(0) + (1/2pi) int_0^R h(r)[psi(alpha + ir/2) + psi(alpha - ir/2)] dr
      by acb.integral (rigorous), plus a bound for r > R from |h(r)| <= s sqrt(2pi) e^{-s^2 (r-w)^2/2}
      and |psi(z)| <= gamma_E + 5|z - 1| (Re z >= 1/4; from psi(z) = -gamma_E + sum_k [1/(k+1) - 1/(k+z)]);
    * zeta: + h(i/2) + h(-i/2).
    N and R are chosen so that each bound is below tol/4 (default 10^{-(dps-8)}).
    Returns arb balls and their midpoints/radii."""
    from flint import acb, arb
    with _flint_dps(dps):
        S, W = arb(float(s)), arb(float(w))
        pi = arb.pi()
        tol_ = arb(10) ** (-(dps - 8)) if tol is None else arb(tol)
        g, h = gaussian_family_arb(s, w)
        # ---- prime-type sum
        A, B, L = coefficient_norm_bound(series, sigma)
        b = arb(sigma) + arb(0.5)

        def tail_P(Nv):
            x = (arb(Nv).log() - b * S * S) / (S * arb(2).sqrt())
            return A * (b * b * S * S / 2).exp() * S * (pi / 2).sqrt() * x.erfc()

        if math.log(2.0) <= (sigma - 0.5) * s * s:
            raise ValueError("s too large for the monotone tail bound at small n")
        Nv = 4
        while not (2 * tail_P(Nv) < tol_ / 4):
            Nv *= 2
            if Nv > N_max:
                raise ValueError(f"N would exceed N_max = {N_max}; raise tol or N_max")
        lo, hi = Nv // 2, Nv                      # bisect for the smallest admissible N
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if 2 * tail_P(mid) < tol_ / 4:
                hi = mid
            else:
                lo = mid
        Nv = max(hi, 2)
        a = log_derivative_coefficients_arb(series, Nv)
        P = arb(0)
        for n in range(2, Nv + 1):
            if not (a[n] == 0):                       # exact zeros (e.g. a_{5k} for DH) skipped
                P += a[n] * arb(n).rsqrt() * g(arb(n).log())
        P_err = tail_P(Nv)
        # ---- archimedean term in r-space
        al = arb(1 + 2 * series.e) / 4
        C0 = 2 * arb.const_euler() + arb(15) / 2       # |psi(al+ir/2) + psi(al-ir/2)| <= C0 + 5 r

        def tail_A(Rv):
            x = arb(Rv) - W
            I0 = (pi / 2).sqrt() / S * (S * x / arb(2).sqrt()).erfc()
            I1 = (-(S * S * x * x) / 2).exp() / (S * S)
            return S * (2 * pi).sqrt() / (2 * pi) * ((C0 + 5 * W) * I0 + 5 * I1)

        if R is None:
            Rv = float(w) + 10.0 / float(s)
            while not (tail_A(Rv) < tol_ / 4):
                Rv += 2.0 / float(s)
        else:
            Rv = float(R)
        half_i = acb(0, 1) / 2

        def integrand(r, analytic):
            return h(r) * ((al + half_i * r).digamma() + (al - half_i * r).digamma())

        I = acb.integral(integrand, 0, Rv, abs_tol=tol_ / 16, rel_tol=tol_ / 16)
        A_err = tail_A(Rv)
        arch_mid = (arb(series.q) / pi).log() * g(arb(0)) + I.real / (2 * pi)
        arch = arch_mid + arb(0, A_err.upper())
        poles = (2 * h(acb(0, 1) / 2)).real if series.pole else arb(0)
        prime = P + arb(0, P_err.upper())
        total = poles - 2 * prime + arch

        def fl(x):
            return {"mid": float(x.mid()), "rad": float(x.rad())}

        return {"series": series.name, "s": float(s), "w": float(w), "dps": dps, "N": Nv, "R": Rv,
                "coefficient_bound_A": float(A.upper()), "sigma": sigma,
                "prime_tail_bound": float(P_err.upper()), "r_tail_bound": float(A_err.upper()),
                "integral_imag": fl(I.imag), "poles": poles, "prime_sum": 2 * prime, "archimedean": arch,
                "arithmetic_side": total, "arithmetic_side_float": fl(total),
                "prime_sum_float": fl(2 * prime), "archimedean_float": fl(arch)}


def zero_side_arb(s: float, w: float, on_line: Sequence[float] = (), off_line: Sequence = (),
                  on_radius: float = 0.0, off_radius: float = 0.0, dps: int = 30) -> dict:
    """Arb enclosure of sum_rho h(tau_rho) over the GIVEN zeros for the Gaussian family,
    each on-line ordinate taken as the ball [gamma +- on_radius], each off-line zero as
    the box re +- off_radius, im +- off_radius (completed to quadruples)."""
    from flint import acb, arb
    with _flint_dps(dps):
        _, h = gaussian_family_arb(s, w)
        on = arb(0)
        for gma in on_line:
            on += 2 * h(acb(arb(float(gma), on_radius))).real
        each = []
        for z in off_line:
            rho = _as_complex(z)
            if rho.imag < 0:
                rho = rho.conjugate()
            R = acb(arb(rho.real, off_radius), arb(rho.imag, off_radius))
            tau = (R - arb(1) / 2) / acb(0, 1)
            tc = tau.conjugate()
            v = h(tau) + h(-tau) + h(tc) + h(-tc)
            each.append(v.real)
        off = sum(each, arb(0))
        total = on + off
        return {"on_line": on, "off_line": off, "off_line_each": each, "total": total,
                "total_float": {"mid": float(total.mid()), "rad": float(total.rad())}}


# ----------------------------------------------------------------------------
# 7. Data
# ----------------------------------------------------------------------------
def load_dh_zeros(path=None) -> dict:
    """Zeros of the Davenport–Heilbronn function with 0 < Im rho < T from
    results/json/davenport_heilbronn_zeros.json: on-line ordinates, off-line zeros
    (Re rho > 1/2; Krawczyk-certified boxes of radius `off_line_radius`), T."""
    import json
    d = json.loads(Path(path or DH_ZEROS_JSON).read_text())
    off = [complex(z["re"], z["im"]) for z in d["off_line_zeros"]]
    rad = max(float(z.get("radius", 0.0)) for z in d["off_line_zeros"]) if off else 0.0
    return {"on_line": np.asarray(d["on_line_zeros"], dtype=float), "off_line": off, "T": float(d["T"]),
            "off_line_radius": rad, "total_count_winding": int(d.get("total_count_winding", -1)),
            "consistent": bool(d.get("consistent", False))}
