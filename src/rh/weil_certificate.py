r"""Rigorous enclosure of Weil's quadratic form at ONE explicit test function.

`rh.weil_form_general` assembles the arithmetic side of Weil's form on a whole basis, with
quadrature errors that are estimated, not bounded.  Here the form is evaluated at a single
explicit real test function f = sum_i c_i phi_i of a `Basis` (rational support a, frequency r0
and coefficients c_i), and EVERY error is bounded, so the result is a certified enclosure.

The quantity (see `rh.weil_form_general` and `rh.explicit_formula_general` for the derivation)
-------------------------------------------------------------------------------------------
For F one of zeta, L(s,(n/5)), Davenport–Heilbronn, g = f * f~ (f~(u) = f(-u)), alpha = (1/2 + e)/2:

    Q(f) = [2 F_L(1/2) F_L(-1/2)]_{zeta only}                       F_L(s) = int f(u) e^{su} du
           - 2 sum_{2 <= n < e^{2a}} a_n n^{-1/2} g(log n)            a_n: coefficients of -F'/F
           + (log(q/pi) - gamma_E - log(1 - e^{-4a})) g(0)
           + 2 int_0^{2a} I(u) du,      I(u) = [g(0) e^{-2u} - g(u) e^{-2 alpha u}] / (1 - e^{-2u}).

By the explicit formula Q(f) = sum_rho f^(tau_rho) f^(-tau_rho) (tau_rho = (rho - 1/2)/i).  A zero
on the critical line has real tau and contributes |f^(tau)|^2 >= 0, whatever its multiplicity.
So a certified Q(f) < 0 PROVES that F has a zero off the critical line, using only the a_n with
n < e^{2a} and the Gamma factor.

How each piece is enclosed
--------------------------
* Closed form of g.  With x = u/a and w = r0 a, f(a x) = (p(x) e^{iwx} + p*(x) e^{-iwx}) / 2,
  where p = Psi_cos - i Psi_sin is a polynomial (Psi_* = sum c_i (1 - x^2)^m P_j(x) over the cos
  or sin members; Legendre basis: p = sum c_i P_j, w = 0) and p* has the conjugate coefficients.
  Then for y = u/a in [0, 2]
      g(a y) = (a/4) [ e^{iwy} A(y) + e^{-iwy} A*(y) + e^{-iwy} B(y) + e^{iwy} B*(y) ],
      A(y) = int_{y-1}^{1} p(x) p*(x - y) dx,     B(y) = int_{y-1}^{1} p(x) p(x - y) e^{2iwx} dx,
  and A*, B* the same with p <-> p*, w -> -w.  The polynomial integrals are exact
  (acb_poly composition and antiderivative); B uses the finite antiderivative
  e^{kx} sum_k (-1)^k q^{(k)}(x) / k^{k+1}.  The expression is analytic in y (no Re), as
  acb.integral requires, and is real for real y.
* a_n: `log_derivative_coefficients_arb` (rigorous balls).
* The integral over [eps, 2a]: `acb.integral` (Petras' algorithm with rigorous error bounds;
  the integrand is meromorphic, its poles are at u = i pi k, k != 0).  On [0, eps] the integrand
  has a removable singularity; by the mean value theorem |I(u)| <= sup|N'| / (2 e^{-2 eps}) with
  N(u) = g(0) e^{-2u} - g(u) e^{-2 alpha u} and |N'| <= (2 + 2 alpha) g(0) + sup|g'|, where
  |g| <= g(0) and |g'| <= ||f||_2 ||f'||_2 + (|f(a-)| + |f(-a+)|) sup|f| (Cauchy–Schwarz, plus the
  jump terms of an f that does not vanish at +-a), so the piece is enclosed by a ball.
* F_L(+-1/2) (zeta): exact antiderivatives of polynomial x exponential.

The enclosure inherits rigour from Arb.  It assumes the explicit formula for the series, which
is a theorem (contour shift; derivation in `rh.explicit_formula_general`; Weil 1952), and the
exact coefficients of the series: kappa for DH is the explicit algebraic number of
`rh.explicit_formula_general`, enclosed by Arb.
"""
from __future__ import annotations

import math
import time
from fractions import Fraction
from typing import Optional, Sequence

from .explicit_formula_general import log_derivative_coefficients_arb
from .weil_form_general import Basis, _flint_dps, get_series

__all__ = ["test_polynomial", "CorrelationEvaluator", "certify_weil_value", "rationalize", "guard_digits"]


def rationalize(coeffs: Sequence, max_den: Optional[int] = None) -> list:
    """Exact rationals for (float / mpmath) coefficients: the binary value of float(c) (exact), or
    the best approximation with denominator <= max_den."""
    out = []
    for c in coeffs:
        fr = Fraction(float(c))
        out.append(fr.limit_denominator(max_den) if max_den else fr)
    return out


def _fmpq(x):
    from flint import fmpq
    x = Fraction(x)
    return fmpq(x.numerator, x.denominator)


def test_polynomial(basis: Basis, parity: str, coeffs: Sequence) -> tuple:
    """(Psi_cos, Psi_sin) as exact fmpq_poly in x = u/a, with f(a x) = Psi_cos(x) cos(w x) +
    Psi_sin(x) sin(w x), w = r0 a (Legendre basis: Psi_sin = 0 and w = 0)."""
    from flint import fmpq_poly
    labels = basis.labels(parity)
    if len(coeffs) != len(labels):
        raise ValueError("one coefficient per basis function of the sector")
    win = fmpq_poly([1, 0, -1]) ** basis.m if basis.kind == "localized" else fmpq_poly([1])
    pc, ps = fmpq_poly([0]), fmpq_poly([0])
    for (j, trig), c in zip(labels, coeffs):
        term = win * fmpq_poly.legendre_p(j) * _fmpq(c)
        if trig == "sin":
            ps += term
        else:
            pc += term
    return pc, ps


def _acb_poly_from(pc, ps, sign: int):
    """p = Psi_cos - i sign Psi_sin as an acb_poly (sign = +1: p, -1: p*)."""
    from flint import acb, acb_poly, arb
    n = max(pc.length(), ps.length())
    cc = [pc[k] if k < pc.length() else 0 for k in range(n)]
    ss = [ps[k] if k < ps.length() else 0 for k in range(n)]
    return acb_poly([acb(arb(c), -sign * arb(s)) for c, s in zip(cc, ss)])


def _int_poly_exp(poly, kappa, lo, hi):
    """int_lo^hi poly(x) e^{kappa x} dx exactly (acb): antiderivative
    e^{kappa x} sum_k (-1)^k poly^{(k)}(x) / kappa^{k+1}; kappa = 0 uses poly.integral()."""
    from flint import acb
    if kappa == 0:
        P = poly.integral()
        return P(hi) - P(lo)
    S, d, ck = None, poly, 1 / kappa
    for k in range(poly.degree() + 1):
        term = d * (ck if k % 2 == 0 else -ck)
        S = term if S is None else S + term
        d = d.derivative()
        ck = ck / kappa
    if S is None:
        return acb(0)
    return (kappa * hi).exp() * S(hi) - (kappa * lo).exp() * S(lo)


class CorrelationEvaluator:
    """g(u) = (f * f~)(u) in closed form, analytic in u, for u = a y with y in [0, 2] (and complex
    balls near that segment).  Built once per (basis, parity, coefficients) at the current flint
    precision."""

    def __init__(self, basis: Basis, parity: str, coeffs: Sequence, pc=None, ps=None):
        from flint import acb, arb
        self.basis, self.parity = basis, parity
        if pc is None:
            pc, ps = test_polynomial(basis, parity, coeffs)
        self.pc, self.ps = pc, ps
        self.p = _acb_poly_from(pc, ps, +1)
        self.pstar = _acb_poly_from(pc, ps, -1)
        # a and r0 are floats in Basis: their exact binary values define the test function (arb
        # holds a double exactly, and the product of two doubles exactly at >= 106 bits)
        self.a = arb(basis.a)
        self.w = self.a * arb(basis.r0)
        self.iw = acb(0, 1) * self.w

    def _shifted(self, poly, y):
        from flint import acb_poly
        return poly(acb_poly([-y, 1]))                       # poly(x - y)

    def g_over_a(self, y):
        """g(a y) / a (acb), y an acb ball."""
        from flint import acb
        y = acb(y)
        one = acb(1)
        lo = y - 1
        p, ps = self.p, self.pstar
        A = _int_poly_exp(p * self._shifted(ps, y), 0, lo, one)
        As = _int_poly_exp(ps * self._shifted(p, y), 0, lo, one)
        if self.w == 0:
            B = _int_poly_exp(p * self._shifted(p, y), 0, lo, one)
            Bs = _int_poly_exp(ps * self._shifted(ps, y), 0, lo, one)
        else:
            B = _int_poly_exp(p * self._shifted(p, y), 2 * self.iw, lo, one)
            Bs = _int_poly_exp(ps * self._shifted(ps, y), -2 * self.iw, lo, one)
        e = (self.iw * y).exp()
        return (e * (A + Bs) + (As + B) / e) / 4

    def g(self, u):
        """g(u) (acb) for u = a y."""
        return self.g_over_a(u / self.a) * self.a

    def norm2(self):
        """||f||_2^2 = g(0) (arb)."""
        return self.g_over_a(0).real * self.a

    def derivative_evaluator(self) -> "CorrelationEvaluator":
        """The same evaluator for f' (f'(a x) = ((p' + i w p) e^{iwx} + c.c.) / (2a)), used for the
        bound ||f'||_2 on the removable-singularity interval."""
        from flint import fmpq_poly
        ev = CorrelationEvaluator.__new__(CorrelationEvaluator)
        ev.basis, ev.parity = self.basis, self.parity
        ev.pc, ev.ps = None, None
        ev.a, ev.w, ev.iw = self.a, self.w, self.iw
        ev.p = (self.p.derivative() + self.p * self.iw) * (1 / self.a)
        ev.pstar = (self.pstar.derivative() - self.pstar * self.iw) * (1 / self.a)
        return ev

    def boundary_values_abs_sum(self):
        """|f(a-)| + |f(-a+)| (arb upper bound), f(a x) = (p(x) e^{iwx} + p*(x) e^{-iwx}) / 2 at x = +-1;
        exactly 0 for a window of order m >= 1."""
        from flint import acb, arb
        tot = arb(0)
        for x in (acb(1), acb(-1)):
            v = (self.p(x) * (self.iw * x).exp() + self.pstar(x) * (-self.iw * x).exp()) / 2
            tot += abs(v)
        return tot

    def sup_norm_bound(self):
        """Upper bound for sup |f| on [-a, a]: sum of the moduli of the coefficients of p (|x| <= 1)."""
        from flint import arb
        tot = arb(0)
        for c in self.p.coeffs():
            tot += abs(c)
        return tot

    def laplace(self, s):
        """F_L(s) = int f(u) e^{su} du = a int_{-1}^{1} f(a x) e^{s a x} dx (acb)."""
        from flint import acb
        one, mone = acb(1), acb(-1)
        k1 = self.iw + s * self.a
        k2 = -self.iw + s * self.a
        return self.a * (_int_poly_exp(self.p, k1, mone, one) + _int_poly_exp(self.pstar, k2, mone, one)) / 2


def guard_digits(basis: Basis) -> int:
    """Extra working digits for the monomial-basis polynomial algebra: composing a polynomial of
    degree 2D (D = basis degree) with x - y, |y| <= 2, inflates coefficients by up to 3^{2D}, and
    the Legendre-to-monomial change of basis by ~2^{2D}."""
    return int(math.ceil(2 * basis.degree * math.log10(6.0))) + 20


def certify_weil_value(series, basis: Basis, parity: str, coeffs: Sequence, dps: int = 30,
                       eps: float = 1e-12, verbose: bool = False, max_attempts: int = 4) -> dict:
    """Certified enclosure of the arithmetic side Q(f) of Weil's form for f = sum coeffs[i] phi_i
    (coeffs exact rationals, or floats taken at their exact binary value).

    dps is the target accuracy (integration tolerance 10^-dps); the working precision adds
    `guard_digits(basis)`.  The enclosure is rigorous at any setting: too little precision only
    widens it.  If the pieces come out wider than ~10^-(dps-5) relative to ||f||^2 (the monomial
    algebra can cancel more than the guard allows, e.g. the pole integral int p(x) e^{kx} dx with
    small k and high degree), the evaluation is repeated with 1.6x the guard digits.
    Returns {'Q': arb, 'norm2': arb (= ||f||^2), 'ratio': arb (Q / ||f||^2), 'upper', 'lower',
    'negative' (True iff the whole enclosure is < 0), 'pieces': {...}, 'seconds', 'working_dps'}."""
    from flint import arb
    guard = guard_digits(basis)
    t0 = time.time()
    out = None
    for _ in range(max_attempts):
        out = _certify_once(series, basis, parity, coeffs, dps, eps, guard)
        width = max(float(x.rad()) for x in out["pieces"].values() if hasattr(x, "rad"))
        if width <= 10.0 ** (-(dps - 5)) * max(1.0, float(abs(out["norm2"]).upper())):
            break
        guard = int(guard * 1.6) + 20
    out["seconds"] = time.time() - t0
    return out


def _certify_once(series, basis: Basis, parity: str, coeffs: Sequence, dps: int, eps: float, guard: int) -> dict:
    from flint import acb, arb, ctx
    ser = get_series(series)
    coeffs = [Fraction(c) if isinstance(c, (int, Fraction)) else Fraction(float(c)) for c in coeffs]
    t0 = time.time()
    work = int(dps) + int(guard)
    with _flint_dps(work):
        ev = CorrelationEvaluator(basis, parity, coeffs)
        A = ev.a
        g0 = ev.norm2()
        alpha = arb(1 + 2 * ser.e) / 4
        # ---- constant term
        const = (arb(ser.q) / arb.pi()).log() - arb.const_euler() - (1 - (-4 * A).exp()).log()
        piece_const = const * g0
        # ---- pole term (zeta)
        piece_pole = arb(0)
        if ser.pole:
            piece_pole = (2 * ev.laplace(acb(0.5)) * ev.laplace(acb(-0.5))).real
        # ---- prime-type terms over n < e^{2a}
        nmax = int(math.floor(math.exp(2 * basis.a))) + 1
        an = log_derivative_coefficients_arb(ser, nmax)
        piece_prime = arb(0)
        used = []
        for n in range(2, nmax + 1):
            x = an[n]
            if x.is_zero():
                continue
            y = arb(n).log() / A
            if y >= 2:                          # g(log n) = 0 (support of g is [-2a, 2a])
                continue
            if not (y < 2):
                raise ArithmeticError(f"cannot decide whether log({n}) < 2a at this precision")
            gv = ev.g_over_a(acb(y)).real * A
            piece_prime -= 2 * x / arb(n).sqrt() * gv
            used.append(n)
        # ---- archimedean integral: 2 int_0^{2a} I(u) du = 2 a int_0^2 I(a y) dy
        e = arb(eps)

        def integrand(y, analytic):
            u = A * y
            num = g0 * (-2 * u).exp() - ev.g_over_a(y) * A * (-2 * alpha * u).exp()
            return num / (1 - (-2 * u).exp())

        t1 = time.time()
        tol = arb(10) ** (-int(dps))
        J = acb.integral(integrand, e / A, acb(2), rel_tol=tol, abs_tol=tol)
        main = 2 * A * J.real
        # [0, eps]: |I| <= sup|N'| / (2 e^{-2 eps}), |N'| <= (2 + 2 alpha) g0 + sup|g'|, and for f smooth on
        # (-a, a) with one-sided boundary values f(+-a) (nonzero only without a window, m = 0),
        # g'(u) = -int f(v) f'(v - u) dv - f(-a+) f(u - a) + f(a-) f(u + a), so
        # |g'| <= ||f||_2 ||f'||_2 + (|f(a-)| + |f(-a+)|) ||f||_inf  (f' the derivative inside (-a, a)).
        fp_norm2 = ev.derivative_evaluator().norm2()
        jump = ev.boundary_values_abs_sum() * ev.sup_norm_bound()
        gprime = (g0 * fp_norm2).sqrt() + jump
        bound = ((2 + 2 * alpha) * g0 + gprime) / (2 * (-2 * e).exp())
        rad = 2 * e * bound                                   # factor 2 of the formula
        piece_arch = main + arb(0, float(rad.upper()) * (1 + 1e-12))
        Q = piece_pole + piece_prime + piece_const + piece_arch
        ratio = Q / g0
        out = {
            "Q": Q, "norm2": g0, "ratio": ratio,
            "upper": float(Q.upper()), "lower": float(Q.lower()),
            "ratio_upper": float(ratio.upper()), "ratio_lower": float(ratio.lower()),
            "negative": bool(Q < 0), "positive": bool(Q > 0),
            "pieces": {"pole": piece_pole, "prime": piece_prime, "constant": piece_const,
                       "archimedean": piece_arch, "eps_interval_bound": rad},
            "terms": used, "dps": dps, "working_dps": work, "prec_bits": ctx.prec, "eps": eps,
            "integration_seconds": time.time() - t1, "seconds": time.time() - t0,
        }
    return out
