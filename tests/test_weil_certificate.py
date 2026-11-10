"""Tests for rh.weil_certificate: rigorous enclosures of Weil's form at one explicit test function.

The certified enclosure is computed by a different method (closed-form correlation, acb.integral
with rigorous error bounds) from the matrices of rh.weil_form_general (Gauss–Legendre, estimated
errors), so agreement between the two is a real check of both.
"""
import mpmath as mp
import numpy as np
import pytest
from flint import acb, arb

from rh import weil_certificate as wc
from rh import weil_form_general as wg


def _matrix_value(S, c, dps=40):
    with mp.workdps(dps):
        Qm = S.matrix_mp("Q")
        cm = [mp.mpf(x.numerator) / x.denominator for x in c]
        return mp.fsum(cm[i] * Qm[i, j] * cm[j] for i in range(S.size) for j in range(S.size))


@pytest.mark.parametrize("series,basis,parity", [
    ("zeta", wg.Basis.legendre(0.8, 6), "even"),
    ("zeta", wg.Basis.localized(0.9, 4, 12.0, 3), "odd"),          # pole term with oscillating f
    ("dh", wg.Basis.localized(1.0, 5, 10.0, 4), "even"),
    ("chi5", wg.Basis.legendre(1.1, 7), "odd"),
])
def test_certified_enclosure_contains_matrix_value(series, basis, parity):
    S = wg.build_sector(series, basis, parity, dps=40)
    c = wc.rationalize(np.random.default_rng(5).standard_normal(S.size))
    val = _matrix_value(S, c)
    r = wc.certify_weil_value(series, basis, parity, c, dps=20)
    assert r["lower"] - 1e-25 <= float(val) <= r["upper"] + 1e-25
    assert r["upper"] - r["lower"] < 1e-9
    assert r["positive"] == (float(val) > 0)


def test_closed_form_correlation_matches_matrix_correlation():
    from rh.weil_form_general import _correlation, _flint_dps, default_nodes
    B = wg.Basis.localized(1.3, 6, 40.0, 3)
    c = wc.rationalize(np.random.default_rng(2).standard_normal(6))
    labels = B.labels("odd")
    nc = default_nodes(B, 40)["n_corr"]
    with _flint_dps(40):
        ev = wc.CorrelationEvaluator(B, "odd", c)
        cm = [arb(x.numerator) / arb(x.denominator) for x in c]
        for u in (0.0, 0.41, 1.7, 2.55):
            sm = _correlation(B, labels, arb(u) / arb(B.a), nc)
            ref = sum(cm[i] * cm[j] * sm[i, j] for i in range(6) for j in range(6))
            g = ev.g(acb(u))
            assert abs(float((g.real - ref).mid())) < 1e-28 * max(1.0, abs(float(ref.mid())))
            assert abs(float(g.imag.mid())) < 1e-28
        # g(0) = ||f||^2 against quadrature of the basis functions
        x, w = np.polynomial.legendre.leggauss(400)
        f = wg.basis_values(B, "odd", B.a * x) @ np.array([float(t) for t in c])
        assert abs(float(ev.norm2().mid()) - float(np.sum(w * B.a * f * f))) < 1e-12


def test_laplace_transform_matches_pole_vectors():
    """F_L(+-1/2) of the closed form against the Gauss–Legendre pole vectors of the matrix module."""
    from rh.weil_form_general import _flint_dps, _pole_vectors
    B = wg.Basis.localized(0.9, 5, 12.0, 2)
    c = wc.rationalize(np.random.default_rng(4).standard_normal(5))
    with _flint_dps(40):
        ev = wc.CorrelationEvaluator(B, "even", c)
        vp, vm = _pole_vectors(B, B.labels("even"), 120)
        cm = [arb(x.numerator) / arb(x.denominator) for x in c]
        Fp = sum(ci * v for ci, v in zip(cm, vp))
        Fm = sum(ci * v for ci, v in zip(cm, vm))
        assert abs(float((ev.laplace(acb(0.5)).real - Fp).mid())) < 1e-30
        assert abs(float((ev.laplace(acb(-0.5)).real - Fm).mid())) < 1e-30


def test_certificate_davenport_heilbronn_off_line_zero():
    """Machine-checked: Q_DH(f) < 0 for an explicit f supported in [-2, 2] (12 wave packets at height
    85.7, window order 2).  By the explicit formula, DH has a zero off the critical line; only its
    coefficients a_n with n <= 54 and its Gamma factor enter.  (Measured enclosure
    [-0.3605718118, -0.3605718116] with ||f|| = 1.)"""
    B = wg.Basis.localized(2.0, 12, 85.7, 2)
    S = wg.build_sector("dh", B, "even", dps=30)
    vals, vecs = wg.smallest_eigenpairs(S, k=1)
    c = wc.rationalize(vecs[0])
    r = wc.certify_weil_value("dh", B, "even", c, dps=10)
    assert r["negative"] and r["upper"] < -0.3
    assert abs(float(r["norm2"].mid()) - 1) < 1e-10
    assert max(r["terms"]) == 54
    # the same f for L(s,(n/5)) (GRH): positive
    r5 = wc.certify_weil_value("chi5", B, "even", c, dps=8)
    assert r5["positive"]


def test_rationalize_with_bits_is_exact_rounding():
    import mpmath as mp
    from fractions import Fraction
    with mp.workdps(120):
        x = mp.mpf(1) / 3
        r = wc.rationalize([x, -x * mp.mpf(10) ** -30], bits=300)
        assert r[0].denominator == 2 ** 300 or (2 ** 300) % r[0].denominator == 0
        assert abs(mp.mpf(r[0].numerator) / r[0].denominator - x) <= mp.mpf(2) ** -301
        assert abs(mp.mpf(r[1].numerator) / r[1].denominator + x * mp.mpf(10) ** -30) <= mp.mpf(2) ** -301
    assert wc.rationalize([0.5]) == [Fraction(1, 2)]                 # default behaviour unchanged
