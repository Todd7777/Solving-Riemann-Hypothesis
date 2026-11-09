"""Tests for rh.weil_form_general: Weil's quadratic form for zeta, L(s,(n/5)) and the
Davenport–Heilbronn function on test functions supported in [-a, a].

Every check is against something independent of the code path being tested: the older
`rh.weil_form` (zeta, Legendre), the zero side computed from the repository's zero lists
(explicit formula), Gauss–Legendre quadrature in Arb, exact rational arithmetic, or the
structure the theory predicts (rank and inertia of an off-line quadruple).
"""
import math
from fractions import Fraction

import mpmath as mp
import numpy as np
import pytest

from rh import weil_form as wf
from rh import weil_form_general as wg


@pytest.fixture(scope="module")
def zero_data():
    out = {s: wg.load_zero_data(s, n_zeta=200000) for s in ("zeta", "chi5", "dh")}
    return out


def _float_gram(B, parity, nodes=2000):
    x, w = np.polynomial.legendre.leggauss(nodes)
    V = wg.basis_values(B, parity, B.a * x)
    return (V * (w * B.a)[:, None]).T @ V


# ----------------------------------------------------------------------------
# bases and exact helpers
# ----------------------------------------------------------------------------
def _legendre_exact(k, x):
    """P_k(x) at a rational x, exactly (three-term recurrence in Fractions)."""
    p0, p1 = Fraction(1), x
    if k == 0:
        return p0
    for n in range(1, k):
        p0, p1 = p1, ((2 * n + 1) * x * p1 - n * p0) / (n + 1)
    return p1


@pytest.mark.parametrize("m,j", [(0, 3), (2, 0), (3, 5), (8, 7)])
def test_window_legendre_expansion_is_exact(m, j):
    b = wg._window_legendre(m, j)
    assert len(b) == j + 2 * m + 1
    for x in (Fraction(0), Fraction(1, 3), Fraction(-5, 7), Fraction(9, 10)):
        lhs = (1 - x * x) ** m * _legendre_exact(j, x)
        rhs = sum(bk * _legendre_exact(k, x) for k, bk in enumerate(b))
        assert lhs == rhs


def test_basis_labels_and_parity():
    B = wg.Basis.localized(1.0, 5, 20.0, 2)
    assert B.labels("even") == ((0, "cos"), (1, "sin"), (2, "cos"), (3, "sin"), (4, "cos"))
    assert B.labels("odd") == ((0, "sin"), (1, "cos"), (2, "sin"), (3, "cos"), (4, "sin"))
    u = np.linspace(-0.9, 0.9, 7)
    for par, sg in (("even", 1), ("odd", -1)):
        V, Vm = wg.basis_values(B, par, u), wg.basis_values(B, par, -u)
        assert np.allclose(Vm, sg * V, atol=1e-14)
    assert np.all(wg.basis_values(B, "even", np.array([1.2, -1.5])) == 0)
    with pytest.raises(ValueError):
        wg.Basis("legendre", 1.0, 4, r0=3.0)


# ----------------------------------------------------------------------------
# assembly
# ----------------------------------------------------------------------------
def test_zeta_legendre_reproduces_weil_form():
    """The zeta / Legendre case must agree with the separately tested rh.weil_form."""
    a, J, dps = 1.0, 12, 40
    W = wf.build_weil_form(a, J, dps=dps)
    for par in ("even", "odd"):
        S = wg.build_sector("zeta", wg.Basis.legendre(a, J), par, dps=dps)
        Qs, Gs, idx = W.sector(par)
        n = len(idx)
        diff = max(abs(float((Qs[i, j] - S.Q[i, j]).mid())) for i in range(n) for j in range(n))
        assert diff < 1e-35
        v_old, _ = wf.smallest_eigenpairs(Qs, Gs, dps, k=1)
        v_new, x = wg.smallest_eigenpairs(S, k=1)
        assert abs(v_old[0] - v_new[0]) < 1e-30 * abs(v_old[0]) + 1e-38
        # normalisation x^T G x = 1
        with mp.workdps(dps):
            Gm = S.matrix_mp("G")
            nrm = mp.fsum(x[0][i] * Gm[i, j] * x[0][j] for i in range(n) for j in range(n))
            assert abs(nrm - 1) < mp.mpf(10) ** (-30)


def test_localized_gram_matches_float_quadrature():
    B = wg.Basis.localized(1.2, 6, 30.0, 4)
    for par in ("even", "odd"):
        S = wg.build_sector("chi5", B, par, dps=30, estimate_errors=False)
        assert np.allclose(S.matrix_float("G"), _float_gram(B, par), rtol=0, atol=1e-13)


@pytest.mark.parametrize("series", ["zeta", "chi5", "dh"])
def test_arithmetic_side_equals_zero_side(series, zero_data):
    """Explicit formula: the matrix built from the a_n (n <= e^{2a}) and the Gamma factor equals
    the sum over the zeros (+ density tail).  A window of order 6 makes f^ decay like r^-7, so the
    tail beyond the known zeros is negligible; measured residuals 1e-16 (zeta, chi5) and 1e-13 (dh,
    limited by the accuracy of its zero list)."""
    B = wg.Basis.localized(1.2, 6, 30.0, 6)
    z = zero_data[series]
    T = float(z["on_line"][-1]) if series == "zeta" else z["T"]
    rng = np.random.default_rng(11)
    for par in ("even", "odd"):
        S = wg.build_sector(series, B, par, dps=30)
        c = rng.standard_normal(S.size)
        arith = wg.quadratic_value(S, c)
        res = wg.zero_side(B, par, c, on_line=z["on_line"], off_line=z["off_line"], series=series, tail=True,
                           T=T, s_correction=True)
        assert abs(arith - res["total"]) < 1e-11 * res["on_line"]
        # the matrix form of the zero side agrees with the direct evaluation
        Z = wg.zero_side_matrix(B, par, on_line=z["on_line"], off_line=z["off_line"], series=series, tail=True,
                                T=T, s_correction=True)["Z"]
        assert abs(float(c @ Z @ c) - res["total"]) < 1e-12 * res["on_line"]


def test_dh_off_line_zeros_are_needed(zero_data):
    """For a wave packet at the first off-line zero of Davenport–Heilbronn (0.8085 + 85.699 i) the
    off-line quadruples contribute ~0.16 to the zero side; the identity holds to ~1e-13 with them and
    fails by their contribution without them."""
    B = wg.Basis.localized(1.5, 6, 85.7, 6)
    z = zero_data["dh"]
    rng = np.random.default_rng(1)
    S = wg.build_sector("dh", B, "even", dps=30)
    c = rng.standard_normal(S.size)
    arith = wg.quadratic_value(S, c)
    full = wg.zero_side(B, "even", c, on_line=z["on_line"], off_line=z["off_line"], series="dh", tail=True,
                        T=z["T"], s_correction=True)
    on_only = wg.zero_side(B, "even", c, on_line=z["on_line"], off_line=(), series="dh", tail=True,
                           T=z["T"], s_correction=True, n_below_T=len(z["on_line"]) + 2 * len(z["off_line"]))
    assert abs(full["off_line"]) > 1e-3
    assert abs(arith - full["total"]) < 1e-10 * abs(full["off_line"])
    assert abs(arith - on_only["total"] - full["off_line"]) < 1e-10 * abs(full["off_line"])


def test_dh_prime_side_uses_non_prime_powers():
    """DH has no Euler product: its a_n with n = 6, 10, 12, ... are nonzero and enter the form."""
    B = wg.Basis.legendre(1.0, 6)
    S_dh = wg.build_sector("dh", B, "even", dps=30)
    S_z = wg.build_sector("zeta", B, "even", dps=30)
    assert 6 in S_dh.terms and 6 not in S_z.terms
    assert S_z.terms == (2, 3, 4, 5, 7)


# ----------------------------------------------------------------------------
# Fourier transforms
# ----------------------------------------------------------------------------
@pytest.mark.parametrize("basis", [wg.Basis.legendre(1.3, 10), wg.Basis.localized(1.1, 7, 25.0, 3)])
def test_fourier_closed_form_against_arb_quadrature(basis):
    for par in ("even", "odd"):
        for z in (0.7, 14.1347, 31.0 - 0.3j, 85.7 - 0.31j):
            closed = wg.fourier_vector_mp(basis, par, z, dps=30)
            quad = wg.fourier_transform_arb(basis, par, z, dps=30)
            flt = wg.fourier_matrix(basis, par, np.array([z]))[0]
            scale = max(abs(complex(x)) for x in closed)
            for c_, q_, f_ in zip(closed, quad, flt):
                q = complex(float(q_.real.mid()), float(q_.imag.mid()))
                assert abs(complex(c_) - q) < 1e-25 * scale
                assert abs(complex(c_) - f_) < 1e-12 * scale
            # parity: phi^(-z) = sgn phi^(z)
            sg = 1 if par == "even" else -1
            minus = wg.fourier_vector_mp(basis, par, -z, dps=30)
            assert max(abs(complex(m_) - sg * complex(c_)) for m_, c_ in zip(minus, closed)) < 1e-25 * scale


@pytest.mark.parametrize("basis", [wg.Basis.legendre(1.3, 12), wg.Basis.legendre(1.4, 160),
                                   wg.Basis.localized(2.0, 20, 85.7, 4)])
def test_far_field_expansion(basis):
    """phi^(z) = e^{iaz} A(z) + e^{-iaz} B(z) exactly; checked against the closed form where the
    expansion is used.  The degree-159 case overflowed float64 before the derivative tables were
    stored scaled (regression test)."""
    for par in ("even", "odd"):
        K = wg._sector_tables(basis, par)["K"]
        T1 = basis.r0 + K * K / basis.a + 50.0
        z = np.array([T1, 2.5 * T1 + 0.4j])
        A, Bc = wg.far_field(basis, par, z)
        ff = np.exp(1j * basis.a * z)[:, None] * A + np.exp(-1j * basis.a * z)[:, None] * Bc
        ref = np.array([[complex(x) for x in wg.fourier_vector_mp(basis, par, complex(zz), 30)] for zz in z])
        assert np.max(np.abs(ff - ref)) < 1e-10 * np.max(np.abs(ref))


def test_boundary_vanishing_subspace():
    B = wg.Basis.legendre(1.3, 24)
    from numpy.polynomial import legendre as npleg
    for par in ("even", "odd"):
        Nb = wg.boundary_vanishing_subspace(B, par, 4)
        assert Nb.shape == (12, 8)
        assert np.allclose(Nb.T @ Nb, np.eye(8), atol=1e-12)
        js = [j for j, _ in B.labels(par)]
        for col in Nb.T:
            coef = np.zeros(24)
            coef[js] = col
            for l in range(4):
                d = npleg.legder(coef, l) if l else coef
                scale = np.sum(np.abs(d)) * 24 ** (2 * l)      # |P_k^{(l)}(+-1)| <= k^{2l}
                assert abs(npleg.legval(1.0, d)) < 1e-11 * scale and abs(npleg.legval(-1.0, d)) < 1e-11 * scale
        # f^ = O(r^-5): |f^(r)| r^5 stays bounded along a geometric sequence of r
        c = Nb[:, 0]
        r = np.array([400.0, 800.0, 1600.0, 3200.0])
        f = np.abs(wg.fourier_transform(B, par, c, r))
        env = [np.max(np.abs(wg.fourier_transform(B, par, c, rr + np.linspace(0, 3, 40)))) * rr ** 5 for rr in r]
        assert max(env) < 4 * min(env) and np.all(np.isfinite(f))


# ----------------------------------------------------------------------------
# zero density and the thought-experiment quadruple
# ----------------------------------------------------------------------------
@pytest.mark.parametrize("series", ["zeta", "dh"])
def test_zero_density_is_theta_prime_over_pi(series):
    from rh.explicit_formula_general import hardy_theta
    ser = wg.get_series(series)
    for T in (30.0, 300.0):
        h = 1e-4
        d_num = (float(hardy_theta(ser, T + h, 30).mid()) - float(hardy_theta(ser, T - h, 30).mid())) / (2 * h) / math.pi
        assert abs(float(np.real(wg.zero_density(series, np.array([T]))[0])) - d_num) < 1e-8


def test_quadruple_matrix_rank_inertia_and_value():
    """An off-line quadruple adds the rank-2 indefinite form 4 sgn Re f^(tau0)^2; an on-line pair adds
    the rank-1 positive form 2 |f^(gamma0)|^2."""
    B = wg.Basis.localized(1.5, 8, 60.0, 4)
    rng = np.random.default_rng(3)
    for par, sg in (("even", 1), ("odd", -1)):
        M = wg.quadruple_matrix(B, par, 60.3, 0.2, dps=30)
        Mf = np.array([[float(M[i, j].mid()) for j in range(M.ncols())] for i in range(M.nrows())])
        ev = np.linalg.eigvalsh(Mf)
        big = np.abs(ev) > 1e-10 * np.max(np.abs(ev))
        assert big.sum() == 2 and ev[big].min() < 0 < ev[big].max()
        c = rng.standard_normal(Mf.shape[0])
        val = wg.zero_side(B, par, c, off_line=[complex(0.7, 60.3)])["off_line"]
        assert abs(float(c @ Mf @ c) - val) < 1e-12 * max(1.0, abs(val))
        P = wg.quadruple_matrix(B, par, 60.3, 0.0, dps=30)
        Pf = np.array([[float(P[i, j].mid()) for j in range(P.ncols())] for i in range(P.nrows())])
        evp = np.linalg.eigvalsh(Pf)
        assert evp.min() > -1e-12 * evp.max() and (evp > 1e-10 * evp.max()).sum() == 1
        on = wg.zero_side(B, par, c, on_line=[60.3])["on_line"]
        assert abs(float(c @ Pf @ c) - on) < 1e-12 * max(1.0, on)


# ----------------------------------------------------------------------------
# the E12 result: Weil positivity fails for Davenport–Heilbronn at support a = 2
# ----------------------------------------------------------------------------
def test_weil_form_detects_davenport_heilbronn_violation(zero_data):
    """Support [-2, 2]: only the coefficients a_n, n <= 54, and the Gamma factor enter.  On 16 wave
    packets at height 85.7 the DH form has a negative eigenvalue (measured -0.29 with 20 packets),
    far outside the estimated numerical error, while the GRH-true L(s,(n/5)) stays positive on the
    same space.  The zero side of the DH minimiser shows why: its first off-line zero
    0.8085 + 85.699 i contributes a negative amount larger than all on-line zeros together."""
    B = wg.Basis.localized(2.0, 16, 85.7, 4)
    S = wg.build_sector("dh", B, "even", dps=30)
    vals, vecs = wg.smallest_eigenpairs(S, k=1)
    eb = wg.eigenvalue_error_bound(S, vals[0])
    assert vals[0] < -0.1 and eb["sign_certified"]
    c = np.array([float(x) for x in vecs[0]])
    z = zero_data["dh"]
    zs = wg.zero_side(B, "even", c, on_line=z["on_line"], off_line=z["off_line"], series="dh", tail=True,
                      T=z["T"], s_correction=True)
    assert abs(zs["total"] - float(vals[0])) < 1e-9
    first = zs["off_line_each"][0]
    assert abs(first["rho"][1] - 85.699) < 1e-3 and first["contribution"] < -zs["on_line"]
    S5 = wg.build_sector("chi5", B, "even", dps=30)
    v5, _ = wg.smallest_eigenpairs(S5, k=1)
    assert v5[0] > 0.3
