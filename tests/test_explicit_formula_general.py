"""Tests for rh.explicit_formula_general: Weil's explicit formula for self-dual
Dirichlet series with periodic coefficients (zeta, L(s,(n/5)), Davenport–Heilbronn)."""
import math

import numpy as np
import pytest
import sympy as sp

import rh.explicit_formula as ef
import rh.explicit_formula_general as G
from rh.data import load_zeros

DH = G.DAVENPORT_HEILBRONN
KAPPA = (math.sqrt(10 - 2 * math.sqrt(5)) - 2) / (math.sqrt(5) - 1)


@pytest.fixture(scope="module")
def dh_zeros():
    return G.load_dh_zeros()


@pytest.fixture(scope="module")
def zeta_zeros():
    return load_zeros(100000)


@pytest.fixture(scope="module")
def chi5_zeros():
    # all zeros of L(s,(n/5)) with 0 < t < 100: 54 sign changes of Z, and N(100) = 54
    return G.on_line_zeros(G.CHI5, 0.0, 100.0, step=0.05)


def _tf(s, w):
    return ef.modulated_gaussian(s, w) if w else ef.gaussian(s)


# ----------------------------------------------------------------------------
# the series and their functional equations
# ----------------------------------------------------------------------------
def test_series_definitions_and_validation():
    assert (G.ZETA.q, G.ZETA.e, G.ZETA.pole) == (1, 0, True)
    assert (G.CHI5.q, G.CHI5.e, G.CHI5.alpha) == (5, 0, 0.25)
    assert (DH.q, DH.e, DH.alpha) == (5, 1, 0.75)
    assert abs(DH.coeffs[1] - KAPPA) < 1e-16
    assert G.coefficient_parity((1, -1, -1, 1, 0)) == 0 and G.coefficient_parity((1, 0.3, -0.3, -1, 0)) == 1
    assert G.coefficient_parity((1, 2, -3, 0, 0)) is None
    with pytest.raises(ValueError):
        G.SelfDualSeries("bad parity", (1, -1, -1, 1, 0), e=1)
    with pytest.raises(ValueError):
        G.SelfDualSeries("pole", (1, 1, 1, 1, 1), e=0)          # sum over a period != 0
    with pytest.raises(ValueError):
        G.SelfDualSeries("c1", (2, -2), e=0)
    # wrapping rh.nyman_beurling_periodic reproduces the parities
    from rh import nyman_beurling_periodic as nbp
    assert G.from_periodic_series(nbp.CHI5).e == 0 and G.from_periodic_series(nbp.DAVENPORT_HEILBRONN).e == 1


def test_completed_functional_equations():
    """Lambda(s) = (q/pi)^{s/2} Gamma((s+e)/2) F(s) = Lambda(1-s) holds for DH with e = 1
    and for (n/5) with e = 0 (Arb, rigorous residual bounds), and fails with the wrong e."""
    pts = (complex(0.3, 10.0), complex(0.8, 85.7), complex(-0.4, 50.2))
    for ser in (DH, G.CHI5):
        good = G.functional_equation_check(ser, pts, dps=40)
        bad = G.functional_equation_check(ser, pts, dps=40, e=1 - ser.e)
        assert max(r["rel_residual"] for r in good) < 1e-30
        assert min(r["rel_residual"] for r in bad) > 0.1
    # the Hurwitz-zeta form of DH agrees with rh.davenport_heilbronn (Dirichlet L-functions)
    from rh import davenport_heilbronn as D
    for s in (complex(0.3, 10.0), complex(2.0, 3.0)):
        assert float(abs(G.series_value(DH, s, dps=30) - D.dh_f(s, dps=30)).upper()) < 1e-25


def test_hardy_Z_and_zero_counts(dh_zeros):
    from rh import davenport_heilbronn as D
    for t in (10.0, 250.3):
        assert float(abs(G.hardy_Z(DH, t, dps=30) - D.dh_Z_acb(t, dps=30)).upper()) < 1e-25
    # N(T) = theta(T)/pi + S(T): DH has 214 zeros below 300 = 204 on the line + 2 x 5 off it
    zc = G.zero_count(DH, 300.0)
    assert zc["N"] == 214 == len(dh_zeros["on_line"]) + 2 * len(dh_zeros["off_line"]) == dh_zeros["total_count_winding"]
    assert G.zero_count(G.ZETA, 100.0)["N"] == 29
    # the first DH ordinates are certified by rigorous sign changes of Z within 1e-11
    cert = G.certify_on_line_zeros(lambda x: D.dh_Z_acb(x, dps=30).real, dh_zeros["on_line"][:25], 1e-11)
    assert cert["all_certified"]


# ----------------------------------------------------------------------------
# the coefficients a_n of -F'/F
# ----------------------------------------------------------------------------
def _naive(series, N):
    c = series.c_array(N)
    a, S = np.zeros(N + 1), np.zeros(N + 1)
    for n in range(2, N + 1):
        a[n] = c[n] * math.log(n) - S[n]
        if 2 * n <= N:
            m = np.arange(2, N // n + 1)
            S[n * m] += a[n] * c[m]
    return a


def test_block_sieve_matches_naive_recursion():
    for ser in (G.ZETA, G.CHI5, DH):
        for N in (2, 3, 17, 1000, 4097):
            G._COEFF_CACHE.clear()
            assert np.max(np.abs(G.log_derivative_coefficients(ser, N) - _naive(ser, N))) < 1e-11


def test_zeta_coefficients_are_von_mangoldt():
    N = 20000
    a = G.log_derivative_coefficients(G.ZETA, N)
    n, L = ef.vonmangoldt_table(N)
    ref = np.zeros(N + 1)
    ref[n] = L
    assert np.max(np.abs(a - ref)) < 1e-12


def test_chi5_coefficients_are_chi_times_von_mangoldt():
    """(n/5) is completely multiplicative, so -L'/L = sum chi(n) Lambda(n) n^{-s}:
    float sieve, Arb balls and the exact recursion all reproduce chi(n) Lambda(n)."""
    N = 20000
    a = G.log_derivative_coefficients(G.CHI5, N)
    n, L = ef.vonmangoldt_table(N)
    ref = np.zeros(N + 1)
    ref[n] = L
    chi = np.array([0, 1, -1, -1, 1])[np.arange(N + 1) % 5]
    assert np.max(np.abs(a - chi * ref)) < 1e-12
    aa = G.log_derivative_coefficients_arb(G.CHI5, 500, dps=30)
    for k in range(2, 501):
        assert aa[k].contains(0) if chi[k] * ref[k] == 0 else abs(float(aa[k].mid()) - chi[k] * ref[k]) < 1e-25
    B = G.log_derivative_coefficients_exact(G.CHI5, 120)
    for k in range(2, 121):
        f = sp.factorint(k)
        expect = {next(iter(f)): int(chi[k])} if (len(f) == 1 and chi[k] != 0) else {}
        assert B[k] == expect, (k, B[k])


def test_dh_coefficients_exact_forms():
    """DH has no Euler product: a_n is not supported on prime powers (a_6 = (1+kappa^2) log 6)."""
    K = sp.Symbol("kappa")
    B = G.log_derivative_coefficients_exact(DH, 30)
    expected = {2: K * sp.log(2), 3: -K * sp.log(3), 4: -(2 + K ** 2) * sp.log(2), 5: 0,
                6: (1 + K ** 2) * sp.log(6), 8: K ** 3 * sp.log(2), 11: sp.log(11), 16: (2 - K ** 4) * sp.log(2),
                22: 0, 27: -K ** 3 * sp.log(3)}
    for n, e in expected.items():
        got = G.exact_coefficient(DH, B[n])
        assert sp.simplify(sp.expand_log(got - e, force=True)) == 0, (n, got)
    a = G.log_derivative_coefficients(DH, 3000)
    aa = G.log_derivative_coefficients_arb(DH, 30, dps=40)
    for n in range(2, 31):
        val = float(G.exact_coefficient(DH, B[n]).subs(K, G._KAPPA_RADICAL).evalf(30))
        assert abs(val - a[n]) < 1e-14 and abs(float(aa[n].mid()) - val) < 1e-15
    # a_p = c_p log p for primes, and the defining recursion holds to rounding
    c = DH.c_array(3000)
    for p in sp.primerange(2, 3000):
        assert abs(a[p] - c[p] * math.log(p)) < 1e-12
    for m in (6, 12, 360, 2310, 2999):
        lhs = sum(a[d] * c[m // d] for d in range(1, m + 1) if m % d == 0)
        assert abs(lhs - c[m] * math.log(m)) < 1e-11
    # kappa satisfies the quartic used to reduce the exact coefficients
    assert abs(KAPPA ** 4 + 2 * KAPPA ** 3 - 6 * KAPPA ** 2 - 2 * KAPPA + 1) < 1e-14


def test_absolute_convergence_bounds():
    assert abs(G.zero_free_abscissa(DH) - 1.3951361582351) < 1e-9
    assert abs(G.zero_free_abscissa(G.ZETA) - 1.7286472389982) < 1e-9
    A, B, L = G.coefficient_norm_bound(DH, 2.0, dps=30)
    assert float(B.upper()) < 1
    a = G.log_derivative_coefficients(DH, 100000)
    n = np.arange(1, 100001)
    assert np.all(np.abs(a[1:]) <= float(A.lower()) * n ** 2.0)           # |a_n| <= A n^2
    assert np.sum(np.abs(a[1:]) / n ** 2.0) <= float(A.lower())


# ----------------------------------------------------------------------------
# archimedean term, zeta special case
# ----------------------------------------------------------------------------
def test_archimedean_u_space_equals_r_space():
    for ser in (G.ZETA, G.CHI5, DH):
        for tf in (ef.gaussian(0.05), ef.gaussian(0.5), ef.modulated_gaussian(0.15, 85.7),
                   ef.modulated_gaussian(1.0, 240.4), ef.weil_two_bumps(0.2, 1.0)):
            r = G.arithmetic_side(ser, tf, arch="both", truncation_check=False)
            assert abs(r["archimedean_u_minus_r"]) < 1e-12, (ser.name, tf.name, r["archimedean_u_minus_r"])


def test_zeta_case_reproduces_explicit_formula_module(zeta_zeros):
    """q = 1, c = (1,), poles: the general code gives rh.explicit_formula's arithmetic side
    (a_n from the recursion = Lambda(n)), and matches the zeta zeros."""
    for tf in (ef.gaussian(0.05), ef.gaussian(0.5), ef.gaussian(1.0), ef.modulated_gaussian(1.0, 100.0),
               ef.modulated_gaussian(0.5, 5000.0), ef.weil_box(1.0), ef.weil_two_bumps(0.2, 2.0),
               ef.weil_gaussian_bump(0.3)):
        mine = G.arithmetic_side(G.ZETA, tf)
        ref = ef.explicit_formula_check(tf, zeros=zeta_zeros, nmax=10 ** 6)
        assert abs(mine["poles"] - ref["h_poles"]) < 1e-13
        assert abs(mine["arithmetic_side"] - ref["arithmetic_side"]) < 1e-13, tf.name
        zs = G.zero_side(tf.h, zeta_zeros, series=G.ZETA, tail=True, T=ef.tail_start(zeta_zeros),
                         r_windows=tf.r_windows, tail_h=tf.tail_average)
        assert abs(zs["total"] - mine["arithmetic_side"]) <= max(1e-10, 2 * abs(ref["residual"])), tf.name


# ----------------------------------------------------------------------------
# Davenport–Heilbronn: arithmetic side = zero side, and the off-line zeros are needed
# ----------------------------------------------------------------------------
@pytest.mark.parametrize("s,w", [(0.05, 0.0), (0.1, 0.0), (0.2, 0.0),
                                 (0.15, 20.0), (0.15, 85.7), (0.15, 114.2), (0.15, 170.0)])
def test_dh_explicit_formula(dh_zeros, s, w):
    """204 on-line zeros + 5 off-line quadruples (+ density tail beyond T = 300, negligible
    here) reproduce the arithmetic side; measured relative residuals 2e-15 .. 1e-13."""
    r = G.explicit_formula_check(DH, _tf(s, w), dh_zeros["on_line"], dh_zeros["off_line"], tail=True,
                                 T=dh_zeros["T"], arch="both")
    assert r["relative_residual"] < 1e-11, r["relative_residual"]
    assert abs(r["arithmetic"]["archimedean_u_minus_r"]) < 1e-12
    assert abs(r["zeros"]["tail"]) < 1e-30 and r["zeros"]["off_line_imag_max"] < 1e-12
    assert 0.5 < r["zero_side"] < 5.5


def test_dh_identity_fails_without_off_line_zeros(dh_zeros):
    on, off = dh_zeros["on_line"], dh_zeros["off_line"]
    for w, rho_im in ((85.7, 85.699), (114.2, 114.163)):
        r = G.explicit_formula_check(DH, ef.modulated_gaussian(0.15, w), on, off)
        each = {round(d["rho"][1], 3): d["contribution"] for d in r["zeros"]["off_line_each"]}
        # the quadruple at the probed height contributes ~ 4 (s sqrt(2pi)/2) = 0.752; the residual
        # without the off-line zeros is exactly minus their total contribution
        assert 0.75 < each[rho_im] < 0.755
        assert r["relative_residual"] < 1e-11
        assert abs(r["residual_without_off_line"] + r["off_line_contribution"]) < 1e-12
        assert r["relative_residual_without_off_line"] > 0.15
    # far from every off-line ordinate the off-line zeros are invisible
    r = G.explicit_formula_check(DH, ef.modulated_gaussian(0.15, 20.0), on, off)
    assert abs(r["off_line_contribution"]) < 1e-20
    # even the plain Gaussian s = 0.05 sees them: 5.16e-5 >> its residual 2e-14
    r = G.explicit_formula_check(DH, ef.gaussian(0.05), on, off)
    assert 5.1e-5 < r["off_line_contribution"] < 5.2e-5 and abs(r["residual"]) < 1e-12


def test_dh_wide_test_functions_where_the_a_n_matter(dh_zeros):
    """u-width s = 0.5, 1.0: the prime-type sum (over ALL n, not prime powers) is O(1)."""
    for s, w in ((0.5, 85.7), (1.0, 114.2), (1.0, 240.4)):
        r = G.explicit_formula_check(DH, _tf(s, w), dh_zeros["on_line"], dh_zeros["off_line"])
        assert abs(r["arithmetic"]["prime_sum"]) > 0.1
        assert abs(r["arithmetic"].get("prime_sum_truncation_estimate", 0.0)) < 1e-15
        assert r["relative_residual"] < 1e-11
        assert r["relative_residual_without_off_line"] > 0.5


def test_bump_test_function_consistency():
    """g = f*f for f = e^{-1/(1-(u/a)^2)} cos(w u): g(0) = int f^2 = (1/2pi) int h, h = f^2-hat,
    quadrature converged (halving the step changes nothing at 1e-13)."""
    import mpmath as mp
    a, w = 1.0, 85.7
    tf = G.bump_test_function(a, w)
    fine = G.bump_test_function(a, w, hx=0.0007)
    zs = np.array([0.0, 5.3, 85.7, 88.8, complex(85.699, -0.3085)])
    assert np.max(np.abs(tf.h(zs) - fine.h(zs))) < 1e-13
    assert max(abs(tf.g(u) - fine.g(u)) for u in (0.0, 0.3, 1.7, 1.99)) < 1e-13
    with mp.workdps(20):
        f = lambda u: mp.exp(-1 / (1 - (u / a) ** 2)) * mp.cos(w * u) if abs(u) < a else mp.mpf(0)
        g0 = mp.quad(lambda u: f(u) ** 2, mp.linspace(-a, a, 40))
    assert abs(tf.g(0.0) - float(g0)) < 1e-13
    assert tf.g(2.0) == 0.0 and tf.g(-2.5) == 0.0 and tf.u_support == 2.0


@pytest.mark.parametrize("a,w", [(1.0, 0.0), (1.0, 85.7), (1.0, 114.2), (2.0, 114.2)])
def test_dh_compactly_supported_g_finite_prime_sum(dh_zeros, a, w):
    """g supported in [-2a, 2a]: the prime-type sum is FINITE (n <= e^{2a}: 7 or 54 terms),
    yet (EF) holds -- provided the off-line quadruples are included."""
    r = G.explicit_formula_check(DH, G.bump_test_function(a, w), dh_zeros["on_line"], dh_zeros["off_line"],
                                 arch="both")
    assert r["arithmetic"]["N"] == int(math.floor(math.exp(2 * a)))
    assert r["relative_residual"] < 1e-10 and abs(r["residual"]) < 1e-12
    assert abs(r["arithmetic"]["archimedean_u_minus_r"]) < 1e-12
    if w:
        assert r["off_line_contribution"] > 0.19 and r["relative_residual_without_off_line"] > 0.5


def test_dh_needs_the_non_prime_power_terms(dh_zeros):
    """Keeping only prime powers n (as an Euler product would give) breaks (EF) by ~0.08."""
    tf = G.bump_test_function(2.0, 114.2)
    r = G.explicit_formula_check(DH, tf, dh_zeros["on_line"], dh_zeros["off_line"])
    a = G.log_derivative_coefficients(DH, 54)
    npp = 2 * math.fsum(a[n] / math.sqrt(n) * tf.g(math.log(n)) for n in range(2, 55)
                        if len(sp.factorint(n)) > 1)
    assert abs(npp) > 0.05 and abs(r["residual"]) < 1e-11


def test_dh_density_tail_reduces_the_residual(dh_zeros):
    # s = 0.012: h is still 2e-3 of its peak at r = 300; the density tail removes 99% of the gap
    tf = ef.gaussian(0.012)
    r0 = G.explicit_formula_check(DH, tf, dh_zeros["on_line"], dh_zeros["off_line"], tail=False)
    r1 = G.explicit_formula_check(DH, tf, dh_zeros["on_line"], dh_zeros["off_line"], tail=True, T=300.0)
    assert abs(r0["residual"]) > 1e-3 and abs(r1["residual"]) < 0.02 * abs(r0["residual"])


# ----------------------------------------------------------------------------
# ball arithmetic
# ----------------------------------------------------------------------------
def test_arb_arithmetic_side_encloses_float_value_and_matches_zeros(dh_zeros):
    for ser, s, w in ((DH, 0.15, 85.7), (DH, 0.2, 0.0), (G.ZETA, 0.2, 0.0)):
        ra = G.arithmetic_side_arb(ser, s, w, dps=30)
        x = ra["arithmetic_side_float"]
        assert x["rad"] < 1e-20
        fl = G.arithmetic_side(ser, _tf(s, w), arch="r")["arithmetic_side"]
        assert abs(x["mid"] - fl) < 1e-13
    ra = G.arithmetic_side_arb(DH, 0.15, 85.7, dps=30)
    za = G.zero_side_arb(0.15, 85.7, dh_zeros["on_line"], dh_zeros["off_line"], on_radius=1e-11,
                         off_radius=dh_zeros["off_line_radius"], dps=30)
    diff = ra["arithmetic_side"] - za["total"]
    assert float(abs(diff).upper()) < 1e-10                   # consistent with (EF) over the listed zeros
    gap = ra["arithmetic_side"] - za["on_line"]               # off-line quadruples omitted
    assert float(abs(gap).lower()) > 0.75


def test_zero_side_bookkeeping():
    tf = ef.modulated_gaussian(0.3, 50.0)
    rho = complex(0.7, 50.2)
    z1 = G.zero_side(tf.h, (), [rho])
    tau = (rho - 0.5) / 1j
    assert abs(z1["off_line"] - 4 * tf.h(np.complex128(tau)).real) < 1e-14
    assert G.zero_side(tf.h, (), [rho.conjugate()])["off_line"] == z1["off_line"]
    assert G.zero_side(tf.h, (), [{"re": 0.7, "im": 50.2}])["off_line"] == z1["off_line"]
    zr = G.zero_side(tf.h, (), [complex(0.8, 0.0)])               # real zero: pair 0.8, 0.2
    assert abs(zr["off_line"] - 2 * tf.h(np.complex128(-0.3j)).real) < 1e-14
    assert abs(G.zero_side(tf.h, [0.0, 3.0])["on_line"] - (tf.h(0.0) + 2 * tf.h(3.0))) < 1e-14
    with pytest.raises(ValueError):
        G.zero_side(tf.h, (), [complex(0.4, 10.0)])


# ----------------------------------------------------------------------------
# L(s, (n/5)): an independent e = 0, q = 5 check with its own zeros
# ----------------------------------------------------------------------------
def test_chi5_explicit_formula_with_its_zeros(chi5_zeros):
    assert len(chi5_zeros) == 54 == G.zero_count(G.CHI5, 100.0)["N"]
    assert abs(chi5_zeros[0] - 6.6484533447277) < 1e-10
    for s, w in ((0.1, 0.0), (0.5, 0.0), (0.3, 40.0), (1.0, 60.0)):
        r = G.explicit_formula_check(G.CHI5, _tf(s, w), chi5_zeros, (), tail=True, T=100.0)
        assert r["relative_residual"] < 1e-11, (s, w, r["relative_residual"])
    r = G.explicit_formula_check(G.CHI5, ef.gaussian(0.5), chi5_zeros, ())
    assert abs(r["arithmetic"]["prime_sum"]) > 0.4         # chi(n) Lambda(n) terms are O(1) here
