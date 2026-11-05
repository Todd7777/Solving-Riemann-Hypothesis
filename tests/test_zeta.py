"""Tests for rh.zeta: theta, Gram points, Riemann–Siegel Z, zero finding.

References are mpmath (high precision) and Arb/python-flint (rigorous balls);
the zero ordinates are compared with the Arb dataset in data/zeros.
"""
import math

import mpmath as mp
import numpy as np
import pytest
from flint import arb, ctx

import rh.zeta as zt
from rh.data import load_zeros

ctx.dps = 30

GAMMA_1 = 14.134725141734693790
GAMMA_2 = 21.022039638771554993
GAMMA_100 = 236.5242296658162058


def n_arb(T):
    return int(round(float(arb.zeta_nzeros(arb(float(T))).mid())))


@pytest.fixture(scope="module")
def arb_zeros():
    return load_zeros(20000)


# ---------------------------------------------------------------- theta ---

@pytest.mark.parametrize("t", [10.0, 100.0, 1e4])
def test_theta_vs_mpmath(t):
    ref = float(zt.theta_mp(t, 40))
    assert abs(zt.theta(t) - ref) <= 1e-14 * max(1.0, abs(ref))
    # pure asymptotic expansion: error ~ exp(-pi t)/2 (1e-14 at t = 10)
    assert abs(float(zt.theta_asymptotic(t)) - ref) <= 2e-14 * max(1.0, abs(ref))


def test_theta_small_t_fallback_and_known_value():
    assert zt.theta(0.0) == 0.0
    assert abs(zt.theta(5.0) - float(zt.theta_mp(5.0))) < 1e-15
    assert abs(zt.theta(-100.0) + zt.theta(100.0)) < 1e-12          # odd
    assert abs(zt.theta(100.0) - 87.97216523178722) < 1e-12           # theta(100)
    assert abs(zt.theta(10.0) - (-3.0670743962898953)) < 1e-14
    v = zt.theta(np.array([5.0, 10.0, 1000.0]))                       # vectorised, mixed regimes
    assert v.shape == (3,) and np.all(np.diff(v) > 0)


def test_theta_inverse_roundtrip():
    t = np.array([10.0, 50.0, 1e3, 1e5])
    back = zt.theta_inverse(zt.theta(t))
    assert np.all(np.abs(back - t) < 1e-9 * t)


# ---------------------------------------------------------- Gram points ---

@pytest.mark.parametrize("n", [-1, 0, 1, 10, 1000, 10 ** 6])
def test_gram_points_vs_arb(n):
    g = zt.gram_point(n)
    ref = arb.gram_point(n)
    assert float(ref.rad()) < 1e-12
    assert abs(float(arb(g) - ref)) < 1e-8
    if n >= 0:
        assert abs(zt.theta(g) - n * math.pi) < 1e-9 * max(1.0, n)


def test_gram_point_known_values_and_index():
    assert abs(zt.gram_point(0) - 17.845599540410860) < 1e-9
    assert abs(zt.gram_point(1) - 23.170282701246309) < 1e-9
    assert abs(zt.gram_point(-1) - 9.666908056130192) < 1e-9
    assert zt.gram_index(1000.0) == 647              # g_647 = 999.6 < 1000 < g_648 = 1000.48
    assert np.all(zt.gram_index(zt.gram_point(np.arange(5, 10)) + 1e-6) == np.arange(5, 10))


# ------------------------------------------- Riemann–Siegel coefficients ---

def test_rs_coefficients_match_haselgrove_tables():
    C = zt.RS_COEFFS
    expected = {
        (0, 0): 0.38268343236508977173, (0, 2): 0.43724046807752044936,
        (0, 4): 0.13237657548034352333, (0, 6): -0.01360502604767418865,
        (1, 1): -0.02682510262837534703, (1, 3): 0.01378477342635185305,
        (1, 5): 0.03849125048223508223,
        (2, 0): 0.00518854283029316849, (2, 2): 0.00030946583880634746,
        (2, 4): -0.01133594107822937338,
        (3, 1): -0.00133971609071945690, (3, 3): 0.00374421513637939370,
        (3, 5): -0.00133031789193214681,
        (4, 0): 0.00046483389361763382, (4, 2): -0.00100566073653404708,
        (4, 4): 0.00024044856573725793,
    }
    for (k, j), v in expected.items():
        assert abs(C[k, j] - v) < 1e-16, (k, j)
    # parity: C_0, C_2, C_4 even in z, C_1, C_3 odd
    assert np.all(C[0, 1::2] == 0) and np.all(C[2, 1::2] == 0) and np.all(C[4, 1::2] == 0)
    assert np.all(C[1, 0::2] == 0) and np.all(C[3, 0::2] == 0)


def test_rs_C0_equals_psi():
    p = np.array([0.02, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9, 0.98])
    assert np.max(np.abs(zt.rs_C(0, p) - zt.psi_direct(p))) < 1e-13
    # removable singularities of Psi at p = 1/4, 3/4 are handled by the series
    assert np.isfinite(zt.rs_C(0, 0.25)) and np.isfinite(zt.rs_C(0, 0.75))


# ------------------------------------------------------------------- Z ---

Z_TEST_POINTS = [30.0, 50.0, 100.0, 150.0, 250.0, 499.9, 500.0, 750.0, 1000.0, 2000.0, 3000.0,
                 5000.0, 7005.0629, 7005.08, 7005.1006, 7005.5, 1e4, 3e4, 74920.5, 1e5]


@pytest.mark.parametrize("t", Z_TEST_POINTS)
def test_Z_vs_mpmath(t):
    ref = float(zt.Z_mp(t, 30))
    err = abs(zt.Z(t) - ref)
    assert err < 1e-6
    assert err <= zt.Z_error_budget(np.array([t]))[0]   # within truncation bound + rounding estimate


def test_Z_near_lehmer_pair_values():
    # Z is tiny but resolved between the Lehmer pair 7005.0629 / 7005.1006
    assert abs(zt.Z(7005.0629) - 1.4010266571959912e-05) < 1e-10
    assert abs(zt.Z(7005.1006) + 1.5100620365297e-05) < 1e-10
    assert zt.Z(7005.08) > 0 and zt.Z(7005.0) < 0 and zt.Z(7005.2) < 0


@pytest.mark.parametrize("t", [200.0, 500.0, 1000.0, 1e4])
def test_Z_rs_error_within_gabcke_bound(t):
    ref = float(zt.Z_mp(t, 30))
    for K in range(5):
        err = abs(zt.Z_rs(t, K) - ref)
        assert err <= zt.rs_remainder_bound(t, K) + zt.Z_rounding_estimate(np.array([t]))[0], K


def test_rs_remainder_bound_shape():
    assert np.isinf(zt.rs_remainder_bound(100.0, 4))
    assert abs(zt.rs_remainder_bound(200.0, 4) - 0.017 * 200 ** -2.75) < 1e-20
    with pytest.raises(ValueError):
        zt.Z_rs(1.0)
    with pytest.raises(ValueError):
        zt.rs_remainder_bound(10.0, 5)


@pytest.mark.parametrize("t", [0.0, 0.5, 14.134725141734693, 100.0, 300.0, 499.9])
def test_Z_em_vs_mpmath(t):
    assert abs(zt.Z_em(t) - float(zt.Z_mp(t, 30))) < 1e-11
    z, bound = zt.zeta_em(np.array([t]))
    assert bound[0] < 1e-16
    with mp.workdps(30):
        ref = complex(mp.zeta(mp.mpc(0.5, t)))
    assert abs(z[0] - ref) < 1e-11


def test_Z_at_zero_is_zeta_half():
    assert abs(zt.Z(0.0) - (-1.4603545088095868)) < 1e-13


def test_Z_scalar_matches_vector():
    t = np.array([15.0, 499.0, 500.0, 5000.3, 74000.7])
    v = zt.Z(t)
    for i, x in enumerate(t):
        assert abs(zt.Z_scalar(float(x)) - v[i]) < 1e-14


@pytest.mark.parametrize("t", [100.0, 7005.0629, 74920.5])
def test_Z_arb_contains_Z(t):
    ball = zt.Z_arb(t)
    assert abs(zt.Z(t) - float(ball.mid())) <= float(ball.rad()) + zt.Z_error_budget(np.array([t]))[0]


# ----------------------------------------------------------- zero finding ---

def test_find_zeros_0_500(arb_zeros):
    expected = n_arb(500.0)
    assert expected == 269
    res = zt.find_zeros(0.0, 500.0, details=True)
    assert len(res) == expected
    assert np.all(np.diff(res.zeros) > 0)
    assert np.all((res.lo < res.zeros) & (res.zeros < res.hi))
    assert np.all(np.sign(res.Z_lo) * np.sign(res.Z_hi) < 0)
    assert np.max(np.abs(res.zeros - arb_zeros[:expected])) < 1e-8
    assert abs(res.zeros[0] - GAMMA_1) < 1e-9
    assert abs(res.zeros[1] - GAMMA_2) < 1e-9
    assert abs(res.zeros[99] - GAMMA_100) < 1e-9
    assert res.unresolved_blocks == []
    # every complete Gram block in range satisfied Rosser's rule
    assert all(found == expected_ for (_, _, expected_, found) in res.gram_blocks)


def test_find_zeros_lehmer_pair(arb_zeros):
    n0, n1 = n_arb(7000.0), n_arb(7010.0)
    res = zt.find_zeros(7000.0, 7010.0, details=True)
    assert len(res) == n1 - n0
    assert np.max(np.abs(res.zeros - arb_zeros[n0:n1])) < 1e-8
    pair = res.zeros[(res.zeros > 7005.0) & (res.zeros < 7005.2)]
    assert pair.size == 2
    assert abs(pair[0] - 7005.062866) < 1e-5 and abs(pair[1] - 7005.100565) < 1e-5
    # the coarse grid (8 points per Gram interval) misses the pair; the
    # Gram-block check must have triggered a local refinement
    assert res.refined_blocks >= 1 and res.unresolved_blocks == []


def test_find_zeros_sparse_grid_still_complete(arb_zeros):
    # even 2 points per mean spacing finds everything thanks to the Gram-block refinement
    n0, n1 = n_arb(7000.0), n_arb(7040.0)
    z = zt.find_zeros(7000.0, 7040.0, pts_per_spacing=2)
    assert z.size == n1 - n0 and np.max(np.abs(z - arb_zeros[n0:n1])) < 1e-8


def test_refine_methods_agree():
    a = zt.find_zeros(2000.0, 2100.0, refine="brent")
    b = zt.find_zeros(2000.0, 2100.0, refine="illinois")
    assert a.size == b.size and np.max(np.abs(a - b)) < 1e-10


def test_find_zeros_argument_errors():
    with pytest.raises(ValueError):
        zt.find_zeros(10.0, 5.0)
    with pytest.raises(ValueError):
        zt.find_zeros(0.0, 100.0, refine="newton")


# ---------------------------------------------------------- N_smooth, S ---

def test_N_smooth_and_S():
    assert abs(zt.N_smooth(1000.0) - 648.6162419444237) < 1e-9
    assert abs(zt.S(1000.0, 649) - 0.3837580555763) < 1e-9
    zeros = np.array([14.13, 21.02, 25.01])
    assert zt.count_below(zeros, 21.02) == 2 and zt.count_below(zeros, 10.0) == 0
    # Gram points are where N_smooth is an integer
    assert abs(zt.N_smooth(zt.gram_point(647)) - 648.0) < 1e-9
