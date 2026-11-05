"""Tests for rh.speiser: argument principle machinery, zeros of zeta', Speiser's criterion."""
import cmath
import math

import mpmath as mp
import pytest
from flint import acb, arb, ctx

from rh import speiser as S

FIRST = complex(2.4631618694543214, 23.29832049276286)


def test_zeta_prime_against_mpmath():
    mp.mp.dps = 20
    for s in [complex(0.7, 30.3), complex(2.0, 100.1), complex(-0.5, 12.0)]:
        ref = complex(mp.zeta(mp.mpc(s.real, s.imag), derivative=1))
        assert abs(S.zeta_prime(s) - ref) < 1e-12 * max(1, abs(ref))
        d1, d2 = S.zeta_prime_jet(s)
        ref2 = complex(mp.zeta(mp.mpc(s.real, s.imag), derivative=2))
        assert abs(d1 - ref) < 1e-12 * max(1, abs(ref)) and abs(d2 - ref2) < 1e-12 * max(1, abs(ref2))


def test_first_zero_of_zeta_prime_by_newton():
    z, steps, res = S.newton_refine(S.zeta_prime_jet, complex(2.5, 23.0))
    assert z is not None and abs(z - FIRST) < 1e-10 and res < 1e-13
    cert = S.certify_zero(S.zeta_prime_ball_jet, z)
    assert cert.certified and cert.re_lower > 2.46 and cert.re_upper < 2.47
    assert cert.im_lower <= FIRST.imag <= cert.im_upper


def test_argument_principle_on_test_functions():
    c = complex(0.3, 0.4)
    assert S.winding_number(lambda z: z - c, 0, 1, 0, 1) == 1
    assert S.winding_number(lambda z: 1 / (z - c), 0, 1, 0, 1) == -1
    assert S.winding_number(lambda z: (z - c) ** 3, 0, 1, 0, 1) == 3
    assert S.winding_number(lambda z: z - c, 2, 3, 2, 3) == 0
    with pytest.raises(S.ArgumentPrincipleError):       # zero on the contour
        S.arg_change(lambda z: z, complex(-1, 0), complex(1, 0), 4)
    with pytest.raises(S.ArgumentPrincipleError):       # zero extremely close to the contour
        S.arg_change(lambda z: z - complex(0.1, 1e-30), complex(-1, 0), complex(1, 0), 4)


def test_locate_zeros_polynomial():
    roots = [complex(0.3, 0.4), complex(0.7, 0.2), complex(0.5, 0.5), complex(0.31, 0.41)]

    def f(z):
        p = 1
        for r in roots:
            p *= (z - r)
        return p

    def jet(z):
        p, dp = 1, 0
        for r in roots:
            dp = dp * (z - r) + p
            p *= (z - r)
        return p, dp

    zs, w, nev = S.locate_zeros(f, jet, 0, 1, 0, 1, dt=0.5, sigma_breaks=[0, 0.5, 1])
    assert w == 4 and len(zs) == 4
    for r in roots:
        assert min(abs(z.z - r) for z in zs) < 1e-10


def test_no_zeros_left_of_half_and_pole_check():
    ctx.dps = 15
    # the double pole of zeta' at s = 1 and no zeros: winding number -2
    assert S.winding_number(S.zeta_prime, -1, 4, -0.25, 0.25, 0.1, 0.05, 0.6) == -2
    # no zeros of zeta' with Re s < 1/2 below t = 100 (Speiser + RH verified there)
    assert S.winding_number(S.zeta_prime, -1, 0.5, 0.25, 100, 0.25, 0.1) == 0


def test_speiser_scan_T100():
    r = S.speiser_scan(100.0, workers=1)
    assert r.count == 19                                   # two independent winding computations
    assert r.winding_total_cells == r.winding_total_independent == r.count
    assert r.count_certified == r.count
    assert r.count_re_lt_half == 0 and r.count_re_lt_half_rigorous == 0
    assert r.min_re > 0.5
    assert all(z["certified"] and z["re_lower"] > 0.5 for z in r.zeros)
    assert r.first_zeros_match_spira
    assert r.pole_check_winding == -2
    assert r.N_T == 29
    assert abs(r.count - r.berndt_main_term) < 5          # O(log T) error term
    assert abs(r.count - r.N_T_minus_log2_term) < 5
    zs = [complex(z["re"], z["im"]) for z in r.zeros]
    assert zs == sorted(zs, key=lambda z: z.imag)
    assert abs(zs[0] - FIRST) < 1e-9
    assert abs(zs[1] - complex(1.2864968222690476, 31.70825008311591)) < 1e-9
    assert all(z["residual"] < 1e-12 for z in r.zeros)


def test_berndt_main_term_and_zero_count():
    assert S.zeta_zero_count(100.0) == 29 and S.zeta_zero_count(1000.0) == 649
    T = 1000.0
    assert abs(S.berndt_main_term(T) - (T / (2 * math.pi) * math.log(T / (4 * math.pi)) - T / (2 * math.pi))) < 1e-9
