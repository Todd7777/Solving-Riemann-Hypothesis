"""Tests for rh.davenport_heilbronn: functional equation, kappa, and the off-line zeros."""
import math

import numpy as np
import pytest
from flint import acb, ctx

from rh import davenport_heilbronn as D

Z1 = complex(0.808517, 85.699348)      # Spira 1994 / Balanzario & Sanchez-Ortiz 2007
Z2 = complex(0.650830, 114.163343)


def test_kappa_from_root_number():
    kc = D.verify_kappa()
    assert kc["difference_upper"] < 1e-30
    assert kc["abs_root_number_minus_1"] < 1e-30
    assert abs(D.KAPPA - 0.2840790438404123) < 1e-15
    assert abs(float(D.KAPPA_MP) - (math.sqrt(10 - 2 * math.sqrt(5)) - 2) / (math.sqrt(5) - 1)) < 1e-15


def test_functional_equation_residual_tiny():
    for s in [complex(0.3, 10.0), complex(0.8, 85.7), complex(1.7, 3.3), complex(-0.4, 50.2), complex(0.5, 200.1)]:
        r = D.functional_equation_residual(s)
        assert r["abs_residual"] < 1e-15 and r["rel_residual"] < 1e-15


def test_dirichlet_series_and_no_euler_product():
    c = D.dh_coefficients(10)
    assert np.allclose(c, [1, D.KAPPA, -D.KAPPA, -1, 0, 1, D.KAPPA, -D.KAPPA, -1, 0])
    assert abs(c[1] * c[2] - c[5]) > 0.9           # c_2 c_3 = -kappa^2 != c_6 = 1: not multiplicative
    s = complex(2.0, 3.0)
    assert abs(D.dh_dirichlet_partial_sum(s, 200000) - D.dh_f_complex(s)) < 1e-6
    # zero-free abscissa and its consequence
    assert abs(D.dh_zero_free_abscissa() - D.SIGMA0) < 1e-9
    assert abs(D.dh_f_complex(complex(3.0, 500.0)) - 1) < 0.2
    # trivial zeros at s = -1, -3 (zeros of the Gamma factor)
    ctx.dps = 20
    assert abs(D.dh_f(acb(-1))) < 1e-15 and abs(D.dh_f(acb(-3))) < 1e-15


def test_Z_is_real():
    for t in [10.0, 85.7, 114.16, 250.3]:
        z = D.dh_Z_acb(t, dps=20)
        assert float(abs(z.imag).upper()) < 1e-15
        assert abs(D.dh_Z(t)) == pytest.approx(abs(D.dh_f_complex(complex(0.5, t))), rel=1e-10)
    Z, dZ = D.dh_Z_jet(85.7)
    h = 1e-5
    assert abs(dZ - (D.dh_Z(85.7 + h) - D.dh_Z(85.7 - h)) / (2 * h)) < 1e-6


def test_first_off_line_zero_found_and_certified():
    off, w, nev = D.off_line_zeros(84.0, 88.0)
    assert w == 1 and len(off) == 1
    z = off[0]
    zc = complex(z["re"], z["im"])
    assert abs(zc - Z1) < 1e-5
    assert z["residual"] < 1e-12 and abs(D.dh_f_complex(zc)) < 1e-12
    assert z["certified"] and z["rigorously_off_line"] and z["re_lower"] > 0.5
    assert z["re"] != 0.5 and abs(z["re"] - 0.5) > 0.3
    # the mirror zero 1 - conj(rho) is also a zero
    assert abs(D.dh_f_complex(complex(1 - z["re"], z["im"]))) < 1e-10


def test_scan_consistency_80_120():
    r = D.dh_scan(T=120.0, t0=80.0)
    assert r.consistent
    assert r.off_line_count_right == 2 and r.off_line_certified == 2
    assert r.total_count_winding == r.on_line_count + 2 * r.off_line_count_right
    assert r.on_line_count == 24
    assert len(r.known_zeros_found) == 2 and all(k["distance"] < 1e-5 for k in r.known_zeros_found)
    assert r.min_offline_distance_to_line > 0.1
    assert all(abs(D.dh_Z(t)) < 1e-9 for t in r.on_line_zeros)
    assert all(d["rigorously_off_line"] for d in r.off_line_zeros)
