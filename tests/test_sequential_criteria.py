import mpmath as mp
import numpy as np
from rh import sequential_criteria as sc


def test_c0_c1_closed_forms():
    # c_0 = 1/zeta(2), c_1 = 1/zeta(2) - 1/zeta(4)
    assert abs(sc.c_k_float(0) - 6 / np.pi ** 2) < 1e-14
    assert abs(sc.c_k_float(1) - (6 / np.pi ** 2 - 90 / np.pi ** 4)) < 1e-14


def test_mu_sum_matches_binomial_sum():
    for k in (5, 20, 60):
        a = float(sc.c_k_binomial(k))
        b = sc.c_k_float(k)
        c = float(sc.c_k_exact(k, dps=30))
        assert abs(a - b) < 1e-13
        assert abs(a - c) < 1e-20 or abs(a - c) < 1e-13


def test_explicit_formula_dominant_zero():
    mp.mp.dps = 30
    rho = [mp.mpc(0.5, mp.zetazero(j).imag) for j in range(1, 6)]
    for k in (2000, 20000):
        exact = float(sc.c_k_exact(k, dps=30))
        pred = float(sc.c_k_explicit(k, rho))
        # relative agreement limited by the O(k^-3/2) remainder relative to k^-3/4
        assert abs(exact - pred) < 5 * k ** -1.5 + 1e-12
