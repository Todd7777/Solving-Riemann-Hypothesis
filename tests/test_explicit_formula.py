"""Tests for rh.explicit_formula (fast: < 60 s in total)."""
import math

import numpy as np
import pytest
import sympy

from rh.data import load_zeros
import rh.explicit_formula as ef


@pytest.fixture(scope="module")
def zeros():
    return load_zeros(100000)


def test_vonmangoldt_table():
    n, L = ef.vonmangoldt_table(300)
    assert np.all(np.diff(n) > 0)
    ref = {k: math.log(sympy.primefactors(k)[0]) for k in range(2, 301) if len(sympy.primefactors(k)) == 1}
    assert set(n.tolist()) == set(ref)
    for k, lk in zip(n, L):
        assert abs(lk - ref[int(k)]) < 1e-14
    # psi(300) = sum Lambda(n), n <= 300
    assert abs(L.sum() - float(sympy.log(sympy.lcm(list(range(1, 301)))))) < 1e-9


def test_explicit_formula_gaussian_residuals(zeros):
    # sigma = 2: g(log n) at n = 1e6 is 4.6e-11, prime truncation error ~3e-8
    r = ef.explicit_formula_check(ef.gaussian(2.0), zeros=zeros, nmax=10 ** 6)
    assert abs(r["residual"]) < 1e-6
    assert abs(r["h_poles"] - 2 * 2 * math.sqrt(2 * math.pi) * math.exp(0.5)) < 1e-12
    for sigma in (1.5, 0.5, 0.2, 0.05, 0.01, 0.001):
        r = ef.explicit_formula_check(ef.gaussian(sigma), zeros=zeros, nmax=10 ** 6)
        assert abs(r["residual"]) < 1e-12, (sigma, r)
    # small sigma: the zeros dominate, h(gamma_1) is not negligible
    r = ef.explicit_formula_check(ef.gaussian(0.01), zeros=zeros, nmax=10 ** 6)
    assert r["zero_side"] > 1.0 and abs(r["prime_sum"]) < 1e-12


def test_explicit_formula_modulated_gaussians(zeros):
    for sigma, omega in ((1.0, 100.0), (1.0, 1000.0), (0.5, 5000.0), (1.0, 70000.0)):
        r = ef.explicit_formula_check(ef.modulated_gaussian(sigma, omega), zeros=zeros, nmax=10 ** 6)
        # both the primes and the zeros contribute at order 1 here
        assert abs(r["prime_sum"]) > 0.05 and r["zero_side"] > 1.0
        assert abs(r["residual"]) < 1e-10, (sigma, omega, r)


def test_archimedean_term_r_and_u_space_agree():
    for tf in (ef.gaussian(2.0), ef.gaussian(0.2), ef.weil_gaussian_bump(0.3), ef.weil_two_bumps(0.2, 2.0)):
        assert abs(ef.archimedean_term(tf, "r") - ef.archimedean_term(tf, "u")) < 1e-12
    # box: u-space only (h decays like 1/r^2); check against a direct r-space quadrature
    tf = ef.weil_box(1.0)
    au = ef.archimedean_term(tf, "u")
    ar = ef.archimedean_term_r(tf.h, ((0.0, 2000.0),), epsabs=1e-10, epsrel=1e-10)
    # r-space truncated at 2000: tail ~ (1/pi) int 2/r^2 (log(r/2pi)) dr ~ 4e-3
    assert abs(au - ar) < 5e-3


def test_density_tail_accelerates_convergence(zeros):
    c = ef.convergence_in_zeros(ef.gaussian(0.003), zeros, [300, 1000, 3000])
    rows = c["rows"]
    assert abs(rows[0]["residual"]) < 1e-2 < abs(rows[0]["residual_no_tail"])
    assert abs(rows[1]["residual"]) < 1e-5 < abs(rows[1]["residual_no_tail"])
    assert abs(rows[2]["residual"]) < 1e-12


def test_weil_positivity_families(zeros):
    for tf in (ef.weil_gaussian_bump(0.05), ef.weil_gaussian_bump(0.3),
               ef.weil_two_bumps(0.2, 0.3466), ef.weil_two_bumps(0.2, 2.0), ef.weil_two_bumps(0.1, 3.0)):
        r = ef.weil_functional(tf, 10 ** 6, zeros)
        assert r["arithmetic_side"] >= 0.0
        assert abs(r["residual"]) < 1e-12, r
    for a in (0.5, 1.0, 3.0):
        tf = ef.weil_box(a)
        assert abs(tf.h_poles() - 32 * math.sinh(a / 2) ** 2) < 1e-12
        r = ef.weil_functional(tf, 10 ** 6, zeros)
        assert r["arithmetic_side"] > 0.0
        assert abs(r["residual"]) < 1e-7, r


def test_weil_scan_min_nonnegative(zeros):
    sc = ef.weil_positivity_scan("two_bumps", np.linspace(0.2, 2.0, 10), nmax=10 ** 5, sigma=0.2)
    assert sc["all_nonnegative"] and sc["min_arithmetic_side"] >= 0.0


def test_weil_functional_from_samples_matches_analytic(zeros):
    u = np.linspace(-6, 6, 6001)
    f = np.exp(-u ** 2 / (2 * 0.3 ** 2))
    rs = ef.weil_functional_from_samples(u, f, 10 ** 6, zeros)
    ra = ef.weil_functional(ef.weil_gaussian_bump(0.3), 10 ** 6, zeros)
    for key in ("h_poles", "prime_sum", "archimedean"):
        assert abs(rs[key] - ra[key]) < 1e-9, key
    assert abs(rs["arithmetic_side"] - ra["arithmetic_side"]) < 1e-9
    assert abs(rs["zero_side_finite"] - ra["zero_side"]) < 1e-9
