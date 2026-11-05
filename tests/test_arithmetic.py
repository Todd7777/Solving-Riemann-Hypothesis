"""Tests for rh.arithmetic: Robin, Lagarias, Nicolas criteria and colossally abundant numbers."""
import math

import numpy as np
import pytest
import sympy

from rh import arithmetic as A

# the 26 exceptions with n >= 3 as usually listed (OEIS A067698); n = 2 is the 27th
ROBIN_EXCEPTIONS_GE_3 = [3, 4, 5, 6, 8, 9, 10, 12, 16, 18, 20, 24, 30, 36, 48, 60, 72, 84,
                         120, 180, 240, 360, 720, 840, 2520, 5040]
CA_FIRST_14 = [2, 6, 12, 60, 120, 360, 2520, 5040, 55440, 720720, 1441440, 4324320,
               21621600, 367567200]


@pytest.fixture(scope="module")
def sigma_1e5():
    return A.sigma_sieve(10 ** 5)


def test_sigma_sieve_matches_sympy(sigma_1e5):
    rng = np.random.default_rng(12345)
    for n in [1, 2, 3, 4, 6, 12, 5040, 65536, 99991, 100000] + rng.integers(1, 10 ** 5, 30).tolist():
        assert int(sigma_1e5[n]) == int(sympy.divisor_sigma(int(n))), n
    assert sigma_1e5[0] == 0


def test_robin_exceptions_are_exactly_the_known_27(sigma_1e5):
    r = A.robin_check(10 ** 5, sigma=sigma_1e5)
    assert r.violators == list(A.ROBIN_EXCEPTIONS)
    assert len(r.violators) == 27
    assert [n for n in r.violators if n >= 3] == ROBIN_EXCEPTIONS_GE_3
    assert r.violators_match_known
    # beyond 5040 Robin's inequality holds, with the maximum of the ratio at n = 10080
    assert r.max_robin_ratio_beyond_5040 < 1
    assert r.argmax_beyond_5040 == 10080
    assert abs(r.max_robin_ratio_beyond_5040 - 0.9858186) < 1e-6
    assert r.ratio_at_5040 > 1
    assert abs(r.max_ratio_beyond_5040 / A.E_GAMMA - r.max_robin_ratio_beyond_5040) < 1e-15


def test_robin_ratio_against_direct_formula(sigma_1e5):
    n = 10080
    direct = sigma_1e5[n] / (n * math.log(math.log(n))) / A.E_GAMMA
    assert abs(direct - A.robin_check(10 ** 5, sigma=sigma_1e5).max_robin_ratio_beyond_5040) < 1e-14


def test_lagarias_no_violators_to_1e5(sigma_1e5):
    r = A.lagarias_check(10 ** 5, sigma=sigma_1e5)
    assert r.violators == []
    assert r.slack_at_1 == 0.0                      # equality only at n = 1
    assert r.min_slack > 0.3 and r.argmin_slack == 2  # H_2 + e^{H_2} log H_2 - 3 = 0.3171...
    assert abs(r.min_slack - 0.31716854341180) < 1e-10
    assert r.max_ratio < 1 and r.argmax_ratio == 12
    assert r.slack_at_5040 > 0 and r.ratio_at_5040 < 1


def test_harmonic_numbers():
    H = A.harmonic_numbers(1000)
    assert float(H[1]) == 1.0
    assert abs(float(H[1000]) - 7.485470860550345) < 1e-12
    assert abs(float(H[10]) - 7381 / 2520) < 1e-15


def test_nicolas_first_2000_primorials():
    r = A.nicolas_check(2000, dps=50)
    assert r.violations == []
    assert r.decreasing_from_2
    assert r.min_ratio > 1
    assert r.ratio[0] < 0                           # k = 1: log log 2 < 0, inequality trivial
    assert r.ratio[1] > 1 and abs(r.ratio[1] - 2.888175572) < 1e-8   # N_2 = 6
    assert all(d > 0 for d in r.difference[1:])
    assert r.primes[-1] == 17389 and r.K == 2000
    # (ratio - 1) sqrt(p_k) log p_k is of order one (~2.17 here)
    assert 1.5 < r.scaled_tail[-1][1] < 3.0


def test_ca_first_14_match_A004490():
    cas = A.ca_numbers(14)
    assert [c.n for c in cas] == CA_FIRST_14
    assert [c.n for c in A.ca_numbers(len(A.CA_NUMBERS_KNOWN))] == list(A.CA_NUMBERS_KNOWN)
    assert cas[7].factorization == {2: 4, 3: 2, 5: 1, 7: 1}          # 5040
    assert all(c.gap_to_runner_up > 1e-6 for c in cas)                # no near-ties in the chain


def test_ca_against_brute_force_maximisation():
    N = 10 ** 6
    sig = A.sigma_sieve(N)
    cas = A.ca_numbers(12)
    for k in range(10):                      # CA numbers <= 10^6: 2 ... 720720
        eps_mid = 0.5 * (cas[k].critical_epsilon + cas[k + 1].critical_epsilon)
        assert A.brute_force_ca_argmax(eps_mid, N, sig) == cas[k].n
        assert A.ca_number_for_epsilon(eps_mid) == cas[k].factorization


def test_ca_exponent_formula():
    # eps = 0.5: a_2 = 1, a_3 = 0  -> n = 2 ; eps = 0.2: 12 = 2^2 3 is optimal
    assert A.ca_exponent(2, 0.5) == 1 and A.ca_exponent(3, 0.5) == 0
    assert A.ca_number_for_epsilon(0.2) == {2: 2, 3: 1}
    assert A.ca_number_for_epsilon(1.0) == {}


def test_ca_robin_and_lagarias_high_precision():
    cas = A.ca_numbers(60, verify_formula=True)
    beyond = [c for c in cas if c.n > 5040]
    assert all(c.robin_ok for c in beyond) and all(c.robin_ratio < 1 for c in beyond)
    assert all(not c.robin_ok for c in cas if c.n <= 5040)   # the small CA numbers violate Robin
    assert all(c.lagarias_ok for c in cas)
    # independent float check of sigma(n)/n for a mid-size CA number
    c = cas[10]
    assert abs(c.sigma_over_n - sympy.divisor_sigma(c.n) / c.n) < 1e-12
    # the Robin ratio of CA numbers creeps back towards 1 from below and Gronwall's ratio towards e^gamma
    assert cas[-1].robin_ratio > cas[8].robin_ratio
    assert 1.7 < cas[-1].gronwall_ratio < A.E_GAMMA
    assert cas[-1].digits > 60
    summ = A.ca_summary(cas)
    assert summ["matches_A004490"] and summ["robin_holds_for_all_n_gt_5040"] and summ["lagarias_holds_for_all"]
