"""Tests for rh.li (Li/Keiper coefficients).  Whole file runs in well under a minute."""
import math

import numpy as np
import pytest

from rh import li
from rh.constants import LI_LAMBDA_1
from rh.data import available_datasets, load_zeros, compute_zeros_arb


@pytest.fixture(scope="module")
def zeros():
    """First 100000 zeros if the dataset is present, else 2000 zeros computed with Arb."""
    sets = available_datasets()
    if sets and max(b for _, b, _ in sets) >= 100000:
        return load_zeros(100000)
    return compute_zeros_arb(1, 2000)


# --------------------------------------------------------------------------- zero sums
def test_lambda1_closed_form_from_zeros(zeros):
    lam, info = li.li_from_zeros(1, zeros, details=True)
    err = abs(lam[0] - LI_LAMBDA_1)
    tol = 1e-9 if len(zeros) >= 100000 else 1e-6
    assert err < tol
    # the rigorous-style error bound must hold (and the true error is far below it)
    assert err < info["error_bound"][0]


def test_partial_sum_matches_direct_formula():
    g = np.array([14.134725141734693, 21.022039638771555, 25.010857580145688])
    for n in (1, 2, 7):
        direct = 2 * sum(1 - math.cos(n * 2 * math.atan(1 / (2 * x))) for x in g)
        assert abs(li.li_partial_sum([n], g)[0] - direct) < 1e-13


def test_effective_cutoff_solves_smooth_count():
    N, last = 100000, 74920.82749899419
    T = li.effective_cutoff(N, last)
    assert abs(float(li.smooth_count(T)) - N) < 1e-6
    spacing = 1 / float(li.smooth_density(T))
    assert -spacing < T - last < 2 * spacing
    assert abs(li.effective_cutoff(N) - T) < 1e-6  # without the starting guess


def test_tail_gauss_legendre_matches_quad():
    T = 74921.2257
    ns = np.array([1, 10, 500, 5000])
    gl = li.li_tail(ns, T)
    qd = li.li_tail(ns, T, method="quad")
    assert np.all(np.abs(gl - qd) / qd < 1e-10)
    # leading behaviour of the tail: ~ (n^2/(2 pi T)) (log(T/2pi) + 1) for n << T
    approx = ns ** 2 / (2 * math.pi * T) * (math.log(T / (2 * math.pi)) + 1)
    assert np.all(np.abs(gl / approx - 1) < 1e-2)


def test_li_from_zeros_rejects_incomplete_zero_list(zeros):
    with pytest.raises(ValueError):
        li.li_from_zeros(3, zeros[1000:1500])


# --------------------------------------------------------------------------- high precision
def test_high_precision_lambda1_to_30_digits():
    from flint import arb, ctx
    balls = li.li_high_precision(5)
    old = ctx.dps
    try:
        ctx.dps = 60
        closed = 1 + arb.const_euler() / 2 - (4 * arb.pi()).log() / 2
        assert float((balls[0] - closed).abs_upper()) < 1e-30   # rigorous ball comparison
    finally:
        ctx.dps = old
    import mpmath as mp
    with mp.workdps(50):
        val = mp.mpf(li.balls_to_str([balls[0]], 45)[0])
        assert abs(val - li.lambda_1_closed_form(50)) < mp.mpf(10) ** -30


def test_high_precision_known_values():
    lam = li.li_high_precision(5, as_float=True)
    # Keiper 1992 / Li 1997 / Maslanka 2004 (digits reproduced by both methods here)
    assert abs(lam[1] - 0.0923457352280466) < 1e-13
    assert abs(lam[2] - 0.2076389205543248) < 1e-13
    assert abs(lam[3] - 0.3687904794922416) < 1e-13
    assert abs(lam[4] - 0.5755427144611774) < 1e-13


def test_high_precision_agrees_with_zero_sum(zeros):
    hp = li.li_high_precision(5, as_float=True)
    lam, info = li.li_from_zeros(5, zeros, details=True)
    assert np.all(np.abs(hp - lam) < 1e-6)
    assert np.all(np.abs(hp - lam) <= info["error_bound"])


def test_positivity_up_to_500(zeros):
    lam = li.li_from_zeros(500, zeros)
    assert np.all(lam > 0)
    rep = li.positivity_report(np.arange(1, 501), lam)
    assert rep["all_positive"] and rep["argmin_n"] == 1
    balls = li.li_high_precision(500)
    assert all(float(b.mid()) - float(b.rad()) > 0 for b in balls)   # rigorous positivity
    assert li.balls_max_radius(balls) < 1e-30


# --------------------------------------------------------------------------- asymptotics
def test_archimedean_part_arb_vs_bombieri_lagarias_formula():
    import mpmath as mp
    arch = li.li_high_precision(40, part="archimedean")
    arith = li.li_high_precision(40, part="arithmetic")
    total = li.li_high_precision(40)
    with mp.workdps(60):
        for n in (1, 2, 3, 10, 25, 40):
            s1_arb = mp.mpf(li.balls_to_str([arch[n - 1]], 55)[0])
            assert abs(s1_arb - li.li_archimedean_exact(n, dps=60)) < mp.mpf(10) ** -25
    # arithmetic part at n = 1 is eta_0 = Euler's constant
    assert abs(float(arith[0].mid()) - 0.5772156649015329) < 1e-15
    from flint import ctx
    old = ctx.dps
    try:
        ctx.dps = 80   # ball arithmetic below runs at the global precision
        assert all(float((arch[i] + arith[i] - total[i]).abs_upper()) < 1e-40 for i in range(40))
    finally:
        ctx.dps = old


def test_smooth_part_asymptotics():
    arch = li.balls_to_float(li.li_high_precision(400, part="archimedean"))
    ns = np.arange(1, 401)
    # harmonic-number form is exact up to exponentially small terms
    assert np.all(np.abs(arch[99:] - li.li_smooth(ns[99:], "harmonic")) < 1e-9)
    # log form: S_1(n) - [(n/2)(log n + gamma - 1 - log 2pi) + 3/4] = -1/(24 n) + O(n^-3)
    d = arch[99:] - li.li_smooth(ns[99:], "log")
    assert np.all(np.abs(d + 1 / (24 * ns[99:])) < 1e-6)
    assert np.all(np.abs(arch[99:] - li.li_smooth(ns[99:], "log3")) < 1e-8)


def test_fit_recovers_asymptotic_constants():
    lam = li.li_high_precision(1000, as_float=True)
    ns = np.arange(1, 1001)
    fit = li.fit_asymptotic(ns, lam)
    assert abs(fit["fit_c1"] - li.LI_C1) < 5e-3
    assert abs(fit["fit_free_a"] - 0.5) < 5e-3
    osc = li.li_oscillation(ns, lam)
    assert np.max(np.abs(osc) / np.sqrt(ns)) < 2.0   # O(sqrt n)-size remainder
