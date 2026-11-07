import math

import mpmath as mp
import numpy as np
import pytest

from rh import weil_form as wf
from rh.data import load_zeros


@pytest.fixture(scope="module")
def zeros():
    return load_zeros(100000)


def test_gram_and_parity_structure():
    W = wf.build_weil_form(0.8, J=10, dps=30, estimate_quad_error=False)
    for i in range(10):
        assert abs(float(W.G[i, i].mid()) - 2 * 0.8 / (2 * i + 1)) < 1e-25
        for j in range(10):
            q_ij, q_ji = float(W.Q[i, j].mid()), float(W.Q[j, i].mid())
            assert abs(q_ij - q_ji) < 1e-20                       # symmetric
            if (i + j) % 2 == 1:                                   # even/odd sectors decouple
                assert abs(q_ij) < 1e-20


def test_arithmetic_side_equals_zero_side(zeros):
    """Explicit formula: the matrix built from primes + Gamma factor + poles must
    reproduce 2 sum |f^(gamma_n)|^2 computed from the zeros (+ density tail)."""
    a = 0.8
    W = wf.build_weil_form(a, J=12, dps=30, estimate_quad_error=False)
    rng = np.random.default_rng(7)
    for parity in ("even", "odd"):
        Qs, Gs, idx = W.sector(parity)
        c = rng.standard_normal(len(idx))
        qa = sum(c[i] * c[j] * float(Qs[i, j].mid()) for i in range(len(idx)) for j in range(len(idx)))
        qz = wf.zero_side_value(c, idx, a, zeros)
        assert abs(qa - qz) / abs(qz) < 1e-6


def test_positivity_without_primes():
    """a < log(2)/2: no prime enters and Q_a is positive (Yoshida; Connes–Consani)."""
    W = wf.build_weil_form(0.3, J=12, dps=30, estimate_quad_error=True)
    assert W.n_prime_powers == 0
    for parity in ("even", "odd"):
        Qs, Gs, idx = W.sector(parity)
        vals, _ = wf.smallest_eigenpairs(Qs, Gs, dps=30, k=1)
        bound, positive = wf.weyl_certificate(W, parity, vals[0], 30)
        assert vals[0] > 0 and positive


def test_minimiser_fourier_transform_vanishes_at_first_zeta_zeros():
    """Support [-1, 1] (prime powers 2, 3, 4, 5, 7 only), 32 Legendre polynomials:
    the even minimiser's Fourier transform has a zero within 1e-12 of gamma_1 and
    within 1e-8 of gamma_2 (measured: ~1e-16 and ~2e-11)."""
    a, J, dps = 1.0, 32, 90
    W = wf.build_weil_form(a, J=J, dps=dps, estimate_quad_error=False)
    Qs, Gs, idx = W.sector("even")
    vals, vecs = wf.smallest_eigenpairs(Qs, Gs, dps=dps, k=1)
    assert 0 < vals[0] < 1e-15
    with mp.workdps(40):
        gam = [mp.zetazero(1).imag, mp.zetazero(2).imag]
        d = wf.zero_match(vecs[0], idx, a, gam, dps=40)
    assert d[0] is not None and d[0] < 1e-12
    assert d[1] is not None and d[1] < 1e-8


def test_spherical_bessel_recurrence():
    with mp.workdps(40):
        for n, z in [(0, 2.5), (7, 2.5), (30, 9.0), (12, 150.0)]:
            ref = mp.sqrt(mp.pi / (2 * mp.mpf(z))) * mp.besselj(n + mp.mpf(1) / 2, z)
            got = wf._spherical_jn_all(n, z, 40)[n]
            assert abs(got - ref) <= mp.mpf(10) ** -35 * abs(ref)
