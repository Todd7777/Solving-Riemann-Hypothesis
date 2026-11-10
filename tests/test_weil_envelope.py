"""Tests for rh.weil_envelope: certified positivity of Weil's form on a whole window [-L, L].

Independent references: mpmath (Bessel functions), the u-space Legendre matrices of rh.weil_form
(tested against the zero side in tests/test_weil_form.py), dense float eigenvalues, and Zhu's
published certificate at L = 0.8.
"""
import math

import mpmath as mp
import numpy as np
import pytest
from flint import arb, arb_mat, ctx

from rh import weil_envelope as we
from rh import weil_form as wf


@pytest.mark.parametrize("x", [0.004, 2.5, 37.75, 211.125, 900.5])
def test_spherical_bessel_against_mpmath(x):
    ctx.dps = 30
    js = we.sph_bessel_j_all(arb(x), 700)
    with mp.workdps(60):
        for n in (0, 1, 13, 120, 350, 699):
            ref = mp.sqrt(mp.pi / (2 * mp.mpf(x))) * mp.besselj(n + mp.mpf(1) / 2, mp.mpf(x))
            ctx.dps = 70
            r = arb(mp.nstr(ref, 65))
            ctx.dps = 30
            assert js[n].overlaps(r)
    assert max(float(v.rad()) for v in js) < 1e-28


def test_comb_constant_and_envelope():
    ctx.dps = 30
    c, A = we.comb(0.8)
    assert [n for n, _ in c] == [2, 3, 4]
    assert abs(float(A.mid()) - 2.9419735) < 1e-6           # Zhu: A_0.8 ~ 2.9420
    c, A = we.comb(0.81)
    assert [n for n, _ in c] == [2, 3, 4, 5]                # log 5 / 2 = 0.8047 < 0.81
    assert abs(float(A.mid()) - 4.3814986) < 1e-6
    # Omega is increasing on t > 0, so Psi_L(t) >= beta for t >= T
    om = [float(we.omega(arb(t)).real.mid()) for t in (0.0, 1.0, 5.0, 20.0, 200.0)]
    assert all(a < b for a, b in zip(om, om[1:]))
    assert float(we.beta_star(0.8, 200.0).mid()) > 0.5
    assert float(we.beta_star(0.81, 400.0).mid()) < 0 < float(we.beta_star(0.81, 600.0).mid())


def _true_form_orthonormal(L, N, parity, dps=40):
    W = wf.build_weil_form(L, 2 * N, dps=dps)
    Qs, Gs, idx = W.sector(parity)
    Q = np.array([[float(Qs[i, j].mid()) for j in range(N)] for i in range(N)])
    nrm = np.array([math.sqrt((2 * n + 1) / (2 * L)) for n in idx])
    return Q * np.outer(nrm, nrm), np.array(idx)


@pytest.mark.parametrize("parity", ["even", "odd"])
def test_envelope_block_is_below_the_true_form(parity):
    """R_T <= Q: on the same Legendre modes, Q_A - R_A is positive semidefinite (it is the matrix of
    (1/pi) int_T^inf (Psi_L - beta) |F|^2 >= 0), and the smallest eigenvalues are ordered."""
    L, T, N = 0.5, 40.0, 10
    blk = we.envelope_block(L, T, N, parity, dps=40)
    R = np.array([[float(blk["R"][i, j].mid()) for j in range(N)] for i in range(N)])
    Q, idx = _true_form_orthonormal(L, N, parity)
    assert list(idx) == blk["orders"]
    assert np.linalg.eigvalsh(Q - R).min() > -1e-14
    assert 0 < np.linalg.eigvalsh(R).min() < np.linalg.eigvalsh(Q).min()
    assert max(float(blk["R"][i, j].rad()) for i in range(N) for j in range(N)) < 1e-30


def test_residual_certificate():
    ctx.dps = 30
    D = [3.0, 1e-12, 2.0]
    rng = np.random.default_rng(0)
    Qm, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    M = Qm @ np.diag(D) @ Qm.T
    A = arb_mat([[arb(float(M[i, j])) for j in range(3)] for i in range(3)])
    ok = we.residual_certify(A, arb(0.5e-12))
    assert ok["certified"] and 0 < float(ok["lower_bound"].lower()) < 1e-12
    bad = we.residual_certify(A, arb(2e-12))            # shift above lambda_min: must not certify
    assert not bad["certified"]
    neg = arb_mat([[arb(1), arb(2)], [arb(2), arb(1)]])  # eigenvalues 3, -1
    assert not we.residual_certify(neg, arb(0))["certified"]


@pytest.mark.parametrize("parity,lo,hi", [("even", 5.6e-7, 5.8e-7), ("odd", 1.06e-4, 1.09e-4)])
def test_certified_positivity_small_window(parity, lo, hi):
    """Q(f) >= c ||f||^2 for every real f of the sector supported in [-1/2, 1/2] (only n = 2 in the
    comb).  The certified c lies below the true minimum (about 9.4e-7 even, 1.96e-4 odd; E10)."""
    r = we.certify_window(0.5, parity, T=40.0, N=24, dps=40)
    assert r["positive"] and lo < r["certified_lower_bound"] < hi
    assert r["eps_B"] < 1e-10 and r["residual_F"] < 1e-25


def test_reproduces_zhu_at_0_8():
    """Zhu (arXiv:2608.24827) certified Q(f) >= 8.9e-18 ||f||^2 on [-0.8, 0.8] with T = 200 and
    200 even modes; the same reduction in ball arithmetic certifies 1.01e-17 (our beta uses Omega(T)
    itself instead of the bound log(T/2pi) - 1/T)."""
    r = we.certify_window(0.8, "even", T=200.0, N=200, dps=40)
    assert r["positive"] and 1.0e-17 < r["certified_lower_bound"] < 1.64e-17


def test_paired_sectors_match_separate_assembly():
    """envelope_blocks shares nodes, symbol and Bessel values between the two sectors; the result must
    agree (ball overlap, every entry) with assembling each sector on its own."""
    both = we.envelope_blocks(0.5, 40.0, {"even": 12, "odd": 12}, dps=40, chunk=50)
    for parity in ("even", "odd"):
        one = we.envelope_block(0.5, 40.0, 12, parity, dps=40, chunk=50)
        A, B = both[parity]["R"], one["R"]
        assert all(A[i, j].overlaps(B[i, j]) for i in range(12) for j in range(12))
        assert both[parity]["orders"] == one["orders"]
