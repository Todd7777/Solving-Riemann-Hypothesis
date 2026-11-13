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


def test_shift_retry_when_the_estimate_is_too_high(monkeypatch):
    """If the Rayleigh-quotient estimate of lambda_min is too high (two close eigenvalues), the first shifted
    LDL^T check fails although the midpoint is positive definite; smaller shifts are then tried on the same
    block, and the certified bound stays valid (below the true lambda_min)."""
    ctx.dps = 40
    good = we.certify_window(0.5, "even", T=40.0, N=37, dps=40)
    real = we.approx_lambda_min_ldl

    def inflated(M, *a, **k):
        lam, x = real(M, *a, **k)
        return lam * 1.5, x
    monkeypatch.setattr(we, "approx_lambda_min_ldl", inflated)
    r = we.certify_window(0.5, "even", T=40.0, N=37, dps=40)
    assert r["positive"] and r["shift_retries"]
    assert r["certified_lower_bound"] <= good["lambda_approx"] * 1.0001
    ctx.dps = 15


def test_panels_at_high_precision_do_not_overflow():
    """The node count of a panel comes from log(const) - log(target). Formed as log(const / target) it overflowed
    a double for targets near 1e-300 (high precision, large H) although both logarithms are moderate. More
    precision needs more nodes, never fewer, on the same panels."""
    fine = we.panels(1.7, 200.0, 100, target=1e-305)
    coarse = we.panels(1.7, 200.0, 100, target=1e-60)
    assert fine and all(p.nu >= 4 for p in fine)
    assert len(fine) == len(coarse)
    assert all(p.lo == q.lo and p.hi == q.hi and p.nu >= q.nu for p, q in zip(fine, coarse))


def test_symblocks_storage():
    """SymBlocks keeps the lower block triangle (last block smaller); reads are symmetric, writes keep diagonal
    blocks symmetric, and the dense round trip is exact."""
    ctx.dps = 30
    rng = np.random.default_rng(1)
    n = 23
    X = rng.standard_normal((n, n))
    X = X + X.T
    M = arb_mat([[arb(float(X[i, j])) for j in range(n)] for i in range(n)])
    S = we.SymBlocks.from_arb_mat(M, b=5)
    assert S.nb == 5 and S.sizes == [5, 5, 5, 5, 3]
    assert sorted(S.blocks) == [(I, J) for I in range(5) for J in range(I + 1)]
    D = S.to_arb_mat()
    assert all(D[i, j] == M[i, j] for i in range(n) for j in range(n))
    assert all(S[i, j] == S[j, i] == M[i, j] for i in range(n) for j in range(n))
    S[3, 17] = arb(7)
    S[11, 13] = arb(-2)                       # inside a diagonal block
    assert S[17, 3] == 7 and S[13, 11] == -2 and S.blocks[(2, 2)][3, 1] == -2 and S.blocks[(2, 2)][1, 3] == -2
    ctx.dps = 15


@pytest.mark.parametrize("parity", ["even", "odd"])
def test_block_storage_matches_dense(parity):
    """storage='blocks' (lower block triangle, one product per block pair, the scale of G_n applied at the
    end) gives the R of the dense assembly: every entry overlaps, midpoints agree to rounding.  The blocked
    LDL^T, the inverse iteration and the residual certificate agree with the dense ones, and a shift above
    lambda_min is refused."""
    L, T, N = 0.5, 40.0, 37
    dn = we.envelope_block(L, T, N, parity, dps=40, chunk=50)
    bl = we.envelope_block(L, T, N, parity, dps=40, chunk=50, storage="blocks", block=8)
    A, B = dn["R"], bl["R"]
    assert isinstance(B, we.SymBlocks) and B.nb == 5 and bl["storage"] == "blocks"
    assert all(A[i, j].overlaps(B[i, j]) for i in range(N) for j in range(N))
    assert max(abs(float((A[i, j] - B[i, j]).mid())) for i in range(N) for j in range(N)) < 1e-37
    ctx.dps = 40
    fd, fb = we.ldl_mid_blocked(A, None), we.ldl_mid_blocked(B, None)
    assert fd["n_negative"] == fb["n_negative"] == 0
    assert max(abs(float(((x - y) / x).mid())) for x, y in zip(fd["d"], fb["d"])) < 1e-30
    lam_d, lam_b = we.approx_lambda_min_ldl(A)[0], we.approx_lambda_min_ldl(B)[0]
    assert abs(float(lam_b.mid()) / float(lam_d.mid()) - 1) < 1e-12
    lam0 = arb(float(lam_d.mid()) * 0.98)
    rd, rb = we.residual_certify(A, lam0), we.residual_certify(B, lam0)
    assert rd["certified"] and rb["certified"] and float(rb["residual_F"].upper()) < 1e-30
    assert not we.residual_certify(B, arb(float(lam_d.mid()) * 1.05))["certified"]
    ctx.dps = 15


def test_block_certify_windows_both_sectors():
    """certify_windows with block storage (shared nodes, both sectors, blocks of 7 with a smaller last one)
    reproduces the certificates of the dense path."""
    rd = we.certify_windows(0.5, 40.0, {"even": 30, "odd": 30}, dps=40, chunk=50, storage="dense")
    rb = we.certify_windows(0.5, 40.0, {"even": 30, "odd": 30}, dps=40, chunk=50, storage="blocks", block=7)
    for p in ("even", "odd"):
        assert rb[p]["positive"] and rb[p]["storage"] == "blocks" and rd[p]["storage"] == "dense"
        assert abs(rb[p]["certified_lower_bound"] / rd[p]["certified_lower_bound"] - 1) < 1e-9


def test_block_ldl_on_an_indefinite_matrix():
    """On an indefinite matrix the blocked LDL^T counts the negative eigenvalues (Sylvester) like the dense one,
    nothing is certified, and the inverse iteration finds the eigenvalue closest to 0."""
    ctx.dps = 30
    rng = np.random.default_rng(3)
    n = 19
    Qm, _ = np.linalg.qr(rng.standard_normal((n, n)))
    ev = np.linspace(1, 3, n)
    ev[4], ev[11], ev[7] = -0.37, -2.0, 0.011
    X = Qm @ np.diag(ev) @ Qm.T
    M = arb_mat([[arb(float(X[i, j])) for j in range(n)] for i in range(n)])
    S = we.SymBlocks.from_arb_mat(M, b=4)
    assert we.inertia_count(S, None) == we.inertia_count(M, None) == 2
    assert not we.residual_certify(S, arb(0))["certified"]
    assert we.approx_lambda_min_ldl(S) is None
    assert abs(float(we.approx_lambda_min(S)[0].mid()) - 0.011) < 1e-9
    assert abs(float(we.approx_lambda_min(M)[0].mid()) - 0.011) < 1e-9
    ctx.dps = 15


def test_block_assembly_checkpoint_and_resume(tmp_path):
    """envelope_blocks(checkpoint=...): an assembly stopped after its first checkpoint and resumed from it gives
    the same R_A as one run through with the same checkpoints (midpoints equal; radii within one unit of the
    30-bit radius mantissa, the restored balls containing the saved ones), and a checkpoint of another assembly
    is not used."""
    from flint import ctx
    L, T, N = 0.5, 40.0, 30
    kw = dict(dps=40, chunk=50, storage="blocks", block=7, checkpoint_every=4)
    ref = we.envelope_blocks(L, T, {"even": N, "odd": N}, checkpoint=str(tmp_path / "ref.pkl"), **kw)
    path = str(tmp_path / "ck.pkl")
    with pytest.raises(we._CheckpointStop):
        we.envelope_blocks(L, T, {"even": N, "odd": N}, checkpoint=path, _stop_after_checkpoint=1, **kw)
    res = we.envelope_blocks(L, T, {"even": N, "odd": N}, checkpoint=path, **kw)
    assert ref["even"]["resumed_from_panel"] == 0 and res["even"]["resumed_from_panel"] == 4
    ctx.dps = 40
    for par in ("even", "odd"):
        A, B = ref[par]["R"], res[par]["R"]
        assert res[par]["n_nodes"] == ref[par]["n_nodes"]
        for i in range(N):
            for j in range(N):
                a, b = A[i, j], B[i, j]
                assert a.mid() == b.mid()
                assert b.rad() >= a.rad() and b.rad() <= a.rad() * (1 + 1e-7) + arb(2) ** -200
    other = we.envelope_blocks(L, T + 1.0, {"even": N, "odd": N}, checkpoint=path, **kw)   # another assembly
    assert other["even"]["resumed_from_panel"] == 0
    ctx.dps = 15
