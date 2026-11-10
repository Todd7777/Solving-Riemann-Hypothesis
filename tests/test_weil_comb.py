"""Tests for rh.weil_comb: non-pointwise certificates A' - P_L(t) + H(t) >= 0 for the prime comb.

Independent references: the Fejér–Egerváry–Szász closed form (one cosine), the Toeplitz eigenvalue
(best separable constant), a dense grid evaluation of A' - P_L + H, the exact identity
int_0^inf cos(w t) sin(t)^2 / t^2 dt = 0 for w >= 2 (via the sine integral), and the true Weil form
of rh.weil_form (the comb-certified envelope form must lie below it).
"""
import math

import mpmath as mp
import numpy as np
import pytest
from flint import acb, arb, ctx

from rh import weil_comb as wcb
from rh import weil_envelope as we
from rh import weil_form as wf


@pytest.mark.parametrize("M", [2, 3, 6, 12])
def test_single_cosine_matches_fejer_egervary_szasz(M):
    R, info = wcb.sos_gram([1.0], M)
    s = wcb._autocorrelations(R, M)
    A = float((s[0] + abs(2 * s[1] + 1)).mid())
    best = wcb.fes_constant(1.0, M)               # 1 / (2 cos(pi/(M+2)))
    assert best - 1e-9 < A < best + 1e-5
    assert abs(info["dual"] - best) < 1e-7        # barrier dual value ~ optimum


def test_prime_comb_and_constants():
    comb = wcb.prime_comb(1.1)                    # e^{2.2} = 9.025: 2, 4, 8 | 3, 9 | 5 | 7
    assert {p: [n for n, _ in t] for p, t in comb.items()} == {2: [2, 4, 8], 3: [3, 9], 5: [5], 7: [7]}
    cert = wcb.comb_certificate(1.1, X=24.0)
    d = cert.describe()
    assert abs(d["A_L"] - 7.0750) < 1e-3
    sep = d["separable_optimum"]                  # sum of Toeplitz lambda_max
    assert abs(sep - 3.7671) < 1e-3
    assert sep < d["A_prime"] < sep * 1.03 < d["A_L"]
    for pp in d["per_prime"]:
        assert pp["dual_degree_M"] - 1e-6 < pp["A_p"] < pp["dual_degree_M"] + 1e-3
        assert pp["lambda_toeplitz"] < pp["A_p"] < pp["sum_c"]
    # every frequency of H exceeds 2L, every coefficient is exact
    assert cert.omega_min > 2 * 1.1
    assert all(h.rad() == 0 for _, _, h in cert.terms)


@pytest.mark.parametrize("L", [0.8, 1.0, 1.2])
def test_certificate_inequality_on_a_grid(L):
    """A' - P_L(t) + H(t) >= 0: dense grid (floats) and Arb at random points, where it is >= S_p >= 0."""
    cert = wcb.comb_certificate(L, X=16.0)
    comb = wcb.prime_comb(L)
    ts = np.linspace(0, 3000, 600001)
    P = sum(float(c.mid()) * np.cos(ts * math.log(n)) for t in comb.values() for n, c in t)
    H = sum(float(h.mid()) * np.cos(ts * math.log(n / d)) for n, d, h in cert.terms)
    v = float(cert.A_prime.mid()) - P + H
    assert v.min() > -1e-9
    assert float(cert.A_L.mid()) - P.max() < 1e-2       # the pointwise constant is nearly attained
    ctx.dps = 30
    rng = np.random.default_rng(1)
    for t in rng.uniform(0, 1e4, 20):
        tt = arb(float(t))
        val = cert.A_prime - sum((c * (tt * arb(n).log()).cos() for t_ in comb.values() for n, c in t_), arb(0))
        val += cert.H(tt).real
        assert val > 0
    ctx.dps = 15


def test_band_limit_identity_kills_H():
    """int_0^inf cos(w t) G_0(t)^2 dt = 0 for w >= 2L (here exactly, via Si): the reason H is free."""
    mp.mp.dps = 30
    L, T = mp.mpf(1), mp.mpf(50)
    w = 3 * mp.log(2)                              # the lowest frequency of H at L = 1

    def tail(a):                                   # int_T^inf cos(a t)/t^2 dt
        return mp.cos(a * T) / T + a * mp.si(a * T) - a * mp.pi / 2
    head = mp.quad(lambda t: mp.cos(w * t) * 2 * (mp.sin(t) / t) ** 2 if t else 2, mp.linspace(0, T, 201))
    tl = 2 * (tail(w) - (tail(w - 2) + tail(w + 2)) / 2) / 2
    assert abs(head) > 1e-4 and abs(head + tl) < 1e-20


def _true_form_orthonormal(L, N, parity, dps=40):
    W = wf.build_weil_form(L, 2 * N, dps=dps)
    Qs, Gs, idx = W.sector(parity)
    Q = np.array([[float(Qs[i, j].mid()) for j in range(N)] for i in range(N)])
    nrm = np.array([math.sqrt((2 * n + 1) / (2 * L)) for n in idx])
    return Q * np.outer(nrm, nrm), np.array(idx)


@pytest.mark.parametrize("L,T", [(0.5, 40.0), (1.0, 300.0)])
def test_comb_envelope_is_below_the_true_form(L, T):
    """R''_T <= Q on the same modes (Q_A - R''_A positive semidefinite), and the comb certificate gives a
    larger lower form than the pointwise envelope where both apply."""
    N = 14
    cert = wcb.comb_certificate(L, X=16.0)
    blk = we.envelope_block(L, T, N, "even", dps=40, comb_cert=cert, wmax=8.0)
    R = np.array([[float(blk["R"][i, j].mid()) for j in range(N)] for i in range(N)])
    Q, idx = _true_form_orthonormal(L, N, "even")
    assert list(idx) == blk["orders"]
    assert np.linalg.eigvalsh(Q - R).min() > -1e-13
    assert max(float(blk["R"][i, j].rad()) for i in range(N) for j in range(N)) < 1e-30
    assert float(blk["beta"].mid()) > 0


def test_comb_certified_window_beats_pointwise():
    ctx.dps = 40
    cert = wcb.comb_certificate(0.5, X=12.0)
    a = we.certify_window(0.5, "even", T=40.0, N=37, dps=40)
    b = we.certify_window(0.5, "even", T=40.0, N=37, dps=40, comb_cert=cert)
    assert a["positive"] and b["positive"]
    assert b["certified_lower_bound"] > a["certified_lower_bound"]
    assert b["A_prime"] < b["A_L"]
    ctx.dps = 15


def test_blocked_ldl_residual_and_inertia():
    ctx.dps = 40
    rng = np.random.default_rng(3)
    n = 150                                        # several 64-blocks
    Qm, _ = np.linalg.qr(rng.standard_normal((n, n)))
    D = np.concatenate([[1e-20, 3e-15], rng.uniform(0.5, 2.0, n - 2)])
    M = Qm @ np.diag(D) @ Qm.T
    from flint import arb_mat
    A = arb_mat([[arb(float(M[i, j])) for j in range(n)] for i in range(n)])
    # the float matrix has lambda_min ~ 1e-20 only to float accuracy: shift well inside the spectrum
    ok = we.residual_certify(A, arb(1e-30))
    lam = np.linalg.eigvalsh(M).min()
    if lam > 1e-28:
        assert ok["certified"]
    assert we.inertia_count(A, arb(0.75)) >= 2
    neg = A - arb_mat(n, n, [arb(1.0) if i == j else 0 for i in range(n) for j in range(n)])
    k = int(np.sum(np.linalg.eigvalsh(M) < 1.0))
    assert we.inertia_count(neg, None) == k
    assert not we.residual_certify(neg, arb(0))["certified"]
    ctx.dps = 15


def test_joint_certificate_is_valid_and_beats_separable():
    """Joint torus SOS certificate at L = 0.9 (primes 2, 3, 5) with monomials n <= e^5: valid on a grid and in
    Arb, every frequency of H is log(n/d) >= 2L, A' is close to its barrier dual value and below the separable
    optimum: the composite constraints (e.g. no mass at frequency log 6 > 2L) are what the separable bound
    ignores."""
    L = 0.9
    cert = wcb.joint_comb_certificate(L, Lam=5.0)
    d = cert.describe()
    assert d["kind"] == "joint" and d["n_terms"] > 0
    assert d["dual"] - 1e-6 < d["A_prime"] < d["dual"] + 5e-3
    assert d["A_prime"] < d["separable_optimum"] - 0.1 < d["A_L"]
    assert cert.omega_min >= 2 * L
    assert all(h.rad() == 0 for _, _, h in cert.terms)
    assert all(n > dd >= 1 and math.log(n / dd) <= 5.0 + 1e-9 for n, dd, _ in cert.terms)
    comb = wcb.prime_comb(L)
    ts = np.linspace(0, 2000, 400001)
    P = sum(float(c.mid()) * np.cos(ts * math.log(n)) for t in comb.values() for n, c in t)
    H = sum(float(h.mid()) * np.cos(ts * math.log(n / dd)) for n, dd, h in cert.terms)
    assert (float(cert.A_prime.mid()) - P + H).min() > -1e-9
    ctx.dps = 30
    for t in np.random.default_rng(2).uniform(0, 1e4, 10):
        tt = arb(float(t))
        val = cert.A_prime - sum((c * (tt * arb(n).log()).cos() for t_ in comb.values() for n, c in t_), arb(0))
        assert val + cert.H(tt).real > 0
    ctx.dps = 15


def test_admm_joint_certificate_matches_barrier_and_is_valid():
    """The ADMM route (used for the large monomial sets at L = 1.5) reaches the barrier optimum on a small
    case, and the certificate built from it is valid: the mismatches it leaves are absorbed into A'."""
    L, Lam = 0.9, 5.0
    bar = wcb.joint_comb_certificate(L, Lam).describe()
    cert = wcb.joint_comb_certificate(L, Lam, solver="admm", admm_iters=4000)
    d = cert.describe()
    assert d["solver"] == "admm"
    assert abs(d["A_prime"] - bar["A_prime"]) < 2e-3
    assert abs(d["A_prime"] - d["admm_value"]) < 1e-9      # the exact certificate reproduces the ADMM value
    comb = wcb.prime_comb(L)
    ts = np.linspace(0, 2000, 400001)
    P = sum(float(c.mid()) * np.cos(ts * math.log(n)) for t in comb.values() for n, c in t)
    H = sum(float(h.mid()) * np.cos(ts * math.log(n / dd)) for n, dd, h in cert.terms)
    assert (float(cert.A_prime.mid()) - P + H).min() > -1e-9
    assert cert.omega_min >= 2 * L


def test_low_rank_alm_joint_certificate():
    """The low-rank augmented Lagrangian (meant for monomial sets of thousands) reaches the barrier optimum on
    a small case, and its factor gives a valid certificate whose A' reproduces the solver's value."""
    L, Lam = 1.0, 4.5
    G = wcb.joint_gram_alm(L, Lam, rank=43, outer=15, inner=200)
    bar = wcb.joint_comb_certificate(L, Lam).describe()["A_prime"]
    assert abs(G["admm_value"] - bar) < 2e-3
    cert = wcb.joint_comb_certificate(L, Lam, gram=G)
    assert abs(float(cert.A_prime.mid()) - G["admm_value"]) < 1e-9
    comb = wcb.prime_comb(L)
    ts = np.linspace(0, 2000, 200001)
    P = sum(float(c.mid()) * np.cos(ts * math.log(n)) for t in comb.values() for n, c in t)
    H = sum(float(h.mid()) * np.cos(ts * math.log(n / dd)) for n, dd, h in cert.terms)
    assert (float(cert.A_prime.mid()) - P + H).min() > -1e-9
