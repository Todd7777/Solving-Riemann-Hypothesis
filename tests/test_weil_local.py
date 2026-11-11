"""Tests for rh.weil_local: wave packets as localised parts of the invisible function H.

Independent references: direct quadrature of int_0^inf H_loc G_0^2 (the band-limit identity), Arb
evaluation against the double-precision evaluator with its error bound, a dense float grid for the
trajectory inequality, and the true Weil form of rh.weil_form (the envelope form must stay below it).
"""
import math

import mpmath as mp
import numpy as np
import pytest
from flint import acb, arb, ctx
from scipy.special import digamma

from rh import weil_comb as wcb
from rh import weil_envelope as we
from rh import weil_form as wf
from rh import weil_local as wl

L12 = 1.2


@pytest.fixture(scope="module")
def l12():
    """Joint certificate at L = 1.2 and wave packets fitted on [130, 200] (where the plain trajectory check
    fails: its last violation is at t ~ 191.6)."""
    cert = wcb.joint_comb_certificate(L12, Lam=5.5)
    comb = wcb.prime_comb(L12)
    ns = np.array([n for t in comb.values() for n, _ in t], float)
    cs = np.array([float(c.mid()) for t in comb.values() for _, c in t])
    w = np.array([math.log(n / d) for n, d, _ in cert.terms])
    h = np.array([float(x.mid()) for _, _, x in cert.terms])

    def G(ts):
        return np.cos(np.outer(ts, np.log(ns))) @ cs - np.cos(np.outer(ts, w)) @ h

    def Om(ts):
        return np.real(digamma(0.25 + 0.5j * ts)) - math.log(math.pi)
    loc = wl.fit_local_correction(G, Om, L12, 130.0, 200.0, 0.05)
    return cert, loc, G, Om


def test_spectrum_check():
    wl.LocalCorrection(L=1.0, b=1.0, m=8, atoms=[(50.0, 3.0, 0, 1.0)])          # w - b = 2 = 2L: allowed
    with pytest.raises(ValueError):
        wl.LocalCorrection(L=1.0, b=1.0, m=8, atoms=[(50.0, 2.75, 1, 1.0)])     # w - b < 2L


def test_packets_are_invisible_to_the_window():
    """int_0^inf (psi(t - tau) + psi(-t - tau)) G_0(t)^2 dt = 0 when w - b >= 2L (G_0^2 has spectrum in
    [-2L, 2L]); a packet with w - b < 2L gives a clearly nonzero integral. Direct quadrature to t = 3000,
    with the remaining tail bounded by int |psi| G_0^2 <= 2 (m/b)^m / t^m * 2/L... (negligible here)."""
    mp.mp.dps = 20
    L = 1.0

    def integral(w, kind):
        loc = wl.LocalCorrection(L=0.0, b=1.0, m=8, atoms=[(60.0, w, kind, 1.0)])

        def f(t):
            ctx.dps = 20
            v = float(loc.H(acb(float(t))).real.mid())
            g0 = 2 * L * (mp.sin(L * t) / (L * t)) ** 2 if t else 2 * L
            return v * g0
        return mp.quad(f, mp.linspace(0, 3000, 1501))
    ok = integral(2 * L + 1.0, 0)
    ok2 = integral(2 * L + 1.5, 1)
    bad = integral(2 * L + 0.2, 0)
    ctx.dps = 15
    assert abs(ok) < 1e-9 and abs(ok2) < 1e-9
    assert abs(bad) > 1e-4


def test_float_evaluation_within_bound(l12):
    _, loc, _, _ = l12
    ctx.prec = 128
    starts = [arb(131.0), arb(166.25)]
    vals, errs = loc.eval_float(starts, 0.0137, 500)
    worst = 0.0
    for b, s in enumerate(starts):
        for k in (0, 3, 250, 499):
            v = loc.H(acb(s + k * arb(0.0137))).real
            assert abs(v - vals[b, k]) < errs[b]
            worst = max(worst, abs(float(v.mid()) - vals[b, k]))
    ctx.prec = 53
    assert worst < 1e-10 and max(errs) < 1e-6


def test_packets_lower_the_trajectory_cutoff(l12):
    """Without packets the trajectory check needs T > 191.6 at L = 1.2; with them it certifies T = 150, and a
    dense float grid confirms Omega - P_L + H + H_loc >= beta there."""
    cert, loc, G, Om = l12
    cert.local, cert.trajectory = None, None
    assert not wcb.trajectory_check(cert, 150.0, 0.05)["certified"]
    H_sup0 = cert.H_sup
    wcb.attach_local(cert, loc)
    try:
        r = wcb.trajectory_check(cert, 150.0, 0.05)
        assert r["certified"] and r["min_margin"] >= 0
        assert r["T2"] > max(t for t, _, _, _ in loc.atoms)
        ts = np.arange(150.0, 260.0, 0.004)
        ctx.prec = 64
        Hl = np.array([float(loc.H(acb(float(t))).real.mid()) for t in ts[::25]])
        ctx.prec = 53
        assert (Om(ts[::25]) - G(ts[::25]) + Hl - 0.05).min() > 0
    finally:
        cert.local, cert.trajectory, cert.H_sup = None, None, H_sup0


def _true_form_orthonormal(L, N, parity, dps=40):
    W = wf.build_weil_form(L, 2 * N, dps=dps)
    Qs, Gs, idx = W.sector(parity)
    Q = np.array([[float(Qs[i, j].mid()) for j in range(N)] for i in range(N)])
    nrm = np.array([math.sqrt((2 * n + 1) / (2 * L)) for n in idx])
    return Q * np.outer(nrm, nrm), np.array(idx)


def test_envelope_with_packets(l12):
    """With the packets in H, the envelope form (packets evaluated in Arb at every node, bounded on the
    ellipses) stays below the true form, and at T = 240 (above the zero-matching floor, ~1.1 x 2 pi e^{1+2L})
    it certifies the even sector."""
    cert, loc, _, _ = l12
    H_sup0 = cert.H_sup
    wcb.attach_local(cert, loc)
    try:
        assert wcb.trajectory_check(cert, 240.0, 0.05)["certified"]
        N = 12
        blk = we.envelope_block(L12, 240.0, N, "even", dps=40, comb_cert=cert, wmax=8.0)
        R = np.array([[float(blk["R"][i, j].mid()) for j in range(N)] for i in range(N)])
        Q, _ = _true_form_orthonormal(L12, N, "even")
        assert np.linalg.eigvalsh(Q - R).min() > -1e-13
        Nfull = int(math.ceil(1.25 * math.e * L12 * 240.0 / 4)) + 20
        res = we.certify_window(L12, "even", T=240.0, N=Nfull, dps=68, comb_cert=cert, wmax=8.0)
        assert res["positive"] and res["certified_lower_bound"] > 1e-51
    finally:
        cert.local, cert.trajectory, cert.H_sup = None, None, H_sup0
        ctx.dps = 15
