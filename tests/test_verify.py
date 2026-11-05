"""Tests for rh.verify (Turing's method)."""
import json
import math
from dataclasses import replace

import mpmath as mp
import numpy as np
import pytest

import rh.verify as rv
import rh.zeta as zt


def test_turing_bound_and_constants():
    assert abs(rv.turing_bound(1000.0) - (2.30 + 0.128 * math.log(1000 / (2 * math.pi)))) < 1e-12
    assert rv.turing_bound(1000.0, "trudgian") < rv.turing_bound(1000.0, "lehman")
    assert rv.TURING_TMIN == 168 * math.pi
    assert zt.gram_point(rv.TURING_M_MIN) > rv.TURING_TMIN >= zt.gram_point(rv.TURING_M_MIN - 1)


def test_N_arb():
    assert rv.N_arb(1000.0) == 649
    assert rv.N_arb(10000.0) == 10142
    assert rv.N_arb(14.0) == 0 and rv.N_arb(15) == 1


def test_theta_integral_vs_mpmath():
    m = 648
    g_a, g_b = zt.gram_point(m), zt.gram_point(m + 3)
    with mp.workdps(30):
        ref = mp.quad(lambda t: zt.theta_mp(t, 30) / mp.pi - m, [g_a, g_b])
    val = rv.theta_integral(g_a, g_b, m)
    assert abs(val - float(ref)) < 1e-9
    # theta/pi - m runs almost linearly from 0 to k = 3, so the integral is ~ k L / 2
    assert 0.45 * 3 * (g_b - g_a) < val < 0.55 * 3 * (g_b - g_a)


def test_verify_rh_up_to_1000():
    res = rv.verify_rh_up_to(1000.0)
    assert res.certified
    assert res.N_T == 649 == res.n_zeros_found
    assert res.arb_N_T == 649 and res.arb_agrees
    assert res.max_abs_diff_vs_arb < 1e-8 and res.n_compared_vs_arb == 649
    c = res.certificate
    assert c.g_m >= 1000.0 and c.g_m > rv.TURING_TMIN
    assert c.k >= 1 and c.quotient < 1.0
    assert c.sign_changes_up_to_g_m == c.m + 1
    assert c.n_brackets_failed == 0 and c.n_fallback_mp == 0
    assert abs(c.L - (c.g_mk - c.g_m)) < 1e-9
    assert abs(c.B - rv.turing_bound(c.g_mk)) < 1e-12
    assert res.search_stats["unresolved_blocks"] == []
    json.dumps(res.to_dict())                       # serialisable summary
    assert "CERTIFIED" in res.summary()


def test_verify_small_T_certifies_beyond_168pi():
    res = rv.verify_rh_up_to(100.0, refine="brent")
    assert res.certified and res.N_T == 29 and res.arb_agrees
    assert res.certified_height > rv.TURING_TMIN
    assert res.certificate.sign_changes_up_to_g_m == res.certificate.m + 1


def test_trudgian_constants_certify_with_fewer_or_equal_intervals():
    a = rv.count_zeros_rigorously(1000.0, constants="lehman")
    b = rv.count_zeros_rigorously(1000.0, constants="trudgian")
    assert a.certified and b.certified and a.N_T == b.N_T == 649
    assert b.k <= a.k and b.B < a.B


def test_count_zeros_rigorously_10000():
    c = rv.count_zeros_rigorously(10000.0)
    assert c.certified and c.N_T == 10142 == rv.N_arb(10000.0)
    assert c.n_brackets_failed == 0


@pytest.fixture(scope="module")
def search_1000():
    return zt.find_zeros(0.0, rv.required_search_height(1000.0, 40, 40), details=True)


def test_missing_zero_breaks_certificate(search_1000):
    s = search_1000
    keep = np.ones(len(s), dtype=bool)
    keep[300] = False
    broken = replace(s, zeros=s.zeros[keep], lo=s.lo[keep], hi=s.hi[keep],
                     Z_lo=s.Z_lo[keep], Z_hi=s.Z_hi[keep])
    c = rv.count_zeros_rigorously(1000.0, search=broken)
    assert not c.certified and c.N_T is None
    assert all("sign changes" in why or "k <=" in why for _, why in c.tried)


def test_spurious_bracket_is_rejected_not_counted(search_1000):
    s = search_1000
    mid = 0.5 * (s.zeros[300] + s.zeros[301])       # no zero here
    i = 301
    ins = lambda arr, v: np.insert(arr, i, v)
    fake = replace(s, zeros=ins(s.zeros, mid), lo=ins(s.lo, mid - 1e-3), hi=ins(s.hi, mid + 1e-3),
                   Z_lo=ins(s.Z_lo, -1.0), Z_hi=ins(s.Z_hi, 1.0))   # lying about the signs
    cert = rv.certify_sign_changes(fake)
    assert cert.n_failed == 1 and not cert.ok[i]
    c = rv.count_zeros_rigorously(1000.0, search=fake)
    assert c.certified and c.N_T == 649 and c.n_brackets_failed == 1


def test_certify_sign_changes_tightens(search_1000):
    cert = rv.certify_sign_changes(search_1000)
    assert cert.ok.all() and cert.tightened.all()
    assert np.all(cert.hi - cert.lo <= 2 * rv.TIGHT_DELTA + 1e-9)
    assert np.all(cert.lo[1:] > cert.hi[:-1])
    assert np.all(np.sign(cert.Z_lo) * np.sign(cert.Z_hi) < 0)


def test_search_must_start_at_zero():
    s = zt.find_zeros(500.0, 1200.0, details=True)
    with pytest.raises(ValueError):
        rv.count_zeros_rigorously(1000.0, search=s)


def test_T_inside_a_bracket_is_certified_at_the_bracket_start():
    """If T lies strictly inside a certified sign-change bracket, N(T) is only
    determined up to +1; the certificate must then certify the count at the
    bracket's left end T' < T and say so (review finding)."""
    import rh.verify as rv
    from rh.data import load_zeros
    z = load_zeros(400)
    T = float(z[300])  # the float64 ordinate of the 301st zero lies inside its bracket
    r = rv.verify_rh_up_to(T, cross_check=True, verbose=False)
    c = r.certificate
    assert c.certified and c.T_ambiguous
    assert c.T_certified is not None and c.T_certified < T
    assert r.N_T == 300 and r.arb_N_T == 300 and r.arb_agrees
    assert "inside a sign-change bracket" in c.message
