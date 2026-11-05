"""Tests for rh.debruijn_newman (all fast: < 60 s in total)."""
import math

import numpy as np
import mpmath as mp
import pytest

from rh.data import load_zeros
import rh.debruijn_newman as dn


@pytest.fixture(scope="module")
def gam():
    return load_zeros(10000)


def test_H0_vanishes_at_first_zero_and_matches_xi():
    g1 = 14.134725141734693
    h00 = dn.H_t(0, 0, dps=20)
    assert abs(float(h00) - 0.06214009727353926) < 1e-12          # xi(1/2)/8
    h = dn.H_t(0, 2 * g1, dps=20)
    assert abs(float(h) / float(h00)) < 1e-10
    for z in (10.0, 30.0, 57.0):
        hq = dn.H_t(0, z, dps=20)
        hx = dn.H0_from_xi(z, dps=20)
        assert abs(float(hq - hx)) < 1e-15 * abs(float(h00))


def test_tail_field_closed_form_vs_quadrature():
    X = 5031.86
    for x in (28.27, 475.5, 4500.0, 5020.0):
        closed = float(dn.tail_field(x, X))
        f = lambda v: mp.log(X / (v * 4 * mp.pi)) / (4 * mp.pi) * 4 * x / (x * x - (X / v) ** 2) * X / (v * v)
        ref = float(mp.quad(f, [0, 0.5, 0.9, 0.99, 1]))
        assert abs(closed - ref) < 1e-9 * max(1.0, abs(ref))
    # derivative
    x = 3000.0
    h = 1e-4
    fd = (dn.tail_field(x + h, X) - dn.tail_field(x - h, X)) / (2 * h)
    assert abs(float(dn.tail_field_derivative(x, X)) - fd) < 1e-8


def test_mean_drift_is_minus_pi_over_4(gam):
    x = 2 * gam[:3000]
    v = dn.velocity_field(x, 0.0, None, dn.tail_start(3000))
    assert abs(np.median(v[:1500]) + math.pi / 4) < 0.03


def test_jacobian_matches_finite_differences(gam):
    x = 2 * gam[:30]
    xf = 2 * gam[30:300]
    X = dn.tail_start(300)
    J = dn.velocity_jacobian(x, 0.0, xf, X)
    h = 1e-6
    for k in (0, 7, 29):
        e = np.zeros_like(x)
        e[k] = h
        col = (dn.velocity_field(x + e, 0, xf, X) - dn.velocity_field(x - e, 0, xf, X)) / (2 * h)
        assert np.abs(J[:, k] - col).max() < 1e-7


def test_forward_flow_keeps_order_and_increases_min_spacing(gam):
    x0 = 2 * gam[:50]
    res = dn.zero_dynamics(x0, (0.0, 0.05), t_eval=[0.0, 0.025, 0.05])
    assert res.status == 0
    for i in range(3):
        assert np.all(np.diff(res.x[i]) > 0)
    m = [res.min_spacing(i) for i in range(3)]
    assert m[0] < m[1] < m[2]
    # all spacings grow where pairs are close: the smallest spacing of the first
    # 50 zeros grows roughly like sqrt(Delta^2 + 8 t)
    assert m[2] > math.sqrt(m[0] ** 2 + 8 * 0.05) * 0.9


def test_backward_flow_decreases_min_spacing(gam):
    x0 = 2 * gam[:50]
    res = dn.zero_dynamics(x0, (0.0, -0.05), t_eval=[0.0, -0.05])
    assert res.min_spacing(1) < res.min_spacing(0)
    assert np.all(np.diff(res.x[1]) > 0)


def test_implicit_methods_agree_with_dop853(gam):
    x0 = 2 * gam[:30]
    a = dn.zero_dynamics(x0, (0.0, 0.05), t_eval=[0.05], method="DOP853")
    b = dn.zero_dynamics(x0, (0.0, 0.05), t_eval=[0.05], method="Radau")
    assert np.abs(a.x[-1] - b.x[-1]).max() < 1e-8


def test_lehmer_pair_7005(gam):
    x = 2 * gam
    r = dn.lehmer_pair_criterion(x, 6708)
    assert r["k"] == 6709
    assert abs(r["gamma_k"] - 7005.0628661749) < 1e-8
    assert abs(r["gamma_k1"] - 7005.1005646726) < 1e-8
    assert r["is_lehmer"]
    assert 0.005 < r["Delta2_g"] < 0.01
    assert abs(r["t_two_body"] + 7.1059e-4) < 1e-7
    assert r["lambda_heuristic"] < r["t_frozen_background"] < r["t_two_body"] < 0
    assert r["lambda_crude"] < r["lambda_heuristic"]


def test_collision_time_of_lehmer_pair(gam):
    x = 2 * gam
    col = dn.estimate_collision_time(x, 6708, n_window=20)
    assert col.status == 1 and col.t_star is not None
    assert col.t_star < col.t_two_body < 0
    assert abs(col.t_star - col.t_two_body) < 0.01 * abs(col.t_two_body)
    assert abs(col.t_star - col.t_frozen_background) < 1e-8
    assert abs(col.x_pair[-1, 1] - col.x_pair[-1, 0]) < 1e-6
    assert np.all(np.diff(col.x_window[-1]) >= 0)


def test_find_lehmer_pairs(gam):
    x = 2 * gam
    pairs = dn.find_lehmer_pairs(x, n_best=3)
    assert len(pairs) == 3
    assert all(p["is_lehmer"] for p in pairs)
    assert 6709 in [p["k"] for p in pairs]


def test_spacing_statistics_rigidity_small(gam):
    rig = dn.flow_spacing_rigidity(gam, n_active=200, n_frozen=300, n_stat=150, t_final=0.2,
                                   t_eval=(0.0, 0.2))
    assert rig["ratio_norm_std"] < 0.9
    assert rig["min_spacing"][-1] > rig["min_spacing"][0]
