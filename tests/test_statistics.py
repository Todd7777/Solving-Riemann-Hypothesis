"""Fast tests for rh.statistics (GUE laws, unfolding, pair correlation, KS, Lehmer)."""
import math

import numpy as np
import pytest
from scipy import integrate

from rh import statistics as st
from rh.data import load_zeros


@pytest.fixture(scope="module")
def zeros_1e5():
    return load_zeros(100000)


@pytest.fixture(scope="module")
def gue():
    return st.gue_table()


# ---------------------------------------------------------------- GUE laws --
def test_gue_density_normalised_and_unit_mean(gue):
    s = np.linspace(0, 6, 6001)
    p = gue.pdf(s)
    assert abs(integrate.simpson(p, x=s) - 1) < 1e-3
    assert abs(integrate.simpson(s * p, x=s) - 1) < 1e-3
    # the Chebyshev-free quadrature of the class agrees too, and tightly
    assert abs(gue.moment(0) - 1) < 1e-8
    assert abs(gue.mean - 1) < 1e-8
    # GUE spacing variance is 0.1800 (Mehta); the Wigner surmise gives 3pi/8-1 = 0.1781
    assert abs(gue.variance - 0.180) < 1e-3
    assert np.all(p >= 0)


def test_gue_gap_probability_converged_and_small_s_series():
    for s in [0.5, 1.0, 2.0, 4.0, 6.0]:
        assert abs(st.gue_gap_probability(s, m=40) - st.gue_gap_probability(s, m=90)) < 1e-13
    assert st.gue_gap_probability(0.0) == 1.0
    # E_2(0; s) = 1 - s + (pi^2/36) s^4 - (pi^4/675) s^6 + O(s^8)   (integrating Mehta's series for p_2)
    for s in [0.02, 0.05]:
        assert abs(st.gue_gap_probability(s) - (1 - s + np.pi ** 2 * s ** 4 / 36 - np.pi ** 4 * s ** 6 / 675)) < 1e-10
    # p_2(s) = (pi^2/3) s^2 - (2 pi^4/45) s^4 + (pi^6/315) s^6 + O(s^8)
    tab = st.gue_table()
    for s in [0.02, 0.05]:
        taylor = (np.pi ** 2 / 3) * s ** 2 - (2 * np.pi ** 4 / 45) * s ** 4 + (np.pi ** 6 / 315) * s ** 6
        assert abs(float(tab.pdf(s)) - taylor) < 1e-8      # Chebyshev E'' is ~1e-9 accurate at the endpoints


def test_gue_jacobi_matches_chebyshev(gue):
    for s in [0.3, 1.0, 1.7, 2.4]:
        E, E1, E2 = st.gue_gap_derivatives_jacobi(s)
        assert abs(E - st.gue_gap_probability(s)) < 1e-14
        assert abs(1 + E1 - float(gue.cdf(s))) < 1e-10
        assert abs(E2 - float(gue.pdf(s))) < 1e-9


def test_gue_close_to_wigner_surmise(gue):
    s = np.linspace(0, 4, 2001)
    dev = np.abs(gue.pdf(s) - st.wigner_surmise(s))
    assert dev.max() < 0.02           # the surmise is good to ~0.005 for beta = 2
    assert dev.max() > 1e-3           # ... but not exact


def test_wigner_surmise_normalised():
    val, _ = integrate.quad(st.wigner_surmise, 0, np.inf)
    assert abs(val - 1) < 1e-10
    mean, _ = integrate.quad(lambda s: s * st.wigner_surmise(s), 0, np.inf)
    assert abs(mean - 1) < 1e-10
    s = np.linspace(0, 5, 501)
    assert np.allclose(np.gradient(st.wigner_surmise_cdf(s), s)[5:-5], st.wigner_surmise(s)[5:-5], atol=2e-3)
    assert abs(float(st.wigner_surmise_cdf(10.0)) - 1) < 1e-12


# --------------------------------------------------------------- unfolding --
def test_theta_against_mpmath():
    mp = pytest.importorskip("mpmath")
    mp.mp.dps = 30
    for t in [14.134725141734693, 100.0, 1000.0, 74920.8]:
        assert abs(float(st.riemann_siegel_theta(t)) - float(mp.siegeltheta(t))) < 2e-9


def test_unfolded_spacings_have_unit_mean(zeros_1e5):
    g = zeros_1e5[:10000]
    for method in ("theta", "density"):
        d = st.normalized_spacings(g, method)
        assert d.shape == (9999,)
        assert np.all(d > 0)
        assert abs(d.mean() - 1) < 0.01
    x = st.unfold(g)
    # N̄(gamma_n) ≈ n - 1/2 up to the bounded fluctuation S(gamma_n)
    n = np.arange(1, len(g) + 1)
    assert abs(np.mean(x - (n - 0.5))) < 0.05
    assert np.std(x - (n - 0.5)) < 0.6


# -------------------------------------------------------- pair correlation --
def test_pair_correlation_near_alpha_one(zeros_1e5):
    pc = st.pair_correlation(zeros_1e5, alpha_max=5.0, bins=100)
    sel = (pc["alpha"] > 0.9) & (pc["alpha"] < 1.1)
    assert sel.sum() >= 3
    assert np.all(np.abs(pc["R2"][sel] - pc["R2_gue"][sel]) < 0.1)
    # level repulsion: almost no pairs at tiny distance, and R_2 -> 1 at large alpha
    assert pc["R2"][0] < 0.05
    assert abs(pc["R2"][(pc["alpha"] > 3)].mean() - 1) < 0.05
    assert pc["n_ref"] > 99900
    assert abs(float(st.pair_correlation_gue(1.0)) - 1) < 1e-12 and float(st.pair_correlation_gue(0.0)) == 0.0


# -------------------------------------------------------------------- KS --
def test_ks_distance_first_1e5_vs_gue(zeros_1e5, gue):
    d = st.normalized_spacings(zeros_1e5)
    ks = st.ks_distance(d, gue.cdf)
    print(f"\nKS distance, first 1e5 zeros vs exact GUE: D = {ks['D']:.4f} at s = {ks['s_at']:.3f} "
          f"(sign {ks['sign']}, noise floor {ks['noise_floor']:.4f})")
    assert ks["D"] < 0.05
    assert ks["D"] > ks["noise_floor"]          # finite-height deviation is real
    assert st.ks_distance(d, st.poisson_cdf)["D"] > 0.2   # and nowhere near Poisson
    # the KS statistic of a sample drawn from the law itself is small
    rng = np.random.default_rng(1)
    u = rng.random(20000)
    s_grid = np.linspace(0, 6, 60001)
    sample = np.interp(u, gue.cdf(s_grid), s_grid)
    assert st.ks_distance(sample, gue.cdf)["D"] < 0.02


def test_ks_by_height_windows(zeros_1e5):
    rows = st.ks_by_height(zeros_1e5, windows=[(1, 10 ** 4), (10 ** 4, 10 ** 5)])
    assert [r["n_spacings"] for r in rows] == [9999, 90000]
    assert all(r["ks_gue"] < 0.05 for r in rows)
    assert rows[0]["logT_mid"] < rows[1]["logT_mid"]
    # the height of a window is the mean of log t over its spacings, not the geometric mean of the
    # end points (which sits far below the bulk of the zeros when the window spans decades)
    g = zeros_1e5[:10 ** 4]
    assert abs(rows[0]["logT_mid"] - np.mean(np.log(0.5 * (g[1:] + g[:-1])))) < 1e-12
    assert abs(rows[0]["logT_mid"] - 8.3548) < 1e-3 and abs(rows[0]["logT_endpoints"] - 5.9233) < 1e-3
    assert all(r["T_lo"] < r["T_mid"] < r["T_hi"] for r in rows)


def test_ks_null_constants():
    from scipy import stats
    k = st.ks_distance(np.linspace(0.01, 3, 400))
    n = k["n"]
    assert abs(k["noise_floor"] * math.sqrt(n) - stats.kstwobign.mean()) < 1e-9     # mean, sqrt(pi/2) log 2
    assert abs(k["null_median"] * math.sqrt(n) - stats.kstwobign.median()) < 1e-9   # 50% quantile 0.8276
    assert abs(k["null_q95"] * math.sqrt(n) - stats.kstwobign.ppf(0.95)) < 1e-9


def _inverse_unfold(x):
    """Ordinates gamma with theta(gamma)/pi + 1 = x (Newton in long double)."""
    x = np.asarray(x, dtype=np.longdouble)
    g = np.asarray(2 * np.pi * x / np.log(x), dtype=np.longdouble) + 100
    for _ in range(50):
        g = g - (st._theta_ld(g) / st.PI_LD + 1 - x) / (np.log(g / st.TWO_PI_LD) / (2 * st.PI_LD))
    return np.asarray(g, dtype=np.float64)


def test_estimators_normalised_on_poisson_process():
    """A unit-density Poisson process, mapped to 'ordinates' through the inverse of the
    unfolding, must give R_2 = 1 and Sigma^2(L) = L: checks the edge handling and the
    normalisation of pair_correlation and number_variance independently of zeta."""
    rng = np.random.default_rng(3)
    x = 2000.0 + np.cumsum(rng.exponential(1.0, 200000))
    g = _inverse_unfold(x)
    assert np.max(np.abs(st.unfold(g) - x)) < 1e-8
    pc = st.pair_correlation(g, alpha_max=4.0, bins=40)
    z = (pc["R2"] - 1) / pc["se"]
    assert abs(pc["R2"].mean() - 1) < 0.005 and np.max(np.abs(z)) < 4.5
    L = np.array([0.5, 2.0, 8.0])
    nv = st.number_variance(g, L, n_windows=40000)
    assert np.allclose(nv["mean_count"], L, rtol=0.02) and np.allclose(nv["sigma2"], L, rtol=0.05)


# --------------------------------------------------------- number variance --
def test_number_variance_formulas():
    L = np.array([0.01, 0.1, 50.0, 500.0])
    gue = st.number_variance_gue(L)
    assert abs(gue[0] - (0.01 - 0.01 ** 2)) < 1e-6
    assert abs(gue[1] - (0.1 - 0.01)) < 2e-4
    for Li, gi in zip(L[2:], gue[2:]):
        assert abs(gi - (math.log(2 * math.pi * Li) + 0.5772156649 + 1) / math.pi ** 2) < 1e-4
    T = 75000.0
    # Berry's semiclassical curve reproduces GUE at short range and saturates
    assert abs(float(st.number_variance_semiclassical(0.1, T)[0]) - float(st.number_variance_gue(0.1))) < 2e-3
    sat = st.number_variance_saturation(T)
    assert 0.3 < sat < 0.45
    big = st.number_variance_semiclassical(np.linspace(30, 60, 31), T)
    assert abs(big.mean() - sat) < 0.03
    assert np.all(big < st.number_variance_gue(30.0))


def test_number_variance_empirical_saturates(zeros_1e5):
    L = np.array([0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 40.0, 60.0, 80.0])
    nv = st.number_variance(zeros_1e5, L, n_windows=50_000)
    assert np.allclose(nv["mean_count"], L, rtol=0.02, atol=0.005)   # jittered-grid Monte Carlo noise
    assert abs(nv["sigma2"][0] - nv["gue"][0]) < 0.02          # GUE-like at short range
    assert nv["sigma2"][-1] < 0.6 * nv["gue"][-1]              # far below GUE at L = 80
    sat = st.detect_saturation(nv["L"], nv["sigma2"], T=float(zeros_1e5[-1]), se=nv["se"])
    assert sat["saturated"]
    assert abs(sat["plateau"] - sat["plateau_berry"]) < 0.08
    assert 0.5 < sat["L_cross"] < 5


# ------------------------------------------------------------------ Lehmer --
def test_lehmer_pair_is_among_smallest_spacings(zeros_1e5):
    rows = st.smallest_spacings(zeros_1e5, k=10)
    assert rows[0]["normalized_spacing"] == min(r["normalized_spacing"] for r in rows)
    idx = [r["index"] for r in rows]
    assert 6709 in idx                                   # Lehmer's pair 7005.063 / 7005.101
    row = next(r for r in rows if r["index"] == 6709)
    assert abs(row["gamma_n"] - 7005.0629) < 1e-3 and abs(row["gamma_n1"] - 7005.1006) < 1e-3
    assert rows[0]["normalized_spacing"] < 0.03
    delta = st.lehmer_pair_statistic(zeros_1e5, 6708)
    assert 0 < delta < 0.8                               # Csordas–Smith–Varga Lehmer pair
    assert abs(delta - 0.006933) < 2e-6


def test_lehmer_statistic_tail_correction(zeros_1e5):
    """Zeros missing beyond the ends of the data bias the CSV sum low by ~Delta^2 rho/distance;
    the smooth-density tail estimate removes this (reference: the same pair with 10^5 zeros)."""
    full = st.lehmer_pair_statistic(zeros_1e5, 6708)
    cut = zeros_1e5[:6800]                                # pair 92 zeros below the upper data edge
    raw = st.lehmer_pair_statistic(cut, 6708, tail_correction=False)
    cor = st.lehmer_pair_statistic(cut, 6708)
    assert raw / full - 1 < -3e-3 and abs(cor / full - 1) < 1e-4
    win = zeros_1e5[6600:6820]                            # a window starting at index 6601
    raw = st.lehmer_pair_statistic(win, 6708 - 6600, tail_correction=False)
    cor = st.lehmer_pair_statistic(win, 6708 - 6600, start_index=6601)
    assert raw / full - 1 < -5e-3 and abs(cor / full - 1) < 2e-4


def test_montgomery_F_shape(zeros_1e5):
    g = zeros_1e5[:5000]
    alphas = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    F = st.montgomery_F(g, alphas, window=20.0)
    assert F[0] > 2                     # ~ log T at alpha = 0
    assert F[1] < F[2]                  # rising for alpha < 1
    assert abs(F[3] - F[4]) < 0.05 and abs(F[2] - F[4]) < 0.1   # flat plateau beyond alpha = 1
