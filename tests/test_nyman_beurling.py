import math
import numpy as np
import pytest

from rh import nyman_beurling as nb
from rh.constants import BAEZ_DUARTE_C, EULER_GAMMA


@pytest.mark.parametrize("h,k", [(1, 1), (1, 2), (2, 3), (3, 5), (4, 7), (5, 12), (7, 11)])
def test_vasyunin_closed_form_matches_direct_integral(h, k):
    """The closed form (Vasyunin) for int_0^inf {u/h}{u/k} u^-2 du is checked
    against an independent direct evaluation of the integral."""
    direct = float(nb.gram_entry_direct_mp(h, k, K=60, dps=30))
    closed = nb.gram_entry(h, k)
    assert abs(direct - closed) < 1e-13


def test_known_values():
    # int_0^inf {u}^2/u^2 du = log(2 pi) - gamma
    assert abs(nb.gram_entry(1, 1) - (math.log(2 * math.pi) - EULER_GAMMA)) < 1e-15
    # gcd reduction
    assert abs(nb.gram_entry(2, 4) - nb.gram_entry(1, 2) / 2) < 1e-15
    # b_1 = 1 - gamma
    assert abs(nb.rhs_vector(1)[0] - (1 - EULER_GAMMA)) < 1e-15


def test_distances_small_N_against_mpmath_reference():
    # reference values computed independently with mpmath (60 digits) in the
    # development notebook: d_10^2 and d_160^2
    d2 = nb.baez_duarte_distances(40)
    assert abs(d2[9] - 0.02377186556) < 2e-11
    assert np.all(np.diff(d2) <= 1e-15)  # non-increasing in N
    assert d2[-1] > 0


def test_gram_matrix_positive_definite():
    G = nb.gram_matrix(60)
    ev = np.linalg.eigvalsh(G)
    assert ev[0] > 0
    assert np.allclose(G, G.T)


def test_arb_enclosure_contains_float_value():
    d2f = nb.baez_duarte_distances(20)[-1]
    d2a, c = nb.baez_duarte_distance_arb(20, dps=40)
    mid, rad = float(d2a.mid()), float(d2a.rad())
    # the rigorous enclosure is ~1e-32 wide; float64 roundoff is ~1e-15
    assert rad < 1e-25
    assert abs(d2f - mid) < 1e-12


def test_mollifier_is_upper_bound():
    G = nb.gram_matrix(50)
    d2 = nb.baez_duarte_distances(50, G)[-1]
    assert nb.mobius_mollifier_bound(50, G) >= d2
