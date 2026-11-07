import json
import math
from pathlib import Path

import mpmath as mp
import numpy as np
import pytest

from rh.nyman_beurling_periodic import CHI5, DAVENPORT_HEILBRONN as DH, gram_entry_direct_mp, offline_floor
from rh import nyman_beurling as nb

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("series", [CHI5, DH])
@pytest.mark.parametrize("h,k", [(1, 1), (1, 2), (2, 3), (3, 5), (4, 7), (5, 6), (7, 11)])
def test_gram_closed_form_matches_direct_evaluation(series, h, k):
    """Jump/digamma closed form vs an independent exact period-sum formula."""
    assert abs(series.gram_coprime(h, k) - float(gram_entry_direct_mp(series, h, k))) < 1e-13


@pytest.mark.parametrize("series", [CHI5, DH])
def test_gram_matrix_and_distances(series):
    G = series.gram_matrix(40)
    assert np.allclose(G, G.T)
    assert np.linalg.eigvalsh(G)[0] > 0
    d2 = series.distances(40, G)
    # d_1^2 = 1 - b_1^2 / G_11 with b_1 = F(1)
    b1 = float(series.F1())
    assert abs(d2[0] - (1 - b1 ** 2 / G[0, 0])) < 1e-14
    assert 0 < d2[-1] < d2[0] < 1


def test_values_at_one_for_chi5():
    with mp.workdps(30):
        assert abs(CHI5.F1() - 2 * mp.log((1 + mp.sqrt(5)) / 2) / mp.sqrt(5)) < mp.mpf(10) ** -25
        chi = [0, 1, -1, -1, 1]
        h = mp.mpf("1e-12")
        deriv = (mp.dirichlet(1 + h, chi) - mp.dirichlet(1 - h, chi)) / (2 * h)
        assert abs(CHI5.F1_prime() - deriv) < 1e-10


def test_zero_sum_constant_matches_davenport_heilbronn_zeros():
    """sum over all zeros of 1/(rho(1-rho)) from the Hadamard formula, minus the
    off-line quadruples, equals the sum over the computed on-line zeros + tail."""
    dh = json.loads((ROOT / "results/json/davenport_heilbronn_zeros.json").read_text())
    off = [mp.mpc(z["re"], z["im"]) for z in dh["off_line_zeros"]]
    on = [float(g) for g in dh["on_line_zeros"]]
    with mp.workdps(30):
        c_on = DH.zero_sum_constant() - sum(4 * mp.re(1 / (r * (1 - r))) for r in off)
        tail = 2 * mp.quad(lambda t: mp.log(5 * t / (2 * mp.pi)) / (2 * mp.pi) / (0.25 + t * t), [dh["T"], mp.inf])
    direct = sum(2 / (0.25 + g * g) for g in on) + float(tail)
    assert abs(float(c_on) - direct) < 1e-4


def test_offline_floor_single_point_and_zeta_version():
    rho = complex(0.8, 50.0)
    assert abs(offline_floor([rho]) - (2 * rho.real - 1) / abs(rho) ** 2) < 1e-15
    # zeta version (with the B/(s-1) correction) ~ 4 delta / gamma^2 for a conjugate pair
    g, d = 1000.0, 0.1
    fl = nb.offline_floor_zeta([complex(0.5 + d, g), complex(0.5 + d, -g)])
    assert abs(fl / (4 * d / g ** 2) - 1) < 1e-3
