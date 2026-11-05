import math
import numpy as np


def test_constants():
    from rh.constants import BAEZ_DUARTE_C, LI_LAMBDA_1, EULER_GAMMA, LOG_4PI
    assert abs(BAEZ_DUARTE_C - (2 + EULER_GAMMA - LOG_4PI)) < 1e-16
    assert abs(LI_LAMBDA_1 - 0.0230957089661210338) < 1e-15
    assert abs(BAEZ_DUARTE_C - 0.0461914179322420676) < 1e-15


def test_zero_dataset_loads_and_is_consistent():
    from rh.data import load_zeros, available_datasets
    g = load_zeros(1000)
    assert len(g) == 1000
    assert abs(g[0] - 14.134725141734693790) < 1e-10
    assert abs(g[1] - 21.022039638771554993) < 1e-10
    assert abs(g[99] - 236.5242296658162058) < 1e-9
    assert np.all(np.diff(g) > 0)
    # Riemann–von Mangoldt consistency: N(gamma_n) = n, |S| small
    T = g[-1]
    smooth = T / (2 * math.pi) * math.log(T / (2 * math.pi * math.e)) + 7 / 8
    assert abs(smooth - 1000) < 3
