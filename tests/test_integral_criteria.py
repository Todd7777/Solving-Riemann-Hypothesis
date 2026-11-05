import math
import numpy as np
from rh import integral_criteria as ic
from rh.data import load_zeros
from rh.constants import EULER_GAMMA


def test_tanh_sinh_log_singularity():
    # int_0^1 log(x) dx = -1 ; int_{-1}^{1} log(1-x^2) dx = 4 log 2 - 4
    x, w = ic.tanh_sinh_rule(40)
    val = np.dot(w, np.log(0.5 * (x + 1))) * 0.5
    assert abs(val + 1) < 1e-12
    val2 = np.dot(w, np.log(1 - x * x))
    assert abs(val2 - (4 * math.log(2) - 4)) < 1e-12


def test_bsy_partial_sums_tend_to_zero():
    z = load_zeros(2000)
    T, I, partial = ic.bsy_integral(z)
    # the identity says I(inf) = 0 under RH; truncation error ~ 1/T
    assert abs(I) < 5e-3
    assert abs(partial[-1] - I) < 1e-15


def test_volchkov_identity_small():
    z = load_zeros(60)
    val = ic.volchkov_integral(z, T=60.0, sigma_max=25.0, n_nodes_t=16, n_nodes_sigma=32)
    target = math.pi * (3 - EULER_GAMMA) / 32
    # truncation at T=60: tail ~ int_60^inf 12 t^2/(64 t^6) * O(1) ~ 3e-6
    assert abs(val - target) < 2e-3
