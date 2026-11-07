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


def test_dataset_boundary_bookkeeping():
    """Zeros read across the boundary of zeros_1_100000.npy / zeros_100001_1000000.npy are the
    consecutive zeros 99997..100004 (no off-by-one), identical to a direct Arb computation."""
    from rh.data import available_datasets, compute_zeros_arb, load_zeros, manifest
    sets = available_datasets()
    if not any(a == 100001 for a, _, _ in sets):
        import pytest
        pytest.skip("second dataset not present")
    g = load_zeros(8, start=99997)
    m1 = manifest(next(p for a, _, p in sets if a == 1))
    m2 = manifest(next(p for a, _, p in sets if a == 100001))
    assert g[3] == m1["last_gamma"] and g[4] == m2["first_gamma"]          # gamma_100000, gamma_100001
    assert np.array_equal(g, compute_zeros_arb(99997, 8))
    assert np.all(np.diff(g) > 0)
    whole = load_zeros(None)
    assert whole.size == 1000000 and np.array_equal(whole[99996:100004], g)
    # every returned array is a fresh writable float64 array (single-file reads used to be read-only views)
    small = load_zeros(10)
    assert small.flags.writeable and small.dtype == np.float64
    assert load_zeros(0).size == 0


def test_mislabelled_dataset_rejected(tmp_path):
    import pytest
    from rh.data import available_datasets
    np.save(tmp_path / "zeros_1_10.npy", np.arange(1.0, 10.0))            # 9 entries, name says 10
    with pytest.raises(ValueError):
        available_datasets(tmp_path)
