"""Tests for the certified zeros of L(s, (n/5)) (scripts/compute_lchi5_zeros.py,
data/zeros/lchi5_zeros.npy/.json).

Independent references: the first zeros are recomputed from scratch with
mpmath (its own Dirichlet-series/Hurwitz-zeta evaluation and log-gamma, a
grid scan and a bracketing root finder -- nothing taken from the dataset or
from Arb); the count at T = 100 is recomputed rigorously in ball arithmetic in
two ways (full-rectangle winding number of L, and theta/pi + S(T)).
"""
import importlib.util
import json
import math
import sys
from pathlib import Path

import mpmath as mp
import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "compute_lchi5_zeros.py"
NPY = REPO / "data" / "zeros" / "lchi5_zeros.npy"
MANIFEST = REPO / "data" / "zeros" / "lchi5_zeros.json"


def _load_script():
    name = "compute_lchi5_zeros"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def lz():
    return _load_script()


@pytest.fixture(scope="module")
def zeros():
    return np.load(NPY)


@pytest.fixture(scope="module")
def manifest():
    return json.loads(MANIFEST.read_text())


# ----------------------------------------------------------------------------
# independent mpmath reference (no Arb, no stored data)
# ----------------------------------------------------------------------------
def _legendre5(n: int) -> int:
    r = n % 5
    return 0 if r == 0 else (1 if r in (1, 4) else -1)       # squares mod 5 are 1, 4


def _Z_mp(t):
    t = mp.mpf(t)
    th = t / 2 * mp.log(5 / mp.pi) + mp.im(mp.loggamma(mp.mpf(1) / 4 + 0.5j * t))
    L = mp.dirichlet(mp.mpf(1) / 2 + 1j * t, [_legendre5(n) for n in range(5)])
    return mp.exp(1j * th) * L


def test_character_is_legendre_symbol_and_root_number_one(lz):
    chi = lz.character()
    assert [complex(chi(n)) for n in range(25)] == [complex(_legendre5(n)) for n in range(25)]
    assert [_legendre5(n) for n in range(1, 6)] == [1, -1, -1, 1, 0]
    rn = lz.root_number()
    assert rn["encloses_1"] and rn["abs_root_number_minus_1_upper"] < 1e-30
    assert all(d["overlap"] and d["rel_diff_upper"] < 1e-30 for d in lz.functional_equation_check())


def test_first_zeros_against_independent_mpmath(zeros):
    """Scan Re Z on (0, 20] with mpmath, bracket every sign change, solve with Illinois at 30 digits."""
    with mp.workdps(30):
        grid = [mp.mpf(k) / 10 for k in range(1, 201)]
        vals = [mp.re(_Z_mp(t)) for t in grid]
        roots = []
        for a, b, fa, fb in zip(grid, grid[1:], vals, vals[1:]):
            if fa * fb < 0:
                roots.append(mp.findroot(lambda t: mp.re(_Z_mp(t)), (a, b), solver="illinois"))
                assert abs(mp.im(_Z_mp(roots[-1]))) < mp.mpf(10) ** -25
    assert len(roots) == int(np.sum(zeros <= 20)) == 6
    for r, g in zip(roots, zeros[:6]):
        assert abs(float(r) - g) < 1e-12, (r, g)
    # values to 12 digits from this independent computation
    assert abs(float(roots[0]) - 6.648453344728) < 1e-11
    assert abs(float(roots[1]) - 9.831444432887) < 1e-11


def test_Z_chi_is_real(lz):
    from flint import ctx
    ctx.dps = 30
    rng = np.random.default_rng(5)
    ts = list(rng.uniform(0.1, 1000.0, 12)) + [6.648453344727714, 999.9]
    for t in ts:
        z = lz.Z_chi(float(t))
        assert z.imag.contains(0)
        assert float(abs(z.imag).upper()) < 1e-24 * max(1.0, float(abs(z).upper()))
    for t in ts[:3]:                                          # mpmath agrees and is real too
        with mp.workdps(30):
            zm = _Z_mp(t)
            assert abs(mp.im(zm)) < mp.mpf(10) ** -24
            assert abs(float(mp.re(zm)) - float(lz.Z_chi(float(t)).real.mid())) < 1e-12
    hz = lz.hardy_z_crosscheck()
    assert hz["all_overlap"] and hz["max_abs_diff_upper"] < 1e-20
    hp = lz.reality_check_high_precision()
    assert hp["Im_contains_0"] and hp["max_rel_Im_upper"] < 1e-40


def test_theta_and_smooth_count(lz):
    from flint import ctx
    ctx.dps = 30
    assert lz.theta_chi(0.0).contains(0)
    for T in (50.0, 100.0, 1000.0):
        # Stirling: theta_chi(T)/pi = (T/2pi) log(5T/(2 pi e)) - 1/8 + O(1/T)
        assert abs(float(lz.n_smooth(T).mid()) - lz.n_smooth_asymptotic(T)) < 1.0 / T


def test_count_at_T100_rigorous_two_ways(lz, zeros):
    n_stored = int(np.sum(zeros <= 100.0))
    b0 = lz.bottom_rectangle()
    assert b0["integer"] == 2                                 # only the trivial zeros 0, -2 in |Im s| <= 1
    B = lz.backlund_count(100.0)
    A = lz.full_rectangle_counts([100.0], workers=1)["counts"][0]
    assert B["N"] == A["N_minus_N1"] == n_stored == 54
    assert abs(B["N"] - B["N_smooth"]) < 1                    # |S(100)| < 1
    assert abs(B["S"] - (B["N"] - B["N_smooth"])) < 1e-9


def test_count_at_T1000_matches_dataset(lz, zeros):
    assert lz.backlund_count(1000.0)["N"] == len(zeros)


def test_small_rectangles_isolate_first_zero(lz):
    from flint import ctx
    ctx.dps = 20
    # [0.25, 0.75] x [6.5, 7] contains exactly the first zero; [0.25, 0.75] x [7, 9] none
    for (t0, t1, expect) in ((6.5, 7.0, 1), (7.0, 9.0, 0)):
        pieces = []
        for z0, z1 in (((0.25, t0), (0.75, t0)), ((0.75, t0), (0.75, t1)),
                       ((0.75, t1), (0.25, t1)), ((0.25, t1), (0.25, t0))):
            a, _ = lz.edge_arg(z0, z1, 0.125)
            pieces.append(a.str(0, radius=True, more=True))
        assert lz.winding(pieces)["integer"] == expect


def test_dataset_and_manifest(zeros, manifest):
    assert zeros.dtype == np.float64 and zeros.ndim == 1
    assert len(zeros) == manifest["count"] == 904
    assert np.all(np.diff(zeros) > 0) and zeros[0] > 0 and zeros[-1] <= manifest["T_max"] == 1000.0
    assert manifest["max_error"] <= 1e-12
    assert len(manifest["eps"]) == len(zeros) and max(manifest["eps"]) == manifest["max_error"]
    assert manifest["all_zeros_on_line_and_simple"]
    assert manifest["certification"]["bottom_rectangle_winding_[-2.5,3.5]x[-1,1]"]["integer"] == 2
    heights = [c["T"] for c in manifest["certification"]["counts"]]
    assert heights == [100.0, 300.0, 600.0, 1000.0]
    for c in manifest["certification"]["counts"]:
        assert c["agree"] and c["sign_changes"] == c["N_backlund"] == c["N_winding"] == int(np.sum(zeros <= c["T"]))
        assert abs(c["N_smooth"] + c["S"] - c["N_backlund"]) < 1e-9
    assert manifest["reality_check"]["high_precision"]["Im_contains_0"]
    assert manifest["independent_mpmath_check"]["max_abs_diff_hp"] < 1e-18
    assert manifest["hp_radius"] <= 1e-20
    assert all(b["N_diff"] == b["sign_changes"] for b in manifest["certification"]["blocks"])
    # the high-precision ordinates round to the stored float64 values
    assert len(manifest["ordinates_hp"]) == len(zeros)
    for s, g in zip(manifest["ordinates_hp"], zeros):
        assert abs(float(s) - g) <= math.ulp(g) / 2


def test_sign_change_certificates_sample(lz, zeros, manifest):
    """Re-certify a sample: Z_chi(x -+ eps) have opposite signs in ball arithmetic.

    The endpoints x -+ eps are formed at 30 digits (tiny balls): at 53 bits their
    rounding radius (~ulp/2) would be comparable to eps itself."""
    from flint import arb, ctx
    ctx.dps = 30
    eps = manifest["eps"]
    r_hp = arb(manifest["hp_radius"])
    for i in [0, 1, 2, 100, 451, 700, len(zeros) - 2, len(zeros) - 1]:
        x, e = float(zeros[i]), eps[i]
        s_lo, _ = lz.Z_sign(arb(x) - arb(e), (30, 60))
        s_hi, _ = lz.Z_sign(arb(x) + arb(e), (30, 60))
        assert s_lo != 0 and s_hi == -s_lo, (i, x, e)
        assert e <= 1e-12
        # the printed high-precision ordinate: |gamma - s| < hp_radius (1e-20)
        s = arb(manifest["ordinates_hp"][i])
        h_lo, _ = lz.Z_sign(s - r_hp, (30, 60))
        h_hi, _ = lz.Z_sign(s + r_hp, (30, 60))
        assert h_lo == s_lo and h_hi == s_hi, (i, manifest["ordinates_hp"][i])
