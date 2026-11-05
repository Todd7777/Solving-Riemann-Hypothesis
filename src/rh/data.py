"""Access to the precomputed datasets of zeta zeros stored under data/zeros.

The files are produced by scripts/compute_zeros.py with Arb (python-flint),
whose zero isolation is rigorous: the n-th file entry is the imaginary part of
the n-th nontrivial zero (ordinate gamma_n > 0), correct to the radius recorded
in the accompanying .json manifest (about 1e-11 absolute).

`load_zeros(n)` returns the first n ordinates as a float64 array, using the
largest available dataset and falling back to a direct Arb computation when no
dataset is present (slow: ~150 zeros/s).
"""
from __future__ import annotations

import glob
import json
import os
import re
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
ZEROS_DIR = REPO_ROOT / "data" / "zeros"

_PAT = re.compile(r"zeros_(\d+)_(\d+)\.npy$")


def available_datasets():
    """List (first_index, last_index, path) for all stored zero files."""
    out = []
    for p in glob.glob(str(ZEROS_DIR / "zeros_*_*.npy")):
        m = _PAT.search(p)
        if m:
            out.append((int(m.group(1)), int(m.group(2)), Path(p)))
    out.sort()
    return out


def manifest(path: Path) -> dict:
    j = Path(str(path)[:-4] + ".json")
    return json.loads(j.read_text()) if j.exists() else {}


def load_zeros(n: int | None = None, start: int = 1) -> np.ndarray:
    """Return ordinates gamma_start, ..., gamma_{start+n-1} (1-based indices).

    Concatenates stored contiguous datasets when needed.  Falls back to Arb.
    """
    sets = available_datasets()
    if n is None:
        if not sets:
            raise FileNotFoundError("no zero datasets in data/zeros; run scripts/compute_zeros.py")
        # largest contiguous coverage from `start`
        n = _contiguous_end(sets, start) - start + 1
    need_last = start + n - 1
    if sets and _contiguous_end(sets, start) >= need_last:
        parts = []
        cur = start
        for a, b, p in sets:
            if b < cur or a > need_last:
                continue
            arr = np.load(p, mmap_mode="r")
            lo = max(cur, a) - a
            hi = min(need_last, b) - a + 1
            parts.append(np.asarray(arr[lo:hi]))
            cur = min(need_last, b) + 1
            if cur > need_last:
                break
        return np.concatenate(parts) if len(parts) > 1 else parts[0]
    # fallback: compute with Arb
    return compute_zeros_arb(start, n)


def _contiguous_end(sets, start):
    """Largest index reachable from `start` through overlapping/adjacent files."""
    end = start - 1
    changed = True
    while changed:
        changed = False
        for a, b, _ in sets:
            if a <= end + 1 and b > end:
                end = b
                changed = True
    return end


def compute_zeros_arb(start: int, n: int, dps: int = 15) -> np.ndarray:
    """Rigorously compute zeros `start..start+n-1` with Arb; return midpoints."""
    from flint import acb, ctx
    old = ctx.dps
    ctx.dps = dps
    try:
        zs = acb.zeta_zeros(start, n)
        return np.array([float(z.imag.mid()) for z in zs], dtype=np.float64)
    finally:
        ctx.dps = old
