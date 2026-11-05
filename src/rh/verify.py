"""Turing's method: rigorous zero counts N(T) from sign changes of Z, and
hence a certificate that all zeros of zeta up to height T are simple and lie
on the critical line.

Everything here is built on :mod:`rh.zeta` (an Arb-independent float64
implementation of the Riemann–Siegel formula).  Arb (python-flint) is used
only for the optional cross-checks (``N_arb`` and the zero dataset in
data/zeros, which was computed with Arb's rigorous zero isolation).

The theorem used
----------------
Let S(t) = pi^{-1} arg zeta(1/2 + i t), the argument obtained by continuous
variation along the horizontal line from +infinity + i t, so that

    N(t) = theta(t)/pi + 1 + S(t)                       (Riemann–von Mangoldt)

for t not the ordinate of a zero, and let S_1(t) = int_0^t S(u) du.

  THEOREM (Turing 1953, Theorem 4, with the constants corrected by Lehman
  1970, Theorem 4; Edwards §8.4; Brent 1979, Lemma/Theorem in §4).
  For 168 pi < t_1 < t_2:
        |S_1(t_2) - S_1(t_1)| <= 2.30 + 0.128 log(t_2 / 2 pi).

  Trudgian (Improvements to Turing's method, Math. Comp. 80 (2011) 2259–2279,
  Theorem 1) improved this to 2.067 + 0.059 log(t_2 / 2 pi) for
  t_2 > t_1 >= 168 pi.  Both constant sets are available
  (``TURING_CONSTANTS``); the default is the classical, weaker pair.

How the count is pinned down (``count_zeros_rigorously``)
---------------------------------------------------------
Let g_m be a Gram point (theta(g_m) = m pi) with g_m > 168 pi.  Suppose we
have disjoint intervals [a_i, b_i], sorted, on each of which Z provably
changes sign; each then contains a zero of odd multiplicity of zeta on the
critical line, so for every t

    N(t) >= Z_sc(t) := #{ i : b_i <= t },                      (lower bound)
    N(t) >= N(g_m) + #{ i : a_i >= g_m, b_i <= t }   for t >= g_m.

Because N_smooth(g_m) := theta(g_m)/pi + 1 = m + 1 is an integer,
s := N(g_m) - (m + 1) = S(g_m) is an integer.  For k >= 1 put
L := g_{m+k} - g_m.  Integrating the second inequality,

    S_1(g_{m+k}) - S_1(g_m) = int_{g_m}^{g_{m+k}} (N(t) - theta(t)/pi - 1) dt
                            >= s L + F_k,
    F_k := sum_{i : a_i >= g_m, b_i <= g_{m+k}} (g_{m+k} - b_i)
           - int_{g_m}^{g_{m+k}} (theta(t)/pi - m) dt.

With the theorem, B := 2.30 + 0.128 log(g_{m+k}/2pi) (or Trudgian's value),
    s <= (B - F_k) / L.
If (B - F_k)/L < 1 then s <= 0, i.e. N(g_m) <= m + 1.  If in addition
Z_sc(g_m) = m + 1 sign changes have been found in (0, g_m], then

    N(g_m) = m + 1 = Z_sc(g_m):

every zero of zeta with 0 < gamma <= g_m is simple and on the critical line
(N counts with multiplicity and each sign change accounts for one zero of odd
multiplicity), and for every T <= g_m not inside a bracket, N(T) = Z_sc(T).

Only the k Gram intervals *above* g_m are needed because the lower bound
N(g_m) >= m + 1 comes for free from the sign changes found (we never have to
prove that zeros were *not* missed below g_m by a left-hand Turing argument
as Brent/Arb do when they work with Gram blocks).

What is rigorous and what is not
--------------------------------
* The inequality above is exact mathematics given (i) the theorem and (ii)
  correct signs of Z at the bracket end points.
* Sign correctness is checked against an error *budget*:
  the proven truncation bound of the formula used (Gabcke's bound for the
  Riemann–Siegel formula for t >= 200, the Euler–Maclaurin remainder below
  500) plus a *heuristic* estimate of float64 rounding
  (``rh.zeta.Z_rounding_estimate``).  Rounding is not tracked rigorously;
  the estimate is ~10–100x larger than the errors measured against mpmath,
  but it is not a proof.  Bracket ends where |Z| does not exceed the budget
  are re-decided with mpmath at 30 digits (``n_fallback_mp`` counts them;
  mpmath has heuristic error control, so these are 'high-precision', not
  rigorous).  Gram points are computed in float64 (error ~1e-10) and the
  integral of theta by Gauss–Legendre quadrature; these errors are far below
  the slack reported in the certificate but are, again, not tracked.
* A fully rigorous implementation would evaluate Z in ball arithmetic.
  That is exactly what Arb does, which is why Arb is kept as the independent
  reference here rather than as a component.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, asdict

import numpy as np

from . import zeta as zt
from .zeta import (PI, TWO_PI, ZeroSearch, Z, Z_error_budget, Z_mp, find_zeros,
                   gram_point, theta)

#: Turing's theorem needs 168 pi < t_1.
TURING_TMIN = 168 * math.pi

#: (A, C): |S_1(t_2) - S_1(t_1)| <= A + C log(t_2 / 2 pi) for 168 pi < t_1 < t_2.
#: The default "lehman" pair is the classical one (Turing 1953 as corrected
#: by Lehman 1970, Theorem 4; also Edwards §8.4, Brent 1979).  The "trudgian"
#: pair is Theorem 1 of Trudgian, Math. Comp. 80 (2011) 2259–2279, quoted
#: here from memory of the paper: check against the published statement
#: before relying on it (it is only offered as an option; every result
#: reported by scripts/run_verify.py uses the classical pair).
TURING_CONSTANTS = {
    "lehman": (2.30, 0.128),     # Turing 1953 / Lehman 1970, Theorem 4
    "trudgian": (2.067, 0.059),  # Trudgian 2011, Theorem 1 (see note above)
}

#: smallest Gram index m with g_m > 168 pi
TURING_M_MIN = int(math.floor(theta(TURING_TMIN) / PI)) + 1
assert gram_point(TURING_M_MIN) > TURING_TMIN >= gram_point(TURING_M_MIN - 1)

#: half-width of the tightened brackets [gamma - delta, gamma + delta]
TIGHT_DELTA = 1e-6
MP_FALLBACK_DPS = 30
MP_FALLBACK_MIN_ABS = 1e-15  # |Z_mp| must exceed this to accept a sign


def turing_bound(t2, constants: str = "lehman") -> float:
    """B(t_2) = A + C log(t_2/2pi) with (A, C) = TURING_CONSTANTS[constants]."""
    A, C = TURING_CONSTANTS[constants]
    return A + C * math.log(float(t2) / TWO_PI)


def N_arb(T) -> int:
    """N(T) from Arb (rigorous): arb.zeta_nzeros(arb(T)).  T is converted to
    an exact arb from a float (passing a Python int to flint crashes)."""
    from flint import arb, ctx
    old = ctx.dps
    ctx.dps = 30
    try:
        v = arb.zeta_nzeros(arb(float(T)))
        if not float(v.rad()) < 0.5:
            raise ValueError(f"Arb could not decide N({T}): {v}")
        return int(round(float(v.mid())))
    finally:
        ctx.dps = old


# --------------------------------------------------------------------------
# sign certification
# --------------------------------------------------------------------------

@dataclass
class SignCertificate:
    """Disjoint brackets [lo_i, hi_i] with certified opposite signs of Z.

    ok[i] is False when neither the tightened nor the grid bracket could be
    certified (even with the mpmath fallback); such brackets are ignored by
    the counting argument (which only makes it harder to certify).
    """
    lo: np.ndarray
    hi: np.ndarray
    Z_lo: np.ndarray
    Z_hi: np.ndarray
    ok: np.ndarray
    tightened: np.ndarray
    n_fallback_mp: int
    n_failed: int

    @property
    def a(self):
        return self.lo[self.ok]

    @property
    def b(self):
        return self.hi[self.ok]


def _certified(Za, Zb, budget_a, budget_b):
    return ((np.sign(Za) * np.sign(Zb)) < 0) & (np.abs(Za) > budget_a) & (np.abs(Zb) > budget_b)


def certify_sign_changes(search: ZeroSearch, K: int = zt.RS_KMAX, delta: float = TIGHT_DELTA,
                         mp_dps: int = MP_FALLBACK_DPS) -> SignCertificate:
    """Certify the sign changes of a ``ZeroSearch``.

    For each zero gamma_i the bracket [gamma_i - delta, gamma_i + delta]
    (intersected with the grid bracket) is tried first: Z is evaluated at
    both ends and the signs are accepted when they are opposite and both
    |Z| exceed ``Z_error_budget``.  Otherwise the original grid bracket is
    tried the same way; failing that, both ends are re-evaluated with mpmath
    at `mp_dps` digits (counted in ``n_fallback_mp``).  Tightened brackets
    that would overlap a neighbour are not used.
    """
    zeros, lo, hi = search.zeros, search.lo, search.hi
    n = zeros.size
    if n == 0:
        e = np.empty(0)
        return SignCertificate(e, e, e, e, np.empty(0, bool), np.empty(0, bool), 0, 0)
    if np.any(lo[1:] < hi[:-1]) or np.any(hi <= lo):
        raise ValueError("brackets must be sorted and disjoint")
    # the tightened bracket is a sub-interval of its grid bracket, so all
    # brackets stay pairwise disjoint whichever of the two is used
    a = np.maximum(zeros - delta, lo)
    b = np.minimum(zeros + delta, hi)
    Za, Zb = Z(a, K), Z(b, K)
    tight_ok = _certified(Za, Zb, Z_error_budget(a, K), Z_error_budget(b, K)) & (a < b)
    # re-evaluate Z at the grid brackets instead of trusting stored values
    Zlo, Zhi = Z(lo, K), Z(hi, K)
    grid_ok = _certified(Zlo, Zhi, Z_error_budget(lo, K), Z_error_budget(hi, K))
    use_tight = tight_ok
    out_lo = np.where(use_tight, a, lo)
    out_hi = np.where(use_tight, b, hi)
    out_Zlo = np.where(use_tight, Za, Zlo)
    out_Zhi = np.where(use_tight, Zb, Zhi)
    ok = tight_ok | grid_ok
    n_fb = 0
    for i in np.nonzero(~ok)[0]:
        n_fb += 1
        for (x, y) in ((a[i], b[i]), (lo[i], hi[i])):
            zx, zy = float(Z_mp(x, mp_dps)), float(Z_mp(y, mp_dps))
            if zx * zy < 0 and abs(zx) > MP_FALLBACK_MIN_ABS and abs(zy) > MP_FALLBACK_MIN_ABS:
                out_lo[i], out_hi[i], out_Zlo[i], out_Zhi[i] = x, y, zx, zy
                ok[i] = True
                break
    return SignCertificate(out_lo, out_hi, out_Zlo, out_Zhi, ok, use_tight, n_fb,
                           int(np.sum(~ok)))


# --------------------------------------------------------------------------
# Turing's criterion
# --------------------------------------------------------------------------

_GL_X, _GL_W = np.polynomial.legendre.leggauss(64)


def theta_integral(g_a: float, g_b: float, m: int) -> float:
    """int_{g_a}^{g_b} (theta(t)/pi - m) dt by 64-point Gauss–Legendre
    (theta is analytic and slowly varying; the quadrature error is far below
    1e-12 for the intervals used here)."""
    h = 0.5 * (g_b - g_a)
    t = h * _GL_X + 0.5 * (g_a + g_b)
    return float(h * np.sum(_GL_W * (theta(t) / PI - m)))


def turing_quotient(m: int, k: int, a: np.ndarray, b: np.ndarray,
                    constants: str = "lehman") -> tuple[float, float, float, float, float, float]:
    """Evaluate the Turing criterion at Gram points g_m, g_{m+k} for certified
    brackets [a_i, b_i] (sorted, disjoint).

    Returns (g_m, g_mk, L, B, F, q) with L = g_{m+k} - g_m,
    B = turing_bound(g_{m+k}), F = sum_{a_i >= g_m, b_i <= g_{m+k}} (g_{m+k} - b_i)
    - int (theta/pi - m), and q = (B - F)/L.  q < 1 proves N(g_m) <= m + 1.
    """
    g_m, g_mk = float(gram_point(m)), float(gram_point(m + k))
    L = g_mk - g_m
    sel = (a >= g_m) & (b <= g_mk)
    F = float(np.sum(g_mk - b[sel])) - theta_integral(g_m, g_mk, m)
    B = turing_bound(g_mk, constants)
    return g_m, g_mk, L, B, F, (B - F) / L


@dataclass
class TuringCertificate:
    """Outcome of ``count_zeros_rigorously``.  When ``certified`` is True:
    N(g_m) = m + 1 = sign_changes_up_to_g_m, hence all zeros with
    0 < gamma <= g_m are simple and on the critical line, and N(T) = N_T."""
    certified: bool
    T: float
    N_T: int | None
    constants: str
    m: int | None = None
    g_m: float | None = None
    k: int | None = None
    g_mk: float | None = None
    L: float | None = None
    B: float | None = None
    F: float | None = None
    quotient: float | None = None
    sign_changes_up_to_g_m: int | None = None
    sign_changes_in_block: int | None = None
    n_brackets_certified: int = 0
    n_brackets_failed: int = 0
    n_fallback_mp: int = 0
    T_ambiguous: bool = False
    message: str = ""
    tried: list = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["tried"] = [list(x) for x in self.tried]
        return d


def _m_start(T: float) -> int:
    """Smallest Gram index m with g_m >= T and g_m > 168 pi."""
    m = int(math.ceil(theta(max(T, 10.0)) / PI - 1e-12)) if T >= 10.0 else 0
    return max(m, TURING_M_MIN)


def required_search_height(T: float, m_search: int, k_max: int) -> float:
    """t_max that a ``find_zeros(0, t_max)`` search needs for
    ``count_zeros_rigorously(T, ..., m_search, k_max)``."""
    return float(gram_point(_m_start(T) + m_search + k_max + 1)) + 1.0


def count_zeros_rigorously(T, search: ZeroSearch | None = None, constants: str = "lehman",
                           m_search: int = 40, k_max: int = 40, K: int = zt.RS_KMAX,
                           pts_per_spacing: int = 8, refine: str = "illinois") -> TuringCertificate:
    """Determine N(T) by Turing's method (see module docstring).

    `search` must be a ``find_zeros(0, t_max, details=True)`` result with
    t_max >= ``required_search_height(T, m_search, k_max)``; if None it is
    computed.  Gram points g_m >= max(T, 168 pi) are tried in increasing
    order (up to `m_search` of them); for each one with exactly m + 1
    certified sign changes below it, k = 1, 2, ..., `k_max` Gram intervals
    above it are used until the criterion (B - F_k)/L < 1 holds.
    """
    T = float(T)
    m0 = _m_start(T)
    need = required_search_height(T, m_search, k_max)
    if search is None:
        search = find_zeros(0.0, need, pts_per_spacing=pts_per_spacing, refine=refine, K=K,
                            details=True)
    if search.t_min != 0.0:
        raise ValueError("the zero search must start at t = 0 so that all sign changes are counted")
    cert = certify_sign_changes(search, K=K)
    a, b = cert.a, cert.b
    base = dict(T=T, constants=constants, n_brackets_certified=int(cert.ok.sum()),
                n_brackets_failed=cert.n_failed, n_fallback_mp=cert.n_fallback_mp)
    T_ambiguous = bool(np.any((a < T) & (T < b)))
    tried = []
    for m in range(m0, m0 + m_search + 1):
        g_m = float(gram_point(m))
        if g_m <= TURING_TMIN:
            tried.append((m, "g_m <= 168 pi"))
            continue
        if np.any((a < g_m) & (g_m < b)):
            tried.append((m, "g_m lies inside a sign-change bracket"))
            continue
        count = int(np.sum(b <= g_m))
        if count != m + 1:
            tried.append((m, f"{count} certified sign changes below g_m, Gram's law needs {m + 1}"))
            continue
        for k in range(1, k_max + 1):
            if gram_point(m + k) > search.t_max:
                tried.append((m, f"search range too short for k = {k}"))
                break
            g_m, g_mk, L, B, F, q = turing_quotient(m, k, a, b, constants)
            if q < 1.0:
                N_T = int(np.sum(b <= T))
                return TuringCertificate(
                    certified=True, N_T=N_T, m=m, g_m=g_m, k=k, g_mk=g_mk, L=L, B=B, F=F,
                    quotient=q, sign_changes_up_to_g_m=count,
                    sign_changes_in_block=int(np.sum((a >= g_m) & (b <= g_mk))),
                    T_ambiguous=T_ambiguous, tried=tried,
                    message=(f"N(g_{m}) = {m + 1} certified with k = {k} Gram intervals "
                             f"({constants} constants): (B - F)/L = ({B:.4f} - {F:.4f})/{L:.4f} "
                             f"= {q:.4f} < 1; N({T:g}) = {N_T}"), **base)
        else:
            tried.append((m, f"Turing criterion not met for k <= {k_max}"))
    return TuringCertificate(certified=False, N_T=None, T_ambiguous=T_ambiguous, tried=tried,
                             message="no Gram point could be certified; see `tried`", **base)


# --------------------------------------------------------------------------
# full verification with cross-checks
# --------------------------------------------------------------------------

LIMITATIONS = [
    "float64 rounding in Z, theta, Gram points and the quadrature is estimated heuristically, not bounded rigorously",
    "truncation of the Riemann–Siegel formula is bounded by Gabcke's theorem (t >= 200) and of Euler–Maclaurin by the classical remainder bound",
    "Turing's theorem is used with the constants of Turing/Lehman (default) or Trudgian 2011",
    "bracket ends with |Z| below the error budget are decided with mpmath at 30 digits (heuristic error control)",
    "Rosser's rule is used only as a heuristic to trigger local grid refinement, never as an assumption in the count",
]


@dataclass
class VerificationResult:
    T: float
    certified: bool
    N_T: int | None
    certified_height: float | None
    n_zeros_found: int
    n_zeros_total: int
    zeros: np.ndarray
    certificate: TuringCertificate
    arb_N_T: int | None = None
    arb_agrees: bool | None = None
    max_abs_diff_vs_arb: float | None = None
    n_compared_vs_arb: int = 0
    timings: dict = field(default_factory=dict)
    search_stats: dict = field(default_factory=dict)
    limitations: list = field(default_factory=lambda: list(LIMITATIONS))

    def summary(self) -> str:
        c = self.certificate
        lines = [
            f"verify_rh_up_to(T = {self.T:g}): {'CERTIFIED' if self.certified else 'NOT certified'}",
            f"  N(T) = {self.N_T}   zeros found below T: {self.n_zeros_found} "
            f"(total found in search range: {self.n_zeros_total})",
        ]
        if self.certified:
            lines += [
                f"  certified height g_m = {self.certified_height:.6f} (m = {c.m}), "
                f"all {c.sign_changes_up_to_g_m} zeros below it simple and on the line",
                f"  Turing block: k = {c.k} Gram intervals to g_{c.m + c.k} = {c.g_mk:.6f}, "
                f"L = {c.L:.4f}, B = {c.B:.4f}, F = {c.F:.4f}, (B-F)/L = {c.quotient:.4f} "
                f"({c.constants} constants)",
            ]
        lines.append(f"  brackets certified: {c.n_brackets_certified}, failed: {c.n_brackets_failed}, "
                     f"mpmath fallbacks: {c.n_fallback_mp}")
        if self.arb_N_T is not None:
            lines.append(f"  Arb cross-check: N(T) = {self.arb_N_T} ({'agrees' if self.arb_agrees else 'DISAGREES'}); "
                         f"max |gamma - gamma_arb| = {self.max_abs_diff_vs_arb:.3e} over {self.n_compared_vs_arb} zeros")
        if self.timings:
            lines.append("  timings: " + ", ".join(f"{k} {v:.2f}s" for k, v in self.timings.items()))
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "T": self.T, "certified": self.certified, "N_T": self.N_T,
            "certified_height": self.certified_height,
            "n_zeros_found": self.n_zeros_found, "n_zeros_total": self.n_zeros_total,
            "certificate": self.certificate.to_dict(),
            "arb_N_T": self.arb_N_T, "arb_agrees": self.arb_agrees,
            "max_abs_diff_vs_arb": self.max_abs_diff_vs_arb,
            "n_compared_vs_arb": self.n_compared_vs_arb,
            "timings": self.timings, "search_stats": self.search_stats,
            "limitations": self.limitations,
        }


def verify_rh_up_to(T, constants: str = "lehman", pts_per_spacing: int = 8,
                    refine: str = "illinois", K: int = zt.RS_KMAX, m_search: int = 40,
                    k_max: int = 40, cross_check: bool = True, arb_zeros=None,
                    verbose: bool = False) -> VerificationResult:
    """Verify the Riemann Hypothesis up to height T (all zeros with
    0 < gamma <= T are simple and on the critical line) and determine N(T).

    (a) ``find_zeros`` on [0, T'] with T' a few dozen Gram intervals beyond
        max(T, 168 pi);  (b) ``count_zeros_rigorously``: Turing's method
        proves that the sign changes found account for all zeros up to a
        Gram point g_m >= T;  (c) optional cross-checks against Arb:
        N(T) from ``arb.zeta_nzeros`` and the ordinates against the Arb
        dataset (``rh.data.load_zeros``; if no dataset is present the first
        2000 zeros are computed with Arb).  `arb_zeros` may be passed to
        skip loading.  See the module docstring for what is and is not
        rigorous.
    """
    T = float(T)
    timings = {}
    t0 = time.time()
    t_hi = required_search_height(T, m_search, k_max)
    search = find_zeros(0.0, t_hi, pts_per_spacing=pts_per_spacing, refine=refine, K=K, details=True)
    timings["find_zeros"] = time.time() - t0
    if verbose:
        print(f"find_zeros(0, {t_hi:.3f}): {len(search)} zeros, grid {search.n_grid}, "
              f"refined blocks {search.refined_blocks}, unresolved {search.unresolved_blocks} "
              f"[{timings['find_zeros']:.1f}s]", flush=True)
    t0 = time.time()
    cert = count_zeros_rigorously(T, search=search, constants=constants, m_search=m_search,
                                  k_max=k_max, K=K)
    timings["turing"] = time.time() - t0
    if verbose:
        print(cert.message, f"[{timings['turing']:.1f}s]", flush=True)
    zeros_T = search.zeros[search.zeros <= T]
    res = VerificationResult(
        T=T, certified=cert.certified, N_T=cert.N_T, certified_height=cert.g_m,
        n_zeros_found=int(zeros_T.size), n_zeros_total=len(search), zeros=zeros_T,
        certificate=cert, timings=timings,
        search_stats=dict(t_max=search.t_max, n_grid=search.n_grid, pts_per_spacing=pts_per_spacing,
                          refine=refine, refined_blocks=search.refined_blocks,
                          unresolved_blocks=[list(x) for x in search.unresolved_blocks],
                          n_gram_blocks=len(search.gram_blocks)))
    if cross_check:
        t0 = time.time()
        res.arb_N_T = N_arb(T)
        res.arb_agrees = (res.arb_N_T == cert.N_T) if cert.N_T is not None else None
        if arb_zeros is None:
            from .data import available_datasets, compute_zeros_arb, load_zeros
            n_cmp = int(zeros_T.size)
            if available_datasets():
                n_avail = max(b for _, b, _ in available_datasets())
                n_cmp = min(n_cmp, n_avail)
                arb_zeros = load_zeros(n_cmp) if n_cmp else np.empty(0)
            else:
                n_cmp = min(n_cmp, 2000)
                arb_zeros = compute_zeros_arb(1, n_cmp) if n_cmp else np.empty(0)
        arb_zeros = np.asarray(arb_zeros, dtype=float)
        n_cmp = min(int(zeros_T.size), int(arb_zeros.size))
        res.n_compared_vs_arb = n_cmp
        res.max_abs_diff_vs_arb = float(np.max(np.abs(zeros_T[:n_cmp] - arb_zeros[:n_cmp]))) if n_cmp else 0.0
        timings["cross_check"] = time.time() - t0
    return res
