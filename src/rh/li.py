r"""Li / Keiper coefficients lambda_n and Li's criterion for the Riemann Hypothesis.

Li's criterion (Li 1997, J. Number Theory 65, 325-333): with

    xi(s) = (1/2) s (s-1) pi^{-s/2} Gamma(s/2) zeta(s),

    lambda_n = 1/(n-1)!  d^n/ds^n [ s^{n-1} log xi(s) ]  at s = 1      (n >= 1),

the Riemann Hypothesis holds if and only if lambda_n > 0 for every n >= 1.
Keiper (1992, Math. Comp. 58, 765-773) had introduced the same numbers (divided
by n) and computed them from the zeros.  Three equivalent descriptions are used
in this module.

1. Sum over the nontrivial zeros (Li 1997; Bombieri-Lagarias 1999):

       lambda_n = sum_rho [ 1 - (1 - 1/rho)^n ],

   the sum over rho and 1 - rho (equivalently rho and conj(rho)) being taken
   together; it is absolutely convergent for every n because
   1 - (1 - 1/rho)^n = n/rho + O(|rho|^{-2}) and sum_rho 1/rho (paired) converges.
   For a zero rho = 1/2 + i*gamma *on the critical line*, (1 - 1/rho) =
   (rho - 1)/rho has modulus one and argument theta_gamma = 2 arctan(1/(2 gamma)),
   so pairing rho with its conjugate gives

       lambda_n = 2 sum_{gamma > 0} [1 - cos(n theta_gamma)]
                = 4 sum_{gamma > 0} sin^2( n arctan(1/(2 gamma)) ).            (1)

   The zero data used here (data/zeros, Arb-certified) lie on the critical line,
   so (1) is exact for the zeros covered.  (For a hypothetical zero off the line,
   |1 - 1/rho| != 1 and the term would differ; (1) is the RH form.)

   Tail.  The zeros above the last available ordinate are replaced by the
   integral against the Riemann-von Mangoldt density dN(t) = (1/2pi) log(t/2pi) dt,

       tail(n) = int_{T_eff}^infty 4 sin^2(n arctan(1/(2u))) (1/2pi) log(u/2pi) du,

   where the starting point T_eff is defined by N_smooth(T_eff) = N (N = number
   of zeros summed, N_smooth(T) = (T/2pi) log(T/(2 pi e)) + 7/8).  Because the
   k-th zero sits, on average, where N_smooth = k - 1/2, this is the midpoint
   rule for the remaining zeros k = N+1, N+2, ...; it is the "midway between the
   last zero and the next expected zero" choice, in the smooth-count variable.
   Abel summation gives for the error of this replacement

       sum_{k > N} f(gamma_k) - tail(n) = - int_{T_eff}^infty f'(t) S(t) dt,
       |error| <= sup_{t >= T_eff} |S(t)| * TV(f; [T_eff, infty)),

   with f(u) = 4 sin^2(n arctan(1/(2u))), S(t) = N(t) - N_smooth(t) the argument
   function, and TV the total variation (= f(T_eff) ~ n^2/T_eff^2 whenever f is
   monotone on the tail, i.e. n <= pi T_eff).  A rigorous bound for S is Trudgian
   (2014, J. Number Theory 134, 280-292): |S(T)| <= 0.112 log T + 0.278 log log T
   + 2.510 for T >= e.  In practice S(t) oscillates about zero and the observed
   error is ~10^3 times smaller than this bound (see scripts/run_li.py).

2. Taylor expansion of log xi at s = 1 (Li 1997, Bombieri-Lagarias 1999):

       d/dz log xi(1/(1-z)) = sum_{n >= 0} lambda_{n+1} z^n,

   i.e. lambda_n = n [z^n] log xi(1/(1-z)).  Writing L(w) = log xi(1 + w) =
   sum_k a_k w^k and w = z/(1 - z) gives the composition
   lambda_n = n sum_{k=1}^n a_k C(n-1, k-1).  The coefficients a_k are obtained
   with Arb (python-flint) power series arithmetic:

       log xi(1 + w) = log(1/2) + log(1 + w) + log[(s-1) zeta(s)]
                       - (s/2) log pi + log Gamma(s/2),           s = 1 + w,

   where (s-1) zeta(s) = 1 + w (zeta(s) - 1/(s-1)) uses Arb's *deflated* Hurwitz
   zeta series (Johansson 2015, Numer. Algorithms 69), which is regular at s = 1
   (its coefficients are the Stieltjes constants (-1)^k gamma_k / k!).  All
   quantities are real on the real axis, so real ball arithmetic (arb_series)
   suffices and the result is a *rigorous enclosure* of each lambda_n.
   Precision: the binomial weights are as large as 2^n, so a_k must be known to
   about 0.30 n + D decimal digits for D correct digits of lambda_n (the a_k decay
   like |1/2 + i gamma_1|^{-k} ~ 14^{-k}, so only their absolute accuracy matters).

3. Archimedean / arithmetic split (Bombieri-Lagarias 1999, Thm 2; Coffey 2004,
   J. Comput. Appl. Math. 166, 525-534):

       lambda_n = S_1(n) + S_2(n),
       S_1(n) = 1 - (n/2)(gamma + log pi + 2 log 2)
                  + sum_{j=2}^n (-1)^j C(n,j) (1 - 2^{-j}) zeta(j),
       S_2(n) = sum_{j=1}^n C(n,j) eta_{j-1},   d/ds log[(s-1) zeta(s)] = sum eta_j (s-1)^j,

   S_1 coming from log(1/2) + log s - (s/2) log pi + log Gamma(s/2) and S_2 from
   log[(s-1) zeta(s)].  Singularity analysis of the generating function of S_1
   (psi(1/(2(1-z)))/(2(1-z)^2) near z = 1) gives, with H_n the harmonic number,

       S_1(n) = (n/2)(H_n - 1 - log 2 pi) + 1/2 + (exponentially small)
              = (n/2)(log n + gamma - 1 - log 2 pi) + 3/4 - 1/(24 n) + O(n^{-3}),

   which this module verifies numerically to 1e-11 (see tests).  This is the
   "smooth part" of lambda_n (Voros 2004, Math. Phys. Anal. Geom. 9, 53-63;
   Maslanka 2004, arXiv:math/0402168; Lagarias 2007, Ann. Inst. Fourier 57):
   under RH lambda_n = (n/2) log n + (1/2)(gamma - 1 - log 2 pi) n + O(sqrt(n) log n),
   whereas a zero off the line would make lambda_n oscillate with exponentially
   growing amplitude (Voros 2004).  The remainder S_2(n) = lambda_n - S_1(n) is the
   "oscillating part"; Bombieri-Lagarias show S_2(n) = O(sqrt(n) log n) under RH.

Nothing here proves RH: positivity of lambda_n for n <= nmax is implied by RH up
to a height that grows roughly like T ~ n, and conversely only lambda_n > 0 for
*all* n is equivalent to RH.
"""
from __future__ import annotations

import math
import time

import numpy as np
from scipy.integrate import quad
from scipy.special import digamma

from .constants import EULER_GAMMA, LOG_2PI, LI_LAMBDA_1

__all__ = [
    "LI_C1", "LI_C0",
    "lambda_1_closed_form", "li_partial_sum", "smooth_count", "smooth_density",
    "effective_cutoff", "li_tail", "li_tail_bound", "s_bound_trudgian",
    "li_from_zeros", "default_dps", "li_high_precision", "balls_to_float",
    "balls_to_str", "li_archimedean_exact", "li_smooth", "li_oscillation",
    "fit_asymptotic", "positivity_report",
]

TWO_PI = 2.0 * math.pi

# Smooth part lambda_n ~ (n/2) log n + LI_C1 * n + LI_C0  (see module docstring, item 3)
LI_C1 = 0.5 * (EULER_GAMMA - 1.0 - LOG_2PI)   # = -1.1303555...
LI_C0 = 0.75


# ----------------------------------------------------------------------------
# exact value of lambda_1
# ----------------------------------------------------------------------------
def lambda_1_closed_form(dps: int = 50):
    """lambda_1 = sum_rho 1/rho = 1 + gamma/2 - log(4 pi)/2 as an mpmath number.

    (Li 1997; this is also -xi'(0)/xi(0) and half of the Baez-Duarte constant.)
    """
    import mpmath as mp
    with mp.workdps(dps):
        return +(1 + mp.euler / 2 - mp.log(4 * mp.pi) / 2)


# ----------------------------------------------------------------------------
# method 1: sum over zeros + Riemann-von Mangoldt tail
# ----------------------------------------------------------------------------
def li_partial_sum(ns, zeros, block: int | None = None) -> np.ndarray:
    """4 sum_gamma sin^2(n arctan(1/(2 gamma))) for each n in `ns` (formula (1)).

    `zeros` are positive ordinates gamma of zeros on the critical line (float64).
    Vectorised in blocks of n; each block sums with numpy's pairwise summation.
    sin^2 is used instead of 1 - cos to keep full relative accuracy of the tiny
    terms n^2/(4 gamma^2) of the far zeros.
    """
    zeros = np.asarray(zeros, dtype=np.float64)
    ns = np.atleast_1d(np.asarray(ns, dtype=np.float64))
    if np.any(ns < 1) or np.any(ns != np.round(ns)):
        raise ValueError("n must be positive integers")
    alpha = np.arctan(0.5 / zeros)            # theta_gamma / 2
    if block is None:
        block = max(1, min(256, int(4_000_000 // max(len(zeros), 1))))
    out = np.empty(len(ns), dtype=np.float64)
    for i in range(0, len(ns), block):
        blk = ns[i:i + block]
        x = np.sin(np.multiply.outer(blk, alpha))
        np.multiply(x, x, out=x)
        out[i:i + block] = 4.0 * np.sum(x, axis=1)
    return out


def smooth_count(T):
    """Riemann-von Mangoldt main term N_smooth(T) = (T/2pi) log(T/(2 pi e)) + 7/8.

    N(T) = N_smooth(T) + S(T) + O(1/T), S(T) = (1/pi) arg zeta(1/2 + iT).
    """
    T = np.asarray(T, dtype=np.float64)
    return T / TWO_PI * np.log(T / (TWO_PI * math.e)) + 7.0 / 8.0


def smooth_density(T):
    """dN_smooth/dT = (1/2pi) log(T/2pi)."""
    T = np.asarray(T, dtype=np.float64)
    return np.log(T / TWO_PI) / TWO_PI


def effective_cutoff(n_zeros: int, last_gamma: float | None = None) -> float:
    """T_eff with N_smooth(T_eff) = n_zeros (start of the integral tail).

    The k-th zero sits on average where N_smooth = k - 1/2, so the remaining
    zeros k > n_zeros are centred at N_smooth = n_zeros + 1/2, n_zeros + 3/2, ...
    and the midpoint rule for them starts at N_smooth = n_zeros.  Solved by
    Newton's method (N_smooth is increasing and convex for T > 2 pi).
    """
    n_zeros = float(n_zeros)
    if last_gamma is not None and last_gamma > 20:
        T = float(last_gamma)
    else:  # asymptotic inversion as a starting point
        T = max(30.0, TWO_PI * n_zeros / max(math.log(max(n_zeros, 3.0)), 1.0))
    for _ in range(100):
        step = (float(smooth_count(T)) - n_zeros) / float(smooth_density(T))
        T -= step
        if abs(step) < 1e-13 * T:
            break
    return float(T)


def s_bound_trudgian(T: float) -> float:
    """Rigorous bound sup_{t>=T}|S(t)| <= 0.112 log T + 0.278 log log T + 2.510 (T >= e).

    Trudgian, J. Number Theory 134 (2014) 280-292.  (The bound is increasing in
    T, so for a *sup over t >= T* it must be evaluated at the largest relevant
    height; for the tail bound here the error integrand decays like 1/t^3, so we
    evaluate it at 10*T and absorb the rest in the generous constant - see
    li_tail_bound.)
    """
    T = float(T)
    return 0.112 * math.log(T) + 0.278 * math.log(math.log(T)) + 2.510


def _tail_integrand_factory(T0: float):
    L = math.log(T0 / TWO_PI)

    def integrand(v, n):
        u = T0 * np.exp(v)
        return 4.0 * np.sin(n * np.arctan(0.5 / u)) ** 2 * (L + v) / TWO_PI * u
    return integrand


_GL_CACHE: dict = {}


def _gl_nodes(v_max: float, panel: float, order: int):
    key = (v_max, panel, order)
    if key not in _GL_CACHE:
        x, w = np.polynomial.legendre.leggauss(order)
        edges = np.arange(0.0, v_max + 1e-12, panel)
        a, b = edges[:-1], edges[1:]
        mid = 0.5 * (a + b)[:, None]
        half = 0.5 * (b - a)[:, None]
        nodes = (mid + half * x[None, :]).ravel()
        weights = (half * w[None, :]).ravel()
        _GL_CACHE[key] = (nodes, weights)
    return _GL_CACHE[key]


def li_tail(ns, T_eff: float, method: str = "gl", v_max: float = 60.0,
            panel: float = 1.0, order: int = 24) -> np.ndarray:
    """Integral tail int_{T_eff}^infty 4 sin^2(n arctan(1/(2u))) (1/2pi) log(u/2pi) du.

    Substituting u = T_eff e^v turns the algebraic log(u)/u^2 decay into an
    e^{-v} decay with a smooth integrand (for n <~ 10 T_eff), integrated over
    0 <= v <= v_max (truncation error ~ e^{-v_max}) by composite Gauss-Legendre
    (method="gl", vectorised over n) or by scipy.integrate.quad per n
    (method="quad", used for validation).
    """
    ns = np.atleast_1d(np.asarray(ns, dtype=np.float64))
    T_eff = float(T_eff)
    integrand = _tail_integrand_factory(T_eff)
    if method == "gl":
        nodes, weights = _gl_nodes(v_max, panel, order)
        out = np.empty(len(ns))
        block = max(1, 2_000_000 // len(nodes))
        for i in range(0, len(ns), block):
            blk = ns[i:i + block]
            vals = integrand(nodes[None, :], blk[:, None])
            out[i:i + block] = vals @ weights
        return out
    elif method == "quad":
        out = np.empty(len(ns))
        for i, n in enumerate(ns):
            out[i], _ = quad(integrand, 0.0, v_max, args=(float(n),), limit=400,
                             epsabs=0.0, epsrel=1e-13)
        return out
    raise ValueError("method must be 'gl' or 'quad'")


def li_tail_bound(ns, T_eff: float, s_bound: float | None = None) -> np.ndarray:
    """Bound sup|S| * TV(f) on the error of the integral tail (module docstring, 1).

    f(u) = 4 sin^2(n arctan(1/(2u))); its total variation on [T_eff, inf) equals
    f(T_eff) ~ n^2/T_eff^2 when n <= pi T_eff (f monotone) and is computed from the
    number of monotone pieces otherwise.  `s_bound` defaults to Trudgian's
    rigorous bound evaluated at 10 T_eff (the error integrand f'(t) S(t) decays
    like t^{-3}, so heights beyond that contribute < 1 percent of the bound).
    """
    ns = np.atleast_1d(np.asarray(ns, dtype=np.float64))
    if s_bound is None:
        s_bound = s_bound_trudgian(10.0 * T_eff)
    x0 = ns * math.atan(0.5 / T_eff)           # argument of sin at the cutoff
    k = np.floor(x0 / (math.pi / 2))
    tv = 4.0 * k + np.abs(4.0 * np.sin(x0) ** 2 - 4.0 * (k % 2))
    return s_bound * tv


def li_from_zeros(nmax: int | None, zeros, tail: bool = True, details: bool = False,
                  ns=None, check: bool = True):
    """lambda_1..lambda_nmax from zeros on the critical line (formula (1)) + tail.

    Parameters
    ----------
    nmax : largest n (ignored if `ns` is given).
    zeros : the ordinates gamma_1 < gamma_2 < ... < gamma_N of the *first* N zeros
        (the tail formula assumes nothing is missing below the cutoff).
    tail : add the Riemann-von Mangoldt integral tail starting at T_eff.
    details : also return a dict with the partial sums, tail, T_eff, error bound.
    ns : explicit array of n values instead of 1..nmax.
    check : verify that len(zeros) is consistent with N_smooth(gamma_N) (|S| < 6).

    Returns the float64 array lambda_n (and the details dict if requested).
    """
    zeros = np.asarray(zeros, dtype=np.float64)
    N = len(zeros)
    if N == 0:
        raise ValueError("need at least one zero")
    if check:
        dev = float(smooth_count(zeros[-1])) - (N - 0.5)
        if abs(dev) > 6.0:
            raise ValueError(
                f"len(zeros)={N} inconsistent with N_smooth(gamma_N)={N - 0.5 + dev:.2f}: "
                "pass the complete list of zeros starting from gamma_1")
    if ns is None:
        ns = np.arange(1, int(nmax) + 1, dtype=np.float64)
    else:
        ns = np.atleast_1d(np.asarray(ns, dtype=np.float64))
    t0 = time.time()
    partial = li_partial_sum(ns, zeros)
    t1 = time.time()
    T_eff = effective_cutoff(N, zeros[-1])
    if tail:
        tl = li_tail(ns, T_eff)
    else:
        tl = np.zeros_like(partial)
    lam = partial + tl
    if not details:
        return lam
    info = {
        "n": ns.astype(np.int64),
        "partial_sum": partial,
        "tail": tl,
        "T_eff": T_eff,
        "n_zeros": N,
        "last_gamma": float(zeros[-1]),
        "mean_spacing_at_cutoff": 1.0 / float(smooth_density(T_eff)),
        "error_bound": li_tail_bound(ns, T_eff),
        "error_bound_S_le_1": li_tail_bound(ns, T_eff, s_bound=1.0),
        "s_bound": s_bound_trudgian(10.0 * T_eff),
        "time_partial_sum": t1 - t0,
        "time_tail": time.time() - t1,
    }
    return lam, info


# ----------------------------------------------------------------------------
# method 2: rigorous high precision via Arb power series
# ----------------------------------------------------------------------------
def default_dps(nmax: int) -> int:
    """Working precision (decimal digits) for li_high_precision.

    The composition multiplies the a_k by binomials up to 2^n (0.301 n digits of
    cancellation); 0.35 n + 60 leaves >~ 40 guaranteed digits for lambda_n, as
    confirmed by the enclosure radii Arb returns.
    """
    return int(0.35 * nmax) + 60


def _log_xi_series(nmax: int, part: str):
    """arb_series of log xi(1 + w) (or of its archimedean/arithmetic part) to O(w^{nmax+1}).

    Must be called with ctx.dps and ctx.cap already set.
    """
    from flint import arb_series, arb
    s = arb_series([1, 1])          # s = 1 + w
    w = arb_series([0, 1])
    if part in ("all", "archimedean"):
        arch = (arb(1) / 2).log() + s.log() - s * (arb.pi().log() / 2) + (s / 2).lgamma()
    if part in ("all", "arithmetic"):
        # (s-1) zeta(s) = 1 + w * (zeta(s) - 1/(s-1)); deflated series is regular at s=1
        arith = (1 + w * s.zeta(deflate=True)).log()
    if part == "all":
        return arch + arith
    if part == "archimedean":
        return arch
    if part == "arithmetic":
        return arith
    raise ValueError("part must be 'all', 'archimedean' or 'arithmetic'")


def li_high_precision(nmax: int, dps: int | None = None, part: str = "all",
                      as_float: bool = False):
    """Rigorous enclosures of lambda_1..lambda_nmax from the Taylor series of log xi.

    Implements method 2 of the module docstring with python-flint (Arb):
    lambda_n = n [z^n] L(z/(1-z)), L(w) = log xi(1+w), the composition being
    Arb's truncated power-series composition.  `part` selects the archimedean
    part S_1(n) (Gamma factor, log s and the constant), the arithmetic part
    S_2(n) (from log[(s-1) zeta(s)]) or their sum (default).

    Returns a list of `flint.arb` balls (midpoint +/- radius, rigorous), or a
    float64 array of midpoints if `as_float`.  Cost: O(nmax^2) ball operations;
    nmax = 1000 takes about a second, nmax = 5000 about half a minute.
    """
    from flint import ctx, arb
    nmax = int(nmax)
    if nmax < 1:
        raise ValueError("nmax must be >= 1")
    if dps is None:
        dps = default_dps(nmax)
    old_dps, old_cap = ctx.dps, ctx.cap
    try:
        ctx.dps = int(dps)
        ctx.cap = nmax + 1
        from flint import arb_series
        L = _log_xi_series(nmax, part)
        z = arb_series([0, 1])
        phi = L(z / (1 - z))                  # phi(z) = L(z/(1-z)), truncated at O(z^{nmax+1})
        c = phi.coeffs()
        balls = [(c[n] if n < len(c) else arb(0)) * n for n in range(1, nmax + 1)]
    finally:
        ctx.dps, ctx.cap = old_dps, old_cap
    if as_float:
        return balls_to_float(balls)
    return balls


def balls_to_float(balls) -> np.ndarray:
    """Midpoints of a list of arb balls as a float64 array."""
    return np.array([float(b.mid()) for b in balls], dtype=np.float64)


def balls_to_str(balls, digits: int = 30) -> list[str]:
    """Decimal strings (midpoints, `digits` significant digits) of arb balls."""
    return [b.mid().str(digits, radius=False) for b in balls]


def balls_max_radius(balls) -> float:
    """Largest enclosure radius in a list of arb balls."""
    return max(float(b.rad()) for b in balls)


# ----------------------------------------------------------------------------
# method 3: archimedean part, smooth asymptotics, oscillating remainder
# ----------------------------------------------------------------------------
def li_archimedean_exact(n: int, dps: int | None = None):
    """S_1(n) by the Bombieri-Lagarias formula (module docstring, 3) with mpmath.

    The alternating binomial sum cancels 2^n, hence dps ~ 0.31 n + 30 digits.
    Independent of the Arb route (li_high_precision(part="archimedean")).
    """
    import mpmath as mp
    n = int(n)
    if dps is None:
        dps = int(0.31 * n) + 30
    with mp.workdps(dps):
        tot = mp.mpf(1) - mp.mpf(n) / 2 * (mp.euler + mp.log(mp.pi) + 2 * mp.log(2))
        for k in range(2, n + 1):
            tot += (-1) ** k * mp.binomial(n, k) * (1 - mp.mpf(2) ** (-k)) * mp.zeta(k)
        return +tot


def li_smooth(n, form: str = "log"):
    """Smooth (archimedean) part of lambda_n.

    form="log"      : (n/2)(log n + gamma - 1 - log 2pi) + 3/4
    form="harmonic" : (n/2)(H_n - 1 - log 2pi) + 1/2   (H_n = psi(n+1) + gamma),
                      equal to S_1(n) up to exponentially small terms
    form="log3"     : the "log" form plus the next correction -1/(24 n)

    Derived from the singularity analysis of the generating function of S_1
    (see module docstring); the slope LI_C1 = (gamma - 1 - log 2pi)/2 is the
    constant c in lambda_n = (n/2) log n + c n + o(n) (Voros 2004; Lagarias 2007).
    """
    n = np.asarray(n, dtype=np.float64)
    if form == "log":
        return 0.5 * n * (np.log(n) + EULER_GAMMA - 1.0 - LOG_2PI) + LI_C0
    if form == "log3":
        return 0.5 * n * (np.log(n) + EULER_GAMMA - 1.0 - LOG_2PI) + LI_C0 - 1.0 / (24.0 * n)
    if form == "harmonic":
        H = digamma(n + 1.0) + EULER_GAMMA
        return 0.5 * n * (H - 1.0 - LOG_2PI) + 0.5
    raise ValueError("form must be 'log', 'log3' or 'harmonic'")


def li_oscillation(ns, lam, form: str = "harmonic") -> np.ndarray:
    """lambda_n minus the smooth part (the arithmetic / oscillating remainder)."""
    return np.asarray(lam, dtype=np.float64) - li_smooth(ns, form=form)


def fit_asymptotic(ns, lam, nmin: int = 1) -> dict:
    """Least-squares fits of the asymptotic law to numerical lambda_n.

    (a) lambda_n - (n/2) log n = c1 n + c0   (slope fixed at 1/2);
    (b) lambda_n = a n log n + b n + c       (all three free).
    Returns the fitted constants, their theoretical values (LI_C1, LI_C0, a=1/2)
    and the RMS residuals.  The oscillating part S_2(n) biases c0 by O(1) and
    c1 by O(1/sqrt(n_max)), so agreement to those levels is what RH predicts.
    """
    ns = np.asarray(ns, dtype=np.float64)
    lam = np.asarray(lam, dtype=np.float64)
    m = ns >= nmin
    n, y = ns[m], lam[m]
    # (a)
    A = np.column_stack([n, np.ones_like(n)])
    (c1, c0), *_ = np.linalg.lstsq(A, y - 0.5 * n * np.log(n), rcond=None)
    res_a = y - (0.5 * n * np.log(n) + c1 * n + c0)
    # (b)
    B = np.column_stack([n * np.log(n), n, np.ones_like(n)])
    (a, b, c), *_ = np.linalg.lstsq(B, y, rcond=None)
    res_b = y - B @ np.array([a, b, c])
    return {
        "nmin": int(nmin), "nmax": int(n.max()), "count": int(len(n)),
        "fit_c1": float(c1), "fit_c0": float(c0),
        "theory_c1": float(LI_C1), "theory_c0": float(LI_C0),
        "rms_residual_fixed_slope": float(np.sqrt(np.mean(res_a ** 2))),
        "fit_free_a": float(a), "fit_free_b": float(b), "fit_free_c": float(c),
        "theory_a": 0.5,
        "rms_residual_free": float(np.sqrt(np.mean(res_b ** 2))),
    }


def positivity_report(ns, lam) -> dict:
    """Minimum of lambda_n, where it occurs, and whether all values are positive."""
    ns = np.asarray(ns)
    lam = np.asarray(lam, dtype=np.float64)
    i = int(np.argmin(lam))
    return {
        "all_positive": bool(np.all(lam > 0)),
        "min_lambda": float(lam[i]),
        "argmin_n": int(ns[i]),
        "count": int(len(lam)),
        "nmax": int(ns.max()),
    }


# ---------------------------------------------------------------------------
# Sensitivity calibration: what an off-line zero would do to lambda_n
# ---------------------------------------------------------------------------
def zero_contribution(rho, n):
    """Contribution to lambda_n of the quadruple {rho, 1-rho, conj(rho), 1-conj(rho)}
    (for a zero on the line the quadruple degenerates to the pair {rho, conj rho}
    and the formula must then be halved; use zero_contribution_pair).
    Each zero r contributes 1 - (1 - 1/r)^n; the four are summed exactly."""
    import mpmath as mp
    rho = mp.mpc(rho)
    tot = mp.mpf(0)
    quad = {rho, 1 - rho, mp.conj(rho), 1 - mp.conj(rho)}
    for r in quad:
        tot += mp.re(1 - (1 - 1 / r) ** n)
    return tot


def detection_horizon(gamma: float, delta: float, safety: float = 1.0) -> float:
    """Smallest n (rough) at which a zero 1/2 - delta + i gamma (mirror of an
    off-line zero at 1/2 + delta + i gamma) would make lambda_n negative.
    The modulus |1 - 1/rho'| for rho' = 1/2 - delta + i gamma is
    sqrt(((1/2+delta)^2 + gamma^2) / ((1/2-delta)^2 + gamma^2)) ≈ 1 + delta/gamma^2,
    so its term grows like exp(n delta/gamma^2); it overtakes the smooth part
    (n/2) log n when  n delta/gamma^2 >~ log(n log n / 2).  Solved by iteration."""
    import math
    g2 = gamma * gamma + 0.25
    rate = 0.5 * math.log1p(2 * delta / ((0.5 - delta) ** 2 + gamma ** 2))
    n = 10.0
    for _ in range(100):
        n_new = (math.log(max(n * math.log(max(n, 2)) / 2, 2)) + math.log(safety)) / rate
        if abs(n_new - n) < 1e-9 * max(n, 1):
            break
        n = max(n_new, 1.0)
    return float(max(n, 1.0))
