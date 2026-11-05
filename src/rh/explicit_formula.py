r"""Weil's explicit formula for the Riemann zeta function and Weil positivity.

The explicit formula (Weil, Comm. Sém. Math. Lund 1952; Iwaniec–Kowalski,
*Analytic Number Theory*, Thm. 5.12; Bombieri, "Remarks on Weil's quadratic
functional in the theory of prime numbers", Rend. Mat. Acc. Lincei 2000).
For an even test function g on R (smooth, decaying faster than any e^{-c|u|}
or compactly supported) with Fourier transform

    h(r) = int_{-inf}^{inf} g(u) e^{i r u} du          (so g(u) = (1/2pi) int h(r) e^{-iru} dr),

and rho = 1/2 + i r_rho running over the nontrivial zeros of zeta (r_rho real iff RH),

    sum_rho h(r_rho) = h(i/2) + h(-i/2) - 2 sum_{n>=1} Lambda(n) n^{-1/2} g(log n)
                       + (1/2pi) int_{-inf}^{inf} h(r) [Re psi(1/4 + i r/2) - log pi] dr,      (EF)

Lambda the von Mangoldt function, psi = Gamma'/Gamma.  Derivation: integrate
(Lambda'/Lambda)(s) H(s), H(s) = h(-i(s-1/2)) = int g(u) e^{(s-1/2)u} du, over the
boundary of -1 <= Re s <= 2 for the completed zeta Lambda(s) = pi^{-s/2} Gamma(s/2)
zeta(s); the poles at s = 0, 1 give -H(0) - H(1) = -h(i/2) - h(-i/2), the
functional equation folds the line Re s = -1 onto Re s = 2, the Dirichlet series
-zeta'/zeta = sum Lambda(n) n^{-s} gives the prime sum, and the Gamma factor,
shifted to Re s = 1/2, gives the archimedean integral.  The archimedean term can
also be written in u-space (useful for compactly supported g):

    (1/2pi) int h(r)[Re psi(1/4+ir/2) - log pi] dr
        = -(gamma_E + log pi) g(0) + 2 int_0^inf [g(0) e^{-2u} - g(u) e^{-u/2}] / (1 - e^{-2u}) du,

obtained from psi(z) = -gamma_E + int_0^inf (e^{-s} - e^{-zs})/(1-e^{-s}) ds with s = 2u.

Weil positivity.  For f in C_c^infty(R) (or rapidly decaying) let f~(u) = conj(f(-u))
and g = f * f~, so that h = |f^|^2 >= 0 on the real line.  Weil's criterion: RH holds
iff the functional W(g) := (right-hand side of (EF)) is >= 0 for every such g.  Under
RH the zero side is sum_rho |f^(r_rho)|^2 >= 0 trivially, so the content of the
criterion is the positivity of the *arithmetic* side (poles + archimedean - primes),
which is what `weil_functional` computes without using any zero.  Yoshida (1992,
"On Hermitian forms attached to zeta functions") and Bombieri (2000) study this
quadratic form on compactly supported test functions; we evaluate it on Gaussian
bumps, pairs of Gaussian bumps (whose g has side lobes at u = +-2a probing the
prime powers near e^{2a}) and on boxes f = 1_[-a,a] (g a triangle, h = 4 sin^2(ar)/r^2).

Numerics: zeros are the rigorous ordinates from data/zeros (first 10^5, accurate to
1.5e-11); the zero sum is completed by the Riemann–von Mangoldt density tail
2 int_T^inf h(r) (1/2pi) log(r/2pi) dr; the prime sum uses a numpy sieve of Lambda(n).
Nothing here proves anything: finitely many zeros and primes, floating point.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Callable, Sequence

import numpy as np
from scipy.integrate import quad
from scipy.special import digamma

EULER_GAMMA = 0.57721566490153286060651209008240243104215933593992
LOG_PI = math.log(math.pi)
TWO_PI = 2.0 * math.pi


# ----------------------------------------------------------------------------
# 1. von Mangoldt function by sieve
# ----------------------------------------------------------------------------
@lru_cache(maxsize=8)
def vonmangoldt_table(nmax: int):
    """Arrays (n, Lambda(n)) over all prime powers n <= nmax, n increasing."""
    nmax = int(nmax)
    is_p = np.ones(nmax + 1, dtype=bool)
    is_p[:2] = False
    for i in range(2, int(math.isqrt(nmax)) + 1):
        if is_p[i]:
            is_p[i * i::i] = False
    primes = np.nonzero(is_p)[0]
    ns = [primes]
    lam = [np.log(primes.astype(float))]
    for p in primes[primes <= math.isqrt(nmax)]:
        pk = int(p) * int(p)
        lp = math.log(p)
        while pk <= nmax:
            ns.append(np.array([pk]))
            lam.append(np.array([lp]))
            pk *= int(p)
    n = np.concatenate(ns)
    L = np.concatenate(lam)
    o = np.argsort(n, kind="stable")
    n, L = n[o], L[o]
    n.setflags(write=False)
    L.setflags(write=False)
    return n, L


def prime_sum(g: Callable, nmax: int) -> float:
    """2 sum_{n <= nmax} Lambda(n) n^{-1/2} g(log n)."""
    n, L = vonmangoldt_table(nmax)
    return float(2.0 * np.sum(L / np.sqrt(n) * g(np.log(n))))


# ----------------------------------------------------------------------------
# 2. Test functions
# ----------------------------------------------------------------------------
@dataclass
class TestFunction:
    """An even real test function g with Fourier transform h (both vectorised).

    h must accept complex arguments (for h(+-i/2)).  r_windows lists intervals
    of r >= 0 outside of which h is negligible (used to steer the archimedean
    and tail quadratures); u_max is where g becomes negligible (None: compact
    support given by u_support); u_points are breakpoints for the u-quadrature.
    tail_average(r) optionally replaces an oscillatory h by its local mean in the
    zero-sum tail (e.g. 4 sin^2(ar)/r^2 -> 2/r^2).
    """
    name: str
    g: Callable
    h: Callable
    params: dict = field(default_factory=dict)
    r_windows: Sequence[tuple] = ((0.0, 50.0),)
    u_max: float | None = None
    u_support: float | None = None
    u_points: Sequence[float] = ()
    tail_average: Callable | None = None
    weil: bool = False               # True if g = f * f~ for some f (h = |f^|^2 >= 0)

    def h_poles(self) -> float:
        """h(i/2) + h(-i/2) (real for even real g)."""
        return float(2.0 * np.real(self.h(np.complex128(0.5j))))

    def u_cut(self) -> float:
        return float(self.u_support if self.u_support is not None else self.u_max)


def gaussian(sigma: float) -> TestFunction:
    """g(u) = exp(-u^2/(2 sigma^2)),  h(r) = sigma sqrt(2 pi) exp(-sigma^2 r^2/2)."""
    s = float(sigma)
    c = s * math.sqrt(TWO_PI)
    return TestFunction(
        name=f"gaussian(sigma={s:g})",
        g=lambda u: np.exp(-np.square(u) / (2 * s * s)),
        h=lambda r: c * np.exp(-s * s * np.square(r) / 2),
        params={"sigma": s}, r_windows=((0.0, 9.0 / s),), u_max=9.0 * s)


def modulated_gaussian(sigma: float, omega: float) -> TestFunction:
    """g(u) = exp(-u^2/(2 sigma^2)) cos(omega u);
    h(r) = (sigma sqrt(2pi)/2) [exp(-sigma^2 (r-omega)^2/2) + exp(-sigma^2 (r+omega)^2/2)].
    Probes the zeros near r = +-omega and the primes with log n ~ 1/omega resolution."""
    s, w = float(sigma), float(omega)
    c = s * math.sqrt(TWO_PI) / 2
    win = 9.0 / s
    return TestFunction(
        name=f"modulated_gaussian(sigma={s:g}, omega={w:g})",
        g=lambda u: np.exp(-np.square(u) / (2 * s * s)) * np.cos(w * u),
        h=lambda r: c * (np.exp(-s * s * np.square(r - w) / 2) + np.exp(-s * s * np.square(r + w) / 2)),
        params={"sigma": s, "omega": w},
        r_windows=((max(0.0, w - win), w + win),) if w > win else ((0.0, w + win),),
        u_max=9.0 * s)


def weil_gaussian_bump(sigma: float) -> TestFunction:
    """f(u) = exp(-u^2/(2 sigma^2)):  g = f*f = sigma sqrt(pi) exp(-u^2/(4 sigma^2)),
    h = |f^|^2 = 2 pi sigma^2 exp(-sigma^2 r^2).
    Note g(log n) ~ e^{-(log n)^2/(4 sigma^2)}: with primes to 10^6 the prime sum
    converges only for sigma <~ 1.1 (10^7: sigma <~ 1.3)."""
    s = float(sigma)
    return TestFunction(
        name=f"weil_gaussian_bump(sigma={s:g})",
        g=lambda u: s * math.sqrt(math.pi) * np.exp(-np.square(u) / (4 * s * s)),
        h=lambda r: TWO_PI * s * s * np.exp(-s * s * np.square(r)),
        params={"sigma": s}, r_windows=((0.0, 6.5 / s),), u_max=13.0 * s, weil=True)


def weil_two_bumps(sigma: float, a: float) -> TestFunction:
    """f(u) = exp(-(u-a)^2/(2 sigma^2)) + exp(-(u+a)^2/(2 sigma^2)):
    g = f*f = sigma sqrt(pi) [e^{-(u-2a)^2/(4s^2)} + 2 e^{-u^2/(4s^2)} + e^{-(u+2a)^2/(4s^2)}],
    h = |f^|^2 = 4 pi sigma^2 e^{-sigma^2 r^2} (1 + cos(2 a r)).
    The side lobes of g at u = +-2a weight the prime powers near n = e^{2a}."""
    s, a = float(sigma), float(a)
    c = s * math.sqrt(math.pi)

    def g(u):
        u = np.asarray(u, dtype=float)
        return c * (np.exp(-np.square(u - 2 * a) / (4 * s * s)) + 2 * np.exp(-np.square(u) / (4 * s * s))
                    + np.exp(-np.square(u + 2 * a) / (4 * s * s)))

    def h(r):
        return 4 * math.pi * s * s * np.exp(-s * s * np.square(r)) * (1 + np.cos(2 * a * r))

    return TestFunction(name=f"weil_two_bumps(sigma={s:g}, a={a:g})", g=g, h=h,
                        params={"sigma": s, "a": a}, r_windows=((0.0, 6.5 / s),),
                        u_max=2 * a + 13.0 * s, u_points=(2 * a,), weil=True)


def weil_box(a: float) -> TestFunction:
    """f = 1_[-a,a]:  g = f*f = (2a - |u|)_+,  h = |f^|^2 = 4 sin^2(a r)/r^2,  h(+-i/2) = 16 sinh^2(a/2).
    h decays only like 1/r^2, so the zero sum converges slowly; the tail uses the
    local mean 2/r^2 of h."""
    a = float(a)

    def g(u):
        return np.maximum(2 * a - np.abs(np.asarray(u, dtype=float)), 0.0)

    def h(r):
        r = np.asarray(r)
        out = np.empty(r.shape, dtype=complex if np.iscomplexobj(r) else float)
        small = np.abs(r) < 1e-8
        out[~small] = 4 * np.sin(a * r[~small]) ** 2 / r[~small] ** 2
        out[small] = 4 * a * a
        return out if out.shape else out[()]

    return TestFunction(name=f"weil_box(a={a:g})", g=g, h=h, params={"a": a},
                        r_windows=((0.0, 400.0 * max(a, 1.0)),), u_support=2 * a, u_points=(),
                        tail_average=lambda r: 2.0 / np.square(r), weil=True)


# ----------------------------------------------------------------------------
# 3. The three arithmetic terms and the zero side
# ----------------------------------------------------------------------------
def archimedean_term_r(h: Callable, r_windows=((0.0, 50.0),), epsabs=1e-14, epsrel=1e-13) -> float:
    """(1/2pi) int_{-inf}^{inf} h(r) [Re psi(1/4 + i r/2) - log pi] dr = (1/pi) int_0^inf ...,
    integrated with quad over the windows where h is not negligible."""
    f = lambda r: float(h(r) * (np.real(digamma(0.25 + 0.5j * r)) - LOG_PI))
    total = 0.0
    for lo, hi in r_windows:
        # split long windows so that quad resolves oscillations / narrow bumps
        n_sub = max(1, int(math.ceil((hi - lo) / 50.0)))
        edges = np.linspace(lo, hi, n_sub + 1)
        for a, b in zip(edges[:-1], edges[1:]):
            v, _ = quad(f, a, b, limit=400, epsabs=epsabs, epsrel=epsrel)
            total += v
    return total / math.pi


def archimedean_term_u(g: Callable, u_cut: float, u_points=(), epsabs=1e-14, epsrel=1e-13) -> float:
    """Same quantity in u-space:
    -(gamma_E + log pi) g(0) + 2 int_0^inf [g(0) e^{-2u} - g(u) e^{-u/2}]/(1 - e^{-2u}) du,
    where beyond u_cut (g negligible or zero) the integral is g(0) * (-1/2) log(1 - e^{-2 u_cut})."""
    g0 = float(g(0.0))

    def f(u):
        if u < 1e-9:
            # limit u -> 0 for even g (g'(0) = 0): numerator g0(1-2u) - g0(1-u/2) + O(u^2)
            # = -(3/2) g0 u, denominator 1 - e^{-2u} ~ 2u, so the integrand -> -3 g0 / 4
            return float(-0.75 * g0)
        return float((g0 * math.exp(-2 * u) - g(u) * math.exp(-u / 2)) / (1.0 - math.exp(-2 * u)))

    pts = sorted(set(float(p) for p in u_points if 0 < p < u_cut))
    edges = [0.0] + pts + [float(u_cut)]
    total = 0.0
    for a, b in zip(edges[:-1], edges[1:]):
        v, _ = quad(f, a, b, limit=400, epsabs=epsabs, epsrel=epsrel)
        total += v
    total += -0.5 * g0 * math.log1p(-math.exp(-2 * u_cut))
    return -(EULER_GAMMA + LOG_PI) * g0 + 2.0 * total


def archimedean_term(tf: TestFunction, method: str = "auto") -> float:
    """Archimedean term of (EF) for a TestFunction; 'r', 'u' or 'auto' (u-space for
    compactly supported g, r-space otherwise)."""
    if method == "auto":
        method = "u" if tf.u_support is not None else "r"
    if method == "r":
        return archimedean_term_r(tf.h, tf.r_windows)
    return archimedean_term_u(tf.g, tf.u_cut(), tf.u_points)


def smooth_zero_count(T):
    """Riemann–von Mangoldt main term N(T) ~ (T/2pi) log(T/(2 pi e)) + 7/8."""
    return T / TWO_PI * np.log(T / (TWO_PI * np.e)) + 7.0 / 8.0


def zero_density(r):
    """(1/2pi) log(r/2pi): density of zero ordinates (one sign) at height r."""
    return np.log(np.asarray(r, dtype=float) / TWO_PI) / TWO_PI


def tail_start(zeros: np.ndarray, n_used: int | None = None) -> float:
    """Height T where the smooth tail starts after the first n_used zeros: the
    midpoint to the next zero if available, otherwise N_smooth(T) = n_used + 1/2."""
    zeros = np.asarray(zeros, dtype=float)
    n = zeros.size if n_used is None else int(n_used)
    if n < zeros.size:
        return float(0.5 * (zeros[n - 1] + zeros[n]))
    from scipy.optimize import brentq
    T0 = float(zeros[n - 1])
    f = lambda T: smooth_zero_count(T) - (n + 0.5)
    # N_smooth is increasing; bracket a sign change around gamma_n.  Because S(T)
    # can be < -1 (13 of the first 1e5 zeros have N_smooth(gamma_n) > n + 1/2),
    # the lower end may have to move below gamma_n.
    lo, hi = T0 - TWO_PI, T0 + 20.0 * TWO_PI
    while f(lo) > 0 and lo > 1.0:
        lo -= 20.0 * TWO_PI
    while f(hi) < 0:
        hi += 20.0 * TWO_PI
    return float(brentq(f, max(lo, 1.0), hi))


def zero_sum(tf_or_h, zeros: np.ndarray, n_used: int | None = None, tail: bool = True):
    """sum_rho h(r_rho) over the first n_used zeros (both signs: 2 sum_n h(gamma_n))
    plus, if tail, the density approximation 2 int_T^inf h(r) (1/2pi) log(r/2pi) dr
    of the remaining zeros.  Returns (finite_sum, tail_value)."""
    tf = tf_or_h if isinstance(tf_or_h, TestFunction) else None
    h = tf.h if tf is not None else tf_or_h
    zeros = np.asarray(zeros, dtype=float)
    n = zeros.size if n_used is None else int(n_used)
    finite = float(2.0 * np.sum(h(zeros[:n])))
    if not tail:
        return finite, 0.0
    T = tail_start(zeros, n)
    windows = tf.r_windows if tf is not None else ((0.0, 50.0),)
    ht = tf.tail_average if (tf is not None and tf.tail_average is not None) else h
    val = 0.0
    for lo, hi in windows:
        a, b = max(lo, T), hi
        if b <= a:
            continue
        n_sub = max(1, int(math.ceil((b - a) / 50.0)))
        edges = np.linspace(a, b, n_sub + 1)
        for p, q in zip(edges[:-1], edges[1:]):
            val += quad(lambda r: float(h(r) * zero_density(r)), p, q, limit=400, epsabs=1e-15, epsrel=1e-12)[0]
    if tf is not None and tf.tail_average is not None:
        # slowly decaying h: beyond the last window use the local mean of h
        R = max(T, max(hi for _, hi in windows))
        if tf.tail_average is not None and tf.name.startswith("weil_box"):
            # int_R^inf (2/r^2) log(r/2pi)/(2pi) dr = (log(R/2pi) + 1)/(pi R)
            val += (math.log(R / TWO_PI) + 1.0) / (math.pi * R)
        else:
            val += quad(lambda r: float(ht(r) * zero_density(r)), R, np.inf, limit=400)[0]
        if T < R and T < windows[-1][1]:
            pass  # the windows already covered [T, R] with the exact h
    return finite, float(2.0 * val)


# ----------------------------------------------------------------------------
# 4. The explicit formula check
# ----------------------------------------------------------------------------
def explicit_formula_check(g, h=None, zeros=None, nmax=10 ** 6, n_zeros=None, tail=True,
                           arch_method="auto", r_windows=None, u_max=None):
    """Evaluate both sides of (EF) and return a dict with every term.

    `g` may be a TestFunction (then h etc. are taken from it) or a callable, in
    which case `h` (complex-capable), `r_windows` and `u_max` describe it.
    zero side  = 2 sum_{n <= n_zeros} h(gamma_n) [+ density tail]
    arithmetic = h(i/2) + h(-i/2) - 2 sum_{n <= nmax} Lambda(n) n^{-1/2} g(log n) + archimedean
    residual   = zero side - arithmetic side.
    """
    if isinstance(g, TestFunction):
        tf = g
    else:
        if h is None:
            raise ValueError("h is required when g is a plain callable")
        tf = TestFunction(name="custom", g=g, h=h, r_windows=r_windows or ((0.0, 50.0),), u_max=u_max or 40.0)
    if zeros is None:
        from rh.data import load_zeros
        zeros = load_zeros(100000)
    zeros = np.asarray(zeros, dtype=float)
    n_used = zeros.size if n_zeros is None else min(int(n_zeros), zeros.size)
    finite, tail_val = zero_sum(tf, zeros, n_used, tail)
    poles = tf.h_poles()
    primes = prime_sum(tf.g, nmax)
    arch = archimedean_term(tf, arch_method)
    arith = poles - primes + arch
    return {"test_function": tf.name, "params": dict(tf.params), "n_zeros": int(n_used), "nmax": int(nmax),
            "zero_sum_finite": finite, "zero_tail": tail_val, "zero_side": finite + tail_val,
            "h_poles": poles, "prime_sum": primes, "archimedean": arch, "arithmetic_side": arith,
            "residual": finite + tail_val - arith, "residual_no_tail": finite - arith}


def convergence_in_zeros(tf: TestFunction, zeros, n_list, nmax=10 ** 6):
    """Residual of (EF) as a function of the number of zeros used (with and without
    the density tail); the arithmetic side is computed once."""
    zeros = np.asarray(zeros, dtype=float)
    arith = tf.h_poles() - prime_sum(tf.g, nmax) + archimedean_term(tf)
    rows = []
    for n in n_list:
        n = int(min(n, zeros.size))
        finite, tail_val = zero_sum(tf, zeros, n, True)
        rows.append({"n_zeros": n, "residual": finite + tail_val - arith, "residual_no_tail": finite - arith,
                     "tail": tail_val})
    return {"test_function": tf.name, "arithmetic_side": arith, "rows": rows}


def convergence_in_primes(tf: TestFunction, zeros, nmax_list):
    """Residual of (EF) as a function of the prime cutoff nmax (zero side fixed)."""
    zeros = np.asarray(zeros, dtype=float)
    finite, tail_val = zero_sum(tf, zeros, None, True)
    base = tf.h_poles() + archimedean_term(tf)
    rows = []
    for nm in nmax_list:
        rows.append({"nmax": int(nm), "residual": finite + tail_val - (base - prime_sum(tf.g, int(nm)))})
    return {"test_function": tf.name, "zero_side": finite + tail_val, "rows": rows}


# ----------------------------------------------------------------------------
# 5. Weil's functional
# ----------------------------------------------------------------------------
def weil_functional(tf: TestFunction, nmax=10 ** 6, zeros=None, arch_method="auto"):
    """Arithmetic side of (EF) for g = f * f~ (the Weil functional W(f*f~)), computed
    without zeros: h(i/2)+h(-i/2) + archimedean - 2 sum Lambda(n) n^{-1/2} g(log n).
    Weil's criterion: RH <=> this is >= 0 for all f.  If `zeros` is given the
    zero side sum_rho |f^(r_rho)|^2 (+ tail) is returned as well for comparison."""
    poles = tf.h_poles()
    primes = prime_sum(tf.g, nmax)
    arch = archimedean_term(tf, arch_method)
    out = {"test_function": tf.name, "params": dict(tf.params), "h_poles": poles, "prime_sum": primes,
           "archimedean": arch, "arithmetic_side": poles - primes + arch}
    if zeros is not None:
        finite, tail_val = zero_sum(tf, np.asarray(zeros, dtype=float), None, True)
        out["zero_side"] = finite + tail_val
        out["zero_tail"] = tail_val
        out["residual"] = finite + tail_val - out["arithmetic_side"]
    return out


def weil_functional_from_samples(u, f, nmax=10 ** 6, zeros=None, n_zeros=None):
    """Weil functional for a *sampled* real f on a uniform grid u (f must be
    negligible at both ends).  g = f * f~ is formed by discrete convolution
    (f~(u) = f(-u)), the prime sum by cubic interpolation of g at log n, the pole
    term as 2 int g(u) cosh(u/2) du and the archimedean term in u-space.  If zeros
    are given, h(gamma) = |f^(gamma)|^2 is evaluated by the trapezoidal rule.
    Accuracy is limited by the grid (O(du^2)); meant for exploring test functions
    that have no closed-form transform."""
    from scipy.interpolate import CubicSpline
    u = np.asarray(u, dtype=float)
    f = np.asarray(f, dtype=float)
    du = u[1] - u[0]
    if not np.allclose(np.diff(u), du):
        raise ValueError("u must be uniform")
    ft = f[::-1]                                  # f~(u) = f(-u) on the grid mirrored about 0
    gfull = np.convolve(f, ft) * du               # support: u[0]+(-u[-1]) ... u[-1]-u[0]
    ug = np.arange(gfull.size) * du + (u[0] - u[-1])
    g_spl = CubicSpline(ug, gfull, extrapolate=False)

    def g(x):
        x = np.asarray(x, dtype=float)
        out = g_spl(np.abs(x))
        return np.where(np.isnan(out), 0.0, out)

    umax = float(ug[-1])
    poles = float(2.0 * np.trapezoid(gfull * np.cosh(ug / 2), ug))
    primes = prime_sum(g, nmax)
    arch = archimedean_term_u(g, umax)
    out = {"test_function": "sampled f", "h_poles": poles, "prime_sum": primes, "archimedean": arch,
           "arithmetic_side": poles - primes + arch, "g_at_0": float(g(0.0))}
    if zeros is not None:
        zeros = np.asarray(zeros, dtype=float)
        n = zeros.size if n_zeros is None else int(n_zeros)
        # the trapezoidal Fourier transform is spectrally accurate only below the
        # Nyquist angular frequency pi/du; beyond it the sampled e^{iru} aliases to
        # a low frequency and |f^|^2 is grossly overestimated.  Cap the zeros used
        # conservatively at HALF the Nyquist frequency.
        r_cap = 0.5 * math.pi / du
        n = int(min(n, np.searchsorted(zeros, r_cap)))
        out["r_cap"] = r_cap
        out["r_nyquist"] = math.pi / du
        zs = zeros[:n]
        # f^(r) = int f(u) e^{iru} du by trapezoid, in chunks of zeros
        total = 0.0
        w = np.full(u.size, du)
        w[0] = w[-1] = du / 2
        fw = f * w
        for i0 in range(0, n, 2000):
            r = zs[i0:i0 + 2000]
            fh = np.exp(1j * np.outer(r, u)) @ fw
            total += float(np.sum(np.abs(fh) ** 2))
        out["zero_side_finite"] = 2.0 * total
        out["n_zeros"] = n
        out["residual_no_tail"] = 2.0 * total - out["arithmetic_side"]
    return out


def weil_positivity_scan(family: str, grid: Sequence[float], nmax=10 ** 6, zeros=None, **kw):
    """Evaluate the Weil functional on a one-parameter family:
    'gaussian_bump' (sigma in grid), 'two_bumps' (a in grid, sigma=kw['sigma']),
    'box' (a in grid).  Returns a list of `weil_functional` dicts plus the minimum."""
    rows = []
    for p in grid:
        if family == "gaussian_bump":
            tf = weil_gaussian_bump(p)
        elif family == "two_bumps":
            tf = weil_two_bumps(kw.get("sigma", 0.2), p)
        elif family == "box":
            tf = weil_box(p)
        else:
            raise ValueError(family)
        r = weil_functional(tf, nmax, zeros)
        r["parameter"] = float(p)
        rows.append(r)
    vals = np.array([r["arithmetic_side"] for r in rows])
    # the arithmetic side is a cancellation of O(1) terms; values below the
    # float64 rounding level of those terms (e.g. -3e-15 where the true value is
    # e^{-200}) are not evidence against positivity, so compare with a tolerance
    tol = np.array([1e-13 * (abs(r["h_poles"]) + abs(r["prime_sum"]) + abs(r["archimedean"])) for r in rows])
    for r, t in zip(rows, tol):
        r["rounding_tolerance"] = float(t)
    return {"family": family, "kw": kw, "rows": rows, "min_arithmetic_side": float(vals.min()),
            "argmin_parameter": float(rows[int(vals.argmin())]["parameter"]),
            "all_nonnegative": bool(np.all(vals >= -tol)),
            "n_negative_beyond_rounding": int(np.sum(vals < -tol)),
            "min_arithmetic_side_in_units_of_tolerance": float(np.min(vals / tol))}


# ----------------------------------------------------------------------------
# 6. Figures
# ----------------------------------------------------------------------------
def plot_residual_convergence(conv_zeros: list, conv_primes: list, path):
    """Left: |residual| vs number of zeros (with/without density tail) for several
    test functions.  Right: |residual| vs prime cutoff nmax."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.0))
    for i, c in enumerate(conv_zeros):
        n = [r["n_zeros"] for r in c["rows"]]
        res = np.abs([r["residual"] for r in c["rows"]])
        res0 = np.abs([r["residual_no_tail"] for r in c["rows"]])
        lab = c["test_function"].replace("gaussian", "Gaussian")
        ax1.loglog(n, np.maximum(res0, 1e-17), "o--", color=f"C{i}", ms=4, lw=1, label=f"{lab}: truncated sum")
        ax1.loglog(n, np.maximum(res, 1e-17), "s-", color=f"C{i}", ms=4, lw=1.5, label=f"{lab}: + density tail")
    ax1.set_xlabel("number of zeros used (both signs)")
    ax1.set_ylabel("|zero side - arithmetic side|")
    ax1.set_title("Explicit formula: convergence in the zeros (primes to $10^6$)", fontsize=10)
    ax1.grid(alpha=0.3, which="both")
    ax1.legend(fontsize=7.5)
    for i, c in enumerate(conv_primes):
        nm = [r["nmax"] for r in c["rows"]]
        res = np.abs([r["residual"] for r in c["rows"]])
        ax2.loglog(nm, np.maximum(res, 1e-17), "o-", color=f"C{i}", ms=4, lw=1.5, label=c["test_function"])
    ax2.set_xlabel(r"prime-power cutoff $n_{\max}$")
    ax2.set_ylabel("|residual|")
    ax2.set_title("Convergence in the primes ($10^5$ zeros + density tail)", fontsize=10)
    ax2.grid(alpha=0.3, which="both")
    ax2.legend(fontsize=7.5)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_weil_positivity(scans: list, path):
    """Arithmetic side of W(f*f~) along one-parameter families (must be >= 0 under RH),
    with the zero side dashed where available and prime powers marked for the
    two-bump family (its side lobes sit at u = 2a = log n)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(scans), figsize=(4.4 * len(scans) + 0.5, 4.6))
    axes = np.atleast_1d(axes)
    for ax, sc in zip(axes, scans):
        p = np.array([r["parameter"] for r in sc["rows"]])
        A = np.array([r["arithmetic_side"] for r in sc["rows"]])
        ax.plot(p, A, color="C3", lw=1.6, label="arithmetic side (no zeros used)")
        if "zero_side" in sc["rows"][0]:
            Z = np.array([r["zero_side"] for r in sc["rows"]])
            ax.plot(p, Z, "k--", lw=1.0, label=r"zero side $\sum_\rho |\hat f(r_\rho)|^2$")
        ax.axhline(0, color="0.5", lw=0.8)
        if "rounding_tolerance" in sc["rows"][0] and sc["family"] == "gaussian_bump":
            tol = np.array([r["rounding_tolerance"] for r in sc["rows"]])
            ax.plot(p, tol, color="0.4", ls=":", lw=1.0, label="float64 rounding level of the cancelled terms")
        if sc["family"] == "two_bumps":
            n, _ = vonmangoldt_table(100)
            pk = 0.5 * np.log(n[n >= 2])
            pk = pk[pk <= p.max()]
            y0, y1 = ax.get_ylim()
            ax.vlines(pk, y0, y0 + 0.06 * (y1 - y0), color="C0", lw=1.0, label=r"$2a=\log p^k$, $p^k\leq 100$")
            ax.set_ylim(y0, y1)
            ax.set_xlabel(r"$a$   (side lobes of $g$ at $u=\pm 2a$ weight the prime powers near $e^{2a}$)")
            ax.set_title(f"two Gaussian bumps at $\\pm a$, $\\sigma$={sc['kw'].get('sigma', 0.2)}", fontsize=10)
        elif sc["family"] == "box":
            ax.set_xlabel(r"$a$   ($f=1_{[-a,a]}$, $g=(2a-|u|)_+$, $h=4\sin^2(ar)/r^2$)")
            ax.set_title("box test functions", fontsize=10)
            ax.set_yscale("log")
        else:
            ax.set_xlabel(r"$\sigma$   ($f=e^{-u^2/2\sigma^2}$)")
            ax.set_title("Gaussian bump", fontsize=10)
            ax.set_xscale("log")
            ax.set_yscale("log")
        ax.set_ylabel(r"$W(f*\tilde f)$")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Weil positivity: the arithmetic side of the explicit formula is nonnegative on "
                 "$g=f*\\tilde f$ (min over the scans: %.3e)" % min(s["min_arithmetic_side"] for s in scans),
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path
