r"""Certified positivity of Weil's quadratic form on a whole window [-L, L] (zeta), by the
envelope reduction, in ball arithmetic throughout.

The form (spectral side)
------------------------
For real f supported in [-L, L] (additive variable u = log x) with F(t) = int f(u) e^{itu} du,

    Q(f) = Pole(f) + (1/pi) int_0^inf Psi_L(t) |F(t)|^2 dt,
    Psi_L(t) = Omega(t) - sum_{log n < 2L} c_n cos(t log n),   c_n = 2 Lambda(n) / sqrt(n),
    Omega(t) = Re psi(1/4 + it/2) - log pi,
    Pole(f) = 2 F(i/2) F(-i/2) = +2 (int f cosh(u/2))^2 (f even),  -2 (int f sinh(u/2))^2 (f odd).

This is the explicit formula of `rh.explicit_formula` written on the spectral side (the archimedean
term is (1/2pi) int h(r) Omega(r) dr, the prime term -2 sum Lambda(n) n^{-1/2} g(log n) with
g(x) = (1/2pi) int h(r) e^{-irx} dr; h = |F|^2 is even in t for real f).  Even and odd f decouple
(no cross terms in either part), so positivity is checked sector by sector.

The envelope reduction (Zhu, arXiv:2608.24827, read via a summary of its HTML version)
--------------------------------------------------------------------------------------
Omega is increasing on t > 0 (d/dt Re psi(1/4 + it/2) = -(1/2) Im psi'(1/4 + it/2) > 0, since
Im psi'(x + iy) = -sum_k 2(x+k)y/((x+k)^2+y^2)^2 < 0 for y > 0), and |cos| <= 1, so for t >= T

    Psi_L(t) >= beta := Omega(T) - A_L,        A_L = sum_{log n < 2L} c_n   (the comb constant).

With Plancherel, (1/pi) int_0^inf |F|^2 = ||f||^2, this gives Q(f) >= R_T(f) for every f, where

    R_T(f) = Pole(f) + (1/pi) int_0^T (Psi_L(t) - beta) |F(t)|^2 dt + beta ||f||^2.

beta > 0 needs T > T_1 ~ 2 pi e^{A_L}: T_1 ~ 119 for L < log(5)/2 (n = 2, 3, 4), ~ 503 for
log(5)/2 <= L < log(7)/2, ~ 2187 at L = 1.

In the orthonormal Legendre basis of a sector, phi_n(u) = sqrt((2n+1)/(2L)) P_n(u/L) (n even or n
odd), F_n(t) = sqrt(2L(2n+1)) i^n j_n(Lt); write G_n(t) = sqrt(2L(2n+1)) (-1)^{floor(n/2)} j_n(Lt)
(a real function with |F_n| = |G_n|).  Then R_T = beta I + s 2 p p^T + C with s = +1 (even) / -1
(odd), p_n = int phi_n cosh(u/2) (even) / int phi_n sinh(u/2) (odd), and

    C_mn = (1/pi) int_0^T (Psi_L(t) - beta) G_m(t) G_n(t) dt.

Split the modes into the first N (block A) and the rest (block D).  With
|j_n(x)| <= x^n / (2n+1)!! the entries of C and p decay super-exponentially once n > e L T / 2:

    eps_D >= ||R_D - beta I||_2,  eps_B >= ||R_AD||_2,
    R_T >= (min(lambda_min(R_A), beta - eps_D) - eps_B) ||f||^2   (2|x_A||x_D| <= |x_A|^2 + |x_D|^2).

What is rigorous here (everything; Arb ball arithmetic)
--------------------------------------------------------
* R_A: composite Gauss–Legendre in t on [0, T] (Arb nodes and weights) with the a-priori error bound
  (Trefethen 2008, Thm 4.5) |I - I_nu| <= (64/15) M rho^{-2nu} / (rho^2 - 1) per unit half-width,
  M a bound for the integrand on the Bernstein ellipse E_rho of the panel:
  |G_m(z) G_n(z)| <= 2L sqrt((2m+1)(2n+1)) e^{2L |Im z|} (from j_n(w) = (1/2i^n) int e^{iws} P_n(s) ds,
  |P_n| <= 1) and sup |Psi_L - beta| over E_rho by Arb on small boxes covering its boundary
  (maximum principle; the poles of Omega, at +-i(1/2 + 2k), are kept outside E_rho).  The error is
  added to each entry as a ball radius.  j_n(Lt) at the nodes: power series (with guard digits for
  its cancellation) for a starting pair and the three-term recurrence, all in ball arithmetic.
* p: modified spherical Bessel functions (Arb bessel_i).
* eps_B, eps_D: explicit bounds from |j_n(x)| <= x^n/(2n+1)!!, i_n(c) <= c^n e^c/(2n+1)!!, and
  sup_{[0,T]} |Psi_L - beta| <= max(Omega(T) - Omega(0), 2 A_L).
* lambda_min(R_A) > lambda_0 by an LDL^T factorisation of R_A - lambda_0 I in ball arithmetic: all
  pivots certainly positive => every symmetric matrix in the balls, the exact one included, is
  positive definite.
Assumed: the explicit formula (a theorem; Weil 1952), Plancherel, and Arb's correctness.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Optional

__all__ = ["comb", "omega", "symbol", "beta_star", "sph_bessel_j_all", "pole_vector", "panels",
           "envelope_block", "tail_bounds", "bessel_guard_digits", "ldl_certify", "approx_lambda_min", "certify_window"]


def _ctx(dps):
    from flint import ctx

    class _C:
        def __enter__(self):
            self.old = ctx.dps
            ctx.dps = int(dps)

        def __exit__(self, *a):
            ctx.dps = self.old
    return _C()


# ----------------------------------------------------------------------------
# 1. The symbol
# ----------------------------------------------------------------------------
def comb(L: float):
    """[(n, c_n)] for the prime powers n with log n < 2L (c_n = 2 Lambda(n)/sqrt(n), arb), and
    A_L = sum c_n (arb).  The comparison log n < 2L is decided rigorously."""
    from flint import arb
    LL = arb(L)
    out = []
    n = 2
    while True:
        ln = arb(n).log()
        if ln > 2 * LL:
            break
        if not (ln < 2 * LL):
            raise ArithmeticError(f"cannot decide log({n}) < 2L at this precision")
        p = _prime_power_base(n)
        if p:
            out.append((n, 2 * arb(p).log() / arb(n).sqrt()))
        n += 1
    A = arb(0)
    for _, c in out:
        A += c
    return out, A


def _prime_power_base(n: int) -> int:
    """p if n = p^k (k >= 1), else 0."""
    for p in range(2, int(math.isqrt(n)) + 2):
        if n % p == 0:
            m = n
            while m % p == 0:
                m //= p
            return p if m == 1 else 0
    return n          # n is prime


def omega(z):
    """Omega(z) = (psi(1/4 + iz/2) + psi(1/4 - iz/2))/2 - log(pi): analytic continuation of
    Re psi(1/4 + it/2) - log pi (acb in, acb out)."""
    from flint import acb, arb
    z = acb(z)
    q = acb(arb(1) / 4)
    h = acb(0, 1) * z / 2
    return ((q + h).digamma() + (q - h).digamma()) / 2 - arb.pi().log()


def symbol(L, z, comb_list=None):
    """Psi_L(z) (acb; real part is the symbol for real z)."""
    from flint import acb, arb
    if comb_list is None:
        comb_list, _ = comb(L)
    z = acb(z)
    s = omega(z)
    for n, c in comb_list:
        s -= c * (z * arb(n).log()).cos()
    return s


def beta_star(L, T):
    """beta = Omega(T) - A_L (arb): Psi_L(t) >= beta for all t >= T."""
    from flint import acb
    _, A = comb(L)
    return omega(acb(T)).real - A


# ----------------------------------------------------------------------------
# 2. Bessel functions
# ----------------------------------------------------------------------------
def _sph_j_series(n: int, x):
    """j_n(x) for real arb x by its power series x^n/(2n+1)!! sum_k (-x^2/2)^k / (k! prod_{i<=k}(2n+2i+1)),
    the tail bounded by the first omitted term once the terms decrease (alternating series).
    The terms peak near k ~ x^2/(4n) at ~ e^{x^2/(4n)} while the sum is ~ e^{-x^2/(4n)}: call at a
    precision with ~ x^2/(2n ln 10) guard digits (`sph_bessel_j_all` does)."""
    from flint import arb
    w = -x * x / 2
    term = arb(1)
    s = arb(1)
    k = 0
    big = arb(1)
    eps = arb(2) ** (-(_prec() + 10))
    while True:
        k += 1
        term = term * w / (k * (2 * n + 2 * k + 1))
        s += term
        if abs(term) > big:
            big = abs(term)
        r = (x * x / 2) / ((k + 1) * (2 * n + 2 * k + 3))   # |t_{k+1}/t_k|, decreasing in k
        if r < 1 and abs(term) < eps * big:
            s += (abs(term) * r) * arb(0, 1)          # alternating, decreasing: |tail| <= next term
            break
        if k > 1000000:
            raise RuntimeError("series did not converge")
    pref = x ** n
    df = arb(1)
    for i in range(1, 2 * n + 2, 2):
        df *= i
    return pref / df * s


def bessel_guard_digits(x: float, n_max: int) -> float:
    """Digits that ball arithmetic loses computing j_0..j_{n_max}(x) by `sph_bessel_j_all` (plus 20):
    * upward (x > n_max + 1): absolute radii grow by lambda_r(a) = (a + sqrt(a^2+4))/2 per step,
      a = (2n+1)/x < 2, while the values oscillate with bounded amplitude (the wrapping effect);
    * downward from n_s = n_max + 1: the series start loses ~ x^2/(2 n_s ln 10) digits; where a >= 2
      the values grow by lambda_v(a) = (a + sqrt(a^2-4))/2 per step but the radii by lambda_r(a); where
      a < 2 as in the upward case.  The input x must be known to the extended precision."""
    import numpy as np
    if x > n_max + 1:
        n = np.arange(1, n_max + 1)
        a = (2 * n + 1) / x
        return float(np.sum(np.log10((a + np.sqrt(a * a + 4)) / 2))) + 20
    ns = n_max + 1
    n = np.arange(1, ns + 1)
    a = (2 * n + 1) / x
    lr = (a + np.sqrt(a * a + 4)) / 2
    lv = np.where(a >= 2, (a + np.sqrt(np.maximum(a * a - 4, 0))) / 2, 1.0)
    loss = float(np.sum(np.log10(lr / lv)))
    return 0.5 * x * x / ns / math.log(10) + loss + 20


def _prec():
    from flint import ctx
    return ctx.prec


def _up(x) -> float:
    """A float >= every point of the arb ball x (round-to-nearest conversion padded upward)."""
    v = float(x.upper())
    return v * (1 + 1e-12) + 1e-300 if v >= 0 else v * (1 - 1e-12) + 1e-300


def sph_bessel_j_all(x, n_max: int, check: bool = True) -> list:
    """[j_0(x), ..., j_{n_max}(x)] (arb) for real arb x > 0, rigorous at any precision.  The
    recurrences run at ctx.prec + `bessel_guard_digits`; pass x accurate to that precision (its own
    radius is amplified like the rounding errors).  With check=True the absolute radii are verified to
    be < 2^-(prec - 16) and the computation is repeated with more guard digits if not.
    x > n_max + 1: upward recurrence from j_0, j_1.  Otherwise: the power series for a starting pair at
    n_s = n_max + 1, n_s + 1 and downward recurrence."""
    from flint import arb, ctx
    xm = float(x.mid())
    if not xm > 0:
        raise ValueError("x > 0 required")
    old = ctx.prec
    guard = bessel_guard_digits(xm, n_max)
    tol = arb(2) ** (-(old - 16))
    res = None
    for attempt in range(4):
        try:
            ctx.prec = old + int(guard * 3.33) + 16
            xx = +x
            if xm > n_max + 1:
                s, c = xx.sin(), xx.cos()
                out = [s / xx]
                if n_max >= 1:
                    out.append(s / (xx * xx) - c / xx)
                for n in range(1, n_max):
                    out.append((2 * n + 1) / xx * out[n] - out[n - 1])
            else:
                ns = n_max + 1
                a = _sph_j_series(ns + 1, xx)
                b = _sph_j_series(ns, xx)
                out = [None] * (ns + 2)
                out[ns + 1], out[ns] = a, b
                for n in range(ns, 0, -1):
                    out[n - 1] = (2 * n + 1) / xx * out[n] - out[n + 1]
                out = out[: n_max + 1]
        finally:
            ctx.prec = old
        res = [+v for v in out]
        if not check or all(v.rad() < tol for v in res):
            return res
        guard += 40
    return res


def pole_vector(L, orders, parity: str) -> list:
    """p_n = int phi_n(u) cosh(u/2) du (even) / sinh (odd) for the orthonormal phi_n
    = sqrt((2n+1)/(2L)) P_n(u/L): equals sqrt((2n+1)/(2L)) 2L i_n(L/2) (modified spherical Bessel,
    i_n(c) = sqrt(pi/(2c)) I_{n+1/2}(c); the parity of n selects cosh or sinh).  Arb bessel_i."""
    from flint import arb
    LL = arb(L)
    c = LL / 2
    pref = (arb.pi() / (2 * c)).sqrt()
    out = []
    for n in orders:
        i_n = pref * c.bessel_i(arb(n) + arb(1) / 2)
        out.append(((2 * n + 1) / (2 * LL)).sqrt() * 2 * LL * i_n)
    return out


# ----------------------------------------------------------------------------
# 3. Quadrature panels with a-priori error bounds
# ----------------------------------------------------------------------------
@dataclass
class Panel:
    lo: float
    hi: float
    nu: int
    rho: float
    err_unit: float        # bound on |I - I_nu| for the integrand sigma G_m G_n divided by 2L sqrt((2m+1)(2n+1))


def _ellipse_sup_sigma(L, comb_list, beta, c, h, rho, n_boxes=96, max_depth=10):
    """Upper bound for sup |Psi_L(z) - beta| on the Bernstein ellipse of the panel [c - h, c + h]
    with parameter rho (maximum principle: on its boundary).  The boundary is covered by arcs whose
    images are enclosed in Arb boxes; an arc whose evaluation is not finite (Arb's complex digamma
    gives up on wide boxes) is bisected, up to max_depth times.  Returns +inf if that fails (e.g. a
    pole of Omega on the boundary)."""
    from flint import acb, arb
    two_pi = 2 * math.pi
    best = 0.0
    stack = [(two_pi * k / n_boxes, two_pi * (k + 1) / n_boxes, 0) for k in range(n_boxes)]
    while stack:
        t0, t1, depth = stack.pop()
        th = arb((t0 + t1) / 2, (t1 - t0) / 2 * 1.0001)
        e = acb(th.cos(), th.sin())                     # e^{i theta} over the arc (ball)
        zeta = (rho * e + e.conjugate() / rho) / 2       # point of the standard ellipse
        z = acb(c) + h * zeta
        v = symbol(L, z, comb_list) - beta
        if v.is_finite():
            best = max(best, _up(abs(v)))
        elif depth < max_depth:
            tm = (t0 + t1) / 2
            stack.append((t0, tm, depth + 1))
            stack.append((tm, t1, depth + 1))
        else:
            return float("inf")
    return best


def panels(L, T, n_max, target=1e-60, w0=0.25, wmax=4.0, rho_choices=(1.5, 2, 3, 4, 6, 8)):
    """Composite rule on [0, T]: widths w0 near t = 0 (the poles of Omega sit at +-i/2), growing to
    wmax.  For each panel the (rho, nu) with the fewest nodes meeting `target` for the worst entry
    (orders <= n_max) is chosen; err_unit is the rigorous per-entry bound divided by
    2L sqrt((2m+1)(2n+1))."""
    from flint import arb
    comb_list, A = comb(L)
    beta = float(beta_star(L, T).mid())
    edges = [0.0]
    t = 0.0
    while t < T - 1e-12:
        w = w0 if t < 2.0 else (min(wmax, max(w0, t / 4.0)))
        t = min(T, t + w)
        edges.append(t)
    out = []
    worst = 2 * L * (2 * n_max + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        c, h = 0.5 * (lo + hi), 0.5 * (hi - lo)
        best = None
        for rho in rho_choices:
            # keep the poles +-i(1/2 + 2k) outside the ellipse: semi-axes a_e = h(rho+1/rho)/2,
            # b_e = h(rho-1/rho)/2; the point i/2 is outside iff (c/a_e)^2 + (1/(2 b_e))^2 > 1 (with margin)
            ae, be = h * (rho + 1 / rho) / 2, h * (rho - 1 / rho) / 2
            if (c / ae) ** 2 + (0.5 / be) ** 2 <= 1.05:
                continue
            M = _ellipse_sup_sigma(L, comb_list, beta, c, h, rho)
            if not math.isfinite(M):
                continue
            growth = math.exp(2 * L * be)               # e^{2L |Im z|}
            # (64/15) M growth worst rho^{-2nu}/(rho^2-1) * h <= target
            const = (64 / 15) * M * growth * worst * h / (rho * rho - 1)
            nu = max(4, int(math.ceil(math.log(const / target) / (2 * math.log(rho)))))
            if best is None or nu < best[1]:
                best = (rho, nu, (64 / 15) * M * growth * h / (rho * rho - 1) * rho ** (-2 * nu))
        if best is None:
            raise RuntimeError(f"no admissible ellipse for panel [{lo}, {hi}]")
        out.append(Panel(lo, hi, best[1], best[0], best[2]))
    return out


# ----------------------------------------------------------------------------
# 4. The leading block R_A
# ----------------------------------------------------------------------------
def sector_orders(parity: str, N: int) -> list:
    first = 0 if parity == "even" else 1
    return [first + 2 * j for j in range(N)]


def envelope_block(L, T, N, parity: str, dps: int = 40, target: Optional[float] = None,
                   chunk: int = 4000, verbose: bool = False) -> dict:
    """R_A = beta I + s 2 p_A p_A^T + C_AA (arb_mat, orthonormal Legendre modes of the sector) with
    rigorous quadrature error radii.  Returns {'R', 'beta', 'A', 'panels', 'n_nodes', 'orders', ...}."""
    from flint import arb, arb_mat, ctx
    from .weil_form import _gl_nodes
    orders = sector_orders(parity, N)
    n_max = orders[-1]
    t0 = time.time()
    with _ctx(dps):
        if target is None:
            target = 10.0 ** (-(dps - 5))
        comb_list, A = comb(L)
        beta = beta_star(L, T)
        LL = arb(L)
        pans = panels(L, T, n_max, target=target)
        # quadrature: accumulate C = sum_k w_k sigma_k G(t_k) G(t_k)^T / pi in chunks of nodes
        C = arb_mat(N, N)
        scale = [((2 * LL * (2 * n + 1)).sqrt()) * (1 if (n // 2) % 2 == 0 else -1) for n in orders]
        rows_G, rows_W = [], []
        n_nodes = 0
        err_unit_total = 0.0

        def flush():
            nonlocal C, rows_G, rows_W
            if not rows_G:
                return
            Gm = arb_mat(rows_G)
            Wm = arb_mat(rows_W)
            C += Gm.transpose() * Wm
            rows_G, rows_W = [], []

        inv_pi = 1 / arb.pi()
        work = ctx.prec
        for P in pans:
            # nodes at an extended precision: the Bessel recurrences amplify the radius of their
            # argument by ~10^guard (see bessel_guard_digits)
            g_dig = max(bessel_guard_digits(L * P.lo + 1e-9, n_max), bessel_guard_digits(L * P.hi, n_max))
            ctx.prec = work + int(g_dig * 3.33) + 32
            try:
                nodes, weights = _gl_nodes(P.nu)
                c, h = arb(P.lo + P.hi) / 2, arb(P.hi - P.lo) / 2
                pts = [(c + h * x, w * h) for x, w in zip(nodes, weights)]
                xs = [LL * t for t, _ in pts]
            finally:
                ctx.prec = work
            for (t, wh), xx in zip(pts, xs):
                sig = symbol(L, +t, comb_list).real - beta
                js = sph_bessel_j_all(xx, n_max)
                g = [scale[i] * js[n] for i, n in enumerate(orders)]
                wk = (+wh) * sig * inv_pi
                rows_G.append(g)
                rows_W.append([gi * wk for gi in g])
                n_nodes += 1
                if len(rows_G) >= chunk:
                    flush()
            err_unit_total += P.err_unit
        flush()
        # quadrature error radii: |E_mn| <= err_unit_total * 2L sqrt((2m+1)(2n+1)) / pi
        for i, m in enumerate(orders):
            for j, n in enumerate(orders):
                r = err_unit_total * 2 * L * math.sqrt((2 * m + 1) * (2 * n + 1)) / math.pi * (1 + 1e-9)
                C[i, j] += arb(0, r)
        p = pole_vector(L, orders, parity)
        s = 1 if parity == "even" else -1
        R = arb_mat(N, N)
        for i in range(N):
            for j in range(N):
                R[i, j] = C[i, j] + s * 2 * p[i] * p[j]
            R[i, i] += beta
        info = {"R": R, "beta": beta, "A": A, "panels": pans, "n_nodes": n_nodes, "orders": orders,
                "quad_error_unit": err_unit_total, "p": p, "comb": [n for n, _ in comb_list],
                "seconds": time.time() - t0, "dps": dps}
    if verbose:
        print(f"envelope_block L={L} T={T} N={N} {parity}: {n_nodes} nodes in {len(pans)} panels, "
              f"quad err unit {err_unit_total:.1e}, {info['seconds']:.0f}s", flush=True)
    return info


# ----------------------------------------------------------------------------
# 5. Tail and coupling bounds
# ----------------------------------------------------------------------------
def tail_bounds(L, T, N, parity: str, n_extra: int = 4000, dps: int = 30) -> dict:
    """Rigorous eps_B >= ||R_AD||_2 and eps_D >= ||R_D - beta I||_2 (D = all modes of the sector beyond
    the first N), from eta_n^2 = (1/pi) int_0^T G_n^2 <= (2L/pi) (LT)^{2n} T / ((2n+1)!!)^2 and
    |p_n| <= sqrt((2n+1)/(2L)) 2L (L/2)^n e^{L/2} / (2n+1)!!.  The sums over n >= N-th order are
    bounded term by term up to n_extra further modes and by a geometric tail beyond (ratio <= 1/2)."""
    from flint import arb, acb
    with _ctx(dps):
        LL, TT = arb(L), arb(T)
        _, A = comb(L)
        beta = beta_star(L, T)
        Om0 = omega(acb(0)).real
        OmT = omega(acb(T)).real
        sigma_max = max(_up(OmT - Om0), _up(2 * A))
        orders = sector_orders(parity, N + n_extra)
        tail = orders[N:]

        def log_df(n):            # log (2n+1)!! = lgamma(2n+2) - n log 2 - lgamma(n+1)
            return (arb(2 * n + 2).lgamma() - n * arb(2).log() - arb(n + 1).lgamma())

        eta2, p2 = arb(0), arb(0)
        last_e, last_p = None, None
        for n in tail:
            le = (2 * LL / arb.pi()).log() + 2 * n * (LL * TT).log() + TT.log() - 2 * log_df(n)
            lp = (((2 * n + 1) / (2 * LL)).log() + 2 * (2 * LL).log() + 2 * n * (LL / 2).log() + LL
                  - 2 * log_df(n))
            e_n, p_n = le.exp(), lp.exp()
            eta2 += e_n
            p2 += p_n
            last_e, last_p = e_n, p_n
        # geometric tail beyond the last term: the ratio of consecutive terms (n -> n+2) is
        # (LT)^4 / ((2n+3)^2 (2n+5)^2) for eta, (L/2)^4/... for p: check it is <= 1/2
        n_last = tail[-1]
        ratio_e = (LL * TT) ** 4 / (arb(2 * n_last + 3) ** 2 * arb(2 * n_last + 5) ** 2)
        if not (ratio_e < arb(1) / 2):
            raise ArithmeticError("increase n_extra: the eta tail is not yet geometric")
        eta2 += last_e      # sum of a geometric tail with ratio <= 1/2 starting below last_e
        p2 += last_p
        sqrtN = math.sqrt(N)
        cosh_norm = (LL + LL.sinh()).sqrt()             # ||cosh(u/2)||_{L^2[-L,L]} >= ||sinh(u/2)||
        eps_B = sigma_max * sqrtN * eta2.sqrt() + 2 * cosh_norm * p2.sqrt()
        eps_D = sigma_max * eta2 + 2 * p2
        return {"eps_B": _up(eps_B), "eps_D": _up(eps_D), "sigma_max": sigma_max,
                "eta_tail2": _up(eta2), "p_tail2": _up(p2), "beta": beta,
                "first_tail_order": tail[0]}


# ----------------------------------------------------------------------------
# 6. Positive definiteness in ball arithmetic
# ----------------------------------------------------------------------------
def ldl_certify(M, shift=None) -> dict:
    """LDL^T of the symmetric arb_mat M - shift I (lower triangle used) in ball arithmetic.
    Returns {'certified': all pivots certainly > 0, 'min_pivot': (lower, upper), 'failed_at'}."""
    from flint import arb
    n = M.nrows()
    A = M.tolist()
    if shift is not None:
        for i in range(n):
            A[i][i] = A[i][i] - shift
    L = [[None] * n for _ in range(n)]
    d = [None] * n
    minp = None
    for k in range(n):
        s = A[k][k]
        Lk = L[k]
        for j in range(k):
            s -= Lk[j] * Lk[j] * d[j]
        d[k] = s
        lo = float(s.lower())
        if minp is None or lo < minp[0]:
            minp = (lo, float(s.upper()))
        if not (s > 0):
            return {"certified": False, "failed_at": k, "pivot": s, "min_pivot": minp}
        for i in range(k + 1, n):
            Li = L[i]
            t = A[i][k]
            for j in range(k):
                t -= Li[j] * Lk[j] * d[j]
            Li[k] = t / s
    return {"certified": True, "failed_at": None, "min_pivot": minp}


def residual_certify(M, shift) -> dict:
    """Certified lower bound for lambda_min(M), M a symmetric arb_mat, by an approximate LDL^T of the
    midpoint of A = M - shift I (exact dyadic numbers, no balls: no wrapping effect) and a rigorous
    residual: if A = Lt Dt Lt^T + E with Lt, Dt exact and Dt >= 0, then Lt Dt Lt^T is positive
    semidefinite, so lambda_min(M) >= shift - ||E||_2 >= shift - ||E||_F.  E is computed in ball
    arithmetic (it contains A - Lt Dt Lt^T for every A in the balls).
    Returns {'certified', 'lower_bound' (arb), 'residual_F' (arb upper bound), 'min_pivot'}."""
    from flint import arb, arb_mat
    n = M.nrows()
    A = M - arb_mat(n, n, [shift if i == j else 0 for i in range(n) for j in range(n)]) if shift is not None \
        else M
    Am = A.mid().tolist()
    L = [[arb(0)] * n for _ in range(n)]
    d = [arb(0)] * n
    for k in range(n):
        Lk = L[k]
        s = Am[k][k]
        for j in range(k):
            s -= Lk[j] * Lk[j] * d[j]
        s = s.mid()
        d[k] = s
        if not (s > 0):
            return {"certified": False, "failed_at": k, "lower_bound": None, "residual_F": None,
                    "min_pivot": float(s)}
        Lk[k] = arb(1)
        for i in range(k + 1, n):
            Li = L[i]
            t = Am[i][k]
            for j in range(k):
                t -= Li[j] * Lk[j] * d[j]
            Li[k] = (t / s).mid()
    Lt = arb_mat(L)
    LD = arb_mat([[L[i][j] * d[j] for j in range(n)] for i in range(n)])
    E = A - LD * Lt.transpose()
    fro2 = arb(0)
    for i in range(n):
        for j in range(n):
            e = abs(E[i, j]).upper()
            fro2 += e * e
    res = fro2.sqrt()
    lower = (arb(shift) if shift is not None else arb(0)) - res
    return {"certified": bool(lower > 0), "lower_bound": lower, "residual_F": res,
            "min_pivot": float(min(x.mid() for x in d)), "failed_at": None}


def approx_lambda_min(M, iters: int = 6):
    """Approximate smallest eigenvalue (and vector) of the symmetric arb_mat M by inverse iteration on
    its midpoint (not rigorous; only used to choose the certified shift)."""
    from flint import arb, arb_mat
    n = M.nrows()
    Mm = M.mid()
    x = arb_mat([[arb(1) / (1 + i)] for i in range(n)])
    lam = None
    for _ in range(iters):
        y = Mm.solve(x)
        nrm = sum((y[i, 0] ** 2 for i in range(n)), arb(0)).sqrt()
        x = arb_mat([[y[i, 0] / nrm] for i in range(n)])
        Mx = Mm * x
        lam = sum((x[i, 0] * Mx[i, 0] for i in range(n)), arb(0))
    return lam, x


# ----------------------------------------------------------------------------
# 7. The certificate
# ----------------------------------------------------------------------------
def certify_window(L, parity: str, T, N, dps: int = 40, lam0=None, safety: float = 0.98,
                   verbose: bool = False) -> dict:
    """Certified lower bound c with Q(f) >= c ||f||^2 for every real f of the sector supported in
    [-L, L]:  c = min(lam0, beta - eps_D) - eps_B, valid when R_A - lam0 I is certified positive
    definite.  lam0 defaults to safety x (approximate lambda_min(R_A))."""
    from flint import arb
    t0 = time.time()
    blk = envelope_block(L, T, N, parity, dps=dps, verbose=verbose)
    with _ctx(dps):
        R = blk["R"]
        lam_approx, _ = approx_lambda_min(R)
        if lam0 is None:
            lam0 = lam_approx * safety if lam_approx > 0 else lam_approx
        cert = residual_certify(R, arb(lam0))
        tb = tail_bounds(L, T, N, parity)
        beta = blk["beta"]
        c = None
        if cert["certified"]:
            lam_A = cert["lower_bound"]                  # certified lambda_min(R_A) >= lam_A > 0
            lower = min(lam_A, beta - tb["eps_D"]) - tb["eps_B"]
            c = float(lower.lower())
        out = {"L": L, "parity": parity, "T": T, "N": N, "dps": dps,
               "orders_max": blk["orders"][-1], "n_nodes": blk["n_nodes"], "n_panels": len(blk["panels"]),
               "quad_error_unit": blk["quad_error_unit"], "beta": float(beta.lower()),
               "comb": blk["comb"], "A_L": float(blk["A"].mid()), "lambda_approx": float(lam_approx.mid()),
               "lambda0": float(arb(lam0).mid()), "ldl_certified": cert["certified"],
               "ldl_min_pivot": cert["min_pivot"],
               "residual_F": float(cert["residual_F"].upper()) if cert["residual_F"] is not None else None,
               "eps_B": tb["eps_B"], "eps_D": tb["eps_D"],
               "certified_lower_bound": c, "positive": bool(c is not None and c > 0),
               "seconds": time.time() - t0}
    if verbose:
        print(f"certify_window L={L} {parity} T={T} N={N}: lambda_min(R_A)~{out['lambda_approx']:.4e} "
              f"ldl={cert['certified']} eps_B={tb['eps_B']:.1e} eps_D={tb['eps_D']:.1e} -> c={c} "
              f"[{out['seconds']:.0f}s]", flush=True)
    return out
