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
           "envelope_block", "envelope_blocks", "certify_windows", "approx_lambda_min_ldl", "tail_bounds", "bessel_guard_digits", "ldl_certify", "ldl_mid_blocked",
           "residual_certify", "inertia_count", "lambda_min_estimate", "approx_lambda_min", "certify_window",
           "SymBlocks"]


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


def symbol(L, z, comb_list=None, comb_cert=None):
    """Psi_L(z) (acb; real part is the symbol for real z), plus H(z) when a comb certificate
    (`rh.weil_comb.CombCertificate`) is given."""
    from flint import acb, arb
    if comb_list is None:
        comb_list, _ = comb(L)
    z = acb(z)
    s = omega(z)
    for n, c in comb_list:
        s -= c * (z * arb(n).log()).cos()
    if comb_cert is not None:
        s += comb_cert.H(z)
    return s


def beta_star(L, T, comb_cert=None):
    """beta = Omega(T) - A_L (arb): Psi_L(t) >= beta for all t >= T.  With a comb certificate
    (A' - P_L + H >= 0 everywhere, H of spectrum outside (-2L, 2L)): beta'' = Omega(T) - A', and
    Psi_L(t) + H(t) >= beta'' for all t >= T.  If the certificate also carries a trajectory check
    (`rh.weil_comb.trajectory_check`: Psi_L + H >= beta_tr on [T_tr, inf)) and T >= T_tr, the larger of
    the two is used; below T_1'' = 2 pi e^{A'} only the trajectory constant is positive."""
    from flint import acb, arb
    if comb_cert is not None:
        b = omega(acb(T)).real - comb_cert.A_prime
        loc = getattr(comb_cert, "local", None)
        if loc is not None:                 # (*) holds for H without the wave packets: subtract their size
            b -= arb(loc.sup_bound())
        tr = getattr(comb_cert, "trajectory", None)
        if tr is not None and tr.get("certified") and T >= tr["T"]:
            bt = arb(tr["beta"])
            if not (b > bt):
                b = bt
        return b
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
    log_err_unit: Optional[float] = None   # its natural log when it is below the double range (err_unit is then 0)


def _ellipse_sup_sigma(L, comb_list, beta, c, h, rho, n_boxes=96, max_depth=10, comb_cert=None):
    """Upper bound for sup |Psi_L(z) (+ H(z)) - beta| on the Bernstein ellipse of the panel
    [c - h, c + h] with parameter rho (maximum principle: on its boundary).  The boundary is covered
    by arcs whose images are enclosed in Arb boxes; an arc whose evaluation is not finite (Arb's
    complex digamma gives up on wide boxes) is bisected, up to max_depth times.  Returns +inf if that
    fails (e.g. a pole of Omega on the boundary).  A comb certificate's H is added through the
    analytic bound sum |h| cosh(omega b_e), b_e = h (rho - 1/rho)/2 the semi-minor axis."""
    from flint import acb, arb
    two_pi = 2 * math.pi
    # H (a finite cosine sum) is bounded analytically on the ellipse: |Im z| <= b_e there, so
    # |H(z)| <= sum |h| cosh(omega b_e); the rest of the symbol is enclosed on Arb boxes
    if comb_cert is None:
        best = 0.0
    elif hasattr(comb_cert, "H_bound_panel"):
        best = comb_cert.H_bound_panel(c, h, rho)
    else:
        best = comb_cert.H_bound(h * (rho - 1 / rho) / 2)
    base = best
    stack = [(two_pi * k / n_boxes, two_pi * (k + 1) / n_boxes, 0) for k in range(n_boxes)]
    while stack:
        t0, t1, depth = stack.pop()
        th = arb((t0 + t1) / 2, (t1 - t0) / 2 * 1.0001)
        e = acb(th.cos(), th.sin())                     # e^{i theta} over the arc (ball)
        zeta = (rho * e + e.conjugate() / rho) / 2       # point of the standard ellipse
        z = acb(c) + h * zeta
        v = symbol(L, z, comb_list) - beta
        if v.is_finite():
            best = max(best, base + _up(abs(v)))
        elif depth < max_depth:
            tm = (t0 + t1) / 2
            stack.append((t0, tm, depth + 1))
            stack.append((tm, t1, depth + 1))
        else:
            return float("inf")
    return best


def panels(L, T, n_max, target=1e-60, w0=0.25, wmax=4.0, rho_choices=(1.5, 2, 3, 4, 6, 8), comb_cert=None,
           log_target: Optional[float] = None, bound_prec: Optional[int] = None):
    """Composite rule on [0, T]: widths w0 near t = 0 (the poles of Omega sit at +-i/2), growing to
    wmax.  For each panel the (rho, nu) with the fewest nodes meeting `target` for the worst entry
    (orders <= n_max) is chosen; err_unit is the rigorous per-entry bound divided by
    2L sqrt((2m+1)(2n+1)) (evaluated in floats from Arb upper bounds; `envelope_block` inflates the
    resulting radii by a relative 1e-9, which covers the float rounding).  log_target: the natural log of the
    target, for targets below the double range (dps > ~300); an error unit below e^-660 is then kept as its log
    (`Panel.log_err_unit`, err_unit = 0) and summed in arb by `envelope_blocks`.  bound_prec: the precision (bits)
    of the ellipse bounds M (ball arithmetic encloses them at any precision; at 280 digits the complex digamma
    near t = 0 took most of half an hour per assembly, at 128 bits seconds)."""
    from flint import ctx
    from flint import arb
    comb_list, A = comb(L)
    beta = beta_star(L, T, comb_cert)  # the exact ball: sup |Psi_L (+ H) - beta| is bounded with it
    edges = [0.0]
    t = 0.0
    while t < T - 1e-12:
        w = w0 if t < 2.0 else (min(wmax, max(w0, t / 4.0)))
        t = min(T, t + w)
        edges.append(t)
    out = []
    lt = math.log(target) if log_target is None else log_target
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
            if bound_prec is None:
                M = _ellipse_sup_sigma(L, comb_list, beta, c, h, rho, comb_cert=comb_cert)
            else:
                old_prec = ctx.prec
                ctx.prec = bound_prec
                try:
                    M = _ellipse_sup_sigma(L, comb_list, beta, c, h, rho, comb_cert=comb_cert)
                finally:
                    ctx.prec = old_prec
            if not math.isfinite(M):
                continue
            growth = math.exp(2 * L * be)               # e^{2L |Im z|}
            # (64/15) M growth worst rho^{-2nu}/(rho^2-1) * h <= target
            const = (64 / 15) * M * growth * worst * h / (rho * rho - 1)
            # log(const) - log(target), not log(const / target): at high precision (target ~ 1e-245) and a large
            # H the quotient overflows a double although both logs are moderate
            nu = max(4, int(math.ceil((math.log(const) - lt) / (2 * math.log(rho)))))
            pre = (64 / 15) * M * growth * h / (rho * rho - 1)
            log_e = math.log(pre) - 2 * nu * math.log(rho)
            if best is None or nu < best[1]:
                best = (rho, nu, pre * rho ** (-2 * nu), None) if log_e > -660.0 else (rho, nu, 0.0, log_e)
        if best is None:
            raise RuntimeError(f"no admissible ellipse for panel [{lo}, {hi}]")
        out.append(Panel(lo, hi, best[1], best[0], best[2], best[3]))
    return out


# ----------------------------------------------------------------------------
# 4. The leading block R_A
# ----------------------------------------------------------------------------
class SymBlocks:
    """A symmetric n x n matrix of arb balls stored as its b x b blocks on and below the block diagonal:
    `blocks[(I, J)]` (I >= J) is an arb_mat holding rows starts[I].. and columns starts[J].. .  A diagonal
    block is stored in full and kept exactly symmetric (every constructor and `__setitem__` maintain this);
    the factorisations below read only its lower triangle.  Memory: n^2/2 + n b/2 entries instead of n^2, and
    every product of the assembly and of the factorisation is a block product (efficient in FLINT for b of
    a few hundred).  `M[i, j]` reads either triangle."""

    def __init__(self, n: int, b: int = 512, blocks: Optional[dict] = None):
        from flint import arb_mat
        if n < 1 or b < 1:
            raise ValueError("n, b >= 1 required")
        self.n, self.b = int(n), int(b)
        self.starts = list(range(0, self.n, self.b))
        self.sizes = [min(self.b, self.n - s) for s in self.starts]
        self.nb = len(self.starts)
        if blocks is None:
            blocks = {(I, J): arb_mat(self.sizes[I], self.sizes[J]) for I in range(self.nb) for J in range(I + 1)}
        self.blocks = blocks

    def nrows(self) -> int:
        return self.n

    def ncols(self) -> int:
        return self.n

    def _loc(self, i, j):
        if i < j:
            i, j = j, i
        if not (0 <= j and i < self.n):
            raise IndexError((i, j))
        I, J = i // self.b, j // self.b
        return (I, J), i - self.starts[I], j - self.starts[J]

    def __getitem__(self, ij):
        key, a, c = self._loc(*ij)
        return self.blocks[key][a, c]

    def __setitem__(self, ij, v):
        key, a, c = self._loc(*ij)
        B = self.blocks[key]
        B[a, c] = v
        if key[0] == key[1]:
            B[c, a] = v

    def to_arb_mat(self):
        """The dense symmetric arb_mat (for tests and small n)."""
        from flint import arb_mat
        M = arb_mat(self.n, self.n)
        for (I, J), B in self.blocks.items():
            si, sj = self.starts[I], self.starts[J]
            for a in range(self.sizes[I]):
                for c in range(self.sizes[J] if I > J else a + 1):
                    v = B[a, c]
                    M[si + a, sj + c] = v
                    M[sj + c, si + a] = v
        return M

    @classmethod
    def from_arb_mat(cls, M, b: int = 512) -> "SymBlocks":
        """The lower triangle of the square arb_mat M (its upper triangle is not read)."""
        S = cls(M.nrows(), b)
        for (I, J), B in S.blocks.items():
            si, sj = S.starts[I], S.starts[J]
            for a in range(S.sizes[I]):
                for c in range(S.sizes[J] if I > J else a + 1):
                    v = M[si + a, sj + c]
                    B[a, c] = v
                    if I == J:
                        B[c, a] = v
        return S


def sector_orders(parity: str, N: int) -> list:
    first = 0 if parity == "even" else 1
    return [first + 2 * j for j in range(N)]


def _bessel_cutoff(x: float, L: float, n_max: int, log10_tau: float) -> int:
    """Smallest order n >= x with sqrt(2L(2n+1)) x^n / (2n+1)!! <= 10^log10_tau (decided in floats with a
    margin of 5 decades), capped at n_max.  For n >= x the bound decreases in n, so it holds for every
    higher order: |G_n(t)| = sqrt(2L(2n+1)) |j_n(Lt)| <= 10^log10_tau there (|j_n(x)| <= x^n/(2n+1)!!)."""
    n = max(0, int(math.ceil(x)))
    lx = math.log10(x) if x > 0 else -1e300
    while n < n_max:
        ldf = (math.lgamma(2 * n + 2) - n * math.log(2) - math.lgamma(n + 1)) / math.log(10)
        if 0.5 * math.log10(2 * L * (2 * n + 1)) + n * lx - ldf <= log10_tau - 5:
            return n
        n += 1
    return n_max


def envelope_block(L, T, N, parity: str, dps: int = 40, target: Optional[float] = None,
                   chunk: int = 4000, verbose: bool = False, comb_cert=None, wmax: float = 4.0,
                   truncate: bool = True, storage: str = "dense", block: int = 512) -> dict:
    """R_A = beta I + s 2 p_A p_A^T + C_AA (arb_mat, orthonormal Legendre modes of the sector) with
    rigorous quadrature error radii.  With a comb certificate (`rh.weil_comb`), beta'' = Omega(T) - A'
    and the symbol on [0, T] is Psi_L + H - beta''.
    truncate: at each node only the orders n <= n_cut(t) with |G_n(t)| possibly above 10^-(dps+40) are
    computed (`_bessel_cutoff`); the rest are dropped and covered by a rigorous radius added to every entry,
    10^-(dps+40) sqrt(2L(2 n_max+1)) sum_k |w_k sigma_k| / pi.  Each chunk of nodes (in increasing t) then
    updates only the leading block of C.
    storage: 'dense' (R an arb_mat) or 'blocks' (R a `SymBlocks` of block size `block`: about a fifth of the
    peak memory and half the products; see `envelope_blocks`).
    Returns {'R', 'beta', 'A', 'panels', 'n_nodes', 'orders', ...}.  (One sector of `envelope_blocks`.)"""
    return envelope_blocks(L, T, {parity: N}, dps=dps, target=target, chunk=chunk, verbose=verbose,
                           comb_cert=comb_cert, wmax=wmax, truncate=truncate, storage=storage, block=block)[parity]


def _add_node_blocks(sec, js, k: int, wk, chunk: int) -> None:
    """Write the Bessel values j_n(L t) of the sector's first k orders (n = orders[i], i < k) into column r of
    the chunk buffers sec['GT'][I] (arb_mat, rows = the orders of block I, one column per node), allocated per
    block on first use; the weight w sigma / pi goes to sec['wts'].  The values are copied, so the node's Python
    objects can be freed at once (a buffer entry is ~160 bytes at 235 digits, a Python arb ~200)."""
    from flint import arb_mat
    S, GT, orders = sec["S"], sec["GT"], sec["orders"]
    r = len(sec["wts"])
    for I in range(S.nb):
        s = S.starts[I]
        if s >= k:
            break
        B = GT[I]
        if B is None:
            B = GT[I] = arb_mat(S.sizes[I], chunk)
        for a in range(min(S.sizes[I], k - s)):
            B[a, r] = js[orders[s + a]]
    sec["wts"].append(wk)


def _flush_blocks(sec) -> None:
    """Add the chunk of nodes in the buffers sec['GT'] (GT_I[i, r] = j_{n_i}(L t_r), zero above a node's cutoff)
    and sec['wts'] (w_r sigma_r / pi) to the lower block triangle sec['S'] of sum_r w_r sigma_r j_m j_n / pi:
    block (I, J) += GT_I W_J with W_J[r, j] = w_r GT_J[j, r].  Blocks above every node's cutoff were never
    allocated and are skipped; unused columns of a last, partial chunk have zero weight rows in W_J."""
    wts = sec["wts"]
    if not wts:
        return
    S, GT = sec["S"], sec["GT"]
    nI = sum(1 for B in GT if B is not None)
    for J in range(nI):
        WJ = GT[J].transpose()                   # chunk x sz_J; row r times w_r
        for r, w in enumerate(wts):
            for c in range(S.sizes[J]):
                WJ[r, c] = WJ[r, c] * w
        for I in range(J, nI):
            S.blocks[(I, J)] = S.blocks[(I, J)] + GT[I] * WJ
        del WJ
    sec["GT"] = [None] * S.nb
    sec["wts"] = []


def _finalize_blocks(S, scale, orders, p, s, beta, drop, qc):
    """R = beta I + s 2 p p^T + C in place on the lower block triangle of S, which holds
    sum_k w_k sigma_k j_m(L t_k) j_n(L t_k) / pi: C_mn = scale_m scale_n S_mn (scale_n = sqrt(2L(2n+1)) times
    the sign of G_n) plus the radius drop + qc sqrt((2m+1)(2n+1)) (dropped orders, quadrature error; both
    padded upward by the caller).  Diagonal blocks are mirrored, so they are exactly symmetric."""
    from flint import arb
    qf = [math.sqrt(2 * m + 1) for m in orders]
    p2 = [s * 2 * x for x in p]
    balls = isinstance(drop, arb) or isinstance(qc, arb)      # radii below the double range (dps > ~300)
    U = arb(0, 1)
    for (I, J), B in S.blocks.items():
        si, sj = S.starts[I], S.starts[J]
        for a in range(S.sizes[I]):
            i = si + a
            sci, pi2, qi = scale[i], p2[i], qf[i]
            for c in range(S.sizes[J] if I > J else a + 1):
                j = sj + c
                rad = (drop + qc * (qi * qf[j])) * U if balls else arb(0, drop + qc * qi * qf[j])
                v = B[a, c] * (sci * scale[j]) + pi2 * p[j] + rad
                if i == j:
                    v += beta
                B[a, c] = v
                if I == J:
                    B[c, a] = v
    return S


class _CheckpointStop(Exception):
    """Raised by `envelope_blocks(_stop_after_checkpoint=k)` after its k-th checkpoint (tests)."""


def _dump_arb_mat(B) -> list:
    """The entries of an arb_mat as Python ints (m, e, rm, re): midpoint m 2^e and radius rm 2^re, exactly."""
    out = []
    for i in range(B.nrows()):
        for j in range(B.ncols()):
            v = B[i, j]
            m, e = v.mid().man_exp()
            rm, re = v.rad().man_exp()
            out.append((int(m), int(e), int(rm), int(re)))
    return out


def _load_arb_mat(nr: int, nc: int, data):
    """Inverse of `_dump_arb_mat`: the midpoints exactly, the radii rounded up (by at most one unit of the 30-bit
    radius mantissa), so that every restored ball contains the saved one."""
    from flint import arb, arb_mat
    B = arb_mat(nr, nc)
    for k, (m, e, rm, re) in enumerate(data):
        B[k // nc, k % nc] = arb(arb((m, e)), arb((rm, re)))
    return B


def _assembly_key(L, T, Ns, dps, target, wmax, truncate, block, chunk, pans, comb_cert) -> str:
    """A fingerprint of an assembly: the parameters, the panels and the comb certificate."""
    import hashlib
    parts = [repr(x) for x in (L, T, sorted(Ns.items()), dps, target, wmax, truncate, block, chunk)]
    parts.append(repr([(P.lo, P.hi, P.nu, P.rho, P.err_unit) for P in pans]))
    if comb_cert is not None:
        parts.append(repr((float(comb_cert.A_prime.upper()), len(getattr(comb_cert, "terms", []) or []),
                           getattr(comb_cert, "H_sup", None))))
        loc = getattr(comb_cert, "local", None)
        if loc is not None:
            parts.append(repr([tuple(a) for a in loc.atoms]))
        tr = getattr(comb_cert, "trajectory", None)
        if tr:
            parts.append(repr((tr.get("T"), tr.get("beta"), tr.get("imposed"))))
    return hashlib.sha256("|".join(parts).encode()).hexdigest()


def _save_assembly(path: str, key: str, sectors: dict, scalars: dict) -> None:
    """Write the block-storage state of an assembly (block by block, to keep the memory spike small)."""
    import os
    import pickle
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump({"key": key, **scalars,
                     "layout": {par: (sec["N"], sec["S"].b, sorted(sec["S"].blocks)) for par, sec in sectors.items()}},
                    f, protocol=5)
        for par, sec in sectors.items():
            for IJ in sorted(sec["S"].blocks):
                B = sec["S"].blocks[IJ]
                pickle.dump((par, IJ, B.nrows(), B.ncols(), _dump_arb_mat(B)), f, protocol=5)
    os.replace(tmp, path)


def _load_assembly(path: str, key: str, sectors: dict):
    """Restore the blocks saved by `_save_assembly` into the sectors' SymBlocks; None (and nothing restored) if
    there is no checkpoint or it belongs to another assembly."""
    import os
    import pickle
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        head = pickle.load(f)
        if head.get("key") != key or set(head["layout"]) != set(sectors):
            return None
        for par, sec in sectors.items():
            N, b, ijs = head["layout"][par]
            if N != sec["N"] or b != sec["S"].b:
                return None
        for par, sec in sectors.items():
            for IJ in head["layout"][par][2]:
                p2, IJ2, nr, nc, data = pickle.load(f)
                if p2 != par or tuple(IJ2) != tuple(IJ):
                    raise ValueError(f"checkpoint {path} is damaged")
                sec["S"].blocks[tuple(IJ)] = _load_arb_mat(nr, nc, data)
    return head


def envelope_blocks(L, T, Ns: dict, dps: int = 40, target: Optional[float] = None, chunk: int = 4000,
                    verbose: bool = False, comb_cert=None, wmax: float = 4.0, truncate: bool = True,
                    storage: str = "dense", block: int = 512, checkpoint: Optional[str] = None,
                    checkpoint_every: int = 16, _stop_after_checkpoint: Optional[int] = None) -> dict:
    """`envelope_block` for several parity sectors at once, Ns = {parity: N}.  The sectors share the panels
    (chosen for the largest order of all), the nodes, the symbol (with H) and the Bessel values at every
    node; only the accumulation is per sector.  Returns {parity: info}.
    storage='blocks': R is a `SymBlocks` (lower block triangle, block size `block`).  The Bessel values of each
    node are copied into per-chunk buffers (`_add_node_blocks`), each chunk adds GT_I W_J to the blocks I >= J
    only (half the products of the dense C + G^T W), the factor sqrt(2L(2n+1)) and the sign of G_n are applied
    once at the end, and R is formed in place: no second n x n matrix.  At most 1024 nodes are held at a time
    (buffers ~ n x chunk entries; larger chunks gain little speed).
    checkpoint (a file name; block storage only, E20): after every `checkpoint_every` panels and after the last
    one, the node buffers are flushed and the blocks saved there (`_save_assembly`: the midpoints exactly, the
    radii rounded up by at most one unit of their 30-bit mantissa, so every restored ball contains the saved one);
    a run started with an existing checkpoint of the same assembly (same parameters, panels and certificate)
    continues after its last panel.  The flushes at the checkpoints are part of the schedule, so a resumed run
    goes through the same arithmetic as an uninterrupted one with checkpoints."""
    from bisect import bisect_right
    from flint import arb, arb_mat, ctx
    from .weil_form import _gl_nodes
    if storage not in ("dense", "blocks"):
        raise ValueError("storage must be 'dense' or 'blocks'")
    use_blocks = storage == "blocks"
    if use_blocks:
        chunk = min(chunk, 1024)
    t0 = time.time()
    sectors = {}
    for parity, N in Ns.items():
        sectors[parity] = {"N": N, "orders": sector_orders(parity, N)}
    n_max = max(sec["orders"][-1] for sec in sectors.values())
    with _ctx(dps):
        log_target = None
        if target is None:
            target = 10.0 ** (-(dps - 5))
            if target < 1e-300:                             # below the double range: carried as a log
                log_target = -(dps - 5) * math.log(10)
        comb_list, A = comb(L)
        beta = beta_star(L, T, comb_cert)
        LL = arb(L)
        pans = panels(L, T, n_max, target=target, wmax=wmax, comb_cert=comb_cert, log_target=log_target,
                      bound_prec=128 if log_target is not None else None)
        tiny = any(P.log_err_unit is not None for P in pans)
        if tiny and not use_blocks:
            raise ValueError("error units below the double range (dps > ~300) need storage='blocks'")
        for sec in sectors.values():
            sec["scale"] = [((2 * LL * (2 * n + 1)).sqrt()) * (1 if (n // 2) % 2 == 0 else -1) for n in sec["orders"]]
            if use_blocks:
                sec["S"] = SymBlocks(sec["N"], block)
                sec["GT"], sec["wts"] = [None] * sec["S"].nb, []
            else:
                sec["C"] = arb_mat(sec["N"], sec["N"])
                sec["rows_G"], sec["rows_W"] = [], []
        n_nodes = 0
        err_unit_total = 0.0
        log10_tau = -(dps + 40)
        sum_abs_wsig = 0.0                      # sum_k |w_k sigma_k| (float upper bound), for the drop radius

        def flush(sec):
            if not sec["rows_G"]:
                return
            k = max(len(r) for r in sec["rows_G"])
            Gm = arb_mat([r + [0] * (k - len(r)) for r in sec["rows_G"]])
            Wm = arb_mat([r + [0] * (k - len(r)) for r in sec["rows_W"]])
            blk = Gm.transpose() * Wm
            del Gm, Wm
            sec["rows_G"], sec["rows_W"] = [], []
            C = sec["C"]
            if k == sec["N"]:
                sec["C"] = C + blk
            else:                               # fold the leading block in now (bounded memory)
                for i in range(k):
                    for j in range(k):
                        C[i, j] += blk[i, j]

        inv_pi = 1 / arb.pi()
        work = ctx.prec
        start, n_saved = 0, 0
        err_small = arb(0)                                  # the error units below the double range, in arb
        if checkpoint is not None and not use_blocks:
            raise ValueError("checkpoint needs storage='blocks'")
        if checkpoint is not None:
            key = _assembly_key(L, T, Ns, dps, target, wmax, truncate, block, chunk, pans, comb_cert)
            head = _load_assembly(checkpoint, key, sectors)
            if head is not None:
                start, n_nodes = head["next_panel"], head["n_nodes"]
                err_unit_total, sum_abs_wsig = head["err_unit_total"], head["sum_abs_wsig"]
                if verbose:
                    print(f"  envelope_blocks: resumed from {checkpoint} after panel {start}/{len(pans)} "
                          f"({n_nodes} nodes)", flush=True)
        for ip, P in enumerate(pans):
            if ip < start:
                if P.log_err_unit is not None:
                    err_small += arb(P.log_err_unit).exp()
                continue
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
                sig = symbol(L, +t, comb_list, comb_cert).real - beta
                ncut = _bessel_cutoff(float(xx.mid()) * (1 + 1e-12), L, n_max, log10_tau) if truncate else n_max
                js = sph_bessel_j_all(xx, ncut)
                wk = (+wh) * sig * inv_pi
                if truncate:
                    sum_abs_wsig += _up(abs(wk))
                for sec in sectors.values():
                    orders, scale = sec["orders"], sec["scale"]
                    k = bisect_right(orders, ncut)         # orders are increasing: those <= ncut
                    if use_blocks:
                        _add_node_blocks(sec, js, k, wk, chunk)
                        if len(sec["wts"]) >= chunk:
                            _flush_blocks(sec)
                        continue
                    g = [scale[i] * js[orders[i]] for i in range(k)]
                    sec["rows_G"].append(g)
                    sec["rows_W"].append([gi * wk for gi in g])
                    if len(sec["rows_G"]) >= chunk:
                        flush(sec)
                n_nodes += 1
            err_unit_total += P.err_unit
            if P.log_err_unit is not None:
                err_small += arb(P.log_err_unit).exp()
            if checkpoint is not None and ((ip + 1) % checkpoint_every == 0 or ip + 1 == len(pans)):
                for sec in sectors.values():
                    _flush_blocks(sec)
                tc = time.time()
                _save_assembly(checkpoint, key, sectors, {"next_panel": ip + 1, "n_nodes": n_nodes,
                                                          "err_unit_total": err_unit_total,
                                                          "sum_abs_wsig": sum_abs_wsig})
                n_saved += 1
                if verbose:
                    print(f"  envelope_blocks: checkpoint after panel {ip + 1}/{len(pans)}, {n_nodes} nodes "
                          f"[{time.time() - t0:.0f}s; saved in {time.time() - tc:.0f}s]", flush=True)
                if _stop_after_checkpoint is not None and n_saved >= _stop_after_checkpoint:
                    raise _CheckpointStop(ip + 1)
        out = {}
        for parity, sec in sectors.items():
            if use_blocks:
                _flush_blocks(sec)
                N, orders = sec["N"], sec["orders"]
                drop = (10.0 ** log10_tau * math.sqrt(2 * L * (2 * n_max + 1)) * sum_abs_wsig * (1 + 1e-9)
                        if truncate else 0.0)
                qc = err_unit_total * 2 * L / math.pi * (1 + 1e-9)
                # the float radius drop + qc ... is valid when qc's padding covers drop (as up to E20, where the
                # quadrature term was ~35 orders above the truncation term); otherwise, and whenever error units
                # fell below the double range, both are balls
                log_drop = (log10_tau * math.log(10) + 0.5 * math.log(2 * L * (2 * n_max + 1))
                            + math.log(max(sum_abs_wsig, 1e-300))) if truncate else -math.inf
                if tiny or (truncate and not (qc > 0 and log_drop < math.log(qc) - 25)):
                    drop = (arb(10) ** log10_tau * arb(math.sqrt(2 * L * (2 * n_max + 1)) * (1 + 1e-12))
                            * arb(sum_abs_wsig) * arb(1 + 1e-9)) if truncate else arb(0)
                    qc = (arb(err_unit_total) + err_small) * 2 * arb(L) / arb.pi() * arb(1 + 1e-9)
                p = pole_vector(L, orders, parity)
                R = _finalize_blocks(sec.pop("S"), sec["scale"], orders, p, 1 if parity == "even" else -1, beta,
                                     drop, qc)
                out[parity] = {"R": R, "beta": beta, "A": A, "panels": pans, "n_nodes": n_nodes, "orders": orders,
                               "A_prime": comb_cert.A_prime if comb_cert is not None else None,
                               "quad_error_unit": err_unit_total, "p": p, "comb": [n for n, _ in comb_list],
                               "seconds": time.time() - t0, "dps": dps, "storage": "blocks",
                               "resumed_from_panel": start,
                               "quad_error_unit_log10": float(((arb(err_unit_total) + err_small).log()
                                                               / arb(10).log()).upper())
                               if (err_unit_total > 0 or tiny) else None}
                continue
            flush(sec)
            N, orders, C = sec["N"], sec["orders"], sec["C"]
            if truncate:
                drop = 10.0 ** log10_tau * math.sqrt(2 * L * (2 * n_max + 1)) * sum_abs_wsig * (1 + 1e-9)
                for i in range(N):
                    for j in range(N):
                        C[i, j] += arb(0, drop)
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
            del C
            sec.pop("C")
            out[parity] = {"R": R, "beta": beta, "A": A, "panels": pans, "n_nodes": n_nodes, "orders": orders,
                           "A_prime": comb_cert.A_prime if comb_cert is not None else None,
                           "quad_error_unit": err_unit_total, "p": p, "comb": [n for n, _ in comb_list],
                           "seconds": time.time() - t0, "dps": dps, "storage": "dense"}
    if verbose:
        q = next(iter(out.values())).get("quad_error_unit_log10")
        qs = f"10^{q:.1f}" if (q is not None and err_unit_total == 0.0) else f"{err_unit_total:.1e}"
        print(f"envelope_blocks L={L} T={T} {Ns}: {n_nodes} nodes in {len(pans)} panels, "
              f"quad err unit {qs}, {time.time() - t0:.0f}s", flush=True)
    return out


# ----------------------------------------------------------------------------
# 5. Tail and coupling bounds
# ----------------------------------------------------------------------------
def tail_bounds(L, T, N, parity: str, n_extra: int = 4000, dps: int = 30, comb_cert=None) -> dict:
    """Rigorous eps_B >= ||R_AD||_2 and eps_D >= ||R_D - beta I||_2 (D = all modes of the sector beyond
    the first N), from eta_n^2 = (1/pi) int_0^T G_n^2 <= (2L/pi) (LT)^{2n} T / ((2n+1)!!)^2 and
    |p_n| <= sqrt((2n+1)/(2L)) 2L (L/2)^n e^{L/2} / (2n+1)!!.  The sums over n >= N-th order are
    bounded term by term up to n_extra further modes and by a geometric tail beyond (ratio <= 1/2)."""
    from flint import arb, acb
    with _ctx(dps):
        LL, TT = arb(L), arb(T)
        _, A = comb(L)
        beta = beta_star(L, T, comb_cert)
        Om0 = omega(acb(0)).real
        OmT = omega(acb(T)).real
        if comb_cert is None:
            # Psi_L - beta = (Omega - Omega(T)) + (A_L - P_L), the second term in [0, 2 A_L]
            sigma_max = max(_up(OmT - Om0), _up(2 * A))
        else:
            # on [0, T]: Omega(0) - A' <= Omega - P_L + H <= Omega(T) + A_L + sup|H| (the lower bound by
            # (*)), so |Psi_L + H - beta| <= max(Omega(T) + A_L + sup|H| - beta, A' + beta - Omega(0)); with
            # beta = Omega(T) - A' this is max(A' + A_L + sup|H|, Omega(T) - Omega(0))
            loc = getattr(comb_cert, "local", None)
            loc_sup = loc.sup_bound() if loc is not None else 0.0     # (*) holds for H without wave packets
            sigma_max = max(_up(OmT + A - beta) + comb_cert.H_sup, _up(comb_cert.A_prime + beta - Om0) + loc_sup)
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


def ldl_mid_blocked(M, shift=None, b: int = 64) -> dict:
    """Approximate LDL^T (no pivoting) of the midpoint of the symmetric arb_mat M - shift I, in
    exact dyadic (midpoint) arithmetic at the working precision: blocked and right-looking.  The working
    matrix is kept as a list of n x b column blocks; after each diagonal block is factored (in Python), the
    blocks to its right are updated with one FLINT product each (full height, with zero rows above the
    panel).  Memory: the column blocks plus L (about two n x n matrices); flops ~ n^3/2 at C speed.
    Negative pivots do not stop it.
    Returns {'L': unit lower triangular arb_mat (exact entries), 'd': pivots (exact arb),
    'n_negative', 'min_pivot', 'first_negative'}.
    For a `SymBlocks` M: `_ldl_mid_symblocks` (b is then the inner block size of its diagonal blocks)."""
    from flint import arb, arb_mat
    if isinstance(M, SymBlocks):
        return _ldl_mid_symblocks(M, shift, inner=b)
    n = M.nrows()
    starts = list(range(0, n, b))
    sh = arb(shift).mid() if shift is not None else None
    blocks = []
    for k0 in starts:                          # column blocks of mid(M) - shift I
        k1 = min(n, k0 + b)
        Sj = arb_mat(n, k1 - k0, [M[i, k0 + j].mid() for i in range(n) for j in range(k1 - k0)])
        if sh is not None:
            for j in range(k1 - k0):
                Sj[k0 + j, j] = Sj[k0 + j, j] - sh
        blocks.append(Sj)
    Lm = arb_mat(n, n)
    d = [None] * n
    n_neg, first_neg = 0, None
    for jb, k0 in enumerate(starts):
        k1 = min(n, k0 + b)
        m = k1 - k0
        Sj = blocks[jb]
        B = [[Sj[k0 + i, j] for j in range(m)] for i in range(m)]
        Lk = [[arb(0)] * m for _ in range(m)]
        dk = [None] * m
        for i in range(m):
            sv = B[i][i]
            Li = Lk[i]
            for j in range(i):
                sv -= Li[j] * Li[j] * dk[j]
            sv = sv.mid()
            dk[i] = sv
            if not (sv > 0):
                n_neg += 1
                if first_neg is None:
                    first_neg = k0 + i
            Li[i] = arb(1)
            for r in range(i + 1, m):
                Lr = Lk[r]
                t = B[r][i]
                for j in range(i):
                    t -= Lr[j] * Li[j] * dk[j]
                Lr[i] = (t / sv).mid()
        for i in range(m):
            d[k0 + i] = dk[i]
            for j in range(i + 1):
                Lm[k0 + i, k0 + j] = Lk[i][j]
        nb = n - k1
        if nb == 0:
            blocks[jb] = None
            break
        # panel below the diagonal block: X (Lk Dk)^T = S[k1:, k0:k1]  <=>  (Lk Dk) X^T = S[k1:, k0:k1]^T
        LD = arb_mat(m, m, [Lk[i][j] * dk[j] if j <= i else 0 for i in range(m) for j in range(m)])
        SbT = arb_mat(m, nb, [Sj[k1 + r, i] for i in range(m) for r in range(nb)])
        blocks[jb] = None                      # this column block is done
        XT = LD.solve(SbT, algorithm="approx").mid()
        W = arb_mat(n, m)                      # X in rows k1.., zero above
        WD = arb_mat(n, m)
        for r in range(nb):
            for i in range(m):
                x = XT[i, r]
                Lm[k1 + r, k0 + i] = x
                W[k1 + r, i] = x
                WD[k1 + r, i] = x * dk[i]
        for jb2 in range(jb + 1, len(starts)):  # S[:, block jb2] -= W (WD[block jb2 rows, :])^T
            c0 = starts[jb2]
            c1 = min(n, c0 + b)
            WDc = arb_mat(m, c1 - c0, [WD[c0 + cc, i] for i in range(m) for cc in range(c1 - c0)])
            blocks[jb2] = (blocks[jb2] - W * WDc).mid()
        del W, WD
    minp = min(float(x) for x in d)
    return {"L": Lm, "d": d, "n_negative": n_neg, "min_pivot": minp, "first_negative": first_neg}


def residual_certify(M, shift, b: int = 64) -> dict:
    """Certified lower bound for lambda_min(M), M a symmetric arb_mat, by an approximate LDL^T of the
    midpoint of A = M - shift I (exact dyadic numbers, no balls: no wrapping effect; `ldl_mid_blocked`)
    and a rigorous residual: if A = Lt Dt Lt^T + E with Lt, Dt exact and Dt >= 0, then Lt Dt Lt^T is
    positive semidefinite, so lambda_min(M) >= shift - ||E||_2 >= shift - ||E||_F.  E is computed in
    ball arithmetic (it contains A - Lt Dt Lt^T for every A in the balls).
    Returns {'certified', 'lower_bound' (arb), 'residual_F' (arb upper bound), 'min_pivot'}.
    For a `SymBlocks` M: `_residual_certify_symblocks` (same certificate, block by block)."""
    from flint import arb, arb_mat
    if isinstance(M, SymBlocks):
        return _residual_certify_symblocks(M, shift, inner=b)
    n = M.nrows()
    f = ldl_mid_blocked(M, shift, b=b)
    if f["n_negative"]:
        return {"certified": False, "failed_at": f["first_negative"], "lower_bound": None, "residual_F": None,
                "min_pivot": f["min_pivot"], "n_negative": f["n_negative"]}
    Lt = f["L"]
    d = f["d"]
    sh = arb(shift) if shift is not None else arb(0)
    min_pivot = f["min_pivot"]
    del f
    # E = M - shift I - Lt D Lt^T, in column blocks of width w (memory: two n x w blocks, not five n x n
    # matrices).  E is symmetric, so only its lower triangle is evaluated and the off-diagonal part counted twice.
    w = 256
    fro2 = arb(0)
    for c0 in range(0, n, w):
        c1 = min(n, c0 + w)
        Bc = arb_mat(n, c1 - c0)             # D (Lt[c0:c1, :])^T
        for j in range(c1 - c0):
            r = c0 + j
            for k in range(r + 1):
                Bc[k, j] = d[k] * Lt[r, k]
        P = Lt * Bc                          # (Lt D Lt^T)[:, c0:c1]
        del Bc
        for j in range(c1 - c0):
            c = c0 + j
            e = abs(M[c, c] - sh - P[c, j]).upper()
            fro2 += e * e
            for i in range(c + 1, n):
                e = abs(M[i, c] - P[i, j]).upper()
                fro2 += 2 * e * e
        del P
    res = fro2.sqrt()
    lower = (arb(shift) if shift is not None else arb(0)) - res
    return {"certified": bool(lower > 0), "lower_bound": lower, "residual_F": res,
            "min_pivot": min_pivot, "failed_at": None, "n_negative": 0}


def inertia_count(M, shift) -> int:
    """Number of eigenvalues of the symmetric arb_mat M below `shift`, from the negative pivots of the
    midpoint LDL^T of M - shift I (Sylvester's law of inertia; floating-point, not rigorous)."""
    return ldl_mid_blocked(M, shift)["n_negative"]


def lambda_min_estimate(M, lo_exp: int = -60, hi_exp: int = 0, digits: float = 0.05) -> dict:
    """Approximate lambda_min of the symmetric arb_mat M (not rigorous; a diagnostic).  Positive
    definite M: inverse iteration (`approx_lambda_min`).  Otherwise bisection on the inertia of
    M + 10^x I in x (log scale) gives -lambda_min to within `digits` decades."""
    from flint import arb
    if inertia_count(M, None) == 0:
        lam, _ = approx_lambda_min(M)
        return {"lambda_min": float(lam.mid()), "n_negative": 0}
    k = inertia_count(M, None)
    a, b = float(lo_exp), float(hi_exp)        # find x with M + 10^x I positive definite iff x > x*
    if inertia_count(M, arb(-(10.0 ** b))) > 0:
        return {"lambda_min": -float("inf"), "n_negative": k}
    while b - a > digits:
        x = (a + b) / 2
        if inertia_count(M, arb(-(10.0 ** x))) > 0:
            a = x
        else:
            b = x
    return {"lambda_min": -(10.0 ** ((a + b) / 2)), "n_negative": k}


def approx_lambda_min_ldl(M, iters: int = 6, b: int = 64):
    """As `approx_lambda_min`, but the inverse iteration reuses one blocked LDL^T of the midpoint of M
    (`ldl_mid_blocked`, O(N^3) at C speed) and triangular solves (O(N^2) each), instead of one full LU
    solve per iteration.  Returns (lambda, x) or None if M has negative pivots (not positive definite).
    For a `SymBlocks` M: `_approx_lambda_min_symblocks`."""
    from flint import arb, arb_mat
    if isinstance(M, SymBlocks):
        return _approx_lambda_min_symblocks(M, iters=iters, inner=b)
    n = M.nrows()
    f = ldl_mid_blocked(M, None, b=b)
    if f["n_negative"]:
        return None
    Lm, d = f["L"], f["d"]
    del f
    x = [arb(1) / (1 + i) for i in range(n)]
    lam = None
    for _ in range(iters):
        z = [None] * n                          # L z = x
        for i in range(n):
            acc = x[i]
            for j in range(i):
                acc -= Lm[i, j] * z[j]
            z[i] = acc.mid()
        w = [(z[i] / d[i]).mid() for i in range(n)]
        y = [None] * n                          # L^T y = w
        for i in range(n - 1, -1, -1):
            acc = w[i]
            for j in range(i + 1, n):
                acc -= Lm[j, i] * y[j]
            y[i] = acc.mid()
        nrm = sum((v * v for v in y), arb(0)).sqrt()
        x = [(v / nrm).mid() for v in y]
        xm = arb_mat([[v] for v in x])
        Mx = M * xm                             # no midpoint copy of M (memory)
        lam = sum((x[i] * Mx[i, 0] for i in range(n)), arb(0)).mid()
    return lam, arb_mat([[v] for v in x])


def approx_lambda_min(M, iters: int = 6):
    """Approximate eigenvalue of the symmetric arb_mat M closest to 0 (and its vector), by inverse
    iteration on the midpoint.  Not rigorous, and it is lambda_min only if M is positive definite:
    it is used only to choose the shift that `residual_certify` then checks rigorously (an indefinite
    M is caught there by a negative pivot).
    For a `SymBlocks` M: inverse iteration with its (possibly indefinite) blocked LDL^T, no dense copy."""
    from flint import arb, arb_mat
    if isinstance(M, SymBlocks):
        return _approx_lambda_min_symblocks(M, iters=iters, allow_indefinite=True)
    n = M.nrows()
    Mm = M.mid()
    x = arb_mat([[arb(1) / (1 + i)] for i in range(n)])
    lam = None
    for _ in range(iters):
        y = Mm.solve(x, algorithm="approx")
        nrm = sum((y[i, 0] ** 2 for i in range(n)), arb(0)).sqrt()
        x = arb_mat([[y[i, 0] / nrm] for i in range(n)])
        Mx = Mm * x
        lam = sum((x[i, 0] * Mx[i, 0] for i in range(n)), arb(0))
    return lam, x


def _ldl_mid_symblocks(M, shift=None, inner: int = 64) -> dict:
    """Approximate LDL^T (no pivoting) of the midpoint of the `SymBlocks` M - shift I, right-looking by blocks
    of size M.b, on a midpoint copy of the lower block triangle that is overwritten by the factor (memory: one
    copy, n^2/2 + n b/2 entries, besides M).  Block column K: the diagonal block by `ldl_mid_blocked` (inner
    block size `inner`), the blocks below it by one approximate solve with L_KK D_K, then
    S_IJ -= L_IK (D_K L_JK^T) for I >= J > K (block products).  Negative pivots do not stop it.
    Returns {'L': {(I, J): block of the unit lower triangular factor, I >= J; exact entries, zero above the
    diagonal of a diagonal block}, 'd': pivots (exact arb), 'n_negative', 'min_pivot', 'first_negative'}."""
    from flint import arb, arb_mat
    nb, st, sz = M.nb, M.starts, M.sizes
    S = {key: B.mid() for key, B in M.blocks.items()}
    if shift is not None:
        sh = arb(shift).mid()
        for K in range(nb):
            B = S[(K, K)]
            for a in range(sz[K]):
                B[a, a] = B[a, a] - sh
    Lb = {}
    d = [None] * M.n
    n_neg, first_neg = 0, None
    for K in range(nb):
        m = sz[K]
        f = ldl_mid_blocked(S.pop((K, K)), None, b=inner)
        LKK, dK = f["L"], f["d"]
        if f["n_negative"]:
            n_neg += f["n_negative"]
            if first_neg is None:
                first_neg = st[K] + f["first_negative"]
        del f
        d[st[K]: st[K] + m] = dK
        Lb[(K, K)] = LKK
        if K == nb - 1:
            break
        below = range(K + 1, nb)
        cols = sum(sz[I] for I in below)
        # L_IK (L_KK D_K)^T = S_IK for all I > K at once:  (L_KK D_K) X = [S_IK^T ...],  L_IK = X_I^T
        LD = arb_mat(m, m, [LKK[a, c] * dK[c] if c <= a else 0 for a in range(m) for c in range(m)])
        RHS = arb_mat(m, cols)
        off = 0
        for I in below:
            B = S.pop((I, K))
            for r in range(sz[I]):
                for a in range(m):
                    RHS[a, off + r] = B[r, a]
            off += sz[I]
            del B
        X = LD.solve(RHS, algorithm="approx")
        del RHS, LD
        X = X.mid()
        off = 0
        for I in below:
            B = arb_mat(sz[I], m)
            for r in range(sz[I]):
                for a in range(m):
                    B[r, a] = X[a, off + r]
            Lb[(I, K)] = B
            off += sz[I]
        del X
        for J in below:                       # trailing update of the lower block triangle
            DLT = Lb[(J, K)].transpose()       # m x sz[J], then row a times d_a: D_K L_JK^T
            for a in range(m):
                da = dK[a]
                for c in range(sz[J]):
                    DLT[a, c] = DLT[a, c] * da
            for I in range(J, nb):
                S[(I, J)] = (S[(I, J)] - Lb[(I, K)] * DLT).mid()
            del DLT
    return {"L": Lb, "d": d, "n_negative": n_neg, "min_pivot": min(float(x) for x in d),
            "first_negative": first_neg}


def _residual_certify_symblocks(M, shift, inner: int = 64) -> dict:
    """`residual_certify` for a `SymBlocks` M: the blocked LDL^T of mid(M) - shift I (`_ldl_mid_symblocks`) and
    the rigorous residual E = M - shift I - L D L^T in ball arithmetic, block by block: for each block column J
    the blocks D_K L_JK^T (K <= J) are formed once, then E_IJ = M_IJ - sum_{K <= J} L_IK (D_K L_JK^T) for I >= J
    and only its Frobenius norm is kept (lower triangle, off-diagonal entries counted twice).  If all pivots
    are positive, L D L^T is positive semidefinite and lambda_min(M) >= shift - ||E||_F."""
    from flint import arb
    f = _ldl_mid_symblocks(M, shift, inner)
    if f["n_negative"]:
        return {"certified": False, "failed_at": f["first_negative"], "lower_bound": None, "residual_F": None,
                "min_pivot": f["min_pivot"], "n_negative": f["n_negative"]}
    Lb, d, min_pivot = f["L"], f["d"], f["min_pivot"]
    del f
    nb, st, sz = M.nb, M.starts, M.sizes
    sh = arb(shift) if shift is not None else arb(0)
    fro2 = arb(0)
    for J in range(nb):
        DLT = []
        for K in range(J + 1):
            Tm = Lb[(J, K)].transpose()        # sz[K] x sz[J]
            for a in range(sz[K]):
                da = d[st[K] + a]
                for c in range(sz[J]):
                    Tm[a, c] = Tm[a, c] * da
            DLT.append(Tm)
        for I in range(J, nb):
            P = Lb[(I, 0)] * DLT[0]
            for K in range(1, J + 1):
                P = P + Lb[(I, K)] * DLT[K]
            E = M.blocks[(I, J)] - P
            del P
            for a in range(sz[I]):
                if I > J:
                    for c in range(sz[J]):
                        u = abs(E[a, c]).upper()
                        fro2 += 2 * u * u
                else:
                    for c in range(a):
                        u = abs(E[a, c]).upper()
                        fro2 += 2 * u * u
                    u = abs(E[a, a] - sh).upper()
                    fro2 += u * u
            del E
        del DLT
    res = fro2.sqrt()
    lower = sh - res
    return {"certified": bool(lower > 0), "lower_bound": lower, "residual_F": res,
            "min_pivot": min_pivot, "failed_at": None, "n_negative": 0}


def _symblocks_matvec(M, x: list) -> list:
    """M x for a `SymBlocks` M and a list x of arb (block products; diagonal blocks are symmetric)."""
    from flint import arb_mat
    nb, st, sz = M.nb, M.starts, M.sizes
    col = [arb_mat(sz[I], 1, x[st[I]: st[I] + sz[I]]) for I in range(nb)]
    row = [arb_mat(1, sz[I], x[st[I]: st[I] + sz[I]]) for I in range(nb)]
    out = []
    for I in range(nb):
        acc = M.blocks[(I, I)] * col[I]
        for J in range(I):
            acc = acc + M.blocks[(I, J)] * col[J]
        for J in range(I + 1, nb):
            acc = acc + (row[J] * M.blocks[(J, I)]).transpose()
        out.extend(acc[a, 0] for a in range(sz[I]))
    return out


def _approx_lambda_min_symblocks(M, iters: int = 6, inner: int = 64, allow_indefinite: bool = False):
    """`approx_lambda_min_ldl` for a `SymBlocks` M: inverse iteration with one blocked LDL^T of mid(M)
    (block triangular solves, the diagonal blocks by substitution) and block products for M x.  Returns
    (lambda, x) or None if a pivot is negative, unless allow_indefinite (then the eigenvalue closest to 0, as
    `approx_lambda_min`).  Not rigorous: it only chooses the shift that `residual_certify` checks."""
    from flint import arb, arb_mat
    f = _ldl_mid_symblocks(M, None, inner)
    if f["n_negative"] and not allow_indefinite:
        return None
    Lb, d = f["L"], f["d"]
    del f
    nb, st, sz, n = M.nb, M.starts, M.sizes, M.n
    x = [arb(1) / (1 + i) for i in range(n)]
    lam = None
    for _ in range(iters):
        z = [None] * n                          # L z = x
        for I in range(nb):
            s = st[I]
            acc = arb_mat(sz[I], 1, x[s: s + sz[I]])
            for K in range(I):
                acc = acc - Lb[(I, K)] * arb_mat(sz[K], 1, z[st[K]: st[K] + sz[K]])
            LII = Lb[(I, I)]
            for a in range(sz[I]):
                v = acc[a, 0]
                for c in range(a):
                    v -= LII[a, c] * z[s + c]
                z[s + a] = v.mid()
        w = [(z[i] / d[i]).mid() for i in range(n)]
        y = [None] * n                          # L^T y = w
        for K in range(nb - 1, -1, -1):
            s = st[K]
            acc = arb_mat(1, sz[K], w[s: s + sz[K]])
            for I in range(K + 1, nb):
                acc = acc - arb_mat(1, sz[I], y[st[I]: st[I] + sz[I]]) * Lb[(I, K)]
            LKK = Lb[(K, K)]
            for a in range(sz[K] - 1, -1, -1):
                v = acc[0, a]
                for c in range(a + 1, sz[K]):
                    v -= LKK[c, a] * y[s + c]
                y[s + a] = v.mid()
        nrm = sum((v * v for v in y), arb(0)).sqrt()
        x = [(v / nrm).mid() for v in y]
        Mx = _symblocks_matvec(M, x)
        lam = sum((x[i] * Mx[i] for i in range(n)), arb(0)).mid()
    return lam, arb_mat([[v] for v in x])


# ----------------------------------------------------------------------------
# 7. The certificate
# ----------------------------------------------------------------------------
def _threads(n):
    """Context: FLINT's thread count set to n (None: unchanged), restored on exit."""
    from flint import ctx

    class _T:
        def __enter__(self):
            self.old = ctx.threads
            if n is not None:
                ctx.threads = int(n)

        def __exit__(self, *a):
            ctx.threads = self.old
    return _T()


def certify_window(L, parity: str, T, N, dps: int = 40, lam0=None, safety: float = 0.98,
                   verbose: bool = False, comb_cert=None, wmax: float = 4.0, chunk: int = 4000,
                   storage: str = "blocks", block: int = 512, threads: Optional[int] = None) -> dict:
    """Certified lower bound c with Q(f) >= c ||f||^2 for every real f of the sector supported in
    [-L, L]:  c = min(lam0, beta - eps_D) - eps_B, valid when R_A - lam0 I is certified positive
    definite.  lam0 defaults to safety x (approximate lambda_min(R_A)).
    storage='blocks' (default): R_A as a `SymBlocks` (lower block triangle, block size `block`), the
    factorisations by blocks in place; peak memory about one n x n matrix of entries instead of about six, and
    fewer products.  'dense' is the arb_mat path used up to E18 (same certificate).  threads: FLINT threads
    for the block products (None: unchanged)."""
    with _threads(threads):
        t0 = time.time()
        blk = envelope_block(L, T, N, parity, dps=dps, verbose=verbose, comb_cert=comb_cert, wmax=wmax, chunk=chunk,
                             storage=storage, block=block)
        return _certify_block(blk, L, parity, T, N, dps, lam0, safety, verbose, comb_cert, t0)


def certify_windows(L, T, Ns: dict, dps: int = 40, safety: float = 0.98, verbose: bool = False, comb_cert=None,
                    wmax: float = 4.0, chunk: int = 4000, storage: str = "blocks", block: int = 512,
                    threads: Optional[int] = None, checkpoint: Optional[str] = None,
                    checkpoint_every: int = 16) -> dict:
    """`certify_window` for several sectors (Ns = {parity: N}) from one shared-node assembly
    (`envelope_blocks`); the sectors are then certified one after the other, each block released after its
    certificate.  checkpoint: see `envelope_blocks` (the assembly resumes after a restart).  Returns
    {parity: result}."""
    with _threads(threads):
        t0 = time.time()
        blks = envelope_blocks(L, T, Ns, dps=dps, verbose=verbose, comb_cert=comb_cert, wmax=wmax, chunk=chunk,
                               storage=storage, block=block, checkpoint=checkpoint,
                               checkpoint_every=checkpoint_every)
        out = {}
        for parity in list(blks):
            blk = blks.pop(parity)
            out[parity] = _certify_block(blk, L, parity, T, Ns[parity], dps, None, safety, verbose, comb_cert, t0)
            del blk
        return out


def _certify_block(blk, L, parity, T, N, dps, lam0, safety, verbose, comb_cert, t0) -> dict:
    from flint import arb
    with _ctx(dps):
        R = blk["R"]
        est = approx_lambda_min_ldl(R)
        # est is None: the midpoint has a negative pivot, so R_A is (numerically) indefinite and the check
        # below must fail; approx_lambda_min then only gives the eigenvalue closest to 0
        lam_approx = est[0] if est is not None else approx_lambda_min(R)[0]
        auto = lam0 is None
        if auto:
            lam0 = lam_approx * safety if lam_approx > 0 else lam_approx
        cert = residual_certify(R, arb(lam0))
        retries = []
        if auto and est is not None and lam_approx > 0 and not cert["certified"]:
            # the midpoint is positive definite, so the shift was too close to (or above) lambda_min: the inverse
            # iteration's Rayleigh quotient overestimates lambda_min when the two smallest eigenvalues are close.
            # Smaller shifts give a smaller but valid bound (each try is one more LDL^T, no re-assembly).
            for fac in (0.5, 0.1, 0.01):
                retries.append(float(arb(lam0).mid()))
                lam0 = lam_approx * fac
                cert = residual_certify(R, arb(lam0))
                if cert["certified"]:
                    break
        tb = tail_bounds(L, T, N, parity, comb_cert=comb_cert)
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
               "A_prime": float(comb_cert.A_prime.upper()) if comb_cert is not None else None,
               "comb_X": comb_cert.X if comb_cert is not None else None,
               "lambda0": float(arb(lam0).mid()), "ldl_certified": cert["certified"],
               "midpoint_indefinite": est is None, "shift_retries": retries,
               "ldl_failed_at": cert.get("failed_at"),
               "ldl_min_pivot": cert["min_pivot"],
               "residual_F": float(cert["residual_F"].upper()) if cert["residual_F"] is not None else None,
               "eps_B": tb["eps_B"], "eps_D": tb["eps_D"],
               "certified_lower_bound": c, "positive": bool(c is not None and c > 0),
               "storage": blk.get("storage", "dense"), "seconds": time.time() - t0}
    if verbose:
        print(f"certify_window L={L} {parity} T={T} N={N}: lambda_min(R_A)~{out['lambda_approx']:.4e} "
              f"ldl={cert['certified']} eps_B={tb['eps_B']:.1e} eps_D={tb['eps_D']:.1e} -> c={c} "
              f"[{out['seconds']:.0f}s]", flush=True)
    return out
