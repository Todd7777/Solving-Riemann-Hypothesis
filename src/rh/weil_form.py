r"""Spectral analysis of Weil's quadratic form on test functions of bounded support.

Setting
-------
For a real test function f supported in [-a, a] (additive variable u = log x)
let g = f * f~ (f~(u) = f(-u)), so g is even, supported in [-2a, 2a], and
h = g^ = |f^|^2 on the real line, with f^(r) = int f(u) e^{iru} du.  Weil's
explicit formula (see `rh.explicit_formula`, verified there to 1e-15) gives

    Q_a(f) := sum_rho f^(tau_rho) f^(-tau_rho)                 (rho = 1/2 + i tau_rho)
            = 2 F(1/2) F(-1/2)                                  (poles; F(s) = int f e^{su} du)
              - 2 sum_{n <= e^{2a}} Lambda(n) n^{-1/2} g(log n)  (primes)
              - (gamma_E + log pi) g(0)
              + 2 int_0^inf [g(0) e^{-2u} - g(u) e^{-u/2}] / (1 - e^{-2u}) du   (archimedean).

Under RH the zero side is 2 sum_{gamma>0} |f^(gamma)|^2 >= 0, and Weil's criterion
says RH holds iff Q_a >= 0 for every a.  The arithmetic side uses only the
primes up to e^{2a} and the Gamma factor: no zero of zeta enters.

What this module computes
-------------------------
Q_a restricted to the J-dimensional space of polynomials of degree < J on
[-a, a] (Legendre basis phi_j(u) = P_j(u/a)) is a J x J symmetric matrix; it
splits into even and odd sectors (Q_a is invariant under f(u) -> f(-u)).  Its
smallest generalised eigenvalue lambda_min(a, J) (normalised by ||f||_2^2) is
an upper bound (Rayleigh–Ritz) for the infimum of Q_a(f)/||f||^2 over all f
supported in [-a, a].

Because under RH Q_a(f) = 2 sum |f^(gamma_n)|^2, a test function with tiny
Q_a(f)/||f||^2 must have f^ (an entire function of exponential type a) nearly
vanishing at every zero gamma_n of zeta: the minimising eigenvector, built from
primes alone, "knows" the zeros.  The experiment script compares the real zeros
of f^_min with the zeros of zeta computed independently.

Related work (2021–2026): Connes–Consani, *Spectral triples and zeta-cycles*
(arXiv:2106.01715) observed extremely small eigenvalues of this restricted form
with eigenvectors close to prolate spheroidal functions; Connes–Consani–
Moscovici, *Zeta zeros and prolate wave operators* (arXiv:2310.18423) and
*Zeta spectral triples* (arXiv:2511.22755); X. Zhu, *Weil positivity in compact
windows* (arXiv:2608.24827) certifies positivity for half-width 0.8 and reports
upper bounds ~3e-283 for the smallest eigenvalue at half-width 2, warning that
float64 evaluation produces spurious negative eigenvalues; arXiv:2610.32704
reports zeros of the Fourier transform of the minimiser approximating the zeros
of xi.  (These were read only through search-engine summaries: arxiv.org is not
reachable from this environment.)  Yoshida (1992) and Connes–Consani (2021,
*Weil positivity and trace formula, the archimedean place*) proved positivity
when no prime contributes, i.e. a <= (log 2)/2.

Numerics
--------
All matrices are assembled in Arb ball arithmetic (python-flint) at a working
precision of `dps` decimal digits, because the smallest eigenvalues are
astronomically small (cancellation between the pole, archimedean and prime
terms is total to many digits).  The correlations c_ij(u) = int phi_i(v)
phi_j(v-u) dv are integrated *exactly* by Gauss–Legendre quadrature (the
integrand is a polynomial).  The archimedean integral is not polynomial; it is
computed with Gauss–Legendre on [0, 2a] (the integrand is analytic in the strip
|Im u| < pi) and its quadrature error is estimated by comparing two node counts.
The eigenproblem is solved with mpmath's symmetric eigensolver at the same
precision.  Nothing here is a proof: the quadrature error is estimated, not
bounded, and only a finite-dimensional subspace is examined.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

__all__ = ["WeilFormMatrices", "build_weil_form", "smallest_eigenpairs", "fourier_transform_mp",
           "fourier_transform_float", "real_zeros_of_fhat", "zero_match", "zero_side_value", "legendre_values",
           "weyl_certificate"]


# ----------------------------------------------------------------------------
# Helpers in Arb
# ----------------------------------------------------------------------------
def _gl_nodes(m: int):
    """Gauss–Legendre nodes and weights on [-1, 1] as arb balls (cached per precision)."""
    from flint import arb, ctx
    key = (m, ctx.prec)
    if key not in _GL_CACHE:
        nodes, weights = [], []
        for k in range(m):
            x, w = arb.legendre_p_root(m, k, weight=True)
            nodes.append(x)
            weights.append(w)
        _GL_CACHE[key] = (nodes, weights)
    return _GL_CACHE[key]


_GL_CACHE: dict = {}


def legendre_values(xs, J: int):
    """Matrix (len(xs) x J) of P_0..P_{J-1} at the arb points xs (three-term recurrence)."""
    from flint import arb, arb_mat
    M = arb_mat(len(xs), J)
    for k, x in enumerate(xs):
        p0, p1 = arb(1), x
        M[k, 0] = p0
        if J > 1:
            M[k, 1] = p1
        for n in range(1, J - 1):
            p2 = ((2 * n + 1) * x * p1 - n * p0) / (n + 1)
            M[k, n + 1] = p2
            p0, p1 = p1, p2
    return M


def _correlation(t, a, J: int, m: int):
    """Symmetrised correlation matrix s(u), u = a t, t in [0, 2]:
    s_ij(u) = (c_ij(u) + c_ji(u))/2,  c_ij(u) = int phi_i(v) phi_j(v - u) dv
            = a int_{t-1}^{1} P_i(x) P_j(x - t) dx   (exact Gauss–Legendre with m >= J nodes)."""
    from flint import arb, arb_mat
    nodes, weights = _gl_nodes(m)
    half = (2 - t) / 2
    xs = [(t - 1) + half * (y + 1) for y in nodes]
    B1 = legendre_values(xs, J)
    B2 = legendre_values([x - t for x in xs], J)
    for k in range(m):                       # scale rows of B2 by the quadrature weights
        wk = weights[k] * half * a
        for j in range(J):
            B2[k, j] = B2[k, j] * wk
    C = B1.transpose() * B2
    return (C + C.transpose()) * arb(0.5)


# ----------------------------------------------------------------------------
# Assembly
# ----------------------------------------------------------------------------
@dataclass
class WeilFormMatrices:
    a: float
    J: int
    dps: int
    n_quad: int
    Q: object            # arb_mat J x J, full quadratic form (arithmetic side)
    G: object            # arb_mat J x J, Gram matrix of the basis (diagonal)
    P: object            # pole part
    A: object            # archimedean part
    Pr: object           # prime part
    quad_error_estimate: float = float("nan")
    n_prime_powers: int = 0
    info: dict = field(default_factory=dict)

    def sector(self, parity: str):
        """(Q, G, indices) restricted to the even ('even') or odd ('odd') basis functions."""
        from flint import arb_mat
        idx = [j for j in range(self.J) if (j % 2 == 0) == (parity == "even")]
        n = len(idx)
        Qs, Gs = arb_mat(n, n), arb_mat(n, n)
        for r, i in enumerate(idx):
            for c, j in enumerate(idx):
                Qs[r, c] = self.Q[i, j]
                Gs[r, c] = self.G[i, j]
        return Qs, Gs, idx


def _archimedean(a, J, m, n_quad):
    """2 int_0^{2a} [s(0) e^{-2u} - s(u) e^{-u/2}]/(1 - e^{-2u}) du by Gauss–Legendre in t = u/a."""
    from flint import arb, arb_mat
    nodes, weights = _gl_nodes(n_quad)
    s0 = _correlation(arb(0), a, J, m)
    total = arb_mat(J, J)
    for y, w in zip(nodes, weights):
        t = y + 1                        # t in (0, 2)
        u = a * t
        su = _correlation(t, a, J, m)
        e2 = (-2 * u).exp()
        eh = (-u / 2).exp()
        den = 1 - e2
        total += (s0 * (e2 / den) - su * (eh / den)) * (w * a)   # dt -> du = a dt; [0,2] has jacobian 1
    return total * 2, s0


def build_weil_form(a: float, J: int, dps: int = 60, n_quad: int | None = None,
                    estimate_quad_error: bool = True) -> WeilFormMatrices:
    """Assemble the arithmetic side of Q_a on span{P_0(u/a), ..., P_{J-1}(u/a)}.

    a      half-width of the support of f (additive variable), primes up to e^{2a} enter
    J      number of Legendre basis functions
    dps    working precision (decimal digits)
    n_quad Gauss–Legendre nodes for the archimedean integral (default J + 60)
    """
    from flint import arb, arb_mat, ctx
    from .explicit_formula import vonmangoldt_table
    old = ctx.dps
    ctx.dps = int(dps)
    try:
        aa = arb(a) if not isinstance(a, str) else arb(a)
        m = J + 2                                    # exact for polynomial correlations
        if n_quad is None:
            # the archimedean integrand is analytic in |Im u| < pi; Gauss–Legendre on
            # [0, 2a] converges like rho^(-2n) with rho = pi/a + sqrt(1 + (pi/a)^2)
            rho = math.pi / float(a) + math.sqrt(1 + (math.pi / float(a)) ** 2)
            n_quad = J + 20 + int(math.ceil(1.15 * dps / (2 * math.log10(rho))))
        # --- Gram matrix (diagonal)
        G = arb_mat(J, J)
        for j in range(J):
            G[j, j] = 2 * aa / (2 * j + 1)
        # --- pole part: Phi_j(+-1/2) = a int_{-1}^{1} P_j(x) e^{+-a x/2} dx (Gauss–Legendre, entire integrand)
        mq = J + 40
        nodes, weights = _gl_nodes(mq)
        Lv = legendre_values(nodes, J)
        Phi_p = [arb(0)] * J
        Phi_m = [arb(0)] * J
        for k in range(mq):
            ep = (aa * nodes[k] / 2).exp()
            em = 1 / ep
            for j in range(J):
                Phi_p[j] += weights[k] * Lv[k, j] * ep
                Phi_m[j] += weights[k] * Lv[k, j] * em
        Phi_p = [x * aa for x in Phi_p]
        Phi_m = [x * aa for x in Phi_m]
        P = arb_mat(J, J)
        for i in range(J):
            for j in range(J):
                P[i, j] = Phi_p[i] * Phi_m[j] + Phi_m[i] * Phi_p[j]
        # --- archimedean part
        arch, s0 = _archimedean(aa, J, m, n_quad)
        logterm = -(1 - (-4 * aa).exp()).log()       # 2 * int_{2a}^inf e^{-2u}/(1-e^{-2u}) du
        A = arch + s0 * (logterm - arb.const_euler() - arb.pi().log())
        quad_err = float("nan")
        if estimate_quad_error:
            arch2, _ = _archimedean(aa, J, m, int(n_quad * 1.5) + 4)
            diff = arch2 - arch
            quad_err = max(abs(float(diff[i, j].mid())) for i in range(J) for j in range(J))
            A = arch2 + s0 * (logterm - arb.const_euler() - arb.pi().log())
        # --- prime part
        nmax = int(math.floor(math.exp(2 * float(a)) + 1e-9))
        Pr = arb_mat(J, J)
        n_pp = 0
        if nmax >= 2:
            ns, lam = vonmangoldt_table(nmax)
            for n in ns.tolist():
                n = int(n)
                if n > nmax:
                    continue
                # Lambda(n) = log p for n = p^k
                p = _smallest_prime_factor(n)
                logn = arb(n).log()
                t = logn / aa
                if t > 2:                    # g vanishes beyond 2a
                    continue
                n_pp += 1
                Pr -= _correlation(t, aa, J, m) * (2 * arb(p).log() / arb(n).sqrt())
        Q = P + A + Pr
        return WeilFormMatrices(a=float(a), J=J, dps=dps, n_quad=n_quad, Q=Q, G=G, P=P, A=A, Pr=Pr,
                                quad_error_estimate=quad_err, n_prime_powers=n_pp,
                                info={"nmax": nmax})
    finally:
        ctx.dps = old


def _smallest_prime_factor(n: int) -> int:
    if n % 2 == 0:
        return 2
    f = 3
    while f * f <= n:
        if n % f == 0:
            return f
        f += 2
    return n


# ----------------------------------------------------------------------------
# Eigenproblem
# ----------------------------------------------------------------------------
def smallest_eigenpairs(Q, G, dps: int, k: int = 3):
    """Smallest k generalised eigenpairs of (Q, G) with G diagonal positive, at
    `dps` digits (mpmath).  Returns (eigenvalues, coefficient vectors) with the
    vectors normalised so that a^T G a = 1, i.e. ||f||_2 = 1; eigenvalues are
    therefore values of Q(f)/||f||^2."""
    import mpmath as mp
    n = Q.nrows()
    with mp.workdps(dps):
        d = [mp.mpf(G[i, i].mid().str(dps + 10, radius=False)) for i in range(n)]
        M = mp.matrix(n, n)
        for i in range(n):
            for j in range(n):
                M[i, j] = mp.mpf(Q[i, j].mid().str(dps + 10, radius=False)) / mp.sqrt(d[i] * d[j])
        # symmetrise exactly
        for i in range(n):
            for j in range(i + 1, n):
                v = (M[i, j] + M[j, i]) / 2
                M[i, j] = v
                M[j, i] = v
        E, V = mp.eigsy(M)
        order = sorted(range(n), key=lambda i: E[i])
        vals, vecs = [], []
        for idx in order[:k]:
            vals.append(E[idx])
            vec = [V[i, idx] / mp.sqrt(d[i]) for i in range(n)]
            # sign convention: first nonzero coefficient positive
            s = 1 if vec[0] >= 0 else -1
            vecs.append([s * x for x in vec])
        return vals, vecs


# ----------------------------------------------------------------------------
# Fourier transform of f = sum a_j P_j(u/a) and its zeros
# ----------------------------------------------------------------------------
def _spherical_jn_all(nmax: int, z, dps: int):
    """[j_0(z), ..., j_nmax(z)] by upward recurrence j_{n+1} = (2n+1)/z j_n - j_{n-1},
    run with enough guard digits to absorb its instability for n > z (the error
    grows at most like the ratio j_0/j_n, i.e. ~ (2n/(e z))^n)."""
    import mpmath as mp
    guard = int(nmax * max(0.0, math.log10(max(2.0 * nmax / (math.e * max(float(z), 1e-30)), 1.0)))) + 20
    with mp.workdps(dps + guard):
        z = mp.mpf(z)
        if z == 0:
            return [mp.mpf(1)] + [mp.mpf(0)] * nmax
        s, c = mp.sin(z), mp.cos(z)
        out = [s / z]
        if nmax >= 1:
            out.append(s / z ** 2 - c / z)
        for n in range(1, nmax):
            out.append((2 * n + 1) / z * out[n] - out[n - 1])
        return [+x for x in out]


def fourier_transform_mp(coeffs, idx, a, r, dps: int = 50):
    """f^(r) = sum_j a_j * 2a i^j j_j(a r) (spherical Bessel), at mpmath precision.
    For an even sector the result is real; for an odd sector it is i times a real
    number and the real number is returned."""
    import mpmath as mp
    with mp.workdps(dps):
        a = mp.mpf(a)
        r = mp.mpf(r)
        jn = _spherical_jn_all(max(idx), a * r, dps)
        tot = mp.mpf(0)
        for c, j in zip(coeffs, idx):
            tot += c * 2 * a * ((-1) ** (j // 2)) * jn[j]   # i^j = (-1)^{j/2} or i(-1)^{(j-1)/2}
        return +tot


def fourier_transform_float(coeffs, idx, a, r):
    """Vectorised float64 version of `fourier_transform_mp` (for scanning)."""
    from scipy.special import spherical_jn
    r = np.asarray(r, dtype=float)
    out = np.zeros_like(r)
    for c, j in zip(coeffs, idx):
        out += float(c) * 2 * a * ((-1) ** (j // 2)) * spherical_jn(j, a * r)
    return out


def real_zeros_of_fhat(coeffs, idx, a, r_max: float, dps: int = 50, grid: int = 20000, refine: bool = True,
                       targets=None, window: float = 0.5):
    """Sign changes of f^ on (0, r_max], refined at `dps` digits (Anderson/secant on
    the bracketing interval).  With `targets` (e.g. zeta zeros) only the sign
    changes within `window` of a target are refined (the nearest one per target)."""
    import mpmath as mp
    rs = np.linspace(1e-6, r_max, grid)
    vals = fourier_transform_float(coeffs, idx, a, rs)
    brackets = [(rs[k], rs[k + 1]) for k in np.nonzero(np.sign(vals[:-1]) * np.sign(vals[1:]) < 0)[0]]
    if targets is not None:
        chosen = []
        for t in targets:
            t = float(t)
            best = min(brackets, key=lambda b: abs(0.5 * (b[0] + b[1]) - t), default=None)
            if best is not None and abs(0.5 * (best[0] + best[1]) - t) <= window and best not in chosen:
                chosen.append(best)
        brackets = sorted(chosen)
    roots = []
    for lo, hi in brackets:
        if refine:
            with mp.workdps(dps):
                f = lambda x: fourier_transform_mp(coeffs, idx, a, x, dps=dps)
                try:
                    root = mp.findroot(f, (mp.mpf(lo), mp.mpf(hi)), solver="anderson")
                except Exception:
                    root = mp.mpf((lo + hi) / 2)
            roots.append(root)
        else:
            roots.append((lo + hi) / 2)
    return roots


def zero_match(coeffs, idx, a, gammas, dps: int = 60, window: float = 0.5, grid: int | None = None):
    """For each target gamma_k return |nearest real zero of f^ - gamma_k| (mpmath),
    or None if f^ has no sign change within `window` of gamma_k."""
    import mpmath as mp
    r_max = float(max(gammas)) + 2 * window
    if grid is None:
        grid = int(max(20000, 200 * a * r_max))
    roots = real_zeros_of_fhat(coeffs, idx, a, r_max, dps=dps, grid=grid, targets=gammas, window=window)
    out = []
    for g in gammas:
        d = min((abs(r - g) for r in roots), default=None)
        out.append(d if (d is not None and d <= window) else None)
    return out


def zero_side_value(coeffs, idx, a, zeros, tail: bool = True, chunk: int = 200000):
    """Zero side 2 sum_n |f^(gamma_n)|^2 (float64) plus the density tail beyond the
    last zero: an independent check of the matrix, valid when the value is far
    above float64 rounding.

    Tail: for f a polynomial on [-a, a], integration by parts gives
    f^(r) = (f(a) e^{iar} - f(-a) e^{-iar})/(ir) + O(r^-2), whose mean square is
    (f(a)^2 + f(-a)^2)/r^2; against the zero density log(r/2pi)/(2pi) this gives
    2 int_T^inf ... dr = (f(a)^2 + f(-a)^2)(log(T/2pi) + 1)/(pi T)  (+ O(T^-2))."""
    zeros = np.asarray(zeros, dtype=float)
    s = 0.0
    for k in range(0, zeros.size, chunk):
        fh = fourier_transform_float(coeffs, idx, a, zeros[k:k + chunk])
        s += 2.0 * float(np.sum(fh ** 2))
    if tail:
        T = float(zeros[-1])
        fa = sum(float(c) for c, j in zip(coeffs, idx))                      # f(a)  = sum a_j P_j(1)
        fma = sum(float(c) * (-1) ** j for c, j in zip(coeffs, idx))        # f(-a) = sum a_j P_j(-1)
        s += (fa ** 2 + fma ** 2) * (math.log(T / (2 * math.pi)) + 1) / (math.pi * T)
    return s


def weyl_certificate(W: WeilFormMatrices, parity: str, lam_min, dps: int):
    """Numerical (not rigorous) positivity certificate for the restricted form.

    Weyl's inequality: if the computed sector matrix differs from the exact one by
    E, every eigenvalue moves by at most ||E||_2 <= ||E||_F.  We take ||E||_F to be
    the Frobenius norm of the Arb ball radii (rounding) plus the archimedean
    quadrature-error estimate (difference between two Gauss–Legendre node counts),
    scaled by the basis normalisation.  Returns (bound, positive) where positive
    means lam_min > bound.  The quadrature error is *estimated*, so this is a
    numerical certificate only."""
    import mpmath as mp
    Qs, Gs, idx = W.sector(parity)
    n = len(idx)
    with mp.workdps(dps):
        rad2 = mp.mpf(0)
        dmin = min(mp.mpf(Gs[i, i].mid().str(30, radius=False)) for i in range(n))
        for i in range(n):
            for j in range(n):
                rad2 += mp.mpf(Qs[i, j].rad().str(10, radius=False)) ** 2
        qerr = mp.mpf(W.quad_error_estimate) if W.quad_error_estimate == W.quad_error_estimate else mp.mpf(0)
        bound = (mp.sqrt(rad2) + n * qerr) / dmin
        return bound, bool(lam_min > bound)
