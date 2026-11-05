r"""Báez-Duarte's sequential Riesz-like criterion and its explicit formula.

Báez-Duarte (2005, Int. J. Math. Math. Sci. 2005:21, 3527–3537) proved:

    c_k := sum_{j=0}^{k} (-1)^j binom(k,j) / zeta(2j+2)     (k = 0, 1, 2, ...)

    RH  <==>  c_k = O(k^{-3/4 + eps}) for every eps > 0.

Using 1/zeta(2j+2) = sum_n mu(n) n^{-2j-2} and the binomial theorem,

    c_k = sum_{n>=1} mu(n) n^{-2} (1 - n^{-2})^k,                       (*)

which is numerically benign (no catastrophic cancellation across j; the only
cancellation is between 6/pi^2 and the first ~sqrt(k) terms).

Explicit formula.  With the Mellin–Barnes representation of (1-x)^k through
the Beta function  B(s, k+1) = Gamma(s)Gamma(k+1)/Gamma(s+k+1),

    c_k = (1/2 pi i) int_{(c)} B(s, k+1) / zeta(2 - 2s) ds,   0 < c < 1/2,

and shifting the contour to the right across the poles at s = 1 - rho/2
(rho the nontrivial zeros of zeta) gives

    c_k = sum_rho  B(1 - rho/2, k+1) / (2 zeta'(rho))
          + sum_{m>=1} B(1 + m, k+1) / (2 zeta'(-2m))  +  (remainder),        (**)

where the second sum comes from the TRIVIAL zeros rho = -2m (poles at s = 1+m);
zeta'(-2m) = (-1)^m (2m)! zeta(2m+1) / (2 (2pi)^{2m}).  The m = 1 term equals
-1/(2 zeta'(-2) (k+1)(k+2)) ~ 16.4/k^2: formally smaller than k^{-3/4} but with a
constant that is ~10^5 times larger than the exponentially small
|Gamma(3/4 - i gamma_1/2)| of the leading nontrivial-zero term, so it dominates
c_k for k up to ~10^5 and must be included (as the numerics in this repository
showed: without it the prediction is off by a factor ~10 at k = 2·10^4).  Since B(1-rho/2, k+1) ~ Gamma(1-rho/2) k^{-(1-rho/2)}
and |Gamma(3/4 - i gamma/2)| decays like exp(-pi gamma/4), the sum is dominated
by the FIRST zero: under RH,

    c_k ~ 2 k^{-3/4} Re[ Gamma(3/4 - i gamma_1/2) k^{i gamma_1/2} / (2 zeta'(rho_1)) ]
          (1 + O(1/k)) + (terms from gamma_2, gamma_3, ... smaller by e^{-pi(gamma_j-gamma_1)/4}).

A zero rho with Re rho = beta < 1/2 (whose mirror 1-rho has real part > 1/2)
would contribute a term of size k^{-(1+beta)/2} >> k^{-3/4}: this is exactly how
the criterion encodes RH.  The module computes c_k exactly (mpmath, (*)) and
compares with the explicit formula (**) truncated to a few zeros.
"""
from __future__ import annotations

import math
import numpy as np
import mpmath as mp

__all__ = ["c_k_exact", "c_k_float", "c_k_explicit", "c_k_binomial", "mobius_array"]


def mobius_array(M: int) -> np.ndarray:
    """mu(1..M) by a linear sieve."""
    mu = np.ones(M + 1, dtype=np.int8)
    is_prime = np.ones(M + 1, dtype=bool)
    is_prime[:2] = False
    for p in range(2, M + 1):
        if is_prime[p]:
            is_prime[p * p::p] = False
            mu[p::p] *= -1
            mu[p * p::p * p] = 0
    mu[0] = 0
    return mu


def c_k_float(k: int, mu: np.ndarray | None = None, M: int | None = None) -> float:
    """c_k via (*) in float64, summing n <= M directly and the tail n > M with
    the expansion 1-(1-n^-2)^k = k n^-2 - k(k-1)/2 n^-4 + ...  (valid for n^2 >> k).
    Accuracy ~1e-15 absolute for k <= 1e6 with M >= 200*sqrt(k)."""
    if M is None:
        M = int(max(2000, 200 * math.sqrt(k)))
    if mu is None or len(mu) <= M:
        mu = mobius_array(M)
    n = np.arange(1, M + 1, dtype=np.float64)
    m1 = mu[1:M + 1].astype(np.float64)
    head = math.fsum(m1 * n ** -2.0 * (1.0 - n ** -2.0) ** k)
    # tail: sum_{n>M} mu(n)/n^2 * (1-n^-2)^k = sum_{n>M} mu(n)/n^2 - sum_{n>M} mu(n)/n^2 [1-(1-n^-2)^k]
    # First part: 6/pi^2 - sum_{n<=M} mu(n)/n^2.  Second: expand in powers of n^-2,
    # with the tails sum_{n>M} mu(n) n^{-2m} (m=2,3,4) summed directly to M2 (error ~ M2^{-3}).
    inv_zeta2 = 6.0 / math.pi ** 2
    tail1 = inv_zeta2 - math.fsum(m1 * n ** -2.0)
    M2 = max(10 * M, 100000)
    if len(mu) <= M2:
        mu = mobius_array(M2)
    n2 = np.arange(M + 1, M2 + 1, dtype=np.float64)
    m2 = mu[M + 1:M2 + 1].astype(np.float64)
    tails = {m: math.fsum(m2 * n2 ** (-2.0 * m)) for m in (2, 3, 4)}
    kk = float(k)
    tail2 = kk * tails[2] - kk * (kk - 1) / 2 * tails[3] + kk * (kk - 1) * (kk - 2) / 6 * tails[4]
    return float(head + tail1 - tail2)


def c_k_exact(k: int, dps: int = 30, M: int | None = None):
    """c_k via (*) with mpmath (dps digits); M as in c_k_float (tail expansion to
    4th order, error ~ k^4/M^9)."""
    with mp.workdps(dps):
        if M is None:
            M = int(max(2000, 200 * math.sqrt(k)))
        mu = mobius_array(M)
        head = mp.mpf(0)
        s2 = mp.mpf(0)
        s = {2: mp.mpf(0), 3: mp.mpf(0), 4: mp.mpf(0)}
        for n in range(1, M + 1):
            m_ = int(mu[n])
            if m_ == 0:
                continue
            n2 = mp.mpf(n) ** 2
            head += m_ / n2 * (1 - 1 / n2) ** k
            s2 += m_ / n2
            for j in (2, 3, 4):
                s[j] += m_ / n2 ** j
        tail1 = 1 / mp.zeta(2) - s2
        t = {j: 1 / mp.zeta(2 * j) - s[j] for j in (2, 3, 4)}
        kk = mp.mpf(k)
        tail2 = kk * t[2] - kk * (kk - 1) / 2 * t[3] + kk * (kk - 1) * (kk - 2) / 6 * t[4]
        return head + tail1 - tail2


def c_k_binomial(k: int, dps: int | None = None):
    """c_k from the defining binomial sum (needs ~0.3k extra digits)."""
    if dps is None:
        dps = int(0.32 * k) + 30
    with mp.workdps(dps):
        return mp.fsum((-1) ** j * mp.binomial(k, j) / mp.zeta(2 * j + 2) for j in range(k + 1))


def c_k_explicit(k, zeros_rho, zeta_prime=None, dps: int = 30, n_trivial: int = 12,
                 include_trivial: bool = True, return_parts: bool = False):
    """Explicit-formula prediction (**): sum over the given nontrivial zeros rho
    (upper half-plane; conjugates added automatically) of
    B(1 - rho/2, k+1) / (2 zeta'(rho)), plus the trivial-zero terms
    B(1+m, k+1) / (2 zeta'(-2m)), m = 1..n_trivial.
    zeros_rho: list of mpc; zeta_prime: optional list of zeta'(rho)."""
    with mp.workdps(dps):
        if zeta_prime is None:
            zeta_prime = [mp.zeta(r, derivative=1) for r in zeros_rho]
        nontriv = mp.mpf(0)
        for r, zp in zip(zeros_rho, zeta_prime):
            s = 1 - r / 2
            B = mp.beta(s, k + 1)
            nontriv += 2 * mp.re(B / (2 * zp))  # rho and conjugate
        triv = mp.mpf(0)
        if include_trivial:
            for m in range(1, n_trivial + 1):
                zp_triv = (-1) ** m * mp.factorial(2 * m) * mp.zeta(2 * m + 1) / (2 * (2 * mp.pi) ** (2 * m))
                triv += mp.beta(1 + m, k + 1) / (2 * zp_triv)
        if return_parts:
            return nontriv + triv, nontriv, triv
        return nontriv + triv
