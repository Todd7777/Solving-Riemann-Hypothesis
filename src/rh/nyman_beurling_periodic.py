r"""Nyman–Beurling–Báez-Duarte distances for Dirichlet series with periodic coefficients.

Purpose: a *controlled* test of how sensitive the Nyman–Beurling criterion is.
We compare two Dirichlet series with period-5 coefficients and the same conductor:

* the L-function of the real character (n/5): coefficients (1, -1, -1, 1, 0) —
  its zeros are on the critical line as far as anyone has computed (GRH);
* the Davenport–Heilbronn function f: coefficients (1, kappa, -kappa, -1, 0) —
  it satisfies a functional equation of the same type but has zeros OFF the
  line (the first at 0.8085 + 85.699i; see `rh.davenport_heilbronn`).

Setting.  Let F(s) = sum_{n>=1} c_n n^{-s} with c periodic of period q and
sum_{n=1}^q c_n = 0 (so F is entire).  Put S(n) = c_1 + ... + c_n (periodic,
bounded) and A(y) = S(floor y).  The functions

    psi_k(x) = A(1/(k x)),      k = 1, 2, ...,

are bounded, supported in (0, 1/k] subset of (0, 1], and have Mellin transforms
int_0^1 psi_k(x) x^{s-1} dx = k^{-s} F(s)/s  (Re s > 0).  As for zeta,

    d_N^2 := min_c || 1_(0,1] - sum_{k<=N} c_k psi_k ||^2_{L^2(0,1)}
           = min_{A_N} (1/2pi) int |1 - F(1/2+it) A_N(1/2+it)|^2 dt / (1/4 + t^2),

with A_N a Dirichlet polynomial of length N.

Easy direction (rigorous, any F of this type).  The Mellin transform maps
L^2(0,1) isometrically onto the Hardy space H^2(Re s > 1/2) with reproducing
kernel K(s, w) = 1/(s + conj(w) - 1).  If F(rho_i) = 0 with Re rho_i > 1/2 then
every candidate takes the value 1/rho_i at rho_i, hence for ALL N

    d_N^2 >= v^* K^{-1} v,   v_i = 1/rho_i,  K_ij = 1/(rho_i + conj(rho_j) - 1)     (floor)

(`offline_floor`).  For a single zero this is (2 Re rho - 1)/|rho|^2.

On-line zeros.  Burnol (2002) proved for zeta that liminf d_N^2 log N >=
sum_{rho on the line} m_rho^2 / |rho|^2.  The heuristic picture for any F of this
type is d_N^2 ~ C_on / log N + floor, where C_on = sum over on-line zeros of
1/|rho|^2.  For F with completed function Lambda(s) = (q/pi)^{s/2}
Gamma((s+e)/2) F(s) = Lambda(1-s) (e = 0 for even, 1 for odd coefficient
sequences), the Hadamard product gives the sum over ALL zeros

    sum_rho 1/(rho (1 - rho)) = 2 Lambda'/Lambda(1) = log(q/pi) + psi((1+e)/2) + 2 F'(1)/F(1),

which equals C_on when every zero is on the line (for zeta the same formula
gives Burnol's constant 2 + gamma - log 4pi).

Gram matrix in closed form.  G_jk = <psi_j, psi_k> = int_0^inf A(u/j) A(u/k) u^{-2} du
reduces to coprime h = j/g, k' = k/g via G_jk = G_{hk'}/g.  The integrand is a
step function; integrating by parts, the integral is sum over its jump points b
of (jump at b)/b.  The jumps sit at u = h m and u = k m' (and at the common
multiples u = h k t), and each of the three resulting series is of the form
sum_m a_m / m with a_m periodic; with a common cutoff u <= U each behaves like
mean(a) log(U/scale) + const, the logarithms cancel, and the constants are
digamma sums:  sum_{m<=M} a_m/m = mean(a) log(M/P) - (1/P) sum_{r=1}^P a_r psi(r/P) + o(1)
for a_m of period P.  Cost O(q (h + k)) per entry.  The formula is checked
against an independent exact evaluation (`gram_entry_direct_mp`, O(q h k)).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from math import gcd

import numpy as np
from scipy.special import digamma

__all__ = ["PeriodicSeries", "CHI5", "DAVENPORT_HEILBRONN", "offline_floor", "gram_entry_direct_mp"]


@lru_cache(maxsize=None)
def _psi_table(P: int) -> np.ndarray:
    """digamma(r/P) for r = 1..P."""
    return digamma(np.arange(1, P + 1, dtype=np.float64) / P)


def _periodic_constant(a: np.ndarray):
    """For a periodic sequence a_1..a_P return (mean, C) with
    sum_{m<=M} a_m/m = mean*log(M) + C + o(1),  C = -mean*log P - (1/P) sum a_r psi(r/P)."""
    P = a.size
    mean = float(a.sum()) / P
    C = -mean * math.log(P) - float(np.dot(a, _psi_table(P))) / P
    return mean, C


@dataclass(frozen=True)
class PeriodicSeries:
    name: str
    coeffs: tuple          # c_1, ..., c_q

    @property
    def q(self) -> int:
        return len(self.coeffs)

    @property
    def c(self) -> np.ndarray:
        return np.asarray(self.coeffs, dtype=np.float64)

    @property
    def S(self) -> np.ndarray:
        """S(0..q-1) as a periodic table: S(n) = S_table[n % q]."""
        s = np.concatenate([[0.0], np.cumsum(self.c)[:-1]])
        return s

    @property
    def parity(self) -> int:
        """0 if c_{q-n} = c_n (even), 1 if c_{q-n} = -c_n (odd)."""
        c = self.c
        q = self.q
        if all(abs(c[(q - n) % q - 1 if (q - n) % q else q - 1] - c[n - 1]) < 1e-15 for n in range(1, q)):
            return 0
        return 1

    # ------------------------------------------------------------------ values at s = 1
    def F1(self):
        """F(1) = sum c_n/n = -(1/q) sum_r c_r psi(r/q)  (mpmath)."""
        import mpmath as mp
        q = self.q
        return -sum(mp.mpf(self.coeffs[r - 1]) * mp.digamma(mp.mpf(r) / q) for r in range(1, q + 1)) / q

    def F1_prime(self):
        """F'(1) = -log(q) F(1) - (1/q) sum_r c_r gamma_1(r/q)  (generalised Stieltjes)."""
        import mpmath as mp
        q = self.q
        g1 = sum(mp.mpf(self.coeffs[r - 1]) * mp.stieltjes(1, mp.mpf(r) / q) for r in range(1, q + 1))
        return -mp.log(q) * self.F1() - g1 / q

    def zero_sum_constant(self):
        """sum over all nontrivial zeros of 1/(rho(1-rho)) = log(q/pi) + psi((1+e)/2) + 2F'(1)/F(1)."""
        import mpmath as mp
        e = self.parity
        return mp.log(mp.mpf(self.q) / mp.pi) + mp.digamma(mp.mpf(1 + e) / 2) + 2 * self.F1_prime() / self.F1()

    # ------------------------------------------------------------------ Gram matrix
    def _S_of(self, n: np.ndarray) -> np.ndarray:
        return self.S[np.mod(n, self.q)] if False else self._S_table()[np.mod(n, self.q)]

    def _S_table(self):
        # S(n) for n mod q, with S(0) = 0, S(n) = c_1 + ... + c_n
        c = self.c
        t = np.zeros(self.q)
        t[1:] = np.cumsum(c)[:-1]
        return t

    def gram_coprime(self, h: int, k: int) -> float:
        """<psi_h, psi_k> for coprime h, k by the jump/digamma formula."""
        q = self.q
        c = self.c
        St = self._S_table()
        tot_mean = 0.0
        val = 0.0
        # (1) jumps of A(u/h) at u = h m, k does not divide m:  a_m = c_m S(floor(h m / k)) / h
        P1 = q * k
        m = np.arange(1, P1 + 1)
        a1 = c[(m - 1) % q] * St[(h * m // k) % q] / h
        a1[m % k == 0] = 0.0
        mean1, C1 = _periodic_constant(a1)
        # (2) jumps of A(u/k) at u = k m, h does not divide m
        P2 = q * h
        m = np.arange(1, P2 + 1)
        a2 = c[(m - 1) % q] * St[(k * m // h) % q] / k
        a2[m % h == 0] = 0.0
        mean2, C2 = _periodic_constant(a2)
        # (3) common jumps at u = h k t
        t = np.arange(1, q + 1)
        a3 = (St[(k * t) % q] * St[(h * t) % q] - St[(k * t - 1) % q] * St[(h * t - 1) % q]) / (h * k)
        mean3, C3 = _periodic_constant(a3)
        tot_mean = mean1 + mean2 + mean3
        if abs(tot_mean) > 1e-9:
            raise ArithmeticError(f"log U terms do not cancel for ({h},{k}): {tot_mean}")
        return -mean1 * math.log(h) + C1 - mean2 * math.log(k) + C2 - mean3 * math.log(h * k) + C3

    def gram_entry(self, j: int, k: int) -> float:
        g = gcd(j, k)
        return self.gram_coprime(j // g, k // g) / g

    def gram_matrix(self, N: int, progress=None) -> np.ndarray:
        G = np.empty((N, N))
        cache = {}
        for j in range(1, N + 1):
            for k in range(j, N + 1):
                g = gcd(j, k)
                key = (j // g, k // g)
                if key not in cache:
                    cache[key] = self.gram_coprime(*key)
                G[j - 1, k - 1] = G[k - 1, j - 1] = cache[key] / g
            if progress and j % 100 == 0:
                progress(j)
        return G

    def rhs(self, N: int) -> np.ndarray:
        """b_k = <1_(0,1], psi_k> = F(1)/k."""
        F1 = float(self.F1())
        return F1 / np.arange(1, N + 1, dtype=np.float64)

    def distances(self, N: int, G: np.ndarray | None = None, return_coeffs: bool = False):
        """d_n^2 for n = 1..N via one Cholesky factorisation (||1_(0,1]||^2 = 1)."""
        import scipy.linalg as sla
        if G is None:
            G = self.gram_matrix(N)
        b = self.rhs(N)
        L = sla.cholesky(G, lower=True)
        y = sla.solve_triangular(L, b, lower=True)
        d2 = 1.0 - np.cumsum(y * y)
        if return_coeffs:
            return d2, sla.solve_triangular(L.T, y, lower=False)
        return d2


def _dh_kappa() -> float:
    return (math.sqrt(10 - 2 * math.sqrt(5)) - 2) / (math.sqrt(5) - 1)


CHI5 = PeriodicSeries("L(s, (n/5))", (1.0, -1.0, -1.0, 1.0, 0.0))
DAVENPORT_HEILBRONN = PeriodicSeries("Davenport-Heilbronn", (1.0, _dh_kappa(), -_dh_kappa(), -1.0, 0.0))


def offline_floor(rhos) -> float:
    """Exact lower bound min ||phi||^2 over phi in L^2(0,1) whose Mellin transform
    takes the value 1/rho at each rho (zeros of F with Re rho > 1/2; pass both
    members of each conjugate pair).  Valid for every N (see module docstring)."""
    import mpmath as mp
    rh = [mp.mpc(r) for r in rhos]
    n = len(rh)
    K = mp.matrix(n, n)
    v = mp.matrix(n, 1)
    for i in range(n):
        v[i] = 1 / rh[i]
        for j in range(n):
            K[i, j] = 1 / (rh[i] + mp.conj(rh[j]) - 1)
    x = mp.lu_solve(K, v)
    val = sum(mp.conj(v[i]) * x[i] for i in range(n))
    return float(mp.re(val))


def gram_entry_direct_mp(series: PeriodicSeries, h: int, k: int, dps: int = 30):
    """Independent exact evaluation for coprime h, k: the integrand
    A(u/h)A(u/k)/u^2 is periodic-step with period L = q h k; summing the
    contributions of all periods in closed form,
        int_0^inf G(u) u^-2 du = (1/L) sum_pieces G_p [psi(x_{p+1}/L) - psi(x_p/L)]."""
    import mpmath as mp
    with mp.workdps(dps):
        q = series.q
        St = [mp.mpf(0)]
        for ci in series.coeffs[:-1]:
            St.append(St[-1] + mp.mpf(ci))
        L = q * h * k
        pts = sorted(set([0, L] + [h * m for m in range(0, q * k + 1)] + [k * m for m in range(0, q * h + 1)]))
        tot = mp.mpf(0)
        for x0, x1 in zip(pts[:-1], pts[1:]):
            Gp = St[(x0 // h) % q] * St[(x0 // k) % q]
            if Gp == 0 or x0 == 0:
                continue
            tot += Gp * (mp.digamma(mp.mpf(x1) / L) - mp.digamma(mp.mpf(x0) / L))
        return tot / L
