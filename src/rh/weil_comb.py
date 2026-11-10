r"""Non-pointwise bounds for the prime comb of Weil's form (input to `rh.weil_envelope`).

The comb
--------
In Weil's form on real test functions supported in [-L, L] the primes enter through the comb

    P_L(t) = sum_{log n < 2L} c_n cos(t log n),        c_n = 2 Lambda(n) / sqrt(n).

Pointwise, sup_t P_L(t) = A_L = sum c_n (the log p are linearly independent, so by Kronecker every
phase vector (t log p mod 2 pi)_p is approached), and the envelope reduction of `rh.weil_envelope`
needs T > 2 pi e^{A_L}, which is doubly exponential in L.

What the reduction really needs
-------------------------------
It needs a constant A' and a function H all of whose frequencies have modulus at least 2L, with

    A' - P_L(t) + H(t) >= 0        for all real t.                                        (*)

For real f supported in [-L, L], |F|^2 is even and is the Fourier transform of the autocorrelation
of f, which vanishes outside (-2L, 2L). Hence int_0^inf H |F|^2 = 0 and

    Q(f) = Pole(f) + (1/pi) int_0^inf (Psi_L + H) |F|^2,      Psi_L + H >= Omega - A' on [0, inf).

Since Omega is increasing, with beta'' = Omega(T) - A' this gives

    Q(f) >= beta'' ||f||^2 + Pole(f) + (1/pi) int_0^T (Psi_L(t) + H(t) - beta'') |F(t)|^2 dt.

This has the same structure as the pointwise reduction. The difference: A_L is replaced by A', and
the symbol on [0, T] is Psi_L + H. Because the phases t log p fill the torus, (*) is an inequality
for trigonometric polynomials on the torus of prime phases.

Separable sum-of-squares certificates (this module)
---------------------------------------------------
Write P_L(t) = sum_p P_p(t log p), where P_p(theta) = sum_{k <= K_p} c_{p^k} cos(k theta) and
K_p = max{k : p^k < e^{2L}}. For each prime choose a degree M_p > K_p and a real matrix R_p, whose
rows are vectors in R^{M_p + 1}. Then

    S_p(theta) = sum_i |sum_j (R_p)_ij e^{i j theta}|^2 = s_0 + 2 sum_{k=1}^{M_p} s_k cos(k theta) >= 0,
    s_k = sum_i sum_j (R_p)_ij (R_p)_{i, j+k}    (exact; computed in ball arithmetic).

Put A_p = s_0 + sum_{k <= K_p} |2 s_k + c_{p^k}| and H_p(theta) = sum_{K_p < k <= M_p} 2 s_k cos(k theta).
Then

    A_p - P_p + H_p = S_p + sum_{k <= K_p} (|2 s_k + c_{p^k}| - (2 s_k + c_{p^k}) cos k theta) >= 0,

and H_p(t log p) has frequencies k log p >= (K_p + 1) log p > 2L. Summing over p gives (*), with
A' = sum_p A_p and the entire function H(t) = sum_p H_p(t log p). H is a finite sum of cosines, so
the same Gauss–Legendre rules with a-priori error bounds that handle the rest of the symbol also
integrate it.

How R_p is chosen. For degree M, the best A_p solves a small semidefinite program. Its dual is:
maximise sum_k c_k m_k over m in R^K, subject to Toep(1, m_1..m_K, 0..0) >= 0, an (M+1) x (M+1)
Toeplitz matrix. We follow the central path of this dual with a log-barrier Newton method;
X = mu Toep(m)^{-1} is the primal Gram matrix (s_k = -c_k/2 at the Newton fixed point), and
R_p = diag(sqrt(lambda)) V^T comes from its eigendecomposition. This floating-point step affects
only how close A_p is to optimal, never whether the certificate is valid.

For K = 1 the optimum is classical (Fejér–Egerváry–Szász): A_p = c / (2 cos(pi / (M + 2))). As
M -> inf, A_p decreases to lambda_max of the (K+1) x (K+1) Toeplitz matrix [c_{p^|j-l|}/2]_{j != l}
(Carathéodory–Toeplitz), the best separable constant. A higher degree means higher frequencies in H
(up to M_p log p), which the quadrature must resolve. We take M_p = max(K_p + 1, floor(X / log p))
for a frequency budget X.

Lower limit. Any certificate (*) implies that the comb operator f -> sum_n c_n (f(. - log n)
+ f(. + log n)) / 2 on L^2[-L, L] has norm at most A'. Its norm A* (`comb_operator_norm`) is
therefore a lower bound for every A', separable or not.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List

__all__ = ["prime_comb", "toeplitz_lambda_max", "fes_constant", "sos_gram", "PrimeCertificate",
           "CombCertificate", "prime_certificate", "comb_certificate", "comb_operator_norm",
           "comb_constants"]


def _primes_upto(x: float) -> List[int]:
    n = int(x)
    sieve = bytearray([1]) * (n + 1)
    sieve[0:2] = b"\x00\x00"
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            sieve[i * i::i] = bytearray(len(sieve[i * i::i]))
    return [i for i in range(2, n + 1) if sieve[i]]


def prime_comb(L: float) -> dict:
    """{p: [(n = p^k, c_n as arb) for k = 1..K_p]} for the prime powers n with log n < 2L.  Every
    comparison with 2L is decided rigorously in Arb, including log p^{K_p + 1} > 2L for the first
    excluded power."""
    from flint import arb
    LL = arb(L)
    out = {}
    for p in _primes_upto(math.exp(2 * L) + 1):
        k, n = 1, p
        while True:
            ln = arb(n).log()
            if ln > 2 * LL:
                break
            if not (ln < 2 * LL):
                raise ArithmeticError(f"cannot decide log({n}) < 2L")
            out.setdefault(p, []).append((n, 2 * arb(p).log() / arb(n).sqrt()))
            k += 1
            n *= p
    return out


def toeplitz_lambda_max(cs) -> float:
    """lambda_max of the (K+1) x (K+1) Toeplitz matrix with entries c_{|j-l|}/2 off the diagonal and 0
    on it (cs = [c_1, ..., c_K], floats).  This is the best separable per-prime constant (degree M -> inf)."""
    import numpy as np
    K = len(cs) + 1
    T = np.zeros((K, K))
    for j in range(K):
        for l in range(K):
            if j != l:
                T[j, l] = cs[abs(j - l) - 1] / 2
    return float(np.linalg.eigvalsh(T)[-1])


def fes_constant(c: float, M: int) -> float:
    """Optimal A for one cosine (K = 1) with H of degree M: c / (2 cos(pi/(M+2))) (Fejér–Egerváry–Szász)."""
    return c / (2 * math.cos(math.pi / (M + 2)))


def sos_gram(cs, M: int, mu_min: float = 1e-10):
    """Near-optimal Gram matrix (floats) for the degree-M certificate of the comb sum_k c_k cos(k theta).

    Uses a log-barrier Newton method on the dual: maximise c.m + mu log det Toep(1, m, 0..0), with
    mu -> mu_min. Returns (R, info): R holds the rows sqrt(lambda_i) v_i of the eigendecomposition of
    X = mu Toep(m)^{-1}, with tiny eigenvalues dropped. info['dual'] is the dual value c.m, a lower bound
    for every degree-M constant (up to the float accuracy), and info['gap'] is the barrier gap (M+1) mu."""
    import numpy as np
    K = len(cs)
    n = M + 1
    if M <= K:
        raise ValueError("degree M must exceed K")
    E = []
    for k in range(1, K + 1):
        Ek = np.zeros((n, n))
        i = np.arange(n - k)
        Ek[i, i + k] = 1.0
        Ek[i + k, i] = 1.0
        E.append(Ek)
    c = np.asarray(cs, dtype=float)

    def Tm(m):
        return np.eye(n) + sum(m[k] * E[k] for k in range(K))

    def logdet(T):
        try:
            Lc = np.linalg.cholesky(T)
        except np.linalg.LinAlgError:
            return None
        return 2.0 * float(np.sum(np.log(np.diag(Lc))))

    m = np.zeros(K)
    mu = 1.0
    while True:
        for _ in range(200):
            T = Tm(m)
            Ti = np.linalg.inv(T)
            g = c + mu * np.array([np.sum(Ti * Ek) for Ek in E])
            TE = [Ti @ Ek for Ek in E]
            Hs = -mu * np.array([[np.sum(TE[k] * TE[l].T) for l in range(K)] for k in range(K)])
            dm = -np.linalg.solve(Hs, g)
            dec = float(g @ dm)                     # Newton decrement^2 (>= 0 for this concave problem)
            if dec < 1e-18:
                break
            f0 = float(c @ m) + mu * logdet(T)
            a = 1.0
            while a > 1e-14:
                mn = m + a * dm
                ld = logdet(Tm(mn))
                if ld is not None and float(c @ mn) + mu * ld >= f0 + 0.25 * a * dec:
                    break
                a *= 0.5
            m = mn
        if mu <= mu_min:
            break
        mu = max(mu / 10.0, mu_min)
    X = mu * np.linalg.inv(Tm(m))
    X = (X + X.T) / 2
    lam, V = np.linalg.eigh(X)
    keep = lam > 1e-13 * lam[-1]
    R = (np.sqrt(lam[keep])[:, None] * V[:, keep].T)
    return R, {"dual": float(c @ m), "gap": n * mu, "m": m.tolist(), "rank": int(keep.sum())}


def _autocorrelations(R, M: int):
    """s_k = sum_i sum_j R_ij R_{i,j+k}, k = 0..M, computed exactly: the entries are floats, so every
    product is exact with 106 bits, and the sums are exact at the 4000-bit precision used here (checked:
    every result has radius zero).  Exactness makes H independent of the working precision."""
    from flint import arb, ctx
    old = ctx.prec
    ctx.prec = 4000
    try:
        s = []
        for k in range(M + 1):
            acc = arb(0)
            for row in R:
                for j in range(len(row) - k):
                    acc += arb(float(row[j])) * arb(float(row[j + k]))
            if acc.rad() != 0:
                raise ArithmeticError("autocorrelation not exact")
            s.append(acc)
    finally:
        ctx.prec = old
    return s


@dataclass
class PrimeCertificate:
    """A_p - P_p(theta) + H_p(theta) >= S_p(theta) >= 0 for the comb of one prime p (see the module docs)."""
    p: int
    K: int
    M: int
    c: list                 # arb c_{p^k}, k = 1..K
    R: list                 # rows of floats (exact): S_p = sum_i |sum_j R_ij e^{ij theta}|^2
    s: list                 # arb autocorrelations s_0..s_M of S_p
    A_p: object             # arb (an exact upper bound, stored as a point)
    lambda_toeplitz: float  # best separable constant (M -> inf), float
    dual: float             # float lower bound for any degree-M constant (from the barrier dual)

    def H_terms(self):
        """[(k, coefficient 2 s_k)] for K < k <= M: H_p(theta) = sum 2 s_k cos(k theta).  The
        coefficients are exact (doubling an exact dyadic number at a precision that holds it)."""
        from flint import ctx
        old = ctx.prec
        ctx.prec = 4000
        try:
            out = [(k, 2 * self.s[k]) for k in range(self.K + 1, self.M + 1)]
        finally:
            ctx.prec = old
        assert all(h.rad() == 0 for _, h in out)
        return out

    def S(self, theta):
        """S_p(theta) = s_0 + 2 sum s_k cos(k theta) (acb)."""
        from flint import acb
        theta = acb(theta)
        v = acb(self.s[0])
        for k in range(1, self.M + 1):
            v += 2 * self.s[k] * (k * theta).cos()
        return v


def prime_certificate(p: int, terms, M: int) -> PrimeCertificate:
    """Degree-M SOS certificate for the comb of the prime p (terms = [(p^k, c_{p^k} arb)])."""
    from flint import arb
    cs = [float(c.mid()) for _, c in terms]
    K = len(cs)
    R, info = sos_gram(cs, M)
    s = _autocorrelations(R, M)
    c_arb = [c for _, c in terms]
    A = s[0]
    for k in range(1, K + 1):
        A += abs(2 * s[k] + c_arb[k - 1])
    return PrimeCertificate(p=p, K=K, M=M, c=c_arb, R=[list(map(float, r)) for r in R], s=s,
                            A_p=arb(float(A.upper()) * (1 + 1e-15)), lambda_toeplitz=toeplitz_lambda_max(cs),
                            dual=info["dual"])


@dataclass
class CombCertificate:
    """A' - P_L(t) + H(t) >= 0 for all real t; H(t) = sum h_j cos(omega_j t) with every omega_j > 2L."""
    L: float
    X: float                                     # frequency budget: every omega_j <= X
    primes: List[PrimeCertificate] = field(default_factory=list)
    A_L: object = None      # arb: the pointwise comb constant
    A_prime: object = None  # arb: the certified non-pointwise constant
    terms: list = field(default_factory=list)    # [(p, k, h)]: h cos(k log(p) t), h an exact dyadic arb
    H_sup: float = 0.0      # upper bound for sup_{t real} |H(t)| (sum |h_j|)
    omega_min: float = 0.0  # lower bound for the smallest frequency of H (> 2L)

    def freqs(self):
        """[(omega = k log p, h)] with omega computed at the current working precision (cached per precision)."""
        from flint import arb, ctx
        cache = self.__dict__.setdefault("_freq_cache", {})
        if ctx.prec not in cache:
            cache[ctx.prec] = [(k * arb(p).log(), h) for p, k, h in self.terms]
        return cache[ctx.prec]

    def H(self, z):
        """H(z) = sum h cos(omega z) (acb) at real or complex z, at the current working precision."""
        from flint import acb
        z = acb(z)
        v = acb(0)
        for om, h in self.freqs():
            v += h * (om * z).cos()
        return v

    def describe(self) -> dict:
        return {"L": self.L, "X": self.X, "A_L": float(self.A_L.mid()), "A_prime": float(self.A_prime.upper()),
                "T1_pointwise": 2 * math.pi * math.exp(float(self.A_L.mid())),
                "T1_certificate": 2 * math.pi * math.exp(float(self.A_prime.upper())),
                "separable_optimum": sum(pc.lambda_toeplitz for pc in self.primes),
                "H_sup": self.H_sup, "n_terms": len(self.terms), "omega_min": self.omega_min,
                "per_prime": [{"p": pc.p, "K": pc.K, "M": pc.M, "A_p": float(pc.A_p.upper()),
                               "lambda_toeplitz": pc.lambda_toeplitz, "dual_degree_M": pc.dual,
                               "sum_c": float(sum(c.mid() for c in pc.c))} for pc in self.primes]}


def comb_certificate(L: float, X: float = 24.0, degrees: dict | None = None) -> CombCertificate:
    """Separable SOS certificate (*) for the comb of the window [-L, L], with every frequency of H at most
    X (degree M_p = max(K_p + 1, floor(X / log p)), unless given in `degrees` = {p: M_p})."""
    from flint import arb
    comb = prime_comb(L)
    LL = arb(L)
    cert = CombCertificate(L=L, X=X)
    A_L, A_prime = arb(0), arb(0)
    omin = float("inf")
    for p, terms in sorted(comb.items()):
        for _, c in terms:
            A_L += c
        K = len(terms)
        M = (degrees or {}).get(p, max(K + 1, int(math.floor(X / math.log(p)))))
        pc = prime_certificate(p, terms, M)
        cert.primes.append(pc)
        A_prime += pc.A_p
        lp = arb(p).log()
        for k, h in pc.H_terms():
            om = k * lp
            if not (om > 2 * LL):          # (K_p + 1) log p > 2L was decided in prime_comb; re-check
                raise ArithmeticError(f"frequency {k} log {p} not > 2L")
            cert.terms.append((p, k, h))
            cert.H_sup += float(abs(h).upper())
            omin = min(omin, float(om.lower()))
    cert.A_L = A_L
    cert.A_prime = arb(float(A_prime.upper()) * (1 + 1e-15))
    cert.H_sup *= (1 + 1e-12)
    cert.omega_min = omin
    return cert


def comb_operator_norm(L: float, J: int = 80) -> float:
    """lambda_max of the comb operator f -> sum c_n (f(. - log n) + f(. + log n))/2 on L^2[-L, L], in the
    Legendre basis of degree < J (numerical, from rh.weil_form's prime matrix).  A lower bound for any
    constant A' in (*)."""
    import numpy as np
    from . import weil_form as wf
    W = wf.build_weil_form(L, J, dps=30, estimate_quad_error=False)
    best = -float("inf")
    for par in ("even", "odd"):
        _, _, idx = W.sector(par)
        Pr = np.array([[float(W.Pr[i, j].mid()) for j in idx] for i in idx])
        g = np.array([2 * L / (2 * j + 1) for j in idx])
        best = max(best, float(np.linalg.eigvalsh(-Pr / np.sqrt(np.outer(g, g)))[-1]))
    return best


def comb_constants(L: float, X: float = 24.0, J: int = 80) -> dict:
    """A_L (pointwise), the certified separable A' (frequency budget X), the separable optimum, and the
    operator norm A* (lower bound), with the envelope cutoffs T_1 = 2 pi e^{A}."""
    cert = comb_certificate(L, X)
    d = cert.describe()
    Astar = comb_operator_norm(L, J)
    d.update({"A_star": Astar, "T1_operator": 2 * math.pi * math.exp(Astar)})
    return d
