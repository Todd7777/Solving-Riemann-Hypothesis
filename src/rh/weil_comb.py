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
           "CombCertificate", "prime_certificate", "comb_certificate", "joint_gram", "joint_gram_admm",
           "joint_comb_certificate", "comb_operator_norm", "comb_constants"]


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
    terms: list = field(default_factory=list)    # [(n, d, h)]: h cos(t log(n/d)), n > d >= 1, h exact dyadic
    H_sup: float = 0.0      # upper bound for sup_{t real} |H(t)| (sum |h_j|)
    omega_min: float = 0.0  # lower bound for the smallest frequency of H (> 2L)
    meta: dict = field(default_factory=dict)     # construction details (joint certificates)

    def freqs(self):
        """[(omega = log(n/d), h)] with omega computed at the current working precision (cached per precision)."""
        from flint import arb, ctx
        cache = self.__dict__.setdefault("_freq_cache", {})
        if ctx.prec not in cache:
            cache[ctx.prec] = [(arb(n).log() - arb(d).log(), h) for n, d, h in self.terms]
        return cache[ctx.prec]

    def H_bound(self, b: float) -> float:
        """Upper bound for sup |H(z)| over the strip |Im z| <= b: sum |h| cosh(omega b)."""
        tot = 0.0
        for n, d, h in self.terms:
            tot += float(abs(h).upper()) * math.cosh(math.log(n / d) * (1 + 1e-12) * b)
        return tot * (1 + 1e-12) + 1e-300

    def H(self, z):
        """H(z) = sum h cos(z log(n/d)) (acb) at real or complex z, at the current working precision.
        All frequencies are logs of ratios of a few integers, so H is evaluated from the powers
        E_n = e^{i z log n}: cos(z log(n/d)) = (E_n / E_d + E_d / E_n) / 2, one exponential per integer and
        two products per term (instead of one cosine per term)."""
        from flint import acb, arb, ctx
        z = acb(z)
        cache = self.__dict__.setdefault("_log_cache", {})
        if ctx.prec not in cache:
            ints = sorted({n for n, _, _ in self.terms} | {d for _, d, _ in self.terms})
            cache[ctx.prec] = {n: arb(n).log() for n in ints}
        logs = cache[ctx.prec]
        if len(self.terms) < 3 * len(logs):           # few terms per integer: cosines are cheaper
            return self.H_direct(z)
        iz = acb(0, 1) * z
        E = {n: (iz * ln).exp() for n, ln in logs.items()}
        Ei = {n: 1 / e for n, e in E.items()}
        v = acb(0)
        for n, d, h in self.terms:
            v += h * (E[n] * Ei[d] + Ei[n] * E[d])
        return v / 2

    def H_direct(self, z):
        """H(z) term by term with one cosine per term (reference for `H`)."""
        from flint import acb
        z = acb(z)
        v = acb(0)
        for om, h in self.freqs():
            v += h * (om * z).cos()
        return v

    def describe(self) -> dict:
        sep = (sum(pc.lambda_toeplitz for pc in self.primes) if self.primes
               else sum(toeplitz_lambda_max([float(c.mid()) for _, c in t]) for t in prime_comb(self.L).values()))
        d = {"L": self.L, "X": self.X, "A_L": float(self.A_L.mid()), "A_prime": float(self.A_prime.upper()),
             "T1_pointwise": 2 * math.pi * math.exp(float(self.A_L.mid())),
             "T1_certificate": 2 * math.pi * math.exp(float(self.A_prime.upper())),
             "separable_optimum": sep,
             "H_sup": self.H_sup, "n_terms": len(self.terms), "omega_min": self.omega_min,
             "per_prime": [{"p": pc.p, "K": pc.K, "M": pc.M, "A_p": float(pc.A_p.upper()),
                            "lambda_toeplitz": pc.lambda_toeplitz, "dual_degree_M": pc.dual,
                            "sum_c": float(sum(c.mid() for c in pc.c))} for pc in self.primes]}
        d.update(self.meta)
        return d


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
            cert.terms.append((p ** k, 1, h))
            cert.H_sup += float(abs(h).upper())
            omin = min(omin, float(om.lower()))
    cert.A_L = A_L
    cert.A_prime = arb(float(A_prime.upper()) * (1 + 1e-15))
    cert.H_sup *= (1 + 1e-12)
    cert.omega_min = omin
    return cert


# ----------------------------------------------------------------------------
# Joint (non-separable) certificates on the torus of prime phases
# ----------------------------------------------------------------------------
def _smooth_exponents(primes, Lam: float):
    """Exponent vectors b >= 0 with sum_p b_p log p <= Lam: the integers n = prod p^{b_p} <= e^Lam built from
    the given primes."""
    lp = [math.log(p) for p in primes]
    out = []

    def rec(i, cur, val):
        if i == len(primes):
            out.append(tuple(cur))
            return
        k = 0
        while val + k * lp[i] <= Lam + 1e-12:
            rec(i + 1, cur + [k], val + k * lp[i])
            k += 1
    rec(0, [], 0.0)
    return out


def joint_gram(L: float, Lam: float, mu_min: float = 1e-9, verbose: bool = False) -> dict:
    """Near-optimal Gram matrix (floats) for a joint certificate on the torus of prime phases.

    Monomials: e^{i <b, theta>} for the exponent vectors b of the integers n <= e^Lam built from the primes
    p < e^{2L}.  Differences b - b' correspond to ratios n/n'.  Dual (moment) problem: maximise
    sum_{p,k <= K_p} c_{p^k} y_{k e_p} subject to the moment matrix M(y)_{b,b'} = y_{b-b'} >= 0, y_0 = 1, and
    y_m = 0 whenever |log(n/n')| >= 2L (those monomials may appear in H).  Log-barrier Newton on y.
    Returns {'B', 'R' (rows of the Gram factor), 'dual', 'gap', 'n_free', ...}."""
    import numpy as np
    import scipy.sparse as sps
    comb = prime_comb(L)
    primes = sorted(comb)
    lp = np.array([math.log(p) for p in primes])
    B = np.array(_smooth_exponents(primes, Lam))
    nB = len(B)
    flat = (B[:, None, :] - B[None, :, :]).reshape(nB * nB, -1)
    first = np.argmax(flat != 0, axis=1)
    sgn = np.sign(flat[np.arange(nB * nB), first])
    sgn[sgn == 0] = 1
    keys, inv = np.unique(flat * sgn[:, None], axis=0, return_inverse=True)
    inv = inv.ravel()
    freq = np.abs(keys @ lp)
    zero = np.all(keys == 0, axis=1)
    free_idx = np.nonzero((~zero) & (freq < 2 * L - 1e-9))[0]
    col = -np.ones(len(keys), dtype=int)
    col[free_idx] = np.arange(len(free_idx))
    mask = col[inv] >= 0
    rows = np.arange(nB * nB)
    S = sps.csr_matrix((np.ones(mask.sum()), (rows[mask], col[inv[mask]])), shape=(nB * nB, len(free_idx)))
    Sc = S.tocsc()
    c = np.zeros(len(free_idx))
    for a, kidx in enumerate(free_idx):
        nz = np.nonzero(keys[kidx])[0]
        if len(nz) == 1:
            p, k = primes[nz[0]], keys[kidx][nz[0]]
            if k <= len(comb[p]):
                c[a] = float(comb[p][k - 1][1].mid())
    nf = len(c)
    pr = []
    for a in range(nf):
        ix = Sc.indices[Sc.indptr[a]:Sc.indptr[a + 1]]
        pr.append((ix // nB, ix % nB))

    def Mof(y):
        return np.eye(nB) + (S @ y).reshape(nB, nB)

    def fval(y, mu):
        try:
            Lc = np.linalg.cholesky(Mof(y))
        except np.linalg.LinAlgError:
            return None
        return float(c @ y) + mu * 2 * float(np.sum(np.log(np.diag(Lc))))

    y = np.zeros(nf)
    mu = 1.0
    while True:
        for _ in range(80):
            W = np.linalg.inv(Mof(y))
            g = c + mu * (S.T @ W.T.ravel())
            Hs = np.empty((nf, nf))
            for a in range(nf):
                r, s_ = pr[a]
                Hs[a] = S.T @ (W[:, r] @ W[s_, :]).T.ravel()
            Hs = -mu * (Hs + Hs.T) / 2
            dy = -np.linalg.solve(Hs, g)
            dec = float(g @ dy)
            if dec < 1e-13:
                break
            f0, step = fval(y, mu), 1.0
            while step > 1e-12:
                f1 = fval(y + step * dy, mu)
                if f1 is not None and f1 >= f0 + 0.25 * step * dec:
                    break
                step *= 0.5
            y = y + step * dy
        if verbose:
            print(f"  joint_gram L={L} Lam={Lam}: mu={mu:.0e} dual {float(c @ y):.6f}", flush=True)
        if mu <= mu_min:
            break
        mu = max(mu / 10.0, mu_min)
    X = mu * np.linalg.inv(Mof(y))
    X = (X + X.T) / 2
    lam, V = np.linalg.eigh(X)
    keep = lam > 1e-13 * lam[-1]
    R = np.sqrt(lam[keep])[:, None] * V[:, keep].T
    return {"B": B, "R": R, "primes": primes, "dual": float(c @ y), "gap": nB * mu, "n_free": nf, "nB": nB,
            "rank": int(keep.sum())}


def joint_gram_admm(L: float, Lam: float, iters: int = 20000, rho: float = 1.0, tol: float = 1e-9,
                    verbose: bool = False, report: int = 1000, alpha: float = 1.0, warm: dict | None = None) -> dict:
    """Near-optimal Gram matrix for the joint certificate by ADMM on the primal SOS problem (for monomial sets
    too large for the barrier method of `joint_gram`, whose Newton systems have one unknown per free moment).

    Primal: minimise tr X over symmetric X >= 0 with, for every free monomial m (|log(n/n')| < 2L), the sum of
    X over the pairs b - b' = m equal to t_m / 2 (t_m = -c_{p^k} on k e_p, 0 elsewhere).  Different monomials
    constrain disjoint sets of entries, so the projection onto the constraints is a uniform shift per
    monomial, and one ADMM step costs one eigendecomposition.  The iterate Z is positive semidefinite by
    construction; its value as a certificate is tr Z + sum_m |s_m(Z) - t_m| (mismatches are absorbed into A'
    by `joint_comb_certificate`), and the best Z seen is returned.  alpha > 1 is over-relaxation; `warm` (the
    result for a smaller Lambda, whose monomials are a subset) embeds its Gram matrix as the starting point."""
    import numpy as np
    comb = prime_comb(L)
    primes = sorted(comb)
    lp = np.array([math.log(p) for p in primes])
    B = np.array(_smooth_exponents(primes, Lam))
    nB = len(B)
    flat = (B[:, None, :] - B[None, :, :]).reshape(nB * nB, -1)
    first = np.argmax(flat != 0, axis=1)
    sgn = np.sign(flat[np.arange(nB * nB), first])
    plus = sgn > 0                                   # one orientation of every nonzero difference
    keys, inv = np.unique(flat[plus], axis=0, return_inverse=True)
    inv = inv.ravel()
    freq = np.abs(keys @ lp)
    free = freq < 2 * L - 1e-9
    fidx = np.nonzero(free)[0]
    col = -np.ones(len(keys), dtype=int)
    col[fidx] = np.arange(len(fidx))
    pos = np.nonzero(plus)[0]                        # flat indices (a nB + b) of the '+' entries
    kf = col[inv]
    sel = kf >= 0
    ent = pos[sel]                                   # entries constrained, one orientation
    kid = kf[sel]                                    # their free-monomial id
    entT = (ent % nB) * nB + ent // nB               # transposed entries
    cnt = np.bincount(kid, minlength=len(fidx)).astype(float)
    target = np.zeros(len(fidx))                     # t_m / 2
    for a, kidx in enumerate(fidx):
        nz = np.nonzero(keys[kidx])[0]
        if len(nz) == 1:
            p, k = primes[nz[0]], keys[kidx][nz[0]]
            if k <= len(comb[p]):
                target[a] = -float(comb[p][k - 1][1].mid()) / 2

    def proj_aff(V):
        v = V.ravel().copy()
        r = np.bincount(kid, weights=v[ent], minlength=len(fidx)) - target
        sh = (r / cnt)[kid]
        v[ent] -= sh
        v[entT] -= sh
        return v.reshape(nB, nB)

    def proj_psd(V):
        V = (V + V.T) / 2
        lam, Q = np.linalg.eigh(V)
        lam = np.maximum(lam, 0.0)
        return (Q * lam) @ Q.T

    def value(Zm):
        r = np.bincount(kid, weights=Zm.ravel()[ent], minlength=len(fidx)) - target
        return float(np.trace(Zm) + 2 * np.sum(np.abs(r))), float(2 * np.sum(np.abs(r)))

    I = np.eye(nB)
    Z = np.zeros((nB, nB))
    if warm is not None:
        pos_of = {tuple(b): i for i, b in enumerate(B.tolist())}
        idx = np.array([pos_of[tuple(b)] for b in warm["B"].tolist()])
        Z[np.ix_(idx, idx)] = warm["R"].T @ warm["R"]
    U = np.zeros((nB, nB))
    best = (float("inf"), None, None)
    for it in range(1, iters + 1):
        X = proj_aff(Z - U - I / rho)
        Zold = Z
        Xh = alpha * X + (1 - alpha) * Z
        Z = proj_psd(Xh + U)
        U = U + Xh - Z
        if it % 20 == 0 or it == iters:
            rp = np.linalg.norm(X - Z)
            rd = rho * np.linalg.norm(Z - Zold)
            val, mis = value(Z)
            if val < best[0]:
                best = (val, Z.copy(), mis)
            if rp > 10 * rd:
                rho *= 2.0
                U /= 2.0
            elif rd > 10 * rp:
                rho /= 2.0
                U *= 2.0
            if verbose and it % report == 0:
                print(f"  admm L={L} Lam={Lam} it={it}: value {val:.6f} (trace {val - mis:.6f}, mismatch {mis:.2e})"
                      f" r_p {rp:.1e} r_d {rd:.1e} rho {rho:.2g}", flush=True)
            if mis < tol and rp < tol and rd < tol:
                break
    val, Zb, mis = best
    lam, Q = np.linalg.eigh((Zb + Zb.T) / 2)
    keep = lam > 1e-14 * lam[-1]
    R = np.sqrt(lam[keep])[:, None] * Q[:, keep].T
    return {"B": B, "R": R, "primes": primes, "dual": float("nan"), "admm_value": val, "admm_mismatch": mis,
            "gap": float("nan"), "n_free": int(len(fidx)), "nB": nB, "rank": int(keep.sum()), "iters": it}


def admm_polish(L: float, Lam: float, G: dict) -> dict:
    """Second candidate from an ADMM Gram factor: project Z = R^T R exactly onto the constraints (zero
    mismatch) and add eps I with eps = max(0, -lambda_min), costing eps |B| in the trace instead of the L1
    mismatch.  Returns G with the better factor (values compared as tr X + sum |mismatch|)."""
    import numpy as np
    comb = prime_comb(L)
    primes = sorted(comb)
    lp = np.array([math.log(p) for p in primes])
    B = G["B"]
    nB = len(B)
    flat = (B[:, None, :] - B[None, :, :]).reshape(nB * nB, -1)
    first = np.argmax(flat != 0, axis=1)
    sgn = np.sign(flat[np.arange(nB * nB), first])
    plus = sgn > 0
    keys, inv = np.unique(flat[plus], axis=0, return_inverse=True)
    inv = inv.ravel()
    free = np.abs(keys @ lp) < 2 * L - 1e-9
    fidx = np.nonzero(free)[0]
    col = -np.ones(len(keys), dtype=int)
    col[fidx] = np.arange(len(fidx))
    pos = np.nonzero(plus)[0]
    kf = col[inv]
    sel = kf >= 0
    ent, kid = pos[sel], kf[sel]
    entT = (ent % nB) * nB + ent // nB
    cnt = np.bincount(kid, minlength=len(fidx)).astype(float)
    target = np.zeros(len(fidx))
    for a, kidx in enumerate(fidx):
        nz = np.nonzero(keys[kidx])[0]
        if len(nz) == 1:
            p, k = primes[nz[0]], keys[kidx][nz[0]]
            if k <= len(comb[p]):
                target[a] = -float(comb[p][k - 1][1].mid()) / 2
    Z = G["R"].T @ G["R"]
    r = np.bincount(kid, weights=Z.ravel()[ent], minlength=len(fidx)) - target
    val_l1 = float(np.trace(Z) + 2 * np.sum(np.abs(r)))
    v = Z.ravel().copy()
    sh = (r / cnt)[kid]
    v[ent] -= sh
    v[entT] -= sh
    X = v.reshape(nB, nB)
    X = (X + X.T) / 2
    lmin = float(np.linalg.eigvalsh(X)[0])
    eps = max(0.0, -lmin) * (1 + 1e-6) + 1e-14
    X = X + eps * np.eye(nB)
    val_proj = float(np.trace(X))
    out = dict(G)
    out["polish"] = {"value_l1": val_l1, "value_projected": val_proj, "eps": eps}
    if val_proj < val_l1:
        lam, Q = np.linalg.eigh(X)
        keep = lam > 0
        out["R"] = np.sqrt(lam[keep])[:, None] * Q[:, keep].T
        out["rank"] = int(keep.sum())
        out["admm_value"] = val_proj
        out["admm_mismatch"] = 0.0
    return out


def joint_comb_certificate(L: float, Lam: float = 5.0, drop: float = 1e-13, mu_min: float = 1e-9,
                           verbose: bool = False, solver: str = "barrier", admm_iters: int = 20000,
                           gram: dict | None = None, admm_alpha: float = 1.6) -> CombCertificate:
    """Joint SOS certificate (*) on the torus of prime phases (see `joint_gram`); every frequency of H is
    log(n/n') with 2L <= log(n/n') <= Lam.

    Rigour: the float Gram factor R is rounded to the fixed-point grid 2^-80 (any real R gives a sum of
    squares S(theta) = sum_i |sum_b R_ib e^{i<b,theta>}|^2 >= 0); X = R^T R is then formed exactly with
    integer matrices, and its coefficients s_m = 2 sum_{b - b' = m} X_{b b'} are exact dyadic numbers.  Every
    coefficient whose frequency |log(n/n')| is not certified >= 2L must match the comb (-c_{p^k} on k e_p, 0
    elsewhere); mismatches are absorbed into A' = s_0 + sum |s_m - target_m|.  The others form H, except
    that coefficients below `drop` are moved into A' as well (|h cos| <= |h|)."""
    import numpy as np
    from flint import arb, ctx, fmpz, fmpz_mat
    if gram is not None:                            # a precomputed (float) Gram factor; only its quality matters
        G = gram
        solver = "admm" if "admm_value" in gram else solver
    elif solver == "admm":
        G = admm_polish(L, Lam, joint_gram_admm(L, Lam, iters=admm_iters, verbose=verbose, alpha=admm_alpha))
    else:
        G = joint_gram(L, Lam, mu_min=mu_min, verbose=verbose)
    B, R, primes = G["B"], G["R"], G["primes"]
    comb = prime_comb(L)
    nB = len(B)
    FB = 80
    Ri = [[int(round(float(v) * 2.0 ** FB)) for v in row] for row in R]
    Rm = fmpz_mat(Ri)
    Xm = Rm.transpose() * Rm                       # exact: X = R^T R on the grid, scaled by 2^{2 FB}
    sums = {}
    for a in range(nB):
        for b in range(nB):
            m = tuple(int(v) for v in (B[a] - B[b]))
            if any(m):
                nz = next(i for i, v in enumerate(m) if v)
                if m[nz] < 0:
                    continue                       # count each unordered pair once (orientation +m)
            sums[m] = sums.get(m, 0) + int(Xm[a, b])
    old = ctx.prec
    ctx.prec = 4000
    try:
        scale = arb(2) ** (-2 * FB)
        coef = {m: (arb(v) * scale if not any(m) else arb(v) * scale) for m, v in sums.items()}
        coef = {m: (v if not any(m) else 2 * v) for m, v in coef.items()}
        assert all(v.rad() == 0 for v in coef.values())
    finally:
        ctx.prec = old
    LL = arb(L)
    A = coef[tuple([0] * len(primes))]
    target = {}
    for i, p in enumerate(primes):
        for k, (_, cval) in enumerate(comb[p], start=1):
            m = [0] * len(primes)
            m[i] = k
            target[tuple(m)] = -cval
    cert = CombCertificate(L=L, X=Lam)
    A_L = arb(0)
    for t in comb.values():
        for _, cval in t:
            A_L += cval
    cert.A_L = A_L
    omin, n_drop = float("inf"), 0
    for m, v in coef.items():
        if not any(m):
            continue
        num = 1
        den = 1
        for p, e in zip(primes, m):
            if e > 0:
                num *= p ** e
            elif e < 0:
                den *= p ** (-e)
        om = arb(max(num, den)).log() - arb(min(num, den)).log()
        if om > 2 * LL or om == 2 * LL:            # certified frequency >= 2L: may stay in H
            if abs(v) < drop:
                A += abs(v)
                n_drop += 1
                continue
            cert.terms.append((max(num, den), min(num, den), v))
            cert.H_sup += float(abs(v).upper())
            omin = min(omin, float(om.lower()))
        else:
            A += abs(v - target.get(m, arb(0)))
    for m in target:                                # comb monomials absent from the differences (not expected)
        if m not in coef:
            A += abs(target[m])
    cert.A_prime = arb(float(A.upper()) * (1 + 1e-15))
    cert.H_sup *= (1 + 1e-12)
    cert.omega_min = omin
    cert.meta = {"kind": "joint", "Lambda": Lam, "nB": nB, "n_free": G["n_free"], "dual": G["dual"],
                 "gap": G["gap"], "rank": G["rank"], "n_dropped": n_drop, "solver": solver}
    if solver == "admm":
        cert.meta.update({"admm_value": G["admm_value"], "admm_mismatch": G["admm_mismatch"], "admm_iters": G["iters"]})
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
