"""Elementary arithmetic equivalents of the Riemann Hypothesis.

This module implements the classical *elementary* reformulations of RH in
terms of the sum-of-divisors function sigma(n), Euler's totient phi(n) and the
harmonic numbers H_n, together with the extremal integers for which the
inequalities are sharpest (primorials and colossally abundant numbers).  All of
them follow from Gronwall's theorem

    limsup_{n -> oo} sigma(n) / (n log log n) = e^gamma           (Gronwall 1913)

made quantitative with/without RH.  Nothing here can *prove* RH: a finite
computation can only exhibit a counterexample (which would disprove RH) or show
that none exists up to the bound checked.

Criteria
--------
Robin (1984), Theorem 1
    RH  <=>  sigma(n) < e^gamma * n * log log n   for all n > 5040.
    Unconditionally (Robin's Theorem 2) sigma(n)/n <= e^gamma log log n
    + 0.6483/log log n for n >= 3.  The inequality fails exactly for the 27
    integers 2 <= n <= 5040 listed in ``ROBIN_EXCEPTIONS`` (for n = 1 the
    right-hand side is undefined, for n = 2 it is negative).
Lagarias (2002)
    RH  <=>  sigma(n) <= H_n + exp(H_n) log H_n   for all n >= 1,
    with equality only for n = 1 (H_n = sum_{j<=n} 1/j).  This version has no
    exceptional set; it is slightly weaker than Robin's for large n because
    exp(H_n) log H_n ~ e^gamma n log(log n + gamma).
Nicolas (1983)
    Let N_k = p_1 p_2 ... p_k be the k-th primorial.  Then
    RH  <=>  N_k / phi(N_k) > e^gamma log log N_k   for all k >= 1.
    (If RH fails the inequality still holds for infinitely many k but also
    fails for infinitely many k.)  For k = 1 the right-hand side is negative
    (log log 2 < 0), so the inequality is only meaningful for k >= 2.
Colossally abundant (CA) numbers (Alaoglu-Erdos 1944)
    n is CA if for some eps > 0, sigma(n)/n^(1+eps) >= sigma(m)/m^(1+eps) for
    all m > 1.  For fixed eps the maximiser is unique (barring ties) and has
    exponents a_p(eps) = floor( log((p^(1+eps)-1)/(p^eps-1)) / log p ) - 1.
    The CA numbers form a chain in which consecutive terms differ by one prime
    factor.  Robin showed that if his inequality fails for some n > 5040 then
    it fails for a CA number, so CA numbers are the natural test set (Briggs
    2006 verified Robin's inequality on CA numbers up to 10^(10^10)).

References
----------
* T. H. Gronwall, Some asymptotic expressions in the theory of numbers,
  Trans. Amer. Math. Soc. 14 (1913) 113-122.
* L. Alaoglu, P. Erdos, On highly composite and similar numbers,
  Trans. Amer. Math. Soc. 56 (1944) 448-469.
* J.-L. Nicolas, Petites valeurs de la fonction d'Euler,
  J. Number Theory 17 (1983) 375-388.
* G. Robin, Grandes valeurs de la fonction somme des diviseurs et hypothese de
  Riemann, J. Math. Pures Appl. 63 (1984) 187-213.
* J. C. Lagarias, An elementary problem equivalent to the Riemann hypothesis,
  Amer. Math. Monthly 109 (2002) 534-543.
* K. Briggs, Abundant numbers and the Riemann hypothesis,
  Experiment. Math. 15 (2006) 251-256.
* OEIS A004490 (colossally abundant numbers), A067698 (Robin exceptions >= 3).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Sequence, Tuple

import mpmath as mp
import numpy as np
import sympy

from .constants import EULER_GAMMA

E_GAMMA = math.exp(EULER_GAMMA)  # 1.7810724179901979...

#: The 27 integers n >= 2 violating Robin's inequality sigma(n) < e^gamma n log log n.
#: n = 2 violates it trivially (log log 2 < 0); n = 1 is excluded (log log 1 undefined).
ROBIN_EXCEPTIONS: Tuple[int, ...] = (
    2, 3, 4, 5, 6, 8, 9, 10, 12, 16, 18, 20, 24, 30, 36, 48, 60, 72, 84,
    120, 180, 240, 360, 720, 840, 2520, 5040,
)

#: First colossally abundant numbers, OEIS A004490.
CA_NUMBERS_KNOWN: Tuple[int, ...] = (
    2, 6, 12, 60, 120, 360, 2520, 5040, 55440, 720720, 1441440, 4324320,
    21621600, 367567200, 6983776800, 160626866400, 321253732800,
    9316358251200, 288807105787200, 2021649740510400, 6064949221531200,
)


# ----------------------------------------------------------------------------
# sigma(n) and H_n by sieving
# ----------------------------------------------------------------------------
def sigma_sieve(N: int) -> np.ndarray:
    """Return an int64 array ``sig`` with ``sig[n] = sigma(n)`` for 1 <= n <= N.

    Divisor-sum sieve (exact integer arithmetic).  Divisors d <= sqrt(N) are
    added with strided slices ``sig[d::d] += d``; divisors d > sqrt(N) are
    added by looping over the cofactor q = n/d < sqrt(N) instead, so the number
    of Python-level iterations is O(sqrt N) while the total work stays
    O(N log N).  N = 10^7 takes about 2 s.  ``sig[0] = 0``.
    """
    if N < 1:
        raise ValueError("N must be >= 1")
    sig = np.zeros(N + 1, dtype=np.int64)
    K = math.isqrt(N)
    for d in range(1, K + 1):
        sig[d::d] += d
    for q in range(1, N // (K + 1) + 1):
        dmax = N // q
        if dmax <= K:
            break
        sig[q * (K + 1): q * dmax + 1: q] += np.arange(K + 1, dmax + 1, dtype=np.int64)
    return sig


def harmonic_numbers(N: int) -> np.ndarray:
    """Return a longdouble array ``H`` with ``H[n] = H_n`` (``H[0] = 0``), 1 <= n <= N.

    Plain cumulative summation in 80-bit extended precision; the accumulated
    rounding error for N = 10^7 is below 1e-12 absolute, far smaller than any
    slack encountered in Lagarias' inequality.
    """
    H = np.zeros(N + 1, dtype=np.longdouble)
    H[1:] = np.cumsum(1.0 / np.arange(1, N + 1, dtype=np.longdouble))
    return H


# ----------------------------------------------------------------------------
# Robin's criterion
# ----------------------------------------------------------------------------
@dataclass
class RobinResult:
    N: int
    violators: List[int]                 # all n in [2, N] with sigma(n) >= e^gamma n log log n
    violators_match_known: bool          # == ROBIN_EXCEPTIONS
    max_ratio_beyond_5040: float         # max_{5040 < n <= N} sigma(n)/(n log log n)
    argmax_beyond_5040: int
    max_robin_ratio_beyond_5040: float   # the same divided by e^gamma (< 1 iff Robin holds)
    ratio_at_5040: float                 # sigma(5040)/(5040 log log 5040) / e^gamma  (> 1)
    top: List[Tuple[int, float]] = field(default_factory=list)  # 10 largest Robin ratios beyond 5040

    def to_dict(self) -> dict:
        d = asdict(self)
        d["top"] = [[int(n), float(r)] for n, r in self.top]
        return d


def robin_check(N: int, sigma: Optional[np.ndarray] = None) -> RobinResult:
    """Check Robin's inequality sigma(n) < e^gamma n log log n for 2 <= n <= N.

    Returns every violator (the strict inequality fails) in [2, N] -- which
    must be exactly ``ROBIN_EXCEPTIONS`` -- and the maximum of
    sigma(n)/(n log log n) over 5040 < n <= N with its argmax.  The quantity
    ``max_robin_ratio_beyond_5040`` is that maximum divided by e^gamma; RH
    predicts it is < 1 for every N (Robin 1984).
    """
    if N < 5041:
        raise ValueError("N must be > 5040 to say anything beyond the exceptional set")
    sig = sigma_sieve(N) if sigma is None else sigma
    n = np.arange(2, N + 1, dtype=np.float64)
    loglog = np.log(np.log(n))
    s = sig[2:N + 1].astype(np.float64)
    viol = (np.nonzero(s >= E_GAMMA * n * loglog)[0] + 2).tolist()
    ratio = s / (n * loglog)                       # sigma(n)/(n log log n), n >= 2
    beyond = ratio[5039:]                          # n = 5041 .. N
    i = int(np.argmax(beyond))
    k = min(10, beyond.size)
    idx = np.argpartition(-beyond, k - 1)[:k]
    idx = idx[np.argsort(-beyond[idx])]
    top = [(int(j) + 5041, float(beyond[j] / E_GAMMA)) for j in idx]
    return RobinResult(
        N=N,
        violators=[int(v) for v in viol],
        violators_match_known=(tuple(viol) == ROBIN_EXCEPTIONS),
        max_ratio_beyond_5040=float(beyond[i]),
        argmax_beyond_5040=i + 5041,
        max_robin_ratio_beyond_5040=float(beyond[i] / E_GAMMA),
        ratio_at_5040=float(ratio[5038] / E_GAMMA),
        top=top,
    )


# ----------------------------------------------------------------------------
# Lagarias' criterion
# ----------------------------------------------------------------------------
@dataclass
class LagariasResult:
    N: int
    violators: List[int]            # n >= 2 with sigma(n) >= H_n + exp(H_n) log H_n, or n = 1 with ">"
    slack_at_1: float               # H_1 + e^{H_1} log H_1 - sigma(1) = 0 exactly
    min_slack: float                # min_{2 <= n <= N} (RHS - sigma(n))
    argmin_slack: int
    max_ratio: float                # max_{2 <= n <= N} sigma(n)/RHS   (< 1 iff no violation)
    argmax_ratio: int
    slack_at_5040: float
    ratio_at_5040: float

    def to_dict(self) -> dict:
        return asdict(self)


def lagarias_check(N: int, sigma: Optional[np.ndarray] = None) -> LagariasResult:
    """Check Lagarias' inequality sigma(n) <= H_n + exp(H_n) log H_n for 1 <= n <= N.

    Equality must hold for n = 1 only.  Returns the violators (none expected),
    the minimal slack RHS - sigma(n) over n >= 2 and the maximal ratio
    sigma(n)/RHS over n >= 2.  Computed in 80-bit extended precision.
    """
    sig = sigma_sieve(N) if sigma is None else sigma
    H = harmonic_numbers(N)[1:]                      # H_1..H_N
    rhs = H + np.exp(H) * np.log(H)                  # n = 1..N
    s = sig[1:N + 1].astype(np.longdouble)
    slack = rhs - s
    viol = (np.nonzero(slack[1:] <= 0)[0] + 2).tolist()
    if slack[0] < 0:
        viol.insert(0, 1)
    ratio = s[1:] / rhs[1:]
    i = int(np.argmin(slack[1:]))
    j = int(np.argmax(ratio))
    return LagariasResult(
        N=N,
        violators=[int(v) for v in viol],
        slack_at_1=float(slack[0]),
        min_slack=float(slack[1:][i]),
        argmin_slack=i + 2,
        max_ratio=float(ratio[j]),
        argmax_ratio=j + 2,
        slack_at_5040=float(slack[5039]) if N >= 5040 else float("nan"),
        ratio_at_5040=float(ratio[5038]) if N >= 5040 else float("nan"),
    )


# ----------------------------------------------------------------------------
# Nicolas' criterion (primorials)
# ----------------------------------------------------------------------------
@dataclass
class NicolasResult:
    K: int
    primes: List[int]               # p_1 .. p_K
    log_primorial: List[float]      # log N_k
    ratio: List[float]              # (N_k/phi(N_k)) / (e^gamma log log N_k); negative for k = 1
    difference: List[float]         # N_k/phi(N_k) - e^gamma log log N_k   (> 0 predicted by RH)
    violations: List[int]           # k >= 2 with ratio <= 1
    decreasing_from_2: bool         # ratio strictly decreasing for k >= 2
    min_ratio: float                # min over k >= 2 (attained at k = K if decreasing)
    scaled_tail: List[Tuple[int, float]]  # (k, (ratio-1) sqrt(p_k) log p_k) for the last few k
    dps: int

    def to_dict(self) -> dict:
        d = asdict(self)
        d["scaled_tail"] = [[int(k), float(v)] for k, v in self.scaled_tail]
        return d


def nicolas_check(K: int, dps: int = 50, keep_every: int = 1) -> NicolasResult:
    """Evaluate Nicolas' inequality for the first K primorials in `dps`-digit precision.

    N_k is never formed: log N_k = sum log p_i and log(N_k/phi(N_k)) =
    -sum log(1 - 1/p_i) are accumulated with mpmath.  The ratio
    (N_k/phi(N_k)) / (e^gamma log log N_k) should exceed 1 for all k >= 2 and
    (empirically, as long as theta(x) < x) decrease monotonically towards 1;
    Mertens' theorem gives N_k/phi(N_k) ~ e^gamma log p_k and the prime number
    theorem gives log log N_k = log theta(p_k) ~ log p_k.
    ``keep_every`` thins the stored sequences (every value is still checked).
    """
    if K < 2:
        raise ValueError("K must be >= 2")
    pK = sympy.prime(K)
    primes = list(sympy.primerange(2, pK + 1))
    assert len(primes) == K
    with mp.workdps(dps):
        eg = mp.exp(mp.euler)
        logN = mp.mpf(0)
        logP = mp.mpf(0)
        ratios: List[float] = []
        diffs: List[float] = []
        logNs: List[float] = []
        kept_primes: List[int] = []
        viol: List[int] = []
        prev = None
        decreasing = True
        min_ratio = mp.inf
        full_ratios = []
        for k, p in enumerate(primes, start=1):
            pm = mp.mpf(p)
            logN += mp.log(pm)
            logP += -mp.log(1 - 1 / pm)
            P = mp.exp(logP)
            ll = mp.log(logN)                # log log N_k (negative for k = 1)
            r = P / (eg * ll)
            dlt = P - eg * ll
            full_ratios.append(r)
            if k >= 2:
                if r <= 1:
                    viol.append(k)
                if prev is not None and r >= prev:
                    decreasing = False
                prev = r
                if r < min_ratio:
                    min_ratio = r
            if (k - 1) % keep_every == 0 or k == K:
                kept_primes.append(int(p))
                logNs.append(float(logN))
                ratios.append(float(r))
                diffs.append(float(dlt))
        tail = []
        for k in range(max(2, K - 4), K + 1):
            p = primes[k - 1]
            tail.append((k, float((full_ratios[k - 1] - 1) * mp.sqrt(p) * mp.log(p))))
    return NicolasResult(
        K=K, primes=kept_primes, log_primorial=logNs, ratio=ratios, difference=diffs,
        violations=viol, decreasing_from_2=decreasing, min_ratio=float(min_ratio),
        scaled_tail=tail, dps=dps,
    )


# ----------------------------------------------------------------------------
# Colossally abundant numbers
# ----------------------------------------------------------------------------
def ca_critical_epsilon(p: int, a: int, dps: int = 60) -> mp.mpf:
    """The value of eps at which the optimal exponent of p jumps from a to a+1.

    With g_p(a) = sigma(p^a)/p^{a(1+eps)} one has g_p(a+1) >= g_p(a) iff
    eps <= log_p((p^{a+2}-1)/(p^{a+1}-1)) - 1 =: eps_p(a).  For a = 0 this is
    log(1 + 1/p)/log p, decreasing in p, so the next new prime to enter a CA
    number is always the smallest prime not yet present.
    """
    with mp.workdps(dps):
        num = mp.mpf(p ** (a + 2) - 1)
        den = mp.mpf(p ** (a + 1) - 1)
        return mp.log(num / den) / mp.log(p) - 1


def ca_exponent(p: int, eps, dps: int = 60) -> int:
    """Alaoglu-Erdos exponent a_p(eps) = floor(log((p^(1+eps)-1)/(p^eps-1))/log p) - 1."""
    with mp.workdps(dps):
        e = mp.mpf(eps)
        pm = mp.mpf(p)
        x = mp.log((pm ** (1 + e) - 1) / (pm ** e - 1)) / mp.log(pm)
        return int(mp.floor(x)) - 1


def ca_number_for_epsilon(eps, dps: int = 60) -> Dict[int, int]:
    """Factorisation {p: a_p(eps)} of the CA number maximising sigma(n)/n^(1+eps).

    Primes are included while a_p(eps) >= 1, i.e. while eps <= log(1+1/p)/log p.
    """
    fact: Dict[int, int] = {}
    for p in sympy.primerange(2, 10 ** 7):
        a = ca_exponent(p, eps, dps)
        if a <= 0:
            break
        fact[int(p)] = a
    return fact


def factored_arithmetic(fact: Dict[int, int], dps: int = 60) -> Dict[str, mp.mpf]:
    """High-precision log n, sigma(n)/n, Robin/Gronwall/Lagarias quantities from a factorisation.

    log n = sum a log p;  sigma(n)/n = prod (p^{a+1}-1)/(p^a (p-1)).
    Lagarias' right-hand side uses H_n = psi(n+1) + gamma (mpmath ``harmonic``),
    exact to working precision also for n with hundreds of digits.
    """
    with mp.workdps(dps):
        logn = mp.mpf(0)
        log_sn = mp.mpf(0)
        n = 1
        for p, a in fact.items():
            logn += a * mp.log(p)
            log_sn += mp.log(mp.mpf(p ** (a + 1) - 1)) - a * mp.log(p) - mp.log(p - 1)
            n *= p ** a
        sn = mp.exp(log_sn)                        # sigma(n)/n
        eg = mp.exp(mp.euler)
        ll = mp.log(logn) if logn > 1 else mp.mpf("-inf")   # log log n (n >= 3 needed)
        if logn > 1:
            robin = sn / (eg * ll)
            gron = sn / ll
            D = mp.sqrt(logn) * (eg * ll - sn)
        else:
            robin = gron = D = mp.mpf("nan")
        H = mp.harmonic(n)
        log_rhs = mp.log(H + mp.exp(H) * mp.log(H))
        log_sigma = logn + log_sn
        return {
            "n": n, "log_n": logn, "sigma_over_n": sn, "loglog_n": ll,
            "robin_ratio": robin, "gronwall_ratio": gron, "robin_scaled": D,
            "lagarias_log_slack": log_rhs - log_sigma,
        }


@dataclass
class CANumber:
    index: int
    factorization: Dict[int, int]
    n: int
    critical_epsilon: float          # n is the maximiser of sigma(m)/m^(1+eps) for eps in (next_eps, this)
    digits: int
    omega: int
    log_n: float
    sigma_over_n: float
    robin_ratio: float               # sigma(n) / (e^gamma n log log n)
    gronwall_ratio: float            # sigma(n) / (n log log n)  -> e^gamma
    one_minus_robin: float
    robin_scaled: float              # sqrt(log n) * (e^gamma log log n - sigma(n)/n)
    lagarias_log_slack: float        # log(H_n + e^{H_n} log H_n) - log sigma(n)   (> 0)
    robin_ok: bool                   # ratio < 1 (only required for n > 5040)
    lagarias_ok: bool
    gap_to_runner_up: float          # eps_best - eps_second at the generating step (no tie if > 0)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["n"] = str(self.n)
        d["factorization"] = [[int(p), int(a)] for p, a in self.factorization.items()]
        return d


def ca_numbers(K: int, dps: int = 60, verify_formula: bool = True) -> List[CANumber]:
    """Generate the first K colossally abundant numbers with exact factorisations.

    Algorithm (Alaoglu-Erdos; cf. Briggs 2006): start from n = 1 and repeatedly
    multiply by the prime p whose critical epsilon eps_p(a_p) is largest among
    the primes already present (a_p = current exponent) and the smallest
    absent prime (a = 0).  Critical epsilons are compared at `dps` digits; the
    smallest gap between the best and second-best candidate over all steps is
    recorded (``gap_to_runner_up``) -- it is positive for all K tested, so no
    tie (which would make the CA number for that eps non-unique) occurs.

    If ``verify_formula`` is True, each factorisation is re-derived from the
    closed-form exponents a_p(eps) at the midpoint of its epsilon interval.
    """
    primes = list(sympy.primerange(2, sympy.prime(K + 2) + 1))
    expo: Dict[int, int] = {}
    out: List[CANumber] = []
    n = 1
    nxt = 0
    eps_list: List[mp.mpf] = []
    gaps: List[float] = []
    with mp.workdps(dps):
        for k in range(1, K + 1):
            cands = [(ca_critical_epsilon(p, a, dps), p) for p, a in expo.items()]
            cands.append((ca_critical_epsilon(primes[nxt], 0, dps), primes[nxt]))
            cands.sort(key=lambda c: -c[0])
            e, p = cands[0]
            gap = float(cands[0][0] - cands[1][0]) if len(cands) > 1 else float("inf")
            if p == primes[nxt]:
                nxt += 1
            expo[p] = expo.get(p, 0) + 1
            n *= p
            eps_list.append(e)
            gaps.append(gap)
            fact = dict(sorted(expo.items()))
            q = factored_arithmetic(fact, dps)
            out.append(CANumber(
                index=k, factorization=fact, n=n, critical_epsilon=float(e),
                digits=len(str(n)), omega=len(fact), log_n=float(q["log_n"]),
                sigma_over_n=float(q["sigma_over_n"]),
                robin_ratio=float(q["robin_ratio"]), gronwall_ratio=float(q["gronwall_ratio"]),
                one_minus_robin=float(1 - q["robin_ratio"]), robin_scaled=float(q["robin_scaled"]),
                lagarias_log_slack=float(q["lagarias_log_slack"]),
                robin_ok=bool(q["robin_ratio"] < 1) if n > 2 else False,
                lagarias_ok=bool(q["lagarias_log_slack"] > 0),
                gap_to_runner_up=gap,
            ))
        if verify_formula:
            # the eps-interval of out[k] is (eps_{k+1}, eps_k); use its midpoint
            nxt_eps = eps_list[1:] + [ca_critical_epsilon(primes[nxt], 0, dps)]
            # the last interval's lower end is the max over the remaining candidates
            last_cands = [ca_critical_epsilon(p, a, dps) for p, a in expo.items()]
            last_cands.append(nxt_eps[-1])
            nxt_eps[-1] = max(last_cands)
            for ca, hi, lo in zip(out, eps_list, nxt_eps):
                mid = (hi + lo) / 2
                fact = ca_number_for_epsilon(mid, dps)
                if fact != ca.factorization:
                    raise AssertionError(f"CA #{ca.index}: chain {ca.factorization} != formula {fact}")
    return out


def brute_force_ca_argmax(eps: float, N: int, sigma: Optional[np.ndarray] = None) -> int:
    """argmax_{1 <= n <= N} sigma(n)/n^(1+eps) by brute force (float64, log scale)."""
    sig = sigma_sieve(N) if sigma is None else sigma
    n = np.arange(1, N + 1, dtype=np.float64)
    val = np.log(sig[1:N + 1].astype(np.float64)) - (1.0 + eps) * np.log(n)
    return int(np.argmax(val)) + 1


def ca_summary(cas: Sequence[CANumber]) -> dict:
    """Aggregate statistics of a CA list for reporting."""
    beyond = [c for c in cas if c.n > 5040]
    return {
        "count": len(cas),
        "largest_digits": cas[-1].digits,
        "largest_omega": cas[-1].omega,
        "largest_log_n": cas[-1].log_n,
        "first_14": [str(c.n) for c in cas[:14]],
        "matches_A004490": [c.n for c in cas[:len(CA_NUMBERS_KNOWN)]] == list(CA_NUMBERS_KNOWN[:len(cas)]),
        "robin_holds_for_all_n_gt_5040": all(c.robin_ok for c in beyond),
        "lagarias_holds_for_all": all(c.lagarias_ok for c in cas),
        "max_robin_ratio_beyond_5040": max(c.robin_ratio for c in beyond) if beyond else None,
        "argmax_robin_ratio_beyond_5040_index": max(beyond, key=lambda c: c.robin_ratio).index if beyond else None,
        "last_robin_ratio": cas[-1].robin_ratio,
        "last_one_minus_robin": cas[-1].one_minus_robin,
        "last_robin_scaled": cas[-1].robin_scaled,
        "last_gronwall_ratio": cas[-1].gronwall_ratio,
        "e_gamma": E_GAMMA,
        "min_gap_to_runner_up": min(c.gap_to_runner_up for c in cas),
        "min_lagarias_log_slack": min(c.lagarias_log_slack for c in cas),
    }
