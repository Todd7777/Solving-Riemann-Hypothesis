r"""Localised high-pass corrections for the prime comb (wave packets), for trajectory certificates.

A trajectory certificate (`rh.weil_comb.trajectory_check`, E15) needs

    Omega(t) - P_L(t) + H(t) >= beta        for t >= T

only, where H is invisible to the window (spectrum outside (-2L, 2L)). A torus certificate H leaves
isolated violations on [T, T3]. They can be removed by adding wave packets

    H_loc(t) = sum_k a_k (psi_k(t - tau_k) + psi_k(-t - tau_k)),   psi_k(s) = e(s) cs_k(w_k s),
    e(s) = (sin(b s / m) / (b s / m))^m,                           cs = cos (kind 0) or sin (kind 1).

e is the m-fold convolution of a box in frequency, so its Fourier transform is supported in [-b, b],
and psi_k has spectrum in +-[w_k - b, w_k + b]. If every w_k - b >= 2L (checked in Arb), H_loc is an
even, integrable function whose spectrum avoids (-2L, 2L). So int_0^inf H_loc |F|^2 = 0 for every f
supported in [-L, L], exactly as for H. Unlike H, it is localised: on the real line
|e(s)| <= min(1, (m / (b |s|))^m).

What is rigorous here:
* `H` evaluates H_loc in Arb at real or complex z, for the envelope's quadrature nodes. Every atom is
  evaluated, because tails are only polynomially small.
* `bound_on_ellipse` bounds |H_loc| on a Bernstein ellipse, using |sin x| <= cosh(Im x), |sin x / x| <=
  cosh(Im x) and |sin x / x| <= cosh(Im x) / |Re x|.
* `eval_float` evaluates H_loc in IEEE double on grids, with an error bound per block. Phases come from
  Arb and products, as in `rh.weil_comb._TrajectoryEvaluator`. Atoms beyond `reach` and all mirror terms
  are bounded by the tail estimate instead of evaluated.
* `M2_on` bounds |H_loc''| on an interval: (w + b)^2 per atom near it (Bernstein's inequality, since
  psi has spectrum in [-(w+b), w+b] and |psi| <= 1), and (w + 2b)^2 (m / (b d))^m at distance d >= m/b.
  The second bound follows from |u|, |u'|, |u''| <= 1/x, 2/x, 5/x for u = sin x / x and x >= 1.

`fit_local_correction` finds the amplitudes by a receding-horizon linear program (scipy HiGHS). This step
is numerical: it affects only how good the correction is, never whether a certificate built on it is
valid.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

__all__ = ["LocalCorrection", "fit_local_correction"]

_U = 2.0 ** -53


def _gam(k: int) -> float:
    return k * _U / (1 - k * _U)


@dataclass
class LocalCorrection:
    """H_loc = sum_k a_k (psi_k(t - tau_k) + psi_k(-t - tau_k)); see the module docstring."""
    L: float
    b: float
    m: int
    atoms: list = field(default_factory=list)         # [(tau, w, kind, a)], floats (exact binary), kind 0/1

    def __post_init__(self):
        self.check_spectrum()

    # ------------------------------------------------------------------ basic bounds
    def check_spectrum(self):
        """Every w - b >= 2L, decided in Arb (the carriers are exact binary floats)."""
        from flint import arb
        LL, bb = arb(self.L), arb(self.b)
        for w in sorted({w for _, w, _, _ in self.atoms}):
            if not (arb(w) - bb >= 2 * LL):
                raise ValueError(f"carrier {w} - b = {w - self.b} is not >= 2L = {2 * self.L}")

    def sum_abs(self) -> float:
        return math.fsum(abs(a) for _, _, _, a in self.atoms) * (1 + 1e-12)

    def sup_bound(self) -> float:
        """sup over real t of |H_loc|: |e| <= 1 and |cs| <= 1 for each of the two terms."""
        return 2 * self.sum_abs()

    def tail_bound(self, t: float) -> float:
        """A bound for |H_loc(s)| for every s >= t, valid when t >= max tau + m / b."""
        mb = self.m / self.b
        tot = 0.0
        for tau, _, _, a in self.atoms:
            if t - tau < mb:
                raise ValueError("tail_bound needs t >= max tau + m/b")
            tot += abs(a) * ((mb / (t - tau)) ** self.m + (mb / (t + tau)) ** self.m)
        return tot * (1 + 1e-9)

    def M2_on(self, lo: float, hi: float) -> float:
        """A bound for |H_loc''| on [lo, hi] (lo >= 0)."""
        mb = self.m / self.b
        tot = 0.0
        for tau, w, _, a in self.atoms:
            for d in (max(0.0, lo - tau, tau - hi), lo + tau):     # the atom and its mirror term
                x = d / mb
                tot += abs(a) * ((w + self.b) ** 2 if x < 1 else (w + 2 * self.b) ** 2 * x ** (-self.m))
        return tot * (1 + 1e-9)

    def bound_on_ellipse(self, c: float, h: float, rho: float) -> float:
        """sup |H_loc(z)| on the Bernstein ellipse of [c - h, c + h] with parameter rho: there
        |Im z| <= b_e = h (rho - 1/rho)/2 and |Re z - c| <= a_e = h (rho + 1/rho)/2."""
        b_e = h * (rho - 1 / rho) / 2 * (1 + 1e-12)
        a_e = h * (rho + 1 / rho) / 2 * (1 + 1e-12)
        mb = self.m / self.b
        ce = math.cosh(b_e / mb) ** self.m
        tot = 0.0
        for tau, w, _, a in self.atoms:
            cw = math.cosh(w * b_e)
            for d in (max(0.0, abs(c - tau) - a_e), max(0.0, c + tau - a_e)):
                tot += abs(a) * ce * cw * min(1.0, mb / d if d > 0 else 1.0) ** self.m
        return tot * (1 + 1e-9) + 1e-300

    # ------------------------------------------------------------------ Arb evaluation
    def _consts(self):
        """Per-precision cache: carrier phases e^{-i w tau}, envelope phases e^{-i (b/m) tau} and inverses."""
        from flint import acb, arb, ctx
        cache = self.__dict__.setdefault("_cache", {})
        if ctx.prec not in cache:
            I = acb(0, 1)
            nu = arb(self.b) / self.m
            ws = sorted({w for _, w, _, _ in self.atoms})
            per = []
            for tau, w, kind, a in self.atoms:
                cw = (-I * arb(w) * arb(tau)).exp()
                ce = (-I * nu * arb(tau)).exp()
                per.append((arb(tau), ws.index(w), kind, arb(a), cw, 1 / cw, ce, 1 / ce))
            cache[ctx.prec] = (nu, [arb(w) for w in ws], per)
        return cache[ctx.prec]

    def H(self, z):
        """H_loc(z) (acb) at real or complex z, every atom evaluated at the current precision."""
        from flint import acb
        z = acb(z)
        nu, ws, per = self._consts()
        I = acb(0, 1)
        Ew = [(I * w * z).exp() for w in ws]
        Ewi = [1 / e for e in Ew]
        Ee = (I * nu * z).exp()
        Eei = 1 / Ee
        m = self.m
        tot = acb(0)
        for tau, j, kind, a, cw, cwi, ce, cei in per:
            for sgn in (1, -1):                          # psi(z - tau) and psi(-z - tau)
                if sgn == 1:
                    x = nu * (z - tau)
                    ep, em = Ee * ce, Eei * cei           # e^{ix}, e^{-ix}
                    qp, qm = Ew[j] * cw, Ewi[j] * cwi     # e^{iw s}, e^{-iw s}, s = z - tau
                else:
                    x = nu * (-z - tau)
                    ep, em = Eei * ce, Ee * cei
                    qp, qm = Ewi[j] * cw, Ew[j] * cwi
                if abs(x) > 0.25:
                    sx = (ep - em) / (2 * I) / x
                else:
                    sx = x.sinc()
                env = sx ** m
                cs = (qp + qm) / 2 if kind == 0 else (qp - qm) / (2 * I)
                tot += a * env * cs
        return tot

    # ------------------------------------------------------------------ float evaluation on grids
    def eval_float(self, starts: list, step: float, K: int, reach: float = 120.0):
        """H_loc at s + j step (j < K) for each exact arb s in `starts`, in IEEE double: (values
        len(starts) x K, error bounds per start). Atoms within `reach` of a block are evaluated; the
        others and every mirror term are bounded by |e(s)| <= (m / (b |s|))^m."""
        import numpy as np
        from flint import acb, arb, ctx
        u = _U
        nb = len(starts)
        vals = np.zeros((nb, K))
        errs = np.zeros(nb)
        if not self.atoms:
            return vals, errs
        tau = np.array([x[0] for x in self.atoms])
        wv = np.array([x[1] for x in self.atoms])
        kind = np.array([x[2] for x in self.atoms])
        av = np.array([x[3] for x in self.atoms])
        ws = sorted(set(wv.tolist()))
        widx = np.array([ws.index(w) for w in wv])
        mb = self.m / self.b
        x0 = 0.05
        old = ctx.prec
        ctx.prec = 128
        try:
            I = acb(0, 1)
            nu = arb(self.b) / self.m
            # per-atom constant phases (rounded; error <= 2^-52 each, generous)
            if "_fconst" not in self.__dict__:
                cw = np.empty(len(tau), complex)
                ce = np.empty(len(tau), complex)
                for k, (t_, w_) in enumerate(zip(tau, wv)):
                    z1 = (-I * arb(w_) * arb(t_)).exp()
                    z2 = (-I * nu * arb(t_)).exp()
                    cw[k] = complex(float(z1.real.mid()), float(z1.imag.mid()))
                    ce[k] = complex(float(z2.real.mid()), float(z2.imag.mid()))
                self._fconst = (cw, ce)
            cw, ce = self._fconst
            eps_c = 2.0 ** -51
            freqs = [arb(w) for w in ws] + [nu]
            st = arb(step)
            steps = []
            for f in freqs:
                z = (I * f * st).exp()
                steps.append(complex(float(z.real.mid()), float(z.imag.mid())))
            eps_w = 2.0 ** -51
            for bi, s in enumerate(starts):
                s_lo = float(s.mid())
                s_hi = s_lo + (K - 1) * step
                near = (tau >= s_lo - reach) & (tau <= s_hi + reach)
                far = ~near
                # far atoms and all mirror terms: bounded, not evaluated
                dfar = np.maximum(np.maximum(s_lo - tau[far], tau[far] - s_hi), 1e-300)
                if far.any() and np.min(dfar) < mb:
                    raise ValueError("reach too small")
                tail = float(np.sum(np.abs(av[far]) * (mb / dfar) ** self.m))
                tail += float(np.sum(np.abs(av) * (mb / (s_lo + tau)) ** self.m)) if s_lo > 0 else float(np.sum(np.abs(av)))
                if not near.any():
                    errs[bi] = tail * (1 + 1e-6)
                    continue
                # grid phases for the carriers in use and for nu, by K - 1 products from Arb starts
                ph = []
                eps0 = 0.0
                for f, sf in zip(freqs, steps):
                    z0 = (I * f * s).exp()
                    zf = complex(float(z0.real.mid()), float(z0.imag.mid()))
                    eps0 = max(eps0, float(abs(z0 - acb(zf)).upper()) + 2.0 ** -52)
                    arr = np.empty(K, complex)
                    arr[0] = zf
                    arr[1:] = sf
                    ph.append(np.cumprod(arr))
                xk = K * (eps_w + 3 * u)
                eps_ph = (1 + 2 * xk) * (eps0 + K * (3 * u * (1 + eps_w) + eps_w))
                Pe = ph[-1]
                tj = s_lo + step * np.arange(K)               # float grid (error <= 4u |t|)
                idx = np.nonzero(near)[0]
                acc = np.zeros(K)
                for k in idx:
                    xs = (self.b / self.m) * (tj - tau[k])
                    sinx = (Pe * ce[k]).imag                  # sin of the exact x
                    small = np.abs(xs) < x0
                    with np.errstate(divide="ignore", invalid="ignore"):
                        sc = np.where(small, 1 - xs * xs / 6 + xs ** 4 / 120, sinx / xs)
                    env = sc.copy()
                    for _ in range(self.m - 1):              # repeated products (no libm pow)
                        env *= sc
                    q = ph[widx[k]] * cw[k]
                    acc += av[k] * env * (q.real if kind[k] == 0 else q.imag)
                vals[bi] = acc
                # error bound
                tmax = abs(s_hi) + reach
                eps_x = (self.b / self.m) * 6 * u * (tmax + reach)     # |x_float - x|
                eps_ph_t = eps_ph + eps_c + 3 * u                       # a product of two phases
                e_big = eps_ph_t / x0 + eps_x / (x0 * (x0 - eps_x)) + 3 * u
                e_small = x0 ** 6 / 5040 + eps_x + 6 * u
                e_sinc = max(e_big, e_small)
                e_env = self.m * (e_sinc + 2 * u) * (1 + e_sinc + 2 * u) ** self.m
                per_atom = e_env + eps_ph_t + 4 * u
                sa = float(np.sum(np.abs(av[idx])))
                errs[bi] = (sa * per_atom * 1.01 + _gam(len(idx) + 1) * sa * 1.01 + tail) * (1 + 1e-6)
        finally:
            ctx.prec = old
        return vals, errs


def fit_local_correction(G_of_t, Om_of_t, L: float, T: float, T_end: float, beta: float, b: float = 1.0,
                         m: int = 8, W: float = 4.0, dw: float = 0.5, dtau: float = 2.0, window: float = 100.0,
                         commit: float = 50.0, step: float = 0.04, amax: float = 8.0, mu: float = 1e-3,
                         verbose: bool = False) -> LocalCorrection:
    """Receding-horizon LP (numerical) for amplitudes a_k making G - H_loc <= Omega - beta on [T, T_end]
    with the largest margin, window by window. G_of_t and Om_of_t give float values of G = P_L - H and
    Omega on arrays of t. Atoms: centres every dtau, carriers 2L + b + j dw (j = 0..W/dw), cos and sin;
    each window optimises the atoms centred in [a, a + window) and keeps those in [a, a + commit)."""
    import numpy as np
    import scipy.sparse as sp
    from scipy.optimize import linprog
    mb = m / b
    reach = mb * 10 ** (8.0 / m)                      # (m/(b d))^m < 1e-8 beyond
    omegas = 2 * L + b + dw * np.arange(int(round(W / dw)) + 1)
    # carriers as exact binary floats with w - b >= 2L
    omegas = np.array([math.ldexp(math.ceil(math.ldexp(w, 20)), -20) for w in omegas])
    ts = np.arange(T - 2 * reach, T_end + 2 * reach, step)
    slack0 = Om_of_t(ts) - beta - G_of_t(ts)

    def env(s):
        x = s / mb
        with np.errstate(invalid="ignore", divide="ignore"):
            u = np.where(np.abs(x) < 1e-8, 1.0, np.sin(x) / x)
        return u ** m

    def cols(taus, t):
        out = []
        for tau in taus:
            e1, e2 = env(t - tau), env(t + tau)
            for w in omegas:
                out.append(e1 * np.cos(w * (t - tau)) + e2 * np.cos(w * (t + tau)))
                out.append(e1 * np.sin(w * (t - tau)) - e2 * np.sin(w * (t + tau)))
        return np.array(out).T

    Hloc = np.zeros_like(ts)
    atoms = []
    a = T - reach / 2
    D = 15.0
    while a < T_end + reach / 2:
        taus = np.arange(a, a + window, dtau)
        core = (ts >= max(T, a - reach / 2)) & (ts < a + window - D)
        side = (~core) & (ts >= T) & (ts >= a - reach) & (ts < a + window + reach)
        pts = core | side
        tc, r, isc = ts[pts], slack0[pts] + Hloc[pts], core[pts]
        C = cols(taus, tc)
        n = C.shape[1]
        rhs = np.where(isc, r, r - np.minimum(r, 0))
        A = sp.hstack([sp.csr_matrix(-C), sp.csr_matrix(C), sp.csr_matrix(-isc.astype(float)[:, None])]).tocsr()
        c = np.concatenate([np.full(2 * n, mu), [1.0]])
        res = linprog(c, A_ub=A, b_ub=rhs, bounds=[(0, amax)] * (2 * n) + [(None, None)], method="highs")
        if res.status != 0:
            raise RuntimeError(f"LP failed at window {a}: {res.message}")
        av = res.x[:n] - res.x[n:2 * n]
        keep = np.repeat(taus < a + commit, 2 * len(omegas))
        av = np.where(keep & (np.abs(av) > 1e-9), av, 0.0)
        av = np.array([math.ldexp(round(math.ldexp(x, 40)), -40) for x in av])   # short binary amplitudes
        near = (ts >= a - reach) & (ts < a + window + reach)
        Hloc[near] += cols(taus, ts[near]) @ av
        for j in np.nonzero(av)[0]:
            k, q = divmod(j, 2 * len(omegas))
            atoms.append((float(taus[k]), float(omegas[q // 2]), int(q % 2), float(av[j])))
        if verbose:
            seg = (ts >= max(T, a - reach / 2)) & (ts < a + commit)
            print(f"  window [{a:.0f}, {a + window:.0f}): s* = {res.x[-1]:+.4f}, committed {int(np.count_nonzero(av))} "
                  f"atoms, min slack so far {(slack0 + Hloc)[seg].min():+.4f}", flush=True)
        a += commit
    return LocalCorrection(L=L, b=b, m=m, atoms=atoms)
