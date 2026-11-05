r"""Heat flow of the Riemann xi function and the de Bruijn–Newman constant.

Definitions (de Bruijn 1950; Newman 1976; notation of Rodgers–Tao and Polymath 15)
----------------------------------------------------------------------------------
    Phi(u) = sum_{n>=1} (2 pi^2 n^4 e^{9u} - 3 pi n^2 e^{5u}) exp(-pi n^2 e^{4u}),
    H_t(z) = int_0^inf e^{t u^2} Phi(u) cos(z u) du.

One has H_0(z) = (1/8) xi(1/2 + i z/2) with xi(s) = (1/2) s (s-1) pi^{-s/2} Gamma(s/2) zeta(s),
so the zeros of H_0 are z = +-2 gamma_n (gamma_n the ordinates of the nontrivial
zeros).  H_t solves the *backward* heat equation d_t H_t = -d_zz H_t.  The de
Bruijn–Newman constant Lambda is the infimum of the t for which H_t has only real
zeros (de Bruijn: Lambda <= 1/2; Newman: Lambda > -infinity).  RH <=> Lambda <= 0.
Rodgers–Tao (2018, arXiv:1801.05914; Acta Math. 225 (2020)) proved Lambda >= 0, so
RH <=> Lambda = 0.  Polymath 15 (Res. Math. Sci. 6 (2019) 31) proved Lambda <= 0.22
and Platt–Trudgian (Bull. LMS 53 (2021)) extended this to Lambda <= 0.2.

Zero dynamics
-------------
Whenever the zeros x_j(t) of H_t are real and simple (in particular for every
t > Lambda) they obey the system of ODEs (Csordas–Smith–Varga, Constr. Approx. 10
(1994) 107–129, Sect. 2; Rodgers–Tao Sect. 2 and Tao's blog "Heat flow and zeroes
of polynomials"):

    dx_j/dt = sum_{k != j} 2 / (x_j - x_k),                                 (ODE)

where the sum runs over *all* zeros, i.e. j, k in Z \ {0} with x_{-k} = -x_k, and
is summed symmetrically (the pair k, -k gives 4 x_j / (x_j^2 - x_k^2), so the sum
converges absolutely).  Derivation: differentiate H_t(x_j(t)) = 0 and use the
backward heat equation, H''/H' = 2 sum_{k != j} 1/(x_j - x_k) at a simple zero.
Consequences: for t > 0 (forward) the zeros repel and the spacings become more
regular (Rodgers–Tao: they approach an equally spaced configuration locally); for
t < 0 (backward) close pairs attract, collide at some t* < 0 and become a complex
conjugate pair.  Since H_t has only real *simple* zeros for every t > Lambda, a
collision at time t* forces Lambda >= t*.  For an isolated pair the ODE gives
d(Delta^2)/dt = 8 exactly, i.e. the collision time is t* = -Delta(0)^2/8.

In this module the first N zeros are evolved explicitly; further zeros from the
rigorous dataset can be kept frozen; everything beyond the explicit zeros is
replaced by the smooth Riemann–von Mangoldt density in the variable x = 2 gamma,
rho(x) dx = (1/(4 pi)) log(x/(4 pi)) dx (half of (1/2pi) log(gamma/2pi) because
dx = 2 dgamma), whose field is known in closed form (`tail_field`).  The tail is
let drift at the mean zero velocity -pi/4 (the leading-order drift of all zeros,
d/dx log|xi(1/2+ix/2)| -> -pi/8, see `velocity_field` docstring).

Lehmer pairs (Csordas–Smith–Varga 1994)
---------------------------------------
For consecutive positive zeros x_k < x_{k+1} of H_0 put Delta_k = x_{k+1} - x_k and
    g_k = sum_{j != k, k+1} [ 1/(x_j - x_k)^2 + 1/(x_j - x_{k+1})^2 ]
(sum over all other zeros, positive and negative).  CSV call {x_k, x_{k+1}} a
*Lehmer pair* if Delta_k^2 g_k < 4/5, and prove (their Theorem 1) that every Lehmer
pair yields an explicit lower bound Lambda >= lambda_k(Delta_k, g_k) of size
-Delta_k^2/8 (1 + O(Delta_k^2 g_k)).  We do not reproduce their exact constant
here; `lehmer_pair_criterion` returns the criterion together with the ODE-based
estimates derived in its docstring, the rigorous version being the CSV theorem.
Historical lower bounds obtained this way (values quoted from memory, approximate):
Csordas–Smith–Varga 1994: Lambda > -4.379e-6; Csordas–Odlyzko–Smith–Varga (ETNA 1,
1993): Lambda > -5.895e-9 (a Lehmer pair near the 10^20-th zero found by Odlyzko);
Odlyzko 2000: Lambda > -2.7e-9; Saouter–Gourdon–Demichel (Math. Comp. 80, 2011):
Lambda > -1.15e-11.  The classical Lehmer pair (Lehmer 1956) is gamma_6709 =
7005.0629, gamma_6710 = 7005.1006; its two-body collision time is about -7.1e-4,
so it is *not* the pair behind the bounds above, which come from much closer pairs
at great heights.

Nothing here is a proof of anything about Lambda: the ODE integration is floating
point and the tail is a smooth approximation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import mpmath as mp
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from scipy.special import spence

LOG_4PI = math.log(4 * math.pi)
MEAN_DRIFT = -math.pi / 4          # leading-order velocity of every zero (x-variable)
LEHMER_THRESHOLD = 0.8             # 4/5 in the CSV definition


# ----------------------------------------------------------------------------
# 1. H_t(z) by quadrature, xi(s), consistency checks
# ----------------------------------------------------------------------------
def Phi(u):
    """Phi(u) = sum_n (2 pi^2 n^4 e^{9u} - 3 pi n^2 e^{5u}) exp(-pi n^2 e^{4u}) (mpmath).

    Uses the current mpmath working precision; the series is truncated when the
    n-th term is below 10^(-dps-10) relative to the first.
    """
    u = mp.mpf(u)
    e4, e5, e9 = mp.exp(4 * u), mp.exp(5 * u), mp.exp(9 * u)
    eps = mp.mpf(10) ** (-(mp.mp.dps + 10))
    s = mp.mpf(0)
    first = None
    for n in range(1, 10000):
        n2 = n * n
        w = mp.exp(-mp.pi * n2 * e4)
        term = (2 * mp.pi ** 2 * n2 * n2 * e9 - 3 * mp.pi * n2 * e5) * w
        s += term
        mag = 2 * mp.pi ** 2 * n2 * n2 * e9 * w
        if first is None:
            first = mag
        if mag < eps * (first if first != 0 else 1):
            break
    return s


def H_t(t, z, dps=30):
    """H_t(z) = int_0^inf e^{t u^2} Phi(u) cos(z u) du by mpmath tanh-sinh quadrature.

    Intended for moderate |z| (say |z| < 200) and |t| <= 1.  Phi(u) decays like
    exp(-pi e^{4u}), so the integral is truncated at u_max where the integrand is
    below 10^-(dps+15); the interval is split into pieces shorter than the period
    of cos(z u) to keep the quadrature accurate.
    """
    with mp.workdps(dps):
        t = mp.mpf(t)
        z = mp.mpf(z)
        umax = 0.25 * math.log(((dps + 15) * math.log(10) + 25.0) / math.pi) + 0.1
        if t > 0:
            umax += float(t) * 0.1
        step = min(0.1, 2 * math.pi / (abs(float(z)) + 1.0))
        npts = int(math.ceil(umax / step)) + 1
        pts = [mp.mpf(umax) * k / (npts - 1) for k in range(npts)]
        f = lambda u: mp.exp(t * u * u) * Phi(u) * mp.cos(z * u)
        return mp.quad(f, pts)


def xi(s, dps=30):
    """Riemann xi(s) = (1/2) s (s-1) pi^{-s/2} Gamma(s/2) zeta(s) (mpmath mpc)."""
    with mp.workdps(dps):
        s = mp.mpc(s)
        return mp.mpf(1) / 2 * s * (s - 1) * mp.pi ** (-s / 2) * mp.gamma(s / 2) * mp.zeta(s)


def H0_from_xi(z, dps=30):
    """H_0(z) = xi(1/2 + i z/2) / 8 evaluated through mpmath's zeta (real part)."""
    with mp.workdps(dps):
        return mp.re(xi(mp.mpf(1) / 2 + mp.mpc(0, 1) * mp.mpf(z) / 2, dps)) / 8


def check_H0_vs_xi(zs=(0.0, 10.0, 30.0, 57.0, 90.0), dps=30):
    """Compare the quadrature H_0(z) with xi(1/2+iz/2)/8 at the given points.

    Returns a list of dicts with both values and the difference relative to H_0(0).
    """
    with mp.workdps(dps):
        h00 = H_t(0, 0, dps)
        out = []
        for z in zs:
            hq = H_t(0, z, dps)
            hx = H0_from_xi(z, dps)
            out.append({"z": float(z), "H0_quad": float(hq), "H0_xi": float(hx),
                        "rel_diff": float(abs(hq - hx) / abs(h00))})
        return out


# ----------------------------------------------------------------------------
# 2. Smooth density of zeros in the variable x = 2 gamma, tail field
# ----------------------------------------------------------------------------
def smooth_density_x(x):
    """Riemann–von Mangoldt density of zeros of H_0 per unit x (x = 2 gamma)."""
    x = np.asarray(x, dtype=float)
    return np.log(x / (4 * np.pi)) / (4 * np.pi)


def smooth_count_x(X):
    """Smooth number of positive zeros of H_0 below X (= N(X/2) without S(T))."""
    T = X / 2.0
    return T / (2 * np.pi) * np.log(T / (2 * np.pi * np.e)) + 7.0 / 8.0


def tail_start(n_explicit):
    """Point X where the smooth tail begins when the first `n_explicit` zeros are
    explicit: smooth_count_x(X) = n_explicit + 1/2 (midpoint rule)."""
    n = float(n_explicit)
    lo, hi = 30.0, 1e3
    while smooth_count_x(hi) < n + 0.5:
        hi *= 4
    return brentq(lambda X: smooth_count_x(X) - (n + 0.5), lo, hi, xtol=1e-10)


def _chi2(r):
    """Legendre chi function chi_2(r) = sum_{m>=0} r^{2m+1}/(2m+1)^2 = (Li2(r)-Li2(-r))/2."""
    return 0.5 * (spence(1.0 - r) - spence(1.0 + r))


def _log_ratio(r):
    """log((1+r)/(1-r)) accurately for 0 <= r < 1."""
    return np.log1p(r) - np.log1p(-r)


def tail_field(x, X):
    r"""Field at 0 < x < X of the smooth tail of zeros beyond X (and their mirrors).

    T(x; X) = int_X^inf rho(y) [2/(x-y) + 2/(x+y)] dy,  rho(y) = log(y/(4 pi))/(4 pi)
           = -(1/(2 pi)) [ log(X/(4 pi)) log((1+r)/(1-r)) + 2 chi_2(r) ],  r = x/X.

    (Substitute y = X/v; the log(X/4pi) part integrates elementarily and the
    log(v) part gives -2 chi_2(r) via int_0^1 2r log v/(1-r^2 v^2) dv.)
    """
    x = np.asarray(x, dtype=float)
    r = x / X
    L = math.log(X / (4 * math.pi))
    return -(L * _log_ratio(r) + 2.0 * _chi2(r)) / (2 * np.pi)


def tail_field_derivative(x, X):
    """d/dx of `tail_field` (used in the Jacobian and in g_k / S_k sums)."""
    x = np.asarray(x, dtype=float)
    r = x / X
    L = math.log(X / (4 * math.pi))
    return -(2.0 * L / (1.0 - r * r) + _log_ratio(r) / r) / (2 * np.pi * X)


def _tail_X(X_tail, t, drift_tail):
    return X_tail + (MEAN_DRIFT * t if drift_tail else 0.0)


# ----------------------------------------------------------------------------
# 3. Velocity field and Jacobian of the ODE system
# ----------------------------------------------------------------------------
def _chunk_size(n_total, budget=2.0e7):
    return max(1, int(budget // max(n_total, 1)))


def velocity_field(x, t=0.0, x_frozen=None, X_tail=None, drift_tail=True):
    r"""Right-hand side of (ODE) for the active zeros `x` (positive, increasing).

    v_j = sum_{k != j} 2/(x_j - x_k) + sum_k 2/(x_j + x_k) + T(x_j; X_tail(t)),
    where k runs over all explicit positive zeros (active and `x_frozen`); the
    k = j term of the second sum, 1/x_j, is the mirror zero -x_j.  T is the
    smooth-tail field (`tail_field`), with X_tail(t) = X_tail - (pi/4) t when
    drift_tail is True.  Check: with 10^5 explicit zeros the median velocity of
    the first 50 000 zeros is -0.7857 (-pi/4 = -0.7854): every zero drifts to the
    left at the leading-order speed pi/4, because H_0(x) ~ A(x) cos(theta(x)) with
    A'/A -> -pi/8 and dx/dt = 2 A'/A + theta''/theta' for the backward heat flow.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    allpos = x if x_frozen is None else np.concatenate([x, np.asarray(x_frozen, dtype=float)])
    v = np.empty(n)
    cs = _chunk_size(allpos.size)
    for i0 in range(0, n, cs):
        xi_ = x[i0:i0 + cs, None]
        d = xi_ - allpos[None, :]
        idx = np.arange(d.shape[0])
        d[idx, i0 + idx] = np.inf                      # exclude the self term
        v[i0:i0 + cs] = (2.0 / d).sum(axis=1) + (2.0 / (xi_ + allpos[None, :])).sum(axis=1)
    if X_tail is not None:
        v += tail_field(x, _tail_X(X_tail, t, drift_tail))
    return v


def velocity_jacobian(x, t=0.0, x_frozen=None, X_tail=None, drift_tail=True):
    """Dense Jacobian d v_j / d x_k of `velocity_field` with respect to the active zeros."""
    x = np.asarray(x, dtype=float)
    n = x.size
    d = x[:, None] - x[None, :]
    np.fill_diagonal(d, np.inf)
    p = x[:, None] + x[None, :]
    J = 2.0 / d ** 2 - 2.0 / p ** 2
    diag = -(2.0 / d ** 2).sum(axis=1) - (2.0 / p ** 2).sum(axis=1) + 2.0 / (2 * x) ** 2 - 1.0 / x ** 2
    # (the k=j entry of the p-sum, 2/(2x)^2, was included in the sum; the true
    #  derivative of the mirror term 1/x_j is -1/x_j^2)
    if x_frozen is not None:
        xf = np.asarray(x_frozen, dtype=float)
        cs = _chunk_size(xf.size)
        for i0 in range(0, n, cs):
            xi_ = x[i0:i0 + cs, None]
            diag[i0:i0 + cs] -= (2.0 / (xi_ - xf[None, :]) ** 2).sum(axis=1)
            diag[i0:i0 + cs] -= (2.0 / (xi_ + xf[None, :]) ** 2).sum(axis=1)
    if X_tail is not None:
        diag += tail_field_derivative(x, _tail_X(X_tail, t, drift_tail))
    np.fill_diagonal(J, diag)
    return J


# ----------------------------------------------------------------------------
# 4. Integration of the zero dynamics
# ----------------------------------------------------------------------------
@dataclass
class FlowResult:
    """Trajectories x_j(t) of the active zeros (shape (n_times, n_active))."""
    t: np.ndarray
    x: np.ndarray
    X_tail: float
    status: int
    message: str
    t_events: list = field(default_factory=list)
    n_rhs: int = 0
    sol: object = None          # dense output (callable) when requested

    def spacings(self, i):
        return np.diff(self.x[i])

    def min_spacing(self, i):
        return float(np.min(np.diff(self.x[i])))


def zero_dynamics(x0, t_span, t_eval=None, x_frozen=None, X_tail="auto", drift_tail=True,
                  method="DOP853", rtol=1e-10, atol=1e-12, min_spacing_stop=1e-6,
                  dense_output=False):
    """Integrate the zero ODE (see module docstring) for the active zeros `x0`.

    Parameters
    ----------
    x0 : positive zeros of H_0 (x = 2 gamma), increasing; these are evolved.
    t_span : (t0, t1); t1 < t0 integrates backward in time.
    x_frozen : further explicit zeros held fixed (their field is included exactly).
    X_tail : start of the smooth tail; "auto" assumes that x0 together with
        x_frozen are the first len(x0)+len(x_frozen) zeros and uses `tail_start`.
        None disables the tail (not recommended: the far field is O(1)).
    method : any solve_ivp method; "Radau"/"LSODA"/"BDF" receive the analytic
        Jacobian.  DOP853 (explicit, 8th order) is the default because the system
        is non-stiff away from collisions and the dense Jacobian costs O(N^3).
    min_spacing_stop : terminal event when a spacing of active zeros falls below
        this (approaching collision, backward flow).
    """
    x0 = np.asarray(x0, dtype=float)
    if np.any(np.diff(x0) <= 0):
        raise ValueError("x0 must be strictly increasing")
    xf = None if x_frozen is None else np.asarray(x_frozen, dtype=float)
    if isinstance(X_tail, str) and X_tail == "auto":
        X_tail = tail_start(x0.size + (0 if xf is None else xf.size))

    counter = {"n": 0}

    def rhs(t, y):
        counter["n"] += 1
        return velocity_field(y, t, xf, X_tail, drift_tail)

    def jac(t, y):
        return velocity_jacobian(y, t, xf, X_tail, drift_tail)

    def ev_collision(t, y):
        return np.min(np.diff(y)) - min_spacing_stop
    ev_collision.terminal = True
    ev_collision.direction = -1

    kw = dict(method=method, rtol=rtol, atol=atol, t_eval=t_eval, events=[ev_collision],
              dense_output=dense_output)
    if method in ("Radau", "BDF", "LSODA"):
        kw["jac"] = jac
    sol = solve_ivp(rhs, t_span, x0, **kw)
    return FlowResult(t=sol.t, x=sol.y.T.copy(), X_tail=float(X_tail) if X_tail is not None else float("nan"),
                      status=sol.status, message=sol.message,
                      t_events=[te.tolist() for te in sol.t_events], n_rhs=counter["n"],
                      sol=sol.sol if dense_output else None)


# ----------------------------------------------------------------------------
# 5. Collision time of a close pair (pair in centre/spacing^2 coordinates)
# ----------------------------------------------------------------------------
def _pair_background_sums(c, s, allo, X, include_tail=True):
    """Sums over the other positive zeros `allo` (and all mirrors) needed for the
    pair (c, s): returns (cdot_part, S) where

      cdot_part = sum_y [1/(x_k - y) + 1/(x_{k+1} - y)] = sum_y 2(c-y)/((c-y)^2 - s/4),
      S         = sum_y 1/((x_{k+1} - y)(x_k - y))     = sum_y 1/((c-y)^2 - s/4),

    y running over allo, -allo, and the mirror pair {-x_k, -x_{k+1}} (whose two
    terms combine to c/(c^2-s/4) + 1/c and 1/(2(c^2-s/4))), plus the smooth tail
    (T(c) for the field, -T'(c)/2 for S).  All expressions are smooth through s=0.
    """
    dm = c - allo
    dp = c + allo
    qm = dm * dm - s / 4.0
    qp = dp * dp - s / 4.0
    cdot = np.sum(2.0 * dm / qm) + np.sum(2.0 * dp / qp) + c / (c * c - s / 4.0) + 1.0 / c
    S = np.sum(1.0 / qm) + np.sum(1.0 / qp) + 1.0 / (2.0 * (c * c - s / 4.0))
    if include_tail and X is not None:
        cdot += float(tail_field(c, X))
        S += -0.5 * float(tail_field_derivative(c, X))
    return cdot, S


@dataclass
class CollisionResult:
    k: int                      # 1-based index of the lower zero of the pair
    gamma_pair: tuple
    Delta0: float               # initial spacing in x = 2 gamma
    t_star: float | None        # collision time from the ODE (None if not reached)
    t_two_body: float           # -Delta0^2/8
    t_frozen_background: float  # log(1 - Delta0^2 S/2)/(4 S), S from the t=0 configuration
    S0: float
    g_k: float
    Delta2_g: float
    is_lehmer_pair: bool
    t: np.ndarray               # sample times (from 0 to t_star)
    x_pair: np.ndarray          # (n_times, 2) trajectories of the pair
    x_window: np.ndarray        # (n_times, n_active) trajectories of all active zeros
    window: tuple               # (lo, hi) 0-based slice of active zeros in x_all
    status: int
    message: str
    n_rhs: int = 0


def estimate_collision_time(x_all, k, n_window=100, X_tail="auto", t_max=None,
                            method="DOP853", rtol=1e-11, atol=1e-14, n_samples=400):
    """Integrate the zero ODE backward in time until the pair (x_k, x_{k+1}) collides.

    `x_all` are the explicit positive zeros (x = 2 gamma, the first len(x_all)
    zeros); `k` is the 0-based index of the lower member of the pair.  The pair is
    represented by its centre c and squared spacing s = Delta^2, in which the ODE
    is smooth through the collision (d s/dt = 8 - 4 s S, d c/dt = regular), so the
    event s = 0 is a regular root.  The `n_window` zeros on each side of the pair
    are evolved; all other explicit zeros are frozen (their drift over the
    collision time scale |t*| ~ Delta^2/8 is negligible) and the smooth tail
    starts at X_tail.  Returns a CollisionResult with the collision time t*
    (Lambda >= t* if the computation were exact), the two-body estimate
    -Delta^2/8, and the frozen-background estimate obtained by keeping S constant:
    s(t) = 2/S + (s0 - 2/S) e^{-4 S t}  =>  t* = log(1 - s0 S/2) / (4 S).
    """
    x_all = np.asarray(x_all, dtype=float)
    N = x_all.size
    if not (0 <= k < N - 1):
        raise ValueError("k out of range")
    if isinstance(X_tail, str) and X_tail == "auto":
        X_tail = tail_start(N)
    lo = max(0, k - n_window)
    hi = min(N, k + 2 + n_window)
    act_idx = np.r_[np.arange(lo, k), np.arange(k + 2, hi)]      # active others
    frozen = np.concatenate([x_all[:lo], x_all[hi:]])
    xk, xk1 = x_all[k], x_all[k + 1]
    c0 = 0.5 * (xk + xk1)
    s0 = (xk1 - xk) ** 2
    D0 = xk1 - xk
    y0 = np.concatenate([x_all[act_idx], [c0, s0]])
    n_act = act_idx.size
    counter = {"n": 0}

    def unpack(y):
        return y[:n_act], y[n_act], y[n_act + 1]

    def rhs(t, y):
        counter["n"] += 1
        xa, c, s = unpack(y)
        X = _tail_X(X_tail, t, True) if X_tail is not None else None
        allo = np.concatenate([xa, frozen])
        # active others: field of all other positive zeros, mirrors, pair, mirror pair, tail
        v = velocity_field(xa, t, frozen, X_tail, True) if X_tail is not None else velocity_field(xa, t, frozen, None)
        dm = xa - c
        dp = xa + c
        v += 4.0 * dm / (dm * dm - s / 4.0) + 4.0 * dp / (dp * dp - s / 4.0)
        cdot, S = _pair_background_sums(c, s, allo, X)
        sdot = 8.0 - 4.0 * s * S
        return np.concatenate([v, [cdot, sdot]])

    def ev_collide(t, y):
        return y[n_act + 1]
    ev_collide.terminal = True
    ev_collide.direction = -1

    # background quantities at t = 0
    allo0 = np.concatenate([x_all[act_idx], frozen])
    _, S0 = _pair_background_sums(c0, s0, allo0, X_tail)
    g_k = lehmer_g(x_all, k, X_tail)
    t_two = -s0 / 8.0
    t_frozen = math.log(1.0 - s0 * S0 / 2.0) / (4.0 * S0) if s0 * S0 < 2 else float("nan")
    if t_max is None:
        t_max = 1.5 * t_frozen if np.isfinite(t_frozen) else 3.0 * t_two
    sol = solve_ivp(rhs, (0.0, t_max), y0, method=method, rtol=rtol, atol=atol,
                    events=[ev_collide], dense_output=True)
    t_star = float(sol.t_events[0][0]) if sol.t_events[0].size else None
    t_end = t_star if t_star is not None else sol.t[-1]
    ts = np.linspace(0.0, t_end, n_samples)
    Y = sol.sol(ts).T
    c = Y[:, n_act]
    s = np.maximum(Y[:, n_act + 1], 0.0)
    x_pair = np.column_stack([c - 0.5 * np.sqrt(s), c + 0.5 * np.sqrt(s)])
    # assemble full window trajectories in increasing order
    xw = np.empty((ts.size, n_act + 2))
    xw[:, :k - lo] = Y[:, :k - lo]
    xw[:, k - lo:k - lo + 2] = x_pair
    xw[:, k - lo + 2:] = Y[:, k - lo:n_act]
    return CollisionResult(k=k + 1, gamma_pair=(xk / 2, xk1 / 2), Delta0=D0, t_star=t_star,
                           t_two_body=t_two, t_frozen_background=t_frozen, S0=float(S0),
                           g_k=float(g_k), Delta2_g=float(D0 * D0 * g_k),
                           is_lehmer_pair=bool(D0 * D0 * g_k < LEHMER_THRESHOLD),
                           t=ts, x_pair=x_pair, x_window=xw, window=(lo, hi),
                           status=sol.status, message=sol.message, n_rhs=counter["n"])


# ----------------------------------------------------------------------------
# 6. Lehmer pairs (Csordas–Smith–Varga 1994)
# ----------------------------------------------------------------------------
def lehmer_g(x_all, k, X_tail="auto"):
    """g_k = sum_{j != k,k+1} [1/(x_j-x_k)^2 + 1/(x_j-x_{k+1})^2] over all zeros
    (explicit positive zeros, their mirrors, and the smooth tail)."""
    x_all = np.asarray(x_all, dtype=float)
    if isinstance(X_tail, str) and X_tail == "auto":
        X_tail = tail_start(x_all.size)
    xk, xk1 = x_all[k], x_all[k + 1]
    others = np.delete(x_all, [k, k + 1])
    g = np.sum(1.0 / (others - xk) ** 2 + 1.0 / (others - xk1) ** 2)
    g += np.sum(1.0 / (x_all + xk) ** 2 + 1.0 / (x_all + xk1) ** 2)       # mirrors (incl. mirror pair)
    if X_tail is not None:
        g += -0.5 * float(tail_field_derivative(xk, X_tail) + tail_field_derivative(xk1, X_tail))
    return float(g)


def lehmer_pair_criterion(x_all, k, X_tail="auto"):
    r"""Lehmer-pair test of Csordas–Smith–Varga for the pair (x_k, x_{k+1}), k 0-based.

    Returns a dict with Delta_k, g_k, Delta_k^2 g_k, is_lehmer (Delta^2 g < 4/5) and
    several estimates of the collision time / lower bound for Lambda:

    * t_two_body = -Delta^2/8  (exact for an isolated pair: d(Delta^2)/dt = 8);
    * t_frozen_background = log(1 - Delta^2 S/2)/(4 S), where
      S = sum_{j != k,k+1} 1/((x_{k+1}-x_j)(x_k-x_j)) is kept at its t=0 value
      (d(Delta^2)/dt = 8 - 4 Delta^2 S exactly, see `estimate_collision_time`);
    * lambda_heuristic = -(Delta^2/8) / (1 - Delta^2 g_k/4): by AM-GM, 0 <= S <= g_k/2,
      hence 8 - 2 Delta^2 g <= d(Delta^2)/dt <= 8 as long as the other zeros stay
      real; if g stays below its initial value the pair must collide no later than
      this time (going backward), so Lambda >= lambda_heuristic.  The Lehmer
      condition Delta^2 g < 4/5 makes the denominator >= 4/5, i.e. the crude bound
      lambda_crude = -(5/32) Delta^2.  Controlling g(t) rigorously is exactly what
      CSV's Theorem 1 does; our numbers are heuristics of the same order.
    """
    x_all = np.asarray(x_all, dtype=float)
    if isinstance(X_tail, str) and X_tail == "auto":
        X_tail = tail_start(x_all.size)
    xk, xk1 = x_all[k], x_all[k + 1]
    D = xk1 - xk
    c = 0.5 * (xk + xk1)
    others = np.delete(x_all, [k, k + 1])
    _, S = _pair_background_sums(c, D * D, others, X_tail)
    g = lehmer_g(x_all, k, X_tail)
    D2g = D * D * g
    t_two = -D * D / 8.0
    t_frozen = math.log(1.0 - D * D * S / 2.0) / (4.0 * S) if D * D * S < 2 else float("nan")
    lam_h = -(D * D / 8.0) / (1.0 - D2g / 4.0) if D2g < 4 else float("nan")
    return {"k": int(k + 1), "gamma_k": float(xk / 2), "gamma_k1": float(xk1 / 2),
            "Delta": float(D), "Delta2": float(D * D), "g_k": float(g), "S_k": float(S),
            "Delta2_g": float(D2g), "is_lehmer": bool(D2g < LEHMER_THRESHOLD),
            "t_two_body": float(t_two), "t_frozen_background": float(t_frozen),
            "lambda_heuristic": float(lam_h), "lambda_crude": float(-5.0 / 32.0 * D * D)}


def find_lehmer_pairs(x_all, n_best=10, X_tail="auto"):
    """Evaluate `lehmer_pair_criterion` for the `n_best` closest consecutive pairs
    (smallest Delta in x); sorted by Delta^2 g_k."""
    x_all = np.asarray(x_all, dtype=float)
    if isinstance(X_tail, str) and X_tail == "auto":
        X_tail = tail_start(x_all.size)
    d = np.diff(x_all)
    cand = np.argsort(d)[:n_best]
    out = [lehmer_pair_criterion(x_all, int(k), X_tail) for k in cand]
    out.sort(key=lambda r: r["Delta2_g"])
    return out


# ----------------------------------------------------------------------------
# 7. Spacing statistics along the flow
# ----------------------------------------------------------------------------
def spacing_statistics(x, lo=0, hi=None):
    """Raw and normalised (by the smooth density) nearest-neighbour spacings of the
    zeros x[lo:hi]; the normalised spacing has mean ~1."""
    x = np.asarray(x, dtype=float)[lo:hi]
    d = np.diff(x)
    mids = 0.5 * (x[1:] + x[:-1])
    s = d * smooth_density_x(mids)
    return {"n": int(d.size), "mean": float(d.mean()), "std": float(d.std()),
            "min": float(d.min()), "max": float(d.max()),
            "norm_mean": float(s.mean()), "norm_std": float(s.std()),
            "norm_min": float(s.min()), "norm_max": float(s.max()),
            "coefficient_of_variation": float(d.std() / d.mean())}


def flow_spacing_rigidity(gammas, n_active=2500, n_frozen=2500, n_stat=2000, t_final=0.2,
                          t_eval=(0.0, 0.05, 0.1, 0.2), method="DOP853", rtol=1e-9, atol=1e-11):
    """Propagate the first n_active zeros (x = 2 gamma) forward to t_final with
    n_frozen further explicit zeros and the smooth tail, and compare the spacing
    statistics of the first n_stat zeros at the sample times.

    Returns dict(times, stats (list of `spacing_statistics` dicts), ratio_std,
    ratio_norm_std, min_spacing, flow (FlowResult)).
    """
    g = np.asarray(gammas, dtype=float)
    x0 = 2.0 * g[:n_active]
    xf = 2.0 * g[n_active:n_active + n_frozen] if n_frozen > 0 else None
    t_eval = np.array(sorted(set([0.0] + list(t_eval) + [t_final])))
    t_eval = t_eval[t_eval <= t_final]
    res = zero_dynamics(x0, (0.0, t_final), t_eval=t_eval, x_frozen=xf, method=method,
                        rtol=rtol, atol=atol)
    stats = [spacing_statistics(res.x[i], 0, n_stat) for i in range(res.t.size)]
    return {"times": res.t.tolist(), "stats": stats,
            "ratio_std": stats[-1]["std"] / stats[0]["std"],
            "ratio_norm_std": stats[-1]["norm_std"] / stats[0]["norm_std"],
            "min_spacing": [s["min"] for s in stats], "flow": res}


# ----------------------------------------------------------------------------
# 8. Figures
# ----------------------------------------------------------------------------
def plot_flow(flow30, collision, path, n_show=30, magnify=10.0):
    """Three panels: (a) trajectories x_j(t)/2 of zeros 1..n_show with the
    displacement from t=0 magnified by `magnify` (the true motion over |t| <= 0.5
    is only a few tenths of a unit); (b) the true displacements; (c) the Lehmer
    pair of `collision` (a CollisionResult) from its collision time t* to 0."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(17, 5.4))
    n_show = min(n_show, flow30.x.shape[1])
    i0 = int(np.argmin(np.abs(flow30.t)))
    cmap = plt.get_cmap("viridis")
    for j in range(n_show):
        col = cmap(j / max(n_show - 1, 1))
        disp = (flow30.x[:, j] - flow30.x[i0, j]) / 2.0
        ax1.plot(flow30.t, flow30.x[i0, j] / 2.0 + magnify * disp, color=col, lw=1.2)
        ax2.plot(flow30.t, disp, color=col, lw=1.2)
    for ax in (ax1, ax2):
        ax.axvline(0.0, color="0.5", lw=0.8, ls="--")
        ax.set_xlabel("t")
        ax.grid(alpha=0.3)
    ax1.set_ylabel(r"$x_j(t)/2$   (zeros of $H_t$ in units of $\gamma$)")
    ax1.set_title(f"Zeros 1..{n_show} under the heat flow (displacement magnified x{magnify:g})\n"
                  r"$\dot x_j = \sum_{k\neq j} 2/(x_j-x_k)$: repulsion for $t>0$, attraction for $t<0$",
                  fontsize=10)
    ax2.set_ylabel(r"$(x_j(t) - x_j(0))/2$")
    ax2.set_title("True displacements: common drift $-\\pi t/8$ (in $\\gamma$ units)\n"
                  "plus repulsion of close pairs (e.g. $\\gamma_{27},\\gamma_{28}$ diverge for $t>0$)", fontsize=10)
    ax2.plot(flow30.t, -np.pi / 8 * flow30.t, "k:", lw=1.2, label=r"$-\pi t/8$")
    ax2.legend(fontsize=8, loc="upper right")

    lo = collision.window[0]
    k0 = collision.k - 1 - lo           # index of lower pair member inside the window
    c_gamma = 0.5 * (collision.gamma_pair[0] + collision.gamma_pair[1])
    for j in (k0, k0 + 1):
        ax3.plot(collision.t, collision.x_window[:, j] / 2.0, lw=2.0, color="C3",
                 label=f"$\\gamma_{{{lo + j + 1}}}(t)$")
    ax3.plot(collision.t, 0.5 * (collision.x_pair[:, 0] + collision.x_pair[:, 1]) / 2.0, "k--", lw=0.8,
             label="centre of the pair")
    if collision.t_star is not None:
        ax3.axvline(collision.t_star, color="C3", ls=":", lw=1)
        ax3.annotate(f"collision $t^*\\approx{collision.t_star:.4e}$\n(two-body $-\\Delta^2/8$ = {collision.t_two_body:.4e})",
                     xy=(collision.t_star, c_gamma), xytext=(25, 40),
                     textcoords="offset points", fontsize=9,
                     arrowprops=dict(arrowstyle="->", color="C3"))
    ax3.set_xlabel("t  (backward heat flow, t < 0)")
    ax3.set_ylabel(r"$x_j(t)/2$")
    d_next = (collision.x_window[0, k0 + 2] - collision.x_window[0, k0 + 1]) / 2.0 if k0 + 2 < collision.x_window.shape[1] else float("nan")
    d_prev = (collision.x_window[0, k0] - collision.x_window[0, k0 - 1]) / 2.0 if k0 >= 1 else float("nan")
    ax3.set_title(f"Lehmer pair $\\gamma_{{{collision.k}}}, \\gamma_{{{collision.k + 1}}}$ = "
                  f"{collision.gamma_pair[0]:.4f}, {collision.gamma_pair[1]:.4f}\n"
                  f"$\\Delta_x$ = {collision.Delta0:.5f}, $\\Delta^2 g_k$ = {collision.Delta2_g:.4f} "
                  f"(Lehmer pair: {collision.is_lehmer_pair}); neighbours at $-${d_prev:.2f}, +{d_next:.2f}",
                  fontsize=9.5)
    ax3.legend(loc="upper left", fontsize=8)
    ax3.grid(alpha=0.3)
    half = 0.75 * collision.Delta0 / 2.0
    ax3.set_ylim(c_gamma - half, c_gamma + half)
    if collision.t_star is not None:
        ax3.set_xlim(1.08 * collision.t_star, -0.02 * collision.t_star)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_spacing_histograms(rig, path, n_stat=2000):
    """Histogram of normalised spacings of the first n_stat zeros at t=0 and at the
    final time of a `flow_spacing_rigidity` result, plus the std ratio."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    res = rig["flow"]
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    bins = np.linspace(0, 3, 61)
    for i, color in ((0, "C0"), (res.t.size - 1, "C3")):
        x = res.x[i][:n_stat]
        s = np.diff(x) * smooth_density_x(0.5 * (x[1:] + x[:-1]))
        ax.hist(s, bins=bins, histtype="step", lw=1.8, color=color, density=True,
                label=f"t = {res.t[i]:.2f}:  std = {s.std():.4f},  min = {s.min():.3f}")
    ax.set_xlabel("normalised nearest-neighbour spacing of zeros of $H_t$ (first %d zeros)" % n_stat)
    ax.set_ylabel("density")
    ax.set_title("Forward heat flow makes the spacings more rigid: "
                 f"std ratio {rig['ratio_norm_std']:.3f}", fontsize=10)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path
