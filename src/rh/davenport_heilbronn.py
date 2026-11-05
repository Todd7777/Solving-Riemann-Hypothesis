"""The Davenport-Heilbronn function: a Riemann-type functional equation without RH.

Let chi be the Dirichlet character mod 5 with chi(2) = i (so chi(3) = -i,
chi(4) = -1; chi is odd and primitive) and put

    kappa = (sqrt(10 - 2 sqrt 5) - 2) / (sqrt 5 - 1) = 0.28407904384041229...,
    f(s)  = (1 - i kappa)/2 * L(s, chi) + (1 + i kappa)/2 * L(s, chibar).

f is an entire Dirichlet series with real coefficients c_n periodic mod 5,
(c_1, ..., c_5) = (1, kappa, -kappa, -1, 0), and it satisfies the functional
equation (G below has |G(1/2 + it)| = 1)

    f(s) = G(s) f(1 - s),   G(s) = (5/pi)^{1/2 - s} Gamma(1 - s/2) / Gamma((1 + s)/2).

The constant kappa is forced by this requirement: with eps(chi) the root
number of chi (eps = tau(chi)/(i sqrt 5), tau the Gauss sum) one needs
(1 + i kappa)/(1 - i kappa) = eps(chi), i.e. kappa = tan(arg eps / 2), which
is verified numerically in ``verify_kappa``.  Thus f has the two analytic
properties that ``explain'' RH for zeta -- a Dirichlet series and the
symmetric functional equation -- and even a real Hardy-type function
Z(t) = e^{i theta(t)} f(1/2 + it) on the critical line.  Yet RH is false for
f: Davenport and Heilbronn (1936) proved that f has infinitely many zeros with
Re s > 1 (and hence, by the functional equation, with Re s < 0), and f has
zeros inside the critical strip off the line, the first ones being

    0.808517 + 85.699348 i   and   0.650830 + 114.163343 i

(Spira 1994; Balanzario & Sanchez-Ortiz 2007).  Bombieri and Ghosh (2011)
studied the distribution of these zeros.  What f lacks is an Euler product:
(c_n) is not multiplicative (c_2 c_3 = -kappa^2 != c_6 = 1).  The moral,
which this module makes concrete, is that any proof of RH must use the Euler
product (multiplicativity of the coefficients), not just the functional
equation and analytic continuation.

Zero-free region (used to bound the search).  Since
sum_{n>=2} |c_n| n^{-sigma} < 1 for sigma > sigma_0 = 1.395136..., f(s) != 0 for
Re s > sigma_0, hence (functional equation) also for Re s < 1 - sigma_0 away
from the trivial zeros s = -1, -3, -5, ... (zeros of G).  All non-trivial
zeros lie in 1 - sigma_0 < Re s < sigma_0.

Implementation: L-functions, Gamma factors and their derivatives are rigorous
Arb enclosures (python-flint ``dirichlet_char.l`` / ``acb_series.dirichlet_l``);
zeros off the line are located with the cell/winding/Newton machinery of
``rh.speiser`` and certified with the Krawczyk test (rigorous Re s > 1/2);
zeros on the line are the sign changes of Z(t).  Consistency: the winding
number of the whole strip rectangle must equal (#sign changes) + 2 (#off-line
zeros with Re s > 1/2 found).

References
----------
* H. Davenport, H. Heilbronn, On the zeros of certain Dirichlet series,
  J. London Math. Soc. 11 (1936) 181-185; II, ibid. 307-312.
* E. C. Titchmarsh, The Theory of the Riemann Zeta-Function, 2nd ed., §10.25.
* R. Spira, Some zeros of the Titchmarsh counterexample, Math. Comp. 63 (1994)
  747-748.
* E. P. Balanzario, J. Sanchez-Ortiz, Zeros of the Davenport-Heilbronn
  counterexample, Math. Comp. 76 (2007) 2045-2049.
* E. Bombieri, A. Ghosh, Around the Davenport-Heilbronn function,
  Uspekhi Mat. Nauk 66 (2011) 15-66 (Russian Math. Surveys 66, 221-270).
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, asdict, field
from typing import List, Optional, Sequence, Tuple

import mpmath as mp
import numpy as np
from flint import acb, arb, ctx, acb_series, dirichlet_char

from .speiser import (ArgumentPrincipleError, Certificate, LocatedZero, certify_zero,
                      locate_zeros, winding_number, _to_complex)

# ----------------------------------------------------------------------------
# constants and characters
# ----------------------------------------------------------------------------
with mp.workdps(50):
    KAPPA_MP = (mp.sqrt(10 - 2 * mp.sqrt(5)) - 2) / (mp.sqrt(5) - 1)
KAPPA = float(KAPPA_MP)                     # 0.2840790438404123
CHI = dirichlet_char(5, 2)                  # chi(2) = i
CHIBAR = dirichlet_char(5, 3)               # conjugate character, chi(2) = -i
SIGMA0 = 1.3951361582351                    # f(s) != 0 for Re s > SIGMA0 (see dh_zero_free_abscissa)
KNOWN_OFF_LINE_ZEROS = (complex(0.808517, 85.699348), complex(0.650830, 114.163343))
DH_COEFFICIENT_PATTERN = (0.0, 1.0, KAPPA, -KAPPA, -1.0)   # c_n for n = 0,1,2,3,4 mod 5


def _kappa_arb() -> arb:
    return ((10 - 2 * arb(5).sqrt()).sqrt() - 2) / (arb(5).sqrt() - 1)


def _coeffs() -> Tuple[acb, acb]:
    k = _kappa_arb()
    return (1 - acb(0, 1) * k) / 2, (1 + acb(0, 1) * k) / 2


# ----------------------------------------------------------------------------
# f, its derivative, the Gamma factor, the functional equation
# ----------------------------------------------------------------------------
def dh_f(s, dps: Optional[int] = None) -> acb:
    """f(s) as a rigorous Arb enclosure (s: acb, complex or float)."""
    if dps is not None:
        ctx.dps = dps
    a, ab = _coeffs()
    s = s if isinstance(s, acb) else acb(complex(s).real, complex(s).imag)
    return a * CHI.l(s) + ab * CHIBAR.l(s)


def dh_f_complex(z: complex) -> complex:
    """f(z) as a Python complex (midpoint)."""
    return _to_complex(dh_f(z))


def dh_jet(z: complex) -> Tuple[complex, complex]:
    """(f(z), f'(z)) as Python complex numbers."""
    a, ab = _coeffs()
    x = acb_series([acb(z.real, z.imag), 1], 2)
    ser = a * x.dirichlet_l(CHI) + ab * x.dirichlet_l(CHIBAR)
    c = ser.coeffs()
    return _to_complex(c[0]), _to_complex(c[1])


def dh_ball_jet(X: acb) -> Tuple[acb, acb]:
    """Rigorous enclosures (f(X), f'(X)) over a complex box X."""
    a, ab = _coeffs()
    x = acb_series([X, 1], 2)
    c = (a * x.dirichlet_l(CHI) + ab * x.dirichlet_l(CHIBAR)).coeffs()
    return c[0], c[1]


def dh_gamma_factor(s: acb) -> acb:
    """G(s) = (5/pi)^{1/2-s} Gamma(1 - s/2) / Gamma((1+s)/2), so that f(s) = G(s) f(1-s)."""
    return (arb(5) / arb.pi()) ** (acb(0.5) - s) * (1 - s / 2).gamma() / ((1 + s) / 2).gamma()


def functional_equation_residual(s: complex, dps: int = 40) -> dict:
    """|f(s) - G(s) f(1-s)| as a rigorous upper bound, plus the relative version."""
    old = ctx.dps
    ctx.dps = dps
    try:
        z = acb(s.real, s.imag)
        lhs = dh_f(z)
        rhs = dh_gamma_factor(z) * dh_f(1 - z)
        diff = abs(lhs - rhs)
        return {"s": [s.real, s.imag], "abs_residual": float(diff.upper()),
                "rel_residual": float((diff / abs(lhs)).upper()), "f": [float(lhs.real), float(lhs.imag)]}
    finally:
        ctx.dps = old


def verify_kappa(dps: int = 40) -> dict:
    """Check kappa = tan(arg(eps)/2) with eps = tau(chi)/(i sqrt 5) the root number of chi."""
    old = ctx.dps
    ctx.dps = dps
    try:
        tau = sum((CHI(a) * acb(0, 2 * arb.pi() * a / 5).exp() for a in range(1, 5)), acb(0))
        eps = tau / (acb(0, 1) * arb(5).sqrt())
        k_num = (eps.arg() / 2).tan()
        k_formula = _kappa_arb()
        return {"kappa": str(k_formula.str(30)), "kappa_from_root_number": str(k_num.str(30)),
                "difference_upper": float(abs(k_num - k_formula).upper()),
                "root_number": [float(eps.real), float(eps.imag)], "abs_root_number_minus_1": float(abs(abs(eps) - 1).upper())}
    finally:
        ctx.dps = old


def dh_coefficients(nmax: int) -> np.ndarray:
    """c_1, ..., c_nmax of the Dirichlet series f(s) = sum c_n n^{-s}."""
    n = np.arange(1, nmax + 1)
    return np.array(DH_COEFFICIENT_PATTERN)[n % 5]


def dh_dirichlet_partial_sum(s: complex, nmax: int) -> complex:
    """Partial sum of the Dirichlet series (converges absolutely for Re s > 1)."""
    n = np.arange(1, nmax + 1, dtype=np.float64)
    return complex(np.sum(dh_coefficients(nmax) * n ** (-s)))


def dh_zero_free_abscissa(dps: int = 30) -> float:
    """sigma_0 with sum_{n>=2} |c_n| n^{-sigma_0} = 1; f(s) != 0 for Re s > sigma_0.

    The sum is evaluated with Hurwitz zeta functions over the residue classes.
    """
    with mp.workdps(dps):
        k = KAPPA_MP

        def tail(sig):
            s = mp.mpf(sig)
            return (5 ** (-s) * (mp.zeta(s, mp.mpf(1) / 5) + mp.zeta(s, mp.mpf(4) / 5)) - 1
                    + k * 5 ** (-s) * (mp.zeta(s, mp.mpf(2) / 5) + mp.zeta(s, mp.mpf(3) / 5)))
        return float(mp.findroot(lambda s: tail(s) - 1, 1.4))


# ----------------------------------------------------------------------------
# The critical line: Z(t)
# ----------------------------------------------------------------------------
def dh_theta(t: float) -> arb:
    """theta(t) = (t/2) log(5/pi) + Im log Gamma(3/4 + it/2);  G(1/2+it) = e^{-2 i theta(t)}."""
    tt = arb(t)
    return tt / 2 * (arb(5) / arb.pi()).log() + acb(arb(3) / 4, tt / 2).lgamma().imag


def dh_Z_acb(t: float, dps: Optional[int] = None) -> acb:
    """Z(t) = e^{i theta(t)} f(1/2 + it) as an Arb enclosure; its imaginary part encloses 0."""
    if dps is not None:
        ctx.dps = dps
    return (acb(0, 1) * dh_theta(t)).exp() * dh_f(acb(0.5, t))


def dh_Z(t: float, dps: Optional[int] = None) -> float:
    """Real Hardy-type function Z(t) = e^{i theta(t)} f(1/2 + it) (midpoint of the real part)."""
    return float(dh_Z_acb(t, dps).real)


def dh_Z_jet(t: float) -> Tuple[float, float]:
    """(Z(t), Z'(t)) with Z'(t) = Re[ e^{i theta} (i theta' f + i f') ], theta' = (1/2) log(5/pi) + (1/2) Re psi(3/4 + it/2)."""
    f, df = dh_ball_jet(acb(0.5, t))
    th = dh_theta(t)
    thp = (arb(5) / arb.pi()).log() / 2 + acb(arb(3) / 4, arb(t) / 2).digamma().real / 2
    e = (acb(0, 1) * th).exp()
    Z = e * f
    dZ = e * (acb(0, 1) * thp * f + acb(0, 1) * df)
    return float(Z.real), float(dZ.real)


def on_line_zeros(t0: float, t1: float, step: float = 0.05, tol: float = 1e-12,
                  dps: int = 20) -> List[float]:
    """Ordinates of the zeros of f on the critical line in (t0, t1): sign changes of Z(t).

    Each sign change is refined by safeguarded Newton (falling back to
    bisection) to absolute accuracy `tol`.  Zeros of even multiplicity, or two
    zeros closer than `step`, would be missed; the caller cross-checks against
    the winding-number count.
    """
    ctx.dps = dps
    grid = np.arange(t0, t1 + step / 2, step)
    grid[-1] = min(grid[-1], t1)
    vals = np.array([dh_Z(float(t)) for t in grid])
    zeros: List[float] = []
    for i in np.nonzero(np.sign(vals[:-1]) * np.sign(vals[1:]) < 0)[0]:
        a, b = float(grid[i]), float(grid[i + 1])
        fa, fb = float(vals[i]), float(vals[i + 1])
        t = a - fa * (b - a) / (fb - fa)
        for _ in range(60):
            Z, dZ = dh_Z_jet(t)
            if Z == 0:
                break
            if (Z < 0) == (fa < 0):
                a, fa = t, Z
            else:
                b, fb = t, Z
            tn = t - Z / dZ if dZ != 0 else 0.5 * (a + b)
            if not (a < tn < b):
                tn = 0.5 * (a + b)
            if abs(tn - t) < tol:
                t = tn
                break
            t = tn
        zeros.append(float(t))
    return zeros


# ----------------------------------------------------------------------------
# Off-line zeros and the full scan
# ----------------------------------------------------------------------------
def off_line_zeros(t0: float, t1: float, sigma_lo: float = 0.5 + 1e-3, sigma_hi: float = 1.5,
                   *, dt: float = 0.5, step_h: float = 0.25, step_v: float = 0.2, threshold: float = 1.0,
                   certify: bool = True, dps: int = 20, certify_dps: int = 30
                   ) -> Tuple[List[dict], int, int]:
    """All zeros of f with sigma_lo < Re s < sigma_hi, t0 < Im s < t1, certified.

    Returns (zeros, cell winding total, evaluations).  Each zero dict holds the
    Newton point, residual, and (if certified) rigorous bounds on Re s, Im s.
    """
    ctx.dps = dps
    breaks = [sigma_lo] + [b for b in (0.75, 1.0, 1.25) if sigma_lo < b < sigma_hi] + [sigma_hi]
    zs, w, nev = locate_zeros(dh_f_complex, dh_jet, sigma_lo, sigma_hi, t0, t1, sigma_breaks=breaks,
                              dt=dt, step_h=step_h, step_v=step_v, threshold=threshold)
    out = []
    for z in sorted(zs, key=lambda z: z.im):
        d = {"re": z.re, "im": z.im, "residual": z.residual, "newton_steps": z.newton_steps,
             "multiplicity": z.multiplicity, "mirror_re": 1.0 - z.re}
        if certify and z.multiplicity == 1:
            c = certify_zero(dh_ball_jet, z.z, dps=certify_dps)
            d.update({"certified": c.certified, "radius": c.radius, "re_lower": c.re_lower,
                      "re_upper": c.re_upper, "im_lower": c.im_lower, "im_upper": c.im_upper,
                      "rigorously_off_line": bool(c.certified and c.re_lower > 0.5)})
        else:
            d.update({"certified": False, "rigorously_off_line": False})
        out.append(d)
    ctx.dps = dps
    return out, w, nev


def total_zero_count(t0: float, t1: float, sigma_lo: float = -0.5, sigma_hi: float = 1.5,
                     step: float = 0.1, dps: int = 20) -> int:
    """Winding number of f around [sigma_lo, sigma_hi] x [t0, t1] (f is entire: = number of zeros)."""
    ctx.dps = dps
    return winding_number(dh_f_complex, sigma_lo, sigma_hi, t0, t1, step, step, 0.8)


@dataclass
class DHResult:
    T: float
    t0: float
    kappa: float
    kappa_check: dict
    sigma0: float
    functional_equation: List[dict]
    on_line_count: int
    off_line_count_right: int          # zeros with Re s > 1/2 (each has a mirror with Re s < 1/2)
    off_line_certified: int
    total_count_winding: int           # winding number of [-0.5, 1.5] x [t0, T]
    consistent: bool                   # total == on_line + 2 * off_line_right
    off_line_zeros: List[dict]
    on_line_zeros: List[float]
    known_zeros_found: List[dict]
    min_offline_distance_to_line: float
    evaluations: int
    wall_seconds: float
    dps: int

    def to_dict(self) -> dict:
        return asdict(self)


def dh_scan(T: float = 300.0, t0: float = 0.0, *, dt: float = 0.5, z_step: float = 0.05,
            dps: int = 20, certify: bool = True, fe_points: Sequence[complex] = (
                complex(0.3, 10.0), complex(0.8, 85.7), complex(1.7, 3.3), complex(-0.4, 50.2), complex(0.5, 200.1))
            ) -> DHResult:
    """Full study of the zeros of f with t0 < Im s < T; see the module docstring."""
    t_start = time.time()
    fe = [functional_equation_residual(s) for s in fe_points]
    kc = verify_kappa()
    sigma0 = dh_zero_free_abscissa()
    off, w_off, nev = off_line_zeros(t0, T, dt=dt, certify=certify, dps=dps)
    on = on_line_zeros(t0, T, step=z_step, dps=dps)
    total = total_zero_count(t0, T, 1.0 - sigma0 - 0.1, sigma0 + 0.1, dps=dps)
    consistent = (total == len(on) + 2 * len(off))
    if not consistent:
        # refine the line scan (close pairs of on-line zeros) once
        on = on_line_zeros(t0, T, step=z_step / 5, dps=dps)
        consistent = (total == len(on) + 2 * len(off))
    known = []
    for k in KNOWN_OFF_LINE_ZEROS:
        if t0 < k.imag < T:
            best = min(off, key=lambda d: abs(complex(d["re"], d["im"]) - k)) if off else None
            known.append({"known": [k.real, k.imag],
                          "found": [best["re"], best["im"]] if best else None,
                          "distance": abs(complex(best["re"], best["im"]) - k) if best else None,
                          "residual": best["residual"] if best else None})
    mind = min((d["re"] - 0.5 for d in off), default=float("nan"))
    return DHResult(
        T=T, t0=t0, kappa=KAPPA, kappa_check=kc, sigma0=sigma0, functional_equation=fe,
        on_line_count=len(on), off_line_count_right=len(off),
        off_line_certified=sum(1 for d in off if d.get("certified")),
        total_count_winding=total, consistent=consistent, off_line_zeros=off, on_line_zeros=on,
        known_zeros_found=known, min_offline_distance_to_line=mind, evaluations=nev,
        wall_seconds=time.time() - t_start, dps=dps,
    )


# ----------------------------------------------------------------------------
# Figure
# ----------------------------------------------------------------------------
def plot_zeros(res: DHResult, path: str, title: Optional[str] = None) -> str:
    """Scatter plot of the zeros in the critical strip; off-line zeros highlighted."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    surface, ink, ink2, muted, grid, axis = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
    blue, orange = "#2a78d6", "#eb6834"
    fig, ax = plt.subplots(figsize=(7.2, 9.0), dpi=150, facecolor=surface)
    ax.set_facecolor(surface)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(axis)
        ax.spines[sp].set_linewidth(0.8)
    ax.grid(axis="y", color=grid, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.axvline(0.5, color=axis, linewidth=0.8, zorder=1)
    for x in (0.0, 1.0):
        ax.axvline(x, color=grid, linewidth=0.8, zorder=1)
    ax.scatter([0.5] * len(res.on_line_zeros), res.on_line_zeros, s=14, color=blue,
               edgecolors=surface, linewidths=1.0, zorder=3,
               label=f"zeros on Re s = 1/2 ({res.on_line_count})")
    xs = [d["re"] for d in res.off_line_zeros] + [1 - d["re"] for d in res.off_line_zeros]
    ys = [d["im"] for d in res.off_line_zeros] * 2
    ax.scatter(xs, ys, s=70, color=orange, edgecolors=surface, linewidths=1.5, zorder=4,
               label=f"zeros off the line ({2 * res.off_line_count_right}, in symmetric pairs)")
    for d in res.off_line_zeros:
        ax.annotate(f"{d['re']:.6f} + {d['im']:.6f} i", (d["re"], d["im"]),
                    xytext=(10, 0), textcoords="offset points", fontsize=7.5, color=ink2, va="center")
    ax.set_xlim(-0.08, 1.32)
    ax.set_ylim(res.t0, res.T)
    ax.set_xlabel("Re s", color=ink2, fontsize=9)
    ax.set_ylabel("t = Im s", color=ink2, fontsize=9)
    ax.tick_params(colors=muted, labelsize=8, length=3)
    ax.set_title(title or f"Zeros of the Davenport–Heilbronn function, {res.t0:g} < t < {res.T:g}",
                 color=ink, fontsize=11, loc="left", pad=34)
    ax.text(0.0, 1.012,
            "f(s) = ½(1−iκ) L(s,χ) + ½(1+iκ) L(s,χ̄),  χ mod 5 with χ(2) = i,  κ = 0.28408\n"
            "a Dirichlet series with a ζ-type functional equation but no Euler product: RH fails for f",
            transform=ax.transAxes, fontsize=7.5, color=ink2, va="bottom")
    leg = ax.legend(loc="lower right", fontsize=8, frameon=True, facecolor=surface, edgecolor=grid)
    for t in leg.get_texts():
        t.set_color(ink2)
    fig.tight_layout()
    fig.savefig(path, facecolor=surface)
    plt.close(fig)
    return path
