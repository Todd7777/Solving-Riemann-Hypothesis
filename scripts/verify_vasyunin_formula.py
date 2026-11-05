#!/usr/bin/env python3
"""Independent high-precision verification of the Vasyunin closed form for
G(h,k) = int_0^inf {u/h}{u/k} u^{-2} du  (h, k coprime), used by rh.nyman_beurling.

The direct evaluation integrates exactly over the first K periods of the
periodic function f(u) = {u/h}{u/k} (period hk) and sums the tail with the
moment expansion  sum_m (-1)^m (m+1) L^{-(m+2)} mu_m zeta(m+2, K)  (Hurwitz zeta),
which converges geometrically.  Run:  python3 scripts/verify_vasyunin_formula.py
"""
import sys, os
import mpmath as mp
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from rh.nyman_beurling import gram_entry_direct_mp

mp.mp.dps = 40

def V(h, k):
    return sum(mp.mpf((m * h) % k) / k * mp.cot(mp.pi * m / k) for m in range(1, k))

def G_closed(h, k):
    return ((mp.log(2 * mp.pi) - mp.euler) / 2 * (mp.mpf(1) / h + mp.mpf(1) / k)
            + mp.mpf(k - h) / (2 * h * k) * mp.log(mp.mpf(h) / k)
            - mp.pi / (2 * h * k) * (V(h, k) + V(k, h)))

worst = mp.mpf(0)
for (h, k) in [(1, 1), (1, 2), (1, 3), (2, 3), (3, 5), (1, 7), (4, 7), (5, 12), (7, 11), (13, 30), (17, 23), (9, 40)]:
    d = gram_entry_direct_mp(h, k, K=200, dps=40)
    c = G_closed(h, k)
    worst = max(worst, abs(d - c))
    print(f"({h:2d},{k:2d})  direct = {mp.nstr(d, 30)}   closed = {mp.nstr(c, 30)}   diff = {mp.nstr(d - c, 3)}")
print("max |direct - closed| =", mp.nstr(worst, 3))
assert worst < mp.mpf(10) ** (-35)
