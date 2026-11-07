#!/usr/bin/env python3
"""Basis-size convergence of the smallest eigenvalue of the restricted Weil form,
to estimate the true infimum lambda*(a) (Rayleigh–Ritz values decrease to it)."""
import json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import mpmath as mp
from rh import weil_form as wf

cases = [(0.8, [32, 48, 64, 96, 128]), (1.0, [96, 128, 160, 192])]
out = []
for a, Js in cases:
    for J in Js:
        t0 = time.time()
        dps = int(max(80, 1.4 * (0.8 * J + 30) + 40))
        W = wf.build_weil_form(a, J=J, dps=dps, estimate_quad_error=False)
        rec = {"a": a, "J": J, "dps": dps}
        for parity in ("even", "odd"):
            Qs, Gs, idx = W.sector(parity)
            vals, _ = wf.smallest_eigenpairs(Qs, Gs, dps=dps, k=2)
            rec[parity] = [mp.nstr(vals[0], 10), mp.nstr(vals[1], 6)]
        rec["seconds"] = round(time.time() - t0)
        out.append(rec)
        print(json.dumps(rec), flush=True)
        json.dump(out, open("results/json/weil_form_convergence.json", "w"), indent=1)
print("done")
