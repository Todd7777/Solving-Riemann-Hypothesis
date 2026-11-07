"""Test of the duality heuristic: does the Weil minimiser on [-a,a] capture the zeta zeros
up to T* = 2 pi e^{2a} (and no further) as the basis grows?"""
import sys, time, json, math
sys.path.insert(0, __import__('os').path.join(__import__('os').path.dirname(__file__), '..', 'src'))
import mpmath as mp
from rh import weil_form as wf
out = []
with mp.workdps(60):
    gam = [mp.zetazero(k).imag for k in range(1, 61)]
def N_smooth(T):
    return T / (2 * math.pi) * math.log(T / (2 * math.pi * math.e)) + 7 / 8
for a, Js in [(1.25, [48, 64, 96, 128]), (1.0, [64, 96])]:
    Tstar = 2 * math.pi * math.exp(2 * a)
    for J in Js:
        t0 = time.time()
        dps = int(max(80, 1.4 * (0.8 * J + 30) + 40))
        W = wf.build_weil_form(a, J=J, dps=dps, estimate_quad_error=False)
        for parity in ("even", "odd"):
            Qs, Gs, idx = W.sector(parity)
            vals, vecs = wf.smallest_eigenpairs(Qs, Gs, dps=dps, k=1)
            with mp.workdps(min(dps, 100)):
                d = wf.zero_match(vecs[0], idx, a, gam, dps=min(dps, 100), window=0.5)
            n3 = 0
            for x in d:
                if x is not None and x < 1e-3:
                    n3 += 1
                else:
                    break
            n_any = sum(1 for x in d if x is not None and x < 1e-3)
            rec = {"a": a, "J": J, "parity": parity, "dps": dps, "lambda1": mp.nstr(vals[0], 6),
                   "T_star": Tstar, "N_smooth_T_star": N_smooth(Tstar), "leading_matched_1e-3": n3,
                   "total_matched_1e-3_first60": n_any,
                   "digits_first10": [round(-float(mp.log10(x)), 1) if (x is not None and x > 0) else None for x in d[:10]],
                   "seconds": round(time.time() - t0)}
            out.append(rec)
            print(json.dumps(rec), flush=True)
json.dump(out, open('results/json/weil_form_capture_law.json', 'w'), indent=1)
print("done")
