"""rh — a research platform for the Riemann Hypothesis.

Nothing in this package proves the Riemann Hypothesis.  It implements, tests
and runs the classical *equivalent criteria* and the standard verification
machinery so that each line of attack can be explored numerically with
rigorous (interval/ball) arithmetic where that is feasible.

Modules
-------
constants       : mathematical constants used across the package
data            : loading of the precomputed zero datasets (data/zeros)
zeta            : Riemann–Siegel Z(t), theta(t), Gram points, zero finding
verify          : Turing's method: certify N(T) and simplicity/location of zeros
li              : Li/Keiper coefficients lambda_n (Li's criterion)
nyman_beurling  : Báez-Duarte distances d_N (Nyman–Beurling criterion)
arithmetic      : Robin, Lagarias, Nicolas criteria; colossally abundant numbers
speiser         : zeros of zeta' (Speiser's criterion)
davenport_heilbronn : a functional-equation function violating RH (barrier illustration)
statistics      : pair correlation and spacing statistics vs. GUE
debruijn_newman : heat-flow dynamics of zeros, de Bruijn–Newman constant
explicit_formula: Weil's explicit formula and Weil positivity numerics
"""
__version__ = "0.1.0"
