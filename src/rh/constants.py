"""Mathematical constants used throughout the package (float64 and mpmath)."""
import math
import mpmath as mp

EULER_GAMMA = 0.57721566490153286060651209008240243104215933593992
LOG_2PI = math.log(2 * math.pi)
LOG_4PI = math.log(4 * math.pi)

# C = sum over ALL nontrivial zeros rho of 1/|rho|^2 = 2 + gamma - log(4 pi)
#   = 0.0461914179322420676...  (this is 2*lambda_1, where lambda_1 = sum_rho 1/rho
#   = 1 + gamma/2 - log(4 pi)/2 = 0.0230957089661210338... is Li's first coefficient).
# It is the constant in Burnol's lower bound liminf d_N^2 log N >= C for the
# Baez-Duarte distances (see docs/02_equivalent_formulations.md).
BAEZ_DUARTE_C = 2 + EULER_GAMMA - LOG_4PI  # 0.0461914179322420676...
LI_LAMBDA_1 = BAEZ_DUARTE_C / 2            # 0.0230957089661210338...

# Height to which RH has been verified rigorously (Platt–Trudgian 2021)
RH_VERIFIED_HEIGHT = 3.0001753328e12
RH_VERIFIED_ZEROS = 12_363_153_437_138

# de Bruijn–Newman constant: 0 <= Lambda <= 0.2 (Rodgers–Tao 2018; Platt–Trudgian 2021)
DBN_LOWER = 0.0
DBN_UPPER = 0.2


def mp_constants(dps=50):
    """Return a dict of the same constants as mpmath numbers at `dps` digits."""
    with mp.workdps(dps):
        g = mp.euler
        return {
            "gamma": +g,
            "log_2pi": mp.log(2 * mp.pi),
            "log_4pi": mp.log(4 * mp.pi),
            "C": 2 + g - mp.log(4 * mp.pi),
        }
