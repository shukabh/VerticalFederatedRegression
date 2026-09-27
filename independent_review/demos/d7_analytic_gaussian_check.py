"""d7: check calibrate_hyperparameters.analytic_gaussian_sigma against an independent scipy
implementation of the Balle-Wang condition, and report the exact (eps, delta) curve."""
import sys
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
sys.path.insert(0, "/home/user/VerticalFederatedRegression/scenario_b")
from calibrate_hyperparameters import analytic_gaussian_sigma, classical_gaussian_sigma

def delta_of(sig, D, eps):   # exact privacy profile of the Gaussian mechanism
    a, b = D / (2 * sig), eps * sig / D
    return norm.cdf(a - b) - np.exp(eps) * norm.cdf(-a - b)

for (D, eps, dl) in [(1, 1, 1e-5), (1, 0.5, 1e-6), (20.57, 1, 1e-5), (23.3, 4, 1e-5), (1, 8, 1e-5)]:
    s_code = analytic_gaussian_sigma(D, eps, dl)
    s_ref = brentq(lambda s: delta_of(s, D, eps) - dl, 1e-6 * D, 1e3 * D)
    print(f"Delta={D:6.2f} eps={eps:4.1f} delta={dl:.0e}: code {s_code:.6f}  scipy {s_ref:.6f}  "
          f"classical {classical_gaussian_sigma(D, eps, dl):.4f}  rel.diff {abs(s_code-s_ref)/s_ref:.1e}")
