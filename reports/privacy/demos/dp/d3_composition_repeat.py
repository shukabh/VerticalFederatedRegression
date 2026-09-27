"""
Report item D4 — what R gets by simply running the protocol again (no budget ledger exists anywhere).

party_o.py draws FRESH noise (np.random.default_rng()) on every connection and has no
counter, so k runs on the same data release k independent Gaussian perturbations of the
same statistic vector s.  k Gaussian releases with noise sigma and sensitivity Delta are
exactly one Gaussian release with sensitivity sqrt(k)*Delta (mu_k = sqrt(k)*Delta/sigma);
the effective (eps, delta) follows from the Balle-Wang profile.  Averaging the k releases
is the sufficient statistic (noise sigma/sqrt(k)).

Run: python d3_composition_repeat.py     (~2 s)
"""
import math, os, sys
import numpy as np
from scipy.special import log_ndtr
from scipy.optimize import brentq

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "../../../../scenario_b")))
import calibrate_hyperparameters as ch  # noqa: E402
sys.path.insert(0, HERE)
from d1_replacement_sensitivity import delta_replace  # noqa: E402


def delta_profile(mu, eps):
    a = log_ndtr(mu / 2 - eps / mu)
    b = eps + log_ndtr(-mu / 2 - eps / mu)
    return 0.0 if b >= a else math.exp(a) * (-math.expm1(b - a))


def eps_of_mu(mu, delta):
    if delta_profile(mu, 0.0) <= delta:
        return 0.0
    return brentq(lambda e: delta_profile(mu, e) - delta, 0.0, 500.0, xtol=1e-10)


delta = 1e-5
BR = BO = By = 1.0
D6 = ch.joint_sensitivity_B(BR, BO, By)
Drep = delta_replace(BR, BO, By)
print(f"bounds B_R=B_O=B_y=1 ; delta={delta}; Delta_6={D6:.3f}, Delta_rep={Drep:.3f}")
print(f"{'eps/run':>7s} {'k runs':>7s} {'eps_total (eq.6 adjacency)':>28s} {'eps_total (Def.4 adjacency)':>29s} {'naive k*eps':>12s}")
for eps in (0.5, 1.0, 4.0):
    sigma = ch.analytic_gaussian_sigma(D6, eps, delta)
    for k in (1, 2, 5, 10, 50, 100):
        e6 = eps_of_mu(math.sqrt(k) * D6 / sigma, delta)
        er = eps_of_mu(math.sqrt(k) * Drep / sigma, delta)
        print(f"{eps:7.1f} {k:7d} {e6:28.2f} {er:29.2f} {k*eps:12.1f}")
    print()

# how many silent re-runs before the targeted per-record SNR B_R*B_y/sigma_avg exceeds 3
for eps in (0.5, 1.0, 4.0, 8.0):
    sigma = ch.analytic_gaussian_sigma(D6, eps, delta)
    k3 = math.ceil((3 * sigma / (BR * By)) ** 2)
    print(f"eps={eps}: sigma={sigma:.3f}; re-runs until a dedicated-column read of y_i has SNR>=3: k={k3}")
