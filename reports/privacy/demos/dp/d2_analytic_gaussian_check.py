"""
Report item D9 — audit calibrate_hyperparameters.analytic_gaussian_sigma against Balle & Wang (2018).

Reference: the exact privacy profile of the Gaussian mechanism (Balle-Wang Thm 8),
    delta(eps; mu) = Phi(mu/2 - eps/mu) - e^eps Phi(-mu/2 - eps/mu),   mu = Delta/sigma,
evaluated with scipy.special.log_ndtr (accurate deep in the tail), root-found on mu.

Checks
  1. accuracy of the code's _Phi(x) = 0.5(1+erf(x/sqrt2)) in the left tail;
  2. sigma_code vs sigma_ref across eps and delta, and the delta ACTUALLY delivered by
     sigma_code (evaluated with the accurate reference).
Run:  python d2_analytic_gaussian_check.py     (~2 s)
"""
import math, os, sys
import numpy as np
from scipy.special import log_ndtr, ndtr
from scipy.optimize import brentq

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "../../../../scenario_b")))
import calibrate_hyperparameters as ch  # noqa: E402


def delta_profile(mu, eps):
    """Accurate Gaussian privacy profile, in log space for both terms."""
    a = log_ndtr(mu / 2 - eps / mu)
    b = eps + log_ndtr(-mu / 2 - eps / mu)
    if b >= a:
        return 0.0
    return math.exp(a) * (-math.expm1(b - a))


def sigma_ref(Delta, eps, delta):
    f = lambda logmu: math.log(max(delta_profile(math.exp(logmu), eps), 1e-320)) - math.log(delta)
    logmu = brentq(f, math.log(1e-4), math.log(1e3), xtol=1e-14, rtol=1e-14, maxiter=500)
    return Delta / math.exp(logmu)


print("=== 1. code _Phi vs scipy ndtr in the left tail")
for x in (-4, -5, -6, -7, -7.5, -8, -8.25, -8.5, -9, -10):
    pc, pr = ch._Phi(x), ndtr(x)
    rel = abs(pc - pr) / pr
    print(f"  x={x:6.2f}  Phi_ref={pr:.3e}  Phi_code={pc:.3e}  rel.err={rel:.1e}")

print("\n=== 2. sigma_code / sigma_ref and delta actually delivered by sigma_code (Delta=1)")
print(f"  {'eps':>5s} {'delta':>8s} {'sigma_ref':>11s} {'sigma_code':>11s} {'ratio':>9s} {'delta_actual/delta':>19s}")
for delta in (1e-5, 1e-8, 1e-12, 1e-14, 1e-15, 1e-16, 1e-17, 1e-20):
    for eps in (0.1, 1.0, 4.0, 8.0, 20.0):
        try:
            sr = sigma_ref(1.0, eps, delta)
        except ValueError:
            sr = float("nan")
        sc = ch.analytic_gaussian_sigma(1.0, eps, delta)
        da = delta_profile(1.0 / sc, eps)
        flag = "   <-- UNDER-NOISED" if da > 1.01 * delta else ""
        print(f"  {eps:5.1f} {delta:8.0e} {sr:11.5f} {sc:11.5f} {sc/sr:9.5f} {da/delta:19.4g}{flag}")

print("\n=== 3. sanity: classical formula vs analytic (eps<1 only is the classical one valid)")
for eps in (0.25, 0.5, 0.9):
    print(f"  eps={eps}: classical/analytic sigma = "
          f"{ch.classical_gaussian_sigma(1, eps, 1e-5)/ch.analytic_gaussian_sigma(1, eps, 1e-5):.3f}")
