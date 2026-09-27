"""
d3_sensitivity_replacement.py -- is Delta_2 in eq. (6) right for the adjacency of Def. 4?

Def. 4: I is fixed/public, datasets differ in ONE matched individual's (x_O, y).
That is replace-one adjacency. Eq. (6) is derived for "adding one record" (equivalently
zero-out: replace (x_O,y) by (0,0)). Here we maximise the l2 norm of the change of the
RELEASED vector [triu(B), C, c_R, c_O, y'y] under replacement, by random restarts +
local optimisation, and compare with eq. (6). Then compute the actual eps the calibrated
sigma delivers under replacement adjacency (analytic Gaussian, Balle-Wang).
"""
import math, sys
import numpy as np
from scipy.optimize import minimize
sys.path.insert(0, "/home/user/VerticalFederatedRegression/scenario_b")
from calibrate_hyperparameters import joint_sensitivity_B, analytic_gaussian_sigma


def released_change(xR, x, y, x2, y2):
    dO = len(x)
    D = np.outer(x, x) - np.outer(x2, x2)
    iu = np.triu_indices(dO)
    parts = [D[iu], np.outer(xR, x - x2).ravel(), xR * (y - y2),
             x * y - x2 * y2, [y * y - y2 * y2]]
    return np.concatenate([np.ravel(p) for p in parts])


def proj(v, B):
    n = np.linalg.norm(v); return v if n <= B else v * (B / n)


def max_replace(BR, BO, By, dR, dO, restarts=300, seed=0):
    rng = np.random.default_rng(seed)
    def unpack(t):
        xR = proj(t[:dR], BR); x = proj(t[dR:dR+dO], BO)
        x2 = proj(t[dR+dO:dR+2*dO], BO)
        y = np.clip(t[-2], -By, By); y2 = np.clip(t[-1], -By, By)
        return xR, x, y, x2, y2
    f = lambda t: -np.sum(released_change(*unpack(t)) ** 2)
    best = 0
    for _ in range(restarts):
        t0 = rng.normal(size=dR + 2*dO + 2) * 2
        r = minimize(f, t0, method="Nelder-Mead", options={"maxiter": 4000, "xatol": 1e-9, "fatol": 1e-12})
        best = max(best, -r.fun)
    return math.sqrt(best)


def eps_for_sigma(sigma, Delta, delta):
    lo, hi = 1e-4, 100.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if analytic_gaussian_sigma(Delta, mid, delta) > sigma:
            lo = mid
        else:
            hi = mid
    return hi


delta = 1e-5
print(f"{'B_R':>5}{'B_O':>5}{'B_y':>5} | {'Delta eq.(6)':>12} {'Delta replace':>13} {'ratio':>6} | "
      f"{'nominal eps':>11} {'actual eps (replace)':>21}")
for (BR, BO, By) in [(1, 1, 1), (1.7, 1.7, 4.5), (3.3, 3.3, 2.6), (1, 1, 5), (2, 2, 1)]:
    D6 = joint_sensitivity_B(BR, BO, By)
    Dr = max_replace(BR, BO, By, 2, 2, restarts=120)
    for eps in (1.0,):
        sig = analytic_gaussian_sigma(D6, eps, delta)
        print(f"{BR:5.1f}{BO:5.1f}{By:5.1f} | {D6:12.3f} {Dr:13.3f} {Dr/D6:6.2f} | "
              f"{eps:11.2f} {eps_for_sigma(sig, Dr, delta):21.2f}")

# closed-form witnesses (valid for any d_O >= 2)
BR, BO, By = 1.0, 1.0, 1.0
xR = np.array([BR, 0]); x = np.array([BO, 0]); x2 = np.array([0, BO])
w1 = np.linalg.norm(released_change(xR, x, By, x2, -By))
w2 = np.linalg.norm(released_change(xR, x, By, -x, -By))
print(f"\nwitness x'=perp(x), y'=-y at B=1: |Delta| = {w1:.3f}  (eq.(6): {joint_sensitivity_B(1,1,1):.3f})")
print(f"witness x'=-x,      y'=-y at B=1: |Delta| = {w2:.3f}")
print("generic upper bound: Delta_replace <= 2 * Delta_zero-out (triangle inequality)")
