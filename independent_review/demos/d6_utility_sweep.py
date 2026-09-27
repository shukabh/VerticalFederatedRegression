"""
d6_utility_sweep.py -- is the privacy/utility trade-off viable at the sizes the paper targets?

Uses the authors' own algebra (phase1_common: aggregate_O, draw_noise, apply_noise,
solve_and_correct in mode='auto', calibrate's analytic sigma) on generate_vfl_data designs
(d_R = d_O = 5, U[0,1] covariates, noise_std 0.1). Bounds are public constants:
  raw:  B_R = B_O = sqrt(5) (exact max norm of U[0,1]^5, no clipping), B_y = 6
  std:  centre 0.5 / scale 1/sqrt(12) (true U[0,1] moments -> public), intercept in A,
        B_R = sqrt(1+B^2), B_O = B = 3.0; y centred/scaled by a public prior-cycle value, B_y = 3
Reports, over noise replications on one dataset per n:
  rel.err = ||beta_bc - beta_OLS|| / ||beta_OLS||   (median), sampling ||beta_OLS - beta_true||,
  the ridge mode and lambda, and the same with y'y dropped from the release (sigma smaller).
"""
import sys, math
import numpy as np
sys.path.insert(0, "/home/user/VerticalFederatedRegression/scenario_b")
import phase1_common as p1
import generate_vfl_data as gvd
from calibrate_hyperparameters import analytic_gaussian_sigma, clip_rows, clip_scalar

dR0 = dO = 5; delta = 1e-5; REPS = 60


def sens(BR, BO, By, with_yty=True):
    return math.sqrt((BO**2 + BR**2) * (BO**2 + By**2) + (By**4 if with_yty else 0.0))


def one(n, eps, standardize, with_yty, seed=11):
    d = gvd.generate_vfl_data(n, dR0, n, dO, n, "B", 0.1, seed)
    XR, XO, y = d.X_R_I.copy(), d.X_O_I.copy(), d.y.copy()
    if standardize:
        s = 1 / math.sqrt(12)
        XR = (XR - 0.5) / s; XO = (XO - 0.5) / s
        yc, ys = 0.5 * (d.beta_R.sum() + d.beta_O.sum()), 1.5     # "prior cycle" constants
        y = (y - yc) / ys
        B = 3.0; BR_rows, BO, By = B, B, 3.0
        XR, _ = clip_rows(XR, BR_rows); XO, _ = clip_rows(XO, BO); y, _ = clip_scalar(y, By)
        XR = np.hstack([np.ones((n, 1)), XR]); BR = math.sqrt(1 + BR_rows**2)
    else:
        BR = BO = math.sqrt(5); By = 6.0
        y, _ = clip_scalar(y, By)
    dR = XR.shape[1]
    b = np.ones(n)
    A = p1.local_A(XR); B_, C, cO, cR, yty = p1.aggregate_O(b, XR, XO, y)
    G, c = p1.assemble(A, B_, C, cR, cO)
    bols = np.linalg.solve(G, c)
    sig = analytic_gaussian_sigma(sens(BR, BO, By, with_yty), eps, delta)
    rng = np.random.default_rng(seed + 1)
    rel, modes, lams = [], [], []
    for _ in range(REPS):
        nz = p1.draw_noise(dR, dO, sig, rng)
        At, Bt, Ct, cRt, cOt, _ = p1.apply_noise(A, B_, C, cR, cO, yty, nz)
        Gt, ct = p1.assemble(At, Bt, Ct, cRt, cOt)
        sol = p1.solve_and_correct(Gt, ct, dR, dO, sig, mode="auto", rho_target=2.0)
        rel.append(np.linalg.norm(sol.beta_bc - bols) / np.linalg.norm(bols))
        modes.append(sol.Psi_mode); lams.append(sol.lam)
    samp = np.linalg.norm(bols[-(dR0 + dO):] - np.concatenate([d.beta_R, d.beta_O])) \
        if not standardize else float('nan')
    return sig, np.median(rel), max(set(modes), key=modes.count), np.median(lams), samp


print(f"{'design':>5} {'eps':>4} {'n':>7} | {'sigma':>7} {'mode':>4} {'lambda':>9} {'rel.err':>8} "
      f"| {'sigma no-yty':>12} {'rel.err no-yty':>14} | {'sampling err':>12}")
for standardize in (False, True):
    for eps in (1.0, 4.0):
        for n in (400, 2000, 10000, 50000):
            s1, r1, m1, l1, samp = one(n, eps, standardize, True)
            s2, r2, _, _, _ = one(n, eps, standardize, False)
            print(f"{'std' if standardize else 'raw':>5} {eps:4.0f} {n:7d} | {s1:7.1f} {m1:>4} {l1:9.1f} "
                  f"{r1:8.3f} | {s2:12.1f} {r2:14.3f} | {samp:12.4f}")
