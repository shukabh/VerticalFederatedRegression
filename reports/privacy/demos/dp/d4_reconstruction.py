"""
D4 — practical disclosure: how well can R reconstruct individual y_i / x_O,i from the
noised c_R = X_R^T y + f_R and C = X_R^T X_O + E_C, knowing X_R of every matched person?

Noise: sigma = analytic_gaussian_sigma(joint_sensitivity_B(B_R,B_O,B_y), eps, 1e-5) — the
code's own calibration. Bounds are public, data-independent 0.99 quantiles of the
generating law (chi_d for standardized N(0,I) rows, 2.576 for standardized y), data
clipped to them, as calibrate/party_* do.

y is generated from X_O only (independent of X_R), so R has NO population-level way to
predict y from x_R: every bit of recovered signal is record-level leakage.

Attacker (Bayes-optimal for the Gaussian model): posterior mean under the prior
y ~ N(0, s^2 I):  y_hat = X_R (X_R^T X_R + (sigma^2/s^2) I)^{-1} c_R   (same for columns of X_O).
Metrics: reconstruction R^2 = 1 - SSE(y_hat)/SSE(prior mean) on the same records (0 = nothing
learned, 1 = exact); for binary y: AUC of the posterior-mean score (0.5 = nothing learned).

Run: python d4_reconstruction.py     (~20 s)
"""
import math, os, sys
import numpy as np
from scipy.stats import chi, norm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "../../../../scenario_b")))
import calibrate_hyperparameters as ch  # noqa: E402
sys.path.insert(0, HERE)
from d1_replacement_sensitivity import delta_replace  # noqa: E402

DELTA = 1e-5
EPSS = (0.5, 1.0, 4.0, 8.0)
rng = np.random.default_rng(2026)


def clip_rows(X, B):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X * np.minimum(1.0, B / np.maximum(n, 1e-300))


def post_mean(X, c, sigma, s2, m0=0.0):
    """posterior mean of v (n-vector) from c = X^T v + N(0,sigma^2), prior N(m0, s2 I)."""
    c0 = c - X.T @ (m0 * np.ones(X.shape[0]))
    k = X.shape[1]
    return m0 + X @ np.linalg.solve(X.T @ X + (sigma**2 / s2) * np.eye(k), c0)


def auc(score, lab):
    r = np.argsort(np.argsort(score)) + 1
    n1 = lab.sum(); n0 = len(lab) - n1
    return (r[lab == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0) if n1 and n0 else float("nan")


def sim(n, dR, dO, eps, reps=150, binary=False, k_runs=1):
    BR = float(chi.ppf(0.99, dR)); BO = float(chi.ppf(0.99, dO))
    By = 1.0 if binary else 2.576
    D6 = ch.joint_sensitivity_B(BR, BO, By)
    sigma = ch.analytic_gaussian_sigma(D6, eps, DELTA)
    sig_eff = sigma / math.sqrt(k_runs)           # averaging k silent re-runs
    r2y, r2x, aucs, accs = [], [], [], []
    for _ in range(reps):
        XR = clip_rows(rng.standard_normal((n, dR)), BR)
        XO = clip_rows(rng.standard_normal((n, dO)), BO)
        lin = XO @ rng.standard_normal(dO)
        if binary:
            p = 1 / (1 + np.exp(-(lin / lin.std() * 1.5 - 0.85)))
            y = (rng.uniform(size=n) < p).astype(float); m0, s2 = 0.3, 0.21
        else:
            y = lin / lin.std() * math.sqrt(0.6) + rng.standard_normal(n) * math.sqrt(0.4)
            y = np.clip(y, -By, By); m0, s2 = 0.0, 1.0
        cR = XR.T @ y + rng.standard_normal(dR) * sig_eff
        C = XR.T @ XO + rng.standard_normal((dR, dO)) * sig_eff
        yh = post_mean(XR, cR, sig_eff, s2, m0)
        if binary:
            aucs.append(auc(yh, y))
        else:   # R^2 against the prior-mean predictor on the SAME records
            r2y.append(1 - np.sum((yh - y) ** 2) / np.sum((m0 - y) ** 2))
        xh = np.column_stack([post_mean(XR, C[:, j], sig_eff, 1.0) for j in range(dO)])
        r2x.append(1 - np.sum((xh - XO) ** 2) / np.sum(XO ** 2))
    out = dict(BR=BR, BO=BO, By=By, sigma=sigma, mu6=D6 / sigma,
               murep=delta_replace(BR, BO, By) / sigma, snr_target=BR * By / sig_eff,
               r2x=np.mean(r2x))
    if binary:
        out.update(auc=np.nanmean(aucs))
    else:
        out.update(r2y=np.mean(r2y))
    return out


print("=== 0. per-record noise-to-signal (typical record vs noise, per block), d_R=20, d_O=5")
dR, dO = 20, 5
BR = chi.ppf(0.99, dR); BO = chi.ppf(0.99, dO); By = 2.576
D6 = ch.joint_sensitivity_B(BR, BO, By)
for eps in EPSS:
    s = ch.analytic_gaussian_sigma(D6, eps, DELTA)
    # typical ||x_R||~sqrt(dR), ||x_O||~sqrt(dO), |y|~0.8 ; noise Frobenius per block
    nsr_c = s * math.sqrt(dR) / (math.sqrt(dR) * 0.8)
    nsr_C = s * math.sqrt(dR * dO) / (math.sqrt(dR) * math.sqrt(dO))
    nsr_B = s * math.sqrt(dO * (dO + 1) / 2) / dO
    print(f"  eps={eps:3.1f} sigma={s:7.2f} | NSR c_R={nsr_c:6.1f}  C={nsr_C:6.1f}  B={nsr_B:6.1f}"
          f"  -> one record is {1/nsr_c:.3f} noise-SDs in c_R")

print("\n=== 1. targeted worst case: R gives the target a dedicated column (x_R,i = B_R e_j)")
print("     c_R[j] = B_R*y_i + N(0,sigma^2), C[j,:] = B_R*x_O,i + noise  (per-coordinate SD sigma/B_R)")
for (dR, dO, lab) in ((20, 5, "d_R=20,d_O=5"), (50, 5, "d_R=50,d_O=5 (R-heavy)"), (10, 15, "d_R=10,d_O=15"),
                      (20, 1, "d_R=20,d_O=1 (few controls)"), (50, 1, "d_R=50,d_O=1 (few controls)")):
    BR = chi.ppf(0.99, dR); BO = chi.ppf(0.99, dO)
    for binary in (False, True):
        By = 1.0 if binary else 2.576
        D6 = ch.joint_sensitivity_B(BR, BO, By)
        row = []
        for eps in EPSS:
            s = ch.analytic_gaussian_sigma(D6, eps, DELTA)
            if binary:   # y in {0,1}, prior 1/2, LR test on c_R[j]
                row.append(f"eps={eps:g}: acc={norm.cdf(BR/(2*s)):.3f}")
            else:        # posterior SD of y_i relative to prior SD 1
                ps = 1 / math.sqrt(1 + (BR / s) ** 2)
                row.append(f"eps={eps:g}: postSD/priorSD={ps:.3f}")
        print(f"  {lab:24s} {'binary y' if binary else 'cont. y '}: " + "  ".join(row))

print("\n=== 2. whole-cohort reconstruction, continuous y (R^2 of y and of X_O; 0 = nothing learned)")
print(f"  {'n':>4s} {'d_R':>4s} {'d_O':>4s} {'eps':>4s} {'sigma':>7s} {'mu_claim':>8s} {'mu_true':>8s} {'R2(y)':>7s} {'R2(X_O)':>8s}")
for (n, dR, dO) in ((20, 20, 5), (50, 50, 5), (50, 10, 5), (200, 20, 5), (20, 20, 1), (50, 50, 1)):
    for eps in EPSS:
        o = sim(n, dR, dO, eps, reps=300)
        print(f"  {n:4d} {dR:4d} {dO:4d} {eps:4.1f} {o['sigma']:7.2f} {o['mu6']:8.3f} {o['murep']:8.3f} "
              f"{o['r2y']:7.3f} {o['r2x']:8.3f}")

print("\n=== 3. binary y (e.g. benefit receipt, prevalence ~0.3), n=d_R=20: AUC of R's per-person score")
for dO in (5, 1):
    row = []
    for eps in EPSS:
        o = sim(20, 20, dO, eps, reps=300, binary=True)
        row.append(f"eps={eps:g}: AUC={o['auc']:.3f}")
    print(f"  d_O={dO}: " + "  ".join(row))

print("\n=== 4. same, after k silent re-runs averaged (no ledger): n=d_R=20, continuous y")
for eps in (1.0, 4.0):
    for k in (1, 10, 100, 1000):
        o = sim(20, 20, 5, eps, reps=150, k_runs=k)
        print(f"  eps/run={eps:3.1f} k={k:5d}  R2(y)={o['r2y']:.3f}  R2(X_O)={o['r2x']:.3f}")
