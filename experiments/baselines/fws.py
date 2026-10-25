"""Ferrando, Wang and Sheldon (AISTATS 2022): parametric-bootstrap confidence intervals for DP OLS, reimplemented.

Reference code: github.com/ceciliaferrando/PB-DP-CIs, DPCIs-OLS.py (commit f0d2bc6e9707501dfc91b231cdd9779bca866e3b),
functions private_OLS and hybrid_bootstrap_OLS.  Their estimator and bootstrap, written for released noisy sufficient
statistics G^ = X'X + V, c^ = X'y + w:

  beta^  = G^^-1 c^                                     (the noisy OLS estimate; no ridge)
  Q^     = G^ / n, replaced by its PSD projection (eigenvalues <= 0 set to 0.1) if not positive definite
  s^2    = residual variance (FWS release it separately; here the plug-in (yy^ - c^' beta^)/(n - p) from the released
           y'y, floored at 0.1 when negative, as FWS floor their private s^2)
  for b = 1..B:  V*, w* fresh draws of the mechanism's noise;  Z* ~ N(0, s^2 Q^)
                 beta*_b = (Q^ + V*/n)^-1 (Q^ beta^ + Z*/sqrt(n) + w*/n)
  interval: the percentile interval [q_(a/2), q_(1-a/2)] of beta*_b, coordinate by coordinate.

The bootstrap resamples the sampling noise of y given X (through Z*, with the design fixed at Q^) and the privacy
noise (V*, w*).  Its target is the OLS coefficient beta of the working model y = X beta + e (homoskedastic e), i.e.
the population least-squares coefficient, not the sample OLS G^-1 c and not the ridge target theta_lambda.

Here the noise model is a general one: an s.d. per entry of the released Gram (zero on R's exact block A, sigma_2 on
the O-dependent entries for v3.1's stage 2) and per entry of c.  The draws are Gaussian (v3.1's mechanism); FWS's code
uses Laplace noise for its own epsilon-DP release.  check_against_reference() runs their bootstrap and this one on the
same inputs and the same random numbers and requires identical draws.
"""
from __future__ import annotations

import importlib.util
import math
import os
import sys

import numpy as np

EXT = os.environ.get('FWS_REF_DIR', '/tmp/claude-0/-home-user-VerticalFederatedRegression/efa2e795-381b-5bb8-8206-13c9bd0ddd5e/scratchpad/ext/PB-DP-CIs')   # local clone (not vendored)
FWS_COMMIT = 'f0d2bc6e9707501dfc91b231cdd9779bca866e3b'
PD_FLOOR = 0.1      # FWS make_pos_def(small_positive=0.1)
S2_FLOOR = 0.1      # FWS: "if sigma_sq_hat_priv < 0: sigma_sq_hat_priv = 0.1"


def psd_fix(Q, floor=PD_FLOOR):
    """FWS make_pos_def: eigenvalues <= 0 replaced by `floor` (symmetric eigendecomposition).  Returns (Q, fixed)."""
    w, U = np.linalg.eigh(Q)
    if np.all(w > 0):
        return Q, False
    return (U * np.where(w > 0, w, floor)) @ U.T, True


def sym_noise(rng, B, sdG):
    """Symmetric Gaussian noise with entry s.d. sdG (p, p) (upper triangle and diagonal drawn, mirrored)."""
    p = sdG.shape[0]
    M = rng.standard_normal((B, p, p))
    E = np.triu(M)
    E = E + np.swapaxes(np.triu(M, 1), -1, -2)
    return E * sdG[None]


def fws_core(beta, Q, s2, n, Vs, ws, Zs):
    """beta*_b = (Q + V*_b / n)^-1 (Q beta + Z*_b / sqrt(n) + w*_b / n) for stacked draws (B, ...)."""
    Qs = Q[None] + Vs / n
    rhs = (Q @ beta)[None] + Zs / math.sqrt(n) + ws / n
    return np.linalg.solve(Qs, rhs[..., None])[..., 0]


def fws_interval(G2, c2, yy2, n, sdG, sdc, rng, B=2000, level=0.955):
    """FWS percentile interval from released noisy statistics (G2, c2, yy2) with noise s.d. sdG (p,p) and sdc (p,).
    Returns dict(beta, lo, hi, s2, pd_fixed)."""
    p = len(c2)
    beta = np.linalg.solve(G2, c2)
    Q, fixed = psd_fix(G2 / n)
    s2 = (yy2 - c2 @ beta) / (n - p)
    s2_floored = s2 < 0
    if s2_floored:
        s2 = S2_FLOOR
    Vs = sym_noise(rng, B, sdG)
    ws = sdc[None] * rng.standard_normal((B, p))
    L = np.linalg.cholesky(s2 * Q + 1e-300 * np.eye(p))
    Zs = rng.standard_normal((B, p)) @ L.T
    bs = fws_core(beta, Q, s2, n, Vs, ws, Zs)
    a = 1 - level
    lo, hi = np.percentile(bs, [100 * a / 2, 100 * (1 - a / 2)], axis=0)
    return dict(beta=beta, lo=lo, hi=hi, s2=float(s2), pd_fixed=bool(fixed), s2_floored=bool(s2_floored))


# ------------------------------------------------------------------------------------------- validation
def load_reference():
    """Import their DPCIs-OLS.py from the scratchpad clone (not vendored).  It parses sys.argv at import time."""
    path = os.path.join(EXT, 'DPCIs-OLS.py')
    argv = sys.argv
    sys.argv = ['DPCIs-OLS.py']
    try:
        spec = importlib.util.spec_from_file_location('fws_ref', path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        sys.argv = argv
    return mod


def check_against_reference(seed=7, trials=3):
    """Their hybrid_bootstrap_OLS against fws_core with the same random numbers: per bootstrap draw their code calls
    compute_dp_noise twice (keeping w of the first call and V of the second) and then multivariate_normal; replaying
    exactly those calls into fws_core must give the same beta* to rounding."""
    ref = load_reference()
    args = ref.args
    args.D, args.N, args.eps, args.gamma, args.zeta, args.num_bootstraps = 4, 500, 2.0, 5, 150, 200
    worst = 0.0
    for t in range(trials):
        np.random.seed(seed + t)
        X, U, y, beta_true = ref.generate_ols_data(args.N, args.D, args.true_beta, args.gamma, args.delta, args.zeta)
        args.N = len(y)
        bhat, Qhat, s2hat = ref.private_OLS(X, y, args)
        np.random.seed(1000 + t)
        theirs = ref.hybrid_bootstrap_OLS(bhat, Qhat, s2hat, args)
        np.random.seed(1000 + t)
        Vs, ws, Zs = [], [], []
        for b in range(args.num_bootstraps):
            w_star = ref.compute_dp_noise(args.D, args.gamma, args.zeta, args.eps)[0]
            V_star = ref.compute_dp_noise(args.D, args.gamma, args.zeta, args.eps)[1]
            Z_star = np.random.multivariate_normal(np.zeros(args.D), s2hat * Qhat)
            Vs.append(V_star); ws.append(w_star); Zs.append(Z_star)
        mine = fws_core(bhat, Qhat, s2hat, args.N, np.array(Vs), np.array(ws), np.array(Zs))
        worst = max(worst, float(np.max(np.abs(mine - theirs) / (np.abs(theirs) + 1e-12))))
        args.N = 500
    return worst


def check_coverage_like_reference(trials=400, seed=3):
    """Their full pipeline (Laplace noise, their sensitivities and data) against fws_interval driven by the same noise
    law: the coverage of the test coordinate should agree up to Monte Carlo error."""
    ref = load_reference()
    args = ref.args
    args.D, args.eps, args.gamma, args.zeta, args.num_bootstraps, args.test_d = 4, 5.0, 5, 150, 300, 2
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    cov_ref, cov_mine, w_ref, w_mine = [], [], [], []
    for N in (1000,):
        for t in range(trials):
            args.N = N
            out = ref.coverage_test_single_trial(args)
            cov_ref.append(out['coverage']['bootstrap'] == 0)
            w_ref.append(out['ci']['bootstrap'][1] - out['ci']['bootstrap'][0])
            # same data law and the same Laplace mechanism, through fws_core
            X, U, y, beta_true = ref.generate_ols_data(N, args.D, args.true_beta, args.gamma, args.delta, args.zeta)
            n = N                       # their private_OLS and bootstrap use the nominal N (rows violating |y| <= zeta
                                        # are dropped by generate_ols_data); mimicked here
            DV, Dw = args.D * (args.D + 1) * args.gamma ** 2, args.D * 2 * args.gamma * args.zeta
            V = np.triu(rng.laplace(0, DV / args.eps / 3, (args.D, args.D)))
            V = V + np.triu(V, 1).T
            w = rng.laplace(0, Dw / args.eps / 3, args.D)
            G2, c2 = X.T @ X + V, X.T @ y + w
            beta = np.linalg.solve(G2, c2)
            Q, _ = psd_fix(G2 / n)
            bols = np.linalg.solve(X.T @ X, X.T @ y)
            width = max((args.zeta + np.sum(-args.gamma * np.abs(bols))) ** 2,
                        (-args.zeta - np.sum(args.gamma * np.abs(bols))) ** 2)
            s2 = np.sum((y - X @ bols) ** 2) / (n - args.D) + rng.laplace(0, width / (n - args.D) / args.eps / 3)
            s2 = S2_FLOOR if s2 < 0 else s2
            Bn = args.num_bootstraps
            Vs = np.triu(rng.laplace(0, DV / args.eps / 3, (Bn, args.D, args.D)))
            Vs = Vs + np.swapaxes(np.triu(Vs, 1), -1, -2)
            ws = rng.laplace(0, Dw / args.eps / 3, (Bn, args.D))
            Zs = rng.multivariate_normal(np.zeros(args.D), s2 * Q, Bn)
            bs = fws_core(beta, Q, s2, n, Vs, ws, Zs)
            lo, hi = np.percentile(bs[:, args.test_d], [2.5, 97.5])
            cov_mine.append(lo <= beta_true[args.test_d] <= hi)
            w_mine.append(hi - lo)
    return dict(coverage_reference=float(np.mean(cov_ref)), coverage_reimplementation=float(np.mean(cov_mine)),
                width_reference=float(np.median(w_ref)), width_reimplementation=float(np.median(w_mine)),
                trials=trials, N=1000, D=4, eps=5.0)


if __name__ == '__main__':
    print('max relative difference of beta* (their bootstrap vs ours, same random numbers):', check_against_reference())
    print(check_coverage_like_reference())
