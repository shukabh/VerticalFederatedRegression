"""The v3.1 release and solve on linked data, as a plaintext simulation built from the repo's own modules.

Same path as experiments/large/sim_large.py (the paper's large-n simulation): stage-1 release and rule
(experiments/v31_settings/scommon.stage1_batch with vfr/solve's exact r_1 quantile), stage-2 release and gate
(experiments/v31_eval/common.stage2_noise / stage2_solve), and under setting I the Wald interval for theta_lambda and
the bias-aware interval for beta (vfr/solve.stage2_infer).  Calibration: the rho split of vfr/dp.calibrate without
the CKKS scale search (experiments/v31_eval/common._calibrate_plain, which the repo documents as agreeing with
dp.calibrate to ~1e-12 relative); OpenFHE is not needed.  The encrypted protocol computes the same statistics up to
CKKS error ~1e-10 (paper, Section 'The cryptographic layer is exact'), so the linked statistics G, c, y'y fully
determine its output distribution.

The population, bounds and true beta are those of the paper's synthetic 'std' population (d_R = 4, d_O = 6).
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'experiments', 'large'))
import lcommon as lc  # noqa: E402
from lcommon import cs, ev, solve  # noqa: E402

P, DA = 11, 5
DELTA = 1e-9
BOUNDS = lc.bounds_for('std')
BETA = lc.beta_true('std')


def calibrate(eps, setting):
    st = lc.setting_dict(setting)
    c = ev._calibrate_plain(eps, DELTA, BOUNDS['B_R'], BOUNDS['B_O'], BOUNDS['B_y'], ev.kappa_for(eps), f1=st['f1'])
    return c, st


def n_off(eps, setting='I'):
    """n above which the stage-1 rule switches the ridge off (Section 'When can the ridge be dropped?'), with the
    population's lambda_min per record."""
    c, st = calibrate(eps, setting)
    ell = population_moments()['lmin_per_record']
    return cs.n_off(c, ell, P, DA, st)[0]


def design(ZR, ZO, y):
    """Z = [1 | z_R | z_O] and y clipped to B_y, as in experiments/large/sim_large.cohort_stats."""
    Z = np.hstack([np.ones((len(y), 1)), ZR, ZO])
    return Z, np.clip(y, -BOUNDS['B_y'], BOUNDS['B_y'])


def stats(Z, yc):
    return Z.T @ Z, Z.T @ yc, float(yc @ yc)


_MOM = {}


def population_moments(n=2_000_000, seed=99):
    """Monte Carlo moments of the population: M = E[z z'], m = E[z y_c] (true pairs) and the same under independent
    pairing of R's block with O's (x_O, y) (a false match), for the contamination formula."""
    if 'M' in _MOM:
        return _MOM
    pop = lc.population()
    rng = np.random.default_rng(seed)
    X = pop._features(n, rng)
    zR, zO, Zs = lc._transform(pop, X, 'std')
    y = Zs @ pop.beta_std + pop.noise_sd * rng.standard_normal(n)
    Z, yc = design(zR, zO, y)
    M = Z.T @ Z / n
    m = Z.T @ yc / n
    # independent pairing: R block from one person, (z_O, y) from another
    perm = rng.permutation(n)
    Zi, yi = design(zR, zO[perm], y[perm])
    Mi = Zi.T @ Zi / n
    mi = Zi.T @ yi / n
    _MOM.update(M=M, m=m, Mi=Mi, mi=mi, lmin_per_record=float(np.linalg.eigvalsh(M)[0]),
                theta_pop=np.linalg.solve(M, m))
    return _MOM


def contamination_limit(pi):
    """Probability limit of least squares when a fraction pi of the linked pairs are false matches with an O row
    drawn independently of the R row: theta*(pi) = ((1-pi) M + pi M_ind)^{-1} ((1-pi) m + pi m_ind)."""
    mo = population_moments()
    G = (1 - pi) * mo['M'] + pi * mo['Mi']
    c = (1 - pi) * mo['m'] + pi * mo['mi']
    return np.linalg.solve(G, c)


def release(G, c, yy, n, eps, setting, reps, rng, intervals=None, Bboot=1000):
    """reps independent runs of the two-stage release on fixed statistics (G, c, yy) of n linked rows.
    Returns dict of arrays: theta (reps, p) (0 when refused), refused, lam, and under I the interval half-widths and
    the coverage of beta (bias-aware) and of theta_lambda (Wald)."""
    cal, st = calibrate(eps, setting)
    Gs = np.broadcast_to(G, (reps, P, P)).copy()
    s1 = cs.stage1_batch(Gs, DA, cal, rng, reps, st)
    E, f, g = ev.stage2_noise(P, DA, cal, rng, reps)
    so = ev.stage2_solve(Gs + E, c[None] + f, s1['lam'], s1['Psi'], cal['sigma2'])
    th = np.where(so['refused'][:, None], 0.0, so['theta'])
    out = dict(theta=th, refused=so['refused'], lam=s1['lam'], sigma1=cal['sigma1'], sigma2=cal['sigma2'])
    do_int = st['intervals'] if intervals is None else intervals
    if do_int:
        cov_b, cov_t, hb, ht = [], [], [], []
        for i in range(reps):
            if so['refused'][i]:
                continue
            sd = dict(lam=s1['lam'][i], Psi=s1['Psi'][i], L_hat=s1['L_hat'][i])
            res = solve.stage2_infer(Gs[i] + E[i], c + f[i], yy + g[i], n, DA, sd, cal['sigma2'], cal['sigma_yy'],
                                     BOUNDS['B_y'], st['alpha'], st['alpha_L'], st['alpha_U'], np.inf, rng,
                                     Bboot=Bboot)
            tl = np.linalg.solve(G + s1['lam'][i] * s1['Psi'][i], c)
            cov_t.append(np.abs(res['theta'] - tl) <= res['half_theta'])
            cov_b.append(np.abs(res['theta'] - BETA) <= res['half_beta'])
            ht.append(res['half_theta']); hb.append(res['half_beta'])
        out.update(cov_beta=np.array(cov_b), cov_theta=np.array(cov_t), half_beta=np.array(hb),
                   half_theta=np.array(ht))
    return out


def summarize(rel, ref=None):
    """Coefficient error of the releases against the population beta (median RMSE over runs), split into the bias of
    the mean release and its spread, plus coverage."""
    th = rel['theta'][~rel['refused']]
    rm = np.sqrt(np.mean((th - BETA) ** 2, axis=1))
    mean = th.mean(0)
    out = dict(rmse_median=float(np.median(rm)), rmse_mean=float(np.sqrt(np.mean((th - BETA) ** 2))),
               bias_norm=float(np.linalg.norm(mean - BETA) / math.sqrt(P)),
               sd_rms=float(np.sqrt(np.mean(th.var(0)))), refused=float(rel['refused'].mean()),
               ridge_on=float(np.mean(rel['lam'] > 0)),
               bias_R=float(np.mean(mean[1:DA] - BETA[1:DA])), bias_O=float(np.mean(mean[DA:] - BETA[DA:])),
               ratio_R=float(np.mean(mean[1:DA] / BETA[1:DA])), ratio_O=float(np.mean(mean[DA:] / BETA[DA:])))
    if 'cov_beta' in rel and len(rel['cov_beta']):
        cb = rel['cov_beta'][:, 1:]                     # slopes only (the intercept's beta is 0)
        out.update(cov_beta=float(cb.mean()), cov_beta_R=float(cb[:, :DA - 1].mean()),
                   cov_beta_O=float(cb[:, DA - 1:].mean()), cov_theta=float(rel['cov_theta'][:, 1:].mean()),
                   half_beta_median=float(np.median(rel['half_beta'][:, 1:])))
    return out


def ols(G, c):
    return np.linalg.solve(G, c)
