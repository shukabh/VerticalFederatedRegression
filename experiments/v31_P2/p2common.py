"""Shared pieces of the re-selection of the prediction setting (P2) of protocol v3.1.

Builds on experiments/v31_settings/scommon.py (imported as `cs`), which builds on experiments/v31_eval/common.py
(`ev`): the plaintext simulation of the v3.1 releases from the implementation's own modules (vfr/dp.py calibrate,
vfr/solve.py masked noise, stage-1 rule and exact radius, the gate).  What is new here:

  * two selection populations built from the simulation population (vfr/data.population(),
    experiments/sim_core.Population):
      'std'  standardization with public constants and the population's 99.5% bounds (as in Section 13.5);
      'bn'   block normalization: each party scales its own standardized feature block of every row to unit norm,
             so B_R^2 = 2 (R's intercept included) and B_O = 1 exactly; B_y is the population's 99.5% bound on
             |y| (y is the same standardized outcome under both preprocessings);
  * population moments of the block-normalized design (test MSE on the population, unclipped y);
  * the P2 candidate grid and its pre-registered extension (reports/v31_P2/settings_choice_P2.md).

Settings are passed as dicts (f1, theta_star, kappa_Psi, alpha, alpha_L, alpha_U, r1_mode, intervals); the
SETTINGS presets of vfr/dp.py are not read for P2.
"""
from __future__ import annotations

import math
import os
import sys
from functools import lru_cache

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, '..', '..'))
SETDIR = os.path.join(ROOT, 'experiments', 'v31_settings')
for _p in (SETDIR, os.path.join(ROOT, 'experiments', 'v31_eval')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import scommon as cs  # noqa: E402  (experiments/v31_settings/scommon.py)
from scommon import ev, dp, solve, sc  # noqa: E402,F401
import sim_accuracy as sa  # noqa: E402  (experiments/v31_eval/sim_accuracy.py: population, moments, n grid)

REPORT = os.path.join(ROOT, 'reports', 'v31_P2')
REPORT_SET = cs.REPORT                       # reports/v31_settings (I and the old P; not rerun)
REPORT_EVAL = cs.REPORT_EVAL                 # reports/v31_eval (previous setting; not rerun)
WORK = os.path.join(ROOT, 'experiments', '_work', 'v31_P2')
FIGDIR = ev.FIGDIR
os.makedirs(REPORT, exist_ok=True)
os.makedirs(WORK, exist_ok=True)

EPS = cs.EPS
DELTA = cs.DELTA
NS = sa.NS
P, DA = 11, 5
POPS = ('std', 'bn')
POP_LABEL = {'std': 'standardized (Section 13.5 bounds)', 'bn': 'block-normalized (B_R^2 = 2, B_O = 1)'}
POP_CODE = {'std': 1, 'bn': 2}

# ------------------------------------------------------------------------------------------- settings
I_SET = cs.make_setting(0.25, 2.5, cs.ALPHA / 20, 'exact')                            # setting I (unchanged)
P_OLD = cs.make_setting(0.2, 0.5, 0.5, 'exact', intervals=False)                      # the old P (replaced)
PREV = dict(cs.PREV)

THETA_GRID = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5]
ALPHAL_GRID = [0.0025, 0.05, 0.25, 0.5]
F1_GRID = [0.1, 0.2, 0.25, 0.35]
# pre-registered one-time extension beyond each edge (used only if the choice lies on that edge)
EXTEND = {'theta_star': {'low': [0.25], 'high': [3.0, 3.5]},
          'alpha_L': {'low': [0.0005], 'high': [0.75]},
          'f1': {'low': [0.05], 'high': [0.45, 0.55]}}


def p_setting(theta, aL, f1):
    return cs.make_setting(f1, theta, aL, 'exact', intervals=False)


def cname(theta, aL, f1):
    return f'P:theta={theta:g},aL={aL:g},f1={f1:g}'


def grid_candidates(thetas=THETA_GRID, alphas=ALPHAL_GRID, f1s=F1_GRID):
    return {cname(t, a, f): p_setting(t, a, f) for t in thetas for a in alphas for f in f1s}


def reference_settings():
    return {'I': dict(I_SET), 'P_old': dict(P_OLD), 'prev': dict(PREV)}


# ------------------------------------------------------------------------------------------- populations
@lru_cache(maxsize=None)
def population(r2):
    return sa.population(r2)


@lru_cache(maxsize=None)
def _bounds(kind, r2):
    pop = population(r2)
    if kind == 'std':
        return dict(B_R=pop.B_R, B_O=pop.B_O, B_y=pop.B_y)
    if kind == 'bn':
        return dict(B_R=math.sqrt(2.0), B_O=1.0, B_y=pop.B_y)
    raise ValueError(kind)


def bounds(kind, r2):
    return dict(_bounds(kind, float(r2)))


def _unit_rows(X):
    nr = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.where(nr > 0, nr, 1.0)


def draw_both(pop, n, rng):
    """One matched cohort of size n under both preprocessings, from the same draw of features and outcome.
    'std' is exactly pop.draw (standardize with the public constants, clip each block to the population's
    99.5% bounds); 'bn' scales each party's standardized block of every row to unit norm.  y is clipped to
    pop.B_y in both (the release uses clipped y; test MSE uses the population's unclipped y)."""
    X = pop._features(n, rng)
    Zs = (X - pop.mu) / pop.s
    y = Zs @ pop.beta_std + pop.noise_sd * rng.standard_normal(n)
    zR, zO = Zs[:, :pop.d_R], Zs[:, pop.d_R:]
    nR, nO = np.linalg.norm(zR, axis=1), np.linalg.norm(zO, axis=1)
    cR = zR * np.minimum(1.0, pop.B_R_rows / np.maximum(nR, 1e-300))[:, None]
    cO = zO * np.minimum(1.0, pop.B_O / np.maximum(nO, 1e-300))[:, None]
    yc = np.clip(y, -pop.B_y, pop.B_y)
    one = np.ones((n, 1))
    Z_std = np.hstack([one, cR, cO])
    Z_bn = np.hstack([one, _unit_rows(zR), _unit_rows(zO)])
    return dict(std=Z_std, bn=Z_bn), yc


def pop_moments(kind, r2, rows=4_000_000, chunk=250_000):
    """Population moments of the preprocessed design: Sigma and ell = lambda_min(Sigma), and the test moments
    (design as released, unclipped y) for the population test MSE.  'std' is sim_accuracy.pop_moments (the
    moments of Section 13.5); 'bn' is computed here the same way (4 million rows, its own seed)."""
    if kind == 'std':
        return sa.pop_moments(r2)
    return _pop_moments_bn(float(r2), rows, chunk)


@lru_cache(maxsize=None)
def _pop_moments_bn(r2, rows, chunk):
    path = os.path.join(WORK, f'pop_moments_bn_r2_{r2}.npz')
    if os.path.exists(path):
        return dict(np.load(path))
    pop = population(r2)
    rng = np.random.default_rng([cs.BASE_SEED, 99, int(round(r2 * 100)), POP_CODE['bn']])
    G = np.zeros((P, P)); c = np.zeros(P); cu = np.zeros(P); yyu = 0.0
    for _ in range(rows // chunk):
        X = pop._features(chunk, rng)
        Zs = (X - pop.mu) / pop.s
        y = Zs @ pop.beta_std + pop.noise_sd * rng.standard_normal(chunk)
        Z = np.hstack([np.ones((chunk, 1)), _unit_rows(Zs[:, :pop.d_R]), _unit_rows(Zs[:, pop.d_R:])])
        yc = np.clip(y, -pop.B_y, pop.B_y)
        G += Z.T @ Z; c += Z.T @ yc; cu += Z.T @ y; yyu += float(y @ y)
    out = dict(Sigma=G / rows, beta_clip=np.linalg.solve(G, c), ell=np.linalg.eigvalsh(G / rows)[0],
               test_G=G / rows, test_c=cu / rows, test_yy=np.array(yyu / rows),
               lam_min_A=np.linalg.eigvalsh(G[:DA, :DA] / rows)[0])
    np.savez(path, **out)
    return out


def pred_mse(th, M):
    """Population test MSE of theta (R, p) (design as released, unclipped y)."""
    return sa.pred_mse(th, M)


def calibrate(kind, r2, eps, f1, delta=DELTA, n_bar_O=32768):
    b = bounds(kind, r2)
    return cs.calibrate(eps, b['B_R'], b['B_O'], b['B_y'], 4, 6, f1, delta=delta, n_bar_O=n_bar_O)


def noise_ratio_table(r2=0.5, f1s=None):
    """Delta, Delta_G, y'y mode and sigma_1 / sigma_2 of both populations (eps = 1; the ratio does not depend on
    eps beyond the scale search's rounding)."""
    f1s = f1s or sorted(set(F1_GRID) | {0.05, 0.45, 0.55})
    out = {}
    for kind in POPS:
        b = bounds(kind, r2)
        D, Dn = dp.delta_rep(b['B_R'], b['B_O'], b['B_y'])
        DG = dp.delta_G(b['B_R'], b['B_O'])
        row = dict(bounds=b, Delta=D, Delta_noyy=Dn, Delta_G=DG, DG_over_D=DG / D, by_f1={})
        for f1 in f1s:
            cal = calibrate(kind, r2, 1.0, f1)
            row['by_f1'][str(f1)] = dict(sigma1=cal['sigma1'], sigma2=cal['sigma2'],
                                         ratio=cal['sigma1'] / cal['sigma2'], joint=cal['joint'])
        M = pop_moments(kind, r2)
        row['ell'] = float(M['ell'])
        row['lam_min_A'] = float(M['lam_min_A'])
        out[kind] = row
    return out


# ------------------------------------------------------------------------------------------- mechanism
def run_setting_on(G, cvec, Z1, Z2, f2, cal, st):
    """One setting on a batch of cohorts (G (R,p,p), c (R,p)) from unit noise (Z1, Z2, f2): stage 1, gate, solve.
    Returns theta with refused -> 0, the stage-1 output and the refusal flags."""
    s1 = cs.stage1_vec(G + cal['sigma1'] * Z1, DA, cal['sigma1'], cal['sigma2'], st)
    th, _, gate, refused, _, _ = cs.stage2_from_unit(G, cvec, Z2, f2, s1['lam'], s1['Psi'], cal['sigma2'])
    return th, s1, gate, refused


def stamp():
    import datetime
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
