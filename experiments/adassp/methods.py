"""
methods.py -- estimators for the AdaSSP benchmark comparison, batched over R noise draws.

Wang (2018) baselines, re-implemented in numpy from their definitions in
<https://github.com/yuxiangw/optimal_dp_linear_regression> (code/*.m; no code is copied):

  trivial      theta = 0                                           (trivial_predictor.m)
  nonprivate   theta = (X'X + I)^-1 X'y                            (linreg.m)
  ssp          s = sqrt(log(4/delta))/eps; iid N(0, s^2) on EVERY entry of X'X + I (not
               symmetrized) and of X'y; theta = XTX_hat^-1 XTy_hat (suffstats_perturb.m)
  adassp       B_X = B_Y = 1, varrho = 0.05, s = sqrt(log(6/delta))/(eps/3):
               eta = sqrt(d log(6/delta) log(2 d^2/varrho)) / (eps/3)
               lmin~ = max(0, lmin(X'X + I) + s N(0,1) - log(6/delta)/(eps/3))
               lam = max(0, eta - lmin~)
               XTy_hat = X'y + s N(0, I_d);  XTX_hat = X'X + I + s (G + G')/2
               theta = (XTX_hat + lam I)^-1 XTy_hat                  (adassp.m)

Vertical-FL methods (the revised Scenario-B protocol, paper/main.tex), through the shared core
experiments/sim_core.py (read-only). R's block must be the first d_R coordinates.

  adassp_matched  AdaSSP's structure under OUR adjacency (replace-one of O-side data (x_O, y),
                  x_R fixed) and accountant (analytic Gaussian, basic composition):
                  * the O-dependent blocks of X'X and X'y (plus y'y) are released jointly with
                    sim_core.release at sigma = analytic_gauss_sigma(delta_rep(1,1,1), 2eps/3,
                    2delta/3); A = X_R'X_R is exact;
                  * lmin(X'X + I) has replace-one sensitivity 1 (||dG||_op <= max ||z||^2 <= 1
                    for unit-norm rows); it is released with the analytic Gaussian at
                    (eps/3, delta/3): sigma_l = analytic_gauss_sigma(1, eps/3, delta/3);
                  * Wang's downward shift is kept as the same number of standard deviations,
                    shift = sigma_l * sqrt(log(6/delta)) (Wang: s*sqrt(log(6/delta)) =
                    log(6/delta)/(eps/3));
                  * Wang's eta = s*sqrt(d log(2d^2/varrho)) is "noise std of the Gram entries x
                    sqrt(d log(2 d^2/varrho))"; we substitute our Gram-noise std:
                    eta = sigma * sqrt(d log(2 d^2/varrho))  (eta_scale=sqrt(2) gives the variant
                    that also accounts for our off-diagonal variance being sigma^2 rather than the
                    sigma^2/2 of Wang's (G+G')/2);
                  * lam = max(0, eta - lmin~); theta = (G~ + I + lam I)^-1 c~.
                  This is an idealized comparison point: nobody holds G in the clear in VFL, so
                  the private lmin step cannot be run there.
  vfl             sigma = analytic_gauss_sigma(delta_rep(1,1,1), eps, delta); fixed ridge
                  lam = fixed_lambda(sigma, p, n, ell, rho_star=2); sim_core.solve gives the
                  uncorrected ridge beta and the bias-corrected beta_bc. ell = 0 is the
                  implementable protocol; ell = 0.8 lmin(G/n) (from the data) is the NON-PRIVATE
                  oracle-ell headroom reference.
  vfl_adaptive    the user's original gate, sim_core.adaptive_gate, on the same release.
"""
from __future__ import annotations

import os
import sys
from functools import lru_cache

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import sim_core as sc  # noqa: E402

VARRHO = 0.05


# -----------------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------------

def bsolve(A, b):
    """Batched solve A[r] x[r] = b[r]; A (R,d,d), b (R,d)."""
    return np.linalg.solve(A, b[..., None])[..., 0]


def sym_half(R, d, rng):
    """Wang's symmetric noise (G + G')/2 with G iid N(0,1): diag var 1, off-diag var 1/2."""
    Gm = rng.standard_normal((R, d, d))
    return 0.5 * (Gm + np.swapaxes(Gm, 1, 2))


@lru_cache(maxsize=None)
def sigma_vfl(eps, delta, B_R=1.0, B_O=1.0, B_y=1.0):
    return sc.analytic_gauss_sigma(sc.delta_rep(B_R, B_O, B_y), eps, delta)


@lru_cache(maxsize=None)
def sigma_matched(eps, delta):
    """(sigma for the joint O-block release at (2eps/3, 2delta/3), sigma_l for lmin at
    (eps/3, delta/3) with sensitivity 1)."""
    return (sc.analytic_gauss_sigma(sc.delta_rep(1.0, 1.0, 1.0), 2 * eps / 3, 2 * delta / 3),
            sc.analytic_gauss_sigma(1.0, eps / 3, delta / 3))


# -----------------------------------------------------------------------------
# Wang baselines
# -----------------------------------------------------------------------------

def nonprivate(G0, c):
    return np.linalg.solve(G0 + np.eye(G0.shape[0]), c)


def ssp(G0, c, eps, delta, R, rng):
    d = G0.shape[0]
    s = np.sqrt(np.log(4 / delta)) / eps
    XTX = G0 + np.eye(d)
    Gh = XTX[None] + s * rng.standard_normal((R, d, d))
    ch = c[None] + s * rng.standard_normal((R, d))
    return bsolve(Gh, ch)


def adassp(G0, c, eps, delta, R, rng, lmin=None, return_lam=False):
    d = G0.shape[0]
    e3 = eps / 3
    logsod = np.log(6 / delta)
    s = np.sqrt(logsod) / e3
    eta = np.sqrt(d * logsod * np.log(2 * d * d / VARRHO)) / e3
    XTX = G0 + np.eye(d)
    if lmin is None:
        lmin = float(np.linalg.eigvalsh(XTX)[0])
    lt = np.maximum(lmin + s * rng.standard_normal(R) - logsod / e3, 0.0)
    lam = np.maximum(0.0, eta - lt)
    ch = c[None] + s * rng.standard_normal((R, d))
    Gh = XTX[None] + s * sym_half(R, d, rng) + lam[:, None, None] * np.eye(d)[None]
    th = bsolve(Gh, ch)
    return (th, lam) if return_lam else th


# -----------------------------------------------------------------------------
# VFL methods
# -----------------------------------------------------------------------------

def vfl_release(G0, c, yty, d_R, sigma, R, rng):
    return sc.release(G0, c, yty, d_R, sigma, R, rng)


def adassp_matched(G0, Gt, ct, eps, delta, rng, lmin=None, eta_scale=1.0, return_lam=False):
    """AdaSSP structure on an existing matched release (Gt, ct) drawn at sigma_matched(eps)[0]."""
    R, d, _ = Gt.shape
    sigma, sigma_l = sigma_matched(eps, delta)
    if lmin is None:
        lmin = float(np.linalg.eigvalsh(G0 + np.eye(d))[0])
    shift = sigma_l * np.sqrt(np.log(6 / delta))
    lt = np.maximum(lmin + sigma_l * rng.standard_normal(R) - shift, 0.0)
    eta = eta_scale * sigma * np.sqrt(d * np.log(2 * d * d / VARRHO))
    lam = np.maximum(0.0, eta - lt)
    th = bsolve(Gt + (1.0 + lam)[:, None, None] * np.eye(d)[None], ct)
    return (th, lam) if return_lam else th


def vfl_fixed(Gt, ct, sigma, d_R, n, ell=0.0, rho_star=2.0):
    """Revised protocol: fixed ridge from public inputs. Returns (beta_ridge, beta_bc, lam)."""
    p = Gt.shape[1]
    lam = sc.fixed_lambda(sigma, p, n, ell, rho_star)
    b, bc, _ = sc.solve(Gt, ct, lam, sigma, d_R)
    return b, bc, lam


def vfl_adaptive(Gt, ct, sigma, d_R):
    """Original adaptive gate, per draw. Returns (beta_ridge, beta_bc, lam)."""
    d_O = Gt.shape[1] - d_R
    return sc.adaptive_gate(Gt, ct, sigma, d_R, d_O)


# -----------------------------------------------------------------------------
# metrics
# -----------------------------------------------------------------------------

def test_mse(Xte, yte, theta):
    """Wang's linreg_err on a batch: theta (R,d) or (d,) -> (R,) or scalar."""
    th = np.atleast_2d(theta)
    r = Xte @ th.T - yte[:, None]
    out = np.mean(r * r, axis=0)
    return out if np.ndim(theta) == 2 else float(out[0])


def coef_rmse(theta, theta_ref):
    th = np.atleast_2d(theta)
    out = np.sqrt(np.mean((th - theta_ref[None]) ** 2, axis=1))
    return out if np.ndim(theta) == 2 else float(out[0])
