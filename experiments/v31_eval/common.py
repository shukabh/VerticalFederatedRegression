"""Shared pieces of the v3.1 evaluation (paper Sections 12.5-12.8).

Everything here is a plaintext simulation of the v3.1 release and solve steps, built from the
implementation's own modules so that the statistics are those of the encrypted protocol:

  * calibration     implementation/vfr/dp.py: Calibrate with the CKKS constants of the
                    implementation's parameter set (sigma_1, sigma_2, sigma_yy, joint or separate
                    y'y mode, tight zCDP conversion, delta slack);
  * stage 1         implementation/vfr/solve.py: masked_noise on (triu B, C) with s.d. sigma_1,
                    stage1_rule (lambda_I / lambda_O, kappa_Psi, r_1, L_hat);
  * stage 2         masked_noise with s.d. sigma_2 on every O-dependent entry of G, Gaussian noise on
                    c and y'y, the gate theta_hat >= 1, the solve, and (where intervals are needed)
                    solve.stage2_infer (bootstrap, RSS correction, Wald and bias-aware intervals);
  * data            implementation/vfr/data.py population() (experiments/sim_core.Population) and
                    implementation/bench/utility_runs.transform (per-party preprocessing).

The encrypted implementation adds exact discrete-Gaussian noise on the coefficient lattice; its
slot-domain noise has the calibrated s.d. (measured 0.996 sigma_2 and 0.976 sigma_1), and every
encrypted median on protein lay inside the simulated 95% range (implementation/bench/results).

Evaluation settings (not protocol defaults): delta = 1e-9, f_1 = 0.1, theta* = 2.5, kappa_Psi = 2,
alpha = 0.05 with alpha_L = alpha_U = alpha/20, y'y released in stage 2 when it is free and
otherwise in its own release with f_y = 0.1 (the implementation's behaviour), bias correction off,
eps in {0.5, 1, 2, 4, 8}.  The flooding parameter is kappa = 40 for eps <= 2, 44 for eps = 4 and 48
for eps = 8, so that the statistical slack (1 + e^eps) gamma stays a negligible part of delta (with
kappa = 40, Calibrate refuses eps = 8 at delta = 1e-9); it changes sigma by under 0.5%.
"""
from __future__ import annotations

import json
import math
import os
import sys
from functools import lru_cache

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, '..', '..'))
IMPL = os.path.join(ROOT, 'implementation')
EXP = os.path.join(ROOT, 'experiments')
for _p in (IMPL, EXP, os.path.join(EXP, 'adassp')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from vfr import common as VC  # noqa: E402
from vfr import dp, solve  # noqa: E402
import sim_core as sc  # noqa: E402

REPORT = os.path.join(ROOT, 'reports', 'v31_eval')
WORK = os.path.join(ROOT, 'experiments', '_work', 'v31_eval')
FIGDIR = os.path.join(ROOT, 'paper', 'figures')
os.makedirs(REPORT, exist_ok=True)
os.makedirs(WORK, exist_ok=True)

DELTA = 1e-9
EPS = [0.5, 1.0, 2.0, 4.0, 8.0]
F1 = 0.1
F_Y = 0.1
THETA_STAR = 2.5
KAPPA_PSI = 2.0
ALPHA = 0.05
ALPHA_L = ALPHA_U = ALPHA / 20
KAPPA = {0.5: 40, 1.0: 40, 2.0: 40, 4.0: 44, 8.0: 48}
BASE_SEED = 20261024


def kappa_for(eps):
    return KAPPA.get(float(eps), 40 if eps <= 2 else 44 if eps <= 4 else 48)


# ------------------------------------------------------------------------------------------- calibration
@lru_cache(maxsize=1)
def _ckks():
    ck = VC.make_ckks(VC.CKKSParams())
    qs = list(ck.qs)
    return ck.N, qs[2], qs[3], qs[0] * qs[1], VC.gadget_constants(ck)


def _calibrate_plain(eps, delta, B_R, B_O, B_y, kappa, f1=F1, f_y=F_Y):
    """The rho split of dp.calibrate without the CKKS scale search (sigma_s = Delta_s / sqrt(2 rho_s));
    agrees with dp.calibrate to ~1e-12 relative.  Used only if dp.calibrate refuses on q0 grounds."""
    Delta, Delta_noyy = dp.delta_rep(B_R, B_O, B_y)
    DG = dp.delta_G(B_R, B_O)
    joint = (Delta - Delta_noyy) <= 1e-9 * Delta
    n_rel = 2 if joint else 3
    gamma = n_rel * (2.0 ** -50 + 2.0 ** -kappa * (1 + 2.0 ** -80) + 2.0 ** -97)
    delta_DP = delta - (1 + math.exp(eps)) * gamma
    rho = dp.rho_tight(eps, delta_DP)
    if joint:
        s1, s2 = DG / math.sqrt(2 * f1 * rho), Delta / math.sqrt(2 * (1 - f1) * rho)
        syy = s2
    else:
        s1 = DG / math.sqrt(2 * f1 * rho)
        s2 = Delta_noyy / math.sqrt(2 * (1 - f1 - f_y) * rho)
        syy = B_y ** 2 / math.sqrt(2 * f_y * rho)
    return dict(sigma1=s1, sigma2=s2, sigma_yy=syy, joint=bool(joint), rho_run=rho, delta_DP=delta_DP,
                Delta=Delta, Delta_noyy=Delta_noyy, Delta_G=DG, source='plain')


@lru_cache(maxsize=None)
def calibrate(eps, B_R, B_O, B_y, d_R, d_O, delta=DELTA, n_bar_O=32768):
    """sigma_1, sigma_2 and sigma_yy (slot domain) from the implementation's Calibrate."""
    kappa = kappa_for(eps)
    N, q1, q2, q0, gadget = _ckks()
    ag = dp.Agreement(n_R_max=1, n_bar_O=n_bar_O, d_R=d_R, d_O=d_O, B_R=B_R, B_O=B_O, B_y=B_y, eps_run=eps,
                      delta_run=delta, f1=F1, f_y=F_Y, theta_star=THETA_STAR, alpha=ALPHA, kappa_Psi=KAPPA_PSI,
                      kappa=kappa)
    try:
        cal = dp.calibrate(ag, N, q1, q2, q0, gadget)
    except RuntimeError:
        out = _calibrate_plain(eps, delta, B_R, B_O, B_y, kappa)
        out['kappa'] = kappa
        return out
    rel = cal['releases']
    s2 = rel['2']['sigma_slot']
    return dict(sigma1=rel['1']['sigma_slot'], sigma2=s2,
                sigma_yy=s2 if cal['joint'] else rel['y']['sigma_slot'], joint=bool(cal['joint']),
                rho_run=cal['rho_run'], delta_DP=cal['delta_DP'], Delta=cal['Delta'],
                Delta_noyy=cal['Delta_noyy'], Delta_G=cal['Delta_G'], kappa=kappa,
                log2_Omega_f=cal['log2_Omega_f'], source='dp.calibrate')


def n_off(cal, ell, p):
    """Section 9.5: n_off = (g* + r_1) / ell."""
    g = 2 * THETA_STAR * cal['sigma2'] * math.sqrt(p)
    r1 = cal['sigma1'] * (2 * math.sqrt(p) + 2 * math.sqrt(math.log(1 / ALPHA_L)))
    return (g + r1) / ell, g, r1


# ------------------------------------------------------------------------------------------- mechanism
def pi_O(p, dA):
    P = np.zeros((p, p))
    P[dA:, dA:] = np.eye(p - dA)
    return P


def stage1_batch(G, dA, cal, rng, R):
    """R independent stage-1 releases and the stage-1 rule.  G: (p,p) or (R,p,p).
    Returns lam (R,), Psi (R,p,p), kind (R,) in {'I','O'}, L_hat (R,), lmin1 (R,)."""
    G = np.asarray(G)
    p = G.shape[-1]
    E1 = solve.masked_noise(rng, R, p, dA, cal['sigma1'])
    Gs = G[None] + E1 if G.ndim == 2 else G + E1
    lam = np.empty(R)
    Psi = np.empty((R, p, p))
    kind = np.empty(R, dtype='<U1')
    Lh = np.empty(R)
    lm = np.empty(R)
    for r in range(R):
        s1 = solve.stage1_rule(Gs[r], dA, cal['sigma1'], cal['sigma2'], ALPHA_L, THETA_STAR, KAPPA_PSI)
        lam[r], Psi[r], kind[r], Lh[r], lm[r] = s1['lam'], s1['Psi'], s1['kind'], s1['L_hat'], s1['lmin1']
    return dict(lam=lam, Psi=Psi, kind=kind, L_hat=Lh, lmin1=lm, r1=s1['r1'], g_star=s1['g_star'])


def stage2_noise(p, dA, cal, rng, R):
    """Stage-2 noise for R releases: E (R,p,p) on the O-dependent entries, f (R,p), g (R,) on y'y."""
    E = solve.masked_noise(rng, R, p, dA, cal['sigma2'])
    f = cal['sigma2'] * rng.standard_normal((R, p))
    g = cal['sigma_yy'] * rng.standard_normal(R)
    return E, f, g


def stage2_solve(G2, c2, lam, Psi, sigma2, bias_correct=False, dA=None):
    """Gate and solve for a batch: G2 (R,p,p), c2 (R,p), lam (R,), Psi (R,p,p).
    Returns theta (R,p), gate (R,), refused (R,) and, if asked, the bias-corrected theta_bc."""
    R, p, _ = G2.shape
    Gl = G2 + lam[:, None, None] * Psi
    gate = np.linalg.eigvalsh(Gl)[:, 0] / (2 * sigma2 * math.sqrt(p))
    refused = gate < 1.0
    th = np.linalg.solve(Gl, c2[..., None])[..., 0]
    out = dict(theta=th, gate=gate, refused=refused)
    if bias_correct:
        P = np.linalg.inv(Gl)
        out['theta_bc'] = th - sigma2 ** 2 * np.einsum('rij,rj->ri', sc.bias_op(P, dA), th)
    return out


def bias_correct(G2, lam, Psi, theta, sigma2, dA):
    """Optional zero-budget correction, eq. (bc): theta - sigma_2^2 M(P~) theta."""
    Gl = G2 + lam[:, None, None] * Psi
    P = np.linalg.inv(Gl)
    return theta - sigma2 ** 2 * np.einsum('rij,rj->ri', sc.bias_op(P, dA), theta)


# ------------------------------------------------------------------------------------------- real data
def per_party_preprocess(F, y, ref, dR, preproc, q=0.995):
    """Per-party preprocessing with constants from the public reference rows `ref`, as in
    implementation/bench/utility_runs.prepare_real: standardize, optional per-party transform, bounds at
    the q-quantile of each block's norm on the reference rows (B_R includes the intercept).
    Returns Z (all rows, clipped, intercept first), yz (standardized, unclipped), yc (clipped), bounds, sy."""
    from bench import utility_runs as U
    mu, sd = F[ref].mean(0), F[ref].std(0)
    sd = np.where(sd > 0, sd, 1.0)
    my, sy = float(y[ref].mean()), float(y[ref].std())
    Fz, yz = (F - mu) / sd, (y - my) / sy
    Fz = U.transform(F, Fz, ref, dR, preproc)
    B_R = float(np.sqrt(1 + np.quantile(np.sum(Fz[ref, :dR] ** 2, 1), q)))
    B_O = float(np.quantile(np.linalg.norm(Fz[ref, dR:], axis=1), q))
    B_y = float(np.quantile(np.abs(yz[ref]), q))
    if preproc == 'blocknorm':          # exact bounds for unit blocks (the quantile is 1 up to rounding)
        B_R, B_O = math.sqrt(2.0), 1.0

    def clip_rows(X, b):
        nr = np.linalg.norm(X, axis=1)
        return X * np.minimum(1.0, b / np.maximum(nr, 1e-300))[:, None]
    Z = np.column_stack([np.ones(len(y)), clip_rows(Fz[:, :dR], math.sqrt(max(B_R ** 2 - 1, 0.0))),
                         clip_rows(Fz[:, dR:], B_O)])
    yc = np.clip(yz, -B_y, B_y)
    return Z, yz, yc, dict(B_R=B_R, B_O=B_O, B_y=B_y), sy


def zscore_design(F, y, ref):
    """Unclipped standardized design with an intercept (the common non-private reference)."""
    mu, sd = F[ref].mean(0), F[ref].std(0)
    sd = np.where(sd > 0, sd, 1.0)
    my, sy = float(y[ref].mean()), float(y[ref].std())
    return np.column_stack([np.ones(len(y)), (F - mu) / sd]), (y - my) / sy, sy


def ols(Z, y):
    return np.linalg.lstsq(Z, y, rcond=None)[0]


def ridge1(Z, y):
    """Non-private reference: (Z'Z + I)^-1 Z'y, Wang's non-private estimator (code/linreg.m).  The unit ridge is
    negligible next to Z'Z at these sizes but keeps rank-deficient designs (constant or rare-valued columns, as in
    elevators and gas) from extrapolating wildly."""
    return np.linalg.solve(Z.T @ Z + np.eye(Z.shape[1]), Z.T @ y)


def test_mse(Z, y, theta):
    th = np.atleast_2d(theta)
    r = Z @ th.T - y[:, None]
    out = np.mean(r * r, axis=0)
    return out if np.ndim(theta) == 2 else float(out[0])


# ------------------------------------------------------------------------------------------- AdaSSP
VARRHO = 0.05


def adassp_sigmas(eps, delta, d, variant):
    """Noise scales of AdaSSP's three releases (lambda_min, X'X, X'y) and its two derived constants.

    'published'  Wang's constants: s = sqrt(log(6/delta))/(eps/3) for all three releases (B_X = B_Y = 1),
                 shift = log(6/delta)/(eps/3) = s sqrt(log(6/delta)), eta = s sqrt(d log(2 d^2/varrho)).
    'replace1'   AdaSSP's algorithm recalibrated so that the three releases are exactly (eps, delta)-DP
                 under replace-one adjacency of a whole record (||x|| <= 1, |y| <= 1) with the same tight
                 zCDP conversion as v3.1: sensitivities 1 (lambda_min), sqrt(2) (X'X with Wang's (Z+Z')/2
                 noise, i.e. the Frobenius norm of x x' - z z'), 2 (X'y); rho_run split equally; the shift
                 and eta keep Wang's number of standard deviations."""
    L6 = math.log(6 / delta)
    if variant == 'published':
        s = math.sqrt(L6) / (eps / 3)
        sl = sx = sy = s
    elif variant == 'replace1':
        rho = dp.rho_tight(eps, delta)
        u = 1.0 / math.sqrt(2 * rho / 3)
        sl, sx, sy = 1.0 * u, math.sqrt(2.0) * u, 2.0 * u
    else:
        raise ValueError(variant)
    return dict(sig_lmin=sl, sig_xx=sx, sig_xy=sy, shift=sl * math.sqrt(L6),
                eta=sx * math.sqrt(d * math.log(2 * d * d / VARRHO)))


def adassp_eps_achieved(eps, delta, adjacency):
    """(eps, delta)-DP actually implied by Wang's published constants under the tight conversion, for
    add/remove adjacency (sensitivities 1, 1, 1) or replace-one adjacency (1, sqrt 2, 2)."""
    s = math.sqrt(math.log(6 / delta)) / (eps / 3)
    sens2 = 3.0 if adjacency == 'add_remove' else 1.0 + 2.0 + 4.0
    rho = sens2 / (2 * s * s)
    return dp.zcdp_to_eps(rho, delta), rho


def adassp_matched_sigmas(eps, delta, d):
    """AdaSSP (matched): AdaSSP's adaptive ridge under the v3.1 adjacency (replace-one of (x_O, y), x_R fixed,
    A exact) and the v3.1 accountant (zCDP, tight conversion).  rho_run = rho_tight(eps, delta) is split as in
    AdaSSP: rho/3 on lambda_min(X'X + I) (replace-one sensitivity 1 for unit-norm rows), 2 rho/3 on one joint
    release of the O-dependent entries (triu B, C, c_R, c_O, y'y) with Delta_rep(1, 1, 1) = sqrt(10).  Wang's
    downward shift and eta keep his number of standard deviations, with our Gram-noise s.d. in eta.
    Idealized: the private lambda_min step needs X'X in the clear, which nobody holds in the vertical setting."""
    rho = dp.rho_tight(eps, delta)
    sig = dp.delta_rep(1.0, 1.0, 1.0)[0] / math.sqrt(2 * (2 * rho / 3))
    sl = 1.0 / math.sqrt(2 * (rho / 3))
    return dict(sigma=sig, sig_lmin=sl, shift=sl * math.sqrt(math.log(6 / delta)),
                eta=sig * math.sqrt(d * math.log(2 * d * d / VARRHO)), rho=rho)


def adassp_matched(G0, c, dA, eps, delta, R, rng, lmin=None):
    """AdaSSP (matched) on X'X = G0 (Wang's preprocessing, no intercept, R's block = first dA coordinates)."""
    d = G0.shape[0]
    k = adassp_matched_sigmas(eps, delta, d)
    if lmin is None:
        lmin = float(np.linalg.eigvalsh(G0 + np.eye(d))[0])
    E = solve.masked_noise(rng, R, d, dA, k['sigma'])
    f = k['sigma'] * rng.standard_normal((R, d))
    lt = np.maximum(lmin + k['sig_lmin'] * rng.standard_normal(R) - k['shift'], 0.0)
    lam = np.maximum(0.0, k['eta'] - lt)
    Gh = G0[None] + E + (1.0 + lam)[:, None, None] * np.eye(d)[None]
    return np.linalg.solve(Gh, (c[None] + f)[..., None])[..., 0], lam


def adassp(G0, c, eps, delta, R, rng, variant='published', lmin=None):
    """AdaSSP (Wang 2018, code/adassp.m) on X'X = G0 (no intercept, Wang's preprocessing), batched over R
    noise draws; same random-number consumption as experiments/adassp/methods.adassp."""
    d = G0.shape[0]
    k = adassp_sigmas(eps, delta, d, variant)
    XTX = G0 + np.eye(d)
    if lmin is None:
        lmin = float(np.linalg.eigvalsh(XTX)[0])
    lt = np.maximum(lmin + k['sig_lmin'] * rng.standard_normal(R) - k['shift'], 0.0)
    lam = np.maximum(0.0, k['eta'] - lt)
    ch = c[None] + k['sig_xy'] * rng.standard_normal((R, d))
    Zm = rng.standard_normal((R, d, d))
    Zs = 0.5 * (Zm + np.swapaxes(Zm, 1, 2))
    Gh = XTX[None] + k['sig_xx'] * Zs + lam[:, None, None] * np.eye(d)[None]
    return np.linalg.solve(Gh, ch[..., None])[..., 0], lam


# ------------------------------------------------------------------------------------------- io / stats
def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        o = float(o)
    if isinstance(o, float) and not math.isfinite(o):
        return None if math.isnan(o) else ('inf' if o > 0 else '-inf')
    return o


def save_json(obj, path):
    with open(path, 'w') as fh:
        json.dump(jsonable(obj), fh, indent=1)


def load_json(path):
    with open(path) as fh:
        return json.load(fh)


def t95(k):
    from scipy import stats
    return float(stats.t.ppf(0.975, k - 1))


def crossing(grid, curve, thr):
    """Smallest n (log-linear interpolation) from which `curve` stays <= thr; None if never."""
    grid = np.asarray(grid, float)
    curve = np.asarray(curve, float)
    below = curve <= thr
    if not below[-1]:
        return None
    i = len(curve) - 1
    while i > 0 and below[i - 1]:
        i -= 1
    if i == 0:
        return float(grid[0])
    x0, x1 = np.log(grid[i - 1]), np.log(grid[i])
    y0, y1 = np.log(curve[i - 1]), np.log(curve[i])
    return float(np.exp(x0 + (np.log(thr) - y0) * (x1 - x0) / (y1 - y0)))
