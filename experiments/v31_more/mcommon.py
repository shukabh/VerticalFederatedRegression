"""Shared pieces of the follow-up experiments (paper check items #27, #30, #31, #34).

Builds on experiments/v31_settings/scommon.py (imported as `cs`) and through it on experiments/v31_eval/common.py
(`ev`): the plaintext simulation of the v3.1 release and solve steps from the implementation's own modules
(vfr/dp.py calibration, vfr/solve.py masked noise, stage-1 rule, gate, stage2_infer with its bootstrap and both
intervals).  New here:

  * releases_fixed(): many independent releases of ONE fixed dataset (G, c, y'y), so that theta_lambda is a fixed
    function of the data and of each release's lambda; every release runs solve.stage2_infer exactly as the
    implementation's R does (Bboot = 2,000 antithetic pairs, truncation on, no public bound on ||beta||);
  * binomial confidence intervals (Wilson and Clopper-Pearson) and a release-clustered interval for pooled coverage
    (coefficients of one release share its noise, so pooling them is not a binomial count).
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, '..', '..'))
SETT = os.path.join(ROOT, 'experiments', 'v31_settings')
if SETT not in sys.path:
    sys.path.insert(0, SETT)

import scommon as cs  # noqa: E402
from scommon import ev, dp, solve  # noqa: E402

REPORT = os.path.join(ROOT, 'reports', 'v31_more')
WORK = os.path.join(ROOT, 'experiments', '_work', 'v31_more')
FIGDIR = ev.FIGDIR
os.makedirs(REPORT, exist_ok=True)
os.makedirs(WORK, exist_ok=True)

DELTA = 1e-9
TARGET = 0.955          # per-coefficient level 1 - alpha_CI of the Wald interval (alpha = 0.05, alpha_L = alpha_U = 0.0025)
BBOOT = 2000            # bootstrap antithetic pairs, the implementation's default (vfr/party_r.py)


def setting(name):
    """Setting as a scommon-style dict from vfr/dp.py SETTINGS (alpha_L None -> alpha/20)."""
    s = dp.setting(name)
    return cs.make_setting(s['f1'], s['theta_star'], s['alpha'] / 20 if s['alpha_L'] is None else s['alpha_L'],
                           s['r1_mode'], kappa_Psi=s['kappa_Psi'], alpha=s['alpha'],
                           alpha_U=s['alpha'] / 20 if s['alpha_U'] is None else s['alpha_U'],
                           intervals=s['intervals'])


# ------------------------------------------------------------------------------------------- statistics
def wilson(k, n, z=1.959963984540054):
    if n == 0:
        return (float('nan'), float('nan'))
    ph = k / n
    den = 1 + z * z / n
    mid = (ph + z * z / (2 * n)) / den
    hw = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return (mid - hw, mid + hw)


def clopper_pearson(k, n, a=0.05):
    from scipy import stats
    lo = 0.0 if k == 0 else float(stats.beta.ppf(a / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(stats.beta.ppf(1 - a / 2, k + 1, n - k))
    return (lo, hi)


def coverage_summary(cov, target=TARGET):
    """cov: (R, p) array of 0/1 with NaN for refused releases.  Pooled coverage with (i) the naive binomial
    (Wilson) interval over all coefficient-releases and (ii) a release-clustered 95% interval (normal approximation
    on the per-release coverage fractions, which are i.i.d. for fixed data); per-coefficient coverage with Wilson
    intervals; a two-sided exact binomial test of each coefficient against the target."""
    from scipy import stats
    cov = np.asarray(cov, float)
    ok = ~np.isnan(cov).any(axis=1)
    C = cov[ok]
    R, p = C.shape
    if R == 0:
        return dict(releases=0)
    k, N = int(C.sum()), int(C.size)
    frac = C.mean(axis=1)
    se_cl = float(frac.std(ddof=1) / math.sqrt(R)) if R > 1 else float('nan')
    pc = C.mean(axis=0)
    kc = C.sum(axis=0).astype(int)
    pv = [float(stats.binomtest(int(x), R, target).pvalue) for x in kc]
    return dict(releases=int(R), refused=int((~ok).sum()), covered=k, total=N, coverage=k / N,
                wilson=wilson(k, N), clustered=(k / N - 1.96 * se_cl, k / N + 1.96 * se_cl),
                design_effect=float((se_cl ** 2) / ((k / N) * (1 - k / N) / N)) if 0 < k < N else None,
                per_coef=pc.tolist(), per_coef_min=float(pc.min()), per_coef_max=float(pc.max()),
                per_coef_wilson=[wilson(int(x), R) for x in kc], per_coef_binom_p=pv,
                per_coef_below_target_p05=int(sum((pc[j] < target) and (pv[j] < 0.05) for j in range(p))),
                pooled_binom_p_naive=float(stats.binomtest(k, N, target).pvalue))


# ------------------------------------------------------------------------------------------- mechanism
def releases_fixed(G, c, yy, n, dA, cal, st, B_y, R, seed, Bboot=BBOOT, beta_ref=None):
    """R independent two-release runs of the v3.1 mechanism on fixed data (G, c, yy), setting st, calibrated
    sigma_1, sigma_2, sigma_yy in `cal`.  Returns per-release arrays: lam, kindO, L_hat, refused, gate, theta,
    theta_lam, half_theta, half_beta, covered (Wald for theta_lambda), and, if beta_ref is given, covered_beta
    (bias-aware interval for beta_ref; an unbounded interval counts as covering) and finite_beta."""
    G = np.asarray(G, float)
    c = np.asarray(c, float)
    p = G.shape[0]
    ss = np.random.SeedSequence(seed)
    s_stage, s_inf = ss.spawn(2)
    rng = np.random.default_rng(s_stage)
    rng_inf = np.random.default_rng(s_inf)
    s1 = cs.stage1_batch(G, dA, cal, rng, R, st)
    E, f, g = ev.stage2_noise(p, dA, cal, rng, R)
    G2, c2, yy2 = G[None] + E, c[None] + f, yy + g
    out = dict(lam=s1['lam'], kindO=(s1['kind'] == 'O'), L_hat=s1['L_hat'], refused=np.zeros(R, bool),
               gate=np.full(R, np.nan), theta=np.full((R, p), np.nan), half_theta=np.full((R, p), np.nan),
               half_beta=np.full((R, p), np.nan), r1=np.array(s1['r1']), g_star=np.array(s1['g_star']))
    out['theta_lam'] = np.linalg.solve(G[None] + s1['lam'][:, None, None] * s1['Psi'], np.broadcast_to(c, (R, p))[..., None])[..., 0]
    for r in range(R):
        sd = dict(lam=s1['lam'][r], Psi=s1['Psi'][r], L_hat=s1['L_hat'][r])
        res = solve.stage2_infer(G2[r], c2[r], yy2[r], n, dA, sd, cal['sigma2'], cal['sigma_yy'], B_y,
                                 st['alpha'], st['alpha_L'], st['alpha_U'], np.inf, rng_inf, Bboot=Bboot,
                                 intervals=st['intervals'])
        out['gate'][r] = res['theta_hat_gate']
        if res['refused']:
            out['refused'][r] = True
            continue
        out['theta'][r] = res['theta']
        if st['intervals']:
            out['half_theta'][r], out['half_beta'][r] = res['half_theta'], res['half_beta']
    cov = (np.abs(out['theta'] - out['theta_lam']) <= out['half_theta']).astype(float)
    cov[out['refused']] = np.nan
    out['covered'] = cov
    if beta_ref is not None:
        cb = (np.abs(out['theta'] - np.asarray(beta_ref)[None]) <= out['half_beta']).astype(float)
        cb[out['refused']] = np.nan
        out['covered_beta'] = cb
        out['finite_beta'] = np.isfinite(out['half_beta'])
    return out


def merge(chunks):
    out = {}
    for k in chunks[0]:
        if k in ('r1', 'g_star'):
            out[k] = chunks[0][k]
        else:
            out[k] = np.concatenate([ch[k] for ch in chunks])
    return out


def releases_parallel(G, c, yy, n, dA, cal, st, B_y, R, seed, workers=4, chunk=250, beta_ref=None, Bboot=BBOOT):
    """releases_fixed in chunks over worker processes; chunk i uses seed (seed..., i)."""
    from concurrent.futures import ProcessPoolExecutor
    seeds = [list(np.atleast_1d(seed)) + [i] for i in range(-(-R // chunk))]
    sizes = [min(chunk, R - i * chunk) for i in range(len(seeds))]
    if workers <= 1:
        return merge([releases_fixed(G, c, yy, n, dA, cal, st, B_y, s, sd, Bboot, beta_ref) for s, sd in zip(sizes, seeds)])
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(releases_fixed, G, c, yy, n, dA, cal, st, B_y, s, sd, Bboot, beta_ref) for s, sd in zip(sizes, seeds)]
        return merge([f.result() for f in futs])
