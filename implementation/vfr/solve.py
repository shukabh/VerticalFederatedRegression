"""Phase 5 at R: Algorithm 6 (Solve and infer), corrected version of paper/algorithms.tex.

Port of the proof review's scratch_stats/alg6.py (same random-number consumption, so the stage-2 estimate
and intervals can be compared draw for draw), with the corrections of the paper:
  * r_1 one-sided radius sigma_1 (2 sqrt p + 2 sqrt(ln(1/alpha_L))) (r1_mode 'union', the default);
  * optionally (r1_mode 'exact') r_1 = sigma_1 q, q an upper confidence bound on the (1 - alpha_L) quantile of
    lambda_max of the unit stage-1 noise matrix, computed offline by Monte Carlo from the public noise structure
    (it depends on (d_A, d_O, alpha_L) only, so it costs no privacy; see r1_quantile_exact);
  * lambda_O by the closed form of Lemma S3(c) instead of bisection (paper tocheck: "replace the bisection");
  * bootstrap truncation ||P~^{1/2} E* P~^{1/2}||_op <= 1/2 (paper: required for the moments to exist);
  * U_yy = min{ yy^ + z_{1-alpha_U/2} sigma_yy + B_y^2 sqrt(n ln(2/alpha_U)/2), n B_y^2 }, sigma_yy = sigma_2 (joint)
    or sigma_y (separate)  (alg6.py's U_spec uses z_{1-alpha_U} and ln(1/alpha_U); both are returned).
"""
from __future__ import annotations

import json
import math
import os
import tempfile
from functools import lru_cache

import numpy as np
from scipy.optimize import brentq
from scipy.stats import binom, norm

Z = norm.ppf


def masked_noise(rng, R, p, dA, sigma):
    """Same draws as alg6.masked_noise."""
    M = rng.standard_normal((R, p, p))
    E = np.triu(M, 1)
    E = E + np.swapaxes(E, -1, -2)
    E[..., np.arange(p), np.arange(p)] = rng.standard_normal((R, p))
    E[..., :dA, :dA] = 0.0
    return sigma * E


# =============================================================================================
# The stage-1 radius r_1: union bound, or the exact (1 - alpha_L) quantile by Monte Carlo
# =============================================================================================

R1_MODES = ('union', 'exact')
R1_CONF = 0.99                 # confidence of the upper bound on the Monte Carlo quantile
R1_MIN_DRAWS = 200_000
R1_EXCEED = 2_500              # draws: at least this many expected exceedances of the quantile
R1_MAX_DRAWS = 4_000_000
R1_SEED = 20261024             # public; the quantile is a function of public quantities only
R1_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'r1_quantiles.json')
_CHUNK = 20_000


def r1_union_unit(p, alpha_L):
    """Union-bound radius in units of sigma_1: 2 sqrt p + 2 sqrt(ln(1/alpha_L))."""
    return 2 * math.sqrt(p) + 2 * math.sqrt(math.log(1 / alpha_L))


def r1_draws(alpha_L):
    """Monte Carlo draws for the exact quantile: at least R1_EXCEED expected exceedances, at least
    R1_MIN_DRAWS, at most R1_MAX_DRAWS."""
    return int(min(R1_MAX_DRAWS, max(R1_MIN_DRAWS, math.ceil(R1_EXCEED / alpha_L))))


def _r1_key(dA, dO, alpha_L, draws, conf, seed):
    return f'dA={dA},dO={dO},alpha_L={float(alpha_L)!r},draws={draws},conf={float(conf)!r},seed={seed}'


def _read_cache(path):
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


def _write_cache(path, key, entry):
    """Atomic read-merge-write; a lost race only costs a recomputation later."""
    try:
        d = _read_cache(path)
        d[key] = entry
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), prefix='.r1_', suffix='.json')
        with os.fdopen(fd, 'w') as fh:
            json.dump(dict(sorted(d.items())), fh, indent=1)
        os.replace(tmp, path)
    except OSError:
        pass


def r1_quantile_mc(dA, dO, alpha_L, draws, conf=R1_CONF, seed=R1_SEED):
    """Monte Carlo of lambda_max(E) for the unit stage-1 noise matrix E: symmetric (p x p, p = dA + dO), zero on
    R's block A (dA x dA), independent N(0, 1) on the released entries (upper triangle of B with its diagonal,
    and C), mirrored.  Returns the point estimate of the (1 - alpha_L) quantile and an upper confidence bound at
    level conf: the k-th order statistic with k - 1 the conf-quantile of Binomial(draws, 1 - alpha_L), so that
    P[ucb >= true quantile] >= conf."""
    p = dA + dO
    rng = np.random.default_rng([seed, dA, dO])
    lm = np.empty(draws)
    for s in range(0, draws, _CHUNK):
        k = min(_CHUNK, draws - s)
        lm[s:s + k] = np.linalg.eigvalsh(masked_noise(rng, k, p, dA, 1.0))[:, -1]
    j = int(binom.ppf(conf, draws, 1 - alpha_L))          # 0-based index of the order statistic
    j = min(j, draws - 1)
    point = float(np.quantile(lm, 1 - alpha_L))
    ucb = float(np.partition(lm, j)[j])
    return dict(point=point, ucb=ucb, draws=draws, conf=conf, seed=seed, index=j, dA=dA, dO=dO,
                alpha_L=float(alpha_L), union=r1_union_unit(p, alpha_L))


@lru_cache(maxsize=None)
def r1_quantile_exact(dA, dO, alpha_L, draws=None, conf=R1_CONF, seed=R1_SEED, cache_path=R1_CACHE):
    """Exact-mode radius in units of sigma_1: the upper confidence bound on the (1 - alpha_L) quantile of
    lambda_max of the unit stage-1 noise matrix (r1_quantile_mc).  Depends on (dA, dO, alpha_L) only.
    Cached in memory and in vfr/r1_quantiles.json (keyed by every input, so a cached value is reproducible)."""
    if not 0 < alpha_L < 1:
        raise ValueError('alpha_L must lie in (0, 1)')
    dA, dO = int(dA), int(dO)
    draws = r1_draws(alpha_L) if draws is None else int(draws)
    key = _r1_key(dA, dO, alpha_L, draws, conf, seed)
    if cache_path:
        hit = _read_cache(cache_path).get(key)
        if hit is not None:
            return float(hit['ucb'])
    entry = r1_quantile_mc(dA, dO, alpha_L, draws, conf, seed)
    if cache_path:
        _write_cache(cache_path, key, entry)
    return entry['ucb']


def r1_unit(p, dA, alpha_L, mode='union'):
    """r_1 / sigma_1 for the given mode."""
    if mode == 'union':
        return r1_union_unit(p, alpha_L)
    if mode == 'exact':
        return r1_quantile_exact(dA, p - dA, alpha_L)
    raise ValueError(f'r1_mode must be one of {R1_MODES}, not {mode!r}')


def lambda_O_closed(A, B1, C1, gamma):
    """min{lam >= 0: lam_min(G1 + lam Pi_O) >= gamma} (Lemma S3(c)); inf if lam_min(A) <= gamma."""
    lamA = np.linalg.eigvalsh(A)[0]
    if lamA <= gamma:
        return np.inf
    S = B1 - C1.T @ np.linalg.solve(A - gamma * np.eye(A.shape[0]), C1)
    return max(0.0, gamma - np.linalg.eigvalsh(S)[0])


def cv_alpha(t, alpha):
    t = abs(t)
    f = lambda c: norm.cdf(c - t) - norm.cdf(-c - t) - (1 - alpha)
    return brentq(f, 0.0, t + 10.0)


def assemble(A, B_triu, C):
    """G from R's exact A, the released upper triangle of B (mirrored) and C."""
    dA, dO = A.shape[0], B_triu.shape[0]
    B = np.triu(B_triu) + np.triu(B_triu, 1).T
    G = np.zeros((dA + dO, dA + dO))
    G[:dA, :dA] = A
    G[:dA, dA:] = C
    G[dA:, :dA] = C.T
    G[dA:, dA:] = B
    return G


def stage1_rule(G1, dA, sigma1, sigma2, alpha_L, theta_star, kappa_Psi, r1_mode='union'):
    """Stage-1 ridge rule.  r1_mode 'union': r_1 = sigma_1 (2 sqrt p + 2 sqrt(ln(1/alpha_L))); 'exact':
    r_1 = sigma_1 times the Monte Carlo (1 - alpha_L) quantile of lambda_max of the unit noise (upper bound)."""
    p = G1.shape[0]
    r1 = sigma1 * r1_unit(p, dA, alpha_L, r1_mode)
    lmin1 = np.linalg.eigvalsh(G1)[0]
    L_hat = max(0.0, lmin1 - r1)
    g_star = 2 * theta_star * sigma2 * np.sqrt(p)
    lam_I = max(0.0, g_star + r1 - lmin1)
    lam_O = lambda_O_closed(G1[:dA, :dA], G1[dA:, dA:], G1[:dA, dA:], g_star + r1)
    PiO = np.zeros((p, p))
    PiO[dA:, dA:] = np.eye(p - dA)
    if lam_O <= kappa_Psi * lam_I:
        lam, Psi, kind = lam_O, PiO, 'O'
    else:
        lam, Psi, kind = lam_I, np.eye(p), 'I'
    return dict(r1=r1, L_hat=L_hat, g_star=g_star, lam=float(lam), Psi=Psi, kind=kind, lam_I=float(lam_I),
                lam_O=float(lam_O), lmin1=float(lmin1), r1_mode=r1_mode)


def rss_fun(G, c, yy, lam, Psi):
    th = np.linalg.solve(G + lam * Psi, c[..., None])[..., 0]
    return yy - np.einsum('...i,...i->...', c, th) - lam * np.einsum('...i,ij,...j->...', th, Psi, th), th


def stage2_infer(G2, c2, yy2, n, dA, s1, sigma2, sigma_yy, B_y, alpha, alpha_L, alpha_U, B_pub, rng,
                 Bboot=2000, truncate=True, antithetic=True, intervals=True):
    """Gate, estimate, parametric bootstrap, RSS correction, both intervals.  rng: numpy Generator.
    intervals=False (a setting without the interval guarantee, e.g. the prediction setting): gate and estimate
    only; alpha and alpha_U are then unused and alpha_L only sized the stage-1 radius."""
    p = G2.shape[0]
    lam, Psi = s1['lam'], s1['Psi']
    Gl2 = G2 + lam * Psi
    gate = np.linalg.eigvalsh(Gl2)[0] / (2 * sigma2 * np.sqrt(p))
    out = dict(theta_hat_gate=float(gate))
    if gate < 1:
        out['refused'] = True
        return out
    out['refused'] = False
    P2 = np.linalg.inv(Gl2)
    th = P2 @ c2
    RSS = yy2 - c2 @ th - lam * th @ Psi @ th
    if not intervals:
        out.update(theta=th, RSS=float(RSS), P2=P2, shrinkage=-lam * P2 @ Psi @ th,
                   sigma2_trP=float(sigma2 * np.trace(P2)), intervals=False)
        return out
    a_CI = alpha - alpha_L - alpha_U
    if not a_CI > 0:
        raise ValueError('alpha_CI = alpha - alpha_L - alpha_U must be positive when intervals are computed')
    Es = masked_noise(rng, Bboot, p, dA, sigma2)
    fs = sigma2 * rng.standard_normal((Bboot, p))
    gs = sigma_yy * rng.standard_normal(Bboot)
    if antithetic:
        Es = np.concatenate([Es, -Es]); fs = np.concatenate([fs, -fs]); gs = np.concatenate([gs, -gs])
    Gb = G2[None] + Es
    keep = np.ones(len(Gb), bool)
    if truncate:
        w, U = np.linalg.eigh(Gl2)
        Wm = U @ np.diag(w ** -0.5) @ U.T
        nr = np.abs(np.linalg.eigvalsh(Wm[None] @ Es @ Wm[None]))
        keep = nr.max(axis=1) <= 0.5
    RSSb, thb = rss_fun(Gb[keep], c2[None] + fs[keep], yy2 + gs[keep], lam, Psi)
    Vmech = np.cov(thb, rowvar=False)
    Corr = RSSb.mean() - RSS
    s2 = max(RSS - Corr, 0.0) / (n - p)
    z = Z(1 - a_CI / 2)
    se_m = np.sqrt(np.diag(Vmech))
    V = Vmech + s2 * P2 @ G2 @ P2
    U_spec = yy2 + Z(1 - alpha_U) * sigma_yy + B_y ** 2 * np.sqrt(n * np.log(1 / alpha_U) / 2)
    U_yy = min(yy2 + Z(1 - alpha_U / 2) * sigma_yy + B_y ** 2 * np.sqrt(n * np.log(2 / alpha_U) / 2), n * B_y ** 2)
    L_hat = s1['L_hat']
    priv = np.sqrt(U_yy / L_hat) if L_hat > 0 else np.inf
    Bb = min(B_pub, priv)
    bbar = lam * np.linalg.norm(Psi @ P2, axis=0) * Bb
    sdV = np.sqrt(np.diag(V))
    half_beta = np.array([cv_alpha(bbar[j] / sdV[j], a_CI) * sdV[j] if np.isfinite(bbar[j]) else np.inf
                          for j in range(p)])
    out.update(theta=th, RSS=float(RSS), Corr=float(Corr), s2=float(s2), Vmech=Vmech, V=V, half_theta=z * se_m,
               half_beta=half_beta, U_yy=float(U_yy), U_spec=float(U_spec), B_beta=float(Bb), bbar=bbar, P2=P2,
               kept=float(keep.mean()), shrinkage=-lam * P2 @ Psi @ th, sigma2_trP=float(sigma2 * np.trace(P2)),
               z=float(z), alpha_CI=float(a_CI), intervals=True)
    return out
