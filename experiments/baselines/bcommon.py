"""Shared pieces of the three baselines (reports/baselines/README.md).

Builds on the existing pipelines, imported unchanged:
  experiments/v31_eval/common.py      (`ev`: plaintext v3.1 release and solve, AdaSSP, per-party preprocessing, io)
  experiments/v31_settings/scommon.py (`cs`: settings, calibration with f_1, vectorized stage-1 rule)
  experiments/v31_P2/p2common.py      (`pc`: the two synthetic populations, population moments)
  experiments/v31_eval/real_data.py   (`rdm`: Wang's 29 datasets, folds, reference split, paired test)
  experiments/v31_settings/real_settings.py (`rs`: unit noise streams and the v3.1 solve of 13.7)

Settings.  I and P are the paper's named settings (implementation/vfr/dp.py SETTINGS; reports/v31_P2/setting_P2.json:
'I' and 'P2', the re-selected P).  Both have f_1 = 0.25, so they share sigma_1 and sigma_2 and differ only in the
stage-1 rule (theta*, alpha_L) and in whether intervals are computed.

The implementation's radius cache (implementation/vfr/r1_quantiles.json) is read but never written: _write_cache is
disabled below.  Radii of the full-record noise matrix (no zero block for A) are computed here by the same Monte Carlo
rule and cached in reports/baselines/r1_full_quantiles.json.
"""
from __future__ import annotations

import os

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import json
import math
import sys
import tempfile
from functools import lru_cache

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, '..', '..'))
for _p in (HERE, os.path.join(ROOT, 'experiments', 'v31_P2'), os.path.join(ROOT, 'experiments', 'v31_settings'),
           os.path.join(ROOT, 'experiments', 'v31_eval'), os.path.join(ROOT, 'experiments', 'adassp')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import p2common as pc  # noqa: E402
from p2common import cs, ev, dp, solve, sc  # noqa: E402,F401
import sens_full  # noqa: E402

REPORT = os.path.join(ROOT, 'reports', 'baselines')
WORK = os.path.join(ROOT, 'experiments', '_work', 'baselines')
FIGDIR = os.path.join(ROOT, 'paper', 'figures')
os.makedirs(REPORT, exist_ok=True)
os.makedirs(WORK, exist_ok=True)

EPS = ev.EPS
F_Y = ev.F_Y
kappa_for = ev.kappa_for


def _no_write(*a, **k):
    raise RuntimeError('the baselines never write implementation/vfr/r1_quantiles.json; a radius is missing there')


solve._write_cache = _no_write


def settings():
    """{'I': ..., 'P': ...}: the paper's settings I and P (P = the re-selected P2)."""
    s = ev.load_json(os.path.join(ROOT, 'reports', 'v31_P2', 'setting_P2.json'))
    return {'I': dict(s['I']), 'P': dict(s['P2'])}


def check_settings_against_dp():
    """The settings used here equal implementation/vfr/dp.py SETTINGS['I'] and ['P']."""
    st = settings()
    for k in ('I', 'P'):
        ref = dp.setting(k)
        for f in ('f1', 'theta_star', 'kappa_Psi', 'alpha', 'alpha_L', 'alpha_U', 'r1_mode', 'intervals'):
            assert ref[f] == st[k][f], (k, f, ref[f], st[k][f])
    return True


# ------------------------------------------------------------------------------------------- radii
def r1_masked(p, dA, alpha_L):
    """v3.1's exact radius (unit noise zero on A), from the implementation's cache only (read-only)."""
    return solve.r1_quantile_exact(dA, p - dA, alpha_L)


R1_FULL_CACHE = os.path.join(REPORT, 'r1_full_quantiles.json')
R1_FULL_SEED = 20261025


def full_noise(rng, R, p, intercept, sigma=1.0):
    """Unit noise of the full-record release: symmetric, independent N(0, 1) on every entry of the upper triangle
    (diagonal included), mirrored; zero only on the (0, 0) entry n when there is an intercept."""
    M = rng.standard_normal((R, p, p))
    E = np.triu(M, 1)
    E = E + np.swapaxes(E, -1, -2)
    E[..., np.arange(p), np.arange(p)] = rng.standard_normal((R, p))
    if intercept:
        E[..., 0, 0] = 0.0
    return sigma * E


def a_block_noise(rng, R, p, dA, intercept):
    """Unit noise on R's block A only (the complement of solve.masked_noise): adding it to a masked draw gives a
    draw of full_noise.  Symmetric, N(0, 1) on A's upper triangle and diagonal, zero at (0, 0) with an intercept."""
    M = rng.standard_normal((R, dA, dA))
    E = np.triu(M, 1)
    E = E + np.swapaxes(E, -1, -2)
    E[..., np.arange(dA), np.arange(dA)] = rng.standard_normal((R, dA))
    if intercept:
        E[..., 0, 0] = 0.0
    out = np.zeros((R, p, p))
    out[:, :dA, :dA] = E
    return out


def _r1_full_mc(p, intercept, alpha_L, draws, conf=solve.R1_CONF):
    from scipy.stats import binom
    rng = np.random.default_rng([R1_FULL_SEED, int(intercept), p])
    lm = np.empty(draws)
    chunk = min(20_000, max(500, int(2e7 / p ** 2)))
    for s in range(0, draws, chunk):
        k = min(chunk, draws - s)
        lm[s:s + k] = np.linalg.eigvalsh(full_noise(rng, k, p, intercept))[:, -1]
    j = min(int(binom.ppf(conf, draws, 1 - alpha_L)), draws - 1)
    return dict(point=float(np.quantile(lm, 1 - alpha_L)), ucb=float(np.partition(lm, j)[j]), draws=draws, conf=conf,
                seed=R1_FULL_SEED, p=p, intercept=bool(intercept), alpha_L=float(alpha_L),
                union=solve.r1_union_unit(p, alpha_L))


R1_FULL_P_LARGE = 64           # above this size (gas only, p = 128, 129) the draws are capped, see r1_full_draws
R1_FULL_DRAWS_LARGE = 200_000


def r1_full_draws(p, alpha_L):
    """solve.r1_draws (at least 2,500 expected exceedances), capped at 200,000 draws for p > 64 (gas): a full p x p
    eigendecomposition at p = 129 costs about 5 ms, so 10^6 draws would take hours.  Fewer draws give a looser upper
    confidence bound (still at confidence 0.99), i.e. a slightly larger radius for the baseline on gas only."""
    d = solve.r1_draws(alpha_L)
    return min(d, R1_FULL_DRAWS_LARGE) if p > R1_FULL_P_LARGE else d


def _r1_full_key(p, intercept, alpha_L):
    return f'p={p},intercept={int(bool(intercept))},alpha_L={float(alpha_L)!r},draws={r1_full_draws(p, alpha_L)}'


def _read(path):
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


@lru_cache(maxsize=None)
def r1_full(p, intercept, alpha_L):
    """Exact-mode radius of the full-record stage-1 noise (upper 99% confidence bound on the (1 - alpha_L) quantile of
    lambda_max, same draws rule as solve.r1_quantile_exact), cached in reports/baselines/r1_full_quantiles.json."""
    key = _r1_full_key(p, intercept, alpha_L)
    hit = _read(R1_FULL_CACHE).get(key)
    if hit is not None:
        return float(hit['ucb'])
    entry = _r1_full_mc(p, intercept, alpha_L, r1_full_draws(p, alpha_L))
    _write_full(key, entry)
    return entry['ucb']


def _write_full(key, entry):
    """Read-merge-write under an exclusive lock (several processes may fill the cache)."""
    import fcntl
    with open(os.path.join(WORK, 'r1_full_quantiles.lock'), 'a') as lf:
        fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
        try:
            d = _read(R1_FULL_CACHE)
            d[key] = entry
            fd, tmp = tempfile.mkstemp(dir=REPORT, prefix='.r1f_', suffix='.json')
            with os.fdopen(fd, 'w') as fh:
                json.dump(dict(sorted(d.items())), fh, indent=1)
            os.replace(tmp, R1_FULL_CACHE)
        finally:
            fcntl.flock(lf.fileno(), fcntl.LOCK_UN)


def _r1_full_job(key):
    p, ic, al = key
    return key, _r1_full_mc(p, ic, al, r1_full_draws(p, al))


def precompute_r1_full(keys, workers=4):
    """Fill the full-record radius cache for (p, intercept, alpha_L) keys in worker processes; the parent writes."""
    from concurrent.futures import ProcessPoolExecutor, as_completed
    have = _read(R1_FULL_CACHE)
    todo = sorted({k for k in keys if _r1_full_key(*k) not in have}, key=lambda k: k[0] ** 3 * r1_full_draws(k[0], k[2]))
    if not todo:
        return 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_r1_full_job, k) for k in todo]
        for fu in as_completed(futs):
            (p, ic, al), entry = fu.result()
            _write_full(_r1_full_key(p, ic, al), entry)
            print(f'r1 full: p={p} intercept={ic} alpha_L={al}: {entry["ucb"]:.4f} (union {entry["union"]:.4f})',
                  flush=True)
    r1_full.cache_clear()
    return len(todo)


# ------------------------------------------------------------------------------------------- calibration
def calibrate_plain(eps, delta, D, Dn, DG, Dy, f1, kappa, f_y=F_Y):
    """The rho split of dp.calibrate (Algorithm Calibrate) for given sensitivities, without the CKKS scale search:
    the same slack gamma at flooding parameter kappa, the same y'y mode rule, rho_run = rho_tight(eps, delta_DP),
    sigma_s = Delta_s / sqrt(2 rho_s).  For v3.1's sensitivities it agrees with dp.calibrate to about 1e-9 relative
    (check_plain_vs_dp)."""
    joint = (D - Dn) <= 1e-9 * D
    n_rel = 2 if joint else 3
    gamma = n_rel * (2.0 ** -50 + 2.0 ** -kappa * (1 + 2.0 ** -80) + 2.0 ** -97)
    delta_DP = delta - (1 + math.exp(eps)) * gamma
    rho = dp.rho_tight(eps, delta_DP)
    if joint:
        s1, s2 = DG / math.sqrt(2 * f1 * rho), D / math.sqrt(2 * (1 - f1) * rho)
        syy = s2
    else:
        s1 = DG / math.sqrt(2 * f1 * rho)
        s2 = Dn / math.sqrt(2 * (1 - f1 - f_y) * rho)
        syy = Dy / math.sqrt(2 * f_y * rho)
    return dict(sigma1=s1, sigma2=s2, sigma_yy=syy, joint=bool(joint), rho_run=rho, delta_DP=delta_DP, Delta=D,
                Delta_noyy=Dn, Delta_G=DG, Delta_y=Dy, kappa=kappa, f1=f1, source='plain')


@lru_cache(maxsize=None)
def _cal_full(eps, B_R, B_O, B_y, intercept, d_u, d_O, f1, delta):
    D, Dn, DG = sens_full.delta_full(B_R, B_O, B_y, intercept, d_u, d_O)
    return calibrate_plain(eps, delta, D, Dn, DG, B_y ** 2, f1, kappa_for(eps))


def cal_full(eps, B_R, B_O, B_y, intercept, d_u, d_O, f1, delta):
    """Noise of the full-record release (every entry noised, whole-record replace-one sensitivity)."""
    return dict(_cal_full(float(eps), float(B_R), float(B_O), float(B_y), bool(intercept), int(d_u), int(d_O),
                          float(f1), float(delta)))


def cal_v31_plain(eps, B_R, B_O, B_y, f1, delta):
    D, Dn = dp.delta_rep(B_R, B_O, B_y)
    return calibrate_plain(eps, delta, D, Dn, dp.delta_G(B_R, B_O), B_y ** 2, f1, kappa_for(eps))


def check_plain_vs_dp(cases=((1.0, 1.0, 1.0, 5, 5), (math.sqrt(2), 1.0, 2.89, 4, 6), (5.59, 6.24, 2.89, 4, 6)),
                      delta=1e-6):
    """calibrate_plain with v3.1's sensitivities against the implementation's Calibrate (cs.calibrate)."""
    worst = 0.0
    for (BR, BO, By, dR, dO) in cases:
        for e in EPS:
            a = cal_v31_plain(e, BR, BO, By, 0.25, delta)
            b = cs.calibrate(e, BR, BO, By, dR, dO, 0.25, delta=delta)
            worst = max(worst, abs(a['sigma1'] / b['sigma1'] - 1), abs(a['sigma2'] / b['sigma2'] - 1))
    return worst


# ------------------------------------------------------------------------------------------- stage 1 and solve
def stage1_q(Gs, dA, sigma1, sigma2, st, q):
    """cs.stage1_vec with the radius r_1 = sigma_1 q given explicitly (q of the noise matrix actually released)."""
    Gs = np.asarray(Gs)
    R, p, _ = Gs.shape
    r1 = sigma1 * q
    lmin1 = np.linalg.eigvalsh(Gs)[:, 0]
    L_hat = np.maximum(0.0, lmin1 - r1)
    g_star = 2 * st['theta_star'] * sigma2 * math.sqrt(p)
    gam = g_star + r1
    lam_I = np.maximum(0.0, gam - lmin1)
    A = Gs[:, :dA, :dA]
    B = Gs[:, dA:, dA:]
    Cm = Gs[:, :dA, dA:]
    lamA = np.linalg.eigvalsh(A)[:, 0]
    ok = lamA > gam
    lam_O = np.full(R, np.inf)
    if ok.any():
        Am = A[ok] - gam * np.eye(dA)[None]
        S = B[ok] - np.swapaxes(Cm[ok], 1, 2) @ np.linalg.solve(Am, Cm[ok])
        lam_O[ok] = np.maximum(0.0, gam - np.linalg.eigvalsh(S)[:, 0])
    useO = lam_O <= st['kappa_Psi'] * lam_I
    lam = np.where(useO, lam_O, lam_I)
    PiO = ev.pi_O(p, dA)
    Psi = np.where(useO[:, None, None], PiO[None], np.eye(p)[None])
    kind = np.where(useO, 'O', 'I')
    return dict(lam=lam, Psi=Psi, kind=kind, L_hat=L_hat, lmin1=lmin1, r1=r1, g_star=g_star, lam_I=lam_I, lam_O=lam_O)


def run_release(G, c, dA, cal, st, q, Z1, Z2, f2):
    """Two releases from unit noise (Z1, Z2 on the Gram, f2 on c), the stage-1 rule with radius sigma_1 q, the gate
    and the solve; a refused release gives theta = 0.  G (p,p) or (R,p,p), c (p,) or (R,p)."""
    G = np.asarray(G)
    G1 = (G[None] if G.ndim == 2 else G) + cal['sigma1'] * Z1
    s1 = stage1_q(G1, dA, cal['sigma1'], cal['sigma2'], st, q)
    th, _, gate, refused, _, _ = cs.stage2_from_unit(G[None] if G.ndim == 2 else G, c[None] if c.ndim == 1 else c,
                                                     Z2, f2, s1['lam'], s1['Psi'], cal['sigma2'])
    return th, s1, gate, refused


def check_stage1_q(seed=0, R=300):
    """stage1_q with q = v3.1's radius reproduces cs.stage1_vec draw for draw."""
    rng = np.random.default_rng(seed)
    pop = sc.Population(r2=0.5, seed=0)
    st = settings()
    for n in (500, 5000, 50000):
        Z, _, _ = pop.draw(n, rng)
        G = Z.T @ Z
        for s in st.values():
            Gs = G[None] + solve.masked_noise(rng, R, 11, 5, 300.0)
            a = cs.stage1_vec(Gs, 5, 300.0, 200.0, s)
            b = stage1_q(Gs, 5, 300.0, 200.0, s, r1_masked(11, 5, s['alpha_L']))
            assert np.array_equal(a['lam'], b['lam']) and np.array_equal(a['kind'], b['kind'])
    return True


# ------------------------------------------------------------------------------------------- io
def save_json(obj, path):
    ev.save_json(obj, path)


def load_json(path):
    return ev.load_json(path)


def paired(a, b):
    """95% CI of the mean paired difference over folds, t with K - 1 df (experiments/v31_eval/real_data.paired)."""
    import real_data as rdm
    return rdm.paired(a, b)


def stamp():
    import datetime
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
