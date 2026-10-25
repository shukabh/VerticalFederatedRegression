"""Shared pieces of the evaluation of the two named settings of protocol v3.1 (inference I, prediction P).

Builds on experiments/v31_eval/common.py (imported as `ev`): the same plaintext simulation of the v3.1 release
and solve steps, from the implementation's own modules (vfr/dp.py Calibrate, vfr/solve.py noise, stage-1 rule,
gate and intervals).  What is new here:

  * a setting is a dict (f1, theta_star, kappa_Psi, alpha_L, alpha_U, r1_mode, intervals), as in vfr/dp.py
    SETTINGS; calibration takes f1 from it;
  * r_1 can be the exact (1 - alpha_L) quantile of lambda_max of the stage-1 noise (vfr/solve.py,
    r1_quantile_exact, an upper confidence bound from a cached Monte Carlo) instead of the union bound;
  * a vectorized stage-1 rule (stage1_vec), checked against solve.stage1_rule draw for draw.

Fixed evaluation choices (as in v31_eval): eps in {0.5, 1, 2, 4, 8}; delta = 1e-9 except 1e-6 in the AdaSSP
benchmark; kappa = 40, 40, 40, 44, 48; y'y in stage 2 when free, otherwise its own release with f_y = 0.1; bias
correction off; the gate theta_hat >= 1 (part of the protocol) with a refused release giving theta = 0.
"""
from __future__ import annotations

import math
import os
import sys
from functools import lru_cache

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, '..', '..'))
EVAL = os.path.join(ROOT, 'experiments', 'v31_eval')
if EVAL not in sys.path:
    sys.path.insert(0, EVAL)

import common as ev  # noqa: E402  (experiments/v31_eval/common.py)
from common import dp, solve, sc  # noqa: E402,F401

REPORT = os.path.join(ROOT, 'reports', 'v31_settings')
REPORT_EVAL = ev.REPORT
WORK = os.path.join(ROOT, 'experiments', '_work', 'v31_settings')
FIGDIR = ev.FIGDIR
os.makedirs(REPORT, exist_ok=True)
os.makedirs(WORK, exist_ok=True)

EPS = ev.EPS
DELTA = ev.DELTA
ALPHA = 0.05
F_Y = ev.F_Y
BASE_SEED = ev.BASE_SEED
kappa_for = ev.kappa_for


def make_setting(f1, theta_star, alpha_L, r1_mode, kappa_Psi=2.0, alpha=ALPHA, alpha_U=None, intervals=True):
    return dict(f1=float(f1), theta_star=float(theta_star), kappa_Psi=float(kappa_Psi), alpha=float(alpha),
                alpha_L=float(alpha_L), alpha_U=float(alpha / 20 if alpha_U is None else alpha_U),
                r1_mode=r1_mode, intervals=bool(intervals))


# The previous evaluation setting (reference only; its results are in reports/v31_eval).
PREV = make_setting(0.1, 2.5, ALPHA / 20, 'union')


def load_settings():
    """The chosen settings I and P (written by select_settings.py into reports/v31_settings/settings.json)."""
    path = os.path.join(REPORT, 'settings.json')
    d = ev.load_json(path)
    return {k: dict(v) for k, v in d['settings'].items()}


def label(st):
    return (f"f1={st['f1']:g}, theta*={st['theta_star']:g}, alpha_L={st['alpha_L']:g}, r1={st['r1_mode']}"
            + ('' if st.get('intervals', True) else ', no intervals'))


# ------------------------------------------------------------------------------------------- calibration
@lru_cache(maxsize=None)
def _calibrate(eps, B_R, B_O, B_y, d_R, d_O, delta, f1, n_bar_O):
    kappa = kappa_for(eps)
    N, q1, q2, q0, gadget = ev._ckks()
    ag = dp.Agreement(n_R_max=1, n_bar_O=n_bar_O, d_R=d_R, d_O=d_O, B_R=B_R, B_O=B_O, B_y=B_y, eps_run=eps,
                      delta_run=delta, f1=f1, f_y=F_Y, kappa=kappa)
    try:
        cal = dp.calibrate(ag, N, q1, q2, q0, gadget)
    except RuntimeError:
        out = ev._calibrate_plain(eps, delta, B_R, B_O, B_y, kappa, f1=f1)
        out['kappa'] = kappa
        return out
    rel = cal['releases']
    s2 = rel['2']['sigma_slot']
    return dict(sigma1=rel['1']['sigma_slot'], sigma2=s2,
                sigma_yy=s2 if cal['joint'] else rel['y']['sigma_slot'], joint=bool(cal['joint']),
                rho_run=cal['rho_run'], delta_DP=cal['delta_DP'], Delta=cal['Delta'],
                Delta_noyy=cal['Delta_noyy'], Delta_G=cal['Delta_G'], kappa=kappa,
                log2_Omega_f=cal['log2_Omega_f'], source='dp.calibrate', f1=f1)


def calibrate(eps, B_R, B_O, B_y, d_R, d_O, f1, delta=DELTA, n_bar_O=32768):
    """sigma_1, sigma_2, sigma_yy (slot domain) from the implementation's Calibrate with stage-1 share f1."""
    return dict(_calibrate(float(eps), float(B_R), float(B_O), float(B_y), int(d_R), int(d_O), float(delta),
                           float(f1), int(n_bar_O)))


# ------------------------------------------------------------------------------------------- stage 1
def r1_unit(p, dA, st):
    return solve.r1_unit(p, dA, st['alpha_L'], st['r1_mode'])


def n_off(cal, ell, p, dA, st):
    """Section 9.5: n_off = (g* + r_1) / ell, with the setting's r_1."""
    g = 2 * st['theta_star'] * cal['sigma2'] * math.sqrt(p)
    r1 = cal['sigma1'] * r1_unit(p, dA, st)
    return (g + r1) / ell, g, r1


def stage1_vec(Gs, dA, sigma1, sigma2, st):
    """Vectorized stage-1 rule over a batch Gs (R, p, p) of noisy stage-1 Grams; same output as
    solve.stage1_rule applied to each (lam, kind, L_hat, lmin1; Psi as an (R, p, p) array)."""
    Gs = np.asarray(Gs)
    R, p, _ = Gs.shape
    r1 = sigma1 * r1_unit(p, dA, st)
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
    return dict(lam=lam, Psi=Psi, kind=kind, L_hat=L_hat, lmin1=lmin1, r1=r1, g_star=g_star,
                lam_I=lam_I, lam_O=lam_O)


def stage1_batch(G, dA, cal, rng, R, st):
    """R independent stage-1 releases (noise from solve.masked_noise) and the stage-1 rule of setting st."""
    G = np.asarray(G)
    p = G.shape[-1]
    E1 = solve.masked_noise(rng, R, p, dA, cal['sigma1'])
    Gs = G[None] + E1 if G.ndim == 2 else G + E1
    return stage1_vec(Gs, dA, cal['sigma1'], cal['sigma2'], st)


def check_stage1_vec(seed=0, R=400):
    """stage1_vec equals solve.stage1_rule draw for draw (both radius modes, both ridge kinds)."""
    rng = np.random.default_rng(seed)
    pop = sc.Population(r2=0.5, seed=0)
    for n in (300, 3000, 30000):
        Z, _, _ = pop.draw(n, rng)
        G = Z.T @ Z
        for st in (PREV, make_setting(0.35, 1.0, 0.25, 'exact'), make_setting(0.2, 0.5, 0.5, 'exact', kappa_Psi=0)):
            for s1s, s2s in ((300.0, 100.0), (30.0, 10.0), (3.0, 1.0)):
                Gs = G[None] + solve.masked_noise(rng, R, 11, 5, s1s)
                v = stage1_vec(Gs, 5, s1s, s2s, st)
                for r in range(0, R, 37):
                    ref = solve.stage1_rule(Gs[r], 5, s1s, s2s, st['alpha_L'], st['theta_star'], st['kappa_Psi'],
                                            r1_mode=st['r1_mode'])
                    assert ref['kind'] == v['kind'][r], (n, st, r)
                    assert abs(ref['lam'] - v['lam'][r]) <= 1e-9 * max(1.0, ref['lam']), (ref['lam'], v['lam'][r])
                    assert abs(ref['L_hat'] - v['L_hat'][r]) <= 1e-9 * max(1.0, ref['L_hat'])
                    assert np.array_equal(ref['Psi'], v['Psi'][r])
    return True


# ------------------------------------------------------------------------------------------- stage 2
def stage2_from_unit(G, c, Z2, f2, lam, Psi, sigma2):
    """Stage-2 releases from unit noise (common random numbers across settings): G (R,p,p) or (p,p) exact,
    Z2 (R,p,p) unit masked noise, f2 (R,p) unit noise on c.  Gate and solve; refused -> theta = 0."""
    G2 = G + sigma2 * Z2
    c2 = c + sigma2 * f2
    st = ev.stage2_solve(G2, c2, lam, Psi, sigma2)
    th = np.where(st['refused'][:, None], 0.0, st['theta'])
    return th, st['theta'], st['gate'], st['refused'], G2, c2


# ------------------------------------------------------------------------------------------- r_1 cache
def _r1_job(key):
    dA, dO, al = key
    return key, solve.r1_quantile_mc(dA, dO, al, solve.r1_draws(al))


def precompute_r1(keys, workers=4):
    """Fill vfr/r1_quantiles.json for the (d_A, d_O, alpha_L) keys before parallel runs start.  Monte Carlo in
    worker processes; the parent writes the cache (no write races)."""
    from concurrent.futures import ProcessPoolExecutor
    have = solve._read_cache(solve.R1_CACHE)
    todo = []
    for dA, dO, al in sorted(set(keys), key=lambda k: -(k[0] + k[1]) ** 3 * solve.r1_draws(k[2])):
        key = solve._r1_key(dA, dO, al, solve.r1_draws(al), solve.R1_CONF, solve.R1_SEED)
        if key not in have:
            todo.append((dA, dO, al))
    if not todo:
        return 0
    from concurrent.futures import as_completed
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_r1_job, k) for k in todo]
        for fu in as_completed(futs):                  # written as each finishes (robust to interruption)
            (dA, dO, al), entry = fu.result()
            key = solve._r1_key(dA, dO, al, entry['draws'], entry['conf'], entry['seed'])
            solve._write_cache(solve.R1_CACHE, key, entry)
            print(f'r1 cache: dA={dA} dO={dO} alpha_L={al}: {entry["ucb"]:.4f} (union {entry["union"]:.4f})', flush=True)
    solve.r1_quantile_exact.cache_clear()
    return len(todo)
