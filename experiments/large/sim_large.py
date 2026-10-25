"""Plaintext simulation of the v3.1 mechanism at 10^5 - 10^6 matched records (accuracy curves at large n).

Same mechanism as the paper's simulations (experiments/v31_eval/common.py, experiments/v31_settings/scommon.py):
calibration from vfr/dp.py (kappa 'auto', single-run ledger, delta = 1e-9, as the encrypted runs), stage-1 release and
rule, stage-2 release, gate, solve; under I also the Wald and bias-aware intervals (solve.stage2_infer, B = 2,000
antithetic pairs).  Population: the synthetic population (R^2 = 0.5) standardized ('std', settings I and P) and block-
normalized ('bn', setting P).  For each n, `reps` fresh cohorts of n matched records; each eps gets an independent
release on every cohort.

Reported per (kind, setting, eps, n): coefficient RMSE against the true beta (median), its ratio to that of
non-private least squares on the same cohorts, the population prediction error relative to least squares (median),
the fraction of releases with the ridge on, gate refusals, and under I the Wald coverage of theta_lambda and the
median half-widths of both intervals.

    python experiments/large/sim_large.py run [--reps 100] [--workers 4]
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lcommon as lc  # noqa: E402
from lcommon import cs, ev, dp, solve  # noqa: E402

NS = [12_000, 60_000, 95_000, 100_000, 200_000, 500_000, 1_000_000]
EPS = [0.5, 1.0, 2.0, 4.0, 8.0]
ARMS = [('std', 'I'), ('std', 'P'), ('bn', 'P')]
P, DA = 11, 5
CHUNK = 250_000


def calibrate(kind, setting, eps, n_bar_O=1_007_616):
    """sigma_1, sigma_2, sigma_yy of the encrypted runs' agreement (kappa auto, single-run ledger)."""
    from vfr import common as VC
    b = lc.bounds_for(kind)
    ag = dp.Agreement(n_R_max=20_000, n_bar_O=n_bar_O, d_R=4, d_O=6, eps_run=eps, delta_run=lc.DELTA, kappa='auto',
                      **b, **dp.setting(setting))
    N, q1, q2, q0, gadget = ev._ckks()
    cal = dp.calibrate(ag, N, q1, q2, q0, gadget)
    rel = cal['releases']
    s2 = rel['2']['sigma_slot']
    return dict(sigma1=rel['1']['sigma_slot'], sigma2=s2, sigma_yy=s2 if cal['joint'] else rel['y']['sigma_slot'],
                joint=bool(cal['joint']), kappa=cal['kappa'])


def cohort_stats(kind, n, rng):
    """G, c, y'y of one cohort of n matched records (clipped y), drawn in chunks."""
    pop = lc.population()
    G = np.zeros((P, P)); c = np.zeros(P); yy = 0.0
    left = n
    while left > 0:
        m = min(CHUNK, left)
        X = pop._features(m, rng)
        zR, zO, Zs = lc._transform(pop, X, kind)
        y = Zs @ pop.beta_std + pop.noise_sd * rng.standard_normal(m)
        yc = np.clip(y, -pop.B_y, pop.B_y)
        Z = np.hstack([np.ones((m, 1)), zR, zO])
        G += Z.T @ Z; c += Z.T @ yc; yy += float(yc @ yc)
        left -= m
    return G, c, yy


def run_cell(kind, n, reps, seed):
    t0 = time.time()
    rng = np.random.default_rng([seed, n, 1 if kind == 'std' else 2])
    data = [cohort_stats(kind, n, rng) for _ in range(reps)]
    Gs = np.array([d[0] for d in data]); cs_ = np.array([d[1] for d in data]); yys = np.array([d[2] for d in data])
    bt = lc.beta_true(kind)
    import p2common as pc
    M = lc.pop_moments(kind)
    th_ols = np.linalg.solve(Gs, cs_[..., None])[..., 0]
    rm_ols = np.sqrt(np.mean((th_ols - bt) ** 2, axis=1))
    mse_ols = pc.pred_mse(th_ols, M)
    out = dict(kind=kind, n=n, reps=reps, rmse_ols_median=float(np.median(rm_ols)), arms={})
    ss = np.random.SeedSequence([seed, n, 7, 1 if kind == 'std' else 2])
    for setting in [s for k, s in ARMS if k == kind]:
        st = lc.setting_dict(setting)
        for eps, sse in zip(EPS, ss.spawn(len(EPS))):
            cal = calibrate(kind, setting, eps)
            r = np.random.default_rng(sse)
            s1 = cs.stage1_batch(Gs, DA, cal, r, reps, st)
            E, f, g = ev.stage2_noise(P, DA, cal, r, reps)
            so = ev.stage2_solve(Gs + E, cs_ + f, s1['lam'], s1['Psi'], cal['sigma2'])
            th = np.where(so['refused'][:, None], 0.0, so['theta'])
            rm = np.sqrt(np.mean((th - bt) ** 2, axis=1))
            pr = pc.pred_mse(th, M) / mse_ols
            row = dict(sigma1=cal['sigma1'], sigma2=cal['sigma2'], kappa=cal['kappa'],
                       rmse_median=float(np.median(rm)), rmse_ratio=float(np.median(rm) / np.median(rm_ols)),
                       pred_ratio_median=float(np.median(pr)), pred_ratio_q90=float(np.quantile(pr, 0.9)),
                       ridge_on=float(np.mean(s1['lam'] > 0)), refused=float(so['refused'].mean()),
                       lam_median=float(np.median(s1['lam'])))
            if st['intervals']:
                cov, hb, ht, fin = [], [], [], []
                for i in range(reps):
                    if so['refused'][i]:
                        continue
                    sd = dict(lam=s1['lam'][i], Psi=s1['Psi'][i], L_hat=s1['L_hat'][i])
                    res = solve.stage2_infer(Gs[i] + E[i], cs_[i] + f[i], yys[i] + g[i], n, DA, sd, cal['sigma2'],
                                             cal['sigma_yy'], lc.bounds_for(kind)['B_y'], st['alpha'], st['alpha_L'],
                                             st['alpha_U'], np.inf, r, Bboot=2000)
                    tl = np.linalg.solve(Gs[i] + s1['lam'][i] * s1['Psi'][i], cs_[i])
                    cov.append(np.abs(res['theta'] - tl) <= res['half_theta'])
                    ht.append(res['half_theta'])
                    hb.append(res['half_beta'])
                    fin.append(bool(np.isfinite(res['half_beta']).all()))
                cov = np.array(cov, float)
                hb = np.array(hb)
                row.update(wald_coverage=float(cov.mean()) if cov.size else None,
                           half_theta_median=float(np.median(ht)) if ht else None,
                           half_beta_median=float(np.median(hb[np.isfinite(hb)])) if np.isfinite(hb).any() else None,
                           half_beta_finite=float(np.mean(fin)) if fin else None,
                           beta_covered=float(np.mean(np.abs(th[~so['refused']] - bt) <= hb)) if hb.size else None)
            out['arms'][f'{setting}_e{eps:g}'] = row
    out['seconds'] = time.time() - t0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['run'])
    ap.add_argument('--reps', type=int, default=100)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--seed', type=int, default=20261025)
    a = ap.parse_args()
    for kind in ('std', 'bn'):
        lc.pop_moments(kind)                        # build the cached moments before the workers start
    jobs = [(kind, n) for n in NS for kind in ('std', 'bn')]
    res = {}
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(run_cell, kind, n, a.reps, a.seed): (kind, n) for kind, n in jobs}
        for fu in futs:
            kind, n = futs[fu]
            r = fu.result()
            res[f'{kind}_{n}'] = r
            print(f"[sim] {kind} n={n}: {r['seconds']:.0f}s " + '; '.join(
                f"{k} rmse x{v['rmse_ratio']:.2f} pred {v['pred_ratio_median']:.4f} ridge {v['ridge_on']:.2f}"
                for k, v in r['arms'].items()), flush=True)
    lc.save_json(dict(stamp=time.strftime('%Y-%m-%d %H:%M'), reps=a.reps, ns=NS, eps=EPS, results=res),
                 os.path.join(lc.REPORT, 'sim_large.json'))
    print('wrote', os.path.join(lc.REPORT, 'sim_large.json'))


if __name__ == '__main__':
    main()
