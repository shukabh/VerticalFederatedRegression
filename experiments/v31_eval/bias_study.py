"""Section 12.6 under v3.1: does the bias correction matter?

Conditional design (the earlier study's E1, experiments/bias_correction_study.py, with the v3.1
mechanism): for each n in {500, ..., 128,000} and eps in {0.5, 1, 2, 4, 8}, five datasets of n matched
records from the R^2 = 0.5 population; for each dataset four independent stage-1 releases, each fixing
(lambda, Psi) by the stage-1 rule; for each (dataset, lambda) 10,000 antithetic pairs of stage-2
releases (E, f) and (-E, -f).  Because (lambda, Psi) comes from stage 1, it is independent of the
stage-2 noise (Proposition 9.1), which is the condition of Theorem 10.2.

Per cell we report, against the ridge target theta_lambda = (G + lambda Psi)^-1 c of each (dataset,
lambda): the mechanism bias E[theta~ | gate] - theta_lambda (antithetic pair means, debiased for
Monte Carlo noise), its share of the mean squared error, the largest |bias_j| / sd_j, and the ratio of
the MSE of the corrected estimate (eq. bc) to that of the uncorrected one; the same ratio against the
sample least-squares estimate; the number of averaged releases K at which the bias would equal the
variance of the average.  The ratio against the true coefficients comes from the cohort-level runs of
sim_accuracy.py (fresh data and releases).

    python experiments/v31_eval/bias_study.py [--pairs 10000] [--workers 4]
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
import common as cm  # noqa: E402
from common import sc, solve  # noqa: E402

N_GRID = [500, 1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000]
SEEDS, DRAWS = 5, 4
P, DA = 11, 5
CHUNK = 5000


def err_stats(ep, em, t):
    """Antithetic pairs (ep, em) of estimates, target t: bias (debiased), per-draw sd, squared errors."""
    m = len(ep)
    pm = 0.5 * (ep + em) - t
    bias = pm.mean(0)
    se = pm.std(0, ddof=1) / math.sqrt(m)
    sd = np.vstack([ep, em]).std(0, ddof=1)
    a = 0.5 * (((ep - t) ** 2).sum(1) + ((em - t) ** 2).sum(1))
    return dict(bias=bias, se=se, sd=sd, a=a, bsq=float(bias @ bias - se @ se), mse=float(a.mean()),
                trV=float((sd ** 2).sum()))


def paired_ratio(a_num, a_den):
    R = a_num.mean() / a_den.mean()
    se = (a_num - R * a_den).std(ddof=1) / (math.sqrt(len(a_num)) * a_den.mean())
    return float(R), float(se)


def compare(U, C):
    R, se = paired_ratio(C['a'], U['a'])
    return dict(mse_ratio=R, mse_ratio_se=se,
                bias_share_unc=max(U['bsq'], 0.0) / U['mse'], bias_share_bc=max(C['bsq'], 0.0) / C['mse'],
                max_bias_over_sd_unc=float(np.max(np.abs(U['bias']) / U['sd'])),
                max_bias_over_sd_bc=float(np.max(np.abs(C['bias']) / C['sd'])),
                max_z_unc=float(np.max(np.abs(U['bias']) / U['se'])),
                gain_bias=(max(U['bsq'], 0.0) - max(C['bsq'], 0.0)) / U['mse'],
                gain_var=(U['trV'] - C['trV']) / U['mse'],
                K_bias_eq_var=(U['trV'] / U['bsq']) if U['bsq'] > 0 else float('inf'),
                mse_unc=U['mse'], mse_bc=C['mse'], bsq_unc=max(U['bsq'], 0.0), trV_unc=U['trV'])


def run_cell(n, eps, pairs):
    t0 = time.time()
    pop = sc.Population(r2=0.5, seed=0)
    cl = cm.calibrate(eps, pop.B_R, pop.B_O, pop.B_y, pop.d_R, pop.d_O)
    s2 = cl['sigma2']
    per = []
    for i in range(SEEDS):
        rd = np.random.default_rng([cm.BASE_SEED, 31, n, i])          # same datasets for every eps
        Z, y, _ = pop.draw(n, rd)
        G, c = Z.T @ Z, Z.T @ y
        b_ols = np.linalg.solve(G, c)
        theta0 = np.linalg.eigvalsh(G)[0] / (2 * s2 * math.sqrt(P))
        rs = np.random.default_rng([cm.BASE_SEED, 32, n, i, int(eps * 10)])
        s1 = cm.stage1_batch(G, DA, cl, rs, DRAWS)
        for t in range(DRAWS):
            lam, Psi = float(s1['lam'][t]), s1['Psi'][t]
            tl = np.linalg.solve(G + lam * Psi, c)
            rng = np.random.default_rng([cm.BASE_SEED, 33, n, i, t, int(eps * 10)])
            parts = {k: [] for k in ('up', 'um', 'cp', 'cm')}
            keep = []
            for s0 in range(0, pairs, CHUNK):
                k = min(CHUNK, pairs - s0)
                E = solve.masked_noise(rng, k, P, DA, s2)
                f = s2 * rng.standard_normal((k, P))
                lamv = np.full(k, lam)
                Psiv = np.broadcast_to(Psi, (k, P, P))
                ok = np.ones(k, bool)
                for sign, ku, kc in ((1, 'up', 'cp'), (-1, 'um', 'cm')):
                    G2, c2 = G[None] + sign * E, c[None] + sign * f
                    st = cm.stage2_solve(G2, c2, lamv, Psiv, s2, bias_correct=True, dA=DA)
                    parts[ku].append(st['theta']); parts[kc].append(st['theta_bc'])
                    ok &= ~st['refused']
                keep.append(ok)
            keep = np.concatenate(keep)
            X = {k: np.concatenate(v)[keep] for k, v in parts.items()}
            U_l, C_l = err_stats(X['up'], X['um'], tl), err_stats(X['cp'], X['cm'], tl)
            U_o, C_o = err_stats(X['up'], X['um'], b_ols), err_stats(X['cp'], X['cm'], b_ols)
            Gl = G + lam * Psi
            per.append(dict(seed=i, draw=t, lam=lam, kind=str(s1['kind'][t]), theta0=float(theta0),
                            theta_lam=float(np.linalg.eigvalsh(Gl)[0] / (2 * s2 * math.sqrt(P))),
                            gate_pass=float(keep.mean()), vs_lam=compare(U_l, C_l), vs_ols=compare(U_o, C_o),
                            rel_bias_unc=float(math.sqrt(max(U_l['bsq'], 0.0)) / np.linalg.norm(tl)),
                            shrink_sq=float(np.sum((tl - b_ols) ** 2)),
                            sampling_sq=float(np.sum((b_ols - pop.beta_true) ** 2)),
                            pred_bias=sc.bias_op(np.linalg.inv(Gl), DA) @ tl * s2 ** 2,
                            bias_vec=U_l['bias'], bias_se=U_l['se']))
    def mean_of(key, sub):
        return float(np.mean([p[sub][key] for p in per]))
    cell = dict(n=n, eps=eps, sigma1=cl['sigma1'], sigma2=s2, theta0=float(np.mean([p['theta0'] for p in per])),
                theta_lam=float(np.mean([p['theta_lam'] for p in per])),
                lam_median=float(np.median([p['lam'] for p in per])),
                frac_ridge_on=float(np.mean([p['lam'] > 0 for p in per])),
                frac_Psi_O=float(np.mean([p['kind'] == 'O' for p in per])),
                gate_pass=float(np.mean([p['gate_pass'] for p in per])),
                pairs=pairs, seeds=SEEDS, draws=DRAWS,
                shrink_share_of_mse=float(np.mean([p['shrink_sq'] for p in per]) /
                                          np.mean([p['shrink_sq'] + p['sampling_sq'] + p['vs_lam']['mse_unc'] for p in per])),
                rel_bias_unc=float(np.mean([p['rel_bias_unc'] for p in per])))
    for sub in ('vs_lam', 'vs_ols'):
        r = np.array([p[sub]['mse_ratio'] for p in per])
        se = np.array([p[sub]['mse_ratio_se'] for p in per])
        cell[sub] = dict(mse_ratio=float(r.mean()), mse_ratio_se=float(np.sqrt((se ** 2).sum()) / len(r)),
                         mse_ratio_min=float(r.min()), mse_ratio_max=float(r.max()),
                         bias_share_unc=mean_of('bias_share_unc', sub), bias_share_unc_max=float(max(p[sub]['bias_share_unc'] for p in per)),
                         bias_share_bc=mean_of('bias_share_bc', sub),
                         max_bias_over_sd_unc=float(max(p[sub]['max_bias_over_sd_unc'] for p in per)),
                         max_bias_over_sd_bc=float(max(p[sub]['max_bias_over_sd_bc'] for p in per)),
                         gain_bias=mean_of('gain_bias', sub), gain_var=mean_of('gain_var', sub),
                         K_bias_eq_var=float(np.median([p[sub]['K_bias_eq_var'] for p in per])))
    # theory check (Theorem 10.2): predicted vs simulated bias, pooled over (dataset, lambda)
    pb = np.array([p['pred_bias'] for p in per]); sb = np.array([p['bias_vec'] for p in per])
    cell['theory_bias_norm_ratio'] = float(np.linalg.norm(sb) / np.linalg.norm(pb)) if np.linalg.norm(pb) > 0 else None
    cell['per'] = [{k: v for k, v in p.items() if k not in ('pred_bias', 'bias_vec', 'bias_se')} for p in per]
    return cell, f'n={n} eps={eps}: theta0={cell["theta0"]:.2f} lam={cell["lam_median"]:.0f} ' \
                  f'ratio={cell["vs_lam"]["mse_ratio"]:.4f} share={cell["vs_lam"]["bias_share_unc"]:.5f} [{time.time() - t0:.0f}s]'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pairs', type=int, default=10000)
    ap.add_argument('--workers', type=int, default=4)
    a = ap.parse_args()
    cells = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(run_cell, n, e, a.pairs) for n in N_GRID for e in cm.EPS]
        for f in futs:
            c, msg = f.result()
            cells.append(c)
            print(msg, flush=True)
    out = dict(settings=dict(delta=cm.DELTA, eps=cm.EPS, n_grid=N_GRID, seeds=SEEDS, stage1_draws=DRAWS,
                             pairs=a.pairs, r2=0.5, f1=cm.F1, theta_star=cm.THETA_STAR, kappa_Psi=cm.KAPPA_PSI),
               cells=cells)
    cm.save_json(out, os.path.join(cm.REPORT, 'bias_study.json'))
    print('wrote', os.path.join(cm.REPORT, 'bias_study.json'))


if __name__ == '__main__':
    main()
