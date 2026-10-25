"""Section 13.6 under setting I: a short check that the mechanism bias stays negligible.

The conditional design of experiments/v31_eval/bias_study.py (the previous setting), with setting I's calibration
(f_1) and stage-1 rule (exact r_1), and 5,000 instead of 10,000 antithetic stage-2 pairs per (dataset, lambda):
n in {500, ..., 128,000} (doubling), eps in {0.5, 1, 2, 4, 8}, five datasets of n matched records (R^2 = 0.5, the
same datasets as before), four stage-1 releases per dataset.  Reported per cell: the squared mechanism bias as a
share of the MSE around theta_lambda, the largest |bias_j| / sd_j, the MSE ratio of the corrected to the uncorrected
estimate against theta_lambda, the validity ratio held by the ridge, and the check of Theorem 10.2.

    python experiments/v31_settings/bias_check.py [--pairs 5000] [--workers 4]
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
import scommon as cs  # noqa: E402
from scommon import ev, sc, solve  # noqa: E402

import bias_study as bs  # noqa: E402  (experiments/v31_eval/bias_study.py)

N_GRID = bs.N_GRID
SEEDS, DRAWS = bs.SEEDS, bs.DRAWS
P, DA = 11, 5
CHUNK = 5000


def run_cell(n, eps, pairs, st):
    t0 = time.time()
    pop = sc.Population(r2=0.5, seed=0)
    cl = cs.calibrate(eps, pop.B_R, pop.B_O, pop.B_y, pop.d_R, pop.d_O, st['f1'])
    s2 = cl['sigma2']
    per = []
    for i in range(SEEDS):
        rd = np.random.default_rng([cs.BASE_SEED, 31, n, i])          # same datasets as bias_study.py
        Z, y, _ = pop.draw(n, rd)
        G, c = Z.T @ Z, Z.T @ y
        theta0 = np.linalg.eigvalsh(G)[0] / (2 * s2 * math.sqrt(P))
        rs = np.random.default_rng([cs.BASE_SEED, 32, n, i, int(eps * 10)])
        s1 = cs.stage1_batch(G, DA, cl, rs, DRAWS, st)
        for t in range(DRAWS):
            lam, Psi = float(s1['lam'][t]), s1['Psi'][t]
            tl = np.linalg.solve(G + lam * Psi, c)
            rng = np.random.default_rng([cs.BASE_SEED, 33, n, i, t, int(eps * 10)])
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
                    so = ev.stage2_solve(G[None] + sign * E, c[None] + sign * f, lamv, Psiv, s2, bias_correct=True, dA=DA)
                    parts[ku].append(so['theta']); parts[kc].append(so['theta_bc'])
                    ok &= ~so['refused']
                keep.append(ok)
            keep = np.concatenate(keep)
            X = {k: np.concatenate(v)[keep] for k, v in parts.items()}
            U_l, C_l = bs.err_stats(X['up'], X['um'], tl), bs.err_stats(X['cp'], X['cm'], tl)
            Gl = G + lam * Psi
            per.append(dict(lam=lam, kind=str(s1['kind'][t]), theta0=float(theta0),
                            theta_lam=float(np.linalg.eigvalsh(Gl)[0] / (2 * s2 * math.sqrt(P))),
                            gate_pass=float(keep.mean()), vs_lam=bs.compare(U_l, C_l),
                            pred_bias=sc.bias_op(np.linalg.inv(Gl), DA) @ tl * s2 ** 2, bias_vec=U_l['bias']))
    on = [p for p in per if p['lam'] > 0]
    cell = dict(n=n, eps=eps, sigma1=cl['sigma1'], sigma2=s2, theta0=float(np.mean([p['theta0'] for p in per])),
                theta_lam=float(np.mean([p['theta_lam'] for p in per])),
                theta_lam_ridge_on=float(np.median([p['theta_lam'] for p in on])) if on else None,
                lam_median=float(np.median([p['lam'] for p in per])),
                frac_ridge_on=float(np.mean([p['lam'] > 0 for p in per])),
                gate_pass=float(np.mean([p['gate_pass'] for p in per])), pairs=pairs,
                mse_ratio=float(np.mean([p['vs_lam']['mse_ratio'] for p in per])),
                mse_ratio_min=float(min(p['vs_lam']['mse_ratio'] for p in per)),
                mse_ratio_max=float(max(p['vs_lam']['mse_ratio'] for p in per)),
                bias_share_unc_max=float(max(p['vs_lam']['bias_share_unc'] for p in per)),
                bias_share_unc=float(np.mean([p['vs_lam']['bias_share_unc'] for p in per])),
                max_bias_over_sd_unc=float(max(p['vs_lam']['max_bias_over_sd_unc'] for p in per)),
                K_bias_eq_var=float(np.median([p['vs_lam']['K_bias_eq_var'] for p in per])))
    pb = np.array([p['pred_bias'] for p in per]); sb = np.array([p['bias_vec'] for p in per])
    cell['theory_bias_norm_ratio'] = float(np.linalg.norm(sb) / np.linalg.norm(pb)) if np.linalg.norm(pb) > 0 else None
    return cell, f'n={n} eps={eps}: theta0={cell["theta0"]:.2f} lam={cell["lam_median"]:.0f} ' \
                  f'theta_lam={cell["theta_lam"]:.2f} share={cell["bias_share_unc_max"]:.5f} [{time.time() - t0:.0f}s]'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pairs', type=int, default=5000)
    ap.add_argument('--workers', type=int, default=4)
    a = ap.parse_args()
    st = cs.load_settings()['I']
    cells = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(run_cell, n, e, a.pairs, st) for n in N_GRID for e in cs.EPS]
        for f in futs:
            c, msg = f.result()
            cells.append(c)
            print(msg, flush=True)
    th = [c for c in cells if c['theta0'] >= 1 and c['theory_bias_norm_ratio'] is not None]
    on = [c for c in cells if c['frac_ridge_on'] > 0.5]
    summary = dict(bias_share_max=max(c['bias_share_unc_max'] for c in cells),
                   max_bias_over_sd=max(c['max_bias_over_sd_unc'] for c in cells),
                   mse_ratio_range=[min(c['mse_ratio_min'] for c in cells), max(c['mse_ratio_max'] for c in cells)],
                   theory_ratio_range=[min(c['theory_bias_norm_ratio'] for c in th), max(c['theory_bias_norm_ratio'] for c in th)],
                   K_min=min(c['K_bias_eq_var'] for c in cells),
                   theta_lam_ridge_on_range=[min(c['theta_lam_ridge_on'] for c in on), max(c['theta_lam_ridge_on'] for c in on)]
                   if on else None,
                   smallest_theta0_ridge_off=min([c['theta0'] for c in cells if c['frac_ridge_on'] == 0], default=None),
                   gate_pass_min=min(c['gate_pass'] for c in cells))
    prev = ev.load_json(os.path.join(cs.REPORT_EVAL, 'bias_study.json'))
    pc = prev['cells']
    summary['previous'] = dict(bias_share_max=max(c['vs_lam']['bias_share_unc_max'] for c in pc),
                               max_bias_over_sd=max(c['vs_lam']['max_bias_over_sd_unc'] for c in pc),
                               K_min=min(c['vs_lam']['K_bias_eq_var'] for c in pc))
    out = dict(settings=dict(setting_I=st, delta=cs.DELTA, eps=cs.EPS, n_grid=N_GRID, seeds=SEEDS, stage1_draws=DRAWS,
                             pairs=a.pairs, r2=0.5), summary=summary, cells=cells)
    path = os.path.join(cs.REPORT, 'bias_check_I.json')
    ev.save_json(out, path)
    print('wrote', path)
    print(summary)


if __name__ == '__main__':
    main()
