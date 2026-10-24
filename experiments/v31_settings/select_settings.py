"""Choose the evaluation settings I (inference) and P (prediction) on the synthetic population only, by the rule
pre-registered in reports/v31_settings/settings_choice.md.  No real dataset is read here.

Design.  Population vfr/data.population() (experiments/sim_core.Population, seed 0); R^2 = 0.5 decides, R^2 = 0.3
and 0.9 are reported as a robustness check only.  delta = 1e-9, eps in {0.5, 1, 2, 4, 8}, n on the grid of
experiments/v31_eval/sim_accuracy.py (100 to 100,000, 14 points), 200 cohorts per n drawn with seeds independent of
the validation runs.  Common random numbers: per (n, eps) one set of unit stage-1 and stage-2 noise draws, scaled by
each candidate's sigma_1 and sigma_2, so candidates differ only through their parameters.  Every candidate runs the
stage-1 rule, the gate theta_hat >= 1 (a refused release outputs theta = 0 and still spends the budget) and the solve.

Candidates.
  I grid: r_1 exact, theta* = 2.5, kappa_Psi = 2, alpha = 0.05, alpha_L = alpha_U = alpha/20, f_1 in
          {0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50}.
  P grid: r_1 exact, kappa_Psi = 2, intervals off, theta* in {0.5, 1, 1.5, 2.5} x alpha_L in {0.0025, 0.25, 0.5}
          x f_1 in {0.1, 0.2, 0.35}.
  The previous setting (union r_1, f_1 = 0.1) is computed for reference.
Criteria (pre-registered).
  C_I(f_1) = mean over the 70 (n, eps) cells of log(RMSE_DP / RMSE_OLS): pooled RMSE of the 10 slopes against the
             true coefficients over the 200 cohorts (theta = 0 if refused), over that of least squares on the same
             cohorts.  I = argmin.
  C_P      = mean over the 70 cells of (mean test MSE of the release) / (mean test MSE of least squares on the same
             cohorts); test MSE on the population (clipped design, unclipped y).  P = argmin.

    python experiments/v31_settings/select_settings.py run [--reps 200] [--workers 4]
    python experiments/v31_settings/select_settings.py choose
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
from scommon import ev, solve  # noqa: E402

import sim_accuracy as sa  # noqa: E402  (experiments/v31_eval/sim_accuracy.py: population moments, grid)

NS = sa.NS
R2S = [0.5, 0.3, 0.9]
P, DA = 11, 5
SELDIR = os.path.join(cs.WORK, 'select')
os.makedirs(SELDIR, exist_ok=True)
SEL_TAG = 5150                     # seed tag of the selection study (validation uses other seeds)

F1_I = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
THETA_P = [0.5, 1.0, 1.5, 2.5]
ALPHAL_P = [cs.ALPHA / 20, 0.25, 0.5]
F1_P = [0.1, 0.2, 0.35]


def candidates():
    out = {'prev': cs.PREV}
    for f in F1_I:
        out[f'I:f1={f:g}'] = cs.make_setting(f, 2.5, cs.ALPHA / 20, 'exact')
    for t in THETA_P:
        for a in ALPHAL_P:
            for f in F1_P:
                out[f'P:theta={t:g},aL={a:g},f1={f:g}'] = cs.make_setting(f, t, a, 'exact', intervals=False)
    return out


def run_cell(r2, n, reps):
    t0 = time.time()
    pop = sa.population(r2)
    M = sa.pop_moments(r2)
    beta = pop.beta_true
    sl = slice(1, P)
    ss = np.random.SeedSequence([cs.BASE_SEED, SEL_TAG, int(round(r2 * 100)), n])
    ss_data, *ss_eps = ss.spawn(1 + len(cs.EPS))
    rd = np.random.default_rng(ss_data)
    Gs = np.empty((reps, P, P)); cvec = np.empty((reps, P))
    for r in range(reps):
        Z, y, _ = pop.draw(n, rd)
        Gs[r], cvec[r] = Z.T @ Z, Z.T @ y
    b_ols = np.linalg.solve(Gs, cvec[..., None])[..., 0]
    ols_sq = np.mean((b_ols[:, sl] - beta[sl]) ** 2)
    ols_mse = float(np.mean(sa.pred_mse(b_ols, M)))
    cands = candidates()
    res = dict(r2=r2, n=n, reps=reps, ols_rmse=float(math.sqrt(ols_sq)), ols_mse=ols_mse,
               lmin_G_median=float(np.median(np.linalg.eigvalsh(Gs)[:, 0])), by_eps={})
    for eps, sse in zip(cs.EPS, ss_eps):
        rng = np.random.default_rng(sse)
        Z1 = solve.masked_noise(rng, reps, P, DA, 1.0)
        Z2 = solve.masked_noise(rng, reps, P, DA, 1.0)
        f2 = rng.standard_normal((reps, P))
        row = {}
        for name, st in cands.items():
            cal = cs.calibrate(eps, pop.B_R, pop.B_O, pop.B_y, pop.d_R, pop.d_O, st['f1'])
            s1 = cs.stage1_vec(Gs + cal['sigma1'] * Z1, DA, cal['sigma1'], cal['sigma2'], st)
            th, _, gate, refused, _, _ = cs.stage2_from_unit(Gs, cvec, Z2, f2, s1['lam'], s1['Psi'], cal['sigma2'])
            sq = np.mean((th[:, sl] - beta[sl]) ** 2)
            mse = sa.pred_mse(th, M)
            row[name] = dict(rmse=float(math.sqrt(sq)), rmse_ratio=float(math.sqrt(sq / ols_sq)),
                             mse=float(mse.mean()), mse_ratio=float(mse.mean() / ols_mse),
                             refused=float(refused.mean()), ridge_on=float(np.mean(s1['lam'] > 0)),
                             lam_median=float(np.median(s1['lam'])), psiO=float(np.mean(s1['kind'] == 'O')),
                             gate_median=float(np.median(gate)), sigma1=cal['sigma1'], sigma2=cal['sigma2'],
                             r1=float(s1['r1']), g_star=float(s1['g_star']))
        res['by_eps'][str(eps)] = row
    ev.save_json(res, os.path.join(SELDIR, f'cell_r2_{r2}_n_{n}.json'))
    return f'r2={r2} n={n}: {time.time() - t0:.0f}s'


def cmd_run(a):
    assert cs.check_stage1_vec(), 'vectorized stage-1 rule differs from solve.stage1_rule'
    # radii used by the candidates (cached on disk before the workers start)
    for al in sorted({st['alpha_L'] for st in candidates().values()}):
        solve.r1_quantile_exact(DA, P - DA, al)
    for r2 in R2S:
        sa.pop_moments(r2)
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = []
        for n in sorted(NS, reverse=True):
            for r2 in R2S:
                if os.path.exists(os.path.join(SELDIR, f'cell_r2_{r2}_n_{n}.json')) and not a.force:
                    continue
                futs.append(ex.submit(run_cell, r2, n, a.reps))
        for f in futs:
            print(f.result(), flush=True)


def criteria(r2):
    cells = [ev.load_json(os.path.join(SELDIR, f'cell_r2_{r2}_n_{n}.json')) for n in NS]
    out = {}
    for name in candidates():
        lr, mr, ref, refmax, on = [], [], [], 0.0, []
        for c in cells:
            for eps in cs.EPS:
                v = c['by_eps'][str(eps)][name]
                lr.append(math.log(v['rmse_ratio']))
                mr.append(v['mse_ratio'])
                ref.append(v['refused'])
                refmax = max(refmax, v['refused'])
                on.append(v['ridge_on'])
        out[name] = dict(C_I=float(np.mean(lr)), C_P=float(np.mean(mr)), refused_mean=float(np.mean(ref)),
                         refused_max=float(refmax), ridge_on_mean=float(np.mean(on)),
                         C_I_arith_ratio=float(np.mean(np.exp(lr))))
    return out


def per_eps(r2, name):
    cells = [ev.load_json(os.path.join(SELDIR, f'cell_r2_{r2}_n_{n}.json')) for n in NS]
    out = {}
    for eps in cs.EPS:
        v = [c['by_eps'][str(eps)][name] for c in cells]
        out[str(eps)] = dict(n=NS, rmse_ratio=[x['rmse_ratio'] for x in v], mse_ratio=[x['mse_ratio'] for x in v],
                             refused=[x['refused'] for x in v], ridge_on=[x['ridge_on'] for x in v],
                             lam_median=[x['lam_median'] for x in v],
                             pred_1p10=ev.crossing(NS, [x['mse_ratio'] for x in v], 1.10),
                             coef_2x=ev.crossing(NS, [x['rmse_ratio'] for x in v], 2.0))
    return out


def cmd_choose(a):
    cands = candidates()
    crit = {str(r2): criteria(r2) for r2 in R2S}
    main = crit['0.5']
    I_names = [k for k in cands if k.startswith('I:')]
    P_names = [k for k in cands if k.startswith('P:')]
    I_best = min(I_names, key=lambda k: main[k]['C_I'])
    P_best = min(P_names, key=lambda k: main[k]['C_P'])
    pop = sa.population(0.5)
    ell = float(sa.pop_moments(0.5)['ell'])
    noff = {}
    for k in ['prev'] + I_names + P_names:
        st = cands[k]
        noff[k] = {str(e): cs.n_off(cs.calibrate(e, pop.B_R, pop.B_O, pop.B_y, pop.d_R, pop.d_O, st['f1']), ell, P, DA,
                                     st)[0] for e in cs.EPS}
    I_st, P_st = dict(cands[I_best]), dict(cands[P_best])
    edge = dict(theta_star_edge=P_st['theta_star'] in (min(THETA_P), max(THETA_P)),
                alpha_L_edge=P_st['alpha_L'] in (min(ALPHAL_P), max(ALPHAL_P)),
                f1_P_edge=P_st['f1'] in (min(F1_P), max(F1_P)), f1_I_edge=I_st['f1'] in (min(F1_I), max(F1_I)))
    r1q = {str(al): dict(exact=solve.r1_quantile_exact(DA, P - DA, al), union=solve.r1_union_unit(P, al))
           for al in sorted({st['alpha_L'] for st in cands.values()})}
    out = dict(rule='reports/v31_settings/settings_choice.md', population='vfr/data.population(), R^2 = 0.5 decides',
               delta=cs.DELTA, eps=cs.EPS, n_grid=NS, reps=ev.load_json(os.path.join(SELDIR, 'cell_r2_0.5_n_100.json'))['reps'],
               candidates=cands, criteria=crit, n_off=noff, chosen=dict(I=I_best, P=P_best), grid_edges=edge,
               r1_unit_p11=r1q, ell=ell,
               per_eps={k: per_eps(0.5, k) for k in ['prev', I_best, P_best]},
               rank_I=sorted(I_names, key=lambda k: main[k]['C_I']),
               rank_P=sorted(P_names, key=lambda k: main[k]['C_P']))
    ev.save_json(out, os.path.join(cs.REPORT, 'selection.json'))
    ev.save_json(dict(settings=dict(I=I_st, P=P_st, prev=cs.PREV), chosen=dict(I=I_best, P=P_best)),
                 os.path.join(cs.REPORT, 'settings.json'))
    print('I grid (R^2 = 0.5; C_I, lower is better; R^2 0.3 / 0.9 for robustness):')
    for k in sorted(I_names, key=lambda k: cands[k]['f1']):
        print(f"  {k:12s} C_I={main[k]['C_I']:.4f}  (0.3: {crit['0.3'][k]['C_I']:.4f}, 0.9: {crit['0.9'][k]['C_I']:.4f})"
              f"  refused {main[k]['refused_mean']:.4f}  n_off(eps=1) {noff[k]['1.0']:.0f}")
    print(f"  prev         C_I={main['prev']['C_I']:.4f}  n_off(eps=1) {noff['prev']['1.0']:.0f}")
    print('P grid, best 12 by C_P:')
    for k in out['rank_P'][:12]:
        print(f"  {k:32s} C_P={main[k]['C_P']:.4f} (0.3: {crit['0.3'][k]['C_P']:.4f}, 0.9: {crit['0.9'][k]['C_P']:.4f})"
              f"  refused mean {main[k]['refused_mean']:.4f} max {main[k]['refused_max']:.3f}")
    print(f"  prev C_P={main['prev']['C_P']:.4f}; I C_P={main[I_best]['C_P']:.4f}")
    print('chosen:', out['chosen'], 'edges:', edge)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('run')
    r.add_argument('--reps', type=int, default=200)
    r.add_argument('--workers', type=int, default=4)
    r.add_argument('--force', action='store_true')
    sub.add_parser('choose')
    a = ap.parse_args()
    {'run': cmd_run, 'choose': cmd_choose}[a.cmd](a)


if __name__ == '__main__':
    main()
