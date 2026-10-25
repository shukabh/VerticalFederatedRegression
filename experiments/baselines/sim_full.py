"""Baseline 1 on the synthetic population (Section 13.5 design): vertical full-record SSP against v3.1.

Same cohorts and the same unit noise as experiments/v31_P2/sim_P2.py (and so as 13.5): population
sim_core.Population(0.5, seed=0), n from 100 to 100,000, 200 cohorts per n, delta = 1e-9, both preprocessings of
the same records ('std' standardized with the 13.5 bounds; 'bn' block-normalized).  For each cohort and eps the same
unit stage-1 noise, unit stage-2 noise and unit noise on c drive v3.1 (I and P) and the full-record release; the
full-record release adds unit noise on R's block A from a separate stream and uses its own sensitivities
(sens_full.delta_full) and radius (the quantile of the full noise matrix).  A refused release gives theta = 0.

Reported per (arm, population, n, eps): population test MSE divided by that of non-private least squares on the same
cohort, refusals, ridge share and median lambda; on 'std' also the median RMSE of the slopes against the true
coefficients and its ratio to the OLS floor (as Table tab:sim_rmse_v31); crossings within 10% (prediction) and of
twice the floor (coefficients); n_off of each arm.  v3.1 recomputed here must reproduce the sim_P2 cells.

    python experiments/baselines/sim_full.py run [--reps 200] [--workers 4]
    python experiments/baselines/sim_full.py summarize
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
import bcommon as bc  # noqa: E402
from bcommon import cs, ev, pc, solve  # noqa: E402

R2 = 0.5
P, DA = pc.P, pc.DA
SIMDIR = os.path.join(bc.WORK, 'sim_full')
os.makedirs(SIMDIR, exist_ok=True)
ARMS = ('v31_I', 'v31_P', 'full_I', 'full_P')
P2CELL = os.path.join(bc.ROOT, 'experiments', '_work', 'v31_P2', 'sim', 'cell_r2_0.5_n_{n}.npz')


def cal_full_pop(kind, eps, f1):
    b = pc.bounds(kind, R2)
    return bc.cal_full(eps, b['B_R'], b['B_O'], b['B_y'], True, 4, 6, f1, pc.DELTA)


def cell_path(n):
    return os.path.join(SIMDIR, f'cell_n_{n}.npz')


def run_cell(n, reps, sts):
    t0 = time.time()
    pop = pc.population(R2)
    M = {k: pc.pop_moments(k, R2) for k in pc.POPS}
    ss = np.random.SeedSequence([cs.BASE_SEED, int(round(R2 * 100)), n])          # = sim_P2.py / 13.5
    ss_data, _ss_inf, *ss_eps = ss.spawn(2 + len(bc.EPS))
    rd = np.random.default_rng(ss_data)
    G = {k: np.empty((reps, P, P)) for k in pc.POPS}
    C = {k: np.empty((reps, P)) for k in pc.POPS}
    for r in range(reps):
        Zd, yc = pc.draw_both(pop, n, rd)
        for k in pc.POPS:
            G[k][r], C[k][r] = Zd[k].T @ Zd[k], Zd[k].T @ yc
    beta = pop.beta_true
    sl = slice(1, P)
    nE = len(bc.EPS)
    out = dict(n=n, reps=reps, arms=np.array(ARMS), pops=np.array(pc.POPS))
    for k in pc.POPS:
        b = np.linalg.solve(G[k], C[k][..., None])[..., 0]
        out[f'{k}_mse_ols'] = pc.pred_mse(b, M[k])
        if k == 'std':
            out['std_rmse_ols'] = np.sqrt(np.mean((b[:, sl] - beta[sl]) ** 2, 1))
        for x, dt in (('mse', float), ('lam', float), ('refused', bool), ('kindO', bool), ('rmse', float)):
            out[f'{k}_{x}'] = np.empty((len(ARMS), nE, reps), dtype=dt)
    ss_a = np.random.SeedSequence([cs.BASE_SEED, int(round(R2 * 100)), n, 777]).spawn(nE)   # extra noise on A
    for e, (eps, sse) in enumerate(zip(bc.EPS, ss_eps)):
        rng = np.random.default_rng(sse)
        Z1 = solve.masked_noise(rng, reps, P, DA, 1.0)
        Z2 = solve.masked_noise(rng, reps, P, DA, 1.0)
        f2 = rng.standard_normal((reps, P))
        ra = np.random.default_rng(ss_a[e])
        ZA1 = bc.a_block_noise(ra, reps, P, DA, True)
        ZA2 = bc.a_block_noise(ra, reps, P, DA, True)
        for k in pc.POPS:
            for a, arm in enumerate(ARMS):
                kind, s = arm.split('_')
                st = sts[s]
                if kind == 'v31':
                    cal = pc.calibrate(k, R2, eps, st['f1'])
                    th, s1, gate, refused = pc.run_setting_on(G[k], C[k], Z1, Z2, f2, cal, st)
                else:
                    cal = cal_full_pop(k, eps, st['f1'])
                    q = bc.r1_full(P, True, st['alpha_L'])
                    th, s1, gate, refused = bc.run_release(G[k], C[k], DA, cal, st, q, Z1 + ZA1, Z2 + ZA2, f2)
                out[f'{k}_mse'][a, e] = pc.pred_mse(th, M[k])
                out[f'{k}_lam'][a, e] = s1['lam']
                out[f'{k}_refused'][a, e] = refused
                out[f'{k}_kindO'][a, e] = s1['kind'] == 'O'
                out[f'{k}_rmse'][a, e] = np.sqrt(np.mean((th[:, sl] - beta[sl]) ** 2, 1))
    np.savez_compressed(cell_path(n), **out)
    return f'n={n}: {time.time() - t0:.0f}s'


def cmd_run(a):
    sts = bc.settings()
    bc.check_settings_against_dp()
    for st in sts.values():
        bc.r1_masked(P, DA, st['alpha_L'])
        bc.r1_full(P, True, st['alpha_L'])
    for k in pc.POPS:
        pc.pop_moments(k, R2)
        for e in bc.EPS:
            for st in sts.values():
                cal_full_pop(k, e, st['f1'])
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        jobs = [ex.submit(run_cell, n, a.reps, sts) for n in sorted(pc.NS, reverse=True)
                if a.force or not os.path.exists(cell_path(n))]
        for j in jobs:
            print(j.result(), flush=True)
    print('done', flush=True)


def n_off_full(kind, eps, st):
    cal = cal_full_pop(kind, eps, st['f1'])
    ell = float(pc.pop_moments(kind, R2)['ell'])
    g = 2 * st['theta_star'] * cal['sigma2'] * math.sqrt(P)
    r1 = cal['sigma1'] * bc.r1_full(P, True, st['alpha_L'])
    return (g + r1) / ell, g, r1


def summarize():
    sts = bc.settings()
    rows = {arm: {k: {str(e): dict(pred=[], refused=[], ridge_on=[], lam_median=[], psiO=[], rmse_median=[],
                                   rmse_ratio=[]) for e in bc.EPS} for k in pc.POPS} for arm in ARMS}
    floor = []
    chk = dict(max_rel_diff_v31_mse_vs_sim_P2=0.0)
    p2_order = None
    for n in pc.NS:
        D = dict(np.load(cell_path(n)))
        floor.append(float(np.median(D['std_rmse_ols'])))
        P2 = dict(np.load(P2CELL.format(n=n)))
        p2_order = list(P2['settings'])
        for a, arm in enumerate(ARMS):
            for k in pc.POPS:
                ols = D[f'{k}_mse_ols'].mean()
                for e, eps in enumerate(bc.EPS):
                    R = rows[arm][k][str(eps)]
                    lam = D[f'{k}_lam'][a, e]
                    R['pred'].append(float(D[f'{k}_mse'][a, e].mean() / ols))
                    R['refused'].append(float(D[f'{k}_refused'][a, e].mean()))
                    R['ridge_on'].append(float(np.mean(lam > 0)))
                    R['lam_median'].append(float(np.median(lam)))
                    R['psiO'].append(float(np.mean(D[f'{k}_kindO'][a, e])))
                    if k == 'std':
                        R['rmse_median'].append(float(np.median(D['std_rmse'][a, e])))
                        R['rmse_ratio'].append(float(np.median(D['std_rmse'][a, e]) / floor[-1]))
                    if arm.startswith('v31'):
                        s = p2_order.index('P2' if arm == 'v31_P' else 'I')
                        dd = float(np.max(np.abs(D[f'{k}_mse'][a, e] / P2[f'{k}_mse'][s, e] - 1)))
                        chk['max_rel_diff_v31_mse_vs_sim_P2'] = max(chk['max_rel_diff_v31_mse_vs_sim_P2'], dd)
    for arm in ARMS:
        kind, s = arm.split('_')
        st = sts[s]
        for k in pc.POPS:
            for eps in bc.EPS:
                R = rows[arm][k][str(eps)]
                R['pred_1p10'] = ev.crossing(pc.NS, R['pred'], 1.10)
                R['pred_1p01'] = ev.crossing(pc.NS, R['pred'], 1.01)
                if k == 'std':
                    R['coef_2x'] = ev.crossing(pc.NS, R['rmse_ratio'], 2.0)
                R['refused_max'] = max(R['refused'])
                if kind == 'v31':
                    cal = pc.calibrate(k, R2, eps, st['f1'])
                    R['n_off'] = cs.n_off(cal, float(pc.pop_moments(k, R2)['ell']), P, DA, st)[0]
                    R['sigma1'], R['sigma2'] = cal['sigma1'], cal['sigma2']
                    R['q'] = bc.r1_masked(P, DA, st['alpha_L'])
                else:
                    cal = cal_full_pop(k, eps, st['f1'])
                    R['n_off'] = n_off_full(k, eps, st)[0]
                    R['sigma1'], R['sigma2'] = cal['sigma1'], cal['sigma2']
                    R['q'] = bc.r1_full(P, True, st['alpha_L'])
    # published 13.5 values of I on 'std' (same cohorts and noise): median slope RMSE must match
    pub = ev.load_json(os.path.join(cs.REPORT, 'sim_I.json'))['by_r2']['0.5']['rows']
    dd = 0.0
    for eps in bc.EPS:
        theirs = [r['rmse_median'] for r in pub if r['eps'] == eps]
        mine = rows['v31_I']['std'][str(eps)]['rmse_median']
        dd = max(dd, float(np.max(np.abs(np.array(mine) / np.array(theirs) - 1))))
    chk['max_rel_diff_v31_I_rmse_vs_sim_I'] = dd
    sens = {}
    for k in pc.POPS:
        b = pc.bounds(k, R2)
        D_, Dn, DG = bc.sens_full.delta_full(b['B_R'], b['B_O'], b['B_y'], True, 4, 6)
        Dv, Dvn = bc.dp.delta_rep(b['B_R'], b['B_O'], b['B_y'])
        sens[k] = dict(bounds=b, full=dict(Delta=D_, Delta_noyy=Dn, Delta_G=DG),
                       v31=dict(Delta=Dv, Delta_noyy=Dvn, Delta_G=bc.dp.delta_G(b['B_R'], b['B_O'])),
                       ell=float(pc.pop_moments(k, R2)['ell']))
    out = dict(settings=dict(I=sts['I'], P=sts['P'], delta=pc.DELTA, eps=bc.EPS, ns=pc.NS, r2=R2, reps=200,
                             populations=pc.POP_LABEL),
               sensitivities=sens, ols_floor_rmse_median=floor, rows=rows, reproduction_check=chk,
               radii=dict(v31={s: bc.r1_masked(P, DA, st['alpha_L']) for s, st in sts.items()},
                          full={s: bc.r1_full(P, True, st['alpha_L']) for s, st in sts.items()}))
    path = os.path.join(bc.REPORT, 'fullrecord_sim.json')
    ev.save_json(out, path)
    print('reproduction check:', chk)
    print('wrote', path)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('run')
    r.add_argument('--reps', type=int, default=200)
    r.add_argument('--workers', type=int, default=4)
    r.add_argument('--force', action='store_true')
    sub.add_parser('summarize')
    a = ap.parse_args()
    cmd_run(a) if a.cmd == 'run' else summarize()


if __name__ == '__main__':
    main()
