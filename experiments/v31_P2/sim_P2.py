"""Synthetic validation of P2 (step 1 of "what happens next" in reports/v31_P2/settings_choice_P2.md).

P2, I and the old P on both selection populations ('std' standardized with the 13.5 bounds; 'bn' block normalized),
R^2 in {0.3, 0.5, 0.9}, n from 100 to 100,000 (the 13.5 grid), 200 cohorts per n.  The cohorts and the unit noise
are those of experiments/v31_settings/sim_settings.py (same seeds, same order of random-number consumption: unit
stage-1 noise, unit stage-2 noise, unit noise on c), so on 'std' every setting reproduces the published 13.5 runs
draw for draw (checked against reports/v31_settings/sim_I.json and sim_P.json); 'bn' preprocesses the same records.
These seeds are independent of the selection (tag 5250).

Reported per (setting, population, R^2, n, eps): test-MSE ratio to non-private least squares (population test MSE,
refused -> theta = 0), gate refusals, ridge share, median lambda, Psi = Pi_O share; crossings within 10% and 1%;
the validity ratio held by the ridge; and the stage-1 switch-off at m * n_off (m = 0.6, ..., 1.4), 200 cohorts, as
sim_settings.run_noff.

    python experiments/v31_P2/sim_P2.py run [--reps 200] [--workers 4]
    python experiments/v31_P2/sim_P2.py summarize
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
import p2common as pc  # noqa: E402
from p2common import cs, ev, solve  # noqa: E402

R2S = [0.5, 0.3, 0.9]
NOFF_MULT = [0.6, 0.8, 1.0, 1.2, 1.4]
SIMDIR = os.path.join(pc.WORK, 'sim')
os.makedirs(SIMDIR, exist_ok=True)
P, DA = pc.P, pc.DA
SET_ORDER = ['P2', 'I', 'P_old']


def settings():
    s = ev.load_json(os.path.join(pc.REPORT, 'setting_P2.json'))
    return {'P2': dict(s['P2']), 'I': dict(s['I']), 'P_old': dict(s['P_old'])}


def cell_path(r2, n):
    return os.path.join(SIMDIR, f'cell_r2_{r2}_n_{n}.npz')


def run_cell(r2, n, reps, sts):
    t0 = time.time()
    pop = pc.population(r2)
    M = {k: pc.pop_moments(k, r2) for k in pc.POPS}
    ss = np.random.SeedSequence([cs.BASE_SEED, int(round(r2 * 100)), n])          # = sim_settings.py / 13.5
    ss_data, _ss_inf, *ss_eps = ss.spawn(2 + len(pc.EPS))
    rd = np.random.default_rng(ss_data)
    G = {k: np.empty((reps, P, P)) for k in pc.POPS}
    C = {k: np.empty((reps, P)) for k in pc.POPS}
    for r in range(reps):
        Zd, yc = pc.draw_both(pop, n, rd)
        for k in pc.POPS:
            G[k][r], C[k][r] = Zd[k].T @ Zd[k], Zd[k].T @ yc
    nE, nS = len(pc.EPS), len(sts)
    out = dict(r2=r2, n=n, reps=reps, settings=np.array(list(sts)), pops=np.array(pc.POPS))
    for k in pc.POPS:
        b = np.linalg.solve(G[k], C[k][..., None])[..., 0]
        out[f'{k}_mse_ols'] = pc.pred_mse(b, M[k])
        out[f'{k}_lmin_G'] = np.linalg.eigvalsh(G[k])[:, 0]
        for x, dt in (('mse', float), ('lam', float), ('gate', float), ('refused', bool), ('kindO', bool)):
            out[f'{k}_{x}'] = np.empty((nS, nE, reps), dtype=dt)
    for e, (eps, sse) in enumerate(zip(pc.EPS, ss_eps)):
        rng = np.random.default_rng(sse)
        Z1 = solve.masked_noise(rng, reps, P, DA, 1.0)       # stage 1 (cs.stage1_batch draws this first)
        Z2 = solve.masked_noise(rng, reps, P, DA, 1.0)       # stage 2 (ev.stage2_noise: E, then f, then g)
        f2 = rng.standard_normal((reps, P))
        for s, (name, st) in enumerate(sts.items()):
            for k in pc.POPS:
                cal = pc.calibrate(k, r2, eps, st['f1'])
                th, s1, gate, refused = pc.run_setting_on(G[k], C[k], Z1, Z2, f2, cal, st)
                out[f'{k}_mse'][s, e] = pc.pred_mse(th, M[k])
                out[f'{k}_lam'][s, e] = s1['lam']
                out[f'{k}_gate'][s, e] = gate
                out[f'{k}_refused'][s, e] = refused
                out[f'{k}_kindO'][s, e] = s1['kind'] == 'O'
    np.savez_compressed(cell_path(r2, n), **out)
    return f'r2={r2} n={n}: {time.time() - t0:.0f}s'


def run_noff(kind, name, st, eps, mult, reps):
    """Stage-1 rule only at n = mult * n_off (R^2 = 0.5), seeds of sim_settings.run_noff."""
    pop = pc.population(0.5)
    M = pc.pop_moments(kind, 0.5)
    cl = pc.calibrate(kind, 0.5, eps, st['f1'])
    no, g, r1 = cs.n_off(cl, float(M['ell']), P, DA, st)
    n = int(round(mult * no))
    ss = np.random.SeedSequence([cs.BASE_SEED, 7, int(eps * 10), int(mult * 100)])
    rd, rn = [np.random.default_rng(s) for s in ss.spawn(2)]
    Gs = np.empty((reps, P, P))
    for r in range(reps):
        Zd, _ = pc.draw_both(pop, n, rd)
        Gs[r] = Zd[kind].T @ Zd[kind]
    s1 = cs.stage1_batch(Gs, DA, cl, rn, reps, st)
    return dict(pop=kind, setting=name, eps=eps, mult=mult, n=n, n_off=no, g_star=g, r1=r1,
                frac_off=float(np.mean(s1['lam'] == 0)), frac_Psi_O=float(np.mean(s1['kind'] == 'O')),
                lam_median=float(np.median(s1['lam'])), lam_pred=max(0.0, g + r1 - n * float(M['ell'])))


def cmd_run(a):
    sts = settings()
    for st in sts.values():
        solve.r1_quantile_exact(DA, P - DA, st['alpha_L'])
    for r2 in R2S:
        for k in pc.POPS:
            pc.pop_moments(k, r2)
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        jobs = [ex.submit(run_cell, r2, n, a.reps, sts) for n in sorted(pc.NS, reverse=True) for r2 in R2S
                if a.force or not os.path.exists(cell_path(r2, n))]
        noff = [ex.submit(run_noff, k, nm, sts[nm], e, m, a.reps) for k in pc.POPS for nm in ('P2',)
                for e in pc.EPS for m in NOFF_MULT]
        for j in jobs:
            print(j.result(), flush=True)
        res = [j.result() for j in noff]
    ev.save_json(res, os.path.join(SIMDIR, 'noff_check.json'))
    print('done', flush=True)


# ------------------------------------------------------------------------------------------- summary
def summarize():
    sts = settings()
    names = list(sts)
    out = dict(settings=sts, eps=pc.EPS, ns=pc.NS, populations=pc.POP_LABEL, by_r2={})
    for r2 in R2S:
        rows = {nm: {k: {str(e): dict(m=[], refused=[], ridge_on=[], lam_median=[], psiO=[], gate_p05=[]) for e in pc.EPS}
                     for k in pc.POPS} for nm in names}
        held = {nm: {k: {str(e): [] for e in pc.EPS} for k in pc.POPS} for nm in names}
        for n in pc.NS:
            D = dict(np.load(cell_path(r2, n)))
            order = list(D['settings'])
            for nm in names:
                s = order.index(nm)
                for k in pc.POPS:
                    ols = D[f'{k}_mse_ols'].mean()
                    for e, eps in enumerate(pc.EPS):
                        R = rows[nm][k][str(eps)]
                        lam = D[f'{k}_lam'][s, e]
                        R['m'].append(float(D[f'{k}_mse'][s, e].mean() / ols))
                        R['refused'].append(float(D[f'{k}_refused'][s, e].mean()))
                        R['ridge_on'].append(float(np.mean(lam > 0)))
                        R['lam_median'].append(float(np.median(lam)))
                        R['psiO'].append(float(np.mean(D[f'{k}_kindO'][s, e])))
                        R['gate_p05'].append(float(np.percentile(D[f'{k}_gate'][s, e], 5)))
                        cal = pc.calibrate(k, r2, eps, sts[nm]['f1'])
                        sel = (lam > 0) & ~D[f'{k}_kindO'][s, e]
                        held[nm][k][str(eps)] += list((D[f'{k}_lmin_G'][sel] + lam[sel]) / (2 * cal['sigma2'] * math.sqrt(P)))
        for nm in names:
            for k in pc.POPS:
                for eps in pc.EPS:
                    R = rows[nm][k][str(eps)]
                    R['pred_1p10'] = ev.crossing(pc.NS, R['m'], 1.10)
                    R['pred_1p01'] = ev.crossing(pc.NS, R['m'], 1.01)
                    R['refused_max'] = max(R['refused'])
                    R['refused_mean'] = float(np.mean(R['refused']))
                    ell = float(pc.pop_moments(k, r2)['ell'])
                    R['n_off'] = cs.n_off(pc.calibrate(k, r2, eps, sts[nm]['f1']), ell, P, DA, sts[nm])[0]
                    v = held[nm][k][str(eps)]
                    R['theta_lambda_held'] = dict(median=float(np.median(v)) if v else None,
                                                  p10=float(np.percentile(v, 10)) if v else None,
                                                  p90=float(np.percentile(v, 90)) if v else None, count=len(v))
        out['by_r2'][str(r2)] = rows
        print('summarized', r2, flush=True)
    noff = os.path.join(SIMDIR, 'noff_check.json')
    if os.path.exists(noff):
        out['noff_check'] = ev.load_json(noff)
    # reproduction check on 'std': the published 13.5 runs of I and the old P (same cohorts and noise)
    chk = {}
    for nm, fn in (('I', 'sim_I.json'), ('P_old', 'sim_P.json')):
        pub = ev.load_json(os.path.join(pc.REPORT_SET, fn))
        diffs = []
        for r2 in R2S:
            pr = pub['by_r2'][str(r2)]['rows']
            for eps in pc.EPS:
                mine = out['by_r2'][str(r2)][nm]['std'][str(eps)]['m']
                theirs = [r['pred_ratio'] for r in pr if r['eps'] == eps]
                diffs += list(np.abs(np.array(mine) / np.array(theirs) - 1))
        chk[nm] = dict(max_rel_diff_pred_ratio=float(np.max(diffs)), cells=len(diffs))
    out['reproduction_check_vs_v31_settings'] = chk
    print('reproduction check:', chk, flush=True)
    path = os.path.join(pc.REPORT, 'sim_P2.json')
    ev.save_json(out, path)
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
    if a.cmd == 'run':
        cmd_run(a)
    else:
        summarize()


if __name__ == '__main__':
    main()
