"""Baseline 2 on real data: FWS intervals against v3.1's intervals (setting I) with a large-n least-squares reference.

Datasets with many rows and a linear signal: 3droad, protein, bike and elevators (Wang's versions,
experiments/adassp/datasets.py).  In fold 0 of the 13.7 split, the reference rows (10%) fix the per-party block
normalization and the bounds (as in 13.7); every other row forms the finite population.  The reference coefficient is
the least-squares fit of the clipped y on the clipped block-normalized design over the whole population (the
working-model coefficient of the clipped data, FWS's target and the bias-aware interval's).  Cohorts of n rows are
drawn without replacement from the population (200 per n), and on each cohort v3.1 under setting I, its two intervals,
FWS on the same stage-2 statistics and FWS on a single full-budget release are computed exactly as in sim_fws.py,
delta = 1e-9.  The cohort is at most a sixth of the population, so the finite-population correction to the sampling
variance is at most 17% (it makes every interval for the population coefficient slightly conservative).

    python experiments/baselines/real_fws.py run [--reps 200] [--workers 4]
    python experiments/baselines/real_fws.py summarize
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bcommon as bc  # noqa: E402
from bcommon import cs, ev  # noqa: E402
import sim_fws as sf  # noqa: E402
import real_data as rdm  # noqa: E402

DELTA = 1e-9
GRID = {'3droad': [1000, 5000, 25000], 'protein': [1000, 2500, 5000], 'bike': [500, 1000, 2000],
        'elevators': [500, 1000, 2000]}
WORKDIR = os.path.join(bc.WORK, 'real_fws')
os.makedirs(WORKDIR, exist_ok=True)


def population(name):
    D = rdm.Data(name)
    te, ref, m = D.split(0)
    pop_rows = np.sort(np.concatenate([te, m]))
    Z, yz, yc, b, sy = ev.per_party_preprocess(D.F, D.y, ref, D.dR, 'blocknorm')
    Zp, ycp = Z[pop_rows], yc[pop_rows]
    beta_pop = np.linalg.lstsq(Zp, ycp, rcond=None)[0]
    return D, Zp, ycp, b, beta_pop


def cell_path(name, n):
    return os.path.join(WORKDIR, f'{name}_n_{n}.npz')


def run_cell(name, n, reps, st):
    t0 = time.time()
    D, Zp, ycp, b, beta_pop = population(name)
    N, p = Zp.shape
    dA = D.dR + 1
    ss = np.random.SeedSequence([cs.BASE_SEED, 31, n, sum(map(ord, name))])
    ss_d, ss_inf, *ss_e = ss.spawn(2 + 4 * len(bc.EPS))
    rd = np.random.default_rng(ss_d)
    Gs = np.empty((reps, p, p)); cvec = np.empty((reps, p)); yys = np.empty(reps)
    b_ols = np.empty((reps, p)); se_ols = np.empty((reps, p))
    for r in range(reps):
        idx = rd.choice(N, n, replace=False)
        Z, y = Zp[idx], ycp[idx]
        Gs[r], cvec[r], yys[r] = Z.T @ Z, Z.T @ y, float(y @ y)
        b_ols[r] = np.linalg.lstsq(Z, y, rcond=None)[0]
        rss = float(np.sum((y - Z @ b_ols[r]) ** 2))
        se_ols[r] = np.sqrt(rss / (n - p) * np.diag(np.linalg.pinv(Gs[r])))
    rng_inf = np.random.default_rng(ss_inf)
    out = dict(name=name, n=n, N=N, reps=reps, b_ols=b_ols, se_ols=se_ols, beta_pop=beta_pop, eps=np.array(bc.EPS),
               bounds=np.array([b['B_R'], b['B_O'], b['B_y']]))
    keys = None
    for k, eps in enumerate(bc.EPS):
        cal = cs.calibrate(eps, b['B_R'], b['B_O'], b['B_y'], D.dR, D.dO, st['f1'], delta=DELTA)
        single = sf.sigma_single(eps, b['B_R'], b['B_O'], b['B_y'], DELTA)
        rs_ = [np.random.default_rng(s) for s in ss_e[4 * k:4 * k + 4]]
        o = sf.intervals_on_cohorts(Gs, cvec, yys, n, dA, cal, single, st, b['B_y'], rs_[0], rng_inf, *rs_[1:])
        if keys is None:
            keys = list(o)
            for kk in keys:
                out[kk] = np.empty((len(bc.EPS),) + np.shape(o[kk]), dtype=np.asarray(o[kk]).dtype)
        for kk in keys:
            out[kk][k] = o[kk]
        out.setdefault('sigma2', []).append(cal['sigma2'])
        out.setdefault('sigma1', []).append(cal['sigma1'])
        out.setdefault('sigma_single', []).append(single['sigma'])
    for kk in ('sigma1', 'sigma2', 'sigma_single'):
        out[kk] = np.array(out[kk])
    np.savez_compressed(cell_path(name, n), **out)
    return f'{name} n={n}: {time.time() - t0:.0f}s'


def cmd_run(a):
    st = bc.settings()['I']
    names = a.datasets or list(GRID)
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        jobs = [ex.submit(run_cell, nm, n, a.reps, st) for nm in names for n in sorted(GRID[nm], reverse=True)
                if a.force or not os.path.exists(cell_path(nm, n))]
        for j in jobs:
            print(j.result(), flush=True)
    print('done', flush=True)


def summarize():
    out = dict(settings=dict(I=bc.settings()['I'], delta=DELTA, eps=bc.EPS, grid=GRID, reps=200, B_fws=sf.B_FWS,
                             preprocessing='per-party block unit norm (13.7 arm (d)), fold-0 reference rows'),
               datasets={})
    for nm in GRID:
        cells = []
        for n in GRID[nm]:
            D = dict(np.load(cell_path(nm, n)))
            cells.append((n, D))
        D0 = cells[0][1]
        p = D0['b_ols'].shape[1]
        sm = sf.summarize_cells(cells, D0['beta_pop'], D0['beta_pop'], nm)
        # beta_true is the same population fit here; keep one name for the reference coefficient
        for r in sm['rows']:
            for mth in r['slopes']:
                r['slopes'][mth].pop('cov_beta_true', None)
                r['slopes'][mth].pop('cov_beta_true_min_coef', None)
        sm['N'] = int(D0['N'])
        sm['p'] = int(p)
        sm['bounds'] = D0['bounds']
        sm['beta_pop'] = D0['beta_pop']
        sm['sigma'] = dict(sigma1=D0['sigma1'], sigma2=D0['sigma2'], sigma_single=D0['sigma_single'])
        out['datasets'][nm] = sm
    path = os.path.join(bc.REPORT, 'fws_real.json')
    ev.save_json(out, path)
    print('wrote', path)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('run')
    r.add_argument('--datasets', nargs='+')
    r.add_argument('--reps', type=int, default=200)
    r.add_argument('--workers', type=int, default=4)
    r.add_argument('--force', action='store_true')
    sub.add_parser('summarize')
    a = ap.parse_args()
    cmd_run(a) if a.cmd == 'run' else summarize()


if __name__ == '__main__':
    main()
