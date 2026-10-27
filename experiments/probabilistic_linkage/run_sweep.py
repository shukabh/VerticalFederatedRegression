"""Controlled linkage errors: the v3.1 regression as a function of the false-match rate pi and the miss rate mu.

For each cell, `reps` fresh cohorts of n0 true pairs from the paper's synthetic population ('std', d_R = 4, d_O = 6):
  * misses: each true pair is lost with probability mu, either at random (MCAR) or with a probability that rises as
    the outcome falls (MNAR: p_i proportional to exp(-y_i), rescaled to mean mu) -- the case where records of some
    groups are harder to link;
  * false matches: a fraction pi of the remaining pairs has its O side (x_O, y) replaced by that of an independent
    person of the scope (exchangeable linkage errors, as in Lahiri and Larsen, 2005);
then one stage-1 / stage-2 release per cohort and eps (setting I with the Wald and bias-aware intervals; setting P
estimate only), plus least squares on the same linked rows.

    python run_sweep.py [--reps 100]          # writes reports/probabilistic_linkage/sweep.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import regression as rg  # noqa: E402

PIS = [0.0, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20]
MISS = [(0.0, 'mcar'), (0.3, 'mcar'), (0.3, 'mnar')]
EPS = [1.0, 4.0]


def cohort(n0, pi, mu, kind, rng):
    pop = rg.lc.population()
    X = pop._features(n0, rng)
    zR, zO, Zs = rg.lc._transform(pop, X, 'std')
    y = Zs @ pop.beta_std + pop.noise_sd * rng.standard_normal(n0)
    if mu > 0:
        if kind == 'mcar':
            keep = rng.random(n0) >= mu
        else:
            w = np.exp(-y)
            p = np.minimum(1.0, mu * w / w.mean())
            keep = rng.random(n0) >= p
        zR, zO, y = zR[keep], zO[keep], y[keep]
    n = len(y)
    nf = int(round(pi * n))
    if nf:
        idx = rng.choice(n, nf, replace=False)
        X2 = pop._features(nf, rng)
        _, zO2, Zs2 = rg.lc._transform(pop, X2, 'std')
        y2 = Zs2 @ pop.beta_std + pop.noise_sd * rng.standard_normal(nf)
        zO = zO.copy(); y = y.copy()
        zO[idx], y[idx] = zO2, y2
    Z, yc = rg.design(zR, zO, y)
    return rg.stats(Z, yc), n


def cell(args):
    pi, mu, kind, n0, reps, seed, bboot = args
    rng = np.random.default_rng([seed, int(pi * 1e4), int(mu * 100), 1 if kind == 'mcar' else 2])
    acc = {f'{st}_e{e:g}': [] for st in ('I', 'P') for e in EPS}
    ols, ns = [], []
    for r in range(reps):
        (G, c, yy), n = cohort(n0, pi, mu, kind, rng)
        ns.append(n)
        ols.append(rg.ols(G, c))
        for e in EPS:
            for st in ('I', 'P'):
                rel = rg.release(G, c, yy, n, e, st, 1, rng, Bboot=bboot)
                acc[f'{st}_e{e:g}'].append(rel)
    out = dict(pi=pi, mu=mu, kind=kind, n_mean=float(np.mean(ns)), reps=reps)
    th = np.array(ols)
    B = rg.BETA
    out['ols'] = dict(rmse_median=float(np.median(np.sqrt(np.mean((th - B) ** 2, 1)))),
                      ratio_R=float(np.mean(th.mean(0)[1:5] / B[1:5])), ratio_O=float(np.mean(th.mean(0)[5:] / B[5:])))
    out['formula'] = dict(ratio_R=float(np.mean(rg.contamination_limit(pi)[1:5] / B[1:5])),
                          ratio_O=float(np.mean(rg.contamination_limit(pi)[5:] / B[5:])))
    for k, rels in acc.items():
        merged = dict(theta=np.concatenate([x['theta'] for x in rels]),
                      refused=np.concatenate([x['refused'] for x in rels]),
                      lam=np.concatenate([x['lam'] for x in rels]))
        if 'cov_beta' in rels[0]:
            for f in ('cov_beta', 'cov_theta', 'half_beta', 'half_theta'):
                parts = [x[f] for x in rels if len(x[f])]
                merged[f] = np.concatenate(parts) if parts else np.zeros((0, rg.P))
        out[k] = rg.summarize(merged)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reps', type=int, default=100)
    ap.add_argument('--n0', type=int, default=40_000)
    ap.add_argument('--seed', type=int, default=20261027)
    ap.add_argument('--bboot', type=int, default=1000)
    ap.add_argument('--workers', type=int, default=3)
    a = ap.parse_args()
    rg.population_moments()
    jobs = [(pi, mu, kind, a.n0, a.reps, a.seed, a.bboot) for (mu, kind) in MISS for pi in PIS]
    t0 = time.time()
    res = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for r in ex.map(cell, jobs):
            res.append(r)
            i1 = r['I_e1']
            print(f"pi={r['pi']:.3f} mu={r['mu']} {r['kind']} n={r['n_mean']:.0f} ols R-ratio {r['ols']['ratio_R']:.3f} "
                  f"(formula {r['formula']['ratio_R']:.3f}) | I e1 rmse {i1['rmse_median']:.4f} covb {i1.get('cov_beta', 0):.3f} "
                  f"covb_R {i1.get('cov_beta_R', 0):.3f} covth {i1.get('cov_theta', 0):.3f} ridge {i1['ridge_on']:.2f} "
                  f"| I e4 covb_R {r['I_e4'].get('cov_beta_R', 0):.3f}", flush=True)
    out = dict(stamp=time.strftime('%Y-%m-%d %H:%M'), n0=a.n0, reps=a.reps, eps=EPS, pis=PIS, miss=MISS,
               n_off={f'{e:g}': {s: rg.n_off(e, s) for s in ('I', 'P')} for e in EPS}, cells=res,
               seconds=time.time() - t0)
    path = os.path.join(rg.ROOT, 'reports', 'probabilistic_linkage', 'sweep.json')
    with open(path, 'w') as fh:
        json.dump(out, fh, indent=1, default=float)
    print('wrote', path, f'{time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
