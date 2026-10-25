"""Baseline 3 on Wang's 29 datasets: TukeyEM (Amin et al., ICLR 2023) at matched (eps, delta = 1e-6).

TukeyEM needs no bound on the features or the labels and runs on the joined table (an idealized, central baseline).
Same folds, reference rows and matched rows as Section 13.7 (experiments/v31_settings/real_settings.py).  Design: the
z-scored features (constants from the reference rows) with an intercept column, and the standardized y, unclipped; this
is the 13.7 "common" design, whose non-private ridge fit (np_common) is TukeyEM's own non-private baseline.  Test MSE is
mapped to Wang's y / max|y| scale as for every other arm.  50 draws per fold (each draw a fresh random partition, PTR
test and sample); the m models of a draw are shared across eps.  A failed PTR test outputs theta = 0.

Number of models m.  TukeyEM's accuracy and its PTR pass rate depend on m.  The primary rule, fixed in advance, takes
subsets of 2p rows, m = floor(n / (2p)) (p = d + 1 coefficients): about what the reference demo uses (n = 22,000,
p = 11, m from 1,000 to 2,000).  We also run m = floor(n / (k p)) for k in {1.5, 3, 5, 10, 25} and report the best k
per dataset and eps, chosen on the test folds, as an optimistic oracle.

    python experiments/baselines/real_tukey.py benchmark [--R 50] [--workers 4] [--datasets ...]
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
from bcommon import cs, ev  # noqa: E402
import tukeyem as te_  # noqa: E402

import real_data as rdm  # noqa: E402
import real_full as rf  # noqa: E402
import datasets as WD  # noqa: E402

K = rdm.K
EPS = bc.EPS
DELTA = 1e-6
KS = [1.5, 2.0, 3.0, 5.0, 10.0, 25.0]
K_PRIMARY = 2.0
PAIRS_TO = ['v31_blocknorm_I', 'v31_blocknorm_P2', 'adassp_pub', 'adassp_matched', 'v31_joint_I', 'v31_joint_P2']


def m_for(n, p, k):
    return int(n // (k * p))


def bench_dataset(name, R):
    t0 = time.time()
    D = rdm.Data(name)
    E = len(EPS)
    p = D.d + 1
    fold = {k: np.zeros((E, K)) for k in KS}
    passed = {k: np.zeros((E, K)) for k in KS}
    dist = {k: np.zeros((E, K)) for k in KS}
    npc = np.zeros(K)
    ms = {}
    fails = {}
    for f in range(K):
        te, ref, m = D.split(f)
        Zc, yzc, syc = ev.zscore_design(D.F, D.y, ref)
        unit = (syc / D.Mw) ** 2
        npc[f] = ev.test_mse(Zc[te], yzc[te], ev.ridge1(Zc[m], yzc[m])) * unit
        X, y = Zc[m], yzc[m]
        n = len(m)
        rng = np.random.default_rng([20261026, f, sum(map(ord, name))])
        for k in KS:
            mm = m_for(n, p, k)
            ms.setdefault(k, []).append(mm)
            mse = np.zeros((E, R)); ok = np.zeros((E, R)); dd = np.full((E, R), np.nan)
            if mm < 1 or n // max(mm, 1) < p:
                fails[k] = 'fewer rows than coefficients per subset'
                mse[:] = ev.test_mse(Zc[te], yzc[te], np.zeros(p)) * unit
            else:
                for r in range(R):
                    models = te_.multiple_regressions(X, y, mm, rng)
                    for e, eps in enumerate(EPS):
                        th, info = te_.dp_tukey(models, eps, DELTA, rng)
                        mse[e, r] = ev.test_mse(Zc[te], yzc[te], th) * unit
                        ok[e, r] = info['passed']
                        if info.get('distance') is not None:
                            dd[e, r] = info['distance']
            fold[k][:, f] = mse.mean(1)
            passed[k][:, f] = ok.mean(1)
            dist[k][:, f] = np.nanmedian(dd, 1) if np.isfinite(dd).any() else np.nan
    return dict(name=name, n=D.n, d=D.d, p=p, n_matched=int(len(m)), fold=fold, passed=passed, dist=dist,
                np_common=npc, m=ms, fails=fails), f'{name}: {time.time() - t0:.0f}s'


def cmd_benchmark(a):
    names = a.datasets or list(WD.PUBLISHED_ORDER)
    pub = rf.published()
    s = ev.load_json(os.path.join(cs.REPORT, 'adassp_benchmark.json'))
    npc_pub = {r['name']: np.asarray(r['fold']['np_common'], float) for r in s['raw']}
    results = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {nm: ex.submit(bench_dataset, nm, a.R) for nm in sorted(names, key=lambda x: -rdm.Data_size(x))}
        for nm in names:
            r, msg = futs[nm].result()
            results.append(r)
            print(msg, flush=True)
    summ = []
    for r in results:
        nm = r['name']
        assert np.allclose(r['np_common'], npc_pub[nm], rtol=1e-9), nm
        F = dict(pub[nm])
        F['np_common'] = r['np_common']
        for k in KS:
            F[f'tukey_k{k:g}'] = r['fold'][k]
        thr = {str(e): te_.ptr_threshold(e, DELTA) for e in EPS}
        row = dict(name=nm, n=r['n'], d=r['d'], p=r['p'], n_matched=r['n_matched'],
                   m={f'{k:g}': int(np.median(r['m'][k])) for k in KS},
                   t={f'{k:g}': int(int(int(np.median(r['m'][k])) / 2) / 2) for k in KS},
                   ptr_threshold=thr, fails={f'{k:g}': v for k, v in r['fails'].items()},
                   np_abs=dict(np_common=float(r['np_common'].mean()), np_wang=float(F['np_wang'].mean()),
                               np_v31_blocknorm=float(F['np_v31_blocknorm'].mean())), by_eps={})
        for e, eps in enumerate(EPS):
            d = dict(tukey={}, passed={}, distance={})
            for k in KS:
                v = r['fold'][k][e]
                ro = v / r['np_common']
                d['tukey'][f'{k:g}'] = dict(abs=float(v.mean()), ratio_own=float(v.mean() / r['np_common'].mean()),
                                            ratio_own_ci_half=float(ev.t95(K) * ro.std(ddof=1) / math.sqrt(K)))
                d['passed'][f'{k:g}'] = float(r['passed'][k][e].mean())
                dk = r['dist'][k][e]
                d['distance'][f'{k:g}'] = float(np.nanmedian(dk)) if np.isfinite(dk).any() else None
            best = min(KS, key=lambda k: r['fold'][k][e].mean())
            d['best_k'] = best
            prim = r['fold'][K_PRIMARY][e]
            bestv = r['fold'][best][e]
            d['paired_abs'] = {f'tukey_vs_{y}': rdm.paired(prim, F[y][e]) for y in PAIRS_TO}
            d['paired_abs'].update({f'tukeybest_vs_{y}': rdm.paired(bestv, F[y][e]) for y in PAIRS_TO})
            own = {y: F[rf.OWN[y]] for y in PAIRS_TO}
            d['paired_ratio_own'] = {f'tukey_vs_{y}': rdm.paired(prim / r['np_common'], F[y][e] / own[y]) for y in PAIRS_TO}
            for y in PAIRS_TO:
                d[y] = dict(abs=float(F[y][e].mean()), ratio_own=float(F[y][e].mean() / F[rf.OWN[y]].mean()))
            row['by_eps'][str(eps)] = d
        summ.append(row)
    verdicts, gaps = {}, {}
    for basis in ('paired_abs', 'paired_ratio_own'):
        verdicts[basis] = {}
        for key in summ[0]['by_eps'][str(EPS[0])][basis]:
            verdicts[basis][key] = {}
            for eps in EPS:
                v = [s['by_eps'][str(eps)][basis][key]['verdict'] for s in summ]
                verdicts[basis][key][str(eps)] = dict(better=v.count('better'), worse=v.count('worse'), none=v.count('none'))
    for eps in EPS:
        g = {}
        for y in PAIRS_TO:
            v = np.array([s['by_eps'][str(eps)]['tukey'][f'{K_PRIMARY:g}']['abs'] / s['by_eps'][str(eps)][y]['abs'] - 1
                          for s in summ])
            g[f'tukey/{y}-1'] = dict(median=float(np.median(v)), min=float(v.min()), max=float(v.max()))
        pr = np.array([s['by_eps'][str(eps)]['passed'][f'{K_PRIMARY:g}'] for s in summ])
        gaps[str(eps)] = dict(abs_gap=g, ptr_pass_mean=float(pr.mean()), datasets_pass_lt_50pct=int((pr < 0.5).sum()),
                              datasets_never_pass=int((pr == 0).sum()), datasets_always_pass=int((pr == 1).sum()),
                              never_pass=[s['name'] for s, x in zip(summ, pr) if x == 0])
    out = dict(settings=dict(delta=DELTA, eps=EPS, R=a.R, folds=K, frac_ref=rdm.FRAC_REF, ks=KS, k_primary=K_PRIMARY,
                             laplace_scale='1/(eps/2) (corrected; the reference code uses eps/2)',
                             design='z-scored features with intercept, standardized y, unclipped (13.7 common design)',
                             reference='google-research/dp_regression/tukey.py @ ba728e288c51126eac25e316b11dfee94d68c301'),
               summary=summ, verdicts=verdicts, gaps=gaps,
               raw=[dict(name=r['name'], fold={f'{k:g}': r['fold'][k] for k in KS},
                         passed={f'{k:g}': r['passed'][k] for k in KS}) for r in results])
    tag = '' if a.datasets is None else '_' + '_'.join(a.datasets)
    path = os.path.join(bc.REPORT, f'tukey_benchmark{tag}.json')
    ev.save_json(out, path)
    print('wrote', path)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    b = sub.add_parser('benchmark')
    b.add_argument('--datasets', nargs='+')
    b.add_argument('--R', type=int, default=50)
    b.add_argument('--workers', type=int, default=4)
    a = ap.parse_args()
    cmd_benchmark(a)


if __name__ == '__main__':
    main()
