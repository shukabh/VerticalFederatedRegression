"""Sections 12.7 and 12.8 under v3.1: benchmark against AdaSSP, and intersection size on real data.

Data: Wang's (2018) UCI regression datasets, read by experiments/adassp/datasets.py (WANG_REPO).

Splits.  10-fold cross-validation with the folds of the earlier study (datasets.kfold, seed 0).  In each
fold the held-out fold is the test set.  The training folds are shuffled once; the first 10% of all rows
(rounded) form the public reference set, which no arm trains on, and the matched set is drawn from the
rest (all of it for the benchmark; its first n rows, nested across n, for the sweep).  Every arm is
trained on the same matched rows and tested on the same test fold.

Arms (Section 12.7; delta = 1e-6 there, as in Wang's published evaluation; delta = 1e-9 in the sweep).
  (a) adassp_pub     AdaSSP, Wang's algorithm and constants, on Wang's preprocessing (z-score with full-data
                     constants, joint row normalization ||x|| = 1, y / max|y|); whole-record guarantee.
  (b) adassp_matched AdaSSP's adaptive ridge under the v3.1 adjacency and accountant (A exact, tight zCDP);
                     idealized: its private lambda_min step needs X'X in the clear.
  (c) v31_joint      v3.1 on Wang's preprocessing (no intercept, B_R = B_O = B_y = 1); idealized: the joint
                     row normalization needs both parties' features, so no party can apply it.
  (d) v31_blocknorm  v3.1, each party scales its own block of every row to unit norm (B_R^2 = 2 with R's
                     intercept, B_O = 1), B_y from the reference set.
  (e) v31_zscore     v3.1, each party standardizes its own columns with reference-set constants; 99.5%
                     bounds on the reference set (as implementation/bench/utility_runs.prepare_real).
  extra: adassp_r1   AdaSSP recalibrated to exactly (eps, delta) under replace-one adjacency of a whole
                     record with the tight conversion (check 32, noise reconciliation).
v3.1 runs the two releases, the stage-1 ridge rule, the gate (a refused release outputs nothing: theta = 0)
and the solve, bias correction off (the corrected estimate is recorded alongside).  R noise draws per fold.

Error.  Each arm's test MSE is reported (i) as a ratio to its own non-private fit under the same
preprocessing (Wang's non-private estimator (Z'Z + I)^-1 Z'y on the matched rows of that arm's design: Wang's
preprocessing for (a)-(c), the clipped per-party design with R's intercept for (d), (e)), and (ii) as an absolute test MSE on one
common y scale, Wang's y / max|y| (each arm's predictions are mapped back through its own y scaling).

    python experiments/v31_eval/real_data.py benchmark [--delta 1e-6] [--R 50] [--workers 4]
    python experiments/v31_eval/real_data.py sweep [--delta 1e-9] [--R 40] [--workers 4]
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

import datasets as WD  # noqa: E402  (experiments/adassp/datasets.py)

DECK = ['housing', 'wine', 'airfoil', 'concrete', 'bike', 'elevators', 'pol', 'kin40k', 'protein']
SWEEP = ['bike', 'elevators', 'pol', 'protein']
K = 10
FRAC_REF = 0.1
V31 = {'v31_zscore': 'zscore', 'v31_blocknorm': 'blocknorm'}
ARMS = ['adassp_pub', 'adassp_matched', 'v31_joint', 'v31_blocknorm', 'v31_zscore']
EXTRA = ['adassp_r1', 'v31_blocknorm_bc', 'v31_zscore_bc']
LETTER = {'adassp_pub': 'a', 'adassp_matched': 'b', 'v31_joint': 'c', 'v31_blocknorm': 'd', 'v31_zscore': 'e'}
LABEL = {'adassp_pub': '(a) AdaSSP, published', 'adassp_matched': '(b) AdaSSP, matched',
         'v31_joint': '(c) v3.1, joint row norm (idealized)', 'v31_blocknorm': '(d) v3.1, block unit norm',
         'v31_zscore': '(e) v3.1, standardize', 'adassp_r1': 'AdaSSP, replace-one recalibrated',
         'v31_blocknorm_bc': '(d) with bias correction', 'v31_zscore_bc': '(e) with bias correction'}
OWN = {'adassp_pub': 'np_wang', 'adassp_matched': 'np_wang', 'v31_joint': 'np_wang', 'adassp_r1': 'np_wang',
       'v31_blocknorm': 'np_v31_blocknorm', 'v31_zscore': 'np_v31_zscore',
       'v31_blocknorm_bc': 'np_v31_blocknorm', 'v31_zscore_bc': 'np_v31_zscore'}
PAIRS = [('v31_blocknorm', 'adassp_pub'), ('v31_joint', 'adassp_pub'), ('v31_joint', 'adassp_matched'),
         ('v31_blocknorm', 'v31_joint'), ('v31_zscore', 'v31_blocknorm'), ('v31_zscore', 'adassp_pub'),
         ('v31_blocknorm_bc', 'v31_blocknorm'), ('v31_zscore_bc', 'v31_zscore')]
NP_KEYS = ['np_wang', 'np_v31_blocknorm', 'np_v31_zscore', 'np_common', 'trivial']

# Earlier mechanism, from the current paper (Table tab:adassp, eps = 1, delta = 1e-6): test MSE / non-private
# ridge on Wang's preprocessing.  Reported for reference, not rerun.
EARLIER_EPS1 = {'housing': dict(np=0.0396, adassp_pub=1.80, adassp_matched=2.07, vfl_earlier=1.97, vfl_uncorr=1.95),
                'wine': dict(np=0.0201, adassp_pub=1.68, adassp_matched=1.95, vfl_earlier=1.84, vfl_uncorr=1.83),
                'airfoil': dict(np=0.0533, adassp_pub=1.09, adassp_matched=1.17, vfl_earlier=1.16, vfl_uncorr=1.16),
                'concrete': dict(np=0.0444, adassp_pub=1.48, adassp_matched=1.74, vfl_earlier=1.69, vfl_uncorr=1.68),
                'bike': dict(np=0.0279, adassp_pub=1.03, adassp_matched=1.07, vfl_earlier=1.06, vfl_uncorr=1.06),
                'elevators': dict(np=0.00868, adassp_pub=1.51, adassp_matched=1.78, vfl_earlier=1.71, vfl_uncorr=1.70),
                'pol': dict(np=0.135, adassp_pub=1.04, adassp_matched=1.08, vfl_earlier=1.06, vfl_uncorr=1.06),
                'kin40k': dict(np=0.0632, adassp_pub=1.00, adassp_matched=1.00, vfl_earlier=1.00, vfl_uncorr=1.00),
                'protein': dict(np=0.119, adassp_pub=1.01, adassp_matched=1.01, vfl_earlier=1.01, vfl_uncorr=1.01)}
EARLIER_VERDICTS = {'vs AdaSSP (published)': {'0.01': '21/0', '0.1': '14/8', '1': '6/19', '10': '1/21'},
                    'vs AdaSSP (matched)': {'0.01': '27/0', '0.1': '27/0', '1': '27/0', '10': '21/3'}}


class Data:
    """One dataset: raw features and y (for v3.1's per-party preprocessing) and Wang's preprocessed version
    (for the arms on Wang's preprocessing), on the same rows."""

    def __init__(self, name):
        self.name = name
        F, y = WD.RAW[name]()
        self.F, self.y = F.astype(float), y.astype(float)
        w = WD.load(name)
        self.Xw, self.yw = w.X, w.y
        self.Mw = float(np.max(np.abs(self.y - self.y.mean())))      # y_w = (y - mean) / Mw: the common y scale
        self.n, self.d = self.F.shape
        self.dR = self.d // 2
        self.dO = self.d - self.dR
        self.folds = WD.kfold(self.n, K, 0)

    def split(self, f, seed=0):
        te = self.folds[f]
        tr = np.random.default_rng([seed, f, 11]).permutation(np.setdiff1d(np.arange(self.n), te))
        n_ref = max(int(round(FRAC_REF * self.n)), 5)
        return te, tr[:n_ref], tr[n_ref:]

    def prep(self, ref, delta):
        out = {}
        for key, pp in V31.items():
            Z, yz, yc, b, sy = cm.per_party_preprocess(self.F, self.y, ref, self.dR, pp)
            out[key] = dict(Z=Z, yz=yz, yc=yc, bounds=b, unit=(sy / self.Mw) ** 2,
                            cal={e: cm.calibrate(e, b['B_R'], b['B_O'], b['B_y'], self.dR, self.dO, delta=delta)
                                 for e in cm.EPS})
        Zc, yzc, syc = cm.zscore_design(self.F, self.y, ref)
        out['common'] = dict(Z=Zc, yz=yzc, unit=(syc / self.Mw) ** 2)
        out['v31_joint'] = dict(cal={e: cm.calibrate(e, 1.0, 1.0, 1.0, self.dR, self.dO, delta=delta) for e in cm.EPS})
        return out


def v31_runs(G, c, dA, cal, R, rng):
    s1 = cm.stage1_batch(G, dA, cal, rng, R)
    p = G.shape[0]
    E, f, g = cm.stage2_noise(p, dA, cal, rng, R)
    st = cm.stage2_solve(G[None] + E, c[None] + f, s1['lam'], s1['Psi'], cal['sigma2'], bias_correct=True, dA=dA)
    th = np.where(st['refused'][:, None], 0.0, st['theta'])
    thbc = np.where(st['refused'][:, None], 0.0, st['theta_bc'])
    return th, thbc, s1, st


def evaluate(D, pre, m, te, eps_list, R, seed_key, delta):
    """All arms on matched rows m, test rows te.  Test MSE on the common y scale (Wang's y / max|y|)."""
    res = {}
    C = pre['common']
    res['np_common'] = cm.test_mse(C['Z'][te], C['yz'][te], cm.ridge1(C['Z'][m], C['yz'][m])) * C['unit']
    res['trivial'] = float(np.mean((D.y[te] - D.y[m].mean()) ** 2)) / D.Mw ** 2
    for key in V31:
        P_ = pre[key]
        res[f'np_{key}'] = cm.test_mse(P_['Z'][te], P_['yz'][te], cm.ridge1(P_['Z'][m], P_['yc'][m])) * P_['unit']
    Xm, ym = D.Xw[m], D.yw[m]
    G0, c0 = Xm.T @ Xm, Xm.T @ ym
    Xt, yt = D.Xw[te], D.yw[te]
    res['np_wang'] = cm.test_mse(Xt, yt, np.linalg.solve(G0 + np.eye(D.d), c0))
    lmin = float(np.linalg.eigvalsh(G0 + np.eye(D.d))[0])
    for e, eps in enumerate(eps_list):
        rng = np.random.default_rng([*seed_key, e])
        for key in V31:
            P_ = pre[key]
            Z, yc = P_['Z'][m], P_['yc'][m]
            th, thbc, s1, st = v31_runs(Z.T @ Z, Z.T @ yc, D.dR + 1, P_['cal'][eps], R, rng)
            res[(key, eps)] = cm.test_mse(P_['Z'][te], P_['yz'][te], th) * P_['unit']
            res[(key + '_bc', eps)] = cm.test_mse(P_['Z'][te], P_['yz'][te], thbc) * P_['unit']
            res[(key + '_lam', eps)] = float(np.median(s1['lam']))
            res[(key + '_psiO', eps)] = float(np.mean(s1['kind'] == 'O'))
            res[(key + '_refused', eps)] = float(st['refused'].mean())
        th, thbc, s1, st = v31_runs(G0, c0, D.dR, pre['v31_joint']['cal'][eps], R, rng)
        res[('v31_joint', eps)] = cm.test_mse(Xt, yt, th)
        res[('v31_joint_lam', eps)] = float(np.median(s1['lam']))
        res[('v31_joint_refused', eps)] = float(st['refused'].mean())
        th, lam = cm.adassp(G0, c0, eps, delta, R, rng, variant='published', lmin=lmin)
        res[('adassp_pub', eps)] = cm.test_mse(Xt, yt, th)
        res[('adassp_pub_lam', eps)] = float(np.median(lam))
        th, lam = cm.adassp_matched(G0, c0, D.dR, eps, delta, R, rng, lmin=lmin)
        res[('adassp_matched', eps)] = cm.test_mse(Xt, yt, th)
        res[('adassp_matched_lam', eps)] = float(np.median(lam))
        th, lam = cm.adassp(G0, c0, eps, delta, R, rng, variant='replace1', lmin=lmin)
        res[('adassp_r1', eps)] = cm.test_mse(Xt, yt, th)
    return res


# ------------------------------------------------------------------------------------------- benchmark
def bench_dataset(name, R, delta):
    t0 = time.time()
    D = Data(name)
    E = len(cm.EPS)
    fold = {k: np.zeros(K) for k in NP_KEYS}
    for mth in ARMS + EXTRA:
        fold[mth] = np.zeros((E, K))
    run_q = {mth: [] for mth in ARMS}
    xkeys = ['v31_zscore_lam', 'v31_blocknorm_lam', 'v31_joint_lam', 'adassp_pub_lam', 'adassp_matched_lam',
             'v31_zscore_psiO', 'v31_blocknorm_psiO', 'v31_zscore_refused', 'v31_blocknorm_refused', 'v31_joint_refused']
    extra = {k: np.zeros((E, K)) for k in xkeys}
    runs = {mth: np.zeros((E, K, R)) for mth in ARMS}
    for f in range(K):
        te, ref, m = D.split(f)
        pre = D.prep(ref, delta)
        if f == 0:
            bounds0 = {k: pre[k]['bounds'] for k in V31}
            cal0 = {k: {str(e): pre[k]['cal'][e] for e in cm.EPS} for k in ('v31_zscore', 'v31_blocknorm', 'v31_joint')}
        res = evaluate(D, pre, m, te, cm.EPS, R, (0, f, sum(map(ord, name))), delta)
        for k in NP_KEYS:
            fold[k][f] = res[k]
        for e, eps in enumerate(cm.EPS):
            for mth in ARMS + EXTRA:
                fold[mth][e, f] = float(np.mean(res[(mth, eps)]))
            for mth in ARMS:
                runs[mth][e, f] = res[(mth, eps)]
            for k in xkeys:
                extra[k][e, f] = res[(k, eps)]
    out = dict(name=name, n=D.n, d=D.d, d_R=D.dR, d_O=D.dO, n_matched=int(len(m)), n_ref=int(len(ref)),
               n_test=int(len(te)), Mw=D.Mw, bounds_fold0=bounds0, calibration_fold0=cal0, fold=fold, extra=extra,
               run_q={mth: np.percentile(runs[mth], [2.5, 97.5], axis=(1, 2)).T for mth in ARMS})
    return out, f'{name}: {time.time() - t0:.0f}s'


def paired(a, b):
    dlt = np.asarray(a) - np.asarray(b)
    k = len(dlt)
    half = cm.t95(k) * dlt.std(ddof=1) / math.sqrt(k)
    mm = dlt.mean()
    return dict(diff=float(mm), half=float(half), rel=float(mm / np.mean(b)),
                verdict='better' if mm + half < 0 else 'worse' if mm - half > 0 else 'none')


def Data_size(name):
    return {'3droad': 434874, 'protein': 45730, 'tamielectric': 45781, 'kin40k': 40000, 'bike': 17379,
            'elevators': 16599, 'pol': 15000, 'pumadyn32nm': 8192, 'parkinsons': 5875}.get(name, 1000)


def cmd_benchmark(a):
    names = a.datasets or WD.PUBLISHED_ORDER
    results = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {nm: ex.submit(bench_dataset, nm, a.R, a.delta) for nm in sorted(names, key=lambda s: -Data_size(s))}
        for nm in names:
            r, msg = futs[nm].result()
            results.append(r)
            print(msg, flush=True)
    summ = []
    for r in results:
        F = r['fold']
        row = dict(name=r['name'], n=r['n'], d=r['d'], d_R=r['d_R'], n_matched=r['n_matched'],
                   np_abs={k: float(F[k].mean()) for k in NP_KEYS}, by_eps={})
        for e, eps in enumerate(cm.EPS):
            d = {}
            for mth in ARMS + EXTRA:
                v = F[mth][e]
                own = F[OWN[mth]]
                ro = v / own                                           # per-fold ratio to own non-private
                d[mth] = dict(abs=float(v.mean()), abs_ci_half=float(cm.t95(K) * v.std(ddof=1) / math.sqrt(K)),
                              ratio_own=float(v.mean() / own.mean()), ratio_own_foldmean=float(ro.mean()),
                              ratio_own_ci_half=float(cm.t95(K) * ro.std(ddof=1) / math.sqrt(K)),
                              ratio_common=float(v.mean() / F['np_common'].mean()))
            d['paired_abs'] = {f'{x}_vs_{y}': paired(F[x][e], F[y][e]) for x, y in PAIRS}
            d['paired_ratio_own'] = {f'{x}_vs_{y}': paired(F[x][e] / F[OWN[x]], F[y][e] / F[OWN[y]]) for x, y in PAIRS}
            d['bc_change'] = {k: float(F[k + '_bc'][e].mean() / F[k][e].mean() - 1) for k in V31}
            d['lam_median'] = {k: float(np.median(r['extra'][k + '_lam'][e]))
                               for k in ('v31_zscore', 'v31_blocknorm', 'v31_joint', 'adassp_pub', 'adassp_matched')}
            d['psiO'] = {k: float(r['extra'][k + '_psiO'][e].mean()) for k in V31}
            d['refused'] = {k: float(r['extra'][k + '_refused'][e].mean()) for k in ('v31_zscore', 'v31_blocknorm', 'v31_joint')}
            row['by_eps'][str(eps)] = d
        summ.append(row)
    verdicts = {}
    for basis in ('paired_abs', 'paired_ratio_own'):
        verdicts[basis] = {}
        for x, y in PAIRS:
            key = f'{x}_vs_{y}'
            verdicts[basis][key] = {}
            for eps in cm.EPS:
                v = [s['by_eps'][str(eps)][basis][key]['verdict'] for s in summ]
                verdicts[basis][key][str(eps)] = dict(better=v.count('better'), worse=v.count('worse'), none=v.count('none'))
    acct = {}
    for e in cm.EPS:
        ar, rr = cm.adassp_eps_achieved(e, a.delta, 'add_remove'), cm.adassp_eps_achieved(e, a.delta, 'replace1')
        acct[str(e)] = dict(eps_achieved_add_remove=ar[0], rho_add_remove=ar[1], eps_achieved_replace_one=rr[0],
                            rho_replace_one=rr[1], rho_tight=cm.dp.rho_tight(e, a.delta),
                            sig_published=cm.adassp_sigmas(e, a.delta, 10, 'published'),
                            sig_replace1=cm.adassp_sigmas(e, a.delta, 10, 'replace1'),
                            sig_matched=cm.adassp_matched_sigmas(e, a.delta, 10),
                            v31_B1=dict((k, cm.calibrate(e, 1.0, 1.0, 1.0, 5, 5, delta=a.delta)[k])
                                        for k in ('sigma1', 'sigma2', 'Delta', 'Delta_G', 'rho_run')))
    out = dict(settings=dict(delta=a.delta, eps=cm.EPS, R=a.R, folds=K, frac_ref=FRAC_REF, arms=LABEL,
                             common_y_scale="Wang's y / max|y| (y centred on the full data)"),
               summary=summ, verdicts=verdicts, adassp_accounting=acct,
               earlier_mechanism_eps1=EARLIER_EPS1, earlier_verdicts=EARLIER_VERDICTS, raw=results)
    tag = '' if a.datasets is None else '_' + '_'.join(a.datasets)
    cm.save_json(out, os.path.join(cm.REPORT, f'adassp_benchmark{tag}.json'))
    print('wrote', os.path.join(cm.REPORT, f'adassp_benchmark{tag}.json'))


# ------------------------------------------------------------------------------------------- sweep
def sweep_fold(name, f, R, points, delta):
    t0 = time.time()
    D = Data(name)
    te, ref, pool = D.split(f)
    pre = D.prep(ref, delta)
    nmax = min(len(D.split(g)[2]) for g in range(K))
    grid = np.unique(np.concatenate([np.round(np.geomspace(100, nmax, points)).astype(int), [1000, 10000]]))
    grid = [int(v) for v in grid if v <= nmax]
    rows = []
    for g, n in enumerate(grid):
        res = evaluate(D, pre, pool[:n], te, cm.EPS, R, (1, f, g, sum(map(ord, name))), delta)
        row = {k: res[k] for k in NP_KEYS}
        row.update(n=n, eps={})
        for eps in cm.EPS:
            row['eps'][str(eps)] = {mth: float(np.mean(res[(mth, eps)])) for mth in ARMS + EXTRA}
            for k in ('v31_blocknorm_lam', 'v31_zscore_lam'):
                row['eps'][str(eps)][k] = res[(k, eps)]
        rows.append(row)
    return dict(name=name, fold=f, grid=grid, rows=rows, n=D.n, d=D.d), f'{name} fold {f}: {time.time() - t0:.0f}s'


def cmd_sweep(a):
    names = a.datasets or SWEEP
    out = {}
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = [(nm, f, ex.submit(sweep_fold, nm, f, a.R, a.points, a.delta)) for nm in names for f in range(K)]
        for nm, f, fu in futs:
            r, msg = fu.result()
            out.setdefault(nm, []).append(r)
            print(msg, flush=True)
    summ = {}
    for nm, folds in out.items():
        grid = folds[0]['grid']
        G = len(grid)
        arr = lambda key: np.array([[fo['rows'][g][key] for fo in folds] for g in range(G)])   # (G, K)
        npk = {k: arr(k) for k in NP_KEYS}
        S = dict(grid=grid, n=folds[0]['n'], d=folds[0]['d'],
                 np_abs={k: v.mean(1) for k, v in npk.items()},
                 trivial_ratio_own={k: (npk['trivial'] / npk[k]).mean(1) for k in ('np_wang', 'np_v31_blocknorm', 'np_v31_zscore')},
                 np_ratio_to_common={k: (npk[k] / npk['np_common']).mean(1) for k in ('np_wang', 'np_v31_blocknorm', 'np_v31_zscore')},
                 eps={})
        for eps in cm.EPS:
            E = {}
            for mth in ARMS + EXTRA:
                v = np.array([[fo['rows'][g]['eps'][str(eps)][mth] for fo in folds] for g in range(G)])
                ro = v / npk[OWN[mth]]
                rc = v / npk['np_common']
                E[mth] = dict(abs=v.mean(1), ratio_own=ro.mean(1), ratio_own_ci=cm.t95(K) * ro.std(1, ddof=1) / math.sqrt(K),
                              ratio_common=rc.mean(1),
                              cross_own_1p10=cm.crossing(grid, ro.mean(1), 1.10),
                              cross_own_1p25=cm.crossing(grid, ro.mean(1), 1.25),
                              cross_common_1p10=cm.crossing(grid, rc.mean(1), 1.10))
            for k in ('v31_blocknorm_lam', 'v31_zscore_lam'):
                E[k] = [float(np.median([fo['rows'][g]['eps'][str(eps)][k] for fo in folds])) for g in range(G)]
            S['eps'][str(eps)] = E
        summ[nm] = S
    cm.save_json(dict(settings=dict(delta=a.delta, eps=cm.EPS, R=a.R, folds=K, frac_ref=FRAC_REF, arms=LABEL),
                      summary=summ), os.path.join(cm.REPORT, 'sweep_n.json'))
    print('wrote', os.path.join(cm.REPORT, 'sweep_n.json'))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    b = sub.add_parser('benchmark')
    b.add_argument('--datasets', nargs='+')
    b.add_argument('--delta', type=float, default=1e-6)
    b.add_argument('--R', type=int, default=50)
    b.add_argument('--workers', type=int, default=4)
    s = sub.add_parser('sweep')
    s.add_argument('--datasets', nargs='+')
    s.add_argument('--delta', type=float, default=1e-9)
    s.add_argument('--R', type=int, default=40)
    s.add_argument('--points', type=int, default=14)
    s.add_argument('--workers', type=int, default=4)
    a = ap.parse_args()
    {'benchmark': cmd_benchmark, 'sweep': cmd_sweep}[a.cmd](a)


if __name__ == '__main__':
    main()
