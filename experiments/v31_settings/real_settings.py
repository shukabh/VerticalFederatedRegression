"""Sections 13.7 and 13.8 under the named settings I (inference) and P (prediction).

Same data, splits, folds, preprocessing and reporting as experiments/v31_eval/real_data.py (the previous setting):
Wang's 29 UCI datasets, 10-fold cross-validation on the folds of the earlier study, 10% public reference rows, every
other training row matched; delta = 1e-6 in the benchmark (as Wang), 1e-9 in the sweep.

Arms.
  (a) adassp_pub      AdaSSP, Wang's algorithm and constants, Wang's preprocessing.
  (b) adassp_matched  AdaSSP's adaptive ridge under the v3.1 adjacency and accountant (idealized).
  (c) v31_joint_{I,P}      v3.1 on Wang's preprocessing (no intercept, unit bounds; idealized), setting I or P.
  (d) v31_blocknorm_{I,P}  v3.1, each party scales its own block of every row to unit norm, setting I or P.
  (e) v31_zscore_I         v3.1, per-party standardization, setting I.
The previous setting's (c), (d), (e) are taken from reports/v31_eval/adassp_benchmark.json and sweep_n.json (not
rerun); its fold-level values are paired with the new arms (same folds and matched rows).

Random numbers.  One stream per (fold, eps, arm family): the v3.1 arms of one preprocessing share their unit noise
across I and P (common random numbers), scaled by each setting's sigma_1 and sigma_2.  A refused release (gate)
outputs theta = 0.

    python experiments/v31_settings/real_settings.py benchmark [--delta 1e-6] [--R 50] [--workers 4]
    python experiments/v31_settings/real_settings.py sweep [--delta 1e-9] [--R 40] [--workers 4]
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

import real_data as rdm  # noqa: E402  (experiments/v31_eval/real_data.py: Data, paired, sizes)
import datasets as WD  # noqa: E402

K = rdm.K
SWEEP = rdm.SWEEP
SET_NAMES = ('I', 'P')
V31_ARMS = ['v31_joint_I', 'v31_joint_P', 'v31_blocknorm_I', 'v31_blocknorm_P', 'v31_zscore_I']
ARMS = ['adassp_pub', 'adassp_matched'] + V31_ARMS
LABEL = {'adassp_pub': '(a) AdaSSP, published', 'adassp_matched': '(b) AdaSSP, matched',
         'v31_joint_I': '(c) v3.1 I, joint row norm (idealized)', 'v31_joint_P': '(c) v3.1 P, joint row norm (idealized)',
         'v31_blocknorm_I': '(d) v3.1 I, block unit norm', 'v31_blocknorm_P': '(d) v3.1 P, block unit norm',
         'v31_zscore_I': '(e) v3.1 I, standardize'}
OWN = {'adassp_pub': 'np_wang', 'adassp_matched': 'np_wang', 'v31_joint_I': 'np_wang', 'v31_joint_P': 'np_wang',
       'v31_blocknorm_I': 'np_v31_blocknorm', 'v31_blocknorm_P': 'np_v31_blocknorm', 'v31_zscore_I': 'np_v31_zscore',
       'v31_joint_prev': 'np_wang', 'v31_blocknorm_prev': 'np_v31_blocknorm', 'v31_zscore_prev': 'np_v31_zscore'}
PAIRS = [('v31_blocknorm_I', 'adassp_pub'), ('v31_blocknorm_P', 'adassp_pub'),
         ('v31_joint_I', 'adassp_pub'), ('v31_joint_P', 'adassp_pub'),
         ('v31_joint_I', 'adassp_matched'), ('v31_joint_P', 'adassp_matched'),
         ('v31_blocknorm_I', 'adassp_matched'), ('v31_blocknorm_P', 'adassp_matched'),
         ('v31_blocknorm_P', 'v31_blocknorm_I'), ('v31_joint_P', 'v31_joint_I'),
         ('v31_blocknorm_I', 'v31_joint_I'), ('v31_blocknorm_P', 'v31_joint_P'),
         ('v31_zscore_I', 'v31_blocknorm_I'),
         ('v31_blocknorm_I', 'v31_blocknorm_prev'), ('v31_blocknorm_P', 'v31_blocknorm_prev'),
         ('v31_joint_I', 'v31_joint_prev'), ('v31_joint_P', 'v31_joint_prev'),
         ('v31_zscore_I', 'v31_zscore_prev')]
NP_KEYS = rdm.NP_KEYS
FAM = {'joint': 1, 'blocknorm': 2, 'zscore': 3, 'adassp_pub': 4, 'adassp_matched': 5}


def settings():
    s = cs.load_settings()
    return {k: s[k] for k in SET_NAMES}


def r1_keys(datasets, alpha_Ls):
    keys = []
    for name in datasets:
        d = WD_dims(name)
        dR, dO = d // 2, d - d // 2
        for al in alpha_Ls:
            keys += [(dR + 1, dO, al), (dR, dO, al)]
    return keys


def WD_dims(name):
    prev = ev.load_json(os.path.join(cs.REPORT_EVAL, 'adassp_benchmark.json'))
    return {s['name']: s['d'] for s in prev['summary']}[name]


def prep(D, ref, delta, sts):
    out = {}
    for key, pp in rdm.V31.items():
        Z, yz, yc, b, sy = ev.per_party_preprocess(D.F, D.y, ref, D.dR, pp)
        out[key] = dict(Z=Z, yz=yz, yc=yc, bounds=b, unit=(sy / D.Mw) ** 2,
                        cal={(s, e): cs.calibrate(e, b['B_R'], b['B_O'], b['B_y'], D.dR, D.dO, sts[s]['f1'], delta=delta)
                             for s in sts for e in cs.EPS})
    Zc, yzc, syc = ev.zscore_design(D.F, D.y, ref)
    out['common'] = dict(Z=Zc, yz=yzc, unit=(syc / D.Mw) ** 2)
    out['v31_joint'] = dict(cal={(s, e): cs.calibrate(e, 1.0, 1.0, 1.0, D.dR, D.dO, sts[s]['f1'], delta=delta)
                                 for s in sts for e in cs.EPS})
    return out


def unit_noise(rng, R, p, dA):
    return (solve.masked_noise(rng, R, p, dA, 1.0), solve.masked_noise(rng, R, p, dA, 1.0), rng.standard_normal((R, p)))


def v31_from_unit(G, c, dA, cal, st, U):
    Z1, Z2, f2 = U
    s1 = cs.stage1_vec(G[None] + cal['sigma1'] * Z1, dA, cal['sigma1'], cal['sigma2'], st)
    th, _, gate, refused, _, _ = cs.stage2_from_unit(G[None], c[None], Z2, f2, s1['lam'], s1['Psi'], cal['sigma2'])
    return th, s1, refused


def evaluate(D, pre, m, te, eps_list, R, seed_key, delta, sts, which=('joint', 'blocknorm', 'zscore', 'adassp')):
    res = {}
    C = pre['common']
    res['np_common'] = ev.test_mse(C['Z'][te], C['yz'][te], ev.ridge1(C['Z'][m], C['yz'][m])) * C['unit']
    res['trivial'] = float(np.mean((D.y[te] - D.y[m].mean()) ** 2)) / D.Mw ** 2
    for key in rdm.V31:
        P_ = pre[key]
        res[f'np_{key}'] = ev.test_mse(P_['Z'][te], P_['yz'][te], ev.ridge1(P_['Z'][m], P_['yc'][m])) * P_['unit']
    Xm, ym = D.Xw[m], D.yw[m]
    G0, c0 = Xm.T @ Xm, Xm.T @ ym
    Xt, yt = D.Xw[te], D.yw[te]
    res['np_wang'] = ev.test_mse(Xt, yt, np.linalg.solve(G0 + np.eye(D.d), c0))
    lmin = float(np.linalg.eigvalsh(G0 + np.eye(D.d))[0])
    for e, eps in enumerate(eps_list):
        for key, sets in (('blocknorm', SET_NAMES), ('zscore', ('I',))):
            if key not in which:
                continue
            P_ = pre[f'v31_{key}']
            Z, yc = P_['Z'][m], P_['yc'][m]
            G, c = Z.T @ Z, Z.T @ yc
            U = unit_noise(np.random.default_rng([*seed_key, e, FAM[key]]), R, G.shape[0], D.dR + 1)
            for s in sets:
                th, s1, refused = v31_from_unit(G, c, D.dR + 1, P_['cal'][(s, eps)], sts[s], U)
                a = f'v31_{key}_{s}'
                res[(a, eps)] = ev.test_mse(P_['Z'][te], P_['yz'][te], th) * P_['unit']
                res[(a + '_lam', eps)] = float(np.median(s1['lam']))
                res[(a + '_psiO', eps)] = float(np.mean(s1['kind'] == 'O'))
                res[(a + '_refused', eps)] = float(refused.mean())
                res[(a + '_ridge_on', eps)] = float(np.mean(s1['lam'] > 0))
        if 'joint' in which:
            U = unit_noise(np.random.default_rng([*seed_key, e, FAM['joint']]), R, D.d, D.dR)
            for s in SET_NAMES:
                th, s1, refused = v31_from_unit(G0, c0, D.dR, pre['v31_joint']['cal'][(s, eps)], sts[s], U)
                a = f'v31_joint_{s}'
                res[(a, eps)] = ev.test_mse(Xt, yt, th)
                res[(a + '_lam', eps)] = float(np.median(s1['lam']))
                res[(a + '_refused', eps)] = float(refused.mean())
                res[(a + '_ridge_on', eps)] = float(np.mean(s1['lam'] > 0))
        if 'adassp' in which:
            th, lam = ev.adassp(G0, c0, eps, delta, R, np.random.default_rng([*seed_key, e, FAM['adassp_pub']]),
                                variant='published', lmin=lmin)
            res[('adassp_pub', eps)] = ev.test_mse(Xt, yt, th)
            res[('adassp_pub_lam', eps)] = float(np.median(lam))
            th, lam = ev.adassp_matched(G0, c0, D.dR, eps, delta, R,
                                        np.random.default_rng([*seed_key, e, FAM['adassp_matched']]), lmin=lmin)
            res[('adassp_matched', eps)] = ev.test_mse(Xt, yt, th)
            res[('adassp_matched_lam', eps)] = float(np.median(lam))
    return res


# ------------------------------------------------------------------------------------------- benchmark
XKEYS = [f'{a}_{x}' for a in V31_ARMS for x in ('lam', 'refused', 'ridge_on')] + \
        [f'v31_blocknorm_{s}_psiO' for s in SET_NAMES] + ['v31_zscore_I_psiO', 'adassp_pub_lam', 'adassp_matched_lam']


def bench_dataset(name, R, delta, sts):
    t0 = time.time()
    D = rdm.Data(name)
    E = len(cs.EPS)
    fold = {k: np.zeros(K) for k in NP_KEYS}
    for a in ARMS:
        fold[a] = np.zeros((E, K))
    extra = {k: np.zeros((E, K)) for k in XKEYS}
    for f in range(K):
        te, ref, m = D.split(f)
        pre = prep(D, ref, delta, sts)
        if f == 0:
            bounds0 = {k: pre[k]['bounds'] for k in rdm.V31}
            cal0 = {k: {f'{s}|{e}': v for (s, e), v in pre[k]['cal'].items()} for k in ('v31_blocknorm', 'v31_zscore', 'v31_joint')}
        res = evaluate(D, pre, m, te, cs.EPS, R, (20261024, f, sum(map(ord, name))), delta, sts)
        for k in NP_KEYS:
            fold[k][f] = res[k]
        for e, eps in enumerate(cs.EPS):
            for a in ARMS:
                fold[a][e, f] = float(np.mean(res[(a, eps)]))
            for k in XKEYS:
                extra[k][e, f] = res[(k, eps)]
    return dict(name=name, n=D.n, d=D.d, d_R=D.dR, d_O=D.dO, n_matched=int(len(m)), Mw=D.Mw, bounds_fold0=bounds0,
                calibration_fold0=cal0, fold=fold, extra=extra), f'{name}: {time.time() - t0:.0f}s'


def cmd_benchmark(a):
    sts = settings()
    names = a.datasets or list(WD.PUBLISHED_ORDER)
    cs.precompute_r1(r1_keys(names, sorted({sts[s]['alpha_L'] for s in sts})), a.workers)
    prev = ev.load_json(os.path.join(cs.REPORT_EVAL, 'adassp_benchmark.json'))
    prev_raw = {r['name']: r for r in prev['raw']}
    results = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {nm: ex.submit(bench_dataset, nm, a.R, a.delta, sts) for nm in sorted(names, key=lambda s: -rdm.Data_size(s))}
        for nm in names:
            r, msg = futs[nm].result()
            results.append(r)
            print(msg, flush=True)
    summ = []
    for r in results:
        F = dict(r['fold'])
        pr = prev_raw[r['name']]['fold']
        for k in ('v31_joint', 'v31_blocknorm', 'v31_zscore'):
            F[f'{k}_prev'] = np.asarray(pr[k], float)
        # the previous run's non-private baselines are on the same rows: check
        for k in NP_KEYS:
            assert np.allclose(F[k], pr[k], rtol=1e-9), (r['name'], k)
        row = dict(name=r['name'], n=r['n'], d=r['d'], d_R=r['d_R'], n_matched=r['n_matched'],
                   np_abs={k: float(np.mean(F[k])) for k in NP_KEYS}, by_eps={})
        allarms = ARMS + ['v31_joint_prev', 'v31_blocknorm_prev', 'v31_zscore_prev']
        for e, eps in enumerate(cs.EPS):
            d = {}
            for mth in allarms:
                v = F[mth][e]
                own = F[OWN[mth]]
                ro = v / own
                d[mth] = dict(abs=float(v.mean()), abs_ci_half=float(ev.t95(K) * v.std(ddof=1) / math.sqrt(K)),
                              ratio_own=float(v.mean() / own.mean()), ratio_own_ci_half=float(ev.t95(K) * ro.std(ddof=1) / math.sqrt(K)),
                              ratio_common=float(v.mean() / F['np_common'].mean()))
            d['paired_abs'] = {f'{x}_vs_{y}': rdm.paired(F[x][e], F[y][e]) for x, y in PAIRS}
            d['paired_ratio_own'] = {f'{x}_vs_{y}': rdm.paired(F[x][e] / F[OWN[x]], F[y][e] / F[OWN[y]]) for x, y in PAIRS}
            d['lam_median'] = {k: float(np.median(r['extra'][k + '_lam'][e])) for k in V31_ARMS + ['adassp_pub', 'adassp_matched']}
            d['refused'] = {k: float(r['extra'][k + '_refused'][e].mean()) for k in V31_ARMS}
            d['refused_max'] = {k: float(r['extra'][k + '_refused'][e].max()) for k in V31_ARMS}
            d['ridge_on'] = {k: float(r['extra'][k + '_ridge_on'][e].mean()) for k in V31_ARMS}
            pl = prev['summary'][[s['name'] for s in prev['summary']].index(r['name'])]['by_eps'][str(eps)]['lam_median']
            d['lam_median_prev'] = {f'{k}_prev': pl[k] for k in ('v31_joint', 'v31_blocknorm', 'v31_zscore')}
            d['lam_median_adassp_prev_run'] = {k: pl[k] for k in ('adassp_pub', 'adassp_matched')}
            row['by_eps'][str(eps)] = d
        summ.append(row)
    verdicts = {}
    for basis in ('paired_abs', 'paired_ratio_own'):
        verdicts[basis] = {}
        for x, y in PAIRS:
            key = f'{x}_vs_{y}'
            verdicts[basis][key] = {}
            for eps in cs.EPS:
                v = [s['by_eps'][str(eps)][basis][key]['verdict'] for s in summ]
                verdicts[basis][key][str(eps)] = dict(better=v.count('better'), worse=v.count('worse'), none=v.count('none'))
    gaps = {}
    for eps in cs.EPS:
        g = {}
        for x, y in [('v31_joint_I', 'adassp_pub'), ('v31_joint_P', 'adassp_pub'), ('v31_joint_I', 'adassp_matched'),
                     ('v31_joint_P', 'adassp_matched'), ('v31_blocknorm_I', 'adassp_pub'), ('v31_blocknorm_P', 'adassp_pub'),
                     ('v31_blocknorm_I', 'adassp_matched'), ('v31_blocknorm_P', 'adassp_matched'),
                     ('v31_joint_prev', 'adassp_pub'), ('v31_blocknorm_prev', 'adassp_pub'), ('v31_joint_prev', 'adassp_matched'),
                     ('v31_blocknorm_P', 'v31_blocknorm_I'), ('v31_blocknorm_I', 'v31_blocknorm_prev'),
                     ('v31_blocknorm_P', 'v31_blocknorm_prev'), ('v31_zscore_I', 'v31_blocknorm_I')]:
            v = np.array([s['by_eps'][str(eps)][x]['abs'] / s['by_eps'][str(eps)][y]['abs'] - 1 for s in summ])
            g[f'{x}/{y}-1'] = dict(median=float(np.median(v)), min=float(v.min()), max=float(v.max()))
        lam = {}
        for x, y in [('v31_joint_I', 'adassp_matched'), ('v31_joint_P', 'adassp_matched'), ('v31_joint_I', 'adassp_pub'),
                     ('v31_joint_P', 'adassp_pub'), ('v31_joint_prev', 'adassp_matched')]:
            rat = []
            for s in summ:
                lx = s['by_eps'][str(eps)]['lam_median'].get(x, s['by_eps'][str(eps)]['lam_median_prev'].get(x))
                ly = s['by_eps'][str(eps)]['lam_median'][y]
                if ly > 0 and lx > 0:
                    rat.append(lx / ly)
            lam[f'{x}/{y}'] = dict(median=float(np.median(rat)) if rat else None,
                                   min=float(np.min(rat)) if rat else None, max=float(np.max(rat)) if rat else None,
                                   count=len(rat))
        gaps[str(eps)] = dict(abs_gap=g, lam_ratio=lam)
    out = dict(settings=dict(delta=a.delta, eps=cs.EPS, R=a.R, folds=K, frac_ref=rdm.FRAC_REF, arms=LABEL,
                             v31_settings=sts, previous_setting=cs.PREV,
                             previous_results='reports/v31_eval/adassp_benchmark.json (not rerun)',
                             common_y_scale="Wang's y / max|y| (y centred on the full data)"),
               summary=summ, verdicts=verdicts, gaps=gaps,
               previous_verdicts=prev['verdicts'], raw=results)
    tag = '' if a.datasets is None else '_' + '_'.join(a.datasets)
    path = os.path.join(cs.REPORT, f'adassp_benchmark{tag}.json')
    ev.save_json(out, path)
    print('wrote', path)


# ------------------------------------------------------------------------------------------- sweep
SW_ARMS = ['v31_blocknorm_I', 'v31_blocknorm_P', 'adassp_pub', 'adassp_matched']


def sweep_fold(name, f, R, points, delta, sts):
    t0 = time.time()
    D = rdm.Data(name)
    te, ref, pool = D.split(f)
    pre = prep(D, ref, delta, sts)
    nmax = min(len(D.split(g)[2]) for g in range(K))
    grid = np.unique(np.concatenate([np.round(np.geomspace(100, nmax, points)).astype(int), [1000, 10000]]))
    grid = [int(v) for v in grid if v <= nmax]
    rows = []
    for g, n in enumerate(grid):
        res = evaluate(D, pre, pool[:n], te, cs.EPS, R, (20261025, f, g, sum(map(ord, name))), delta, sts,
                       which=('blocknorm', 'adassp'))
        row = {k: res[k] for k in NP_KEYS}
        row.update(n=n, eps={})
        for eps in cs.EPS:
            row['eps'][str(eps)] = {mth: float(np.mean(res[(mth, eps)])) for mth in SW_ARMS}
            for s in SET_NAMES:
                for x in ('lam', 'refused', 'ridge_on'):
                    row['eps'][str(eps)][f'v31_blocknorm_{s}_{x}'] = res[(f'v31_blocknorm_{s}_{x}', eps)]
        rows.append(row)
    return dict(name=name, fold=f, grid=grid, rows=rows, n=D.n, d=D.d), f'{name} fold {f}: {time.time() - t0:.0f}s'


def cmd_sweep(a):
    sts = settings()
    names = a.datasets or SWEEP
    cs.precompute_r1(r1_keys(names, sorted({sts[s]['alpha_L'] for s in sts})), a.workers)
    out = {}
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = [(nm, f, ex.submit(sweep_fold, nm, f, a.R, a.points, a.delta, sts)) for nm in names for f in range(K)]
        for nm, f, fu in futs:
            r, msg = fu.result()
            out.setdefault(nm, []).append(r)
            print(msg, flush=True)
    prev = ev.load_json(os.path.join(cs.REPORT_EVAL, 'sweep_n.json'))['summary']
    summ = {}
    for nm, folds in out.items():
        grid = folds[0]['grid']
        G = len(grid)
        arr = lambda key: np.array([[fo['rows'][g][key] for fo in folds] for g in range(G)])   # (G, K)
        npk = {k: arr(k) for k in NP_KEYS}
        assert grid == prev[nm]['grid']
        S = dict(grid=grid, n=folds[0]['n'], d=folds[0]['d'], np_abs={k: v.mean(1) for k, v in npk.items()}, eps={})
        for eps in cs.EPS:
            E = {}
            for mth in SW_ARMS:
                v = np.array([[fo['rows'][g]['eps'][str(eps)][mth] for fo in folds] for g in range(G)])
                ro = v / npk[OWN[mth]]
                rc = v / npk['np_common']
                E[mth] = dict(abs=v.mean(1), ratio_own=ro.mean(1), ratio_own_ci=ev.t95(K) * ro.std(1, ddof=1) / math.sqrt(K),
                              ratio_common=rc.mean(1),
                              cross_own_1p10=ev.crossing(grid, ro.mean(1), 1.10),
                              cross_own_1p25=ev.crossing(grid, ro.mean(1), 1.25),
                              cross_common_1p10=ev.crossing(grid, rc.mean(1), 1.10))
            for s in SET_NAMES:
                for x in ('lam', 'refused', 'ridge_on'):
                    k = f'v31_blocknorm_{s}_{x}'
                    vals = [[fo['rows'][g]['eps'][str(eps)][k] for fo in folds] for g in range(G)]
                    E[k] = [float(np.median(v)) if x == 'lam' else float(np.mean(v)) for v in vals]
            pe = prev[nm]['eps'][str(eps)]['v31_blocknorm']
            E['v31_blocknorm_prev'] = dict(ratio_own=pe['ratio_own'], ratio_own_ci=pe['ratio_own_ci'],
                                           ratio_common=pe['ratio_common'], cross_own_1p10=pe['cross_own_1p10'],
                                           cross_own_1p25=pe['cross_own_1p25'], cross_common_1p10=pe['cross_common_1p10'])
            E['v31_blocknorm_prev_lam'] = prev[nm]['eps'][str(eps)]['v31_blocknorm_lam']
            pa = prev[nm]['eps'][str(eps)]['adassp_pub']
            E['adassp_pub_prev_run'] = dict(cross_own_1p10=pa['cross_own_1p10'])
            S['eps'][str(eps)] = E
        summ[nm] = S
    path = os.path.join(cs.REPORT, 'sweep_n.json')
    ev.save_json(dict(settings=dict(delta=a.delta, eps=cs.EPS, R=a.R, folds=K, frac_ref=rdm.FRAC_REF,
                                    arms={k: LABEL[k] for k in SW_ARMS}, v31_settings=sts,
                                    previous_results='reports/v31_eval/sweep_n.json (not rerun)'),
                      summary=summ), path)
    print('wrote', path)


def cmd_radii(a):
    """Fill the r_1 cache for every dataset of the benchmark and the sweep (both settings' alpha_L)."""
    sts = settings()
    n = cs.precompute_r1(r1_keys(list(WD.PUBLISHED_ORDER), sorted({sts[s]['alpha_L'] for s in sts})), a.workers)
    print(f'computed {n} radii', flush=True)


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
    r = sub.add_parser('radii')
    r.add_argument('--workers', type=int, default=4)
    a = ap.parse_args()
    {'benchmark': cmd_benchmark, 'sweep': cmd_sweep, 'radii': cmd_radii}[a.cmd](a)


if __name__ == '__main__':
    main()
