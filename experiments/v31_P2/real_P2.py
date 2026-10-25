"""Sections 13.7 and 13.8 under P2 (steps 2 and 3 of "what happens next" in reports/v31_P2/settings_choice_P2.md).

Same data, folds, 10% public reference rows, matched rows, preprocessing and reporting as
experiments/v31_settings/real_settings.py.  Only the v3.1 arms (c) joint normalization and (d) block normalization
are run under P2.  Their unit noise comes from the streams of the earlier run (same seed keys and arm families), so
P2 is paired draw for draw with I and the old P.  As a check, I is recomputed on the same path and must reproduce
the published fold values.  (a) AdaSSP published, (b) AdaSSP matched, I and the old P under (c) and (d), and (e)
are read from reports/v31_settings/adassp_benchmark.json and sweep_n.json; the previous setting from
reports/v31_eval.  None of them is rerun.

    python experiments/v31_P2/real_P2.py benchmark [--delta 1e-6] [--R 50] [--workers 4]
    python experiments/v31_P2/real_P2.py sweep [--delta 1e-9] [--R 40] [--workers 4]
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
from p2common import cs, ev  # noqa: E402

import real_data as rdm  # noqa: E402  (experiments/v31_eval/real_data.py)
import real_settings as rs  # noqa: E402  (experiments/v31_settings/real_settings.py: unit noise, solve, families)
import datasets as WD  # noqa: E402

K = rdm.K
EPS = cs.EPS
NEW = ['v31_joint_P2', 'v31_blocknorm_P2']
CHK = {'v31_joint_Ichk': 'v31_joint_I', 'v31_blocknorm_Ichk': 'v31_blocknorm_I'}
OLD = ['adassp_pub', 'adassp_matched', 'v31_joint_I', 'v31_joint_P', 'v31_blocknorm_I', 'v31_blocknorm_P', 'v31_zscore_I']
PREV = ['v31_joint_prev', 'v31_blocknorm_prev', 'v31_zscore_prev']
OWN = dict(rs.OWN, v31_joint_P2='np_wang', v31_blocknorm_P2='np_v31_blocknorm')
LABEL = dict(rs.LABEL, v31_joint_P2='(c) v3.1 P2, joint row norm (idealized)', v31_blocknorm_P2='(d) v3.1 P2, block unit norm',
             v31_joint_prev='(c) previous', v31_blocknorm_prev='(d) previous', v31_zscore_prev='(e) previous')
PAIRS = [('v31_blocknorm_P2', 'adassp_pub'), ('v31_blocknorm_P2', 'adassp_matched'),
         ('v31_joint_P2', 'adassp_pub'), ('v31_joint_P2', 'adassp_matched'),
         ('v31_blocknorm_P2', 'v31_blocknorm_I'), ('v31_blocknorm_P2', 'v31_blocknorm_P'),
         ('v31_joint_P2', 'v31_joint_I'), ('v31_joint_P2', 'v31_joint_P'),
         ('v31_blocknorm_P2', 'v31_joint_P2'), ('v31_blocknorm_P2', 'v31_blocknorm_prev'),
         ('v31_blocknorm_P2', 'v31_zscore_I'),
         # reference pairs recomputed from the published fold values (must equal the published verdicts)
         ('v31_blocknorm_I', 'adassp_pub'), ('v31_blocknorm_P', 'adassp_pub'), ('v31_joint_I', 'adassp_pub'),
         ('v31_joint_P', 'adassp_pub'), ('v31_joint_I', 'adassp_matched'), ('v31_joint_P', 'adassp_matched'),
         ('v31_blocknorm_I', 'adassp_matched'), ('v31_blocknorm_P', 'adassp_matched'),
         ('v31_blocknorm_prev', 'adassp_pub'), ('v31_joint_prev', 'adassp_pub'), ('v31_joint_prev', 'adassp_matched')]
GAP_PAIRS = [('v31_blocknorm_P2', 'adassp_pub'), ('v31_blocknorm_P2', 'adassp_matched'), ('v31_joint_P2', 'adassp_pub'),
             ('v31_joint_P2', 'adassp_matched'), ('v31_blocknorm_P2', 'v31_blocknorm_I'),
             ('v31_blocknorm_P2', 'v31_blocknorm_P'), ('v31_joint_P2', 'v31_joint_I'), ('v31_joint_P2', 'v31_joint_P'),
             ('v31_blocknorm_P2', 'v31_blocknorm_prev'), ('v31_blocknorm_P2', 'v31_joint_P2'),
             ('v31_blocknorm_I', 'adassp_pub'), ('v31_blocknorm_P', 'adassp_pub'), ('v31_blocknorm_prev', 'adassp_pub'),
             ('v31_joint_I', 'adassp_pub'), ('v31_joint_P', 'adassp_pub'), ('v31_joint_I', 'adassp_matched'),
             ('v31_joint_P', 'adassp_matched'), ('v31_blocknorm_I', 'adassp_matched'), ('v31_blocknorm_P', 'adassp_matched'),
             ('v31_joint_prev', 'adassp_matched')]
XK = ('lam', 'refused', 'ridge_on', 'psiO')


def load_sets():
    s = ev.load_json(os.path.join(pc.REPORT, 'setting_P2.json'))
    return {'P2': dict(s['P2']), 'I': dict(s['I'])}


def dims(name):
    prev = ev.load_json(os.path.join(cs.REPORT_EVAL, 'adassp_benchmark.json'))
    return {s['name']: s['d'] for s in prev['summary']}[name]


def r1_keys(names, alpha_L):
    keys = []
    for nm in names:
        d = dims(nm)
        dR, dO = d // 2, d - d // 2
        keys += [(dR + 1, dO, alpha_L), (dR, dO, alpha_L)]
    return keys


def prep(D, ref, delta, sts):
    Z, yz, yc, b, sy = ev.per_party_preprocess(D.F, D.y, ref, D.dR, 'blocknorm')
    bn = dict(Z=Z, yz=yz, yc=yc, bounds=b, unit=(sy / D.Mw) ** 2,
              cal={(s, e): cs.calibrate(e, b['B_R'], b['B_O'], b['B_y'], D.dR, D.dO, sts[s]['f1'], delta=delta)
                   for s in sts for e in EPS})
    jt = dict(cal={(s, e): cs.calibrate(e, 1.0, 1.0, 1.0, D.dR, D.dO, sts[s]['f1'], delta=delta) for s in sts for e in EPS})
    return bn, jt


def evaluate(D, bn, jt, m, te, R, seed_key, sts, which=('joint', 'blocknorm')):
    """P2 (and the I check) under (c) and (d) on matched rows m, test rows te; unit noise as real_settings.evaluate."""
    res = {}
    Xm, ym = D.Xw[m], D.yw[m]
    G0, c0 = Xm.T @ Xm, Xm.T @ ym
    Xt, yt = D.Xw[te], D.yw[te]
    res['np_v31_blocknorm'] = ev.test_mse(bn['Z'][te], bn['yz'][te], ev.ridge1(bn['Z'][m], bn['yc'][m])) * bn['unit']
    res['np_wang'] = ev.test_mse(Xt, yt, np.linalg.solve(G0 + np.eye(D.d), c0))
    for e, eps in enumerate(EPS):
        if 'blocknorm' in which:
            Z, yc = bn['Z'][m], bn['yc'][m]
            G, c = Z.T @ Z, Z.T @ yc
            U = rs.unit_noise(np.random.default_rng([*seed_key, e, rs.FAM['blocknorm']]), R, G.shape[0], D.dR + 1)
            for s, arm in (('P2', 'v31_blocknorm_P2'), ('I', 'v31_blocknorm_Ichk')):
                th, s1, refused = rs.v31_from_unit(G, c, D.dR + 1, bn['cal'][(s, eps)], sts[s], U)
                res[(arm, eps)] = float(np.mean(ev.test_mse(bn['Z'][te], bn['yz'][te], th) * bn['unit']))
                res[(arm + '_lam', eps)] = float(np.median(s1['lam']))
                res[(arm + '_refused', eps)] = float(refused.mean())
                res[(arm + '_ridge_on', eps)] = float(np.mean(s1['lam'] > 0))
                res[(arm + '_psiO', eps)] = float(np.mean(s1['kind'] == 'O'))
        if 'joint' in which:
            U = rs.unit_noise(np.random.default_rng([*seed_key, e, rs.FAM['joint']]), R, D.d, D.dR)
            for s, arm in (('P2', 'v31_joint_P2'), ('I', 'v31_joint_Ichk')):
                th, s1, refused = rs.v31_from_unit(G0, c0, D.dR, jt['cal'][(s, eps)], sts[s], U)
                res[(arm, eps)] = float(np.mean(ev.test_mse(Xt, yt, th)))
                res[(arm + '_lam', eps)] = float(np.median(s1['lam']))
                res[(arm + '_refused', eps)] = float(refused.mean())
                res[(arm + '_ridge_on', eps)] = float(np.mean(s1['lam'] > 0))
                res[(arm + '_psiO', eps)] = float(np.mean(s1['kind'] == 'O'))
    return res


# ------------------------------------------------------------------------------------------- benchmark
def bench_dataset(name, R, delta, sts):
    t0 = time.time()
    D = rdm.Data(name)
    E = len(EPS)
    arms = NEW + list(CHK)
    fold = {a: np.zeros((E, K)) for a in arms}
    extra = {f'{a}_{x}': np.zeros((E, K)) for a in arms for x in XK}
    npv = {k: np.zeros(K) for k in ('np_wang', 'np_v31_blocknorm')}
    for f in range(K):
        te, ref, m = D.split(f)
        bn, jt = prep(D, ref, delta, sts)
        if f == 0:
            cal0 = {k: {f'{s}|{e}': v for (s, e), v in d['cal'].items() if s == 'P2'} for k, d in (('v31_blocknorm', bn), ('v31_joint', jt))}
            b0 = bn['bounds']
        res = evaluate(D, bn, jt, m, te, R, (20261024, f, sum(map(ord, name))), sts)
        for k in npv:
            npv[k][f] = res[k]
        for e, eps in enumerate(EPS):
            for a in arms:
                fold[a][e, f] = res[(a, eps)]
                for x in XK:
                    extra[f'{a}_{x}'][e, f] = res[(f'{a}_{x}', eps)]
    return dict(name=name, n=D.n, d=D.d, d_R=D.dR, d_O=D.dO, n_matched=int(len(m)), fold=fold, extra=extra, np=npv,
                bounds_fold0=b0, calibration_fold0=cal0), f'{name}: {time.time() - t0:.0f}s'


def cmd_benchmark(a):
    sts = load_sets()
    names = a.datasets or list(WD.PUBLISHED_ORDER)
    cs.precompute_r1(r1_keys(names, sts['P2']['alpha_L']), a.workers)
    pub = ev.load_json(os.path.join(pc.REPORT_SET, 'adassp_benchmark.json'))
    pub_raw = {r['name']: r for r in pub['raw']}
    pub_sum = {s['name']: s for s in pub['summary']}
    prev = ev.load_json(os.path.join(cs.REPORT_EVAL, 'adassp_benchmark.json'))
    prev_raw = {r['name']: r for r in prev['raw']}
    results = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {nm: ex.submit(bench_dataset, nm, a.R, a.delta, sts) for nm in sorted(names, key=lambda s: -rdm.Data_size(s))}
        for nm in names:
            r, msg = futs[nm].result()
            results.append(r)
            print(msg, flush=True)
    summ, check = [], {}
    for r in results:
        nm = r['name']
        F = {k: np.asarray(v, float) for k, v in pub_raw[nm]['fold'].items()}
        for k in ('v31_joint', 'v31_blocknorm', 'v31_zscore'):
            F[f'{k}_prev'] = np.asarray(prev_raw[nm]['fold'][k], float)
        for k in ('np_wang', 'np_v31_blocknorm'):
            assert np.allclose(r['np'][k], F[k], rtol=1e-9), (nm, k)
        for k in NEW:
            F[k] = r['fold'][k]
        check[nm] = {c: float(np.max(np.abs(r['fold'][c] / F[o] - 1))) for c, o in CHK.items()}
        row = dict(name=nm, n=r['n'], d=r['d'], d_R=r['d_R'], n_matched=r['n_matched'],
                   np_abs={k: float(np.mean(F[k])) for k in rdm.NP_KEYS}, by_eps={},
                   bounds_fold0=r['bounds_fold0'],
                   sigma_ratio_fold0={k: {e: v['sigma1'] / v['sigma2'] for e, v in d.items()} for k, d in r['calibration_fold0'].items()})
        for e, eps in enumerate(EPS):
            d = {}
            for mth in NEW + OLD + PREV:
                v = F[mth][e]
                own = F[OWN[mth]]
                ro = v / own
                d[mth] = dict(abs=float(v.mean()), ratio_own=float(v.mean() / own.mean()),
                              ratio_own_ci_half=float(ev.t95(K) * ro.std(ddof=1) / math.sqrt(K)),
                              ratio_common=float(v.mean() / F['np_common'].mean()))
            d['paired_abs'] = {f'{x}_vs_{y}': rdm.paired(F[x][e], F[y][e]) for x, y in PAIRS}
            d['paired_ratio_own'] = {f'{x}_vs_{y}': rdm.paired(F[x][e] / F[OWN[x]], F[y][e] / F[OWN[y]]) for x, y in PAIRS}
            ps = pub_sum[nm]['by_eps'][str(eps)]
            d['lam_median'] = {k: float(np.median(r['extra'][k + '_lam'][e])) for k in NEW}
            d['lam_median'].update({k: ps['lam_median'][k] for k in ps['lam_median']})
            d['refused'] = {k: float(r['extra'][k + '_refused'][e].mean()) for k in NEW}
            d['refused_max'] = {k: float(r['extra'][k + '_refused'][e].max()) for k in NEW}
            d['refused_published'] = ps['refused']
            d['ridge_on'] = {k: float(r['extra'][k + '_ridge_on'][e].mean()) for k in NEW}
            d['psiO'] = {k: float(r['extra'][k + '_psiO'][e].mean()) for k in NEW}
            row['by_eps'][str(eps)] = d
        summ.append(row)
    verdicts = {}
    for basis in ('paired_abs', 'paired_ratio_own'):
        verdicts[basis] = {}
        for x, y in PAIRS:
            key = f'{x}_vs_{y}'
            verdicts[basis][key] = {}
            for eps in EPS:
                v = [s['by_eps'][str(eps)][basis][key]['verdict'] for s in summ]
                verdicts[basis][key][str(eps)] = dict(better=v.count('better'), worse=v.count('worse'), none=v.count('none'))
    # the recomputed reference verdicts must equal the published ones
    pubv = pub['verdicts']['paired_abs']
    ref_ok = all(verdicts['paired_abs'][k] == pubv[k] for k in pubv if k in verdicts['paired_abs'])
    gaps = {}
    for eps in EPS:
        g = {}
        for x, y in GAP_PAIRS:
            v = np.array([s['by_eps'][str(eps)][x]['abs'] / s['by_eps'][str(eps)][y]['abs'] - 1 for s in summ])
            g[f'{x}/{y}-1'] = dict(median=float(np.median(v)), min=float(v.min()), max=float(v.max()),
                                   argmin=summ[int(np.argmin(v))]['name'], argmax=summ[int(np.argmax(v))]['name'])
        lam = {}
        for x, y in [('v31_joint_P2', 'adassp_matched'), ('v31_joint_P2', 'adassp_pub'), ('v31_joint_I', 'adassp_matched'),
                     ('v31_joint_P', 'adassp_matched'), ('v31_blocknorm_P2', 'v31_blocknorm_I')]:
            rat = []
            for s in summ:
                lx, ly = s['by_eps'][str(eps)]['lam_median'][x], s['by_eps'][str(eps)]['lam_median'][y]
                if lx > 0 and ly > 0:
                    rat.append(lx / ly)
            lam[f'{x}/{y}'] = dict(median=float(np.median(rat)) if rat else None, min=float(np.min(rat)) if rat else None,
                                   max=float(np.max(rat)) if rat else None, count=len(rat))
        refs = {}
        for k in NEW + ['v31_blocknorm_P', 'v31_joint_P', 'v31_blocknorm_I', 'v31_joint_I']:
            if k in NEW:
                v = np.array([s['by_eps'][str(eps)]['refused'][k] for s in summ])
            else:
                v = np.array([s['by_eps'][str(eps)]['refused_published'][k] for s in summ])
            refs[k] = dict(mean=float(v.mean()), max=float(v.max()), argmax=summ[int(np.argmax(v))]['name'],
                           datasets_ge_25pct=int((v >= 0.25).sum()), datasets_ge_1pct=int((v > 0.01).sum()),
                           datasets_all=int((v >= 0.999).sum()))
        gaps[str(eps)] = dict(abs_gap=g, lam_ratio=lam, refused=refs)
    overall = {}
    for k in NEW + ['v31_blocknorm_P', 'v31_joint_P']:
        key = 'refused' if k in NEW else 'refused_published'
        v = np.array([[s['by_eps'][str(e)][key][k] for e in EPS] for s in summ])     # (29, 5)
        overall[k] = dict(mean_all=float(v.mean()), datasets_ge_25pct_any_eps=int((v.max(1) >= 0.25).sum()),
                          datasets_mean_ge_25pct=int((v.mean(1) >= 0.25).sum()),
                          worst=[(summ[i]['name'], float(v[i].mean())) for i in np.argsort(-v.mean(1))[:8]])
    out = dict(settings=dict(delta=a.delta, eps=EPS, R=a.R, folds=K, frac_ref=rdm.FRAC_REF,
                             arms={k: LABEL[k] for k in NEW + OLD + PREV}, P2=sts['P2'],
                             reference='reports/v31_settings/adassp_benchmark.json (a, b, I, P, e: not rerun); '
                                       'reports/v31_eval/adassp_benchmark.json (previous setting: not rerun)'),
               reproduction_check=dict(max_rel_diff_I_recomputed_vs_published=check,
                                       reference_verdicts_equal_published=bool(ref_ok)),
               summary=summ, verdicts=verdicts, gaps=gaps, refused_overall=overall,
               raw=[dict(name=r['name'], fold={k: r['fold'][k] for k in NEW},
                         extra={k: v for k, v in r['extra'].items() if k.startswith(tuple(NEW))}) for r in results])
    tag = '' if a.datasets is None else '_' + '_'.join(a.datasets)
    path = os.path.join(pc.REPORT, f'adassp_benchmark_P2{tag}.json')
    ev.save_json(out, path)
    print('reference verdicts equal published:', ref_ok, '; I recomputed vs published, max rel diff:',
          max(max(v.values()) for v in check.values()))
    print('wrote', path)


# ------------------------------------------------------------------------------------------- sweep
def sweep_fold(name, f, R, points, delta, sts):
    t0 = time.time()
    D = rdm.Data(name)
    te, ref, pool = D.split(f)
    bn, jt = prep(D, ref, delta, sts)
    nmax = min(len(D.split(g)[2]) for g in range(K))
    grid = np.unique(np.concatenate([np.round(np.geomspace(100, nmax, points)).astype(int), [1000, 10000]]))
    grid = [int(v) for v in grid if v <= nmax]
    rows = []
    for g, n in enumerate(grid):
        res = evaluate(D, bn, jt, pool[:n], te, R, (20261025, f, g, sum(map(ord, name))), sts, which=('blocknorm',))
        row = dict(n=n, np_v31_blocknorm=res['np_v31_blocknorm'], eps={})
        for eps in EPS:
            row['eps'][str(eps)] = {k: res[(k, eps)] for k in ('v31_blocknorm_P2', 'v31_blocknorm_Ichk')}
            for x in XK:
                row['eps'][str(eps)][f'v31_blocknorm_P2_{x}'] = res[(f'v31_blocknorm_P2_{x}', eps)]
        rows.append(row)
    return dict(name=name, fold=f, grid=grid, rows=rows, n=D.n, d=D.d,
                sigma_ratio={str(e): bn['cal'][('P2', e)]['sigma1'] / bn['cal'][('P2', e)]['sigma2'] for e in EPS}), \
        f'{name} fold {f}: {time.time() - t0:.0f}s'


def cmd_sweep(a):
    sts = load_sets()
    names = a.datasets or rdm.SWEEP
    cs.precompute_r1(r1_keys(names, sts['P2']['alpha_L']), a.workers)
    out = {}
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = [(nm, f, ex.submit(sweep_fold, nm, f, a.R, a.points, a.delta, sts)) for nm in names for f in range(K)]
        for nm, f, fu in futs:
            r, msg = fu.result()
            out.setdefault(nm, []).append(r)
            print(msg, flush=True)
    pub = ev.load_json(os.path.join(pc.REPORT_SET, 'sweep_n.json'))['summary']
    summ, check = {}, {}
    for nm, folds in out.items():
        grid = folds[0]['grid']
        G = len(grid)
        assert grid == pub[nm]['grid']
        npb = np.array([[fo['rows'][g]['np_v31_blocknorm'] for fo in folds] for g in range(G)])
        assert np.allclose(npb.mean(1), pub[nm]['np_abs']['np_v31_blocknorm'], rtol=1e-9)
        S = dict(grid=grid, n=folds[0]['n'], d=folds[0]['d'], sigma_ratio=folds[0]['sigma_ratio'], eps={})
        mx = 0.0
        for eps in EPS:
            E = {}
            for arm in ('v31_blocknorm_P2', 'v31_blocknorm_Ichk'):
                v = np.array([[fo['rows'][g]['eps'][str(eps)][arm] for fo in folds] for g in range(G)])
                ro = v / npb
                E[arm] = dict(abs=v.mean(1), ratio_own=ro.mean(1), ratio_own_ci=ev.t95(K) * ro.std(1, ddof=1) / math.sqrt(K),
                              cross_own_1p10=ev.crossing(grid, ro.mean(1), 1.10),
                              cross_own_1p25=ev.crossing(grid, ro.mean(1), 1.25))
            mx = max(mx, float(np.max(np.abs(E['v31_blocknorm_Ichk']['ratio_own'] /
                                             np.array(pub[nm]['eps'][str(eps)]['v31_blocknorm_I']['ratio_own']) - 1))))
            for x in XK:
                vals = [[fo['rows'][g]['eps'][str(eps)][f'v31_blocknorm_P2_{x}'] for fo in folds] for g in range(G)]
                E[f'v31_blocknorm_P2_{x}'] = [float(np.median(v)) if x == 'lam' else float(np.mean(v)) for v in vals]
            pe = pub[nm]['eps'][str(eps)]
            E['reference'] = dict(
                I=dict(cross_own_1p10=pe['v31_blocknorm_I']['cross_own_1p10'], ratio_own=pe['v31_blocknorm_I']['ratio_own']),
                P_old=dict(cross_own_1p10=pe['v31_blocknorm_P']['cross_own_1p10'], ratio_own=pe['v31_blocknorm_P']['ratio_own'],
                           refused=pe['v31_blocknorm_P_refused']),
                prev=dict(cross_own_1p10=pe['v31_blocknorm_prev']['cross_own_1p10'], ratio_own=pe['v31_blocknorm_prev']['ratio_own']),
                adassp_pub=dict(cross_own_1p10=pe['adassp_pub']['cross_own_1p10'], ratio_own=pe['adassp_pub']['ratio_own']),
                I_lam=pe['v31_blocknorm_I_lam'])
            S['eps'][str(eps)] = E
        check[nm] = mx
        summ[nm] = S
    path = os.path.join(pc.REPORT, 'sweep_n_P2.json')
    ev.save_json(dict(settings=dict(delta=a.delta, eps=EPS, R=a.R, folds=K, frac_ref=rdm.FRAC_REF, P2=sts['P2'],
                                    reference='reports/v31_settings/sweep_n.json (I, P, previous, AdaSSP: not rerun)'),
                      reproduction_check=dict(max_rel_diff_I_recomputed_vs_published=check), summary=summ), path)
    print('I recomputed vs published (ratio_own), max rel diff:', check)
    print('wrote', path)


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
