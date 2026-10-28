"""Baseline 1 on Wang's 29 datasets: vertical full-record SSP against v3.1 (Section 13.7 design).

Full-record SSP aggregates the WHOLE Gram matrix securely (Karr et al. 2009; Hall et al. 2011; Gascon et al. 2017)
and adds noise to every released entry, R's block A included, under replace-one adjacency of a whole record
(x_R, x_O, y).  Everything else is v3.1's: the same two releases and budget split, the same accountant (tight zCDP
conversion, the same slack), the same stage-1 rule and gate, settings I and P.  The sensitivities are those of
experiments/baselines/sens_full.py, and the exact radius r_1 is the quantile of the full noise matrix (no zero block).
So the only difference from v3.1 is that A is noised too, with the larger sensitivity that this requires.

Data, folds, reference rows, matched rows and reporting are those of experiments/v31_settings/real_settings.py and
experiments/v31_P2/real_P2.py (delta = 1e-6, 10 folds, 50 noise draws, eps in {0.5, 1, 2, 4, 8}).  Arms:
  full_blocknorm_{I,P}  (d) per-party block unit norm (B_R^2 = 2 with R's intercept, B_O = 1, B_y from the reference)
  full_joint_{I,P}      (c) Wang's joint row normalization, no intercept, B_R = B_O = B_y = 1 (idealized)
The v3.1 arms (d) and (c) under I and P are recomputed on the same unit noise streams as the published runs and must
reproduce reports/v31_settings/adassp_benchmark.json (I) and reports/v31_P2/adassp_benchmark_P2.json (P) to 1e-12.
The full-record arms add noise on A from a separate stream to the SAME unit noise on the O-dependent entries, so each
full-record release is paired draw for draw with the v3.1 release it extends.

    python experiments/baselines/real_full.py radii [--workers 4]
    python experiments/baselines/real_full.py benchmark [--R 50] [--workers 4] [--datasets ...]
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

import real_data as rdm  # noqa: E402
import real_settings as rs  # noqa: E402
import datasets as WD  # noqa: E402

K = rdm.K
EPS = bc.EPS
DELTA = 1e-6
FAM_A = {'joint': 101, 'blocknorm': 102}          # streams of the extra noise on A
SETS = ('I', 'P')
V31 = {('blocknorm', 'I'): 'v31_blocknorm_I', ('blocknorm', 'P'): 'v31_blocknorm_P2',
       ('joint', 'I'): 'v31_joint_I', ('joint', 'P'): 'v31_joint_P2'}
FULL = {('blocknorm', 'I'): 'full_blocknorm_I', ('blocknorm', 'P'): 'full_blocknorm_P',
        ('joint', 'I'): 'full_joint_I', ('joint', 'P'): 'full_joint_P'}
OWN = {'v31_blocknorm_I': 'np_v31_blocknorm', 'v31_blocknorm_P2': 'np_v31_blocknorm',
       'full_blocknorm_I': 'np_v31_blocknorm', 'full_blocknorm_P': 'np_v31_blocknorm',
       'v31_joint_I': 'np_wang', 'v31_joint_P2': 'np_wang', 'full_joint_I': 'np_wang', 'full_joint_P': 'np_wang',
       'adassp_pub': 'np_wang', 'adassp_matched': 'np_wang'}
LABEL = {'full_blocknorm_I': 'full-record SSP (d) block norm, I', 'full_blocknorm_P': 'full-record SSP (d) block norm, P',
         'full_joint_I': 'full-record SSP (c) joint norm, I (idealized)',
         'full_joint_P': 'full-record SSP (c) joint norm, P (idealized)',
         'v31_blocknorm_I': 'v3.1 (d) block norm, I', 'v31_blocknorm_P2': 'v3.1 (d) block norm, P',
         'v31_joint_I': 'v3.1 (c) joint norm, I (idealized)', 'v31_joint_P2': 'v3.1 (c) joint norm, P (idealized)',
         'adassp_pub': '(a) AdaSSP, published', 'adassp_matched': '(b) AdaSSP, matched'}
NEW = list(FULL.values())
PAIRS = [('full_blocknorm_I', 'v31_blocknorm_I'), ('full_blocknorm_P', 'v31_blocknorm_P2'),
         ('full_joint_I', 'v31_joint_I'), ('full_joint_P', 'v31_joint_P2'),
         ('full_blocknorm_P', 'v31_blocknorm_I'), ('full_joint_I', 'v31_blocknorm_I'), ('full_joint_P', 'v31_blocknorm_P2'),
         ('full_blocknorm_I', 'adassp_pub'), ('full_blocknorm_P', 'adassp_pub'),
         ('full_joint_I', 'adassp_matched'), ('full_joint_P', 'adassp_matched'),
         ('v31_blocknorm_I', 'adassp_pub'), ('v31_blocknorm_P2', 'adassp_pub')]
XK = ('lam', 'refused', 'ridge_on', 'psiO')


def dims(name):
    prev = ev.load_json(os.path.join(cs.REPORT_EVAL, 'adassp_benchmark.json'))
    return {s['name']: (s['d'], s['d_R']) for s in prev['summary']}[name]


def r1_keys(names, sts):
    keys = []
    for nm in names:
        d, dR = dims(nm)
        for st in sts.values():
            keys += [(d + 1, True, st['alpha_L']), (d, False, st['alpha_L'])]
    return keys


def prep(D, ref, sts):
    Z, yz, yc, b, sy = ev.per_party_preprocess(D.F, D.y, ref, D.dR, 'blocknorm')
    bn = dict(Z=Z, yz=yz, yc=yc, bounds=b, unit=(sy / D.Mw) ** 2, cal={}, cal_full={})
    jt = dict(cal={}, cal_full={})
    for s, st in sts.items():
        for e in EPS:
            bn['cal'][(s, e)] = cs.calibrate(e, b['B_R'], b['B_O'], b['B_y'], D.dR, D.dO, st['f1'], delta=DELTA)
            bn['cal_full'][(s, e)] = bc.cal_full(e, b['B_R'], b['B_O'], b['B_y'], True, D.dR, D.dO, st['f1'], DELTA)
            jt['cal'][(s, e)] = cs.calibrate(e, 1.0, 1.0, 1.0, D.dR, D.dO, st['f1'], delta=DELTA)
            jt['cal_full'][(s, e)] = bc.cal_full(e, 1.0, 1.0, 1.0, False, D.dR, D.dO, st['f1'], DELTA)
    return bn, jt


def evaluate(D, bn, jt, m, te, R, seed_key, sts):
    res = {}
    Xm, ym = D.Xw[m], D.yw[m]
    G0, c0 = Xm.T @ Xm, Xm.T @ ym
    Xt, yt = D.Xw[te], D.yw[te]
    res['np_v31_blocknorm'] = ev.test_mse(bn['Z'][te], bn['yz'][te], ev.ridge1(bn['Z'][m], bn['yc'][m])) * bn['unit']
    res['np_wang'] = ev.test_mse(Xt, yt, np.linalg.solve(G0 + np.eye(D.d), c0))
    for e, eps in enumerate(EPS):
        for pp in ('blocknorm', 'joint'):
            if pp == 'blocknorm':
                Z, yc = bn['Z'][m], bn['yc'][m]
                G, c = Z.T @ Z, Z.T @ yc
                dA, ic, P_ = D.dR + 1, True, bn
                tmse = lambda th: ev.test_mse(bn['Z'][te], bn['yz'][te], th) * bn['unit']
            else:
                G, c = G0, c0
                dA, ic, P_ = D.dR, False, jt
                tmse = lambda th: ev.test_mse(Xt, yt, th)
            p = G.shape[0]
            U = rs.unit_noise(np.random.default_rng([*seed_key, e, rs.FAM[pp]]), R, p, dA)   # published stream
            ra = np.random.default_rng([*seed_key, e, FAM_A[pp]])
            ZA1, ZA2 = bc.a_block_noise(ra, R, p, dA, ic), bc.a_block_noise(ra, R, p, dA, ic)
            for s, st in sts.items():
                # v3.1, recomputed on the published stream (check)
                th, s1, refused = rs.v31_from_unit(G, c, dA, P_['cal'][(s, eps)], st, U)
                a = V31[(pp, s)]
                res[(a, eps)] = float(np.mean(tmse(th)))
                # full-record SSP: the same unit noise on the O-dependent entries plus noise on A
                cf = P_['cal_full'][(s, eps)]
                q = bc.r1_full(p, ic, st['alpha_L'])
                th, s1, gate, refused = bc.run_release(G, c, dA, cf, st, q, U[0] + ZA1, U[1] + ZA2, U[2])
                a = FULL[(pp, s)]
                res[(a, eps)] = float(np.mean(tmse(th)))
                res[(a + '_lam', eps)] = float(np.median(s1['lam']))
                res[(a + '_refused', eps)] = float(refused.mean())
                res[(a + '_ridge_on', eps)] = float(np.mean(s1['lam'] > 0))
                res[(a + '_psiO', eps)] = float(np.mean(s1['kind'] == 'O'))
    return res


def bench_dataset(name, R, sts):
    t0 = time.time()
    D = rdm.Data(name)
    E = len(EPS)
    chk = list(V31.values())
    fold = {a: np.zeros((E, K)) for a in NEW + chk}
    extra = {f'{a}_{x}': np.zeros((E, K)) for a in NEW for x in XK}
    npv = {k: np.zeros(K) for k in ('np_wang', 'np_v31_blocknorm')}
    cal0 = None
    for f in range(K):
        te, ref, m = D.split(f)
        bn, jt = prep(D, ref, sts)
        if f == 0:
            cal0 = {pp: {f'{s}|{e}': dict(v31=P_['cal'][(s, e)], full=P_['cal_full'][(s, e)]) for s in sts for e in EPS}
                    for pp, P_ in (('blocknorm', bn), ('joint', jt))}
            b0 = bn['bounds']
        res = evaluate(D, bn, jt, m, te, R, (20261024, f, sum(map(ord, name))), sts)
        for k in npv:
            npv[k][f] = res[k]
        for e, eps in enumerate(EPS):
            for a in NEW + chk:
                fold[a][e, f] = res[(a, eps)]
            for a in NEW:
                for x in XK:
                    extra[f'{a}_{x}'][e, f] = res[(f'{a}_{x}', eps)]
    return dict(name=name, n=D.n, d=D.d, d_R=D.dR, d_O=D.dO, n_matched=int(len(m)), fold=fold, extra=extra, np=npv,
                bounds_fold0=b0, calibration_fold0=cal0), f'{name}: {time.time() - t0:.0f}s'


def published():
    """Fold values of the published arms: I, (a), (b) from reports/v31_settings; P from reports/v31_P2."""
    s = ev.load_json(os.path.join(cs.REPORT, 'adassp_benchmark.json'))
    p2 = ev.load_json(os.path.join(bc.ROOT, 'reports', 'v31_P2', 'adassp_benchmark_P2.json'))
    out = {}
    for r in s['raw']:
        out[r['name']] = {k: np.asarray(r['fold'][k], float) for k in
                          ('np_wang', 'np_v31_blocknorm', 'adassp_pub', 'adassp_matched', 'v31_joint_I', 'v31_blocknorm_I')}
    for r in p2['raw']:
        for k in ('v31_joint_P2', 'v31_blocknorm_P2'):
            out[r['name']][k] = np.asarray(r['fold'][k], float)
    return out


def cmd_radii(a):
    sts = bc.settings()
    names = a.datasets or list(WD.PUBLISHED_ORDER)
    keys = r1_keys(names, sts) + [(11, True, st['alpha_L']) for st in sts.values()]     # + the synthetic population
    n = bc.precompute_r1_full(keys, a.workers)
    print(f'computed {n} full-record radii', flush=True)


def cmd_benchmark(a):
    sts = bc.settings()
    bc.check_settings_against_dp()
    names = a.datasets or list(WD.PUBLISHED_ORDER)
    bc.precompute_r1_full(r1_keys(names, sts), a.workers)
    pub = published()
    results = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {nm: ex.submit(bench_dataset, nm, a.R, sts) for nm in sorted(names, key=lambda s: -rdm.Data_size(s))}
        for nm in names:
            r, msg = futs[nm].result()
            results.append(r)
            print(msg, flush=True)
    summ, check = [], {}
    for r in results:
        nm = r['name']
        F = dict(pub[nm])
        for k in ('np_wang', 'np_v31_blocknorm'):
            assert np.allclose(r['np'][k], F[k], rtol=1e-9), (nm, k)
        check[nm] = {k: float(np.max(np.abs(r['fold'][k] / F[k] - 1))) for k in V31.values()}
        for k in NEW:
            F[k] = r['fold'][k]
        row = dict(name=nm, n=r['n'], d=r['d'], d_R=r['d_R'], n_matched=r['n_matched'],
                   np_abs={k: float(np.mean(F[k])) for k in ('np_wang', 'np_v31_blocknorm')}, by_eps={},
                   bounds_fold0=r['bounds_fold0'],
                   noise_fold0={pp: {k: dict(sigma1_ratio=v['full']['sigma1'] / v['v31']['sigma1'],
                                             sigma2_ratio=v['full']['sigma2'] / v['v31']['sigma2'],
                                             Delta_full=v['full']['Delta'], Delta_v31=v['v31']['Delta'],
                                             Delta_G_full=v['full']['Delta_G'], Delta_G_v31=v['v31']['Delta_G'],
                                             joint_full=v['full']['joint'], joint_v31=v['v31']['joint'])
                                      for k, v in d.items() if k.endswith('|1.0')}
                                 for pp, d in r['calibration_fold0'].items()})
        for e, eps in enumerate(EPS):
            d = {}
            for mth in NEW + list(V31.values()) + ['adassp_pub', 'adassp_matched']:
                v = F[mth][e]
                own = F[OWN[mth]]
                ro = v / own
                d[mth] = dict(abs=float(v.mean()), ratio_own=float(v.mean() / own.mean()),
                              ratio_own_ci_half=float(ev.t95(K) * ro.std(ddof=1) / math.sqrt(K)))
            d['paired_abs'] = {f'{x}_vs_{y}': rdm.paired(F[x][e], F[y][e]) for x, y in PAIRS}
            d['paired_ratio_own'] = {f'{x}_vs_{y}': rdm.paired(F[x][e] / F[OWN[x]], F[y][e] / F[OWN[y]]) for x, y in PAIRS}
            d['lam_median'] = {k: float(np.median(r['extra'][k + '_lam'][e])) for k in NEW}
            d['refused'] = {k: float(r['extra'][k + '_refused'][e].mean()) for k in NEW}
            d['refused_max'] = {k: float(r['extra'][k + '_refused'][e].max()) for k in NEW}
            d['ridge_on'] = {k: float(r['extra'][k + '_ridge_on'][e].mean()) for k in NEW}
            d['psiO'] = {k: float(r['extra'][k + '_psiO'][e].mean()) for k in NEW}
            row['by_eps'][str(eps)] = d
        summ.append(row)
    verdicts, gaps = {}, {}
    for basis in ('paired_abs', 'paired_ratio_own'):
        verdicts[basis] = {}
        for x, y in PAIRS:
            key = f'{x}_vs_{y}'
            verdicts[basis][key] = {}
            for eps in EPS:
                v = [s['by_eps'][str(eps)][basis][key]['verdict'] for s in summ]
                verdicts[basis][key][str(eps)] = dict(better=v.count('better'), worse=v.count('worse'), none=v.count('none'))
    for eps in EPS:
        g = {}
        for x, y in PAIRS:
            v = np.array([s['by_eps'][str(eps)][x]['abs'] / s['by_eps'][str(eps)][y]['abs'] - 1 for s in summ])
            g[f'{x}/{y}-1'] = dict(median=float(np.median(v)), min=float(v.min()), max=float(v.max()),
                                   argmin=summ[int(np.argmin(v))]['name'], argmax=summ[int(np.argmax(v))]['name'])
        refs = {k: dict(mean=float(np.mean([s['by_eps'][str(eps)]['refused'][k] for s in summ])),
                        max=float(np.max([s['by_eps'][str(eps)]['refused'][k] for s in summ])),
                        argmax=summ[int(np.argmax([s['by_eps'][str(eps)]['refused'][k] for s in summ]))]['name'])
                for k in NEW}
        gaps[str(eps)] = dict(abs_gap=g, refused=refs)
    out = dict(settings=dict(delta=DELTA, eps=EPS, R=a.R, folds=K, frac_ref=rdm.FRAC_REF, I=sts['I'], P=sts['P'],
                             arms={k: LABEL[k] for k in LABEL},
                             reference='v3.1 I, (a), (b): reports/v31_settings/adassp_benchmark.json; v3.1 P: '
                                       'reports/v31_P2/adassp_benchmark_P2.json (recomputed here as a check)'),
               reproduction_check=dict(max_rel_diff_v31_recomputed_vs_published=check,
                                       max_overall=max(max(v.values()) for v in check.values())),
               summary=summ, verdicts=verdicts, gaps=gaps,
               raw=[dict(name=r['name'], fold={k: r['fold'][k] for k in NEW}, extra=r['extra']) for r in results])
    tag = '' if a.datasets is None else '_' + '_'.join(a.datasets)
    path = os.path.join(bc.REPORT, f'fullrecord_benchmark{tag}.json')
    ev.save_json(out, path)
    print('v3.1 recomputed vs published, max rel diff:', out['reproduction_check']['max_overall'])
    print('wrote', path)


def _sens_dataset(name):
    """Exact full-record sensitivities (sens_full.delta_full) against the earlier grid values (delta_full_grid) for
    every fold and both preprocessings of one dataset."""
    D = rdm.Data(name)
    rows = []
    for f in range(K):
        te, ref, m = D.split(f)
        _, _, _, b, _ = ev.per_party_preprocess(D.F, D.y, ref, D.dR, 'blocknorm')
        for pp, bnd, ic in (('blocknorm', (b['B_R'], b['B_O'], b['B_y']), True), ('joint', (1.0, 1.0, 1.0), False)):
            if pp == 'joint' and f > 0:
                continue                                  # the same bounds in every fold
            info = bc.sens_full.delta_full_info(*map(float, bnd), ic, D.dR, D.dO)
            new = (info['Delta'], info['Delta_noyy'], info['Delta_G'])
            old = bc.sens_full.delta_full_grid(*map(float, bnd), ic, D.dR, D.dO)
            rows.append(dict(fold=f, preprocessing=pp, bounds=list(bnd), method=info['method'], exact=list(new),
                             grid=list(old), rel=[a / b_ - 1 for a, b_ in zip(new, old)]))
    return dict(name=name, d=D.d, d_R=D.dR, d_O=D.dO, rows=rows)


def cmd_sens(a):
    names = a.datasets or list(WD.PUBLISHED_ORDER)
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        res = list(ex.map(_sens_dataset, names))
    rel = [x for r in res for row in r['rows'] for x in row['rel']]
    dim1 = sorted({r['name'] for r in res for row in r['rows'] if row['method'] == 'dim1'})
    out = dict(note='exact (Theorem S10c / Lemma S10d) vs the earlier grid + local search, both rounded up by 1e-12',
               datasets_dim1=dim1, max_abs_rel=max(abs(x) for x in rel), min_rel=min(rel), max_rel=max(rel),
               max_abs_rel_dim1=max(abs(x) for r in res for row in r['rows'] if row['method'] == 'dim1'
                                    for x in row['rel']),
               datasets=res)
    path = os.path.join(bc.ROOT, 'reports', 'baselines_v2', 'sens_full_datasets.json')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ev.save_json(out, path)
    print({k: v for k, v in out.items() if k != 'datasets'})
    print('wrote', path)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('radii')
    r.add_argument('--datasets', nargs='+')
    r.add_argument('--workers', type=int, default=4)
    b = sub.add_parser('benchmark')
    b.add_argument('--datasets', nargs='+')
    b.add_argument('--R', type=int, default=50)
    b.add_argument('--workers', type=int, default=4)
    s = sub.add_parser('sens')
    s.add_argument('--datasets', nargs='+')
    s.add_argument('--workers', type=int, default=4)
    a = ap.parse_args()
    {'radii': cmd_radii, 'benchmark': cmd_benchmark, 'sens': cmd_sens}[a.cmd](a)


if __name__ == '__main__':
    main()
