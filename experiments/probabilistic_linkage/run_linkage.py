"""Probabilistic linkage through exact-match OPRF-PSI on derived keys: linkage quality, leakage, and the effect on the
v3.1 regression.  Plaintext simulation (see linkage.py for why a dictionary lookup on the keys is exactly what the
labeled PSI outputs).

    python run_linkage.py synthetic [--seed 1] [--mnar 0] [--n-scope 1000000]   # main experiment
    python run_linkage.py febrl                                               # FEBRL4 check (linkage only)
    python run_linkage.py scale                                               # quality vs register size

Writes reports/probabilistic_linkage/{synthetic_s<seed>[_mnar].json, febrl.json, scale.json}.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, HERE)
import generator as gen  # noqa: E402
import keys as KY  # noqa: E402
import linkage as LK  # noqa: E402

OUT = os.path.join(ROOT, 'reports', 'probabilistic_linkage')
os.makedirs(OUT, exist_ok=True)


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    return o


def linkages(O, R, true_pos, n_overlap, fs=True, log=print, lsh_multi=True):
    """All linkage variants on one pair of files.  Returns {name: dict(acc, eval, leak, cnt, info)}."""
    res = {}
    t0 = time.time()
    kO = {s: KY.keys_for(O, s) for s in ('exact', 'multikey', 'lsh')}
    kR = {s: KY.keys_for(R, s) for s in ('exact', 'multikey', 'lsh')}
    log(f'  keys: {time.time() - t0:.0f}s')
    variants = [('exact', 'unique', 1), ('multikey', 'unique', 1), ('multikey', 'unique', 2),
                ('multikey', 'multi', 1), ('lsh', 'unique', 1), ('lsh', 'unique', 2), ('lsh', 'multi', 1),
                ('lsh', 'multi', 2)]
    if not lsh_multi:                                 # O(10^6) candidate pairs at 50k queries: run in `scale` only
        variants = [v for v in variants if v[:2] != ('lsh', 'multi')]
    caches = {s: {} for s in kO}
    for sch, pol, mk in variants:
        acc, cnt, info = LK.link_keys(kO[sch], kR[sch], pol, p_accept=0.0, min_keys=mk, cache=caches[sch])
        name = f'{sch}/{pol}/k>={mk}'
        pairs = info.get('pairs', np.zeros((0, 2), int))
        tp = (true_pos[pairs[:, 0]] == pairs[:, 1]).sum() if len(pairs) else 0
        res[name] = dict(acc=acc, eval=LK.evaluate(acc, true_pos, n_overlap), leak=LK.leakage(pairs, true_pos),
                         cnt=cnt, em=dict(m=info.get('m'), u=info.get('u'), prior=info.get('prior')),
                         candidate_recall=float(tp / n_overlap), K=KY.K_OF[sch])
        e = res[name]['eval']
        log(f'  {name:24s} prec {e["precision"]:.4f} rec {e["recall"]:.4f} (cand recall {tp / n_overlap:.4f}) '
            f'acc {e["accepted"]} cand {len(pairs)}')
        if sch == 'multikey' and pol == 'multi':
            block = pairs
    if fs:
        # FS on per-field tokens, candidates from the multikey blocking PSI (policy multi)
        fR = [KY.fields(R, i) for i in range(len(R))]
        fO = {}
        for i, t in block:
            if t not in fO:
                fO[t] = KY.fields(O, t)
        u_fields = LK.u_from_file(fR)                 # random-pair agreement within R's own file
        for pa in (0.5, 0.999):
            acc, info = LK.link_fields(fO, fR, block, p_accept=pa, u_fixed=u_fields)
            name = f'fs-fields/multi/p{pa:g}'
            res[name] = dict(acc=acc, eval=LK.evaluate(acc, true_pos, n_overlap), leak=LK.leakage(block, true_pos),
                             em=dict(m=info['m'], u=info['u'], prior=info['prior']), K=len(KY.FIELD_NAMES),
                             agreement_vectors_revealed=int(len(block)), fields=KY.FIELD_NAMES)
            e = res[name]['eval']
            log(f'  {name:24s} prec {e["precision"]:.4f} rec {e["recall"]:.4f} acc {e["accepted"]}')
    return res


def pr_curve(O, R, true_pos, n_overlap, scheme='multikey', policy='unique'):
    """Precision / recall of the 1:1 resolution as the minimum number of matched key types varies."""
    kO, kR = KY.keys_for(O, scheme), KY.keys_for(R, scheme)
    out, cache = [], {}
    for mk in range(1, KY.K_OF[scheme] + 1):
        acc, _, _ = LK.link_keys(kO, kR, policy, p_accept=0.0, min_keys=mk, cache=cache)
        e = LK.evaluate(acc, true_pos, n_overlap)
        out.append(dict(min_keys=mk, precision=e['precision'], recall=e['recall'], accepted=e['accepted']))
    return out


def cmd_synthetic(a):
    import regression as rg
    t0 = time.time()
    W = gen.make_world(n_scope=a.n_scope, n_R=a.n_R, n_overlap=a.n_overlap, seed=a.seed, mnar=a.mnar)
    print(f'world: {time.time() - t0:.0f}s', flush=True)
    res = linkages(W.O, W.R, W.true_pos, a.n_overlap, log=lambda s: print(s, flush=True), lsh_multi=False)
    true = np.flatnonzero(W.in_scope_R)
    res['perfect (shared ID)'] = dict(acc=np.c_[true, W.true_pos[true]],
                                      eval=LK.evaluate(np.c_[true, W.true_pos[true]], W.true_pos, a.n_overlap), K=1)
    rng = np.random.default_rng([a.seed, 77])
    out = dict(meta=dict(W.meta, n_R=a.n_R, reps_I=a.reps_I, reps_P=a.reps_P, eps=a.eps,
                         rates_R=gen.RATES_R, rates_O=gen.RATES_O, name_stats=gen.name_stats(gen.tables()),
                         n_off={f'{e:g}': {s: rg.n_off(e, s) for s in ('I', 'P')} for e in a.eps}),
               linkages={})
    for name, r in res.items():
        acc = r['acc']
        pidR, pidO = W.R.pid[acc[:, 0]], W.O.pid[acc[:, 1]]
        Z, yc = rg.design(W.ZR[pidR], W.ZO[pidO], W.y[pidO])
        G, c, yy = rg.stats(Z, yc)
        th = rg.ols(G, c)
        rec = {k: v for k, v in r.items() if k != 'acc'}
        rec['n_linked'] = int(len(acc))
        rec['ols'] = dict(theta=th, rmse=float(np.sqrt(np.mean((th - rg.BETA) ** 2))),
                          ratio_R=float(np.mean(th[1:5] / rg.BETA[1:5])), ratio_O=float(np.mean(th[5:] / rg.BETA[5:])),
                          formula_ratio_R=float(np.mean(rg.contamination_limit(r['eval'].get('false_rate', 0.0) or 0.0)[1:5]
                                                        / rg.BETA[1:5])))
        rec['dp'] = {}
        for eps in a.eps:
            for st, reps in (('I', a.reps_I), ('P', a.reps_P)):
                rel = rg.release(G, c, yy, len(acc), eps, st, reps, rng, Bboot=a.bboot)
                rec['dp'][f'{st}_e{eps:g}'] = rg.summarize(rel)
        out['linkages'][name] = rec
        d1 = rec['dp'].get('I_e1', {})
        print(f'{name:24s} n={len(acc)} ols rmse {rec["ols"]["rmse"]:.4f} R-ratio {rec["ols"]["ratio_R"]:.3f} | '
              f'I e1 rmse {d1.get("rmse_median", float("nan")):.4f} cov_beta {d1.get("cov_beta", float("nan")):.3f} '
              f'cov_beta_R {d1.get("cov_beta_R", float("nan")):.3f}', flush=True)
    out['pr_curve'] = dict(multikey=pr_curve(W.O, W.R, W.true_pos, a.n_overlap, 'multikey'),
                           lsh=pr_curve(W.O, W.R, W.true_pos, a.n_overlap, 'lsh'))
    out['seconds'] = time.time() - t0
    tag = f'synthetic_s{a.seed}' + (f'_mnar{a.mnar:g}' if a.mnar else '')
    with open(os.path.join(OUT, tag + '.json'), 'w') as fh:
        json.dump(jsonable(out), fh, indent=1)
    print('wrote', tag, f'{time.time() - t0:.0f}s')


def cmd_febrl(a):
    import febrl
    A, B, true_pos = febrl.load(a.febrl_dir)
    n_overlap = int((true_pos >= 0).sum())
    print('FEBRL4', len(A), len(B), 'true pairs', n_overlap)
    res = linkages(A, B, true_pos, n_overlap)
    out = {k: {kk: vv for kk, vv in v.items() if kk != 'acc'} for k, v in res.items()}
    out['pr_curve'] = dict(multikey=pr_curve(A, B, true_pos, n_overlap, 'multikey'),
                           lsh=pr_curve(A, B, true_pos, n_overlap, 'lsh'))
    with open(os.path.join(OUT, 'febrl.json'), 'w') as fh:
        json.dump(jsonable(out), fh, indent=1)


def cmd_scale(a):
    out = []
    for n_scope in a.sizes:
        t0 = time.time()
        W = gen.make_world(n_scope=n_scope, n_R=5000, n_overlap=4000, seed=a.seed)
        for sch in ('exact', 'multikey', 'lsh'):
            kO, kR = KY.keys_for(W.O, sch), KY.keys_for(W.R, sch)
            for pol in ('unique', 'multi'):
                cand, cnt = LK.psi_output(kO, kR, pol)
                pairs = np.array(list(cand.keys()), int).reshape(-1, 2)
                cache = {('psi', pol): (cand, cnt)}
                for mk in (1, 2):
                    acc, _, _ = LK.link_keys(kO, kR, pol, p_accept=0.0, min_keys=mk, cache=cache)
                    e = LK.evaluate(acc, W.true_pos, 4000)
                    lk = LK.leakage(pairs, W.true_pos)
                    out.append(dict(n_scope=n_scope, scheme=sch, policy=pol, min_keys=mk, **e,
                                    candidate_pairs=int(len(pairs)),
                                    candidate_recall=float((W.true_pos[pairs[:, 0]] == pairs[:, 1]).sum() / 4000),
                                    noncohort_positions=lk.get('positions_of_noncohort_people', 0),
                                    O_items_dropped_nonunique=cnt['O_items_dropped_nonunique'], O_items=cnt['O_items']))
                    print(n_scope, sch, pol, mk, f"prec {e['precision']:.4f} rec {e['recall']:.4f} "
                          f"cand {len(pairs)} noncohort {lk.get('positions_of_noncohort_people', 0)} "
                          f"dropped {cnt['O_items_dropped_nonunique']}", flush=True)
            del kO, kR, cand, cache
            for fn in (KY.norm, KY.canon, KY.soundex, KY.nysiis, KY.metaphone, KY._LSH.signature):
                fn.cache_clear()
            import gc
            gc.collect()
        print(f'  {n_scope}: {time.time() - t0:.0f}s', flush=True)
        with open(os.path.join(OUT, 'scale.json'), 'w') as fh:     # after every size (a 3M world is memory-heavy)
            json.dump(jsonable(out), fh, indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['synthetic', 'febrl', 'scale'])
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--mnar', type=float, default=0.0)
    ap.add_argument('--n-scope', type=int, default=1_000_000)
    ap.add_argument('--n-R', type=int, default=50_000)
    ap.add_argument('--n-overlap', type=int, default=40_000)
    ap.add_argument('--eps', type=float, nargs='+', default=[1.0, 4.0])
    ap.add_argument('--reps-I', type=int, default=200)
    ap.add_argument('--reps-P', type=int, default=400)
    ap.add_argument('--bboot', type=int, default=1000)
    ap.add_argument('--sizes', type=int, nargs='+', default=[100_000, 300_000, 1_000_000, 3_000_000])
    ap.add_argument('--febrl-dir', default=None)
    a = ap.parse_args()
    dict(synthetic=cmd_synthetic, febrl=cmd_febrl, scale=cmd_scale)[a.cmd](a)


if __name__ == '__main__':
    main()
