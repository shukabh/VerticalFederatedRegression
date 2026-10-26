"""Check the refreshed cost model (implementation/bench/cost_model.py with the committed bench/results/ inputs)
against the measured large runs (reports/large/stage1-5.json, 78 runs, 4 threads per party, 4 parallel entries).

Those runs used the item-by-item Python OPRF and no release padding, so the model is evaluated for that code
(predict_run(oprf='python', pad_margin=None)), with each cell's own dimensions (3droad: d_R = 1, d_O = 2).  The runs
of this round with the parallel OPRF and release padding (reports/eng/run_large_eng.json, if present) are checked
against predict_run(oprf='native', pad_margin=2).  Writes reports/eng/model_check.json.

    python experiments/eng/model_check.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'implementation', 'bench'))
sys.path.insert(0, os.path.join(ROOT, 'implementation'))
import cost_model as CM  # noqa: E402

GATED = 2.0      # phases of at least 2 s (the large runs' cost gate)


def inputs():
    mu1, mu4 = CM.load('micro_t1.json'), CM.load('micro_t4.json')
    cm = CM.load('cost_model.json')
    return mu1, mu4, cm['fit'], cm['speedups_4cores']['aggregate']


def check_cell(c, mu1, mu4, fit, sp, oprf='python', pad=None):
    cell = c['cell']
    dR, dO = (1, 2) if cell['dataset'] == '3droad' else (4, 6)
    nb = c.get('n_bar_O', cell.get('n_bar_O'))
    pr = CM.predict_run(nb, cell['n_R_max'], mu1, mu4, fit, sp, d_R=dR, d_O=dO, oprf=oprf, pad_margin=pad)
    meas = {k: v['measured'] for k, v in c['cost_remeasured']['phases'].items()}
    ph = {k: dict(measured=meas[k], predicted=pr['phases'][k], ratio=meas[k] / pr['phases'][k]) for k in meas}
    O_meas = c['cost_remeasured']['O_total']['measured']
    return dict(id=cell['id'], stage=cell['stage'], n_bar_O=nb, n_R_max=cell['n_R_max'], phases=ph,
                O_total=dict(measured=O_meas, predicted=pr['O_total'], ratio=O_meas / pr['O_total']),
                old_remeasured_ratio=c['cost_remeasured']['O_total']['ratio'],
                old_published_ratio=c['cost']['O_total']['ratio'])


if __name__ == '__main__':
    mu1, mu4, fit, sp = inputs()
    out = dict(source='bench/results/ (refreshed)', speedup_aggregate=sp, stages={})
    for st in range(1, 6):
        d = json.load(open(os.path.join(ROOT, 'reports', 'large', f'stage{st}.json')))
        cells = [check_cell(c, mu1, mu4, fit, sp) for c in d['cells'].values()]
        phases = sorted(cells[0]['phases'])
        summ = {}
        for k in phases:
            r = [c['phases'][k]['ratio'] for c in cells]
            m = np.mean([c['phases'][k]['measured'] for c in cells])
            summ[k] = dict(ratio_min=min(r), ratio_max=max(r), ratio_mean=float(np.mean(r)), measured_mean=float(m),
                           predicted=cells[0]['phases'][k]['predicted'], gated=bool(m >= GATED))
        tot = [c['O_total']['ratio'] for c in cells]
        out['stages'][st] = dict(cells=cells, phases=summ, O_total_ratio=[min(tot), max(tot)],
                                 O_total_predicted=float(np.mean([c['O_total']['predicted'] for c in cells])),
                                 O_total_measured=float(np.mean([c['O_total']['measured'] for c in cells])),
                                 within_25pct_gated=all(0.75 <= c['phases'][k]['ratio'] <= 1.25 for c in cells
                                                        for k in phases if summ[k]['gated']),
                                 misses=sorted({(k, round(c['phases'][k]['ratio'], 2)) for c in cells for k in phases
                                                if summ[k]['gated'] and not 0.75 <= c['phases'][k]['ratio'] <= 1.25}))
        print(f"stage {st}: O total measured/predicted {min(tot):.2f}-{max(tot):.2f}; "
              + '; '.join(f"{k} {v['ratio_min']:.2f}-{v['ratio_max']:.2f}" for k, v in summ.items() if v['gated']))
    p = os.path.join(ROOT, 'reports', 'eng', 'run_large_eng.json')
    if os.path.exists(p):
        new = json.load(open(p))
        out['new_runs'] = []
        for r in new['runs']:
            pr = CM.predict_run(r['n_bar_O'], r['n_R_max'], mu1, mu4, fit, sp, oprf='native', pad_margin=2.0)
            ph = {k: dict(measured=v, predicted=pr['phases'][k], ratio=v / pr['phases'][k]) for k, v in r['phases'].items()}
            out['new_runs'].append(dict(id=r['id'], n_bar_O=r['n_bar_O'], phases=ph,
                                        O_total=dict(measured=r['O_total'], predicted=pr['O_total'],
                                                     ratio=r['O_total'] / pr['O_total'])))
            print(r['id'], f"O total {r['O_total']:.1f} / {pr['O_total']:.1f}",
                  '; '.join(f"{k} {v['ratio']:.2f}" for k, v in ph.items() if v['measured'] >= GATED))
    with open(os.path.join(ROOT, 'reports', 'eng', 'model_check.json'), 'w') as fh:
        json.dump(out, fh, indent=1)
