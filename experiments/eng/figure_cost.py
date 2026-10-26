"""paper/figures/cost_large.png, redrawn with the refreshed cost model and the parallel OPRF.

Left: O's wall time per phase in the large runs (bars: setting I, eps = 1, mean of 3 runs at 100k, 300k and 1M rows,
n_R^max = 20,000, 4 threads; single-threaded OPRF, no release padding), against the cost model with the inputs
published before those runs (filled diamonds) and the refreshed model for the same code (open triangles,
reports/eng/model_check.json); stars: O's total in one run each with the parallel OPRF and release padding
(reports/eng/run_large_eng.json), open circles: the refreshed model's prediction for it.  Right: traffic per run.

    python experiments/eng/figure_cost.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'experiments', 'large'))
import figures_large as FL  # noqa: E402
import lcommon as lc  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

if __name__ == '__main__':
    mc = json.load(open(os.path.join(ROOT, 'reports', 'eng', 'model_check.json')))
    new = {r['id']: r for r in mc.get('new_runs', [])}
    meas, pub, ref, labels, traffic, newm, newp = [], [], [], [], [], [], []
    for tag, k in FL.SIZES:
        st = FL._stage(k)
        c = [v for v in st['cells'].values() if v['cell']['setting'] == 'I' and v['cell']['kind'] == 'std'
             and v['cell']['eps'] == 1.0][0]
        ph = c['cost']['phases']
        row = {p: ph[p]['measured'] for p, _ in FL.PHASES[:-1]}
        row['O.other'] = sum(ph[p]['measured'] for p in ('O.keygen', 'O.oprf_requests', 'O.release'))
        meas.append(row)
        pub.append(c['cost']['O_total']['predicted'])
        ref.append([x for x in mc['stages'][str(k)]['cells'] if x['id'] == c['cell']['id']][0]['O_total']['predicted'])
        labels.append(f"{tag}\n($\\bar n_O$ = {c['n_bar_O']:,})")
        traffic.append({k2: v['MB'] for k2, v in c['cost']['traffic'].items()})
        r = new.get(c['cell']['id'])
        newm.append(r['O_total']['measured'] if r else np.nan)
        newp.append(r['O_total']['predicted'] if r else np.nan)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw=dict(width_ratios=[1.6, 1]))
    x = np.arange(len(meas))
    bottom = np.zeros(len(meas))
    for (p, lab), col in zip(FL.PHASES, FL.COLORS):
        v = np.array([m[p] for m in meas])
        ax.bar(x - 0.08, v, 0.5, bottom=bottom, color=col, label=lab)
        bottom += v
    ax.plot(x - 0.08, pub, 'kD', ms=7, label='model, inputs published before the runs')
    ax.plot(x - 0.08, ref, 'k^', ms=7, mfc='white', label='refreshed model')
    ax.plot(x + 0.3, newm, 'r*', ms=12, label='measured, parallel OPRF (one run)')
    ax.plot(x + 0.3, newp, 'o', color='r', ms=7, mfc='white', label='refreshed model, parallel OPRF')
    ax.set_xticks(x, labels)
    ax.set_ylabel("O's wall time per phase (s), 4 threads")
    ax.set_title('O, setting I, $\\varepsilon = 1$, $n_R^{\\max} = 20{,}000$')
    ax.legend(fontsize=7, loc='upper left')
    keys = ['sent:upload', 'sent:psi_query', 'sent:keygen_R', 'sent:psi_keys', 'recv:psi_response']
    names = ["R's upload", 'PSI queries', 'key shares', 'BFV keys', 'PSI replies']
    bottom = np.zeros(len(traffic))
    for kk, nm, col in zip(keys, names, FL.COLORS):
        v = np.array([t.get(kk, 0.0) for t in traffic])
        ax2.bar(x, v, 0.55, bottom=bottom, color=col, label=nm)
        bottom += v
    rest = np.array([sum(t.values()) for t in traffic]) - bottom
    ax2.bar(x, rest, 0.55, bottom=bottom, color='#9a9a9a', label='other')
    ax2.set_xticks(x, [l.split('\n')[0] for l in labels])
    ax2.set_ylabel('traffic per run (MB)')
    ax2.legend(fontsize=7, loc='upper left')
    fig.tight_layout()
    fig.savefig(os.path.join(lc.FIGDIR, 'cost_large.png'), dpi=150)
    print('wrote', os.path.join(lc.FIGDIR, 'cost_large.png'))
