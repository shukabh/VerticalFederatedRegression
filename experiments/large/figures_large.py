"""Figures of the large-scale evaluation (paper/figures/*_large.png).

  cost_large.png      O's wall time per phase at 100k, 300k and 1M register rows (n_R^max = 20,000, four threads),
                      measured (bars, mean of the runs) against the cost model (markers: published inputs, and
                      re-measured inputs with the interpolation at the actual partition fill); right: traffic.
  noise_large.png     the decrypted noise of every release of every encrypted run, standardized by the calibrated
                      sigma (all slots), against N(0, 1); one histogram per stage.
  accuracy_large.png  plaintext simulation at 12k-1M matched records: population prediction error relative to
                      non-private least squares and coefficient RMSE relative to least squares, settings I and P, with
                      the encrypted runs' values at 12k (regime A) and 60k / 95k (regime B).

    python experiments/large/figures_large.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lcommon as lc  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

PHASES = [('O.oprf_self', 'OPRF of the register'), ('O.psi_preprocess', 'PSI binning + interpolation'),
          ('O.psi_eval', 'PSI evaluation'), ('O.psi_sanitize', 'SanitizeBFV'), ('O.upload_convert', 'upload conversion'),
          ('O.aggregate', 'aggregation'), ('O.other', 'keygen, OPRF answers, releases')]
COLORS = ['#4C72B0', '#55A868', '#C44E52', '#8172B2', '#CCB974', '#64B5CD', '#9a9a9a']
SIZES = [('100k', 1), ('300k', 2), ('1M', 3)]


def _stage(k):
    p = os.path.join(lc.REPORT, f'stage{k}.json')
    return lc.load_json(p) if os.path.exists(p) else None


def fig_cost():
    meas, pub, rem, labels, traffic = [], [], [], [], []
    for tag, k in SIZES:
        st = _stage(k)
        if not st:
            continue
        cells = [v for v in st['cells'].values() if v['cell']['setting'] == 'I' and v['cell']['kind'] == 'std'
                 and v['cell']['eps'] == 1.0]
        c = cells[0]
        ph = c['cost']['phases']
        row = {p: ph[p]['measured'] for p, _ in PHASES[:-1]}
        row['O.other'] = sum(ph[p]['measured'] for p in ('O.keygen', 'O.oprf_requests', 'O.release'))
        meas.append(row)
        pub.append(c['cost']['O_total']['predicted'])
        rem.append(c['cost_remeasured']['O_total']['predicted'] if c.get('cost_remeasured') else np.nan)
        labels.append(f"{tag}\n($\\bar n_O$ = {c['n_bar_O']:,})")
        traffic.append({k2: v['MB'] for k2, v in c['cost']['traffic'].items()})
    if not meas:
        return
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw=dict(width_ratios=[1.6, 1]))
    x = np.arange(len(meas))
    bottom = np.zeros(len(meas))
    for (p, lab), col in zip(PHASES, COLORS):
        v = np.array([m[p] for m in meas])
        ax.bar(x, v, 0.55, bottom=bottom, color=col, label=lab)
        bottom += v
    ax.plot(x, pub, 'kD', ms=7, label='cost model (published inputs)')
    ax.plot(x, rem, 'k^', ms=7, mfc='white', label='cost model (re-measured inputs)')
    ax.set_xticks(x, labels)
    ax.set_ylabel("O's wall time per phase (s), 4 threads")
    ax.set_title('O, setting I, $\\varepsilon = 1$, $n_R^{\\max} = 20{,}000$')
    ax.legend(fontsize=7, loc='upper left')
    keys = ['sent:upload', 'sent:psi_query', 'sent:keygen_R', 'sent:psi_keys', 'recv:psi_response']
    names = ["R's upload", 'PSI queries', 'key shares', 'BFV keys', 'PSI replies']
    bottom = np.zeros(len(traffic))
    for kk, nm, col in zip(keys, names, COLORS):
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
    plt.close(fig)


def fig_noise():
    from scipy import stats
    fig, axes = plt.subplots(1, 5, figsize=(14, 3), sharey=True)
    edges = np.linspace(-5, 5, 41)
    mid = 0.5 * (edges[1:] + edges[:-1])
    for ax, k in zip(axes, range(1, 6)):
        H = np.zeros(40)
        n = 0
        for cid in [c['id'] for c in lc.stage_cells(k)]:
            for r in lc.load_runs(cid):
                for s, v in r.get('slots', {}).items():
                    H += np.array(v['hist'])
                    n += v['n']
        if n == 0:
            ax.set_visible(False)
            continue
        dens = H / (n * (edges[1] - edges[0]))
        ax.semilogy(mid, np.maximum(dens, 1e-9), 'o', ms=3, label='decrypted noise')
        ax.semilogy(mid, stats.norm.pdf(mid), 'k-', lw=1, label='N(0, 1)')
        ax.set_ylim(1e-7, 1)
        ax.set_title(f'stage {k} ({n / 1e6:.1f}M slot values)', fontsize=9)
        ax.set_xlabel('standardized noise')
    axes[0].set_ylabel('density')
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(lc.FIGDIR, 'noise_large.png'), dpi=150)
    plt.close(fig)


def fig_accuracy():
    p = os.path.join(lc.REPORT, 'sim_large.json')
    if not os.path.exists(p):
        return
    S = lc.load_json(p)
    ns = S['ns']
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.5), sharex=True)
    arms = [('std', 'I', 'standardized, setting I'), ('std', 'P', 'standardized, setting P'),
            ('bn', 'P', 'block-normalized, setting P')]
    cmap = plt.get_cmap('viridis')
    for j, (kind, st, title) in enumerate(arms):
        for i, eps in enumerate(S['eps']):
            key = f'{st}_e{eps:g}'
            rows = [S['results'][f'{kind}_{n}']['arms'].get(key) for n in ns]
            col = cmap(i / (len(S['eps']) - 1))
            pr = [r['pred_ratio_median'] - 1 for r in rows]
            rm = [r['rmse_ratio'] for r in rows]
            on = [r['ridge_on'] > 0.5 for r in rows]
            for ax, y in ((axes[0, j], pr), (axes[1, j], rm)):
                ax.plot(ns, y, '-', color=col, label=f'$\\varepsilon = {eps:g}$')
                ax.scatter(ns, y, c=[col], s=18, facecolors=['white' if o else col for o in on], edgecolors=[col])
        axes[0, j].set_title(title, fontsize=10)
        axes[0, j].set_xscale('log'); axes[0, j].set_yscale('log')
        axes[1, j].set_xscale('log')
        axes[1, j].set_xlabel('matched records $n$')
    # encrypted spot checks (median over the runs of a cell)
    for k in (1, 2, 3, 4):
        st = _stage(k)
        if not st:
            continue
        for v in st['cells'].values():
            c = v['cell']
            if c['dataset'] != 'synthetic':
                continue
            j = {('std', 'I'): 0, ('std', 'P'): 1, ('bn', 'P'): 2}[(c['kind'], c['setting'])]
            runs = [r for r in v['stats']['accuracy']['runs'] if not r.get('refused')]
            if not runs:
                continue
            i = S['eps'].index(c['eps']) if c['eps'] in S['eps'] else 0
            col = cmap(i / (len(S['eps']) - 1))
            pr = np.median([r['pred_ratio'] for r in runs]) - 1
            rm = np.median([r['rmse'] for r in runs]) / v['stats']['accuracy']['rmse_ols']
            axes[0, j].plot(v['n'] * 1.03, max(pr, 1e-6), 'x', color=col, ms=8, mew=2)
            axes[1, j].plot(v['n'] * 1.03, rm, 'x', color=col, ms=8, mew=2)
    axes[0, 0].set_ylabel('prediction error / least squares $- 1$')
    axes[1, 0].set_ylabel('coefficient RMSE / least squares')
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(lc.FIGDIR, 'accuracy_large.png'), dpi=150)
    plt.close(fig)


def main():
    fig_cost()
    fig_noise()
    fig_accuracy()
    print('figures written to', lc.FIGDIR)


if __name__ == '__main__':
    main()
