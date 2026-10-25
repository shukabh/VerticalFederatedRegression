"""Figures of the P2 evaluation, from the JSON files in reports/v31_P2/ (and the reference runs in reports/v31_settings).

    python experiments/v31_P2/figures_P2.py

Writes paper/figures/{sim_pred_vs_n,adassp_mse_vs_eps,sweep_n}_v31p2.png.  Style as experiments/v31_settings/
figures_settings.py: the first three categorical slots of the validated palette (I blue, P2 orange, AdaSSP aqua),
each with its own marker and line style; P0 (the first choice of P, replaced) is a neutral grey reference line; P2 takes P's slot
and is labelled 'setting P', the name it has in the paper.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p2common as pc  # noqa: E402
from p2common import ev  # noqa: E402

import figures as F  # noqa: E402  (experiments/v31_eval/figures.py: style, helpers)
from figures import plt, Line2D, FuncFormatter  # noqa: E402

C_GREY = '#7a7975'
SET = {'I': dict(color=F.C_BLUE, ls='-', marker='o', label='setting I (inference)'),
       'P2': dict(color=F.C_ORANGE, ls='-', marker='D', label='setting P (prediction; P2 in reports/v31_P2)'),
       'P_old': dict(color=C_GREY, ls=(0, (4, 2)), marker=None, label='P₀, first choice of P (replaced)'),
       'A': dict(color=F.C_AQUA, ls=(0, (3, 2)), marker='^', label='AdaSSP, published'),
       'B': dict(color=F.C_AQUA, ls=(0, (1.2, 1.6)), marker='v', label='AdaSSP, matched')}


def save(fig, name):
    path = os.path.join(pc.FIGDIR, name)
    fig.savefig(path, dpi=220, bbox_inches='tight')
    plt.close(fig)
    print('wrote', path)


def fig_pred(S):
    """Test MSE / non-private least squares against n, rows = population, columns = eps; I, P2 and the old P."""
    R = S['by_r2']['0.5']
    fig, axes = plt.subplots(2, 5, figsize=(10.6, 5.4), sharey='row')
    for i, k in enumerate(pc.POPS):
        for j, e in enumerate(pc.EPS):
            ax = axes[i, j]
            for key in ('P_old', 'I', 'P2'):
                v = R[key][k][str(e)]
                x, y = np.array(pc.NS), np.array(v['m'])
                st = SET[key]
                ax.plot(x, y, color=st['color'], ls=st['ls'], lw=1.3 if key != 'P_old' else 1.1, zorder=4)
                if st['marker']:
                    act = np.array(v['ridge_on']) >= 0.5
                    ax.plot(x[~act], y[~act], ls='none', marker=st['marker'], ms=3.4, mfc=st['color'], mec='white', mew=0.6, zorder=5)
                    ax.plot(x[act], y[act], ls='none', marker=st['marker'], ms=3.4, mfc='white', mec=st['color'], mew=0.9, zorder=5)
            ax.axhline(1.0, color=F.FLOOR, lw=0.9)
            ax.axhline(1.1, color=F.MUTED, lw=0.8, ls=(0, (1, 1.5)))
            F.style(ax, logy=True, xt=(100, 1000, 10000, 100000))
            top = 2.2 if k == 'std' else 2.6
            F.log_ticks(ax, 0.97, top, mults=(1, 1.1, 1.25, 1.5, 2))
            ax.set_xlim(80, 1.25e5)
            ax.set_title(f"{'standardized' if k == 'std' else 'block-normalized'}, ε = {e:g}", fontsize=8.0, loc='left')
            if i == 1:
                ax.set_xlabel('n (log)')
            ax.tick_params(axis='x', labelsize=7)
        axes[i, 0].set_ylabel('test MSE / non-private')
    h = [Line2D([], [], color=SET[k]['color'], ls=SET[k]['ls'], lw=1.3, marker=SET[k]['marker'], ms=3.6, mec='white',
                label=SET[k]['label']) for k in ('I', 'P2', 'P_old')]
    h += [Line2D([], [], ls='none', marker='o', ms=3.6, mfc='white', mec=F.INK2, mew=0.9, label='hollow: ridge on in most releases'),
          Line2D([], [], color=F.MUTED, lw=0.8, ls=(0, (1, 1.5)), label='within 10%')]
    fig.legend(handles=h, loc='lower center', ncol=5, bbox_to_anchor=(0.5, -0.04), fontsize=7.6)
    fig.subplots_adjust(wspace=0.1, hspace=0.32, bottom=0.13)
    save(fig, 'sim_pred_vs_n_v31p2.png')


def fig_adassp(A, pubA):
    summ = {r['name']: r for r in A['summary']}
    raw = {r['name']: r for r in A['raw']}
    praw = {r['name']: r for r in pubA['raw']}
    names = [n for n in ['housing', 'wine', 'airfoil', 'concrete', 'bike', 'elevators', 'pol', 'kin40k', 'protein'] if n in summ]
    fig, axes = plt.subplots(3, 3, figsize=(8.6, 7.6))
    K = A['settings']['folds']
    ARM = {'v31_blocknorm_P': 'P_old', 'adassp_matched': 'B', 'adassp_pub': 'A', 'v31_blocknorm_I': 'I', 'v31_blocknorm_P2': 'P2'}
    own = {'adassp_pub': 'np_wang', 'adassp_matched': 'np_wang', 'v31_blocknorm_I': 'np_v31_blocknorm',
           'v31_blocknorm_P': 'np_v31_blocknorm', 'v31_blocknorm_P2': 'np_v31_blocknorm'}
    for ax, nm in zip(axes.flat, names):
        r = summ[nm]
        Fd = {k: np.asarray(v, float) for k, v in praw[nm]['fold'].items()}
        Fd['v31_blocknorm_P2'] = np.asarray(raw[nm]['fold']['v31_blocknorm_P2'], float)
        tops = []
        for mth in ['v31_blocknorm_P', 'adassp_matched', 'adassp_pub', 'v31_blocknorm_I', 'v31_blocknorm_P2']:
            st = SET[ARM[mth]]
            v = Fd[mth] / Fd[own[mth]][None, :]
            m = v.mean(1)
            half = ev.t95(K) * v.std(1, ddof=1) / math.sqrt(K)
            if mth == 'v31_blocknorm_P':
                ax.plot(pc.EPS, m, color=st['color'], ls=st['ls'], lw=1.1, zorder=3)
            else:
                tops.append(float(np.max(m + half)))
                ax.errorbar(pc.EPS, m, yerr=half, color=st['color'], ls=st['ls'], lw=1.3, marker=st['marker'], ms=3.4,
                            mec='white', mew=0.6, capsize=1.8, elinewidth=0.8, zorder=4)
        ax.axhline(1.0, color=F.FLOOR, lw=0.9)
        F.style(ax, logy=True, xt=pc.EPS)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
        hi = max(1.3, min(1.1 * max(tops), 12))
        F.log_ticks(ax, 0.97, hi, mults=(1, 1.5, 2, 3, 5))
        ax.set_ylim(top=hi)
        ax.set_title(f"{nm}  (n = {r['n']:,}, d = {r['d']})", fontsize=8.3, loc='left')
    for ax in axes[-1]:
        ax.set_xlabel('ε (log)')
    for ax in axes[:, 0]:
        ax.set_ylabel('test MSE / own non-private')
    lab = {'v31_blocknorm_I': 'v3.1 block norm, setting I', 'v31_blocknorm_P2': 'v3.1 block norm, setting P',
           'adassp_pub': 'AdaSSP, published', 'adassp_matched': 'AdaSSP, matched',
           'v31_blocknorm_P': 'v3.1 block norm, P₀ (clipped at the panel top)'}
    h = [Line2D([], [], color=SET[ARM[m]]['color'], ls=SET[ARM[m]]['ls'], lw=1.3, marker=SET[ARM[m]]['marker'], ms=3.6,
                mec='white', label=lab[m]) for m in ['v31_blocknorm_I', 'v31_blocknorm_P2', 'adassp_pub', 'adassp_matched',
                                                      'v31_blocknorm_P']]
    fig.legend(handles=h, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.05), fontsize=7.6)
    fig.subplots_adjust(hspace=0.42, wspace=0.28, bottom=0.1)
    save(fig, 'adassp_mse_vs_eps_v31p2.png')


def fig_sweep(W, eps_rows=(1.0, 4.0)):
    summ = W['summary']
    names = [n for n in ['bike', 'elevators', 'pol', 'protein'] if n in summ]
    fig, axes = plt.subplots(len(eps_rows), 4, figsize=(10.4, 5.4))
    for i, e in enumerate(eps_rows):
        for j, nm in enumerate(names):
            ax = axes[i, j]
            S = summ[nm]
            grid = S['grid']
            E = S['eps'][str(e)]
            ref = E['reference']
            ax.plot(grid, ref['P_old']['ratio_own'], color=C_GREY, ls=SET['P_old']['ls'], lw=1.1, zorder=3)
            ax.plot(grid, ref['adassp_pub']['ratio_own'], color=F.C_AQUA, ls=SET['A']['ls'], lw=1.2, marker='^', ms=2.8,
                    mec='white', mew=0.4, zorder=4)
            ax.plot(grid, ref['I']['ratio_own'], color=F.C_BLUE, lw=1.3, marker='o', ms=2.8, mec='white', mew=0.4, zorder=5)
            Ek = E['v31_blocknorm_P2']
            ax.errorbar(grid, Ek['ratio_own'], yerr=Ek['ratio_own_ci'], color=F.C_ORANGE, lw=1.3, marker='D', ms=2.8,
                        mec='white', mew=0.4, capsize=1.3, elinewidth=0.6, zorder=6)
            ax.axhline(1.0, color=F.FLOOR, lw=0.9)
            ax.axhline(1.1, color=F.MUTED, lw=0.8, ls=(0, (1, 1.5)))
            F.style(ax, logy=True, xt=(100, 1000, 10000))
            curves = [max(ref['I']['ratio_own']), max(Ek['ratio_own']), max(ref['adassp_pub']['ratio_own'])]
            top = max(1.3, min(6, 1.1 * max(curves)))
            F.log_ticks(ax, 0.97, top, mults=(1, 1.1, 1.25, 1.5, 2, 3, 4, 5))
            ax.set_ylim(top=top)
            ax.set_title(f"{nm}, ε = {e:g}", fontsize=8.3, loc='left')
            if i == len(eps_rows) - 1:
                ax.set_xlabel('matched records n (log)')
        axes[i, 0].set_ylabel('test MSE / own non-private')
    h = [Line2D([], [], color=SET[k]['color'], ls=SET[k]['ls'], lw=1.3, marker=SET[k]['marker'], ms=3.4, mec='white',
                label=lab) for k, lab in (('I', 'v3.1 block norm, setting I'), ('P2', 'v3.1 block norm, setting P'),
                                          ('A', 'AdaSSP, published'), ('P_old', 'v3.1 block norm, P₀ (clipped)'))]
    h += [Line2D([], [], color=F.MUTED, lw=0.8, ls=(0, (1, 1.5)), label='within 10%')]
    fig.legend(handles=h, loc='lower center', ncol=5, bbox_to_anchor=(0.5, -0.06), fontsize=7.6)
    fig.subplots_adjust(hspace=0.38, wspace=0.25, bottom=0.14)
    save(fig, 'sweep_n_v31p2.png')


def main():
    p = lambda f: os.path.join(pc.REPORT, f)
    if os.path.exists(p('sim_P2.json')):
        fig_pred(ev.load_json(p('sim_P2.json')))
    if os.path.exists(p('adassp_benchmark_P2.json')):
        fig_adassp(ev.load_json(p('adassp_benchmark_P2.json')), ev.load_json(os.path.join(pc.REPORT_SET, 'adassp_benchmark.json')))
    if os.path.exists(p('sweep_n_P2.json')):
        fig_sweep(ev.load_json(p('sweep_n_P2.json')))


if __name__ == '__main__':
    main()
