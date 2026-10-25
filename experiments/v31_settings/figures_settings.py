"""Figures of the evaluation of settings I and P, from the JSON summaries in reports/v31_settings/.

    python experiments/v31_settings/figures_settings.py

Writes paper/figures/{sim_rmse_vs_n,sim_pred_vs_n,sim_efficiency_ratio,adassp_mse_vs_eps,sweep_n}_v31s.png (the
_v31.png files of the previous setting are kept).  Style and colour as experiments/v31_eval/figures.py: eps is
ordinal (one blue ramp, validated --ordinal); the settings use the first three categorical slots (validated
all-pairs): I blue, P orange, AdaSSP aqua, each with its own marker and line style; the previous setting is a
neutral grey reference line.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scommon as cs  # noqa: E402
from scommon import ev  # noqa: E402

import figures as F  # noqa: E402  (experiments/v31_eval/figures.py: style, helpers, eps ramp)
from figures import plt, Line2D, Patch, FuncFormatter  # noqa: E402

C_I, C_P, C_A = F.C_BLUE, F.C_ORANGE, F.C_AQUA
C_PREV = '#7a7975'
SET = {'I': dict(color=C_I, ls='-', marker='o', label='setting I (inference)'),
       'P': dict(color=C_P, ls='-', marker='D', label='setting P (prediction)'),
       'prev': dict(color=C_PREV, ls=(0, (1.2, 1.6)), marker=None, label='previous setting (reference)'),
       'A': dict(color=C_A, ls=(0, (3, 2)), marker='^', label='AdaSSP, published'),
       'B': dict(color=C_A, ls=(0, (1.2, 1.6)), marker='v', label='AdaSSP, matched')}


def save(fig, name):
    path = os.path.join(cs.FIGDIR, name)
    fig.savefig(path, dpi=220, bbox_inches='tight')
    plt.close(fig)
    print('wrote', path)


# ------------------------------------------------------------------------------------------- 13.5
def fig_rmse(S, name='sim_rmse_vs_n_v31s.png', tag='I'):
    R = S['by_r2']['0.5']
    rows = R['rows']
    ns = sorted({r['n'] for r in rows})
    fig = plt.figure(figsize=(7.2, 4.9))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 3.0], hspace=0.08)
    a0 = fig.add_subplot(gs[0]); ax = fig.add_subplot(gs[1], sharex=a0)
    ols = [next(r for r in rows if r['n'] == n) for n in ns]
    ax.fill_between(ns, [r['ols_rmse_p025'] for r in ols], [r['ols_rmse_p975'] for r in ols], color=F.FLOOR,
                    alpha=0.12, lw=0, zorder=2)
    ax.plot(ns, [r['ols_rmse_median'] for r in ols], color=F.FLOOR, lw=F.LW, marker='s', ms=F.MS - 0.5, mec='white',
            mew=0.7, zorder=6)
    for k, e in enumerate(cs.EPS):
        rs = [next(r for r in rows if r['n'] == n and r['eps'] == e) for n in ns]
        act = [r['frac_ridge_on'] >= 0.5 for r in rs]
        ax.fill_between(ns, [r['rmse_p025'] for r in rs], [r['rmse_p975'] for r in rs], color=F.EPS_COL[k],
                        alpha=0.10, lw=0, zorder=2)
        F.eps_line(ax, ns, [r['rmse_median'] for r in rs], k, act)
        a0.plot(ns, [r['frac_ridge_on'] for r in rs], color=F.EPS_COL[k], lw=F.LW, zorder=5)
        a0.axvline(R['meta']['n_off'][str(e)][0], color=F.EPS_COL[k], lw=0.9, ls=(0, (2, 2)), zorder=3)
    trivial = float(np.sqrt(np.mean(np.asarray(R['meta']['beta_true'][1:]) ** 2)))
    ax.axhline(trivial, color=F.REF, lw=0.9, ls=(0, (4, 3)), zorder=1)
    ax.text(1.2e5, trivial * 1.06, 'all-zero estimate', ha='right', va='bottom', fontsize=7.5, color=F.INK2)
    F.style(ax, logy=True); F.style(a0)
    F.log_ticks(ax, 1e-3, 1.0)
    ax.set_xlim(80, 1.25e5)
    a0.set_ylim(-0.05, 1.05)
    a0.set_yticks([0, 0.5, 1]); a0.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
    a0.tick_params(labelbottom=False)
    a0.set_ylabel('share with\nλ > 0', fontsize=8)
    a0.text(0.0, 1.08, f'Setting {tag}. Dashed verticals: n_off = (g* + r₁)/ℓ of Section 9.5', transform=a0.transAxes,
            fontsize=7.5, color=F.INK2)
    ax.set_xlabel('matched records n (log scale)')
    ax.set_ylabel('RMSE of slopes vs true β (standardized, log)')
    h = [Line2D([], [], color=F.EPS_COL[k], lw=F.LW, marker='o', ms=F.MS, mec='white', mew=0.7, label=f'{tag}, ε = {e:g}')
         for k, e in enumerate(cs.EPS)]
    h += [Line2D([], [], color=F.FLOOR, lw=F.LW, marker='s', ms=F.MS - 0.5, mec='white', label='non-private least squares'),
          F.HOLLOW, Patch(facecolor=F.EPS_COL[2], alpha=0.25, lw=0, label='2.5–97.5% of runs')]
    ax.legend(handles=h, loc='lower left', fontsize=7.4, ncol=2, handlelength=1.8, columnspacing=1.0)
    save(fig, name)


def fig_pred(SI, SP, SO):
    """Test MSE / non-private least squares against n, one panel per eps: I, P and the previous setting."""
    fig, axes = plt.subplots(1, 5, figsize=(10.6, 2.9), sharey=True)
    for ax, e in zip(axes, cs.EPS):
        for key, S in (('prev', SO), ('I', SI), ('P', SP)):
            rows = S['by_r2']['0.5']['rows']
            rs = sorted([r for r in rows if r['eps'] == e], key=lambda r: r['n'])
            x = np.array([r['n'] for r in rs]); y = np.array([r['pred_ratio'] for r in rs])
            st = SET[key]
            ax.plot(x, y, color=st['color'], ls=st['ls'], lw=1.3 if key != 'prev' else 1.1, zorder=4)
            if st['marker']:
                act = np.array([r['frac_ridge_on'] >= 0.5 for r in rs])
                ax.plot(x[~act], y[~act], ls='none', marker=st['marker'], ms=3.4, mfc=st['color'], mec='white', mew=0.6, zorder=5)
                ax.plot(x[act], y[act], ls='none', marker=st['marker'], ms=3.4, mfc='white', mec=st['color'], mew=0.9, zorder=5)
        ax.axhline(1.0, color=F.FLOOR, lw=0.9)
        ax.axhline(1.1, color=F.MUTED, lw=0.8, ls=(0, (1, 1.5)))
        F.style(ax, logy=True, xt=(100, 1000, 10000, 100000))
        F.log_ticks(ax, 0.97, 2.2, mults=(1, 1.1, 1.25, 1.5, 2))
        ax.set_xlim(80, 1.25e5)
        ax.set_title(f'ε = {e:g}', fontsize=8.5, loc='left')
        ax.set_xlabel('n (log)')
        ax.tick_params(axis='x', labelsize=7)
    axes[0].set_ylabel('test MSE / non-private')
    h = [Line2D([], [], color=SET[k]['color'], ls=SET[k]['ls'], lw=1.3, marker=SET[k]['marker'], ms=3.6, mec='white',
                label=SET[k]['label']) for k in ('I', 'P', 'prev')]
    h += [Line2D([], [], ls='none', marker='o', ms=3.6, mfc='white', mec=F.INK2, mew=0.9, label='hollow: ridge on in most releases'),
          Line2D([], [], color=F.MUTED, lw=0.8, ls=(0, (1, 1.5)), label='within 10%')]
    fig.legend(handles=h, loc='lower center', ncol=5, bbox_to_anchor=(0.5, -0.1), fontsize=7.6)
    fig.subplots_adjust(wspace=0.1, bottom=0.25)
    save(fig, 'sim_pred_vs_n_v31s.png')


def fig_eff(S):
    orig = F.save
    F.save = lambda fig, name: save(fig, name.replace('_v31.png', '_v31s.png'))
    try:
        F.fig_eff(S)
    finally:
        F.save = orig


# ------------------------------------------------------------------------------------------- 13.7
ARM = {'adassp_pub': 'A', 'adassp_matched': 'B', 'v31_blocknorm_I': 'I', 'v31_blocknorm_P': 'P', 'v31_blocknorm_prev': 'prev'}
LAB = {'adassp_pub': '(a) AdaSSP, published', 'adassp_matched': '(b) AdaSSP, matched',
       'v31_blocknorm_I': '(d) v3.1 block norm, setting I', 'v31_blocknorm_P': '(d) v3.1 block norm, setting P',
       'v31_blocknorm_prev': '(d) v3.1 block norm, previous setting'}


def fig_adassp(A, prevA):
    summ = {r['name']: r for r in A['summary']}
    raw = {r['name']: r for r in A['raw']}
    praw = {r['name']: r for r in prevA['raw']}
    names = [n for n in ['housing', 'wine', 'airfoil', 'concrete', 'bike', 'elevators', 'pol', 'kin40k', 'protein'] if n in summ]
    fig, axes = plt.subplots(3, 3, figsize=(8.6, 7.6))
    K = A['settings']['folds']
    for ax, nm in zip(axes.flat, names):
        r = summ[nm]
        Fd = dict(raw[nm]['fold'])
        Fd['v31_blocknorm_prev'] = praw[nm]['fold']['v31_blocknorm']
        own = {'adassp_pub': 'np_wang', 'adassp_matched': 'np_wang', 'v31_blocknorm_I': 'np_v31_blocknorm',
               'v31_blocknorm_P': 'np_v31_blocknorm', 'v31_blocknorm_prev': 'np_v31_blocknorm'}
        tops = []
        for mth in ['v31_blocknorm_prev', 'adassp_matched', 'adassp_pub', 'v31_blocknorm_I', 'v31_blocknorm_P']:
            st = SET[ARM[mth]]
            v = np.asarray(Fd[mth]) / np.asarray(Fd[own[mth]])[None, :]
            m = v.mean(1)
            half = ev.t95(K) * v.std(1, ddof=1) / math.sqrt(K)
            tops.append(float(np.max(m + half)))
            if mth == 'v31_blocknorm_prev':
                ax.plot(cs.EPS, m, color=st['color'], ls=st['ls'], lw=1.1, zorder=3)
            else:
                ax.errorbar(cs.EPS, m, yerr=half, color=st['color'], ls=st['ls'], lw=1.3, marker=st['marker'], ms=3.4,
                            mec='white', mew=0.6, capsize=1.8, elinewidth=0.8, zorder=4)
        ax.axhline(1.0, color=F.FLOOR, lw=0.9)
        F.style(ax, logy=True, xt=cs.EPS)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
        hi = max(1.3, min(1.1 * max(tops), 12))
        F.log_ticks(ax, 0.97, hi, mults=(1, 1.5, 2, 3, 5))
        ax.set_title(f"{nm}  (n = {r['n']:,}, d = {r['d']})", fontsize=8.3, loc='left')
    for ax in axes[-1]:
        ax.set_xlabel('ε (log)')
    for ax in axes[:, 0]:
        ax.set_ylabel('test MSE / own non-private')
    h = [Line2D([], [], color=SET[ARM[m]]['color'], ls=SET[ARM[m]]['ls'], lw=1.3, marker=SET[ARM[m]]['marker'], ms=3.6,
                mec='white', label=LAB[m]) for m in ['v31_blocknorm_I', 'v31_blocknorm_P', 'adassp_pub', 'adassp_matched',
                                                      'v31_blocknorm_prev']]
    fig.legend(handles=h, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.05), fontsize=7.6)
    fig.subplots_adjust(hspace=0.42, wspace=0.28, bottom=0.1)
    save(fig, 'adassp_mse_vs_eps_v31s.png')


# ------------------------------------------------------------------------------------------- 13.8
def fig_sweep(W, eps_rows=(1.0, 4.0)):
    summ = W['summary']
    names = [n for n in ['bike', 'elevators', 'pol', 'protein'] if n in summ]
    fig, axes = plt.subplots(len(eps_rows), 4, figsize=(10.4, 5.4), sharex=False)
    for i, e in enumerate(eps_rows):
        for j, nm in enumerate(names):
            ax = axes[i, j]
            S = summ[nm]
            grid = S['grid']
            E = S['eps'][str(e)]
            ax.plot(grid, E['v31_blocknorm_prev']['ratio_own'], color=C_PREV, ls=SET['prev']['ls'], lw=1.1, zorder=3)
            ax.plot(grid, E['adassp_pub']['ratio_own'], color=C_A, ls=SET['A']['ls'], lw=1.2, marker='^', ms=2.8,
                    mec='white', mew=0.4, zorder=4)
            for key in ('I', 'P'):
                st = SET[key]
                Ek = E[f'v31_blocknorm_{key}']
                ax.errorbar(grid, Ek['ratio_own'], yerr=Ek['ratio_own_ci'], color=st['color'], lw=1.3, marker=st['marker'],
                            ms=2.8, mec='white', mew=0.4, capsize=1.3, elinewidth=0.6, zorder=5)
            ax.axhline(1.0, color=F.FLOOR, lw=0.9)
            ax.axhline(1.1, color=F.MUTED, lw=0.8, ls=(0, (1, 1.5)))
            F.style(ax, logy=True, xt=(100, 1000, 10000))
            curves = [max(E[k]['ratio_own']) for k in ('v31_blocknorm_I', 'v31_blocknorm_P', 'adassp_pub', 'v31_blocknorm_prev')]
            top = max(1.3, min(6, 1.1 * max(curves)))
            F.log_ticks(ax, 0.97, top, mults=(1, 1.1, 1.25, 1.5, 2, 3, 4, 5))
            ax.set_title(f"{nm}, ε = {e:g}", fontsize=8.3, loc='left')
            if i == len(eps_rows) - 1:
                ax.set_xlabel('matched records n (log)')
        axes[i, 0].set_ylabel('test MSE / own non-private')
    h = [Line2D([], [], color=SET[k]['color'], ls=SET[k]['ls'], lw=1.3, marker=SET[k]['marker'], ms=3.4, mec='white',
                label=lab) for k, lab in (('I', 'v3.1 block norm, setting I'), ('P', 'v3.1 block norm, setting P'),
                                          ('A', 'AdaSSP, published'), ('prev', 'v3.1 block norm, previous setting'))]
    h += [Line2D([], [], color=F.MUTED, lw=0.8, ls=(0, (1, 1.5)), label='within 10%')]
    fig.legend(handles=h, loc='lower center', ncol=5, bbox_to_anchor=(0.5, -0.06), fontsize=7.6)
    fig.subplots_adjust(hspace=0.38, wspace=0.25, bottom=0.14)
    save(fig, 'sweep_n_v31s.png')


def main():
    p = lambda f: os.path.join(cs.REPORT, f)
    q = lambda f: os.path.join(cs.REPORT_EVAL, f)
    SI, SP, SO = ev.load_json(p('sim_I.json')), ev.load_json(p('sim_P.json')), ev.load_json(q('sim_accuracy.json'))
    fig_rmse(SI)
    fig_pred(SI, SP, SO)
    fig_eff(SI)
    if os.path.exists(p('adassp_benchmark.json')):
        fig_adassp(ev.load_json(p('adassp_benchmark.json')), ev.load_json(q('adassp_benchmark.json')))
    if os.path.exists(p('sweep_n.json')):
        fig_sweep(ev.load_json(p('sweep_n.json')))


if __name__ == '__main__':
    main()
