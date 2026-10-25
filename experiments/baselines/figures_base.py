"""Figures of the three baselines, from the JSON files in reports/baselines/.

    python experiments/baselines/figures_base.py

Writes paper/figures/{fullrecord_sim,fullrecord_real,fws_coverage,tukey}_base.png.  Style and colours as the other
evaluation figures (experiments/v31_eval/figures.py): the validated categorical slots in fixed order, colour for the
setting (I blue, P orange) or the method, line style for the mechanism (solid v3.1, dashed full-record SSP).
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bcommon as bc  # noqa: E402
from bcommon import ev, pc  # noqa: E402

import figures as F  # noqa: E402  (experiments/v31_eval/figures.py)
from figures import plt, Line2D  # noqa: E402

C_YELLOW, C_MAGENTA = '#eda100', '#e87ba4'
GREY = '#7a7975'
ARM = {'v31_I': dict(color=F.C_BLUE, ls='-', marker='o', label='v3.1, setting I'),
       'full_I': dict(color=F.C_BLUE, ls=(0, (4, 2)), marker=None, label='full-record SSP, setting I'),
       'v31_P': dict(color=F.C_ORANGE, ls='-', marker='D', label='v3.1, setting P'),
       'full_P': dict(color=F.C_ORANGE, ls=(0, (4, 2)), marker=None, label='full-record SSP, setting P')}


def save(fig, name):
    path = os.path.join(bc.FIGDIR, name)
    fig.savefig(path, dpi=220, bbox_inches='tight')
    plt.close(fig)
    print('wrote', path)


def _line(ax, x, y, key, ridge=None):
    st = ARM[key]
    x, y = np.asarray(x, float), np.asarray(y, float)
    ax.plot(x, y, color=st['color'], ls=st['ls'], lw=1.4, zorder=4)
    if st['marker']:
        act = np.zeros(len(x), bool) if ridge is None else np.asarray(ridge) >= 0.5
        ax.plot(x[~act], y[~act], ls='none', marker=st['marker'], ms=3.4, mfc=st['color'], mec='white', mew=0.6, zorder=5)
        ax.plot(x[act], y[act], ls='none', marker=st['marker'], ms=3.4, mfc='white', mec=st['color'], mew=0.9, zorder=5)


def fig_full_sim():
    S = ev.load_json(os.path.join(bc.REPORT, 'fullrecord_sim.json'))
    R, ns = S['rows'], S['settings']['ns']
    eps_show = ['1.0', '4.0', '8.0']
    fig, axes = plt.subplots(3, len(eps_show), figsize=(9.2, 7.6), sharex=True)
    for j, e in enumerate(eps_show):
        ax = axes[0, j]
        for key in ('v31_I', 'full_I', 'v31_P', 'full_P'):
            v = R[key]['std'][e]
            _line(ax, ns, v['rmse_ratio'], key, v['ridge_on'])
        ax.axhline(1.0, color=F.FLOOR, lw=0.9)
        ax.axhline(2.0, color=F.MUTED, lw=0.8, ls=(0, (1, 1.5)))
        F.style(ax, logy=True)
        F.log_ticks(ax, 0.9, 40, mults=(1, 2, 5))
        ax.set_title(f'ε = {float(e):g}')
        for i, k in ((1, 'std'), (2, 'bn')):
            ax = axes[i, j]
            for key in ('v31_I', 'full_I', 'v31_P', 'full_P'):
                v = R[key][k][e]
                _line(ax, ns, v['pred'], key, v['ridge_on'])
            ax.axhline(1.0, color=F.FLOOR, lw=0.9)
            ax.axhline(1.1, color=F.MUTED, lw=0.8, ls=(0, (1, 1.5)))
            F.style(ax, logy=True)
            F.log_ticks(ax, 0.97, 2.2 if k == 'std' else 1.9, mults=(1, 1.1, 1.25, 1.5, 2))
    axes[0, 0].set_ylabel('slope RMSE / OLS floor\n(standardized)')
    axes[1, 0].set_ylabel('test MSE / non-private\n(standardized)')
    axes[2, 0].set_ylabel('test MSE / non-private\n(block-normalized)')
    for ax in axes[-1]:
        ax.set_xlabel('matched records n')
    hs = [Line2D([], [], color=ARM[k]['color'], ls=ARM[k]['ls'], marker=ARM[k]['marker'], ms=4, lw=1.4,
                 label=ARM[k]['label']) for k in ('v31_I', 'full_I', 'v31_P', 'full_P')]
    hs.append(Line2D([], [], ls='none', marker='o', ms=4, mfc='white', mec=F.INK2, mew=0.9,
                     label='hollow: ridge on in most releases (v3.1)'))
    fig.tight_layout()
    fig.legend(handles=hs, loc='lower center', ncol=3, bbox_to_anchor=(0.5, 1.0))
    save(fig, 'fullrecord_sim_base.png')


def fig_full_real():
    path = os.path.join(bc.REPORT, 'fullrecord_benchmark.json')
    if not os.path.exists(path):
        return
    S = ev.load_json(path)
    eps = S['settings']['eps']
    summ = S['summary']
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4), gridspec_kw=dict(width_ratios=[1, 1.6]))
    ax = axes[0]
    pairs = [('full_blocknorm_I', 'v31_blocknorm_I', F.C_BLUE, '-', '(d) block norm, I'),
             ('full_blocknorm_P', 'v31_blocknorm_P2', F.C_ORANGE, '-', '(d) block norm, P'),
             ('full_joint_I', 'v31_joint_I', F.C_BLUE, (0, (1.2, 1.6)), '(c) joint norm, I'),
             ('full_joint_P', 'v31_joint_P2', F.C_ORANGE, (0, (1.2, 1.6)), '(c) joint norm, P')]
    for x, y, c, ls, lab in pairs:
        med, lo, hi = [], [], []
        for e in eps:
            v = np.array([s['by_eps'][str(e)][x]['abs'] / s['by_eps'][str(e)][y]['abs'] - 1 for s in summ]) * 100
            med.append(np.median(v)); lo.append(np.percentile(v, 25)); hi.append(np.percentile(v, 75))
        ax.plot(eps, med, color=c, ls=ls, lw=1.4, marker='o', ms=3.4, label=lab, zorder=4)
    ax.axhline(0, color=F.FLOOR, lw=0.9)
    F.style(ax, logx=True, xt=eps)
    ax.set_xlabel('ε')
    ax.set_ylabel('test MSE of full-record SSP\nover v3.1, minus 1 (%)')
    ax.set_title('median over the 29 datasets')
    ax.set_ylim(-0.1, None)
    ax.legend(loc='lower right')
    ax = axes[1]
    e = '1.0'
    order = sorted(summ, key=lambda s: s['n'])
    xs = np.arange(len(order))
    for k, (x, y, c, ls, lab) in enumerate(pairs[:2]):
        v = np.array([s['by_eps'][e][x]['abs'] / s['by_eps'][e][y]['abs'] - 1 for s in order]) * 100
        sig = np.array([s['by_eps'][e]['paired_abs'][f'{x}_vs_{y}']['verdict'] != 'none' for s in order])
        off = (k - 0.5) * 0.36
        ax.bar(xs + off, v, width=0.34, color=c, zorder=3, label=lab + ', ε = 1')
        ax.plot(xs[sig] + off, v[sig] + np.where(v[sig] >= 0, 0.35, -0.35), ls='none', marker='*', ms=4,
                color=F.INK2, zorder=4)
    ax.axhline(0, color=F.FLOOR, lw=0.9)
    ax.set_xticks(xs)
    ax.set_xticklabels([s['name'] for s in order], rotation=60, fontsize=6.5, ha='right', rotation_mode='anchor')
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    ax.grid(True, axis='y', color=F.GRID, lw=0.6, zorder=0)
    ax.set_ylabel('full-record / v3.1 − 1 (%)')
    ax.set_title('per dataset, ordered by n (*: paired 95% CI excludes 0)')
    ax.legend(loc='upper left')
    fig.tight_layout()
    save(fig, 'fullrecord_real_base.png')


FWS_M = {'wald': dict(color=F.C_BLUE, ls='-', marker='o', label='v3.1 Wald, for θ_λ'),
         'bias_aware': dict(color=F.C_AQUA, ls='-', marker='s', label='v3.1 bias-aware, for β (where finite)'),
         'fws_s2': dict(color=F.C_ORANGE, ls='-', marker='D', label='FWS on v3.1\'s stage-2 release, for β'),
         'fws_1': dict(color=C_YELLOW, ls=(0, (4, 2)), marker='^', label='FWS, one full-budget release, for β')}


def fig_fws():
    path = os.path.join(bc.REPORT, 'fws_sim.json')
    if not os.path.exists(path):
        return
    S = ev.load_json(path)
    rows = S['rows']
    ns = S['settings']['ns']
    eps_show = [1.0, 8.0]
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 5.8), sharex=True)
    for j, e in enumerate(eps_show):
        rs = [r for r in rows if r['eps'] == e]
        ridge = [r['ridge_on'] for r in rs]
        ax = axes[0, j]
        for m, est in (('wald', 'theta_lam'), ('bias_aware', 'beta_clip'), ('fws_s2', 'beta_clip'), ('fws_1', 'beta_clip')):
            y = np.array([r['slopes'][m][f'cov_{est}'] for r in rs], float)
            if m == 'bias_aware':
                fin = np.array([r['slopes'][m]['finite'] for r in rs]) > 0.999
                y = np.where(fin, y, np.nan)
            st = FWS_M[m]
            ax.plot(ns, y, color=st['color'], ls=st['ls'], lw=1.4, marker=st['marker'], ms=3.4, mec='white', mew=0.5,
                    zorder=4)
        ax.axhline(0.955, color=F.FLOOR, lw=0.9)
        F.style(ax)
        ax.set_ylim(0.9, 1.005)
        ax.set_title(f'ε = {e:g}: coverage of the slopes')
        ax = axes[1, j]
        for m in ('wald', 'bias_aware', 'fws_s2', 'fws_1'):
            fin = np.array([r['slopes'][m]['finite'] for r in rs])
            y = np.array([r['slopes'][m]['half_median'] if f >= 0.5 else np.nan for r, f in zip(rs, fin)], float)
            st = FWS_M[m]
            ax.plot(ns, y, color=st['color'], ls=st['ls'], lw=1.4, zorder=4)
            part = fin < 0.999
            ax.plot(np.array(ns)[~part], y[~part], ls='none', marker=st['marker'], ms=3.4, mfc=st['color'], mec='white',
                    mew=0.5, zorder=5)
            ax.plot(np.array(ns)[part], y[part], ls='none', marker=st['marker'], ms=3.6, mfc='white', mec=st['color'],
                    mew=0.9, zorder=5)
        F.style(ax, logy=True)
        on = np.array(ridge) >= 0.5
        if on.any():
            last_on = np.array(ns)[on].max()
            for a in axes[:, j]:
                a.axvspan(ns[0] * 0.8, last_on * 1.15, color=F.GRID, alpha=0.5, lw=0, zorder=0)
        ax.set_title(f'ε = {e:g}: median half-width (hollow: finite in 50-99%)')
        ax.set_xlabel('matched records n')
    axes[0, 0].set_ylabel('coverage')
    axes[1, 0].set_ylabel('half-width (standardized scale)')
    hs = [Line2D([], [], color=FWS_M[k]['color'], ls=FWS_M[k]['ls'], marker=FWS_M[k]['marker'], ms=4, lw=1.4,
                 label=FWS_M[k]['label']) for k in FWS_M]
    hs.append(Line2D([], [], color=F.FLOOR, lw=0.9, label='target 0.955'))
    hs.append(plt.Rectangle((0, 0), 1, 1, color=F.GRID, alpha=0.5, label='shaded: v3.1 ridge on in most releases'))
    fig.legend(handles=hs, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout()
    save(fig, 'fws_coverage_base.png')


def fig_tukey():
    path = os.path.join(bc.REPORT, 'tukey_benchmark.json')
    if not os.path.exists(path):
        return
    S = ev.load_json(path)
    summ = S['summary']
    kp = f"{S['settings']['k_primary']:g}"
    x = np.array([s['n_matched'] / s['p'] for s in summ])
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.5))
    styles = (('1.0', F.C_BLUE, 'o', -0.025, 4.5), ('8.0', F.C_ORANGE, 'D', 0.025, 4.2))
    ax = axes[0]
    for e, c, mk, dy, ms in styles:
        y = np.array([s['by_eps'][e]['passed'][kp] for s in summ])
        ax.plot(x, y + dy, ls='none', marker=mk, ms=ms, mfc=c, mec='white', mew=0.6, label=f'ε = {float(e):g}', zorder=4)
    ax.set_xscale('log')
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    ax.grid(True, color=F.GRID, lw=0.6, zorder=0)
    ax.set_ylim(-0.08, 1.08)
    ax.set_xlabel('matched records per coefficient, n / p')
    ax.set_ylabel('share of runs passing the PTR test')
    ax.set_title('TukeyEM, m = ⌊n/(2p)⌋ subsets (ε = 1 drawn slightly lower)')
    ax.legend(loc='center left')
    ax = axes[1]
    for e, c, mk, dy, ms in styles:
        y = np.array([s['by_eps'][e]['tukey'][kp]['abs'] / s['by_eps'][e]['v31_blocknorm_P2']['abs'] for s in summ])
        ok = np.array([s['by_eps'][e]['passed'][kp] >= 0.5 for s in summ])
        ax.plot(x[ok], y[ok], ls='none', marker=mk, ms=ms, mfc=c, mec='white', mew=0.6, label=f'ε = {float(e):g}, PTR passes',
                zorder=4)
        ax.plot(x[~ok], y[~ok], ls='none', marker=mk, ms=ms, mfc='white', mec=c, mew=0.9,
                label=f'ε = {float(e):g}, PTR fails (outputs 0)', zorder=4)
    ax.axhline(1.0, color=F.FLOOR, lw=0.9)
    ax.set_xscale('log')
    ax.set_yscale('log')
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    ax.grid(True, color=F.GRID, lw=0.6, zorder=0)
    ax.set_xlabel('matched records per coefficient, n / p')
    ax.set_ylabel('test MSE, TukeyEM / v3.1 (d) under P')
    ax.set_title("absolute test MSE (Wang's scale)")
    ax.legend(loc='upper right', fontsize=7)
    fig.tight_layout()
    save(fig, 'tukey_base.png')


if __name__ == '__main__':
    fig_full_sim()
    fig_full_real()
    fig_fws()
    fig_tukey()
