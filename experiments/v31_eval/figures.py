"""Figures of the v3.1 evaluation, from the JSON summaries in reports/v31_eval/.

    python experiments/v31_eval/figures.py

Writes paper/figures/{sim_rmse_vs_n,sim_efficiency_ratio,bias_mse_ratio,adassp_mse_vs_eps,sweep_n}_v31.png.
Colour: eps is ordinal, so it uses one blue ramp light -> dark (validated with the dataviz validator,
--ordinal); methods use the first three categorical slots (validated all-pairs) plus line style.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as cm  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter, NullLocator  # noqa: E402

EPS_COL = ['#86b6ef', '#5598e7', '#2a78d6', '#1c5cab', '#0d366b']
C_BLUE, C_ORANGE, C_AQUA = '#2a78d6', '#eb6834', '#1baf7a'
FLOOR = '#3b3b3b'
INK, INK2, MUTED = '#0b0b0b', '#52514e', '#8a8984'
GRID, SPINE, REF = '#e4e3df', '#bdbcb7', '#9a9994'
LW, MS = 1.5, 4.5

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 8.5, 'axes.edgecolor': SPINE, 'axes.linewidth': 0.7,
    'axes.labelcolor': INK, 'axes.titlecolor': INK, 'axes.titlesize': 9, 'axes.labelsize': 8.5,
    'xtick.color': INK2, 'ytick.color': INK2, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
    'xtick.minor.size': 1.2, 'ytick.minor.size': 1.2, 'xtick.major.width': 0.7, 'ytick.major.width': 0.7,
    'legend.frameon': False, 'legend.fontsize': 7.8, 'text.color': INK, 'savefig.facecolor': 'white',
    'figure.facecolor': 'white', 'lines.solid_capstyle': 'round', 'lines.solid_joinstyle': 'round',
})


def fmt_n(x, _=None):
    return f'{x / 1000:g}k' if x >= 1000 else f'{x:g}'


def style(ax, logx=True, logy=False, xt=(100, 1000, 10000, 100000)):
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    if logx:
        ax.set_xscale('log')
        ax.xaxis.set_major_locator(FixedLocator(list(xt)))
        ax.xaxis.set_major_formatter(FuncFormatter(fmt_n))
        ax.xaxis.set_minor_formatter(NullFormatter())
    if logy:
        ax.set_yscale('log')
    ax.grid(True, which='major', color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)


def log_ticks(ax, lo, hi, mults=(1, 2, 5)):
    t = [m * 10.0 ** d for d in range(int(math.floor(math.log10(lo))) - 1, int(math.ceil(math.log10(hi))) + 1)
         for m in mults if lo <= m * 10.0 ** d <= hi]
    ax.yaxis.set_major_locator(FixedLocator(t))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_ylim(lo, hi)


def eps_line(ax, x, y, k, active, ms=MS, lw=LW, z=5):
    x, y, active = np.asarray(x), np.asarray(y), np.asarray(active, bool)
    c = EPS_COL[k]
    ax.plot(x, y, color=c, lw=lw, zorder=z)
    ax.plot(x[~active], y[~active], ls='none', marker='o', ms=ms, mfc=c, mec='white', mew=0.8, zorder=z + 0.1)
    ax.plot(x[active], y[active], ls='none', marker='o', ms=ms, mfc='white', mec=c, mew=1.1, zorder=z + 0.1)


HOLLOW = Line2D([], [], ls='none', marker='o', ms=MS, mfc='white', mec=INK2, mew=1.1,
                label='hollow: ridge on in most releases')


def save(fig, name):
    path = os.path.join(cm.FIGDIR, name)
    fig.savefig(path, dpi=220, bbox_inches='tight')
    plt.close(fig)
    print('wrote', path)


# ------------------------------------------------------------------------------------------- 12.5
def fig_rmse(S):
    R = S['by_r2']['0.5']
    rows = R['rows']
    ns = sorted({r['n'] for r in rows})
    fig = plt.figure(figsize=(7.2, 4.9))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 3.0], hspace=0.08)
    a0 = fig.add_subplot(gs[0]); ax = fig.add_subplot(gs[1], sharex=a0)
    ols = [next(r for r in rows if r['n'] == n) for n in ns]
    ax.fill_between(ns, [r['ols_rmse_p025'] for r in ols], [r['ols_rmse_p975'] for r in ols], color=FLOOR,
                    alpha=0.12, lw=0, zorder=2)
    ax.plot(ns, [r['ols_rmse_median'] for r in ols], color=FLOOR, lw=LW, marker='s', ms=MS - 0.5, mec='white',
            mew=0.7, zorder=6)
    for k, e in enumerate(cm.EPS):
        rs = [next(r for r in rows if r['n'] == n and r['eps'] == e) for n in ns]
        act = [r['frac_ridge_on'] >= 0.5 for r in rs]
        ax.fill_between(ns, [r['rmse_p025'] for r in rs], [r['rmse_p975'] for r in rs], color=EPS_COL[k],
                        alpha=0.10, lw=0, zorder=2)
        eps_line(ax, ns, [r['rmse_median'] for r in rs], k, act)
        a0.plot(ns, [r['frac_ridge_on'] for r in rs], color=EPS_COL[k], lw=LW, zorder=5)
        no = R['meta']['n_off'][str(e)][0]
        a0.axvline(no, color=EPS_COL[k], lw=0.9, ls=(0, (2, 2)), zorder=3)
    trivial = float(np.sqrt(np.mean(np.asarray(R['meta']['beta_true'][1:]) ** 2)))
    ax.axhline(trivial, color=REF, lw=0.9, ls=(0, (4, 3)), zorder=1)
    ax.text(1.2e5, trivial * 1.06, 'all-zero estimate', ha='right', va='bottom', fontsize=7.5, color=INK2)
    style(ax, logy=True); style(a0)
    log_ticks(ax, 1e-3, 1.0)
    ax.set_xlim(80, 1.25e5)
    a0.set_ylim(-0.05, 1.05)
    a0.set_yticks([0, 0.5, 1]); a0.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
    a0.tick_params(labelbottom=False)
    a0.set_ylabel('share with\nλ > 0', fontsize=8)
    a0.text(0.0, 1.08, 'Dashed verticals: n_off = (g* + r₁)/ℓ of Section 9.5', transform=a0.transAxes,
            fontsize=7.5, color=INK2)
    ax.set_xlabel('matched records n (log scale)')
    ax.set_ylabel('RMSE of slopes vs true β (standardized, log)')
    h = [Line2D([], [], color=EPS_COL[k], lw=LW, marker='o', ms=MS, mec='white', mew=0.7, label=f'v3.1, ε = {e:g}')
         for k, e in enumerate(cm.EPS)]
    h += [Line2D([], [], color=FLOOR, lw=LW, marker='s', ms=MS - 0.5, mec='white', label='non-private least squares'),
          HOLLOW, Patch(facecolor=EPS_COL[2], alpha=0.25, lw=0, label='2.5–97.5% of runs')]
    ax.legend(handles=h, loc='lower left', fontsize=7.4, ncol=2, handlelength=1.8, columnspacing=1.0)
    save(fig, 'sim_rmse_vs_n_v31.png')


def fig_eff(S):
    R = S['by_r2']['0.5']
    pc = R['percoef']
    rows = R['rows']
    ns = sorted({r['n'] for r in rows})
    names = ['intercept', 'R1 score', 'R2 binary', 'R3 income', 'R4 count',
             'O1 income', 'O2 age', 'O3 binary', 'O4 count', 'O5 score', 'O6 assets']
    beta = R['meta']['beta_true']
    fig, axes = plt.subplots(2, 6, figsize=(10.5, 4.3), sharex=True, sharey=True)
    pos = {j: axes[0, j] for j in range(5)}
    pos.update({5 + j: axes[1, j] for j in range(6)})
    axes[0, 5].axis('off')
    for j, ax in pos.items():
        ax.axhline(1.0, color=REF, lw=0.9, zorder=1)
        for k, e in enumerate(cm.EPS):
            v = [next(p for p in pc if p['n'] == n and p['eps'] == e and p['j'] == j)['eff_ratio'] for n in ns]
            act = [next(r for r in rows if r['n'] == n and r['eps'] == e)['frac_ridge_on'] >= 0.5 for n in ns]
            eps_line(ax, ns, v, k, act, ms=3.2, lw=1.2)
        style(ax, logy=True, xt=(100, 1000, 10000, 100000))
        log_ticks(ax, 0.2, 2000, mults=(1, 10))
        ax.set_xlim(80, 1.25e5)
        sub = f'β = {beta[j]:+.2f}' if j else 'β = 0'
        ax.set_title(f'{names[j]}  ({sub})', fontsize=8, loc='left')
        ax.tick_params(axis='x', labelsize=7)
    for ax in axes[1]:
        ax.set_xlabel('n (log)')
    axes[0, 0].set_ylabel("R's block\nRMSE_DP,j / SE_OLS,j")
    axes[1, 0].set_ylabel("O's block\nRMSE_DP,j / SE_OLS,j")
    h = [Line2D([], [], color=EPS_COL[k], lw=1.2, marker='o', ms=3.5, mec='white', label=f'ε = {e:g}')
         for k, e in enumerate(cm.EPS)] + [HOLLOW, Line2D([], [], color=REF, lw=0.9, label='least squares (= 1)')]
    axes[0, 5].legend(handles=h, loc='center left', fontsize=7.6)
    fig.subplots_adjust(wspace=0.12, hspace=0.38)
    save(fig, 'sim_efficiency_ratio_v31.png')


# ------------------------------------------------------------------------------------------- 12.6
def fig_bias(B, S):
    cells = B['cells']
    rows = S['by_r2']['0.5']['rows']
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.2))
    for k, e in enumerate(cm.EPS):
        cs = sorted([c for c in cells if c['eps'] == e], key=lambda c: c['theta0'])
        x = [c['theta0'] for c in cs]
        act = [c['frac_ridge_on'] >= 0.5 for c in cs]
        eps_line(axes[0], x, [max(100 * c['vs_lam']['bias_share_unc'], 1e-5) for c in cs], k, act, ms=3.6, lw=1.2)
        eps_line(axes[1], x, [c['vs_lam']['mse_ratio'] for c in cs], k, act, ms=3.6, lw=1.2)
        rs = sorted([r for r in rows if r['eps'] == e], key=lambda r: r['theta0_median'])
        rs = [r for r in rs if r['theta0_median'] > 0.03]
        eps_line(axes[2], [r['theta0_median'] for r in rs], [r['bc_ratio_true'] ** 2 for r in rs], k,
                 [r['frac_ridge_on'] >= 0.5 for r in rs], ms=3.6, lw=1.2)
    for ax in axes:
        style(ax, xt=(0.1, 1, 10, 100))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
        ax.set_xlabel(r'$\vartheta_0 = \lambda_{\min}(G)\,/\,(2\sigma_2\sqrt{p})$  (log)')
    axes[0].set_yscale('log')
    log_ticks(axes[0], 1e-5, 1.0, mults=(1,))
    axes[0].set_ylabel('bias² / MSE, uncorrected (%)')
    axes[0].set_title(r'(a) Mechanism bias, share of MSE around $\theta_\lambda$', loc='left', fontsize=8.5)
    for ax, t in ((axes[1], r'(b) MSE ratio, corrected / uncorrected, vs $\theta_\lambda$'),
                  (axes[2], r'(c) MSE ratio, corrected / uncorrected, vs true $\beta$')):
        ax.axhline(1.0, color=REF, lw=0.9, zorder=1)
        ax.set_title(t, loc='left', fontsize=8.5)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:.3f}'))
    lo = min(min(c['vs_lam']['mse_ratio'] for c in cells), min(r['bc_ratio_true'] ** 2 for r in rows if r['theta0_median'] > 0.03))
    hi = max(max(c['vs_lam']['mse_ratio'] for c in cells), max(r['bc_ratio_true'] ** 2 for r in rows if r['theta0_median'] > 0.03))
    pad = 0.08 * (hi - lo)
    for ax in axes[1:]:
        ax.set_ylim(min(lo - pad, 0.995), max(hi + pad, 1.005))
    h = [Line2D([], [], color=EPS_COL[k], lw=1.2, marker='o', ms=3.5, mec='white', label=f'ε = {e:g}')
         for k, e in enumerate(cm.EPS)] + [HOLLOW]
    fig.legend(handles=h, loc='lower center', ncol=6, bbox_to_anchor=(0.5, -0.06), fontsize=7.6)
    fig.subplots_adjust(wspace=0.32, bottom=0.2)
    save(fig, 'bias_mse_ratio_v31.png')


# ------------------------------------------------------------------------------------------- 12.7
MSTYLE = {'adassp_pub': dict(color=C_ORANGE, ls='-', marker='^', label='(a) AdaSSP, published'),
          'adassp_matched': dict(color=C_ORANGE, ls=(0, (3, 2)), marker='v', label='(b) AdaSSP, matched'),
          'v31_joint': dict(color=C_BLUE, ls=(0, (3, 2)), marker='D', label='(c) v3.1, joint row norm (idealized)'),
          'v31_blocknorm': dict(color=C_BLUE, ls='-', marker='o', label='(d) v3.1, block unit norm'),
          'v31_zscore': dict(color=C_AQUA, ls='-', marker='s', label='(e) v3.1, standardize')}
OWN = {'adassp_pub': 'np_wang', 'adassp_matched': 'np_wang', 'v31_joint': 'np_wang',
       'v31_blocknorm': 'np_v31_blocknorm', 'v31_zscore': 'np_v31_zscore'}


def fig_adassp(A):
    summ = {r['name']: r for r in A['summary']}
    raw = {r['name']: r for r in A['raw']}
    early = A.get('earlier_mechanism_eps1', {})
    names = [n for n in ['housing', 'wine', 'airfoil', 'concrete', 'bike', 'elevators', 'pol', 'kin40k', 'protein'] if n in summ]
    fig, axes = plt.subplots(3, 3, figsize=(8.6, 7.6))
    K = A['settings']['folds']
    for ax, nm in zip(axes.flat, names):
        r = summ[nm]
        F = raw[nm]['fold']
        tops = []
        for mth in ['v31_zscore', 'adassp_matched', 'v31_joint', 'adassp_pub', 'v31_blocknorm']:
            st = MSTYLE[mth]
            v = np.asarray(F[mth]) / np.asarray(F[OWN[mth]])[None, :]      # (E, K) ratio to own non-private, per fold
            m = v.mean(1)
            half = cm.t95(K) * v.std(1, ddof=1) / math.sqrt(K)
            tops.append(float(np.max(m + half)))
            ax.errorbar(cm.EPS, m, yerr=half, color=st['color'], ls=st['ls'], lw=1.3, marker=st['marker'], ms=3.4,
                        mec='white', mew=0.6, capsize=1.8, elinewidth=0.8, zorder=4)
        if nm in early:
            ax.plot([1.0], [early[nm]['vfl_earlier']], ls='none', marker='*', ms=8, mfc=INK, mec='white', mew=0.6, zorder=6)
        ax.axhline(1.0, color=FLOOR, lw=0.9)
        style(ax, logy=True, xt=cm.EPS)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
        hi = max(1.3, min(1.1 * max(tops), 12))
        log_ticks(ax, 0.97, hi, mults=(1, 1.5, 2, 3, 5))
        ax.set_title(f"{nm}  (n = {r['n']:,}, d = {r['d']})", fontsize=8.3, loc='left')
    for ax in axes[-1]:
        ax.set_xlabel('ε (log)')
    for ax in axes[:, 0]:
        ax.set_ylabel('test MSE / own non-private')
    h = [Line2D([], [], color=MSTYLE[m]['color'], ls=MSTYLE[m]['ls'], lw=1.3, marker=MSTYLE[m]['marker'], ms=3.6,
                mec='white', label=MSTYLE[m]['label']) for m in ['adassp_pub', 'adassp_matched', 'v31_joint', 'v31_blocknorm', 'v31_zscore']]
    h += [Line2D([], [], ls='none', marker='*', ms=8, mfc=INK, mec='white', label='earlier mechanism, ε = 1 (current paper)')]
    fig.legend(handles=h, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.05), fontsize=7.6)
    fig.subplots_adjust(hspace=0.42, wspace=0.28, bottom=0.1)
    save(fig, 'adassp_mse_vs_eps_v31.png')


# ------------------------------------------------------------------------------------------- 12.8
def fig_sweep(W):
    summ = W['summary']
    names = [n for n in ['bike', 'elevators', 'pol', 'protein'] if n in summ]
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 6.2))
    for ax, nm in zip(axes.flat, names):
        S = summ[nm]
        grid = S['grid']
        for k, e in enumerate(cm.EPS):
            E = S['eps'][str(e)]
            ax.plot(grid, E['adassp_pub']['ratio_own'], color=EPS_COL[k], lw=1.0, ls=(0, (3, 2)), zorder=3)
            ax.errorbar(grid, E['v31_blocknorm']['ratio_own'], yerr=E['v31_blocknorm']['ratio_own_ci'],
                        color=EPS_COL[k], lw=1.4, marker='o', ms=3.0, mec='white', mew=0.5, capsize=1.5,
                        elinewidth=0.7, zorder=4)
        triv = S['trivial_ratio_own']['np_v31_blocknorm']
        ax.plot(grid, triv, color=REF, lw=0.9, ls=(0, (4, 3)))
        ax.axhline(1.0, color=FLOOR, lw=0.9)
        ax.axhline(1.1, color=MUTED, lw=0.8, ls=(0, (1, 1.5)))
        style(ax, logy=True, xt=(100, 1000, 10000))
        curves = [max(S['eps'][str(e)][m]['ratio_own']) for e in cm.EPS for m in ('adassp_pub', 'v31_blocknorm')]
        top = max(1.3, min(6, 1.1 * max(max(triv), max(curves))))
        log_ticks(ax, 0.97, top, mults=(1, 1.1, 1.25, 1.5, 2, 3, 4, 5))
        ax.set_title(f"{nm}  (d = {S['d']}, up to n = {grid[-1]:,})", fontsize=8.3, loc='left')
        ax.set_xlabel('matched records n (log)')
    for ax in axes[:, 0]:
        ax.set_ylabel('test MSE / own non-private, same n')
    h = [Line2D([], [], color=EPS_COL[k], lw=2, label=f'ε = {e:g}') for k, e in enumerate(cm.EPS)]
    h += [Line2D([], [], color=INK2, lw=1.4, marker='o', ms=3, label='v3.1, block unit norm'),
          Line2D([], [], color=INK2, lw=1.0, ls=(0, (3, 2)), label='AdaSSP, published'),
          Line2D([], [], color=REF, lw=0.9, ls=(0, (4, 3)), label='predict the training mean (block norm)'),
          Line2D([], [], color=MUTED, lw=0.8, ls=(0, (1, 1.5)), label='within 10%')]
    fig.legend(handles=h, loc='lower center', ncol=5, bbox_to_anchor=(0.5, -0.05), fontsize=7.6)
    fig.subplots_adjust(hspace=0.35, wspace=0.25, bottom=0.12)
    save(fig, 'sweep_n_v31.png')


def main():
    p = lambda f: os.path.join(cm.REPORT, f)
    S = cm.load_json(p('sim_accuracy.json')) if os.path.exists(p('sim_accuracy.json')) else None
    if S:
        fig_rmse(S)
        fig_eff(S)
    if S and os.path.exists(p('bias_study.json')):
        fig_bias(cm.load_json(p('bias_study.json')), S)
    if os.path.exists(p('adassp_benchmark.json')):
        fig_adassp(cm.load_json(p('adassp_benchmark.json')))
    if os.path.exists(p('sweep_n.json')):
        fig_sweep(cm.load_json(p('sweep_n.json')))


if __name__ == '__main__':
    main()
