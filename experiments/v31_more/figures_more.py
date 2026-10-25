"""Figures of the follow-up checks, from reports/v31_more/*.json and encrypted_more_noise.npz.

    python experiments/v31_more/figures_more.py

Writes paper/figures/{coverage_real,coverage_repeat,noise}_more.png.  Style and colours as in
experiments/v31_eval/figures.py (eps: the validated blue ramp; encrypted against plaintext: categorical slots 1 and 2).
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mcommon as mc  # noqa: E402
sys.path.insert(0, os.path.join(mc.ROOT, 'experiments', 'v31_eval'))
import figures as F  # noqa: E402
from figures import plt, Line2D  # noqa: E402

EPS_C = {1.0: F.EPS_COL[1], 4.0: F.EPS_COL[3], 8.0: F.EPS_COL[4]}
DS_ORDER = ['protein', 'bike', 'elevators', 'pol', '3droad']


def load(name):
    with open(os.path.join(mc.REPORT, name)) as fh:
        return json.load(fh)


def fig_real():
    d = load('coverage_real.json')
    cells = d['cells']
    rp = os.path.join(mc.REPORT, 'coverage_real_recheck.json')
    recheck = json.load(open(rp)) if os.path.exists(rp) else []
    fig, ax = plt.subplots(figsize=(7.0, 2.7))
    F.style(ax, logx=False)
    x0, ticks, labs, seps = 0, [], [], []
    for ds in DS_ORDER:
        cc = [c for c in cells if c['dataset'] == ds]
        if not cc:
            continue
        ns = sorted(set(c['n'] for c in cc))
        for n in ns:
            for k, eps in enumerate((1.0, 4.0, 8.0)):
                c = [c for c in cc if c['n'] == n and c['eps'] == eps]
                if not c:
                    continue
                c = c[0]
                cv = c['coverage_all']
                xx = x0 + (k - 1) * 0.22
                on = c['ridge_on'] > 0.5
                ax.plot([xx, xx], cv['clustered'], color=EPS_C[eps], lw=1.1, zorder=4)
                ax.plot([xx], [cv['coverage']], ls='none', marker='o', ms=4.2, mfc='white' if on else EPS_C[eps],
                        mec=EPS_C[eps], mew=1.1, zorder=5)
                rc = [r for r in recheck if r['dataset'] == ds and r['n'] == n and r['eps'] == eps]
                if rc:
                    cv2 = rc[0]['recheck']
                    ax.plot([xx + 0.09] * 2, cv2['clustered'], color=F.MUTED, lw=1.0, zorder=4)
                    ax.plot([xx + 0.09], [cv2['coverage']], ls='none', marker='D', ms=3.6, mfc='white', mec=F.MUTED, mew=1.0, zorder=5)
            ticks.append(x0)
            labs.append(f'{n / 1000:.0f}k' if n >= 1000 else f'{n}')
            x0 += 1
        seps.append((x0 - len(ns), x0 - 1, ds))
        x0 += 0.6
    ax.axhline(mc.TARGET, color=F.FLOOR, lw=0.9, zorder=3)
    ax.set_xticks(ticks)
    ax.set_xticklabels(labs, fontsize=7.2)
    for a, b, ds in seps:
        ax.text((a + b) / 2, 0.9335, ds, ha='center', va='bottom', fontsize=8, color=F.INK)
    ax.set_ylim(0.93, 0.975)
    ax.set_xlim(-0.6, x0 - 0.4)
    ax.set_ylabel('Wald coverage of $\\theta_\\lambda$')
    ax.set_xlabel('matched records $n$ (fixed cohort)')
    hs = [Line2D([], [], color=EPS_C[e], marker='o', ms=4.2, lw=1.1, label=f'$\\varepsilon = {e:g}$') for e in (1.0, 4.0, 8.0)]
    hs += [Line2D([], [], ls='none', marker='o', ms=4.2, mfc='white', mec=F.INK2, mew=1.1, label='hollow: ridge on'),
           Line2D([], [], color=F.MUTED, marker='D', ms=3.6, mfc='white', lw=1.0, label='re-run, 5,000 releases'),
           Line2D([], [], color=F.FLOOR, lw=0.9, label='target 0.955')]
    ax.legend(handles=hs, ncol=6, loc='upper center', bbox_to_anchor=(0.5, 1.13), handlelength=1.6, columnspacing=1.2)
    fig.tight_layout()
    path = os.path.join(mc.FIGDIR, 'coverage_real_more.png')
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print('wrote', path)


def fig_repeat():
    d = load('coverage_repeat.json')
    enc = load('encrypted_more.json') if os.path.exists(os.path.join(mc.REPORT, 'encrypted_more.json')) else None
    rows = []
    lab = {'synthetic': 'synthetic', 'protein_zscore': 'protein\nstandardize', 'protein_winsor2': 'protein\nwinsorize',
           'protein_blocknorm': 'protein\nblock norm'}
    for r in d['results']:
        rows.append((f"{lab[r['tag']]}\n$\\varepsilon = {r['eps']:g}$", r['coverage']['coverage'], r['coverage']['clustered'],
                     r['encrypted']['coverage'], r['encrypted']['wilson'], f"{r['encrypted']['covered']}/{r['encrypted']['total']}"))
    fig, ax = plt.subplots(figsize=(7.0, 2.6))
    F.style(ax, logx=False)
    for i, (lb, pc, pci, ec, eci, et) in enumerate(rows):
        ax.plot([i - 0.12] * 2, pci, color=F.C_BLUE, lw=1.2, zorder=4)
        ax.plot([i - 0.12], [pc], ls='none', marker='o', ms=4.4, color=F.C_BLUE, zorder=5)
        ax.plot([i + 0.12] * 2, eci, color=F.C_ORANGE, lw=1.0, zorder=4)
        ax.plot([i + 0.12], [ec], ls='none', marker='D', ms=4.0, mfc='white', mec=F.C_ORANGE, mew=1.1, zorder=5)
        ax.text(i + 0.12, 0.792, et, ha='center', va='bottom', fontsize=6.6, color=F.INK2)
    ax.axhline(mc.TARGET, color=F.FLOOR, lw=0.9, zorder=3)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([r[0] for r in rows], fontsize=6.8)
    ax.set_ylim(0.785, 1.01)
    ax.set_ylabel('Wald coverage of $\\theta_\\lambda$')
    hs = [Line2D([], [], color=F.C_BLUE, marker='o', ms=4.4, lw=1.2, label='plaintext simulation, 10,000 releases (95% CI)'),
          Line2D([], [], color=F.C_ORANGE, marker='D', ms=4.0, mfc='white', lw=1.0, label='earlier encrypted runs (Wilson 95% CI)'),
          Line2D([], [], color=F.FLOOR, lw=0.9, label='target 0.955')]
    ax.legend(handles=hs, ncol=3, loc='upper center', bbox_to_anchor=(0.5, 1.14), handlelength=1.8)
    fig.tight_layout()
    path = os.path.join(mc.FIGDIR, 'coverage_repeat_more.png')
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print('wrote', path)


def fig_noise():
    from scipy import stats
    z = dict(np.load(os.path.join(mc.REPORT, 'encrypted_more_noise.npz')))
    enc = load('encrypted_more.json')
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.9), gridspec_kw=dict(width_ratios=[1.0, 0.9, 1.0]))
    # (a) histogram of all released entries against N(0, 1)
    ax = axs[0]
    F.style(ax, logx=False)
    allz = np.concatenate([v.ravel() for k, v in z.items() if not k.endswith('_yy')])
    bins = np.linspace(-4.5, 4.5, 37)
    ax.hist(allz, bins=bins, density=True, color=F.EPS_COL[0], edgecolor='white', lw=0.6, zorder=3)
    xs = np.linspace(-4.5, 4.5, 300)
    ax.plot(xs, stats.norm.pdf(xs), color=F.FLOOR, lw=1.1, zorder=4)
    ax.set_xticks([-4, -2, 0, 2, 4])
    ax.set_xlabel('standardized noise')
    ax.set_ylabel('density')
    ax.set_title(f'all released entries ({allz.size:,})', fontsize=8.2)
    # (b) QQ plot
    ax = axs[1]
    F.style(ax, logx=False)
    srt = np.sort(allz)
    q = stats.norm.ppf((np.arange(1, srt.size + 1) - 0.5) / srt.size)
    ax.plot([-4.2, 4.2], [-4.2, 4.2], color=F.REF, lw=0.9, zorder=3)
    ax.plot(q, srt, ls='none', marker='o', ms=1.6, color=F.C_BLUE, zorder=4)
    ax.set_xticks([-4, -2, 0, 2, 4])
    ax.set_xlabel('N(0, 1) quantile')
    ax.set_ylabel('sample quantile')
    ax.set_title('quantile plot', fontsize=8.2)
    # (c) mean per configuration and release, 95% CI
    ax = axs[2]
    F.style(ax, logx=False)
    labels, i = [], 0
    for cfg, r in enc['results'].items():
        for s, col in (('stage1', F.C_BLUE), ('stage2', F.C_ORANGE)):
            t = r['noise'][s]
            ax.plot(t['ci95'], [i, i], color=col, lw=1.2, zorder=4)
            ax.plot([t['mean']], [i], ls='none', marker='o', ms=4.2, color=col, zorder=5)
            labels.append(f"{cfg}, rel. {s[-1]}")
            i += 1
    for s, col in (('stage1', F.C_BLUE), ('stage2', F.C_ORANGE)):
        t = enc['noise_pooled'][s]
        ax.plot(t['ci95'], [i, i], color=col, lw=1.6, zorder=4)
        ax.plot([t['mean']], [i], ls='none', marker='s', ms=4.6, color=col, zorder=5)
        labels.append(f"pooled, rel. {s[-1]}")
        i += 1
    old = enc['noise_pooled'].get('earlier_spot_checks', {}).get('per_check', {})
    for key, lab_ in (('I2_old', 'earlier I2'), ('P1_old', 'earlier P1')):
        if key not in old:
            continue
        for s, col in (('stage1', F.C_BLUE), ('stage2', F.C_ORANGE)):
            t = old[key][s]
            ax.plot(t['ci95'], [i, i], color=F.MUTED, lw=1.0, zorder=4)
            ax.plot([t['mean']], [i], ls='none', marker='o', ms=3.8, mfc='white', mec=col, mew=1.1, zorder=5)
            labels.append(f"{lab_}, rel. {s[-1]}")
            i += 1
    ax.axvline(0, color=F.FLOOR, lw=0.9, zorder=3)
    ax.xaxis.set_major_formatter(F.FuncFormatter(lambda v, _: f'{v:g}'))
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=6.8)
    ax.invert_yaxis()
    ax.set_xlabel('mean (95% CI)')
    ax.set_title('mean of the noise', fontsize=8.2)
    fig.tight_layout()
    path = os.path.join(mc.FIGDIR, 'noise_more.png')
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print('wrote', path)


if __name__ == '__main__':
    fig_real()
    fig_repeat()
    if os.path.exists(os.path.join(mc.REPORT, 'encrypted_more_noise.npz')):
        fig_noise()
