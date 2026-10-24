"""Tables and key numbers of the v3.1 evaluation, generated from the JSON summaries in reports/v31_eval/.

    python experiments/v31_eval/tables.py

Writes reports/v31_eval/tables_v31.tex (LaTeX table bodies used in paper/drafts/eval_v31.tex) and
reports/v31_eval/key_numbers.md (Markdown versions plus the numbers quoted in the text).
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as cm  # noqa: E402

P = lambda f: os.path.join(cm.REPORT, f)
TEX, MD = [], []


def fmt_n(v):
    if v is None:
        return None
    return f'{v / 1000:.1f}k' if v < 10000 else f'{v / 1000:.0f}k'


def tex_n(v, nmax=None):
    if v is None:
        return f'$>${fmt_n(nmax)}' if nmax else '$>$100k'
    return fmt_n(v)


def sim_tables(S):
    R = S['by_r2']['0.5']
    rows = R['rows']
    get = lambda n, e: next(r for r in rows if r['n'] == n and r['eps'] == e)
    TEX.append('% tab:sim_rmse_v31 body: median RMSE (ratio to the non-private floor), R^2 = 0.5')
    MD.append('## 12.5 Median RMSE of the slopes vs true beta, R^2 = 0.5 (ratio to the OLS floor)\n')
    MD.append('| n | ' + ' | '.join(f'eps = {e:g}' for e in cm.EPS) + ' | OLS floor |')
    MD.append('|---|' + '---|' * (len(cm.EPS) + 1))
    for n in (1000, 10000, 100000):
        cells = []
        for e in cm.EPS:
            r = get(n, e)
            mark = '$^\\ast$' if r['frac_ridge_on'] >= 0.5 else ''
            v = r['rmse_median']
            cells.append(f"{v:.3f} ({r['ratio_median']:.1f}){mark}" if v >= 0.01 else f"{v:.4f} ({r['ratio_median']:.1f}){mark}")
        floor = get(n, 1.0)['ols_rmse_median']
        TEX.append(f"    $10^{{{int(math.log10(n))}}}$ & " + ' & '.join(cells) + f' & {floor:.4f} \\\\')
        MD.append(f'| {n:,} | ' + ' | '.join(c.replace('$^\\ast$', '*') for c in cells) + f' | {floor:.4f} |')
    MD.append('\n(* ridge on in most releases)\n')
    MD.append('### Crossings (log-linear interpolation of the median curves)\n')
    MD.append('| R^2 | eps | n_off (9.5) | coef. RMSE <= 2x floor | coef. <= 1.25x floor | prediction within 10% | within 1% |')
    MD.append('|---|---|---|---|---|---|---|')
    for r2 in ('0.3', '0.5', '0.9'):
        for e in cm.EPS:
            c = S['by_r2'][r2]['crossings'][str(e)]
            MD.append(f"| {r2} | {e:g} | {fmt_n(c['n_off'])} | {fmt_n(c['coef_2x_median']) or '>100k'} | "
                      f"{fmt_n(c['coef_1p25x_median']) or '>100k'} | {fmt_n(c['pred_1p10']) or '>100k'} | {fmt_n(c['pred_1p01']) or '>100k'} |")
    MD.append('')
    # ratio at 100k for eps where the 2x crossing is not reached
    for r2 in ('0.5', '0.9'):
        rs = S['by_r2'][r2]['rows']
        MD.append(f"R^2 = {r2}, ratio to floor at n = 100k: " +
                  ', '.join(f"eps {e:g}: {next(r for r in rs if r['n'] == 100000 and r['eps'] == e)['ratio_median']:.2f}" for e in cm.EPS))
    MD.append('')
    # ridge switch-off check
    TEX.append('% tab:noff_check_v31 body: share of releases with lambda = 0 at n = m n_off (stage-1 rule, 200 cohorts)')
    MD.append('### Stage-1 ridge: share of releases with lambda = 0 at n = m * n_off (200 cohorts each)\n')
    nc = S['noff_check']
    mults = sorted({x['mult'] for x in nc})
    MD.append('| eps | n_off | ' + ' | '.join(f'{m:g} n_off' for m in mults) + ' | median lambda at 0.8 n_off (linear prediction) |')
    MD.append('|---|---|' + '---|' * (len(mults) + 1))
    for e in cm.EPS:
        xs = sorted([x for x in nc if x['eps'] == e], key=lambda x: x['mult'])
        x8 = next(x for x in xs if abs(x['mult'] - 0.8) < 1e-9)
        TEX.append(f"    {e:g} & {xs[0]['n_off']:,.0f} & " + ' & '.join(f"{x['frac_off']:.2f}" for x in xs) +
                   f" & {x8['lam_median']:,.0f} & {x8['lam_pred']:,.0f} \\\\".replace(',', '{,}'))
        MD.append(f"| {e:g} | {xs[0]['n_off']:,.0f} | " + ' | '.join(f"{x['frac_off']:.3f}" for x in xs) +
                  f" | {x8['lam_median']:,.0f} ({x8['lam_pred']:,.0f}) |")
    MD.append('')
    # inference summary
    off = [r for r in rows if r['frac_ridge_on'] == 0]
    on = [r for r in rows if r['frac_ridge_on'] >= 0.5]
    MD.append('### Intervals, R^2 = 0.5 (per (n, eps) cell, 200 releases x 11 coefficients)\n')
    MD.append(f"- Wald interval for theta_lambda: pooled coverage {min(r['wald_cov_theta_lam'] for r in rows):.3f}-"
              f"{max(r['wald_cov_theta_lam'] for r in rows):.3f} over all {len(rows)} cells (target 0.955).")
    MD.append(f"- Ridge off in every release ({len(off)} cells): bias-aware interval finite in all, covers true beta "
              f"{min(r['ba_cov_beta_finite'] for r in off):.3f}-{max(r['ba_cov_beta_finite'] for r in off):.3f}, clipped-population beta "
              f"{min(r['ba_cov_beta_clip'] for r in off):.3f}-{max(r['ba_cov_beta_clip'] for r in off):.3f}.")
    fin_on = [r['ba_frac_finite'] for r in on]
    MD.append(f"- Ridge on in most releases ({len(on)} cells): bias-aware interval finite in {min(fin_on):.3f}-{max(fin_on):.3f} "
              f"of coefficients (no public bound on ||beta||); where finite, median half-width up to "
              f"{max((r['ba_halfwidth_median'] or 0) for r in on):.2f} vs Wald {np.median([r['wald_halfwidth_median'] for r in on]):.3f}.")
    MD.append(f"- Naive reading of the theta_lambda Wald interval as an interval for beta: coverage "
              f"{min(r['wald_naive_cov_beta'] for r in on):.3f}-{max(r['wald_naive_cov_beta'] for r in on):.3f} with the ridge on, "
              f"{min(r['wald_naive_cov_beta'] for r in off):.3f}-{max(r['wald_naive_cov_beta'] for r in off):.3f} with it off.")
    MD.append(f"- Refused releases (gate): {max(r['refused'] for r in rows):.4f} at most, over all cells and R^2 = 0.5.")
    ref_all = max(r['refused'] for r2 in S['by_r2'] for r in S['by_r2'][r2]['rows'])
    MD.append(f"- Refused releases over all R^2: {ref_all:.4f} at most.")
    # efficiency ratio spread once lambda = 0
    pc = R['percoef']
    MD.append('\n### Efficiency ratio RMSE_DP,j / SE_OLS,j once lambda = 0 in every release (slopes only)\n')
    for e in cm.EPS:
        ns = [r['n'] for r in rows if r['eps'] == e and r['frac_ridge_on'] == 0]
        if not ns:
            continue
        for n in ns[:1] + ns[-1:]:
            eff = np.array([p['eff_ratio'] for p in pc if p['n'] == n and p['eps'] == e and p['j'] > 0])
            Rb, Ob = eff[:4], eff[4:]
            MD.append(f"- eps {e:g}, n = {n:,}: median {np.median(eff):.2f}, range {eff.min():.2f}-{eff.max():.2f} "
                      f"(spread {100 * (eff.max() / np.median(eff) - 1):+.0f}% / {100 * (eff.min() / np.median(eff) - 1):+.0f}%), "
                      f"O-block / R-block mean {Ob.mean() / Rb.mean():.2f}")
    MD.append('')
    # shrinkage share and correction vs true beta
    MD.append('### Bias correction against the true beta (cohort-level runs, R^2 = 0.5)\n')
    MD.append(f"- MSE ratio corrected/uncorrected vs true beta, ridge on: {min(r['bc_ratio_true'] ** 2 for r in on):.4f}-"
              f"{max(r['bc_ratio_true'] ** 2 for r in on):.4f}; ridge off: {min(r['bc_ratio_true'] ** 2 for r in off):.4f}-"
              f"{max(r['bc_ratio_true'] ** 2 for r in off):.4f}.")
    MD.append(f"- Shrinkage share E||theta_lambda - beta_hat||^2 / E||theta~ - beta||^2 with the ridge on: "
              f"{min(r['shrink_share'] for r in on if r['n'] >= 1000):.2f}-{max(r['shrink_share'] for r in on if r['n'] >= 1000):.2f} (n >= 1000).")
    MD.append('')


def bias_tables(B):
    cells = B['cells']
    MD.append('## 12.6 Bias correction, conditional study (5 datasets x 4 stage-1 draws x 10,000 antithetic pairs per cell)\n')
    bins = [(0, 1), (1, 2.5), (2.5, 7), (7, 15), (15, 1e9)]
    TEX.append('% tab:bias_v31 body')
    MD.append('| theta0 | cells | ridge on | bias^2/MSE (max) | max_j |bias_j|/sd_j | MSE ratio bc/unc vs theta_lambda | vs sample OLS | K: bias^2 = Var/K (min) |')
    MD.append('|---|---|---|---|---|---|---|---|')
    for lo, hi in bins:
        cs = [c for c in cells if lo <= c['theta0'] < hi]
        if not cs:
            continue
        on = sum(c['frac_ridge_on'] >= 0.5 for c in cs)
        sh = max(c['vs_lam']['bias_share_unc'] for c in cs)
        mb = max(c['vs_lam']['max_bias_over_sd_unc'] for c in cs)
        rl = [c['vs_lam']['mse_ratio'] for c in cs]
        ro = [c['vs_ols']['mse_ratio'] for c in cs]
        K = min(c['vs_lam']['K_bias_eq_var'] for c in cs)
        lab = f'$\\ge {lo:g}$' if hi > 1e8 else f'{lo:g}--{hi:g}'
        TEX.append(f"    {lab} & {len(cs)} & {on} & {100 * sh:.3f}\\% & {mb:.3f} & {min(rl):.4f}--{max(rl):.4f} & "
                   f"{min(ro):.4f}--{max(ro):.4f} & {K:,.0f} \\\\".replace(',', '{,}'))
        MD.append(f"| {lab.replace('$', '').replace(chr(92) + 'ge', '>=')} | {len(cs)} | {on} | {100 * sh:.4f}% | {mb:.3f} | "
                  f"{min(rl):.4f}-{max(rl):.4f} | {min(ro):.4f}-{max(ro):.4f} | {K:,.0f} |")
    tr = [c['theory_bias_norm_ratio'] for c in cells if c['theta0'] >= 1 and c['theory_bias_norm_ratio']]
    MD.append(f"\n- Max bias^2 share over all cells: {100 * max(c['vs_lam']['bias_share_unc'] for c in cells):.4f}%; "
              f"max |bias_j|/sd_j {max(c['vs_lam']['max_bias_over_sd_unc'] for c in cells):.3f}.")
    MD.append(f"- MSE ratio vs theta_lambda over all cells: {min(c['vs_lam']['mse_ratio'] for c in cells):.4f}-"
              f"{max(c['vs_lam']['mse_ratio'] for c in cells):.4f}; part from bias removal at most "
              f"{100 * max(c['vs_lam']['gain_bias'] for c in cells):.3f}%, from variance up to {100 * max(c['vs_lam']['gain_var'] for c in cells):.3f}%.")
    on = [c for c in cells if c['frac_ridge_on'] >= 0.5]
    MD.append(f"- With the ridge on: vs sample OLS the correction changes MSE by {100 * (min(c['vs_ols']['mse_ratio'] for c in on) - 1):+.2f}% to "
              f"{100 * (max(c['vs_ols']['mse_ratio'] for c in on) - 1):+.2f}%; deterministic shrinkage is "
              f"{min(c['vs_ols']['bias_share_unc'] for c in on):.3f}-{max(c['vs_ols']['bias_share_unc'] for c in on):.3f} of the MSE around the sample OLS.")
    MD.append(f"- Theorem 10.2 check (theta0 >= 1): simulated / predicted bias norm {min(tr):.3f}-{max(tr):.3f}.")
    MD.append(f"- Smallest K (releases averaged) at which bias^2 equals the variance of the average: "
              f"{min(c['vs_lam']['K_bias_eq_var'] for c in cells):,.0f}.")
    sw = [(c['theta0'], c['eps'], c['n']) for c in cells if c['frac_ridge_on'] < 0.5]
    MD.append(f"- Smallest theta0 at which the stage-1 ridge was off: {min(sw)[0]:.2f} (eps {min(sw)[1]:g}, n {min(sw)[2]:,}).")
    MD.append('')


ARMS = ['adassp_pub', 'adassp_matched', 'v31_joint', 'v31_blocknorm', 'v31_zscore']
LET = dict(zip(ARMS, 'abcde'))
DECK = ['housing', 'wine', 'airfoil', 'concrete', 'bike', 'elevators', 'pol', 'kin40k', 'protein']
PAIRS = [('v31_blocknorm', 'adassp_pub'), ('v31_joint', 'adassp_pub'), ('v31_joint', 'adassp_matched'),
         ('v31_blocknorm', 'v31_joint'), ('v31_zscore', 'v31_blocknorm')]


def adassp_tables(A):
    summ = {s['name']: s for s in A['summary']}
    early = A['earlier_mechanism_eps1']
    e = '1.0'
    TEX.append(f"% tab:adassp_v31 body: eps = 1, delta = {A['settings']['delta']:g}; ratio to own non-private; bold = lowest implementable private arm")
    MD.append(f"## 12.7 AdaSSP benchmark, eps = 1, delta = {A['settings']['delta']:g} (test MSE / own non-private)\n")
    MD.append('| dataset | n | d | np Wang | np block | np std | (a) | (b) | (c) | (d) | (e) | earlier VFL | AdaSSP replace-one |')
    MD.append('|---|---|---|---|---|---|---|---|---|---|---|---|---|')
    for nm in DECK:
        s = summ[nm]
        d = s['by_eps'][e]
        r = {m: d[m]['ratio_own'] for m in ARMS}
        best = min(('adassp_pub', 'v31_blocknorm', 'v31_zscore'), key=lambda m: d[m]['abs'])
        cells = []
        for m in ARMS:
            v = f'{r[m]:.2f}'
            cells.append(v)
        npw, npb, npz = s['np_abs']['np_wang'], s['np_abs']['np_v31_blocknorm'], s['np_abs']['np_v31_zscore']
        TEX.append(f"    {nm} & {s['n']:,} & {s['d']} & {npw:.3g} & {npb:.3g} & {npz:.3g} & ".replace(',', '{,}') +
                   ' & '.join(cells) + f" & {early[nm]['vfl_earlier']:.2f} \\\\")
        MD.append(f"| {nm} | {s['n']:,} | {s['d']} | {npw:.4g} | {npb:.4g} | {npz:.4g} | " + ' | '.join(cells) +
                  f" | {early[nm]['vfl_earlier']:.2f} | {d['adassp_r1']['ratio_own']:.2f} |")
    MD.append('')
    # absolute at eps = 1 on the common scale
    TEX.append('% tab:adassp_abs_v31 body: absolute test MSE on Wang\'s y scale, eps = 1')
    MD.append("### Absolute test MSE at eps = 1 on Wang's y / max|y| scale\n")
    MD.append('| dataset | np Wang | np block | np std | (a) | (b) | (c) | (d) | (e) |')
    MD.append('|---|---|---|---|---|---|---|---|---|')
    for nm in DECK:
        s = summ[nm]
        d = s['by_eps'][e]
        best = min(('adassp_pub', 'v31_blocknorm', 'v31_zscore'), key=lambda m: d[m]['abs'])
        vals = [s['np_abs']['np_wang'], s['np_abs']['np_v31_blocknorm'], s['np_abs']['np_v31_zscore']] + [d[m]['abs'] for m in ARMS]
        MD.append(f'| {nm} | ' + ' | '.join(f'{v:.4g}' for v in vals) + ' |')
        TEX.append(f'    {nm} & ' + ' & '.join(f'{v:.3g}' for v in vals) + ' \\\\')
    MD.append('')
    # verdicts
    V = A['verdicts']
    TEX.append('% tab:verdicts_v31 body: paired over folds, absolute MSE; better / worse (of 29)')
    MD.append('### Paired fold-level comparisons over all 29 datasets: better / worse / no significant difference\n')
    MD.append('| pair | basis | ' + ' | '.join(f'eps = {x:g}' for x in cm.EPS) + ' |')
    MD.append('|---|---|' + '---|' * len(cm.EPS))
    for x, y in PAIRS:
        key = f'{x}_vs_{y}'
        for basis in ('paired_abs', 'paired_ratio_own'):
            v = V[basis][key]
            MD.append(f"| ({LET[x]}) vs ({LET[y]}) | {'absolute' if basis == 'paired_abs' else 'ratio to own'} | " +
                      ' | '.join(f"{v[str(x_)]['better']} / {v[str(x_)]['worse']} / {v[str(x_)]['none']}" for x_ in cm.EPS) + ' |')
        v = V['paired_abs'][key]
        TEX.append(f'    ({LET[x]}) vs ({LET[y]}) & ' + ' & '.join(f"{v[str(x_)]['better']} / {v[str(x_)]['worse']}" for x_ in cm.EPS) + ' \\\\')
    for key in ('v31_blocknorm_bc_vs_v31_blocknorm', 'v31_zscore_bc_vs_v31_zscore'):
        v = V['paired_abs'][key]
        MD.append(f"| {key} | absolute | " + ' | '.join(f"{v[str(x_)]['better']} / {v[str(x_)]['worse']} / {v[str(x_)]['none']}" for x_ in cm.EPS) + ' |')
    MD.append('')
    # median gaps
    for eps in cm.EPS:
        g_c_a = [summ[n]['by_eps'][str(eps)]['v31_joint']['abs'] / summ[n]['by_eps'][str(eps)]['adassp_pub']['abs'] - 1 for n in summ]
        g_d_a = [summ[n]['by_eps'][str(eps)]['v31_blocknorm']['abs'] / summ[n]['by_eps'][str(eps)]['adassp_pub']['abs'] - 1 for n in summ]
        g_c_b = [summ[n]['by_eps'][str(eps)]['v31_joint']['abs'] / summ[n]['by_eps'][str(eps)]['adassp_matched']['abs'] - 1 for n in summ]
        g_d_c = [summ[n]['by_eps'][str(eps)]['v31_blocknorm']['abs'] / summ[n]['by_eps'][str(eps)]['v31_joint']['abs'] - 1 for n in summ]
        MD.append(f"- eps {eps:g}: abs MSE gap, median over 29 [range]: (c)/(a)-1 {100 * np.median(g_c_a):+.1f}% "
                  f"[{100 * min(g_c_a):+.0f}, {100 * max(g_c_a):+.0f}]; (c)/(b)-1 {100 * np.median(g_c_b):+.1f}%; "
                  f"(d)/(a)-1 {100 * np.median(g_d_a):+.1f}% [{100 * min(g_d_a):+.0f}, {100 * max(g_d_a):+.0f}]; (d)/(c)-1 {100 * np.median(g_d_c):+.1f}%")
    bc = [summ[n]['by_eps'][str(eps)]['bc_change'][k] for n in summ for eps in cm.EPS for k in ('v31_blocknorm', 'v31_zscore')]
    MD.append(f"- Bias correction changes v3.1 test MSE by {100 * min(bc):+.2f}% to {100 * max(bc):+.2f}% (all datasets, eps, both preprocessings).")
    lam = {k: [summ[n]['by_eps']['1.0']['lam_median'][k] for n in DECK] for k in ('v31_joint', 'adassp_pub', 'adassp_matched')}
    MD.append('- Median ridge at eps = 1 (deck datasets), (c) / (a) / (b): ' +
              ', '.join(f"{n} {lam['v31_joint'][i]:.0f}/{lam['adassp_pub'][i]:.0f}/{lam['adassp_matched'][i]:.0f}" for i, n in enumerate(DECK)))
    ref = max(summ[n]['by_eps'][str(eps)]['refused'][k] for n in summ for eps in cm.EPS for k in ('v31_zscore', 'v31_blocknorm', 'v31_joint'))
    MD.append(f'- Largest refused share (gate) of any v3.1 arm: {ref:.4f}.')
    MD.append('')
    MD.append('### AdaSSP accounting (check 32)\n')
    MD.append('| eps | eps achieved by published constants, add/remove | replace-one | AdaSSP s (published) | AdaSSP X\'y s.d., replace-one recal. | matched sigma | v3.1 sigma_2, B = 1 | v3.1 sigma_1 |')
    MD.append('|---|---|---|---|---|---|---|---|')
    for e_, a in A['adassp_accounting'].items():
        MD.append(f"| {float(e_):g} | {a['eps_achieved_add_remove']:.3f} | {a['eps_achieved_replace_one']:.3f} | "
                  f"{a['sig_published']['sig_xx']:.2f} | {a['sig_replace1']['sig_xy']:.2f} | {a['sig_matched']['sigma']:.2f} | "
                  f"{a['v31_B1']['sigma2']:.2f} | {a['v31_B1']['sigma1']:.2f} |")
    r1 = [summ[n]['by_eps'][str(eps)]['adassp_r1']['abs'] / summ[n]['by_eps'][str(eps)]['adassp_pub']['abs'] - 1 for n in summ for eps in cm.EPS]
    r1e = {eps: [summ[n]['by_eps'][str(eps)]['adassp_r1']['abs'] / summ[n]['by_eps'][str(eps)]['adassp_pub']['abs'] - 1 for n in summ] for eps in cm.EPS}
    MD.append(f"\n- Recalibrating AdaSSP to replace-one adjacency changes its test MSE by a median " +
              ', '.join(f"{100 * np.median(v):+.1f}% (eps {k:g})" for k, v in r1e.items()) +
              f"; range {100 * min(r1):+.1f}% to {100 * max(r1):+.1f}%.")
    MD.append('')


def sweep_tables(W):
    summ = W['summary']
    TEX.append('% tab:sweep_v31 body: n to come within 10% of own non-private; v3.1 block norm (AdaSSP published)')
    MD.append(f"## 12.8 Matched records needed to come within 10% of own non-private (delta = {W['settings']['delta']:g})\n")
    MD.append('v3.1 block unit norm, with AdaSSP (published) in brackets and v3.1 standardize after the slash.\n')
    MD.append('| dataset | n available | ' + ' | '.join(f'eps = {e:g}' for e in cm.EPS) + ' |')
    MD.append('|---|---|' + '---|' * len(cm.EPS))
    for nm in ['bike', 'elevators', 'pol', 'protein']:
        if nm not in summ:
            continue
        S = summ[nm]
        nmax = S['grid'][-1]
        cells, tcells = [], []
        for e in cm.EPS:
            E = S['eps'][str(e)]
            v, a, z = E['v31_blocknorm']['cross_own_1p10'], E['adassp_pub']['cross_own_1p10'], E['v31_zscore']['cross_own_1p10']
            cells.append(f"{fmt_n(v) or '>' + fmt_n(nmax)} ({fmt_n(a) or '>' + fmt_n(nmax)}) / {fmt_n(z) or '>' + fmt_n(nmax)}")
            tcells.append(f"{tex_n(v, nmax)} ({tex_n(a, nmax)})")
        MD.append(f"| {nm} | {nmax:,} | " + ' | '.join(cells) + ' |')
        TEX.append(f"    {nm} & {nmax:,} & ".replace(',', '{,}') + ' & '.join(tcells) + ' \\\\')
    MD.append('')
    MD.append('Same, relative to the common reference (non-private unit-ridge least squares on standardized features with an intercept, same n):\n')
    MD.append('| dataset | ' + ' | '.join(f'eps = {e:g}' for e in cm.EPS) + ' |')
    MD.append('|---|' + '---|' * len(cm.EPS))
    for nm in ['bike', 'elevators', 'pol', 'protein']:
        S = summ[nm]
        nmax = S['grid'][-1]
        MD.append(f'| {nm} | ' + ' | '.join(
            f"{fmt_n(S['eps'][str(e)]['v31_blocknorm']['cross_common_1p10']) or '>' + fmt_n(nmax)} "
            f"({fmt_n(S['eps'][str(e)]['adassp_pub']['cross_common_1p10']) or '>' + fmt_n(nmax)}) / "
            f"{fmt_n(S['eps'][str(e)]['v31_zscore']['cross_common_1p10']) or '>' + fmt_n(nmax)}" for e in cm.EPS) + ' |')
    MD.append('')
    MD.append('Non-private baselines relative to each other (mean over n): ' + '; '.join(
        f"{nm}: block/std {np.mean(np.array(summ[nm]['np_abs']['np_v31_blocknorm']) / np.array(summ[nm]['np_abs']['np_v31_zscore'])):.2f}, "
        f"Wang/std {np.mean(np.array(summ[nm]['np_abs']['np_wang']) / np.array(summ[nm]['np_abs']['np_v31_zscore'])):.2f}" for nm in summ))
    MD.append('')
    MD.append('Ratio to own non-private at selected n (v3.1 block norm / AdaSSP published / v3.1 standardize):\n')
    MD.append('| dataset | n | ' + ' | '.join(f'eps = {e:g}' for e in cm.EPS) + ' |')
    MD.append('|---|---|' + '---|' * len(cm.EPS))
    for nm, S in summ.items():
        grid = S['grid']
        for n in [1000, 10000, grid[-1]]:
            if n not in grid:
                continue
            g = grid.index(n)
            MD.append(f'| {nm} | {n:,} | ' + ' | '.join(
                f"{S['eps'][str(e)]['v31_blocknorm']['ratio_own'][g]:.3f} / {S['eps'][str(e)]['adassp_pub']['ratio_own'][g]:.3f} / "
                f"{S['eps'][str(e)]['v31_zscore']['ratio_own'][g]:.3f}" for e in cm.EPS) + ' |')
    MD.append('')
    lam = {nm: [S['eps'][str(e)]['v31_blocknorm_lam'][-1] for e in cm.EPS] for nm, S in summ.items()}
    MD.append('Median stage-1 ridge at the largest n (block norm): ' + '; '.join(f"{nm}: " + ', '.join(f'{v:.0f}' for v in l) for nm, l in lam.items()))
    MD.append('')


def main():
    if os.path.exists(P('sim_accuracy.json')):
        sim_tables(cm.load_json(P('sim_accuracy.json')))
    if os.path.exists(P('bias_study.json')):
        bias_tables(cm.load_json(P('bias_study.json')))
    if os.path.exists(P('adassp_benchmark.json')):
        adassp_tables(cm.load_json(P('adassp_benchmark.json')))
    if os.path.exists(P('sweep_n.json')):
        sweep_tables(cm.load_json(P('sweep_n.json')))
    open(P('tables_v31.tex'), 'w').write('\n'.join(TEX) + '\n')
    open(P('key_numbers.md'), 'w').write('# Key numbers of the v3.1 evaluation (generated by experiments/v31_eval/tables.py)\n\n' + '\n'.join(MD) + '\n')
    print('\n'.join(MD))


if __name__ == '__main__':
    main()
