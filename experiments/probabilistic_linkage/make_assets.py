"""Tables (LaTeX), macros and figures of reports/probabilistic_linkage/report.tex from the experiment JSONs.

    python make_assets.py
"""
from __future__ import annotations

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, '..', '..'))
OUT = os.path.join(ROOT, 'reports', 'probabilistic_linkage')


def load(name):
    p = os.path.join(OUT, name)
    return json.load(open(p)) if os.path.exists(p) else None


def pct(x, d=1):
    return '--' if x is None or (isinstance(x, float) and np.isnan(x)) else f'{100 * x:.{d}f}\\%'


def f3(x):
    return '--' if x is None or (isinstance(x, float) and np.isnan(x)) else f'{x:.3f}'


def ints(x):
    return f'{int(x):,}'.replace(',', '{,}')


LABEL = {
    'perfect (shared ID)': 'clean shared ID (today)',
    'exact/unique/k>=1': 'exact QIs ($K{=}1$)',
    'multikey/unique/k>=1': 'multikey, filter, $\\ge1$ key',
    'multikey/unique/k>=2': 'multikey, filter, $\\ge2$ keys',
    'multikey/multi/k>=1': 'multikey, multi-label$^\\dagger$',
    'lsh/unique/k>=1': 'LSH, filter, $\\ge1$ band',
    'lsh/unique/k>=2': 'LSH, filter, $\\ge2$ bands',
    'lsh/multi/k>=1': 'LSH, multi-label$^\\dagger$, $\\ge1$',
    'lsh/multi/k>=2': 'LSH, multi-label$^\\dagger$, $\\ge2$',
    'fs-fields/multi/p0.5': 'FS on field tokens (c1), $p\\ge0.5$',
    'fs-fields/multi/p0.999': 'FS on field tokens (c1), $p\\ge0.999$',
}
ORDER = list(LABEL)


def tab_linkage(s1, s2, mac):
    L = s1['linkages']; M = s2['linkages'] if s2 else {}
    rows = []
    for k in ORDER:
        if k not in L or k.startswith('perfect'):
            continue
        r = L[k]; e = r['eval']; lk = r.get('leak', {}) or {}
        m = M.get(k, {}).get('eval', {}) if M else {}
        rows.append(f"{LABEL[k]} & {r.get('K', '')} & {pct(e['precision'], 2)} & {pct(e['recall'], 1)} & "
                    f"{pct(r.get('candidate_recall'), 1)} & {ints(lk.get('candidate_pairs', 0))} & "
                    f"{ints(lk.get('positions_of_noncohort_people', 0))} & {pct(m.get('precision'), 2)} & {pct(m.get('recall'), 1)} \\\\")
    mk = L['multikey/unique/k>=1']; ex = L['exact/unique/k>=1']; ls = L['lsh/unique/k>=1']
    mac.update(MKprec=pct(mk['eval']['precision'], 2), MKrec=pct(mk['eval']['recall'], 1),
               EXrec=pct(ex['eval']['recall'], 1), EXprec=pct(ex['eval']['precision'], 2),
               MKnoncohort=ints(mk['leak']['positions_of_noncohort_people']),
               LSHnoncohort=ints(ls['leak']['positions_of_noncohort_people']),
               LSHprec=pct(ls['eval']['precision'], 1), LSHrec=pct(ls['eval']['recall'], 1),
               MKtwoprec=pct(L['multikey/unique/k>=2']['eval']['precision'], 2),
               MKtworec=pct(L['multikey/unique/k>=2']['eval']['recall'], 1),
               MKfalse=pct(mk['eval']['false_rate'], 2))
    if M:
        mac.update(MKrecMNAR=pct(M['multikey/unique/k>=1']['eval']['recall'], 1),
                   EXrecMNAR=pct(M['exact/unique/k>=1']['eval']['recall'], 1))
    return r"""\begin{table}[t]\centering\small
\caption{Linkage on the synthetic world ($\Op$: $10^6$ rows; $\Rp$: 50{,}000 records, 40{,}000 in scope). Precision and recall
of the accepted 1:1 links against the truth; candidate recall: true pairs among the PSI candidates (the ceiling of any
resolution); candidate pairs and non-cohort positions: what $\Rp$ learns from the PSI (Section~\ref{sec:leak}). Last two
columns: the MNAR world (corruption doubled for outcomes below the median). The FS rows score the candidates of the multi-label multikey blocking. $^\dagger$Not supported by the current PSI
(one label per item).}\label{tab:linkage}
\begin{tabular}{@{}lrrrrrrrr@{}}\toprule
scheme & $K$ & precision & recall & cand.\ recall & cand.\ pairs & non-cohort pos. & prec.\ MNAR & rec.\ MNAR\\\midrule
""" + '\n'.join(rows) + r"""
\bottomrule\end{tabular}\end{table}
"""


def tab_febrl(fb):
    rows = []
    for k in ORDER:
        if k in fb:
            e = fb[k]['eval']
            rows.append(f"{LABEL[k]} & {pct(e['precision'], 2)} & {pct(e['recall'], 1)} & "
                        f"{pct(fb[k].get('candidate_recall'), 1)} \\\\")
    return r"""\begin{table}[t]\centering\small
\caption{FEBRL4 (5{,}000 originals as $\Op$, 5{,}000 corrupted duplicates as $\Rp$; no sex field). FEBRL's duplicates
carry several modifications each, so recall is lower than in our world; with only 5{,}000 rows there are almost no
collisions, so precision is near 1 for the deterministic keys.}\label{tab:febrl}
\begin{tabular}{@{}lrrr@{}}\toprule scheme & precision & recall & cand.\ recall\\\midrule
""" + '\n'.join(rows) + r"""
\bottomrule\end{tabular}\end{table}
"""


def tab_scale(sc, mac):
    keep = [('exact', 'unique', 1), ('multikey', 'unique', 1), ('multikey', 'multi', 1), ('lsh', 'unique', 1),
            ('lsh', 'multi', 1)]
    names = {('exact', 'unique', 1): 'exact QIs', ('multikey', 'unique', 1): 'multikey, filter',
             ('multikey', 'multi', 1): 'multikey, multi-label', ('lsh', 'unique', 1): 'LSH, filter',
             ('lsh', 'multi', 1): 'LSH, multi-label'}
    sizes = sorted({r['n_scope'] for r in sc})
    rows = []
    for key in keep:
        for n in sizes:
            r = [x for x in sc if (x['scheme'], x['policy'], x['min_keys']) == key and x['n_scope'] == n]
            if not r:
                continue
            r = r[0]
            rows.append(f"{names[key]} & {ints(n)} & {pct(r['precision'], 2)} & {pct(r['recall'], 1)} & "
                        f"{ints(r['candidate_pairs'])} & {ints(r['noncohort_positions'])} & "
                        f"{pct(r['O_items_dropped_nonunique'] / max(r['O_items'], 1), 1)} \\\\")
        rows.append('\\addlinespace')
    lm = [x for x in sc if (x['scheme'], x['policy'], x['min_keys']) == ('lsh', 'multi', 1)]
    if lm:
        big = max(lm, key=lambda x: x['n_scope'])
        mac.update(LSHmultiCand=ints(big['candidate_pairs']), LSHmultiN=ints(big['n_scope']))
    return r"""\begin{table}[t]\centering\small
\caption{Quality and leakage against register size ($\Rp$: 5{,}000 records, 4{,}000 in scope; at least one agreeing key,
1:1 resolution). Dropped: share of $\Op$'s key items removed by the uniqueness filter. Deterministic multikey linkage
degrades slowly with $\bar n_O$; LSH keys either lose most of their items to the filter or flood $\Rp$ with candidates.}
\label{tab:scale}
\begin{tabular}{@{}lrrrrrr@{}}\toprule scheme & $\bar n_O$ & precision & recall & cand.\ pairs & non-cohort pos. & dropped\\\midrule
""" + '\n'.join(rows[:-1]) + r"""
\bottomrule\end{tabular}\end{table}
"""


def tab_regression(s1, s2, mac):
    def rows_of(S, tag):
        out = []
        for k in ORDER[:1] + ['perfect (shared ID)'] + ORDER[1:]:
            if k not in S['linkages'] or k == 'perfect (shared ID)' and False:
                continue
        for k in ['perfect (shared ID)'] + ORDER[1:]:
            if k not in S['linkages']:
                continue
            r = S['linkages'][k]; o = r['ols']; d = r['dp']
            i1, i4, p1 = d.get('I_e1', {}), d.get('I_e4', {}), d.get('P_e1', {})
            out.append(f"{LABEL[k]} & {ints(r['n_linked'])} & {pct(r['eval'].get('false_rate'), 2)} & "
                       f"{f3(o['ratio_R'])} & {f3(o['ratio_O'])} & {f3(i1.get('rmse_median'))} & "
                       f"{f3(i1.get('cov_beta_R'))} & {f3(i4.get('rmse_median'))} & {f3(i4.get('cov_beta_R'))} & "
                       f"{f3(p1.get('rmse_median'))} \\\\")
        return out
    body = ['\\multicolumn{10}{@{}l}{\\emph{World 1 (corruption independent of the data)}}\\\\'] + rows_of(s1, 's1')
    if s2:
        body += ['\\addlinespace', '\\multicolumn{10}{@{}l}{\\emph{World 2 (MNAR: corruption doubled for $y$ below the median)}}\\\\'] + rows_of(s2, 's2')
    L = s1['linkages']
    mac.update(PerfIrmse=f3(L['perfect (shared ID)']['dp']['I_e1']['rmse_median']),
               MKIrmse=f3(L['multikey/unique/k>=1']['dp']['I_e1']['rmse_median']),
               EXIrmse=f3(L['exact/unique/k>=1']['dp']['I_e1']['rmse_median']),
               LSHIrmse=f3(L['lsh/unique/k>=1']['dp']['I_e1']['rmse_median']),
               LSHcovR=f3(L['lsh/unique/k>=1']['dp']['I_e1'].get('cov_beta_R')),
               LSHcovRfour=f3(L['lsh/unique/k>=1']['dp']['I_e4'].get('cov_beta_R')),
               LSHratioR=f3(L['lsh/unique/k>=1']['ols']['ratio_R']))
    return r"""\begin{table}[t]\centering\small
\caption{The regression on each linkage (one world; the DP columns are medians or rates over 200 releases of setting~I
and 400 of setting~P on the same linked rows). Ratio R/O: mean ratio of least-squares coefficients to the true $\bbeta$
for $\Rp$'s and $\Op$'s slopes (sampling noise of one cohort is about $\pm0.01$--$0.02$). RMSE: median coefficient RMSE
against $\bbeta$. cov.\ $\beta_R$: coverage of $\Rp$'s slopes by the bias-aware interval (nominal $0.95$).}
\label{tab:regression}
\begin{tabular}{@{}lrrrrrrrrr@{}}\toprule
 & & & \multicolumn{2}{c}{least squares} & \multicolumn{2}{c}{I, $\varepsilon=1$} & \multicolumn{2}{c}{I, $\varepsilon=4$} & P, $\varepsilon{=}1$\\
\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}\cmidrule(lr){10-10}
linkage & $n$ & false & ratio R & ratio O & RMSE & cov.\ $\beta_R$ & RMSE & cov.\ $\beta_R$ & RMSE\\\midrule
""" + '\n'.join(body) + r"""
\bottomrule\end{tabular}\end{table}
"""


def tab_sweep(sw, mac):
    rows = []
    for c in sw['cells']:
        i1, i4 = c['I_e1'], c['I_e4']
        lab = 'none' if c['mu'] == 0 else f"{int(100 * c['mu'])}\\% {c['kind'].upper()}"
        rows.append(f"{lab} & {c['pi']:.3f} & {ints(round(c['n_mean']))} & {f3(c['ols']['ratio_R'])} & "
                    f"{f3(c['formula']['ratio_R'])} & {f3(c['ols']['ratio_O'])} & {f3(i1['ratio_R'])} & "
                    f"{f3(i1.get('cov_beta_R'))} & {f3(i1.get('cov_theta'))} & {f3(i4.get('cov_beta_R'))} & "
                    f"{f3(i4.get('cov_beta_O'))} & {pct(i1['ridge_on'], 0)} \\\\")
        if c['pi'] == sw['pis'][-1]:
            rows.append('\\addlinespace')
    cells = {(c['mu'], c['kind'], c['pi']): c for c in sw['cells']}
    def g(pi, key, e='I_e4', mu=0.0, kind='mcar'):
        c = cells.get((mu, kind, pi));
        return f3(c[e].get(key)) if c else '--'
    mac.update(SwCovRfourTwo=g(0.02, 'cov_beta_R'), SwCovRfourFive=g(0.05, 'cov_beta_R'),
               SwCovRfourTen=g(0.10, 'cov_beta_R'), SwCovRoneFive=g(0.05, 'cov_beta_R', 'I_e1'),
               SwCovRoneTen=g(0.10, 'cov_beta_R', 'I_e1'), SwCovRoneTwenty=g(0.20, 'cov_beta_R', 'I_e1'),
               SwCovRfourZero=g(0.0, 'cov_beta_R'), SwCovRoneZero=g(0.0, 'cov_beta_R', 'I_e1'),
               SwMNARratio=f3(cells[(0.3, 'mnar', 0.0)]['ols']['ratio_R']) if (0.3, 'mnar', 0.0) in cells else '--',
               SwMNARratioO=f3(cells[(0.3, 'mnar', 0.0)]['ols']['ratio_O']) if (0.3, 'mnar', 0.0) in cells else '--',
               SwMNARcovRfour=g(0.0, 'cov_beta_R', 'I_e4', 0.3, 'mnar'),
               SwMNARcovOfour=g(0.0, 'cov_beta_O', 'I_e4', 0.3, 'mnar'))
    return r"""\begin{table}[t]\centering\small
\caption{Controlled linkage errors ($n_0=40{,}000$ true pairs, 100 cohorts per row, one release per cohort and
$\varepsilon$). Misses: none, 30\% at random, or 30\% with probability rising as $y$ falls (MNAR). Ratio R/O: mean
coefficient ratio to $\bbeta$ for $\Rp$'s/$\Op$'s slopes (least squares, the formula \eqref{eq:contam}, and setting I at
$\varepsilon=1$). cov.\ $\beta_R$/$\beta_O$: bias-aware interval coverage of $\Rp$'s/$\Op$'s slopes; cov.\
$\theta_\lambda$: Wald coverage of the protocol estimand on the linked rows. Nominal $0.95$.}\label{tab:sweep}
\begin{tabular}{@{}lrrrrrrrrrrr@{}}\toprule
 & & & \multicolumn{3}{c}{least squares / formula} & \multicolumn{3}{c}{I, $\varepsilon=1$} & \multicolumn{2}{c}{I, $\varepsilon=4$} & \\
\cmidrule(lr){4-6}\cmidrule(lr){7-9}\cmidrule(lr){10-11}
misses & $\pi$ & $n$ & ratio R & formula & ratio O & ratio R & cov.\ $\beta_R$ & cov.\ $\theta_\lambda$ & cov.\ $\beta_R$ & cov.\ $\beta_O$ & ridge on\\\midrule
""" + '\n'.join(rows[:-1]) + r"""
\bottomrule\end{tabular}\end{table}
"""


def tab_cost(cost, mac):
    def get(layout, nR, K):
        r = [x for x in cost if x['layout'] == layout and x['n_R_max'] == nR and x['K'] == K]
        return r[0] if r else None
    rows = []
    for nR in (20_000, 100_000):
        for K in (1, 2, 4, 7, 8, 12):
            j, p = get('joint', nR, K), get('per_type', nR, K)
            if not j or not p:
                continue
            rows.append(f"{ints(nR)} & {K} & {j['O_wall'] / 60:.1f} & {j['comm_MB'] / 1024:.2f} & {j['tables']} / {j['alpha']} & "
                        f"{p['O_wall'] / 60:.1f} & {p['comm_MB'] / 1024:.2f} \\\\")
        rows.append('\\addlinespace')
    b, p7, j7 = get('per_type', 20_000, 1), get('per_type', 20_000, 7), get('joint', 20_000, 7)
    p7b, j7b = get('per_type', 100_000, 7), get('joint', 100_000, 7)
    mac.update(CostBase=f"{b['O_wall']:.0f}", CostPerSeven=f"{p7['O_wall'] / 60:.1f}", CostJointSeven=f"{j7['O_wall'] / 60:.1f}",
               CostPerSevenBig=f"{p7b['O_wall'] / 60:.0f}", CostJointSevenBig=f"{j7b['O_wall'] / 3600:.1f}",
               CommPerSeven=f"{p7['comm_MB'] / 1024:.1f}", CommJointSevenBig=f"{j7b['comm_MB'] / 1024:.0f}",
               CommPerSevenBig=f"{p7b['comm_MB'] / 1024:.1f}")
    return r"""\begin{table}[t]\centering\small
\caption{Modelled cost of one run with $K$ keys per record at $\bar n_O=1{,}007{,}616$ (refreshed cost model; 4 threads per
party; $\Op$'s wall time in minutes, total traffic in GB). Joint: one PSI over all keys (tables/partitions per bin
shown); per type: $K$ PSI instances of today's size. $K=1$ is today's protocol ($85$\,s modelled, $91$\,s measured at
$n_R^{\max}=20{,}000$). Beyond $K=1$ these are model extrapolations, not measurements.}\label{tab:cost}
\begin{tabular}{@{}rrrrrrr@{}}\toprule
 & & \multicolumn{3}{c}{joint PSI} & \multicolumn{2}{c}{per-type PSI}\\\cmidrule(lr){3-5}\cmidrule(lr){6-7}
$n_R^{\max}$ & $K$ & $\Op$ (min) & traffic (GB) & tables / $\alpha$ & $\Op$ (min) & traffic (GB)\\\midrule
""" + '\n'.join(rows[:-1]) + r"""
\bottomrule\end{tabular}\end{table}
"""


def fig_sweep(sw):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    cs = [c for c in sw['cells'] if c['mu'] == 0.0]
    pis = [c['pi'] for c in cs]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].plot(pis, [c['formula']['ratio_R'] for c in cs], 'k-', label='formula (eq. 1)')
    ax[0].plot(pis, [c['ols']['ratio_R'] for c in cs], 'o', color='#555555', label='least squares')
    ax[0].plot(pis, [c['I_e1']['ratio_R'] for c in cs], 's--', color='#1f6fb4', label='setting I, $\\varepsilon=1$')
    ax[0].plot(pis, [c['I_e4']['ratio_R'] for c in cs], '^--', color='#d9822b', label='setting I, $\\varepsilon=4$')
    ax[0].set_xlabel('false-match rate $\\pi$'); ax[0].set_ylabel("mean $\\theta_R/\\beta_R$ (R's slopes)")
    ax[0].legend(frameon=False, fontsize=8)
    ax[1].axhline(0.95, color='k', lw=0.8, ls=':')
    ax[1].plot(pis, [c['I_e1'].get('cov_beta_R') for c in cs], 's-', color='#1f6fb4', label="$\\beta_R$, $\\varepsilon=1$")
    ax[1].plot(pis, [c['I_e4'].get('cov_beta_R') for c in cs], '^-', color='#d9822b', label="$\\beta_R$, $\\varepsilon=4$")
    ax[1].plot(pis, [c['I_e4'].get('cov_beta_O') for c in cs], 'v:', color='#d9822b', label="$\\beta_O$, $\\varepsilon=4$")
    ax[1].plot(pis, [c['I_e1'].get('cov_theta') for c in cs], 'o:', color='#555555', label="Wald $\\theta_\\lambda$, $\\varepsilon=1$")
    ax[1].set_xlabel('false-match rate $\\pi$'); ax[1].set_ylabel('coverage'); ax[1].set_ylim(0, 1.02)
    ax[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig_sweep.pdf'))


def main():
    mac = {}
    s1, s2 = load('synthetic_s1.json'), load('synthetic_s2_mnar2.json')
    fb, sc, sw, cost = load('febrl.json'), load('scale.json'), load('sweep.json'), load('cost.json')
    files = {}
    if s1:
        files['tab_linkage.tex'] = tab_linkage(s1, s2, mac)
        files['tab_regression.tex'] = tab_regression(s1, s2, mac)
        no = s1['meta']['n_off']
        mac.update(NoffIone=ints(round(no['1']['I'])), NoffIfour=ints(round(no['4']['I'])))
    if fb:
        files['tab_febrl.tex'] = tab_febrl(fb)
        mac.update(FBmkrec=pct(fb['multikey/unique/k>=1']['eval']['recall'], 1),
                   FBexrec=pct(fb['exact/unique/k>=1']['eval']['recall'], 1))
    if sc:
        files['tab_scale.tex'] = tab_scale(sc, mac)
    if sw:
        files['tab_sweep.tex'] = tab_sweep(sw, mac)
        fig_sweep(sw)
    if cost:
        files['tab_cost.tex'] = tab_cost(cost, mac)
    import sys
    sys.path.insert(0, HERE)
    import regression as rg
    B = rg.BETA
    for pi, nm in ((0.01, 'one'), (0.05, 'five'), (0.10, 'ten')):
        mac['FormulaR' + nm] = f3(float(np.mean(rg.contamination_limit(pi)[1:5] / B[1:5])))
    for name, txt in files.items():
        if name in ('tab_linkage.tex', 'tab_regression.tex', 'tab_sweep.tex', 'tab_scale.tex'):
            txt = txt.replace('\\begin{tabular}', '\\resizebox{\\textwidth}{!}{\\begin{tabular}').replace(
                '\\end{tabular}', '\\end{tabular}}')
        open(os.path.join(OUT, name), 'w').write(txt)
    with open(os.path.join(OUT, 'numbers.tex'), 'w') as fh:
        for k, v in sorted(mac.items()):
            fh.write(f'\\newcommand{{\\{k}}}{{{v}}}\n')
    json.dump(mac, open(os.path.join(OUT, 'numbers.json'), 'w'), indent=1)
    print(json.dumps(mac, indent=1))


if __name__ == '__main__':
    main()
