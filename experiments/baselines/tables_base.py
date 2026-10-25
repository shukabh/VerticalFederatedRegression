"""Key numbers and tables of the three baselines, from the JSON files in reports/baselines/.

    python experiments/baselines/tables_base.py

Writes reports/baselines/key_numbers.md (every table the README and the paper draft quote).
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bcommon as bc  # noqa: E402
from bcommon import ev  # noqa: E402

EPS = [0.5, 1.0, 2.0, 4.0, 8.0]
DECK = ['housing', 'wine', 'airfoil', 'concrete', 'bike', 'elevators', 'pol', 'kin40k', 'protein']
OUT = []


def w(s=''):
    OUT.append(s)


def fk(x):
    if x is None:
        return '>100k'
    return f'{x / 1000:.1f}k' if x >= 1000 else f'{x:.0f}'


def pct(x):
    return f'{100 * x:+.1f}%'


def table(head, rows):
    w('| ' + ' | '.join(head) + ' |')
    w('|' + '|'.join(['---'] * len(head)) + '|')
    for r in rows:
        w('| ' + ' | '.join(str(c) for c in r) + ' |')
    w()


def load(name):
    p = os.path.join(bc.REPORT, name)
    return ev.load_json(p) if os.path.exists(p) else None


# ------------------------------------------------------------------------------------------- baseline 1
def full_sim():
    S = load('fullrecord_sim.json')
    if S is None:
        return
    w('## Baseline 1: full-record SSP, synthetic population (13.5 cohorts and noise, delta = 1e-9, R^2 = 0.5)')
    w()
    s = S['sensitivities']
    rows = []
    for k, lab in (('std', 'standardized'), ('bn', 'block-normalized')):
        f, v = s[k]['full'], s[k]['v31']
        rows.append([lab, f"{v['Delta']:.2f} / {f['Delta']:.2f} ({f['Delta'] / v['Delta'] - 1:+.1%})",
                     f"{v['Delta_noyy']:.2f} / {f['Delta_noyy']:.2f} ({f['Delta_noyy'] / v['Delta_noyy'] - 1:+.1%})",
                     f"{v['Delta_G']:.2f} / {f['Delta_G']:.2f} ({f['Delta_G'] / v['Delta_G'] - 1:+.1%})"])
    table(['population', 'Δ v3.1 / full', 'Δ without y\'y', 'Δ_G'], rows)
    r = S['radii']
    w(f"Exact radius q (p = 11): I {r['v31']['I']:.3f} (v3.1, zero block on A) vs {r['full']['I']:.3f} (full noise); "
      f"P {r['v31']['P']:.3f} vs {r['full']['P']:.3f}.")
    w()
    R = S['rows']
    for k, lab in (('std', 'standardized'), ('bn', 'block-normalized')):
        rows = []
        for e in EPS:
            row = [f'{e:g}']
            for sname in ('I', 'P'):
                a, b = R[f'v31_{sname}'][k][str(e)], R[f'full_{sname}'][k][str(e)]
                row += [f"{fk(a['n_off'])} / {fk(b['n_off'])}", f"{fk(a['pred_1p10'])} / {fk(b['pred_1p10'])}"]
                if k == 'std':
                    row.append(f"{fk(a['coef_2x'])} / {fk(b['coef_2x'])}")
            rows.append(row)
        head = ['ε'] + [f'{x} {s}' for s in ('I', 'P') for x in (['n_off', 'pred ≤1.1', 'coef 2×'] if k == 'std'
                                                                 else ['n_off', 'pred ≤1.1'])]
        w(f'### {lab}: v3.1 / full-record (matched records)')
        w()
        table(head, rows)
        ratios = []
        for e in EPS:
            for sname in ('I', 'P'):
                a, b = R[f'v31_{sname}'][k][str(e)], R[f'full_{sname}'][k][str(e)]
                if a['pred_1p10'] and b['pred_1p10']:
                    ratios.append(b['pred_1p10'] / a['pred_1p10'])
        w(f"Records for prediction within 10%: full-record needs {min(ratios):.2f}-{max(ratios):.2f}x those of v3.1.")
        w(f"Max refusal over n: v3.1 P {max(R['v31_P'][k][str(e)]['refused_max'] for e in EPS):.3f}, "
          f"full P {max(R['full_P'][k][str(e)]['refused_max'] for e in EPS):.3f}.")
        w()
    w(f"Reproduction: {S['reproduction_check']}")
    w()


def full_real():
    S = load('fullrecord_benchmark.json')
    if S is None:
        return
    w('## Baseline 1: full-record SSP on the 29 datasets (delta = 1e-6, 10 folds, 50 draws)')
    w()
    w(f"v3.1 recomputed vs published fold values: max relative difference {S['reproduction_check']['max_overall']:.2e}.")
    w()
    V = S['verdicts']['paired_abs']
    rows = []
    labels = {'full_blocknorm_I_vs_v31_blocknorm_I': 'full (d) I vs v3.1 (d) I',
              'full_blocknorm_P_vs_v31_blocknorm_P2': 'full (d) P vs v3.1 (d) P',
              'full_joint_I_vs_v31_joint_I': 'full (c) I vs v3.1 (c) I',
              'full_joint_P_vs_v31_joint_P2': 'full (c) P vs v3.1 (c) P',
              'full_blocknorm_P_vs_v31_blocknorm_I': 'full (d) P vs v3.1 (d) I',
              'full_joint_I_vs_v31_blocknorm_I': 'full (c) I vs v3.1 (d) I',
              'full_joint_P_vs_v31_blocknorm_P2': 'full (c) P vs v3.1 (d) P',
              'full_blocknorm_I_vs_adassp_pub': 'full (d) I vs (a) AdaSSP pub.',
              'full_blocknorm_P_vs_adassp_pub': 'full (d) P vs (a) AdaSSP pub.',
              'full_joint_I_vs_adassp_matched': 'full (c) I vs (b) AdaSSP matched',
              'full_joint_P_vs_adassp_matched': 'full (c) P vs (b) AdaSSP matched',
              'v31_blocknorm_I_vs_adassp_pub': 'v3.1 (d) I vs (a) [reference]',
              'v31_blocknorm_P2_vs_adassp_pub': 'v3.1 (d) P vs (a) [reference]'}
    for key, lab in labels.items():
        row = [lab]
        for e in EPS:
            v = V[key][str(e)]
            g = S['gaps'][str(e)]['abs_gap'][key.replace('_vs_', '/') + '-1']['median']
            row.append(f"{v['better']} / {v['worse']} ({pct(g)})")
        rows.append(row)
    w('Paired verdicts, absolute test MSE: first arm significantly lower / higher (median gap of the absolute MSE).')
    w()
    table(['comparison'] + [f'ε = {e:g}' for e in EPS], rows)
    V2 = S['verdicts']['paired_ratio_own']
    rows = []
    for key in list(labels)[:4]:
        rows.append([labels[key]] + [f"{V2[key][str(e)]['better']} / {V2[key][str(e)]['worse']}" for e in EPS])
    w('Paired verdicts on the ratio to each arm\'s own non-private fit (same own fit for each pair).')
    w()
    table(['comparison'] + [f'ε = {e:g}' for e in EPS], rows)
    summ = {s['name']: s for s in S['summary']}
    for e in ('1.0', '8.0'):
        rows = []
        for nm in DECK:
            s = summ[nm]
            b = s['by_eps'][e]
            rows.append([nm, s['n'], s['d']] + [f"{b[a]['ratio_own']:.3f}" for a in
                                                ('v31_blocknorm_I', 'full_blocknorm_I', 'v31_blocknorm_P2', 'full_blocknorm_P',
                                                 'v31_joint_I', 'full_joint_I', 'v31_joint_P2', 'full_joint_P')] +
                        [f"{b['v31_blocknorm_P2']['abs']:.4g}", f"{b['full_blocknorm_P']['abs']:.4g}"])
        w(f'### ε = {float(e):g}: ratio to own non-private fit; last two columns absolute test MSE (Wang scale), (d) under P')
        w()
        table(['dataset', 'n', 'd', 'v31 (d) I', 'full (d) I', 'v31 (d) P', 'full (d) P', 'v31 (c) I', 'full (c) I',
               'v31 (c) P', 'full (c) P', 'abs v31 (d) P', 'abs full (d) P'], rows)
    for e in ('1.0', '8.0'):
        ref = S['gaps'][e]['refused']
        w(f"ε = {float(e):g}, gate refusals of full-record arms: " +
          ', '.join(f"{k} mean {v['mean']:.4f} max {v['max']:.3f} ({v['argmax']})" for k, v in ref.items()))
        w()
    nz = []
    for s in S['summary']:
        nf = s['noise_fold0']
        nz.append((s['name'], nf['blocknorm']['P|1.0']['sigma1_ratio'], nf['blocknorm']['P|1.0']['sigma2_ratio'],
                   nf['joint']['P|1.0']['sigma1_ratio'], nf['joint']['P|1.0']['sigma2_ratio']))
    a = np.array([x[1:] for x in nz])
    w(f"Noise ratio full/v3.1 (fold 0, eps = 1): block norm sigma_1 {a[:, 0].min():.3f}-{a[:, 0].max():.3f}, "
      f"sigma_2 {a[:, 1].min():.3f}-{a[:, 1].max():.3f}; joint norm sigma_1 {a[:, 2].min():.3f}-{a[:, 2].max():.3f}, "
      f"sigma_2 {a[:, 3].min():.3f}-{a[:, 3].max():.3f}.")
    w()
    return S


# ------------------------------------------------------------------------------------------- baseline 2
def fws_tables(S, title, real=False):
    rows_all = S['rows']
    w(title)
    w()
    for e in (1.0, 8.0):
        rows = []
        for r in [x for x in rows_all if x['eps'] == e]:
            sl = r['slopes']
            row = [r['n'], f"{r['ridge_on']:.2f}"]
            row += [f"{sl['wald']['cov_theta_lam']:.3f}", f"{sl['wald']['cov_beta_clip']:.3f}",
                    (f"{sl['bias_aware']['cov_beta_clip']:.3f}" if sl['bias_aware']['finite'] > 0.999 else
                     f"{sl['bias_aware']['finite']:.2f} fin."),
                    f"{sl['fws_s2']['cov_beta_clip']:.3f}", f"{sl['fws_s2']['cov_b_ols']:.3f}",
                    f"{sl['fws_1']['cov_beta_clip']:.3f}", f"{sl['ols_t']['cov_beta_clip']:.3f}"]
            hw = lambda m: (f"{sl[m]['half_median']:.3g}" if sl[m]['half_median'] is not None else '-')
            ba = sl['bias_aware']
            row += [hw('wald'), hw('bias_aware') if ba['finite'] > 0.999 else
                    ('inf' if ba['half_median'] is None else f"{ba['half_median']:.3g} ({ba['finite']:.2f} fin.)"), hw('fws_s2'),
                    hw('fws_1'), hw('ols_t'), f"{r['fws_s2_pdfix']:.2f}", f"{r['fws_s2_s2floor']:.2f}"]
            rows.append(row)
        w(f'ε = {e:g}. Coverage of the slopes (target 0.955); median half-width; FWS-S2 PSD fix and s^2 floor shares.')
        w()
        table(['n', 'ridge on', 'Wald θ_λ', 'Wald β', 'BA β', 'FWS-S2 β', 'FWS-S2 b_OLS', 'FWS-1 β', 'OLS-t β',
               'hw Wald', 'hw BA', 'hw FWS-S2', 'hw FWS-1', 'hw OLS', 'PSD fix', 's² floor'], rows)
    w('Ranges by regime (all eps): coverage [min, max] over cells; median half-width [min, max].')
    w()
    rows = []
    for regime, A in S['by_regime'].items():
        if not A['cells']:
            continue
        for m in ('wald', 'bias_aware', 'fws_s2', 'fws_1', 'ols_t'):
            cells = []
            for est in ('theta_lam', 'b_ols', 'beta_clip') + (() if real else ('beta_true',)):
                v = A.get(f'{m}|{est}')
                cells.append(f'{v[0]:.3f}-{v[1]:.3f}' if v else '-')
            h = A.get(f'{m}|half_median')
            fin = A.get(f'{m}|finite')
            rows.append([regime, A['cells'], m] + cells + [f'{h[0]:.3g}-{h[1]:.3g}' if h else '-',
                                                           f'{fin[0]:.2f}-{fin[1]:.2f}' if fin else '-'])
    table(['regime', 'cells', 'method', 'θ_λ', 'sample OLS', 'β (clip)'] + ([] if real else ['β true']) +
          ['half-width', 'finite'], rows)


def fws_all():
    S = load('fws_sim.json')
    if S is not None:
        fws_tables(S, '## Baseline 2: FWS against v3.1 intervals, synthetic population (setting I, delta = 1e-9)')
        w(f"Reproduction of the published v3.1 half-widths: max abs diff "
          f"{max(v for v in S['reproduction_check']['max_abs_diff_half_theta_vs_published'].values() if v == v):.2e}")
        w()
    R = load('fws_real.json')
    if R is not None:
        for nm, D in R['datasets'].items():
            fws_tables(D, f"## Baseline 2 on {nm} (population N = {D['N']}, p = {D['p']}, block norm, setting I)",
                       real=True)


# ------------------------------------------------------------------------------------------- baseline 3
def tukey():
    S = load('tukey_benchmark.json')
    if S is None:
        return
    kp = f"{S['settings']['k_primary']:g}"
    w('## Baseline 3: TukeyEM on the 29 datasets (delta = 1e-6, same folds, m = floor(n / (2p)))')
    w()
    rows = []
    for s in sorted(S['summary'], key=lambda s: s['n']):
        b1, b8 = s['by_eps']['1.0'], s['by_eps']['8.0']
        rows.append([s['name'], s['n_matched'], s['p'], s['m'][kp], s['t'][kp],
                     f"{float(s['ptr_threshold']['1.0']):.1f} / {float(s['ptr_threshold']['8.0']):.1f}",
                     f"{b1['passed'][kp]:.2f} / {b8['passed'][kp]:.2f}",
                     f"{b1['tukey'][kp]['ratio_own']:.3f} / {b8['tukey'][kp]['ratio_own']:.3f}",
                     f"{b1['tukey'][kp]['abs']:.4g} / {b8['tukey'][kp]['abs']:.4g}",
                     f"{b1['v31_blocknorm_P2']['abs']:.4g} / {b8['v31_blocknorm_P2']['abs']:.4g}",
                     b1['paired_abs']['tukey_vs_v31_blocknorm_P2']['verdict'] + ' / ' +
                     b8['paired_abs']['tukey_vs_v31_blocknorm_P2']['verdict'],
                     f"{b1['best_k']:g} / {b8['best_k']:g}"])
    table(['dataset', 'n', 'p', 'm', 't', 'PTR threshold ε=1/8', 'PTR pass ε=1/8', 'ratio own ε=1/8',
           'abs ε=1/8', 'v3.1 (d) P abs', 'vs (d) P', 'best k'], rows)
    V = S['verdicts']['paired_abs']
    rows = []
    for key in V:
        rows.append([key] + [f"{V[key][str(e)]['better']} / {V[key][str(e)]['worse']}" +
                             (f" ({pct(S['gaps'][str(e)]['abs_gap'][key.replace('tukey_vs_', 'tukey/') + '-1']['median'])})"
                              if key.startswith('tukey_vs_') else '') for e in EPS])
    w('Paired verdicts, absolute test MSE: TukeyEM significantly lower / higher (median gap).')
    w()
    table(['comparison'] + [f'ε = {e:g}' for e in EPS], rows)
    rows = []
    for e in EPS:
        g = S['gaps'][str(e)]
        rows.append([f'{e:g}', f"{g['ptr_pass_mean']:.2f}", g['datasets_never_pass'], g['datasets_pass_lt_50pct'],
                     g['datasets_always_pass'], ', '.join(g['never_pass'])])
    table(['ε', 'mean pass', 'never pass', 'pass < 50%', 'always pass', 'never pass (datasets)'], rows)


def verdict_matrix():
    """Every baseline arm against v3.1 (d) under I and under P, paired over folds (abs and ratio to own fit)."""
    B, T = load('fullrecord_benchmark.json'), load('tukey_benchmark.json')
    if B is None or T is None:
        return None
    import real_full as rf
    import real_data as rdm
    pub = rf.published()
    s = ev.load_json(os.path.join(bc.ROOT, 'reports', 'v31_settings', 'adassp_benchmark.json'))
    npc = {r['name']: np.asarray(r['fold']['np_common'], float) for r in s['raw']}
    arms = {}
    for r in B['raw']:
        for k in ('full_blocknorm_I', 'full_blocknorm_P', 'full_joint_I', 'full_joint_P'):
            arms.setdefault(k, {})[r['name']] = (np.asarray(r['fold'][k], float), pub[r['name']][rf.OWN[k]])
    for r in T['raw']:
        arms.setdefault('tukey', {})[r['name']] = (np.asarray(r['fold']['2'], float), npc[r['name']])
    lab = {'full_blocknorm_I': 'full-record (d), I', 'full_blocknorm_P': 'full-record (d), P',
           'full_joint_I': 'full-record (c), I (idealized)', 'full_joint_P': 'full-record (c), P (idealized)',
           'tukey': 'TukeyEM, m = n/(2p) (idealized)'}
    out = {}
    for basis in ('abs', 'ratio_own'):
        out[basis] = {}
        for a, D in arms.items():
            for ref in ('v31_blocknorm_I', 'v31_blocknorm_P2'):
                key = f'{a}_vs_{ref}'
                out[basis][key] = {}
                for e, eps in enumerate(EPS):
                    v = []
                    g = []
                    for nm, (F, own) in D.items():
                        x, y = F[e], pub[nm][ref][e]
                        if basis == 'ratio_own':
                            x, y = x / own, y / pub[nm]['np_v31_blocknorm']
                        v.append(rdm.paired(x, y)['verdict'])
                        g.append(F[e].mean() / pub[nm][ref][e].mean() - 1)
                    out[basis][key][str(eps)] = dict(better=v.count('better'), worse=v.count('worse'),
                                                     none=v.count('none'), median_gap=float(np.median(g)))
    w('## Every baseline arm against v3.1 (d) block normalization under I and P (paired over folds, 95% CI, t with 9 df)')
    w()
    w('Cells: datasets where the baseline arm is significantly lower / higher (median gap of the absolute test MSE).')
    w()
    for basis in ('abs', 'ratio_own'):
        rows = []
        for a in arms:
            for ref, rl in (('v31_blocknorm_I', 'v3.1 (d) I'), ('v31_blocknorm_P2', 'v3.1 (d) P')):
                d = out[basis][f'{a}_vs_{ref}']
                rows.append([lab[a], rl] + [f"{d[str(e)]['better']} / {d[str(e)]['worse']}" +
                                            (f" ({pct(d[str(e)]['median_gap'])})" if basis == 'abs' else '') for e in EPS])
        w(f'Basis: {"absolute test MSE (Wang scale)" if basis == "abs" else "ratio to each arm own non-private fit"}')
        w()
        table(['arm', 'against'] + [f'ε = {e:g}' for e in EPS], rows)
    ev.save_json(out, os.path.join(bc.REPORT, 'verdicts_vs_v31d.json'))
    return out


def main():
    full_sim()
    full_real()
    fws_all()
    tukey()
    verdict_matrix()
    path = os.path.join(bc.REPORT, 'key_numbers.md')
    with open(path, 'w') as fh:
        fh.write('# Baselines: key numbers (generated by experiments/baselines/tables_base.py)\n\n' + '\n'.join(OUT) + '\n')
    print('wrote', path)


if __name__ == '__main__' and len(sys.argv) == 1:
    main()


# ------------------------------------------------------------------------------------------- LaTeX rows for the draft
def _hw(x):
    if x is None:
        return '--'
    return f'{x:.2g}' if x < 0.01 else (f'{x:.3f}' if x < 1 else (f'{x:.2f}' if x < 10 else f'{x:.1f}'))


def _cell(sl, est):
    c = sl[f'cov_{est}']
    if sl['finite'] < 0.001:
        return 'unbounded'
    if sl['finite'] < 0.999:
        return f"${c:.3f}$ [{100 * sl['finite']:.0f}\\% fin.; ${_hw(sl['half_median'])}$]"
    return f"${c:.3f}$ (${_hw(sl['half_median'])}$)"


def latex_fws_rows(rows, pick, est_beta='beta_clip'):
    out = []
    for e, ns in pick:
        first = True
        for n in ns:
            r = [x for x in rows if x['eps'] == e and x['n'] == n][0]
            sl = r['slopes']
            lab = f'{e:g}' if first else ''
            first = False
            out.append(f"    {lab} & {n:,} & {r['ridge_on']:.2f} & {_cell(sl['wald'], 'theta_lam')} & "
                       f"${sl['wald'][f'cov_{est_beta}']:.3f}$ & {_cell(sl['bias_aware'], est_beta)} & "
                       f"{_cell(sl['fws_s2'], est_beta)} & {_cell(sl['fws_1'], est_beta)} \\\\".replace(',', '{,}'))
        out.append('    \\addlinespace')
    return '\n'.join(out[:-1])


def latex():
    L = {}
    S = load('fws_sim.json')
    if S is not None:
        L['FWSROWS'] = latex_fws_rows(S['rows'], [(1.0, [1000, 5000, 7500, 10000, 25000, 50000, 100000]),
                                                 (8.0, [500, 1000, 2500, 10000, 100000])])
    R = load('fws_real.json')
    if R is not None:
        out = []
        show = {'3droad': [1000, 5000, 25000], 'protein': [1000, 5000], 'bike': [500, 2000]}
        for nm, ns_ in show.items():
            D = R['datasets'][nm]
            first = True
            for e in (1.0, 8.0):
                for n in ns_:
                    r = [x for x in D['rows'] if x['eps'] == e and x['n'] == n][0]
                    sl = r['slopes']
                    lab = nm if first else ''
                    first = False
                    out.append(f"    {lab} & {e:g} & {n:,} & {r['ridge_on']:.2f} & {_cell(sl['wald'], 'theta_lam')} & "
                               f"{_cell(sl['bias_aware'], 'beta_clip')} & {_cell(sl['fws_s2'], 'beta_clip')} & "
                               f"{_cell(sl['fws_1'], 'beta_clip')} \\\\".replace(',', '{,}'))
            out.append('    \\addlinespace')
        L['FWSREALROWS'] = '\n'.join(out[:-1])
    F = load('fullrecord_sim.json')
    if F is not None:
        R_ = F['rows']
        k_ = lambda x: '$>$100k' if x is None else (f'{x / 1000:.1f}k' if x >= 1000 else f'{x:.0f}')
        out = []
        for e in EPS:
            c = []
            for sname in ('I', 'P'):
                a, b = R_[f'v31_{sname}']['std'][str(e)], R_[f'full_{sname}']['std'][str(e)]
                c += [f"{k_(a['n_off'])} / {k_(b['n_off'])}", f"{k_(a['pred_1p10'])} / {k_(b['pred_1p10'])}",
                      f"{k_(a['coef_2x'])} / {k_(b['coef_2x'])}"]
            out.append(f'    {e:g} & ' + ' & '.join(c) + ' \\\\')
        L['FULLSIMROWS'] = '\n'.join(out)
        out = []
        for e in EPS:
            a, b = R_['v31_I']['bn'][str(e)], R_['full_I']['bn'][str(e)]
            a2, b2 = R_['v31_P']['bn'][str(e)], R_['full_P']['bn'][str(e)]
            out.append(f"    {e:g} & & {k_(a['pred_1p10'])} / {k_(b['pred_1p10'])} & & & "
                       f"{k_(a2['pred_1p10'])} / {k_(b2['pred_1p10'])} & \\\\")
        L['FULLSIMBNROWS'] = '\n'.join(out)
    B = load('fullrecord_benchmark.json')
    if B is not None:
        V = B['verdicts']['paired_abs']
        lab = [('full_blocknorm_I_vs_v31_blocknorm_I', '(d) block norm, I'),
               ('full_blocknorm_P_vs_v31_blocknorm_P2', '(d) block norm, P'),
               ('full_joint_I_vs_v31_joint_I', '(c) joint norm, I'),
               ('full_joint_P_vs_v31_joint_P2', '(c) joint norm, P'),
               ('full_blocknorm_P_vs_v31_blocknorm_I', '(d), P vs v3.1 (d), I'),
               ('full_blocknorm_I_vs_adassp_pub', '(d), I vs (a) AdaSSP, published'),
               ('full_blocknorm_P_vs_adassp_pub', '(d), P vs (a)'),
               ('full_joint_P_vs_adassp_matched', '(c), P vs (b) AdaSSP, matched')]
        out = []
        for key, lb in lab:
            cells = []
            for e in EPS:
                v = V[key][str(e)]
                g = B['gaps'][str(e)]['abs_gap'][key.replace('_vs_', '/') + '-1']['median']
                cells.append(f"{v['better']}/{v['worse']} (${100 * g:+.1f}$)")
            out.append(f'    {lb} & ' + ' & '.join(cells) + ' \\\\')
        L['FULLVERDICTROWS'] = '\n'.join(out)
    T = load('tukey_benchmark.json')
    if T is not None:
        kp = f"{T['settings']['k_primary']:g}"
        out = []
        passing = [x for x in sorted(T['summary'], key=lambda x: x['n_matched'])
                   if max(x['by_eps'][str(e)]['passed'][kp] for e in EPS) > 0]
        for x in passing:
            b1, b8 = x['by_eps']['1.0'], x['by_eps']['8.0']
            r1 = b1['tukey'][kp]['abs'] / b1['v31_blocknorm_P2']['abs']
            r8 = b8['tukey'][kp]['abs'] / b8['v31_blocknorm_P2']['abs']
            out.append(f"    {x['name']} & {x['n_matched']:,} & {x['p']} & {x['m'][kp]:,} & "
                       f"{b1['passed'][kp]:.2f} / {b8['passed'][kp]:.2f} & "
                       f"{b1['tukey'][kp]['ratio_own']:.3f} / {b8['tukey'][kp]['ratio_own']:.3f} & "
                       f"{r1:.3f} / {r8:.3f} \\\\".replace(',', '{,}'))
        L['TUKEYROWS'] = '\n'.join(out)
        L['TUKEYNFAIL'] = str(len(T['summary']) - len(passing))
    path = os.path.join(bc.REPORT, 'latex_rows.json')
    import json
    with open(path, 'w') as fh:
        json.dump(L, fh, indent=1)
    print('wrote', path)


if __name__ == '__main__' and len(sys.argv) > 1 and sys.argv[1] == 'latex':
    latex()
