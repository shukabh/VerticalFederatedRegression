"""Key numbers of the P2 evaluation, from the JSON files in reports/v31_P2/ (references from reports/v31_settings).

    python experiments/v31_P2/tables_P2.py        # writes reports/v31_P2/key_numbers.md
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p2common as pc  # noqa: E402
from p2common import ev  # noqa: E402

EPS = pc.EPS
P = lambda f: os.path.join(pc.REPORT, f)


def kfmt(v, avail=None):
    if v is None:
        return '>' + kfmt(avail) if avail else '—'
    if v >= 19950:
        return f'{v / 1000:.0f}k'
    if v >= 1000:
        return f'{v / 1000:.1f}k'
    return f'{v:.0f}'


def pct(x, d=1):
    return f'{100 * x:+.{d}f}%'


def sec_selection(L):
    d = ev.load_json(P('selection_P2.json'))
    c, s = d['criteria_all'], d['candidates']
    ch = d['chosen']
    L += ['## Selection (synthetic populations only; rule: settings_choice_P2.md, commit 0d7e7a0)', '',
          f"- P2 = {ch} (r1 exact, kappa_Psi = 2, intervals off); stage {d['stage']}; edges of the choice: {d['edges_of_choice']}",
          f"- grid-stage choice: {d['result_grid_stage']['chosen']} (tied: {d['result_grid_stage']['tied']})",
          f"- {d['result']['n_feasible']} of {len(c)} candidates feasible; {d['result']['n_binding']}/140 cells binding", '',
          '| candidate | C | C std | C bn | max refusal std | max refusal bn | mean refusal bn | feasible |',
          '|---|---|---|---|---|---|---|---|']
    for nm in d['rank_all'][:20]:
        x = c[nm]
        L.append(f"| {nm} | {x['C']:.4f} | {x['C_std']:.4f} | {x['C_bn']:.4f} | {x['refused_max_std']:.3f} | "
                 f"{x['refused_max_bn']:.3f} | {x['refused_mean_bn']:.4f} | {'yes' if x['feasible'] else 'no'} |")
    for nm, x in d['reference'].items():
        L.append(f"| ref {nm} | {x['C']:.4f} | {x['C_std']:.4f} | {x['C_bn']:.4f} | {x['refused_max_std']:.3f} | "
                 f"{x['refused_max_bn']:.3f} | {x['refused_mean_bn']:.4f} | — |")
    L += ['', 'Robustness (R^2 = 0.3, 0.9; not used):']
    for r2, v in d['robustness'].items():
        vv = v['values']
        L.append(f"- R^2 = {r2}: rule applied here picks {v['choice_if_applied_here']}; P2 rank {v['rank_of_P2_here']} by C, "
                 f"feasible {v['P2_feasible_here']}; C: P2 {vv[ch]['C']:.4f}, I {vv['I']['C']:.4f}, old P {vv['P_old']['C']:.4f}; "
                 f"P2 max refusal bn {vv[ch]['refused_max_bn']:.3f}")
    nr = d['noise_ratio']['0.5']
    L += ['', 'Noise ratio sigma1/sigma2 (R^2 = 0.5): ' + '; '.join(
        f"{k}: " + ', '.join(f"f1={f}: {v['ratio']:.2f}" for f, v in nr[k]['by_f1'].items()) for k in pc.POPS), '']


def sec_sim(L):
    S = ev.load_json(P('sim_P2.json'))
    L += ['## Synthetic validation (13.5 cohorts and noise; delta = 1e-9; 200 cohorts per n)', '',
          f"Reproduction of the published 13.5 runs on 'std': {S['reproduction_check_vs_v31_settings']}", '']
    for r2 in ('0.5', '0.3', '0.9'):
        R = S['by_r2'][r2]
        L += [f'### R^2 = {r2}: records for test MSE within 10% (1%) of non-private', '',
              '| population | setting | ' + ' | '.join(f'eps = {e:g}' for e in EPS) + ' | max refusal (any n, eps) | mean refusal |',
              '|---|---|' + '---|' * len(EPS) + '---|---|']
        for k in pc.POPS:
            for nm in ('P2', 'I', 'P_old'):
                v = [R[nm][k][str(e)] for e in EPS]
                L.append(f"| {k} | {nm} | " + ' | '.join(f"{kfmt(x['pred_1p10'], 1e5)} ({kfmt(x['pred_1p01'], 1e5)})" for x in v)
                         + f" | {max(x['refused_max'] for x in v):.3f} | {np.mean([x['refused_mean'] for x in v]):.4f} |")
        L.append('')
    R = S['by_r2']['0.5']
    L += ['n_off (eq. noff): ' + '; '.join(f"{nm} {k}: " + ', '.join(kfmt(R[nm][k][str(e)]['n_off']) for e in EPS)
                                         for nm in ('P2', 'I', 'P_old') for k in pc.POPS), '']
    L += ['Validity ratio held by the ridge (Psi = I, ridge on), median over n (range over eps): ' + '; '.join(
        f"{nm} {k}: {min(R[nm][k][str(e)]['theta_lambda_held']['median'] for e in EPS):.2f}-"
        f"{max(R[nm][k][str(e)]['theta_lambda_held']['median'] for e in EPS):.2f}" for nm in ('P2', 'I', 'P_old') for k in pc.POPS), '']
    L += ['### Switch-off of P2 (share of 200 stage-1 releases with lambda = 0 at m n_off)', '',
          '| population | eps | n_off | 0.6 | 0.8 | 1.0 | 1.2 | 1.4 | median lambda at 0.8 n_off (linear prediction) | Psi = Pi_O at 1.0 |',
          '|---|---|---|---|---|---|---|---|---|---|']
    for k in pc.POPS:
        for e in EPS:
            rs = sorted([r for r in S['noff_check'] if r['pop'] == k and r['eps'] == e], key=lambda r: r['mult'])
            m08 = [r for r in rs if r['mult'] == 0.8][0]
            m10 = [r for r in rs if r['mult'] == 1.0][0]
            L.append(f"| {k} | {e:g} | {rs[0]['n_off']:,.0f} | " + ' | '.join(f"{r['frac_off']:.3f}" for r in rs)
                     + f" | {m08['lam_median']:,.0f} ({m08['lam_pred']:,.0f}) | {m10['frac_Psi_O']:.2f} |")
    L.append('')


def sec_bench(L):
    A = ev.load_json(P('adassp_benchmark_P2.json'))
    V = A['verdicts']['paired_abs']
    G = A['gaps']
    L += ['## AdaSSP benchmark (delta = 1e-6, 29 datasets, 10 folds, 50 draws; (c), (d) under P2 rerun, the rest published)', '',
          f"Reproduction: {A['reproduction_check']['reference_verdicts_equal_published'] and 'reference verdicts equal the published ones' or 'REFERENCE VERDICTS DIFFER'}; "
          f"I recomputed on P2's noise streams vs published, max relative difference "
          f"{max(max(v.values()) for v in A['reproduction_check']['max_rel_diff_I_recomputed_vs_published'].values()):.1e}", '',
          '### Paired verdicts on absolute test MSE: better / worse / no significant difference', '',
          '| pair | ' + ' | '.join(f'eps = {e:g}' for e in EPS) + ' |', '|---|' + '---|' * len(EPS)]
    for key in ['v31_blocknorm_P2_vs_adassp_pub', 'v31_blocknorm_I_vs_adassp_pub', 'v31_blocknorm_P_vs_adassp_pub',
                'v31_blocknorm_prev_vs_adassp_pub', 'v31_blocknorm_P2_vs_adassp_matched', 'v31_blocknorm_I_vs_adassp_matched',
                'v31_blocknorm_P_vs_adassp_matched', 'v31_joint_P2_vs_adassp_pub', 'v31_joint_I_vs_adassp_pub',
                'v31_joint_P_vs_adassp_pub', 'v31_joint_prev_vs_adassp_pub', 'v31_joint_P2_vs_adassp_matched',
                'v31_joint_I_vs_adassp_matched', 'v31_joint_P_vs_adassp_matched', 'v31_joint_prev_vs_adassp_matched',
                'v31_blocknorm_P2_vs_v31_blocknorm_I', 'v31_blocknorm_P2_vs_v31_blocknorm_P', 'v31_blocknorm_P2_vs_v31_blocknorm_prev',
                'v31_joint_P2_vs_v31_joint_I', 'v31_joint_P2_vs_v31_joint_P', 'v31_blocknorm_P2_vs_v31_joint_P2',
                'v31_blocknorm_P2_vs_v31_zscore_I']:
        L.append(f'| {key} | ' + ' | '.join(f"{V[key][str(e)]['better']} / {V[key][str(e)]['worse']} / {V[key][str(e)]['none']}"
                                             for e in EPS) + ' |')
    L += ['', '### Median absolute-MSE gap over the 29 datasets [min, max]', '',
          '| ratio - 1 | ' + ' | '.join(f'eps = {e:g}' for e in EPS) + ' |', '|---|' + '---|' * len(EPS)]
    for key in G['1.0']['abs_gap']:
        L.append(f'| {key} | ' + ' | '.join(
            f"{pct(G[str(e)]['abs_gap'][key]['median'])} [{100 * G[str(e)]['abs_gap'][key]['min']:+.0f}, "
            f"{100 * G[str(e)]['abs_gap'][key]['max']:+.0f}]" for e in EPS) + ' |')
    L += ['', '### Median ridge ratio (datasets with both ridges positive)', '']
    for key in G['1.0']['lam_ratio']:
        L.append(f'- {key}: ' + '; '.join(
            f"eps {e:g}: {G[str(e)]['lam_ratio'][key]['median']:.2f} (n={G[str(e)]['lam_ratio'][key]['count']})"
            if G[str(e)]['lam_ratio'][key]['median'] else f'eps {e:g}: —' for e in EPS))
    L += ['', '### Gate refusals (share of releases), per eps: mean over datasets; max (dataset); datasets > 1%; >= 25%; all', '']
    for k in ('v31_blocknorm_P2', 'v31_joint_P2', 'v31_blocknorm_P', 'v31_joint_P'):
        L.append(f'- {k}: ' + '; '.join(
            f"eps {e:g}: {G[str(e)]['refused'][k]['mean']:.4f}, {G[str(e)]['refused'][k]['max']:.3f} ({G[str(e)]['refused'][k]['argmax']}), "
            f"{G[str(e)]['refused'][k]['datasets_ge_1pct']}, {G[str(e)]['refused'][k]['datasets_ge_25pct']}, {G[str(e)]['refused'][k]['datasets_all']}"
            for e in EPS))
    L += ['', 'Overall (mean over 29 datasets and 5 eps): ' + '; '.join(
        f"{k}: {v['mean_all']:.4f} (worst: " + ', '.join(f'{n} {x:.3f}' for n, x in v['worst'][:4]) + ')'
        for k, v in A['refused_overall'].items()), '']
    summ = {r['name']: r for r in A['summary']}
    L += ['### eps = 1: ratio to own non-private (absolute test MSE)', '',
          '| dataset | n | d | (a) | (b) | (c) I | (c) old P | (c) P2 | (d) I | (d) old P | (d) P2 | (d) P2 refused | (d) P2 sigma1/sigma2 |',
          '|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for nm in [r['name'] for r in A['summary']]:
        r = summ[nm]
        b = r['by_eps']['1.0']
        cell = lambda a: f"{b[a]['ratio_own']:.2f} ({b[a]['abs']:.3g})"
        L.append(f"| {nm} | {r['n']:,} | {r['d']} | " + ' | '.join(cell(a) for a in
                 ('adassp_pub', 'adassp_matched', 'v31_joint_I', 'v31_joint_P', 'v31_joint_P2', 'v31_blocknorm_I',
                  'v31_blocknorm_P', 'v31_blocknorm_P2'))
                 + f" | {b['refused']['v31_blocknorm_P2']:.3f} | {r['sigma_ratio_fold0']['v31_blocknorm']['P2|1.0']:.2f} |")
    L.append('')


def sec_sweep(L):
    W = ev.load_json(P('sweep_n_P2.json'))
    L += ['## Real-data sweep (delta = 1e-9; block normalization): records to come within 10% of own non-private', '',
          f"Reproduction (I recomputed vs published ratio_own, max relative difference): {W['reproduction_check']}", '',
          '| dataset | arm | ' + ' | '.join(f'eps = {e:g}' for e in EPS) + ' |', '|---|---|' + '---|' * len(EPS)]
    for nm, S in W['summary'].items():
        avail = S['grid'][-1]
        for arm, get in (('P2', lambda E: E['v31_blocknorm_P2']['cross_own_1p10']),
                         ('I', lambda E: E['reference']['I']['cross_own_1p10']),
                         ('old P', lambda E: E['reference']['P_old']['cross_own_1p10']),
                         ('previous', lambda E: E['reference']['prev']['cross_own_1p10']),
                         ('AdaSSP', lambda E: E['reference']['adassp_pub']['cross_own_1p10'])):
            L.append(f"| {nm} ({avail:,}) | {arm} | " + ' | '.join(kfmt(get(S['eps'][str(e)]), avail) for e in EPS) + ' |')
    L += ['', 'P2 gate refusals by dataset, eps (max over n; mean over n while the ridge is on in most releases); ratio at the largest n (P2 / I):', '']
    for nm, S in W['summary'].items():
        parts = []
        for e in EPS:
            E = S['eps'][str(e)]
            ref = np.array(E['v31_blocknorm_P2_refused'])
            on = np.array(E['v31_blocknorm_P2_ridge_on']) >= 0.5
            parts.append(f"eps {e:g}: max {ref.max():.3f}, on-mean {ref[on].mean() if on.any() else 0:.3f}, "
                         f"last {E['v31_blocknorm_P2']['ratio_own'][-1]:.2f}/{E['reference']['I']['ratio_own'][-1]:.2f}")
        L.append(f"- {nm} (sigma1/sigma2 = {S['sigma_ratio']['1.0']:.2f}): " + '; '.join(parts))
    L.append('')


def sec_enc(L):
    E = ev.load_json(P('encrypted_P2.json'))
    L += ['## Encrypted spot check (OpenFHE; P2 at eps = %g; n = %d in a %d-row register padded to %d)' %
          (E['eps'], E['data']['n'], E['data']['n_scope'], E['data']['n_bar_O']), '',
          f"- lambda in the simulated 95% range: {E['lam_in_range']} (range {E['sim']['lam_q'][0]:,.0f}-{E['sim']['lam_q'][2]:,.0f}; "
          f"runs: " + ', '.join(f"{r['lam']:,.0f} ({r['kind']})" for r in E['runs']) + ')',
          f"- theta in the simulated 95% range: {E['theta_in_range']}; within 2.5 s.d. of the simulation given lambda: "
          f"{E['within_2p5sd']} (max |z| {E['max_abs_z_given_lambda']:.2f}); refused {E['refused']} (simulated refusal "
          f"{E['sim']['refused']:.4f})",
          f"- r1 = {E['runs'][0]['r1']:,.0f}; n_off of the matched design {E['n_off_matched']:,.0f}; wall "
          + ', '.join(f"{r['wall']:.0f}s" for r in E['runs']),
          f"- raw decrypted noise: " + '; '.join(f"{k}: n = {v['n']}, mean {v['mean']:.3f}, s.d. {v['sd']:.3f}, max |z| {v['max_abs']:.2f}, "
                                                 f"KS p = {v['ks_p']:.2f}" for k, v in E['noise_pooled'].items())
          + f"; A exact: {E['A_exact']}", '']


def main():
    L = ['# Key numbers: setting P2 (generated by experiments/v31_P2/tables_P2.py)', '']
    for f, fn in ((sec_selection, 'selection_P2.json'), (sec_sim, 'sim_P2.json'), (sec_bench, 'adassp_benchmark_P2.json'),
                  (sec_sweep, 'sweep_n_P2.json'), (sec_enc, 'encrypted_P2.json')):
        if os.path.exists(P(fn)):
            f(L)
    with open(P('key_numbers.md'), 'w') as fh:
        fh.write('\n'.join(L) + '\n')
    print('wrote', P('key_numbers.md'))


if __name__ == '__main__':
    main()
