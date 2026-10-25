"""Key numbers of the large-scale evaluation: reports/large/key_numbers.md and summary.json, from the stage JSON files
(summarize_large.py), the simulation (sim_large.py) and the re-measured cost-model inputs (micro_large.py).

    python experiments/large/tables_large.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lcommon as lc  # noqa: E402

PH_O = ['O.oprf_self', 'O.psi_preprocess', 'O.oprf_requests', 'O.psi_eval', 'O.psi_sanitize', 'O.upload_convert',
        'O.aggregate', 'O.release', 'O.keygen']
PH_R = ['R.oprf', 'R.psi_bfv', 'R.upload_encrypt', 'R.keygen', 'R.release']


def _stage(k):
    p = os.path.join(lc.REPORT, f'stage{k}.json')
    return lc.load_json(p) if os.path.exists(p) else None


def f(x, d=2):
    if x is None:
        return '-'
    if isinstance(x, str):
        return x
    return f'{x:.{d}f}'


def cost_tables(L, summ):
    L += ['## Cost (four threads per party; O also aggregates four entries in parallel)', '']
    L += ['Measured wall time per phase (mean of the runs of the setting I, eps = 1 cell; every other cell of a size has the '
          'same operations) against the cost model with its published inputs (pub) and with re-measured inputs and the '
          'interpolation at the actual partition fill (rem).', '']
    hdr = '| phase |'
    sep = '|---|'
    cols = []
    for k, tag in ((1, '100k'), (2, '300k'), (3, '1M'), (4, '1M, n_R^max 100k'), (5, '3droad 311k')):
        st = _stage(k)
        if not st:
            continue
        cands = [v for v in st['cells'].values() if v['cell']['setting'] == 'I' and v['cell']['eps'] in (1.0,)]
        if not cands:
            continue
        cols.append((tag, cands[0]))
        hdr += f' {tag} meas | pub | rem |'
        sep += '---|---|---|'
    if not cols:
        return
    L += [hdr, sep]
    for ph in PH_O + PH_R:
        row = f'| {ph} |'
        for tag, c in cols:
            p = c['cost']['phases'][ph]
            r = c['cost_remeasured']['phases'][ph] if c.get('cost_remeasured') else None
            row += f" {f(p['measured'], 1)} | {f(p['predicted'], 1)} | {f(r['predicted'], 1) if r else '-'} |"
        L.append(row)
    row = '| **O total** |'
    for tag, c in cols:
        r = c['cost_remeasured']['O_total']['predicted'] if c.get('cost_remeasured') else None
        row += f" **{f(c['cost']['O_total']['measured'], 1)}** | {f(c['cost']['O_total']['predicted'], 1)} | {f(r, 1) if r else '-'} |"
    L.append(row)
    row = '| run wall (end to end) |'
    for tag, c in cols:
        row += f" {f(c['cost']['wall_run']['mean'], 0)} | | |"
    L.append(row)
    row = '| peak memory O / R (MB) |'
    for tag, c in cols:
        row += f" {c['cost']['peak_rss_MB']['O']:.0f} / {c['cost']['peak_rss_MB']['R']:.0f} | | |"
    L.append(row)
    row = '| traffic (MB), model |'
    for tag, c in cols:
        row += f" {c['cost']['comm_MB']:.1f} | {c['cost']['comm_model_MB']:.1f} | |"
    L.append(row)
    row = '| O CPU (s, all threads) |'
    for tag, c in cols:
        cpu = sum(v['cpu'] for v in c['cost']['cpu_marks']['O'].values())
        row += f" {cpu:.0f} | {c['cost']['model']['cpu_O_1core']:.0f} (1 core) | |"
        summ.setdefault('cpu_O', {})[tag] = cpu
    L.append(row)
    L.append('')
    for tag, c in cols:
        summ.setdefault('cost', {})[tag] = dict(O_total=c['cost']['O_total'], wall_run=c['cost']['wall_run']['mean'],
                                               peak_rss_MB=c['cost']['peak_rss_MB'], comm_MB=c['cost']['comm_MB'],
                                               comm_model_MB=c['cost']['comm_model_MB'],
                                               O_total_remeasured=c['cost_remeasured']['O_total'] if c.get('cost_remeasured') else None,
                                               traffic=c['cost']['traffic'], model=c['cost']['model'])
    L += ['Traffic per message type (MB, R\'s view; model in brackets):', '']
    for tag, c in cols:
        L.append(f'- {tag}: ' + ', '.join(f"{k} {v['MB']:.1f}" + (f" ({v['model_MB']:.1f})" if v['model_MB'] is not None else '')
                                         for k, v in c['cost']['traffic'].items() if v['MB'] >= 0.05))
    L.append('')


def gate_table(L, summ):
    L += ['## Gates', '']
    L += ['| stage | passed (published model) | cost | PSI | lambda | estimates | noise | cost with re-measured inputs | passed with re-measured inputs |',
          '|---|---|---|---|---|---|---|---|---|']
    for k in (1, 2, 3):
        st = _stage(k)
        if not st or not st.get('gate'):
            continue
        g = st['gate']
        ch = g['checks']
        L.append(f"| {k} | {g['passed']} | {ch['cost']} | {ch['psi_exact']} | {ch['lam']} ({g['lam_misses']} of {g['runs']} outside) | "
                 f"{ch['theta']} ({f(g['theta_in_95'], 3)} inside, max abs z {f(g['max_abs_z'], 2)}) | {ch['noise']} | "
                 f"{g.get('cost_remeasured_ok')} | {g.get('passed_with_remeasured_inputs')} |")
        summ.setdefault('gates', {})[k] = g
    L.append('')
    for k in (1, 2, 3):
        st = _stage(k)
        if st and st.get('gate') and st['gate']['cost_fails']:
            L.append(f"- stage {k}, phases outside 25% of the published model: " +
                     '; '.join(f"{cid}: {', '.join(v)}" for cid, v in st['gate']['cost_fails'].items()))
        if st and st.get('gate') and st['gate'].get('cost_fails_remeasured'):
            L.append(f"- stage {k}, phases outside 25% with re-measured inputs: " +
                     '; '.join(f"{cid}: {', '.join(v)}" for cid, v in st['gate']['cost_fails_remeasured'].items()))
    L.append('')


def cell_table(L, summ):
    L += ['## Cells', '']
    L += ['| cell | n | runs | PSI exact | lambda (runs) | sim 95% range | ridge on (sim) | noise s.d. st1 / st2 | KS p min | '
          'Wald covered (enc.) | Wald coverage (sim, 400) | accuracy |',
          '|---|---|---|---|---|---|---|---|---|---|---|---|']
    for k in (1, 2, 3, 4, 5):
        st = _stage(k)
        if not st:
            continue
        for cid, v in st['cells'].items():
            s = v['stats']
            lam = ', '.join(f"{d['lam']:.0f}" for d in s['runs'])
            lq = s['sim']['lam_q']
            nz = s['noise']
            ks = min(x['ks_p_min'] for x in nz.values())
            ce = s.get('coverage_encrypted')
            cs_ = s.get('coverage_sim')
            acc = s['accuracy']
            if v['cell']['dataset'] == 'synthetic':
                runs = [r for r in acc['runs'] if not r.get('refused')]
                a = (f"RMSE {np.median([r['rmse'] for r in runs]):.4f} (OLS {acc['rmse_ols']:.4f}); pred x{np.median([r['pred_ratio'] for r in runs]):.4f}"
                     f" (sim x{acc['sim']['pred_ratio_median']:.4f})") if runs else 'refused'
            else:
                runs = [r for r in acc['runs'] if not r.get('refused')]
                a = (f"test MSE x{np.median([r['ratio'] for r in runs]):.4f} (sim x{acc['sim']['ratio_median']:.4f})") if runs else 'refused'
            L.append(f"| {cid} | {v['n']:,} | {len(s['runs'])} | {s['psi_exact']} | {lam} | {lq[0]:.0f}-{lq[2]:.0f} | {s['sim']['ridge_on']:.2f} | "
                     f"{nz['1']['sd']:.4f} / {nz['2']['sd']:.4f} | {ks:.3f} | "
                     f"{(str(ce['covered']) + '/' + str(ce['total'])) if ce else '-'} | "
                     f"{f(cs_['wald']['coverage'], 3) if cs_ else '-'} | {a} |")
            summ.setdefault('cells', {})[cid] = dict(n=v['n'], lam=[d['lam'] for d in s['runs']], lam_q=lq,
                                                     ridge_on_sim=s['sim']['ridge_on'], noise=nz, psi_exact=s['psi_exact'],
                                                     coverage_encrypted=ce, coverage_sim=cs_, accuracy=acc,
                                                     intervals=s.get('intervals'))
    L.append('')


def sim_table(L, summ):
    p = os.path.join(lc.REPORT, 'sim_large.json')
    if not os.path.exists(p):
        return
    S = lc.load_json(p)
    L += [f"## Plaintext simulation at large n ({S['reps']} cohorts per n)", '']
    for kind, st in (('std', 'I'), ('std', 'P'), ('bn', 'P')):
        L += [f'### {kind}, setting {st}: population prediction error / least squares (coefficient RMSE / least squares; ridge-on fraction)', '']
        L += ['| eps | ' + ' | '.join(f'{n:,}' for n in S['ns']) + ' |', '|---|' + '---|' * len(S['ns'])]
        for eps in S['eps']:
            row = f'| {eps:g} |'
            for n in S['ns']:
                r = S['results'][f'{kind}_{n}']['arms'][f'{st}_e{eps:g}']
                row += f" {r['pred_ratio_median']:.4f} ({r['rmse_ratio']:.2f}; {r['ridge_on']:.2f}) |"
            L.append(row)
        L.append('')
        if st == 'I':
            L += ['Wald coverage of theta_lambda (pooled) and median bias-aware half-width for beta:', '',
                  '| eps | ' + ' | '.join(f'{n:,}' for n in S['ns']) + ' |', '|---|' + '---|' * len(S['ns'])]
            for eps in S['eps']:
                row = f'| {eps:g} |'
                for n in S['ns']:
                    r = S['results'][f'{kind}_{n}']['arms'][f'{st}_e{eps:g}']
                    row += f" {f(r.get('wald_coverage'), 3)} / {f(r.get('half_beta_median'), 4)} |"
                L.append(row)
            L.append('')
    summ['sim'] = S


def main():
    L = ['# Large-scale evaluation: key numbers', '', '(generated by experiments/large/tables_large.py)', '']
    summ = {}
    gate_table(L, summ)
    cost_tables(L, summ)
    cell_table(L, summ)
    sim_table(L, summ)
    mp = os.path.join(lc.REPORT, 'micro_large.json')
    if os.path.exists(mp):
        M = lc.load_json(mp)
        L += ['## Interpolation at the actual partition fill (s)', '', '| n_bar_O | threads | alpha | fill | seconds |',
              '|---|---|---|---|---|']
        for r in M['polys_at_fill']:
            L.append(f"| {r['n_bar_O']:,} | {r['threads']} | {r['alpha']} | {r['fill']:.2f} | {r['seconds']:.2f} |")
        L.append('')
    with open(os.path.join(lc.REPORT, 'key_numbers.md'), 'w') as fh:
        fh.write('\n'.join(L) + '\n')
    lc.save_json({k: v for k, v in summ.items() if k != 'sim'}, os.path.join(lc.REPORT, 'summary.json'))
    print('\n'.join(L))


if __name__ == '__main__':
    main()
