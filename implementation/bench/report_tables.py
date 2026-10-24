"""Render the benchmark results as markdown: bench/results/benchmarks.md."""
from __future__ import annotations

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results')
MB = 2 ** 20


def load(n):
    p = os.path.join(RES, n)
    return json.load(open(p)) if os.path.exists(p) else None


def main():
    L = ['# Benchmarks (4-core Xeon @ 2.8 GHz, 15 GB; N = 2^14 CKKS, N_B = 2^14 BFV; d_R = 4, d_O = 6, n_R^max = 2000)', '']
    e1, e4 = load('e2e_A_t1_w1.json'), load('e2e_A_t1_w4.json')
    if e1:
        L += ['## End to end, per phase (seconds; circuit A; DP noise on; single thread per party)', '',
              '| n_bar_O | m | PSI D, alpha | O: OPRF self | O: PSI preproc | O: PSI eval | O: sanitize | R: upload enc | O: convert | O: aggregate | O: aggregate, 4 threads | O: 2 releases | R: 2 releases | R: solve | wall | wall (4 thr agg) |',
              '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
        for r, r4 in zip(e1, e4 or e1):
            O, R = r['O'], r['R']
            rel = sum(v['sample'] + v['rerand_pdec'] for v in O['release'].values())
            relR = sum(R['release_R'].values())
            L.append(f"| {r['n_bar_O']} | {O['aggregate_detail']['m']} | {r['psi']['D']}, {r['psi']['alpha']} | {O['psi_oprf_self']:.2f} | "
                     f"{O['psi_preprocess']:.2f} | {O['psi_eval']:.2f} | {O['psi_sanitize']:.2f} | {R['upload_encrypt']:.2f} | "
                     f"{O['upload_convert']:.2f} | {O['aggregate']:.2f} | {r4['O']['aggregate']:.2f} | {rel:.2f} | {relR:.2f} | "
                     f"{R['solve']:.2f} | {r['wall']:.1f} | {r4['wall']:.1f} |")
        L += ['', 'Keygen: R 1.9 s (rotation-key shares), O 2.0 s (shares + joint keys); setup/calibration 0.15 s.', '',
              '## Messages (bytes, from the transcript)', '']
        types = list(e1[0]['messages'].keys())
        L.append('| message (R view) | ' + ' | '.join(str(r['n_bar_O']) for r in e1) + ' |')
        L.append('|---' * (len(e1) + 1) + '|')
        for t in types:
            L.append(f'| {t} | ' + ' | '.join(f"{r['messages'].get(t, {}).get('bytes', 0) / MB:.3f} MB" for r in e1) + ' |')
        L.append('| total | ' + ' | '.join(f"{sum(v['bytes'] for v in r['messages'].values()) / MB:.1f} MB" for r in e1) + ' |')
    for th in (1, 4):
        mu = load(f'micro_t{th}.json')
        if not mu:
            continue
        L += ['', f'## Microbenchmarks, {th} thread(s) (median, seconds)', '', '| operation | time |', '|---|---|']
        for k, v in mu.items():
            if isinstance(v, (int, float)) and not k.startswith('size') and k not in ('threads', 'psi_D', 'psi_alpha_1M', 'psi_W', 'dg_log2_sigma_c'):
                L.append(f'| {k} | {v:.4g} |')
        if th == 1:
            L += ['', '| size | bytes |', '|---|---|'] + [f'| {k} | {v:,} |' for k, v in mu.items() if k.startswith('size')]
            L.append(f"| (sigma_c of the release, log2) | {mu['dg_log2_sigma_c']:.2f} |")
    m15 = load('micro_t1_logN15.json')
    if m15:
        L += ['', '## CKKS at N = 2^15 (1 thread)', '', '| operation | time |', '|---|---|']
        for k, v in m15.items():
            if isinstance(v, float) and not k.startswith('size'):
                L.append(f'| {k} | {v:.4g} |')
    alt = load('scale_options.json')
    if alt and 'alt_backends' in alt:
        L += ['', '## Scale options: OpenFHE-only operation costs (1 thread)', '',
              '| backend | Q limbs (bits) | P limbs | log QP | native scale | ct-pt mult | rescale | rotation | EvalSum (13 rot.) |', '|---|---|---|---|---|---|---|---|---|']
        for r in alt['alt_backends']:
            if 'error' in r:
                continue
            name = ('128-bit build ' if r['native_int'] == 128 else '64-bit build ') + r['tech']
            L.append(f"| {name} | {r['towers_Q']} ({','.join(map(str, r['q_bits']))}) | {r['towers_P']} | {r['logQP']} | 2^{r['scale_log2']:.1f} | "
                     f"{r['t_mul_plain'] * 1e3:.2f} ms | {r['t_rescale'] * 1e3:.2f} ms | {r['t_rotate'] * 1e3:.1f} ms | {r['t_evalsum']:.3f} s |")
        L += ['', '| circuit | n_bar_O | eps | log2 Omega_f | log2 B_tot (stage 1, 2) | sigma_2 (slot) | q0 needed: spec / (Q) (bits) |', '|---|---|---|---|---|---|---|']
        for r in alt['rows']:
            L.append(f"| {r['variant']} | {r['n_bar_O']:,} | {r['eps']} | {r['log2_Omega_f']} | {r['log2_Btot_1']}, {r['log2_Btot_2']} | {r['sigma_slot_2']} | {r['q0_need_bits']} / {r['q0_need_Q_bits']} |")
    with open(os.path.join(RES, 'benchmarks.md'), 'w') as fh:
        fh.write('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
