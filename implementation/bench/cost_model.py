"""Cost model of one run (paper Table 12 structure) fitted to the measurements, validated against the end-to-end
runs at 1k-64k, and extrapolated to n_bar_O = 100k, 300k, 1M.  Nothing is run at those sizes.

Inputs: bench/results/micro_t1.json (single-thread op costs, N = 2^14), micro_t4.json, micro_t1_logN15.json,
        e2e_A_t1_w1.json and e2e_A_t1_w4.json (end-to-end runs).
Output: bench/results/cost_model.json and cost_model.md

    python bench/cost_model.py [--nRmax 2000]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results')
sys.path.insert(0, os.path.normpath(os.path.join(HERE, '..')))
from vfr import psi as P  # noqa: E402

D_R, D_O = 4, 6
N_ENT = D_O * (D_O + 1) // 2 + (D_R + 1) * D_O + (D_R + 1) + D_O + 1   # 63 (joint mode)
N_VEC = D_R + 2
N_REL = 2
MB = 2 ** 20


def load(name):
    with open(os.path.join(RES, name)) as fh:
        return json.load(fh)


def model(nbar, nRmax, mu, e2e_fit, logN=14, mu15=None, felts=None, legacy_batches=False):
    """Single-thread seconds per phase and bytes per message.  mu: micro_t1 (N = 2^14).  felts: field elements per
    PSI item (default PSIParams().felts); legacy_batches: tables filled to 0.9 N_bins (the runs before 2026-10-25)."""
    m14 = mu
    N = 2 ** logN
    Ns = N // 2
    m = -(-nbar // Ns)
    pp = P.PSIParams() if felts is None else P.PSIParams(felts=felts)
    D, alpha, _ = pp.sizes(nbar)
    W = pp.windows(D)
    F = pp.felts                                        # field elements per item: F ciphertexts per query/reply set
    batches = -(-nRmax // int(pp.cuckoo_load * pp.n_bins)) if legacy_batches else pp.tables(nRmax)[0]
    ck = m14 if logN == 14 else mu15
    t = {}
    # ---------------- O
    t['O.keygen'] = ck['ckks_rotkey_share'] + ck['ckks_make_rotkeys'] + ck['ckks_pk_share']
    t['O.psi.oprf_self'] = nbar * m14['oprf_server_evaluate_per_item']
    t['O.psi.binning'] = nbar * e2e_fit['bin_per_item']
    t['O.psi.polys'] = F * m14['psi_build_polys_1M_shape'] * (alpha * D * D) / (m14['psi_alpha_1M'] * m14['psi_D'] ** 2)
    t['O.psi.oprf_requests'] = nRmax * m14['oprf_server_blind_eval_per_item']
    per_mult = m14['bfv_compute_powers_D64'] / (m14['psi_D'] - m14['psi_W'])
    per_coef = m14['bfv_eval_partition_D64'] / (m14['psi_D'] + 1)
    t['O.psi.eval'] = batches * F * (max(D - W, 0) * per_mult + alpha * (D + 1) * per_coef)
    t['O.psi.sanitize'] = batches * 2 * alpha * F * (m14['bfv_flood_sample_N'] + m14['bfv_sanitize_excl_sampling'])
    t['O.upload_convert'] = N_VEC * m * ck['ckks_convert_O']
    t['O.agg.encode'] = N_ENT * m * ck['ckks_encode_dd']
    t['O.agg.mul_add'] = N_ENT * m * (ck['ckks_mul_plain_L0'] + ck['ckks_add_L0'])
    t['O.agg.evalsum'] = N_ENT * ck['ckks_evalsum_L1']
    t['O.agg.rescale_mask'] = N_ENT * (ck['ckks_rescale_det_L0'] + ck['ckks_mul_plain_L1'] + ck['ckks_rescale_det_L1'])
    t['O.release'] = N_REL * (ck['dg_sample_N'] + ck['dg_residues_N'] + ck['ckks_rerand_partial_dec'])
    # ---------------- R
    t['R.keygen'] = ck['ckks_rotkey_share'] + ck['ckks_pk_share']
    t['R.psi.oprf'] = nRmax * (m14['oprf_client_blind_per_item'] + m14['oprf_client_finalize_per_item'])
    t['R.psi.bfv'] = e2e_fit['bfv_keygen'] + batches * F * (W * m14['bfv_encrypt'] + 2 * alpha * m14['bfv_decrypt'])
    t['R.upload_encrypt'] = N_VEC * m * ck['ckks_encrypt_seeded_R']
    t['R.release'] = N_REL * (ck['ckks_combine_R'] + ck['ckks_decode_dd_R'])
    # ---------------- communication
    ct_up = ck['size_seeded_upload_bytes']
    b = {}
    b['R->O keygen (pk + rotation-key shares)'] = ck['size_rotkey_share_bytes'] + ct_up
    b['O->R keygen (pk share)'] = ct_up
    b['R->O OPRF'] = 32 * nRmax
    b['O->R OPRF'] = 32 * nRmax
    b['R->O BFV keys'] = m14['size_bfv_pk_relin_bytes']
    b['R->O PSI query'] = batches * F * W * m14['size_bfv_query_ct_bytes']
    b['O->R PSI reply'] = batches * 2 * alpha * F * m14['size_bfv_reply_ct_bytes']
    b['R->O upload'] = N_VEC * m * ct_up
    b['O->R releases'] = N_REL * ck['size_release_bytes']
    # ---------------- memory at O (peak, approximate)
    ct_full = ck['size_ckks_ct_L0_bytes']
    mem = dict(uploaded_cts=N_VEC * m * ct_full, rotation_keys=2 * ck['size_rotkey_share_bytes'],
               psi_tables=alpha * (2 * D + 1) * pp.n_slots * 8, bfv_powers=D * m14['size_bfv_query_ct_bytes'],
               register=nbar * (D_O + 1) * 8 * 4 + nbar * 64)
    return dict(n_bar_O=nbar, m=m, D=D, alpha=alpha, W=W, batches=batches, felts=F, t=t, bytes=b, mem=mem,
                cpu_O=sum(v for k, v in t.items() if k.startswith('O.')),
                cpu_R=sum(v for k, v in t.items() if k.startswith('R.')),
                comm=sum(b.values()), mem_O=sum(mem.values()))


def parallel(rec, cores, sp):
    """Wall-clock estimate with `cores` cores at O: measured speedups where we have them (sp), else 1."""
    t = rec['t']
    w = 0.0
    for k, v in t.items():
        if not k.startswith('O.'):
            continue
        s = 1.0
        if k.startswith('O.agg'):
            s = sp['aggregate'] * cores / 4
        elif k == 'O.psi.polys':
            s = sp['polys'] * cores / 4
        elif k == 'O.psi.eval':
            s = sp['bfv']
        elif k in ('O.psi.oprf_self', 'O.psi.binning', 'O.upload_convert'):
            s = cores * 0.9          # embarrassingly parallel; NOT implemented (assumption)
        w += v / max(1.0, s)
    return w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--nRmax', type=int, default=2000)
    a = ap.parse_args()
    mu = load('micro_t1.json')
    mu4 = load('micro_t4.json')
    mu15 = load('micro_t1_logN15.json')
    e1 = load('e2e_A_t1_w1.json')
    e4 = load('e2e_A_t1_w4.json')
    # fits from the end-to-end runs: binning cost per register row, R's BFV key generation
    big = max(e1, key=lambda r: r['n_bar_O'])
    pp = P.PSIParams()
    Dg, ag_, _ = pp.sizes(big['n_bar_O'])
    polys = mu['psi_build_polys_1M_shape'] * (ag_ * Dg * Dg) / (mu['psi_alpha_1M'] * mu['psi_D'] ** 2)
    # binning (pure Python) measured directly on random OPRF outputs
    import time as _t
    outs = [os.urandom(64) for _ in range(20000)]
    t0 = _t.time()
    P.bin_items(outs, list(range(1, 20001)), pp, 20000)
    bin_per_item = (_t.time() - t0) / 20000
    fit = dict(bin_per_item=bin_per_item,
               bfv_keygen=max(0.0, min(r['R']['psi_query'] for r in e1) - pp.windows(pp.sizes(1024)[0]) * mu['bfv_encrypt']))
    sp = dict(aggregate=sum(r['O']['aggregate_detail']['entries_wall'] for r in e1) /
              sum(r['O']['aggregate_detail']['entries_wall'] for r in e4),
              polys=mu['psi_build_polys_1M_shape'] / mu4['psi_build_polys_1M_shape'],
              bfv=mu['bfv_compute_powers_D64'] / mu4['bfv_compute_powers_D64'])
    out = dict(fit=fit, speedups_4cores=sp, validation=[], extrapolation=[], nRmax=a.nRmax)
    # ---------------- validation against the measured single-thread runs
    for r in e1:
        mrec = model(r['n_bar_O'], r['n_R_max'], mu, fit, felts=r.get('felts', 1), legacy_batches='felts' not in r)
        O = r['O']
        meas = dict(agg=O['aggregate'], psi_eval=O['psi_eval'], psi_san=O['psi_sanitize'],
                    oprf_self=O['psi_oprf_self'], upload=O['upload_convert'] + r['R']['upload_encrypt'])
        pred = dict(agg=sum(v for k, v in mrec['t'].items() if k.startswith('O.agg')), psi_eval=mrec['t']['O.psi.eval'],
                    psi_san=mrec['t']['O.psi.sanitize'], oprf_self=mrec['t']['O.psi.oprf_self'],
                    upload=mrec['t']['O.upload_convert'] + mrec['t']['R.upload_encrypt'])
        out['validation'].append(dict(n_bar_O=r['n_bar_O'], measured=meas, predicted=pred,
                                      comm_measured=sum(v['bytes'] for v in r['messages'].values()),
                                      comm_predicted=mrec['comm'], wall_measured=r['wall']))
    # ---------------- extrapolation
    for nbar in (100_000, 300_000, 1_000_000):
        for nR in (a.nRmax, 10_000):
            rec = model(nbar, nR, mu, fit)
            rec['wall_O_4cores'] = parallel(rec, 4, sp)
            rec['wall_O_16cores'] = parallel(rec, 16, sp)
            rec15 = model(nbar, nR, mu, fit, logN=15, mu15=mu15)
            rec['cpu_O_N15'] = rec15['cpu_O']
            rec['agg_N14'] = sum(v for k, v in rec['t'].items() if k.startswith('O.agg'))
            rec['agg_N15'] = sum(v for k, v in rec15['t'].items() if k.startswith('O.agg'))
            rec['comm_N15'] = rec15['comm']
            rec['n_R_max'] = nR
            out['extrapolation'].append(rec)
    with open(os.path.join(RES, 'cost_model.json'), 'w') as fh:
        json.dump(out, fh, indent=1)
    # ---------------- markdown
    L = ['# Cost model (single thread unless stated; N = 2^14, d_R = 4, d_O = 6, |E| = 63, 2 releases)', '']
    L.append(f"Fitted: binning {fit['bin_per_item'] * 1e6:.1f} us/row; R's BFV keygen {fit['bfv_keygen']:.2f} s. "
             f"Measured 4-core speedups: aggregation {sp['aggregate']:.2f}x (parallel entries), PSI polynomials "
             f"{sp['polys']:.2f}x, BFV powers {sp['bfv']:.2f}x (OpenFHE OpenMP).")
    L += ['', '## Validation against the measured end-to-end runs (single thread)', '',
          '| n_bar_O | aggregate meas/pred (s) | PSI eval meas/pred | sanitize meas/pred | O OPRF self meas/pred | upload meas/pred | comm meas/pred (MB) | wall (s) |',
          '|---|---|---|---|---|---|---|---|']
    for v in out['validation']:
        me, pr = v['measured'], v['predicted']
        L.append(f"| {v['n_bar_O']} | {me['agg']:.1f} / {pr['agg']:.1f} | {me['psi_eval']:.1f} / {pr['psi_eval']:.1f} | "
                 f"{me['psi_san']:.1f} / {pr['psi_san']:.1f} | {me['oprf_self']:.1f} / {pr['oprf_self']:.1f} | "
                 f"{me['upload']:.2f} / {pr['upload']:.2f} | {v['comm_measured'] / MB:.1f} / {v['comm_predicted'] / MB:.1f} | {v['wall_measured']:.1f} |")
    L += ['', '## Extrapolation (not run)', '',
          '| n_bar_O | n_R^max | m | D, alpha | O CPU 1 core (s) | O wall 4 cores (s) | O wall 16 cores (s) | R CPU (s) | comm (MB) | O memory (MB) | aggregate N=2^14 / 2^15 (s) |',
          '|---|---|---|---|---|---|---|---|---|---|---|']
    for r in out['extrapolation']:
        L.append(f"| {r['n_bar_O']:,} | {r['n_R_max']:,} | {r['m']} | {r['D']}, {r['alpha']} | {r['cpu_O']:.0f} | "
                 f"{r['wall_O_4cores']:.0f} | {r['wall_O_16cores']:.0f} | {r['cpu_R']:.0f} | {r['comm'] / MB:.0f} | "
                 f"{r['mem_O'] / MB:.0f} | {r['agg_N14']:.0f} / {r['agg_N15']:.0f} |")
    L += ['', '### Breakdown at n_bar_O = 1M, n_R^max = %d (single thread, seconds)' % a.nRmax, '']
    r1m = [r for r in out['extrapolation'] if r['n_bar_O'] == 1_000_000 and r['n_R_max'] == a.nRmax][0]
    for k, v in sorted(r1m['t'].items(), key=lambda kv: -kv[1]):
        L.append(f'- {k}: {v:.1f}')
    L += ['', 'Communication at 1M:', '']
    for k, v in r1m['bytes'].items():
        L.append(f'- {k}: {v / MB:.1f} MB')
    L += ['', 'Memory at O at 1M:', '']
    for k, v in r1m['mem'].items():
        L.append(f'- {k}: {v / MB:.0f} MB')
    with open(os.path.join(RES, 'cost_model.md'), 'w') as fh:
        fh.write('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
