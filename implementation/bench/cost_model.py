"""Cost model of one run (paper Table 12 structure) fitted to the measurements, validated against the end-to-end
runs at 1k-64k, and extrapolated to n_bar_O = 100k, 300k, 1M.  Nothing is run at those sizes here; the large runs of
reports/large/ are compared with it by experiments/eng/model_check.py.

Inputs: bench/results/micro_t1.json (single-thread op costs, N = 2^14), micro_t4.json (4 OpenMP threads),
        micro_t1_logN15.json, agg_speed.json (O's aggregation at (threads, parallel entries) = (1, 1), (4, 4), (1, 4)),
        e2e_A_t1_w1.json and e2e_A_t1_w4.json (end-to-end runs).
Output: bench/results/cost_model.json and cost_model.md

Terms refreshed on 2026-10-26 (experiments/eng/refresh_cost_model.py):
  * O's PSI interpolation at the actual partition fill: the membership polynomials always have degree D (dummy
    roots), the label interpolation costs O(n_a^2) in the count n_a of a partition.  The two parts are measured
    separately (full and empty partitions at the 1M shape) and the label part is scaled by E[sum_a n_a^2] under
    bin loads Binomial(n_bar_O h, 1/N_bins), partitions filled in order (psi.bin_items), instead of alpha D^2.
  * The aggregation speedup is the one of the configuration that is run, 4 OpenMP threads x 4 parallel entries
    (agg_speed.json), not one thread per entry.
  * O's OPRF over its register and its answer to the blinded queries are batched in the native layer and parallel
    (vfr/oprf.py); micro_t{1,4} time that code ('oprf_server_*_per_item'); the earlier single-threaded Python loop is
    kept as '*_py' for the runs made before it (oprf='python').
  * predict_run(): the per-phase wall time of a run in the configuration of the large runs (4 OpenMP threads per
    party, 4 parallel entries), with optional release padding (Agreement.release_pad).

    python bench/cost_model.py [--nRmax 2000]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

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


def n_entries(d_R, d_O):
    """released entries in joint mode: triu B, C, c_R, c_O, y'y."""
    return d_O * (d_O + 1) // 2 + (d_R + 1) * d_O + (d_R + 1) + d_O + 1


def fill_sq(nbar, pp, D, alpha):
    """Per bin: (E[n_a^2] for each partition a, mean fill).  Load L ~ Binomial(nbar h, 1/N_bins) (capped at alpha D,
    which the load bound exceeds with probability <= delta_psi); partition a holds min(max(L - a D, 0), D) items."""
    from scipy.stats import binom
    hi = alpha * D
    L = np.arange(0, hi + 1)
    n_tr = nbar * pp.h
    pmf = binom.pmf(L, n_tr, 1.0 / pp.n_bins)
    pmf[-1] += binom.sf(hi, n_tr, 1.0 / pp.n_bins)
    e2 = [float((pmf * np.clip(L - a * D, 0, D).astype(float) ** 2).sum()) for a in range(alpha)]
    return e2, float((pmf * L).sum()) / hi


def polys_time(nbar, mu, pp=None, fill=True, mu4=None):
    """O's PSI interpolation (all felts).  fill=True: membership part alpha D^2 scaled from the empty-partition
    measurement, label part sum_a E[n_a^2] scaled from the full-minus-empty measurement (mu: single-thread inputs).
    With mu4 (4-thread inputs), the wall time on 4 threads under the static OpenMP schedule of psi_build_polys, which
    splits the (alpha, F N_bins) iterations into 4 contiguous blocks: partition a = 0 is the fullest, so the threads'
    shares differ; wall = (largest share) x 4 / s, s the measured 4-thread speedup at full partitions.
    fill=False, or inputs without the empty-partition measurement: the earlier term (full partitions, alpha D^2)."""
    pp = pp or P.PSIParams()
    D, alpha, _ = pp.sizes(nbar)
    full, a1, D1 = mu['psi_build_polys_1M_shape'], mu['psi_alpha_1M'], mu['psi_D']
    if not fill or 'psi_build_polys_1M_shape_empty' not in mu:
        m = mu4 or mu
        return pp.felts * m['psi_build_polys_1M_shape'] * (alpha * D * D) / (a1 * D1 ** 2)
    empty = mu['psi_build_polys_1M_shape_empty']
    e2, _ = fill_sq(nbar, pp, D, alpha)
    w = [pp.felts * (empty * D * D + max(full - empty, 0.0) * e2[a]) / (a1 * D1 ** 2) for a in range(alpha)]
    if mu4 is None:
        return sum(w)
    T = 4
    s4 = full / mu4['psi_build_polys_1M_shape']
    shares = []
    for k in range(T):                                   # thread k: rows [k alpha / T, (k + 1) alpha / T)
        lo, hi = k * alpha / T, (k + 1) * alpha / T
        shares.append(sum(w[a] * max(0.0, min(hi, a + 1) - max(lo, a)) for a in range(alpha)))
    return max(shares) * T / s4


def model(nbar, nRmax, mu, e2e_fit, logN=14, mu15=None, felts=None, legacy_batches=False, d_R=D_R, d_O=D_O,
          oprf='native', fill=True, mu1=None):
    """Single-thread seconds per phase and bytes per message.  mu: micro_t1 (N = 2^14).  felts: field elements per
    PSI item (default PSIParams().felts); legacy_batches: tables filled to 0.9 N_bins (the runs before 2026-10-25).
    d_R, d_O: the run's dimensions (released entries, upload vectors).  oprf: 'native' (the batched OPRF of the current
    code) or 'python' (the item-by-item loop of the runs before 2026-10-26).  fill: interpolation at the actual
    partition fill (polys_time).  mu1: the single-thread inputs when mu holds 4-thread ones (the interpolation is
    then evaluated under the 4-thread schedule)."""
    m14 = mu
    N_ENT_ = n_entries(d_R, d_O)
    N_VEC_ = d_R + 2
    sfx = '_py' if oprf == 'python' and 'oprf_server_evaluate_per_item_py' in m14 else ''
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
    t['O.psi.oprf_self'] = nbar * m14['oprf_server_evaluate_per_item' + sfx]
    t['O.psi.binning'] = nbar * e2e_fit['bin_per_item']
    t['O.psi.polys'] = polys_time(nbar, mu1, pp, fill, mu4=m14) if mu1 is not None else polys_time(nbar, m14, pp, fill)
    t['O.psi.oprf_requests'] = nRmax * m14['oprf_server_blind_eval_per_item' + sfx]
    per_mult = m14['bfv_compute_powers_D64'] / (m14['psi_D'] - m14['psi_W'])
    per_coef = m14['bfv_eval_partition_D64'] / (m14['psi_D'] + 1)
    t['O.psi.eval'] = batches * F * (max(D - W, 0) * per_mult + alpha * (D + 1) * per_coef)
    t['O.psi.sanitize'] = batches * 2 * alpha * F * (m14['bfv_flood_sample_N'] + m14['bfv_sanitize_excl_sampling'])
    t['O.upload_convert'] = N_VEC_ * m * ck['ckks_convert_O']
    t['O.agg.encode'] = N_ENT_ * m * ck['ckks_encode_dd']
    t['O.agg.mul_add'] = N_ENT_ * m * (ck['ckks_mul_plain_L0'] + ck['ckks_add_L0'])
    t['O.agg.evalsum'] = N_ENT_ * ck['ckks_evalsum_L1']
    t['O.agg.rescale_mask'] = N_ENT_ * (ck['ckks_rescale_det_L0'] + ck['ckks_mul_plain_L1'] + ck['ckks_rescale_det_L1'])
    t['O.release'] = N_REL * (ck['dg_sample_N'] + ck['dg_residues_N'] + ck['ckks_rerand_partial_dec'])
    # ---------------- R
    t['R.keygen'] = ck['ckks_rotkey_share'] + ck['ckks_pk_share']
    t['R.psi.oprf'] = nRmax * (m14['oprf_client_blind_per_item'] + m14['oprf_client_finalize_per_item'])
    t['R.psi.bfv'] = e2e_fit['bfv_keygen'] + batches * F * (W * m14['bfv_encrypt'] + 2 * alpha * m14['bfv_decrypt'])
    t['R.upload_encrypt'] = N_VEC_ * m * ck['ckks_encrypt_seeded_R']
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
    b['R->O upload'] = N_VEC_ * m * ct_up
    b['O->R releases'] = N_REL * ck['size_release_bytes']
    # ---------------- memory at O (peak, approximate)
    ct_full = ck['size_ckks_ct_L0_bytes']
    mem = dict(uploaded_cts=N_VEC_ * m * ct_full, rotation_keys=2 * ck['size_rotkey_share_bytes'],
               psi_tables=alpha * (2 * D + 1) * pp.n_slots * 8, bfv_powers=D * m14['size_bfv_query_ct_bytes'],
               register=nbar * (d_O + 1) * 8 * 4 + nbar * 64)
    return dict(n_bar_O=nbar, m=m, D=D, alpha=alpha, W=W, batches=batches, felts=F, t=t, bytes=b, mem=mem,
                cpu_O=sum(v for k, v in t.items() if k.startswith('O.')),
                cpu_R=sum(v for k, v in t.items() if k.startswith('R.')),
                comm=sum(b.values()), mem_O=sum(mem.values()))


def predict_run(nbar, nRmax, mu1, mu4, fit, sp_agg, threads=4, workers=4, d_R=D_R, d_O=D_O, oprf='native',
                fill=True, pad_margin=None):
    """Per-phase wall seconds of one run in the configuration of the large runs (reports/large/): `threads` OpenMP
    threads per party (OpenFHE, the PSI interpolation and the batched OPRF use them: micro_t4 inputs), `workers`
    parallel entries in O's aggregation (the single-thread aggregation divided by the measured speedup sp_agg of
    that configuration), single-threaded Python phases (binning; the item-by-item OPRF with oprf='python') at their
    single-thread inputs.  pad_margin: release padding with release_pad='auto' (each release held for pad_margin x
    its calibrated time).  Returns {phase: seconds} on the implementation's phases, plus the totals."""
    mt = mu4 if threads == 4 else mu1
    m1 = model(nbar, nRmax, mu1, fit, d_R=d_R, d_O=d_O, oprf=oprf, fill=fill)
    m4 = model(nbar, nRmax, mt, fit, d_R=d_R, d_O=d_O, oprf=oprf, fill=fill, mu1=mu1 if threads == 4 else None)
    t = dict(m4['t'])
    single = ['O.psi.binning']
    if oprf == 'python':
        single += ['O.psi.oprf_self', 'O.psi.oprf_requests']
    single += ['R.psi.oprf']                                # R's OPRF stays item by item (n_R^max items)
    for k in single:
        t[k] = m1['t'][k]
    agg1 = sum(v for k, v in m1['t'].items() if k.startswith('O.agg'))
    rel = t['O.release']
    if pad_margin:                                      # every release held to its deadline, plus the calibration
        # (3 timed samplings with their residues and one partial decryption, party_o.calibrate_release_time)
        rel = rel * pad_margin + 3 * (mt['dg_sample_N'] + mt['dg_residues_N']) + mt['ckks_rerand_partial_dec']
    ph = {
        'O.oprf_self': t['O.psi.oprf_self'],
        'O.psi_preprocess': t['O.psi.binning'] + t['O.psi.polys'],
        'O.oprf_requests': t['O.psi.oprf_requests'],
        'O.psi_eval': t['O.psi.eval'],
        'O.psi_sanitize': t['O.psi.sanitize'],
        'O.upload_convert': t['O.upload_convert'],
        'O.aggregate': agg1 / (sp_agg if workers > 1 else 1.0),
        'O.release': rel,
        'O.keygen': t['O.keygen'],
        'R.oprf': t['R.psi.oprf'],
        'R.psi_bfv': t['R.psi.bfv'],
        'R.upload_encrypt': t['R.upload_encrypt'],
        'R.keygen': t['R.keygen'],
        'R.release': t['R.release'],
    }
    return dict(phases=ph, O_total=sum(v for k, v in ph.items() if k.startswith('O.')), comm=m1['comm'],
                bytes=m1['bytes'], cpu_O_1core=m1['cpu_O'], D=m1['D'], alpha=m1['alpha'], tables=m1['batches'],
                m=m1['m'], polys_1thread=m1['t']['O.psi.polys'])


def speedups(mu, mu4, e1, e4):
    """Measured 4-core speedups.  aggregate: the configuration that is run (4 OpenMP threads x 4 parallel entries,
    agg_speed.json); aggregate_1x4: one thread per entry (the earlier, published figure, from the e2e runs or
    agg_speed.json)."""
    sp = dict(polys=mu['psi_build_polys_1M_shape'] / mu4['psi_build_polys_1M_shape'],
              bfv=mu['bfv_compute_powers_D64'] / mu4['bfv_compute_powers_D64'],
              oprf=mu['oprf_server_evaluate_per_item'] / mu4['oprf_server_evaluate_per_item'])
    sp['aggregate_1x4_e2e'] = (sum(r['O']['aggregate_detail']['entries_wall'] for r in e1) /
                               sum(r['O']['aggregate_detail']['entries_wall'] for r in e4))
    path = os.path.join(RES, 'agg_speed.json')
    if os.path.exists(path):
        ag = {(r['threads'], r['workers']): r['seconds'] for r in load('agg_speed.json')['runs']}
        sp['aggregate'] = ag[(1, 1)] / ag[(4, 4)]
        sp['aggregate_1x4'] = ag[(1, 1)] / ag[(1, 4)]
        sp['aggregate_source'] = 'agg_speed.json: (1 thread, 1 entry) / (4 threads, 4 entries)'
    else:
        sp['aggregate'] = sp['aggregate_1x4'] = sp['aggregate_1x4_e2e']
        sp['aggregate_source'] = 'e2e runs: 1 thread x 4 entries (agg_speed.json missing)'
    return sp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--nRmax', type=int, default=2000)
    a = ap.parse_args()
    mu = load('micro_t1.json')
    mu4 = load('micro_t4.json')
    mu15 = load('micro_t1_logN15.json')
    e1 = load('e2e_A_t1_w1.json')
    e4 = load('e2e_A_t1_w4.json')
    pp = P.PSIParams()
    # binning (pure Python) measured directly on random OPRF outputs, at 200,000 rows (at 20,000 rows the per-row
    # time varied by 40% between runs)
    import time as _t
    nbin = 200_000
    outs = [os.urandom(64) for _ in range(nbin)]
    t0 = _t.time()
    P.bin_items(outs, list(range(1, nbin + 1)), pp, nbin)
    bin_per_item = (_t.time() - t0) / nbin
    fit = dict(bin_per_item=bin_per_item,
               bfv_keygen=max(0.0, min(r['R']['psi_query'] for r in e1) - pp.windows(pp.sizes(1024)[0]) * mu['bfv_encrypt']))
    sp = speedups(mu, mu4, e1, e4)
    out = dict(fit=fit, speedups_4cores=sp, validation=[], extrapolation=[], nRmax=a.nRmax,
               micro_stamp=dict(t1=mu.get('stamp'), t4=mu4.get('stamp')))
    # ---------------- validation against the measured single-thread runs (item-by-item OPRF at the time)
    for r in e1:
        mrec = model(r['n_bar_O'], r['n_R_max'], mu, fit, felts=r.get('felts', 1), legacy_batches='felts' not in r,
                     oprf='python')
        O = r['O']
        meas = dict(agg=O['aggregate'], psi_eval=O['psi_eval'], psi_san=O['psi_sanitize'],
                    oprf_self=O['psi_oprf_self'], upload=O['upload_convert'] + r['R']['upload_encrypt'])
        pred = dict(agg=sum(v for k, v in mrec['t'].items() if k.startswith('O.agg')), psi_eval=mrec['t']['O.psi.eval'],
                    psi_san=mrec['t']['O.psi.sanitize'], oprf_self=mrec['t']['O.psi.oprf_self'],
                    upload=mrec['t']['O.upload_convert'] + mrec['t']['R.upload_encrypt'])
        out['validation'].append(dict(n_bar_O=r['n_bar_O'], measured=meas, predicted=pred,
                                      comm_measured=sum(v['bytes'] for v in r['messages'].values()),
                                      comm_predicted=mrec['comm'], wall_measured=r['wall']))
    # ---------------- extrapolation (batched, parallel OPRF; release padding with margin 2)
    for nbar in (100_000, 300_000, 1_000_000):
        for nR in (a.nRmax, 20_000):
            rec = model(nbar, nR, mu, fit)
            run = predict_run(nbar, nR, mu, mu4, fit, sp['aggregate'], pad_margin=2.0)
            rec['wall_O_4cores'] = run['O_total']
            rec['wall_O_4cores_phases'] = run['phases']
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
    L.append(f"Inputs measured {mu.get('stamp', '?')} (1 thread) and {mu4.get('stamp', '?')} (4 threads) on the idle "
             f"machine. Fitted: binning {fit['bin_per_item'] * 1e6:.1f} us/row; R's BFV keygen {fit['bfv_keygen']:.2f} s. "
             f"Measured 4-core speedups: aggregation {sp['aggregate']:.2f}x in the run configuration (4 threads x 4 "
             f"parallel entries; {sp['aggregate_1x4']:.2f}x with one thread per entry), PSI polynomials "
             f"{sp['polys']:.2f}x, BFV powers {sp['bfv']:.2f}x (OpenFHE OpenMP), O's OPRF {sp['oprf']:.2f}x (batched, "
             f"OpenMP). The PSI interpolation is evaluated at the actual partition fill.")
    L += ['', '## Validation against the measured end-to-end runs (single thread; item-by-item OPRF at the time)', '',
          '| n_bar_O | aggregate meas/pred (s) | PSI eval meas/pred | sanitize meas/pred | O OPRF self meas/pred | upload meas/pred | comm meas/pred (MB) | wall (s) |',
          '|---|---|---|---|---|---|---|---|']
    for v in out['validation']:
        me, pr = v['measured'], v['predicted']
        L.append(f"| {v['n_bar_O']} | {me['agg']:.1f} / {pr['agg']:.1f} | {me['psi_eval']:.1f} / {pr['psi_eval']:.1f} | "
                 f"{me['psi_san']:.1f} / {pr['psi_san']:.1f} | {me['oprf_self']:.1f} / {pr['oprf_self']:.1f} | "
                 f"{me['upload']:.2f} / {pr['upload']:.2f} | {v['comm_measured'] / MB:.1f} / {v['comm_predicted'] / MB:.1f} | {v['wall_measured']:.1f} |")
    L += ['', '## Extrapolation', '',
          'O wall 4 cores: predict_run (4 OpenMP threads per party, 4 parallel entries, batched parallel OPRF, release '
          'padding with margin 2). The large runs of reports/large/ are compared with the model by '
          'experiments/eng/model_check.py (reports/eng/model_check.json).', '',
          '| n_bar_O | n_R^max | m | D, alpha | O CPU 1 core (s) | O wall 4 cores (s) | R CPU (s) | comm (MB) | O memory (MB) | aggregate N=2^14 / 2^15 (s) |',
          '|---|---|---|---|---|---|---|---|---|---|']
    for r in out['extrapolation']:
        L.append(f"| {r['n_bar_O']:,} | {r['n_R_max']:,} | {r['m']} | {r['D']}, {r['alpha']} | {r['cpu_O']:.0f} | "
                 f"{r['wall_O_4cores']:.0f} | {r['cpu_R']:.0f} | {r['comm'] / MB:.0f} | "
                 f"{r['mem_O'] / MB:.0f} | {r['agg_N14']:.0f} / {r['agg_N15']:.0f} |")
    L += ['', '### Breakdown at n_bar_O = 1M, n_R^max = %d (single thread, seconds)' % a.nRmax, '']
    r1m = [r for r in out['extrapolation'] if r['n_bar_O'] == 1_000_000 and r['n_R_max'] == a.nRmax][0]
    for k, v in sorted(r1m['t'].items(), key=lambda kv: -kv[1]):
        L.append(f'- {k}: {v:.1f}')
    L += ['', '### O wall time per phase at n_bar_O = 1M, n_R^max = 20,000 (4 cores, seconds)', '']
    r1m20 = [r for r in out['extrapolation'] if r['n_bar_O'] == 1_000_000 and r['n_R_max'] == 20_000][0]
    for k, v in sorted(r1m20['wall_O_4cores_phases'].items(), key=lambda kv: -kv[1]):
        if k.startswith('O.'):
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
