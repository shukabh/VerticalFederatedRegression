"""Microbenchmarks at full parameters (N = 2^14, Q = 60+3x59 bits, P = 2x60 bits, dnum = 2; BFV N_B = 2^14,
7 x 60-bit chain, t ~ 2^32; ristretto255 OPRF).  Single thread by default (--threads to change OpenFHE's
OpenMP threads).  Writes bench/results/micro_t{threads}.json.

    python bench/bench_micro.py [--threads 1] [--reps 5]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, '..')))

from vfr import common as C  # noqa: E402
from vfr import dp, oprf, psi  # noqa: E402
from vfr import sampler as S  # noqa: E402
from vfr import vfr_native as vn  # noqa: E402


def timeit(fn, reps):
    ts = []
    out = None
    for _ in range(reps):
        t = time.perf_counter()
        out = fn()
        ts.append(time.perf_counter() - t)
    return float(np.median(ts)), out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--threads', type=int, default=1)
    ap.add_argument('--reps', type=int, default=5)
    ap.add_argument('--logN', type=int, default=14, help='CKKS ring; BFV/OPRF parts only at 14')
    a = ap.parse_args()
    vn.set_threads(a.threads)
    R = {}
    reps = a.reps
    cp = C.CKKSParams(logN=a.logN)
    ck = C.make_ckks(cp)
    N, Ns = ck.N, ck.N // 2
    qs = list(ck.qs)
    q1, q2 = qs[2], qs[3]
    ag = dp.Agreement(n_R_max=2000, n_bar_O=10 ** 6, d_R=4, d_O=6, B_R=5.59, B_O=6.24, B_y=2.89)
    cal = dp.calibrate(ag, N, q1, q2, qs[0] * qs[1], C.gadget_constants(ck))
    OmU, Omu, OmV = C.dd(q2), C.dd(q1), C.dd(cal['Omega_V'])
    rng = np.random.default_rng(0)
    crs = os.urandom(32)
    sR, sO = ck.sample_secret(), ck.sample_secret()
    R['ckks_pk_share'], p0 = timeit(lambda: ck.pk_share(sR, crs), reps)
    R['ckks_rotkey_share'], bR = timeit(lambda: ck.rotkey_share(sR, crs), max(1, reps // 2))
    bO = ck.rotkey_share(sO, crs)
    R['ckks_make_rotkeys'], km = timeit(lambda: ck.make_rotkeys([bR, bO], crs), max(1, reps // 2))
    pk = ck.make_pk([p0, ck.pk_share(sO, crs)], crs)
    u = (rng.uniform(size=Ns) < 0.3).astype(float)
    V = rng.uniform(-38, 38, Ns)
    R['ckks_encode_dd'], _ = timeit(lambda: ck.encode_plain(V, OmV, 4), reps)
    seed = os.urandom(32)
    R['ckks_encrypt_seeded_R'], c0 = timeit(lambda: ck.encrypt_seeded(sR, u, OmU, seed), reps)
    R['ckks_convert_O'], ct = timeit(lambda: ck.convert(c0, seed, sO, OmU), reps)
    pt = ck.encode_plain(V, OmV, 4)
    R['ckks_mul_plain_L0'], prod = timeit(lambda: ck.mul_plain(ct, pt), reps)
    R['ckks_add_L0'], _ = timeit(lambda: ck.add(prod, prod), reps)
    R['ckks_rescale_det_L0'], r1 = timeit(lambda: ck.rescale(prod, False), reps)
    R['ckks_rescale_stoch_L0'], _ = timeit(lambda: ck.rescale(prod, True), reps)
    R['ckks_rotate_L1'], _ = timeit(lambda: ck.rotate_once(r1, km, 0), reps)
    R['ckks_rotate_L0'], _ = timeit(lambda: ck.rotate_once(prod, km, 0), reps)
    R['ckks_evalsum_L1'], T = timeit(lambda: ck.eval_sum(r1, km), max(1, reps // 2))
    R['ckks_evalsum_L0'], _ = timeit(lambda: ck.eval_sum(prod, km), max(1, reps // 2))
    mk = np.zeros(Ns); mk[3] = 1
    pm = ck.encode_plain(mk, Omu, 3)
    R['ckks_mul_plain_L1'], tm = timeit(lambda: ck.mul_plain(T, pm), reps)
    R['ckks_rescale_det_L1'], M = timeit(lambda: ck.rescale(tm, False), reps)
    sig2 = cal['releases']['2']['sigma_c2']
    srng = S.CSPRNG()
    # the protocol's budgeted sampler returns the residues directly (DEVIATIONS A24): dg_sample_N includes them
    R['dg_sample_N'], er = timeit(lambda: S.sample_residues(sig2, N, qs[:2], srng), max(1, reps // 2))
    R['dg_residues_N'] = 0.0
    R['dg_sample_plain_N'], eta = timeit(lambda: S.sample_vector(sig2, N, srng), max(1, reps // 2))
    R['dg_residues_plain_N'], _ = timeit(lambda: S.residues(eta, qs[:2]), reps)
    R['dg_log2_sigma_c'] = 0.5 * math.log2(sig2)
    R['ckks_rerand_partial_dec'], (c1p, h) = timeit(lambda: ck.rerand_partial_decrypt(M, pk, sO, er), reps)
    R['ckks_combine_R'], (hi, lo) = timeit(lambda: ck.combine(c1p, h, sR), reps)
    R['ckks_decode_dd_R'], _ = timeit(lambda: ck.decode_i128(hi, lo, OmV), reps)
    # ------------------------------------------------------------------ sizes
    R['size_ckks_ct_L0_bytes'] = 2 * 4 * N * 8
    R['size_seeded_upload_bytes'] = 4 * N * 8 + 32
    R['size_release_bytes'] = 2 * 2 * N * 8
    R['size_rotkey_share_bytes'] = int(bR.nbytes)
    if a.logN != 14:
        R['logN'] = a.logN
        os.makedirs(os.path.join(HERE, 'results'), exist_ok=True)
        with open(os.path.join(HERE, 'results', f'micro_t{a.threads}_logN{a.logN}.json'), 'w') as fh:
            json.dump(R, fh, indent=1)
        for kk, v in R.items():
            print(f'{kk:36s} {v}')
        return
    # ------------------------------------------------------------------ OPRF
    # O's side as party O runs it: batched in the native layer, OpenMP over the items (a.threads); the earlier
    # item-by-item Python loop is kept as *_py (single-threaded whatever a.threads); R's side is item by item
    k = oprf.keygen()
    xs = [str(i).encode() for i in range(2000)]
    xb = [str(i).encode() for i in range(40000)]
    t = time.perf_counter(); oprf.evaluate_batch(k, xb, a.threads); R['oprf_server_evaluate_per_item'] = (time.perf_counter() - t) / len(xb)
    t = time.perf_counter(); [oprf.evaluate(k, x) for x in xs]; R['oprf_server_evaluate_per_item_py'] = (time.perf_counter() - t) / len(xs)
    t = time.perf_counter(); bl = [oprf.blind(x) for x in xs]; R['oprf_client_blind_per_item'] = (time.perf_counter() - t) / len(xs)
    t = time.perf_counter(); ev = [oprf.blind_evaluate(k, b[1]) for b in bl]; R['oprf_server_blind_eval_per_item_py'] = (time.perf_counter() - t) / len(xs)
    blob = b''.join(b[1] for b in bl) * 10
    t = time.perf_counter(); oprf.blind_evaluate_batch(k, blob, a.threads); R['oprf_server_blind_eval_per_item'] = (time.perf_counter() - t) / (len(blob) // 32)
    t = time.perf_counter(); [oprf.finalize(x, b[0], e) for x, b, e in zip(xs, bl, ev)]; R['oprf_client_finalize_per_item'] = (time.perf_counter() - t) / len(xs)
    # ------------------------------------------------------------------ BFV PSI
    pp = psi.PSIParams()
    D, alpha, _ = pp.sizes(10 ** 6)
    W = pp.windows(D)
    R['psi_D'], R['psi_alpha_1M'], R['psi_W'] = D, alpha, W
    bf = vn.BFVCore(14, pp.t, pp.bfv_depth, True)
    keys = bf.keygen()
    pkb, rkb = bf.public_bytes(keys)
    bo = vn.BFVCore(bf.context_bytes())
    bo.load_public(pkb, rkb)
    R['size_bfv_pk_relin_bytes'] = len(pkb) + len(rkb)
    q = rng.integers(0, pp.t - 1, pp.N_B, dtype=np.uint64)
    wv = psi.window_values(q, W, pp.t)
    R['bfv_encrypt'], w0 = timeit(lambda: bf.encrypt_sym(keys, wv[0]), reps)
    wins = [bf.encrypt_sym(keys, w) for w in wv]
    R['size_bfv_query_ct_bytes'] = len(w0)
    R['bfv_compute_powers_D64'], pw = timeit(lambda: bo.compute_powers(wins, D), 1)
    Pc = rng.integers(0, pp.t, (D + 1, pp.N_B), dtype=np.uint64)
    Qc = rng.integers(0, pp.t, (D + 1, pp.N_B), dtype=np.uint64)
    R['bfv_eval_partition_D64'], (Sct, Qct) = timeit(lambda: bo.eval_partition(pw, Pc, Qc), max(1, reps // 2))
    fl = pp.flood_sigma2()
    keep = bo.moduli()[:pp.reply_towers]
    R['bfv_flood_sample_N'], fe = timeit(lambda: S.sample_residues(fl, pp.N_B, keep, srng), max(1, reps // 2))
    R['bfv_sanitize_excl_sampling'], rep = timeit(lambda: bo.sanitize(Sct, pp.reply_towers, fe), reps)
    R['size_bfv_reply_ct_bytes'] = len(rep)
    R['bfv_decrypt'], _ = timeit(lambda: bf.decrypt(keys, rep), reps)
    # interpolation at the 1M load: (alpha, N_bins, D) items
    xs_ = rng.integers(0, pp.t - 1, (alpha, pp.N_B, D), dtype=np.uint64)
    ys_ = rng.integers(0, pp.t, (alpha, pp.N_B, D), dtype=np.uint64)
    cnt = np.full((alpha, pp.N_B), D, dtype=np.int32)
    R['psi_build_polys_1M_shape'], _ = timeit(lambda: vn.psi_build_polys(xs_, ys_, cnt, pp.t), 1)
    # the same shape with empty partitions: the membership polynomials (D dummy roots, always degree D) without
    # any label interpolation; the cost model splits the full-partition time into these two parts (partition fill)
    cnt0 = np.zeros((alpha, pp.N_B), dtype=np.int32)
    R['psi_build_polys_1M_shape_empty'], _ = timeit(lambda: vn.psi_build_polys(xs_, ys_, cnt0, pp.t), 1)
    R['threads'] = a.threads
    R['stamp'] = time.strftime('%Y-%m-%d %H:%M')
    R['calibration_1M'] = dict(log2_Omega_f=cal['log2_Omega_f'], log2_B_tot={s: math.log2(v['B_tot']) for s, v in cal['releases'].items()},
                               sigma_slot={s: v['sigma_slot'] for s, v in cal['releases'].items()})
    os.makedirs(os.path.join(HERE, 'results'), exist_ok=True)
    with open(os.path.join(HERE, 'results', f'micro_t{a.threads}.json'), 'w') as fh:
        json.dump(R, fh, indent=1)
    for kk, v in R.items():
        print(f'{kk:36s} {v}')


if __name__ == '__main__':
    main()
