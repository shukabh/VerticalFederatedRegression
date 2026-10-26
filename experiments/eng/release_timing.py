"""Padded release timing: how often does a release overrun its deadline?

For the evaluation's largest sigma_c^2 (n_bar_O = 1,007,616, settings I and P, eps = 1 and 4), O's deadline is
computed as in party_o.release_deadline (margin 2 x [worst of 3 timed dummy samplings + one re-randomized partial
decryption]), then K releases' work (exact sampling of N = 2^14 values, residues, re-randomized partial decryption
of a ciphertext at q0, with eta) is timed against it, on the idle machine and, as a stress case, with four CPU-bound
processes competing for the four cores (the deadline still calibrated idle).  Writes reports/eng/release_timing.json.

    python experiments/eng/release_timing.py [--K 1000] [--K_loaded 200]
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'experiments', 'large'))
import lcommon as lc  # noqa: E402
from vfr import common as C  # noqa: E402
from vfr import dp  # noqa: E402
from vfr import sampler as S  # noqa: E402
from vfr.party_o import calibrate_release_time  # noqa: E402


def hog(stop_at):
    x = 0
    while time.time() < stop_at:
        x += 1


def q0_ciphertext(ck, cal):
    sR, sO = ck.sample_secret(), ck.sample_secret()
    crs = os.urandom(32)
    pk = ck.make_pk([ck.pk_share(sR, crs), ck.pk_share(sO, crs)], crs)
    seed = os.urandom(32)
    OmU = C.dd(cal['Omega_U'])
    ct = ck.convert(ck.encrypt_seeded(sR, np.zeros(ck.N // 2), OmU, seed), seed, sO, OmU)
    ct = ck.rescale(ck.rescale(ct, False), False)
    assert ck.ctxt_towers(ct) == 2
    return ct, pk, sO


def time_releases(ck, s2, ct, pk, sO, q0t, K, rng):
    ts = []
    for _ in range(K):
        t0 = time.perf_counter()
        eta = S.residues(S.sample_vector(s2, ck.N, rng), q0t)
        ck.rerand_partial_decrypt(ct, pk, sO, eta)
        ts.append(time.perf_counter() - t0)
    return ts


def summary(ts, deadline):
    a = np.array(ts)
    over = int((a > deadline).sum())
    return dict(K=len(a), mean=float(a.mean()), sd=float(a.std()), max=float(a.max()), p99=float(np.quantile(a, 0.99)),
                max_over_deadline=float(a.max() / deadline), overruns=over,
                # one-sided 95% upper bound on the overrun probability (exact binomial; 3/K when none)
                p_overrun_95=float(1 - 0.05 ** (1 / len(a))) if over == 0 else None,
                sd_margin=float((deadline - a.mean()) / a.std()))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--K', type=int, default=1000)
    ap.add_argument('--K_loaded', type=int, default=200)
    ap.add_argument('--margin', type=float, default=2.0)
    a = ap.parse_args()
    from vfr import vfr_native as vn
    vn.set_threads(4)
    ck = C.make_ckks(C.CKKSParams())
    qs = list(ck.qs)
    q0t = [qs[0], qs[1]]
    s2s = {}
    for st, kind in (('I', 'std'), ('P', 'std'), ('P', 'bn')):
        for eps in (1.0, 4.0):
            ag = dp.Agreement(n_R_max=20000, n_bar_O=1_007_616, d_R=4, d_O=6, eps_run=eps, delta_run=1e-9,
                              **lc.bounds_for(kind), **dp.setting(st))
            cal = dp.calibrate(ag, ck.N, qs[2], qs[3], qs[0] * qs[1], C.gadget_constants(ck))
            for s, r in cal['releases'].items():
                s2s[f'{st}{kind}_e{eps:g}_s{s}'] = int(r['sigma_c2'])
    key = max(s2s, key=s2s.get)
    s2 = s2s[key]
    ct, pk, sO = q0_ciphertext(ck, cal)
    res = dict(stamp=time.strftime('%Y-%m-%d %H:%M'), load_before=os.getloadavg(), margin=a.margin,
               sigma_c2_log2={k: v.bit_length() for k, v in s2s.items()}, sigma_c2_used=key,
               log2_sigma_c=s2.bit_length() / 2)
    c = calibrate_release_time(ck, s2, ct, pk, sO, q0t)
    deadline = a.margin * (c['sample_max'] + c['rerand_pdec'])
    res['calibration'] = c
    res['deadline_s'] = deadline
    rng = S.CSPRNG()
    res['idle'] = summary(time_releases(ck, s2, ct, pk, sO, q0t, a.K, rng), deadline)
    print('idle', res['idle'], flush=True)
    stop = time.time() + 3600
    hogs = [mp.Process(target=hog, args=(stop,)) for _ in range(4)]
    for h in hogs:
        h.start()
    try:
        res['loaded_4_hogs'] = summary(time_releases(ck, s2, ct, pk, sO, q0t, a.K_loaded, rng), deadline)
    finally:
        for h in hogs:
            h.terminate()
    print('loaded', res['loaded_4_hogs'], flush=True)
    os.makedirs(os.path.join(ROOT, 'reports', 'eng'), exist_ok=True)
    with open(os.path.join(ROOT, 'reports', 'eng', 'release_timing.json'), 'w') as fh:
        json.dump(res, fh, indent=1)
