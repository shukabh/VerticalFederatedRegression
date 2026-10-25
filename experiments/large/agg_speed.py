"""Aggregation wall time at n_bar_O = 106,496 in one process, for a given (OpenMP threads, parallel entries), on random
data with real keys and ciphertexts (the same AG.aggregate call as party O).  Used by micro_large.py to measure the
speedup of the run configuration (4 OpenMP threads and 4 parallel entries) over one thread.

    OMP_NUM_THREADS=t python experiments/large/agg_speed.py --threads t --workers w
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lcommon as lc  # noqa: E402
from vfr import aggregate as AG  # noqa: E402
from vfr import common as C  # noqa: E402
from vfr import dp  # noqa: E402

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--threads', type=int, default=1)
    ap.add_argument('--workers', type=int, default=1)
    ap.add_argument('--n_bar', type=int, default=106496)
    a = ap.parse_args()
    from vfr import vfr_native as vn
    vn.set_threads(a.threads)
    b = lc.bounds_for('std')
    ag = dp.Agreement(n_R_max=20000, n_bar_O=a.n_bar, d_R=4, d_O=6, eps_run=1.0, delta_run=1e-9, **b, **dp.setting('I'))
    ck = C.make_ckks(C.CKKSParams())
    qs = list(ck.qs)
    cal = dp.calibrate(ag, ck.N, qs[2], qs[3], qs[0] * qs[1], C.gadget_constants(ck))
    crs = os.urandom(32)
    sR, sO = ck.sample_secret(), ck.sample_secret()
    km = ck.make_rotkeys([ck.rotkey_share(sR, crs), ck.rotkey_share(sO, crs)], crs)
    Ns = ck.N // 2
    m = -(-a.n_bar // Ns)
    OmU = C.dd(cal['Omega_U'])
    rng = np.random.default_rng(0)
    U = []
    for v in range(6):
        row = []
        for h in range(m):
            seed = os.urandom(32)
            row.append(ck.convert(ck.encrypt_seeded(sR, rng.uniform(-1, 1, Ns), OmU, seed), seed, sO, OmU))
        U.append(row)
    XO = rng.uniform(-0.5, 0.5, (a.n_bar, 6))
    yt = rng.uniform(-1, 1, a.n_bar)
    ents = C.entries(4, 6, cal['joint'])
    t0 = time.time()
    AG.aggregate(ck, km, U, XO, yt, ents, [1, 2], 'A', cal['Omega_V'], cal['Omega_u'], a.n_bar, workers=a.workers)
    print(json.dumps(dict(threads=a.threads, workers=a.workers, n_bar=a.n_bar, seconds=time.time() - t0)))
