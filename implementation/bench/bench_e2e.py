"""End-to-end runs (two processes over a socket, DP noise on, production path) at small and medium register
sizes; per-phase wall time at each party and the bytes of every message type.  Never run at >= 100k here.

    python bench/bench_e2e.py --sizes 1024 4096 16384 32768 65536 --nRmax 2000 --threads 4 [--variant A]
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, '..')))

from vfr import common as C  # noqa: E402
from vfr import data, dp, protocol  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sizes', type=int, nargs='+', default=[1024, 4096, 16384])
    ap.add_argument('--nRmax', type=int, default=2000)
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--variant', default='A')
    ap.add_argument('--workers', type=int, default=1, help='parallel entries in O\'s aggregation')
    ap.add_argument('--tag', default='')
    a = ap.parse_args()
    out = []
    for nbar in a.sizes:
        if nbar >= 100_000:
            raise SystemExit('large runs are deliberately not supported here')
        n_scope = int(0.9 * nbar)
        n_R = min(a.nRmax - 100, n_scope)
        n = min(n_R, max(50, nbar // 10))
        dR_, dO_, truth, b = data.synthetic(nbar, n_scope, n_R, n, seed=nbar)
        ag = dp.Agreement(n_R_max=a.nRmax, n_bar_O=nbar, d_R=4, d_O=6, variant=a.variant, **b)
        wd = os.path.join(HERE, 'results', 'work')
        os.makedirs(wd, exist_ok=True)
        t = time.time()
        omp = dict(omp_threads=a.threads)
        R, O = protocol.run(ag, C.CKKSParams(), dR_, dO_, dict(threads=a.threads, Bboot=1000, **omp),
                            dict(threads=a.threads, entry_workers=a.workers, **omp,
                                 ledger_path=os.path.join(wd, f'ledger_{nbar}.json')))
        wall = time.time() - t
        ok = set(R['I']) == truth['matched']
        rec = dict(n_bar_O=nbar, n_R_max=a.nRmax, n=n, threads=a.threads, workers=a.workers, variant=a.variant, wall=wall, psi_exact=ok,
                   O=O['timings'], R=R['timings'], messages=R['summary'],
                   log2_Omega_f=R['calibration']['log2_Omega_f'],
                   sigma=R['sigma'], psi=O['debug']['psi'])
        out.append(rec)
        print(json.dumps(dict(n_bar_O=nbar, wall=round(wall, 1), psi_exact=ok,
                              aggregate=round(O['timings']['aggregate'], 2)), default=float), flush=True)
    os.makedirs(os.path.join(HERE, 'results'), exist_ok=True)
    fn = os.path.join(HERE, 'results', f'e2e_{a.variant}_t{a.threads}_w{a.workers}{a.tag}.json')
    with open(fn, 'w') as fh:
        json.dump(out, fh, indent=1, default=float)
    print('wrote', fn)


if __name__ == '__main__':
    main()
