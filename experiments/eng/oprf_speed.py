"""O's OPRF evaluation of its whole register (Link, phase 'oprf_self'): the earlier item-by-item Python loop
(single-threaded) against the native batched evaluation at 1 and 4 OpenMP threads, at n_bar_O = 106,496 and
1,007,616 identifiers (as party O builds them: the register's ids plus 'dummy:' padding).  One timed run each, the
same fresh key for all three; the outputs are compared in full.  Also times the answer to R's n_R^max = 20,000
blinded queries.  Writes reports/eng/oprf_speed.json.

    python experiments/eng/oprf_speed.py [--sizes 106496 1007616]
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'implementation'))
from vfr import oprf  # noqa: E402

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--sizes', type=int, nargs='+', default=[106_496, 1_007_616])
    ap.add_argument('--n_real', type=int, nargs='+', default=[100_000, 1_000_000])
    a = ap.parse_args()
    res = dict(stamp=time.strftime('%Y-%m-%d %H:%M'), cpus=os.cpu_count(), load_before=os.getloadavg(), sizes=[])
    for nbar, nreal in zip(a.sizes, a.n_real):
        ids = [str(i) for i in range(nreal)] + [f'dummy:{j}' for j in range(nbar - nreal)]
        k = oprf.keygen()
        rec = dict(n_bar_O=nbar)
        t0 = time.perf_counter()
        ref = [oprf.evaluate(k, i.encode()) for i in ids]              # the earlier party_o code
        rec['python_1thread_s'] = time.perf_counter() - t0
        for th in (1, 4):
            t0 = time.perf_counter()
            out = oprf.evaluate_batch(k, [i.encode() for i in ids], th)  # the current party_o code
            rec[f'native_{th}threads_s'] = time.perf_counter() - t0
            rec[f'native_{th}threads_identical'] = out == ref
            del out
        rec['speedup_4threads_vs_python'] = rec['python_1thread_s'] / rec['native_4threads_s']
        rec['speedup_4threads_vs_native1'] = rec['native_1threads_s'] / rec['native_4threads_s']
        del ref
        bl = b''.join(oprf.blind(str(i).encode())[1] for i in range(20_000))
        t0 = time.perf_counter()
        ev_ref = b''.join(oprf.blind_evaluate(k, bl[32 * i:32 * i + 32]) for i in range(20_000))
        rec['requests_python_s'] = time.perf_counter() - t0
        t0 = time.perf_counter()
        ev = oprf.blind_evaluate_batch(k, bl, 4)
        rec['requests_native_4threads_s'] = time.perf_counter() - t0
        rec['requests_identical'] = ev == ev_ref
        print(json.dumps(rec), flush=True)
        res['sizes'].append(rec)
    res['load_after'] = os.getloadavg()
    with open(os.path.join(ROOT, 'reports', 'eng', 'oprf_speed.json'), 'w') as fh:
        json.dump(res, fh, indent=1)
