"""Re-measure the cost model's inputs on the idle machine (diagnosis of the stage-1 cost gate).

1. bench/bench_micro.py at 1 and 4 threads, unchanged; its outputs (which it writes to implementation/bench/results/
   micro_t{1,4}.json) are moved to reports/large/micro_t{1,4}_remeasured.json and the committed files are restored, so
   the published model's inputs stay as they are.
2. O's PSI interpolation (vn.psi_build_polys) at the partition fill of each evaluation size: bin loads drawn from
   Binomial(n_bar_O h, 1/N_bins), each bin filled partition by partition in order (as psi.bin_items), F N_bins slots.
   The published model scales one measurement at full partitions (every partition holds D items) by alpha D^2, which
   is the worst case; the label interpolation costs O(n_a^2) in the actual count n_a of a partition.
3. The aggregation at n_bar_O = 106,496 at (1 thread, 1 entry), (4 OpenMP threads, 4 parallel entries: the run
   configuration) and (1 thread, 4 entries: the configuration of the published speedup), agg_speed.py.

    python experiments/large/micro_large.py [--skip_bench]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lcommon as lc  # noqa: E402

IMPL = os.path.join(lc.ROOT, 'implementation')
SIZES = [106_496, 303_104, 1_007_616, 311_296]


def polys_at_fill(n_bar_O, threads, reps=2, seed=0):
    from vfr import psi
    from vfr import vfr_native as vn
    vn.set_threads(threads)
    pp = psi.PSIParams()
    D, alpha, B = pp.sizes(n_bar_O)
    rng = np.random.default_rng(seed)
    nb, F = pp.n_bins, pp.felts
    load = rng.binomial(n_bar_O * pp.h, 1.0 / nb, size=nb)       # items per bin (distinct bins per item ignored)
    load = np.minimum(load, alpha * D)
    cnt1 = np.stack([np.clip(load - a * D, 0, D) for a in range(alpha)])          # (alpha, nb)
    cnt = np.concatenate([cnt1] * F, axis=1).astype(np.int32)                      # (alpha, F nb)
    xs = np.zeros((alpha, F * nb, D), dtype=np.uint64)
    ys = rng.integers(0, pp.t, (alpha, F * nb, D), dtype=np.uint64)
    # distinct values per (partition, slot): a random start plus distinct offsets
    xs[:] = (rng.integers(0, pp.t - 1 - D, (alpha, F * nb, 1), dtype=np.uint64) + np.arange(D, dtype=np.uint64)[None, None, :])
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        vn.psi_build_polys(xs, ys, cnt, pp.t)
        ts.append(time.perf_counter() - t0)
    return dict(n_bar_O=n_bar_O, threads=threads, D=D, alpha=alpha, B=B, mean_load=float(load.mean()),
                fill=float(cnt1.sum() / (alpha * D * nb)), seconds=float(np.median(ts)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--skip_bench', action='store_true')
    a = ap.parse_args()
    res = dict(stamp=time.strftime('%Y-%m-%d %H:%M'), resources=lc.resources())
    if not a.skip_bench:
        for th in (1, 4):
            dst = os.path.join(IMPL, 'bench', 'results', f'micro_t{th}.json')
            keep = dst + '.published'
            shutil.copy(dst, keep)
            try:
                env = dict(os.environ, OMP_NUM_THREADS=str(th))
                subprocess.run([sys.executable, os.path.join(IMPL, 'bench', 'bench_micro.py'), '--threads', str(th)],
                               check=True, env=env, cwd=IMPL)
                shutil.move(dst, os.path.join(lc.REPORT, f'micro_t{th}_remeasured.json'))
            finally:
                shutil.move(keep, dst)
            print(f'bench_micro --threads {th}: done', flush=True)
    if not a.skip_bench:
        res['aggregate'] = []
        for th, w in ((1, 1), (4, 4), (1, 4)):
            env = dict(os.environ, OMP_NUM_THREADS=str(th))
            out = subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'agg_speed.py'),
                                  '--threads', str(th), '--workers', str(w)], check=True, env=env, capture_output=True, text=True)
            res['aggregate'].append(json.loads(out.stdout.strip().splitlines()[-1]))
            print(res['aggregate'][-1], flush=True)
    res['polys_at_fill'] = [polys_at_fill(n, th) for n in SIZES for th in (1, 4)]
    for r in res['polys_at_fill']:
        print(r, flush=True)
    lc.save_json(res, os.path.join(lc.REPORT, 'micro_large.json'))


if __name__ == '__main__':
    main()
