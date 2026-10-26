"""Refresh the committed cost-model inputs on the idle machine (implementation/bench/results/).

1. bench/bench_micro.py at 1 and 4 OpenMP threads: writes micro_t1.json and micro_t4.json in place (now with the
   batched OPRF, the earlier Python OPRF as *_py, and the empty-partition interpolation).
2. O's aggregation at n_bar_O = 106,496 for (threads, parallel entries) = (1, 1), (4, 4) and (1, 4)
   (experiments/large/agg_speed.py): agg_speed.json.  The model uses (1, 1) / (4, 4), the configuration that is run.
3. The PSI interpolation measured at the actual partition fill of each large-run size (experiments/large/
   micro_large.polys_at_fill) against the model's fill term (cost_model.polys_time): polys_fill.json.
4. bench/cost_model.py: cost_model.json and cost_model.md.

    python experiments/eng/refresh_cost_model.py [--skip_micro]
"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, '..', '..'))
IMPL = os.path.join(ROOT, 'implementation')
RES = os.path.join(IMPL, 'bench', 'results')
LARGE = os.path.join(ROOT, 'experiments', 'large')
sys.path.insert(0, LARGE)
sys.path.insert(0, os.path.join(IMPL, 'bench'))
SIZES = [106_496, 303_104, 1_007_616, 311_296]

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--skip_micro', action='store_true')
    a = ap.parse_args()
    if not a.skip_micro:
        for th in (1, 4):
            env = dict(os.environ, OMP_NUM_THREADS=str(th))
            subprocess.run([sys.executable, os.path.join(IMPL, 'bench', 'bench_micro.py'), '--threads', str(th)],
                           check=True, env=env, cwd=IMPL, stdout=subprocess.DEVNULL)
            print(f'bench_micro --threads {th}: done', flush=True)
        runs = []
        for th, w in ((1, 1), (4, 4), (1, 4)):
            env = dict(os.environ, OMP_NUM_THREADS=str(th))
            out = subprocess.run([sys.executable, os.path.join(LARGE, 'agg_speed.py'), '--threads', str(th),
                                  '--workers', str(w)], check=True, env=env, capture_output=True, text=True)
            runs.append(json.loads(out.stdout.strip().splitlines()[-1]))
            print(runs[-1], flush=True)
        with open(os.path.join(RES, 'agg_speed.json'), 'w') as fh:
            json.dump(dict(stamp=time.strftime('%Y-%m-%d %H:%M'), source='experiments/large/agg_speed.py', runs=runs),
                      fh, indent=1)
    import micro_large as ML  # noqa: E402
    import cost_model as CM  # noqa: E402
    mu1, mu4 = CM.load('micro_t1.json'), CM.load('micro_t4.json')
    rows = []
    for n in SIZES:
        for th, mu in ((1, mu1), (4, mu4)):
            r = ML.polys_at_fill(n, th)
            r['model_fill'] = CM.polys_time(n, mu1, fill=True, mu4=mu4 if th == 4 else None)
            r['model_full_partitions'] = CM.polys_time(n, mu1, fill=False, mu4=mu4 if th == 4 else None)
            r['ratio_measured_over_model_fill'] = r['seconds'] / r['model_fill']
            rows.append(r)
            print(r, flush=True)
    with open(os.path.join(RES, 'polys_fill.json'), 'w') as fh:
        json.dump(dict(stamp=time.strftime('%Y-%m-%d %H:%M'), rows=rows), fh, indent=1)
    subprocess.run([sys.executable, os.path.join(IMPL, 'bench', 'cost_model.py'), '--nRmax', '2000'], check=True,
                   cwd=IMPL, stdout=subprocess.DEVNULL)
    print('cost_model.py: done', flush=True)
