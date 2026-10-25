"""Check item #30: the coverage repeat, in plaintext simulation at the exact settings of the earlier encrypted runs.

The earlier encrypted end-to-end runs (implementation/bench/utility_runs.py, results in
implementation/bench/results/utility_*.json) used the previous statistical setting (f_1 = 0.1, union-bound radius,
theta* = 2.5, kappa_Psi = 2, alpha_L = alpha_U = alpha/20), delta = 1e-9 and Bboot = 2,000 antithetic pairs:

  synthetic  the simulation population, 20,000 matched records, register 24,000 rows (padded to 24,576), cohort
             21,000, seed 5; eps = 4 (kappa = 40) and eps = 8 (kappa = 48); 20 runs each; lambda = 0 in every run;
             Wald coverage of theta_lambda 209/220 and 204/220;
  protein    19,206 matched records (seed 0 split), standardize / winsorize |z| <= 2 / block unit norm, eps = 1, 4,
             kappa = 40; 5 runs each; coverage 50, 49, 48, 46, 47 and 46 of 50.

Here the same data are rebuilt with the same code (utility_runs.prepare_synthetic / prepare_real and
vfr.data.plaintext_stats), the noise s.d. are the ones the encrypted runs used (read from those JSON files), and each
setting gets --reps independent releases of the plaintext mechanism (stage 1, stage 2, gate, stage2_infer).  For
each setting: pooled Wald coverage of theta_lambda with a release-clustered 95% interval and the naive Wilson
interval, per-coefficient coverage, the standardized errors, and the probability that a batch of encrypted-run size
(20 or 5 releases) drawn from the plaintext releases covers no more coefficients than the encrypted batch did.

    python experiments/v31_more/coverage_repeat.py [--reps 10000] [--workers 4]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mcommon as mc  # noqa: E402
from mcommon import ev  # noqa: E402

sys.path.insert(0, os.path.join(mc.ROOT, 'implementation', 'bench'))
import utility_runs as U  # noqa: E402
from vfr import data  # noqa: E402

RES = os.path.join(mc.ROOT, 'implementation', 'bench', 'results')
PREPROCS = (('zscore', 'utility_real_protein.json'), ('winsor2', 'utility_real_protein_winsor2.json'),
            ('blocknorm', 'utility_real_protein_blocknorm.json'))


def enc_counts(R):
    rs = [r for r in R['runs'] if 'covered' in r]
    return int(sum(sum(r['covered']) for r in rs)), int(sum(len(r['covered']) for r in rs)), len(rs)


def batch_prob(cov, runs, observed, B=200000, seed=0):
    """P(a batch of `runs` releases drawn with replacement from the plaintext releases covers <= observed)."""
    ok = cov[~np.isnan(cov).any(axis=1)]
    per = ok.sum(axis=1)
    rng = np.random.default_rng(seed)
    tot = per[rng.integers(0, len(per), size=(B, runs))].sum(axis=1)
    return float(np.mean(tot <= observed))


def one_setting(tag, G, c, yy, n, dA, sig, eps, kappa, B_y, enc, reps, workers, seed):
    st = mc.setting('prev')
    cal = dict(sigma1=sig['1'], sigma2=sig['2'], sigma_yy=sig.get('y', sig['2']))
    t0 = time.time()
    o = mc.releases_parallel(G, c, yy, n, dA, cal, st, B_y, reps, seed, workers=workers)
    cs_ = mc.coverage_summary(o['covered'])
    ok = ~o['refused']
    z = (o['theta'][ok] - o['theta_lam'][ok]) / (o['half_theta'][ok] / mc.solve.Z(1 - (st['alpha'] - st['alpha_L'] - st['alpha_U']) / 2))
    k, N, runs = enc
    p = G.shape[0]
    rec = dict(tag=tag, eps=eps, kappa=kappa, n=int(n), p=int(p), sigma=sig, reps=reps,
               lam_q=np.quantile(o['lam'], [0.025, 0.5, 0.975]).tolist(), ridge_on=float(np.mean(o['lam'] > 0)),
               refused=float(o['refused'].mean()), coverage=cs_,
               z=dict(mean=float(z.mean()), sd=float(z.std()), per_coef_sd=z.std(axis=0).tolist(),
                      per_coef_mean=z.mean(axis=0).tolist()),
               encrypted=dict(covered=k, total=N, runs=runs, coverage=k / N, wilson=mc.wilson(k, N),
                              clopper_pearson=mc.clopper_pearson(k, N),
                              p_batch_le_observed=batch_prob(o['covered'], runs, k)),
               seconds=time.time() - t0)
    print(f"[{tag}] eps={eps}: coverage {cs_['coverage']:.4f} clustered CI ({cs_['clustered'][0]:.4f}, {cs_['clustered'][1]:.4f}) "
          f"per-coef {cs_['per_coef_min']:.3f}-{cs_['per_coef_max']:.3f}; z sd {z.std():.3f}; ridge on {rec['ridge_on']:.2f}; "
          f"encrypted {k}/{N}, P(batch <= obs) = {rec['encrypted']['p_batch_le_observed']:.3f}  ({rec['seconds']:.0f}s)", flush=True)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reps', type=int, default=10000)
    ap.add_argument('--workers', type=int, default=4)
    a = ap.parse_args()
    out = dict(setting=mc.setting('prev'), delta=mc.DELTA, Bboot=mc.BBOOT, reps=a.reps, target=mc.TARGET, results=[])
    # synthetic, ridge off
    D = json.load(open(os.path.join(RES, 'utility_synthetic.json')))
    m = D['meta']
    dR_, dO_, b, meta = U.prepare_synthetic(m['n_bar_O'], m['n_register'], m['n_cohort'], m['n_matched'], m['seed'])
    assert all(abs(b[k] - m['bounds'][k]) < 1e-12 for k in b)
    ps = data.plaintext_stats(dR_, dO_, b['B_R'], b['B_O'], b['B_y'])
    for e, R in D['results'].items():
        eps = float(e)
        chk = ev._calibrate_plain(eps, mc.DELTA, b['B_R'], b['B_O'], b['B_y'], R['kappa'], f1=0.1)
        assert abs(chk['sigma2'] / R['sigma']['2'] - 1) < 1e-3, (chk['sigma2'], R['sigma'])
        out['results'].append(one_setting('synthetic', ps['G'], ps['c'], ps['yy'], ps['n'], 5, R['sigma'], eps, R['kappa'],
                                          b['B_y'], enc_counts(R), a.reps, a.workers, [mc.cs.BASE_SEED, 30, int(eps)]))
        out['results'][-1]['data'] = dict(meta)
        ev.save_json(out, os.path.join(mc.REPORT, 'coverage_repeat.json'))
    # protein, ridge on
    for ip, (pp, fn) in enumerate(PREPROCS):
        D = json.load(open(os.path.join(RES, fn)))
        m = D['meta']
        dR_, dO_, b, meta, test = U.prepare_real('protein', seed=m['seed'], preproc=pp)
        assert all(abs(b[k] - m['bounds'][k]) < 1e-12 for k in b) and meta['n_matched'] == m['n_matched']
        ps = data.plaintext_stats(dR_, dO_, b['B_R'], b['B_O'], b['B_y'])
        for e, R in D['results'].items():
            eps = float(e)
            out['results'].append(one_setting(f'protein_{pp}', ps['G'], ps['c'], ps['yy'], ps['n'], 5, R['sigma'], eps,
                                              R['kappa'], b['B_y'], enc_counts(R), a.reps, a.workers,
                                              [mc.cs.BASE_SEED, 31, int(eps), ip]))
            out['results'][-1]['data'] = dict(meta)
            ev.save_json(out, os.path.join(mc.REPORT, 'coverage_repeat.json'))
    print('wrote', os.path.join(mc.REPORT, 'coverage_repeat.json'))


if __name__ == '__main__':
    main()
