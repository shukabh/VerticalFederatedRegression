"""Encrypted spot check of the plaintext simulation in the ridge-on regime (synthetic population).

The existing encrypted validation (implementation/bench/results/utility*.json) covers protein with the
ridge on and the synthetic population with the ridge off.  This adds the synthetic population with the
stage-1 ridge on: n = 10,000 matched records in a 12,000-row register (padded to 16,384), eps = 2
(n_off = 21.5k), fixed data, a few full encrypted runs of implementation/vfr/protocol.py (OPRF, labeled
PSI, threshold CKKS, two releases with exact discrete-Gaussian noise in O's share, solve).  Each run's
lambda and estimate are compared with a plaintext simulation of the same mechanism on the same data.

    python experiments/v31_eval/encrypted_spot.py [--reps 4] [--eps 2] [--n 10000]
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as cm  # noqa: E402
from vfr import common as VC  # noqa: E402
from vfr import data, dp, protocol  # noqa: E402

SLOTS = 8192


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reps', type=int, default=4)
    ap.add_argument('--eps', type=float, default=2.0)
    ap.add_argument('--n', type=int, default=10000)
    ap.add_argument('--seed', type=int, default=11)
    a = ap.parse_args()
    n_scope = int(1.2 * a.n)
    n_bar = int(math.ceil(n_scope / SLOTS) * SLOTS)
    n_R = a.n + 1000
    pop = data.population()
    dR_, dO_, truth, b = data.synthetic(n_bar, n_scope, n_R, a.n, seed=a.seed, pop=pop)
    ps = data.plaintext_stats(dR_, dO_, b['B_R'], b['B_O'], b['B_y'])
    G, c, yy = ps['G'], ps['c'], ps['yy']
    ag = dp.Agreement(n_R_max=n_R, n_bar_O=n_bar, d_R=4, d_O=6, eps_run=a.eps, delta_run=cm.DELTA,
                      kappa=cm.kappa_for(a.eps), **b)
    runs = []
    for k in range(a.reps):
        with tempfile.TemporaryDirectory() as td:
            t0 = time.time()
            R, O = protocol.run(ag, VC.CKKSParams(), dR_, dO_, dict(omp_threads=1, solve_seed=500 + k),
                                dict(omp_threads=1, ledger_path=os.path.join(td, 'ledger.json')))
            wall = time.time() - t0
        res = R['result'] or {}
        s1 = R['stage1']
        mask = np.ones((11, 11), bool)
        mask[:5, :5] = False
        iu = np.triu_indices(11)
        sel = mask[iu]
        zG2 = ((np.asarray(R['G2']) - G)[iu][sel]) / R['sigma']['2']
        zG1 = ((np.asarray(R['G1']) - G)[iu][sel]) / R['sigma']['1']
        zc = (np.asarray(R['c2']) - c) / R['sigma']['2']
        zz2 = np.concatenate([zG2, zc])
        noise = dict(stage2=dict(n=int(zz2.size), mean=float(zz2.mean()), sd=float(zz2.std()), max_abs=float(np.abs(zz2).max())),
                     stage1=dict(n=int(zG1.size), mean=float(zG1.mean()), sd=float(zG1.std()), max_abs=float(np.abs(zG1).max())),
                     A_exact=bool(np.allclose(np.asarray(R['G2'])[:5, :5], G[:5, :5], rtol=1e-9, atol=1e-6)),
                     z_stage2=zz2.tolist(), z_stage1=zG1.tolist())
        runs.append(dict(k=k, wall=wall, n=R['n'], lam=float(s1['lam']), kind=s1['kind'], sigma=R['sigma'],
                         refused=bool(res.get('refused', True)), theta=np.asarray(res.get('theta')).tolist(),
                         half_theta=np.asarray(res.get('half_theta')).tolist(),
                         theta_lam=np.linalg.solve(G + s1['lam'] * np.asarray(s1['Psi']), c).tolist(), noise=noise))
        print(f"run {k}: {wall:.0f}s lam={s1['lam']:.0f} ({s1['kind']}) refused={runs[-1]['refused']} "
              f"noise z stage2 sd={noise['stage2']['sd']:.3f} max={noise['stage2']['max_abs']:.2f}; stage1 sd={noise['stage1']['sd']:.3f} "
              f"max={noise['stage1']['max_abs']:.2f}; A exact={noise['A_exact']}", flush=True)
    sig = runs[0]['sigma']
    cal = dict(sigma1=sig['1'], sigma2=sig['2'], sigma_yy=sig.get('y', sig['2']))
    rng = np.random.default_rng(3)
    S = 4000
    s1 = cm.stage1_batch(G, 5, cal, rng, S)
    E, f, g = cm.stage2_noise(11, 5, cal, rng, S)
    st = cm.stage2_solve(G[None] + E, c[None] + f, s1['lam'], s1['Psi'], cal['sigma2'])
    th = st['theta'][~st['refused']]
    mu, sd = th.mean(0), th.std(0)
    lq = np.quantile(s1['lam'], [0.025, 0.5, 0.975])
    for r in runs:
        Psi_r = np.eye(11) if r['kind'] == 'I' else cm.pi_O(11, 5)
        Ec, fc, _ = cm.stage2_noise(11, 5, cal, rng, S)
        stc = cm.stage2_solve(G[None] + Ec, c[None] + fc, np.full(S, r['lam']), np.broadcast_to(Psi_r, (S, 11, 11)), cal['sigma2'])
        r['z_vs_sim_given_lambda'] = ((np.array(r['theta']) - stc['theta'].mean(0)) / stc['theta'].std(0)).tolist()
        r['z_vs_sim'] = ((np.array(r['theta']) - mu) / sd).tolist()
        r['lam_in_sim_95'] = bool(lq[0] <= r['lam'] <= lq[2])
        r['covered'] = (np.abs(np.array(r['theta']) - np.array(r['theta_lam'])) <= np.array(r['half_theta'])).tolist()
    cal_impl = cm.calibrate(a.eps, b['B_R'], b['B_O'], b['B_y'], 4, 6, n_bar_O=n_bar)
    out = dict(setting=dict(n=a.n, n_scope=n_scope, n_bar_O=n_bar, n_R=n_R, eps=a.eps, delta=cm.DELTA,
                            kappa=cm.kappa_for(a.eps), seed=a.seed),
               sigma_encrypted_run=sig, sigma_eval_calibrate=dict(sigma1=cal_impl['sigma1'], sigma2=cal_impl['sigma2']),
               sim=dict(draws=S, lam_q=lq.tolist(), refused=float(st['refused'].mean()), theta_mean=mu.tolist(),
                        theta_sd=sd.tolist()),
               runs=runs,
               max_abs_z=float(max(abs(z) for r in runs for z in r['z_vs_sim'])),
               max_abs_z_given_lambda=float(max(abs(z) for r in runs for z in r['z_vs_sim_given_lambda'])),
               noise_pooled=dict((st, dict(n=int(len(v)), mean=float(np.mean(v)), sd=float(np.std(v)), max_abs=float(np.max(np.abs(v)))))
                                 for st, v in (('stage2', [z for r in runs for z in r['noise']['z_stage2']]),
                                               ('stage1', [z for r in runs for z in r['noise']['z_stage1']]))),
               coverage=f"{sum(sum(r['covered']) for r in runs)}/{sum(len(r['covered']) for r in runs)}")
    cm.save_json(out, os.path.join(cm.REPORT, 'encrypted_spot.json'))
    print('max |z| vs simulation', out['max_abs_z'], 'lambda 95% range', lq, 'coverage', out['coverage'])


if __name__ == '__main__':
    main()
