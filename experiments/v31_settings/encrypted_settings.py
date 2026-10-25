"""Encrypted spot check of settings I and P (full OpenFHE protocol, implementation/vfr/protocol.py).

One synthetic dataset from the simulation population: n = 10,000 matched records in a 12,000-row register padded to
16,384 (never more), cohort n + 1,000.  Each setting runs at one eps where its stage-1 ridge is on (I at eps = 2,
P at eps = 1 by default).  Each run's lambda and estimate are compared with a plaintext simulation of the same
mechanism on the same data (4,000 draws): lambda against the simulated 95% range, theta against the simulated
distribution given the run's lambda.  The raw decrypted noise of every release is recorded (standardized by the
calibrated sigma), and A must be exact.

    python experiments/v31_settings/encrypted_settings.py [--reps 5] [--eps_I 2] [--eps_P 1] [--n 10000]
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
import scommon as cs  # noqa: E402
from scommon import ev, dp, solve  # noqa: E402
from vfr import common as VC  # noqa: E402
from vfr import data, protocol  # noqa: E402

SLOTS = 8192
P, DA = 11, 5


def simulate(G, c, cal, st, S, seed):
    rng = np.random.default_rng(seed)
    s1 = cs.stage1_batch(G, DA, cal, rng, S, st)
    E, f, g = ev.stage2_noise(P, DA, cal, rng, S)
    so = ev.stage2_solve(G[None] + E, c[None] + f, s1['lam'], s1['Psi'], cal['sigma2'])
    return s1, so


def run_setting(name, st, eps, reps, dR_, dO_, b, n_R, n_bar, G, c, seed0):
    ag = dp.Agreement(n_R_max=n_R, n_bar_O=n_bar, d_R=4, d_O=6, eps_run=eps, delta_run=cs.DELTA,
                      kappa=cs.kappa_for(eps), **b, **{k: v for k, v in dp.setting(name).items()})
    assert all(getattr(ag, k) == st[k] or (k in ('alpha_L', 'alpha_U') and abs(getattr(ag, k) - st[k]) < 1e-15)
               for k in ('f1', 'theta_star', 'kappa_Psi', 'r1_mode', 'intervals', 'alpha_L')), 'dp.SETTINGS != settings.json'
    runs = []
    iu = np.triu_indices(P)
    mask = np.ones((P, P), bool)
    mask[:DA, :DA] = False
    sel = mask[iu]
    for k in range(reps):
        with tempfile.TemporaryDirectory() as td:
            t0 = time.time()
            R, O = protocol.run(ag, VC.CKKSParams(), dR_, dO_, dict(omp_threads=1, solve_seed=seed0 + k),
                                dict(omp_threads=1, ledger_path=os.path.join(td, 'ledger.json')))
            wall = time.time() - t0
        res = R['result'] or {}
        s1 = R['stage1']
        zG2 = ((np.asarray(R['G2']) - G)[iu][sel]) / R['sigma']['2']
        zG1 = ((np.asarray(R['G1']) - G)[iu][sel]) / R['sigma']['1']
        zc = (np.asarray(R['c2']) - c) / R['sigma']['2']
        zz2 = np.concatenate([zG2, zc])
        noise = dict(stage2=dict(n=int(zz2.size), mean=float(zz2.mean()), sd=float(zz2.std()), max_abs=float(np.abs(zz2).max())),
                     stage1=dict(n=int(zG1.size), mean=float(zG1.mean()), sd=float(zG1.std()), max_abs=float(np.abs(zG1).max())),
                     A_exact=bool(np.allclose(np.asarray(R['G2'])[:DA, :DA], G[:DA, :DA], rtol=1e-9, atol=1e-6)),
                     z_stage2=zz2.tolist(), z_stage1=zG1.tolist())
        refused = bool(res.get('refused', True))
        runs.append(dict(k=k, wall=wall, n=R['n'], lam=float(s1['lam']), kind=s1['kind'], r1=float(s1['r1']),
                         r1_mode=s1['r1_mode'], sigma=R['sigma'], refused=refused, gate=res.get('theta_hat_gate'),
                         theta=np.asarray(res.get('theta')).tolist(),
                         half_theta=(np.asarray(res['half_theta']).tolist() if 'half_theta' in res else None),
                         theta_lam=np.linalg.solve(G + s1['lam'] * np.asarray(s1['Psi']), c).tolist(), noise=noise))
        print(f"[{name}] run {k}: {wall:.0f}s r1={s1['r1']:.0f} ({s1['r1_mode']}) lam={s1['lam']:.0f} ({s1['kind']}) "
              f"refused={refused} noise sd stage2={noise['stage2']['sd']:.3f} stage1={noise['stage1']['sd']:.3f} "
              f"A exact={noise['A_exact']}", flush=True)
    sig = runs[0]['sigma']
    cal = dict(sigma1=sig['1'], sigma2=sig['2'], sigma_yy=sig.get('y', sig['2']))
    S = 4000
    s1s, so = simulate(G, c, cal, st, S, 3)
    lq = np.quantile(s1s['lam'], [0.025, 0.5, 0.975])
    rng = np.random.default_rng(4)
    for r in runs:
        r['lam_in_sim_95'] = bool(lq[0] <= r['lam'] <= lq[2])
        r['lam_sim_quantile'] = float(np.mean(s1s['lam'] <= r['lam']))
        if r['refused']:
            continue
        Psi_r = np.eye(P) if r['kind'] == 'I' else ev.pi_O(P, DA)
        Ec, fc, _ = ev.stage2_noise(P, DA, cal, rng, S)
        stc = ev.stage2_solve(G[None] + Ec, c[None] + fc, np.full(S, r['lam']), np.broadcast_to(Psi_r, (S, P, P)), cal['sigma2'])
        tc = stc['theta'][~stc['refused']]
        r['z_vs_sim_given_lambda'] = ((np.array(r['theta']) - tc.mean(0)) / tc.std(0)).tolist()
        th_all = so['theta'][~so['refused']]
        r['theta_in_sim_95'] = ((np.array(r['theta']) >= np.quantile(th_all, 0.025, axis=0)) &
                                (np.array(r['theta']) <= np.quantile(th_all, 0.975, axis=0))).tolist()
        if r['half_theta'] is not None:
            r['covered'] = (np.abs(np.array(r['theta']) - np.array(r['theta_lam'])) <= np.array(r['half_theta'])).tolist()
    ok = [r for r in runs if not r['refused']]
    zall = [z for r in ok for z in r['z_vs_sim_given_lambda']]
    out = dict(setting=name, values=st, eps=eps, sigma_encrypted=sig,
               sim=dict(draws=S, lam_q=lq.tolist(), refused=float(so['refused'].mean()),
                        ridge_on=float(np.mean(s1s['lam'] > 0)), psiO=float(np.mean(s1s['kind'] == 'O')),
                        r1=float(s1s['r1'])),
               runs=runs, lam_in_range=f"{sum(r['lam_in_sim_95'] for r in runs)}/{len(runs)}",
               theta_in_range=f"{sum(sum(r['theta_in_sim_95']) for r in ok)}/{sum(len(r['theta_in_sim_95']) for r in ok)}",
               max_abs_z_given_lambda=float(np.max(np.abs(zall))) if zall else None,
               within_2p5sd=f"{sum(abs(z) <= 2.5 for z in zall)}/{len(zall)}",
               refused=sum(r['refused'] for r in runs),
               noise_pooled=dict((s, dict(n=int(len(v)), mean=float(np.mean(v)), sd=float(np.std(v)), max_abs=float(np.max(np.abs(v)))))
                                 for s, v in (('stage2', [z for r in runs for z in r['noise']['z_stage2']]),
                                              ('stage1', [z for r in runs for z in r['noise']['z_stage1']]))),
               A_exact=all(r['noise']['A_exact'] for r in runs))
    if ok and ok[0]['half_theta'] is not None:
        out['coverage'] = f"{sum(sum(r['covered']) for r in ok)}/{sum(len(r['covered']) for r in ok)}"
    print(f"[{name}] lambda in sim 95%: {out['lam_in_range']} (range {lq[0]:.0f}-{lq[2]:.0f}); theta in sim 95%: "
          f"{out['theta_in_range']}; |z| <= 2.5: {out['within_2p5sd']}; coverage {out.get('coverage')}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reps', type=int, default=5)
    ap.add_argument('--eps_I', type=float, default=2.0)
    ap.add_argument('--eps_P', type=float, default=1.0)
    ap.add_argument('--n', type=int, default=10000)
    ap.add_argument('--seed', type=int, default=11)
    a = ap.parse_args()
    n_scope = int(1.2 * a.n)
    n_bar = int(math.ceil(n_scope / SLOTS) * SLOTS)
    assert n_bar <= 16384, 'the encrypted spot check stays at or below 16,384 register rows'
    n_R = a.n + 1000
    pop = data.population()
    dR_, dO_, truth, b = data.synthetic(n_bar, n_scope, n_R, a.n, seed=a.seed, pop=pop)
    ps = data.plaintext_stats(dR_, dO_, b['B_R'], b['B_O'], b['B_y'])
    G, c = ps['G'], ps['c']
    sts = cs.load_settings()
    ell = float(np.linalg.eigvalsh(G / a.n)[0])
    out = dict(data=dict(n=a.n, n_scope=n_scope, n_bar_O=n_bar, n_R=n_R, seed=a.seed, delta=cs.DELTA,
                         ell_matched=ell), results={})
    for name, eps, seed0 in (('I', a.eps_I, 700), ('P', a.eps_P, 800)):
        cal = cs.calibrate(eps, b['B_R'], b['B_O'], b['B_y'], 4, 6, sts[name]['f1'], n_bar_O=n_bar)
        out['results'][name] = run_setting(name, sts[name], eps, a.reps, dR_, dO_, b, n_R, n_bar, G, c, seed0)
        out['results'][name]['n_off_matched'] = cs.n_off(cal, ell, P, DA, sts[name])[0]
        out['results'][name]['kappa'] = cs.kappa_for(eps)
    ev.save_json(out, os.path.join(cs.REPORT, 'encrypted_spot.json'))
    print('wrote', os.path.join(cs.REPORT, 'encrypted_spot.json'))


if __name__ == '__main__':
    main()
