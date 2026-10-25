"""Encrypted spot check of P2 (step 4 of "what happens next" in reports/v31_P2/settings_choice_P2.md).

Full OpenFHE runs of the protocol (implementation/vfr/protocol.py) with P2's parameters passed to dp.Agreement
directly (the SETTINGS preset 'P' of vfr/dp.py still holds the old P and is not read).  Data: the synthetic dataset of
the earlier check (experiments/v31_settings/encrypted_settings.py): n = 10,000 matched records of the simulation
population, a 12,000-row register padded to 16,384 (never more), cohort n + 1,000, seed 11.  eps = 1.

For each run: lambda and theta against a plaintext simulation of the same mechanism on the same data (4,000 draws):
lambda against the simulated 95% range, theta against the simulated 95% range and against the simulated distribution
given the run's lambda.  The raw decrypted noise of every release is recorded (standardized by the calibrated sigma),
with a Kolmogorov-Smirnov test against N(0, 1); A must be exact.

    python experiments/v31_P2/encrypted_P2.py [--reps 5] [--eps 1] [--n 10000]
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
import p2common as pc  # noqa: E402
from p2common import cs, ev, dp  # noqa: E402
from vfr import common as VC  # noqa: E402
from vfr import data, protocol  # noqa: E402

SLOTS = 8192
P, DA = pc.P, pc.DA
AG_KEYS = ('f1', 'theta_star', 'kappa_Psi', 'alpha', 'alpha_L', 'alpha_U', 'r1_mode', 'intervals')


def simulate(G, c, cal, st, S, seed):
    rng = np.random.default_rng(seed)
    s1 = cs.stage1_batch(G, DA, cal, rng, S, st)
    E, f, g = ev.stage2_noise(P, DA, cal, rng, S)
    so = ev.stage2_solve(G[None] + E, c[None] + f, s1['lam'], s1['Psi'], cal['sigma2'])
    return s1, so


def main():
    from scipy import stats
    ap = argparse.ArgumentParser()
    ap.add_argument('--reps', type=int, default=5)
    ap.add_argument('--eps', type=float, default=1.0)
    ap.add_argument('--n', type=int, default=10000)
    ap.add_argument('--seed', type=int, default=11)
    a = ap.parse_args()
    n_scope = int(1.2 * a.n)
    n_bar = int(math.ceil(n_scope / SLOTS) * SLOTS)
    assert n_bar <= 16384, 'the encrypted spot check stays at or below 16,384 register rows'
    n_R = a.n + 1000
    st = ev.load_json(os.path.join(pc.REPORT, 'setting_P2.json'))['P2']
    pop = data.population()
    dR_, dO_, truth, b = data.synthetic(n_bar, n_scope, n_R, a.n, seed=a.seed, pop=pop)
    ps = data.plaintext_stats(dR_, dO_, b['B_R'], b['B_O'], b['B_y'])
    G, c = ps['G'], ps['c']
    eps = a.eps
    ag = dp.Agreement(n_R_max=n_R, n_bar_O=n_bar, d_R=4, d_O=6, eps_run=eps, delta_run=cs.DELTA,
                      kappa=cs.kappa_for(eps), **b, **{k: st[k] for k in AG_KEYS})
    iu = np.triu_indices(P)
    mask = np.ones((P, P), bool)
    mask[:DA, :DA] = False
    sel = mask[iu]
    runs = []
    for k in range(a.reps):
        with tempfile.TemporaryDirectory() as td:
            t0 = time.time()
            R, O = protocol.run(ag, VC.CKKSParams(), dR_, dO_, dict(omp_threads=1, solve_seed=900 + k),
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
                         theta=np.asarray(res.get('theta')).tolist(), noise=noise,
                         theta_lam=np.linalg.solve(G + s1['lam'] * np.asarray(s1['Psi']), c).tolist()))
        print(f"[P2] run {k}: {wall:.0f}s r1={s1['r1']:.0f} ({s1['r1_mode']}) lam={s1['lam']:.0f} ({s1['kind']}) "
              f"refused={refused} gate={res.get('theta_hat_gate')} noise sd stage2={noise['stage2']['sd']:.3f} "
              f"stage1={noise['stage1']['sd']:.3f} A exact={noise['A_exact']}", flush=True)
    sig = runs[0]['sigma']
    cal = dict(sigma1=sig['1'], sigma2=sig['2'], sigma_yy=sig.get('y', sig['2']))
    calp = cs.calibrate(eps, b['B_R'], b['B_O'], b['B_y'], 4, 6, st['f1'], n_bar_O=n_bar)
    S = 4000
    s1s, so = simulate(G, c, cal, st, S, 3)
    lq = np.quantile(s1s['lam'], [0.025, 0.5, 0.975])
    rng = np.random.default_rng(4)
    th_all = so['theta'][~so['refused']]
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
        r['theta_in_sim_95'] = ((np.array(r['theta']) >= np.quantile(th_all, 0.025, axis=0)) &
                                (np.array(r['theta']) <= np.quantile(th_all, 0.975, axis=0))).tolist()
        r['sim_refused_given_lambda'] = float(stc['refused'].mean())
    ok = [r for r in runs if not r['refused']]
    zall = [z for r in ok for z in r['z_vs_sim_given_lambda']]
    z1 = np.array([z for r in runs for z in r['noise']['z_stage1']])
    z2 = np.array([z for r in runs for z in r['noise']['z_stage2']])
    zp = np.concatenate([z1, z2])
    ell = float(np.linalg.eigvalsh(G / a.n)[0])
    out = dict(setting='P2', values=st, eps=eps, kappa=cs.kappa_for(eps),
               data=dict(n=a.n, n_scope=n_scope, n_bar_O=n_bar, n_R=n_R, seed=a.seed, delta=cs.DELTA, ell_matched=ell),
               sigma_encrypted=sig, sigma_plaintext_calibration=dict(sigma1=calp['sigma1'], sigma2=calp['sigma2']),
               n_off_matched=cs.n_off(calp, ell, P, DA, st)[0],
               sim=dict(draws=S, lam_q=lq.tolist(), refused=float(so['refused'].mean()), ridge_on=float(np.mean(s1s['lam'] > 0)),
                        psiO=float(np.mean(s1s['kind'] == 'O')), r1=float(s1s['r1']),
                        theta_q025=np.quantile(th_all, 0.025, axis=0).tolist(), theta_q975=np.quantile(th_all, 0.975, axis=0).tolist()),
               runs=runs, lam_in_range=f"{sum(r['lam_in_sim_95'] for r in runs)}/{len(runs)}",
               theta_in_range=f"{sum(sum(r['theta_in_sim_95']) for r in ok)}/{sum(len(r['theta_in_sim_95']) for r in ok)}",
               max_abs_z_given_lambda=float(np.max(np.abs(zall))) if zall else None,
               within_2p5sd=f"{sum(abs(z) <= 2.5 for z in zall)}/{len(zall)}",
               refused=sum(r['refused'] for r in runs),
               noise_pooled=dict(stage1=dict(n=int(z1.size), mean=float(z1.mean()), sd=float(z1.std()), max_abs=float(np.abs(z1).max()),
                                             ks_p=float(stats.kstest(z1, 'norm').pvalue)),
                                 stage2=dict(n=int(z2.size), mean=float(z2.mean()), sd=float(z2.std()), max_abs=float(np.abs(z2).max()),
                                             ks_p=float(stats.kstest(z2, 'norm').pvalue)),
                                 all=dict(n=int(zp.size), mean=float(zp.mean()), sd=float(zp.std()), max_abs=float(np.abs(zp).max()),
                                          ks_p=float(stats.kstest(zp, 'norm').pvalue))),
               A_exact=all(r['noise']['A_exact'] for r in runs))
    print(f"[P2] lambda in sim 95%: {out['lam_in_range']} (range {lq[0]:.0f}-{lq[2]:.0f}); theta in sim 95%: "
          f"{out['theta_in_range']}; |z| <= 2.5: {out['within_2p5sd']} (max {out['max_abs_z_given_lambda']}); "
          f"refused {out['refused']}; noise {out['noise_pooled']['all']}", flush=True)
    path = os.path.join(pc.REPORT, 'encrypted_P2.json')
    ev.save_json(out, path)
    print('wrote', path)


if __name__ == '__main__':
    main()
