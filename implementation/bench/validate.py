"""Small-scale validation numbers for the report (the same checks as tests/, with the measured values printed).

    python bench/validate.py     -> bench/results/validation.json
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, '..')))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, '..', 'tests')))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, '..', 'tests', 'reference')))

from helpers import Pipeline, tau_vectors, true_entries  # noqa: E402
from vfr import common as C  # noqa: E402
from vfr import data, dp, protocol  # noqa: E402


def main():
    out = {}
    wd = os.path.join(HERE, 'results', 'work')
    os.makedirs(wd, exist_ok=True)
    os.environ['VFR_DEBUG_NO_DP'] = '1'
    # ---------------- toy example
    dR_, dO_, b = data.toy()
    ag = dp.Agreement(n_R_max=4, n_bar_O=6, d_R=2, d_O=2, **b)
    R, O = protocol.run(ag, C.CKKSParams(), dR_, dO_, {}, dict(debug_no_dp=True, tau_override=data.TOY_TAU,
                                                                 ledger_path=os.path.join(wd, 'v_toy.json')))
    toy = json.load(open(os.path.normpath(os.path.join(HERE, '..', '..', 'reports', 'toy_example', 'toy.json'))))
    out['toy'] = dict(I=R['I'], pi=R['pi'], max_abs_err_G=float(np.abs(R['G2'] - np.array(toy['G'])).max()),
                      max_abs_err_c=float(np.abs(R['c2'] - np.array(toy['c'])).max()), err_yy=abs(R['yy'] - toy['yy']),
                      log2_Omega_f=R['calibration']['log2_Omega_f'])
    # ---------------- synthetic, noise off, both circuits
    for variant in ('A', 'C'):
        dR_, dO_, truth, b = data.synthetic(4096, 3500, 500, 300, seed=31)
        ag = dp.Agreement(n_R_max=600, n_bar_O=4096, d_R=4, d_O=6, variant=variant, **b)
        R, O = protocol.run(ag, C.CKKSParams(), dR_, dO_, {}, dict(debug_no_dp=True, ledger_path=os.path.join(wd, f'v_{variant}.json')))
        ps = data.plaintext_stats(dR_, dO_, b['B_R'], b['B_O'], b['B_y'])
        out[f'synthetic_noise_off_{variant}'] = dict(
            psi_exact=set(R['I']) == truth['matched'], n=R['n'],
            alignment_exact=all(R['pi'][i] == O['tau'][i] for i in R['I']),
            max_rel_err_G=float(np.abs(R['G2'] - ps['G']).max() / np.abs(ps['G']).max()),
            max_rel_err_c=float(np.abs(R['c2'] - ps['c']).max() / np.abs(ps['G']).max()),
            log2_Omega_f=R['calibration']['log2_Omega_f'], wall_O=sum(v for v in O['timings'].values() if isinstance(v, float)))
    os.environ.pop('VFR_DEBUG_NO_DP')
    # ---------------- noise on: s.d. over repeated releases; random error vs worst-case bound
    n_bar = 4096
    dR_, dO_, truth, b = data.synthetic(n_bar, 3500, 500, 300, seed=32)
    ag = dp.Agreement(n_R_max=600, n_bar_O=n_bar, d_R=4, d_O=6, **b)
    perm = np.random.default_rng(0).permutation(n_bar)[:len(dO_['ids'])] + 1
    tau = {i: int(p) for i, p in zip(dO_['ids'], perm)}
    vecs, XO, yt = tau_vectors(dR_, dO_, tau, ag)
    ps = data.plaintext_stats(dR_, dO_, b['B_R'], b['B_O'], b['B_y'])
    tru = true_entries(ps['G'], ps['c'], ps['yy'], 4, 6)
    pl = Pipeline(ag, C.CKKSParams(), vecs, XO, yt)
    nz = {}
    for stage, reps in ((2, 60), (1, 30)):
        sig = pl.cal['releases'][str(stage)]['sigma_slot']
        names = [e['name'] for e in C.release_entries(pl.ents, stage)]
        rows = []
        for _ in range(reps):
            vals = pl.release(stage)[0]
            rows.append([vals[k] - tru[k] for k in names])
        Dm = np.array(rows)
        per = Dm.std(axis=0) / sig
        nz[str(stage)] = dict(releases=reps, entries=len(names), sigma_slot=sig, pooled_sd=float(np.sqrt((Dm ** 2).mean())),
                              ratio=float(np.sqrt((Dm ** 2).mean()) / sig), mean_over_sigma=float(Dm.mean() / sig),
                              per_entry_ratio_min=float(per.min()), per_entry_ratio_max=float(per.max()))
    out['noise_sd'] = nz
    p2 = Pipeline(ag, C.CKKSParams(), vecs, XO, yt)
    marg = {}
    for s in (1, 2):
        _, _, v1 = pl.release(s, debug_no_dp=True)
        _, _, v2 = p2.release(s, debug_no_dp=True)
        d = np.array([x - y for x, y in zip(v1, v2)], dtype=float) / math.sqrt(2)
        marg[str(s)] = dict(log2_measured_l2=float(math.log2(np.linalg.norm(d))),
                            log2_B_tot=float(math.log2(pl.cal['releases'][str(s)]['B_tot'])))
    out['random_error_vs_Btot'] = marg
    # ---------------- noise on, full protocol, compare with alg6
    import alg6
    dR_, dO_, truth, b = data.synthetic(4096, 3500, 500, 300, seed=33)
    ag = dp.Agreement(n_R_max=600, n_bar_O=4096, d_R=4, d_O=6, B_beta_pub=5.0, **b)
    R, O = protocol.run(ag, C.CKKSParams(), dR_, dO_, dict(solve_seed=7, Bboot=1000), dict(ledger_path=os.path.join(wd, 'v_n.json')))
    res = R['result']

    class ZeroFirst:
        def __init__(self, seed, k):
            self.g, self.k = np.random.default_rng(seed), k

        def standard_normal(self, size=None):
            if self.k > 0:
                self.k -= 1
                return np.zeros(size) if size is not None else 0.0
            return self.g.standard_normal(size)
    sig = R['sigma']
    s1 = alg6.stage1(ZeroFirst(0, 2), R['G1'], 5, sig['1'], sig['2'], ag.alpha / 20, ag.theta_star, ag.kappa_Psi)
    ref = alg6.solve_infer(ZeroFirst(7, 4), R['G2'], R['c2'], R['yy'], R['n'], 5, s1, sig['2'], sig['2'], ag.B_y, ag.alpha,
                           ag.alpha / 20, ag.alpha / 20, ag.B_beta_pub, Bboot=1000, truncate=True, antithetic=True)
    out['alg6_equivalence'] = dict(
        refused=res['refused'], lam=R['stage1']['lam'], Psi=R['stage1']['kind'], gate=res['theta_hat_gate'],
        max_abs_diff_theta=float(np.abs(ref['theta'] - res['theta']).max()),
        max_abs_diff_half_theta=float(np.abs(ref['half_theta'] - res['half_theta']).max()),
        max_abs_diff_half_beta=float(np.abs(ref['half_beta'] - res['half_beta']).max()),
        max_rel_diff_Vmech=float(np.abs(ref['Vmech'] - res['Vmech']).max() / np.abs(ref['Vmech']).max()),
        U_yy_corrected=res['U_yy'], U_yy_spec=res['U_spec'], theta=res['theta'].tolist(),
        half_theta=res['half_theta'].tolist())
    with open(os.path.join(HERE, 'results', 'validation.json'), 'w') as fh:
        json.dump(out, fh, indent=1, default=str)
    print(json.dumps(out, indent=1, default=str))


if __name__ == '__main__':
    main()
