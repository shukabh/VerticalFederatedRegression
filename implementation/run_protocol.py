"""Run protocol v3.1 with R and O as two processes over a socket, with the DP noise on (production path).

    python run_protocol.py toy
    python run_protocol.py synthetic --n_bar_O 4096 --n_R 500 --n 300 [--variant A|C] [--log2_Omega_U auto|q2|68]
                                     [--eps 1.0] [--delta 1e-9] [--kappa auto|40] [--zeta 0.1]
                                     [--ledger_eps 10] [--ledger_delta 1e-6] [--ledger_xi 1e-3|none]
                                     [--transcript out.json]
                                     [--setting prev|I|P] [--f1 F] [--theta_star T] [--kappa_Psi K]
                                     [--alpha_L A] [--r1_mode union|exact] [--no_intervals]

--setting picks a named evaluation setting of the statistical parameters (vfr/dp.py SETTINGS: 'I' inference, the
default; 'P' prediction, without intervals; 'prev' the earlier one); the explicit options override single fields.

kappa 'auto' lets Calibrate derive the flooding parameter from (eps, delta, zeta, kappa_min) and the ledger cap.
The default ledger file holds a multi-run cap (ledger_eps, ledger_delta) with the hockey-stick accounting
(--ledger_xi, the share of ledger_eps reserved for smudging; 'none' selects the statistical-distance accounting);
--ledger_eps none gives a single-run ledger whose cap is this run's budget.  The ledger charges a run's full cost
when it admits the run.

There is deliberately no switch for the noise-free debug path (tests only).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from vfr import common as C  # noqa: E402
from vfr import data, dp, protocol  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dataset', choices=['toy', 'synthetic'])
    ap.add_argument('--n_bar_O', type=int, default=4096)
    ap.add_argument('--n_R', type=int, default=500)
    ap.add_argument('--n_R_max', type=int, default=None)
    ap.add_argument('--n', type=int, default=300)
    ap.add_argument('--variant', default='A', choices=['A', 'C'])
    ap.add_argument('--log2_Omega_U', default='auto',
                    help="R's encoding scale: auto (2^68 for circuit A, q2 for C), q2, or log2 of the scale")
    ap.add_argument('--eps', type=float, default=1.0)
    ap.add_argument('--delta', type=float, default=1e-9)
    ap.add_argument('--kappa', default='auto', help="flooding parameter: auto (derived by Calibrate) or an integer")
    ap.add_argument('--zeta', type=float, default=0.1, help='kappa=auto: slack share of delta_run')
    ap.add_argument('--kappa_min', type=int, default=40, help='kappa=auto: statistical security floor')
    ap.add_argument('--ledger_eps', default='10', help="ledger cap eps of the base register, or 'none'")
    ap.add_argument('--ledger_delta', type=float, default=1e-6)
    ap.add_argument('--ledger_xi', default='1e-3',
                    help="share of ledger_eps reserved for smudging (hockey-stick ledger), or 'none' (statistical "
                         "distance)")
    ap.add_argument('--B_beta_pub', type=float, default=float('inf'))
    ap.add_argument('--ledger', default=os.path.join(HERE, 'ledger_O.json'))
    ap.add_argument('--transcript', default=None)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--setting', default='I', choices=sorted(dp.SETTINGS),
                    help='named evaluation setting of the statistical parameters (vfr/dp.py SETTINGS)')
    ap.add_argument('--f1', type=float, default=None, help='stage-1 share of rho_run (overrides the setting)')
    ap.add_argument('--theta_star', type=float, default=None)
    ap.add_argument('--kappa_Psi', type=float, default=None)
    ap.add_argument('--alpha_L', type=float, default=None)
    ap.add_argument('--r1_mode', default=None, choices=['union', 'exact'])
    ap.add_argument('--no_intervals', action='store_true', help='estimate only (no interval guarantee)')
    a = ap.parse_args()
    a.log2_Omega_U = {'auto': 'auto', 'q2': None}.get(a.log2_Omega_U) if a.log2_Omega_U in ('auto', 'q2') \
        else float(a.log2_Omega_U)
    priv = dict(eps_run=a.eps, delta_run=a.delta, kappa=a.kappa if a.kappa == 'auto' else int(a.kappa),
                zeta=a.zeta, kappa_min=a.kappa_min,
                ledger_eps=None if a.ledger_eps.lower() == 'none' else float(a.ledger_eps),
                ledger_delta=None if a.ledger_eps.lower() == 'none' else a.ledger_delta,
                ledger_xi=None if a.ledger_xi.lower() == 'none' else float(a.ledger_xi))
    stat = dp.setting(a.setting)
    for k in ('f1', 'theta_star', 'kappa_Psi', 'alpha_L', 'r1_mode'):
        if getattr(a, k) is not None:
            stat[k] = getattr(a, k)
    if a.no_intervals:
        stat['intervals'] = False
    priv.update(stat)
    if a.dataset == 'toy':
        dR_, dO_, b = data.toy()
        ag = dp.Agreement(n_R_max=4, n_bar_O=6, d_R=2, d_O=2, **priv, **b)
        truth = None
    else:
        dR_, dO_, truth, b = data.synthetic(a.n_bar_O, int(0.9 * a.n_bar_O), a.n_R, a.n, seed=a.seed)
        ag = dp.Agreement(n_R_max=a.n_R_max or a.n_R + 100, n_bar_O=a.n_bar_O, d_R=4, d_O=6, variant=a.variant,
                          log2_Omega_U=a.log2_Omega_U, B_beta_pub=a.B_beta_pub, **priv, **b)
    R, O = protocol.run(ag, C.CKKSParams(), dR_, dO_, {}, dict(ledger_path=a.ledger))
    if 'refused' in R:
        print('refused:', R['refused'])
        return
    print(f"matched n = {R['n']}" + ('' if truth is None else f"  (PSI exact: {set(R['I']) == truth['matched']})"))
    cal = R['calibration']
    print(f"kappa {cal['kappa']} (slack {100 * cal['slack_fraction']:.2f}% of delta); "
          f"log QP {cal['log2_QP']:.0f} <= {cal['log2_QP_max']}; q0 2^{cal['q0_bits']:.0f} > 2^{cal['q0_need_bits']:.1f}")
    print(f"mode {'joint' if cal['joint'] else 'separate'}; rho_run {cal['rho_run']:.5f}; Omega_f 2^{cal['log2_Omega_f']:.2f}; "
          + '; '.join(f"sigma_{s} = {v['sigma_slot']:.1f}" for s, v in cal['releases'].items()))
    s1 = R['stage1']
    print(f"setting {a.setting}: " + ', '.join(f'{k} = {v}' for k, v in stat.items()))
    print(f"stage 1: r_1 = {s1['r1']:.4g} ({s1['r1_mode']}), lambda = {s1['lam']:.4g}, Psi = {s1['kind']}")
    res = R['result']
    if res is None:
        print('n <= p: no estimate')
    elif res['refused']:
        print(f"refused by the gate (theta_hat = {res['theta_hat_gate']:.3f}); the budget is spent")
    else:
        np.set_printoptions(precision=4, suppress=True)
        print('theta~       ', res['theta'])
        if res.get('intervals', True):
            print('Wald half    ', res['half_theta'])
            print('beta half    ', res['half_beta'])
        else:
            print('(no intervals in this setting)')
        print(f"gate theta_hat = {res['theta_hat_gate']:.3f}; sigma_2 tr(P~) = {res['sigma2_trP']:.3f}")
    print('messages (R view):')
    for k, v in R['summary'].items():
        print(f"  {k:28s} {v['count']:3d} x  {v['bytes'] / 2 ** 20:9.3f} MB")
    if a.transcript:
        with open(a.transcript, 'w') as fh:
            json.dump(dict(R=R['transcript'], O=O['transcript']), fh, indent=1)
        print('transcript written to', a.transcript)


if __name__ == '__main__':
    main()
