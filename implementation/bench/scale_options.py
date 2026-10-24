"""The CKKS scale question: what each option implies.

1. Calibrate (worst-case B_tot of the actual OpenFHE gadget) for circuit A (as written, with R at q2, masks at q1
   and O's data at Omega_V = Omega_f) and circuit C (restructured), at several register sizes.
2. Collects the OpenFHE-only microbenchmarks of the alternative backends (composite scaling in the 64-bit build,
   FIXEDMANUAL in a 128-bit native build) written by alt_scaling.cpp into results/alt_scaling.jsonl.

    python bench/scale_options.py
"""
from __future__ import annotations

import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, '..')))
from vfr import btot as bt  # noqa: E402
from vfr import common as C  # noqa: E402
from vfr import dp  # noqa: E402


def main():
    ck = C.make_ckks(C.CKKSParams())
    qs = list(ck.qs)
    g = C.gadget_constants(ck)
    rows = []
    for variant, lu in (('A', None), ('A', 68.0), ('C', None)):
        for nbar in (4096, 100_000, 1_000_000):
            for eps in (1.0, 4.0):
                ag = dp.Agreement(n_R_max=2000, n_bar_O=nbar, d_R=4, d_O=6, B_R=5.5911, B_O=6.2418, B_y=2.8880,
                                  eps_run=eps, variant=variant, log2_Omega_U=lu)
                cal = dp.calibrate(ag, ck.N, qs[2], qs[3], qs[0] * qs[1], g)
                r2 = cal['releases']['2']
                cp = bt.CircuitParams(N=ck.N, n_bar=nbar, d_R=4, d_O=6, B_R=ag.B_R, B_O=ag.B_O, B_y=ag.B_y, q1=qs[2],
                                      q2=qs[3], Omega_U=float(cal['Omega_U']), Omega_V=cal['Omega_V'], Omega_u=qs[2], **g)
                terms = {k: round(math.log2(v), 1) if v > 0 else None for k, v in bt.bounds(cp, 2, True, variant).items()}
                rows.append(dict(variant=variant + ('' if lu is None else f' (Omega_U=2^{lu:.0f})'), n_bar_O=nbar, eps=eps, log2_Omega_f=round(cal['log2_Omega_f'], 2),
                                 log2_Btot_1=round(math.log2(cal['releases']['1']['B_tot']), 2),
                                 log2_Btot_2=round(math.log2(r2['B_tot']), 2), sigma_slot_2=round(r2['sigma_slot'], 1),
                                 log2_sigma_c_2=round(math.log2(r2['sigma_c']), 2), log2_Delta_x_2=round(math.log2(r2['Delta_x']), 1),
                                 q0_need_bits=round(cal['q0_need_bits'], 1), q0_need_Q_bits=round(cal['q0_need_Q_bits'], 1),
                                 terms_stage2=terms))
    out = dict(chain=dict(q_bits=[q.bit_length() for q in qs], p_bits=[p.bit_length() for p in ck.ps], gadget=g), rows=rows)
    alt = os.path.join(HERE, 'results', 'alt_scaling.jsonl')
    if os.path.exists(alt):
        out['alt_backends'] = [json.loads(l) for l in open(alt) if l.strip().startswith('{')]
    with open(os.path.join(HERE, 'results', 'scale_options.json'), 'w') as fh:
        json.dump(out, fh, indent=1)
    print('| circuit | n_bar_O | eps | log2 Omega_f | log2 B_tot (st1, st2) | sigma_2 (slot) | log2 sigma_c,2 | q0 needed (spec / (Q)) bits |')
    print('|---|---|---|---|---|---|---|---|')
    for r in rows:
        print(f"| {r['variant']} | {r['n_bar_O']:,} | {r['eps']} | {r['log2_Omega_f']} | {r['log2_Btot_1']}, {r['log2_Btot_2']} | "
              f"{r['sigma_slot_2']} | {r['log2_sigma_c_2']} | {r['q0_need_bits']} / {r['q0_need_Q_bits']} |")
    print()
    for r in rows:
        if r['eps'] == 1.0:
            print(r['variant'], r['n_bar_O'], r['terms_stage2'])


if __name__ == '__main__':
    main()
