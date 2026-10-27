"""Cost of K match keys per record, from the repo's refreshed cost model (implementation/bench/cost_model.py).

With K keys O inserts K * n_bar_O items (each labelled with its row's shuffled position) and R sends K * n_R^max padded
OPRF queries and PSI items.  The CKKS part (upload, aggregation, releases) is unchanged: it runs over the n_bar_O
positions whatever the linkage.  So the Link phases are evaluated at (K n_bar_O, K n_R^max) and the CKKS phases at
(n_bar_O, n_R^max), both with predict_run (4 threads per party, 4 parallel entries, parallel OPRF, release padding),
and the PSI parameter check (vfr/psi.PSIParams.check: load bound, partitions, cuckoo tables, false-match bound
p_pass <= delta_psi) is run at the scaled sizes.  Extrapolation beyond the measured 1M rows / 100k queries is the
model's, not a measurement.

Two layouts give R exactly the same output (keys are domain-separated by type, so a key of type k can only meet a key
of type k):
  'joint'     one PSI over all K n_bar_O items and K n_R^max queries.  The evaluation and the replies scale with
              (cuckoo tables) x (partitions per bin), both proportional to K, so their cost grows like K^2 (and faster
              at large n_R^max, where the delta_psi bound forces more, emptier cuckoo tables);
  'per_type'  K independent PSI instances, one per key type, each of the present size (n_bar_O items, n_R^max
              queries): every Link phase costs K times the present one.  This is the layout to use.

    python cost.py        # writes reports/probabilistic_linkage/cost.json
"""
from __future__ import annotations

import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'implementation'))
sys.path.insert(0, os.path.join(ROOT, 'implementation', 'bench'))
import cost_model as CM  # noqa: E402
from vfr import psi as PSI  # noqa: E402

LINK = ('O.oprf_self', 'O.psi_preprocess', 'O.oprf_requests', 'O.psi_eval', 'O.psi_sanitize', 'R.oprf', 'R.psi_bfv')
LINK_BYTES = ('R->O OPRF', 'O->R OPRF', 'R->O PSI query', 'O->R PSI reply')
MB = 2 ** 20


def run(n_bar_O, n_R_max, K, mu1, mu4, fit, sp, layout='joint'):
    base = CM.predict_run(n_bar_O, n_R_max, mu1, mu4, fit, sp, pad_margin=2.0)
    big = CM.predict_run(K * n_bar_O, K * n_R_max, mu1, mu4, fit, sp, pad_margin=2.0) if layout == 'joint' else base
    f = 1 if layout == 'joint' else K
    ph = dict(base['phases'])
    for k in LINK:
        ph[k] = f * big['phases'][k]
    by = dict(base['bytes'])
    for k in LINK_BYTES:
        by[k] = f * big['bytes'][k]
    pp = PSI.PSIParams()
    chk = pp.check(K * n_bar_O, K * n_R_max) if layout == 'joint' else pp.check(n_bar_O, n_R_max)
    O_tot = sum(v for k, v in ph.items() if k.startswith('O.'))
    R_tot = sum(v for k, v in ph.items() if k.startswith('R.'))
    return dict(layout=layout, K=K, n_bar_O=n_bar_O, n_R_max=n_R_max, O_items=K * n_bar_O, R_queries=K * n_R_max,
                O_wall=O_tot, R_wall=R_tot, link_O=sum(ph[k] for k in LINK if k.startswith('O.')),
                comm_MB=sum(by.values()) / MB, link_comm_MB=sum(by[k] for k in LINK_BYTES) / MB,
                D=big['D'], alpha=big['alpha'], tables=big['tables'], psi_check_ok=bool(chk['ok']),
                psi_check_reasons=chk.get('reasons', []),
                log2_p_pass=chk['false_match']['log2_p_pass'], log2_p_cuckoo=chk['log2_p_cuckoo'], phases=ph)


def main():
    mu1, mu4 = CM.load('micro_t1.json'), CM.load('micro_t4.json')
    cm = CM.load('cost_model.json')
    fit, sp = cm['fit'], cm['speedups_4cores']['aggregate']
    out = []
    for n_bar_O, n_R_max in ((1_007_616, 20_000), (1_007_616, 100_000)):
        for K, layout in [(k, l) for l in ('joint', 'per_type') for k in (1, 2, 4, 7, 8, 12)]:
            r = run(n_bar_O, n_R_max, K, mu1, mu4, fit, sp, layout)
            out.append(r)
            print(f"{layout}: n_bar_O={n_bar_O} n_R^max={n_R_max} K={K}: O {r['O_wall']:.0f}s (link {r['link_O']:.0f}s), "
                  f"R {r['R_wall']:.0f}s, comm {r['comm_MB']:.0f} MB (link {r['link_comm_MB']:.0f}), D={r['D']} "
                  f"alpha={r['alpha']} tables={r['tables']} psi ok={r['psi_check_ok']} {r['psi_check_reasons']}",
                  flush=True)
    os.makedirs(os.path.join(ROOT, 'reports', 'probabilistic_linkage'), exist_ok=True)
    with open(os.path.join(ROOT, 'reports', 'probabilistic_linkage', 'cost.json'), 'w') as fh:
        json.dump(out, fh, indent=1)


if __name__ == '__main__':
    main()
