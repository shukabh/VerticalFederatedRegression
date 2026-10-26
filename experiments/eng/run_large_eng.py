"""Full protocol runs with the parallel OPRF and padded release timing (release_pad='auto', margin 2), in the
configuration of the large runs (experiments/large/run_large.one_run: 4 OpenMP threads per party, 4 parallel
entries, a fresh single-run ledger, setting I, eps = 1): one run each of cells A100k_Istd_e1 and A1M_Istd_e1 of
reports/large (n_bar_O = 106,496 and 1,007,616, n_R^max = 20,000).  Never above 1M rows.

O's release phase here is the deadline calibration plus every release held to its deadline.  Records go to
experiments/_work/eng/run_large_eng.jsonl; the summary to reports/eng/run_large_eng.json.

    python experiments/eng/run_large_eng.py [--cells A100k_Istd_e1 A1M_Istd_e1]
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'experiments', 'large'))
import lcommon as lc  # noqa: E402
import run_large as RL  # noqa: E402
from lcommon import ev  # noqa: E402

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--cells', nargs='+', default=['A100k_Istd_e1', 'A1M_Istd_e1'])
    a = ap.parse_args()
    work = os.path.join(ROOT, 'experiments', '_work', 'eng')
    os.makedirs(work, exist_ok=True)
    out = dict(stamp=time.strftime('%Y-%m-%d %H:%M'), git_head=lc.git_head(), runs=[])
    for cid in a.cells:
        cell = lc.CELLS[cid]
        assert cell['n_bar_O'] <= 1_007_616
        built = lc.build(cell)
        ref = RL.reference(cell, built)
        ag = lc.agreement(cell, built)
        assert ag.release_pad == 'auto'
        rec = RL.one_run(cell, built, ag, ref, 0, work)
        with open(os.path.join(work, 'run_large_eng.jsonl'), 'a') as fh:
            fh.write(json.dumps(ev.jsonable(rec)) + '\n')
        TO = rec['timings_O']
        ph = dict(rec['phases'])
        rel = TO['release']
        ph['O.release'] = TO['release_calibrate'] + sum(r['busy'] + r['held'] for r in rel.values())
        lat = rec['timings_R'].get('release_latency_R', {})
        old = [c for c in json.load(open(os.path.join(ROOT, 'reports', 'large', f"stage{cell['stage']}.json")))['cells'].values()
               if c['cell']['id'] == cid][0]
        r = dict(id=cid, n_bar_O=cell['n_bar_O'], n_R_max=cell['n_R_max'], wall_run=rec['wall'],
                 psi_exact=rec['psi_exact'], phases=ph, O_total=sum(v for k, v in ph.items() if k.startswith('O.')),
                 release=dict(deadline=TO['release_deadline'], calibrate=TO['release_calibrate'], per_release=rel,
                              latency_R=lat),
                 peak_rss_MB=rec['peak_rss_MB'], comm_MB=sum(s['bytes'] for s in rec['messages_R'].values()) / 2 ** 20,
                 slots_sd={s: v['sd'] for s, v in rec['slots'].items()},
                 earlier=dict(O_total=old['cost_remeasured']['O_total']['measured'],
                              oprf_self=old['cost_remeasured']['phases']['O.oprf_self']['measured'],
                              wall_run=old['cost_remeasured']['wall_run']))
        print(json.dumps(dict(id=cid, O_total=r['O_total'], oprf=ph['O.oprf_self'], wall=rec['wall'],
                              deadline=r['release']['deadline'], earlier=r['earlier']), default=float), flush=True)
        out['runs'].append(r)
        del built
    with open(os.path.join(ROOT, 'reports', 'eng', 'run_large_eng.json'), 'w') as fh:
        json.dump(ev.jsonable(out), fh, indent=1)
