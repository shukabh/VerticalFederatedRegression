"""Encrypted runs of the large-scale evaluation (full OpenFHE protocol, implementation/vfr/protocol.py).

Each run is the production path: two processes over a socket, exact discrete-Gaussian noise, a single-run ledger in
a fresh file, OpenMP with THREADS threads per party and THREADS parallel entries in O's aggregation.  One JSON line per
run is appended to experiments/_work/large/<cell>.jsonl (resumable; git-ignored): per-phase wall time (the parties'
timers) and CPU time and peak memory (their phase marks), traffic per message type, PSI correctness against the
plaintext join (match set and every position), the raw decrypted noise of the released entries and of all 8,192 slots
(standardized by the calibrated sigma), lambda, the estimate, the intervals and theta_lambda.  A run's ciphertexts
live only in the two processes' memory and nothing large is written to disk.

    python experiments/large/run_large.py --stage 1 [--reps 3] [--cells A100k_Istd_e1 ...]
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lcommon as lc  # noqa: E402
from lcommon import ev  # noqa: E402
from vfr import common as VC  # noqa: E402
from vfr import protocol  # noqa: E402


def slot_stats(z, used, sigma):
    """Noise in all slots of one decoded release (real parts outside the packed entries and every imaginary part),
    standardized by sigma, as experiments/v31_more/encrypted_more.slot_stats."""
    from scipy import stats
    z = np.asarray(z)
    mask = np.ones(z.size, bool)
    mask[list(used)] = False
    v = np.concatenate([z.real[mask], z.imag]) / sigma
    return dict(n=int(v.size), mean=float(v.mean()), sd=float(v.std()), max_abs=float(np.abs(v).max()),
                ks_p=float(stats.kstest(v, 'norm').pvalue), ks_stat=float(stats.kstest(v, 'norm').statistic),
                hist=np.histogram(v, bins=np.linspace(-5, 5, 41))[0].tolist())


def reference(cell, built):
    """Plaintext reference of the cell's data (the same for every run): G, c, y'y, n, the non-private fit and, for
    3droad, the test design."""
    ps = built['ps']
    G, c = ps['G'], ps['c']
    ref = dict(G=G.tolist(), c=c.tolist(), yy=ps['yy'], n=ps['n'], d_R=built['d_R'], d_O=built['d_O'],
               n_bar_O=built['n_bar_O'], bounds=built['bounds'], theta_ols=np.linalg.solve(G, c).tolist(),
               lmin_G=float(np.linalg.eigvalsh(G)[0]))
    if cell['dataset'] == 'synthetic':
        ref['beta_true'] = built['beta_true'].tolist()
    else:
        T, M = built['test'], built['matched_design']
        th_np = ev.ridge1(M['Z'], M['y'])                 # the arm's own non-private fit (as the paper's real-data runs)
        ref.update(meta=built['meta'], theta_np=th_np.tolist(), test_mse_np=ev.test_mse(T['Z'], T['y'], th_np),
                   test_mse_trivial=float(np.mean(T['y'] ** 2)))
    return ref


def one_run(cell, built, ag, ref, k, workdir):
    G, c, yy = np.asarray(ref['G']), np.asarray(ref['c']), ref['yy']
    p, dA = G.shape[0], built['d_R'] + 1
    seed = 50_000 + 1000 * cell['stage'] + 10 * k + list(lc.CELLS).index(cell['id'])
    opts = dict(omp_threads=lc.THREADS, threads=lc.THREADS)
    with tempfile.TemporaryDirectory(dir=workdir) as td:
        t0 = time.time()
        R, O = protocol.run(ag, VC.CKKSParams(), built['data_R'], built['data_O'], dict(opts, solve_seed=seed),
                            dict(opts, entry_workers=lc.THREADS, ledger_path=os.path.join(td, 'ledger.json')),
                            timeout=7200)
        wall = time.time() - t0
    if 'refused' in R:
        return dict(cell=cell['id'], k=k, refused_run=R['refused'], wall=wall)
    # PSI against the plaintext join: the match set and every position
    truth = built['truth']['matched']
    pos_ok = all(R['pi'][i] == O['tau'][i] for i in R['pi'])
    psi_exact = bool(set(R['I']) == truth and pos_ok and R['n'] == len(truth))
    iu = np.triu_indices(p)
    mask = np.ones((p, p), bool)
    mask[:dA, :dA] = False
    sel = mask[iu]
    sg = R['sigma']
    z1 = ((np.asarray(R['G1']) - G)[iu][sel]) / sg['1']
    z2 = np.concatenate([((np.asarray(R['G2']) - G)[iu][sel]) / sg['2'], (np.asarray(R['c2']) - c) / sg['2']])
    zyy = (float(R['yy']) - yy) / sg.get('y', sg['2'])
    ents = VC.entries(built['d_R'], built['d_O'], R['joint'])
    stages = (1, 2) if R['joint'] else (1, 2, 'y')
    used = {s: [e['slot'] for e in VC.release_entries(ents, s)] for s in stages}
    res = R['result'] or {}
    s1 = R['stage1']
    theta_lam = np.linalg.solve(G + s1['lam'] * np.asarray(s1['Psi']), c)
    rec = dict(cell=cell['id'], k=k, solve_seed=seed, git_head=lc.git_head(), threads=lc.THREADS, workers=lc.THREADS,
               wall=wall, n=R['n'], psi_exact=psi_exact, psi_positions_exact=bool(pos_ok),
               psi_stats=R.get('psi_stats'), psi_check=R.get('psi_check'), psi_O=O['debug'].get('psi'),
               timings_O=O['timings'], timings_R=R['timings'],
               marks_O=lc.marks_table(O['marks']), marks_R=lc.marks_table(R['marks']),
               peak_rss_MB=dict(O=max(m[3] for m in O['marks']) / 1024.0, R=max(m[3] for m in R['marks']) / 1024.0),
               messages_R=R['summary'], messages_O=O['summary'],
               phases=lc.measured_phases(R, O),
               calibration=dict(kappa=R['calibration']['kappa'], log2_Omega_f=R['calibration']['log2_Omega_f'],
                                joint=R['joint'], rho_run=R['calibration']['rho_run'],
                                q0_bits=R['calibration']['q0_bits'], q0_need_bits=R['calibration']['q0_need_bits']),
               sigma=sg, lam=float(s1['lam']), kind=s1['kind'], r1=float(s1['r1']), r1_mode=s1['r1_mode'],
               L_hat=float(s1['L_hat']), refused=bool(res.get('refused', True)), gate=res.get('theta_hat_gate'),
               theta=np.asarray(res['theta']).tolist() if 'theta' in res else None,
               half_theta=np.asarray(res['half_theta']).tolist() if 'half_theta' in res else None,
               half_beta=np.asarray(res['half_beta']).tolist() if 'half_beta' in res else None,
               sigma2_trP=res.get('sigma2_trP'), theta_lam=theta_lam.tolist(),
               z_stage1=z1.tolist(), z_stage2=z2.tolist(), z_yy=zyy,
               A_exact=bool(np.allclose(np.asarray(R['G2'])[:dA, :dA], G[:dA, :dA], rtol=1e-9, atol=1e-6)
                            and np.allclose(np.asarray(R['G1'])[:dA, :dA], G[:dA, :dA], rtol=1e-9, atol=1e-6)),
               slots={str(s): slot_stats(R['decoded'][str(s)], used[s], sg[str(s)]) for s in stages})
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', type=int, required=True)
    ap.add_argument('--reps', type=int, default=3)
    ap.add_argument('--cells', nargs='*', default=None)
    a = ap.parse_args()
    cells = [c for c in lc.stage_cells(a.stage) if not a.cells or c['id'] in a.cells]
    workdir = os.path.join(lc.WORK, 'tmp')
    os.makedirs(workdir, exist_ok=True)
    print(f'[stage {a.stage}] resources before: {lc.resources()}', flush=True)
    groups = {}
    for c in cells:
        groups.setdefault(lc.data_key(c), []).append(c)
    for key, cs_ in groups.items():
        todo = [c for c in cs_ if len(lc.load_runs(c['id'])) < a.reps]
        if not todo:
            continue
        t0 = time.time()
        built = lc.build(cs_[0])
        print(f'[stage {a.stage}] data {key}: built in {time.time() - t0:.0f}s, n = {built["ps"]["n"]}, '
              f'n_bar_O = {built["n_bar_O"]}', flush=True)
        for c in todo:
            ref = reference(c, built)
            lc.save_json(ref, os.path.join(lc.WORK, f'ref_{c["id"]}.json'))
            ag = lc.agreement(c, built)
            path = os.path.join(lc.WORK, f'{c["id"]}.jsonl')
            for k in range(len(lc.load_runs(c['id'])), a.reps):
                rec = one_run(c, built, ag, ref, k, workdir)
                rec['resources_after'] = lc.resources()
                with open(path, 'a') as fh:
                    fh.write(json.dumps(ev.jsonable(rec)) + '\n')
                if 'refused_run' in rec:
                    print(f"[{c['id']}] run {k}: REFUSED {rec['refused_run']}", flush=True)
                    continue
                print(f"[{c['id']}] run {k}: wall {rec['wall']:.0f}s psi_exact={rec['psi_exact']} n={rec['n']} "
                      f"lam={rec['lam']:.0f} ({rec['kind']}) refused={rec['refused']} "
                      f"slots1 sd {rec['slots']['1']['sd']:.4f} slots2 sd {rec['slots']['2']['sd']:.4f} "
                      f"agg {rec['phases']['O.aggregate']:.1f}s psi_eval {rec['phases']['O.psi_eval']:.1f}s "
                      f"peak O {rec['peak_rss_MB']['O']:.0f}MB R {rec['peak_rss_MB']['R']:.0f}MB", flush=True)
                gc.collect()
        del built
        gc.collect()
    print(f'[stage {a.stage}] resources after: {lc.resources()}', flush=True)


if __name__ == '__main__':
    main()
