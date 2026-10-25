"""Summaries and gates of the large-scale evaluation (one stage at a time).

For every cell of a stage: measured per-phase wall and CPU time against the cost model (lcommon.predict), peak memory
per party, traffic per message type against the model's exact counts, PSI correctness, the decrypted noise (all
slots of every release, standardized by the calibrated sigma; KS against N(0, 1)), lambda and the estimate against a
plaintext simulation of the same mechanism on the same data (4,000 draws, the runs' sigma), Wald coverage of
theta_lambda under I (the encrypted runs, and 400 plaintext releases with the full bootstrap on the same data),
accuracy (synthetic: coefficient RMSE against the true beta and population prediction error against non-private
least squares on the same cohort; 3droad: test MSE ratio against the arm's own non-private fit), and, in regime B,
whether the ridge is off and the widths of the bias-aware interval.

Gate of stages 1-3 (fixed before the runs; README of reports/large):
  * cost: every phase whose prediction or mean measurement is at least 2 s, and O's total, within 25% of the model;
  * PSI exact (match set and every position) in every run;
  * lambda inside the simulated 95% range in every run but at most max(1, 10%) of a stage's runs, and the estimates
    inside the simulated 95% ranges in at least 85% of (run, coefficient) pairs, with |z| < 4 against the simulated law
    given the run's lambda;
  * noise s.d. (pooled over the runs of a cell, per release) within 3% of 1.

    python experiments/large/summarize_large.py --stage 1 [--sim 4000] [--cov 400] [--workers 4]
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lcommon as lc  # noqa: E402
from lcommon import cs, ev, mc  # noqa: E402

MB = 2 ** 20
GATE_MIN_S = 2.0
GATE_TOL = 0.25
MSG_MAP = {'keygen_R': 'R->O keygen (pk + rotation-key shares)', 'keygen_O': 'O->R keygen (pk share)',
           'oprf_request': 'R->O OPRF', 'oprf_response': 'O->R OPRF', 'psi_keys': 'R->O BFV keys',
           'psi_query': 'R->O PSI query', 'psi_response': 'O->R PSI reply', 'upload': 'R->O upload',
           'release': 'O->R releases'}


def pooled(stats_list):
    """Pool per-run slot statistics (n, mean, sd): overall mean and s.d."""
    n = np.array([s['n'] for s in stats_list], float)
    m = np.array([s['mean'] for s in stats_list])
    sd = np.array([s['sd'] for s in stats_list])
    N = n.sum()
    mu = float((n * m).sum() / N)
    var = float((n * (sd ** 2 + m ** 2)).sum() / N - mu ** 2)
    return dict(n=int(N), mean=mu, sd=math.sqrt(var), ks_p_min=float(min(s['ks_p'] for s in stats_list)),
                ks_p_runs=[s['ks_p'] for s in stats_list], max_abs=float(max(s['max_abs'] for s in stats_list)))


def simulate(G, c, cal, st, S, seed, dA):
    rng = np.random.default_rng(seed)
    p = G.shape[0]
    s1 = cs.stage1_batch(G, dA, cal, rng, S, st)
    E, f, g = ev.stage2_noise(p, dA, cal, rng, S)
    so = ev.stage2_solve(G[None] + E, c[None] + f, s1['lam'], s1['Psi'], cal['sigma2'])
    return s1, so


def cost_block(cell, runs, ref, inputs='published', polys='model'):
    pr = lc.predict(ref['n_bar_O'], cell['n_R_max'], inputs=inputs, polys=polys)
    meas = {k: float(np.mean([r['phases'][k] for r in runs])) for k in runs[0]['phases']}
    rows, fails = {}, []
    for k, v in pr['phases'].items():
        m = meas[k]
        ratio = m / v if v > 0 else float('nan')
        gated = max(v, m) >= GATE_MIN_S
        ok = (not gated) or abs(ratio - 1) <= GATE_TOL
        rows[k] = dict(predicted=v, measured=m, ratio=ratio, gated=gated, ok=ok,
                       measured_runs=[r['phases'][k] for r in runs])
        if not ok:
            fails.append(k)
    tot_p = sum(v for k, v in pr['phases'].items() if k.startswith('O.'))
    tot_m = sum(v for k, v in meas.items() if k.startswith('O.'))
    tot_ok = abs(tot_m / tot_p - 1) <= GATE_TOL
    if not tot_ok:
        fails.append('O.total')
    # CPU time and peak memory from the phase marks
    cpu = {}
    for side in ('O', 'R'):
        acc = {}
        for r in runs:
            for m in r[f'marks_{side}']:
                acc.setdefault(m['phase'], []).append((m['wall'], m['cpu'], m['peak_rss_MB']))
        cpu[side] = {ph: dict(wall=float(np.mean([x[0] for x in v])), cpu=float(np.mean([x[1] for x in v])),
                              peak_rss_MB=float(np.max([x[2] for x in v]))) for ph, v in acc.items()}
    # traffic per message type (R's view) against the model
    traffic = {}
    for mt, s in runs[0]['messages_R'].items():
        traffic[mt] = dict(count=s['count'], MB=s['bytes'] / MB,
                           model_MB=pr['bytes'][MSG_MAP[mt]] / MB if mt in MSG_MAP else None)
    comm = sum(s['bytes'] for s in runs[0]['messages_R'].values())
    return dict(phases=rows, O_total=dict(predicted=tot_p, measured=tot_m, ratio=tot_m / tot_p, ok=tot_ok),
                wall_run=dict(mean=float(np.mean([r['wall'] for r in runs])), runs=[r['wall'] for r in runs]),
                cpu_marks=cpu, peak_rss_MB=dict(O=float(max(r['peak_rss_MB']['O'] for r in runs)),
                                                R=float(max(r['peak_rss_MB']['R'] for r in runs))),
                traffic=traffic, comm_MB=comm / MB, comm_model_MB=pr['comm'] / MB,
                model=dict(D=pr['D'], alpha=pr['alpha'], W=pr['W'], tables=pr['tables'], felts=pr['felts'], m=pr['m'],
                           cpu_O_1core=pr['cpu_O_1core'], cpu_R_1core=pr['cpu_R_1core'], mem_O_MB=sum(pr['mem'].values()) / MB),
                inputs=inputs, polys=polys, fails=fails, ok=not fails)


def stats_block(cell, runs, ref, S, ncov, workers):
    G, c, yy, n = np.asarray(ref['G']), np.asarray(ref['c']), ref['yy'], ref['n']
    p, dA = G.shape[0], ref['d_R'] + 1
    st = lc.setting_dict(cell['setting'])
    sig = runs[0]['sigma']
    cal = dict(sigma1=sig['1'], sigma2=sig['2'], sigma_yy=sig.get('y', sig['2']))
    s1s, so = simulate(G, c, cal, st, S, 3, dA)
    lq = np.quantile(s1s['lam'], [0.025, 0.5, 0.975])
    okm = ~so['refused']
    th_ok = so['theta'][okm]
    tq = np.quantile(th_ok, [0.025, 0.975], axis=0) if okm.any() else None
    rng = np.random.default_rng(4)
    per_run = []
    for r in runs:
        d = dict(k=r['k'], lam=r['lam'], kind=r['kind'], refused=r['refused'],
                 lam_in_95=bool(lq[0] - 1e-9 * max(1, lq[2]) <= r['lam'] <= lq[2] + 1e-9 * max(1, lq[2])),
                 lam_quantile=float(np.mean(s1s['lam'] <= r['lam'])))
        if not r['refused']:
            th = np.array(r['theta'])
            d['theta_in_95'] = ((th >= tq[0]) & (th <= tq[1])).tolist()
            Psi_r = np.eye(p) if r['kind'] == 'I' else ev.pi_O(p, dA)
            Ec, fc, _ = ev.stage2_noise(p, dA, cal, rng, S)
            stc = ev.stage2_solve(G[None] + Ec, c[None] + fc, np.full(S, r['lam']), np.broadcast_to(Psi_r, (S, p, p)),
                                  cal['sigma2'])
            tc = stc['theta'][~stc['refused']]
            d['z_given_lambda'] = ((th - tc.mean(0)) / tc.std(0)).tolist()
            if r.get('half_theta') is not None:
                d['wald_covered'] = (np.abs(th - np.array(r['theta_lam'])) <= np.array(r['half_theta'])).tolist()
        per_run.append(d)
    out = dict(sigma=sig, sim=dict(draws=S, lam_q=lq.tolist(), ridge_on=float(np.mean(s1s['lam'] > 0)),
                                   psiO=float(np.mean(s1s['kind'] == 'O')), refused=float(so['refused'].mean()),
                                   r1=float(s1s['r1'])),
               runs=per_run)
    # noise
    stages = sorted(runs[0]['slots'])
    out['noise'] = {s: pooled([r['slots'][s] for r in runs]) for s in stages}
    from scipy import stats
    z1 = np.concatenate([r['z_stage1'] for r in runs])
    z2 = np.concatenate([r['z_stage2'] for r in runs])
    out['noise_entries'] = dict(stage1=dict(n=int(z1.size), mean=float(z1.mean()), sd=float(z1.std()),
                                            ks_p=float(stats.kstest(z1, 'norm').pvalue)),
                                stage2=dict(n=int(z2.size), mean=float(z2.mean()), sd=float(z2.std()),
                                            ks_p=float(stats.kstest(z2, 'norm').pvalue)))
    out['A_exact'] = all(r['A_exact'] for r in runs)
    out['psi_exact'] = all(r['psi_exact'] for r in runs)
    out['psi'] = dict(stats=[r['psi_stats'] for r in runs], check=runs[0]['psi_check'], O=runs[0]['psi_O'])
    # accuracy
    th_ols = np.asarray(ref['theta_ols'])
    acc = {}
    if cell['dataset'] == 'synthetic':
        bt = np.asarray(ref['beta_true'])
        M = lc.pop_moments(cell['kind'])
        rm = lambda th: float(np.sqrt(np.mean((np.asarray(th) - bt) ** 2)))
        import p2common as pc
        mse_ols = float(pc.pred_mse(th_ols[None], M)[0])
        acc['rmse_ols'] = rm(th_ols)
        acc['runs'] = [dict(rmse=rm(r['theta']), pred_ratio=float(pc.pred_mse(np.asarray(r['theta'])[None], M)[0] / mse_ols))
                       if not r['refused'] else dict(refused=True) for r in runs]
        thz = np.where(so['refused'][:, None], 0.0, so['theta'])
        rms = np.sqrt(np.mean((thz - bt) ** 2, axis=1))
        pr_ = pc.pred_mse(thz, M) / mse_ols
        acc['sim'] = dict(rmse_median=float(np.median(rms)), rmse_mean=float(rms.mean()),
                          rmse_ratio_to_ols=float(np.median(rms) / acc['rmse_ols']),
                          pred_ratio_median=float(np.median(pr_)), pred_ratio_mean=float(pr_.mean()))
    else:
        _, _, _, _, _, test, _ = lc.road3d(cell['n'], cell['n_cohort'], cell['seed'])
        mnp = ref['test_mse_np']
        acc['test_mse_np'] = mnp
        acc['test_mse_trivial'] = ref['test_mse_trivial']
        acc['runs'] = [dict(test_mse=ev.test_mse(test['Z'], test['y'], np.asarray(r['theta'])),
                            ratio=ev.test_mse(test['Z'], test['y'], np.asarray(r['theta'])) / mnp)
                       if not r['refused'] else dict(refused=True) for r in runs]
        thz = np.where(so['refused'][:, None], 0.0, so['theta'])
        tm = ev.test_mse(test['Z'], test['y'], thz)
        acc['sim'] = dict(ratio_median=float(np.median(tm / mnp)), ratio_mean=float(np.mean(tm / mnp)))
    out['accuracy'] = acc
    # Wald coverage under I: plaintext releases with the full bootstrap on the same data
    if st['intervals'] and ncov:
        t0 = time.time()
        rel = mc.releases_parallel(G, c, yy, n, dA, cal, st, ref['bounds']['B_y'], ncov, [lc.SEED_DATA.get('B', 0), 7,
                                   list(lc.CELLS).index(cell['id'])], workers=workers, chunk=max(1, ncov // workers),
                                   beta_ref=ref.get('beta_true'))
        cov = mc.coverage_summary(rel['covered'])
        out['coverage_sim'] = dict(releases=ncov, wald=dict(coverage=cov.get('coverage'), clustered=cov.get('clustered'),
                                                           per_coef_min=cov.get('per_coef_min'), per_coef_max=cov.get('per_coef_max')),
                                   ridge_on=float(np.mean(rel['lam'] > 0)), refused=int(rel['refused'].sum()),
                                   half_beta_finite=float(np.mean(np.isfinite(rel['half_beta']).all(axis=1))),
                                   half_beta_median=float(np.nanmedian(np.where(np.isfinite(rel['half_beta']), rel['half_beta'], np.nan))),
                                   half_theta_median=float(np.nanmedian(rel['half_theta'])), seconds=time.time() - t0)
        if 'covered_beta' in rel:
            cb = mc.coverage_summary(rel['covered_beta'])
            out['coverage_sim']['beta'] = dict(coverage=cb.get('coverage'), clustered=cb.get('clustered'))
    enc_cov = [x for d in per_run for x in d.get('wald_covered', [])]
    out['coverage_encrypted'] = dict(covered=int(sum(enc_cov)), total=len(enc_cov)) if enc_cov else None
    if st['intervals']:
        hb = [np.array(r['half_beta'], dtype=float) for r in runs if r.get('half_beta') is not None]
        ht = [np.array(r['half_theta'], dtype=float) for r in runs if r.get('half_theta') is not None]
        out['intervals'] = dict(half_beta_runs=[h.tolist() for h in hb], half_theta_runs=[h.tolist() for h in ht],
                                half_beta_median=float(np.median(np.concatenate(hb))) if hb else None,
                                half_beta_median_finite=float(np.median(np.concatenate(hb)[np.isfinite(np.concatenate(hb))])) if hb and np.isfinite(np.concatenate(hb)).any() else None,
                                half_beta_finite=[bool(np.isfinite(h).all()) for h in hb],
                                ridge_off_runs=int(sum(r['lam'] == 0 for r in runs)), runs=len(runs))
    return out


def gate(stage, cells_out):
    runs_total = sum(len(v['stats']['runs']) for v in cells_out.values())
    lam_miss = sum(not d['lam_in_95'] for v in cells_out.values() for d in v['stats']['runs'])
    th_in = [x for v in cells_out.values() for d in v['stats']['runs'] for x in d.get('theta_in_95', [])]
    zs = [abs(x) for v in cells_out.values() for d in v['stats']['runs'] for x in d.get('z_given_lambda', [])]
    noise_sd = {cid: {s: v['stats']['noise'][s]['sd'] for s in v['stats']['noise']} for cid, v in cells_out.items()}
    rem = [v.get('cost_remeasured') for v in cells_out.values()]
    checks = dict(
        cost=all(v['cost']['ok'] for v in cells_out.values()),
        psi_exact=all(v['stats']['psi_exact'] for v in cells_out.values()),
        lam=lam_miss <= max(1, int(0.1 * runs_total)),
        theta=(np.mean(th_in) >= 0.85 if th_in else True) and (max(zs) < 4 if zs else True),
        noise=all(abs(sd - 1) <= 0.03 for d in noise_sd.values() for sd in d.values()))
    other = all(v for k, v in checks.items() if k != 'cost')
    cost_rem = all(r['ok'] for r in rem) if all(r is not None for r in rem) else None
    return dict(stage=stage, passed=all(checks.values()), checks=checks, lam_misses=lam_miss, runs=runs_total,
                cost_remeasured_ok=cost_rem, passed_with_remeasured_inputs=(other and cost_rem) if cost_rem is not None else None,
                cost_fails_remeasured={cid: v['cost_remeasured']['fails'] for cid, v in cells_out.items()
                                       if v.get('cost_remeasured') and v['cost_remeasured']['fails']},
                theta_in_95=float(np.mean(th_in)) if th_in else None, max_abs_z=float(max(zs)) if zs else None,
                noise_sd=noise_sd, cost_fails={cid: v['cost']['fails'] for cid, v in cells_out.items() if v['cost']['fails']})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', type=int, required=True)
    ap.add_argument('--sim', type=int, default=4000)
    ap.add_argument('--cov', type=int, default=400)
    ap.add_argument('--workers', type=int, default=4)
    a = ap.parse_args()
    out = {}
    for cell in lc.stage_cells(a.stage):
        runs = [r for r in lc.load_runs(cell['id']) if 'refused_run' not in r]
        if not runs:
            continue
        ref = lc.load_json(os.path.join(lc.WORK, f'ref_{cell["id"]}.json'))
        t0 = time.time()
        have_rem = os.path.exists(os.path.join(lc.REPORT, 'micro_t4_remeasured.json'))
        out[cell['id']] = dict(cell=cell, n=ref['n'], n_bar_O=ref['n_bar_O'], cost=cost_block(cell, runs, ref),
                               cost_remeasured=cost_block(cell, runs, ref, 'remeasured', 'fill') if have_rem else None,
                               stats=stats_block(cell, runs, ref, a.sim, a.cov, a.workers), git_heads=sorted({r['git_head'] for r in runs}))
        c = out[cell['id']]
        print(f"[{cell['id']}] {len(runs)} runs, {time.time() - t0:.0f}s: cost ok={c['cost']['ok']} "
              f"(O total {c['cost']['O_total']['measured']:.1f}/{c['cost']['O_total']['predicted']:.1f}s, fails {c['cost']['fails']}); "
              f"psi exact={c['stats']['psi_exact']}; noise sd " +
              ', '.join(f"{s}: {v['sd']:.4f}" for s, v in c['stats']['noise'].items()) +
              f"; lam in 95%: {sum(d['lam_in_95'] for d in c['stats']['runs'])}/{len(runs)}", flush=True)
    g = gate(a.stage, out) if a.stage <= 3 else None
    res = dict(stage=a.stage, stamp=time.strftime('%Y-%m-%d %H:%M'), cells=out, gate=g, threads=lc.THREADS)
    lc.save_json(res, os.path.join(lc.REPORT, f'stage{a.stage}.json'))
    if g:
        print('GATE', 'PASSED' if g['passed'] else 'FAILED', g['checks'], 'lam misses', g['lam_misses'], '/', g['runs'],
              'theta in 95%', g['theta_in_95'], 'max|z|', g['max_abs_z'], 'cost fails', g['cost_fails'], flush=True)
        print('  with re-measured inputs: cost ok', g['cost_remeasured_ok'], 'passed', g['passed_with_remeasured_inputs'],
              'fails', g['cost_fails_remeasured'], flush=True)


if __name__ == '__main__':
    main()
