"""Section 13.5 under the named settings I and P: coefficient accuracy, prediction, switch-off and coverage on the
synthetic population.

Same design as experiments/v31_eval/sim_accuracy.py (the previous setting), with the same cohorts and the same unit
noise draws (same seeds and the same order of random-number consumption; only sigma_1, sigma_2 and the stage-1 rule
change), so every difference from the previous run is the setting's:
  * population sim_core.Population(r2, seed=0), R^2 in {0.3, 0.5, 0.9}, n from 100 to 100,000, 200 cohorts per n;
  * per cohort and eps an independent run of v3.1 under the setting: stage-1 release (sigma_1 from f_1), the
    setting's stage-1 rule (r_1 exact or union, theta*, alpha_L, kappa_Psi), stage-2 release, the gate, the solve;
  * at R^2 = 0.5, if the setting has intervals, solve.stage2_infer on every release (B = 2000 antithetic pairs,
    no public bound on ||beta||);
  * switch-off check: stage-1 rule only at n = m n_off(setting), m in {0.6, 0.8, 1.0, 1.2, 1.4}, 200 cohorts.

    python experiments/v31_settings/sim_settings.py run --setting I [--reps 200] [--workers 4]
    python experiments/v31_settings/sim_settings.py summarize --setting I
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scommon as cs  # noqa: E402
from scommon import ev, solve  # noqa: E402

import sim_accuracy as sa  # noqa: E402  (experiments/v31_eval/sim_accuracy.py)

NS = sa.NS
R2S = sa.R2S
INFER_R2 = sa.INFER_R2
NOFF_MULT = sa.NOFF_MULT
P, DA = 11, 5


def simdir(name):
    d = os.path.join(cs.WORK, f'sim_{name}')
    os.makedirs(d, exist_ok=True)
    return d


def cal_for(pop, st):
    return {e: cs.calibrate(e, pop.B_R, pop.B_O, pop.B_y, pop.d_R, pop.d_O, st['f1']) for e in cs.EPS}


def run_cell(name, st, r2, n, reps, infer):
    t0 = time.time()
    pop = sa.population(r2)
    cal = cal_for(pop, st)
    M = sa.pop_moments(r2)
    ss = np.random.SeedSequence([cs.BASE_SEED, int(round(r2 * 100)), n])       # = sim_accuracy.py
    ss_data, ss_inf, *ss_eps = ss.spawn(2 + len(cs.EPS))
    rng_d = np.random.default_rng(ss_data)
    Gs = np.empty((reps, P, P)); cvec = np.empty((reps, P)); yys = np.empty(reps)
    b_ols = np.empty((reps, P))
    for r in range(reps):
        Z, y, _ = pop.draw(n, rng_d)
        Gs[r], cvec[r], yys[r] = Z.T @ Z, Z.T @ y, float(y @ y)
        b_ols[r] = np.linalg.solve(Gs[r], cvec[r])
    nE = len(cs.EPS)
    out = dict(r2=r2, n=n, reps=reps, b_ols=b_ols, lmin_G=np.linalg.eigvalsh(Gs)[:, 0],
               mse_ols=sa.pred_mse(b_ols, M),
               theta=np.empty((nE, reps, P)), theta_lam=np.empty((nE, reps, P)),
               lam=np.empty((nE, reps)), kind=np.empty((nE, reps), dtype='<U1'), L_hat=np.empty((nE, reps)),
               gate=np.empty((nE, reps)), refused=np.empty((nE, reps), bool), mse_pred=np.empty((nE, reps)))
    infer = infer and st['intervals']
    if infer:
        for k in ('half_theta', 'half_beta'):
            out[k] = np.empty((nE, reps, P))
        out['sigma2_trP'] = np.empty((nE, reps))
    rng_inf = np.random.default_rng(ss_inf)
    for k, (eps, sse) in enumerate(zip(cs.EPS, ss_eps)):
        cl = cal[eps]
        rng = np.random.default_rng(sse)
        s1 = cs.stage1_batch(Gs, DA, cl, rng, reps, st)
        E, f, g = ev.stage2_noise(P, DA, cl, rng, reps)
        G2, c2, yy2 = Gs + E, cvec + f, yys + g
        so = ev.stage2_solve(G2, c2, s1['lam'], s1['Psi'], cl['sigma2'])
        out['theta'][k] = so['theta']
        out['gate'][k], out['refused'][k] = so['gate'], so['refused']
        out['lam'][k], out['kind'][k], out['L_hat'][k] = s1['lam'], s1['kind'], s1['L_hat']
        out['theta_lam'][k] = np.linalg.solve(Gs + s1['lam'][:, None, None] * s1['Psi'], cvec[..., None])[..., 0]
        thz = np.where(so['refused'][:, None], 0.0, so['theta'])
        out['mse_pred'][k] = sa.pred_mse(thz, M)
        if infer:
            for r in range(reps):
                sd = dict(lam=s1['lam'][r], Psi=s1['Psi'][r], L_hat=s1['L_hat'][r])
                res = solve.stage2_infer(G2[r], c2[r], yy2[r], n, DA, sd, cl['sigma2'], cl['sigma_yy'], pop.B_y,
                                         st['alpha'], st['alpha_L'], st['alpha_U'], np.inf, rng_inf)
                if res['refused']:
                    out['half_theta'][k, r] = np.nan; out['half_beta'][k, r] = np.nan; out['sigma2_trP'][k, r] = np.nan
                    continue
                assert np.allclose(res['theta'], so['theta'][r], rtol=1e-6, atol=1e-9)
                out['half_theta'][k, r] = res['half_theta']
                out['half_beta'][k, r] = res['half_beta']
                out['sigma2_trP'][k, r] = res['sigma2_trP']
    np.savez_compressed(os.path.join(simdir(name), f'cell_r2_{r2}_n_{n}.npz'), **out)
    return f'[{name}] r2={r2} n={n}: {time.time() - t0:.0f}s'


def run_noff(name, st, eps, mult, reps):
    pop = sa.population(0.5)
    M = sa.pop_moments(0.5)
    cl = cs.calibrate(eps, pop.B_R, pop.B_O, pop.B_y, pop.d_R, pop.d_O, st['f1'])
    no, g, r1 = cs.n_off(cl, float(M['ell']), P, DA, st)
    n = int(round(mult * no))
    ss = np.random.SeedSequence([cs.BASE_SEED, 7, int(eps * 10), int(mult * 100)])     # = sim_accuracy.py
    rd, rn = [np.random.default_rng(s) for s in ss.spawn(2)]
    Gs = np.empty((reps, P, P))
    for r in range(reps):
        Z, _, _ = pop.draw(n, rd)
        Gs[r] = Z.T @ Z
    s1 = cs.stage1_batch(Gs, DA, cl, rn, reps, st)
    return dict(eps=eps, mult=mult, n=n, n_off=no, g_star=g, r1=r1, frac_off=float(np.mean(s1['lam'] == 0)),
                frac_Psi_O=float(np.mean(s1['kind'] == 'O')), lam_median=float(np.median(s1['lam'])),
                lam_pred=max(0.0, g + r1 - n * float(M['ell'])))


def get_setting(name):
    if name == 'prev':
        return dict(cs.PREV)
    return cs.load_settings()[name]


def cmd_run(a):
    st = get_setting(a.setting)
    print(a.setting, cs.label(st), flush=True)
    solve.r1_quantile_exact(DA, P - DA, st['alpha_L']) if st['r1_mode'] == 'exact' else None
    for r2 in R2S:
        sa.pop_moments(r2)
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        jobs = []
        for n in sorted(NS, reverse=True):
            for r2 in R2S:
                if os.path.exists(os.path.join(simdir(a.setting), f'cell_r2_{r2}_n_{n}.npz')) and not a.force:
                    continue
                jobs.append(ex.submit(run_cell, a.setting, st, r2, n, a.reps, r2 == INFER_R2))
        noff = [ex.submit(run_noff, a.setting, st, e, m, a.noff_reps) for e in cs.EPS for m in NOFF_MULT]
        for j in jobs:
            print(j.result(), flush=True)
        res = [j.result() for j in noff]
    ev.save_json(res, os.path.join(simdir(a.setting), 'noff_check.json'))
    print('done', flush=True)


# ------------------------------------------------------------------------------------------- summaries
def summarize_r2(name, st, r2):
    pop = sa.population(r2)
    cal = cal_for(pop, st)
    M = sa.pop_moments(r2)
    beta = pop.beta_true
    bclip = M['beta_clip']
    ell = float(M['ell'])
    sl = slice(1, P)
    rows, percoef = [], []
    rng = np.random.default_rng(1)
    for n in NS:
        D = dict(np.load(os.path.join(simdir(name), f'cell_r2_{r2}_n_{n}.npz')))
        reps = int(D['reps'])
        rr_ols = np.sqrt(np.mean((D['b_ols'][:, sl] - beta[sl]) ** 2, 1))
        sd_ols = D['b_ols'].std(0, ddof=1)
        base = dict(r2=r2, n=n, ols_rmse_median=float(np.median(rr_ols)), ols_rmse_pooled=float(np.sqrt(np.mean(rr_ols ** 2))),
                    ols_rmse_p025=float(np.percentile(rr_ols, 2.5)), ols_rmse_p975=float(np.percentile(rr_ols, 97.5)),
                    ols_mse_pred=float(D['mse_ols'].mean()), lmin_G_over_n=float(np.median(D['lmin_G']) / n))
        for k, eps in enumerate(cs.EPS):
            cl = cal[eps]
            ok = ~D['refused'][k]
            th = D['theta'][k]
            thz = np.where(ok[:, None], th, 0.0)
            rrz = np.sqrt(np.mean((thz[:, sl] - beta[sl]) ** 2, 1))       # refused -> theta = 0
            rr = np.sqrt(np.mean((th[ok][:, sl] - beta[sl]) ** 2, 1)) if ok.any() else np.array([np.nan])
            tl = D['theta_lam'][k]
            lam = D['lam'][k]
            row = dict(base, eps=eps, sigma1=cl['sigma1'], sigma2=cl['sigma2'],
                       theta0_median=float(np.median(D['lmin_G'])) / (2 * cl['sigma2'] * math.sqrt(P)),
                       refused=float(np.mean(~ok)), frac_ridge_on=float(np.mean(lam > 0)),
                       frac_Psi_O=float(np.mean(D['kind'][k] == 'O')), lam_median=float(np.median(lam)),
                       lam_p10=float(np.percentile(lam, 10)), lam_p90=float(np.percentile(lam, 90)),
                       frac_Lhat_pos=float(np.mean(D['L_hat'][k] > 0)), gate_median=float(np.median(D['gate'][k])),
                       rmse_median=float(np.median(rrz)), rmse_p025=float(np.percentile(rrz, 2.5)),
                       rmse_p975=float(np.percentile(rrz, 97.5)), rmse_pooled=float(np.sqrt(np.mean(rrz ** 2))),
                       rmse_median_passed=float(np.median(rr)),
                       ratio_median=float(np.median(rrz) / np.median(rr_ols)),
                       ratio_pooled=float(np.sqrt(np.mean(rrz ** 2)) / np.sqrt(np.mean(rr_ols ** 2))),
                       mse_pred=float(np.mean(D['mse_pred'][k])),
                       pred_ratio=float(np.mean(D['mse_pred'][k]) / D['mse_ols'].mean()),
                       shrink_share=float(np.mean(np.sum((tl - D['b_ols']) ** 2, 1)) / np.mean(np.sum((thz - beta) ** 2, 1))))
            if 'half_theta' in D and ok.any():
                ht, hb = D['half_theta'][k][ok], D['half_beta'][k][ok]
                cov_lam = np.abs(th[ok] - tl[ok]) <= ht
                cov_naive = np.abs(th[ok] - beta) <= ht
                fin = np.isfinite(hb)
                cov_b = np.abs(th[ok] - beta) <= hb
                cov_bc = np.abs(th[ok] - bclip) <= hb
                off = lam[ok] == 0
                row.update(wald_cov_theta_lam=float(cov_lam.mean()),
                           wald_cov_theta_lam_min=float(cov_lam.mean(0).min()),
                           wald_cov_theta_lam_ridge_on=float(cov_lam[~off].mean()) if (~off).any() else None,
                           wald_cov_theta_lam_ridge_off=float(cov_lam[off].mean()) if off.any() else None,
                           wald_naive_cov_beta=float(cov_naive[:, sl].mean()),
                           ba_frac_finite=float(fin.mean()),
                           ba_cov_beta=float(cov_b[:, sl].mean()),
                           ba_cov_beta_finite=float(cov_b[:, sl][fin[:, sl]].mean()) if fin[:, sl].any() else None,
                           ba_cov_beta_clip=float(cov_bc[:, sl].mean()),
                           ba_cov_beta_ridge_off=float(cov_b[off][:, sl].mean()) if off.any() else None,
                           ba_cov_beta_clip_ridge_off=float(cov_bc[off][:, sl].mean()) if off.any() else None,
                           ba_halfwidth_median=float(np.median(hb[:, sl][fin[:, sl]])) if fin[:, sl].any() else None,
                           wald_halfwidth_median=float(np.median(ht[:, sl])),
                           sigma2_trP_median=float(np.nanmedian(D['sigma2_trP'][k])))
            rows.append(row)
            if r2 == INFER_R2 and ok.any():
                err = th[ok] - beta
                rmse_j = np.sqrt(np.mean(err ** 2, 0))
                for j in range(P):
                    percoef.append(dict(r2=r2, n=n, eps=eps, j=j, beta=float(beta[j]), rmse=float(rmse_j[j]),
                                        sd_ols=float(sd_ols[j]), eff_ratio=float(rmse_j[j] / sd_ols[j])))
    cross = {}
    for eps in cs.EPS:
        rs = [r for r in rows if r['eps'] == eps]
        ns = [r['n'] for r in rs]
        cross[str(eps)] = dict(
            coef_2x_median=ev.crossing(ns, [r['ratio_median'] for r in rs], 2.0),
            coef_2x_pooled=ev.crossing(ns, [r['ratio_pooled'] for r in rs], 2.0),
            coef_1p25x_median=ev.crossing(ns, [r['ratio_median'] for r in rs], 1.25),
            pred_1p10=ev.crossing(ns, [r['pred_ratio'] for r in rs], 1.10),
            pred_1p01=ev.crossing(ns, [r['pred_ratio'] for r in rs], 1.01),
            ridge_off_half=ev.crossing(ns, [max(r['frac_ridge_on'], 1e-9) for r in rs], 0.5),
            n_off=cs.n_off(cal[eps], ell, P, DA, st)[0])
    meta = dict(r2=r2, setting=st, B_R=pop.B_R, B_O=pop.B_O, B_y=pop.B_y, beta_true=beta, beta_clip=bclip, ell=ell,
                p=P, d_R_block=DA, calibration={str(e): cal[e] for e in cs.EPS},
                n_off={str(e): cs.n_off(cal[e], ell, P, DA, st) for e in cs.EPS},
                r1_unit=cs.r1_unit(P, DA, st))
    return dict(meta=meta, rows=rows, percoef=percoef, crossings=cross)


def theta_lambda_held(name, st, r2=0.5):
    """Validity ratio lambda_min(G + lambda Psi)/(2 sigma_2 sqrt p) that the stage-1 ridge holds (ridge-on releases)."""
    pop = sa.population(r2)
    cal = cal_for(pop, st)
    out = {}
    for eps in cs.EPS:
        vals = []
        for n in NS:
            D = np.load(os.path.join(simdir(name), f'cell_r2_{r2}_n_{n}.npz'))
            k = cs.EPS.index(eps)
            lam, kind = D['lam'][k], D['kind'][k]
            on = lam > 0
            if not on.any():
                continue
            lmin = D['lmin_G']           # lambda_min(G + lambda I) = lambda_min(G) + lambda (Psi = I releases only)
            sel = on & (kind == 'I')
            vals += list((lmin[sel] + lam[sel]) / (2 * cal[eps]['sigma2'] * math.sqrt(P)))
        out[str(eps)] = dict(median=float(np.median(vals)) if vals else None,
                             p10=float(np.percentile(vals, 10)) if vals else None,
                             p90=float(np.percentile(vals, 90)) if vals else None, count=len(vals))
    return out


def cmd_summarize(a):
    st = get_setting(a.setting)
    out = dict(setting_name=a.setting, settings=dict(st, delta=cs.DELTA, eps=cs.EPS, kappa=ev.KAPPA, ns=NS,
                                                     r1_unit=cs.r1_unit(P, DA, st)), by_r2={})
    for r2 in R2S:
        out['by_r2'][str(r2)] = summarize_r2(a.setting, st, r2)
        print('summarized', r2, flush=True)
    out['theta_lambda_held'] = theta_lambda_held(a.setting, st)
    noff = os.path.join(simdir(a.setting), 'noff_check.json')
    if os.path.exists(noff):
        out['noff_check'] = ev.load_json(noff)
    path = os.path.join(cs.REPORT, f'sim_{a.setting}.json')
    ev.save_json(out, path)
    print('wrote', path)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('run')
    r.add_argument('--setting', required=True)
    r.add_argument('--reps', type=int, default=200)
    r.add_argument('--noff-reps', type=int, default=200)
    r.add_argument('--workers', type=int, default=4)
    r.add_argument('--force', action='store_true')
    s = sub.add_parser('summarize')
    s.add_argument('--setting', required=True)
    a = ap.parse_args()
    {'run': cmd_run, 'summarize': cmd_summarize}[a.cmd](a)


if __name__ == '__main__':
    main()
