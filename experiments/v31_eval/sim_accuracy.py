"""Section 12.5 under v3.1: coefficient accuracy against intersection size (synthetic population).

Design (as the earlier study, experiments/coef_accuracy_figures.py, with the v3.1 mechanism):
  * population experiments/sim_core.Population(r2, seed=0) (= implementation/vfr/data.population()):
    d_R = 4, d_O = 6 features on realistic raw scales, standardized with public constants and clipped
    to the population's 99.5% bounds; R^2 in {0.3, 0.5, 0.9};
  * n from 100 to 100,000 (the earlier grid plus points around the ridge threshold); 200 cohorts per n,
    each a fresh draw of n matched records, shared across eps; for each eps an independent run of
    v3.1: a stage-1 release of (triu B, C) with s.d. sigma_1, the stage-1 ridge rule, a stage-2 release
    of every O-dependent entry with s.d. sigma_2, the gate theta_hat >= 1, the solve; bias correction off
    (the corrected estimate is computed alongside, for Section 12.6);
  * at R^2 = 0.5, solve.stage2_infer is run on every release (bootstrap with B = 2000 antithetic pairs,
    Wald interval for theta_lambda, bias-aware interval for beta with no public bound on ||beta||);
  * stage-1 only: the fraction of releases with lambda = 0 at n = m * n_off (Section 9.5 check).

    python experiments/v31_eval/sim_accuracy.py run [--reps 200] [--workers 4]
    python experiments/v31_eval/sim_accuracy.py summarize
Raw per-replicate arrays go to experiments/_work/v31_eval/sim/ (git-ignored); summaries to
reports/v31_eval/sim_accuracy.json.
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
os.environ.setdefault('MKL_NUM_THREADS', '1')

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as cm  # noqa: E402
from common import sc, solve  # noqa: E402

NS = [100, 250, 500, 1000, 2500, 5000, 7500, 10000, 15000, 25000, 35000, 50000, 75000, 100000]
R2S = [0.3, 0.5, 0.9]
INFER_R2 = 0.5
NOFF_MULT = [0.6, 0.8, 1.0, 1.2, 1.4]
SIMDIR = os.path.join(cm.WORK, 'sim')
os.makedirs(SIMDIR, exist_ok=True)
P, DA = 11, 5


def population(r2):
    return sc.Population(r2=r2, seed=0)


def cal_for(pop):
    return {e: cm.calibrate(e, pop.B_R, pop.B_O, pop.B_y, pop.d_R, pop.d_O) for e in cm.EPS}


def pop_moments(r2, rows=4_000_000, chunk=250_000):
    """Population second moments of the clipped design: ell = lambda_min(Sigma), the clipped-population
    coefficient (target with clipped y) and test moments (clipped design, unclipped y) for prediction MSE."""
    path = os.path.join(SIMDIR, f'pop_moments_r2_{r2}.npz')
    if os.path.exists(path):
        return dict(np.load(path))
    pop = population(r2)
    rng = np.random.default_rng([cm.BASE_SEED, 99, int(round(r2 * 100))])
    G = np.zeros((P, P)); c = np.zeros(P); cu = np.zeros(P); yyu = 0.0
    for _ in range(rows // chunk):
        X = pop._features(chunk, rng)
        Zs = (X - pop.mu) / pop.s
        y = Zs @ pop.beta_std + pop.noise_sd * rng.standard_normal(chunk)
        zR, zO = Zs[:, :pop.d_R], Zs[:, pop.d_R:]
        nR, nO = np.linalg.norm(zR, axis=1), np.linalg.norm(zO, axis=1)
        zR = zR * np.minimum(1.0, pop.B_R_rows / np.maximum(nR, 1e-300))[:, None]
        zO = zO * np.minimum(1.0, pop.B_O / np.maximum(nO, 1e-300))[:, None]
        Z = np.hstack([np.ones((chunk, 1)), zR, zO])
        yc = np.clip(y, -pop.B_y, pop.B_y)
        G += Z.T @ Z; c += Z.T @ yc; cu += Z.T @ y; yyu += float(y @ y)
    out = dict(Sigma=G / rows, beta_clip=np.linalg.solve(G, c), ell=np.linalg.eigvalsh(G / rows)[0],
               test_G=G / rows, test_c=cu / rows, test_yy=np.array(yyu / rows),
               lam_min_A=np.linalg.eigvalsh(G[:DA, :DA] / rows)[0])
    np.savez(path, **out)
    return out


def pred_mse(th, M):
    """Test MSE of theta (R,p) on the population (clipped design, unclipped y)."""
    return M['test_yy'] - 2 * th @ M['test_c'] + np.einsum('ri,ij,rj->r', th, M['test_G'], th)


def run_cell(r2, n, reps, infer):
    t0 = time.time()
    pop = population(r2)
    cal = cal_for(pop)
    M = pop_moments(r2)
    ss = np.random.SeedSequence([cm.BASE_SEED, int(round(r2 * 100)), n])
    ss_data, ss_inf, *ss_eps = ss.spawn(2 + len(cm.EPS))
    rng_d = np.random.default_rng(ss_data)
    Gs = np.empty((reps, P, P)); cs = np.empty((reps, P)); yys = np.empty(reps)
    b_ols = np.empty((reps, P)); se_ols = np.empty((reps, P))
    for r in range(reps):
        Z, y, _ = pop.draw(n, rng_d)
        G, c, yy = Z.T @ Z, Z.T @ y, float(y @ y)
        Gs[r], cs[r], yys[r] = G, c, yy
        b = np.linalg.solve(G, c)
        rss = yy - 2 * c @ b + b @ G @ b
        b_ols[r] = b
        se_ols[r] = np.sqrt(max(rss, 0.0) / (n - P) * np.diag(np.linalg.inv(G)))
    lmin_G = np.linalg.eigvalsh(Gs)[:, 0]
    nE = len(cm.EPS)
    out = dict(r2=r2, n=n, reps=reps, b_ols=b_ols, se_ols=se_ols, lmin_G=lmin_G,
               mse_ols=pred_mse(b_ols, M),
               theta=np.empty((nE, reps, P)), theta_bc=np.empty((nE, reps, P)), theta_lam=np.empty((nE, reps, P)),
               lam=np.empty((nE, reps)), kind=np.empty((nE, reps), dtype='<U1'), L_hat=np.empty((nE, reps)),
               gate=np.empty((nE, reps)), refused=np.empty((nE, reps), bool), mse_pred=np.empty((nE, reps)))
    if infer:
        for k in ('half_theta', 'half_beta'):
            out[k] = np.empty((nE, reps, P))
        out['sigma2_trP'] = np.empty((nE, reps))
    rng_inf = np.random.default_rng(ss_inf)
    for k, (eps, sse) in enumerate(zip(cm.EPS, ss_eps)):
        cl = cal[eps]
        rng = np.random.default_rng(sse)
        s1 = cm.stage1_batch(Gs, DA, cl, rng, reps)
        E, f, g = cm.stage2_noise(P, DA, cl, rng, reps)
        G2, c2, yy2 = Gs + E, cs + f, yys + g
        st = cm.stage2_solve(G2, c2, s1['lam'], s1['Psi'], cl['sigma2'], bias_correct=True, dA=DA)
        out['theta'][k], out['theta_bc'][k] = st['theta'], st['theta_bc']
        out['gate'][k], out['refused'][k] = st['gate'], st['refused']
        out['lam'][k], out['kind'][k], out['L_hat'][k] = s1['lam'], s1['kind'], s1['L_hat']
        out['theta_lam'][k] = np.linalg.solve(Gs + s1['lam'][:, None, None] * s1['Psi'], cs[..., None])[..., 0]
        out['mse_pred'][k] = pred_mse(st['theta'], M)
        if infer:
            for r in range(reps):
                sd = dict(lam=s1['lam'][r], Psi=s1['Psi'][r], L_hat=s1['L_hat'][r])
                res = solve.stage2_infer(G2[r], c2[r], yy2[r], n, DA, sd, cl['sigma2'], cl['sigma_yy'], pop.B_y,
                                         cm.ALPHA, cm.ALPHA_L, cm.ALPHA_U, np.inf, rng_inf)
                if res['refused']:
                    out['half_theta'][k, r] = np.nan; out['half_beta'][k, r] = np.nan; out['sigma2_trP'][k, r] = np.nan
                    continue
                assert np.allclose(res['theta'], st['theta'][r], rtol=1e-6, atol=1e-9)
                out['half_theta'][k, r] = res['half_theta']
                out['half_beta'][k, r] = res['half_beta']
                out['sigma2_trP'][k, r] = res['sigma2_trP']
    np.savez_compressed(os.path.join(SIMDIR, f'cell_r2_{r2}_n_{n}.npz'), **out)
    return f'r2={r2} n={n}: {time.time() - t0:.0f}s'


def run_noff(eps, mult, reps):
    """Stage-1 rule only, at n = mult * n_off (R^2 = 0.5 population; G does not depend on R^2)."""
    pop = population(0.5)
    M = pop_moments(0.5)
    cl = cm.calibrate(eps, pop.B_R, pop.B_O, pop.B_y, pop.d_R, pop.d_O)
    no, g, r1 = cm.n_off(cl, float(M['ell']), P)
    n = int(round(mult * no))
    ss = np.random.SeedSequence([cm.BASE_SEED, 7, int(eps * 10), int(mult * 100)])
    rd, rn = [np.random.default_rng(s) for s in ss.spawn(2)]
    Gs = np.empty((reps, P, P))
    for r in range(reps):
        Z, _, _ = pop.draw(n, rd)
        Gs[r] = Z.T @ Z
    s1 = cm.stage1_batch(Gs, DA, cl, rn, reps)
    return dict(eps=eps, mult=mult, n=n, n_off=no, frac_off=float(np.mean(s1['lam'] == 0)),
                frac_Psi_O=float(np.mean(s1['kind'] == 'O')), lam_median=float(np.median(s1['lam'])),
                lam_pred=max(0.0, g + r1 - n * float(M['ell'])))


def cmd_run(a):
    for r2 in R2S:
        pop_moments(r2)
    jobs = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        # largest cells first
        for n in sorted(NS, reverse=True):
            for r2 in R2S:
                path = os.path.join(SIMDIR, f'cell_r2_{r2}_n_{n}.npz')
                if os.path.exists(path) and not a.force:
                    continue
                jobs.append(ex.submit(run_cell, r2, n, a.reps, r2 == INFER_R2))
        noff = [ex.submit(run_noff, e, m, a.noff_reps) for e in cm.EPS for m in NOFF_MULT]
        for j in jobs:
            print(j.result(), flush=True)
        res = [j.result() for j in noff]
    cm.save_json(res, os.path.join(SIMDIR, 'noff_check.json'))
    print('done')


# ------------------------------------------------------------------------------------------- summaries
def boot_ci(x, B=2000, seed=0, stat=np.mean):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), (B, len(x)))
    v = np.array([stat(x[i]) for i in idx])
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def summarize_r2(r2):
    pop = population(r2)
    cal = cal_for(pop)
    M = pop_moments(r2)
    beta = pop.beta_true
    bclip = M['beta_clip']
    ell = float(M['ell'])
    sl = slice(1, P)
    rows, percoef = [], []
    rng = np.random.default_rng(1)
    for n in NS:
        D = dict(np.load(os.path.join(SIMDIR, f'cell_r2_{r2}_n_{n}.npz')))
        reps = int(D['reps'])
        W = rng.multinomial(reps, np.full(reps, 1 / reps), size=2000).astype(float)  # paired bootstrap weights
        rr_ols = np.sqrt(np.mean((D['b_ols'][:, sl] - beta[sl]) ** 2, 1))
        sd_ols = D['b_ols'].std(0, ddof=1)
        base = dict(r2=r2, n=n, ols_rmse_median=float(np.median(rr_ols)), ols_rmse_p025=float(np.percentile(rr_ols, 2.5)),
                    ols_rmse_p975=float(np.percentile(rr_ols, 97.5)), ols_rmse_mean=float(rr_ols.mean()),
                    ols_rmse_pooled=float(np.sqrt(np.mean(rr_ols ** 2))), ols_mse_pred=float(D['mse_ols'].mean()),
                    lmin_G_over_n=float(np.median(D['lmin_G']) / n))
        for k, eps in enumerate(cm.EPS):
            cl = cal[eps]
            ok = ~D['refused'][k]
            th, thbc = D['theta'][k], D['theta_bc'][k]
            thz = np.where(ok[:, None], th, 0.0)        # a refused release outputs nothing: theta = 0
            rr = np.sqrt(np.mean((th[ok][:, sl] - beta[sl]) ** 2, 1))
            rrz = np.sqrt(np.mean((thz[:, sl] - beta[sl]) ** 2, 1))
            mse_u = np.mean((th[:, sl] - beta[sl]) ** 2, 1)
            mse_c = np.mean((thbc[:, sl] - beta[sl]) ** 2, 1)
            tl = D['theta_lam'][k]
            msel_u = np.sum((th - tl) ** 2, 1)
            msel_c = np.sum((thbc - tl) ** 2, 1)
            ratio_true = np.sqrt((W @ mse_c) / (W @ mse_u))
            lam = D['lam'][k]
            rmean = [boot_ci(rr, seed=k)] if len(rr) > 2 else [(np.nan, np.nan)]
            row = dict(base, eps=eps, sigma1=cl['sigma1'], sigma2=cl['sigma2'],
                       theta0_pop=n * ell / (2 * cl['sigma2'] * math.sqrt(P)),
                       theta0_median=float(np.median(D['lmin_G'])) / (2 * cl['sigma2'] * math.sqrt(P)),
                       refused=float(np.mean(~ok)), frac_ridge_on=float(np.mean(lam > 0)),
                       frac_Psi_O=float(np.mean(D['kind'][k] == 'O')), lam_median=float(np.median(lam)),
                       lam_p10=float(np.percentile(lam, 10)), lam_p90=float(np.percentile(lam, 90)),
                       frac_Lhat_pos=float(np.mean(D['L_hat'][k] > 0)),
                       gate_median=float(np.median(D['gate'][k])),
                       rmse_median=float(np.median(rr)), rmse_p025=float(np.percentile(rr, 2.5)),
                       rmse_p975=float(np.percentile(rr, 97.5)), rmse_mean=float(rr.mean()),
                       rmse_mean_ci=rmean[0], rmse_pooled=float(np.sqrt(np.mean(rr ** 2))),
                       rmse_median_zero_if_refused=float(np.median(rrz)),
                       ratio_median=float(np.median(rr) / np.median(rr_ols)),
                       ratio_pooled=float(np.sqrt(np.mean(rr ** 2)) / np.sqrt(np.mean(rr_ols ** 2))),
                       rmse_vs_clip_pooled=float(np.sqrt(np.mean((th[ok][:, sl] - bclip[sl]) ** 2))),
                       mse_pred=float(np.mean(D['mse_pred'][k])), pred_ratio=float(np.mean(D['mse_pred'][k]) / D['mse_ols'].mean()),
                       bc_ratio_true=float(np.sqrt(mse_c.mean() / mse_u.mean())),
                       bc_ratio_true_ci=(float(np.percentile(ratio_true, 2.5)), float(np.percentile(ratio_true, 97.5))),
                       bc_msratio_lam=float(msel_c.mean() / msel_u.mean()),
                       shrink_share=float(np.mean(np.sum((tl - D['b_ols']) ** 2, 1)) / np.mean(np.sum((th - beta) ** 2, 1))))
            if 'half_theta' in D:
                ht, hb = D['half_theta'][k][ok], D['half_beta'][k][ok]
                cov_lam = np.abs(th[ok] - tl[ok]) <= ht
                cov_naive = np.abs(th[ok] - beta) <= ht
                fin = np.isfinite(hb)
                cov_b = np.abs(th[ok] - beta) <= hb
                cov_bc = np.abs(th[ok] - bclip) <= hb
                row.update(wald_cov_theta_lam=float(cov_lam.mean()),
                           wald_cov_theta_lam_min=float(cov_lam.mean(0).min()),
                           wald_naive_cov_beta=float(cov_naive[:, sl].mean()),
                           ba_frac_finite=float(fin.mean()),
                           ba_cov_beta=float(cov_b[:, sl].mean()),
                           ba_cov_beta_finite=float(cov_b[:, sl][fin[:, sl]].mean()) if fin[:, sl].any() else None,
                           ba_cov_beta_clip=float(cov_bc[:, sl].mean()),
                           ba_halfwidth_median=float(np.median(hb[:, sl][fin[:, sl]])) if fin[:, sl].any() else None,
                           wald_halfwidth_median=float(np.median(ht[:, sl])),
                           sigma2_trP_median=float(np.nanmedian(D['sigma2_trP'][k])))
            rows.append(row)
            if r2 == INFER_R2:
                err = th[ok] - beta
                rmse_j = np.sqrt(np.mean(err ** 2, 0))
                bw = W[:, ok] if ok.all() else rng.multinomial(ok.sum(), np.full(ok.sum(), 1 / ok.sum()), size=2000).astype(float)
                m1 = bw @ D['b_ols'][ok] / ok.sum(); m2 = bw @ D['b_ols'][ok] ** 2 / ok.sum()
                sdb = np.sqrt(np.maximum(m2 - m1 ** 2, 1e-300) * ok.sum() / (ok.sum() - 1))
                effb = np.sqrt(bw @ err ** 2 / ok.sum()) / sdb
                for j in range(P):
                    percoef.append(dict(r2=r2, n=n, eps=eps, j=j, beta=float(beta[j]), rmse=float(rmse_j[j]),
                                        bias=float(err[:, j].mean()), sd_ols=float(sd_ols[j]),
                                        eff_ratio=float(rmse_j[j] / sd_ols[j]),
                                        eff_ci=(float(np.percentile(effb[:, j], 2.5)), float(np.percentile(effb[:, j], 97.5)))))
    # crossings of the median-RMSE ratio and of the prediction ratio
    cross = {}
    for eps in cm.EPS:
        rs = [r for r in rows if r['eps'] == eps]
        ns = [r['n'] for r in rs]
        cross[str(eps)] = dict(
            coef_2x_median=cm.crossing(ns, [r['ratio_median'] for r in rs], 2.0),
            coef_2x_pooled=cm.crossing(ns, [r['ratio_pooled'] for r in rs], 2.0),
            coef_1p25x_median=cm.crossing(ns, [r['ratio_median'] for r in rs], 1.25),
            pred_1p10=cm.crossing(ns, [r['pred_ratio'] for r in rs], 1.10),
            pred_1p01=cm.crossing(ns, [r['pred_ratio'] for r in rs], 1.01),
            ridge_off_half=cm.crossing(ns, [max(r['frac_ridge_on'], 1e-9) for r in rs], 0.5),
            n_off=cm.n_off(cal[eps], ell, P)[0])
    meta = dict(r2=r2, B_R=pop.B_R, B_O=pop.B_O, B_y=pop.B_y, beta_true=beta, beta_clip=bclip, ell=ell,
                lam_min_A=float(M['lam_min_A']), p=P, d_R_block=DA,
                calibration={str(e): cal[e] for e in cm.EPS},
                n_off={str(e): cm.n_off(cal[e], ell, P) for e in cm.EPS}, types=pop.types)
    return dict(meta=meta, rows=rows, percoef=percoef, crossings=cross)


def cmd_summarize(a):
    out = dict(settings=dict(delta=cm.DELTA, eps=cm.EPS, f1=cm.F1, theta_star=cm.THETA_STAR, kappa_Psi=cm.KAPPA_PSI,
                             alpha=cm.ALPHA, alpha_L=cm.ALPHA_L, alpha_U=cm.ALPHA_U, kappa=cm.KAPPA, ns=NS,
                             bias_correction='off (corrected estimate reported separately)'),
               by_r2={})
    for r2 in R2S:
        out['by_r2'][str(r2)] = summarize_r2(r2)
        print('summarized', r2, flush=True)
    noff = os.path.join(SIMDIR, 'noff_check.json')
    if os.path.exists(noff):
        out['noff_check'] = cm.load_json(noff)
    cm.save_json(out, os.path.join(cm.REPORT, 'sim_accuracy.json'))
    print('wrote', os.path.join(cm.REPORT, 'sim_accuracy.json'))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('run')
    r.add_argument('--reps', type=int, default=200)
    r.add_argument('--noff-reps', type=int, default=200)
    r.add_argument('--workers', type=int, default=4)
    r.add_argument('--force', action='store_true')
    sub.add_parser('summarize')
    a = ap.parse_args()
    {'run': cmd_run, 'summarize': cmd_summarize}[a.cmd](a)


if __name__ == '__main__':
    main()
