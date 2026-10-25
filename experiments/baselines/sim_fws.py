"""Baseline 2 on the synthetic population: Ferrando-Wang-Sheldon (FWS) parametric-bootstrap intervals against v3.1's
intervals under setting I (Section 13.5 cohorts; R^2 = 0.5; delta = 1e-9; n from 100 to 100,000; 200 cohorts per n).

For each cohort and eps the v3.1 release is the one of experiments/v31_settings/sim_settings.py under setting I
(same seeds, same order of random numbers): stage 1, the stage-1 rule, stage 2, the gate, and solve.stage2_infer
(bootstrap of 2,000 antithetic pairs, no public bound on ||beta||), which must reproduce the published half-widths.
On the same releases we compute
  wald        v3.1's Wald interval for theta_lambda (target 1 - alpha_CI = 0.955);
  bias_aware  v3.1's bias-aware interval for beta (unbounded while the private bound L^ is 0);
  fws_s2      FWS's percentile bootstrap applied to the SAME stage-2 statistics (G^_2 with R's exact A, c^_2, y'y^_2),
              with the mechanism's noise law (sigma_2 on the O-dependent entries, zero on A), at level 0.955;
  fws_1       FWS on a single release that spends the whole budget on the stage-2 statistics (no stage 1; the natural
              FWS mechanism under our adjacency and accountant), same level;
  ols_t       non-private least squares with its usual interval (reference only).
Estimands: theta_lambda = (G + lambda Psi)^-1 c on the cohort (defined for v3.1; equal to the sample OLS when
lambda = 0); the sample OLS b = G^-1 c; the population least-squares coefficient of the clipped data beta_clip (the
working-model target of FWS and of the bias-aware interval); and the generating coefficients beta_true (unclipped).

    python experiments/baselines/sim_fws.py run [--reps 200] [--workers 4]
    python experiments/baselines/sim_fws.py summarize
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
import bcommon as bc  # noqa: E402
from bcommon import cs, ev, pc, solve, dp  # noqa: E402
import fws  # noqa: E402

R2 = 0.5
P, DA = pc.P, pc.DA
NS = pc.NS
SIMDIR = os.path.join(bc.WORK, 'sim_fws')
os.makedirs(SIMDIR, exist_ok=True)
PUBCELL = os.path.join(bc.ROOT, 'experiments', '_work', 'v31_settings', 'sim_I', 'cell_r2_0.5_n_{n}.npz')
B_FWS = 2000
METHODS = ('wald', 'bias_aware', 'fws_s2', 'fws_1', 'ols_t')
ESTIMANDS = ('theta_lam', 'b_ols', 'beta_clip', 'beta_true')


def sigma_single(eps, B_R, B_O, B_y, delta, f_y=bc.F_Y):
    """One release of every O-dependent entry with the whole budget (y'y in it when free, else in its own release)."""
    D, Dn = dp.delta_rep(B_R, B_O, B_y)
    joint = (D - Dn) <= 1e-9 * D
    n_rel = 1 if joint else 2
    kappa = bc.kappa_for(eps)
    gamma = n_rel * (2.0 ** -50 + 2.0 ** -kappa * (1 + 2.0 ** -80) + 2.0 ** -97)
    rho = dp.rho_tight(eps, delta - (1 + math.exp(eps)) * gamma)
    if joint:
        s = D / math.sqrt(2 * rho)
        return dict(sigma=s, sigma_yy=s, joint=True, rho_run=rho)
    return dict(sigma=Dn / math.sqrt(2 * (1 - f_y) * rho), sigma_yy=B_y ** 2 / math.sqrt(2 * f_y * rho), joint=False,
                rho_run=rho)


def o_mask(p, dA):
    m = np.ones((p, p))
    m[:dA, :dA] = 0.0
    return m


def intervals_on_cohorts(Gs, cvec, yys, n, dA, cal, single, st, B_y, rng_stage, rng_inf, rng_f2, rng_1, rng_f1):
    """v3.1 release + its intervals, FWS on the same stage-2 statistics, and FWS on a single full-budget release, for a
    batch of cohorts (Gs (R,p,p), cvec (R,p), yys (R,)).  rng_stage draws v3.1's noise in the order of sim_settings."""
    R, p, _ = Gs.shape
    s1 = cs.stage1_batch(Gs, dA, cal, rng_stage, R, st)
    E, f, g = ev.stage2_noise(p, dA, cal, rng_stage, R)
    G2, c2, yy2 = Gs + E, cvec + f, yys + g
    so = ev.stage2_solve(G2, c2, s1['lam'], s1['Psi'], cal['sigma2'])
    out = dict(theta=so['theta'], refused=so['refused'], lam=s1['lam'], kindO=s1['kind'] == 'O', L_hat=s1['L_hat'],
               theta_lam=np.linalg.solve(Gs + s1['lam'][:, None, None] * s1['Psi'], cvec[..., None])[..., 0],
               half_theta=np.full((R, p), np.nan), half_beta=np.full((R, p), np.nan))
    for r in range(R):
        sd = dict(lam=s1['lam'][r], Psi=s1['Psi'][r], L_hat=s1['L_hat'][r])
        res = solve.stage2_infer(G2[r], c2[r], yy2[r], n, dA, sd, cal['sigma2'], cal['sigma_yy'], B_y,
                                 st['alpha'], st['alpha_L'], st['alpha_U'], np.inf, rng_inf)
        if res['refused']:
            continue
        out['half_theta'][r], out['half_beta'][r] = res['half_theta'], res['half_beta']
    level = 1 - (st['alpha'] - st['alpha_L'] - st['alpha_U'])
    sdG = cal['sigma2'] * o_mask(p, dA)
    sdc = np.full(p, cal['sigma2'])
    for tag, (GG, cc, yy, sG, sc_, rf) in (
            ('fws_s2', (G2, c2, yy2, sdG, sdc, rng_f2)),
            ('fws_1', (Gs + solve.masked_noise(rng_1, R, p, dA, single['sigma']),
                       cvec + single['sigma'] * rng_1.standard_normal((R, p)),
                       yys + single['sigma_yy'] * rng_1.standard_normal(R),
                       single['sigma'] * o_mask(p, dA), np.full(p, single['sigma']), rng_f1))):
        bet, lo, hi = np.empty((R, p)), np.empty((R, p)), np.empty((R, p))
        pdf, s2f = np.zeros(R, bool), np.zeros(R, bool)
        for r in range(R):
            o = fws.fws_interval(GG[r], cc[r], yy[r], n, sG, sc_, rf, B=B_FWS, level=level)
            bet[r], lo[r], hi[r], pdf[r], s2f[r] = o['beta'], o['lo'], o['hi'], o['pd_fixed'], o['s2_floored']
        out.update({f'{tag}_beta': bet, f'{tag}_lo': lo, f'{tag}_hi': hi, f'{tag}_pdfix': pdf, f'{tag}_s2floor': s2f})
    return out


def cell_path(n):
    return os.path.join(SIMDIR, f'cell_n_{n}.npz')


def run_cell(n, reps, st):
    t0 = time.time()
    pop = pc.population(R2)
    cal = {e: cs.calibrate(e, pop.B_R, pop.B_O, pop.B_y, pop.d_R, pop.d_O, st['f1']) for e in bc.EPS}
    ss = np.random.SeedSequence([cs.BASE_SEED, int(round(R2 * 100)), n])           # = sim_settings.py
    ss_data, ss_inf, *ss_eps = ss.spawn(2 + len(bc.EPS))
    rng_d = np.random.default_rng(ss_data)
    Gs = np.empty((reps, P, P)); cvec = np.empty((reps, P)); yys = np.empty(reps)
    b_ols = np.empty((reps, P)); se_ols = np.empty((reps, P))
    for r in range(reps):
        Z, y, _ = pop.draw(n, rng_d)
        Gs[r], cvec[r], yys[r] = Z.T @ Z, Z.T @ y, float(y @ y)
        b_ols[r] = np.linalg.solve(Gs[r], cvec[r])
        rss = yys[r] - cvec[r] @ b_ols[r]
        se_ols[r] = np.sqrt(max(rss, 0.0) / (n - P) * np.diag(np.linalg.inv(Gs[r])))
    rng_inf = np.random.default_rng(ss_inf)
    ss_x = np.random.SeedSequence([cs.BASE_SEED, int(round(R2 * 100)), n, 4242]).spawn(3 * len(bc.EPS))
    out = dict(n=n, reps=reps, b_ols=b_ols, se_ols=se_ols, eps=np.array(bc.EPS))
    keys = None
    for k, eps in enumerate(bc.EPS):
        single = sigma_single(eps, pop.B_R, pop.B_O, pop.B_y, pc.DELTA)
        o = intervals_on_cohorts(Gs, cvec, yys, n, DA, cal[eps], single, st, pop.B_y,
                                 np.random.default_rng(ss_eps[k]), rng_inf, *[np.random.default_rng(s) for s in
                                                                             ss_x[3 * k:3 * k + 3]])
        if keys is None:
            keys = list(o)
            for kk in keys:
                out[kk] = np.empty((len(bc.EPS),) + np.shape(o[kk]), dtype=np.asarray(o[kk]).dtype)
        for kk in keys:
            out[kk][k] = o[kk]
    # the v3.1 intervals must be the published ones (experiments/v31_settings/sim_settings.py, setting I)
    chk = None
    if os.path.exists(PUBCELL.format(n=n)):
        pub = dict(np.load(PUBCELL.format(n=n)))
        a, b = out['half_theta'], pub['half_theta']
        chk = float(np.nanmax(np.abs(a - b))) if np.isfinite(a).any() else 0.0
        assert np.array_equal(np.isnan(a), np.isnan(b))
        assert np.allclose(out['theta'], pub['theta'], rtol=1e-10, atol=1e-12)
    out['check_half_theta_vs_published'] = np.array(np.nan if chk is None else chk)
    np.savez_compressed(cell_path(n), **out)
    return f'n={n}: {time.time() - t0:.0f}s (check {chk})'


def cmd_run(a):
    st = bc.settings()['I']
    bc.r1_masked(P, DA, st['alpha_L'])
    pc.pop_moments('std', R2)
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        jobs = [ex.submit(run_cell, n, a.reps, st) for n in sorted(NS, reverse=True)
                if a.force or not os.path.exists(cell_path(n))]
        for j in jobs:
            print(j.result(), flush=True)
    print('done', flush=True)


# ------------------------------------------------------------------------------------------- summary
def lohi(D, method, k):
    """(lo, hi, centre, ok) arrays (reps, p) of a method at eps index k; ok marks releases the method reports."""
    if method == 'wald':
        th, h = D['theta'][k], D['half_theta'][k]
        return th - h, th + h, th, ~D['refused'][k]
    if method == 'bias_aware':
        th, h = D['theta'][k], D['half_beta'][k]
        return th - h, th + h, th, ~D['refused'][k]
    if method in ('fws_s2', 'fws_1'):
        lo, hi = D[f'{method}_lo'][k], D[f'{method}_hi'][k]
        return lo, hi, D[f'{method}_beta'][k], np.ones(len(lo), bool)
    if method == 'ols_t':
        z = 2.005374
        return D['b_ols'] - z * D['se_ols'], D['b_ols'] + z * D['se_ols'], D['b_ols'], np.ones(len(D['b_ols']), bool)
    raise ValueError(method)


def coverage_table(D, k, targets, sl):
    """Coverage and width per method and estimand over the coefficients in sl and the releases each method reports."""
    res = {}
    for mth in METHODS:
        lo, hi, ctr, ok = lohi(D, mth, k)
        lo, hi = lo[ok][:, sl], hi[ok][:, sl]
        half = (hi - lo) / 2
        fin = np.isfinite(half)
        row = dict(reported=float(ok.mean()), finite=float(fin.mean()),
                   half_median=float(np.median(half[fin])) if fin.any() else None,
                   half_median_all=float(np.median(half)) if fin.any() else float('inf'))
        for est, tv in targets.items():
            t = tv[ok][:, sl] if tv.ndim == 2 else np.broadcast_to(tv[sl], lo.shape)
            cov = (lo <= t) & (t <= hi)
            row[f'cov_{est}'] = float(cov.mean())
            row[f'cov_{est}_min_coef'] = float(cov.mean(0).min())
        res[mth] = row
    return res


def summarize_cells(cells, beta_clip, beta_true, label):
    sl = slice(1, P)
    rows = []
    for n, D in cells:
        for k, eps in enumerate(bc.EPS):
            targets = dict(theta_lam=D['theta_lam'][k], b_ols=D['b_ols'], beta_clip=beta_clip, beta_true=beta_true)
            lam = D['lam'][k]
            row = dict(n=n, eps=eps, ridge_on=float(np.mean(lam > 0)), refused=float(D['refused'][k].mean()),
                       Lhat_pos=float(np.mean(D['L_hat'][k] > 0)),
                       fws_s2_pdfix=float(D['fws_s2_pdfix'][k].mean()), fws_1_pdfix=float(D['fws_1_pdfix'][k].mean()),
                       fws_s2_s2floor=float(D['fws_s2_s2floor'][k].mean()), fws_1_s2floor=float(D['fws_1_s2floor'][k].mean()),
                       slopes=coverage_table(D, k, targets, sl),
                       all11_theta_lam_wald=coverage_table(D, k, dict(theta_lam=D['theta_lam'][k]), slice(0, P))['wald'])
            rows.append(row)
    # pooled ranges by regime
    agg = {}
    for regime, sel in (('ridge_off', lambda r: r['ridge_on'] == 0), ('ridge_on', lambda r: r['ridge_on'] == 1),
                        ('mixed', lambda r: 0 < r['ridge_on'] < 1)):
        rs = [r for r in rows if sel(r)]
        A = dict(cells=len(rs))
        for mth in METHODS:
            for est in ESTIMANDS:
                v = [r['slopes'][mth][f'cov_{est}'] for r in rs]
                A[f'{mth}|{est}'] = [float(min(v)), float(max(v))] if v else None
            w = [r['slopes'][mth]['half_median'] for r in rs if r['slopes'][mth]['half_median'] is not None]
            A[f'{mth}|half_median'] = [float(min(w)), float(max(w))] if w else None
            A[f'{mth}|finite'] = [float(min(r['slopes'][mth]['finite'] for r in rs)),
                                  float(max(r['slopes'][mth]['finite'] for r in rs))] if rs else None
        agg[regime] = A
    return dict(label=label, rows=rows, by_regime=agg)


def summarize():
    pop = pc.population(R2)
    M = pc.pop_moments('std', R2)
    cells, checks = [], {}
    for n in NS:
        D = dict(np.load(cell_path(n)))
        cells.append((n, D))
        checks[str(n)] = float(D['check_half_theta_vs_published'])
    out = summarize_cells(cells, M['beta_clip'], pop.beta_true, 'synthetic population, R^2 = 0.5, setting I')
    st = bc.settings()['I']
    out.update(settings=dict(I=st, delta=pc.DELTA, eps=bc.EPS, ns=NS, reps=int(cells[0][1]['reps']), B_fws=B_FWS,
                             level=1 - (st['alpha'] - st['alpha_L'] - st['alpha_U']),
                             fws_reference='github.com/ceciliaferrando/PB-DP-CIs @ ' + fws.FWS_COMMIT),
               reproduction_check=dict(max_abs_diff_half_theta_vs_published=checks),
               sigma_single={str(e): sigma_single(e, pop.B_R, pop.B_O, pop.B_y, pc.DELTA) for e in bc.EPS},
               sigma_v31={str(e): {k: v for k, v in cs.calibrate(e, pop.B_R, pop.B_O, pop.B_y, 4, 6, st['f1']).items()
                                   if k in ('sigma1', 'sigma2')} for e in bc.EPS},
               beta_clip=M['beta_clip'], beta_true=pop.beta_true)
    path = os.path.join(bc.REPORT, 'fws_sim.json')
    ev.save_json(out, path)
    print('check:', checks)
    print('wrote', path)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('run')
    r.add_argument('--reps', type=int, default=200)
    r.add_argument('--workers', type=int, default=4)
    r.add_argument('--force', action='store_true')
    sub.add_parser('summarize')
    a = ap.parse_args()
    cmd_run(a) if a.cmd == 'run' else summarize()


if __name__ == '__main__':
    main()
