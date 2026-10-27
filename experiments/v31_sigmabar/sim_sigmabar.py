"""Check item #9: the bias-aware interval with the public variance bound sigma_bar^2 against the one with the
data-driven s~^2, on the synthetic population and on two populations where clipping of y binds.

Plaintext simulation of the release and solve steps of v3.1 under setting I (vfr/dp.py SETTINGS['I']), as in
experiments/v31_settings/sim_settings.py: per cohort and eps a stage-1 release, the stage-1 rule (exact r_1), a
stage-2 release, the gate, and vfr/solve.stage2_infer with the implementation's bootstrap (B = 2,000 antithetic
pairs, no public bound on ||beta||).  sigma_1, sigma_2 and sigma_yy come from the rho split of Calibrate at
delta = 1e-9 (v31_eval/common._calibrate_plain, which agrees with dp.calibrate to 1e-12 relative and needs no
native module), with each population's own B_y.

Populations (features, feature bounds B_R, B_O and slopes of sim_core.Population(r2=0.5, seed=0)):
  base    the synthetic population of the paper: y = z'beta + e, e ~ N(0, 0.5), B_y the 99.5% quantile of |y|
          (2.89; y clipped in about 0.5% of rows);
  yclip   the same y, B_y the 80% quantile of |y| (about 20% of rows clipped): the clipped errors are
          heteroskedastic although e is not;
  hetero  e ~ N(0, s(z)^2) with s(z) proportional to 0.3 + |z_O1| (z_O1: O's first feature, lognormal),
          scaled so that E[s^2] = 0.5, and B_y the 90% quantile of |y|: heteroskedastic before clipping, with
          the noisier rows at high leverage.

Intervals for beta, all from the same release and bootstrap (no extra random numbers):
  guaranteed  sigma_bar^2 = B_y^2 (the Agreement default, valid with no model; proofs Part B, Prop. S10b);
  declared    sigma_bar^2 = 0.5, the pre-clipping error variance (valid for base and yclip, where e is
              homoskedastic and clipping is 1-Lipschitz; not admissible for hetero, where no bound is declared);
  model       the data-driven s~^2 (valid only under the working model).
Targets: the projection coefficient beta* = G^-1 X~' E[y~ | X] of each cohort (the estimand of Proposition S10b,
computed in closed form from the Gaussian errors), and the population least-squares coefficient of the clipped
data (4 million rows; the target of the paper's Table fws_cov).  Coverage is over the ten slopes.

    python experiments/v31_sigmabar/sim_sigmabar.py run [--reps 1000] [--workers 4]
    python experiments/v31_sigmabar/sim_sigmabar.py summarize
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import numpy as np  # noqa: E402
from scipy.stats import norm  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'experiments', 'v31_settings'))
import scommon as cs  # noqa: E402
from scommon import ev, solve, sc, dp  # noqa: E402

REPORT = os.path.join(ROOT, 'reports', 'v31_sigmabar')
WORK = os.path.join(ROOT, 'experiments', '_work', 'v31_sigmabar')
os.makedirs(REPORT, exist_ok=True)
os.makedirs(WORK, exist_ok=True)

P, DA = 11, 5
SETTING = dp.setting('I')
EPS = (1.0, 8.0)
NS = {1.0: (5000, 25000, 50000, 100000), 8.0: (1000, 2500, 10000, 25000, 100000)}
POPS = ('base', 'yclip', 'hetero')
Y_QUANT = {'base': 0.995, 'yclip': 0.80, 'hetero': 0.90}
HET_COL = 4          # column of the standardized features (0-based, without intercept): O's first feature (lognormal)
SIGMA0_2 = 0.5       # pre-clipping error variance (base, yclip), = 1 - R^2
SEED = 20261027


class Pop:
    """sim_core.Population(r2=0.5, seed=0) with its own error model and y bound."""

    def __init__(self, name):
        self.name = name
        self.base = sc.Population(r2=0.5, seed=0)
        b = self.base
        self.B_R, self.B_O, self.d_R, self.d_O = b.B_R, b.B_O, b.d_R, b.d_O
        rng = np.random.default_rng([SEED, 1])
        X = b._features(b.pop_size, rng)
        Zs = (X - b.mu) / b.s
        if name == 'hetero':
            h = 0.3 + np.abs(Zs[:, HET_COL])
            self.h_scale = math.sqrt(SIGMA0_2 / float(np.mean(h ** 2)))
        else:
            self.h_scale = None
        m, s = Zs @ b.beta_std, self.err_sd(Zs)
        y = m + s * rng.standard_normal(len(m))
        self.B_y = float(b.B_y) if name == 'base' else float(np.quantile(np.abs(y), Y_QUANT[name]))

    def err_sd(self, Zs):
        if self.h_scale is None:
            return np.full(len(Zs), math.sqrt(SIGMA0_2))
        return self.h_scale * (0.3 + np.abs(Zs[:, HET_COL]))

    def draw(self, n, rng):
        """Clipped design Z (intercept first), clipped y, and the mean m and s.d. s of y before clipping."""
        b = self.base
        X = b._features(n, rng)
        Zs = (X - b.mu) / b.s
        m, s = Zs @ b.beta_std, self.err_sd(Zs)
        y = m + s * rng.standard_normal(n)
        zR, zO = Zs[:, :b.d_R], Zs[:, b.d_R:]
        nR, nO = np.linalg.norm(zR, axis=1), np.linalg.norm(zO, axis=1)
        zR = zR * np.minimum(1.0, b.B_R_rows / np.maximum(nR, 1e-300))[:, None]
        zO = zO * np.minimum(1.0, b.B_O / np.maximum(nO, 1e-300))[:, None]
        Z = np.hstack([np.ones((n, 1)), zR, zO])
        return Z, np.clip(y, -self.B_y, self.B_y), m, s


def clip_mean_var(m, s, B):
    """Mean and variance of clip(m + s Z, -B, B), Z ~ N(0, 1) (vectorized)."""
    a, b = (-B - m) / s, (B - m) / s
    Pa, Pb, pa, pb = norm.cdf(a), norm.cdf(b), norm.pdf(a), norm.pdf(b)
    mid = Pb - Pa
    e1 = m * mid + s * (pa - pb)
    e2 = (m * m + s * s) * mid + 2 * m * s * (pa - pb) + s * s * (a * pa - b * pb)
    mean = e1 + B * (1 - Pb) - B * Pa
    var = e2 + B * B * (Pa + 1 - Pb) - mean ** 2
    return mean, np.maximum(var, 0.0)


def pop_target(name, rows=4_000_000, chunk=250_000):
    """Population least-squares coefficient of clipped y on the clipped design, ell = lambda_min(Sigma), the
    clipping rate of y and the mean conditional variance of the clipped y (cached)."""
    path = os.path.join(WORK, f'pop_{name}.npz')
    if os.path.exists(path):
        return dict(np.load(path))
    pop = Pop(name)
    rng = np.random.default_rng([SEED, 2, POPS.index(name)])
    G = np.zeros((P, P)); c = np.zeros(P); clip = 0.0; v = 0.0; vmax = 0.0
    for _ in range(rows // chunk):
        Z, y, m, s = pop.draw(chunk, rng)
        G += Z.T @ Z; c += Z.T @ y
        _, var = clip_mean_var(m, s, pop.B_y)
        v += float(var.sum()); vmax = max(vmax, float(var.max()))
        a, b = (-pop.B_y - m) / s, (pop.B_y - m) / s
        clip += float(np.sum(norm.cdf(a) + norm.sf(b)))
    out = dict(beta_pop=np.linalg.solve(G, c), ell=np.linalg.eigvalsh(G / rows)[0], clip_rate=np.array(clip / rows),
               mean_var=np.array(v / rows), max_var=np.array(vmax), B_y=np.array(pop.B_y))
    np.savez(path, **out)
    return out


def calib(pop, eps):
    out = ev._calibrate_plain(eps, cs.DELTA, pop.B_R, pop.B_O, pop.B_y, cs.kappa_for(eps), f1=SETTING['f1'])
    out['kappa'] = cs.kappa_for(eps)
    return out


def run_cell(name, eps, n, reps):
    t0 = time.time()
    path = os.path.join(WORK, f'cell_{name}_eps{eps:g}_n{n}.npz')
    pop = Pop(name)
    cl = calib(pop, eps)
    ss = np.random.SeedSequence([SEED, POPS.index(name), int(eps * 10), n])
    ss_d, ss_n, ss_i = ss.spawn(3)
    rd, rn, ri = (np.random.default_rng(s) for s in (ss_d, ss_n, ss_i))
    Gs = np.empty((reps, P, P)); cv = np.empty((reps, P)); yys = np.empty(reps); bstar = np.empty((reps, P))
    for r in range(reps):
        Z, y, m, s = pop.draw(n, rd)
        Gs[r], cv[r], yys[r] = Z.T @ Z, Z.T @ y, float(y @ y)
        mu, _ = clip_mean_var(m, s, pop.B_y)
        bstar[r] = np.linalg.solve(Gs[r], Z.T @ mu)
    s1 = cs.stage1_batch(Gs, DA, cl, rn, reps, SETTING)
    E, f, g = ev.stage2_noise(P, DA, cl, rn, reps)
    G2, c2, yy2 = Gs + E, cv + f, yys + g
    keys = ('guaranteed', 'declared', 'model')
    out = dict(theta=np.full((reps, P), np.nan), s2=np.full(reps, np.nan), lam=s1['lam'], L_hat=s1['L_hat'],
               refused=np.zeros(reps, bool), bstar=bstar, half_theta=np.full((reps, P), np.nan),
               theta_lam=np.linalg.solve(Gs + s1['lam'][:, None, None] * s1['Psi'], cv[..., None])[..., 0],
               sigma1=cl['sigma1'], sigma2=cl['sigma2'], B_y=pop.B_y)
    for k in keys:
        out['half_' + k] = np.full((reps, P), np.nan)
    sb2 = pop.B_y ** 2
    for r in range(reps):
        sd = dict(lam=s1['lam'][r], Psi=s1['Psi'][r], L_hat=s1['L_hat'][r])
        res = solve.stage2_infer(G2[r], c2[r], yy2[r], n, DA, sd, cl['sigma2'], cl['sigma_yy'], pop.B_y,
                                 SETTING['alpha'], SETTING['alpha_L'], SETTING['alpha_U'], np.inf, ri,
                                 sigma_bar2=sb2)
        if res['refused']:
            out['refused'][r] = True
            continue
        out['theta'][r], out['s2'][r], out['half_theta'][r] = res['theta'], res['s2'], res['half_theta']
        out['half_guaranteed'][r] = res['half_beta_guaranteed']
        out['half_model'][r] = res['half_beta_model']
        PGP = res['P2'] @ G2[r] @ res['P2']
        out['half_declared'][r] = solve.bias_aware_half(res['bbar'], res['Vmech'] + SIGMA0_2 * PGP, res['alpha_CI'])
    np.savez_compressed(path, **out)
    return f'{name} eps={eps:g} n={n}: {time.time() - t0:.0f}s'


def cmd_run(a):
    for name in POPS:
        pop_target(name)
    jobs = [(name, eps, n) for name in POPS for eps in EPS for n in NS[eps]]
    jobs.sort(key=lambda j: -j[2])
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(run_cell, *j, a.reps) for j in jobs
                if a.force or not os.path.exists(os.path.join(WORK, f'cell_{j[0]}_eps{j[1]:g}_n{j[2]}.npz'))]
        for fu in futs:
            print(fu.result(), flush=True)


# ------------------------------------------------------------------------------------------- summaries
def wilson(k, m, z=1.96):
    if m == 0:
        return (float('nan'), float('nan'))
    ph = k / m
    d = 1 + z * z / m
    c = (ph + z * z / (2 * m)) / d
    h = z * math.sqrt(ph * (1 - ph) / m + z * z / (4 * m * m)) / d
    return (c - h, c + h)


def cell_summary(name, eps, n, T):
    D = dict(np.load(os.path.join(WORK, f'cell_{name}_eps{eps:g}_n{n}.npz')))
    ok = ~D['refused']
    sl = slice(1, P)                                   # the ten slopes
    th = D['theta'][ok][:, sl]
    out = dict(pop=name, eps=eps, n=n, reps=int(len(ok)), refused=float(np.mean(~ok)),
               ridge=float(np.mean(D['lam'] > 0)), sigma2=float(D['sigma2']), B_y=float(D['B_y']),
               s2_median=float(np.nanmedian(D['s2'])))
    targets = dict(cond=D['bstar'][ok][:, sl], pop=np.broadcast_to(T['beta_pop'][sl], th.shape))
    for k in ('guaranteed', 'declared', 'model'):
        h = D['half_' + k][ok][:, sl]
        fin = np.isfinite(h)
        out[k] = dict(finite=float(fin.mean()),
                      median_half=float(np.median(h[fin])) if fin.any() else float('inf'))
        for tn, tv in targets.items():
            cov = np.abs(th - tv) <= h                  # an unbounded interval covers
            per = cov.mean(axis=0)
            out[k]['cov_' + tn] = dict(pooled=float(cov.mean()), min=float(per.min()), max=float(per.max()),
                                       per_slope=per.tolist(),
                                       min_ci=wilson(int(cov[:, per.argmin()].sum()), cov.shape[0]))
    ht = D['half_theta'][ok][:, sl]
    covw = np.abs(th - D['theta_lam'][ok][:, sl]) <= ht
    out['wald'] = dict(pooled=float(covw.mean()), min=float(covw.mean(0).min()), max=float(covw.mean(0).max()),
                       median_half=float(np.median(ht)))
    g, m_ = D['half_guaranteed'][ok][:, sl], D['half_model'][ok][:, sl]
    both = np.isfinite(g) & np.isfinite(m_)
    out['width_ratio_g_over_m'] = float(np.median(g[both] / m_[both])) if both.any() else float('nan')
    return out


def cmd_summarize(a):
    res = dict(setting=SETTING, eps=list(EPS), ns={str(k): list(v) for k, v in NS.items()}, populations={}, cells=[])
    for name in POPS:
        T = pop_target(name)
        pop = Pop(name)
        res['populations'][name] = dict(B_y=float(T['B_y']), B_y2=float(T['B_y']) ** 2,
                                        clip_rate_y=float(T['clip_rate']), mean_cond_var=float(T['mean_var']),
                                        max_cond_var=float(T['max_var']), ell=float(T['ell']),
                                        beta_pop=T['beta_pop'].tolist(), y_quantile=Y_QUANT[name],
                                        sigma=dict({f'{e:g}': {k: calib(pop, e)[k] for k in ('sigma1', 'sigma2')}
                                                    for e in EPS}))
        for eps in EPS:
            for n in NS[eps]:
                if os.path.exists(os.path.join(WORK, f'cell_{name}_eps{eps:g}_n{n}.npz')):
                    res['cells'].append(cell_summary(name, eps, n, T))
    with open(os.path.join(REPORT, 'sim_sigmabar.json'), 'w') as fh:
        json.dump(res, fh, indent=1)
    print('wrote', os.path.join(REPORT, 'sim_sigmabar.json'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['run', 'summarize'])
    ap.add_argument('--reps', type=int, default=1000)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--force', action='store_true')
    a = ap.parse_args()
    (cmd_run if a.cmd == 'run' else cmd_summarize)(a)


if __name__ == '__main__':
    main()
