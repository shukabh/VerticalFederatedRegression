"""End-to-end utility runs of the full encrypted protocol (both parties, real OpenFHE, exact DP noise).

  real       a UCI dataset with per-party preprocessing: a public reference split gives the standardization
             constants and the 99.5% clipping bounds, a test split measures prediction error, the rest is O's
             register; R's cohort is a random part of the register plus people O does not hold.  Each encrypted
             run is compared with plaintext least squares on the same matched, clipped data, with the plaintext
             ridge target for the run's (lambda, Psi), and with a plaintext simulation of the same mechanism.
  synthetic  the simulation population with enough matched records for the stage-1 ridge to switch off; fixed
             data, repeated encrypted runs; checks that the estimate approaches least squares and that the Wald
             interval covers theta_lambda at its nominal rate (pooled over runs and coefficients).
  report     writes bench/results/utility.md from the JSON results.

    python bench/utility_runs.py real --dataset protein --eps 1 4 --reps 3
    python bench/utility_runs.py synthetic --eps 4 8 --reps 20 --n 20000
    python bench/utility_runs.py report
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, '..'))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.normpath(os.path.join(ROOT, '..', 'experiments', 'adassp')))
from vfr import common as C  # noqa: E402
from vfr import data, dp, protocol, solve  # noqa: E402

RES = os.path.join(HERE, 'results')
SLOTS = 8192                     # N/2 for N = 2^14: O pads its register to a multiple of this (public size)


# ----------------------------------------------------------------------------------------------- inputs
def _ids(rng, k, taken):
    out = []
    while len(out) < k:
        c = f"{int(rng.integers(10 ** 8, 10 ** 9))}"
        if c not in taken:
            taken.add(c)
            out.append(c)
    return out


def prepare_real(name, seed=0, frac_ref=0.1, frac_test=0.2, cohort_frac=0.6, n_R_extra=1000, q=0.995):
    import datasets as wang
    F, y = wang.RAW[name]()
    ok = np.isfinite(F).all(1) & np.isfinite(y)
    F, y = F[ok].astype(float), y[ok].astype(float)
    rng = np.random.default_rng(seed)
    n = len(y)
    perm = rng.permutation(n)
    n_ref, n_test = int(frac_ref * n), int(frac_test * n)
    ref, test, pool = perm[:n_ref], perm[n_ref:n_ref + n_test], perm[n_ref + n_test:]
    mu, sd = F[ref].mean(0), F[ref].std(0)
    sd = np.where(sd > 0, sd, 1.0)
    my, sy = y[ref].mean(), y[ref].std()
    Fz, yz = (F - mu) / sd, (y - my) / sy
    d = F.shape[1]
    dR = d // 2
    # bounds from the public reference split only
    B_R = float(np.sqrt(1 + np.quantile(np.sum(Fz[ref, :dR] ** 2, 1), q)))
    B_O = float(np.quantile(np.linalg.norm(Fz[ref, dR:], axis=1), q))
    B_y = float(np.quantile(np.abs(yz[ref]), q))
    taken = set()
    ids_pool = _ids(rng, len(pool), taken)
    n_coh = int(cohort_frac * len(pool))
    coh = rng.choice(len(pool), n_coh, replace=False)
    ids_extra = _ids(rng, n_R_extra, taken)
    XR = np.vstack([Fz[pool[coh], :dR], Fz[rng.choice(ref, n_R_extra), :dR]])   # extra people: features irrelevant
    ids_R = [ids_pool[i] for i in coh] + ids_extra
    pr = rng.permutation(len(ids_R))
    data_R = dict(ids=[ids_R[i] for i in pr], X=XR[pr])
    data_O = dict(ids=ids_pool, X=Fz[pool, dR:], y=yz[pool])
    bounds = dict(B_R=B_R, B_O=B_O, B_y=B_y)

    def clip_rows(X, b):
        nr = np.linalg.norm(X, axis=1)
        return X * np.minimum(1.0, b / np.maximum(nr, 1e-300))[:, None]
    Zt = np.column_stack([np.ones(len(test)), clip_rows(Fz[test, :dR], math.sqrt(B_R ** 2 - 1)),
                          clip_rows(Fz[test, dR:], B_O)])
    meta = dict(dataset=name, n_rows=int(n), d=int(d), d_R=int(dR), d_O=int(d - dR), n_ref=int(n_ref),
                n_test=int(n_test), n_register=int(len(pool)), n_cohort=int(len(ids_R)), n_matched=int(n_coh),
                n_bar_O=int(math.ceil(len(pool) / SLOTS) * SLOTS), bounds=bounds, quantile=q, seed=seed)
    return data_R, data_O, bounds, meta, dict(Z=Zt, y=yz[test])


def prepare_synthetic(n_bar_O, n_scope, n_R, n, seed):
    pop = data.population()
    dR_, dO_, truth, b = data.synthetic(n_bar_O, n_scope, n_R, n, seed=seed, pop=pop)
    meta = dict(dataset='synthetic', n_bar_O=n_bar_O, n_register=n_scope, n_cohort=n_R, n_matched=n, d_R=pop.d_R,
                d_O=pop.d_O, bounds=b, seed=seed, beta_true=pop.beta_true.tolist())
    return dR_, dO_, b, meta


# ----------------------------------------------------------------------------------------------- runs
def one_run(ag, data_R, data_O, k):
    with tempfile.TemporaryDirectory() as td:
        t0 = time.time()
        R, O = protocol.run(ag, C.CKKSParams(), data_R, data_O,
                            dict(omp_threads=1, solve_seed=1000 + k), dict(omp_threads=1, ledger_path=os.path.join(td, 'ledger.json')))
        wall = time.time() - t0
    res = R['result'] or {}
    s1 = R['stage1']
    out = dict(k=k, wall=wall, n=R['n'], lam=s1['lam'], kind=s1['kind'], Psi=np.asarray(s1['Psi']).tolist(),
               sigma=R['sigma'], log2_Omega_f=math.log2(R['Omega_f']), refused=bool(res.get('refused', True)),
               gate=res.get('theta_hat_gate'))
    if not out['refused']:
        out.update(theta=np.asarray(res['theta']).tolist(), half_theta=np.asarray(res['half_theta']).tolist(),
                   half_beta=np.asarray(res['half_beta']).tolist(), z=res['z'], sigma2_trP=res['sigma2_trP'])
    return out


def run_many(ag, data_R, data_O, reps, workers):
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(one_run, ag, data_R, data_O, k) for k in range(reps)]
        return [f.result() for f in futs]


def simulate_mechanism(G, c, yy, dA, sig1, sig2, ag, S=2000, seed=7):
    """Plaintext simulation of the same two releases (Gaussian slot noise with the calibrated s.d.) and the
    same stage-1 rule and gate; returns the simulated estimates and their (lambda, refused)."""
    rng = np.random.default_rng(seed)
    p = G.shape[0]
    a_L = ag.alpha / 20
    ths, lams, refused = [], [], 0
    for _ in range(S):
        G1 = G + solve.masked_noise(rng, 1, p, dA, sig1)[0]
        s1 = solve.stage1_rule(G1, dA, sig1, sig2, a_L, ag.theta_star, ag.kappa_Psi)
        G2 = G + solve.masked_noise(rng, 1, p, dA, sig2)[0]
        c2 = c + sig2 * rng.standard_normal(p)
        Gl = G2 + s1['lam'] * s1['Psi']
        if np.linalg.eigvalsh(Gl)[0] / (2 * sig2 * math.sqrt(p)) < 1:
            refused += 1
            continue
        ths.append(np.linalg.solve(Gl, c2))
        lams.append(s1['lam'])
    return np.array(ths), np.array(lams), refused / S


def analyse(runs, ps, dA, ag, test=None):
    G, c = ps['G'], ps['c']
    ols = np.linalg.solve(G, c)
    rows = []
    for r in runs:
        row = dict(k=r['k'], wall=r['wall'], lam=r['lam'], kind=r['kind'], refused=r['refused'], gate=r['gate'])
        if not r['refused']:
            th = np.array(r['theta'])
            th_lam = np.linalg.solve(G + r['lam'] * np.array(r['Psi']), c)
            half = np.array(r['half_theta'])
            sd = half / r['z']
            row.update(theta=th.tolist(), theta_lam=th_lam.tolist(), zscore=((th - th_lam) / sd).tolist(),
                       covered=(np.abs(th - th_lam) <= half).tolist(), rel_err_ols=float(np.linalg.norm(th - ols) / np.linalg.norm(ols)),
                       half_theta=half.tolist(), half_beta=r['half_beta'], sigma2_trP=r['sigma2_trP'])
            if test is not None:
                row.update(mse=float(np.mean((test['y'] - test['Z'] @ th) ** 2)),
                           mse_theta_lam=float(np.mean((test['y'] - test['Z'] @ th_lam) ** 2)))
        rows.append(row)
    out = dict(ols=ols.tolist(), n=int(ps['n']), runs=rows)
    if test is not None:
        out.update(mse_ols=float(np.mean((test['y'] - test['Z'] @ ols) ** 2)), mse_zero=float(np.mean(test['y'] ** 2)))
    sig = runs[0]['sigma']
    ths, lams, rf = simulate_mechanism(G, c, ps['yy'], dA, sig['1'], sig['2'], ag)
    sim = dict(refused=rf, lam_median=float(np.median(lams)) if len(lams) else None,
               theta_mean=ths.mean(0).tolist() if len(ths) else None, theta_sd=ths.std(0).tolist() if len(ths) else None)
    if test is not None and len(ths):
        mses = np.mean((test['y'][None] - ths @ test['Z'].T) ** 2, axis=1)
        sim.update(mse_q=[float(x) for x in np.quantile(mses, [0.025, 0.5, 0.975])])
    if len(ths):
        for row in rows:
            if 'theta' in row:
                row['z_vs_sim'] = ((np.array(row['theta']) - ths.mean(0)) / ths.std(0)).tolist()
    out['sim'] = sim
    return out


def cmd_real(a):
    data_R, data_O, b, meta, test = prepare_real(a.dataset, seed=a.seed)
    ps = data.plaintext_stats(data_R, data_O, b['B_R'], b['B_O'], b['B_y'])
    out = dict(meta=meta, results={})
    for eps in a.eps:
        ag = dp.Agreement(n_R_max=meta['n_cohort'], n_bar_O=meta['n_bar_O'], d_R=meta['d_R'], d_O=meta['d_O'],
                          eps_run=eps, delta_run=a.delta, kappa=a.kappa, **b)
        t0 = time.time()
        runs = run_many(ag, data_R, data_O, a.reps, a.workers)
        res = analyse(runs, ps, meta['d_R'] + 1, ag, test)
        res.update(eps=eps, kappa=a.kappa, delta=a.delta, sigma=runs[0]['sigma'], log2_Omega_f=runs[0]['log2_Omega_f'],
                   elapsed=time.time() - t0)
        out['results'][str(eps)] = res
        print(f"eps={eps}: {a.reps} runs in {time.time() - t0:.0f}s; lam={[round(r['lam']) for r in runs]}; "
              f"refused={[r['refused'] for r in runs]}", flush=True)
    path = os.path.join(RES, f'utility_real_{a.dataset}.json')
    json.dump(out, open(path, 'w'), indent=1)
    print('wrote', path)


def cmd_synthetic(a):
    n_scope = int(a.n * a.scope_ratio)
    n_bar = int(math.ceil(n_scope / SLOTS) * SLOTS)
    n_R = a.n + a.n_R_extra
    data_R, data_O, b, meta = prepare_synthetic(n_bar, n_scope, n_R, a.n, a.seed)
    ps = data.plaintext_stats(data_R, data_O, b['B_R'], b['B_O'], b['B_y'])
    out = dict(meta=meta, results={})
    for eps in a.eps:
        kappa = 48 if eps >= 8 else a.kappa
        ag = dp.Agreement(n_R_max=n_R, n_bar_O=n_bar, d_R=4, d_O=6, eps_run=eps, delta_run=a.delta, kappa=kappa, **b)
        t0 = time.time()
        runs = run_many(ag, data_R, data_O, a.reps, a.workers)
        res = analyse(runs, ps, 5, ag)
        res.update(eps=eps, kappa=kappa, delta=a.delta, sigma=runs[0]['sigma'], log2_Omega_f=runs[0]['log2_Omega_f'],
                   elapsed=time.time() - t0)
        out['results'][str(eps)] = res
        print(f"eps={eps}: {a.reps} runs in {time.time() - t0:.0f}s; lam={sorted(set(round(r['lam']) for r in runs))}; "
              f"refused={sum(r['refused'] for r in runs)}", flush=True)
        json.dump(out, open(os.path.join(RES, 'utility_synthetic.json'), 'w'), indent=1)
    print('wrote', os.path.join(RES, 'utility_synthetic.json'))


# ----------------------------------------------------------------------------------------------- report
def _fmt(x, d=3):
    return '—' if x is None else f'{x:.{d}f}'


def cmd_report(a):
    L = ['# End-to-end utility runs of the full encrypted protocol', '',
         'Generated by `bench/utility_runs.py`. Every run is a complete two-process run: OPRF and labeled PSI, threshold',
         'CKKS upload and aggregation, two releases with exact discrete-Gaussian noise, and the solver. delta = 1e-9.', '']
    pr = os.path.join(RES, 'utility_real_protein.json')
    if os.path.exists(pr):
        D = json.load(open(pr))
        m = D['meta']
        L += [f"## Real data: {m['dataset']} ({m['n_rows']:,} rows, {m['d']} features; R holds {m['d_R']}, O holds {m['d_O']} and y)", '',
              f"Per-party preprocessing from a public reference split of {m['n_ref']:,} rows (standardization constants and 99.5% bounds "
              f"B_R = {m['bounds']['B_R']:.2f} incl. intercept, B_O = {m['bounds']['B_O']:.2f}, B_y = {m['bounds']['B_y']:.2f}); "
              f"test split {m['n_test']:,} rows; O's register {m['n_register']:,} rows padded to {m['n_bar_O']:,}; "
              f"R's cohort {m['n_cohort']:,} people, of whom {m['n_matched']:,} are in the register.", '',
              '| eps | sigma_1 | sigma_2 | Omega_f | run | lambda (Psi) | gate | test MSE | MSE of theta_lambda | rel. distance to OLS | Wald covers theta_lambda |',
              '|---|---|---|---|---|---|---|---|---|---|---|']
        for e, R in D['results'].items():
            for r in R['runs']:
                cov = f"{sum(r['covered'])}/{len(r['covered'])}" if 'covered' in r else '—'
                L.append(f"| {e} | {R['sigma']['1']:.0f} | {R['sigma']['2']:.0f} | 2^{R['log2_Omega_f']:.2f} | {r['k']} | "
                         f"{r['lam']:.0f} ({r['kind']}) | {_fmt(r['gate'], 2)} | {_fmt(r.get('mse'), 4)} | {_fmt(r.get('mse_theta_lam'), 4)} | "
                         f"{_fmt(r.get('rel_err_ols'), 3)} | {cov} |")
        L += ['']
        for e, R in D['results'].items():
            S = R['sim']
            zs = [abs(z) for r in R['runs'] if 'z_vs_sim' in r for z in r['z_vs_sim']]
            L.append(f"- eps = {e}: plaintext least squares test MSE {R['mse_ols']:.4f}, predicting zero {R['mse_zero']:.4f}. "
                     f"Plaintext simulation of the same mechanism (2,000 draws): median lambda {_fmt(S['lam_median'], 0)}, "
                     f"refused {100 * S['refused']:.1f}%, test MSE 95% range {_fmt(S.get('mse_q', [None])[0], 4)}–{_fmt(S.get('mse_q', [None, None, None])[2], 4)}. "
                     f"Encrypted estimates against the simulated distribution: max |z| = {_fmt(max(zs) if zs else None, 2)} over {len(zs)} coefficients.")
        L += ['']
    sy = os.path.join(RES, 'utility_synthetic.json')
    if os.path.exists(sy):
        D = json.load(open(sy))
        m = D['meta']
        L += [f"## Synthetic: ridge switched off ({m['n_matched']:,} matched, register {m['n_register']:,} padded to {m['n_bar_O']:,}, "
              f"cohort {m['n_cohort']:,}; d_R = 4, d_O = 6, p = 11)", '',
              'Fixed data, repeated encrypted runs (fresh keys, OPRF key, shuffle and noise each run). Coverage is of theta_lambda, '
              'the protocol estimand for this data, by the Wald interval (target 0.955), pooled over runs and the 11 coefficients.', '',
              '| eps | kappa | sigma_2 | Omega_f | runs | refused | lambda = 0 | Wald coverage | mean z | sd z | median rel. distance to OLS | median wall (s) |',
              '|---|---|---|---|---|---|---|---|---|---|---|---|']
        for e, R in D['results'].items():
            rs = [r for r in R['runs'] if 'zscore' in r]
            zs = np.array([z for r in rs for z in r['zscore']])
            cov = np.array([c for r in rs for c in r['covered']])
            L.append(f"| {e} | {R['kappa']} | {R['sigma']['2']:.1f} | 2^{R['log2_Omega_f']:.2f} | {len(R['runs'])} | "
                     f"{sum(r['refused'] for r in R['runs'])} | {sum(r['lam'] == 0 for r in R['runs'])} | "
                     f"{cov.mean():.3f} ({cov.sum()}/{cov.size}) | {zs.mean():+.3f} | {zs.std():.3f} | "
                     f"{np.median([r['rel_err_ols'] for r in rs]):.4f} | {np.median([r['wall'] for r in R['runs']]):.0f} |")
        L += ['']
    path = os.path.join(RES, 'utility.md')
    open(path, 'w').write('\n'.join(L) + '\n')
    print(open(path).read())


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('real')
    r.add_argument('--dataset', default='protein')
    r.add_argument('--eps', type=float, nargs='+', default=[1.0, 4.0])
    r.add_argument('--reps', type=int, default=3)
    r.add_argument('--delta', type=float, default=1e-9)
    r.add_argument('--kappa', type=int, default=40)
    r.add_argument('--workers', type=int, default=2)
    r.add_argument('--seed', type=int, default=0)
    s = sub.add_parser('synthetic')
    s.add_argument('--eps', type=float, nargs='+', default=[4.0, 8.0])
    s.add_argument('--reps', type=int, default=20)
    s.add_argument('--n', type=int, default=20000)
    s.add_argument('--scope_ratio', type=float, default=1.2)
    s.add_argument('--n_R_extra', type=int, default=1000)
    s.add_argument('--delta', type=float, default=1e-9)
    s.add_argument('--kappa', type=int, default=40)
    s.add_argument('--workers', type=int, default=2)
    s.add_argument('--seed', type=int, default=5)
    sub.add_parser('report')
    a = ap.parse_args()
    {'real': cmd_real, 'synthetic': cmd_synthetic, 'report': cmd_report}[a.cmd](a)


if __name__ == '__main__':
    main()
