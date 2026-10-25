"""Check item #27: coverage of the Wald interval for theta_lambda on real data with calibrated noise (setting I).

Data.  Wang's datasets (experiments/adassp/datasets.py) with the fold-0 split of Sections 13.7-13.8 (10% public
reference rows, the test fold, the rest matched; experiments/v31_eval/real_data.py Data.split(0)) and per-party block
normalization with constants and bounds from the reference rows (ev.per_party_preprocess(..., 'blocknorm')).
Datasets: protein and the other sweep datasets (bike, elevators, pol), where block normalization keeps the ridge on
at every size we can reach, and 3droad (p = 4), where the ridge is off at moderate n, so that both regimes are
covered on real data.

Fixed data.  A cell is (dataset, n, eps): the cohort is the first n matched rows (the matched rows are in random
order) or all of them; G, c, y'y are computed once, so theta_lambda = (G + lambda Psi)^-1 c is a fixed function of the
data and of each release's lambda.  Each cell gets --reps independent releases of the plaintext mechanism under
setting I (vfr/dp.py SETTINGS['I']) with sigma_1, sigma_2, sigma_yy from the implementation's Calibrate at
delta = 1e-9 and eps in {1, 4, 8} (kappa = 40, 44, 48 as in the rest of the evaluation), and the implementation's
stage2_infer (Bboot = 2,000 antithetic pairs).

Also recorded (descriptive, see the README): the bias-aware interval against the least-squares fit of the clipped y
on the clipped design over all non-reference rows (the reference coefficient of the paper's real-data FWS
comparison).  With fixed data this is coverage conditional on one sample, not the interval's nominal guarantee.

    python experiments/v31_more/coverage_real.py run [--reps 1000] [--workers 4]
    python experiments/v31_more/coverage_real.py summarize
    python experiments/v31_more/coverage_real.py recheck [--reps 5000]   # cells whose 95% CI lies below the target
"""
from __future__ import annotations

import argparse
import os
import sys
import time

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mcommon as mc  # noqa: E402
from mcommon import cs, ev  # noqa: E402

import real_data as rdm  # noqa: E402  (experiments/v31_eval/real_data.py)

EPS = [1.0, 4.0, 8.0]
GRID = {'protein': [1000, 5000, 20000, 'all'], 'bike': [1000, 5000, 'all'], 'elevators': [1000, 5000, 'all'],
        'pol': [1000, 5000, 'all'], '3droad': [1000, 5000, 25000]}
BETA_MEANINGFUL = {'protein': True, 'bike': False, 'elevators': False, 'pol': False, '3droad': True}
WORKDIR = os.path.join(mc.WORK, 'coverage_real')
os.makedirs(WORKDIR, exist_ok=True)


def dataset(name):
    D = rdm.Data(name)
    te, ref, m = D.split(0)
    Z, yz, yc, b, sy = ev.per_party_preprocess(D.F, D.y, ref, D.dR, 'blocknorm')
    pop = np.sort(np.concatenate([te, m]))
    beta_pop = np.linalg.lstsq(Z[pop], yc[pop], rcond=None)[0]
    return D, Z, yc, b, m, beta_pop


def cell_path(name, n, eps):
    return os.path.join(WORKDIR, f'{name}_n_{n}_eps_{eps:g}.npz')


def run_cell(name, n, eps, reps, workers):
    t0 = time.time()
    D, Z, yc, b, m, beta_pop = dataset(name)
    rows = m if n == 'all' else m[:n]
    Zc, y = Z[rows], yc[rows]
    G, c, yy = Zc.T @ Zc, Zc.T @ y, float(y @ y)
    nn = len(rows)
    p = G.shape[0]
    dA = D.dR + 1
    st = mc.setting('I')
    cal = cs.calibrate(eps, b['B_R'], b['B_O'], b['B_y'], D.dR, D.dO, st['f1'], delta=mc.DELTA)
    seed = [cs.BASE_SEED, 27, sum(map(ord, name)), nn, int(eps)]
    o = mc.releases_parallel(G, c, yy, nn, dA, cal, st, b['B_y'], reps, seed, workers=workers, beta_ref=beta_pop)
    ell = float(np.linalg.eigvalsh(G / nn)[0])
    beta_hat = np.linalg.lstsq(Zc, y, rcond=None)[0]
    np.savez_compressed(cell_path(name, n, eps), covered=o['covered'], covered_beta=o['covered_beta'],
                        finite_beta=o['finite_beta'], lam=o['lam'], kindO=o['kindO'], L_hat=o['L_hat'],
                        refused=o['refused'], gate=o['gate'], half_theta=o['half_theta'], theta=o['theta'],
                        theta_lam=o['theta_lam'], n=nn, p=p, dA=dA, eps=eps, sigma1=cal['sigma1'], sigma2=cal['sigma2'],
                        sigma_yy=cal['sigma_yy'], joint=cal['joint'], kappa=cal['kappa'], ell=ell, beta_pop=beta_pop,
                        beta_hat=beta_hat, bounds=np.array([b['B_R'], b['B_O'], b['B_y']]),
                        n_off=cs.n_off(cal, max(ell, 1e-300), p, dA, st)[0], r1=o['r1'])
    cv = mc.coverage_summary(o['covered'])
    return (f"{name} n={nn} eps={eps:g}: coverage {cv['coverage']:.4f} ({cv['clustered'][0]:.3f}-{cv['clustered'][1]:.3f}) "
            f"ridge on {np.mean(o['lam'] > 0):.2f} refused {o['refused'].mean():.3f} [{time.time() - t0:.0f}s]")


def cmd_recheck(a):
    """Re-run, with --reps fresh releases (new seed), every cell whose clustered 95% interval lies below the target."""
    st = mc.setting('I')
    out = []
    for nm in GRID:
        for n in GRID[nm]:
            for eps in EPS:
                f = cell_path(nm, n, eps)
                if not os.path.exists(f):
                    continue
                cv = mc.coverage_summary(np.load(f)['covered'])
                if cv['clustered'][1] >= mc.TARGET:
                    continue
                D, Z, yc, b, m, beta_pop = dataset(nm)
                rows = m if n == 'all' else m[:n]
                Zc, y = Z[rows], yc[rows]
                cal = cs.calibrate(eps, b['B_R'], b['B_O'], b['B_y'], D.dR, D.dO, st['f1'], delta=mc.DELTA)
                o = mc.releases_parallel(Zc.T @ Zc, Zc.T @ y, float(y @ y), len(rows), D.dR + 1, cal, st, b['B_y'], a.reps,
                                         [cs.BASE_SEED, 2727, len(rows)], workers=a.workers)
                c2 = mc.coverage_summary(o['covered'])
                out.append(dict(dataset=nm, n=len(rows), eps=eps, first=cv, recheck=c2, recheck_reps=a.reps))
                print(f"recheck {nm} n={len(rows)} eps={eps:g}: first {cv['coverage']:.4f} {cv['clustered']}, "
                      f"recheck {c2['coverage']:.4f} {c2['clustered']}", flush=True)
    ev.save_json(out, os.path.join(mc.REPORT, 'coverage_real_recheck.json'))


def cmd_run(a):
    names = a.datasets or list(GRID)
    for nm in names:
        for n in GRID[nm]:
            for eps in EPS:
                if a.force or not os.path.exists(cell_path(nm, n, eps)):
                    print(run_cell(nm, n, eps, a.reps, a.workers), flush=True)
    print('done', flush=True)


def summarize():
    st = mc.setting('I')
    out = dict(setting=st, delta=mc.DELTA, eps=EPS, grid={k: [str(x) for x in v] for k, v in GRID.items()},
               target=mc.TARGET, Bboot=mc.BBOOT, preprocessing='per-party block unit norm, fold-0 reference rows',
               cells=[])
    allcov = []
    for nm in GRID:
        for n in GRID[nm]:
            for eps in EPS:
                f = cell_path(nm, n, eps)
                if not os.path.exists(f):
                    continue
                d = dict(np.load(f))
                cv = mc.coverage_summary(d['covered'])
                slopes = mc.coverage_summary(d['covered'][:, 1:])
                ok = ~d['refused']
                fin = d['finite_beta'][ok].all(axis=1) if ok.any() else np.array([])
                rec = dict(dataset=nm, n=int(d['n']), n_label=str(n), p=int(d['p']), eps=float(eps), kappa=int(d['kappa']),
                           sigma1=float(d['sigma1']), sigma2=float(d['sigma2']), joint=bool(d['joint']),
                           ell=float(d['ell']), n_off=(float(d['n_off']) if d['ell'] > 1e-10 else None),   # singular design: no switch-off
                           reps=int(len(d['lam'])),
                           ridge_on=float(np.mean(d['lam'] > 0)), psiO=float(np.mean(d['kindO'][d['lam'] > 0])) if np.any(d['lam'] > 0) else None,
                           lam_median=float(np.median(d['lam'])), refused=float(d['refused'].mean()),
                           coverage_all=cv, coverage_slopes=slopes,
                           half_theta_median=float(np.nanmedian(d['half_theta'][:, 1:])),
                           beta_meaningful=BETA_MEANINGFUL[nm],
                           bias_aware=dict(all_finite_share=float(fin.mean()) if fin.size else None,
                                           any_finite_share=float(d['finite_beta'][ok].any(axis=1).mean()) if ok.any() else None,
                                           coverage_beta_pop=mc.coverage_summary(d['covered_beta'][:, 1:])['coverage'],
                                           note='conditional on one fixed sample; unbounded intervals count as covering'),
                           beta_hat_minus_beta_pop_max=float(np.max(np.abs(d['beta_hat'] - d['beta_pop']))))
                out['cells'].append(rec)
                allcov.append(d['covered'][~d['refused']])
    cells = out['cells']
    out['overall'] = dict(cells=len(cells),
                          coverage_range=[min(c['coverage_all']['coverage'] for c in cells), max(c['coverage_all']['coverage'] for c in cells)],
                          cells_ci_excludes_target_below=[f"{c['dataset']} n={c['n']} eps={c['eps']:g}" for c in cells
                                                          if c['coverage_all']['clustered'][1] < mc.TARGET],
                          cells_ci_excludes_target_above=[f"{c['dataset']} n={c['n']} eps={c['eps']:g}" for c in cells
                                                          if c['coverage_all']['clustered'][0] > mc.TARGET],
                          ridge_off_cells=[f"{c['dataset']} n={c['n']} eps={c['eps']:g}" for c in cells if c['ridge_on'] == 0],
                          per_coef_min=min(c['coverage_all']['per_coef_min'] for c in cells),
                          per_coef_max=max(c['coverage_all']['per_coef_max'] for c in cells))
    path = os.path.join(mc.REPORT, 'coverage_real.json')
    ev.save_json(out, path)
    print('wrote', path)
    for c in cells:
        cv = c['coverage_all']
        print(f"{c['dataset']:9s} n={c['n']:6d} eps={c['eps']:g} ridge={c['ridge_on']:.2f} refused={c['refused']:.3f} "
              f"cov={cv['coverage']:.4f} ({cv['clustered'][0]:.3f}-{cv['clustered'][1]:.3f}) coef {cv['per_coef_min']:.3f}-{cv['per_coef_max']:.3f} "
              f"deff={cv['design_effect']:.2f} hw={c['half_theta_median']:.3g} BA fin={c['bias_aware']['all_finite_share']} "
              f"covB={c['bias_aware']['coverage_beta_pop']:.3f}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('run')
    r.add_argument('--datasets', nargs='+')
    r.add_argument('--reps', type=int, default=1000)
    r.add_argument('--workers', type=int, default=4)
    r.add_argument('--force', action='store_true')
    sub.add_parser('summarize')
    rc = sub.add_parser('recheck')
    rc.add_argument('--reps', type=int, default=5000)
    rc.add_argument('--workers', type=int, default=2)
    a = ap.parse_args()
    {'run': cmd_run, 'summarize': lambda a: summarize(), 'recheck': cmd_recheck}[a.cmd](a)


if __name__ == '__main__':
    main()
