"""Choose P2 by the rule pre-registered in reports/v31_P2/settings_choice_P2.md (commit 0d7e7a0).

Two selection populations from vfr/data.population(): (a) 'std' standardized with the 13.5 bounds, (b) 'bn' block
normalized (B_R^2 = 2, B_O = 1).  R^2 = 0.5 decides; R^2 = 0.3 and 0.9 are robustness only.  delta = 1e-9,
eps in {0.5, 1, 2, 4, 8}, 14 values of n from 100 to 100,000, 200 cohorts per n (seed tag 5250).  Per (R^2, n, eps)
one set of unit noise, shared by every candidate and both populations.  No real dataset is read.

Rule.  rho(c, k) = refusal share, m(c, k) = mean test MSE / mean non-private test MSE, k = (population, eps, n) at
R^2 = 0.5 (140 cells).  Binding cell: some grid candidate has rho <= 0.01.  Feasible: rho <= 0.01 in every binding
cell.  P2 = argmin of C = mean_k m(c, k) over feasible candidates; ties within 0.1% go to the larger theta*, then the
smaller alpha_L, then f1 closest to 0.25, then the smaller f1.  On a grid edge: extend once (p2common.EXTEND),
recompute on the union, keep the result.

    python experiments/v31_P2/select_P2.py run [--reps 200] [--workers 4]
    python experiments/v31_P2/select_P2.py choose          # writes the extension plan if P2 is on an edge
    python experiments/v31_P2/select_P2.py run --ext       # only if choose asked for it; then choose again
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
import p2common as pc  # noqa: E402
from p2common import cs, ev, solve  # noqa: E402

R2S = [0.5, 0.3, 0.9]
DECIDE_R2 = 0.5
SEL_TAG = 5250
SELDIR = os.path.join(pc.WORK, 'select')
os.makedirs(SELDIR, exist_ok=True)
EXT_PLAN = os.path.join(SELDIR, 'extension_plan.json')
REF_MAX = 0.01
TIE_REL = 1e-3
P, DA = pc.P, pc.DA


# ------------------------------------------------------------------------------------------- candidates
def grid():
    return pc.grid_candidates()


def ext_axes():
    if not os.path.exists(EXT_PLAN):
        return None
    return ev.load_json(EXT_PLAN)['axes']


def extended_candidates(axes):
    """Full cross product of the extended axes (the grid's values plus the pre-registered extension values in each
    direction where the choice was on the edge)."""
    th = sorted(set(pc.THETA_GRID) | set(axes.get('theta_star', [])))
    al = sorted(set(pc.ALPHAL_GRID) | set(axes.get('alpha_L', [])))
    f1 = sorted(set(pc.F1_GRID) | set(axes.get('f1', [])))
    return pc.grid_candidates(th, al, f1)


def cell_path(r2, n, tag):
    return os.path.join(SELDIR, f'cell_{tag}_r2_{r2}_n_{n}.json')


# ------------------------------------------------------------------------------------------- one cell
def run_cell(r2, n, reps, cands, tag):
    t0 = time.time()
    pop = pc.population(r2)
    M = {k: pc.pop_moments(k, r2) for k in pc.POPS}
    ss = np.random.SeedSequence([cs.BASE_SEED, SEL_TAG, int(round(r2 * 100)), n])
    ss_data, *ss_eps = ss.spawn(1 + len(pc.EPS))
    rd = np.random.default_rng(ss_data)
    G = {k: np.empty((reps, P, P)) for k in pc.POPS}
    C = {k: np.empty((reps, P)) for k in pc.POPS}
    for r in range(reps):
        Zd, yc = pc.draw_both(pop, n, rd)
        for k in pc.POPS:
            G[k][r], C[k][r] = Zd[k].T @ Zd[k], Zd[k].T @ yc
    res = dict(r2=r2, n=n, reps=reps, tag=tag, ols={}, by_eps={})
    for k in pc.POPS:
        b = np.linalg.solve(G[k], C[k][..., None])[..., 0]
        res['ols'][k] = dict(mse=float(np.mean(pc.pred_mse(b, M[k]))),
                             lmin_G_median=float(np.median(np.linalg.eigvalsh(G[k])[:, 0])))
    for eps, sse in zip(pc.EPS, ss_eps):
        rng = np.random.default_rng(sse)
        Z1 = solve.masked_noise(rng, reps, P, DA, 1.0)
        Z2 = solve.masked_noise(rng, reps, P, DA, 1.0)
        f2 = rng.standard_normal((reps, P))
        row = {}
        for k in pc.POPS:
            ols = res['ols'][k]['mse']
            rk = {}
            for name, st in cands.items():
                cal = pc.calibrate(k, r2, eps, st['f1'])
                th, s1, gate, refused = pc.run_setting_on(G[k], C[k], Z1, Z2, f2, cal, st)
                mse = pc.pred_mse(th, M[k])
                rk[name] = dict(m=float(mse.mean() / ols), nref=int(refused.sum()), on=float(np.mean(s1['lam'] > 0)),
                                lam=float(np.median(s1['lam'])), psiO=float(np.mean(s1['kind'] == 'O')),
                                gate=float(np.median(gate)))
            row[k] = rk
        res['by_eps'][str(eps)] = row
    ev.save_json(res, cell_path(r2, n, tag))
    return f'[{tag}] r2={r2} n={n}: {time.time() - t0:.0f}s'


def cmd_run(a):
    assert cs.check_stage1_vec(), 'vectorized stage-1 rule differs from solve.stage1_rule'
    if a.ext:
        axes = ext_axes()
        assert axes, 'no extension plan: run choose first (it writes one only if P2 is on a grid edge)'
        g = grid()
        cands = {k: v for k, v in extended_candidates(axes).items() if k not in g}
        tag = 'ext'
    else:
        cands = dict(grid())
        cands.update(pc.reference_settings())
        tag = 'grid'
    print(f'{len(cands)} candidates ({tag})', flush=True)
    for al in sorted({st['alpha_L'] for st in cands.values() if st['r1_mode'] == 'exact'}):
        print(f'r1/sigma1 at alpha_L={al:g}: {solve.r1_quantile_exact(DA, P - DA, al):.4f}', flush=True)
    for r2 in R2S:
        for k in pc.POPS:
            pc.pop_moments(k, r2)
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = []
        for n in sorted(pc.NS, reverse=True):
            for r2 in R2S:
                if os.path.exists(cell_path(r2, n, tag)) and not a.force:
                    continue
                futs.append(ex.submit(run_cell, r2, n, a.reps, cands, tag))
        for f in futs:
            print(f.result(), flush=True)


# ------------------------------------------------------------------------------------------- the rule
def load_cells(r2):
    cells = []
    for n in pc.NS:
        c = ev.load_json(cell_path(r2, n, 'grid'))
        pe = cell_path(r2, n, 'ext')
        if os.path.exists(pe):
            e = ev.load_json(pe)
            assert e['reps'] == c['reps'] and np.isclose(e['ols']['std']['mse'], c['ols']['std']['mse'], rtol=1e-12)
            for eps in c['by_eps']:
                for k in pc.POPS:
                    c['by_eps'][eps][k].update(e['by_eps'][eps][k])
        cells.append(c)
    return cells


def table(cells, names):
    """rho (refusal counts) and m over the cells, as arrays (n_cand, n_cells), cell order (pop, eps, n)."""
    keys = [(k, str(e), i) for k in pc.POPS for e in pc.EPS for i in range(len(cells))]
    nref = np.array([[cells[i]['by_eps'][e][k][nm]['nref'] for (k, e, i) in keys] for nm in names])
    m = np.array([[cells[i]['by_eps'][e][k][nm]['m'] for (k, e, i) in keys] for nm in names])
    on = np.array([[cells[i]['by_eps'][e][k][nm]['on'] for (k, e, i) in keys] for nm in names])
    return keys, nref, m, on


def apply_rule(cells, cands):
    """The pre-registered rule on candidate set `cands` (dict name -> setting).  Returns the choice and criteria."""
    names = list(cands)
    reps = cells[0]['reps']
    keys, nref, m, on = table(cells, names)
    max_ref = math.floor(REF_MAX * reps + 1e-9)
    ok = nref <= max_ref
    binding = ok.any(0)
    viol = (~ok & binding[None]).sum(1)
    feasible = viol == 0
    fallback = False
    if not feasible.any():
        feasible = viol == viol.min()
        fallback = True
    Cv = m.mean(1)
    pops = np.array([k for (k, _, _) in keys])
    crit = {}
    for j, nm in enumerate(names):
        d = dict(C=float(Cv[j]), feasible=bool(feasible[j]), binding_violated=int(viol[j]),
                 ridge_on_mean=float(on[j].mean()))
        for k in pc.POPS:
            s = pops == k
            d[f'C_{k}'] = float(m[j, s].mean())
            d[f'refused_max_{k}'] = float(nref[j, s].max() / reps)
            d[f'refused_mean_{k}'] = float(nref[j, s].mean() / reps)
            d[f'cells_over_1pct_{k}'] = int((nref[j, s] > max_ref).sum())
        crit[nm] = d
    fe = [nm for j, nm in enumerate(names) if feasible[j]]
    cmin = min(crit[nm]['C'] for nm in fe)
    tied = [nm for nm in fe if crit[nm]['C'] <= (1 + TIE_REL) * cmin]
    best_C = min(fe, key=lambda nm: crit[nm]['C'])

    def tiekey(nm):
        st = cands[nm]
        return (-st['theta_star'], st['alpha_L'], abs(st['f1'] - 0.25), st['f1'])
    chosen = sorted(tied, key=tiekey)[0]
    rank = sorted(fe, key=lambda nm: crit[nm]['C'])
    return dict(chosen=chosen, best_by_C=best_C, tied=sorted(tied, key=tiekey), C_min=cmin, fallback=fallback,
                n_binding=int(binding.sum()), n_cells=len(keys), binding_by_pop={k: int(binding[pops == k].sum()) for k in pc.POPS},
                criteria=crit, rank_feasible=rank, n_feasible=len(fe),
                rank_all=sorted(names, key=lambda nm: crit[nm]['C']))


def edges(st, thetas, alphas, f1s):
    out = {}
    for ax, vals, v in (('theta_star', thetas, st['theta_star']), ('alpha_L', alphas, st['alpha_L']), ('f1', f1s, st['f1'])):
        if v == min(vals):
            out[ax] = 'low'
        elif v == max(vals):
            out[ax] = 'high'
    return out


def per_eps_curves(cells, name):
    out = {}
    for k in pc.POPS:
        out[k] = {}
        for eps in pc.EPS:
            v = [c['by_eps'][str(eps)][k][name] for c in cells]
            mr = [x['m'] for x in v]
            out[k][str(eps)] = dict(n=pc.NS, m=mr, refused=[x['nref'] / cells[0]['reps'] for x in v],
                                    ridge_on=[x['on'] for x in v], lam_median=[x['lam'] for x in v],
                                    psiO=[x['psiO'] for x in v], gate_median=[x['gate'] for x in v],
                                    pred_1p10=ev.crossing(pc.NS, mr, 1.10))
    return out


def cmd_choose(a):
    g = grid()
    refs = pc.reference_settings()
    cells = {r2: load_cells(r2) for r2 in R2S}
    plan = ext_axes()
    have_ext = os.path.exists(cell_path(DECIDE_R2, pc.NS[0], 'ext'))
    if plan and have_ext:
        cands = extended_candidates(plan)
        th = sorted(set(pc.THETA_GRID) | set(plan.get('theta_star', [])))
        al = sorted(set(pc.ALPHAL_GRID) | set(plan.get('alpha_L', [])))
        f1 = sorted(set(pc.F1_GRID) | set(plan.get('f1', [])))
        stage = 'extended'
    else:
        cands, th, al, f1 = g, pc.THETA_GRID, pc.ALPHAL_GRID, pc.F1_GRID
        stage = 'grid'
    rule = apply_rule(cells[DECIDE_R2], cands)
    ch = rule['chosen']
    e = edges(cands[ch], th, al, f1)
    if stage == 'grid' and e:
        axes = {ax: pc.EXTEND[ax][side] for ax, side in e.items()}
        ev.save_json(dict(axes=axes, edges=e, chosen_on_grid=ch), EXT_PLAN)
        print(f'grid choice {ch} lies on the edge {e}: extension plan written, run `run --ext` then `choose` again')
    # first-stage (grid only) result is always reported
    rule_grid = rule if stage == 'grid' else apply_rule(cells[DECIDE_R2], g)
    # reference settings, evaluated on the same cells (not candidates)
    allc = dict(cands)
    allc.update(refs)
    ref_crit = {}
    keys, nref, m, on = table(cells[DECIDE_R2], list(refs))
    for j, nm in enumerate(refs):
        pops = np.array([k for (k, _, _) in keys])
        ref_crit[nm] = dict(C=float(m[j].mean()), **{f'C_{k}': float(m[j, pops == k].mean()) for k in pc.POPS},
                            **{f'refused_max_{k}': float(nref[j, pops == k].max() / cells[DECIDE_R2][0]['reps']) for k in pc.POPS},
                            **{f'refused_mean_{k}': float(nref[j, pops == k].mean() / cells[DECIDE_R2][0]['reps']) for k in pc.POPS})
    robust = {}
    for r2 in R2S:
        if r2 == DECIDE_R2:
            continue
        rr = apply_rule(cells[r2], cands)
        keysr, nrefr, mr, _ = table(cells[r2], [ch] + list(refs))
        pops = np.array([k for (k, _, _) in keysr])
        robust[str(r2)] = dict(choice_if_applied_here=rr['chosen'], n_feasible=rr['n_feasible'],
                               rank_of_P2_here=(rr['rank_feasible'].index(ch) + 1) if ch in rr['rank_feasible'] else None,
                               P2_feasible_here=rr['criteria'][ch]['feasible'],
                               values={nm: dict(C=float(mr[j].mean()),
                                                **{f'C_{k}': float(mr[j, pops == k].mean()) for k in pc.POPS},
                                                **{f'refused_max_{k}': float(nrefr[j, pops == k].max() / cells[r2][0]['reps']) for k in pc.POPS})
                                       for j, nm in enumerate([ch] + list(refs))},
                               top10=[(nm, rr['criteria'][nm]['C']) for nm in rr['rank_feasible'][:10]])
    r1q = {str(x): solve.r1_quantile_exact(DA, P - DA, x) for x in sorted({s['alpha_L'] for s in allc.values()})}
    noise = {str(r2): pc.noise_ratio_table(r2) for r2 in R2S}
    ell = {k: float(pc.pop_moments(k, DECIDE_R2)['ell']) for k in pc.POPS}
    noff = {}
    for nm in [ch, 'I', 'P_old', 'prev']:
        st = allc[nm]
        noff[nm] = {k: {str(eps): cs.n_off(pc.calibrate(k, DECIDE_R2, eps, st['f1']), ell[k], P, DA, st)[0] for eps in pc.EPS}
                    for k in pc.POPS}
    top = rule['rank_feasible'][:15]
    out = dict(rule='reports/v31_P2/settings_choice_P2.md (pre-registered, commit 0d7e7a0)',
               stage=stage, extension_plan=plan, delta=pc.DELTA, eps=pc.EPS, n_grid=pc.NS,
               reps=cells[DECIDE_R2][0]['reps'], populations=pc.POP_LABEL, noise_ratio=noise, ell=ell,
               r1_unit_p11=r1q,
               chosen=ch, chosen_setting=allc[ch], edges_of_choice=e,
               result=dict({k: v for k, v in rule.items() if k not in ('criteria', 'rank_all', 'rank_feasible')},
                           top15_feasible=[(nm, rule['criteria'][nm]) for nm in top]),
               result_grid_stage=dict(chosen=rule_grid['chosen'], best_by_C=rule_grid['best_by_C'],
                                      tied=rule_grid['tied'], n_feasible=rule_grid['n_feasible'],
                                      n_binding=rule_grid['n_binding'], fallback=rule_grid['fallback']),
               criteria_all=rule['criteria'], rank_all=rule['rank_all'], reference=ref_crit,
               candidates=cands, references=refs, robustness=robust, n_off=noff,
               per_eps={nm: per_eps_curves(cells[DECIDE_R2], nm) for nm in [ch, 'I', 'P_old', 'prev']})
    ev.save_json(out, os.path.join(pc.REPORT, 'selection_P2.json'))
    ev.save_json(dict(P2=allc[ch], name=ch, I=refs['I'], P_old=refs['P_old'], prev=refs['prev']),
                 os.path.join(pc.REPORT, 'setting_P2.json'))
    crit = rule['criteria']
    print(f"stage {stage}: {len(cands)} candidates, {rule['n_feasible']} feasible, {rule['n_binding']}/{rule['n_cells']} "
          f"binding cells {rule['binding_by_pop']}, fallback={rule['fallback']}")
    print('top feasible by C (C, C_std, C_bn, max refusal std/bn):')
    for nm in top:
        c = crit[nm]
        print(f"  {nm:34s} C={c['C']:.4f} std={c['C_std']:.4f} bn={c['C_bn']:.4f} "
              f"ref max {c['refused_max_std']:.3f}/{c['refused_max_bn']:.3f}")
    print('best infeasible by C:')
    for nm in [x for x in rule['rank_all'] if not crit[x]['feasible']][:8]:
        c = crit[nm]
        print(f"  {nm:34s} C={c['C']:.4f} std={c['C_std']:.4f} bn={c['C_bn']:.4f} "
              f"ref max {c['refused_max_std']:.3f}/{c['refused_max_bn']:.3f} viol {c['binding_violated']}")
    for nm, c in ref_crit.items():
        print(f"  ref {nm:8s} C={c['C']:.4f} std={c['C_std']:.4f} bn={c['C_bn']:.4f} ref max {c['refused_max_std']:.3f}/{c['refused_max_bn']:.3f}")
    print('tied:', rule['tied'])
    print('chosen:', ch, 'edges:', e)
    for r2, v in robust.items():
        print(f'R2={r2}: choice if applied here {v["choice_if_applied_here"]}; P2 rank {v["rank_of_P2_here"]}, feasible {v["P2_feasible_here"]}')


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('run')
    r.add_argument('--reps', type=int, default=200)
    r.add_argument('--workers', type=int, default=4)
    r.add_argument('--force', action='store_true')
    r.add_argument('--ext', action='store_true')
    sub.add_parser('choose')
    a = ap.parse_args()
    {'run': cmd_run, 'choose': cmd_choose}[a.cmd](a)


if __name__ == '__main__':
    main()
