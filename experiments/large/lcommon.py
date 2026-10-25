"""Shared pieces of the large-scale evaluation of protocol v3.1 (register sizes 100k, 300k and 1M rows).

Builds on experiments/v31_more/mcommon.py (`mc`), and through it on experiments/v31_settings/scommon.py (`cs`) and
experiments/v31_eval/common.py (`ev`): the plaintext simulation of the v3.1 releases from the implementation's own
modules.  New here: the cells of the staged runs, the datasets at large register sizes, and the cost-model
prediction for the thread configuration that is actually run.

Datasets
  synthetic  the simulation population of implementation/vfr/data.py (experiments/sim_core.Population, R^2 = 0.5,
             d_R = 4, d_O = 6): features on realistic raw scales, standardized with public population constants.
             'std' clips each block to the population's 99.5% bounds (vfr/data.synthetic); 'bn' scales each party's
             standardized block of every row to unit norm (the P2 selection's population (b),
             experiments/v31_P2/p2common.draw_both): B_R^2 = 2, B_O = 1.  Both come from one draw per (size, seed).
  3droad     434,874 rows, 3 features (experiments/adassp/datasets.py): 10% public reference rows (constants and
             bounds), 20% test rows, the rest is O's register (304,413 rows, padded to 311,296); R holds the first
             feature (plus the intercept), O the other two and y; block normalization as in the paper's real-data
             runs (bench/utility_runs.transform, experiments/v31_eval/common.per_party_preprocess).

Parameters (approved design): N = 2^14, chain [60,59,59,59] + P[60,60]; delta = 1e-9, kappa = 'auto'; a single-run
ledger with a fresh file per run; settings I and P of vfr/dp.py; kappa_Psi = 2; bias correction off; PSI with h = 3,
delta_psi = 2^-40 and 2 felts per item (the implementation's defaults).
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from functools import lru_cache

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, '..', '..'))
for _p in (os.path.join(ROOT, 'experiments', 'v31_more'), os.path.join(ROOT, 'experiments', 'v31_P2'),
           os.path.join(ROOT, 'implementation'), os.path.join(ROOT, 'implementation', 'bench')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import mcommon as mc  # noqa: E402  (experiments/v31_more/mcommon.py)
from mcommon import cs, ev, dp, solve  # noqa: E402,F401

REPORT = os.path.join(ROOT, 'reports', 'large')
WORK = os.path.join(ROOT, 'experiments', '_work', 'large')
FIGDIR = os.path.join(ROOT, 'paper', 'figures')
os.makedirs(REPORT, exist_ok=True)
os.makedirs(WORK, exist_ok=True)

SLOTS = 8192
DELTA = 1e-9
THREADS = 4                 # OpenMP threads per party (OpenFHE, PSI polynomials) and parallel entries at O
N_R_MAX_A = 20_000
N_COHORT_A = 15_000         # real cohort rows (padded by R to n_R^max)
N_MATCHED_A = 12_000
SIZES_A = {'100k': 100_000, '300k': 300_000, '1M': 1_000_000}
SEED_DATA = {'100k': 101, '300k': 301, '1M': 1001, 'B': 2001, '3droad': 3001}


def n_bar(n_scope):
    return int(math.ceil(n_scope / SLOTS) * SLOTS)


# ------------------------------------------------------------------------------------------- cells
def _cells():
    cells = {}
    for tag, ns in SIZES_A.items():
        for st, kind in (('I', 'std'), ('P', 'std'), ('P', 'bn')):
            for eps in (1.0, 4.0):
                cid = f'A{tag}_{st}{kind}_e{eps:g}'
                cells[cid] = dict(id=cid, stage={'100k': 1, '300k': 2, '1M': 3}[tag], regime='A', dataset='synthetic',
                                  kind=kind, setting=st, eps=eps, n_scope=ns, n_bar_O=n_bar(ns), n_R_max=N_R_MAX_A,
                                  n_cohort=N_COHORT_A, n=N_MATCHED_A, seed=SEED_DATA[tag], size=tag)
    for n in (60_000, 95_000):
        for eps in (0.5, 1.0):
            cid = f'B{n // 1000}k_Istd_e{eps:g}'
            cells[cid] = dict(id=cid, stage=4, regime='B', dataset='synthetic', kind='std', setting='I', eps=eps,
                              n_scope=1_000_000, n_bar_O=n_bar(1_000_000), n_R_max=100_000, n_cohort=100_000, n=n,
                              seed=SEED_DATA['B'] + n // 1000, size='1M')
    for st in ('I', 'P'):
        for eps in (1.0, 4.0):
            cid = f'R3d_{st}bn_e{eps:g}'
            cells[cid] = dict(id=cid, stage=5, regime='3droad', dataset='3droad', kind='bn', setting=st, eps=eps,
                              n_R_max=N_R_MAX_A, n_cohort=N_COHORT_A, n=N_MATCHED_A, seed=SEED_DATA['3droad'],
                              size='300k')
    return cells


CELLS = _cells()


def stage_cells(stage):
    return [c for c in CELLS.values() if c['stage'] == stage]


def data_key(cell):
    """Cells with the same key share one dataset (the settings and eps of a stage run on the same data)."""
    return (cell['dataset'], cell.get('n_scope'), cell['n'], cell['n_cohort'], cell['seed'], cell['kind'])


# ------------------------------------------------------------------------------------------- synthetic data
@lru_cache(maxsize=1)
def population():
    from vfr import data
    return data.population()


def _unit_rows(X):
    nr = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.where(nr > 0, nr, 1.0)


def _transform(pop, X, kind):
    Zs = (X - pop.mu) / pop.s
    zR, zO = Zs[:, :pop.d_R], Zs[:, pop.d_R:]
    if kind == 'std':                    # exactly sim_core.Population.draw
        nR, nO = np.linalg.norm(zR, axis=1), np.linalg.norm(zO, axis=1)
        zR = zR * np.minimum(1.0, pop.B_R_rows / np.maximum(nR, 1e-300))[:, None]
        zO = zO * np.minimum(1.0, pop.B_O / np.maximum(nO, 1e-300))[:, None]
        return zR, zO, Zs
    if kind == 'bn':
        return _unit_rows(zR), _unit_rows(zO), Zs
    raise ValueError(kind)


def bounds_for(kind):
    pop = population()
    if kind == 'std':
        return dict(B_R=pop.B_R, B_O=pop.B_O, B_y=pop.B_y)
    return dict(B_R=math.sqrt(2.0), B_O=1.0, B_y=pop.B_y)


def synthetic(n_scope, n_R, n, seed, kind):
    """O's scope of n_scope real rows (padded by O to n_bar_O), R's cohort of n_R rows, n of them matched; one draw of
    features and outcome, preprocessed as `kind`.  Matched rows carry R's and O's features and y from the same draw;
    identifiers are random 9-digit strings (vfr/data._ids).  Returns data_R, data_O, truth, bounds."""
    from vfr import data
    pop = population()
    rng = np.random.default_rng(seed)
    taken = set()
    ids_m = data._ids(rng, n, taken)
    ids_O_only = data._ids(rng, n_scope - n, taken)
    ids_R_only = data._ids(rng, n_R - n, taken)
    X = pop._features(n_scope + n_R - n, rng)
    zR, zO, Zs = _transform(pop, X, kind)
    y = Zs[:n_scope] @ pop.beta_std + pop.noise_sd * rng.standard_normal(n_scope)
    # rows 0..n-1 matched, n..n_scope-1 O only, n_scope.. R only
    XR = np.vstack([zR[:n], zR[n_scope:]])
    XO = zO[:n_scope]
    pR, pO = rng.permutation(n_R), rng.permutation(n_scope)
    ids_R = ids_m + ids_R_only
    ids_O = ids_m + ids_O_only
    data_R = dict(ids=[ids_R[i] for i in pR], X=XR[pR])
    data_O = dict(ids=[ids_O[i] for i in pO], X=XO[pO], y=y[pO])
    truth = dict(matched=set(ids_m))
    return data_R, data_O, truth, bounds_for(kind)


@lru_cache(maxsize=None)
def pop_moments(kind):
    """Population moments of the preprocessed design (R^2 = 0.5): 'std' from experiments/v31_eval/sim_accuracy.py,
    'bn' from experiments/v31_P2/p2common.py (4 million rows each, cached in experiments/_work)."""
    import p2common as pc
    return pc.pop_moments(kind, 0.5)


def beta_true(kind):
    """The reference coefficient: the population's true standardized coefficient for 'std' (intercept 0); for 'bn',
    whose design is a nonlinear map of the features, the population least-squares coefficient of the clipped
    outcome on the block-normalized design."""
    if kind == 'std':
        return np.asarray(population().beta_true, float)
    return np.asarray(pop_moments('bn')['beta_clip'], float)


# ------------------------------------------------------------------------------------------- 3droad
@lru_cache(maxsize=1)
def _raw_3droad():
    import datasets as WD                      # experiments/adassp/datasets.py (on ev's path)
    F, y = WD.RAW['3droad']()
    ok = np.isfinite(F).all(1) & np.isfinite(y)
    return F[ok].astype(float), y[ok].astype(float)


def road3d(n, n_cohort, seed, preproc='blocknorm', d_R=1):
    """3droad: public reference 10%, test 20%, the rest O's register.  R's cohort: n register rows (matched) and
    n_cohort - n further people (features drawn from the reference rows, as bench/utility_runs.prepare_real).
    Returns data_R, data_O, truth, bounds, meta, test (Z, y) and the matched design (Z, y clipped)."""
    from vfr import data
    F, y = _raw_3droad()
    rng = np.random.default_rng(seed)
    N = len(y)
    perm = rng.permutation(N)
    n_ref, n_test = int(0.1 * N), int(0.2 * N)
    ref, test, pool = perm[:n_ref], perm[n_ref:n_ref + n_test], perm[n_ref + n_test:]
    Z, yz, yc, bounds, sy = ev.per_party_preprocess(F, y, ref, d_R, preproc)
    taken = set()
    ids_pool = data._ids(rng, len(pool), taken)
    coh = rng.choice(len(pool), n, replace=False)
    ids_extra = data._ids(rng, n_cohort - n, taken)
    XR = np.vstack([Z[pool[coh], 1:1 + d_R], Z[rng.choice(ref, n_cohort - n), 1:1 + d_R]])
    ids_R = [ids_pool[i] for i in coh] + ids_extra
    pr = rng.permutation(len(ids_R))
    data_R = dict(ids=[ids_R[i] for i in pr], X=XR[pr])
    data_O = dict(ids=ids_pool, X=Z[pool, 1 + d_R:], y=yz[pool])
    truth = dict(matched=set(ids_pool[i] for i in coh))
    meta = dict(dataset='3droad', n_rows=int(N), d=int(F.shape[1]), d_R=d_R, d_O=int(F.shape[1] - d_R), n_ref=int(n_ref),
                n_test=int(n_test), n_register=int(len(pool)), n_bar_O=n_bar(len(pool)), n_cohort=int(n_cohort),
                n_matched=int(n), preproc=preproc, bounds=bounds, seed=seed)
    return data_R, data_O, truth, bounds, meta, dict(Z=Z[test], y=yz[test]), dict(Z=Z[pool[coh]], y=yc[pool[coh]])


# ------------------------------------------------------------------------------------------- build a cell
def build(cell):
    """Dataset, agreement inputs and plaintext reference of a cell."""
    from vfr import data
    if cell['dataset'] == 'synthetic':
        dR_, dO_, truth, b = synthetic(cell['n_scope'], cell['n_cohort'], cell['n'], cell['seed'], cell['kind'])
        d_R, d_O, nbo = 4, 6, cell['n_bar_O']
        extra = dict(beta_true=beta_true(cell['kind']))
    else:
        dR_, dO_, truth, b, meta, test, mat = road3d(cell['n'], cell['n_cohort'], cell['seed'])
        d_R, d_O, nbo = meta['d_R'], meta['d_O'], meta['n_bar_O']
        extra = dict(meta=meta, test=test, matched_design=mat)
    ps = data.plaintext_stats(dR_, dO_, b['B_R'], b['B_O'], b['B_y'])
    return dict(data_R=dR_, data_O=dO_, truth=truth, bounds=b, d_R=d_R, d_O=d_O, n_bar_O=nbo, ps=ps, **extra)


def agreement(cell, built):
    return dp.Agreement(n_R_max=cell['n_R_max'], n_bar_O=built['n_bar_O'], d_R=built['d_R'], d_O=built['d_O'],
                        eps_run=cell['eps'], delta_run=DELTA, kappa='auto', **built['bounds'], **dp.setting(cell['setting']))


def setting_dict(name):
    return mc.setting(name)


# ------------------------------------------------------------------------------------------- cost model
@lru_cache(maxsize=None)
def _cost_inputs(inputs='published'):
    """Cost-model inputs: 'published' = implementation/bench/results/micro_t{1,4}.json (the committed model);
    'remeasured' = the same microbenchmarks re-run on the idle machine (reports/large/micro_t{1,4}_remeasured.json,
    experiments/large/micro_large.py).  The fit (binning per row, BFV keygen) and the measured 4-core speedups are the
    committed ones in both cases."""
    import cost_model as CM
    if inputs == 'published':
        mu1, mu4 = CM.load('micro_t1.json'), CM.load('micro_t4.json')
    else:
        mu1 = load_json(os.path.join(REPORT, 'micro_t1_remeasured.json'))
        mu4 = load_json(os.path.join(REPORT, 'micro_t4_remeasured.json'))
    fitj = CM.load('cost_model.json')
    return CM, mu1, mu4, fitj['fit'], fitj['speedups_4cores']


def polys_at_fill(n_bar_O, threads=THREADS):
    """Measured interpolation time at the actual partition fill (reports/large/micro_large.json), or None."""
    path = os.path.join(REPORT, 'micro_large.json')
    if not os.path.exists(path):
        return None
    for r in load_json(path)['polys_at_fill']:
        if r['n_bar_O'] == n_bar_O and r['threads'] == threads:
            return r['seconds']
    return None


def predict(n_bar_O, n_R_max, threads=THREADS, workers=THREADS, inputs='published', polys='model'):
    """Cost-model prediction for the run configuration: per-phase seconds of bench/cost_model.model evaluated with
    the 4-thread microbenchmarks (micro_t4: OpenFHE's and the PSI interpolation's OpenMP), and, for the aggregation,
    the single-thread model divided by the measured speedup of parallel entries (cost_model.json,
    speedups_4cores['aggregate']).  Communication is exact.  inputs: 'published' or 'remeasured' microbenchmarks
    (_cost_inputs); polys: 'model' (the published term: one full-partition measurement scaled by alpha D^2) or
    'fill' (the interpolation measured at the size's actual partition fill, micro_large.py).  Returns the model
    dicts and the mapping to the implementation's phases."""
    CM, mu1, mu4, fit, sp = _cost_inputs(inputs)
    m1 = CM.model(n_bar_O, n_R_max, mu1, fit)
    m4 = CM.model(n_bar_O, n_R_max, mu4 if threads == 4 else mu1, fit)
    t = dict(m4['t'])
    sp_agg = sp['aggregate']
    if inputs == 'remeasured':
        # the OPRF and binning phases are single-threaded Python (libsodium, hashlib): OpenMP does not touch them, so
        # their single-thread inputs apply (the 4-thread OPRF microbenchmark is a 200-item timing, noisy)
        for k in ('O.psi.oprf_self', 'O.psi.oprf_requests', 'O.psi.binning', 'R.psi.oprf'):
            t[k] = m1['t'][k]
        # the aggregation speedup measured in the run configuration (4 OpenMP threads x 4 parallel entries, which
        # oversubscribes the 4 cores) instead of the published one (1 thread x 4 entries)
        mp = os.path.join(REPORT, 'micro_large.json')
        if os.path.exists(mp):
            ag_ = {(r['threads'], r['workers']): r['seconds'] for r in load_json(mp).get('aggregate', [])}
            if (1, 1) in ag_ and (threads, workers) in ag_:
                sp_agg = ag_[(1, 1)] / ag_[(threads, workers)]
    if polys == 'fill':
        pf = polys_at_fill(n_bar_O, threads)
        if pf is not None:
            t['O.psi.polys'] = pf
    agg1 = sum(v for k, v in m1['t'].items() if k.startswith('O.agg'))
    agg = agg1 / (sp_agg if workers == 4 else 1.0)
    phases = {
        'O.oprf_self': t['O.psi.oprf_self'],
        'O.psi_preprocess': t['O.psi.binning'] + t['O.psi.polys'],
        'O.oprf_requests': t['O.psi.oprf_requests'],
        'O.psi_eval': t['O.psi.eval'],
        'O.psi_sanitize': t['O.psi.sanitize'],
        'O.upload_convert': t['O.upload_convert'],
        'O.aggregate': agg,
        'O.release': t['O.release'],
        'O.keygen': t['O.keygen'],
        'R.oprf': t['R.psi.oprf'],
        'R.psi_bfv': t['R.psi.bfv'],
        'R.upload_encrypt': t['R.upload_encrypt'],
        'R.keygen': t['R.keygen'],
        'R.release': t['R.release'],
    }
    return dict(phases=phases, single_thread=dict(m1['t'], **{'O.agg_total': agg1}), bytes=m1['bytes'],
                comm=m1['comm'], mem=m1['mem'], cpu_O_1core=m1['cpu_O'], cpu_R_1core=m1['cpu_R'],
                D=m1['D'], alpha=m1['alpha'], W=m1['W'], tables=m1['batches'], felts=m1['felts'], m=m1['m'],
                inputs=inputs, polys=polys, speedup_aggregate=sp_agg)


def measured_phases(R, O):
    """Per-phase seconds of a run (see measured_phases_from)."""
    return measured_phases_from(R, O)


def _mark_cpu(marks, name):
    """CPU seconds of the phase that ends at mark `name` (single-threaded phases: their compute time without the
    time spent waiting for the other party, which the wall timers of these two phases include)."""
    for (n0, w0, c0, _), (n1, w1, c1, _) in zip(marks[:-1], marks[1:]):
        if n1 == name:
            return c1 - c0
    raise KeyError(name)


def measured_phases_from(R, O):
    """Per-phase seconds of a run, on the phases of `predict`: the parties' own wall timers, except R's OPRF phase and
    O's answer to the OPRF request, whose timers include waiting for the other party; for these two single-threaded
    phases the CPU time between the phase marks is used."""
    TO, TR = O['timings'], R['timings']
    rel = TO.get('release', {})
    return {
        'O.oprf_self': TO['psi_oprf_self'],
        'O.psi_preprocess': TO['psi_preprocess'],
        'O.oprf_requests': _mark_cpu(O['marks'], 'oprf_requests'),
        'O.psi_eval': TO['psi_eval'],
        'O.psi_sanitize': TO['psi_sanitize'],
        'O.upload_convert': TO['upload_convert'],
        'O.aggregate': TO['aggregate'],
        'O.release': sum(v['sample'] + v['rerand_pdec'] for v in rel.values()),
        'O.keygen': TO['keygen_O'],
        'R.oprf': _mark_cpu(R['marks'], 'oprf'),
        'R.psi_bfv': TR['psi_query'] + TR['psi_decode'],
        'R.upload_encrypt': TR['upload_encrypt'],
        'R.keygen': TR['keygen_R'],
        'R.release': sum(TR.get('release_R', {}).values()),
    }


def marks_table(marks):
    """[(name, wall, cpu, peak_rss_MB)] -> per-phase wall/CPU deltas and the peak RSS reached by each mark."""
    out = []
    for (n0, w0, c0, _), (n1, w1, c1, r1) in zip(marks[:-1], marks[1:]):
        out.append(dict(phase=n1, wall=w1 - w0, cpu=c1 - c0, peak_rss_MB=r1 / 1024.0))
    return out


# ------------------------------------------------------------------------------------------- misc
def git_head():
    try:
        return subprocess.check_output(['git', '-C', ROOT, 'rev-parse', 'HEAD'], text=True).strip()
    except Exception:  # noqa: BLE001
        return None


def resources():
    import shutil
    du = shutil.disk_usage(ROOT)
    mem = {}
    with open('/proc/meminfo') as fh:
        for line in fh:
            k, v = line.split(':', 1)
            mem[k] = int(v.split()[0])
    return dict(disk_used_frac=du.used / du.total, disk_free_GB=du.free / 2 ** 30,
                mem_available_GB=mem['MemAvailable'] / 2 ** 20, mem_total_GB=mem['MemTotal'] / 2 ** 20)


def load_runs(cell_id):
    path = os.path.join(WORK, f'{cell_id}.jsonl')
    if not os.path.exists(path):
        return []
    return [json.loads(l) for l in open(path)]


def save_json(obj, path):
    ev.save_json(obj, path)


def load_json(path):
    return ev.load_json(path)
