"""Check item #31: about 50 encrypted runs per configuration (full OpenFHE protocol, implementation/vfr/protocol.py).

Data: the synthetic dataset of the earlier encrypted checks (experiments/v31_settings/encrypted_settings.py,
experiments/v31_P2/encrypted_P2.py): n = 10,000 matched records of the simulation population, a 12,000-row register
padded to 16,384 (never more), cohort n + 1,000, seed 11, delta = 1e-9, kappa = 40 for eps <= 2 and 48 at eps = 8.

Configurations (setting values from vfr/dp.py SETTINGS):
  I2  setting I at eps = 2   (stage-1 ridge on; the earlier check's configuration)
  P1  setting P at eps = 1   (ridge on; the earlier P2 check's configuration)
  I8  setting I at eps = 8   (ridge off: the matched design's n_off under I is about 3.9k)

`run` appends one JSON line per encrypted run to experiments/_work/v31_more/encrypted_<cfg>.jsonl (resumable), with
the raw decrypted noise of every released entry standardized by the calibrated sigma (stage 1: the 51 O-dependent
entries of G; stage 2: those 51, the 11 entries of c and y'y), and summary statistics of the noise in all 8,192 slots
of each decrypted release (the slots outside the packed entries hold the plaintext 0 plus the same noise).
`summarize` compares every run with a plaintext simulation of the same mechanism on the same data (4,000 draws;
lambda and theta against the simulated 95% ranges, theta against the simulated law given the run's lambda), computes
Wald coverage under I, and tests the noise: mean zero per release type (entry-level z test and run-level t test),
and Kolmogorov-Smirnov against the calibrated law.  The released slot noise is the canonical embedding of an exact
discrete Gaussian on the coefficient lattice with log2 sigma_c around 60, so its calibrated law in a slot is
N(0, sigma_slot^2) to far better than the KS test can resolve; the test is against that law.

    python experiments/v31_more/encrypted_more.py run --cfg I2 [--reps 50]
    python experiments/v31_more/encrypted_more.py summarize [--first 50]
    python experiments/v31_more/encrypted_more.py plaincov [--reps 10000]   # plaintext coverage at I2 and I8

The check was specified with about 50 runs per configuration; after those (run times were about 20 s, well inside the
time budget) each configuration was extended to 100 runs, for all three configurations alike; only a preliminary summary of the
first 22-26 runs had been looked at.  Both summaries are written: encrypted_more_first50.json and encrypted_more.json (all runs).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mcommon as mc  # noqa: E402
from mcommon import cs, ev, dp, solve  # noqa: E402
from vfr import common as VC  # noqa: E402
from vfr import data, protocol  # noqa: E402

SLOTS = 8192
P, DA = 11, 5
N_MATCHED, SEED = 10000, 11
CFG = {'I2': ('I', 2.0, 7200), 'P1': ('P', 1.0, 7100), 'I8': ('I', 8.0, 7800),
       'I2r': ('I', 2.0, 9000)}   # I2r: replication batch of I2 (see the docstring), run in parts of 50 with seeds 9000 + 1000 part + k
OLD = {'I_eps2': os.path.join(mc.ROOT, 'reports', 'v31_settings', 'encrypted_spot.json'),
       'P2_eps1': os.path.join(mc.ROOT, 'reports', 'v31_P2', 'encrypted_P2.json')}


def dataset():
    n_scope = int(1.2 * N_MATCHED)
    n_bar = int(math.ceil(n_scope / SLOTS) * SLOTS)
    assert n_bar <= 16384, 'the encrypted runs stay at or below 16,384 register rows'
    n_R = N_MATCHED + 1000
    dR_, dO_, truth, b = data.synthetic(n_bar, n_scope, n_R, N_MATCHED, seed=SEED, pop=data.population())
    ps = data.plaintext_stats(dR_, dO_, b['B_R'], b['B_O'], b['B_y'])
    return dR_, dO_, b, ps, dict(n=N_MATCHED, n_scope=n_scope, n_bar_O=n_bar, n_R=n_R, seed=SEED, delta=mc.DELTA)


def agreement(name, eps, meta, b):
    return dp.Agreement(n_R_max=meta['n_R'], n_bar_O=meta['n_bar_O'], d_R=4, d_O=6, eps_run=eps, delta_run=mc.DELTA,
                        kappa=cs.kappa_for(eps), **b, **dp.setting(name))


def git_head():
    try:
        return subprocess.check_output(['git', '-C', mc.ROOT, 'rev-parse', 'HEAD'], text=True).strip()
    except Exception:  # noqa: BLE001
        return None


def slot_stats(z, used, sigma):
    """Noise in all slots of one decoded release: real and imaginary parts outside the used slots, standardized."""
    from scipy import stats
    z = np.asarray(z)
    mask = np.ones(z.size, bool)
    mask[list(used)] = False
    v = np.concatenate([z.real[mask], z.imag]) / sigma
    return dict(n=int(v.size), mean=float(v.mean()), sd=float(v.std()), max_abs=float(np.abs(v).max()),
                ks_p=float(stats.kstest(v, 'norm').pvalue), hist=np.histogram(v, bins=np.linspace(-5, 5, 41))[0].tolist())


def cmd_run(a):
    name, eps, seed0 = CFG[a.cfg]
    dR_, dO_, b, ps, meta = dataset()
    G, c, yy = ps['G'], ps['c'], ps['yy']
    ag = agreement(name, eps, meta, b)
    seed0 += 1000 * a.part
    path = os.path.join(mc.WORK, f'encrypted_{a.cfg}.jsonl' if not a.part and a.cfg != 'I2r' else f'encrypted_{a.cfg}_part{a.part}.jsonl')
    done = 0
    if os.path.exists(path):
        done = sum(1 for _ in open(path))
    iu = np.triu_indices(P)
    mask = np.ones((P, P), bool)
    mask[:DA, :DA] = False
    sel = mask[iu]
    ents = VC.entries(4, 6, True)
    used = {s: [e['slot'] for e in VC.release_entries(ents, s)] for s in (1, 2)}
    head = git_head()
    for k in range(done, a.reps):
        with tempfile.TemporaryDirectory() as td:
            t0 = time.time()
            R, O = protocol.run(ag, VC.CKKSParams(), dR_, dO_, dict(omp_threads=1, solve_seed=seed0 + k),
                                dict(omp_threads=1, ledger_path=os.path.join(td, 'ledger.json')))
            wall = time.time() - t0
        assert R['joint'], 'y-y is expected in the stage-2 release'
        res = R['result'] or {}
        s1 = R['stage1']
        sg = R['sigma']
        z1 = ((np.asarray(R['G1']) - G)[iu][sel]) / sg['1']
        z2 = np.concatenate([((np.asarray(R['G2']) - G)[iu][sel]) / sg['2'], (np.asarray(R['c2']) - c) / sg['2']])
        zyy = (float(R['yy']) - yy) / sg['2']
        rec = dict(cfg=a.cfg, part=a.part, solve_seed=seed0 + k, setting=name, eps=eps, k=k, wall=wall, git_head=head, n=R['n'], lam=float(s1['lam']),
                   kind=s1['kind'], r1=float(s1['r1']), r1_mode=s1['r1_mode'], sigma=sg,
                   refused=bool(res.get('refused', True)), gate=res.get('theta_hat_gate'),
                   theta=np.asarray(res.get('theta')).tolist() if 'theta' in res else None,
                   half_theta=np.asarray(res['half_theta']).tolist() if 'half_theta' in res else None,
                   theta_lam=np.linalg.solve(G + s1['lam'] * np.asarray(s1['Psi']), c).tolist(),
                   z_stage1=z1.tolist(), z_stage2=z2.tolist(), z_yy=zyy,
                   A_exact=bool(np.allclose(np.asarray(R['G2'])[:DA, :DA], G[:DA, :DA], rtol=1e-9, atol=1e-6)
                                and np.allclose(np.asarray(R['G1'])[:DA, :DA], G[:DA, :DA], rtol=1e-9, atol=1e-6)),
                   slots={str(s): slot_stats(R['decoded'][str(s)], used[s], sg[str(s)]) for s in (1, 2)},
                   timings_O_aggregate=O['timings'].get('aggregate'))
        with open(path, 'a') as fh:
            fh.write(json.dumps(ev.jsonable(rec)) + '\n')
        print(f"[{a.cfg}] run {k}: {wall:.0f}s lam={rec['lam']:.0f} ({rec['kind']}) refused={rec['refused']} "
              f"z1 mean {z1.mean():+.3f} sd {z1.std():.3f}; z2 mean {z2.mean():+.3f} sd {z2.std():.3f}; "
              f"slots1 mean {rec['slots']['1']['mean']:+.4f} sd {rec['slots']['1']['sd']:.4f}; A exact {rec['A_exact']}", flush=True)


FIRST = None      # summarize --first N: only the first N runs of each configuration (the pre-specified 50)


def load_runs(cfg):
    import glob
    paths = [os.path.join(mc.WORK, f'encrypted_{cfg}.jsonl')] + sorted(glob.glob(os.path.join(mc.WORK, f'encrypted_{cfg}_part*.jsonl')))
    runs = [json.loads(l) for path in paths if os.path.exists(path) for l in open(path)]
    runs.sort(key=lambda r: (r.get('part', 0), r['k']))
    return runs[:FIRST] if FIRST else runs


def mean_tests(z_by_run):
    """Mean-zero tests for standardized noise: entry-level z test (entries independent N(0,1) under H0) and
    run-level one-sample t test on the per-run means (robust to any within-release dependence)."""
    from scipy import stats
    allz = np.concatenate(z_by_run)
    rm = np.array([np.mean(z) for z in z_by_run])
    n = allz.size
    se = 1 / math.sqrt(n)
    t = stats.ttest_1samp(rm, 0.0) if len(rm) > 1 else None
    return dict(entries=int(n), runs=len(z_by_run), mean=float(allz.mean()), se_entry=se, z=float(allz.mean() / se),
                p_entry=float(2 * stats.norm.sf(abs(allz.mean() / se))), ci95=(float(allz.mean() - 1.96 * se), float(allz.mean() + 1.96 * se)),
                sd=float(allz.std()), sd_ci95=_sd_ci(allz), run_t=float(t.statistic) if t else None,
                run_p=float(t.pvalue) if t else None, ks_p=float(stats.kstest(allz, 'norm').pvalue),
                ks_stat=float(stats.kstest(allz, 'norm').statistic), max_abs=float(np.abs(allz).max()),
                beyond_3=int(np.sum(np.abs(allz) > 3)), expected_beyond_3=float(n * 2 * stats.norm.sf(3)))


def _sd_ci(z):
    from scipy import stats
    n = z.size
    s2 = float(np.sum(z * z) / n)          # mean known (0) under H0
    lo, hi = stats.chi2.ppf([0.025, 0.975], n)
    return (math.sqrt(n * s2 / hi), math.sqrt(n * s2 / lo))


def simulate(G, c, cal, st, S, seed):
    rng = np.random.default_rng(seed)
    s1 = cs.stage1_batch(G, DA, cal, rng, S, st)
    E, f, g = ev.stage2_noise(P, DA, cal, rng, S)
    so = ev.stage2_solve(G[None] + E, c[None] + f, s1['lam'], s1['Psi'], cal['sigma2'])
    return s1, so


def reproduce(runs, G, c, yy, b, st, seed0):
    """Feed each run's decrypted releases (rebuilt from the recorded noise) to the plaintext solver with the run's
    solve seed: the encrypted lambda, estimate and half-widths must come back exactly (relative error)."""
    iu = np.triu_indices(P)
    mask = np.ones((P, P), bool)
    mask[:DA, :DA] = False
    sel = mask[iu]

    def build(z, s):
        M = np.zeros((P, P))
        vals = np.zeros(len(sel))
        vals[sel] = np.asarray(z) * s
        M[iu] = vals
        return G + M + np.triu(M, 1).T
    err = dict(lam=0.0, theta=0.0, half_theta=0.0)
    for r in runs:
        s1, s2 = r['sigma']['1'], r['sigma']['2']
        z2 = np.asarray(r['z_stage2'])
        G1, G2 = build(r['z_stage1'], s1), build(z2[:51], s2)
        S1 = solve.stage1_rule(G1, DA, s1, s2, st['alpha_L'], st['theta_star'], st['kappa_Psi'], r1_mode=st['r1_mode'])
        res = solve.stage2_infer(G2, c + z2[51:] * s2, yy + r['z_yy'] * s2, r['n'], DA, S1, s2, s2, b['B_y'], st['alpha'],
                                 st['alpha_L'], st['alpha_U'], np.inf, np.random.default_rng(r.get('solve_seed', seed0 + r['k'])),
                                 intervals=st['intervals'])
        err['lam'] = max(err['lam'], abs(S1['lam'] - r['lam']) / max(1.0, r['lam']))
        if not res['refused']:
            th = np.asarray(r['theta'])
            err['theta'] = max(err['theta'], float(np.max(np.abs(res['theta'] - th) / np.maximum(np.abs(th), 1e-12))))
            if st['intervals']:
                err['half_theta'] = max(err['half_theta'], float(np.max(np.abs(res['half_theta'] / np.asarray(r['half_theta']) - 1))))
    return err


def summarize_cfg(cfg, G, c, meta, b, yy=None):
    from scipy import stats
    runs = load_runs(cfg)
    if not runs:
        return None
    name, eps, _ = CFG[cfg]
    st = mc.setting(name)
    sig = runs[0]['sigma']
    cal = dict(sigma1=sig['1'], sigma2=sig['2'], sigma_yy=sig.get('y', sig['2']))
    calp = cs.calibrate(eps, b['B_R'], b['B_O'], b['B_y'], 4, 6, st['f1'], n_bar_O=meta['n_bar_O'])
    S = 4000
    s1s, so = simulate(G, c, cal, st, S, 3)
    lq = np.quantile(s1s['lam'], [0.025, 0.5, 0.975])
    th_ok = so['theta'][~so['refused']]
    tq = np.quantile(th_ok, [0.025, 0.975], axis=0)
    rng = np.random.default_rng(4)
    ok = [r for r in runs if not r['refused']]
    zgl, inr = [], []
    for r in ok:
        Psi_r = np.eye(P) if r['kind'] == 'I' else ev.pi_O(P, DA)
        Ec, fc, _ = ev.stage2_noise(P, DA, cal, rng, S)
        stc = ev.stage2_solve(G[None] + Ec, c[None] + fc, np.full(S, r['lam']), np.broadcast_to(Psi_r, (S, P, P)), cal['sigma2'])
        tc = stc['theta'][~stc['refused']]
        zgl.append((np.array(r['theta']) - tc.mean(0)) / tc.std(0))
        inr.append((np.array(r['theta']) >= tq[0]) & (np.array(r['theta']) <= tq[1]))
    zgl = np.array(zgl)
    inr = np.array(inr)
    lam_in = np.array([lq[0] <= r['lam'] <= lq[2] for r in runs])
    lam_q = np.array([np.mean(s1s['lam'] <= r['lam']) for r in runs])
    out = dict(cfg=cfg, setting=name, values=st, eps=eps, kappa=cs.kappa_for(eps), runs=len(runs),
               git_heads=sorted(set(r['git_head'] for r in runs if r.get('git_head'))),
               wall=dict(median=float(np.median([r['wall'] for r in runs])), min=float(min(r['wall'] for r in runs)),
                         max=float(max(r['wall'] for r in runs))),
               sigma_encrypted=sig, sigma_plaintext_calibration=dict(sigma1=calp['sigma1'], sigma2=calp['sigma2']),
               r1=float(runs[0]['r1']), refused=int(sum(r['refused'] for r in runs)),
               ridge_on_runs=int(sum(r['lam'] > 0 for r in runs)), psiO_runs=int(sum(r['kind'] == 'O' and r['lam'] > 0 for r in runs)),
               lam=dict(median=float(np.median([r['lam'] for r in runs])), min=float(min(r['lam'] for r in runs)),
                        max=float(max(r['lam'] for r in runs))),
               sim=dict(draws=S, lam_q=lq.tolist(), refused=float(so['refused'].mean()), ridge_on=float(np.mean(s1s['lam'] > 0)),
                        theta_q025=tq[0].tolist(), theta_q975=tq[1].tolist()),
               lam_in_sim_95=f"{int(lam_in.sum())}/{len(runs)}",
               lam_sim_quantile_ks_p=(float(stats.kstest(lam_q, 'uniform').pvalue) if np.any(s1s['lam'] > 0) else None),
               theta_in_sim_95=f"{int(inr.sum())}/{inr.size}" if inr.size else None,
               theta_in_sim_95_frac=float(inr.mean()) if inr.size else None,
               theta_in_sim_95_wilson=mc.wilson(int(inr.sum()), int(inr.size)) if inr.size else None,
               z_given_lambda=dict(n=int(zgl.size), mean=float(zgl.mean()), sd=float(zgl.std()), max_abs=float(np.abs(zgl).max()),
                                   within_2p5=f"{int(np.sum(np.abs(zgl) <= 2.5))}/{zgl.size}",
                                   ks_p=float(stats.kstest(zgl.ravel(), 'norm').pvalue)) if zgl.size else None,
               A_exact=all(r['A_exact'] for r in runs),
               reproduced_from_decrypted_noise=reproduce(runs, G, c, yy, b, st, CFG[cfg][2]))
    if st['intervals'] and ok:
        cov = np.array([(np.abs(np.array(r['theta']) - np.array(r['theta_lam'])) <= np.array(r['half_theta'])) for r in ok], float)
        out['coverage'] = mc.coverage_summary(cov)
        out['coverage']['clopper_pearson_naive'] = mc.clopper_pearson(int(cov.sum()), int(cov.size))
    z1 = [np.array(r['z_stage1']) for r in runs]
    z2 = [np.array(r['z_stage2']) for r in runs]
    z2y = [np.append(np.array(r['z_stage2']), r['z_yy']) for r in runs]
    out['noise'] = dict(stage1=mean_tests(z1), stage2=mean_tests(z2), stage2_with_yy=mean_tests(z2y),
                        all=mean_tests([np.concatenate([a, b_]) for a, b_ in zip(z1, z2y)]))
    sl = {}
    for s in ('1', '2'):
        mu = np.array([r['slots'][s]['mean'] for r in runs])
        nn = np.array([r['slots'][s]['n'] for r in runs])
        sd = np.array([r['slots'][s]['sd'] for r in runs])
        pooled_mean = float(np.sum(mu * nn) / nn.sum())
        pooled_var = float(np.sum(nn * (sd ** 2 + mu ** 2)) / nn.sum())
        hist = np.sum([r['slots'][s]['hist'] for r in runs], axis=0)
        sl[s] = dict(values=int(nn.sum()), mean=pooled_mean, se=float(1 / math.sqrt(nn.sum())), z=float(pooled_mean * math.sqrt(nn.sum())),
                     sd=math.sqrt(pooled_var), run_mean_t_p=float(stats.ttest_1samp(mu, 0).pvalue) if len(mu) > 1 else None,
                     per_release_ks_p_min=float(min(r['slots'][s]['ks_p'] for r in runs)),
                     per_release_ks_p_uniform_ks=float(stats.kstest([r['slots'][s]['ks_p'] for r in runs], 'uniform').pvalue),
                     max_abs=float(max(r['slots'][s]['max_abs'] for r in runs)), hist=hist.tolist())
    out['noise_all_slots'] = sl
    return out


def old_noise():
    """Raw standardized noise of the earlier spot checks (I at eps = 2, P_0 and P at eps = 1, 5 runs each)."""
    z1, z2 = {}, {}
    if os.path.exists(OLD['I_eps2']):
        dd = ev.load_json(OLD['I_eps2'])['results']
        for key, tag in (('I', 'I2_old'), ('P', 'P0_eps1_old')):      # 'P' there is the first choice P_0
            z1[tag] = [np.array(r['noise']['z_stage1']) for r in dd[key]['runs']]
            z2[tag] = [np.array(r['noise']['z_stage2']) for r in dd[key]['runs']]
    if os.path.exists(OLD['P2_eps1']):
        d = ev.load_json(OLD['P2_eps1'])
        z1['P1_old'] = [np.array(r['noise']['z_stage1']) for r in d['runs']]
        z2['P1_old'] = [np.array(r['noise']['z_stage2']) for r in d['runs']]
    return z1, z2


def cmd_summarize(a):
    global FIRST
    FIRST = a.first
    dR_, dO_, b, ps, meta = dataset()
    G, c = ps['G'], ps['c']
    meta['ell_matched'] = float(np.linalg.eigvalsh(G / meta['n'])[0])
    out = dict(data=meta, results={})
    for cfg in CFG:
        r = summarize_cfg(cfg, G, c, meta, b, ps['yy'])
        if r is None:
            continue
        name, eps, _ = CFG[cfg]
        st = mc.setting(name)
        calp = cs.calibrate(eps, b['B_R'], b['B_O'], b['B_y'], 4, 6, st['f1'], n_bar_O=meta['n_bar_O'])
        r['n_off_matched'] = cs.n_off(calp, meta['ell_matched'], P, DA, st)[0]
        out['results'][cfg] = r
    # pooled over the new configurations, per release type; then also with the earlier spot checks
    runs = {cfg: load_runs(cfg) for cfg in CFG}
    z1 = [np.array(r['z_stage1']) for v in runs.values() for r in v]
    z2 = [np.array(r['z_stage2']) for v in runs.values() for r in v]
    pooled = dict(stage1=mean_tests(z1), stage2=mean_tests(z2))
    o1, o2 = old_noise()
    oz1 = [z for v in o1.values() for z in v]
    oz2 = [z for v in o2.values() for z in v]
    if oz1:
        pooled['earlier_spot_checks'] = dict(stage1=mean_tests(oz1), stage2=mean_tests(oz2),
                                             per_check={k: dict(stage1=mean_tests(o1[k]), stage2=mean_tests(o2[k])) for k in o1})
        pooled['new_plus_earlier'] = dict(stage1=mean_tests(z1 + oz1), stage2=mean_tests(z2 + oz2))
    out['noise_pooled'] = pooled
    # I2 and its replication I2r together (same configuration and data)
    both = [r for cfg in ('I2', 'I2r') for r in runs.get(cfg, []) if not r['refused']]
    if both:
        cov = np.array([(np.abs(np.array(r['theta']) - np.array(r['theta_lam'])) <= np.array(r['half_theta'])) for r in both], float)
        out['I2_with_replication'] = dict(runs=len(both), coverage=mc.coverage_summary(cov),
                                          lam_in_sim_95=int(sum(out['results'][c]['lam_in_sim_95'].split('/')[0] and int(out['results'][c]['lam_in_sim_95'].split('/')[0]) for c in ('I2', 'I2r') if c in out['results'])))
    out['first_runs_only'] = FIRST
    path = os.path.join(mc.REPORT, 'encrypted_more.json' if not FIRST else f'encrypted_more_first{FIRST}.json')
    ev.save_json(out, path)
    if FIRST:
        print('wrote', path)
        return
    # raw standardized noise of every released entry (small: about 120 values per run)
    raw = {f'{cfg}_{s}': np.array([r[f'z_stage{s}'] for r in v], float) for cfg, v in runs.items() if v for s in (1, 2)}
    raw.update({f'{cfg}_yy': np.array([r['z_yy'] for r in v], float) for cfg, v in runs.items() if v})
    np.savez_compressed(os.path.join(mc.REPORT, 'encrypted_more_noise.npz'), **raw)
    print('wrote', path)
    for cfg, r in out['results'].items():
        print(cfg, 'runs', r['runs'], 'lam in', r['lam_in_sim_95'], 'theta in', r['theta_in_sim_95'], 'cov',
              r.get('coverage', {}).get('coverage'), 'noise1', {k: round(v, 4) if isinstance(v, float) else v for k, v in r['noise']['stage1'].items() if k in ('mean', 'z', 'sd', 'ks_p')},
              'noise2', {k: round(v, 4) if isinstance(v, float) else v for k, v in r['noise']['stage2'].items() if k in ('mean', 'z', 'sd', 'ks_p')})
    print(json.dumps(ev.jsonable({k: {kk: vv for kk, vv in v.items() if kk in ('entries', 'mean', 'z', 'p_entry', 'run_p', 'sd', 'ks_p')}
                                  for k, v in pooled.items() if 'entries' in v or True and isinstance(v, dict) and 'mean' in v}), indent=1))


def cmd_plaincov(a):
    """Plaintext Wald coverage of theta_lambda at the encrypted configurations with intervals (I2, I8): --reps
    releases of the plaintext mechanism (stage2_infer, Bboot = 2,000) on the same data with the encrypted runs' sigma."""
    dR_, dO_, b, ps, meta = dataset()
    out = {}
    for cfg in ('I2', 'I8'):   # I2r has the same mechanism and data as I2
        name, eps, _ = CFG[cfg]
        st = mc.setting(name)
        runs = load_runs(cfg)
        sig = runs[0]['sigma'] if runs else None
        cal = cs.calibrate(eps, b['B_R'], b['B_O'], b['B_y'], 4, 6, st['f1'], n_bar_O=meta['n_bar_O'])
        if sig:
            assert abs(sig['1'] / cal['sigma1'] - 1) < 1e-9 and abs(sig['2'] / cal['sigma2'] - 1) < 1e-9
        t0 = time.time()
        o = mc.releases_parallel(ps['G'], ps['c'], ps['yy'], ps['n'], DA, cal, st, b['B_y'], a.reps,
                                 [cs.BASE_SEED, 31, int(eps)], workers=a.workers)
        cv = mc.coverage_summary(o['covered'])
        ok = ~o['refused']
        z = (o['theta'][ok] - o['theta_lam'][ok]) / (o['half_theta'][ok] / mc.solve.Z(1 - (st['alpha'] - st['alpha_L'] - st['alpha_U']) / 2))
        per = np.nansum(o['covered'], axis=1)
        rng = np.random.default_rng(0)
        enc = {}
        for nr in (50, 100):
            tot = per[rng.integers(0, len(per), size=(200000, nr))].sum(axis=1)
            enc[str(nr)] = dict(q025=float(np.quantile(tot, 0.025)), q975=float(np.quantile(tot, 0.975)), mean=float(tot.mean()))
        out[cfg] = dict(setting=name, eps=eps, reps=a.reps, coverage=cv, ridge_on=float(np.mean(o['lam'] > 0)),
                        refused=float(o['refused'].mean()), z_mean=float(z.mean()), z_sd=float(z.std()),
                        batch_totals=enc, lam_q=np.quantile(o['lam'], [0.025, 0.5, 0.975]).tolist(), seconds=time.time() - t0)
        print(cfg, f"coverage {cv['coverage']:.4f} {cv['clustered']} z sd {z.std():.3f} batch 50: {enc['50']}", flush=True)
    ev.save_json(out, os.path.join(mc.REPORT, 'encrypted_plaincov.json'))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('run')
    r.add_argument('--cfg', required=True, choices=list(CFG))
    r.add_argument('--reps', type=int, default=50)
    r.add_argument('--part', type=int, default=0, help='I2r: part index (separate file and seeds)')
    sm = sub.add_parser('summarize')
    sm.add_argument('--first', type=int, default=None, help='only the first N runs of each configuration')
    pc_ = sub.add_parser('plaincov')
    pc_.add_argument('--reps', type=int, default=10000)
    pc_.add_argument('--workers', type=int, default=2)
    a = ap.parse_args()
    {'run': cmd_run, 'summarize': cmd_summarize, 'plaincov': cmd_plaincov}[a.cmd](a)


if __name__ == '__main__':
    main()
