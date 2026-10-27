"""Budgeted sampler (DEVIATIONS A24, paper check item T8): cost, empirical budget failures, value independence of
its work, and the padded release against its deadline.

At the largest sigma_c^2 of the 10^6-row runs (sigma_c = 2^68.5; as in experiments/eng/release_timing.py) and N = 2^14:
  cost     the budgeted sampler (sampler.sample_residues: T fixed trials, limb arithmetic, residues) against the plain
           CKS20 sampler with its residues, interleaved; also at the PSI flooding variance sigma_psi^2;
  failures K calls of the budgeted sampler: failures (expected none; the bound is 2^-80 per call), the surplus
           A - N of accepted trials, and the acceptance rate against the bound p_lo;
  values   the time of one call's trials on crafted coins that force the extreme values (every v = 0, every
           v = J_v - 1, every u = 0, random), against the per-value time of the plain sampler by |eta| / sigma;
  release  the deadline D computed as in party_o.release_deadline with the budgeted sampler, and K releases' work
           (budgeted sampling, residues, re-randomized partial decryption) timed against it, then the same with
           four CPU-bound processes competing for the cores (D still calibrated before them).
Writes reports/v31_timing/timing.json.

    python experiments/eng/release_timing_budget.py [--K_fail 1000] [--K_release 300] [--K_loaded 100]
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'experiments', 'large'))
sys.path.insert(0, HERE)
import lcommon as lc  # noqa: E402
from release_timing import hog, q0_ciphertext, summary  # noqa: E402
from vfr import common as C  # noqa: E402
from vfr import dp, psi  # noqa: E402
from vfr import sampler as S  # noqa: E402
from vfr.party_o import calibrate_release_time  # noqa: E402


def largest_sigma(ck):
    qs = list(ck.qs)
    s2s, cal = {}, None
    for st, kind in (('I', 'std'), ('P', 'std'), ('P', 'bn')):
        for eps in (1.0, 4.0):
            ag = dp.Agreement(n_R_max=20000, n_bar_O=1_007_616, d_R=4, d_O=6, eps_run=eps, delta_run=1e-9,
                              **lc.bounds_for(kind), **dp.setting(st))
            cal = dp.calibrate(ag, ck.N, qs[2], qs[3], qs[0] * qs[1], C.gadget_constants(ck))
            for s, r in cal['releases'].items():
                s2s[f'{st}{kind}_e{eps:g}_s{s}'] = int(r['sigma_c2'])
    key = max(s2s, key=s2s.get)
    return key, s2s[key], cal


def stats(a):
    a = np.asarray(a)
    return dict(n=len(a), median=float(np.median(a)), mean=float(a.mean()), sd=float(a.std()),
                min=float(a.min()), max=float(a.max()))


def cost(s2, N, q0t, reps, rng):
    old, new = [], []
    for _ in range(reps):
        t0 = time.perf_counter()
        S.residues(S.sample_vector(s2, N, rng), q0t)
        old.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        S.sample_residues(s2, N, q0t, rng)
        new.append(time.perf_counter() - t0)
    o, n = stats(old), stats(new)
    return dict(plain_cks=o, budgeted=n, ratio_median=n['median'] / o['median'], ratio_mean=n['mean'] / o['mean'])


def failures(s2, N, K, rng):
    bud = S.budget(s2, N)
    acc, fails, und = [], 0, 0
    for _ in range(K):
        out = S.run_budgeted(s2, N, rng)
        acc.append(out['accepted'])
        fails += out['failed']
        und += out['n_undecided']
    a = np.array(acc)
    sd = np.sqrt(bud['T'] * bud['p_lo'] * (1 - bud['p_lo']))
    return dict(K=K, failures=int(fails), undecided_trials=int(und), T=bud['T'], N=N, J_v=bud['J_v'],
                k_max=bud['k_max'], xi_bound=bud['xi_bound'], log2_xi_bound=bud['log2_xi_bound'], p_lo=bud['p_lo'],
                accepted_mean=float(a.mean()), accepted_min=int(a.min()), accepted_max=int(a.max()),
                surplus_min=int(a.min() - N), acceptance_rate=float(a.mean() / bud['T']),
                surplus_min_in_sd=float((a.min() - N) / sd), trials_per_value=bud['T'] / N)


def crafted(s2, N, reps, rng):
    """Time _trials on coins forcing extreme values (same budget, same shapes)."""
    bud = S.budget(s2, N)
    T, Ly, Jv = bud['T'], bud['Ly'], bud['J_v']
    nl = S.coin_limbs(bud)

    def coins(kind):
        raw = np.frombuffer(rng.bytes(4 * nl * T), dtype='<u4').reshape(nl, T).astype(np.uint64)
        if kind == 'v0':                                  # U >= e^-1: every v = 0
            raw[1 + Ly:5 + Ly] = 0xFFFFFFFF
        elif kind == 'vmax':                              # every v = J_v - 1 (largest |y|; almost all rejected)
            x = S.EXP_TABLE[Jv - 1] - 1
            for i in range(4):
                raw[1 + Ly + i] = (x >> (32 * i)) & 0xFFFFFFFF
        elif kind == 'u0':
            raw[1:1 + Ly] = 0
        return raw
    kinds = ('random', 'v0', 'vmax', 'u0')
    ts = {k: [] for k in kinds}
    acc = {}
    for _ in range(reps):
        for k in kinds:
            raw = coins(k)
            t0 = time.perf_counter()
            _, _, a, _ = S._trials(bud, raw=raw)
            ts[k].append(time.perf_counter() - t0)
            acc[k] = int(a.sum())
    st = {k: stats(v) for k, v in ts.items()}
    med = [st[k]['median'] for k in kinds]
    return dict(per_kind=st, accepted_last=acc, max_rel_diff_median=float((max(med) - min(med)) / min(med)))


def plain_by_value(s2, n, rng):
    """Per-value time of the plain CKS20 sampler, binned by |eta| / sigma (its work grows with |eta|)."""
    sig = float(s2) ** 0.5
    ts, zs = [], []
    for _ in range(n):
        t0 = time.perf_counter()
        y = S.discrete_gaussian(rng, s2)
        ts.append(time.perf_counter() - t0)
        zs.append(abs(float(y)) / sig)
    ts, zs = np.array(ts), np.array(zs)
    out = {}
    for lo, hi in ((0, 1), (1, 2), (2, 3), (3, 99)):
        m = (zs >= lo) & (zs < hi)
        out[f'{lo}-{hi}'] = dict(n=int(m.sum()), median_us=float(np.median(ts[m]) * 1e6),
                                 mean_us=float(ts[m].mean() * 1e6))
    out['corr_time_abs_eta'] = float(np.corrcoef(ts, zs)[0, 1])
    return out


def time_releases(ck, s2, ct, pk, sO, q0t, K, rng):
    ts = []
    for _ in range(K):
        t0 = time.perf_counter()
        eta = S.sample_residues(s2, ck.N, q0t, rng)
        ck.rerand_partial_decrypt(ct, pk, sO, eta)
        ts.append(time.perf_counter() - t0)
    return ts


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--K_cost', type=int, default=40)
    ap.add_argument('--K_fail', type=int, default=1000)
    ap.add_argument('--K_fail_psi', type=int, default=300)
    ap.add_argument('--K_crafted', type=int, default=30)
    ap.add_argument('--n_plain', type=int, default=20000)
    ap.add_argument('--K_release', type=int, default=300)
    ap.add_argument('--K_loaded', type=int, default=100)
    ap.add_argument('--margin', type=float, default=2.0)
    ap.add_argument('--out', default=os.path.join(ROOT, 'reports', 'v31_timing', 'timing.json'))
    a = ap.parse_args()
    from vfr import vfr_native as vn
    vn.set_threads(4)
    ck = C.make_ckks(C.CKKSParams())
    qs = list(ck.qs)
    q0t = [qs[0], qs[1]]
    key, s2, cal = largest_sigma(ck)
    pp = psi.PSIParams()
    s2psi = pp.flood_sigma2()
    keep_psi = [(1 << 60) - 93, (1 << 59) + 12345677]        # two ~60-bit residues, as the reply's two limbs
    rng = S.CSPRNG()
    res = dict(stamp=time.strftime('%Y-%m-%d %H:%M'), load_before=os.getloadavg(), cpu_count=os.cpu_count(),
               sigma_c2_used=key, log2_sigma_c=s2.bit_length() / 2, N=ck.N, log2_sigma_psi=s2psi.bit_length() / 2,
               xi=S.XI_DEFAULT, budget_release=S.budget(s2, ck.N), budget_psi=S.budget(s2psi, pp.N_B))
    res['budget_release'].pop('C')
    res['budget_psi'].pop('C')
    res['cost_release'] = cost(s2, ck.N, q0t, a.K_cost, rng)
    print('cost', res['cost_release'], flush=True)
    res['cost_psi'] = cost(s2psi, pp.N_B, keep_psi, max(10, a.K_cost // 2), rng)
    print('cost psi', res['cost_psi'], flush=True)
    res['failures_release'] = failures(s2, ck.N, a.K_fail, rng)
    print('failures', res['failures_release'], flush=True)
    res['failures_psi'] = failures(s2psi, pp.N_B, a.K_fail_psi, rng)
    print('failures psi', res['failures_psi'], flush=True)
    res['crafted_values'] = crafted(s2, ck.N, a.K_crafted, rng)
    print('crafted', res['crafted_values'], flush=True)
    res['plain_by_value'] = plain_by_value(s2, a.n_plain, rng)
    print('plain', res['plain_by_value'], flush=True)
    ct, pk, sO = q0_ciphertext(ck, cal)
    c = calibrate_release_time(ck, s2, ct, pk, sO, q0t)
    deadline = a.margin * (c['sample_max'] + c['rerand_pdec'])
    res['calibration'] = c
    res['deadline_s'] = deadline
    res['load_release'] = os.getloadavg()
    res['release'] = summary(time_releases(ck, s2, ct, pk, sO, q0t, a.K_release, rng), deadline)
    print('release', res['release'], flush=True)
    stop = time.time() + 3600
    hogs = [mp.Process(target=hog, args=(stop,)) for _ in range(4)]
    for h in hogs:
        h.start()
    try:
        res['release_loaded_4_hogs'] = summary(time_releases(ck, s2, ct, pk, sO, q0t, a.K_loaded, rng), deadline)
    finally:
        for h in hogs:
            h.terminate()
    print('loaded', res['release_loaded_4_hogs'], flush=True)
    res['load_after'] = os.getloadavg()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, 'w') as fh:
        json.dump(res, fh, indent=1)
