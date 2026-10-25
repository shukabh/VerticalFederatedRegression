"""Does correcting FWS's plug-in residual variance change FWS's coverage where the ridge is on?
Same cohorts, releases and FWS random numbers as experiments/baselines/sim_fws.py (synthetic population, setting I,
fws_s2 = FWS on the stage-2 release).  For every cohort FWS is run twice with the same bootstrap draws:
  plug-in:   s2 = (yy2 - c2' b) / (n - p)                                   (as published; floor 0.1)
  corrected: s2 = (yy2 - c2' b + sigma2^2 (tr P + b' C_chi(P) b)) / (n - p), P = G2^{-1}, b = P c2   (floor 0.1)
where the added term is the leading-order bias of the plug-in RSS (stats proofs, Part B, proposition on the FWS
residual variance).  v3.1's own bootstrap (solve.stage2_infer) is skipped: it consumes a separate random stream.
    python3 experiments/proofs_check/fws_s2_corrected.py [reps]
"""
import math, os, sys, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'baselines'))
import bcommon as bc
from bcommon import cs, ev, pc, solve
import fws, sim_fws as sf

reps = int(sys.argv[1]) if len(sys.argv) > 1 else 200
P, DA = pc.P, pc.DA
mask = np.ones((P, P)); mask[:DA, :DA] = 0
PiO = np.diag([0.0] * DA + [1.0] * (P - DA))

def C_chi(Pm):
    return mask * Pm + np.diag(mask @ np.diag(Pm)) - PiO @ np.diag(np.diag(Pm))

def fws_pair(G2, c2, yy2, n, sdG, sdc, rng, sigma2, level):
    st = rng.bit_generator.state
    o1 = fws.fws_interval(G2, c2, yy2, n, sdG, sdc, rng, B=sf.B_FWS, level=level)
    rng.bit_generator.state = st
    b = np.linalg.solve(G2, c2); Pm = np.linalg.inv(G2)
    corr = sigma2 ** 2 * (np.trace(Pm) + b @ C_chi(Pm) @ b)
    o2 = fws.fws_interval(G2, c2, yy2 + corr, n, sdG, sdc, rng, B=sf.B_FWS, level=level)   # adds corr to the RSS
    return o1, o2

st = bc.settings()['I']
pop = pc.population(sf.R2); M = pc.pop_moments('std', sf.R2); beta_clip = np.asarray(M['beta_clip'])
level = 1 - (st['alpha'] - st['alpha_L'] - st['alpha_U'])
results = []
for n in sorted(sf.NS):
    cal = {e: cs.calibrate(e, pop.B_R, pop.B_O, pop.B_y, pop.d_R, pop.d_O, st['f1']) for e in bc.EPS}
    ss = np.random.SeedSequence([cs.BASE_SEED, int(round(sf.R2 * 100)), n])
    ss_data, ss_inf, *ss_eps = ss.spawn(2 + len(bc.EPS))
    rng_d = np.random.default_rng(ss_data)
    Gs = np.empty((reps, P, P)); cvec = np.empty((reps, P)); yys = np.empty(reps)
    for r in range(reps):
        Z, y, _ = pop.draw(n, rng_d)
        Gs[r], cvec[r], yys[r] = Z.T @ Z, Z.T @ y, float(y @ y)
    ss_x = np.random.SeedSequence([cs.BASE_SEED, int(round(sf.R2 * 100)), n, 4242]).spawn(3 * len(bc.EPS))
    for k, eps in enumerate(bc.EPS):
        rng_stage = np.random.default_rng(ss_eps[k]); rng_f2 = np.random.default_rng(ss_x[3 * k])
        s1 = cs.stage1_batch(Gs, DA, cal[eps], rng_stage, reps, st)
        if not np.all(s1['lam'] > 0):
            continue                                  # ridge-on cells only
        E, f, g = ev.stage2_noise(P, DA, cal[eps], rng_stage, reps)
        G2, c2, yy2 = Gs + E, cvec + f, yys + g
        sdG = cal[eps]['sigma2'] * mask; sdc = np.full(P, cal[eps]['sigma2'])
        cov = {'plug': [], 'corr': []}; s2v = {'plug': [], 'corr': []}; half = {'plug': [], 'corr': []}
        for r in range(reps):
            o1, o2 = fws_pair(G2[r], c2[r], yy2[r], n, sdG, sdc, rng_f2, cal[eps]['sigma2'], level)
            for tag, o in (('plug', o1), ('corr', o2)):
                cov[tag].append(((o['lo'] <= beta_clip) & (beta_clip <= o['hi']))[1:])
                s2v[tag].append(o['s2']); half[tag].append(((o['hi'] - o['lo']) / 2)[1:])
        row = dict(n=n, eps=eps, **{f'cov_{t}': float(np.mean(cov[t])) for t in cov},
                   **{f's2_median_{t}': float(np.median(s2v[t])) for t in s2v},
                   **{f'half_median_{t}': float(np.median(half[t])) for t in half},
                   s2_floored_plug=float(np.mean(np.array(s2v['plug']) == 0.1)))
        results.append(row)
        print(json.dumps(row), flush=True)
os.makedirs(os.path.join(HERE, 'out'), exist_ok=True)
json.dump(results, open(os.path.join(HERE, 'out', 'fws_s2_corrected.json'), 'w'), indent=1)
print('cells', len(results), 'coverage plug-in', [round(r['cov_plug'], 3) for r in results])
print('coverage corrected', [round(r['cov_corr'], 3) for r in results])
