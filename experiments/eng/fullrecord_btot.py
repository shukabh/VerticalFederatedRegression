"""Check item (full-record SSP, former #13): B_tot and the derived parameters of the FULL-RECORD circuit, re-derived.

The benchmark path of experiments/eng/fullrecord_bench.py (commit 49b4f7d) aggregated and released the full-record
statistics with the v3.1 calibration, and experiments/v31_more/cost_fullrecord.py re-evaluated Calibrate with the
entry counts increased only.  Here Calibrate (vfr/dp.py, fullrecord_sens=...) runs on the full-record circuit as
re-derived in Lemma S10e (docs/proofs/stats_s7_s10.tex; vfr/btot.py, CircuitParams.fullrecord):
  * B_tot: |E| + n_A entries and Gamma_E + n_A (O's vectors of ones); R's vectors do not enter any error term;
  * Delta_x: nu_U with the U-side bounds of the new vectors (B_R for the first moments, B_R^2 for the products) and
    the whole-record adjacency (R's row changes too: Omega_U nu_dU + sqrt(N) Omega_V nu_Vmax more), epsilon_x doubled;
  * the decoding offset d_max (diagnostic condition (Q)): nu_U, Gamma_E, |E| and zeta_E of the full-record circuit;
  * the whole-record sensitivities of the full-record release (experiments/baselines/sens_full.py, exact).
Three calibrations at the benchmark's agreement (simulation population, d_R = 4, d_O = 6, n_R^max = 2,000, the
default setting): 'v31' (what the benchmark path used), 'entries_only' (the earlier re-evaluation: v3.1 Calibrate
with 14 more entries per release) and 'fullrecord' (re-derived).  Also at the paper's evaluation agreement
(eps = 1, delta = 1e-9, setting I, n_bar_O = 65,536, as cost_fullrecord.calib_effect) and at 10^6 register rows.

    python experiments/eng/fullrecord_btot.py          # writes reports/eng/fullrecord_btot.json
"""
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'implementation'))
sys.path.insert(0, os.path.join(ROOT, 'experiments', 'baselines'))
from vfr import btot as bt, common as C, data, dp  # noqa: E402
import sens_full  # noqa: E402

NR_MAX = 2000


def summarize(cal):
    rel = cal['releases']
    return dict(log2_Omega_f=cal['log2_Omega_f'], Omega_V=cal['Omega_V'], kappa=cal['kappa'], joint=cal['joint'],
                entries=cal['entries'], Delta=cal['Delta'], Delta_noyy=cal['Delta_noyy'], Delta_G=cal['Delta_G'],
                q0_bits=cal['q0_bits'], q0_need_bits=cal['q0_need_bits'], q0_need_Q_bits=cal['q0_need_Q_bits'],
                log2_d_max=cal['log2_d_max'],
                releases={s: dict(log2_B_tot=math.log2(r['B_tot']), log2_Delta_x=math.log2(r['Delta_x']),
                                  sigma_c2=r['sigma_c2'], log2_sigma_c=math.log2(r['sigma_c']),
                                  log2_flood_needed=math.log2(r['flood_needed']),
                                  flood_margin_bits=math.log2(r['sigma_c']) - math.log2(r['flood_needed']),
                                  sigma_slot=r['sigma_slot'], log2_d_max=math.log2(r['d_max']),
                                  epsilon_x=r.get('epsilon_x'))
                          for s, r in rel.items()})


def three(ag, ck):
    qs = list(ck.qs)
    args = (ck.N, qs[2], qs[3], qs[0] * qs[1], C.gadget_constants(ck))
    out = {'v31': summarize(dp.calibrate(ag, *args))}
    orig = bt.entry_counts
    nA = bt.n_A(ag.d_R)
    bt.entry_counts = lambda dR, dO, joint, fullrecord=False: {k: v + (nA if k != 'y' else 0)
                                                              for k, v in orig(dR, dO, joint).items()}
    try:
        out['entries_only'] = summarize(dp.calibrate(ag, *args))
    finally:
        bt.entry_counts = orig
    sens = sens_full.delta_full(ag.B_R, ag.B_O, ag.B_y, True, ag.d_R, ag.d_O)
    out['fullrecord'] = summarize(dp.calibrate(ag, *args, fullrecord_sens=sens))
    out['fullrecord_sens_method'] = sens_full.delta_full_info(float(ag.B_R), float(ag.B_O), float(ag.B_y), True,
                                                              ag.d_R, ag.d_O)['method']
    # B_tot of the full-record circuit at the v3.1 scale (the flooding test the benchmark's releases would face)
    v = out['v31']
    cp = bt.CircuitParams(N=ck.N, n_bar=ag.n_bar_O, d_R=ag.d_R, d_O=ag.d_O, B_R=ag.B_R, B_O=ag.B_O, B_y=ag.B_y,
                          q1=qs[2], q2=qs[3], Omega_U=float(dp.scales_from_primes(qs[2], qs[3],
                          dp.resolve_log2_Omega_U(ag.log2_Omega_U, ag.variant))[0]), Omega_V=v['Omega_V'],
                          Omega_u=float(qs[2]), delta_B=ag.delta_B, fullrecord=True, **C.gadget_constants(ck))
    out['fullrecord_Btot_at_v31_scale'] = {
        s: dict(log2_B_tot=math.log2(bt.B_tot(cp, int(s), v['joint'], ag.variant)),
                floods_with_v31_sigma=dp.floods(r['sigma_c2'], bt.B_tot(cp, int(s), v['joint'], ag.variant), v['kappa']))
        for s, r in v['releases'].items() if s in ('1', '2')}
    return out


def main():
    ck = C.make_ckks(C.CKKSParams())
    res = dict(note=__doc__.split('\n\n')[0], n_A=bt.n_A(4), rows=[])
    for nbar in (16384, 65536):
        n_scope = int(0.9 * nbar)
        n_R = NR_MAX - 100
        n = min(n_R, max(50, nbar // 10))
        _, _, _, b = data.synthetic(nbar, n_scope, n_R, n, seed=nbar)       # as experiments/eng/fullrecord_bench.py
        ag = dp.Agreement(n_R_max=NR_MAX, n_bar_O=nbar, d_R=4, d_O=6, release_pad=None, **b)
        res['rows'].append(dict(agreement='benchmark (fullrecord_bench.py)', n_bar_O=nbar, bounds=b,
                                eps_run=ag.eps_run, delta_run=ag.delta_run, **three(ag, ck)))
    for nbar in (65536, 10 ** 6):
        ag = dp.Agreement(n_R_max=NR_MAX, n_bar_O=nbar, d_R=4, d_O=6, B_R=5.59, B_O=6.24, B_y=2.89, eps_run=1.0,
                          delta_run=1e-9, **dp.setting('I'))
        res['rows'].append(dict(agreement='paper evaluation, setting I, eps 1, delta 1e-9', n_bar_O=nbar,
                                bounds=dict(B_R=5.59, B_O=6.24, B_y=2.89), eps_run=1.0, delta_run=1e-9, **three(ag, ck)))
    path = os.path.join(ROOT, 'reports', 'eng', 'fullrecord_btot.json')
    with open(path, 'w') as fh:
        json.dump(res, fh, indent=1, default=float)
    for r in res['rows']:
        print(r['agreement'], r['n_bar_O'])
        for k in ('v31', 'entries_only', 'fullrecord'):
            x = r[k]
            print(f"  {k:13s} log2 Omega_f {x['log2_Omega_f']:.4f} kappa {x['kappa']} q0 need {x['q0_need_bits']:.2f}/{x['q0_bits']:.2f} "
                  + ' '.join(f"[{s}: B_tot 2^{v['log2_B_tot']:.3f} Dx 2^{v['log2_Delta_x']:.2f} sig_slot {v['sigma_slot']:.4g} "
                             f"margin {v['flood_margin_bits']:.3f}b dmax 2^{v['log2_d_max']:.2f}]"
                             for s, v in x['releases'].items()))
        print('  full-record B_tot at the v3.1 scale:', r['fullrecord_Btot_at_v31_scale'])
    print('wrote', path)


if __name__ == '__main__':
    main()
