"""Check item #34: the cryptographic cost of full-record SSP inside the v3.1 protocol, from the implementation's cost
model (implementation/bench/cost_model.py, per-operation costs measured by bench_micro.py in
implementation/bench/results/micro_t1.json, single thread, N = 2^14).  Nothing is run at scale.

What full-record SSP adds.  Its releases also contain triu A except the constant n, where A is the Gram block of R's
features with the intercept: (d_R + 1)(d_R + 2)/2 - 1 = d_R(d_R + 1)/2 + d_R more entries per release (14 at
d_R = 4).  In the protocol's architecture (vfr/aggregate.py: T_e = EvalSum(sum_h U_e[h] * Ecd(V_e[h])), packed
one entry per slot) they are aggregated as follows:
  * sum_t b_t x_{R,j,t} (d_R entries): R's existing upload U = b * xdot_j, O's vector V = 1;
  * sum_t b_t x_{R,j,t} x_{R,k,t} (d_R(d_R + 1)/2 entries): R uploads d_R(d_R + 1)/2 more encrypted vectors
    b * x_j * x_k (R knows b and its features in O's order), O's vector V = 1.
So per run: d_R(d_R + 1)/2 more upload vectors (R encrypts, O converts, more bytes) and d_R(d_R + 1)/2 + d_R more
aggregated entries (mul_plain + add per chunk, one EvalSum, rescale and mask).  The encode of O's vector is counted
per chunk as the implementation does it (upper bound); with V = 1 it can be one cached plaintext (lower bound).
The releases do not change: each is one ciphertext whose noise is sampled per ring coefficient, and 65 and 77 entries
fit in the 8,192 slots as 51 and 63 do; R decodes a few more slots.  A secure-aggregation protocol of another kind
(Karr et al., Gascon et al.) would have its own cost, not modelled here.

The worst-case encryption error B_tot (vfr/btot.py) grows with the number of packed entries; its effect on the
calibration is reported by re-running Calibrate with the entry counts increased (the U-side bound of the new
vectors, |b x_j x_k| <= B_R^2, is not re-derived; see the README).

    python experiments/v31_more/cost_fullrecord.py
"""
from __future__ import annotations

import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, '..', '..'))
BENCH = os.path.join(ROOT, 'implementation', 'bench')
sys.path.insert(0, BENCH)
sys.path.insert(0, os.path.join(ROOT, 'implementation'))
import cost_model as CM  # noqa: E402
from vfr import btot as bt, common as C, dp  # noqa: E402

REPORT = os.path.join(ROOT, 'reports', 'v31_more')
MB = 2 ** 20
SIZES = (4096, 16384, 65536)
NR_MAX = 2000


def extra(nbar, d_R, mu, cached_ones=False):
    """Extra single-thread seconds and bytes of full-record SSP over v3.1 at register size nbar."""
    Ns = 2 ** 14 // 2
    m = -(-nbar // Ns)
    n_new_vec = d_R * (d_R + 1) // 2
    n_new_ent = n_new_vec + d_R
    t = {}
    t['O.agg.encode'] = (1 if cached_ones else n_new_ent * m) * mu['ckks_encode_dd']
    t['O.agg.mul_add'] = n_new_ent * m * (mu['ckks_mul_plain_L0'] + mu['ckks_add_L0'])
    t['O.agg.evalsum'] = n_new_ent * mu['ckks_evalsum_L1']
    t['O.agg.rescale_mask'] = n_new_ent * (mu['ckks_rescale_det_L0'] + mu['ckks_mul_plain_L1'] + mu['ckks_rescale_det_L1'])
    t['O.upload_convert'] = n_new_vec * m * mu['ckks_convert_O']
    t['R.upload_encrypt'] = n_new_vec * m * mu['ckks_encrypt_seeded_R']
    t['O.release'] = 0.0
    t['R.release'] = 0.0
    b = {'R->O upload': n_new_vec * m * mu['size_seeded_upload_bytes'], 'O->R releases': 0}
    return dict(m=m, new_vectors=n_new_vec, new_entries=n_new_ent, t=t, bytes=b,
                agg=sum(v for k, v in t.items() if k.startswith('O.agg')),
                cpu_O=sum(v for k, v in t.items() if k.startswith('O.')),
                cpu_R=sum(v for k, v in t.items() if k.startswith('R.')), comm=sum(b.values()))


def calib_effect(d_R, d_O, n_new):
    """Calibrate with v3.1's entry counts and with n_new more entries in each release."""
    ck = C.make_ckks(C.CKKSParams())
    qs = list(ck.qs)
    ag = dp.Agreement(n_R_max=NR_MAX, n_bar_O=65536, d_R=d_R, d_O=d_O, B_R=5.59, B_O=6.24, B_y=2.89, eps_run=1.0,
                      delta_run=1e-9, **dp.setting('I'))
    out = {}
    orig = bt.entry_counts
    for tag, add in (('v31', 0), ('full_record', n_new)):
        bt.entry_counts = (lambda a: (lambda dR, dO, joint: {k: v + (a if k != 'y' else 0) for k, v in orig(dR, dO, joint).items()}))(add)
        try:
            cal = dp.calibrate(ag, ck.N, qs[2], qs[3], qs[0] * qs[1], C.gadget_constants(ck))
            out[tag] = dict(entries=cal.get('entries'), log2_Omega_f=cal['log2_Omega_f'], kappa=cal.get('kappa'),
                            log2_B_tot={s: math.log2(r['B_tot']) for s, r in cal['releases'].items() if 'B_tot' in r},
                            sigma_slot={s: r['sigma_slot'] for s, r in cal['releases'].items()})
        except Exception as e:  # noqa: BLE001
            out[tag] = dict(error=repr(e))
        finally:
            bt.entry_counts = orig
    return out


def main():
    mu = CM.load('micro_t1.json')
    fit = CM.load('cost_model.json')['fit']
    e2e = {r['n_bar_O']: r for r in CM.load('e2e_A_t1_w1.json')}
    rows = []
    for nbar in SIZES:
        base = CM.model(nbar, NR_MAX, mu, fit)
        assert CM.N_ENT == 63 and CM.N_VEC == 6 and CM.D_R == 4 and CM.D_O == 6
        ex = extra(nbar, CM.D_R, mu)
        exc = extra(nbar, CM.D_R, mu, cached_ones=True)
        agg_b = sum(v for k, v in base['t'].items() if k.startswith('O.agg'))
        meas = e2e.get(nbar)
        rel_b = base['t']['O.release'] + base['t']['R.release']
        rows.append(dict(
            n_bar_O=nbar, m=base['m'], d_R=CM.D_R, d_O=CM.D_O, n_R_max=NR_MAX,
            v31=dict(entries=CM.N_ENT, upload_vectors=CM.N_VEC, agg=agg_b, cpu_O=base['cpu_O'], cpu_R=base['cpu_R'],
                     comm_MB=base['comm'] / MB, upload_MB=base['bytes']['R->O upload'] / MB, release_s=rel_b,
                     measured_wall=meas['wall'] if meas else None, measured_agg=meas['O']['aggregate'] if meas else None),
            full_record=dict(entries=CM.N_ENT + ex['new_entries'], upload_vectors=CM.N_VEC + ex['new_vectors'],
                             extra_agg=ex['agg'], extra_agg_cached_ones=exc['agg'], extra_cpu_O=ex['cpu_O'],
                             extra_cpu_R=ex['cpu_R'], extra_comm_MB=ex['comm'] / MB, extra_release_s=0.0,
                             breakdown=ex['t']),
            ratio=dict(agg=(agg_b + ex['agg']) / agg_b, agg_cached=(agg_b + exc['agg']) / agg_b,
                       cpu_O=(base['cpu_O'] + ex['cpu_O']) / base['cpu_O'], cpu_R=(base['cpu_R'] + ex['cpu_R']) / base['cpu_R'],
                       comm=(base['comm'] + ex['comm']) / base['comm'],
                       wall_single_thread_est=((meas['wall'] + ex['cpu_O'] + ex['cpu_R']) / meas['wall']) if meas else None)))
    # the paper's two-party dimensions at other sizes: the e2e register sizes of Section 12 (24,576 and 32,768)
    extra_rows = []
    for nbar, dR in ((24576, 4), (32768, 4)):
        base = CM.model(nbar, NR_MAX, mu, fit)
        ex = extra(nbar, dR, mu)
        agg_b = sum(v for k, v in base['t'].items() if k.startswith('O.agg'))
        extra_rows.append(dict(n_bar_O=nbar, extra_agg=ex['agg'], agg=agg_b, ratio_agg=(agg_b + ex['agg']) / agg_b,
                               extra_cpu=ex['cpu_O'] + ex['cpu_R'], cpu=base['cpu_O'] + base['cpu_R'],
                               extra_comm_MB=ex['comm'] / MB, comm_MB=base['comm'] / MB))
    calib = calib_effect(4, 6, 14)
    out = dict(source='implementation/bench/cost_model.py with micro_t1.json (single thread, N = 2^14)', rows=rows,
               other_sizes=extra_rows, calibration=calib,
               note=('extra entries per release: d_R(d_R+1)/2 + d_R = 14 at d_R = 4 (triu A without n, intercept '
                     'included); extra upload vectors d_R(d_R+1)/2 = 10; releases unchanged (one ciphertext each)'))
    with open(os.path.join(REPORT, 'cost_fullrecord.json'), 'w') as fh:
        json.dump(out, fh, indent=1)
    for r in rows:
        v, f, q = r['v31'], r['full_record'], r['ratio']
        print(f"n_bar={r['n_bar_O']:6d} m={r['m']}: agg {v['agg']:.1f}s + {f['extra_agg']:.2f}s (x{q['agg']:.3f}; cached x{q['agg_cached']:.3f}); "
              f"O CPU {v['cpu_O']:.1f}s + {f['extra_cpu_O']:.2f}s (x{q['cpu_O']:.3f}); R CPU {v['cpu_R']:.2f}s + {f['extra_cpu_R']:.2f}s; "
              f"comm {v['comm_MB']:.1f} MB + {f['extra_comm_MB']:.1f} MB (x{q['comm']:.3f}); wall est x{q['wall_single_thread_est']:.3f}")
    print(extra_rows)
    print(json.dumps(calib, indent=1))


if __name__ == '__main__':
    main()
