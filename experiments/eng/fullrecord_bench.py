"""Check item #14: the measured cryptographic cost of full-record SSP inside the v3.1 architecture.

A benchmark path, not a protocol mode.  Full-record SSP also releases triu A without the constant n: 14 more entries
at d_R = 4.  The four sums sum_t b_t x_{R,j,t} come from R's existing uploads (U = xdot_j, O's vector V = 1); the ten
products sum_t b_t x_{R,j,t} x_{R,k,t} need ten more encrypted upload vectors b * x_j * x_k from R (V = 1).  This
script measures, at n_bar_O = 16,384 and 65,536 (n_R^max = 2,000, d_R = 4, d_O = 6, single thread, as the cost model's
x1.16 / x1.10-1.55 estimate in experiments/v31_more/cost_fullrecord.py):

  1. a full v3.1 run of the implementation (two processes, noise on, release padding off as in the model): the
     baseline wall time, phases and traffic;
  2. in one process with real keys: R's seeded encryption and O's conversion of the 6 v3.1 vectors and of the 10
     extra ones; O's aggregation (circuit A, the same AG.aggregate call as party O) of the 63 v3.1 entries and of the
     77 full-record entries (the extra entries use O's vector V = 1, supplied by wrapping AG.v_vector here); one
     release of each packed ciphertext (exact sampling, re-randomization, partial decryption; R's combination and
     decoding); and the exact byte count of the extra uploads.  A noiseless decryption (test-only path, both key
     shares in this process) checks that the 14 extra entries decode to triu A.

The measured full-record cost is the baseline run plus the measured differences.  Writes reports/eng/fullrecord.json.

Calibration (since the re-derivation of B_tot for the full-record circuit, Lemma S10e of docs/proofs/stats_s7_s10.tex):
the full-record path is aggregated, released and decoded with its OWN calibration, dp.calibrate(...,
fullrecord_sens=...): the whole-record sensitivities (experiments/baselines/sens_full.py), the full-record entry counts
and B_tot, Delta_x and offset of the full-record circuit (vfr/btot.py, CircuitParams.fullrecord).  The v3.1 path keeps
the v3.1 calibration.  The first version (commit 49b4f7d) used the v3.1 calibration for both, under which the
full-record stage-2 release does not flood the full-record B_tot (experiments/eng/fullrecord_btot.py).
--calib v31 reproduces that first version.

    python experiments/eng/fullrecord_bench.py [--sizes 16384 65536] [--calib fullrecord|v31]
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, '..', '..'))
IMPL = os.path.join(ROOT, 'implementation')
sys.path.insert(0, IMPL)
sys.path.insert(0, os.path.join(IMPL, 'tests'))
sys.path.insert(0, os.path.join(ROOT, 'experiments', 'baselines'))
import sens_full  # noqa: E402
from vfr import aggregate as AG  # noqa: E402
from vfr import common as C  # noqa: E402
from vfr import data, dp, protocol  # noqa: E402
from vfr import sampler as S  # noqa: E402
import helpers  # noqa: E402  (implementation/tests/helpers.py: tau_vectors)

NR_MAX = 2000
MB = 2 ** 20
_v_orig = AG.v_vector


def _v_vector(spec, XO, yt):
    if spec[0] == 'one':                                 # benchmark path only: O's vector of ones
        return np.ones(XO.shape[0])
    return _v_orig(spec, XO, yt)


def fullrecord_entries(d_R, d_O, joint):
    """v3.1's entries plus triu A without n: sum b xdot_j (j = 1..d_R, U = 1 + j) and sum b x_j x_k (U = new vector)."""
    ents = C.entries(d_R, d_O, joint)
    extra, prods = [], []
    for j in range(1, d_R + 1):
        extra.append(dict(name=f'A_0{j}', kind='A', U=1 + j, V=('one',), rel={1, 2}))
    v = d_R + 2
    for j in range(1, d_R + 1):
        for k in range(j, d_R + 1):
            extra.append(dict(name=f'A_{j}{k}', kind='A', U=v, V=('one',), rel={1, 2}))
            prods.append((j, k))
            v += 1
    out = ents + extra
    for i, e in enumerate(out):
        e['slot'] = i
    return out, prods


def baseline_run(nbar, wd):
    n_scope = int(0.9 * nbar)
    n_R = NR_MAX - 100
    n = min(n_R, max(50, nbar // 10))
    dR_, dO_, truth, b = data.synthetic(nbar, n_scope, n_R, n, seed=nbar)
    ag = dp.Agreement(n_R_max=NR_MAX, n_bar_O=nbar, d_R=4, d_O=6, release_pad=None, **b)
    omp = dict(omp_threads=1)
    led = os.path.join(wd, f'ledger_fr_{nbar}.json')
    if os.path.exists(led):                              # a fresh single-run ledger for every run
        os.remove(led)
    t = time.time()
    R, O = protocol.run(ag, C.CKKSParams(), dR_, dO_, dict(threads=1, Bboot=1000, **omp),
                        dict(threads=1, entry_workers=1, **omp, ledger_path=led))
    if 'refused' in R:
        raise RuntimeError(R['refused'])
    wall = time.time() - t
    comm = sum(v['bytes'] for v in R['summary'].values())
    return dict(wall=wall, comm_bytes=comm, upload_bytes=sum(v['bytes'] for k, v in R['summary'].items() if k.endswith(':upload')), O=O['timings'],
                R=R['timings'], psi_exact=set(R['I']) == truth['matched']), (dR_, dO_, b, ag)


def bench_path(nbar, dR_, dO_, b, ag, calib='fullrecord'):
    from vfr import vfr_native as vn
    vn.set_threads(1)
    ck = C.make_ckks(C.CKKSParams())
    qs = list(ck.qs)
    q0t = [qs[0], qs[1]]
    cal = dp.calibrate(ag, ck.N, qs[2], qs[3], qs[0] * qs[1], C.gadget_constants(ck))
    if calib == 'fullrecord':                            # the full-record circuit's own calibration (Lemma S10e)
        sens = sens_full.delta_full(ag.B_R, ag.B_O, ag.B_y, True, ag.d_R, ag.d_O)
        cal_f = dp.calibrate(ag, ck.N, qs[2], qs[3], qs[0] * qs[1], C.gadget_constants(ck), fullrecord_sens=sens)
    else:
        cal_f = cal
    assert cal_f['Omega_U'] == cal['Omega_U'] and cal_f['Omega_u'] == cal['Omega_u']
    crs = os.urandom(32)
    sR, sO = ck.sample_secret(), ck.sample_secret()
    pk = ck.make_pk([ck.pk_share(sR, crs), ck.pk_share(sO, crs)], crs)
    km = ck.make_rotkeys([ck.rotkey_share(sR, crs), ck.rotkey_share(sO, crs)], crs)
    ids_O = list(dO_['ids']) + [f'dummy:{i}' for i in range(nbar - len(dO_['ids']))]
    perm = np.random.default_rng(nbar).permutation(nbar) + 1
    tau = {i: int(p) for i, p in zip(ids_O, perm)}
    vecs, XOt, yt = helpers.tau_vectors(dR_, dO_, tau, ag)
    ents, prods = fullrecord_entries(ag.d_R, ag.d_O, cal_f['joint'])
    extra_vecs = np.array([vecs[0] * vecs[1 + j] * vecs[1 + k] for j, k in prods])   # b * x_j * x_k (xdot_j = x_j)
    Ns = ck.N // 2
    m = -(-nbar // Ns)
    OmU = C.dd(cal['Omega_U'])

    def upload(V):
        enc = conv = 0.0
        nbytes = 0
        out = []
        for v in range(V.shape[0]):
            row = []
            for h in range(m):
                seg = np.zeros(Ns)
                ch = V[v, h * Ns:(h + 1) * Ns]
                seg[:len(ch)] = ch
                seed = os.urandom(32)
                t0 = time.perf_counter()
                c0 = ck.encrypt_seeded(sR, seg, OmU, seed)
                t1 = time.perf_counter()
                row.append(ck.convert(c0, seed, sO, OmU))
                t2 = time.perf_counter()
                enc += t1 - t0
                conv += t2 - t1
                nbytes += c0.nbytes + len(seed)
            out.append(row)
        return out, dict(encrypt_R=enc, convert_O=conv, bytes=nbytes, vectors=int(V.shape[0]))

    U, up_base = upload(vecs)
    Ux, up_extra = upload(extra_vecs)
    stages = [1, 2]
    AG.v_vector = _v_vector
    try:
        t0 = time.perf_counter()
        Ms_b, tm_b = AG.aggregate(ck, km, U, XOt, yt, C.entries(ag.d_R, ag.d_O, cal['joint']), stages, 'A',
                                  cal['Omega_V'], cal['Omega_u'], nbar)
        agg_b = time.perf_counter() - t0
        t0 = time.perf_counter()
        Ms_f, tm_f = AG.aggregate(ck, km, U + Ux, XOt, yt, ents, stages, 'A', cal_f['Omega_V'], cal_f['Omega_u'],
                                  nbar)
        agg_f = time.perf_counter() - t0
    finally:
        AG.v_vector = _v_orig
    from fractions import Fraction

    def omf_dd(c):
        return C.dd(Fraction(c['Omega_U']) * Fraction(c['Omega_V']) * Fraction(c['Omega_u']) / (qs[2] * qs[3]))

    rng = S.CSPRNG()

    def release(M, s, c):
        s2 = c['releases'][str(s)]['sigma_c2']
        Omf_dd = omf_dd(c)
        t0 = time.perf_counter()
        eta = S.residues(S.sample_vector(s2, ck.N, rng), q0t)
        c1p, h = ck.rerand_partial_decrypt(M, pk, sO, eta)
        t1 = time.perf_counter()
        hi, lo = ck.combine(c1p, h, sR)
        z = ck.decode_i128(hi, lo, Omf_dd)
        t2 = time.perf_counter()
        return t1 - t0, t2 - t1, len(c1p) + len(h) if isinstance(c1p, (bytes, bytearray)) else c1p.nbytes + h.nbytes

    rel = {}
    for tag, Ms, c in (('v31', Ms_b, cal), ('full', Ms_f, cal_f)):
        rel[tag] = [release(Ms[s], s, c) for s in stages]
    # correctness of the benchmark path: noiseless decryption of stage 2, the 14 A entries against the plaintext
    zero = np.zeros((2, ck.N), dtype=np.uint64)
    c1p, h = ck.rerand_partial_decrypt(Ms_f[2], pk, sO, zero)
    hi, lo = ck.combine(c1p, h, sR)
    z = ck.decode_i128(hi, lo, omf_dd(cal_f))
    A_true = vecs[1:] @ (vecs[1:] * vecs[0]).T          # sum_t b xdot xdot^T over the matched rows
    err = max(abs(float(z[e['slot']].real) - A_true[int(e['name'][2]), int(e['name'][3])])
              for e in ents if e['kind'] == 'A')
    ref = {e['name']: float(z[e['slot']].real) for e in ents if e['kind'] == 'B'}
    errB = max(abs(ref[f'B_{j}{k}'] - float(XOt[:, j] @ (XOt[:, k] * vecs[0]))) for j in range(6) for k in range(j, 6))
    return dict(m=m, entries_v31=len(ents) - len(prods) - ag.d_R, entries_full=len(ents),
                upload_v31=up_base, upload_extra=up_extra, aggregate_v31=agg_b, aggregate_full=agg_f,
                aggregate_detail_v31=tm_b, aggregate_detail_full=tm_f,
                release_v31=[dict(O=a, R=b_, bytes=c) for a, b_, c in rel['v31']],
                release_full=[dict(O=a, R=b_, bytes=c) for a, b_, c in rel['full']],
                noiseless_max_abs_err_A=err, noiseless_max_abs_err_B=errB,
                calibration=dict(mode=calib, **{tag: dict(
                    log2_Omega_f=c['log2_Omega_f'], Omega_V=c['Omega_V'], kappa=c['kappa'], joint=c['joint'],
                    entries=c['entries'], Delta=c['Delta'], Delta_G=c['Delta_G'],
                    sigma_c2={s: r['sigma_c2'] for s, r in c['releases'].items()},
                    log2_B_tot={s: float(np.log2(r['B_tot'])) for s, r in c['releases'].items()},
                    sigma_slot={s: r['sigma_slot'] for s, r in c['releases'].items()}) for tag, c in
                    (('v31', cal), ('full', cal_f))}))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--sizes', type=int, nargs='+', default=[16384, 65536])
    ap.add_argument('--calib', choices=('fullrecord', 'v31'), default='fullrecord')
    ap.add_argument('--out', default=os.path.join(ROOT, 'reports', 'eng', 'fullrecord.json'))
    a = ap.parse_args()
    wd = os.path.join(ROOT, 'experiments', '_work', 'eng')
    os.makedirs(wd, exist_ok=True)
    model = {r['n_bar_O']: r for r in json.load(open(os.path.join(ROOT, 'reports', 'v31_more', 'cost_fullrecord.json')))['rows']}
    res = dict(stamp=time.strftime('%Y-%m-%d %H:%M'), load_before=os.getloadavg(), n_R_max=NR_MAX, threads=1, rows=[])
    for nbar in a.sizes:
        base, (dR_, dO_, b, ag) = baseline_run(nbar, wd)
        bp = bench_path(nbar, dR_, dO_, b, ag, a.calib)
        d_agg = bp['aggregate_full'] - bp['aggregate_v31']
        d_up = bp['upload_extra']['encrypt_R'] + bp['upload_extra']['convert_O']
        d_rel = sum(r['O'] + r['R'] for r in bp['release_full']) - sum(r['O'] + r['R'] for r in bp['release_v31'])
        d_O = d_agg + bp['upload_extra']['convert_O'] + sum(r['O'] for r in bp['release_full']) - sum(r['O'] for r in bp['release_v31'])
        cpu_O_base = sum(v for k, v in base['O'].items() if isinstance(v, (int, float)) and k not in ('release_deadline',))
        cpu_O_base += sum(r['sample'] + r['rerand_pdec'] for r in base['O'].get('release', {}).values())
        row = dict(n_bar_O=nbar, baseline=base, bench=bp,
                   extra=dict(aggregate=d_agg, upload_encrypt_R=bp['upload_extra']['encrypt_R'],
                              upload_convert_O=bp['upload_extra']['convert_O'], releases=d_rel,
                              upload_bytes=bp['upload_extra']['bytes']),
                   measured=dict(agg_ratio=bp['aggregate_full'] / bp['aggregate_v31'],
                                 time_ratio=(base['wall'] + d_agg + d_up + d_rel) / base['wall'],
                                 O_time_ratio=(cpu_O_base + d_O) / cpu_O_base,
                                 comm_ratio=(base['comm_bytes'] + bp['upload_extra']['bytes']) / base['comm_bytes'],
                                 extra_upload_MB=bp['upload_extra']['bytes'] / MB),
                   model=dict(agg_ratio=model[nbar]['ratio']['agg'], time_ratio=model[nbar]['ratio']['wall_single_thread_est'],
                              O_cpu_ratio=model[nbar]['ratio']['cpu_O'], comm_ratio=model[nbar]['ratio']['comm'],
                              extra_upload_MB=model[nbar]['full_record']['extra_comm_MB']) if nbar in model else None)
        print(json.dumps(dict(n_bar_O=nbar, measured=row['measured'], model=row['model'],
                              errA=bp['noiseless_max_abs_err_A'], wall=base['wall'])), flush=True)
        res['rows'].append(row)
    res.update(calib=a.calib, load_after=os.getloadavg())
    with open(a.out, 'w') as fh:
        json.dump(res, fh, indent=1, default=float)
