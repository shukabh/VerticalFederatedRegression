"""Party R (researcher): holds the cohort (ids, X_R) and receives the output.

Phases: verify O's public parameters (recomputes Calibrate), threshold keygen share, Link (padded OPRF
queries, labeled PSI under its own BFV key, I and pi, exact local A), Upload (seeded symmetric encryption
under s_R of b and xdot in tau-order), Release reconstruction (v = h_O + c1' s_R, decode with the exact final
scale), Solve and infer (Algorithm 6).  R never sends a decryption share.
"""
from __future__ import annotations

import math
import os
import time
from fractions import Fraction

import numpy as np

from . import common as C
from . import dp, oprf, psi, solve
from .wire import Channel, Transcript


def run_R(sock, ag: dp.Agreement, cp: C.CKKSParams, data: dict, opts: dict):
    tr = Transcript('R', keep_raw=opts.get('keep_raw', False))
    ch = Channel(sock, tr)
    T = {}
    if 'threads' in opts:
        from . import vfr_native as vn
        vn.set_threads(int(opts['threads']))
    nprng = np.random.default_rng()                     # cuckoo-hashing kicks

    # ------------------------------------------------------------------ Phase 0: verify public parameters
    t0 = time.time()
    mtype, pub, _ = ch.recv()
    if mtype == 'refuse':
        return dict(refused=pub['reason'], transcript=tr.records)
    if mtype != 'public_params':
        raise RuntimeError(mtype)
    a2 = pub['agreement']
    for f in ('n_R_max', 'n_bar_O', 'd_R', 'd_O', 'B_R', 'B_O', 'B_y', 'eps_run', 'delta_run', 'f1', 'f_y', 'kappa',
              'zeta', 'kappa_min', 'delta_B', 't_eta', 'lambda_sec', 'ledger_eps', 'ledger_delta', 'ledger_zeta',
              'ledger_mode', 'variant', 'theta_star', 'kappa_Psi', 'alpha', 'alpha_L', 'alpha_U', 'r1_mode',
              'intervals'):
        if a2[f] != getattr(ag, f):
            raise RuntimeError(f'agreement mismatch on {f}')
    a_L, a_U, _ = dp.alpha_split(ag)                    # validates the statistical setting
    if ag.r1_mode == 'exact':                           # public: (d_A, d_O, alpha_L) only; cached on disk
        solve.r1_quantile_exact(ag.d_R + 1, ag.d_O, a_L)
    ck = C.make_ckks(cp)
    if [int(q) for q in ck.qs] != pub['qs'] or [int(p) for p in ck.ps] != pub['ps']:
        raise RuntimeError('CKKS moduli differ')
    qs = list(ck.qs)
    q0, q1, q2 = qs[0] * qs[1], qs[2], qs[3]
    cal = dp.calibrate(ag, ck.N, q1, q2, q0, C.gadget_constants(ck))       # data independent: recompute
    for s, rel in cal['releases'].items():
        if rel['sigma_c2'] != pub['calibration']['releases'][s]['sigma_c2']:
            raise RuntimeError('calibration mismatch')
    if cal['Omega_V'] != pub['Omega_V']:
        raise RuntimeError('scale mismatch')
    joint = cal['joint']
    # exact final scale Omega_f = Omega_U Omega_V Omega_u / (q1 q2)
    Omf = Fraction(cal['Omega_U']) * Fraction(pub['Omega_V']) * Fraction(cal['Omega_u']) / (q1 * q2)
    Omf_dd = C.dd(Omf)
    pp = psi.PSIParams(**{k: v for k, v in pub['psi'].items()})
    D, alpha, W = pub['psi_sizes']['D'], pub['psi_sizes']['alpha'], pub['psi_sizes']['W']
    crs = bytes.fromhex(pub['crs'])
    T['setup_verify'] = time.time() - t0

    t0 = time.time()
    sR = ck.sample_secret()
    p0R = ck.pk_share(sR, crs)
    bR = ck.rotkey_share(sR, crs)
    ch.send('keygen_R', blobs={'p0_R': p0R, 'rotkey_b_R': bR})
    ch.recv('keygen_O')
    T['keygen_R'] = time.time() - t0

    # ------------------------------------------------------------------ Phase 1: link
    t0 = time.time()
    ids = [str(i) for i in data['ids']]
    XR = np.asarray(data['X'], float).reshape(len(ids), ag.d_R)
    if len(ids) > ag.n_R_max:
        raise ValueError('cohort larger than n_R_max')
    padded = ids + ['rdummy:' + os.urandom(16).hex() for _ in range(ag.n_R_max - len(ids))]
    blinds, alphas = zip(*(oprf.blind(i.encode()) for i in padded))
    ch.send('oprf_request', blobs={'blinded': b''.join(alphas)})
    _, _, ob = ch.recv('oprf_response')
    ev = ob['evaluated']
    outs = [oprf.finalize(padded[i].encode(), blinds[i], ev[32 * i:32 * i + 32]) for i in range(len(padded))]
    T['psi_oprf'] = time.time() - t0

    t0 = time.time()
    from . import vfr_native as vn
    bf = vn.BFVCore(int(math.log2(pp.N_B)), pp.t, pp.bfv_depth, True)
    bkeys = bf.keygen()
    pkb, rkb = bf.public_bytes(bkeys)
    ch.send('psi_keys', blobs={'context': bf.context_bytes(), 'pk': pkb, 'rk': rkb})
    batches = psi.receiver_batches(outs, pp, nprng)
    found, amb, bad = {}, set(), set()
    T['psi_query'] = 0.0
    T['psi_decode'] = 0.0
    for bi, bt_ in enumerate(batches):
        ta = time.time()
        wins = psi.window_values(bt_['query'], W, pp.t)
        ch.send('psi_query', meta=dict(batch=bi), blobs={f'w{i}': bf.encrypt_sym(bkeys, w) for i, w in enumerate(wins)})
        T['psi_query'] += time.time() - ta
        _, _, rb = ch.recv('psi_response')
        ta = time.time()
        Sl = [bf.decrypt(bkeys, rb[f'S{a}']) for a in range(alpha)]
        Ql = [bf.decrypt(bkeys, rb[f'Q{a}']) for a in range(alpha)]
        f, am, bd = psi.receiver_decode(Sl, Ql, bt_['slot_item'], outs, pp.t, ag.n_bar_O)
        found.update(f)
        amb |= am
        bad |= bd
        T['psi_decode'] += time.time() - ta
    if amb or bad:
        raise RuntimeError(f'PSI produced ambiguous or invalid labels: {len(amb)} ambiguous, {len(bad)} invalid')
    I = sorted(i for i in found if i < len(ids))
    pi = {ids[i]: int(found[i]) for i in I}
    n = len(I)
    # xdot = (1, clip(x_R; sqrt(B_R^2 - 1)))
    lim = math.sqrt(ag.B_R ** 2 - 1)
    nr = np.linalg.norm(XR, axis=1)
    XRc = XR * np.minimum(1.0, lim / np.maximum(nr, 1e-300))[:, None]
    Xdot = np.hstack([np.ones((len(ids), 1)), XRc])
    A = Xdot[I].T @ Xdot[I]
    T['link_total_R'] = T['psi_oprf'] + T['psi_query'] + T['psi_decode']

    # ------------------------------------------------------------------ Phase 2: upload
    t0 = time.time()
    Ns = ck.N // 2
    m = -(-ag.n_bar_O // Ns)
    vecs = np.zeros((ag.d_R + 2, ag.n_bar_O))
    for i in I:
        tpos = found[i] - 1
        vecs[0, tpos] = 1.0
        vecs[1:, tpos] = Xdot[i]
    OmU = C.dd(cal['Omega_U'])
    blobs = {}
    for v in range(ag.d_R + 2):
        for h in range(m):
            seg = np.zeros(Ns)
            chunk = vecs[v, h * Ns:(h + 1) * Ns]
            seg[:len(chunk)] = chunk
            seed = os.urandom(32)
            blobs[f'c0_{v}_{h}'] = ck.encrypt_seeded(sR, seg, OmU, seed)
            blobs[f'seed_{v}_{h}'] = seed
    T['upload_encrypt'] = time.time() - t0
    ch.send('upload', meta=dict(m=m, nvec=ag.d_R + 2), blobs=blobs)

    # ------------------------------------------------------------------ Phase 4/5: releases, solve, infer
    ents = C.entries(ag.d_R, ag.d_O, joint)
    sig = {s: cal['releases'][s]['sigma_slot'] for s in cal['releases']}
    fp = {}

    def release(stage):
        ch.send('release_request', meta=dict(stage=str(stage)))
        _, rm, rb = ch.recv('release')
        ta = time.time()
        if opts.get('test_partial_share') and os.environ.get('VFR_DEBUG_NO_DP') == '1':
            fp.setdefault(str(stage), ck._test_partial_share(rb['c1p'], sR).tobytes()[:256])
        hi, lo = ck.combine(rb['c1p'], rb['h'], sR)
        z = ck.decode_i128(hi, lo, Omf_dd)
        T.setdefault('release_R', {})[str(stage)] = time.time() - ta
        return {e['name']: float(z[e['slot']].real) for e in C.release_entries(ents, stage)}, z

    dO, dA = ag.d_O, ag.d_R + 1

    def unpack(vals):
        B = np.zeros((dO, dO))
        for j in range(dO):
            for k in range(j, dO):
                B[j, k] = vals[f'B_{j}{k}']
        Cm = np.array([[vals[f'C_{j}{k}'] for k in range(dO)] for j in range(dA)])
        return B, Cm

    rel1, z1 = release(1)
    B1, C1 = unpack(rel1)
    G1 = solve.assemble(A, B1, C1)
    s1 = solve.stage1_rule(G1, dA, sig['1'], sig['2'], a_L, ag.theta_star, ag.kappa_Psi, r1_mode=ag.r1_mode)
    rel2, z2 = release(2)
    B2, C2 = unpack(rel2)
    cR = np.array([rel2[f'cR_{j}'] for j in range(dA)])
    cO = np.array([rel2[f'cO_{k}'] for k in range(dO)])
    if joint:
        yy2 = rel2['yy']
        sig_yy = sig['2']
        zy = None
    else:
        rely, zy = release('y')
        yy2 = rely['yy']
        sig_yy = sig['y']
    ch.send('done')
    G2 = solve.assemble(A, B2, C2)
    c2 = np.concatenate([cR, cO])
    t0 = time.time()
    res = None
    if n > G2.shape[0]:
        res = solve.stage2_infer(G2, c2, yy2, n, dA, s1, sig['2'], sig_yy, ag.B_y, ag.alpha, a_L, a_U,
                                 ag.B_beta_pub, np.random.default_rng(opts.get('solve_seed')), Bboot=opts.get('Bboot', 2000),
                                 truncate=True, antithetic=opts.get('antithetic', True), intervals=ag.intervals)
    T['solve'] = time.time() - t0
    out = dict(I=[ids[i] for i in I], pi=pi, n=n, A=A, rel1=rel1, rel2=rel2, yy=yy2, G1=G1, G2=G2, c2=c2,
               stage1={k: v for k, v in s1.items()}, result=res, sigma=sig, Omega_f=float(Omf),
               calibration=cal, timings=T, transcript=tr.records, summary=tr.summary(), joint=joint,
               decoded={'1': z1, '2': z2, 'y': zy})
    if opts.get('keep_raw'):
        out['raw'] = tr.raw
    if fp:
        out['partial_share_fingerprints'] = fp
    return out
