"""Party O (organization): holds the register (ids, X_O, y).  Never decrypts anything.

Phases (spec Algorithms 1-5): Setup (scope padding, fresh shuffle tau, Calibrate, ledger admission, which charges
the run's full planned cost atomically, threshold keygen), Link (fresh OPRF key, PSI preprocessing and evaluation
with SanitizeBFV), Upload conversion, Aggregate and pack, Release (re-randomize, exact discrete Gaussian eta,
h_O = c0' + c1' s_O + eta, send (c1', h_O) only).
"""
from __future__ import annotations

import math
import os
import time
from dataclasses import asdict

import numpy as np

from . import aggregate as AG
from . import common as C
from . import dp, oprf, psi
from . import sampler as S
from .wire import Channel, Transcript

DEBUG_ENV = 'VFR_DEBUG_NO_DP'


def _clip_rows(X, B):
    n = np.linalg.norm(X, axis=1)
    return X * np.minimum(1.0, B / np.maximum(n, 1e-300))[:, None]


def _shuffle(n, rng: S.CSPRNG):
    """fresh uniformly random permutation of 1..n (Fisher-Yates with the CSPRNG)."""
    perm = list(range(1, n + 1))
    for i in range(n - 1, 0, -1):
        j = rng.randbelow(i + 1)
        perm[i], perm[j] = perm[j], perm[i]
    return perm


def _hold_until(t_end: float):
    """Sleep until perf_counter() >= t_end (the last few ms by a short spin, so the overshoot stays small)."""
    while True:
        rem = t_end - time.perf_counter()
        if rem <= 0:
            return
        time.sleep(rem - 0.002 if rem > 0.004 else 0)


def calibrate_release_time(ck, sigma_c2: int, M, pk, sO, q0_towers, reps: int = dp.RELEASE_PAD_REPS) -> dict:
    """Time one release on public inputs only: `reps` exact samplings of N values at sigma_c2 (with an independent
    generator; the values are discarded) plus their residues, and one re-randomized partial decryption of M with
    eta = 0 (its output is discarded, never sent).  Returns the worst sampling time and the decryption time."""
    rng = S.CSPRNG()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        S.residues(S.sample_vector(sigma_c2, ck.N, rng), q0_towers)
        ts.append(time.perf_counter() - t0)
    t0 = time.perf_counter()
    out = ck.rerand_partial_decrypt(M, pk, sO, np.zeros((2, ck.N), dtype=np.uint64))
    t_pdec = time.perf_counter() - t0
    del out
    return dict(sample_max=max(ts), samples=ts, rerand_pdec=t_pdec)


def release_deadline(ag: dp.Agreement, ck, cal, M, pk, sO, q0_towers, debug_no_dp=False):
    """The release deadline in seconds (Agreement.release_pad), or None without padding.  It is fixed before the
    first release and depends only on public parameters (N, the run's sigma_c^2, the margin) and the machine."""
    if ag.release_pad is None:
        return None
    if ag.release_pad != 'auto':
        return float(ag.release_pad)
    s2 = max(int(r['sigma_c2']) for r in cal['releases'].values())
    c = calibrate_release_time(ck, s2, M, pk, sO, q0_towers)
    return float(ag.release_pad_margin) * (c['sample_max'] + c['rerand_pdec'])


def run_O(sock, ag: dp.Agreement, cp: C.CKKSParams, data: dict, opts: dict):
    tr = Transcript('O', keep_raw=opts.get('keep_raw', False))
    ch = Channel(sock, tr)
    T = {}
    MK = []
    C.mark(MK, 'start')
    debug = {}
    rng = S.CSPRNG()
    debug_no_dp = bool(opts.get('debug_no_dp', False))
    if debug_no_dp and os.environ.get(DEBUG_ENV) != '1':
        raise RuntimeError(f'debug_no_dp requires {DEBUG_ENV}=1 (test-only path)')
    if 'threads' in opts:
        from . import vfr_native as vn
        vn.set_threads(int(opts['threads']))

    # ------------------------------------------------------------------ Phase 0: setup
    t0 = time.time()
    ids = [str(i) for i in data['ids']]
    X = _clip_rows(np.asarray(data['X'], float).reshape(len(ids), ag.d_O), ag.B_O)
    y = np.clip(np.asarray(data['y'], float), -ag.B_y, ag.B_y)
    if len(ids) > ag.n_bar_O:
        raise ValueError('scope larger than n_bar_O')
    n_dummy = ag.n_bar_O - len(ids)
    all_ids = ids + ['dummy:' + os.urandom(16).hex() for _ in range(n_dummy)]
    Xall = np.vstack([X, np.zeros((n_dummy, ag.d_O))])
    yall = np.concatenate([y, np.zeros(n_dummy)])
    if opts.get('tau_override'):                       # test hook (worked toy example only)
        tov = opts['tau_override']
        tau = [tov[i] if i in tov else None for i in all_ids]
        rest = sorted(set(range(1, ag.n_bar_O + 1)) - {v for v in tau if v is not None})
        tau = [v if v is not None else rest.pop(0) for v in tau]
    else:
        tau = _shuffle(ag.n_bar_O, rng)
    ck = C.make_ckks(cp)
    qs = list(ck.qs)
    q0, q1, q2 = qs[0] * qs[1], qs[2], qs[3]
    gadget = C.gadget_constants(ck)
    try:
        dp.alpha_split(ag)                              # the statistical setting must be well formed
    except ValueError as exc:
        ch.send('refuse', meta=dict(reason=f'Setting: {exc}'))
        return dict(refused=f'Setting: {exc}', transcript=tr.records)
    try:
        cal = dp.calibrate(ag, ck.N, q1, q2, q0, gadget)
    except dp.CalibrationRefused as exc:                # parameters not admissible: refuse the run
        ch.send('refuse', meta=dict(reason=f'Calibrate: {exc}'))
        return dict(refused=f'Calibrate: {exc}', transcript=tr.records)
    joint = cal['joint']
    pp = psi.PSIParams(**opts.get('psi', {}))
    from . import vfr_native as vn
    try:                                                # PSI failure probabilities and SanitizeBFV, before any charge
        local_moduli = vn.BFVCore(int(math.log2(pp.N_B)), pp.t, pp.bfv_depth, True).moduli()
        psi_check = pp.check(ag.n_bar_O, ag.n_R_max, moduli=local_moduli)
    except psi.PSIParamsRefused as exc:
        psi_check = dict(ok=False, reasons=[str(exc)])
    if not psi_check['ok']:
        why = 'PSI parameters: ' + '; '.join(psi_check['reasons'])
        ch.send('refuse', meta=dict(reason=why))
        return dict(refused=why, transcript=tr.records)
    led = cal['ledger']                                 # public cap of the base register
    ledger = dp.Ledger(opts.get('ledger_path') or os.path.join(opts.get('workdir', '.'), 'ledger_O.json'),
                       eps_cap=led['eps'], delta_cap=led['delta'], zeta=led['zeta'], mode=led['mode'],
                       xi=led['xi'], delta_prime=led['delta_prime'], quota=opts.get('quota', 10 ** 6))
    # admission charges the run's full planned cost (every release) atomically; nothing is refunded on an abort
    ok, why = ledger.admit_run(cal['ledger_cost'], ag.researcher, ag.project,
                               meta=dict(kappa=cal['kappa'], n_rel=cal['n_rel'], eps_run=ag.eps_run,
                                         delta_run=ag.delta_run))
    if not ok:
        ch.send('refuse', meta=dict(reason=why))
        return dict(refused=why, transcript=tr.records)
    D, alpha, Bload = pp.sizes(ag.n_bar_O)
    W = pp.windows(D)
    n_tables = psi_check['tables']
    crs = os.urandom(32)
    Omega_V = cal['Omega_V']
    pub = dict(agreement=asdict(ag), ckks=cp.as_dict(), qs=[int(q) for q in qs], ps=[int(p) for p in ck.ps],
               calibration=cal, psi=asdict(pp), psi_sizes=dict(D=D, alpha=alpha, W=W, load_bound=Bload, tables=n_tables, felts=pp.felts),
               crs=crs.hex(), Omega_V=Omega_V)
    T['setup_calibrate'] = time.time() - t0
    C.mark(MK, 'setup')
    ch.send('public_params', meta=pub)

    # threshold key generation (shares with the common random string)
    t0 = time.time()
    _, _, kb = ch.recv('keygen_R')
    sO = ck.sample_secret()
    p0O = ck.pk_share(sO, crs)
    bO = ck.rotkey_share(sO, crs)
    pk = ck.make_pk([kb['p0_R'], p0O], crs)
    km = ck.make_rotkeys([kb['rotkey_b_R'], bO], crs)
    ch.send('keygen_O', blobs={'p0_O': p0O})
    T['keygen_O'] = time.time() - t0
    C.mark(MK, 'keygen')

    # ------------------------------------------------------------------ Phase 1: link
    t0 = time.time()
    k = oprf.keygen()                                   # fresh OPRF key for this run
    outs = oprf.evaluate_batch(k, [i.encode() for i in all_ids], int(opts.get('threads', 0)))   # parallel, native
    T['psi_oprf_self'] = time.time() - t0
    C.mark(MK, 'oprf_self')
    t0 = time.time()
    tables = psi.sender_tables(outs, tau, pp, ag.n_bar_O)
    T['psi_preprocess'] = time.time() - t0
    C.mark(MK, 'psi_preprocess')
    debug['psi'] = dict(D=tables['D'], alpha=tables['alpha'], max_load=tables['max_load'], n_dup=tables['n_dup'],
                        n_bins_redone=tables['n_bins_redone'], t_bin=tables['t_bin'], t_polys=tables['t_polys'],
                        felts=pp.felts, tables=n_tables, check=psi_check)

    t0 = time.time()
    _, _, ob = ch.recv('oprf_request')
    blinded = ob['blinded']
    if len(blinded) != 32 * ag.n_R_max:
        raise RuntimeError('OPRF request must contain exactly n_R_max elements')
    ev = oprf.blind_evaluate_batch(k, blinded, int(opts.get('threads', 0)))
    ch.send('oprf_response', blobs={'evaluated': ev})
    ledger.log_oprf_run(ag.researcher, ag.project)
    T['psi_oprf_requests'] = time.time() - t0
    C.mark(MK, 'oprf_requests')

    _, _, kb2 = ch.recv('psi_keys')
    bo = vn.BFVCore(kb2['context'])
    if bo.t != pp.t or bo.N != pp.N_B:
        raise RuntimeError('BFV context does not match the public PSI parameters')
    sc_R = pp.sanitize_check(bo.moduli())               # R's chain must also meet SanitizeBFV's conditions
    if not (sc_R['flood_ok'] and sc_R['switch_ok']):
        raise RuntimeError('BFV chain from R does not meet the SanitizeBFV conditions')
    bo.load_public(kb2['pk'], kb2['rk'])
    keep = bo.moduli()[:pp.reply_towers]
    flood2 = pp.flood_sigma2()
    T['psi_eval'] = 0.0
    T['psi_sanitize'] = 0.0
    nb, F = pp.n_bins, pp.felts
    for bi in range(n_tables):
        _, _, qb = ch.recv('psi_query')
        t0 = time.time()
        layers = psi.masked_layers(tables['Pc'], tables['Lc'], rng, pp.t)
        evals = {}
        for f in range(F):                              # felt f: its own ciphertexts, powers and coefficient slots
            pw = bo.compute_powers([qb[f'w{i}_{f}'] for i in range(W)], D)
            for a, (Sc, Qc) in enumerate(layers):
                evals[(a, f)] = bo.eval_partition(pw, psi.felt_slice(Sc, f, nb), psi.felt_slice(Qc, f, nb))
            del pw
        T['psi_eval'] += time.time() - t0
        t0 = time.time()
        rep = {}
        for (a, f), (Sct, Qct) in sorted(evals.items()):
            rep[f'S{a}_{f}'] = bo.sanitize(Sct, pp.reply_towers, S.residues(S.sample_vector(flood2, pp.N_B, rng), keep))
            rep[f'Q{a}_{f}'] = bo.sanitize(Qct, pp.reply_towers, S.residues(S.sample_vector(flood2, pp.N_B, rng), keep))
        T['psi_sanitize'] += time.time() - t0
        ch.send('psi_response', meta=dict(batch=bi), blobs=rep)

    C.mark(MK, 'psi_eval')
    # ------------------------------------------------------------------ Phase 2: upload conversion
    _, um, ub = ch.recv('upload')
    C.mark(MK, 'upload_recv')
    t0 = time.time()
    Ns = ck.N // 2
    m = -(-ag.n_bar_O // Ns)
    nvec = ag.d_R + 2
    if um['m'] != m or um['nvec'] != nvec:
        raise RuntimeError('upload shape mismatch')
    OmU = C.dd(cal['Omega_U'])
    U = [[ck.convert(ub[f'c0_{v}_{h}'], ub[f'seed_{v}_{h}'], sO, OmU) for h in range(m)] for v in range(nvec)]
    T['upload_convert'] = time.time() - t0
    C.mark(MK, 'upload_convert')

    # ------------------------------------------------------------------ Phase 3: aggregate and pack
    t0 = time.time()
    pos = np.asarray(tau) - 1
    XO = np.zeros((ag.n_bar_O, ag.d_O))
    yt = np.zeros(ag.n_bar_O)
    XO[pos] = Xall
    yt[pos] = yall
    ents = C.entries(ag.d_R, ag.d_O, joint)

    stages = [1, 2] if joint else [1, 2, 'y']
    Ms, tm = AG.aggregate(ck, km, U, XO, yt, ents, stages, ag.variant, Omega_V, cal['Omega_u'], ag.n_bar_O,
                          workers=int(opts.get('entry_workers', 1)))
    T['aggregate'] = time.time() - t0
    T['aggregate_detail'] = tm
    C.mark(MK, 'aggregate')
    debug['final_scale'] = {str(s): ck.ctxt_scale(Ms[s]) for s in stages}
    debug['final_towers'] = {str(s): ck.ctxt_towers(Ms[s]) for s in stages}

    # ------------------------------------------------------------------ Phase 4: releases
    q0_towers = [qs[0], qs[1]]
    released = set()
    T['release'] = {}
    t0 = time.time()
    deadline = release_deadline(ag, ck, cal, Ms[stages[0]], pk, sO, q0_towers, debug_no_dp)
    T['release_deadline'] = deadline
    T['release_calibrate'] = time.time() - t0
    while True:
        mtype, meta, _ = ch.recv()
        if mtype == 'done':
            break
        if mtype != 'release_request':
            raise RuntimeError(f'unexpected message {mtype}')
        s = C.stage_key(meta['stage'])
        if s in released or s not in Ms:
            raise RuntimeError('each release is made once')
        t0 = time.time()
        tq = time.perf_counter()                        # the deadline runs from the receipt of the request
        rel = cal['releases'][str(s)]
        if debug_no_dp:
            eta_res = np.zeros((2, ck.N), dtype=np.uint64)      # DEBUG ONLY (guarded above)
        else:
            eta = S.sample_vector(rel['sigma_c2'], ck.N, rng)  # exact N_Z(0, sigma_c^2), fresh per release
            eta_res = S.residues(eta, q0_towers)
            del eta
        t1 = time.time()
        if opts.get('test_record_c0p') and os.environ.get(DEBUG_ENV) == '1':
            c1p, h, c0p = ck._test_rerand_partial_decrypt_c0p(Ms[s], pk, sO, eta_res)
            debug.setdefault('c0p_fingerprints', {})[str(s)] = c0p.tobytes()[:256]
        else:
            c1p, h = ck.rerand_partial_decrypt(Ms[s], pk, sO, eta_res)
        del eta_res                                     # (the release was charged at admission)
        released.add(s)
        T['release'][str(s)] = dict(sample=t1 - t0, rerand_pdec=time.time() - t1)
        if deadline is not None:
            busy = time.perf_counter() - tq
            T['release'][str(s)].update(busy=busy, deadline=deadline, overrun=busy > deadline)
            if busy > deadline:                         # rule: an overrun release is refused, never sent late
                del c1p, h
                ch.send('refuse', meta=dict(reason='release deadline overrun', stage=str(s)))
                break
            _hold_until(tq + deadline)
            T['release'][str(s)]['held'] = time.perf_counter() - tq - busy
        ch.send('release', meta=dict(stage=str(s), deadline=deadline), blobs={'c1p': c1p, 'h': h})
    C.mark(MK, 'releases')
    out = dict(timings=T, transcript=tr.records, summary=tr.summary(), calibration=cal, debug=debug, marks=MK,
               tau={i: p for i, p in zip(all_ids, tau) if not i.startswith('dummy:')})
    if opts.get('keep_raw'):
        out['raw'] = tr.raw
    return out
