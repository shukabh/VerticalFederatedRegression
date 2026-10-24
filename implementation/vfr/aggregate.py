"""Phase 3 at O: Aggregate and pack (spec Algorithm 4), and the release primitive of Algorithm 5.

Circuit 'A' (as written in the spec):
    T_e = EvalSum(Rescale(sum_h U_e[h] * Ecd_{Omega_V}(V_e[h])));   P_e = Rescale(T_e * Ecd_{Omega_u}(u_slot(e)))
    M_s = sum_{e in E_s} P_e
Circuit 'C' (proof review's restructured circuit):
    T_e = Rescale_stoch(EvalSum(sum_h U_e[h] * Ecd(V_e[h])));   M_s = Rescale_stoch(sum_{e in E_s} T_e * Ecd(u_slot(e)))
Scales: R encodes at Omega_U (2^68 by default for circuit A, q2 for circuit C), masks at Omega_u = q1, O's data at
Omega_V, so Omega_f = Omega_U Omega_V Omega_u / (q1 q2), tracked exactly.
"""
from __future__ import annotations

import time

import numpy as np

from . import common as C
from . import sampler as S


def v_vector(spec, XO, yt):
    if spec[0] == 'xx':
        return XO[:, spec[1]] * XO[:, spec[2]]
    if spec[0] == 'x':
        return XO[:, spec[1]]
    if spec[0] == 'y':
        return yt
    if spec[0] == 'xy':
        return XO[:, spec[1]] * yt
    if spec[0] == 'yy':
        return yt * yt
    raise ValueError(spec)


def aggregate(ck, km, U, XO, yt, ents, stages, variant, Omega_V, Omega_u, n_bar, workers=1):
    """U[v][h]: converted ciphertexts of R's vectors (v = 0: b, 1 + j: xdot_j), chunk h.
    XO, yt: O's columns in tau-order (length n_bar).  Returns ({stage: packed ciphertext}, timing dict).
    workers > 1 processes entries in parallel threads (the native calls release the GIL; set
    OMP_NUM_THREADS=1 for the process to avoid oversubscription)."""
    Ns = ck.N // 2
    m = -(-n_bar // Ns)
    OmV, Omu = C.dd(Omega_V), C.dd(Omega_u)
    masks = {}
    for e in ents:                                   # one-hot masks (shared, read only)
        u = np.zeros(Ns)
        u[e['slot']] = 1.0
        masks[e['slot']] = ck.encode_plain(u, Omu, ck.nQ - 1)

    def one_entry(e):
        tm = dict(encode=0.0, mul_add=0.0, evalsum=0.0, rescale_mask=0.0)
        V = v_vector(e['V'], XO, yt)
        acc = None
        for h in range(m):
            seg = np.zeros(Ns)
            chunk = V[h * Ns:(h + 1) * Ns]
            seg[:len(chunk)] = chunk
            ta = time.time()
            ptx = ck.encode_plain_v(seg.tolist(), OmV[0], OmV[1], ck.nQ)
            tb = time.time()
            prod = ck.mul_plain(U[e['U']][h], ptx)
            if acc is None:
                acc = prod
            else:
                ck.add_inplace(acc, prod)
            tc = time.time()
            tm['encode'] += tb - ta
            tm['mul_add'] += tc - tb
        ta = time.time()
        if variant == 'A':
            r = ck.rescale(acc, False)
            tb = time.time()
            Tsum = ck.eval_sum(r, km)
            tc = time.time()
            res = ck.rescale(ck.mul_plain(Tsum, masks[e['slot']]), False)
            tm['rescale_mask'] += (tb - ta) + (time.time() - tc)
            tm['evalsum'] += tc - tb
        elif variant == 'C':
            Tsum = ck.eval_sum(acc, km)
            tb = time.time()
            res = ck.rescale(Tsum, True)
            tm['evalsum'] += tb - ta
            tm['rescale_mask'] += time.time() - tb
        else:
            raise ValueError(variant)
        return e['name'], res, tm

    t0 = time.time()
    if workers > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(one_entry, ents))
    else:
        results = [one_entry(e) for e in ents]
    tm = dict(encode=0.0, mul_add=0.0, evalsum=0.0, rescale_mask=0.0)
    Tent = {}
    for name, res, t in results:
        Tent[name] = res
        for k in tm:
            tm[k] += t[k]
    tm['entries_wall'] = time.time() - t0
    Ms = {}
    ta = time.time()
    for s in stages:
        acc = None
        for e in C.release_entries(ents, s):
            term = Tent[e['name']] if variant == 'A' else ck.mul_plain(Tent[e['name']], masks[e['slot']])
            acc = term if acc is None else ck.add(acc, term)
        Ms[s] = acc if variant == 'A' else ck.rescale(acc, True)
    tm['pack'] = time.time() - ta
    tm['n_entries'] = len(ents)
    tm['m'] = m
    tm['workers'] = workers
    return Ms, tm


def release_share(ck, M, pk, sO, sigma_c2, q0_towers, rng, debug_no_dp=False):
    """Algorithm 5 at O: fresh (u, e0, e1) and exact eta inside the native call; returns (c1', h_O)."""
    if debug_no_dp:
        eta_res = np.zeros((len(q0_towers), ck.N), dtype=np.uint64)
    else:
        eta_res = S.residues(S.sample_vector(sigma_c2, ck.N, rng), q0_towers)
    return ck.rerand_partial_decrypt(M, pk, sO, eta_res)
