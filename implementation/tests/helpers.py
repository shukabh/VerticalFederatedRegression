"""In-process harness for statistical tests of the release mechanism.

Runs the same native/aggregate code as the parties, with both key shares in one process (test only), so that
many releases of one packed ciphertext can be drawn quickly.  Each release uses fresh (u, e0, e1, eta).
"""
from __future__ import annotations

import os
from fractions import Fraction

import numpy as np

from vfr import aggregate as AG
from vfr import common as C
from vfr import dp
from vfr import sampler as S


class Pipeline:
    def __init__(self, ag: dp.Agreement, cp: C.CKKSParams, vecs: np.ndarray, XO: np.ndarray, yt: np.ndarray):
        """vecs: R's (d_R + 2) x n_bar vectors in tau order (b, xdot_0..); XO, yt: O's columns in tau order."""
        self.ag, self.cp = ag, cp
        ck = self.ck = C.make_ckks(cp)
        qs = list(ck.qs)
        self.q0t = [qs[0], qs[1]]
        self.cal = dp.calibrate(ag, ck.N, qs[2], qs[3], qs[0] * qs[1], C.gadget_constants(ck))
        crs = os.urandom(32)
        self.sR, self.sO = ck.sample_secret(), ck.sample_secret()
        self.pk = ck.make_pk([ck.pk_share(self.sR, crs), ck.pk_share(self.sO, crs)], crs)
        self.km = ck.make_rotkeys([ck.rotkey_share(self.sR, crs), ck.rotkey_share(self.sO, crs)], crs)
        Ns = ck.N // 2
        m = -(-ag.n_bar_O // Ns)
        OmU = C.dd(self.cal['Omega_U'])
        U = []
        for v in range(vecs.shape[0]):
            row = []
            for h in range(m):
                seg = np.zeros(Ns)
                ch = vecs[v, h * Ns:(h + 1) * Ns]
                seg[:len(ch)] = ch
                seed = os.urandom(32)
                c0 = ck.encrypt_seeded(self.sR, seg, OmU, seed)
                row.append(ck.convert(c0, seed, self.sO, OmU))
            U.append(row)
        self.joint = self.cal['joint']
        self.ents = C.entries(ag.d_R, ag.d_O, self.joint)
        self.stages = [1, 2] if self.joint else [1, 2, 'y']
        self.Ms, _ = AG.aggregate(ck, self.km, U, XO, yt, self.ents, self.stages, ag.variant,
                                  self.cal['Omega_V'], self.cal['Omega_u'], ag.n_bar_O)
        Omf = Fraction(self.cal['Omega_U']) * Fraction(self.cal['Omega_V']) * Fraction(self.cal['Omega_u']) / (qs[2] * qs[3])
        self.Omf = Omf
        self.Omf_dd = C.dd(Omf)
        self.rng = S.CSPRNG()

    def release(self, stage, debug_no_dp=False):
        rel = self.cal['releases'][str(stage)]
        c1p, h = AG.release_share(self.ck, self.Ms[stage], self.pk, self.sO, rel['sigma_c2'], self.q0t, self.rng,
                                  debug_no_dp=debug_no_dp)
        hi, lo = self.ck.combine(c1p, h, self.sR)
        z = self.ck.decode_i128(hi, lo, self.Omf_dd)
        coeffs = [(int(a) << 64) + int(b) for a, b in zip(hi, lo)]
        return {e['name']: float(z[e['slot']].real) for e in C.release_entries(self.ents, stage)}, z, coeffs


def tau_vectors(data_R, data_O, tau, ag):
    """R's and O's vectors in tau order, with both parties' clipping (as the parties build them)."""
    Omap = {i: k for k, i in enumerate(data_O['ids'])}
    XR = np.asarray(data_R['X'], float)
    lim = np.sqrt(ag.B_R ** 2 - 1)
    nr = np.linalg.norm(XR, axis=1)
    XRc = XR * np.minimum(1.0, lim / np.maximum(nr, 1e-300))[:, None]
    XO = np.asarray(data_O['X'], float)
    no = np.linalg.norm(XO, axis=1)
    XOc = XO * np.minimum(1.0, ag.B_O / np.maximum(no, 1e-300))[:, None]
    yc = np.clip(np.asarray(data_O['y'], float), -ag.B_y, ag.B_y)
    vecs = np.zeros((ag.d_R + 2, ag.n_bar_O))
    XOt = np.zeros((ag.n_bar_O, ag.d_O))
    yt = np.zeros(ag.n_bar_O)
    for j, i in enumerate(data_O['ids']):
        p = tau[i] - 1
        XOt[p] = XOc[j]
        yt[p] = yc[j]
    for k, i in enumerate(data_R['ids']):
        if i in Omap:
            p = tau[i] - 1
            vecs[0, p] = 1.0
            vecs[1, p] = 1.0
            vecs[2:, p] = XRc[k]
    return vecs, XOt, yt


def true_entries(G, c, yy, d_R, d_O):
    dA = d_R + 1
    out = {}
    for j in range(d_O):
        for k in range(j, d_O):
            out[f'B_{j}{k}'] = G[dA + j, dA + k]
    for j in range(dA):
        for k in range(d_O):
            out[f'C_{j}{k}'] = G[j, dA + k]
    for j in range(dA):
        out[f'cR_{j}'] = c[j]
    for k in range(d_O):
        out[f'cO_{k}'] = c[dA + k]
    out['yy'] = yy
    return out
