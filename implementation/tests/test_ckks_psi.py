"""CKKS core (threshold keys, seeded upload, conversion, circuits A and C, release) and labeled PSI, in process."""
import math
import os

import numpy as np
import pytest

from helpers import Pipeline, tau_vectors, true_entries
from vfr import common as C
from vfr import data, dp, oprf, psi
from vfr import sampler as S


@pytest.fixture(scope='module')
def ck():
    return C.make_ckks(C.CKKSParams())


def _i128(hi, lo):
    return [(int(a) << 64) + int(b) for a, b in zip(hi, lo)]


def test_encoder_exact_scale_roundtrip(ck):
    rng = np.random.default_rng(1)
    z = rng.uniform(-40, 40, ck.N // 2)
    for sc in (2.0 ** 40, 2.0 ** 61.5, float(ck.qs[3])):
        c = ck.encode_coeffs(z, C.dd(sc))
        assert max(abs(x) for x in c) < 2 ** 120
        back = ck.decode_coeffs(c, C.dd(sc))
        assert np.abs(back.real - z).max() < 64 * 2 ** 7 / sc * ck.N + 1e-12


def test_seeded_upload_and_key_conversion(ck):
    crs = os.urandom(32)
    sR, sO = ck.sample_secret(), ck.sample_secret()
    z = (np.random.default_rng(2).uniform(size=ck.N // 2) < 0.3).astype(float)
    OmU = C.dd(ck.qs[3])
    seed = os.urandom(32)
    c0 = ck.encrypt_seeded(sR, z, OmU, seed)
    ct = ck.convert(c0, seed, sO, OmU)
    v = _i128(*ck._test_decrypt_both(ct, sR, sO, 2))
    assert np.abs(ck.decode_coeffs(v, OmU).real - z).max() < 1e-12
    # converting with the wrong share (or the wrong seed) does not give a ciphertext under s_R + s_O
    bad = ck.convert(c0, seed, ck.sample_secret(), OmU)
    vb = _i128(*ck._test_decrypt_both(bad, sR, sO, 2))
    assert np.abs(ck.decode_coeffs(vb, OmU).real - z).max() > 1e3
    bad2 = ck.convert(c0, os.urandom(32), sO, OmU)
    vb2 = _i128(*ck._test_decrypt_both(bad2, sR, sO, 2))
    assert np.abs(ck.decode_coeffs(vb2, OmU).real - z).max() > 1e3


def test_evalsum_with_joint_keys_is_the_trace(ck):
    crs = os.urandom(32)
    sR, sO = ck.sample_secret(), ck.sample_secret()
    km = ck.make_rotkeys([ck.rotkey_share(sR, crs), ck.rotkey_share(sO, crs)], crs)
    z = np.random.default_rng(3).uniform(-1, 1, ck.N // 2)
    Om = C.dd(2.0 ** 50)
    seed = os.urandom(32)
    ct = ck.convert(ck.encrypt_seeded(sR, z, Om, seed), seed, sO, Om)
    T = ck.eval_sum(ct, km)
    v = _i128(*ck._test_decrypt_both(T, sR, sO, 2))
    dec = ck.decode_coeffs(v, Om)
    assert np.abs(dec.real - z.sum()).max() < 1e-6 and np.abs(dec.imag).max() < 1e-6


def _small_problem(n_bar=4096, n=300, seed=3, variant='A', log2_Omega_U='auto'):
    dR_, dO_, truth, b = data.synthetic(n_bar, int(0.8 * n_bar), 500, n, seed=seed)
    ag = dp.Agreement(n_R_max=600, n_bar_O=n_bar, d_R=4, d_O=6, variant=variant, log2_Omega_U=log2_Omega_U, **b)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_bar)[:len(dO_['ids'])] + 1
    tau = {i: int(p) for i, p in zip(dO_['ids'], perm)}
    vecs, XO, yt = tau_vectors(dR_, dO_, tau, ag)
    ps = data.plaintext_stats(dR_, dO_, b['B_R'], b['B_O'], b['B_y'])
    return ag, vecs, XO, yt, ps


@pytest.mark.parametrize('variant,lu', [('A', 'auto'), ('A', None), ('C', 'auto')])
def test_circuit_noise_off_matches_plaintext(variant, lu):
    ag, vecs, XO, yt, ps = _small_problem(variant=variant, log2_Omega_U=lu)
    pl = Pipeline(ag, C.CKKSParams(), vecs, XO, yt)
    truth = true_entries(ps['G'], ps['c'], ps['yy'], ag.d_R, ag.d_O)
    for s in pl.stages:
        vals, z, _ = pl.release(s, debug_no_dp=True)
        err = max(abs(vals[k] - truth[k]) for k in vals)
        assert err < 1e-9 * max(1.0, max(abs(truth[k]) for k in vals)), (s, err)
        # slots not used by the release decode to ~0
        used = {e['slot'] for e in C.release_entries(pl.ents, s)}
        other = np.array([abs(z[k]) for k in range(64) if k not in used])
        assert other.max() < 1e-6
    # the exact final scale O tracked equals R's rational Omega_f
    assert abs(pl.ck.ctxt_scale(pl.Ms[2])[0] / float(pl.Omf) - 1) < 1e-15


def test_random_error_within_worst_case_bound():
    """Two runs with independent keys and randomness on the same data: the deterministic centre cancels in the
    difference, which isolates the randomness-dependent error e_tot (incl. the fresh re-randomization error).
    Its l2 norm must stay below the worst-case B_tot (the proofs report a margin of 6-8 bits)."""
    ag, vecs, XO, yt, ps = _small_problem(n_bar=4096, n=300, seed=4)
    p1 = Pipeline(ag, C.CKKSParams(), vecs, XO, yt)
    p2 = Pipeline(ag, C.CKKSParams(), vecs, XO, yt)
    for s in (1, 2):
        _, _, v1 = p1.release(s, debug_no_dp=True)
        _, _, v2 = p2.release(s, debug_no_dp=True)
        d = np.array([a - b for a, b in zip(v1, v2)], dtype=float) / math.sqrt(2)
        meas = np.linalg.norm(d)
        bound = p1.cal['releases'][str(s)]['B_tot']
        assert meas < bound, (s, math.log2(meas), math.log2(bound))
        assert math.log2(bound) - math.log2(meas) > 3


def test_rerandomization_is_fresh():
    ag, vecs, XO, yt, ps = _small_problem(n_bar=1024, n=100, seed=5)
    pl = Pipeline(ag, C.CKKSParams(), vecs, XO, yt)
    rel = pl.cal['releases']['1']
    from vfr import aggregate as AG
    c1a, ha = AG.release_share(pl.ck, pl.Ms[1], pl.pk, pl.sO, rel['sigma_c2'], pl.q0t, pl.rng, debug_no_dp=True)
    c1b, hb = AG.release_share(pl.ck, pl.Ms[1], pl.pk, pl.sO, rel['sigma_c2'], pl.q0t, pl.rng, debug_no_dp=True)
    assert not np.array_equal(c1a, c1b) and not np.array_equal(ha, hb)
    assert c1a.shape == (2, pl.ck.N) and ha.shape == (2, pl.ck.N)


def test_labeled_psi_exact_recovery():
    from vfr import vfr_native as vn
    pp = psi.PSIParams()
    rng = S.CSPRNG()
    n_bar, n_R = 2000, 300
    O_ids = [f'o{i}' for i in range(1500)] + [f'dummy:{os.urandom(8).hex()}' for _ in range(n_bar - 1500)]
    R_ids = [f'o{i}' for i in range(0, 1500, 7)][:200] + [f'r{i}' for i in range(n_R - 200)]
    tau = list(range(1, n_bar + 1))
    np.random.default_rng(0).shuffle(tau)
    k = oprf.keygen()
    outsO = [oprf.evaluate(k, i.encode()) for i in O_ids]
    tables = psi.sender_tables(outsO, tau, pp, n_bar)
    outsR = []
    for i in R_ids:
        r, a = oprf.blind(i.encode())
        outsR.append(oprf.finalize(i.encode(), r, oprf.blind_evaluate(k, a)))
    bf = vn.BFVCore(14, pp.t, pp.bfv_depth, True)
    keys = bf.keygen()
    pkb, rkb = bf.public_bytes(keys)
    bo = vn.BFVCore(bf.context_bytes())
    bo.load_public(pkb, rkb)
    D, alpha = tables['D'], tables['alpha']
    W = pp.windows(D)
    batches = psi.receiver_batches(outsR, pp, np.random.default_rng(1))
    keep = bo.moduli()[:pp.reply_towers]
    found = {}
    for bt_ in batches:
        wins = [bf.encrypt_sym(keys, w) for w in psi.window_values(bt_['query'], W, pp.t)]
        pw = bo.compute_powers(wins, D)
        Sl, Ql = [], []
        for Sc, Qc in psi.masked_layers(tables['Pc'], tables['Lc'], rng, pp.t):
            Sct, Qct = bo.eval_partition(pw, Sc, Qc)
            fl = pp.flood_sigma2()
            Sl.append(bf.decrypt(keys, bo.sanitize(Sct, pp.reply_towers, S.residues(S.sample_vector(fl, pp.N_B, rng), keep))))
            Ql.append(bf.decrypt(keys, bo.sanitize(Qct, pp.reply_towers, S.residues(S.sample_vector(fl, pp.N_B, rng), keep))))
        f, amb, bad = psi.receiver_decode(Sl, Ql, bt_['slot_item'], outsR, pp.t, n_bar)
        assert not amb and not bad
        found.update(f)
    expect = {ri: tau[O_ids.index(R_ids[ri])] for ri in range(len(R_ids)) if R_ids[ri] in O_ids}
    assert found == expect
