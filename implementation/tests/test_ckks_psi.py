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
    """Two felts per item (64-bit items), the full BFV path with SanitizeBFV: R recovers exactly the matches and
    their positions, and no felt-wise false match passes the label-consistency test."""
    from vfr import vfr_native as vn
    pp = psi.PSIParams()
    assert pp.felts == 2
    rng = S.CSPRNG()
    n_bar, n_R = 2000, 300
    O_ids = [f'o{i}' for i in range(1500)] + [f'dummy:{os.urandom(8).hex()}' for _ in range(n_bar - 1500)]
    R_ids = [f'o{i}' for i in range(0, 1500, 7)][:200] + [f'r{i}' for i in range(n_R - 200)]
    tau = list(range(1, n_bar + 1))
    np.random.default_rng(0).shuffle(tau)
    k = oprf.keygen()
    outsO = [oprf.evaluate(k, i.encode()) for i in O_ids]
    tables = psi.sender_tables(outsO, tau, pp, n_bar)
    assert tables['Pc'].shape[2] == pp.felts * pp.N_B
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
    nb, F = pp.n_bins, pp.felts
    batches = psi.receiver_batches(outsR, pp, np.random.default_rng(1))
    keep = bo.moduli()[:pp.reply_towers]
    fl = pp.flood_sigma2()
    found, stats = {}, {}
    for bt_ in batches:
        wins = psi.window_values(bt_['query'], W, pp.t)
        Sl = [[] for _ in range(alpha)]
        Ql = [[] for _ in range(alpha)]
        layers = psi.masked_layers(tables['Pc'], tables['Lc'], rng, pp.t)
        for f in range(F):
            pw = bo.compute_powers([bf.encrypt_sym(keys, w[f * nb:(f + 1) * nb]) for w in wins], D)
            for a, (Sc, Qc) in enumerate(layers):
                Sct, Qct = bo.eval_partition(pw, psi.felt_slice(Sc, f, nb), psi.felt_slice(Qc, f, nb))
                Sl[a].append(bf.decrypt(keys, bo.sanitize(Sct, pp.reply_towers, S.residues(S.sample_vector(fl, pp.N_B, rng), keep))))
                Ql[a].append(bf.decrypt(keys, bo.sanitize(Qct, pp.reply_towers, S.residues(S.sample_vector(fl, pp.N_B, rng), keep))))
        f_, amb, bad = psi.receiver_decode([np.concatenate(x) for x in Sl], [np.concatenate(x) for x in Ql],
                                           bt_['slot_item'], outsR, pp.t, n_bar, felts=F, n_bins=nb, stats=stats)
        assert not amb and not bad
        found.update(f_)
    expect = {ri: tau[O_ids.index(R_ids[ri])] for ri in range(len(R_ids)) if R_ids[ri] in O_ids}
    assert found == expect
    assert stats['felt_inconsistent'] == 0


def _plain_replies(tables, query, t, rng):
    """Masked replies evaluated in the clear (as R decrypts them, without BFV): S, Q per partition."""
    S_l, Q_l = [], []
    for Sc, Qc in psi.masked_layers(tables['Pc'], tables['Lc'], rng, t):
        acc_s = np.zeros(Sc.shape[1], dtype=object)
        acc_q = np.zeros(Sc.shape[1], dtype=object)
        w = np.array([int(v) for v in query], dtype=object)
        p = np.ones(Sc.shape[1], dtype=object)
        for d in range(Sc.shape[0]):
            acc_s = (acc_s + p * Sc[d].astype(object)) % t
            acc_q = (acc_q + p * Qc[d].astype(object)) % t
            p = (p * w) % t
        S_l.append(acc_s.astype(np.uint64))
        Q_l.append(acc_q.astype(np.uint64))
    return S_l, Q_l


def test_psi_two_felts_forced_near_collisions():
    """A match on one felt alone is not a match; nor are two felts matched by two different items of a partition
    (all membership values 0, label felts inconsistent).  An item whose two felts both equal the query's, with a
    consistent label, is a match.  O's tables are built by bin_items and psi_build_polys, the replies are evaluated
    in the clear."""
    from vfr import vfr_native as vn
    pp = psi.PSIParams()
    t, nb, F = pp.t, pp.n_bins, pp.felts
    n_bar = 600
    k = oprf.keygen()
    outsO = [oprf.evaluate(k, f'o{i}'.encode()) for i in range(n_bar)]
    tau = list(range(1, n_bar + 1))
    outsR = [oprf.evaluate(k, f'r{i}'.encode()) for i in range(4)]        # four non-members
    bt = psi.receiver_batches(outsR, pp, np.random.default_rng(2))[0]
    bi = psi.bin_items(outsO, tau, pp, n_bar)
    xs, ys, counts = bi['xs'], bi['ys'], bi['counts']
    by_item = {i: b for b, i in bt['slot_item'].items()}

    def plant(b, felts, labels):
        a = int(np.argmin(counts[:, b]))
        c = counts[a, b]
        assert c < bi['D']
        for f in range(F):
            xs[a, f * nb + b, c] = felts[f]
            ys[a, f * nb + b, c] = labels[f]
            counts[a, f * nb + b] += 1
        return a

    rng = np.random.default_rng(3)
    other = lambda v: (v + 1 + int(rng.integers(t - 3))) % (t - 1)   # a different felt value
    x = {i: psi.item_felts(outsR[i], t, F) for i in range(4)}
    key = {i: [psi.label_key(outsR[i], t, f) for f in range(F)] for i in range(4)}
    # item 0: one O item shares felt 0 only;  item 1: one O item shares felt 1 only
    plant(by_item[0], [x[0][0], other(x[0][1])], [(77 + key[0][0]) % t, (77 + key[0][1]) % t])
    plant(by_item[1], [other(x[1][0]), x[1][1]], [(78 + key[1][0]) % t, (78 + key[1][1]) % t])
    # item 2: felt 0 from one O item and felt 1 from another item of the same partition (cross-item)
    b2 = by_item[2]
    a2 = int(np.argmin(counts[:, b2]))
    for f_match, lab in ((0, 91), (1, 92)):
        fel = [x[2][f] if f == f_match else other(x[2][f]) for f in range(F)]
        c = counts[a2, b2]
        for f in range(F):
            xs[a2, f * nb + b2, c] = fel[f]
            ys[a2, f * nb + b2, c] = (lab + key[2][f]) % t
            counts[a2, f * nb + b2] += 1
    # item 3: a full 64-bit collision with a consistent label (control: decodes to the planted position 55)
    plant(by_item[3], x[3], [(55 + key[3][f]) % t for f in range(F)])
    Pc, Lc = vn.psi_build_polys(xs, ys, counts, t)
    S_l, Q_l = _plain_replies(dict(Pc=Pc, Lc=Lc), bt['query'], t, S.CSPRNG())
    # one felt alone: some membership value is 0, but not all
    b0, b1 = by_item[0], by_item[1]
    assert any(int(S_[b0]) == 0 for S_ in S_l) and any(int(S_[b1 + nb]) == 0 for S_ in S_l)
    stats = {}
    found, amb, bad = psi.receiver_decode(S_l, Q_l, bt['slot_item'], outsR, t, n_bar, felts=F, n_bins=nb, stats=stats)
    assert found == {3: 55} and not amb and not bad
    assert stats['felt_partial'] >= 2 and stats['felt_inconsistent'] == 1
    # with one felt per item the same replies would have matched items 0 and 2 (the old behaviour)
    found1, _, _ = psi.receiver_decode([S_[:nb] for S_ in S_l], [Q_[:nb] for Q_ in Q_l], bt['slot_item'], outsR, t,
                                       n_bar, felts=1, n_bins=nb)
    assert {0, 2, 3} <= set(found1)


def test_psi_rejects_repeated_position_two_felts():
    """The repeated-position check also applies with two felts: two identifiers decoding to one position."""
    t, n_bar, nb = psi.T_DEFAULT, 100, 16
    outs = [oprf.evaluate(oprf.keygen(), f'id{i}'.encode()) for i in range(3)]
    slot_item = {5: 0, 9: 1, 11: 2}
    pos = {0: 17, 1: 17, 2: 40}
    S_ = np.ones(2 * nb, dtype=np.uint64)
    Q_ = np.zeros(2 * nb, dtype=np.uint64)
    for b, i in slot_item.items():
        for f in range(2):
            S_[f * nb + b] = 0
            Q_[f * nb + b] = (pos[i] + psi.label_key(outs[i], t, f)) % t
    found, amb, bad = psi.receiver_decode([S_], [Q_], slot_item, outs, t, n_bar, felts=2, n_bins=nb)
    assert found == pos and not amb and not bad
    with pytest.raises(psi.PSIOutputRejected, match='not injective'):
        psi.check_injective(found)


def test_psi_parameter_check():
    """False-match, cuckoo and SanitizeBFV requirements: two felts meet delta_psi = 2^-40 at every evaluation size
    (1M rows, n_R^max up to 100,000); one felt does not, even at 4,096 rows; the exact bounds behave as stated."""
    from vfr import vfr_native as vn
    pp = psi.PSIParams()
    mod = vn.BFVCore(14, pp.t, pp.bfv_depth, True).moduli()
    for n_bar, n_R in ((4096, 600), (106496, 20000), (1007616, 20000), (1007616, 100000)):
        c = pp.check(n_bar, n_R, moduli=mod)
        assert c['ok'], c['reasons']
        fm = c['false_match']
        assert fm['p_silent'] < fm['p_pass'] < fm['p_felt'] and fm['p_item'] <= fm['p_felt']
        assert fm['p_pass'] <= pp.delta_psi and c['p_cuckoo'] <= pp.delta_psi
        assert c['sanitize']['flood_ok'] and c['sanitize']['switch_ok']
    # 1M rows, n_R^max = 20,000: the cross-item events alone exceed delta_psi (so the consistency test is needed)
    fm = pp.false_match(1007616, 20000)
    assert fm['p_felt'] > pp.delta_psi and fm['p_pass'] < 2.0 ** -60
    assert pp.tables(2000)[0] == 1 and pp.tables(20000)[:2] == (2, 10000) and pp.tables(100000)[0] == 10
    p1 = psi.PSIParams(felts=1)
    c1 = p1.check(4096, 600)
    assert not c1['ok'] and 'felt' in c1['reasons'][0]
    assert p1.false_match(1007616, 2000)['p_pass'] > 1e-4          # the old 32-bit items at 1M
    # a too short chain after the switch is refused (one 60-bit limb: q'/(2t) < t_eta sigma_psi)
    bad = psi.PSIParams(reply_towers=1).sanitize_check(mod)
    assert bad['flood_ok'] and not bad['switch_ok']


def test_psi_balanced_tables():
    pp = psi.PSIParams()
    outs = [os.urandom(64) for _ in range(20000)]
    bts = psi.receiver_batches(outs, pp, np.random.default_rng(0))
    assert [len(b['slot_item']) for b in bts] == [10000, 10000]
    assert sorted(i for b in bts for i in b['slot_item'].values()) == list(range(20000))


def test_psi_rejects_repeated_position():
    """R aborts when two identifiers decode to the same position (only a false match can cause it)."""
    t, n_bar = psi.T_DEFAULT, 100
    outs = [oprf.evaluate(oprf.keygen(), f'id{i}'.encode()) for i in range(3)]
    slot_item = {5: 0, 9: 1, 11: 2}
    pos = {0: 17, 1: 17, 2: 40}                          # items 0 and 1 both decode to position 17
    S_ = np.ones(32, dtype=np.uint64)
    Q_ = np.zeros(32, dtype=np.uint64)
    for b, i in slot_item.items():
        S_[b] = 0
        Q_[b] = (pos[i] + psi.label_key(outs[i], t)) % t
    found, amb, bad = psi.receiver_decode([S_], [Q_], slot_item, outs, t, n_bar)
    assert found == pos and not amb and not bad          # the per-item checks pass
    with pytest.raises(psi.PSIOutputRejected, match='not injective'):
        psi.check_injective(found)
    psi.check_injective({0: 17, 1: 18, 2: 40})            # distinct positions are accepted


def _valid_assignment(tab, cand):
    assert tab is not None and len(tab) == len(cand)
    assert sorted(tab.values()) == list(range(len(cand)))
    for b, i in tab.items():
        assert b in cand[i]


def test_cuckoo_fallback_finds_assignment():
    # The random walk is cut off (max_kicks = 1) on dense tables (54 items, 64 bins), so it fails; the
    # augmenting-path fallback then returns an assignment exactly when one exists (checked against scipy's
    # maximum bipartite matching), also with max_kicks = 0 (everything placed by augmenting paths).
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import maximum_bipartite_matching
    n_bins, h, n_exist, n_none = 64, 3, 0, 0
    for seed in range(20):
        rng = np.random.default_rng(seed)
        cand = [[int(b) for b in rng.integers(n_bins, size=h)] for _ in range(54)]
        rows = [i for i, c in enumerate(cand) for _ in c]
        M = csr_matrix((np.ones(len(rows)), (rows, [b for c in cand for b in c])), shape=(len(cand), n_bins))
        exists = bool((maximum_bipartite_matching(M, perm_type='column') >= 0).sum() == len(cand))
        for kicks in (1, 0):
            stats = {}
            tab = psi.cuckoo(cand, n_bins, h, np.random.default_rng(1), max_kicks=kicks, stats=stats)
            assert stats['fallback']
            if exists:
                _valid_assignment(tab, cand)
            else:
                assert tab is None
        n_exist += exists
        n_none += not exists
    assert n_exist > 0 and n_none > 0


def test_cuckoo_fallback_on_a_walk_that_cycles():
    # Items 0-2 compete for bins {0, 1} and {0, 1, 2}; the walk from item 2 only ever kicks inside {0, 1} when
    # its own candidates are {0, 1, 1}, but an assignment exists through item 1's third bin.
    cand = [[0, 1, 0], [1, 0, 2], [0, 1, 1]]
    stats = {}
    tab = psi.cuckoo(cand, 3, 3, np.random.default_rng(0), max_kicks=2, stats=stats)
    _valid_assignment(tab, cand)
    # no assignment exists: four items with candidates in three bins
    cand4 = [[0, 1, 2], [0, 1, 2], [0, 1, 2], [2, 1, 0]]
    assert psi.cuckoo(cand4, 3, 3, np.random.default_rng(0), max_kicks=50, stats=stats) is None
    assert stats['fallback']


def test_cuckoo_without_fallback_is_unchanged():
    # At the evaluation's load (2,000 items in 16,384 bins) the walk succeeds and the fallback does not run.
    rng = np.random.default_rng(3)
    cand = [list(rng.integers(2 ** 14, size=3)) for _ in range(2000)]
    stats = {}
    tab = psi.cuckoo(cand, 2 ** 14, 3, np.random.default_rng(5), stats=stats)
    assert not stats['fallback']
    _valid_assignment(tab, cand)
    assert tab == psi.cuckoo(cand, 2 ** 14, 3, np.random.default_rng(5))
