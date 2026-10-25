"""Unit tests: OPRF (RFC 9497 vectors), exact sampler, sensitivities and calibration, B_tot port."""
import json
import math
import os
import sys

import numpy as np
import pytest

from vfr import btot as bt
from vfr import dp, oprf
from vfr import sampler as S

HERE = os.path.dirname(os.path.abspath(__file__))


# ------------------------------------------------------------------------------------------- OPRF
def test_oprf_rfc9497_vectors():
    V = json.load(open(os.path.join(HERE, 'data', 'rfc9497_ristretto255_sha512_oprf.json')))[0]
    sk = oprf.derive_key_pair(bytes.fromhex(V['seed']), bytes.fromhex(V['keyInfo']))
    assert sk.hex() == V['skSm']
    for t in V['vectors']:
        x, r = bytes.fromhex(t['Input']), bytes.fromhex(t['Blind'])
        _, a = oprf.blind(x, r)
        assert a.hex() == t['BlindedElement']
        b = oprf.blind_evaluate(sk, a)
        assert b.hex() == t['EvaluationElement']
        assert oprf.finalize(x, r, b).hex() == t['Output']
        assert oprf.evaluate(sk, x).hex() == t['Output']


def test_oprf_blinding_consistency():
    k = oprf.keygen()
    for x in (b'104', b'221', b'some-identifier'):
        r, a = oprf.blind(x)
        assert oprf.finalize(x, r, oprf.blind_evaluate(k, a)) == oprf.evaluate(k, x)
    assert oprf.evaluate(k, b'1') != oprf.evaluate(oprf.keygen(), b'1')


# ------------------------------------------------------------------------------------------- sampler
def test_discrete_gaussian_small_sigma_pmf():
    rng = S.CSPRNG()
    s2 = 3
    n = 60000
    xs = np.array(S.sample_vector(s2, n, rng))
    ks = np.arange(-10, 11)
    p = np.exp(-ks ** 2 / (2 * s2))
    p /= np.exp(-np.arange(-60, 61) ** 2 / (2 * s2)).sum()
    emp = np.array([(xs == k).sum() for k in ks])
    chi2 = (((emp - n * p) ** 2) / (n * p)).sum()
    assert chi2 < 60          # 21 cells; P[chi2_21 > 60] ~ 1e-5
    assert abs(xs.var() - s2) < 0.1


def test_discrete_gaussian_large_sigma_moments_and_low_bits():
    rng = S.CSPRNG()
    sig2 = (2 ** 55 + 12345) ** 2
    xs = S.sample_vector(sig2, 16384, rng)
    a = np.array([float(x) for x in xs]) / math.sqrt(sig2)
    assert abs(a.mean()) < 0.05 and abs(a.std() - 1) < 0.03
    low = np.array([x & 0xFF for x in xs])     # the low bits are noised (no floating-point lattice)
    assert abs(low.mean() - 127.5) < 3 and len(set(low.tolist())) == 256


# ------------------------------------------------------------------------------------------- sensitivity
@pytest.mark.parametrize('BR,BO,By', [(5.59, 6.24, 2.89), (3 ** .5, 2 ** .5, 1.0), (1.0, 1.0, 1.0), (4.0, 1.0, 1.0)])
def test_delta_rep_closed_forms(BR, BO, By):
    D, Dn = dp.delta_rep(BR, BO, By)
    cf = dp.delta_rep_closed(BR, BO, By)
    assert cf is not None and abs(D - cf) / cf < 1e-10
    assert Dn <= D * (1 + 1e-12)


def test_delta_rep_regime3_and_free_yy():
    D, Dn = dp.delta_rep(1, 1, 5)            # paper Remark yyfree: 26.07 with y'y, 14.14 without
    assert abs(D - 26.07) < 0.01 and abs(Dn - 14.14) < 0.01
    sys.path.insert(0, os.path.normpath(os.path.join(HERE, '..', '..', 'experiments')))
    import sim_core as sc
    for (a, b, c) in [(1, 1, 5), (1, 2, 4), (0.5, 0.5, 3)]:
        assert dp.delta_rep(a, b, c)[0] >= sc.delta_rep(a, b, c) * (1 - 1e-8)
        assert abs(dp.delta_rep(a, b, c)[0] - sc.delta_rep(a, b, c)) / sc.delta_rep(a, b, c) < 1e-6
    assert abs(dp.delta_G(5.59, 6.24) - dp.delta_rep(5.59, 6.24, 0)[0]) < 1e-9


def test_delta_rep_is_an_upper_bound_by_random_search():
    rng = np.random.default_rng(0)
    BR, BO, By = 1.7, 1.2, 1.5
    D, Dn = dp.delta_rep(BR, BO, By)
    best = 0
    for _ in range(20000):
        xd = rng.normal(size=3); xd *= BR / np.linalg.norm(xd)
        x, xp = rng.normal(size=2), rng.normal(size=2)
        x *= BO * rng.uniform() ** 0.1 / np.linalg.norm(x); xp *= BO * rng.uniform() ** 0.1 / np.linalg.norm(xp)
        y, yp = rng.uniform(-By, By), rng.uniform(-By, By)
        dB = np.outer(x, x) - np.outer(xp, xp)
        v = np.concatenate([dB[np.triu_indices(2)], np.outer(xd, x - xp).ravel(), xd * (y - yp), x * y - xp * yp,
                            [y * y - yp * yp]])
        best = max(best, np.linalg.norm(v))
    assert best <= D


def test_delta_rep_regime3_against_brute_force():
    """Regime 3 (w - r > 2o): the exact enumeration over the roots of three cubics agrees with a direct maximization
    over the vectors (d_O = 2, ||xd|| = B_R; multistart L-BFGS-B) and is never below it."""
    from scipy.optimize import minimize

    def brute(BR, BO, By, starts=25, seed=0):
        rng = np.random.default_rng(seed)

        def neg(z):
            a, b, ra, rb, s, sp = z
            x = BO / (1 + np.exp(-np.clip(ra, -50, 50))) * np.array([np.cos(a), np.sin(a)])
            xp = BO / (1 + np.exp(-np.clip(rb, -50, 50))) * np.array([np.cos(b), np.sin(b)])
            y, yp = By * np.tanh(s), By * np.tanh(sp)
            dB = np.outer(x, x) - np.outer(xp, xp)
            return -(np.sum(dB[np.triu_indices(2)] ** 2) + BR ** 2 * (np.sum((x - xp) ** 2) + (y - yp) ** 2)
                     + np.sum((x * y - xp * yp) ** 2) + (y * y - yp * yp) ** 2)
        best = 0.0
        for _ in range(starts):
            z0 = rng.normal(size=6) * np.array([3, 3, 3, 3, 1, 1])
            r = minimize(neg, z0, method='L-BFGS-B', options=dict(ftol=1e-15, gtol=1e-12, maxiter=5000))
            best = max(best, -r.fun)
        return math.sqrt(best)

    for BR, BO, By in [(1.0, 1.0, 5.0), (0.5, 0.5, 3.0)]:
        assert By ** 2 - BR ** 2 > 2 * BO ** 2                     # regime 3
        D = dp.delta_rep(BR, BO, By)[0]
        bf = brute(BR, BO, By)
        assert bf <= D and bf >= D * (1 - 1e-8)


def test_rho_tight_matches_proofs():
    gamma = 3 * (2 ** -50 + 2 ** -40)
    rho = dp.rho_tight(1.0, 1e-9 - (1 + math.e) * gamma)
    assert abs(rho - 0.01496) < 5e-5


# ------------------------------------------------------------------------------------------- B_tot
def test_btot_reproduces_proof_table():
    """Example parameters of crypto/03b_error_bound.tex, Table btot: (A) 25.34 -> 61.5, (C') 17.56."""
    Om = 2 ** 52.5
    base = dict(N=2 ** 14, n_bar=10 ** 6, d_R=4, d_O=6, B_R=5.59, B_O=6.24, B_y=2.89, q1=Om, q2=Om, Omega_U=Om,
                Omega_V=Om, Omega_u=Om, Qtilde_max_A=1, Qtilde_max_C=1, P=1, k_digit_A=1, k_digit_C=1, dnum_A=2,
                dnum_C=2, k_P=1, varsigma_up=1, varsigma_down=1)
    cp = bt.CircuitParams(**base)
    assert abs(math.log2(bt.B_tot(cp, 2, True, 'A')) - 25.34) < 0.01
    cpc = bt.CircuitParams(**{**base, 'P': 2 ** 10, 'Omega_U': 2 ** 60, 'q2': 2 ** 60})
    assert abs(math.log2(bt.B_tot(cpc, 2, True, 'C')) - 17.56) < 0.01
    assert abs(math.log2(bt.delta_x(cp, 2, True, 86.18)) - 18.8) < 0.1


def test_calibrate_flooding_and_q0():
    from vfr import common as C
    ck = C.make_ckks(C.CKKSParams())
    qs = list(ck.qs)
    for variant in ('A', 'C'):
        ag = dp.Agreement(n_R_max=2000, n_bar_O=10 ** 6, d_R=4, d_O=6, B_R=5.59, B_O=6.24, B_y=2.89, variant=variant)
        cal = dp.calibrate(ag, ck.N, qs[2], qs[3], qs[0] * qs[1], C.gadget_constants(ck))
        assert cal['joint']
        for s, r in cal['releases'].items():
            assert r['sigma_c'] >= 2 ** 40 * r['B_tot'] / math.sqrt(2 * math.pi)
            assert r['rho_exact'] <= r['rho'] * (1 + 1e-12)
        assert abs(cal['releases']['2']['sigma_slot'] - 525) < 2
        assert cal['q0_bits'] > cal['q0_need_bits']
    # restructured circuit needs about 2^53.7 (proof review)
    assert 53.3 < cal['log2_Omega_f'] < 54.2
