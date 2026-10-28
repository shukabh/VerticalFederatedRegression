"""The budgeted exact sampler (vfr/sampler.py, `budget`, `sample_residues`; DEVIATIONS A24).

The sampler runs a fixed number T of candidate trials with fixed-width limb arithmetic.  These tests check
(i) the limb arithmetic and every decision against an independent big-integer reference on the same coins,
including adversarial coins (large v, near-threshold values); (ii) the output law (exact pmf at small sigma, moments
and low bits at large sigma); (iii) the residues; (iv) the budget and its failure bound; (v) that a failure raises.
"""
import decimal
import math

import numpy as np
import pytest
from scipy.stats import binom

from vfr import sampler as S


def _ref_trials(bud, raw):
    """Big-integer reference of one call's trials on the given coins: (|y|, sign, accepted, undecided) lists."""
    Sg, b, C, Jv, km, Ly = bud['sigma2'], bud['b'], bud['C'], bud['J_v'], bud['k_max'], bud['Ly']
    r = S.RBITS
    E = S.EXP_TABLE
    T = raw.shape[1]

    def word(row, j, L=4):
        return sum(int(raw[row + i, j]) << (32 * i) for i in range(L))

    out = []
    for j in range(T):
        s = int(raw[0, j]) & 1
        u = word(1, j, Ly) & ((1 << b) - 1)
        RV, RA = word(1 + Ly, j), word(5 + Ly, j)
        RK = [word(9 + Ly + 4 * k, j) for k in range(km)]
        v = sum(1 for jj in range(1, Jv + 1) if RV < E[jj])
        bad = any(RV == E[jj] for jj in range(1, Jv + 1)) or v >= Jv
        absy = u + (v << b) + s
        G = ((absy * absy) << (r - 1)) // Sg - (v << r) + C
        assert G >= 0                                       # c >= S / (2 t^2) makes g >= 0
        n, F = G >> r, G & ((1 << r) - 1)
        if n == 0:
            acc = True
        else:
            th = E[min(n, S.J_ZERO)]
            acc = RA < th
            bad |= RA == th
        alive, lead = True, 0
        for k in range(1, km + 1):
            th = F // k
            alive = alive and RK[k - 1] < th
            lead += alive
            bad |= RK[k - 1] == th
        bad |= lead == km
        out.append((absy, s, acc and lead % 2 == 0, bad))
    return out


def _coins(bud, T, rng):
    nl = S.coin_limbs(bud)
    return np.frombuffer(rng.bytes(4 * nl * T), dtype='<u4').reshape(nl, T).astype(np.uint64)


def _set_word(raw, row, j, x):
    for i in range(4):
        raw[row + i, j] = (x >> (32 * i)) & 0xFFFFFFFF


@pytest.mark.parametrize('sigma2', [3, 1000003, (2 ** 55 + 12345) ** 2, 2 ** 122, int(2 ** 137) + 98765, 2 ** 180 + 7])
def test_trials_match_bigint_reference(sigma2):
    rng = S.CSPRNG()
    bud = S.budget(sigma2, 512)
    T = 1500
    raw = _coins(bud, T, rng)
    Ly = bud['Ly']
    # adversarial coins: large v (up to and beyond J_v), the smallest and largest u, coins equal to thresholds
    E = S.EXP_TABLE
    for j in range(0, 200):
        jj = 1 + (j % (bud['J_v'] + 1))
        _set_word(raw, 1 + Ly, j, max(0, E[min(jj, S.J_ZERO)] - 1))            # v = jj (fails at jj >= J_v)
    for j in range(200, 230):
        for i in range(Ly):
            raw[1 + i, j] = 0 if j % 2 else 0xFFFFFFFF
    for j in range(230, 240):
        _set_word(raw, 1 + Ly, j, E[1 + j % 5])                                # a tie in v
    Y, sgn, acc, bad = S._trials(bud, raw=raw)
    got = S.limbs_to_ints(Y, np.zeros(T, dtype=np.uint64))
    ref = _ref_trials(bud, raw)
    assert [r[0] for r in ref] == got
    assert [r[1] for r in ref] == [int(x) for x in sgn]
    assert [r[2] for r in ref] == acc.tolist()
    assert [r[3] for r in ref] == bad.tolist()
    assert bad[230:240].all() and not bad[300:].all()


def test_div_small_is_exact():
    """The reciprocal digit step is exact for every divisor and every partial value; the 128-bit division too."""
    for k in range(1, 64):
        x = np.arange(0, k << 16, dtype=np.uint64)
        M = np.uint64(-(-(1 << S._DIV_S) // k))
        assert ((x * M) >> np.uint64(S._DIV_S) == x // np.uint64(k)).all()
    rng = np.random.default_rng(1)
    vals = [int(v) for v in rng.integers(0, 2 ** 63, 500)] + [2 ** 128 - 1, 0, 1, 2 ** 127, 2 ** 96 + 5]
    vals += [(int(a) << 64) | int(b) for a, b in zip(rng.integers(0, 2 ** 63, 500), rng.integers(0, 2 ** 63, 500))]
    A = np.array([[(v >> (32 * i)) & 0xFFFFFFFF for v in vals] for i in range(4)], dtype=np.uint64)
    for k in (1, 2, 3, 7, 29, 63):
        q = S._div_small(A, k)
        assert [sum(int(q[i, j]) << (32 * i) for i in range(4)) for j in range(len(vals))] == [v // k for v in vals]


def test_exp_table_against_decimal():
    decimal.getcontext().prec = 120
    for j in range(1, S.J_ZERO + 1):
        x = (decimal.Decimal(2) ** 128) * decimal.Decimal(-j).exp()
        assert S.EXP_TABLE[j] == int(x.to_integral_value(rounding=decimal.ROUND_FLOOR))
    assert S.EXP_TABLE[S.J_ZERO] == 0 and S.EXP_TABLE[S.J_ZERO - 1] > 0


def test_small_sigma_pmf():
    rng = S.CSPRNG()
    s2, n = 3, 60000
    xs = np.array(S.sample_budgeted(s2, n, rng))
    ks = np.arange(-10, 11)
    p = np.exp(-ks ** 2 / (2 * s2))
    p /= np.exp(-np.arange(-60, 61) ** 2 / (2 * s2)).sum()
    emp = np.array([(xs == k).sum() for k in ks])
    chi2 = (((emp - n * p) ** 2) / (n * p)).sum()
    assert chi2 < 60          # 21 cells; P[chi2_21 > 60] ~ 1e-5
    assert abs(xs.var() - s2) < 0.1


def test_moderate_sigma_pmf_binned():
    """sigma^2 = 50 (t = 8): a chi-square over 15 bins covering the proposal's blocks and tails."""
    rng = S.CSPRNG()
    s2, n = 50, 80000
    xs = np.array(S.sample_budgeted(s2, n, rng))
    edges = np.array([-np.inf, -21, -15, -11, -8, -5, -3, -1, 1, 3, 5, 8, 11, 15, 21, np.inf])
    grid = np.arange(-200, 201)
    w = np.exp(-grid ** 2 / (2 * s2))
    w /= w.sum()
    p = np.array([w[(grid >= lo) & (grid < hi)].sum() for lo, hi in zip(edges[:-1], edges[1:])])
    emp = np.histogram(xs, bins=edges)[0]
    chi2 = (((emp - n * p) ** 2) / (n * p)).sum()
    assert chi2 < 50          # 15 bins; P[chi2_14 > 50] ~ 6e-6
    assert abs(xs.mean()) < 0.1


def test_large_sigma_moments_and_low_bits():
    rng = S.CSPRNG()
    sig2 = (2 ** 55 + 12345) ** 2
    xs = S.sample_budgeted(sig2, 16384, rng)
    a = np.array([float(x) for x in xs]) / math.sqrt(sig2)
    assert abs(a.mean()) < 0.05 and abs(a.std() - 1) < 0.03
    assert abs(np.mean(a ** 4) - 3) < 0.25
    low = np.array([x & 0xFF for x in xs])
    assert abs(low.mean() - 127.5) < 3 and len(set(low.tolist())) == 256


def test_residues_match_integers():
    rng = S.CSPRNG()
    qs = [(1 << 60) - 93, (1 << 59) + 12345677, 65537]
    for s2 in (7, 2 ** 122, int(2 ** 137) + 3):
        out = S.run_budgeted(s2, 2000, rng)
        assert not out['failed']
        vals = S.limbs_to_ints(out['Y'], out['sign'])
        neg = out['sign'] == 1
        for q in qs:
            rm = S._mod_q(out['Y'], q)
            res = np.where(neg & (rm != 0), np.uint64(q) - rm, rm)
            assert [int(x) for x in res] == [v % q for v in vals]
    r = S.sample_residues(2 ** 122, 1000, qs[:2], rng)
    assert r.shape == (2, 1000) and r.dtype == np.uint64


@pytest.mark.parametrize('sigma2,n', [(3, 100), (2 ** 84, 4096), (2 ** 122, 16384), (int(2 ** 137), 16384)])
def test_budget_bound(sigma2, n):
    bud = S.budget(sigma2, n)
    assert bud['xi_bound'] <= S.XI_DEFAULT
    # the Chernoff bound used for T is above the exact binomial tail
    assert binom.logcdf(n - 1, bud['T'], bud['p_lo']) <= math.log(S.XI_DEFAULT / 2)
    assert binom.logcdf(n - 1, bud['T'] - 200, bud['p_lo']) > binom.logcdf(n - 1, bud['T'], bud['p_lo'])
    # every other term of the bound
    T = bud['T']
    assert T * math.exp(-bud['J_v']) <= S.XI_DEFAULT / 8
    assert T / math.factorial(bud['k_max']) <= S.XI_DEFAULT / 8
    # p_lo is below the exact acceptance probability (computed by summing the target over Z)
    s2 = bud['sigma2']
    if s2 < 10 ** 6:
        grid = np.arange(-60 * int(math.isqrt(s2) + 1), 60 * int(math.isqrt(s2) + 1) + 1)
        tot = np.exp(-grid.astype(float) ** 2 / (2 * s2)).sum()
        p_exact = (1 - math.exp(-1)) * math.exp(-bud['C'] / 2.0 ** 128) * tot / 2 ** (bud['b'] + 1)
        assert bud['p_lo'] <= p_exact
    # a smaller xi needs more trials
    assert S.budget(sigma2, n, 2.0 ** -100)['T'] > T


def test_empirical_acceptance_matches_bound():
    rng = S.CSPRNG()
    s2 = int(2 ** 137)
    accs = [S.run_budgeted(s2, 16384, rng)['accepted'] for _ in range(3)]
    bud = S.budget(s2, 16384)
    for a in accs:
        assert a >= 16384
        # the acceptance probability is close to the lower bound (Poisson summation is tight at large sigma)
        assert abs(a / bud['T'] - bud['p_lo']) < 6 * math.sqrt(bud['p_lo'] / bud['T'])


def test_failure_raises(monkeypatch):
    real = S.budget

    def tight(sigma2, n, xi=S.XI_DEFAULT):
        b = real(sigma2, n, xi)
        b['T'] = n                                          # far too few trials: A < n
        return b
    monkeypatch.setattr(S, 'budget', tight)
    with pytest.raises(S.SamplerBudgetExhausted):
        S.sample_residues(2 ** 122, 4096, [(1 << 60) - 93])
    assert S.run_budgeted(2 ** 122, 4096)['failed']


def test_rejects_non_integer_variance():
    with pytest.raises(ValueError):
        S.budget(2.5, 10)
    with pytest.raises(ValueError):
        S.budget(0, 10)
