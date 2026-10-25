"""Calibrate (paper, Algorithm 7 and Sections 7.3-7.4): the derived flooding parameter (kappa='auto'), the
security bound on the modulus chain, refusals, the floor on sigma_c, and the ledger's composition rules (hockey-stick
and statistical-distance accounting)."""
import math
from fractions import Fraction

import pytest

from vfr import common as C
from vfr import dp

B = dict(B_R=5.59, B_O=6.24, B_y=2.89)        # simulation population (evaluation setting)


@pytest.fixture(scope='module')
def ctx():
    ck = C.make_ckks(C.CKKSParams())
    qs = list(ck.qs)
    return ck.N, qs[0] * qs[1], qs[2], qs[3], C.gadget_constants(ck)


def _ag(**kw):
    base = dict(n_R_max=2000, n_bar_O=4096, d_R=4, d_O=6, delta_run=1e-9, **B)
    base.update(kw)
    return dp.Agreement(**base)


def _cal(ctx, gadget=None, q0=None, **kw):
    N, q0_, q1, q2, g = ctx
    return dp.calibrate(_ag(**kw), N, q1, q2, q0 or q0_, gadget or g)


@pytest.mark.parametrize('eps,kappa', [(0.5, 40), (1, 40), (2, 40), (4, 41), (8, 46)])
def test_kappa_auto_is_the_smallest_admissible(ctx, eps, kappa):
    c = _cal(ctx, eps_run=eps)
    assert c['kappa'] == kappa and c['n_rel'] == 2
    assert c['slack_fraction'] <= 0.1                      # (1 + e^eps) gamma <= zeta delta_run
    if kappa > 40:                                         # one less violates the rule unless the floor binds
        assert (1 + math.exp(eps)) * 2 * dp.slack_per_release(kappa - 1, 2.0 ** -50) > 0.1 * 1e-9
    for r in c['releases'].values():
        assert r['sigma_c'] >= 2.0 ** kappa * r['B_tot'] / math.sqrt(2 * math.pi)
    assert c['q0_bits'] > c['q0_need_bits']
    assert c['log2_QP'] <= c['log2_QP_max'] == 438


def test_eps8_feasible_at_one_million_rows(ctx):
    c = _cal(ctx, eps_run=8.0, n_bar_O=10 ** 6)
    assert c['kappa'] == 46
    assert 71 < c['log2_Omega_f'] < 73
    assert c['q0_bits'] > c['q0_need_bits'] and c['log2_QP'] <= 438


def test_explicit_kappa(ctx):
    with pytest.raises(dp.CalibrationRefused, match='delta_run too small'):
        _cal(ctx, eps_run=8.0, kappa=40)                   # slack exceeds delta_run
    c = _cal(ctx, eps_run=8.0, kappa=48)
    assert c['kappa'] == 48 and c['kappa_info']['rule'] == 'fixed'
    assert abs(c['log2_Omega_f'] - 73.62) < 0.1            # DEVIATIONS B10 data point


def test_refusals_security_bound_and_q0(ctx):
    N, q0, q1, q2, g = ctx
    with pytest.raises(dp.CalibrationRefused, match='exceeds the 128-bit bound'):
        _cal(ctx, gadget=dict(g, P=2.0 ** 210))            # log QP = 237 + 210 > 438
    with pytest.raises(dp.CalibrationRefused, match='too small'):
        _cal(ctx, q0=2 ** 80 + 1)
    with pytest.raises(dp.CalibrationRefused, match='lower delta_B'):
        _cal(ctx, eps_run=8.0, delta_run=1e-13)            # n_rel (1 + e^8) delta_B alone exceeds zeta delta_run


@pytest.mark.parametrize('eps,kappa_L,kappa', [(0.5, 22, 40), (1, 21, 40), (2, 19, 40), (4, 17, 41), (8, 15, 46)])
def test_kappa_with_hockey_stick_ledger(ctx, eps, kappa_L, kappa):
    """Hockey-stick ledger (default xi_L = 1e-3, delta' = 2^-100) at the cap (10, 1e-6), zeta_L = 0.1: kappa_L is the
    smallest kappa with n_rel eps'(kappa, delta') <= (rho~_run / rho_L) E'_L (composition proof, Proposition 5.5), and
    kappa is then the single-run kappa."""
    c = _cal(ctx, eps_run=eps, ledger_eps=10.0, ledger_delta=1e-6)
    ki = c['kappa_info']
    assert ki['ledger_rule'] == 'hs' and ki['kappa_L'] == kappa_L and c['kappa'] == kappa
    assert c['kappa'] == _cal(ctx, eps_run=eps)['kappa']                      # the ledger no longer raises kappa
    share = dp.rho_tight(eps, 0.9e-9) / ki['rho_L']
    assert ki['E_L'] <= 1e-3 * 10.0 / 2
    assert 2 * dp.eps_smudge(kappa_L, 2.0 ** -100) <= share * ki['E_L'] < 2 * dp.eps_smudge(kappa_L - 1, 2.0 ** -100)
    assert 2 * (2.0 ** -50 + 2.0 ** -100) <= share * ki['Gamma_L']           # the delta_B admissibility constraint
    cost = c['ledger_cost']
    assert cost['rho'] == c['rho_run'] and cost['eps_prime'] >= 2 * dp.eps_smudge(c['kappa'], 2.0 ** -100)
    assert cost['gamma'] >= 2 * (2.0 ** -50 + 2.0 ** -100)


def test_kappa_with_statistical_distance_ledger(ctx):
    """ledger_xi=None keeps the earlier rule: kappa_L is the smallest kappa with gamma <= (rho~_run / rho_L) Gamma_L."""
    c = _cal(ctx, eps_run=1.0, ledger_eps=10.0, ledger_delta=1e-6, ledger_xi=None)
    ki = c['kappa_info']
    assert ki['ledger_rule'] == 'sd' and ki['kappa_run'] == 37 and ki['kappa_L'] == 46 and c['kappa'] == 46
    assert c['gamma'] <= c['rho_run'] / ki['rho_L'] * ki['Gamma_L']
    assert c['ledger_cost'] == dict(rho=c['rho_run'], eps_prime=0.0, gamma=c['gamma'])


@pytest.mark.parametrize('xi', [1e-3, None])
def test_ledger_filter_exhausted_by_rho_not_slack(ctx, tmp_path, xi):
    c = _cal(ctx, eps_run=1.0, ledger_eps=10.0, ledger_delta=1e-6, ledger_xi=xi)
    led = c['ledger']
    L = dp.Ledger(str(tmp_path / 'l.json'), eps_cap=10.0, delta_cap=1e-6, zeta=0.1, mode='filter', xi=xi,
                  delta_prime=led['delta_prime'])
    runs = 0
    while L.admit_run(c['ledger_cost'], 'R', 'p')[0]:
        runs += 1
    cost, sp = c['ledger_cost'], L.spent()
    assert runs == math.floor(Fraction(L.d['rho_cap']) / Fraction(cost['rho']))      # exact, no tolerance
    assert sp['gamma'] + Fraction(cost['gamma']) <= Fraction(L.d['gamma_cap'])       # refused for rho, not for slack
    if xi is not None:
        assert sp['eps'] + Fraction(cost['eps_prime']) <= Fraction(L.d['E_cap'])     # nor for smudging
    # reopening with a different cap is refused
    with pytest.raises(RuntimeError, match='cap differs'):
        dp.Ledger(str(tmp_path / 'l.json'), eps_cap=5.0, delta_cap=1e-6, zeta=0.1, mode='filter', xi=xi)


def test_ledger_slack_converted_at_the_cap_eps(ctx, tmp_path):
    """With a fixed kappa = 40, the statistical-distance ledger converts the slack with 1 + e^eps_cap and refuses after
    two runs; the hockey-stick ledger is still exhausted by rho (102 runs)."""
    c = _cal(ctx, eps_run=1.0, kappa=40, ledger_eps=10.0, ledger_delta=1e-6, ledger_xi=None)
    L = dp.Ledger(str(tmp_path / 'l.json'), eps_cap=10.0, delta_cap=1e-6, zeta=0.1, mode='filter', xi=None)
    runs = 0
    while L.admit_run(c['ledger_cost'], 'R', 'p')[0]:
        runs += 1
    assert runs == 2 and L.spent()['rho'] + Fraction(c['rho_run']) <= Fraction(L.d['rho_cap'])
    c2 = _cal(ctx, eps_run=1.0, kappa=40, ledger_eps=10.0, ledger_delta=1e-6)
    L2 = dp.Ledger(str(tmp_path / 'l2.json'), eps_cap=10.0, delta_cap=1e-6, zeta=0.1, mode='filter', xi=1e-3)
    runs = 0
    while L2.admit_run(c2['ledger_cost'], 'R', 'p')[0]:
        runs += 1
    assert runs == math.floor(Fraction(L2.d['rho_cap']) / Fraction(c2["rho_run"])) == 102


def test_single_run_ledger_is_filled_exactly(ctx, tmp_path):
    c = _cal(ctx, eps_run=4.0)
    led = dp.ledger_cap(_ag(eps_run=4.0))
    assert (led['eps'], led['delta'], led['mode'], led['xi']) == (4.0, 1e-9, 'fixed', None)
    assert c['ledger'] == led
    L = dp.Ledger(str(tmp_path / 'l.json'), eps_cap=led['eps'], delta_cap=led['delta'], zeta=led['zeta'],
                  mode=led['mode'], xi=led['xi'])
    assert L.admit_run(c['ledger_cost'], 'R', 'p')[0]
    assert not L.admit_run(dict(rho=1e-6, gamma=0.0), 'R', 'p')[0]


def test_sigma_c_floor_is_2_to_the_42(ctx, monkeypatch):
    """The floor on sigma_c is 2^42 (the factor 1 + 2^-80 of the smudging lemma needs sigma_c >= 2.1 * 2^40).  With a
    tiny error bound the floor binds, and Calibrate raises the scale until every sigma_c reaches it."""
    assert dp.SIGMA_C_FLOOR == 2.0 ** 42
    c = _cal(ctx, eps_run=1.0)
    for r in c['releases'].values():
        assert r['flood_needed'] >= 2.0 ** 42 and r['sigma_c'] >= r['flood_needed']
    monkeypatch.setattr(dp.bt, 'B_tot', lambda *a, **k: 1.0)                 # 2^40 * 1 / sqrt(2 pi) < 2^42
    c = _cal(ctx, eps_run=8.0, delta_run=1e-6)                                # kappa = 40; small sigma_s
    assert c['kappa'] == 40 and c['log2_Omega_f'] > 40.5                      # the search started below
    assert all(r['flood_needed'] == 2.0 ** 42 for r in c['releases'].values())
    assert min(r['sigma_c'] for r in c['releases'].values()) >= 2.0 ** 42
    assert min(r['sigma_c'] for r in c['releases'].values()) < 2.0 ** (42 + 1 / 16)   # smallest scale on the grid


def test_flooding_test_is_conservative():
    # A case that a double-precision test passes but exact arithmetic with the margin refuses: sigma_c^2 equal to
    # the ceiling of the float value of (2^kappa B_tot / sqrt(2 pi))^2.
    kappa, Bt = 40, 2.0 ** 26.59
    need_float = 2.0 ** kappa * Bt / math.sqrt(2 * math.pi)
    s2 = math.ceil(Fraction(need_float) ** 2)
    assert math.sqrt(s2) >= need_float                     # the old test admits it
    assert not dp.floods(s2, Bt, kappa)                    # the conservative test refuses it
    assert Fraction(dp.flood_needed(Bt, kappa)) > Fraction(need_float) * (1 + Fraction(1, 2 ** 40))
    s2_ok = math.ceil(Fraction(need_float) ** 2 * (1 + Fraction(1, 2 ** 37)))
    assert dp.floods(s2_ok, Bt, kappa)
    assert not dp.floods(int(dp.SIGMA_C_FLOOR ** 2) - 1, 1.0, 0)   # the floor binds and is tested exactly
    assert dp.floods(int(dp.SIGMA_C_FLOOR ** 2), 1.0, 0)


def test_calibrate_floods_in_exact_arithmetic(ctx):
    # every release Calibrate returns satisfies sigma_c^2 >= (2^kappa B_tot (1 + 2^-40)/sqrt(2 pi))^2 exactly,
    # with sqrt(2 pi) bounded above by a rational
    sqrt2pi_up = Fraction(2506628274631001, 10 ** 15)      # > sqrt(2 pi) = 2.50662827463100050...
    assert sqrt2pi_up ** 2 > 2 * Fraction(math.pi)
    for eps in (1, 4, 8):
        c = _cal(ctx, eps_run=eps)
        for r in c['releases'].values():
            need = Fraction(2) ** c['kappa'] * Fraction(r['B_tot']) * (1 + Fraction(1, 2 ** 40)) / sqrt2pi_up
            assert Fraction(r['sigma_c2']) >= need ** 2
            assert Fraction(r['sigma_c2']) >= Fraction(dp.SIGMA_C_FLOOR) ** 2


@pytest.mark.parametrize('n_bar,n_R_max,log2_dmax', [(10 ** 6, 2000, 30.3), (10 ** 6, 10 ** 6, 38.6),
                                                     (4096, 2000, 29.7)])
def test_q0_diagnostic_uses_proved_offset(ctx, n_bar, n_R_max, log2_dmax):
    # crypto proofs, Proposition 4.5: d_max = 2^30.3 at n_bar_O = 10^6 with n_R^max = 2,000 and 2^38.6 if every
    # row may match (previous setting, eps = 1); it does not change the q0 decision: q0 = 2^119 against a need of
    # 2^81.0 (4k) and 2^89.0 (1M), and the sharper condition (Q) 2^74.1 and 2^81.9
    c = _cal(ctx, n_bar_O=n_bar, n_R_max=n_R_max, eps_run=1)
    assert c['n_max_offset'] == min(n_bar, n_R_max)
    assert abs(c['log2_d_max'] - log2_dmax) < 0.05
    assert abs(c['q0_need_Q_bits'] - (81.94 if n_bar == 10 ** 6 else 74.08)) < 0.01
    assert c['q0_need_Q_bits'] < c['q0_need_bits'] < c['q0_bits']
    for r in c['releases'].values():                       # d_max is far below the dominant term of (Q)
        assert math.log2(r['d_max']) < c['q0_need_Q_bits'] - 40
