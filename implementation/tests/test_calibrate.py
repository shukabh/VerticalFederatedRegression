"""Calibrate (paper, Algorithm 7 and Sections 7.3-7.4): the derived flooding parameter (kappa='auto'), the
security bound on the modulus chain, refusals, and the ledger's composition rules."""
import math

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


def test_kappa_with_ledger_cap(ctx):
    c = _cal(ctx, eps_run=1.0, ledger_eps=10.0, ledger_delta=1e-6)
    ki = c['kappa_info']
    assert ki['kappa_run'] == 37 and ki['kappa_L'] == 46 and c['kappa'] == 46
    assert c['gamma'] <= c['rho_run'] / ki['rho_L'] * ki['Gamma_L']


def test_ledger_filter_exhausted_by_rho_not_slack(ctx, tmp_path):
    c = _cal(ctx, eps_run=1.0, ledger_eps=10.0, ledger_delta=1e-6)
    L = dp.Ledger(str(tmp_path / 'l.json'), eps_cap=10.0, delta_cap=1e-6, zeta=0.1, mode='filter')
    runs = 0
    while L.check_run(c['rho_run'], c['gamma'], 'R', 'p')[0]:
        for s, r in c['releases'].items():
            L.charge(s, r['rho'], c['gamma'] / c['n_rel'])
        runs += 1
    assert runs == math.floor(L.d['rho_cap'] / c['rho_run'] * (1 + 1e-12))
    assert L.d['gamma_spent'] + c['gamma'] <= L.d['gamma_cap']      # refused for rho, not for slack
    # reopening with a different cap is refused
    with pytest.raises(RuntimeError, match='cap differs'):
        dp.Ledger(str(tmp_path / 'l.json'), eps_cap=5.0, delta_cap=1e-6, zeta=0.1, mode='filter')


def test_ledger_slack_converted_at_the_cap_eps(ctx, tmp_path):
    """With a fixed kappa = 40 the slack, converted with 1 + e^eps_cap, binds after two runs."""
    c = _cal(ctx, eps_run=1.0, kappa=40)
    L = dp.Ledger(str(tmp_path / 'l.json'), eps_cap=10.0, delta_cap=1e-6, zeta=0.1, mode='filter')
    runs = 0
    while L.check_run(c['rho_run'], c['gamma'], 'R', 'p')[0]:
        L.charge(1, c['rho_run'], c['gamma'])
        runs += 1
    assert runs == 2 and L.d['rho_spent'] + c['rho_run'] <= L.d['rho_cap']


def test_single_run_ledger_is_filled_exactly(ctx, tmp_path):
    c = _cal(ctx, eps_run=4.0)
    eps_L, delta_L, zeta_L, mode = dp.ledger_cap(_ag(eps_run=4.0))
    assert (eps_L, delta_L, mode) == (4.0, 1e-9, 'fixed')
    L = dp.Ledger(str(tmp_path / 'l.json'), eps_cap=eps_L, delta_cap=delta_L, zeta=zeta_L, mode=mode)
    assert L.check_run(c['rho_run'], c['gamma'], 'R', 'p')[0]
    L.charge(1, c['rho_run'], c['gamma'])
    assert not L.check_run(1e-6, 0.0, 'R', 'p')[0]
