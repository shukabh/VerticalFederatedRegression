"""The stage-1 radius r_1: the exact (Monte Carlo) quantile and the union bound; the statistical setting of the
agreement and its threading through both parties."""
import math

import numpy as np
import pytest

from vfr import common as C
from vfr import data, dp, protocol, solve

P, DA = 11, 5


def lam_max(rng, R, p, dA, chunk=20000):
    out = []
    for s in range(0, R, chunk):
        out.append(np.linalg.eigvalsh(solve.masked_noise(rng, min(chunk, R - s), p, dA, 1.0))[:, -1])
    return np.concatenate(out)


@pytest.mark.parametrize('alpha_L', [0.0025, 0.25])
def test_exact_quantile_gives_the_one_sided_failure_rate(alpha_L):
    """P[lambda_max(G1^ - G) > r_1] <= alpha_L, and close to it (not a loose bound), on fresh noise draws
    independent of the Monte Carlo that produced the quantile."""
    q = solve.r1_quantile_exact(DA, P - DA, alpha_L)
    R = 400_000
    lm = lam_max(np.random.default_rng(777), R, P, DA)
    rate = float(np.mean(lm > q))
    sd = math.sqrt(alpha_L * (1 - alpha_L) / R)
    assert rate <= alpha_L + 3 * sd
    assert rate >= 0.8 * alpha_L
    # the same with a sigma_1 scaling, through the stage-1 rule's radius
    s1 = solve.stage1_rule(np.eye(P) * 1e6, DA, 3.0, 1.0, alpha_L, 2.5, 2.0, r1_mode='exact')
    assert s1['r1'] == pytest.approx(3.0 * q, rel=1e-12) and s1['r1_mode'] == 'exact'


def test_exact_quantile_upper_bound_is_conservative():
    """The order-statistic upper bound covers the true quantile with probability >= R1_CONF (small p for speed)."""
    p, dA, a = 3, 1, 0.05
    truth = np.quantile(lam_max(np.random.default_rng(5), 1_000_000, p, dA), 1 - a)
    reps, miss = 150, 0
    for k in range(reps):
        e = solve.r1_quantile_mc(dA, p - dA, a, draws=2000, seed=1000 + k)
        assert e['ucb'] >= e['point']
        miss += e['ucb'] < truth
    assert miss <= 6                       # expected 1.5 at conf 0.99


def test_union_bound_is_still_available():
    lmin_shift = 1e5
    G1 = np.eye(P) * lmin_shift
    for a in (0.0025, 0.25):
        s_def = solve.stage1_rule(G1, DA, 2.0, 1.0, a, 2.5, 2.0)
        s_uni = solve.stage1_rule(G1, DA, 2.0, 1.0, a, 2.5, 2.0, r1_mode='union')
        s_ex = solve.stage1_rule(G1, DA, 2.0, 1.0, a, 2.5, 2.0, r1_mode='exact')
        r_union = 2.0 * (2 * math.sqrt(P) + 2 * math.sqrt(math.log(1 / a)))
        assert s_def['r1'] == s_uni['r1'] == pytest.approx(r_union, rel=1e-14)
        assert s_def['r1_mode'] == 'union'
        assert s_ex['r1'] < s_uni['r1']
    # p = 11, alpha_L = 0.0025: about 7.4 sigma_1 instead of 11.53 sigma_1
    assert solve.r1_union_unit(P, 0.0025) == pytest.approx(11.5287, abs=1e-3)
    assert 7.3 < solve.r1_quantile_exact(DA, P - DA, 0.0025) < 7.6
    with pytest.raises(ValueError):
        solve.r1_unit(P, DA, 0.0025, 'bogus')


def test_agreement_settings():
    ag = dp.Agreement(n_R_max=10, n_bar_O=20, d_R=4, d_O=6, B_R=2.0, B_O=1.0, B_y=1.0)
    assert dp.alpha_split(ag) == pytest.approx((0.0025, 0.0025, 0.045))
    prev = dp.setting('prev')
    assert all(getattr(ag, k) == v for k, v in prev.items())          # the defaults are the previous setting
    no_ci = dp.Agreement(n_R_max=10, n_bar_O=20, d_R=4, d_O=6, B_R=2.0, B_O=1.0, B_y=1.0, alpha_L=0.5,
                         intervals=False, r1_mode='exact')
    assert dp.alpha_split(no_ci)[0] == 0.5
    with pytest.raises(ValueError):
        dp.alpha_split(dp.Agreement(n_R_max=10, n_bar_O=20, d_R=4, d_O=6, B_R=2.0, B_O=1.0, B_y=1.0, alpha_L=0.5))
    with pytest.raises(ValueError):
        dp.alpha_split(dp.Agreement(n_R_max=10, n_bar_O=20, d_R=4, d_O=6, B_R=2.0, B_O=1.0, B_y=1.0,
                                    r1_mode='bogus'))
    with pytest.raises(ValueError):
        dp.setting('bogus')
    for name in dp.SETTINGS:
        dp.alpha_split(dp.Agreement(n_R_max=10, n_bar_O=20, d_R=4, d_O=6, B_R=2.0, B_O=1.0, B_y=1.0,
                                    **dp.setting(name)))


def test_protocol_threads_the_setting(tmp_path):
    """Both parties run with the exact radius, a larger alpha_L and intervals off; R's radius is sigma_1 times the
    cached quantile and the result carries an estimate without intervals.  An ill-formed setting is refused."""
    dR_, dO_, truth, b = data.synthetic(4096, 3500, 500, 300, seed=21)
    ag = dp.Agreement(n_R_max=600, n_bar_O=4096, d_R=4, d_O=6, f1=0.35, alpha_L=0.25, r1_mode='exact',
                      intervals=False, **b)
    R, O = protocol.run(ag, C.CKKSParams(), dR_, dO_, dict(solve_seed=5), dict(ledger_path=str(tmp_path / 'l.json')))
    s1 = R['stage1']
    assert s1['r1_mode'] == 'exact'
    assert s1['r1'] == pytest.approx(R['sigma']['1'] * solve.r1_quantile_exact(5, 6, 0.25), rel=1e-12)
    res = R['result']
    assert res is not None
    if not res['refused']:
        assert res['intervals'] is False and 'half_theta' not in res and len(res['theta']) == 11
    bad = dp.Agreement(n_R_max=600, n_bar_O=4096, d_R=4, d_O=6, alpha_L=0.5, **b)       # intervals on: alpha_CI < 0
    R2, _ = protocol.run(bad, C.CKKSParams(), dR_, dO_, {}, dict(ledger_path=str(tmp_path / 'l2.json')), timeout=300)
    assert R2['refused'].startswith('Setting')
