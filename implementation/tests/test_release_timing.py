"""Release timing (DEVIATIONS A24): the budgeted sampler and the padded deadline.

The noise of every release (and the flooding of SanitizeBFV) comes from the budgeted sampler, whose work does not
depend on the values drawn; its failure (probability at most Agreement.sampler_xi per release, charged in gamma)
refuses the release.  As defence in depth O also holds every release until a deadline fixed before the first
release, measured from the receipt of the request; an overrun release is refused, never sent late.
"""
import os
import time

import numpy as np
import pytest

from vfr import common as C
from vfr import data, dp, protocol
from vfr.party_o import _hold_until

SLACK = 0.25        # socket, framing and scheduling time on top of the deadline (two local processes)


def _toy_run(tmp_path, name, opts_O=None, **kw):
    dR_, dO_, b = data.toy()
    ag = dp.Agreement(n_R_max=4, n_bar_O=6, d_R=2, d_O=2, **b, **kw)
    return protocol.run(ag, C.CKKSParams(), dR_, dO_, {},
                        dict(ledger_path=str(tmp_path / f'{name}.json'), **(opts_O or {})), timeout=300)


def test_hold_until_masks_the_work_time():
    """The hold ends at the deadline whatever the work before it took (to a few ms)."""
    ends = []
    for work in (0.0, 0.02, 0.11, 0.19):
        t0 = time.perf_counter()
        time.sleep(work)
        _hold_until(t0 + 0.25)
        ends.append(time.perf_counter() - t0)
    assert min(ends) >= 0.25
    assert max(ends) - min(ends) < 0.01


def test_fixed_deadline(tmp_path):
    R, O = _toy_run(tmp_path, 'fixed', release_pad=1.5)
    assert 'refused' not in R
    TO = O['timings']
    assert TO['release_deadline'] == 1.5
    for s, rec in TO['release'].items():
        assert rec['deadline'] == 1.5 and not rec['overrun']
        assert rec['busy'] < 1.5 and rec['held'] > 0
        assert 1.5 <= rec['busy'] + rec['held'] < 1.5 + 0.01
    lat = R['timings']['release_latency_R']
    assert set(lat) == {'1', '2'}
    # R sends the first request right after its upload, so its latency also contains O's aggregation (which does
    # not depend on the noise); the timer starts when O takes the request up.  The second request finds O idle.
    assert lat['1'] >= 1.5
    assert 1.5 <= lat['2'] < 1.5 + SLACK
    # the deadline is announced in each release (public), and R checked it against the agreement
    rel = [r for r in O['transcript'] if r['type'] == 'release']
    assert len(rel) == 2


def test_auto_deadline_latency_is_the_deadline(tmp_path):
    R, O = _toy_run(tmp_path, 'auto')                    # release_pad='auto' is the default
    TO = O['timings']
    d = TO['release_deadline']
    assert d is not None and d > 0
    busy = [rec['busy'] for rec in TO['release'].values()]
    assert max(busy) < d                                 # margin 2 over the calibrated worst case
    lat = R['timings']['release_latency_R']
    assert lat['1'] >= d and d <= lat['2'] < d + SLACK
    # on O's side every release takes exactly the deadline (to the scheduling jitter), whatever its sampling took
    tot = [rec['busy'] + rec['held'] for rec in TO['release'].values()]
    assert all(d <= v < d + 0.01 for v in tot)


def test_overrun_is_refused(tmp_path):
    R, O = _toy_run(tmp_path, 'overrun', release_pad=1e-4)
    assert 'deadline overrun' in R['refused']
    rec = O['timings']['release']['1']
    assert rec['overrun'] and rec['busy'] > 1e-4
    types = [r['type'] for r in O['transcript']]
    assert 'release' not in types and types[-1] == 'refuse'
    # the run's cost was charged at admission and stays charged (as for any abort)
    assert (tmp_path / 'overrun.json').exists()


def test_no_padding_and_bad_values(tmp_path):
    R, O = _toy_run(tmp_path, 'off', release_pad=None)
    assert O['timings']['release_deadline'] is None
    assert 'busy' not in O['timings']['release']['1']
    for bad in (-1.0, 0, 'fast', float('inf')):
        with pytest.raises(ValueError):
            dp.check_release_pad(dp.Agreement(n_R_max=4, n_bar_O=6, d_R=2, d_O=2, B_R=1, B_O=1, B_y=1, release_pad=bad))
    with pytest.raises(ValueError):
        dp.check_release_pad(dp.Agreement(n_R_max=4, n_bar_O=6, d_R=2, d_O=2, B_R=1, B_O=1, B_y=1, release_pad_margin=0.5))
    R, O = _toy_run(tmp_path, 'neg', release_pad=-1.0)
    assert R['refused'].startswith('Setting')


def test_sampler_budget_is_public_and_charged(tmp_path):
    """Calibrate publishes each release's work budget (R checks it) and charges sampler_xi per release in gamma and
    in the ledger cost."""
    dR_, dO_, b = data.toy()
    ag = dp.Agreement(n_R_max=4, n_bar_O=6, d_R=2, d_O=2, **b)
    ag0 = dp.Agreement(n_R_max=4, n_bar_O=6, d_R=2, d_O=2, **b, sampler_xi=2.0 ** -100)
    ck = C.make_ckks(C.CKKSParams())
    qs = list(ck.qs)
    cal = dp.calibrate(ag, ck.N, qs[2], qs[3], qs[0] * qs[1], C.gadget_constants(ck))
    cal0 = dp.calibrate(ag0, ck.N, qs[2], qs[3], qs[0] * qs[1], C.gadget_constants(ck))
    n_rel = cal['n_rel']
    for s, rel in cal['releases'].items():
        sb = rel['sampler']
        assert sb['xi_bound'] <= ag.sampler_xi and sb['T'] > ck.N
        assert sb['T'] == dp.smp.budget(rel['sigma_c2'], ck.N, ag.sampler_xi)['T']
    assert cal['gamma'] - cal0['gamma'] == pytest.approx(n_rel * (ag.sampler_xi - ag0.sampler_xi), rel=1e-2)
    assert cal['gamma'] == dp._ceil_float(n_rel * dp.Fraction(dp.slack_per_release(cal['kappa'], ag.delta_B, ag.sampler_xi)))
    with pytest.raises(ValueError):
        dp.alpha_split(dp.Agreement(n_R_max=4, n_bar_O=6, d_R=2, d_O=2, **b, sampler_xi=0.0))
    with pytest.raises(dp.CalibrationRefused):          # too small for the 128-bit coins: Calibrate refuses
        dp.calibrate(dp.Agreement(n_R_max=4, n_bar_O=6, d_R=2, d_O=2, **b, sampler_xi=2.0 ** -200),
                     ck.N, qs[2], qs[3], qs[0] * qs[1], C.gadget_constants(ck))


def test_sampler_failure_refuses_the_release(tmp_path, debug_env):
    R, O = _toy_run(tmp_path, 'sfail', opts_O=dict(test_sampler_fail='release'))
    assert 'sampler budget exhausted' in R['refused']
    assert O['timings']['release']['1'] == dict(sampler_failed=True)
    types = [r['type'] for r in O['transcript']]
    assert 'release' not in types and types[-1] == 'refuse'


def test_sampler_failure_in_psi_aborts(tmp_path, debug_env):
    R, O = _toy_run(tmp_path, 'pfail', opts_O=dict(test_sampler_fail='psi'))
    assert 'sampler budget exhausted' in R['refused'] and 'sampler budget exhausted' in O['refused']
    types = [r['type'] for r in O['transcript']]
    assert 'psi_response' not in types and types[-1] == 'refuse'


def test_psi_check_charges_the_flooding_sampler():
    from vfr import psi
    pp = psi.PSIParams()
    chk = pp.check(1_000_000, 20_000)
    D, alpha, _ = pp.sizes(1_000_000)
    assert chk['sampler_calls'] == 2 * alpha * pp.felts * chk['tables']
    assert chk['p_sampler'] == chk['sampler_calls'] * pp.sampler_xi <= pp.delta_psi
    bad = psi.PSIParams(sampler_xi=2.0 ** -30).check(1_000_000, 20_000)
    assert not bad['ok'] and any('sampler' in r for r in bad['reasons'])
