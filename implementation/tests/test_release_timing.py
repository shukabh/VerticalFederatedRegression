"""Padded release timing (Agreement.release_pad; DEVIATIONS A24).

O holds every release until a deadline fixed before the first release, measured from the receipt of the request,
so that the latency R observes does not depend on the noise drawn by the (not constant-time) exact sampler.  An
overrun release is refused, never sent late.
"""
import os
import time

import numpy as np
import pytest

from vfr import common as C
from vfr import data, dp, protocol
from vfr.party_o import _hold_until

SLACK = 0.25        # socket, framing and scheduling time on top of the deadline (two local processes)


def _toy_run(tmp_path, name, **kw):
    dR_, dO_, b = data.toy()
    ag = dp.Agreement(n_R_max=4, n_bar_O=6, d_R=2, d_O=2, **b, **kw)
    return protocol.run(ag, C.CKKSParams(), dR_, dO_, {}, dict(ledger_path=str(tmp_path / f'{name}.json')), timeout=300)


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
