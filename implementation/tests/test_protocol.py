"""End-to-end runs with R and O as separate processes over a socket.

  * worked toy example: linkage and, with eta = 0 (guarded debug flag), the plaintext statistics of toy.json;
  * synthetic cohort (n_bar_O = 4096, n = 300, d_R = 4, d_O = 6): exact match set and alignment, noise-off
    decryption equals plaintext G and c, and the transcript checks (what crosses, and what never does);
  * noise on: the stage-2 estimate and intervals equal alg6.py fed the same decoded noisy statistics, and O's
    ledger charged the run's full cost once, at admission;
  * the debug flag cannot be used without the test-only environment variable.
"""
import json
import os
from fractions import Fraction

import numpy as np
import pytest

from vfr import common as C
from vfr import data, dp, protocol

HERE = os.path.dirname(os.path.abspath(__file__))
TOY_JSON = os.path.normpath(os.path.join(HERE, '..', '..', 'reports', 'toy_example', 'toy.json'))

R_TO_O = {'keygen_R', 'oprf_request', 'psi_keys', 'psi_query', 'upload', 'release_request', 'done'}
O_TO_R = {'public_params', 'keygen_O', 'oprf_response', 'psi_response', 'release', 'refuse'}


def test_toy_example(debug_env, workdir):
    dR_, dO_, b = data.toy()
    ag = dp.Agreement(n_R_max=4, n_bar_O=6, d_R=2, d_O=2, **b)
    R, O = protocol.run(ag, C.CKKSParams(), dR_, dO_, {},
                        dict(debug_no_dp=True, tau_override=data.TOY_TAU, ledger_path=os.path.join(workdir, 'l_toy.json')))
    toy = json.load(open(TOY_JSON))
    assert sorted(R['I']) == sorted(toy['I']) == ['104', '221']
    assert R['pi'] == {'104': 6, '221': 3} == toy['psi']
    G = np.array(toy['G'])
    c = np.array(toy['c'])
    assert np.abs(R['G1'] - G).max() < 1e-10
    assert np.abs(R['G2'] - G).max() < 1e-10
    assert np.abs(R['c2'] - c).max() < 1e-10
    assert abs(R['yy'] - toy['yy']) < 1e-10
    assert np.allclose(R['A'], np.array(toy['A']), atol=1e-12)
    assert abs(R['calibration']['Delta'] - toy['Delta']) < 1e-6


@pytest.fixture(scope='module')
def synthetic_noise_off(tmp_path_factory):
    os.environ['VFR_DEBUG_NO_DP'] = '1'
    try:
        wd = str(tmp_path_factory.mktemp('syn'))
        dR_, dO_, truth, b = data.synthetic(4096, 3500, 500, 300, seed=11)
        ag = dp.Agreement(n_R_max=600, n_bar_O=4096, d_R=4, d_O=6, **b)
        R, O = protocol.run(ag, C.CKKSParams(), dR_, dO_, dict(keep_raw=True, test_partial_share=True),
                            dict(debug_no_dp=True, keep_raw=True, test_record_c0p=True,
                                 ledger_path=os.path.join(wd, 'l.json')))
    finally:
        os.environ.pop('VFR_DEBUG_NO_DP', None)
    return dR_, dO_, truth, b, ag, R, O


def test_synthetic_linkage_exact(synthetic_noise_off):
    dR_, dO_, truth, b, ag, R, O = synthetic_noise_off
    assert set(R['I']) == truth['matched'] and R['n'] == 300
    assert all(R['pi'][i] == O['tau'][i] for i in R['I'])


def test_synthetic_noise_off_statistics(synthetic_noise_off):
    dR_, dO_, truth, b, ag, R, O = synthetic_noise_off
    ps = data.plaintext_stats(dR_, dO_, b['B_R'], b['B_O'], b['B_y'])
    scale = np.abs(ps['G']).max()
    assert np.abs(R['G1'] - ps['G']).max() < 1e-11 * scale
    assert np.abs(R['G2'] - ps['G']).max() < 1e-11 * scale
    assert np.abs(R['c2'] - ps['c']).max() < 1e-11 * scale
    assert abs(R['yy'] - ps['yy']) < 1e-11 * scale


def test_transcript(synthetic_noise_off):
    dR_, dO_, truth, b, ag, R, O = synthetic_noise_off
    tr_R, tr_O = R['transcript'], O['transcript']
    sentR = {r['type'] for r in tr_R if r['dir'] == 'sent'}
    sentO = {r['type'] for r in tr_O if r['dir'] == 'sent'}
    assert sentR <= R_TO_O and sentO <= O_TO_R
    # the two transcripts agree message by message
    assert [(r['type'], r['bytes'], r['sha256']) for r in tr_R if r['dir'] == 'sent'] == \
           [(r['type'], r['bytes'], r['sha256']) for r in tr_O if r['dir'] == 'recv']
    # a release carries exactly (c1', h_O): two ring elements modulo q0 (2 towers)
    N = 2 ** C.CKKSParams().logN
    for r in tr_R:
        if r['type'] == 'release':
            assert r['dir'] == 'recv' and r['fields'] == {'c1p': 2 * N * 8, 'h': 2 * N * 8}
    # O's OPRF evaluation count is exactly n_R_max
    req = [r for r in tr_O if r['type'] == 'oprf_request']
    assert len(req) == 1 and req[0]['fields'] == {'blinded': 32 * ag.n_R_max}
    # R never receives c0' (fingerprints recorded by O in the guarded test path)
    recvR = b''.join(f for d, t, f in R['raw'] if d == 'recv')
    assert len(O['debug']['c0p_fingerprints']) == 2
    for s, fp in O['debug']['c0p_fingerprints'].items():
        assert len(fp) == 256 and recvR.find(fp[:64]) < 0 and recvR.find(fp[128:192]) < 0
    # O never receives a decryption share of R (c1' s_R, computed by R in the guarded test path)
    recvO = b''.join(f for d, t, f in O['raw'] if d == 'recv')
    assert len(R['partial_share_fingerprints']) == 2
    for s, fp in R['partial_share_fingerprints'].items():
        assert recvO.find(fp[:64]) < 0 and recvO.find(fp[128:192]) < 0
    # nothing in R -> O names a decryption share
    for r in tr_O:
        if r['dir'] == 'recv':
            assert not any(k in f for f in r['fields'] for k in ('share', 'pdec', 'h_R', 'dec'))


def test_noise_on_matches_alg6(tmp_path):
    import alg6
    dR_, dO_, truth, b = data.synthetic(4096, 3500, 500, 300, seed=12)
    ag = dp.Agreement(n_R_max=600, n_bar_O=4096, d_R=4, d_O=6, B_beta_pub=5.0, **b)
    R, O = protocol.run(ag, C.CKKSParams(), dR_, dO_, dict(solve_seed=123, Bboot=1000),
                        dict(ledger_path=str(tmp_path / 'l.json')))
    res = R['result']
    assert res is not None and not res['refused']
    # O's ledger charged the run's full planned cost once, at admission, and nothing per release
    led = dp.Ledger(str(tmp_path / 'l.json'), eps_cap=1.0, delta_cap=1e-9, zeta=0.1, mode='fixed', xi=None)
    cost = O['calibration']['ledger_cost']
    assert len(led.d['runs']) == 1 and 'releases' not in led.d
    assert led.spent() == dict(rho=Fraction(cost['rho']), eps=Fraction(0), gamma=Fraction(cost['gamma']))
    assert cost['rho'] == O['calibration']['rho_run'] and cost['gamma'] == O['calibration']['gamma']
    p, dA = 11, 5
    sig = R['sigma']
    aL = aU = ag.alpha / 20

    class ZeroFirst:
        """numpy Generator whose first `k` standard_normal calls return zeros (alg6 draws the release noise
        itself; here the noise is already in the decoded statistics)."""
        def __init__(self, seed, k):
            self.g, self.k = np.random.default_rng(seed), k

        def standard_normal(self, size=None):
            if self.k > 0:
                self.k -= 1
                return np.zeros(size) if size is not None else 0.0
            return self.g.standard_normal(size)

    s1ref = alg6.stage1(ZeroFirst(0, 2), R['G1'], dA, sig['1'], sig['2'], aL, ag.theta_star, ag.kappa_Psi)
    for k in ('lam', 'kind', 'L_hat', 'r1', 'lam_I'):
        assert s1ref[k] == R['stage1'][k] or abs(s1ref[k] - R['stage1'][k]) < 1e-9 * max(1, abs(s1ref[k]))
    ref = alg6.solve_infer(ZeroFirst(123, 4), R['G2'], R['c2'], R['yy'], R['n'], dA, s1ref, sig['2'], sig['2'],
                           ag.B_y, ag.alpha, aL, aU, ag.B_beta_pub, Bboot=1000, truncate=True, antithetic=True)
    assert ref['refused'] == res['refused']
    for k in ('theta', 'Vmech', 'V', 'half_theta', 'half_beta'):
        assert np.allclose(ref[k], res[k], rtol=1e-10, atol=1e-12), k
    for k in ('RSS', 'Corr', 's2', 'U_spec'):
        assert abs(ref[k] - res[k]) <= 1e-9 * max(1.0, abs(ref[k])), k
    # the noise is on: decoded statistics differ from the plaintext ones by about sigma
    ps = data.plaintext_stats(dR_, dO_, b['B_R'], b['B_O'], b['B_y'])
    dev = (R['G2'] - ps['G'])[dA:, dA:]
    assert 0.3 * sig['2'] < np.std(dev[np.triu_indices(6)]) < 3 * sig['2']


def test_debug_flag_is_guarded(monkeypatch, tmp_path):
    monkeypatch.delenv('VFR_DEBUG_NO_DP', raising=False)
    dR_, dO_, b = data.toy()
    ag = dp.Agreement(n_R_max=4, n_bar_O=6, d_R=2, d_O=2, **b)
    with pytest.raises(RuntimeError, match='VFR_DEBUG_NO_DP'):
        protocol.run(ag, C.CKKSParams(), dR_, dO_, {}, dict(debug_no_dp=True, ledger_path=str(tmp_path / 'l.json')),
                     timeout=120)


def test_setup_refuses_short_psi_items(tmp_path):
    """O's Setup refuses (before the ledger is charged) when the PSI items are too short for delta_psi: one felt
    per item (32-bit items) at 4,096 rows and n_R^max = 600."""
    dR_, dO_, truth, b = data.synthetic(4096, 3500, 500, 300, seed=22)
    ag = dp.Agreement(n_R_max=600, n_bar_O=4096, d_R=4, d_O=6, **b)
    led = tmp_path / 'l.json'
    R, O = protocol.run(ag, C.CKKSParams(), dR_, dO_, {}, dict(ledger_path=str(led), psi=dict(felts=1)), timeout=300)
    assert R['refused'].startswith('PSI parameters') and 'felt' in R['refused']
    assert not led.exists()
