"""O's ledger (paper, Section 7.4; composition proof, Definitions 2.3 and 5.3, Theorem 5.2, Proposition 5.5):
the hockey-stick smudging cost eps'(kappa, delta'), the capacities and their rounding, admission without tolerance,
the full cost charged atomically at admission (no refund), and concurrent admissions that cannot exceed the cap."""
import json
import math
import multiprocessing as mp
from fractions import Fraction

import pytest

from vfr import dp

DP100 = 2.0 ** -100


# ------------------------------------------------------------------------------------- eps'(kappa, delta')
@pytest.mark.parametrize('kappa,delta_p,eps_ref', [(6, 1e-6, 0.15886), (6, 1e-9, 0.21364), (8, 1e-6, 0.036278),
                                                   (8, 1e-9, 0.050799), (10, 1e-6, 0.0081637), (10, 1e-9, 0.012044),
                                                   (20, DP100, 2.4620e-5), (30, DP100, 2.2433e-8),
                                                   (40, DP100, 2.0217e-11)])
def test_eps_smudge_is_the_tight_conversion(kappa, delta_p, eps_ref):
    """Tight values of the composition proof (Table 1 and the kappa script), the closed form as an upper bound, and
    delta(pi 4^-kappa, eps') <= delta' where log_delta_cks can evaluate it."""
    e = dp.eps_smudge(kappa, delta_p)
    assert e == pytest.approx(eps_ref, rel=5e-5) and e >= eps_ref * (1 - 5e-5)
    rho = math.pi * 4.0 ** -kappa
    assert e <= rho + 2 * math.sqrt(rho * math.log(1 / delta_p))
    if kappa <= 10:
        assert dp.log_delta_cks(rho, e) <= math.log(delta_p) + 1e-9


def test_eps_smudge_decreases_and_stays_valid():
    vals = [dp.eps_smudge(k, DP100) for k in range(1, 80)]
    assert all(b < a for a, b in zip(vals, vals[1:]))
    assert all(v >= 0 for v in vals)


# ------------------------------------------------------------------------------------- capacities and rounding
@pytest.mark.parametrize('eps,delta', [(10.0, 0.9e-6), (1.0, 0.9e-9), (0.5, 1e-9), (8.0, 1e-9), (9.99, 0.8e-6)])
def test_rho_tight_rounds_down(eps, delta):
    """rho_tight never returns a point above the largest rho with delta(rho, eps) <= delta, and it is tight."""
    r = dp.rho_tight(eps, delta)
    assert dp.log_delta_cks(r, eps) <= math.log(delta) - 2e-13
    assert dp.log_delta_cks(r * (1 + 1e-9), eps) > math.log(delta)


def test_capacities_are_rounded_down_and_add_up():
    sd = dp.ledger_capacities(10.0, 1e-6, 0.1, None)
    assert Fraction(sd['Gamma']) * (1 + Fraction(math.exp(10.0))) <= Fraction(1, 10) * Fraction(1e-6)
    assert sd['rho'] == pytest.approx(1.5297129439, rel=1e-9)
    hs = dp.ledger_capacities(10.0, 1e-6, 0.1, 1e-3)
    assert Fraction(hs['E']) <= Fraction(1e-3) * 10 / 2 and hs['E'] == pytest.approx(0.005, rel=1e-11)
    assert hs['rho'] == pytest.approx(1.52667697, rel=1e-8)                  # 0.198% below the SD rho_L
    assert hs['Gamma'] == pytest.approx(4.5625e-12, rel=1e-4)
    # Corollary 5.4 at eps = eps_I: e^E delta(rho_L, eps_I) + (1 + e^(eps_I + E)) Gamma' <= delta_L, eps_I + 2E <= eps_L
    assert hs['eps_I'] + 2 * hs['E'] <= 10.0
    total = math.exp(hs['E'] + dp.log_delta_cks(hs['rho'], hs['eps_I'])) + (1 + math.exp(hs['eps_I'] + hs['E'])) * hs['Gamma']
    assert total <= 1e-6


def _ledger(tmp_path, name='l.json', **kw):
    kw = dict(dict(eps_cap=10.0, delta_cap=1e-6, zeta=0.1, mode='filter', xi=1e-3, delta_prime=DP100), **kw)
    return dp.Ledger(str(tmp_path / name), **kw)


def test_filter_has_no_tolerance(tmp_path):
    """A run that fills a capacity exactly is admitted; any excess, however small, is refused (the earlier test
    accepted a relative excess of 1e-12)."""
    L = _ledger(tmp_path)
    cap = dict(rho=L.d['rho_cap'], eps_prime=L.d['E_cap'], gamma=L.d['gamma_cap'])
    for k in cap:
        L2 = _ledger(tmp_path, f'over_{k}.json')
        over = dict(rho=0.0, eps_prime=0.0, gamma=0.0)
        over[k] = cap[k] * (1 + 1e-13)
        assert not L2.admit_run(over, 'R', 'p')[0], k
    assert L.admit_run(cap, 'R', 'p')[0]                                      # exactly full
    for k in cap:
        tiny = dict(rho=0.0, eps_prime=0.0, gamma=0.0)
        tiny[k] = 5e-324
        assert not L.admit_run(tiny, 'R', 'p')[0], k
    assert L.spent() == dict(rho=Fraction(cap['rho']), eps=Fraction(cap['eps_prime']), gamma=Fraction(cap['gamma']))


def test_statistical_distance_fixed_rule_has_no_tolerance(tmp_path):
    """Runs fixed in advance: delta(rho, eps_L) + (1 + e^eps_L) Gamma <= delta_L, tested without the earlier
    +1e-9 tolerance on log delta."""
    L = _ledger(tmp_path, mode='fixed', xi=None, delta_prime=None)
    gam = 1e-13
    rho_max = dp.rho_tight(10.0, 1e-6 - (1 + math.exp(10.0)) * gam * (1 + 1e-9))
    assert L.admits(dict(rho=rho_max, gamma=gam))
    assert not L.admits(dict(rho=rho_max * (1 + 1e-9), gamma=gam))


def test_hockey_stick_fixed_rule(tmp_path):
    """Runs fixed in advance under the hockey-stick accounting: e^E delta(rho, eps_L - 2E) + (1 + e^(eps_L - E)) Gamma
    <= delta_L.  Such a ledger admits at least as many runs as the filter with the same cap."""
    cost = dict(rho=0.02, eps_prime=4e-11, gamma=2 * (2.0 ** -50 + DP100))
    counts = {}
    for mode in ('fixed', 'filter'):
        L = _ledger(tmp_path, f'{mode}.json', mode=mode)
        n = 0
        while L.admit_run(cost, 'R', 'p')[0]:
            n += 1
        counts[mode] = n
        sp = L.spent()
        if mode == 'fixed':
            E, rho, gam = float(sp['eps']), float(sp['rho']), float(sp['gamma'])
            lhs = math.exp(E + dp.log_delta_cks(rho, 10.0 - 2 * E)) + (1 + math.exp(10.0 - E)) * gam
            assert lhs <= 1e-6
            E2, rho2, gam2 = E + cost['eps_prime'], rho + cost['rho'], gam + cost['gamma']
            lhs2 = math.exp(E2 + dp.log_delta_cks(rho2, 10.0 - 2 * E2)) + (1 + math.exp(10.0 - E2)) * gam2
            assert lhs2 > 1e-6
    assert counts['fixed'] >= counts['filter'] == 76


def test_smudging_budget_can_bind(tmp_path):
    """A run whose kappa is below its kappa_L can exhaust E'_L before rho_L; the filter then refuses it."""
    L = _ledger(tmp_path)
    cost = dict(rho=1e-3, eps_prime=1e-3, gamma=1e-15)                         # 5 runs fill E'_L = 0.005
    n = 0
    while L.admit_run(cost, 'R', 'p')[0]:
        n += 1
    assert n == 4                                                              # E'_L is rounded down below 0.005
    assert L.spent()['rho'] + Fraction(cost['rho']) <= Fraction(L.d['rho_cap'])


# ------------------------------------------------------------------------------------- charged at admission
def test_full_cost_charged_at_admission_and_never_refunded(tmp_path):
    L = _ledger(tmp_path)
    cost = dict(rho=0.0149673, eps_prime=4.05e-11, gamma=1.78e-15)
    ok, why = L.admit_run(cost, 'R', 'p', meta=dict(kappa=40, n_rel=2))
    assert ok and why == 'ok'
    # charged in full immediately, before any release is made, and visible to every other handle on the file
    other = _ledger(tmp_path)
    for led in (L, other):
        assert led.spent() == dict(rho=Fraction(cost['rho']), eps=Fraction(cost['eps_prime']),
                                   gamma=Fraction(cost['gamma']))
        assert len(led.d['runs']) == 1 and led.d['runs'][0]['kappa'] == 40
    assert not hasattr(L, 'charge') and not hasattr(L, 'refund')               # no per-release charge, no refund
    on_disk = json.load(open(tmp_path / 'l.json'))
    assert on_disk['version'] == 3 and Fraction(on_disk['rho_spent']) == Fraction(cost['rho'])


def test_older_ledger_files_are_refused(tmp_path):
    p = tmp_path / 'old.json'
    p.write_text(json.dumps(dict(version=2, eps_cap=10.0, delta_cap=1e-6, zeta=0.1, mode='filter')))
    with pytest.raises(RuntimeError, match='older format'):
        dp.Ledger(str(p), eps_cap=10.0, delta_cap=1e-6)


# ------------------------------------------------------------------------------------- concurrency
def test_stale_handles_cannot_exceed_the_cap(tmp_path):
    """Two handles opened before either admits (as two O processes serving two researchers would be): each admission
    re-reads the state under the lock, so together they admit exactly what the cap allows."""
    A, B = _ledger(tmp_path), _ledger(tmp_path)
    cost = dict(rho=0.1, eps_prime=1e-11, gamma=1e-15)
    allowed = math.floor(Fraction(A.d['rho_cap']) / Fraction(cost['rho']))
    n, turn = 0, 0
    while True:
        ok = (A if turn % 2 == 0 else B).admit_run(cost, 'R%d' % (turn % 2), 'p')[0]
        if not ok:
            ok = (B if turn % 2 == 0 else A).admit_run(cost, 'R', 'p')[0]
            if not ok:
                break
        n += 1
        turn += 1
    assert n == allowed == 15
    A.refresh()
    assert A.spent()['rho'] == allowed * Fraction(cost['rho']) <= Fraction(A.d['rho_cap'])


def _admit_many(path, n_try, q):
    L = dp.Ledger(path, eps_cap=10.0, delta_cap=1e-6, zeta=0.1, mode='filter', xi=1e-3, delta_prime=DP100)
    cost = dict(rho=0.04, eps_prime=1e-11, gamma=1e-15)
    q.put(sum(L.admit_run(cost, 'R', 'p')[0] for _ in range(n_try)))


def test_concurrent_processes_cannot_exceed_the_cap(tmp_path):
    path = str(tmp_path / 'shared.json')
    L = dp.Ledger(path, eps_cap=10.0, delta_cap=1e-6, zeta=0.1, mode='filter', xi=1e-3, delta_prime=DP100)
    allowed = math.floor(Fraction(L.d['rho_cap']) / Fraction(0.04))
    ctx = mp.get_context('fork')
    q = ctx.Queue()
    procs = [ctx.Process(target=_admit_many, args=(path, 20, q)) for _ in range(4)]
    for p in procs:
        p.start()
    total = sum(q.get(timeout=120) for _ in procs)
    for p in procs:
        p.join(timeout=60)
    L.refresh()
    assert total == allowed == 38 == len(L.d['runs'])
    assert L.spent()['rho'] == allowed * Fraction(0.04) <= Fraction(L.d['rho_cap'])
