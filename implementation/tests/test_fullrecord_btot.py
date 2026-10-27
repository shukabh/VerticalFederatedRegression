"""B_tot, Delta_x and the decoding offset of the FULL-RECORD circuit (the full-record SSP baseline; not a protocol
mode): vfr/btot.py with CircuitParams.fullrecord=True and dp.calibrate(..., fullrecord_sens=...), as re-derived in
Lemma S10e of docs/proofs/stats_s7_s10.tex.  The v3.1 circuit (fullrecord=False) is unchanged."""
import dataclasses
import math

import pytest

from vfr import btot as bt
from vfr import common as C
from vfr import dp

Om = 2 ** 52.5
BASE = dict(N=2 ** 14, n_bar=65536, d_R=4, d_O=6, B_R=5.59, B_O=6.24, B_y=2.89, q1=2 ** 59, q2=2 ** 59,
            Omega_U=2 ** 68, Omega_V=2 ** 53, Omega_u=2 ** 59, Qtilde_max_A=2 ** 118, Qtilde_max_C=2 ** 118,
            P=2 ** 119, k_digit_A=1, k_digit_C=1, dnum_A=2, dnum_C=2, k_P=2)
# the full-record sensitivities at the simulation population (Theorem S10c; experiments/baselines/sens_full.py)
SENS_SIM = (92.67, 92.67, 85.82)


def _cp(**kw):
    return bt.CircuitParams(**{**BASE, **kw})


def test_entry_counts():
    assert bt.n_A(4) == 14 and bt.n_A(1) == 2
    v, f = bt.entry_counts(4, 6, True), bt.entry_counts(4, 6, True, fullrecord=True)
    assert (v[1], v[2], v['y']) == (51, 63, 1) and (f[1], f[2], f['y']) == (65, 77, 1)
    f = bt.entry_counts(4, 6, False, fullrecord=True)
    assert (f[1], f[2], f['y']) == (65, 76, 1)


def test_v31_circuit_unchanged_by_the_flag_default():
    cp = _cp()
    assert not cp.fullrecord
    for s in (1, 2):
        assert bt.gamma_E(cp, s, True) == bt.gamma_E(cp, s, True, ones=False)


def test_btot_full_record_is_v31_with_more_entries_and_gamma():
    """Lemma S10e(a): B_tot changes only through |E| + n_A and Gamma_E + n_A."""
    cpv, cpf = _cp(), _cp(fullrecord=True)
    for s in (1, 2):
        assert bt.gamma_E(cpf, s, True) == bt.gamma_E(cpv, s, True) + 14
        Tv, Tf = bt.bounds(cpv, s, True), bt.bounds(cpf, s, True)
        assert Tf['T6'] == Tv['T6'] and Tf['T7'] == Tv['T7']
        for k in ('T1', 'T2', 'T3', 'T4', 'T5', 'T8'):
            assert Tf[k] > Tv[k]
        assert bt.B_tot(cpf, s, True) > bt.B_tot(cpv, s, True)
    # T2..T5 scale with the entry count only: sqrt(77/63) for T2-T4 (their leading parts), 77/63 for T5
    Tv, Tf = bt.bounds(cpv, 2, True), bt.bounds(cpf, 2, True)
    assert Tf['T5'] / Tv['T5'] == pytest.approx(77 / 63, rel=1e-12)
    assert Tf['T3'] / Tv['T3'] == pytest.approx(math.sqrt(77 / 63), rel=1e-12)
    assert bt.bounds(cpv, 'y', False) == bt.bounds(_cp(fullrecord=True), 'y', False)


def test_btot_full_record_has_no_U_side():
    """Lemma S10e(a): no bound on R's vectors enters B_tot (B_R appears only through U)."""
    for s in (1, 2):
        assert bt.B_tot(_cp(fullrecord=True, B_R=5.59), s, True) == bt.B_tot(_cp(fullrecord=True, B_R=50.0), s, True)
        assert bt.B_tot(_cp(B_R=5.59), s, True) == bt.B_tot(_cp(B_R=50.0), s, True)


def test_U_side_terms():
    """nu_U gains d_R B_R^2 + d_R(d_R+1)/2 B_R^4; zeta_E^2 gains B_R^4; nu_Vmax counts the ones."""
    cpv, cpf = _cp(), _cp(fullrecord=True)
    r = 5.59 ** 2
    for s in (1, 2):
        assert bt.nu_U(cpf, s, True) ** 2 == pytest.approx(bt.nu_U(cpv, s, True) ** 2 + 4 * r + 10 * r * r, rel=1e-12)
        assert bt.zeta_E(cpf, s, True) ** 2 == pytest.approx(bt.zeta_E(cpv, s, True) ** 2 + r * r, rel=1e-12)
        assert bt.nu_V_max(cpf, s, True) ** 2 == pytest.approx(bt.nu_V_max(cpv, s, True) ** 2 + 14, rel=1e-12)
        assert bt.nu_V_max(cpv, s, True) ** 2 >= bt.gamma_E(cpv, s, True)       # per-entry maxima >= row bound


def test_delta_x_full_record():
    """Lemma S10e(b): the whole-record adjacency adds Omega_U nu_dU + sqrt(N) Omega_V nu_Vmax to the bracket; with the
    extra terms removed (nu_U of the full circuit, v3.1 bracket) delta_x is between the two."""
    cpv, cpf = _cp(), _cp(fullrecord=True)
    for s in (1, 2):
        dv, df = bt.delta_x(cpv, s, True, 86.18), bt.delta_x(cpf, s, True, 92.67)
        assert df > dv
        N, q12 = cpf.N, cpf.q1 * cpf.q2
        nE = bt.entry_counts(4, 6, True, True)[s]
        nuV, nuU = 2 * math.sqrt(bt.gamma_E(cpv, s, True)), bt.nu_U(cpf, s, True)
        K = cpf.Omega_V * nuV + math.sqrt(N) * cpf.Omega_U * nuU + 2 * cpf.Omega_U * nuU \
            + math.sqrt(N) * cpf.Omega_V * bt.nu_V_max(cpf, s, True)
        expect = (math.sqrt(N / 2) * cpf.Omega_u / q12 * K + math.sqrt(N) / 2 * cpf.Omega_U * cpf.Omega_V / q12
                  * math.sqrt(nE) * 92.67 + N ** 1.5 / (4 * q12) * math.sqrt(nE) * K + 1.0)
        assert df == pytest.approx(expect, rel=1e-12)
        eps_x = cpf.Omega_u / q12 * N ** 1.5 * math.sqrt(nE) / (2 * math.sqrt(2)) + N ** 2.5 * nE / (8 * q12)
        assert bt.epsilon_x(cpf, s, True) == pytest.approx(2 * eps_x, rel=1e-12) and 2 * eps_x < 2 ** -30


def test_offset_full_record_larger():
    cpv, cpf = _cp(), _cp(fullrecord=True)
    for s in (1, 2):
        ov, of = bt.offset_bounds(cpv, s, True, 2000), bt.offset_bounds(cpf, s, True, 2000)
        assert of['d_max'] > ov['d_max'] and of['d_bar'] > ov['d_bar']


@pytest.fixture(scope='module')
def ctx():
    ck = C.make_ckks(C.CKKSParams())
    qs = list(ck.qs)
    return ck.N, qs[2], qs[3], qs[0] * qs[1], C.gadget_constants(ck)


def test_calibrate_full_record(ctx):
    """At the paper's evaluation agreement (setting I, eps 1, delta 1e-9, n_bar_O 65,536): the full-record circuit's
    B_tot is not flooded by the v3.1 stage-2 noise, and Calibrate for the full-record circuit admits it one grid step
    higher (reports/eng/fullrecord_btot.json)."""
    ag = dp.Agreement(n_R_max=2000, n_bar_O=65536, d_R=4, d_O=6, B_R=5.59, B_O=6.24, B_y=2.89, eps_run=1.0,
                      delta_run=1e-9, **dp.setting('I'))
    v = dp.calibrate(ag, *ctx)
    f = dp.calibrate(ag, *ctx, fullrecord_sens=SENS_SIM)
    assert 'fullrecord' not in v and f['fullrecord']
    assert f['entries'] == {'1': 65, '2': 77, 'y': 1} and f['joint'] and f['kappa'] == v['kappa'] == 40
    assert (f['Delta'], f['Delta_noyy'], f['Delta_G']) == SENS_SIM
    for s, r in f['releases'].items():                                   # flooding holds, tested exactly
        assert dp.floods(r['sigma_c2'], r['B_tot'], f['kappa'])
        assert r['epsilon_x'] < 2 ** -30
    assert f['q0_bits'] > f['q0_need_bits']
    assert abs(v['log2_Omega_f'] - 62.625) < 1e-5 and abs(f['log2_Omega_f'] - 62.65625) < 1e-5   # one grid step
    # the v3.1 stage-2 noise at the v3.1 scale does not flood the full-record B_tot there
    cp = dataclasses.replace(bt.CircuitParams(N=ctx[0], n_bar=65536, d_R=4, d_O=6, B_R=5.59, B_O=6.24, B_y=2.89,
                                              q1=ctx[1], q2=ctx[2], Omega_U=float(dp.admissible_scale(68)),
                                              Omega_V=v['Omega_V'], Omega_u=float(ctx[1]), **ctx[4]), fullrecord=True)
    assert not dp.floods(v['releases']['2']['sigma_c2'], bt.B_tot(cp, 2, True), 40)
    assert dp.floods(v['releases']['2']['sigma_c2'], v['releases']['2']['B_tot'], 40)
