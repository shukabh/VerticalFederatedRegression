"""Exact full-record sensitivity of the full-record SSP baseline (experiments/baselines/sens_full.py): Theorem S10c
for blocks of dimension >= 2 and Lemma S10d (docs/proofs/stats_s7_s10.tex) for one block of dimension 1."""
import math
import os
import sys

import numpy as np
import pytest

pytest.importorskip('sympy')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'experiments', 'baselines'))
import sens_full as sf  # noqa: E402

BLOCK_TAMI = (math.sqrt(2), 1.0, 1.7188210326789277, True)       # tamielectric, block normalization, fold 0
BLOCK_3DROAD = (math.sqrt(2), 1.0, 3.4330469390257967, True)     # 3droad, block normalization, fold 0


def test_closed_form_unit_bounds():
    D, Dn, DG = sf.delta_full(1.0, 1.0, 1.0, False, 3, 3)
    assert D ** 2 == pytest.approx(32 / 3, rel=1e-11) and DG ** 2 == pytest.approx(6, rel=1e-11)


@pytest.mark.parametrize('b', [BLOCK_TAMI, BLOCK_3DROAD, (1.0, 1.0, 1.0, False)])
def test_dim1_values_and_order(b):
    D, Dn, DG = sf.delta_full(*b, 1, 2)
    info = sf.delta_full_info(*b, 1, 2)
    assert info['method'] == 'dim1'
    cl = [math.sqrt(sf.sup2_closed(*b, 2, True)[0]), math.sqrt(sf.sup2_closed(*b, 2, False)[0]),
          math.sqrt(sf.sup2_closed(*b, 1)[0])]
    for e, c in zip((D, Dn, DG), cl):
        assert e <= c * (1 + 1e-11)                    # Theorem S10c is a valid bound for a 1-dimensional block
    assert DG == pytest.approx(math.sqrt(10) if b[3] else math.sqrt(4.5), rel=1e-11)
    assert DG < cl[2] * (1 - 1e-3)                     # and not tight there


def test_dim1_known_values():
    assert sf.delta_full(*BLOCK_TAMI, 1, 2)[0] == pytest.approx(5.999903776050534, rel=1e-13)
    assert sf.delta_full(*BLOCK_3DROAD, 1, 2)[0] == pytest.approx(13.574342765516606, rel=1e-13)
    assert sf.delta_full(*BLOCK_3DROAD, 1, 2)[1] == pytest.approx(11.892423446334881, rel=1e-13)


def test_dim1_never_below_a_dense_search():
    for b in (BLOCK_3DROAD, (3.0, 0.5, 1.0, True), (1.2, 2.5, 0.4, True)):
        iota = 1.0 if b[3] else 0.0
        Ru, o = b[0] ** 2 - iota, b[1] ** 2
        ex = sf.delta_full(*b, 1, 2)
        for val, (wy, wyy) in zip(ex, ((True, True), (True, False), (False, False))):
            br = math.sqrt(sf.brute_dim1(Ru, o, b[2], iota, wy, wyy, n=17, starts=10))
            assert val >= br * (1 - 1e-13)
            assert val <= br * (1 + 1e-6)              # and the search gets close: the value is attained


def test_dim1_swap_symmetry():
    b = (2.0, 1.5, 3.0, True)
    a = sf.delta_full(*b, 1, 3)
    # d_O = 1: exchange the roles of R_u = B_R^2 - 1 and o = B_O^2 (B_R' with B_R'^2 - 1 = o, B_O' = sqrt(R_u))
    c = sf.delta_full(math.sqrt(b[1] ** 2 + 1), math.sqrt(b[0] ** 2 - 1), b[2], True, 3, 1)
    assert np.allclose(a, c, rtol=1e-12)


def test_both_dim1_refused():
    with pytest.raises(NotImplementedError):
        sf.delta_full(1.0, 1.0, 1.0, False, 1, 1)
