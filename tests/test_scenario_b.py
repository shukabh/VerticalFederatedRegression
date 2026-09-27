"""Tests for the user's ORIGINAL scenario_b modules (psi_common, phase1_common, he_backend)
and the parts of party_o.py / party_r.py they drive.

    python -m pytest tests/ -q                      # plaintext backend; OpenFHE tests skip
    <python with openfhe> -m pytest tests/ -q       # also runs the OpenFHE tests

The he_backend factories default to backend="openfhe", so every plaintext call below
passes backend="plaintext" explicitly.
"""
import os
import socket
import sys
import threading

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "scenario_b"))
import he_backend as hb  # noqa: E402
import phase1_common as p1  # noqa: E402
import psi_common as pc  # noqa: E402
from party_o import power_from_windows  # noqa: E402

needs_openfhe = pytest.mark.skipif(not hb._HAVE_OPENFHE, reason="openfhe not importable")

D_R, D_O = 3, 4
P_DIM = D_R + D_O


# ---------------------------------------------------------------------------
# phase1_common
# ---------------------------------------------------------------------------

def _bias_matrix(P, dR, dO):
    """Materialise the closure returned by phase1_common._bias_operator."""
    M = p1._bias_operator(P, dR, dO)
    return np.column_stack([M(e) for e in np.eye(dR + dO)])


def test_bias_operator_matches_exact_second_moment():
    """sigma^-2 E[P E P E] for the block-masked single-draw mechanism, summed exactly."""
    rng = np.random.default_rng(1)
    basis = []
    for i in range(P_DIM):
        for j in range(i, P_DIM):
            if i < D_R and j < D_R:            # A block is never noised
                continue
            S = np.zeros((P_DIM, P_DIM)); S[i, j] = S[j, i] = 1.0
            basis.append(S)
    X = rng.normal(size=(200, P_DIM))
    P = np.linalg.inv(X.T @ X)
    exact = sum(P @ S @ P @ S for S in basis)
    assert np.abs(exact - _bias_matrix(P, D_R, D_O)).max() < 1e-18


def test_draw_noise_structure():
    rng = np.random.default_rng(3)
    nz = p1.draw_noise(D_R, D_O, 1.0, rng)
    assert nz.E_B.shape == (D_O, D_O) and nz.E_C.shape == (D_R, D_O)
    assert nz.f_R.shape == (D_R,) and nz.f_O.shape == (D_O,) and np.isscalar(nz.e_yty)
    assert np.array_equal(nz.E_B, nz.E_B.T)
    sig, N = 2.0, 6000
    EB2 = np.zeros((D_O, D_O)); EC2 = np.zeros((D_R, D_O)); f2 = 0.0
    for _ in range(N):
        nz = p1.draw_noise(D_R, D_O, sig, rng)
        EB2 += nz.E_B ** 2; EC2 += nz.E_C ** 2; f2 += np.mean(nz.f_R ** 2) + np.mean(nz.f_O ** 2)
    # single draw, NOT GOE: the diagonal of E_B has variance sigma^2, not 2 sigma^2
    assert np.allclose(EB2 / N, sig ** 2, rtol=0.08)
    assert np.allclose(EC2 / N, sig ** 2, rtol=0.08)
    assert abs(f2 / (2 * N) - sig ** 2) < 0.1 * sig ** 2
    A = np.eye(D_R)
    out = p1.apply_noise(A, np.zeros((D_O, D_O)), np.zeros((D_R, D_O)), np.zeros(D_R),
                         np.zeros(D_O), 0.0, nz)
    assert out[0] is A                          # A stays clean


def _gram_with_gap():
    """Gram whose lambda_min is well below lambda_min(A) (near-collinear O block),
    so the three gate branches are separable by choice of sigma."""
    rng = np.random.default_rng(2)
    X = rng.uniform(size=(400, P_DIM))
    X[:, -1] = X[:, -2] + 0.05 * rng.normal(size=400)
    G = X.T @ X
    c = G @ rng.normal(size=P_DIM)
    lmin, lminA = np.linalg.eigvalsh(G)[0], np.linalg.eigvalsh(G[:D_R, :D_R])[0]
    assert lmin < 0.5 * lminA
    return G, c, lmin, lminA


def _edge(sigma):
    return 2.0 * sigma * np.sqrt(P_DIM)


def _on_grid(lam, lam0=1e-3, grow=1.5):
    k = np.log(lam / lam0) / np.log(grow)
    return abs(k - round(k)) < 1e-9


def test_select_ridge_sigma_zero():
    G, _, _, _ = _gram_with_gap()
    assert p1.select_ridge(G, D_R, D_O, 0.0) == ("O", 0.0, np.inf)


def test_select_ridge_no_ridge_needed_returns_floor():
    """rho(0) >= target: the geometric search stops at its first point, lambda = 1e-3."""
    G, c, lmin, _ = _gram_with_gap()
    sigma = lmin / _edge(1.0) / 3.0             # rho(0) = 3
    mode, lam, rho = p1.select_ridge(G, D_R, D_O, sigma, target=2.0)
    assert mode == "O" and lam == 1e-3 and rho >= 2.0
    sol = p1.solve_and_correct(G, c, D_R, D_O, sigma, mode="auto", rho_target=2.0)
    assert sol.Psi_mode == "O" and sol.lam == 1e-3


def test_select_ridge_O_block_branch_is_minimal_on_grid():
    G, c, lmin, lminA = _gram_with_gap()
    sigma = lminA / _edge(1.0) / 2.5            # ceiling 2.5 >= 2 > rho(0)
    assert lmin / _edge(sigma) < 2.0
    PiO = p1.ridge_selector(D_R, D_O, "O")
    mode, lam, rho = p1.select_ridge(G, D_R, D_O, sigma, target=2.0)
    assert mode == "O" and lam > 1e-3 and rho >= 2.0 and _on_grid(lam)
    assert p1.rho_lambda(G, lam / 1.5, PiO, sigma, P_DIM) < 2.0      # previous grid point fails
    # overshoot vs the exact minimal lambda is below the growth factor
    lo, hi = 0.0, lam
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        lo, hi = (lo, mid) if p1.rho_lambda(G, mid, PiO, sigma, P_DIM) >= 2.0 else (mid, hi)
    assert hi <= lam < 1.5 * hi


def test_select_ridge_escalates_to_full_ridge():
    G, c, lmin, lminA = _gram_with_gap()
    sigma = lminA / _edge(1.0) / 1.5            # ceiling 1.5 < 2: O-block ridge cannot validate
    mode, lam, rho = p1.select_ridge(G, D_R, D_O, sigma, target=2.0)
    lam_exact = 2.0 * _edge(sigma) - lmin       # full ridge: lambda_min(G + lam I) = lmin + lam
    assert mode == "I" and rho >= 2.0 and _on_grid(lam)
    assert lam_exact <= lam < 1.5 * lam_exact + 1e-9


def test_solve_and_correct_matches_its_formulae():
    G, c, lmin, lminA = _gram_with_gap()
    for sigma in (lmin / _edge(1.0) / 3.0, lminA / _edge(1.0) / 2.5, lminA / _edge(1.0) / 1.5):
        mode, lam, _ = p1.select_ridge(G, D_R, D_O, sigma, target=2.0)
        sol = p1.solve_and_correct(G, c, D_R, D_O, sigma, mode="auto", rho_target=2.0)
        assert (sol.Psi_mode, sol.lam) == (mode, lam)
        Glam = G + lam * p1.ridge_selector(D_R, D_O, mode)
        br = np.linalg.solve(Glam, c)
        Mm = _bias_matrix(np.linalg.inv(Glam), D_R, D_O)
        assert np.allclose(sol.beta_ridge, br, rtol=1e-10, atol=1e-12)
        assert np.allclose(sol.beta_bc, br - sigma ** 2 * Mm @ br, rtol=1e-9, atol=1e-12)
        assert np.allclose(sol.beta_ridge, p1.oracle_ridge(G, c, D_R, D_O, lam, mode), rtol=1e-10)


# ---------------------------------------------------------------------------
# psi_common + he_backend (plaintext): full labeled-PSI round trip
# ---------------------------------------------------------------------------

def test_framing_roundtrip_and_eof():
    a, b = socket.socketpair()
    payload = os.urandom(3_000_000)
    th = threading.Thread(target=pc.send_msg, args=(a, payload))
    th.start()
    assert pc.recv_msg(b) == payload
    th.join()
    a.close()
    assert pc.recv_msg(b) is None               # EOF -> None (callers must check)
    b.close()


def test_lagrange_and_membership_polys():
    t = pc.PLAIN_MODULUS
    rng = np.random.default_rng(0)
    xs = [int(v) for v in rng.integers(0, t - 1, 9)]
    ys = [int(v) for v in rng.integers(0, 5000, 9)]
    P = pc.poly_from_roots_mod(xs, t)
    L = pc.lagrange_coeffs(list(zip(xs, ys)), t)
    assert all(pc.poly_eval_mod(P, x, t) == 0 for x in xs)
    assert all(pc.poly_eval_mod(L, x, t) == y for x, y in zip(xs, ys))
    assert pc.lagrange_coeffs([(5, 1), (5, 2), (7, 3)], t) == pc.lagrange_coeffs([(5, 1), (7, 3)], t)


def _psi_round_trip(ids_R, ids_O, t, NB, NH=3, DCAP=4, W=4, backend="plaintext"):
    """party_r.py lines 59-95 and party_o.py lines 64-96, in one process."""
    own = hb.bfv_owner(plain_modulus=t, mult_depth=4, n_slots=NB, ring_dim=NB, backend=backend)
    pub = hb.bfv_from_public(own.public_blob(), backend=backend)
    tb = pc.build_cuckoo_table(ids_R, np.random.default_rng(0), NB, NH)
    assert tb is not None
    P_layers, L_layers, D, alpha, coll = pc.bin_and_interpolate(ids_O, t, NB, NH, DCAP)
    assert 1 < D <= (1 << (W - 1))
    yv = [pc.DUMMY_Y] * NB
    for slot, x in tb.items():
        yv[slot] = pc.id_to_field(int(x), t)
    wins = [pub.ct_from_bytes(own.ct_bytes(own.encrypt([pow(v, 1 << i, t) if v else 0 for v in yv])))
            for i in range(W)]
    powers = {d: power_from_windows(d, wins, pub) for d in range(1, D + 1)}
    rng = np.random.default_rng(7)
    found = {}
    for a in range(alpha):
        ctP = pub.mul_pt(powers[1], P_layers[a][1])
        for d in range(2, D + 1):
            ctP = pub.add(ctP, pub.mul_pt(powers[d], P_layers[a][d]))
        ctP = pub.add_pt(ctP, P_layers[a][0])
        ctL = pub.mul_pt(powers[1], L_layers[a][1])
        for d in range(2, D):
            ctL = pub.add(ctL, pub.mul_pt(powers[d], L_layers[a][d]))
        ctL = pub.add_pt(ctL, L_layers[a][0])
        ct_s = pub.mul_pt(ctP, rng.integers(1, t, NB).tolist())
        ct_q = pub.add(pub.mul_pt(ctP, rng.integers(1, t, NB).tolist()), ctL)
        sdec = own.decrypt(own.ct_from_bytes(pub.ct_bytes(ct_s)), NB)
        qdec = own.decrypt(own.ct_from_bytes(pub.ct_bytes(ct_q)), NB)
        for slot, x in tb.items():
            if sdec[slot] % t == 0 and x not in found:
                found[x] = qdec[slot] % t
        # the HE result equals the plaintext mirror in psi_common
        for slot in list(tb)[:5]:
            Pv, Lv = pc.eval_layers_plaintext(P_layers, L_layers, D, alpha, slot, yv[slot], t)[a]
            assert Pv == pc.poly_eval_mod([P_layers[a][d][slot] for d in range(D + 1)], yv[slot], t)
    return found, alpha, D, coll


def test_plaintext_psi_round_trip_recovers_matches_and_row_labels():
    rng = np.random.default_rng(1)
    pool = [int(v) for v in rng.choice(10**15, 700, replace=False)]
    ids_R = pool[:150]
    ids_O = pool[100:600]                       # 50 shared
    rng.shuffle(ids_O)
    found, alpha, D, coll = _psi_round_trip(ids_R, ids_O, pc.PLAIN_MODULUS, NB=512)
    truth = {x: j for j, x in enumerate(ids_O) if x in set(ids_R)}
    assert alpha >= 2                           # d_cap = 4 forces several partitions
    assert coll == 0
    assert found == truth                       # set AND O-row label for every match


def test_plaintext_ckks_semantics():
    own = hb.ckks_owner(mult_depth=2, scale_bits=50, batch_size=16, backend="plaintext")
    pub = hb.ckks_from_public(own.public_blob(), backend="plaintext")
    b = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 1.0]); w = np.arange(10.0)
    ct = pub.add_scalar(pub.inner_product(pub.ct_from_bytes(own.ct_bytes(own.encrypt(b))), w), 0.5)
    assert own.decrypt_slot0(ct) == pytest.approx(float(b @ w) + 0.5)
    assert np.all(ct[1:] == 0.0)                # plaintext backend: result lives in slot 0 only


def test_factories_default_to_openfhe():
    if hb._HAVE_OPENFHE:
        pytest.skip("openfhe importable: default backend works")
    with pytest.raises(RuntimeError):
        hb.bfv_owner(plain_modulus=pc.PLAIN_MODULUS, mult_depth=2, n_slots=16)


# ---------------------------------------------------------------------------
# OpenFHE backend (skipped when openfhe is not importable)
# ---------------------------------------------------------------------------

@needs_openfhe
def test_openfhe_bfv_ops_including_upper_half_values():
    t = 366379009                               # calibrated t for the base config, 1 mod 32768
    own = hb.bfv_owner(plain_modulus=t, mult_depth=4, n_slots=16384, ring_dim=16384, backend="openfhe")
    pub = hb.bfv_from_public(own.public_blob(), backend="openfhe")
    assert own.n_slots() == 16384
    v = [1, 2, t - 1, t // 2 + 5, 12345]
    m = [3, 3, t - 2, 3, 1]
    ct = pub.ct_from_bytes(own.ct_bytes(own.encrypt(v)))
    r = pub.add_pt(pub.mul_pt(pub.mul_ct(ct, ct), m), [7] * 5)
    got = [x % t for x in own.decrypt(own.ct_from_bytes(pub.ct_bytes(r)), 5)]
    assert got == [(a * a * k + 7) % t for a, k in zip(v, m)]


@needs_openfhe
def test_openfhe_psi_round_trip():
    rng = np.random.default_rng(4)
    pool = [int(v) for v in rng.choice(10**15, 400, replace=False)]
    ids_R, ids_O = pool[:100], pool[60:360]
    t = 366379009
    found, _, _, _ = _psi_round_trip(ids_R, ids_O, t, NB=16384, DCAP=2, backend="openfhe")
    assert found == {x: j for j, x in enumerate(ids_O) if x in set(ids_R)}


@needs_openfhe
def test_openfhe_ckks_inner_product():
    own = hb.ckks_owner(mult_depth=2, scale_bits=50, batch_size=4096, backend="openfhe")
    pub = hb.ckks_from_public(own.public_blob(), backend="openfhe")
    rng = np.random.default_rng(0)
    b = (rng.uniform(size=4000) < 0.3).astype(float); x = rng.uniform(size=4000)
    ct = pub.add_scalar(pub.inner_product(pub.ct_from_bytes(own.ct_bytes(own.encrypt(b))), x * x), 1.25)
    assert own.decrypt_slot0(own.ct_from_bytes(pub.ct_bytes(ct))) == pytest.approx(float(b @ (x * x)) + 1.25,
                                                                                   abs=1e-6)
