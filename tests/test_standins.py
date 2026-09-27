"""Checks for the STAND-IN modules scenario_b/{psi_common,phase1_common,he_backend}.py.

    python -m pytest tests/ -q
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "scenario_b"))
import he_backend as hb  # noqa: E402
import phase1_common as p1  # noqa: E402
import psi_common as pc  # noqa: E402

T = 1073872897          # prime, = 1 mod 2*16384 (60-bit safe path is exercised below too)


def _ev(coeffs, x, t):
    return sum(c * pow(x, k, t) for k, c in enumerate(coeffs)) % t


def test_framing_roundtrip():
    import socket
    import threading
    a, b = socket.socketpair()
    payload = os.urandom(3_000_000)
    th = threading.Thread(target=pc.send_msg, args=(a, payload))
    th.start()
    assert pc.recv_msg(b) == payload
    th.join()
    a.close(); b.close()


def test_poly_and_interpolation():
    rng = np.random.default_rng(0)
    xs = [int(v) for v in rng.integers(1, T, 9)]
    ys = [int(v) for v in rng.integers(0, 5000, 9)]
    P = pc._poly_from_roots(xs, T)
    L = pc._interpolate(xs, ys, T)
    assert P[-1] == 1 and all(_ev(P, x, T) == 0 for x in xs)
    assert all(_ev(L, x, T) == y for x, y in zip(xs, ys))


def _psi_in_process(ids_R, ids_O, t, NB, NH=3, DCAP=4, W=4):
    """R's and O's Phase-0 logic (as in party_r.py / party_o.py) on the plaintext backend."""
    own = hb.bfv_owner(t, 4, NB, NB, "plaintext")
    pub = hb.bfv_from_public(own.public_blob(), "plaintext")
    tb = pc.build_cuckoo_table(ids_R, np.random.default_rng(0), NB, NH)
    assert tb is not None
    P_layers, L_layers, D, alpha, coll = pc.bin_and_interpolate(ids_O, t, NB, NH, DCAP)
    assert D <= (1 << (W - 1)) and D > 1
    yv = [pc.DUMMY_Y] * NB
    for s, x in tb.items():
        yv[s] = pc.id_to_field(x, t)
    wins = [own.encrypt([pow(v, 1 << i, t) if v else 0 for v in yv]) for i in range(W)]

    def power(d):
        acc = None
        for i in range(d.bit_length()):
            if (d >> i) & 1:
                acc = wins[i] if acc is None else pub.mul_ct(acc, wins[i])
        return acc

    powers = {d: power(d) for d in range(1, D + 1)}
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
        r = np.random.default_rng(a)
        s = own.decrypt(pub.mul_pt(ctP, r.integers(1, t, NB).tolist()), NB)
        q = own.decrypt(pub.add(pub.mul_pt(ctP, r.integers(1, t, NB).tolist()), ctL), NB)
        for slot, x in tb.items():
            if s[slot] % t == 0:
                found.setdefault(x, q[slot] % t)
    return found, alpha, D, coll


@pytest.mark.parametrize("t", [T, 1152921504606683137])   # 30-bit and 60-bit primes
def test_labeled_psi_recovers_matches_and_labels(t):
    rng = np.random.default_rng(1)
    pool = [int(v) for v in rng.choice(10**15, 700, replace=False)]
    ids_R = pool[:150]
    ids_O = pool[100:600]                                 # 50 shared
    rng.shuffle(ids_O)
    found, alpha, D, coll = _psi_in_process(ids_R, ids_O, t, NB=512)
    truth = {x: j for j, x in enumerate(ids_O) if x in set(ids_R)}
    assert alpha >= 2                                     # DCAP=4 forces partitioning
    assert coll == 0
    assert found == truth


def test_ridge_gate_three_branches():
    rng = np.random.default_rng(2)
    d_R, d_O = 3, 4
    X = rng.uniform(size=(400, d_R + d_O))
    X[:, -1] = X[:, -2] + 0.05 * rng.normal(size=400)    # near-collinear O block
    G = X.T @ X
    assert np.linalg.eigvalsh(G)[0] < 0.5 * np.linalg.eigvalsh(G[:d_R, :d_R])[0]
    c = G @ rng.normal(size=d_R + d_O)
    lmin = np.linalg.eigvalsh(G)[0]
    lminA = np.linalg.eigvalsh(G[:d_R, :d_R])[0]
    edge = lambda s: 2 * s * np.sqrt(d_R + d_O)
    s_ols = lmin / edge(1.0) / 3.0                        # rho(0) = 3 >= 2
    sol = p1.solve_and_correct(G, c, d_R, d_O, s_ols, rho_target=2.0)
    assert sol.Psi_mode == "none" and sol.lam == 0.0
    s_O = lminA / edge(1.0) / 2.5                         # ceiling 2.5 > 2 > rho(0)
    assert lmin / edge(s_O) < 2.0
    sol = p1.solve_and_correct(G, c, d_R, d_O, s_O, rho_target=2.0)
    assert sol.Psi_mode == "O" and sol.lam > 0 and sol.rho >= 2.0
    PiO = np.diag([0.0] * d_R + [1.0] * d_O)              # minimality of lambda
    assert np.linalg.eigvalsh(G + 0.999 * sol.lam * PiO)[0] < 2.0 * edge(s_O)
    s_I = lminA / edge(1.0) / 1.5                         # ceiling 1.5 < 2 -> escalate
    sol = p1.solve_and_correct(G, c, d_R, d_O, s_I, rho_target=2.0)
    assert sol.Psi_mode == "I" and sol.rho >= 2.0 and abs(sol.rho - 2.0) < 1e-6


def test_bias_operator_matches_exact_second_moment():
    rng = np.random.default_rng(1)
    d_R, d_O = 3, 4
    p = d_R + d_O
    basis = []
    for i in range(p):
        for j in range(i, p):
            if i < d_R and j < d_R:
                continue
            S = np.zeros((p, p)); S[i, j] = S[j, i] = 1.0
            basis.append(S)
    X = rng.normal(size=(200, p))
    P = np.linalg.inv(X.T @ X)
    exact = sum(P @ S @ P @ S for S in basis)
    assert np.abs(exact - p1.M(P, d_R)).max() < 1e-18


def test_noise_structure():
    rng = np.random.default_rng(3)
    nz = p1.draw_noise(2, 3, 1.0, rng)
    assert np.allclose(nz.E_B, nz.E_B.T)
    acc = np.zeros((3, 3))
    for _ in range(4000):
        acc += p1.draw_noise(2, 3, 2.0, rng).E_B ** 2
    assert np.allclose(acc / 4000, 4.0, rtol=0.1)          # diagonal too: single draw, not GOE


def test_plaintext_ckks_inner_product():
    own = hb.ckks_owner(2, 50, 16, "plaintext")
    pub = hb.ckks_from_public(own.public_blob(), "plaintext")
    b = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 1.0])
    w = np.arange(10.0)
    ct = pub.add_scalar(pub.inner_product(pub.ct_from_bytes(own.ct_bytes(own.encrypt(b))), w), 0.5)
    assert own.decrypt_slot0(ct) == pytest.approx(float(b @ w) + 0.5)
