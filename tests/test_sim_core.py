"""Checks of experiments/sim_core.py against known values and the user's modules."""
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "experiments"))
sys.path.insert(0, os.path.join(HERE, "..", "scenario_b"))

import sim_core as sc  # noqa: E402
import phase1_common as p1  # noqa: E402
from calibrate_hyperparameters import analytic_gaussian_sigma  # noqa: E402


@pytest.mark.parametrize("b, expected", [((1, 1, 1), np.sqrt(10)),
                                         ((3.16, 3, 3), 29.3889),
                                         ((np.sqrt(5), np.sqrt(5), 5), np.sqrt(1000)),
                                         ((5, 1, 1), np.sqrt(200))])
def test_delta_rep_known_values(b, expected):
    assert sc.delta_rep(*b) == pytest.approx(expected, rel=1e-4)


def test_delta_rep_bounds():
    rng = np.random.default_rng(0)
    for _ in range(50):
        b = rng.uniform(0.2, 6, 3)
        d, a = sc.delta_rep(*b), sc.delta_add(*b)
        assert a * (1 - 1e-9) <= d <= 2 * a * (1 + 1e-9)


@pytest.mark.parametrize("eps", [0.25, 1.0, 4.0])
def test_analytic_sigma_matches_calibration_script(eps):
    assert sc.analytic_gauss_sigma(30.0, eps, 1e-5) == pytest.approx(
        analytic_gaussian_sigma(30.0, eps, 1e-5), rel=1e-6)


def test_release_noise_structure():
    rng = np.random.default_rng(1)
    p, dR, sig = 6, 2, 3.0
    Gt, ct, _ = sc.release(np.zeros((p, p)), np.zeros(p), 0.0, dR, sig, 40000, rng)
    assert np.allclose(Gt, np.swapaxes(Gt, 1, 2))
    assert np.all(Gt[:, :dR, :dR] == 0)
    var = Gt.var(0)
    chi = sc.mask(p, dR)
    assert np.allclose(var[chi == 1], sig**2, rtol=0.05)          # diagonal included
    assert ct.var(0) == pytest.approx(np.full(p, sig**2), rel=0.05)


def test_bias_op_matches_user_module():
    rng = np.random.default_rng(2)
    X = rng.standard_normal((50, 7)); P = np.linalg.inv(X.T @ X)
    beta = rng.standard_normal(7)
    ref = p1._bias_operator(P, 3, 4)(beta)
    assert np.allclose(sc.bias_op(P, 3) @ beta, ref, atol=1e-14)


def test_mech_cov_matches_monte_carlo():
    rng = np.random.default_rng(3)
    p, dR, sig = 5, 2, 1.0
    beta = rng.standard_normal(p)
    Gt, _, _ = sc.release(np.zeros((p, p)), np.zeros(p), 0.0, dR, sig, 200000, rng)
    Eb = np.einsum("rij,j->ri", Gt, beta)
    emp = np.cov(Eb, rowvar=False)
    th = sc.mech_cov_Ebeta(beta[None], sig, dR)[0]
    assert np.allclose(emp, th, atol=0.05 * np.abs(th).max())


def test_statistics_match_protocol_aggregation():
    pop = sc.Population(seed=4, pop_size=50_000)
    Z, y, _ = pop.draw(300, np.random.default_rng(5))
    dR = pop.d_R_block
    XdotR, XO = Z[:, :dR], Z[:, dR:]
    b = np.ones(len(y))
    B, C, cO, cR, yty = p1.aggregate_O(b, XdotR, XO, y)
    G_ref, c_ref = p1.assemble(p1.local_A(XdotR), B, C, cR, cO)
    G, c, yty2 = sc.stats(Z, y)
    assert np.allclose(G, G_ref) and np.allclose(c, c_ref) and yty == pytest.approx(yty2)


def test_tiny_sigma_recovers_ols():
    pop = sc.Population(seed=6, pop_size=50_000)
    Z, y, _ = pop.draw(2000, np.random.default_rng(7))
    G, c, yty = sc.stats(Z, y)
    Gt, ct, _ = sc.release(G, c, yty, pop.d_R_block, 1e-9, 3, np.random.default_rng(8))
    b, bc, _ = sc.solve(Gt, ct, 0.0, 1e-9, pop.d_R_block)
    ols = np.linalg.solve(G, c)
    assert np.allclose(b, ols, atol=1e-8) and np.allclose(bc, ols, atol=1e-8)


def test_population_targets():
    pop = sc.Population(seed=9, r2=0.5, pop_size=200_000)
    Z, y, clip = pop.draw(100_000, np.random.default_rng(10))
    ols = np.linalg.lstsq(Z, y, rcond=None)[0]
    r2 = 1 - np.var(y - Z @ ols) / np.var(y)
    assert r2 == pytest.approx(0.5, abs=0.03)
    assert np.abs(ols[1:] - pop.beta_std).max() < 0.02
    assert max(clip.values()) < 0.02
