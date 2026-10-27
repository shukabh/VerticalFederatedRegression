"""Check item #9: the bias-aware interval with the public variance bound sigma_bar^2 (Agreement.sigma_bar2).

The interval reported as guaranteed uses V = Vmech + sigma_bar^2 P~ G2 P~ (proofs Part B, Proposition S10b); the
one with the data-driven s~^2 is still returned, labelled as valid under the working model only.  These tests need
no native module."""
import math
from dataclasses import asdict

import numpy as np
import pytest
from scipy.stats import norm

from vfr import dp, solve

BASE = dict(n_R_max=10, n_bar_O=20, d_R=2, d_O=2, B_R=2.0, B_O=1.0, B_y=1.5)


# ------------------------------------------------------------------------------------------- the agreement
def test_sigma_bar2_default_is_B_y_squared():
    ag = dp.Agreement(**BASE)
    assert ag.sigma_bar2 is None
    assert dp.sigma_bar2_of(ag) == 1.5 ** 2
    assert 'sigma_bar2' in asdict(ag)                  # sent to R in the public parameters, checked there


def test_sigma_bar2_declared_value_is_kept():
    ag = dp.Agreement(**BASE, sigma_bar2=0.5)
    assert dp.sigma_bar2_of(ag) == 0.5
    assert dp.sigma_bar2_of(dp.Agreement(**BASE, sigma_bar2=2.25)) == 2.25      # exactly B_y^2 is allowed
    dp.alpha_split(ag)                                  # a well-formed setting


@pytest.mark.parametrize('bad', [0.0, -1.0, float('nan'), float('inf'), 2.26, 10.0])
def test_sigma_bar2_refused(bad):
    ag = dp.Agreement(**BASE, sigma_bar2=bad)
    with pytest.raises(ValueError, match='sigma_bar2'):
        dp.sigma_bar2_of(ag)
    with pytest.raises(ValueError, match='sigma_bar2'):
        dp.alpha_split(ag)                              # both parties call it in Setup


# ------------------------------------------------------------------------------------------- stage2_infer
def _release(seed, n=2000, p=6, dA=3, sigma2=2.0, lam=0.0):
    rng = np.random.default_rng(seed)
    X = np.column_stack([np.ones(n), rng.standard_normal((n, p - 1))])
    beta = np.linspace(-0.5, 0.5, p)
    y = np.clip(X @ beta + 0.7 * rng.standard_normal(n), -1.5, 1.5)
    G, c, yy = X.T @ X, X.T @ y, float(y @ y)
    E = solve.masked_noise(rng, 1, p, dA, sigma2)[0]
    G2, c2, yy2 = G + E, c + sigma2 * rng.standard_normal(p), yy + sigma2 * rng.standard_normal()
    lmin = float(np.linalg.eigvalsh(G)[0])
    s1 = dict(lam=lam, Psi=np.eye(p), L_hat=0.9 * lmin)
    return G2, c2, yy2, n, dA, s1, sigma2


@pytest.mark.parametrize('lam', [0.0, 50.0])
def test_both_bias_aware_intervals_and_unchanged_draws(lam):
    G2, c2, yy2, n, dA, s1, s2 = _release(1, lam=lam)
    args = (G2, c2, yy2, n, dA, s1, s2, s2, 1.5, 0.05, 0.0025, 0.0025, np.inf)
    old = solve.stage2_infer(*args, np.random.default_rng(9), Bboot=300)
    new = solve.stage2_infer(*args, np.random.default_rng(9), Bboot=300, sigma_bar2=0.8)
    # the random-number consumption, the estimate, the Wald interval and the working-model interval are unchanged
    for k in ('theta', 'Vmech', 'half_theta', 'V', 'half_beta'):
        assert np.array_equal(old[k], new[k]), k
    assert np.array_equal(new['half_beta'], new['half_beta_model'])
    assert np.array_equal(new['V'], new['V_model'])
    # default sigma_bar^2 = B_y^2
    assert old['sigma_bar2'] == 1.5 ** 2 and new['sigma_bar2'] == 0.8
    PGP = new['P2'] @ G2 @ new['P2']
    assert np.allclose(new['V_guaranteed'], new['Vmech'] + 0.8 * PGP, rtol=1e-12, atol=0)
    assert np.allclose(new['V_guaranteed'] - new['V_model'], (0.8 - new['s2']) * PGP, rtol=1e-9, atol=1e-15)
    # sigma_bar^2 above s~^2 (here s~^2 is about 0.45): the guaranteed interval is wider, coordinate by coordinate
    assert new['s2'] < 0.8
    assert np.all(new['half_beta_guaranteed'] > new['half_beta_model'])
    assert np.all(old['half_beta_guaranteed'] > new['half_beta_guaranteed'])
    # with sigma_bar^2 = s~^2 the two intervals coincide
    same = solve.stage2_infer(*args, np.random.default_rng(9), Bboot=300, sigma_bar2=new['s2'])
    assert np.allclose(same['half_beta_guaranteed'], same['half_beta_model'], rtol=1e-12)


def test_unbounded_bias_stays_unbounded():
    G2, c2, yy2, n, dA, s1, s2 = _release(2, lam=50.0)
    s1 = dict(s1, L_hat=0.0)                            # no private bound on ||beta||, no public one
    r = solve.stage2_infer(G2, c2, yy2, n, dA, s1, s2, s2, 1.5, 0.05, 0.0025, 0.0025, np.inf,
                           np.random.default_rng(0), Bboot=200)
    assert np.all(np.isinf(r['half_beta_guaranteed'])) and np.all(np.isinf(r['half_beta_model']))
    assert np.all(np.isfinite(r['half_theta']))


def test_sigma_bar2_must_be_positive_in_solve():
    G2, c2, yy2, n, dA, s1, s2 = _release(3)
    with pytest.raises(ValueError, match='sigma_bar2'):
        solve.stage2_infer(G2, c2, yy2, n, dA, s1, s2, s2, 1.5, 0.05, 0.0025, 0.0025, np.inf,
                           np.random.default_rng(0), Bboot=100, sigma_bar2=0.0)


def test_no_intervals_setting_ignores_sigma_bar2():
    G2, c2, yy2, n, dA, s1, s2 = _release(4)
    r = solve.stage2_infer(G2, c2, yy2, n, dA, s1, s2, s2, 1.5, 0.05, 0.75, 0.0025, np.inf,
                           np.random.default_rng(0), intervals=False, sigma_bar2=0.3)
    assert r['intervals'] is False and 'half_beta_guaranteed' not in r


# ------------------------------------------------------------------------------------------- coverage under binding clipping
def _clip_mean_var(m, s, B):
    """Mean and variance of clip(m + s Z, -B, B), Z ~ N(0, 1)."""
    a, b = (-B - m) / s, (B - m) / s
    Pa, Pb, pa, pb = norm.cdf(a), norm.cdf(b), norm.pdf(a), norm.pdf(b)
    mid = Pb - Pa
    e1 = m * mid + s * (pa - pb)                                         # E[Y 1{a<Z<b}]
    e2 = (m * m + s * s) * mid + 2 * m * s * (pa - pb) + s * s * (a * pa - b * pb)
    mean = e1 - B * Pa + B * (1 - Pb)
    second = e2 + B * B * (Pa + 1 - Pb)
    return mean, second - mean ** 2


def test_clip_moments_formula():
    rng = np.random.default_rng(0)
    for m, s, B in ((0.3, 1.0, 1.0), (-1.2, 0.5, 1.0), (2.0, 2.0, 1.5)):
        y = np.clip(m + s * rng.standard_normal(2_000_000), -B, B)
        mu, v = _clip_mean_var(m, s, B)
        assert abs(y.mean() - mu) < 3e-3 and abs(y.var() - v) < 3e-3


def test_guaranteed_interval_covers_projection_coefficient_when_clipping_binds():
    """Fixed design; the error s.d. grows with |x_1| (high-leverage rows are noisier), y clipped at B_y (binding for
    about 14% of rows); lambda = 0; small mechanism noise, so that the sampling variance dominates.  The interval
    with sigma_bar^2 = B_y^2 covers the projection coefficient beta* = G^-1 X' E[y~ | X]; the working-model
    interval under-covers the slope on x_1 (about 0.82 expected from the sandwich variance)."""
    rng = np.random.default_rng(11)
    n, p, dA, B, sig2 = 600, 3, 2, 1.5, 0.05
    X = np.column_stack([np.ones(n), rng.standard_normal(n), rng.standard_normal(n)])
    m = X @ np.array([0.1, 0.4, -0.3])
    s = 0.1 + 0.9 * np.abs(X[:, 1]) ** 1.5
    mu, var = _clip_mean_var(m, s, B)
    G = X.T @ X
    bstar = np.linalg.solve(G, X.T @ mu)
    assert var.max() <= B * B
    reps, cov_g, cov_m = 600, np.zeros(p), np.zeros(p)
    s1 = dict(lam=0.0, Psi=np.eye(p), L_hat=0.5 * float(np.linalg.eigvalsh(G)[0]))
    for r in range(reps):
        y = np.clip(m + s * rng.standard_normal(n), -B, B)
        c, yy = X.T @ y, float(y @ y)
        G2 = G + solve.masked_noise(rng, 1, p, dA, sig2)[0]
        c2, yy2 = c + sig2 * rng.standard_normal(p), yy + sig2 * rng.standard_normal()
        res = solve.stage2_infer(G2, c2, yy2, n, dA, s1, sig2, sig2, B, 0.05, 0.0025, 0.0025, np.inf, rng,
                                 Bboot=100)
        err = np.abs(res['theta'] - bstar)
        cov_g += err <= res['half_beta_guaranteed']
        cov_m += err <= res['half_beta_model']
    cov_g, cov_m = cov_g / reps, cov_m / reps
    assert np.all(cov_g >= 0.95)                        # conservative: B_y^2 is far above every variance here
    assert cov_m[1] < 0.90                              # the s~^2 interval is not guaranteed under binding clipping
