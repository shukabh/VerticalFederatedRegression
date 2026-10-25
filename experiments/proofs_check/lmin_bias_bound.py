"""A rigorous bound on the downward bias of lambda_min of the noisy stage-1 Gram, and the switch-off size it implies.

Lemma (proved in the stats proofs, Part B): for a fixed symmetric G and a masked symmetric Gaussian noise matrix E of scale
sigma (off-diagonal variances <= sigma^2, diagonal <= 2 sigma^2),
    E lambda_min(G + E) >= m(G, sigma) := E min_{|u|=1} (u'Gu - 2 sigma g'u),  g ~ N(0, I_p),
by the Sudakov-Fernique inequality for Gaussian processes with equal means; and lambda_min(G + E) is sqrt(2) sigma-Lipschitz
in the Gaussian coordinates, so P[lambda_min(G + E) < m - t] <= exp(-t^2 / (4 sigma^2)).  Hence the stage-1 rule switches
the ridge off with probability >= 1 - a' whenever  m(G, sigma_1) - 2 sigma_1 sqrt(ln(1/a')) >= g* + r_1.
m(G, sigma) is the mean of a trust-region subproblem: min_{|u|=1} u'Gu - 2h'u = theta - sum_k h_k^2/(mu_k - theta), with
theta < mu_1 the root of sum_k h_k^2/(mu_k - theta)^2 = 1 (h = sigma g in the eigenbasis of G; the 'hard case' has
probability 0).  It depends only on the spectrum of G and on sigma.
This script evaluates the switch-off bound for the simulation population (G = n Sigma) at settings I and P and compares it
with a direct simulation of the stage-1 rule.  Run from the repository root.
"""
import math, sys
import numpy as np
from scipy.optimize import brentq
sys.path.insert(0, 'experiments')
import sim_core as sc

def tr_min(mu, h):
    """min over the unit sphere of u'diag(mu)u - 2h'u."""
    f = lambda th: np.sum(h ** 2 / (mu - th) ** 2) - 1.0
    hi = mu[0] - 1e-14 * max(1.0, abs(mu[0]))
    lo = mu[0] - np.linalg.norm(h) - 1.0
    while f(lo) > 0: lo -= np.linalg.norm(h) + 1.0
    if f(hi) < 0:            # hard case (measure zero): take theta = mu_1
        th = mu[0]
    else:
        th = brentq(f, lo, hi, xtol=1e-12 * max(1, abs(lo)))
    return th - np.sum(h ** 2 / (mu - th))

def m_bound(mu, sigma, M, rng):
    g = rng.standard_normal((M, len(mu)))
    vals = np.array([tr_min(mu, sigma * gi) for gi in g])
    return vals.mean(), vals.std() / math.sqrt(M)

def masked_noise(p, dA, sigma, rng, size):
    E = np.zeros((size, p, p))
    iu = np.triu_indices(p)
    mask = np.array([not (i < dA and j < dA) for i, j in zip(*iu)])
    Z = rng.standard_normal((size, mask.sum())) * sigma
    for k, (i, j) in enumerate(zip(iu[0][mask], iu[1][mask])):
        E[:, i, j] = Z[:, k]; E[:, j, i] = Z[:, k]
    return E

pop = sc.Population(r2=0.5, seed=0)
rng = np.random.default_rng(7)
Z, y, _ = pop.draw(400_000, rng)          # clipped design: Sigma of the preprocessed, clipped population
Sigma = Z.T @ Z / Z.shape[0]
mu_S = np.linalg.eigvalsh(Sigma)
p, dA = Sigma.shape[0], pop.d_R_block
print('p', p, 'ell = lambda_min(Sigma) =', round(mu_S[0], 4), ' next eigenvalues', np.round(mu_S[:4], 3))
alpha_p = 0.05
for name, (sig1, sig2, theta, q) in {
        'I  (eps=1)': (892.0, 575.0, 2.5, 7.415),
        'P  (eps=1)': (892.0, 575.0, 1.5, 4.442)}.items():
    gstar = 2 * theta * sig2 * math.sqrt(p); r1 = q * sig1; target = gstar + r1
    noff = target / mu_S[0]
    print(f'\nsetting {name}: g*+r1 = {target:.0f}, n_off = {noff:.0f}, sigma_1/(g*+r1) = {sig1/target:.4f}')
    # bracket of prop:inf.noffbracket (upper end), exact quantile q'_{0.95} = 6.303
    print(f'   bracket upper end (worst case over spectra): {1 + 6.303 * sig1 / target:.3f} n_off')
    for fac in (1.0, 1.1, 1.2, 1.3, 1.4, 1.5):
        n = fac * noff
        mu = n * mu_S
        m, se = m_bound(mu, sig1, 4000, rng)
        guaranteed = m - 2 * sig1 * math.sqrt(math.log(1 / alpha_p)) >= target
        E = masked_noise(p, dA, sig1, rng, 20000)
        lmins = np.linalg.eigvalsh(n * Sigma[None] + E)[:, 0]
        off = np.mean(lmins >= target)
        print(f'   n = {fac:.1f} n_off: bias bound mu1 - m = {mu[0]-m:7.0f} (+-{se:.0f}) = {(mu[0]-m)/sig1:.2f} sigma_1;'
              f' simulated bias = {mu[0]-lmins.mean():6.0f};  off w.p. >= 0.95 guaranteed: {guaranteed};'
              f'  simulated share off = {off:.3f}')
