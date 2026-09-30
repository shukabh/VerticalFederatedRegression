"""
sim_core.py -- shared simulation core for the revised Scenario-B protocol (paper/main.tex).

The HE and PSI layers reproduce plaintext aggregation to ~1e-10 (reports/accuracy), so the
accuracy experiments work directly on the sufficient statistics:

    G = Z^T Z,  c = Z^T y,  Z = [ 1 | Z_R | Z_O ]   (R's block includes the intercept)

and apply the revised mechanism exactly as specified:
  * replace-one sensitivity Delta_rep (closed form, Proposition 5.1) and analytic-Gaussian sigma
  * single-draw symmetric Gaussian noise on the O-dependent blocks (A exact), same sigma for all
  * ridge fixed from public inputs, lambda = max{0, 2 rho* sigma sqrt(p) - n ell}, Psi = I
  * zero-budget bias correction beta_bc = beta - sigma^2 M(P~) beta
  * first-order variance (sampling + mechanism) for Wald intervals
The original adaptive gate (phase1_common.select_ridge, the user's module) is exposed as a
baseline.

Everything is batched over noise draws: arrays of shape (R, p, p) / (R, p).
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import log_ndtr, ndtr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "scenario_b"))

# =============================================================================
# Sensitivity and calibration
# =============================================================================

def delta_add(B_R, B_O, B_y):
    """Add/remove bound, eq. (6) of the draft (what the original code used)."""
    return float(np.sqrt((B_O**2 + B_R**2) * (B_O**2 + B_y**2) + B_y**4))


def delta_rep(B_R, B_O, B_y):
    """Replace-one sensitivity of (triu B, C, c_R, c_O, y'y), x_R fixed (Proposition 5.1).

    Regime 3 needs a 1-D maximisation; it is done on a grid, refined locally, and rounded up
    so that sigma is never calibrated to an underestimate."""
    R, O, Y = B_R**2, B_O**2, B_y**2
    if abs(R - Y) <= 2 * O:
        return float(np.sqrt(0.5 * (2 * O + R + Y) ** 2 + 2 * R * Y))
    if R - Y >= 2 * O:
        return float(np.sqrt(4 * R * (O + Y)))

    def F(t):
        c = np.clip(-(R + B_y * t) / (2 * O), -1.0, 1.0)
        return (2 * O * O * (1 - c * c) + 2 * R * O * (1 - c) + R * (B_y - t) ** 2
                + O * (Y + t * t) - 2 * O * c * B_y * t + (Y - t * t) ** 2)

    ts = np.linspace(-B_y, B_y, 4001)
    vals = F(ts)
    k = int(np.argmax(vals))
    lo, hi = ts[max(k - 1, 0)], ts[min(k + 1, len(ts) - 1)]
    res = minimize_scalar(lambda t: -F(t), bounds=(lo, hi), method="bounded",
                          options={"xatol": 1e-12})
    best = max(vals[k], -res.fun, F(-B_y), F(B_y))
    return float(np.sqrt(best) * (1 + 1e-9))


def analytic_gauss_sigma(Delta, eps, delta, tol=1e-12):
    """Balle-Wang analytic Gaussian mechanism: smallest sigma with
    Phi(D/2s - e s/D) - e^eps Phi(-D/2s - e s/D) <= delta (log-space tail)."""
    def gap(s):
        a, b = Delta / (2 * s), eps * s / Delta
        return ndtr(a - b) - np.exp(eps + log_ndtr(-a - b)) - delta
    hi = max(Delta / eps, 1e-12)
    while gap(hi) > 0:
        hi *= 2
    lo = hi / 2
    while gap(lo) <= 0:
        hi, lo = lo, lo / 2
    while hi - lo > tol * hi:
        mid = 0.5 * (lo + hi)
        if gap(mid) <= 0:
            hi = mid
        else:
            lo = mid
    return float(hi)


# =============================================================================
# Realistic synthetic population
# =============================================================================

def _toeplitz(k, r):
    idx = np.arange(k)
    return r ** np.abs(idx[:, None] - idx[None, :])


FEATURE_TYPES_R = ["gaussian", "binary", "lognormal", "count"]
FEATURE_TYPES_O = ["lognormal", "age", "binary", "count", "gaussian", "lognormal"]
# standardized-effect pattern (rescaled to hit the target R^2); mixed magnitudes and signs
BETA_PATTERN = np.array([0.50, -0.30, 0.20, 0.10,          # R features
                         0.40, -0.25, 0.15, 0.05, -0.10, 0.30])  # O features


def _transform(u, kind):
    """Map a standard-normal latent column to a feature on a realistic scale."""
    if kind == "gaussian":
        return 10.0 + 3.0 * u                         # e.g. a test score
    if kind == "binary":
        return (u > 0.5).astype(float)                # ~31% prevalence
    if kind == "lognormal":
        return np.exp(10.5 + 0.6 * u)                 # income-like, dollars
    if kind == "count":
        return np.floor(np.exp(0.8 + 0.5 * u))        # visits / children
    if kind == "age":
        return 18.0 + 62.0 * ndtr(u)                  # 18..80
    raise ValueError(kind)


@dataclass
class Population:
    """A known population, so that 'true beta', public standardization constants and committed
    bounds are all data-independent (as the protocol requires)."""
    d_R: int = 4
    d_O: int = 6
    r2: float = 0.5
    corr: float = 0.3
    quantile: float = 0.995        # committed bounds cover this fraction of the population
    seed: int = 0
    pop_size: int = 400_000
    y_mean: float = 50_000.0       # outcome on an income-like scale
    y_sd: float = 20_000.0
    types_R: list = field(default_factory=lambda: list(FEATURE_TYPES_R))
    types_O: list = field(default_factory=lambda: list(FEATURE_TYPES_O))

    def __post_init__(self):
        assert len(self.types_R) == self.d_R and len(self.types_O) == self.d_O
        self.types = self.types_R + self.types_O
        self.k = self.d_R + self.d_O
        self.L = np.linalg.cholesky(_toeplitz(self.k, self.corr))
        rng = np.random.default_rng(self.seed)
        X = self._features(self.pop_size, rng)
        # public standardization constants (population reference values)
        self.mu = X.mean(0)
        self.s = X.std(0)
        Zs = (X - self.mu) / self.s
        Rz = np.corrcoef(Zs, rowvar=False)
        pattern = BETA_PATTERN[: self.k]
        scale = np.sqrt(self.r2 / (pattern @ Rz @ pattern))
        self.beta_std = pattern * scale                     # true standardized slopes
        self.noise_sd = np.sqrt(1.0 - self.r2)
        ystd = Zs @ self.beta_std + self.noise_sd * rng.standard_normal(self.pop_size)
        self.mu_ystd, self.s_ystd = float(ystd.mean()), float(ystd.std())
        # raw-scale coefficients (for the "different scales" view)
        self.beta_raw = self.y_sd * self.beta_std / self.s
        self.intercept_raw = self.y_mean - float(self.beta_raw @ self.mu)
        # committed bounds on the standardized scale
        zR, zO = Zs[:, : self.d_R], Zs[:, self.d_R:]
        self.B_R_rows = float(np.quantile(np.linalg.norm(zR, axis=1), self.quantile))
        self.B_R = float(np.sqrt(1.0 + self.B_R_rows**2))   # intercept column included
        self.B_O = float(np.quantile(np.linalg.norm(zO, axis=1), self.quantile))
        self.B_y = float(np.quantile(np.abs(ystd), self.quantile))
        # population second-moment matrix of [1, z] and its smallest eigenvalue (for ell)
        Zfull = np.hstack([np.ones((self.pop_size, 1)), Zs])
        self.Sigma_pop = Zfull.T @ Zfull / self.pop_size
        self.lam_min_pop = float(np.linalg.eigvalsh(self.Sigma_pop)[0])
        self.d_R_block = self.d_R + 1                       # R's block incl. intercept
        self.p = self.k + 1

    def _features(self, n, rng):
        U = rng.standard_normal((n, self.k)) @ self.L.T
        return np.column_stack([_transform(U[:, j], t) for j, t in enumerate(self.types)])

    def draw(self, n, rng):
        """One matched cohort of size n: standardized with public constants, clipped to the
        committed bounds. Returns Z (n x p, intercept first), y (n,), clip rates."""
        X = self._features(n, rng)
        Zs = (X - self.mu) / self.s
        y = Zs @ self.beta_std + self.noise_sd * rng.standard_normal(n)
        zR, zO = Zs[:, : self.d_R], Zs[:, self.d_R:]
        nR, nO = np.linalg.norm(zR, axis=1), np.linalg.norm(zO, axis=1)
        hitR, hitO, hity = nR > self.B_R_rows, nO > self.B_O, np.abs(y) > self.B_y
        zR = zR * np.minimum(1.0, self.B_R_rows / np.maximum(nR, 1e-300))[:, None]
        zO = zO * np.minimum(1.0, self.B_O / np.maximum(nO, 1e-300))[:, None]
        y = np.clip(y, -self.B_y, self.B_y)
        Z = np.hstack([np.ones((n, 1)), zR, zO])
        return Z, y, {"X_R": hitR.mean(), "X_O": hitO.mean(), "y": hity.mean()}

    @property
    def beta_true(self):
        """True coefficient vector on the standardized scale, intercept first (~0)."""
        return np.concatenate([[0.0], self.beta_std])

    def sigma_for(self, eps, delta=1e-5, sensitivity="rep"):
        D = (delta_rep if sensitivity == "rep" else delta_add)(self.B_R, self.B_O, self.B_y)
        return analytic_gauss_sigma(D, eps, delta), D


# =============================================================================
# Mechanism
# =============================================================================

def stats(Z, y):
    return Z.T @ Z, Z.T @ y, float(y @ y)


def mask(p, d_R):
    """chi_ij = 0 iff both i, j in R's block."""
    chi = np.ones((p, p))
    chi[:d_R, :d_R] = 0.0
    return chi


def release(G, c, yty, d_R, sigma, R, rng):
    """R independent releases of the O-dependent statistics (single-draw symmetric noise,
    A exact). Returns Gt (R,p,p), ct (R,p), ytyt (R,)."""
    p = G.shape[0]
    M = rng.standard_normal((R, p, p)) * sigma
    E = np.triu(M) + np.swapaxes(np.triu(M, 1), 1, 2)     # symmetric, every entry var sigma^2
    E[:, :d_R, :d_R] = 0.0
    f = rng.standard_normal((R, p)) * sigma
    ytyt = yty + sigma * rng.standard_normal(R)
    return G[None] + E, c[None] + f, ytyt


def fixed_lambda(sigma, p, n, ell, rho_star=2.0):
    return max(0.0, 2.0 * rho_star * sigma * np.sqrt(p) - n * ell)


def bias_op(P, d_R):
    """Batched M(P) of Theorem 1 (P: (R,p,p) or (p,p))."""
    P = np.asarray(P)
    single = P.ndim == 2
    if single:
        P = P[None]
    p = P.shape[-1]
    PiO = np.zeros((p, p)); PiO[d_R:, d_R:] = np.eye(p - d_R)
    PiR = np.eye(p) - PiO
    trP = np.trace(P, axis1=1, axis2=2)[:, None, None]
    trOP = np.trace(PiO[None] @ P, axis1=1, axis2=2)[:, None, None]
    dP = np.einsum("rii->ri", P)
    PdP = P * dP[:, None, :]                                # P diag(P)
    M = (P @ P + trP * P - PdP) @ PiO + (P @ PiO @ P + trOP * P) @ PiR
    return M[0] if single else M


def solve(Gt, ct, lam, sigma, d_R, Psi=None):
    """Ridge solve and bias correction for a batch. Returns beta_ridge, beta_bc, P."""
    Rn, p, _ = Gt.shape
    if Psi is None:
        Psi = np.eye(p)
    lam = np.broadcast_to(np.asarray(lam, float), (Rn,))
    Gl = Gt + lam[:, None, None] * Psi[None]
    P = np.linalg.inv(Gl)
    b = np.einsum("rij,rj->ri", P, ct)
    bc = b - sigma**2 * np.einsum("rij,rj->ri", bias_op(P, d_R), b)
    return b, bc, P


def rho_hat(Gt, lam, sigma, Psi=None):
    Rn, p, _ = Gt.shape
    if Psi is None:
        Psi = np.eye(p)
    lam = np.broadcast_to(np.asarray(lam, float), (Rn,))
    lm = np.linalg.eigvalsh(Gt + lam[:, None, None] * Psi[None])[:, 0]
    return lm / (2 * sigma * np.sqrt(p))


def mech_cov_Ebeta(beta, sigma, d_R):
    """Cov(E beta) under the single-draw masked mechanism, batched over beta (R,p)."""
    p = beta.shape[-1]
    chi = mask(p, d_R)
    b2 = beta**2
    term1 = np.einsum("ij,rj->ri", chi, b2)                    # diag entries
    K = chi[None] * beta[:, :, None] * beta[:, None, :]
    K[:, np.arange(p), np.arange(p)] += term1 - np.diag(chi)[None] * b2
    return sigma**2 * K


def first_order_cov(Gt, ct, ytyt, beta, P, sigma, n, d_R):
    """Plug-in Var(beta~) ~= s^2 P G P + P (sigma^2 I + Cov(E beta)) P  (sampling + mechanism).
    s^2 from RSS~ = y'y~ - 2 c~'b + b'G~b, floored at a small positive value."""
    Rn, p, _ = Gt.shape
    rss = ytyt - 2 * np.einsum("ri,ri->r", ct, beta) + np.einsum("ri,rij,rj->r", beta, Gt, beta)
    s2 = np.maximum(rss, 1e-6 * n) / max(n - p, 1)
    samp = s2[:, None, None] * (P @ Gt @ P)
    mech = P @ (sigma**2 * np.eye(p)[None] + mech_cov_Ebeta(beta, sigma, d_R)) @ P
    return samp + mech, s2


def adaptive_gate(Gt, ct, sigma, d_R, d_O, rho_target=2.0):
    """The user's original gate (phase1_common.solve_and_correct, mode='auto'), per draw."""
    import phase1_common as p1
    out_b, out_bc, lams = [], [], []
    for G1, c1 in zip(Gt, ct):
        sol = p1.solve_and_correct(G1, c1, d_R, d_O, sigma, mode="auto", rho_target=rho_target)
        out_b.append(sol.beta_ridge); out_bc.append(sol.beta_bc); lams.append(sol.lam)
    return np.array(out_b), np.array(out_bc), np.array(lams)
