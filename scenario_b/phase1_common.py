"""
phase1_common.py  —  Phase 1 shared math (pure numpy, NO TenSEAL import)

The entire non-crypto pipeline lives here so the TenSEAL parties and the
plaintext validator run identical algebra:

    aggregate  ->  sensitivity  ->  DP noise (O-blocks only)  ->  assemble
    (clean A)  ->  O-block ridge  ->  rho validity gate  ->  solve  ->  bias-correct

Scenario B (O holds y). The block A = XdotR^T XdotR is R's exact, un-noised,
locally-held Gram; only the O-dependent blocks (B, C, c_R, c_O, y^T y) are
noised. See vfl_ols_scenarioB.tex for the theory (Theorem 1/2, Remark on rho).
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


# ==============================================================================
# Aggregation (plaintext reference == what the HE path computes)
# ==============================================================================

def local_A(XdotR):
    """A = XdotR^T XdotR.  Equals X_R_I^T X_R_I because XdotR is zero off the
    matched rows.  Computed by R locally, in the clear; never noised."""
    return XdotR.T @ XdotR


def aggregate_O(b, XdotR, X_O, y_O):
    """The O-side homomorphic aggregates, as plaintext reference.
        B   = sum_i b_i x_O x_O^T            (b-masked self-Gram)
        C   = sum_i xdot_i x_O^T             (cross-Gram)
        c_O = sum_i b_i x_O y_i              (b-masked)
        c_R = sum_i xdot_i y_i               (masked via xdot zeros)
        yty = sum_i b_i y_i^2
    """
    B   = X_O.T @ (b[:, None] * X_O)
    C   = XdotR.T @ X_O
    c_O = X_O.T @ (b * y_O)
    c_R = XdotR.T @ y_O
    yty = float(np.sum(b * y_O ** 2))
    return B, C, c_O, c_R, yty


# ==============================================================================
# DP: sensitivity, sigma, symmetric noise on the O-dependent blocks
# ==============================================================================

def joint_sensitivity_B(B_R, B_O, B_y):
    """Scenario-B joint L2 sensitivity of (B, C, c_O, c_R, y^T y):
        Delta_2^2 = (B_O^2 + B_R^2)(B_O^2 + B_y^2) + B_y^4 .
    (No B_R^4 term: block A is un-noised.)"""
    return float(np.sqrt((B_O ** 2 + B_R ** 2) * (B_O ** 2 + B_y ** 2) + B_y ** 4))


def sigma_gaussian(Delta2, eps, delta):
    """Standard Gaussian-mechanism sigma (Balle-Wang analytic calibration is
    tighter; this is the classical bound)."""
    return float(Delta2 * np.sqrt(2.0 * np.log(1.25 / delta)) / eps)


@dataclass
class Noise:
    E_B: np.ndarray      # (dO,dO) symmetric
    E_C: np.ndarray      # (dR,dO)
    f_R: np.ndarray      # (dR,)
    f_O: np.ndarray      # (dO,)
    e_yty: float


def draw_noise(dR, dO, sigma, rng):
    """Symmetric single-draw mechanism: upper triangle of B iid N(0,sigma^2),
    mirrored; C, c_R, c_O, y^T y iid N(0,sigma^2). Block A gets none."""
    M = rng.normal(0.0, sigma, size=(dO, dO))
    E_B = np.triu(M) + np.triu(M, 1).T
    E_C = rng.normal(0.0, sigma, size=(dR, dO))
    f_R = rng.normal(0.0, sigma, size=dR)
    f_O = rng.normal(0.0, sigma, size=dO)
    e_yty = float(rng.normal(0.0, sigma))
    return Noise(E_B, E_C, f_R, f_O, e_yty)


# ==============================================================================
# Assembly (R splices in the clean local A)
# ==============================================================================

def assemble(A, B, C, c_R, c_O):
    """Return (G, c) with G = [[A, C],[C^T, B]], c = [c_R; c_O]."""
    G = np.block([[A, C], [C.T, B]])
    c = np.concatenate([c_R, c_O])
    return G, c


def apply_noise(A, B, C, c_R, c_O, yty, nz: Noise):
    """Add DP noise to the O-dependent blocks (A stays clean)."""
    return (A,
            B + nz.E_B,
            C + nz.E_C,
            c_R + nz.f_R,
            c_O + nz.f_O,
            yty + nz.e_yty)


# ==============================================================================
# Ridge selector, validity gate, solve, bias corrector
# ==============================================================================

def ridge_selector(dR, dO, mode="O"):
    """Psi in {I, Pi_O}. 'O' ridges only the noised O-block (recommended)."""
    p = dR + dO
    if mode == "I":
        return np.eye(p)
    Psi = np.zeros((p, p)); Psi[dR:, dR:] = np.eye(dO)   # Pi_O
    return Psi


def rho_lambda(G, lam, Psi, sigma, p):
    """rho_lambda = lambda_min(G + lam*Psi) / (2 sigma sqrt p).  The leading-order
    bias expansion is valid w.h.p. once rho_lambda > 1 (Remark, paper)."""
    lm = np.linalg.eigvalsh(G + lam * Psi)[0]
    return float(lm / (2.0 * sigma * np.sqrt(p))) if sigma > 0 else np.inf


def auto_lambda(G, Psi, sigma, p, target=2.0, lam0=1e-3, grow=1.5, cap=1e12):
    """Smallest lambda (geometric search) giving rho_lambda >= target."""
    if sigma <= 0:
        return 0.0
    lam = lam0
    while lam < cap:
        if rho_lambda(G, lam, Psi, sigma, p) >= target:
            return lam
        lam *= grow
    return cap


def select_ridge(G, dR, dO, sigma, target=2.0):
    """Choose the ridge selector and lambda that clear rho >= target.

    O-block-only ridge (Psi = Pi_O) leaves A un-shrunk but has a VALIDITY
    CEILING: as lambda -> inf, lambda_min(G + lambda*Pi_O) -> lambda_min(A),
    so O-block ridge can reach rho_target only if lambda_min(A)/(2 sigma sqrt p)
    >= target. When the noise is too large for that (small cohort / small eps /
    raw moments), we escalate to FULL ridge (Psi = I), which lifts A too and can
    always reach the target, at the cost of shrinking R's coefficients.

    Returns (mode, lambda, rho_achieved).
    """
    p = dR + dO
    if sigma <= 0:
        return "O", 0.0, np.inf
    lamA = float(np.linalg.eigvalsh(G[:dR, :dR])[0])       # lambda_min(A)
    rho_ceiling_O = lamA / (2.0 * sigma * np.sqrt(p))
    if rho_ceiling_O >= target:
        Psi = ridge_selector(dR, dO, "O")
        lam = auto_lambda(G, Psi, sigma, p, target)
        return "O", lam, rho_lambda(G, lam, Psi, sigma, p)
    # escalate: O-block ridge cannot validate here
    Psi = ridge_selector(dR, dO, "I")
    lam = auto_lambda(G, Psi, sigma, p, target)
    return "I", lam, rho_lambda(G, lam, Psi, sigma, p)


def _bias_operator(Plam, dR, dO):
    """M(.) : beta -> M beta, leading-order bias operator (Theorem 1/2), for the
    SINGLE-DRAW symmetric Gaussian mechanism actually injected by O (each
    independent entry, diagonal INCLUDED, ~ N(0, sigma^2); see draw_noise).

    The GOE-convention contraction gives
        M_GOE beta = (P^2 + tr(P) P) Pi_O beta + (P Pi_O P + tr(Pi_O P) P) Pi_R beta,
    but GOE has diagonal variance 2 sigma^2. The deployed single-draw mechanism
    has diagonal variance sigma^2, so the true operator is
        M_single = M_GOE - P diag(P) Pi_O
    (the diagonal-overcount term -delta_ij delta_ik delta_ib in the single-draw
    covariance forces i=j=k=b, contributing -P_ab P_bb 1[b in O]).
    Scalar check (p=1, all-O): M_single beta = a^-2 beta -> bias sigma^2 a^-2 beta.
    """
    p = dR + dO
    PiO = np.zeros((p, p)); PiO[dR:, dR:] = np.eye(dO); PiR = np.eye(p) - PiO
    trP, trOP = np.trace(Plam), np.trace(PiO @ Plam)
    A1 = Plam @ Plam + trP * Plam
    A2 = Plam @ PiO @ Plam + trOP * Plam
    dP = np.diag(Plam)                                     # diagonal of P
    return lambda beta: (A1 @ (PiO @ beta) + A2 @ (PiR @ beta)
                         - Plam @ (dP * (PiO @ beta)))      # single-draw correction


@dataclass
class Solution:
    beta_ridge: np.ndarray       # private ridge estimate  (Theorem-2 target)
    beta_bc:    np.ndarray       # bias-corrected ridge estimate
    lam:        float
    rho:        float
    Psi_mode:   str


def solve_and_correct(Gt, ct, dR, dO, sigma, *, mode="O",
                      lam=None, rho_target=2.0):
    """R's endgame: pick/keep ridge, gate on rho, solve, bias-correct.

    Gt, ct : ASSEMBLED private statistics (clean A already spliced in).
    sigma  : the DP noise std used by O (public constant).
    lam    : fixed ridge; if None, auto-select to clear rho_target.
    Returns Solution.  Bias correction is post-processing (zero extra budget).
    """
    p = dR + dO
    if mode == "auto":
        mode, lam_sel, _ = select_ridge(Gt, dR, dO, sigma, target=rho_target)
        if lam is None:
            lam = lam_sel
    Psi = ridge_selector(dR, dO, mode)
    if lam is None:
        lam = auto_lambda(Gt, Psi, sigma, p, target=rho_target)
    Glam = Gt + lam * Psi
    Plam = np.linalg.inv(Glam)
    beta_ridge = Plam @ ct
    M = _bias_operator(Plam, dR, dO)
    beta_bc = beta_ridge - (sigma ** 2) * M(beta_ridge)     # unbiased thru O(sigma^4)
    rho = rho_lambda(Gt, lam, Psi, sigma, p)
    return Solution(beta_ridge, beta_bc, float(lam), float(rho), mode)


# ==============================================================================
# Oracle helpers (for validation only)
# ==============================================================================

def oracle_ols(X_R_I, X_O_I, y):
    Z = np.hstack([X_R_I, X_O_I])
    return np.linalg.solve(Z.T @ Z, Z.T @ y)


def oracle_ridge(G, c, dR, dO, lam, mode="O"):
    Psi = ridge_selector(dR, dO, mode)
    return np.linalg.solve(G + lam * Psi, c)


def shrinkage_bias(G, c, dR, dO, lam, mode="O"):
    """Deterministic ridge shrinkage relative to OLS: -lam P_lam Psi beta_ols."""
    Psi = ridge_selector(dR, dO, mode)
    Plam = np.linalg.inv(G + lam * Psi)
    beta_ols = np.linalg.solve(G, c)
    return -lam * Plam @ Psi @ beta_ols
