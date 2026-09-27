"""
phase1_common.py -- Phase-1 statistics, DP noise, and the solve/correct step.

##############################################################################
#  STAND-IN written from call sites in party_r.py/party_o.py; the user's      #
#  original module was not available. Replace with the original.             #
##############################################################################

Interfaces used by the party files
----------------------------------
  local_A(Xdot)                     -> Xdot.T @ Xdot   (R's clean block, never sent)
  assemble(A, B, C, c_R, c_O)       -> (G_tilde, c_tilde), R's block FIRST:
                                         G = [[A, C], [C.T, B]],  c = [c_R; c_O]
  draw_noise(d_R, d_O, sigma, rng)  -> NoiseDraw(E_B, E_C, f_R, f_O, e_yty)
        single-draw symmetric Gaussian (paper eq. (8)): every entry of E_B on or
        above the diagonal is iid N(0, sigma^2) and mirrored (so the diagonal
        also has variance sigma^2, NOT the GOE 2 sigma^2); E_C, f_R, f_O, e_yty iid.
  solve_and_correct(Gt, ct, d_R, d_O, sigma, mode="auto", rho_target=2.0)
        -> Solution(beta_ridge, beta_bc, lam, Psi_mode, rho, ...)

Ridge gate (paper Remark 4), mode="auto"
----------------------------------------
  rho(lambda) = lambda_min(Gt + lambda Psi) / (2 sigma sqrt(p)),  p = d_R + d_O.
  1. rho(0) >= rho_target                      -> lambda = 0,   Psi_mode = 'none'
  2. else if lambda_min(A)/(2 sigma sqrt p) > rho_target   (A = Gt[:d_R,:d_R], exact)
                                               -> Psi = Pi_O,   Psi_mode = 'O',
        smallest lambda with rho(lambda) >= rho_target (bisection; lambda_min of
        Gt + lambda Pi_O is nondecreasing in lambda and tends to lambda_min(A)).
  3. else                                      -> Psi = I,      Psi_mode = 'I',
        lambda = rho_target * 2 sigma sqrt(p) - lambda_min(Gt).
  Other modes (for experiments): "none" forces lambda = 0.

Estimators (paper Theorem 2 / eq. (10))
---------------------------------------
  beta_ridge = solve(Gt + lambda Psi, ct)
  beta_bc    = beta_ridge - sigma^2 M(inv(Gt + lambda Psi)) beta_ridge
  M(P) = (P^2 + tr(P) P - P diag(P)) Pi_O + (P Pi_O P + tr(Pi_O P) P) Pi_R
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def local_A(Xdot):
    """R's clean Gram block A = Xdot^T Xdot (zero rows off the match contribute 0)."""
    Xdot = np.asarray(Xdot, dtype=float)
    return Xdot.T @ Xdot


def assemble(A, B, C, c_R, c_O):
    """Splice R's clean A into the released blocks; R's coordinates first."""
    A = np.asarray(A, dtype=float); B = np.asarray(B, dtype=float)
    C = np.asarray(C, dtype=float)
    Gt = np.block([[A, C], [C.T, B]])
    ct = np.concatenate([np.asarray(c_R, dtype=float).ravel(),
                         np.asarray(c_O, dtype=float).ravel()])
    return Gt, ct


# ---------------------------------------------------------------------------
# Noise
# ---------------------------------------------------------------------------

@dataclass
class NoiseDraw:
    E_B: np.ndarray    # (d_O, d_O) symmetric
    E_C: np.ndarray    # (d_R, d_O)
    f_R: np.ndarray    # (d_R,)
    f_O: np.ndarray    # (d_O,)
    e_yty: float


def draw_noise(d_R, d_O, sigma, rng):
    """One draw of the single-draw symmetric Gaussian mechanism."""
    U = rng.normal(0.0, sigma, size=(d_O, d_O))
    E_B = np.triu(U) + np.triu(U, 1).T                  # upper incl. diag, mirrored
    E_C = rng.normal(0.0, sigma, size=(d_R, d_O))
    f_R = rng.normal(0.0, sigma, size=d_R)
    f_O = rng.normal(0.0, sigma, size=d_O)
    e_yty = float(rng.normal(0.0, sigma))
    return NoiseDraw(E_B=E_B, E_C=E_C, f_R=f_R, f_O=f_O, e_yty=e_yty)


# ---------------------------------------------------------------------------
# Bias operator (Theorem 1/2 bracket)
# ---------------------------------------------------------------------------

def projectors(p, d_R):
    PiR = np.diag([1.0] * d_R + [0.0] * (p - d_R))
    return PiR, np.eye(p) - PiR


def M(P, d_R):
    """sigma^{-2} E[P E P E] for the block-masked single-draw mechanism."""
    p = P.shape[0]
    PiR, PiO = projectors(p, d_R)
    D = np.diag(np.diag(P))
    return (P @ P + np.trace(P) * P - P @ D) @ PiO \
        + (P @ PiO @ P + np.trace(PiO @ P) * P) @ PiR


# ---------------------------------------------------------------------------
# Solve + correct
# ---------------------------------------------------------------------------

@dataclass
class Solution:
    beta_ridge: np.ndarray
    beta_bc: np.ndarray
    lam: float
    Psi_mode: str            # 'none' | 'O' | 'I'
    rho: float               # rho at the chosen (lambda, Psi), from the released Gram
    rho0: float = float("nan")
    rho_ceiling_O: float = float("nan")
    extra: dict = field(default_factory=dict)


def _lmin(S):
    return float(np.linalg.eigvalsh(0.5 * (S + S.T))[0])


def _smallest_lambda_O(Gt, PiO, target, max_doublings=200, iters=200):
    """Smallest lambda >= 0 with lambda_min(Gt + lambda PiO) >= target."""
    f = lambda lam: _lmin(Gt + lam * PiO)
    lo, hi = 0.0, max(target - f(0.0), 1e-12)
    k = 0
    while f(hi) < target:
        lo, hi = hi, 2.0 * hi
        k += 1
        if k > max_doublings:
            raise RuntimeError("O-block ridge cannot reach rho_target (ceiling too close)")
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if f(mid) >= target:
            hi = mid
        else:
            lo = mid
        if hi - lo <= 1e-12 * max(1.0, hi):
            break
    return hi


def solve_and_correct(Gt, ct, d_R, d_O, sigma, mode="auto", rho_target=2.0):
    Gt = np.asarray(Gt, dtype=float)
    Gt = 0.5 * (Gt + Gt.T)
    ct = np.asarray(ct, dtype=float)
    p = d_R + d_O
    assert Gt.shape == (p, p), (Gt.shape, p)
    PiR, PiO = projectors(p, d_R)
    edge = 2.0 * float(sigma) * np.sqrt(p)              # semicircle edge 2 sigma sqrt(p)
    rho_of = lambda S: (_lmin(S) / edge) if edge > 0 else np.inf

    rho0 = rho_of(Gt)
    lam_A = _lmin(Gt[:d_R, :d_R])
    ceil_O = lam_A / edge if edge > 0 else np.inf
    target = rho_target * edge

    if mode == "none" or rho0 >= rho_target:
        lam, Psi, Psi_mode = 0.0, np.zeros((p, p)), "none"
    elif mode == "auto":
        if ceil_O > rho_target:
            lam, Psi, Psi_mode = _smallest_lambda_O(Gt, PiO, target), PiO, "O"
        else:
            # exact arithmetic gives rho(lambda) == rho_target; the 1e-10 relative
            # margin keeps eigvalsh rounding from landing a hair BELOW the target
            # (party_r.py warns on `sol.rho < rho_target`).
            lam = target - _lmin(Gt) + 1e-10 * max(1.0, abs(target))
            Psi, Psi_mode = np.eye(p), "I"
    else:
        raise ValueError(f"unknown mode {mode!r}")

    G_lam = Gt + lam * Psi
    beta_ridge = np.linalg.solve(G_lam, ct)
    P_lam = np.linalg.inv(G_lam)
    beta_bc = beta_ridge - float(sigma) ** 2 * (M(P_lam, d_R) @ beta_ridge)
    return Solution(beta_ridge=beta_ridge, beta_bc=beta_bc, lam=float(lam),
                    Psi_mode=Psi_mode, rho=float(rho_of(G_lam)), rho0=float(rho0),
                    rho_ceiling_O=float(ceil_O))
