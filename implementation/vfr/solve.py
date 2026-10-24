"""Phase 5 at R: Algorithm 6 (Solve and infer), corrected version of paper/algorithms.tex.

Port of the proof review's scratch_stats/alg6.py (same random-number consumption, so the stage-2 estimate
and intervals can be compared draw for draw), with the corrections of the paper:
  * r_1 one-sided radius sigma_1 (2 sqrt p + 2 sqrt(ln(1/alpha_L)));
  * lambda_O by the closed form of Lemma S3(c) instead of bisection (paper tocheck: "replace the bisection");
  * bootstrap truncation ||P~^{1/2} E* P~^{1/2}||_op <= 1/2 (paper: required for the moments to exist);
  * U_yy = min{ yy^ + z_{1-alpha_U/2} sigma_yy + B_y^2 sqrt(n ln(2/alpha_U)/2), n B_y^2 }, sigma_yy = sigma_2 (joint)
    or sigma_y (separate)  (alg6.py's U_spec uses z_{1-alpha_U} and ln(1/alpha_U); both are returned).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

Z = norm.ppf


def masked_noise(rng, R, p, dA, sigma):
    """Same draws as alg6.masked_noise."""
    M = rng.standard_normal((R, p, p))
    E = np.triu(M, 1)
    E = E + np.swapaxes(E, -1, -2)
    E[..., np.arange(p), np.arange(p)] = rng.standard_normal((R, p))
    E[..., :dA, :dA] = 0.0
    return sigma * E


def lambda_O_closed(A, B1, C1, gamma):
    """min{lam >= 0: lam_min(G1 + lam Pi_O) >= gamma} (Lemma S3(c)); inf if lam_min(A) <= gamma."""
    lamA = np.linalg.eigvalsh(A)[0]
    if lamA <= gamma:
        return np.inf
    S = B1 - C1.T @ np.linalg.solve(A - gamma * np.eye(A.shape[0]), C1)
    return max(0.0, gamma - np.linalg.eigvalsh(S)[0])


def cv_alpha(t, alpha):
    t = abs(t)
    f = lambda c: norm.cdf(c - t) - norm.cdf(-c - t) - (1 - alpha)
    return brentq(f, 0.0, t + 10.0)


def assemble(A, B_triu, C):
    """G from R's exact A, the released upper triangle of B (mirrored) and C."""
    dA, dO = A.shape[0], B_triu.shape[0]
    B = np.triu(B_triu) + np.triu(B_triu, 1).T
    G = np.zeros((dA + dO, dA + dO))
    G[:dA, :dA] = A
    G[:dA, dA:] = C
    G[dA:, :dA] = C.T
    G[dA:, dA:] = B
    return G


def stage1_rule(G1, dA, sigma1, sigma2, alpha_L, theta_star, kappa_Psi):
    p = G1.shape[0]
    r1 = sigma1 * (2 * np.sqrt(p) + 2 * np.sqrt(np.log(1 / alpha_L)))
    lmin1 = np.linalg.eigvalsh(G1)[0]
    L_hat = max(0.0, lmin1 - r1)
    g_star = 2 * theta_star * sigma2 * np.sqrt(p)
    lam_I = max(0.0, g_star + r1 - lmin1)
    lam_O = lambda_O_closed(G1[:dA, :dA], G1[dA:, dA:], G1[:dA, dA:], g_star + r1)
    PiO = np.zeros((p, p))
    PiO[dA:, dA:] = np.eye(p - dA)
    if lam_O <= kappa_Psi * lam_I:
        lam, Psi, kind = lam_O, PiO, 'O'
    else:
        lam, Psi, kind = lam_I, np.eye(p), 'I'
    return dict(r1=r1, L_hat=L_hat, g_star=g_star, lam=float(lam), Psi=Psi, kind=kind, lam_I=float(lam_I),
                lam_O=float(lam_O), lmin1=float(lmin1))


def rss_fun(G, c, yy, lam, Psi):
    th = np.linalg.solve(G + lam * Psi, c[..., None])[..., 0]
    return yy - np.einsum('...i,...i->...', c, th) - lam * np.einsum('...i,ij,...j->...', th, Psi, th), th


def stage2_infer(G2, c2, yy2, n, dA, s1, sigma2, sigma_yy, B_y, alpha, alpha_L, alpha_U, B_pub, rng,
                 Bboot=2000, truncate=True, antithetic=True):
    """Gate, estimate, parametric bootstrap, RSS correction, both intervals.  rng: numpy Generator."""
    p = G2.shape[0]
    lam, Psi = s1['lam'], s1['Psi']
    a_CI = alpha - alpha_L - alpha_U
    Gl2 = G2 + lam * Psi
    gate = np.linalg.eigvalsh(Gl2)[0] / (2 * sigma2 * np.sqrt(p))
    out = dict(theta_hat_gate=float(gate))
    if gate < 1:
        out['refused'] = True
        return out
    out['refused'] = False
    P2 = np.linalg.inv(Gl2)
    th = P2 @ c2
    RSS = yy2 - c2 @ th - lam * th @ Psi @ th
    Es = masked_noise(rng, Bboot, p, dA, sigma2)
    fs = sigma2 * rng.standard_normal((Bboot, p))
    gs = sigma_yy * rng.standard_normal(Bboot)
    if antithetic:
        Es = np.concatenate([Es, -Es]); fs = np.concatenate([fs, -fs]); gs = np.concatenate([gs, -gs])
    Gb = G2[None] + Es
    keep = np.ones(len(Gb), bool)
    if truncate:
        w, U = np.linalg.eigh(Gl2)
        Wm = U @ np.diag(w ** -0.5) @ U.T
        nr = np.abs(np.linalg.eigvalsh(Wm[None] @ Es @ Wm[None]))
        keep = nr.max(axis=1) <= 0.5
    RSSb, thb = rss_fun(Gb[keep], c2[None] + fs[keep], yy2 + gs[keep], lam, Psi)
    Vmech = np.cov(thb, rowvar=False)
    Corr = RSSb.mean() - RSS
    s2 = max(RSS - Corr, 0.0) / (n - p)
    z = Z(1 - a_CI / 2)
    se_m = np.sqrt(np.diag(Vmech))
    V = Vmech + s2 * P2 @ G2 @ P2
    U_spec = yy2 + Z(1 - alpha_U) * sigma_yy + B_y ** 2 * np.sqrt(n * np.log(1 / alpha_U) / 2)
    U_yy = min(yy2 + Z(1 - alpha_U / 2) * sigma_yy + B_y ** 2 * np.sqrt(n * np.log(2 / alpha_U) / 2), n * B_y ** 2)
    L_hat = s1['L_hat']
    priv = np.sqrt(U_yy / L_hat) if L_hat > 0 else np.inf
    Bb = min(B_pub, priv)
    bbar = lam * np.linalg.norm(Psi @ P2, axis=0) * Bb
    sdV = np.sqrt(np.diag(V))
    half_beta = np.array([cv_alpha(bbar[j] / sdV[j], a_CI) * sdV[j] if np.isfinite(bbar[j]) else np.inf
                          for j in range(p)])
    out.update(theta=th, RSS=float(RSS), Corr=float(Corr), s2=float(s2), Vmech=Vmech, V=V, half_theta=z * se_m,
               half_beta=half_beta, U_yy=float(U_yy), U_spec=float(U_spec), B_beta=float(Bb), bbar=bbar, P2=P2,
               kept=float(keep.mean()), shrinkage=-lam * P2 @ Psi @ th, sigma2_trP=float(sigma2 * np.trace(P2)),
               z=float(z), alpha_CI=float(a_CI))
    return out
