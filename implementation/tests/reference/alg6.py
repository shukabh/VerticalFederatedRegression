# Verbatim copy of the proof review's reference implementation (v31/proofs/scratch_stats/alg6.py), kept for tests.
"""Faithful implementation of Algorithm 6 (Solve and infer) of protocol v3.1, Gaussian slot noise,
plus the quantities needed to check Lemmas S3, S4, S6 and Proposition S5.

Conventions: index set of R's block (incl. intercept) = first dA coordinates; Pi_O projects on
the last dO coordinates.  Noise: masked symmetric matrix with sd sigma on every entry outside the
A block (diagonal of B included), sd sigma on every entry of c, sd sigma_yy on y'y.
"""
import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

Z = norm.ppf


def masked_noise(rng, R, p, dA, sigma):
    M = rng.standard_normal((R, p, p))
    E = np.triu(M, 1)
    E = E + np.swapaxes(E, -1, -2)
    E[..., np.arange(p), np.arange(p)] = rng.standard_normal((R, p))
    E[..., :dA, :dA] = 0.0
    return sigma * E


def chi_mask(p, dA):
    chi = np.ones((p, p)); chi[:dA, :dA] = 0.0
    return chi


def K_mat(theta, chi):
    """K(theta)_ik = delta_ik (sum_j chi_ij theta_j^2 - chi_ii theta_i^2) + chi_ik theta_i theta_k."""
    p = len(theta)
    t2 = theta ** 2
    K = chi * np.outer(theta, theta)
    K[np.arange(p), np.arange(p)] += chi @ t2 - np.diag(chi) * t2
    return K


def M_chi(P, chi):
    """E[E P E]/sigma^2 = chi o P + diag(sum_j chi_ij P_jj - chi_ii P_ii)."""
    p = P.shape[0]
    dP = np.diag(P)
    out = chi * P
    out[np.arange(p), np.arange(p)] += chi @ dP - np.diag(chi) * dP
    return out


def rss_fun(G, c, yy, lam, Psi):
    """R(G,c,yy) = yy - c'theta - lam theta'Psi theta, theta = (G + lam Psi)^{-1} c (batched)."""
    th = np.linalg.solve(G + lam * Psi, c[..., None])[..., 0]
    return yy - np.einsum("...i,...i->...", c, th) - lam * np.einsum("...i,ij,...j->...", th, Psi, th), th


def rss_bias_first_order(G, theta, lam, Psi, sigma, chi):
    """Second-order mean of RSS~ - RSS_lambda (Lemma S6): -sigma^2 [tr((G+2 lam Psi) V) + 2 lam theta'Psi P M theta],
    V = P (I + K(theta)) P."""
    p = G.shape[0]
    P = np.linalg.inv(G + lam * Psi)
    V = P @ (np.eye(p) + K_mat(theta, chi)) @ P
    return -sigma ** 2 * (np.trace((G + 2 * lam * Psi) @ V) + 2 * lam * theta @ Psi @ P @ M_chi(P, chi) @ theta)


def lambda_O_closed(A, B1, C1, gamma):
    """min{lam >= 0: lam_min(G1 + lam Pi_O) >= gamma} in closed form (Lemma S3)."""
    lamA = np.linalg.eigvalsh(A)[0]
    if lamA <= gamma:
        return np.inf
    S = B1 - C1.T @ np.linalg.solve(A - gamma * np.eye(A.shape[0]), C1)
    return max(0.0, gamma - np.linalg.eigvalsh(S)[0])


def lambda_O_bisect(G1, PiO, gamma, hi=1e12, it=200):
    f = lambda lam: np.linalg.eigvalsh(G1 + lam * PiO)[0] - gamma
    if f(0.0) >= 0:
        return 0.0
    if f(hi) < 0:
        return np.inf
    lo = 0.0
    for _ in range(it):
        mid = 0.5 * (lo + hi)
        if f(mid) >= 0:
            hi = mid
        else:
            lo = mid
    return hi


def cv_alpha(t, alpha):
    """(1-alpha) quantile of |Z + t|."""
    t = abs(t)
    f = lambda c: norm.cdf(c - t) - norm.cdf(-c - t) - (1 - alpha)
    return brentq(f, 0.0, t + 10.0)


def stage1(rng, G, dA, sigma1, sigma2, alpha_L, theta_star, kappa_Psi):
    p = G.shape[0]
    E1 = masked_noise(rng, 1, p, dA, sigma1)[0]
    G1 = G + E1
    r1 = sigma1 * (2 * np.sqrt(p) + 2 * np.sqrt(np.log(1 / alpha_L)))
    lmin1 = np.linalg.eigvalsh(G1)[0]
    L_hat = max(0.0, lmin1 - r1)
    g_star = 2 * theta_star * sigma2 * np.sqrt(p)
    lam_I = max(0.0, g_star + r1 - lmin1)
    lam_O = lambda_O_closed(G1[:dA, :dA], G1[dA:, dA:], G1[:dA, dA:], g_star + r1)
    PiO = np.zeros((p, p)); PiO[dA:, dA:] = np.eye(p - dA)
    if lam_O <= kappa_Psi * lam_I:
        lam, Psi, kind = lam_O, PiO, "O"
    else:
        lam, Psi, kind = lam_I, np.eye(p), "I"
    return dict(E1=E1, G1=G1, r1=r1, L_hat=L_hat, g_star=g_star, lam=lam, Psi=Psi, kind=kind,
                lam_I=lam_I, lam_O=lam_O)


def solve_infer(rng, G, c, yy, n, dA, s1, sigma2, sigma_yy, B_y, alpha, alpha_L, alpha_U,
                B_pub, Bboot=300, truncate=False, antithetic=False):
    """Stage 2 + inference given the stage-1 output s1. Returns a dict (or None if refused)."""
    p = G.shape[0]
    lam, Psi = s1["lam"], s1["Psi"]
    a_CI = alpha - alpha_L - alpha_U
    E2 = masked_noise(rng, 1, p, dA, sigma2)[0]
    f2 = sigma2 * rng.standard_normal(p)
    g2 = sigma_yy * rng.standard_normal()
    G2, c2, yy2 = G + E2, c + f2, yy + g2
    Gl2 = G2 + lam * Psi
    theta_hat = np.linalg.eigvalsh(Gl2)[0] / (2 * sigma2 * np.sqrt(p))
    out = dict(theta_hat_gate=theta_hat, E2=E2, f2=f2, g2=g2)
    if theta_hat < 1:
        out["refused"] = True
        return out
    out["refused"] = False
    P2 = np.linalg.inv(Gl2)
    th = P2 @ c2
    RSS = yy2 - c2 @ th - lam * th @ Psi @ th
    # parametric bootstrap
    Es = masked_noise(rng, Bboot, p, dA, sigma2)
    fs = sigma2 * rng.standard_normal((Bboot, p))
    gs = sigma_yy * rng.standard_normal(Bboot)
    if antithetic:
        Es = np.concatenate([Es, -Es]); fs = np.concatenate([fs, -fs]); gs = np.concatenate([gs, -gs])
    Gb = G2[None] + Es
    keep = np.ones(len(Gb), bool)
    if truncate:   # symmetric truncation: ||Gl2^{-1/2} E* Gl2^{-1/2}|| <= 1/2
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
    U_corr = yy2 + np.sqrt(2 * np.log(1 / alpha_U) * (n * B_y ** 4 / 4 + sigma_yy ** 2))
    U_corr = min(U_corr, n * B_y ** 2)
    L_hat = s1["L_hat"]
    priv = np.sqrt(U_spec / L_hat) if L_hat > 0 else np.inf
    Bb = min(B_pub, priv)
    bbar = lam * np.linalg.norm(Psi @ P2, axis=0) * Bb
    sdV = np.sqrt(np.diag(V))
    half_beta = np.array([cv_alpha(bbar[j] / sdV[j], a_CI) * sdV[j] if np.isfinite(bbar[j]) else np.inf
                          for j in range(p)])
    out.update(theta=th, RSS=RSS, Corr=Corr, s2=s2, Vmech=Vmech, V=V, half_theta=z * se_m,
               half_beta=half_beta, U_spec=U_spec, U_corr=U_corr, Bb=Bb, bbar=bbar, P2=P2,
               kept=keep.mean(), boot_mean_theta=thb.mean(0))
    return out
