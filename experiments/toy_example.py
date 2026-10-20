"""
toy_example.py -- every number of the worked-example deck (slides/build_toy_deck.js).

R holds 3 cohort records with 2 features; O holds 5 register records with 2 features and y.
The script follows the protocol phase by phase and records, in the clear, what each step computes:
  0 setup      bounds, sensitivities, the protocol's calibration (calibration_protocol.py)
  1 link       padding, O's secret shuffle tau, the PSI outcome, R's exact block A
  2 upload     R's vectors b and xdot in O's shuffled order
  3 aggregate  the encrypted sums B, C, c_R, c_O, y'y, and their slots in the packed ciphertexts
  4 release    the release algebra on one number (toy modulus, no security), then the two DP releases
  5 solve      stage-1 ridge choice, gate, estimate, bootstrap Wald intervals
  scale-up     the same steps on 20,000 matched people from a population like the toy's
Usage: python experiments/toy_example.py   (writes reports/toy_example/toy.json)
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import sys

import numpy as np
from scipy.special import ndtri

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import calibration_protocol as cp  # noqa: E402
import sim_core as sc  # noqa: E402

EPS, DELTA, F1, THETA_STAR, ALPHA = 1.0, 1e-9, 0.1, 2.5, 0.05
ALPHA_L = ALPHA_U = ALPHA / 20
ALPHA_CI = ALPHA - ALPHA_L - ALPHA_U
SEED = 20261020        # first seed tried; the run uses the first seed whose noise draws are all within +-2.5 sd
TYPICAL = 2.5

# ---------------------------------------------------------------- data
R_ROWS = [("104", 0.5, -0.2), ("221", -0.4, 0.8), ("317", 0.9, 0.1)]                 # id, education, health
O_ROWS = [("088", 0.3, 1.0, -0.5), ("104", -0.6, 1.0, 0.7), ("150", 0.1, 1.0, -0.2),  # id, age, benefits, income
          ("221", 0.8, 0.0, 0.4), ("399", -0.2, 0.0, 0.9)]
N_R_MAX, N_O_BAR = 4, 6                       # public caps: R pads its ids to 4, O its register to 6
R_DUMMY, O_DUMMY = "9d1", "7e5"               # random identifiers that never match
TAU = {"150": 1, O_DUMMY: 2, "221": 3, "088": 4, "399": 5, "104": 6}   # O's secret shuffle: id -> position


def oprf(i):
    """Stand-in for F_k(id): a keyed hash, shown truncated (illustration only)."""
    return hmac.new(b"run-key", i.encode(), hashlib.sha256).hexdigest()[:6]


def r(x, k=3):
    return np.round(np.asarray(x, float), k).tolist()


def sym_noise(rng, sigma, d_R, p):
    """Symmetric noise on the O-dependent entries of G (B and C blocks), zero on R's block A."""
    E = np.zeros((p, p))
    iu = np.triu_indices(p)
    E[iu] = sigma * rng.standard_normal(len(iu[0]))
    E = np.triu(E) + np.triu(E, 1).T
    E[:d_R, :d_R] = 0.0
    return E


def stage_rule(G1, sigma1, sigma2, p, d_R):
    r1 = sigma1 * (2 * np.sqrt(p) + 2 * np.sqrt(np.log(1 / ALPHA_L)))
    g = 2 * THETA_STAR * sigma2 * np.sqrt(p)
    lmin1 = float(np.linalg.eigvalsh(G1)[0])
    lam_I = max(0.0, g + r1 - lmin1)
    PiO = np.diag([0.0] * d_R + [1.0] * (p - d_R))
    # lambda_O: smallest ridge on O's block alone that lifts lambda_min to g + r1. As it grows, lambda_min tends
    # to lambda_min(A) (R's exact block), so it is infinite when A itself is below the target.
    if lmin1 >= g + r1:
        lam_O = 0.0
    elif np.linalg.eigvalsh(G1[:d_R, :d_R])[0] < g + r1:
        lam_O = float("inf")
    else:
        lo, hi = 0.0, 1e9
        for _ in range(200):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if np.linalg.eigvalsh(G1 + mid * PiO)[0] < g + r1 else (lo, mid)
        lam_O = hi
    lam, Psi = (lam_O, PiO) if lam_O <= 2 * lam_I else (lam_I, np.eye(p))
    return dict(r1=r1, g=g, lmin1=lmin1, lam_I=lam_I, lam_O=lam_O, lam=lam, Psi=Psi,
                Psi_name="Π_O" if Psi is PiO else "I")


def solve(G2, c2, lam, Psi, sigma2, d_R, rng, B=4000):
    p = len(c2)
    P = np.linalg.inv(G2 + lam * Psi)
    theta = P @ c2
    gate = float(np.linalg.eigvalsh(G2 + lam * Psi)[0] / (2 * sigma2 * np.sqrt(p)))
    draws = np.array([np.linalg.solve(G2 + sym_noise(rng, sigma2, d_R, p) + lam * Psi,
                                      c2 + sigma2 * rng.standard_normal(p)) for _ in range(B)])
    sd = draws.std(axis=0, ddof=1)
    z = float(ndtri(1 - ALPHA_CI / 2))
    return theta, gate, sd, z


def draws(seed, sigma1, sigma2, d_R, p):
    """The run's noise, in protocol order: the toy release's eta, stage 1, stage 2 (G, c, y'y)."""
    rng = np.random.default_rng(seed)
    eta = sigma2 * rng.standard_normal()
    E1, E2 = sym_noise(rng, sigma1, d_R, p), sym_noise(rng, sigma2, d_R, p)
    f2, gy = sigma2 * rng.standard_normal(p), sigma2 * rng.standard_normal()
    iu = np.triu_indices(p)
    zs = np.concatenate([[eta / sigma2], E1[iu] / sigma1, E2[iu] / sigma2, f2 / sigma2, [gy / sigma2]])
    return rng, eta, E1, E2, f2, gy, float(np.abs(zs).max())


def main():
    d_R, d_O = 3, 2                     # R's columns: intercept, education, health; O's: age, benefits
    p = d_R + d_O
    names = ["intercept", "education", "health", "age", "benefits"]

    # ---------------- the answer in the clear (what pooling would give)
    Rmap = {i: (a, b) for i, a, b in R_ROWS}
    Omap = {i: (a, b, y) for i, a, b, y in O_ROWS}
    I = [i for i, *_ in R_ROWS if i in Omap]
    Z = np.array([[1.0, *Rmap[i], *Omap[i][:2]] for i in I])
    yI = np.array([Omap[i][2] for i in I])
    G, c, yy = Z.T @ Z, Z.T @ yI, float(yI @ yI)

    # ---------------- 0 setup: bounds, sensitivities, calibration
    B_R, B_O, B_y = np.sqrt(3.0), np.sqrt(2.0), 1.0      # |every entry| <= 1; B_R includes the intercept
    Delta, Delta_G = sc.delta_rep(B_R, B_O, B_y), sc.delta_rep(B_R, B_O, 0.0)
    rho = cp.rho_tight(EPS, DELTA)
    sigma1, sigma2 = Delta_G / np.sqrt(2 * F1 * rho), Delta / np.sqrt(2 * (1 - F1) * rho)
    seed = SEED
    while (run := draws(seed, sigma1, sigma2, d_R, p))[-1] > TYPICAL:
        seed += 1
    rng, eta_slot, E1, E2, f2, gy, zmax = run

    # ---------------- 1 link
    R_ids = [i for i, *_ in R_ROWS] + [R_DUMMY]
    O_ids = [i for i, *_ in O_ROWS] + [O_DUMMY]
    psi = {i: TAU.get(i) for i in R_ids if i in TAU}                 # R learns: matched id -> position
    A = Z[:, :d_R].T @ Z[:, :d_R]

    # ---------------- 2 upload: vectors over the n_O_bar positions, in tau order
    b = np.zeros(N_O_BAR)
    X = np.zeros((d_R, N_O_BAR))
    for i, t in psi.items():
        b[t - 1] = 1.0
        X[:, t - 1] = [1.0, *Rmap[i]]
    # O's columns in tau order (dummy row: zeros)
    Ocol = np.zeros((3, N_O_BAR))
    for i, t in TAU.items():
        if i in Omap:
            Ocol[:, t - 1] = Omap[i]
    order = sorted(TAU, key=TAU.get)

    # ---------------- 3 aggregate (what the ciphertexts hold)
    Bm = np.array([[np.sum(b * Ocol[j] * Ocol[k]) for k in range(d_O)] for j in range(d_O)])
    Cm = np.array([[np.sum(X[j] * Ocol[k]) for k in range(d_O)] for j in range(d_R)])
    cR = np.array([np.sum(X[j] * Ocol[2]) for j in range(d_R)])
    cO = np.array([np.sum(b * Ocol[k] * Ocol[2]) for k in range(d_O)])
    yy_enc = float(np.sum(b * Ocol[2] ** 2))
    assert np.allclose(Bm, G[d_R:, d_R:]) and np.allclose(Cm, G[:d_R, d_R:]) and np.allclose(cR, c[:d_R])
    assert np.allclose(cO, c[d_R:]) and np.isclose(yy_enc, yy) and np.allclose(A, G[:d_R, :d_R])
    slots1 = [(f"B_{j+1}{k+1}", Bm[j, k]) for j in range(d_O) for k in range(j, d_O)] + \
             [(f"C_{j}{k+1}", Cm[j, k]) for j in range(d_R) for k in range(d_O)]
    slots2 = slots1 + [(f"c_R{j}", cR[j]) for j in range(d_R)] + [(f"c_O{k+1}", cO[k]) for k in range(d_O)] + [("yᵀy", yy)]

    # ---------------- 4 release algebra on one number (toy: one coefficient, small modulus, no security)
    q, Om = 1_000_003, 1000
    zc = lambda v: int((v + q // 2) % q - q // 2)                    # centred representative
    z = float(Cm[1, 0])                                              # C_11 = sum education x age
    s_R, s_O, a, e, e_pk, u, e0, e1 = 3, 5, 723_481, 2, 1, 1, 1, -2
    s = s_R + s_O
    ecd = round(Om * z)
    c0, c1 = zc(-a * s + e + ecd), a
    p0 = zc(-a * s + e_pk)
    c0p, c1p = zc(c0 + p0 * u + e0), zc(c1 + a * u + e1)
    eta = int(round(Om * eta_slot))
    hO = zc(c0p + c1p * s_O + eta)
    v = zc(hO + c1p * s_R)
    e_tot = e + e_pk * u + e0 + e1 * s
    assert v == ecd + e_tot + eta
    toy = dict(q=q, Omega=Om, z=z, ecd=ecd, s_R=s_R, s_O=s_O, s=s, a=a, e=e, e_pk=e_pk, u=u, e0=e0, e1=e1,
               c0=c0, c1=c1, p0=p0, c0p=c0p, c1p=c1p, eta=eta, hO=hO, v=v, e_tot=e_tot, decoded=v / Om)

    # ---------------- 4-5 stage 1 and the ridge
    G1 = G + E1
    st = stage_rule(G1, sigma1, sigma2, p, d_R)

    # ---------------- stage 2, gate, estimate, intervals
    G2, c2, yy2 = G + E2, c + f2, yy + gy
    theta, gate, sd, zq = solve(G2, c2, st["lam"], st["Psi"], sigma2, d_R, rng)
    theta_lam = np.linalg.solve(G + st["lam"] * st["Psi"], c)

    # ---------------- scale-up: the same steps on 20,000 matched people
    n = 20_000
    srng = np.random.default_rng(SEED + 1)
    XR = srng.uniform(-1, 1, (n, 2))
    XO = np.column_stack([srng.uniform(-1, 1, n), srng.binomial(1, 0.5, n)])
    beta_true = np.array([0.1, 0.3, -0.2, 0.4, 0.2])
    Zs = np.column_stack([np.ones(n), XR, XO])
    ys = np.clip(Zs @ beta_true + 0.2 * srng.standard_normal(n), -1, 1)
    Gs, cs = Zs.T @ Zs, Zs.T @ ys
    G1s = Gs + sym_noise(srng, sigma1, d_R, p)
    sts = stage_rule(G1s, sigma1, sigma2, p, d_R)
    G2s, c2s = Gs + sym_noise(srng, sigma2, d_R, p), cs + sigma2 * srng.standard_normal(p)
    theta_s, gate_s, sd_s, _ = solve(G2s, c2s, sts["lam"], sts["Psi"], sigma2, d_R, srng, B=2000)
    ols_s = np.linalg.solve(Gs, cs)

    out = dict(
        seed=seed, max_abs_z=zmax, names=names, eps=EPS, delta=DELTA, f1=F1, theta_star=THETA_STAR, alpha=ALPHA, alpha_ci=ALPHA_CI, z=zq,
        R_rows=R_ROWS, O_rows=O_ROWS, n_R_max=N_R_MAX, n_O_bar=N_O_BAR, R_dummy=R_DUMMY, O_dummy=O_DUMMY,
        oprf={i: oprf(i) for i in sorted(set(R_ids + O_ids))}, tau=TAU, order=order, I=I, psi=psi,
        Z=r(Z), yI=r(yI), G=r(G), c=r(c), yy=round(yy, 3), rankG=int(np.linalg.matrix_rank(G)),
        bounds=dict(B_R=B_R, B_O=B_O, B_y=B_y), Delta=Delta, Delta_G=Delta_G, rho=rho, sigma1=sigma1, sigma2=sigma2,
        A=r(A), b=r(b), X=r(X), Ocol=r(Ocol), Bm=r(Bm), Cm=r(Cm), cR=r(cR), cO=r(cO),
        slots1=[(k, round(v_, 3)) for k, v_ in slots1], slots2=[(k, round(v_, 3)) for k, v_ in slots2],
        toy=toy,
        stage1=dict(Bhat=r(G1[d_R:, d_R:], 1), Chat=r(G1[:d_R, d_R:], 1), G1=r(G1, 1), lmin1=st["lmin1"], r1=st["r1"],
                    g=st["g"], lam_I=st["lam_I"], lam_O=None if np.isinf(st["lam_O"]) else st["lam_O"],
                    lam=st["lam"], Psi=st["Psi_name"], lminA=float(np.linalg.eigvalsh(A)[0])),
        stage2=dict(G2=r(G2, 1), c2=r(c2, 1), yy2=round(float(yy2), 1), gate=gate,
                    theta=r(theta, 4), theta_lam=r(theta_lam, 4), sd=r(sd, 4)),
        scale=dict(n=n, beta_true=r(beta_true), ols=r(ols_s, 4), theta=r(theta_s, 4), sd=r(sd_s, 4),
                   lmin1=sts["lmin1"], r1=sts["r1"], g=sts["g"], lam=sts["lam"], Psi=sts["Psi_name"], gate=gate_s,
                   lminG=float(np.linalg.eigvalsh(Gs)[0]), G11=float(Gs[1, 1])),
    )
    os.makedirs(os.path.join(HERE, "..", "reports", "toy_example"), exist_ok=True)
    path = os.path.join(HERE, "..", "reports", "toy_example", "toy.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False, default=float)
    print(json.dumps({k: out[k] for k in ["I", "psi", "rankG", "Delta", "Delta_G", "rho", "sigma1", "sigma2"]}, default=float))
    print("G =", out["G"]); print("c =", out["c"], "yy =", out["yy"]); print("toy:", toy)
    print("stage1:", {k: v_ for k, v_ in out["stage1"].items() if k not in ("G1",)})
    print("stage2:", {k: v_ for k, v_ in out["stage2"].items() if k not in ("G2",)})
    print("scale:", out["scale"])


if __name__ == "__main__":
    main()
