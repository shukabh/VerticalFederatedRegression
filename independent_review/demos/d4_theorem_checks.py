"""
d4_theorem_checks.py -- Monte Carlo checks of Theorem 1/2, Remark 2 and the plug-in correction.

 A. Theorem 1 at large rho: empirical E[beta~] - beta^ (antithetic pairs) vs M beta^ from
    phase1_common._bias_operator; also vs the GOE variant (no -P diag(P) Pi_O term).
 B. Remark 2 tail bound Pr(||P E|| >= 1) <= exp(-p (rho-1)^2).
 C. Heavy tails: without conditioning, beta~ has no mean (sample mean does not settle).
 D. Ridge chosen from the released G~ (as solve_and_correct does) vs a deterministic ridge:
    does the bias correction still remove the O(sigma^2) bias?
"""
import sys
import numpy as np
sys.path.insert(0, "/home/user/VerticalFederatedRegression/scenario_b")
import phase1_common as p1

rng = np.random.default_rng(7)
dR, dO = 3, 3; p = dR + dO
n = 300
Z = np.column_stack([rng.normal(size=(n, p))]) + 0.3 * rng.normal(size=(n, 1))
beta_true = rng.normal(size=p)
y = Z @ beta_true + 0.5 * rng.normal(size=n)
G = Z.T @ Z; c = Z.T @ y
P = np.linalg.inv(G); bhat = P @ c
lmin = np.linalg.eigvalsh(G)[0]
iu = np.triu_indices(p)
mask = np.ones((p, p)); mask[:dR, :dR] = 0


def draw_E(sig, m):
    Mx = rng.normal(0, sig, size=(m, p, p))
    E = np.triu(Mx) + np.transpose(np.triu(Mx, 1), (0, 2, 1))
    return E * mask, rng.normal(0, sig, size=(m, p))


def M_goe(Pm):
    PiO = np.zeros((p, p)); PiO[dR:, dR:] = np.eye(dO); PiR = np.eye(p) - PiO
    return ((Pm @ Pm + np.trace(Pm) * Pm) @ PiO
            + (Pm @ PiO @ Pm + np.trace(PiO @ Pm) * Pm) @ PiR)


# ---------------- A ----------------
print("A. Theorem 1, single-draw mechanism, antithetic Monte Carlo")
for rho in (8.0, 4.0):
    sig = lmin / (2 * rho * np.sqrt(p))
    m, reps = 200000, 5
    acc = np.zeros(p); acc2 = []
    for _ in range(reps):
        E, f = draw_E(sig, m)
        bp = np.linalg.solve(G + E, (c + f)[..., None])[..., 0]
        bm = np.linalg.solve(G - E, (c - f)[..., None])[..., 0]
        pair = 0.5 * (bp + bm) - bhat
        acc += pair.mean(0); acc2.append(pair)
    emp = acc / reps
    se = np.concatenate(acc2).std(0) / np.sqrt(m * reps)
    th = sig ** 2 * p1._bias_operator(P, dR, dO)(bhat)
    goe = sig ** 2 * M_goe(P) @ bhat
    print(f"  rho={rho}: |emp|={np.linalg.norm(emp):.3e}  |emp - Thm1|={np.linalg.norm(emp-th):.2e}"
          f"  |emp - GOE variant|={np.linalg.norm(emp-goe):.2e}  (MC se ~{np.linalg.norm(se):.1e})")
    print(f"           relative error of Thm 1: {np.linalg.norm(emp-th)/np.linalg.norm(emp):.3f}")

# ---------------- B ----------------
print("\nB. Remark 2 tail bound")
for rho in (1.1, 1.3, 1.6):
    sig = lmin / (2 * rho * np.sqrt(p))
    E, _ = draw_E(sig, 100000)
    Gh = np.linalg.cholesky(P)            # P = L L^T ; spectral radius of PE = ||L^T E L||
    S = np.einsum('ji,mjk,kl->mil', Gh, E, Gh)
    r = np.abs(np.linalg.eigvalsh(S)).max(1)
    print(f"  rho={rho}: empirical Pr(rho(PE)>=1) = {np.mean(r>=1):.4f}   bound exp(-p(rho-1)^2) = "
          f"{np.exp(-p*(rho-1)**2):.4f}")

# ---------------- C ----------------
print("\nC. Unconditional mean of beta~ at rho=0.5 (no ridge): running means of coordinate 0")
sig = lmin / (2 * 0.5 * np.sqrt(p))
E, f = draw_E(sig, 400000)
b0 = np.linalg.solve(G + E, (c + f)[..., None])[..., 0][:, 0]
for k in (10**3, 10**4, 10**5, 4 * 10**5):
    print(f"  first {k:>6} draws: mean={b0[:k].mean():+10.3f}  max|.|={np.abs(b0[:k]).max():.1f}"
          f"   (beta^_0={bhat[0]:+.3f})")

# ---------------- D ----------------
print("\nD. Full ridge chosen from the released G~ vs deterministic ridge (target rho_hat=2)")
sig = lmin / (2 * 0.35 * np.sqrt(p))            # noise-dominated: rho(lambda=0) = 0.35
edge = 2 * sig * np.sqrt(p); target = 2.0
m = 200000
E, f = draw_E(sig, m)
Gt = G + E; ctt = c + f
lmin_t = np.linalg.eigvalsh(Gt)[:, 0]
lam_dd = np.maximum(0.0, target * edge - lmin_t)          # exact smallest lambda: rho_hat = 2
lam_fix = float(np.median(lam_dd))                          # a deterministic lambda of same size
I = np.eye(p)


def run(lams):
    k_ = len(lams); Gt_, ct = Gt[:k_], ctt[:k_]
    Glam = Gt_ + lams[:, None, None] * I
    Pl = np.linalg.inv(Glam)
    br = np.einsum('mij,mj->mi', Pl, ct)
    bc = np.array([br[i] - sig**2 * p1._bias_operator(Pl[i], dR, dO)(br[i]) for i in range(len(lams))])
    tgt = np.linalg.solve(G[None] + lams[:, None, None] * I, np.broadcast_to(c, (len(lams), p))[..., None])[..., 0]
    return br - tgt, bc - tgt


k = 40000
for name, lams in (("deterministic lambda", np.full(k, lam_fix)), ("lambda from G~ (as code)", lam_dd[:k])):
    er, ec = run(lams)
    ser = er.std(0) / np.sqrt(k); sec = ec.std(0) / np.sqrt(k)
    print(f"  {name:26s}: |bias raw ridge|={np.linalg.norm(er.mean(0)):.4f} (se {np.linalg.norm(ser):.4f})"
          f"   |bias after correction|={np.linalg.norm(ec.mean(0)):.4f} (se {np.linalg.norm(sec):.4f})")
print(f"  (lambda median {lam_fix:.1f}; lambda from G~ ranges {np.percentile(lam_dd,5):.1f}.."
      f"{np.percentile(lam_dd,95):.1f}; |beta^_lambda| = "
      f"{np.linalg.norm(np.linalg.solve(G+lam_fix*I, c)):.3f}, |beta^_OLS| = {np.linalg.norm(bhat):.3f})")
print(f"  shrinkage bias |beta^_lambda - beta^_OLS| = "
      f"{np.linalg.norm(np.linalg.solve(G+lam_fix*I, c) - bhat):.3f}")
