# Run: python3 reports/checks/theorem1_and_adaptive_ridge_check.py   (~1 min; numpy only)
# Exact check of Lemma 1 / Theorem 1 (single-draw masked symmetric Gaussian, A block un-noised),
# plus a Monte Carlo check that the zero-budget correction removes the O(sigma^2) bias,
# and that choosing lambda FROM the noised Gram (Remark 4 gate) breaks that.
import numpy as np
rng = np.random.default_rng(1)
dR, dO = 3, 4; p = dR+dO
PiR = np.diag([1.]*dR+[0.]*dO); PiO = np.eye(p)-PiR
def M(P):
    D = np.diag(np.diag(P))
    return (P@P + np.trace(P)*P - P@D)@PiO + (P@PiO@P + np.trace(PiO@P)*P)@PiR
basis = []
for i in range(p):
    for j in range(i, p):
        if i < dR and j < dR: continue
        S = np.zeros((p,p)); S[i,j] = S[j,i] = 1; basis.append(S)
X = rng.normal(size=(200,p)); G = X.T@X; P = np.linalg.inv(G)
exact = sum(P@S@P@S for S in basis)            # E[PEPE]/sigma^2, exact
print("Theorem 1 operator vs exact E[PEPE]/sigma^2:  max|diff| =", np.abs(exact-M(P)).max())

def draw(sig):
    z = rng.normal(0, sig, len(basis)); return sum(zk*S for zk,S in zip(z,basis))
beta = rng.normal(size=p); c = G@beta + rng.normal(size=p)*3; bhat = P@c
sig = 0.08*np.linalg.eigvalsh(G)[0]/(2*np.sqrt(p))*6   # rho ~ 2
print("rho =", np.linalg.eigvalsh(G)[0]/(2*sig*np.sqrt(p)))
N = 200000; acc = np.zeros(p); accbc = np.zeros(p)
for _ in range(N//2):
    E = draw(sig); f = rng.normal(0, sig, p)
    for s in (1,-1):                                # antithetic pair
        Gt = G+s*E; bt = np.linalg.solve(Gt, c+s*f)
        acc += bt; accbc += bt - sig**2*M(np.linalg.inv(Gt))@bt
pred = sig**2*M(P)@bhat
print("MC bias        :", np.round(acc/N-bhat, 6))
print("Theorem 1 bias :", np.round(pred, 6))
print("after correction:", np.round(accbc/N-bhat, 6), "(should be ~0 up to O(sigma^4)+MC)")

# adaptive full ridge: lambda chosen so lambda_min(Gt+lam I) = T  (Remark 4-style gate)
T = np.linalg.eigvalsh(G)[0] + 3*sig*np.sqrt(p)
lam0 = T - np.linalg.eigvalsh(G)[0]; P0 = np.linalg.inv(G+lam0*np.eye(p)); target = P0@c
acc_f = np.zeros(p); acc_a = np.zeros(p)
for _ in range(N//2):
    E = draw(sig); f = rng.normal(0, sig, p)
    for s in (1,-1):
        Gt = G+s*E
        for lam, a in ((lam0, acc_f), (max(0., T-np.linalg.eigvalsh(Gt)[0]), acc_a)):
            Pl = np.linalg.inv(Gt+lam*np.eye(p)); bt = Pl@(c+s*f)
            a += bt - sig**2*M(Pl)@bt
print("\nfixed lambda, corrected - ridge target   :", np.round(acc_f/N-target, 6))
print("adaptive lambda, corrected - ridge target:", np.round(acc_a/N-target, 6))
print("scale of the O(sigma^2) bias being corrected:", np.round(sig**2*M(P0)@target, 6))
