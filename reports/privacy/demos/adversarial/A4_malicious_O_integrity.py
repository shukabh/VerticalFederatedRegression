"""
A4 — Malicious O: undetectable output tampering (LOCAL SIMULATION ONLY).

O is the only party that computes the O-dependent blocks, R never has an independent
view of O's data, and there is no MAC / proof-of-correct-computation on O's reply
(party_o.py:133 just pickles and sends). So O can return fabricated aggregates and
bias the published model arbitrarily; R decrypts, assembles and solves (party_r.py:
119-137) with no integrity check. This shows O silently zeroing R's coefficient of
interest -- R concludes 'R's feature has no effect', the opposite of the truth, and
cannot tell.

Also demonstrates label poisoning (PSI misalignment) as a bias vector.
"""
import numpy as np

rng = np.random.default_rng(3)
n, d_R, d_O = 400, 2, 2
X_R = rng.uniform(-1, 1, (n, d_R))
X_O = rng.uniform(-1, 1, (n, d_O))
beta_true = np.array([2.0, -1.5, 0.7, 0.4])          # R-feature 0 has a STRONG effect
Z = np.hstack([X_R, X_O])
y = Z @ beta_true + rng.normal(0, 0.1, n)

# Honest aggregates (tiny noise for legibility; the point is integrity, not DP).
A = X_R.T @ X_R
B = X_O.T @ X_O
C = X_R.T @ X_O
cR = X_R.T @ y
cO = X_O.T @ y

def solve(A, B, C, cR, cO):
    G = np.block([[A, C], [C.T, B]])
    c = np.concatenate([cR, cO])
    return np.linalg.solve(G, c)

beta_honest = solve(A, B, C, cR, cO)

# Malicious O erases R-feature-0's cross terms and moment so its coefficient collapses.
C_bad = C.copy(); C_bad[0, :] = 0.0
cR_bad = cR.copy(); cR_bad[0] = 0.0
beta_tampered = solve(A, B, C_bad, cR_bad, cO)

print("Malicious O — biasing R's headline coefficient:")
print(f"  true beta            = {np.round(beta_true,3)}")
print(f"  honest fit           = {np.round(beta_honest,3)}")
print(f"  O-tampered fit       = {np.round(beta_tampered,3)}")
print(f"  R-feature-0: true {beta_true[0]:.2f}  honest {beta_honest[0]:.2f}  "
      f"tampered {beta_tampered[0]:.2f}  <-- O suppressed it to ~0")
print("  R has NO independent view of O's data and NO integrity proof, so R cannot")
print("  distinguish the tampered fit from a legitimate one.\n")

# Label poisoning: O returns wrong row indices, misaligning X_R against O's rows.
perm = rng.permutation(n)                              # O maps matches to wrong O-rows
C_mis = X_R.T @ X_O[perm]
cO_mis = X_O[perm].T @ y                               # O's own blocks stay self-consistent
# (cR uses y, which under misalignment pairs X_R with the wrong individuals' y)
cR_mis = X_R.T @ y[perm]
beta_mis = solve(A, B, C_mis, cR_mis, cO_mis)
print("Malicious O — PSI label poisoning (records misaligned):")
print(f"  misaligned fit       = {np.round(beta_mis,3)}  (arbitrary bias; R cannot detect")
print("  because R never learns O's true row<->id map; the code's only 'check' is a")
print("  match COUNT against a ground-truth file that does not exist in deployment).")
