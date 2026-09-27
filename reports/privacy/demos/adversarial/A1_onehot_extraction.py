"""
A1 — Malicious-R single-record extraction (LOCAL SIMULATION ONLY).

This reproduces Party O's EXACT Phase-4 computation (paper eqs. 3-4; party_o.py
lines ~109-131) in a numpy plaintext model, and shows that a malicious R who sends
a crafted Enc(Xdot_R) / Enc(b) --- which O cannot inspect because they are
encrypted --- recovers an individual O-record's (x_O,i, y_i) essentially exactly,
DESPITE the DP noise, by amplifying its own plaintext scale.

Why it works: O adds a FIXED absolute Gaussian noise (sigma from the code's own
calibration). The sensitivity that fixes sigma assumes ||x_R,i|| <= B_R and
b in {0,1}. Xdot_R and b are encrypted, so O never checks these bounds. R sets a
one-hot column with amplitude s; the target's value comes back multiplied by s
while the noise stays at sigma, so dividing by s drives the error to sigma/s -> 0.

No network, no crypto, no pickle: this is the arithmetic O performs, nothing else.
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "scenario_b"))
from calibrate_hyperparameters import joint_sensitivity_B, analytic_gaussian_sigma

rng = np.random.default_rng(7)

# ---- O's private data (synthetic, stands in for the agency's records) ----
n_O, d_O, d_R = 200, 4, 3
B_R, B_O, B_y, eps, delta = 3.0, 3.0, 20.0, 1.0, 1e-5
X_O = rng.uniform(-1, 1, size=(n_O, d_O)) * 1.5          # O features
y_O = rng.uniform(-1, 1, size=n_O) * 15.0                # O response (e.g. income)

sigma = analytic_gaussian_sigma(joint_sensitivity_B(B_R, B_O, B_y), eps, delta)
print(f"calibrated sigma = {sigma:.2f}  (Delta2={joint_sensitivity_B(B_R,B_O,B_y):.1f}, "
      f"eps={eps}, delta={delta})")


def O_phase4(b, Xdot, X_O, y_O, sigma, rng):
    """EXACT replica of party_o.py:116-131 (b-masked Gram + noise). Returns the
    decrypted aggregates R would receive. Every independent entry ~ N(0,sigma^2)."""
    d_O = X_O.shape[1]; d_R = Xdot.shape[1]
    C = np.zeros((d_R, d_O)); cR = np.zeros(d_R)
    B = np.zeros((d_O, d_O)); cO = np.zeros(d_O)
    for j in range(d_R):                                  # C[j,k]=sum_i Xdot[i,j] X_O[i,k]
        for k in range(d_O):
            C[j, k] = Xdot[:, j] @ X_O[:, k] + rng.normal(0, sigma)
        cR[j] = Xdot[:, j] @ y_O + rng.normal(0, sigma)   # c_R[j]=sum_i Xdot[i,j] y_i
    for j in range(d_O):
        for k in range(j, d_O):
            B[j, k] = B[k, j] = (b * X_O[:, j]) @ X_O[:, k] + rng.normal(0, sigma)
        cO[j] = (b * X_O[:, j]) @ y_O + rng.normal(0, sigma)
    yty = (b * y_O) @ y_O + rng.normal(0, sigma)
    return dict(B=B, C=C, cR=cR, cO=cO, yty=yty)


target = 137                                              # the individual R wants to read
print(f"\nGROUND TRUTH  x_O[{target}] = {np.round(X_O[target],4)}   y[{target}] = {y_O[target]:.4f}\n")

print(f"{'scale s':>10} | {'||x_O_hat - x_O||':>18} | {'|y_hat - y|':>12} | {'rel err':>9}")
print("-" * 60)
results = []
for s in [1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e8]:
    # Malicious R: one-hot column 0 at the target row, amplitude s; other cols zero.
    Xdot = np.zeros((n_O, d_R)); Xdot[target, 0] = s
    b = np.zeros(n_O)                                     # b unused here (attack via C/cR)
    agg = O_phase4(b, Xdot, X_O, y_O, sigma, rng)
    x_hat = agg["C"][0, :] / s                            # recovers x_O[target] + noise/s
    y_hat = agg["cR"][0] / s                              # recovers y[target]  + noise/s
    ex = np.linalg.norm(x_hat - X_O[target])
    ey = abs(y_hat - y_O[target])
    rel = ex / max(np.linalg.norm(X_O[target]), 1e-12)
    results.append((s, ex, ey, rel))
    print(f"{s:>10.0e} | {ex:>18.6f} | {ey:>12.6f} | {rel:>9.2e}")

print("\nInterpretation:")
print(" * scale s=1  : one legit-magnitude probe. Error ~ sigma, the single record is")
print("               BURIED in noise (this is what DP is supposed to guarantee).")
print(" * scale s>=1e4: error ~ sigma/s. The target's exact O-features AND response are")
print("               recovered to many digits. DP is fully defeated; O never saw s.")
print(" * The target row can be ANY O index (matched or not) -- b is not checked either.")
print(f" * At s=1e6, recovered y = {results[6][2]:.2e} away from the true value "
      f"{y_O[target]:.4f}.")
