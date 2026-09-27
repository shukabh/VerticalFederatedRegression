"""
D5 — the sensitivity assumes ||x_R,i|| <= B_R, b in {0,1}^n_O and Xdot_R = 0 off the match
set. O cannot check any of this (it only sees Enc(b), Enc(Xdot_R)), and party_o.py does not
try. This script replays party_o.py's Phase-4 arithmetic in plaintext (HE is linear, so the
decrypted values are the same up to CKKS error) with the code's own sigma, and shows what
an R that deviates from the protocol obtains.

Note: R also generates the CKKS keys/parameters (party_r.py: hb.ckks_owner(...)), so R
controls the plaintext dynamic range available for the scale factor M.

Run: python d5_unenforced_bounds.py     (<1 s)
"""
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "../../../../scenario_b")))
import calibrate_hyperparameters as ch  # noqa: E402

rng = np.random.default_rng(11)
n_O, d_O, d_R, n = 5000, 5, 10, 400
B_R, B_O, B_y, eps, delta = 3.0, 3.0, 3.0, 1.0, 1e-5
sigma = ch.analytic_gaussian_sigma(ch.joint_sensitivity_B(B_R, B_O, B_y), eps, delta)

X_O = ch.clip_rows(rng.standard_normal((n_O, d_O)), B_O)[0]
y_O = ch.clip_scalar(X_O @ rng.standard_normal(d_O) + rng.standard_normal(n_O), B_y)[0]
matched = rng.choice(n_O, n, replace=False)


def party_o_phase4(b, Xdot):
    """party_o.py lines 110-131 in plaintext, one N(0,sigma^2) per released entry."""
    z = lambda *s: rng.standard_normal(s) * sigma
    iu = np.triu_indices(d_O)
    B = (X_O.T * b) @ X_O
    return dict(B=B[iu] + z(len(iu[0])), C=Xdot.T @ X_O + z(Xdot.shape[1], d_O),
                cR=Xdot.T @ y_O + z(Xdot.shape[1]), cO=(X_O.T * b) @ y_O + z(d_O),
                yty=float(b @ (y_O ** 2)) + z(1)[0])


print(f"sigma (eps={eps}, bounds {B_R},{B_O},{B_y}) = {sigma:.3f}")

# ---- (a) honest R: dedicated-column read of one matched person is buried in noise
b = np.zeros(n_O); b[matched] = 1
Xdot = np.zeros((n_O, d_R)); Xdot[matched] = ch.clip_rows(rng.standard_normal((n, d_R)), B_R)[0]
tgt = matched[0]
Xh = Xdot.copy(); Xh[tgt] = 0; Xh[tgt, 0] = B_R        # legal: row norm = B_R, only tgt uses col 0
Xh[:, 0][np.arange(n_O) != tgt] = 0
out = party_o_phase4(b, Xh)
print(f"(a) honest, bound-respecting dedicated column: y_hat={out['cR'][0]/B_R:+.3f}  true y={y_O[tgt]:+.3f}"
      f"   (SD of estimate sigma/B_R = {sigma/B_R:.2f})")

# ---- (b) R scales its column by M >> B_R, and points columns at ANY O row (matched or not)
M = 1e4
targets = np.r_[matched[:5], np.setdiff1d(np.arange(n_O), matched)[:5]]   # 5 matched + 5 UNMATCHED
Xm = np.zeros((n_O, d_R))
for j, i in enumerate(targets):
    Xm[i, j] = M
out = party_o_phase4(b, Xm)
y_hat = out["cR"] / M
xO_hat = out["C"] / M
print(f"(b) R violates B_R (M={M:g}) and addresses 10 rows (5 matched, 5 NOT in the intersection):")
print(f"    max |y_hat - y| = {np.max(np.abs(y_hat - y_O[targets])):.2e}   "
      f"max |x_O_hat - x_O| = {np.max(np.abs(xO_hat - X_O[targets])):.2e}")
D6 = ch.joint_sensitivity_B(B_R, B_O, B_y)
mu_eff = M * np.sqrt(4 * B_O**2 + 4 * B_y**2) / sigma     # replace (x_O,y)->(-x_O,-y) in an M-row
print(f"    Gaussian-DP parameter mu: claimed Delta/sigma={D6/sigma:.3f}; actual >= {mu_eff:.0f}  (no meaningful eps)")

# ---- (c) R sends a non-binary b = M * e_i  -> O-only blocks expose one record
i = targets[7]
bm = np.zeros(n_O); bm[i] = M
out = party_o_phase4(bm, np.zeros((n_O, d_R)))
print(f"(c) b = M*e_i for an unmatched row: y_i^2 from y'y: {out['yty']/M:.4f} vs {y_O[i]**2:.4f};"
      f"  x_O,i*y_i from c_O: max err {np.max(np.abs(out['cO']/M - X_O[i]*y_O[i])):.1e}")

# ---- (d) no check that Xdot rows are zero off the match set: pull in the whole database
Xall = np.zeros((n_O, 1)); Xall[:, 0] = M                         # 'intercept' on every O row
out = party_o_phase4(np.ones(n_O), Xall)
print(f"(d) Xdot = M on all {n_O} rows (not just the {n} matched): c_R/M = {out['cR'][0]/M:.3f} vs "
      f"sum of y over O's ENTIRE register {y_O.sum():.3f}; unmatched people enter the release")
