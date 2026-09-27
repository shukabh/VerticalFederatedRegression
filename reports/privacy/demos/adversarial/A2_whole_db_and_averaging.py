"""
A2 / A5 — Whole-database reconstruction and repeated-run noise averaging
(LOCAL SIMULATION ONLY). Same exact O Phase-4 arithmetic as A1.

Part 1 (whole-DB, needs scaling): with d_R probe columns per run, R reads d_R
O-records exactly per protocol run. ceil(n_O/d_R) runs reconstruct the ENTIRE O
database (all X_O and y). There is no rate limit or DP budget ledger in party_o.py.

Part 2 (no scaling needed): even a perfectly well-formed, in-bounds R (b binary,
Xdot the true PSI alignment) recovers the EXACT noiseless sufficient statistics by
running the protocol N times and averaging: O draws fresh N(0,sigma^2) each run, so
the averaged estimate has stdev sigma/sqrt(N). N runs cost N * the intended budget,
but nothing at O tracks or caps this.
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "scenario_b"))
from calibrate_hyperparameters import joint_sensitivity_B, analytic_gaussian_sigma

rng = np.random.default_rng(11)
B_R, B_O, B_y, eps, delta = 3.0, 3.0, 20.0, 1.0, 1e-5
sigma = analytic_gaussian_sigma(joint_sensitivity_B(B_R, B_O, B_y), eps, delta)


def O_C_cR(Xdot, X_O, y_O, sigma, rng):
    """Just the C / c_R blocks O returns (party_o.py:120-129), with noise."""
    d_R, d_O = Xdot.shape[1], X_O.shape[1]
    C = np.empty((d_R, d_O)); cR = np.empty(d_R)
    for j in range(d_R):
        for k in range(d_O):
            C[j, k] = Xdot[:, j] @ X_O[:, k] + rng.normal(0, sigma)
        cR[j] = Xdot[:, j] @ y_O + rng.normal(0, sigma)
    return C, cR


# =================== Part 1: reconstruct the whole O database ===================
n_O, d_O, d_R = 60, 5, 10
X_O = rng.uniform(-1, 1, size=(n_O, d_O)) * 1.5
y_O = rng.uniform(-1, 1, size=n_O) * 15.0
s = 1e7
rec_X = np.zeros((n_O, d_O)); rec_y = np.zeros(n_O)
n_runs = int(np.ceil(n_O / d_R))
for r in range(n_runs):
    rows = range(r * d_R, min((r + 1) * d_R, n_O))
    Xdot = np.zeros((n_O, d_R))
    for j, i in enumerate(rows):                 # column j is a one-hot on row i, amp s
        Xdot[i, j] = s
    C, cR = O_C_cR(Xdot, X_O, y_O, sigma, rng)
    for j, i in enumerate(rows):
        rec_X[i] = C[j] / s
        rec_y[i] = cR[j] / s
errX = np.linalg.norm(rec_X - X_O) / np.linalg.norm(X_O)
erry = np.linalg.norm(rec_y - y_O) / np.linalg.norm(y_O)
print("Part 1 — whole-database reconstruction (scaled one-hot probes)")
print(f"  n_O={n_O}, d_R={d_R}: {n_runs} runs reconstruct the FULL database")
print(f"  relative error  ||X_hat-X_O||/||X_O|| = {errX:.2e}   "
      f"||y_hat-y_O||/||y_O|| = {erry:.2e}")
print(f"  => for a real agency n_O=5,000,000 and d_R=10: "
      f"{int(np.ceil(5_000_000/10)):,} runs steal everything; no ledger stops it.\n")


# =================== Part 2: average away the noise, no scaling ==================
n_O, d_O, d_R = 300, 4, 3
X_O = rng.uniform(-1, 1, size=(n_O, d_O)) * 1.5
y_O = rng.uniform(-1, 1, size=n_O) * 15.0
b = (rng.random(n_O) < 0.5).astype(float)        # a legitimate binary selection
Xdot = np.zeros((n_O, d_R))
Xdot[b == 1] = np.clip(rng.uniform(-1, 1, size=(int(b.sum()), d_R)), -1, 1)  # in-bounds
true_cR = Xdot.T @ y_O                            # the exact (noiseless) statistic
print("Part 2 — repeated-run averaging with a WELL-FORMED, in-bounds R")
print(f"{'N runs':>8} | {'stdev of c_R estimate':>22} | {'expected sigma/sqrt(N)':>22}")
print("-" * 60)
for N in [1, 10, 100, 1000, 10000]:
    acc = np.zeros(d_R)
    for _ in range(N):
        _, cR = O_C_cR(Xdot, X_O, y_O, sigma, rng)
        acc += cR
    est = acc / N
    emp = np.linalg.norm(est - true_cR) / np.sqrt(d_R)
    print(f"{N:>8} | {emp:>22.4f} | {sigma/np.sqrt(N):>22.4f}")
print("  The noiseless O-sufficient-statistics are recovered to arbitrary precision;")
print("  N runs = N x the DP budget, and party_o.py keeps no state across invocations.")
