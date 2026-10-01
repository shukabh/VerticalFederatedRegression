"""
calibration_heuristic.py -- from a privacy budget to the protocol's noise and ridge, and to the
matched-set size it needs.

The chain (all inputs public, committed before any data are seen):

  1. delta     <= 1/(10 N), N = people O protects (its register size n_O or the population).
  2. budget    one release: (eps, delta) directly. k planned releases: convert the cap to zCDP,
               sqrt(rho_cap) = sqrt(L + eps) - sqrt(L), L = ln(1/delta), and give each release
               rho = rho_cap / k (minus a small share for the CKKS flooding noise).
  3. Delta     replace-one sensitivity from the committed bounds (R = B_R^2, O = B_O^2, Y = B_y^2):
               |R - Y| <= 2 O:  Delta^2 = (2 O + R + Y)^2 / 2 + 2 R Y
               R - Y  >= 2 O:  Delta^2 = 4 R (O + Y)
               Y - R  >  2 O:  1-D maximisation (sim_core.delta_rep).
  4. sigma     one release: sigma = Delta * s(eps, delta), the analytic-Gaussian multiplier
               (tabulated by this script); k releases: sigma = Delta / sqrt(2 rho).
  5. ridge     lambda = max{0, 2 rho* sigma sqrt(p) - n ell}, rho* = 2.5; it switches off at
               n_off = 2 rho* sigma sqrt(p) / ell. Release gate: rho_hat >= 1.
  6. utility   first order, the release perturbs beta by P (f - E beta) with P = (G + lambda I)^-1:
               Cov ~ P (sigma^2 I + Cov(E beta)) P, Cov(E beta) ~ sigma^2 ||beta||^2 I.
               With G = n Sigma (Sigma: second moments of the standardized, clipped design),
               s^2 the residual variance and lambda = 0:
                 coefficients  RMSE within r x OLS at  n_coef ~ sigma^2 (1+||b||^2) tr(S^-2) / (s^2 tr(S^-1) (r^2-1))
               (for a well-conditioned design). Scaling laws, exact at first order:
                 prediction    excess test MSE depends on n only through n / sigma when ell = 0
                               (lambda is proportional to sigma), so n_pred(eps) = n_pred(eps0) sigma(eps)/sigma(eps0);
                 coefficients  once the ridge is off, DP MSE / sampling MSE ~ sigma^2 / n, so
                               n_coef(eps) = n_coef(eps0) (sigma(eps)/sigma(eps0))^2.
               With sigma ~ Delta/eps, prediction needs n ~ Delta/eps and coefficients n ~ (Delta/eps)^2.
               For ill-conditioned designs use the first-order eigen-sums below (coef_ratio, pred_ratio).

The script tabulates s(eps, delta), and checks the first-order predictions against the Monte Carlo
of reports/coef_accuracy (simulation) and the real-data sweep of reports/adassp/sweep.
Usage: python experiments/calibration_heuristic.py [--quick] > reports/calibration/heuristic.md
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sim_core as sc  # noqa: E402


# ----------------------------------------------------------------------------- calibration
def zcdp_rho(eps, delta):
    """Largest rho with rho-zCDP => (eps, delta)-DP via eps = rho + 2 sqrt(rho ln(1/delta))."""
    L = np.log(1.0 / delta)
    return (np.sqrt(L + eps) - np.sqrt(L)) ** 2


def calibrate(eps, delta, B_R, B_O, B_y, p, n, ell=0.0, rho_star=2.5, k=1, err_share=0.0):
    """The full chain for a budget (eps, delta) spread over k releases."""
    Delta = sc.delta_rep(B_R, B_O, B_y)
    if k == 1 and err_share == 0.0:
        sigma = sc.analytic_gauss_sigma(Delta, eps, delta)
        rho = Delta ** 2 / (2 * sigma ** 2)                  # its zCDP cost, for the ledger
    else:
        rho = (1 - err_share) * zcdp_rho(eps, delta) / k
        sigma = Delta / np.sqrt(2 * rho)
    lam = sc.fixed_lambda(sigma, p, n, ell, rho_star)
    n_off = 2 * rho_star * sigma * np.sqrt(p) / ell if ell > 0 else np.inf
    return dict(Delta=Delta, sigma=sigma, rho=rho, lam=lam, n_off=n_off)


# ----------------------------------------------------------------------------- first-order utility
def coef_ratio(n, Sigma, beta, s2, sigma, lam, d_R, idx):
    """RMSE(private) / RMSE(OLS) over coordinates idx, against the true beta (first order)."""
    G = n * Sigma
    P = np.linalg.inv(G + lam * np.eye(len(beta)))
    mech = sc.mech_cov_Ebeta(beta[None], sigma, d_R)[0]
    cov = s2 * P @ G @ P + P @ (sigma ** 2 * np.eye(len(beta)) + mech) @ P
    bias = -lam * P @ beta
    mse = np.trace(cov[np.ix_(idx, idx)]) + bias[idx] @ bias[idx]
    floor = s2 * np.trace(np.linalg.inv(G)[np.ix_(idx, idx)])
    return np.sqrt(mse / floor)


def pred_ratio(n, Sigma, beta, s2, sigma, lam, d_R):
    """Test MSE(private) / test MSE(non-private ridge with +I), first order."""
    d = len(beta)
    G = n * Sigma

    def excess(P, extra):
        cov = s2 * P @ G @ P + extra
        b = -(np.eye(d) - P @ G) @ beta                     # shrinkage bias of P G beta
        return np.trace(Sigma @ cov) + b @ Sigma @ b

    P = np.linalg.inv(G + lam * np.eye(d))
    mech = sc.mech_cov_Ebeta(beta[None], sigma, d_R)[0]
    priv = excess(P, P @ (sigma ** 2 * np.eye(d) + mech) @ P)
    Pn = np.linalg.inv(G + np.eye(d))
    return (s2 + priv) / (s2 + excess(Pn, 0.0))


def crossing(ns, vals, thr):
    """Smallest n (log-linear interpolation) from which vals stay <= thr."""
    vals = np.asarray(vals)
    if vals[-1] > thr:
        return None
    i = len(vals) - 1
    while i > 0 and vals[i - 1] <= thr:
        i -= 1
    if i == 0:
        return float(ns[0])
    x0, x1 = np.log(ns[i - 1]), np.log(ns[i])
    return float(np.exp(x0 + (thr - vals[i - 1]) * (x1 - x0) / (vals[i] - vals[i - 1])))


def fmt(v):
    if v is None:
        return "not reached"
    return f"{v:,.0f}" if v < 1000 else f"{v / 1000:.1f}k" if v < 1e4 else f"{v / 1000:.0f}k"


# ----------------------------------------------------------------------------- report
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="smaller population for the simulation check")
    args = ap.parse_args()
    out = print
    out("# Calibration heuristic: from ε to σ, λ and the matched-set size\n")
    out("Generated by `experiments/calibration_heuristic.py`.\n")

    out("## 1. Noise multiplier s(ε, δ) = σ / Δ for one release\n")
    out("Analytic Gaussian (exact), with the classical bound and the zCDP closed form for comparison.\n")
    eps_grid = [0.1, 0.25, 0.5, 1, 2, 4, 8]
    out("| δ | " + " | ".join(f"ε = {e:g}" for e in eps_grid) + " |")
    out("|---|" + "---:|" * len(eps_grid))
    for d in (1e-5, 1e-6, 1e-7, 1e-8):
        out(f"| {d:g} | " + " | ".join(f"{sc.analytic_gauss_sigma(1.0, e, d):.3g}" for e in eps_grid) + " |")
    out("")
    out("| δ = 10⁻⁶ | " + " | ".join(f"ε = {e:g}" for e in eps_grid) + " |")
    out("|---|" + "---:|" * len(eps_grid))
    L = np.log(1e6)
    out("| classical √(2 ln(1.25/δ))/ε | " + " | ".join(f"{np.sqrt(2 * np.log(1.25e6)) / e:.3g}" for e in eps_grid) + " |")
    out("| zCDP (√(L+ε)+√L)/(√2 ε) | " + " | ".join(f"{(np.sqrt(L + e) + np.sqrt(L)) / (np.sqrt(2) * e):.3g}" for e in eps_grid) + " |")
    out("| zCDP / exact | " + " | ".join(
        f"{(np.sqrt(L + e) + np.sqrt(L)) / (np.sqrt(2) * e) / sc.analytic_gauss_sigma(1.0, e, 1e-6):.2f}" for e in eps_grid) + " |")
    out("")

    out("## 2. Check against the simulation (reports/coef_accuracy)\n")
    out("Matched records at which the slope RMSE reaches 2× the OLS floor: first-order prediction vs Monte Carlo "
        "(200 runs per n; ρ* = 2, ℓ = 0.8 λ_min of the population, δ = 10⁻⁵).\n")
    mc = {0.3: {1: "75k", 2: "21k", 4: "7.1k", 8: "1.9k"}, 0.5: {1: "≈105k", 2: "34k", 4: "9.4k", 8: "3.2k"},
          0.9: {1: "> 100k", 2: "66k", 4: "30k", 8: "13k"}}
    out("| R² | ε | σ | first order (exact mask) | rule of thumb n_coef | σ² transfer from ε = 2 | Monte Carlo |")
    out("|---|---:|---:|---:|---:|---:|---:|")
    ns = np.unique(np.round(np.geomspace(100, 1e6, 400)))
    for r2 in (0.3, 0.5, 0.9):
        pop = sc.Population(r2=r2, seed=0, pop_size=100_000 if args.quick else 400_000)
        S, b, s2, p, dR = pop.Sigma_pop, pop.beta_true, pop.noise_sd ** 2, pop.p, pop.d_R_block
        idx = list(range(1, p))
        ell = 0.8 * pop.lam_min_pop
        Si = np.linalg.inv(S)[np.ix_(idx, idx)]
        Si2 = (np.linalg.inv(S) @ np.linalg.inv(S))[np.ix_(idx, idx)]
        sig2, _ = pop.sigma_for(2, 1e-5)
        mc2 = {0.3: 21e3, 0.5: 34e3, 0.9: 66e3}[r2]
        for eps in (1, 2, 4, 8):
            sig, _ = pop.sigma_for(eps, 1e-5)
            vals = [coef_ratio(n, S, b, s2, sig, sc.fixed_lambda(sig, p, n, ell, 2.0), dR, idx) for n in ns]
            thumb = sig ** 2 * (1 + b @ b) * np.trace(Si2) / (s2 * np.trace(Si) * 3)
            transfer = "(anchor)" if eps == 2 else fmt(mc2 * (sig / sig2) ** 2)
            out(f"| {r2} | {eps} | {sig:.0f} | {fmt(crossing(ns, vals, 2.0))} | {fmt(max(thumb, sc.fixed_lambda(sig, p, 0, 0, 2.0) / ell))} | {transfer} | {mc[r2][eps]} |")
    out("\nThe rule of thumb is floored at n_off, below which the ridge is on. At R² = 0.9 clipping bias at the "
        "committed bounds raises the Monte Carlo OLS floor, so the ratio reaches 2 sooner than first order predicts; "
        "the first-order numbers ignore clipping. The σ² transfer anchors on the Monte Carlo value at ε = 2.\n")

    out("## 3. Check against the real-data sweep (reports/adassp/sweep)\n")
    sys.path.insert(0, os.path.join(HERE, "adassp"))
    import datasets as D  # noqa: E402
    import methods as M  # noqa: E402
    X_csv = list(csv.DictReader(open(os.path.join(HERE, "..", "reports", "adassp", "sweep", "crossings.csv"))))
    out("Matched records at which test MSE comes within 10% of non-private: first-order prediction vs the sweep "
        "(Wang's preprocessing, B = 1, Δ = √10, ρ* = 2, ℓ = 0, δ = 10⁻⁶). Σ, β and s² are taken from the full "
        "dataset, as a pilot study or public reference data would supply them.\n")
    out("| dataset | ε | σ | first order | σ transfer from ε = 1 | sweep |")
    out("|---|---:|---:|---:|---:|---:|")
    ns = np.unique(np.round(np.geomspace(100, 2e5, 300)))
    for nm in ("bike", "elevators", "pol", "protein"):
        ds = D.load(nm)
        S = ds.X.T @ ds.X / ds.n
        b = np.linalg.solve(ds.X.T @ ds.X + np.eye(ds.d), ds.X.T @ ds.y)
        s2 = float(np.mean((ds.y - ds.X @ b) ** 2))
        p, dR = ds.d, ds.d // 2
        sig1 = M.sigma_vfl(1, 1e-6)
        row1 = next(r for r in X_csv if r["dataset"] == nm and float(r["threshold"]) == 1.1
                    and float(r["eps"]) == 1 and r["method"] == "vfl")
        for eps in (1, 2, 4):
            sig = M.sigma_vfl(eps, 1e-6)
            vals = [pred_ratio(n, S, b, s2, sig, sc.fixed_lambda(sig, p, n, 0.0, 2.0), dR) for n in ns]
            if eps == 1:
                transfer = "(anchor)"
            else:
                transfer = "—" if row1["n_cross"] == "" else fmt(float(row1["n_cross"]) * sig / sig1)
            row = next(r for r in X_csv if r["dataset"] == nm and float(r["threshold"]) == 1.1
                       and float(r["eps"]) == eps and r["method"] == "vfl")
            sweep = f"> {int(row['n_max']) // 1000}k" if row["n_cross"] == "" else fmt(float(row["n_cross"]))
            out(f"| {nm} | {eps} | {sig:.1f} | {fmt(crossing(ns, vals, 1.1))} | {transfer} | {sweep} |")
    out("\nThe first-order column evaluates the eigen-sums of `pred_ratio` (sampling, shrinkage and DP terms); the "
        "transfer column scales the sweep's ε = 1 value by σ(ε)/σ(1). A closed form with tr(Σ⁻¹) is not usable here: "
        "pol and elevators are ill-conditioned or rank-deficient, and the ridge damps exactly the directions that "
        "would dominate it.\n")

    out("## 4. Worked example: the simulation population's bounds\n")
    pop = sc.Population(r2=0.5, seed=0, pop_size=100_000 if args.quick else 400_000)
    ell = 0.8 * pop.lam_min_pop
    out(f"B_R = {pop.B_R:.2f}, B_O = {pop.B_O:.2f}, B_y = {pop.B_y:.2f}, p = {pop.p}, ℓ = {ell:.3f}, ρ* = 2.5, "
        f"n = 10,000. One release uses the analytic Gaussian; k = 5 releases split a zCDP budget equal to the same (ε, δ).\n")
    out("| ε | δ | Δ | σ (k = 1) | ρ per release | λ at n = 10k | ridge off from n | σ (k = 5) |")
    out("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for eps in (0.5, 1, 2, 4, 8):
        for delta in (1e-6,):
            c1 = calibrate(eps, delta, pop.B_R, pop.B_O, pop.B_y, pop.p, 10_000, ell=ell)
            c5 = calibrate(eps, delta, pop.B_R, pop.B_O, pop.B_y, pop.p, 10_000, ell=ell, k=5)
            out(f"| {eps} | {delta:g} | {c1['Delta']:.1f} | {c1['sigma']:.0f} | {c1['rho']:.4f} | {c1['lam']:.0f} | "
                f"{fmt(c1['n_off'])} | {c5['sigma']:.0f} |")
    out("")


if __name__ == "__main__":
    main()
