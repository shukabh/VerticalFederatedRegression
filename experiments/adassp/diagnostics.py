"""
diagnostics.py -- why the VFL protocol trails AdaSSP where it does (paper, Section "Empirical evaluation").

1. Ridge ablation at full n: VFL with the implementable ridge (ell = 0), with an oracle ridge
   (ell = 0.8 lambda_min(G/n), non-private, a headroom reference) and with no ridge, against AdaSSP.
2. How well y fills its privacy bound after Wang's scaling y / max|y|: rms |y| and the tail
   max|y| / q_0.995(|y|). The noise is calibrated to the bound |y| <= 1, so unused headroom is
   lost signal.
3. Conditioning: lambda_min(X'X/n) and lambda_max/lambda_min.

Usage (repo root, WANG_REPO set):  python experiments/adassp/diagnostics.py > reports/adassp/diagnostics.md
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import datasets as D  # noqa: E402
import methods as M  # noqa: E402

sc = M.sc
R, DELTA = 30, 1e-6


def ablation(name, eps):
    ds = D.load(name)
    X, y, d = ds.X, ds.y, ds.d
    d_R = d // 2
    acc = {k: [] for k in ("adassp", "ell0", "oracle", "none")}
    for f, te in enumerate(D.kfold(ds.n)):
        tr = np.setdiff1d(np.arange(ds.n), te)
        Xtr, ytr, n = X[tr], y[tr], len(tr)
        G0, c, yty = Xtr.T @ Xtr, Xtr.T @ ytr, float(ytr @ ytr)
        rng = np.random.default_rng([f, int(10 * eps)])
        m_np = M.test_mse(X[te], y[te], M.nonprivate(G0, c))
        acc["adassp"].append(M.test_mse(X[te], y[te], M.adassp(G0, c, eps, DELTA, R, rng)).mean() / m_np)
        sig = M.sigma_vfl(eps, DELTA)
        Gt, ct, _ = sc.release(G0, c, yty, d_R, sig, R, rng)
        for key, ell in (("ell0", 0.0), ("oracle", 0.8 * np.linalg.eigvalsh(G0 / n)[0])):
            _, bc, _ = M.vfl_fixed(Gt, ct, sig, d_R, n, ell=ell)
            acc[key].append(M.test_mse(X[te], y[te], bc).mean() / m_np)
        with np.errstate(all="ignore"):
            acc["none"].append(np.median(M.test_mse(X[te], y[te], M.bsolve(Gt, ct))) / m_np)
    return {k: float(np.mean(v)) for k, v in acc.items()}


def main():
    print("# Diagnostics: where the VFL protocol loses to AdaSSP\n")
    print(f"10-fold CV at full n, R = {R} noise draws per fold, delta = {DELTA:g}. Test MSE / non-private.\n")
    print("## 1. Ridge ablation\n")
    print("| dataset | ε | AdaSSP (published) | VFL, ℓ = 0 (protocol) | VFL, oracle ℓ | VFL, no ridge (median) |")
    print("|---|---:|---:|---:|---:|---:|")
    for nm in ["housing", "wine", "elevators", "bike", "pol"]:
        for eps in (1, 10):
            a = ablation(nm, eps)
            print(f"| {nm} | {eps} | {a['adassp']:.3f} | {a['ell0']:.3f} | {a['oracle']:.3f} | {a['none']:.3g} |")
    print("\n## 2. How well y fills its bound, and conditioning\n")
    print("| dataset | n | d | rms abs(y) | q99.5 abs(y) | max / q99.5 | λmin(XᵀX/n) | λmax/λmin |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for nm in ["elevators", "wine", "bike", "housing", "protein", "pol"]:
        ds = D.load(nm)
        a = np.abs(ds.y)
        w = np.linalg.eigvalsh(ds.X.T @ ds.X / ds.n)
        q = np.quantile(a, 0.995)
        rank_def = w[0] < 1e-12 * w[-1]
        print(f"| {nm} | {ds.n:,} | {ds.d} | {np.sqrt(np.mean(ds.y ** 2)):.3f} | {q:.3f} | {a.max() / q:.2f} | "
              + ("0 (rank-deficient) | ∞ |" if rank_def else f"{w[0]:.1e} | {w[-1] / w[0]:.1e} |"))


if __name__ == "__main__":
    main()
