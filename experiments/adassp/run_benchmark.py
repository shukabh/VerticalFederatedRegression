"""
run_benchmark.py -- AdaSSP (Wang 2018) vs the revised Scenario-B VFL protocol on Wang's UCI data.

Setup, following Wang's code/exp_uci.m:
  * data preprocessed exactly as Wang (datasets.load): z-scored X, unit-norm rows, y / max|y|;
  * 10-fold cross-validation, test MSE of each estimator on the held-out fold;
  * eps in {0.01, ..., 10}, delta = 1e-6; R noise draws per (fold, eps).
Vertical split: R holds the first d_R = floor(d/2) features (--d-r to change); O holds the rest
and y. The VFL release adds noise to every O-dependent block (B, C, c_R, c_O); R's block
A = X_R'X_R stays exact. Wang's joint row normalisation gives ||x_R||, ||x_O||, |y| <= 1, so the
protocol's committed bounds are B_R = B_O = B_y = 1.

Methods (see methods.py):
  trivial        theta = 0
  nonprivate     (X'X + I)^-1 X'y                                  Wang's non-private reference
  ssp            Wang's sufficient-statistics perturbation
  adassp         Wang's AdaSSP: DP for the whole record (x, y) -- a stronger guarantee than ours
  adassp_matched AdaSSP's adaptive ridge under OUR adjacency and accountant (idealised: its
                 private lambda_min step needs X'X in the clear, which nobody holds in VFL)
  vfl            revised protocol: fixed ridge lambda = 2 rho* sigma sqrt(p), bias-corrected;
                 a draw that fails the release gate rho_hat >= 1 is not released (theta = 0)
  vfl_uncorrected  the same release and ridge, without the bias correction

Usage (repo root; WANG_REPO points at a clone of github.com/yuxiangw/optimal_dp_linear_regression):
  python experiments/adassp/run_benchmark.py --quick     # 3 small datasets, 3 eps, a few seconds
  python experiments/adassp/run_benchmark.py             # the 9 datasets of the deck
  python experiments/adassp/run_benchmark.py --all       # all 29 datasets of Wang's published run
Outputs go to reports/adassp/ (or --out): results.csv, compare.csv, summary.md, fig_mse_vs_eps.png.
Uncertainty: a run is one CV fold x one noise draw. The CI of a method's mean uses the t distribution
over the k fold means (the folds, not the draws, are the independent units). Method comparisons
(compare.csv) are paired over folds, because the fold-to-fold variation is shared by all methods.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import datasets as D  # noqa: E402
import methods as M  # noqa: E402

sc = M.sc
DECK = ["housing", "wine", "airfoil", "concrete", "bike", "elevators", "pol", "kin40k", "protein"]
QUICK = ["housing", "airfoil", "concrete"]
ORDER = ["trivial", "nonprivate", "ssp", "adassp", "adassp_matched", "vfl", "vfl_uncorrected"]
LABEL = {"trivial": "Trivial (θ = 0)", "nonprivate": "Non-private", "ssp": "SSP",
         "adassp": "AdaSSP (published)", "adassp_matched": "AdaSSP (matched)",
         "vfl": "VFL, ours", "vfl_uncorrected": "VFL, uncorrected"}
PUB_INDEX = {"trivial": 0, "nonprivate": 1, "ssp": 2, "adassp": 6}   # rows of exp_results.mat
PAIRS = [("vfl", "adassp"), ("vfl", "adassp_matched"), ("vfl", "vfl_uncorrected")]


def t95(k):
    return float(stats.t.ppf(0.975, k - 1))


def paired(fold_a, fold_b):
    """Paired comparison over folds: mean difference a - b, its CI half-width, verdict for a."""
    dlt = fold_a - fold_b
    k = len(dlt)
    half = t95(k) * dlt.std(ddof=1) / np.sqrt(k)
    m = dlt.mean()
    return m, half, ("better" if m + half < 0 else "worse" if m - half > 0 else "no difference")


def run_dataset(name, eps_list, delta, R, k, d_R_arg, rho_star, seed):
    ds = D.load(name)
    X, y, n, d = ds.X, ds.y, ds.n, ds.d
    d_R = max(1, min(d - 1, d // 2 if d_R_arg is None else d_R_arg))
    folds = D.kfold(n, k, seed)
    mse = {m: np.zeros((len(eps_list), k, R)) for m in ORDER}      # every run: eps x fold x draw
    gate = np.zeros((len(eps_list), k))
    lam_vfl = np.zeros((len(eps_list), k))
    lam_ada = np.zeros((len(eps_list), k))
    for f, te in enumerate(folds):
        tr = np.setdiff1d(np.arange(n), te)
        Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]
        G0, c, yty = Xtr.T @ Xtr, Xtr.T @ ytr, float(ytr @ ytr)
        lmin = float(np.linalg.eigvalsh(G0 + np.eye(d))[0])
        m_triv = M.test_mse(Xte, yte, np.zeros(d))
        m_np = M.test_mse(Xte, yte, M.nonprivate(G0, c))
        for e, eps in enumerate(eps_list):
            rng = np.random.default_rng([seed, f, e, sum(map(ord, name))])
            mse["trivial"][e, f] = m_triv
            mse["nonprivate"][e, f] = m_np
            mse["ssp"][e, f] = M.test_mse(Xte, yte, M.ssp(G0, c, eps, delta, R, rng))
            th, lam = M.adassp(G0, c, eps, delta, R, rng, lmin=lmin, return_lam=True)
            mse["adassp"][e, f] = M.test_mse(Xte, yte, th)
            lam_ada[e, f] = lam.mean()
            sig_m = M.sigma_matched(eps, delta)[0]
            Gt, ct, _ = sc.release(G0, c, yty, d_R, sig_m, R, rng)
            th = M.adassp_matched(G0, Gt, ct, eps, delta, rng, lmin=lmin)
            mse["adassp_matched"][e, f] = M.test_mse(Xte, yte, th)
            sig = M.sigma_vfl(eps, delta)
            Gt, ct, _ = sc.release(G0, c, yty, d_R, sig, R, rng)
            b, bc, lam = M.vfl_fixed(Gt, ct, sig, d_R, len(tr), ell=0.0, rho_star=rho_star)
            ok = sc.rho_hat(Gt, lam, sig) >= 1.0
            b[~ok] = 0.0
            bc[~ok] = 0.0
            mse["vfl"][e, f] = M.test_mse(Xte, yte, bc)
            mse["vfl_uncorrected"][e, f] = M.test_mse(Xte, yte, b)
            gate[e, f] = ok.mean()
            lam_vfl[e, f] = lam
    return dict(name=name, n=n, d=d, d_R=d_R, runs=mse, mse={m: v.mean(2) for m, v in mse.items()},
                gate=gate, lam_vfl=lam_vfl, lam_ada=lam_ada)


def published_lookup():
    try:
        err, _ = D.published()
    except Exception as exc:                      # exp_results.mat missing from the clone
        print(f"(published results unavailable: {exc})")
        return None
    return lambda m, eps, name: (float(err[PUB_INDEX[m], D.EPS_LIST.index(eps), D.PUBLISHED_ORDER.index(name)])
                                 if m in PUB_INDEX and eps in D.EPS_LIST and name in D.PUBLISHED_ORDER else None)


def write_csv(path, results, eps_list, pub):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["dataset", "n", "d", "d_R", "eps", "method", "test_mse", "se_over_folds", "ci95_lo", "ci95_hi",
                    "run_p2.5", "run_p97.5", "ratio_to_nonprivate", "published_mse", "vfl_gate_pass", "lambda_mean"])
        for r in results:
            k = r["gate"].shape[1]
            for e, eps in enumerate(eps_list):
                npv = r["mse"]["nonprivate"][e].mean()
                for m in ORDER:
                    v = r["mse"][m][e]
                    p = pub(m, eps, r["name"]) if pub else None
                    lam = r["lam_vfl"][e].mean() if m.startswith("vfl") else (r["lam_ada"][e].mean() if m == "adassp" else "")
                    se = v.std(ddof=1) / np.sqrt(k)
                    lo, hi = np.percentile(r["runs"][m][e], [2.5, 97.5])
                    w.writerow([r["name"], r["n"], r["d"], r["d_R"], eps, m, f"{v.mean():.6g}",
                                f"{se:.3g}", f"{v.mean() - t95(k) * se:.6g}", f"{v.mean() + t95(k) * se:.6g}",
                                f"{lo:.6g}", f"{hi:.6g}", f"{v.mean() / npv:.4g}",
                                "" if p is None else f"{p:.6g}",
                                f"{r['gate'][e].mean():.3f}" if m.startswith("vfl") else "",
                                lam if lam == "" else f"{lam:.4g}"])


def write_compare(path, results, eps_list):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["dataset", "eps", "method", "vs", "mean_diff", "ci95_halfwidth", "rel_diff", "verdict"])
        for r in results:
            for e, eps in enumerate(eps_list):
                for a, b in PAIRS:
                    m, half, verdict = paired(r["mse"][a][e], r["mse"][b][e])
                    w.writerow([r["name"], eps, a, b, f"{m:.4g}", f"{half:.3g}",
                                f"{m / r['mse'][b][e].mean():.4g}", verdict])


def write_summary(path, results, eps_list, pub, args):
    lines = ["# AdaSSP vs revised VFL protocol: benchmark summary", "",
             f"Command: `{' '.join(sys.argv)}`  ", f"10-fold CV, R = {args.R} noise draws per fold, "
             f"δ = {args.delta:g}, ρ* = {args.rho_star:g}, R holds the first ⌊d/2⌋ features. "
             "Cells: mean test MSE (ratio to non-private).", ""]
    show = [e for e in (0.1, 1, 10) if e in eps_list] or eps_list[:1]
    for eps in show:
        e = eps_list.index(eps)
        lines += [f"## ε = {eps:g}", "", "| dataset | n | d | " + " | ".join(LABEL[m] for m in ORDER) + " | VFL gate pass |",
                  "|---|---:|---:|" + "---:|" * (len(ORDER) + 1)]
        for r in results:
            npv = r["mse"]["nonprivate"][e].mean()
            cells = [f"{r['mse'][m][e].mean():.4g} ({r['mse'][m][e].mean() / npv:.2f})" for m in ORDER]
            lines.append(f"| {r['name']} | {r['n']:,} | {r['d']} | " + " | ".join(cells) + f" | {r['gate'][e].mean():.0%} |")
        lines.append("")
    lines += ["## Paired comparisons (95% CI over folds)", "",
              "Number of datasets where the first method has lower / higher test MSE than the second, "
              "or no significant difference.", "",
              "| comparison | " + " | ".join(f"ε = {eps:g}" for eps in eps_list) + " |",
              "|---|" + "---|" * len(eps_list)]
    for a, b in PAIRS:
        cells = []
        for e in range(len(eps_list)):
            v = [paired(r["mse"][a][e], r["mse"][b][e])[2] for r in results]
            cells.append(f"{v.count('better')} / {v.count('worse')} / {v.count('no difference')}")
        lines.append(f"| {LABEL[a]} vs {LABEL[b]} | " + " | ".join(cells) + " |")
    lines.append("")
    if pub:
        lines += ["## Check against Wang's published run (code/exp_results.mat)", "",
                  "Ratio ours / published, median and range over datasets and ε. Values near 1 mean the "
                  "re-implementation reproduces Wang's numbers (different random folds and noise, so "
                  "small-n, small-ε cells differ most).", "", "| method | median | min | max |", "|---|---:|---:|---:|"]
        for m in ("trivial", "nonprivate", "adassp"):
            rat = [r["mse"][m][e].mean() / pub(m, eps, r["name"]) for r in results for e, eps in enumerate(eps_list)
                   if pub(m, eps, r["name"])]
            if rat:
                lines.append(f"| {LABEL[m]} | {np.median(rat):.3f} | {min(rat):.3f} | {max(rat):.3f} |")
        lines.append("")
    open(path, "w").write("\n".join(lines))


def make_figure(path, results, eps_list):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    style = {"trivial": dict(color="#8a8f98", ls=":", lw=1.5), "nonprivate": dict(color="#1f2937", ls="--", lw=1.5),
             "vfl": dict(color="#2a78d6", lw=2, marker="o", ms=4), "vfl_uncorrected": dict(color="#eb6834", lw=1.5, ls="--", marker="o", ms=3),
             "adassp": dict(color="#1baf7a", lw=2, marker="s", ms=4), "adassp_matched": dict(color="#eda100", lw=2, marker="^", ms=4),
             "ssp": dict(color="#e87ba4", lw=1.2, marker=".", ms=4)}
    nc = min(3, len(results))
    nr = int(np.ceil(len(results) / nc))
    fig, axes = plt.subplots(nr, nc, figsize=(4.2 * nc, 3.3 * nr), squeeze=False)
    for ax, r in zip(axes.flat, results):
        k = r["mse"]["vfl"].shape[1]
        for m in ("adassp", "vfl"):                           # middle 95% of individual runs
            lo, hi = np.percentile(r["runs"][m].reshape(len(eps_list), -1), [2.5, 97.5], axis=1)
            ax.fill_between(eps_list, lo, hi, color=style[m]["color"], alpha=0.12, lw=0)
        for m in ["trivial", "nonprivate", "ssp", "adassp", "adassp_matched", "vfl_uncorrected", "vfl"]:
            mean = r["mse"][m].mean(1)
            if m in ("trivial", "nonprivate", "ssp"):         # references; SSP's CI spans decades
                ax.plot(eps_list, mean, label=LABEL[m], **style[m])
            else:                                             # 95% CI of the mean, t over folds
                half = t95(k) * r["mse"][m].std(1, ddof=1) / np.sqrt(k)
                ax.errorbar(eps_list, mean, yerr=half, capsize=2, elinewidth=1, label=LABEL[m], **style[m])
        top = 3 * r["mse"]["trivial"].mean()
        ax.set(xscale="log", yscale="log", title=f"{r['name']}  (n = {r['n']:,}, d = {r['d']})", xlabel="ε")
        ax.set_ylim(top=top)
        ax.grid(True, which="major", color="#e5e7eb", lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    for ax in axes.flat[len(results):]:
        ax.axis("off")
    axes[0, 0].set_ylabel("test MSE (10-fold CV)")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=len(l), frameon=False, fontsize=9)
    fig.suptitle("Test MSE vs ε: AdaSSP (Wang 2018) and the revised VFL protocol\n"
                 "points: mean over folds × noise draws;  bars: 95% CI of the mean (t over the 10 folds);  "
                 "bands: middle 95% of individual runs (VFL, AdaSSP)", fontsize=11, y=0.995)
    fig.tight_layout(rect=(0, 0.05, 1, 0.965))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--quick", action="store_true", help="3 small datasets, eps in {0.1, 1, 10}, R = 10")
    g.add_argument("--all", action="store_true", help="all 29 datasets of Wang's published run")
    g.add_argument("--datasets", nargs="+", help="explicit dataset names")
    ap.add_argument("--eps", nargs="+", type=float, default=None, help=f"default {D.EPS_LIST}")
    ap.add_argument("--delta", type=float, default=D.DELTA)
    ap.add_argument("--R", type=int, default=None, help="noise draws per fold (default 50; 10 with --quick)")
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--d-r", type=int, default=None, help="number of features held by R (default floor(d/2))")
    ap.add_argument("--rho-star", type=float, default=2.0, help="ridge constant of the protocol (default 2)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(HERE, "..", "..", "reports", "adassp"))
    ap.add_argument("--no-figure", action="store_true")
    args = ap.parse_args()

    names = QUICK if args.quick else D.PUBLISHED_ORDER if args.all else args.datasets or DECK
    eps_list = args.eps or ([0.1, 1.0, 10.0] if args.quick else [float(e) for e in D.EPS_LIST])
    eps_list = [int(e) if float(e).is_integer() else e for e in eps_list]      # match EPS_LIST keys
    args.R = args.R or (10 if args.quick else 50)
    os.makedirs(args.out, exist_ok=True)

    results = []
    for nm in names:
        t = time.time()
        r = run_dataset(nm, eps_list, args.delta, args.R, args.folds, args.d_r, args.rho_star, args.seed)
        results.append(r)
        e1 = eps_list.index(1) if 1 in eps_list else len(eps_list) // 2
        print(f"{nm:12s} n={r['n']:7,d} d={r['d']:3d}  eps={eps_list[e1]:g}:  "
              + "  ".join(f"{m}={r['mse'][m][e1].mean():.4g}" for m in ORDER) + f"   [{time.time() - t:.1f}s]",
              flush=True)

    pub = published_lookup()
    write_csv(os.path.join(args.out, "results.csv"), results, eps_list, pub)
    write_compare(os.path.join(args.out, "compare.csv"), results, eps_list)
    write_summary(os.path.join(args.out, "summary.md"), results, eps_list, pub, args)
    if not args.no_figure:
        make_figure(os.path.join(args.out, "fig_mse_vs_eps.png"), results, eps_list)
    print(f"wrote {os.path.abspath(args.out)}/results.csv, compare.csv, summary.md" + ("" if args.no_figure else ", fig_mse_vs_eps.png"))


if __name__ == "__main__":
    main()
