"""
sweep_n.py -- test MSE against intersection size n on real data (Wang's UCI sets).

For each CV fold, the training folds are shuffled once and the first n rows form the matched
set, so the subsets are nested across n; the held-out fold is the test set. At each n the
revised VFL protocol (bias-corrected, fixed ridge, release gate) and AdaSSP (published) are run
for each eps, with non-private and trivial references. Everything is reported relative to the
non-private test MSE at the same n and fold ("privacy cost"); 95% CIs use t over the folds.

Usage (repo root, WANG_REPO set as for run_benchmark.py):
  python experiments/adassp/sweep_n.py                  # bike, elevators, pol, protein: ~1 min
  python experiments/adassp/sweep_n.py --quick          # 2 datasets, 5 values of n, R = 10
Outputs (default reports/adassp/sweep/): sweep.csv, crossings.csv, summary.md, fig_sweep_n.png.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import datasets as D  # noqa: E402
import methods as M  # noqa: E402
from run_benchmark import t95  # noqa: E402

sc = M.sc
DEFAULT = ["bike", "elevators", "pol", "protein"]     # n >= 15k and a real linear signal
EPS = [0.5, 1, 2, 4]
THRESH = [1.10, 1.25]                                 # "within 10% / 25% of non-private"


def run(name, eps_list, n_grid, R, k, rho_star, delta, seed):
    ds = D.load(name)
    X, y, d = ds.X, ds.y, ds.d
    d_R = d // 2
    folds = D.kfold(ds.n, k, seed)
    nmax = min(ds.n - len(te) for te in folds)
    grid = [int(v) for v in n_grid if v <= nmax]
    if grid[-1] != nmax:
        grid.append(nmax)
    E, G = len(eps_list), len(grid)
    out = {m: np.zeros((E, G, k)) for m in ("vfl", "adassp")}      # fold-mean ratio to non-private
    mse_np = np.zeros((G, k))
    ratio_triv = np.zeros((G, k))
    gate = np.zeros((E, G, k))
    for f, te in enumerate(folds):
        pool = np.random.default_rng([seed, f, 7]).permutation(np.setdiff1d(np.arange(ds.n), te))
        Xte, yte = X[te], y[te]
        for g, n in enumerate(grid):
            tr = pool[:n]
            Xtr, ytr = X[tr], y[tr]
            G0, c, yty = Xtr.T @ Xtr, Xtr.T @ ytr, float(ytr @ ytr)
            lmin = float(np.linalg.eigvalsh(G0 + np.eye(d))[0])
            m_np = M.test_mse(Xte, yte, M.nonprivate(G0, c))
            mse_np[g, f] = m_np
            ratio_triv[g, f] = M.test_mse(Xte, yte, np.zeros(d)) / m_np
            for e, eps in enumerate(eps_list):
                rng = np.random.default_rng([seed, f, g, e, sum(map(ord, name))])
                th = M.adassp(G0, c, eps, delta, R, rng, lmin=lmin)
                out["adassp"][e, g, f] = M.test_mse(Xte, yte, th).mean() / m_np
                sig = M.sigma_vfl(eps, delta)
                Gt, ct, _ = sc.release(G0, c, yty, d_R, sig, R, rng)
                _, bc, lam = M.vfl_fixed(Gt, ct, sig, d_R, n, ell=0.0, rho_star=rho_star)
                ok = sc.rho_hat(Gt, lam, sig) >= 1.0
                bc[~ok] = 0.0
                out["vfl"][e, g, f] = M.test_mse(Xte, yte, bc).mean() / m_np
                gate[e, g, f] = ok.mean()
    return dict(name=name, n=ds.n, d=d, grid=grid, ratio=out, mse_np=mse_np, ratio_triv=ratio_triv, gate=gate)


def crossing(grid, curve, thr):
    """Smallest n (log-linear interpolation) with curve <= thr and staying below; None if never."""
    below = curve <= thr
    if not below[-1]:
        return None
    i = len(curve) - 1
    while i > 0 and below[i - 1]:
        i -= 1
    if i == 0:
        return float(grid[0])
    x0, x1, y0, y1 = np.log(grid[i - 1]), np.log(grid[i]), curve[i - 1], curve[i]
    return float(np.exp(x0 + (thr - y0) * (x1 - x0) / (y1 - y0)))


def fmt_n(v, nmax):
    if v is None:
        return f"> {nmax / 1000:.0f}k" if nmax >= 1000 else f"> {nmax}"
    return f"{v:,.0f}" if v < 1000 else f"{v / 1000:.1f}k" if v < 10000 else f"{v / 1000:.0f}k"


def write_outputs(results, eps_list, out, args):
    with open(os.path.join(out, "sweep.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["dataset", "n_matched", "eps", "method", "ratio_to_nonprivate", "ci95_lo", "ci95_hi", "nonprivate_mse",
                    "trivial_ratio", "vfl_gate_pass"])
        for r in results:
            k = r["mse_np"].shape[1]
            for g, n in enumerate(r["grid"]):
                for e, eps in enumerate(eps_list):
                    for m in ("vfl", "adassp"):
                        v = r["ratio"][m][e, g]
                        half = t95(k) * v.std(ddof=1) / np.sqrt(k)
                        w.writerow([r["name"], n, eps, m, f"{v.mean():.5g}", f"{v.mean() - half:.5g}", f"{v.mean() + half:.5g}",
                                    f"{r['mse_np'][g].mean():.5g}", f"{r['ratio_triv'][g].mean():.4g}",
                                    f"{r['gate'][e, g].mean():.3f}" if m == "vfl" else ""])
    with open(os.path.join(out, "crossings.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["dataset", "threshold", "eps", "method", "n_cross", "n_max"])
        for r in results:
            for thr in THRESH:
                for e, eps in enumerate(eps_list):
                    for m in ("vfl", "adassp"):
                        v = crossing(r["grid"], r["ratio"][m][e].mean(1), thr)
                        w.writerow([r["name"], thr, eps, m, "" if v is None else f"{v:.0f}", r["grid"][-1]])
    lines = ["# Intersection-size sweep on real data", "",
             f"Command: `{' '.join(sys.argv)}`  ",
             f"10-fold CV; nested random subsets of the training folds; R = {args.R} noise draws per fold and n; "
             f"δ = {args.delta:g}; ρ* = {args.rho_star:g}. Ratios are test MSE ÷ non-private test MSE at the same n.", ""]
    for thr in THRESH:
        lines += [f"## Matched records needed to come within {100 * (thr - 1):.0f}% of non-private",
                  "", "VFL, ours (AdaSSP published in brackets). Log-linear interpolation of the mean curve.", "",
                  "| dataset | n available | " + " | ".join(f"ε = {e:g}" for e in eps_list) + " |",
                  "|---|---:|" + "---:|" * len(eps_list)]
        for r in results:
            nmax = r["grid"][-1]
            cells = []
            for e in range(len(eps_list)):
                a = crossing(r["grid"], r["ratio"]["vfl"][e].mean(1), thr)
                b = crossing(r["grid"], r["ratio"]["adassp"][e].mean(1), thr)
                cells.append(f"{fmt_n(a, nmax)} ({fmt_n(b, nmax)})")
            lines.append(f"| {r['name']} | {nmax:,} | " + " | ".join(cells) + " |")
        lines.append("")
    lines += ["## Ratio at selected n (VFL, ours / AdaSSP published)", "",
              "| dataset | n | " + " | ".join(f"ε = {e:g}" for e in eps_list) + " |", "|---|---:|" + "---:|" * len(eps_list)]
    for r in results:
        for n in sorted({g for g in r["grid"] if g in (1000, 10000)} | {r["grid"][-1]}):
            g = r["grid"].index(n)
            lines.append(f"| {r['name']} | {n:,} | " + " | ".join(
                f"{r['ratio']['vfl'][e, g].mean():.3f} / {r['ratio']['adassp'][e, g].mean():.3f}" for e in range(len(eps_list))) + " |")
    lines.append("")
    open(os.path.join(out, "summary.md"), "w").write("\n".join(lines))


def figure(results, eps_list, path, deck=False):
    """deck=True: no title, larger type, for a slide."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker  # noqa: F401
    from matplotlib.lines import Line2D
    ramp = ["#86b6ef", "#5598e7", "#2a78d6", "#0d366b"][-len(eps_list):]
    nc = 2
    nr = int(np.ceil(len(results) / nc))
    plt.rcParams.update({"font.size": 11 if deck else 10})
    fig, axes = plt.subplots(nr, nc, figsize=(8.4, 3.3 * nr + 0.4) if deck else (10.5, 3.9 * nr), squeeze=False)
    for ax, r in zip(axes.flat, results):
        grid = np.array(r["grid"])
        k = r["mse_np"].shape[1]
        ax.plot(grid, r["ratio_triv"].mean(1), color="#8a8f98", ls=":", lw=1.5)
        ax.axhline(1.0, color="#1f2937", lw=1.2)
        ax.axhline(1.1, color="#b8bec8", lw=1.0, ls=(0, (1, 1.5)))
        for e, eps in enumerate(eps_list):
            for m, ls, lw, mk in (("adassp", (0, (4, 2)), 1.3, None), ("vfl", "-", 2.0, "o")):
                v = r["ratio"][m][e]
                half = t95(k) * v.std(1, ddof=1) / np.sqrt(k)
                ax.errorbar(grid, v.mean(1), yerr=half, color=ramp[e], ls=ls, lw=lw, marker=mk, ms=3.5,
                            capsize=2, elinewidth=0.9)
        ax.set(xscale="log", yscale="log", xlabel="matched records n (log)",
               title=f"{r['name']}  (d = {r['d']})" if deck else f"{r['name']}  (d = {r['d']}, up to n = {grid[-1]:,})")
        top = max(1.3, min(6, 1.15 * r["ratio_triv"].mean(1).max()))
        ax.set_ylim(0.97, top)
        ticks = [t for t in (1, 1.1, 1.25, 1.5, 2, 3, 4, 5) if t <= top]
        ax.yaxis.set_major_locator(matplotlib.ticker.FixedLocator(ticks))
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.grid(True, which="major", color="#e5e7eb", lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    for ax in axes.flat[len(results):]:
        ax.axis("off")
    for ax in axes[:, 0]:
        ax.set_ylabel("MSE ÷ non-private" if deck else "test MSE ÷ non-private at the same n")
    handles = [Line2D([], [], color=c, lw=2.5, label=f"ε = {e:g}") for c, e in zip(ramp, eps_list)] + [
        Line2D([], [], color="#1f2937", lw=2, marker="o", ms=4, label="VFL, ours"),
        Line2D([], [], color="#1f2937", lw=1.3, ls=(0, (4, 2)), label="AdaSSP (published)"),
        Line2D([], [], color="#8a8f98", ls=":", lw=1.5, label="Trivial (θ = 0)"),
        Line2D([], [], color="#1f2937", lw=1.2, label="Non-private (= 1)"),
        Line2D([], [], color="#b8bec8", lw=1.0, ls=(0, (1, 1.5)), label="within 10%")]
    fig.legend(handles=handles, loc="lower center", ncol=4 if deck else 5, frameon=False, fontsize=10 if deck else 9)
    if not deck:
        fig.suptitle("Privacy cost against intersection size on real data\n"
                     "mean over 10 folds × noise draws; bars: 95% CI over folds", fontsize=11, y=0.995)
    fig.tight_layout(rect=(0, 0.1 if deck else 0.07, 1, 1 if deck else 0.965))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=None)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--eps", nargs="+", type=float, default=None)
    ap.add_argument("--R", type=int, default=None, help="noise draws per fold and n (default 40; 10 with --quick)")
    ap.add_argument("--points", type=int, default=14, help="values of n on the log grid")
    ap.add_argument("--rho-star", type=float, default=2.0)
    ap.add_argument("--delta", type=float, default=D.DELTA)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(HERE, "..", "..", "reports", "adassp", "sweep"))
    args = ap.parse_args()
    names = args.datasets or (DEFAULT[:2] if args.quick else DEFAULT)
    eps_list = [int(e) if float(e).is_integer() else e for e in (args.eps or EPS)]
    args.R = args.R or (10 if args.quick else 40)
    n_grid = np.unique(np.concatenate([np.round(np.geomspace(100, 50000, 5 if args.quick else args.points)).astype(int),
                                       [1000, 10000]]))           # 1k and 10k for the summary table
    os.makedirs(args.out, exist_ok=True)
    results = []
    for nm in names:
        t = time.time()
        r = run(nm, eps_list, n_grid, args.R, 10, args.rho_star, args.delta, args.seed)
        results.append(r)
        print(f"{nm:10s} grid {r['grid'][0]}..{r['grid'][-1]} ({len(r['grid'])} points)  [{time.time() - t:.1f}s]", flush=True)
    write_outputs(results, eps_list, args.out, args)
    figure(results, eps_list, os.path.join(args.out, "fig_sweep_n.png"))
    figure(results, eps_list, os.path.join(args.out, "fig_sweep_n_deck.png"), deck=True)
    print(f"wrote {os.path.abspath(args.out)}/sweep.csv, summary.md, fig_sweep_n.png")


if __name__ == "__main__":
    main()
