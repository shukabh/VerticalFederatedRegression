"""
coef_accuracy_figures.py -- coefficient-accuracy figures for the revised Scenario-B protocol
(paper/main.tex, Sections 5 and 7), answering the reviewer request:

  "a graphic which compares our VFL beta (with bias correction) with true beta (like RMSE)
   against the intersection size (log scale), for different epsilon values, with confidence
   intervals showing the variation across runs ... the beta coefficients would have different
   scales, so is there a way to compare them separately?"

Monte Carlo design (uses experiments/sim_core.py, read-only):
  * n in {100, ..., 100000}, eps in {0.5, 1, 2, 4, 8}, delta = 1e-5, replace-one sigma.
  * Each replicate draws a NEW matched cohort of size n from the known population (features on
    heterogeneous raw scales, standardized with public constants, clipped to committed bounds).
    The dataset is shared across eps; each eps gets its own independent DP release.
  * Estimators: non-private OLS on the true matches (the floor; plaintext linkage + OLS gives the
    same number), VFL bias-corrected (main), VFL uncorrected. Also OLS on the *unclipped* data, to
    show the (small) cost of the committed bounds themselves.
  * Fixed ridge lambda = max{0, 2 rho* sigma sqrt(p) - n ell}, rho* = 2, ell = 0.8 lam_min_pop.
  * Release gate rho_hat >= 1 is recorded (estimate still computed; the withhold rate is reported).
  * Wald intervals from sim_core.first_order_cov (sampling + mechanism).

Outputs go to reports/coef_accuracy/: figures (PNG 200 dpi + PDF), CSV/JSON summaries and the raw
per-replicate estimates (npz).

Usage:
  python experiments/coef_accuracy_figures.py                 # full run (r2 = 0.5, 0.3, 0.9)
  python experiments/coef_accuracy_figures.py --plots-only    # re-plot from saved npz
  python experiments/coef_accuracy_figures.py --reps 20 --out /tmp/x   # quick smoke test
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sim_core as sc  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter, NullLocator  # noqa: E402

# =============================================================================
# Configuration
# =============================================================================
NS = np.array([100, 250, 500, 1000, 2500, 5000, 10000, 25000, 50000, 100000])
EPS = np.array([0.5, 1.0, 2.0, 4.0, 8.0])
DELTA = 1e-5
RHO_STAR = 2.0
ELL_FRAC = 0.8                 # committed eigenvalue floor ell = 0.8 * pop.lam_min_pop
POP_SEED = 0                   # the population (true beta, public constants, bounds)
BASE_SEED = 20260930           # Monte Carlo streams
N_BOOT = 2000
Z975 = 1.959963984540054

# style (conventions given by the project)
EPS_COLORS = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]
OLS_COLOR = "#3b3b3b"
C_CORR, C_UNCORR = "#2a78d6", "#eb6834"
INK, INK2 = "#0b0b0b", "#52514e"
GRID = "#e4e3df"
SPINE = "#bdbcb7"
REF = "#9a9994"
BAND_ALPHA = 0.18
LW, MS = 2.0, 5.0

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9.5,
    "axes.edgecolor": SPINE, "axes.linewidth": 0.8, "axes.labelcolor": INK,
    "axes.titlecolor": INK, "axes.titlesize": 10, "axes.labelsize": 9.5,
    "xtick.color": INK2, "ytick.color": INK2, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "xtick.major.size": 3, "ytick.major.size": 3, "xtick.minor.size": 1.5, "ytick.minor.size": 1.5,
    "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "legend.frameon": False, "legend.fontsize": 8.5, "text.color": INK,
    "savefig.facecolor": "white", "figure.facecolor": "white",
    "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
})

KIND_LABEL = {"gaussian": "score", "binary": "binary", "lognormal": "income ($)",
              "count": "count", "age": "age (yr)"}


def coef_labels(pop):
    """Short names for the p = 1 + d_R + d_O coefficients (intercept first)."""
    names, blocks = ["intercept"], ["R"]
    for j, t in enumerate(pop.types_R):
        names.append(f"R{j + 1} {KIND_LABEL[t]}"); blocks.append("R")
    seen_ln = 0
    for j, t in enumerate(pop.types_O):
        lab = KIND_LABEL[t]
        if t == "lognormal":
            seen_ln += 1
            lab = "income ($)" if seen_ln == 1 else "assets ($)"
        names.append(f"O{j + 1} {lab}"); blocks.append("O")
    return names, blocks


# =============================================================================
# Simulation
# =============================================================================

def draw_both(pop, n, rng):
    """Mirror of Population.draw (identical RNG consumption and output), additionally returning
    the unclipped design/outcome so that the no-bounds plaintext OLS can be reported."""
    X = pop._features(n, rng)
    Zs = (X - pop.mu) / pop.s
    y = Zs @ pop.beta_std + pop.noise_sd * rng.standard_normal(n)
    zR, zO = Zs[:, : pop.d_R], Zs[:, pop.d_R:]
    nR, nO = np.linalg.norm(zR, axis=1), np.linalg.norm(zO, axis=1)
    clip = np.array([(nR > pop.B_R_rows).mean(), (nO > pop.B_O).mean(), (np.abs(y) > pop.B_y).mean()])
    zRc = zR * np.minimum(1.0, pop.B_R_rows / np.maximum(nR, 1e-300))[:, None]
    zOc = zO * np.minimum(1.0, pop.B_O / np.maximum(nO, 1e-300))[:, None]
    yc = np.clip(y, -pop.B_y, pop.B_y)
    one = np.ones((n, 1))
    return np.hstack([one, zRc, zOc]), yc, np.hstack([one, Zs]), y, clip


def clipped_target(pop, rng, total=4_000_000, chunk=250_000):
    """Population OLS coefficient on the clipped data (the estimand the protocol targets)."""
    G = np.zeros((pop.p, pop.p)); c = np.zeros(pop.p)
    for _ in range(total // chunk):
        Z, y, _ = pop.draw(chunk, rng)
        G += Z.T @ Z; c += Z.T @ y
    return np.linalg.solve(G, c), float(np.linalg.eigvalsh(G / total)[0])


def simulate(r2, reps, seed_offset=0, verbose=True):
    pop = sc.Population(r2=r2, seed=POP_SEED)
    p, dR = pop.p, pop.d_R_block
    ell = ELL_FRAC * pop.lam_min_pop
    sig = np.array([pop.sigma_for(e, DELTA)[0] for e in EPS])
    Dlt = pop.sigma_for(EPS[0], DELTA)[1]
    beta = pop.beta_true
    ss = np.random.SeedSequence([BASE_SEED, int(round(r2 * 100)), seed_offset])
    ss_target, ss_mc = ss.spawn(2)
    beta_clip, lam_min_clip = clipped_target(pop, np.random.default_rng(ss_target))

    nN, nE = len(NS), len(EPS)
    out = {
        "beta_ols": np.empty((nN, reps, p)), "se_ols": np.empty((nN, reps, p)),
        "beta_olsu": np.empty((nN, reps, p)),
        "beta_bc": np.empty((nN, nE, reps, p)), "beta_un": np.empty((nN, nE, reps, p)),
        "se_bc": np.empty((nN, nE, reps, p)), "se_un": np.empty((nN, nE, reps, p)),
        "rho_hat": np.empty((nN, nE, reps)), "rho_clean": np.empty((nN, nE, reps)),
        "lam": np.empty((nN, nE)), "clip": np.empty((nN, 3)),
        "lam_min_sample": np.empty((nN, reps)),
    }
    for i, (n, ss_n) in enumerate(zip(NS, ss_mc.spawn(nN))):
        t0 = time.time()
        ss_data, *ss_eps = ss_n.spawn(1 + nE)
        rng_d = np.random.default_rng(ss_data)
        Gs = np.empty((reps, p, p)); cs = np.empty((reps, p)); ytys = np.empty(reps)
        clips = np.empty((reps, 3))
        for r in range(reps):
            Z, y, Zu, yu, clips[r] = draw_both(pop, int(n), rng_d)
            G, c, yty = sc.stats(Z, y)
            Gs[r], cs[r], ytys[r] = G, c, yty
            b_ols = np.linalg.solve(G, c)
            rss = yty - 2 * c @ b_ols + b_ols @ G @ b_ols
            s2 = max(rss, 0.0) / (n - p)
            out["beta_ols"][i, r] = b_ols
            out["se_ols"][i, r] = np.sqrt(s2 * np.diag(np.linalg.inv(G)))
            out["beta_olsu"][i, r] = np.linalg.lstsq(Zu, yu, rcond=None)[0]
        out["clip"][i] = clips.mean(0)
        out["lam_min_sample"][i] = np.linalg.eigvalsh(Gs / n)[:, 0]
        for k, (e, s_k) in enumerate(zip(EPS, sig)):
            rng_e = np.random.default_rng(ss_eps[k])
            # independent release per replicate: release() on zero statistics returns exactly the
            # mechanism's noise (symmetric masked E, f, y'y noise); add it to each replicate's stats.
            E, f, gnoise = sc.release(np.zeros((p, p)), np.zeros(p), 0.0, dR, s_k, reps, rng_e)
            Gt, ct, ytyt = Gs + E, cs + f, ytys + gnoise
            lam = sc.fixed_lambda(s_k, p, int(n), ell, RHO_STAR)
            b, bc, P = sc.solve(Gt, ct, lam, s_k, dR)
            cov_bc, _ = sc.first_order_cov(Gt, ct, ytyt, bc, P, s_k, int(n), dR)
            cov_un, _ = sc.first_order_cov(Gt, ct, ytyt, b, P, s_k, int(n), dR)
            d_bc = np.einsum("rii->ri", cov_bc); d_un = np.einsum("rii->ri", cov_un)
            out["beta_bc"][i, k], out["beta_un"][i, k] = bc, b
            out["se_bc"][i, k] = np.sqrt(np.where(d_bc > 0, d_bc, np.nan))
            out["se_un"][i, k] = np.sqrt(np.where(d_un > 0, d_un, np.nan))
            out["rho_hat"][i, k] = sc.rho_hat(Gt, lam, s_k)
            out["rho_clean"][i, k] = sc.rho_hat(Gs, lam, s_k)
            out["lam"][i, k] = lam
        if verbose:
            print(f"  r2={r2} n={n:>6d}: {time.time() - t0:5.1f}s  clip={np.round(out['clip'][i], 4)}",
                  flush=True)
    meta = {
        "r2": r2, "reps": reps, "p": p, "d_R_block": dR, "ell": ell, "lam_min_pop": pop.lam_min_pop,
        "lam_min_clipped_pop": lam_min_clip, "sigma": sig.tolist(), "Delta_rep": Dlt,
        "B_R": pop.B_R, "B_O": pop.B_O, "B_y": pop.B_y, "beta_true": beta.tolist(),
        "beta_clip": beta_clip.tolist(), "beta_raw": pop.beta_raw.tolist(), "s": pop.s.tolist(),
        "y_sd": pop.y_sd, "types": pop.types,
        "n_ridge_off": (2 * RHO_STAR * sig * np.sqrt(p) / ell).tolist(),
    }
    return out, meta, pop


# =============================================================================
# Metrics
# =============================================================================

def boot_weights(reps, B, rng):
    """Multinomial resampling weights (B, reps); row b gives counts of each replicate."""
    return rng.multinomial(reps, np.full(reps, 1.0 / reps), size=B).astype(float)


def interp_cross(ns, ratio, target=2.0):
    """Smallest n (log-log interpolated) after which ratio stays <= target."""
    ok = ratio <= target
    if ok.all():
        return float(ns[0]), "<="
    if not ok[-1]:
        return float("nan"), ">"
    j = int(np.max(np.nonzero(~ok)[0]))       # last violation; j + 1 is the first stay-ok point
    x0, x1 = np.log(ns[j]), np.log(ns[j + 1])
    y0, y1 = np.log(ratio[j]), np.log(ratio[j + 1])
    t = (np.log(target) - y0) / (y1 - y0)
    return float(np.exp(x0 + t * (x1 - x0))), "="


def compute_metrics(out, meta, names, blocks, rng):
    beta = np.array(meta["beta_true"]); bclip = np.array(meta["beta_clip"])
    p = len(beta); sl = slice(1, p)
    reps = out["beta_ols"].shape[1]
    W = boot_weights(reps, N_BOOT, rng)                     # shared across methods: paired
    nN, nE = len(NS), len(EPS)

    def run_rmse(b):                                        # per-run slope RMSE, (..., reps)
        return np.sqrt(np.mean((b[..., sl] - beta[sl]) ** 2, axis=-1))

    def agg(b, se=None):
        """Aggregate one method at one (n, eps): b (reps, p)."""
        err = b - beta
        rr = run_rmse(b)
        mse_run = np.mean(err[:, sl] ** 2, axis=1)
        boot_mean = W @ rr / reps
        boot_pool = np.sqrt(W @ mse_run / reps)
        d = {
            "rmse_median": float(np.median(rr)), "rmse_p025": float(np.percentile(rr, 2.5)),
            "rmse_p975": float(np.percentile(rr, 97.5)), "rmse_mean": float(rr.mean()),
            "rmse_mean_ci_lo": float(np.percentile(boot_mean, 2.5)),
            "rmse_mean_ci_hi": float(np.percentile(boot_mean, 97.5)),
            "rmse_pooled": float(np.sqrt(mse_run.mean())),
            "rmse_pooled_ci_lo": float(np.percentile(boot_pool, 2.5)),
            "rmse_pooled_ci_hi": float(np.percentile(boot_pool, 97.5)),
            "rmse_clip_target_pooled": float(np.sqrt(np.mean((b[:, sl] - bclip[sl]) ** 2))),
            "intercept_rmse": float(np.sqrt(np.mean(err[:, 0] ** 2))),
            "intercept_bias": float(np.mean(err[:, 0])),
        }
        if se is not None:
            cov = np.abs(err) <= Z975 * se                   # NaN se -> not covered
            d["coverage_mean_slopes"] = float(np.mean(cov[:, sl]))
            d["coverage_min_slope"] = float(np.min(cov[:, sl].mean(0)))
            d["se_invalid_frac"] = float(np.mean(~np.isfinite(se)))
        return d, rr, mse_run

    overall, percoef = [], []
    curves = {"rr": {}, "mse_run": {}}
    for i, n in enumerate(NS):
        b_ols = out["beta_ols"][i]
        sd_ols = b_ols.std(0, ddof=1)                        # SE_OLS,j (empirical)
        # bootstrap of sd_ols for paired efficiency-ratio CIs
        m1 = W @ b_ols / reps; m2 = W @ b_ols ** 2 / reps
        sd_ols_boot = np.sqrt(np.maximum(m2 - m1 ** 2, 0) * reps / (reps - 1))
        methods = [("ols", None, b_ols, out["se_ols"][i]),
                   ("ols_unclipped", None, out["beta_olsu"][i], None)]
        for k, e in enumerate(EPS):
            methods += [("vfl_bc", k, out["beta_bc"][i, k], out["se_bc"][i, k]),
                        ("vfl_uncorrected", k, out["beta_un"][i, k], out["se_un"][i, k])]
        ols_d = None
        for mname, k, b, se in methods:
            d, rr, mse_run = agg(b, se)
            curves["rr"][(mname, i, k)] = rr
            curves["mse_run"][(mname, i, k)] = mse_run
            row = {"r2": meta["r2"], "n": int(n), "eps": (float(EPS[k]) if k is not None else None),
                   "method": mname, **d}
            if mname == "ols":
                ols_d = d
            if k is not None:
                lam = float(out["lam"][i, k]); rh = out["rho_hat"][i, k]
                row.update({"lambda": lam, "ridge_active": bool(lam > 0),
                            "rho_hat_median": float(np.median(rh)),
                            "rho_hat_p025": float(np.percentile(rh, 2.5)),
                            "rho_clean_median": float(np.median(out["rho_clean"][i, k])),
                            "gate_withhold_frac": float(np.mean(rh < 1.0)),
                            "ratio_to_ols_median": d["rmse_median"] / ols_d["rmse_median"],
                            "ratio_to_ols_pooled": d["rmse_pooled"] / ols_d["rmse_pooled"]})
            overall.append(row)
            # per-coefficient
            err = b - beta
            rmse_j = np.sqrt(np.mean(err ** 2, 0))
            boot_rmse_j = np.sqrt(W @ err ** 2 / reps)
            eff_boot = boot_rmse_j / sd_ols_boot
            cov = (np.abs(err) <= Z975 * se) if se is not None else None
            cov_clip = (np.abs(b - bclip) <= Z975 * se) if se is not None else None
            for j in range(p):
                bj = beta[j]
                pc = {"r2": meta["r2"], "n": int(n), "eps": row["eps"], "method": mname, "j": j,
                      "coef": names[j], "block": blocks[j], "beta_std": float(bj),
                      "beta_raw": (float(meta["beta_raw"][j - 1]) if j > 0 else None),
                      "rmse_std": float(rmse_j[j]), "bias_std": float(err[:, j].mean()),
                      "sd": float(b[:, j].std(ddof=1)), "sd_ols": float(sd_ols[j]),
                      "eff_ratio": float(rmse_j[j] / sd_ols[j]),
                      "eff_ratio_ci_lo": float(np.percentile(eff_boot[:, j], 2.5)),
                      "eff_ratio_ci_hi": float(np.percentile(eff_boot[:, j], 97.5)),
                      "rmse_std_ci_lo": float(np.percentile(boot_rmse_j[:, j], 2.5)),
                      "rmse_std_ci_hi": float(np.percentile(boot_rmse_j[:, j], 97.5))}
                if j > 0:
                    pc["raw_rmse"] = float(meta["y_sd"] * rmse_j[j] / meta["s"][j - 1])
                    pc["sign_rate"] = float(np.mean(np.sign(b[:, j]) == np.sign(bj)))
                    if abs(bj) >= 0.1:
                        rel = np.abs(err[:, j]) / abs(bj)
                        pc["relerr_median"] = float(np.median(rel))
                        pc["relerr_p90"] = float(np.percentile(rel, 90))
                        pc["rel_rmse"] = float(rmse_j[j] / abs(bj))
                if cov is not None:
                    pc["coverage"] = float(np.mean(cov[:, j]))
                    pc["coverage_clip_target"] = float(np.mean(cov_clip[:, j]))
                percoef.append(pc)
    # paired bootstrap for the corrected/uncorrected pooled-RMSE ratio
    ratio = []
    for i, n in enumerate(NS):
        for k, e in enumerate(EPS):
            a = curves["mse_run"][("vfl_bc", i, k)]; u = curves["mse_run"][("vfl_uncorrected", i, k)]
            rb = np.sqrt((W @ a) / (W @ u))
            ratio.append({"r2": meta["r2"], "n": int(n), "eps": float(e),
                          "ratio_pooled": float(np.sqrt(a.mean() / u.mean())),
                          "ci_lo": float(np.percentile(rb, 2.5)), "ci_hi": float(np.percentile(rb, 97.5)),
                          "median_run_ratio": float(np.median(np.sqrt(a / u))),
                          "frac_runs_bc_better": float(np.mean(a < u)),
                          "ridge_active": bool(out["lam"][i, k] > 0)})
    return overall, percoef, ratio


def lookup(rows, **kw):
    for r in rows:
        if all((r.get(k) == v) if not isinstance(v, float) else
               (r.get(k) is not None and abs(r.get(k) - v) < 1e-12) for k, v in kw.items()):
            return r
    raise KeyError(kw)


def key_numbers(overall, meta):
    kn = {"rmse": {}, "n_within_2x_floor": {}, "ridge_active_n": {}, "gate": {}}
    for n in (1000, 10000, 100000):
        o = lookup(overall, n=n, method="ols")
        kn["rmse"][str(n)] = {"ols_median": o["rmse_median"], "ols_pooled": o["rmse_pooled"]}
        for e in EPS:
            v = lookup(overall, n=n, eps=float(e), method="vfl_bc")
            kn["rmse"][str(n)][f"eps{e:g}"] = {k: v[k] for k in
                                             ("rmse_median", "rmse_pooled", "rmse_p025", "rmse_p975",
                                              "ratio_to_ols_median", "ratio_to_ols_pooled",
                                              "coverage_mean_slopes")}
    for k, e in enumerate(EPS):
        rat_med = np.array([lookup(overall, n=int(n), eps=float(e), method="vfl_bc")["ratio_to_ols_median"]
                            for n in NS])
        rat_pool = np.array([lookup(overall, n=int(n), eps=float(e), method="vfl_bc")["ratio_to_ols_pooled"]
                             for n in NS])
        rat_unc = np.array([lookup(overall, n=int(n), eps=float(e), method="vfl_bc")["rmse_pooled"]
                            / lookup(overall, n=int(n), method="ols_unclipped")["rmse_pooled"] for n in NS])
        kn["n_within_2x_floor"][f"eps{e:g}"] = {"median_line": interp_cross(NS, rat_med),
                                               "pooled": interp_cross(NS, rat_pool),
                                               "pooled_vs_unclipped_ols": interp_cross(NS, rat_unc)}
        kn["ridge_active_n"][f"eps{e:g}"] = {"n_ridge_off": meta["n_ridge_off"][k],
                                            "grid_n_active": [int(n) for n in NS if n < meta["n_ridge_off"][k]]}
        wf = {int(n): lookup(overall, n=int(n), eps=float(e), method="vfl_bc")["gate_withhold_frac"]
              for n in NS}
        kn["gate"][f"eps{e:g}"] = {"max_withhold": max(wf.values()),
                                  "cells_with_withhold": {n: f for n, f in wf.items() if f > 0}}
    return kn


# =============================================================================
# Plot helpers
# =============================================================================

def fmt_n(x, _pos=None):
    if x >= 1000:
        v = x / 1000
        return f"{v:g}k"
    return f"{x:g}"


def style(ax, logx=True, logy=False, grid_y=True, grid_x=True):
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    if logx:
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(FixedLocator([100, 1000, 10000, 100000]))
        ax.xaxis.set_major_formatter(FuncFormatter(fmt_n))
        ax.xaxis.set_minor_formatter(NullFormatter())
    if logy:
        ax.set_yscale("log")
    ax.grid(False)
    if grid_y:
        ax.grid(True, axis="y", which="major", color=GRID, lw=0.7, zorder=0)
    if grid_x:
        ax.grid(True, axis="x", which="major", color=GRID, lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", which="both", colors=INK2)


def log_y_ticks(ax, lo, hi):
    """Decade + 2/5 ticks with plain labels (0.001, 0.002, ...)."""
    ticks = []
    for d in range(int(np.floor(np.log10(lo))) - 1, int(np.ceil(np.log10(hi))) + 1):
        for m in (1, 2, 5):
            t = m * 10.0 ** d
            if lo <= t <= hi:
                ticks.append(t)
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:g}"))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_ylim(lo, hi)


def save(fig, outdir, name):
    fig.savefig(os.path.join(outdir, name + ".png"), dpi=200, bbox_inches="tight")
    fig.savefig(os.path.join(outdir, name + ".pdf"), bbox_inches="tight")
    plt.close(fig)


def eps_label(e):
    return f"ε = {e:g}"


def ridge_mask(meta, k):
    """Grid points where the fixed ridge is active (lambda > 0) for eps index k."""
    return NS < meta["n_ridge_off"][k]


def eps_line(ax, y, k, meta, ms=MS, z=5):
    """eps-coloured 2px line; filled markers where lambda = 0, hollow where the ridge is active."""
    c = EPS_COLORS[k]
    act = ridge_mask(meta, k)
    ax.plot(NS, y, color=c, lw=LW, zorder=z)
    ax.plot(NS[~act], y[~act], ls="none", marker="o", ms=ms, mfc=c, mec="white", mew=0.6, zorder=z + 0.1)
    ax.plot(NS[act], y[act], ls="none", marker="o", ms=ms, mfc="white", mec=c, mew=1.3, zorder=z + 0.1)


TEN_PCT = Line2D([], [], color=REF, lw=1.0, ls=(0, (4, 3)), label="10% relative error")
RIDGE_HANDLE = Line2D([], [], ls="none", marker="o", ms=MS, mfc="white", mec=INK2, mew=1.3,
                      label="hollow marker: ridge active (λ > 0)")


def series(overall, key, method, eps=None):
    return np.array([lookup(overall, n=int(n), method=method,
                            **({"eps": float(eps)} if eps is not None else {}))[key] for n in NS])


# =============================================================================
# Figure A: RMSE vs n
# =============================================================================

def fig_A(overall, meta, outdir, tag, reps):
    beta = np.array(meta["beta_true"])
    trivial = float(np.sqrt(np.mean(beta[1:] ** 2)))
    fig = plt.figure(figsize=(10.4, 6.3))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 5.2], hspace=0.07, right=0.66)
    axs = fig.add_subplot(gs[0]); ax = fig.add_subplot(gs[1], sharex=axs)
    xl = (75, 135000)
    dodge = np.exp(np.linspace(-0.045, 0.045, len(EPS)))

    # OLS floor
    med = series(overall, "rmse_median", "ols")
    lo, hi = series(overall, "rmse_p025", "ols"), series(overall, "rmse_p975", "ols")
    ax.fill_between(NS, lo, hi, color=OLS_COLOR, alpha=BAND_ALPHA * 0.8, lw=0, zorder=2)
    ax.plot(NS, med, color=OLS_COLOR, lw=LW, marker="s", ms=MS - 0.5, zorder=6,
            mec="white", mew=0.6)
    for k, e in enumerate(EPS):
        c = EPS_COLORS[k]
        med = series(overall, "rmse_median", "vfl_bc", e)
        lo, hi = series(overall, "rmse_p025", "vfl_bc", e), series(overall, "rmse_p975", "vfl_bc", e)
        mn = series(overall, "rmse_mean", "vfl_bc", e)
        clo, chi = series(overall, "rmse_mean_ci_lo", "vfl_bc", e), series(overall, "rmse_mean_ci_hi", "vfl_bc", e)
        ax.fill_between(NS, lo, hi, color=c, alpha=BAND_ALPHA, lw=0, zorder=2)
        eps_line(ax, med, k, meta)
        ax.errorbar(NS * dodge[k], mn, yerr=[mn - clo, chi - mn], fmt="none", ecolor=c,
                    elinewidth=1.0, capsize=2.2, capthick=1.0, zorder=7)
    ax.plot(NS, series(overall, "rmse_median", "ols_unclipped"), color=INK2, lw=1.2, ls=(0, (1, 1.6)),
            zorder=6)
    ax.axhline(trivial, color=REF, lw=1.0, ls=(0, (4, 3)), zorder=1)
    ax.text(xl[1] / 1.08, trivial * 1.07, "β̂ = 0 (all-zero estimate)", ha="right", va="bottom",
            fontsize=8, color=INK2)
    style(ax, logy=True)
    log_y_ticks(ax, 1e-3, 1.0)
    ax.set_xlim(*xl)
    ax.set_xlabel("Intersection size n (matched records, log scale)")
    ax.set_ylabel("RMSE of slopes vs true β (standardized scale, log)")

    handles = [Line2D([], [], color=EPS_COLORS[k], lw=LW, marker="o", ms=MS, mec="white", mew=0.6,
                      label=f"VFL, bias-corrected, {eps_label(e)}") for k, e in enumerate(EPS)]
    handles += [Line2D([], [], color=OLS_COLOR, lw=LW, marker="s", ms=MS - 0.5, mec="white", mew=0.6,
                       label="Non-private OLS on true matches (floor)"),
                Line2D([], [], color=INK2, lw=1.2, ls=(0, (1, 1.6)),
                       label="OLS on unclipped data (no committed bounds)"),
                RIDGE_HANDLE,
                Patch(facecolor="#5598e7", alpha=BAND_ALPHA + 0.07, lw=0, label="2.5–97.5% of runs"),
                Line2D([], [], color=INK2, lw=0, marker="|", ms=8, mew=1.0,
                       label="95% bootstrap CI of mean RMSE")]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8,
              handlelength=2.2, borderaxespad=0.0, labelspacing=0.7)

    # ridge-active strip
    for k, e in enumerate(EPS):
        nstar = meta["n_ridge_off"][k]
        yk = len(EPS) - 1 - k
        axs.plot([xl[0], min(nstar, xl[1])], [yk, yk], color=EPS_COLORS[k], lw=4.5,
                 solid_capstyle="butt")
        axs.text(min(nstar, xl[1]) * 1.12, yk, f"λ > 0 for n < {fmt_n(round(nstar, -2))}",
                 va="center", ha="left", fontsize=7.5, color=INK2)
    axs.set_yticks(range(len(EPS)))
    axs.set_yticklabels([eps_label(e) for e in EPS[::-1]], fontsize=7.5)
    axs.set_ylim(-0.8, len(EPS) - 0.2)
    axs.tick_params(axis="y", length=0)
    for s in ("top", "right", "left"):
        axs.spines[s].set_visible(False)
    axs.spines["bottom"].set_visible(False)
    axs.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    axs.set_title("Fixed ridge active (λ = 2ρ*σ√p − nℓ > 0)", fontsize=8.5, color=INK2, loc="left",
                  pad=3)
    fig.suptitle(f"Coefficient error vs intersection size, R² = {meta['r2']:g}", x=0.125, ha="left",
                 y=0.985, fontsize=11.5, color=INK, fontweight="bold")
    fig.text(0.125, 0.938, f"Line: median of per-run RMSE over {reps} replicates (new data + new DP "
             f"release each run); δ = 1e-5, replace-one σ, ρ* = 2",
             fontsize=8, color=INK2, ha="left")
    save(fig, outdir, f"figA_rmse_vs_n_{tag}")


def fig_A2(ratio_rows, overall, meta, outdir, tag):
    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    dodge = np.exp(np.linspace(-0.05, 0.05, len(EPS)))
    ax.axhline(1.0, color=REF, lw=1.0, zorder=1)
    for k, e in enumerate(EPS):
        rows = [lookup(ratio_rows, n=int(n), eps=float(e)) for n in NS]
        r = np.array([x["ratio_pooled"] for x in rows])
        lo = np.array([x["ci_lo"] for x in rows]); hi = np.array([x["ci_hi"] for x in rows])
        act = np.array([x["ridge_active"] for x in rows])
        c = EPS_COLORS[k]
        ax.fill_between(NS * dodge[k], lo, hi, color=c, alpha=BAND_ALPHA, lw=0, zorder=2)
        ax.plot(NS * dodge[k], r, color=c, lw=LW, zorder=4)
        ax.plot((NS * dodge[k])[act], r[act], ls="none", marker="o", ms=MS, mfc="white", mec=c,
                mew=1.4, zorder=5)
        ax.plot((NS * dodge[k])[~act], r[~act], ls="none", marker="o", ms=MS, mfc=c, mec="white",
                mew=0.6, zorder=5)
    style(ax)
    ax.set_xlim(75, 135000)
    ax.set_xlabel("Intersection size n (log scale)")
    ax.set_ylabel("RMSE(corrected) / RMSE(uncorrected)")
    lo_all = min(x["ci_lo"] for x in ratio_rows); hi_all = max(x["ci_hi"] for x in ratio_rows)
    pad = 0.1 * (hi_all - lo_all)
    ax.set_ylim(min(lo_all - pad, 0.985), max(hi_all + pad, 1.015))
    handles = [Line2D([], [], color=EPS_COLORS[k], lw=LW, label=eps_label(e)) for k, e in enumerate(EPS)]
    handles += [Line2D([], [], ls="none", marker="o", ms=MS, mfc="white", mec=INK2, mew=1.4,
                       label="ridge active (λ > 0)"),
                Line2D([], [], ls="none", marker="o", ms=MS, mfc=INK2, mec="white", mew=0.6,
                       label="no ridge (λ = 0)"),
                Patch(facecolor="#5598e7", alpha=BAND_ALPHA + 0.07, lw=0, label="paired bootstrap 95% CI")]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8)
    ax.set_title(f"Effect of the zero-budget bias correction, R² = {meta['r2']:g}", loc="left",
                 fontsize=11, fontweight="bold", pad=18)
    ax.text(0, 1.02, "Pooled slope RMSE vs true β; below 1 means the correction helps",
            transform=ax.transAxes, fontsize=8, color=INK2)
    save(fig, outdir, f"figA2_correction_ratio_{tag}")


# =============================================================================
# Figure B / C: per-coefficient small multiples
# =============================================================================

def pc_series(percoef, key, method, j, eps=None):
    return np.array([lookup(percoef, n=int(n), method=method, j=j,
                            **({"eps": float(eps)} if eps is not None else {}))[key] for n in NS])


def grid_axes(meta, names, include_intercept=True, sharey=True, figsize=(15.5, 6.4)):
    """2 x 6 grid: row 0 = R block (intercept + d_R slopes), row 1 = O block; returns
    (fig, {j: ax}, legend_ax)."""
    p = len(names)
    dRb = meta["d_R_block"]
    r_idx = list(range(0 if include_intercept else 1, dRb))
    o_idx = list(range(dRb, p))
    ncol = max(len(r_idx) + 1, len(o_idx))
    fig, axes = plt.subplots(2, ncol, figsize=figsize, sharex=True, sharey=sharey)
    mp = {}
    for c, j in enumerate(r_idx):
        mp[j] = axes[0, c]
    for c, j in enumerate(o_idx):
        mp[j] = axes[1, c]
    used = set(id(a) for a in mp.values())
    spare = [a for a in axes.ravel() if id(a) not in used]
    for a in spare:
        a.axis("off")
    return fig, mp, spare[0], axes


def panel_title(ax, meta, names, j):
    b = meta["beta_true"][j]
    sub = f"β = {b:+.3f}" if j > 0 else "β = 0 (intercept)"
    ax.set_title(f"{names[j]}\n", fontsize=9, color=INK, loc="left", pad=2)
    ax.text(0, 1.015, sub, transform=ax.transAxes, fontsize=7.8, color=INK2, va="bottom")


def row_labels(fig, axes, offset_in=0.78):
    """Block labels left of each row; call after subplots_adjust."""
    w = fig.get_figwidth()
    for row, lab in zip(axes, ["R block (researcher)", "O block (organization, holds y)"]):
        pos = row[0].get_position()
        fig.text(pos.x0 - offset_in / w, 0.5 * (pos.y0 + pos.y1), lab, rotation=90, va="center",
                 ha="center", fontsize=9.5, color=INK2)


def eps_handles(ols_label=None, band_label=None, extra=()):
    h = [Line2D([], [], color=EPS_COLORS[k], lw=LW, marker="o", ms=MS - 1, mec="white", mew=0.5,
                label=f"VFL corrected, {eps_label(e)}") for k, e in enumerate(EPS)]
    if ols_label:
        h.append(Line2D([], [], color=OLS_COLOR, lw=LW, marker="s", ms=MS - 1.5, mec="white", mew=0.5,
                        label=ols_label))
    if band_label:
        h.append(Patch(facecolor="#5598e7", alpha=BAND_ALPHA + 0.07, lw=0, label=band_label))
    h.extend(extra)
    return h


def fig_B_metric(percoef, meta, names, outdir, fname, key, lo_key, hi_key, ylabel, title, subtitle,
                 ylim, ols_key=None, refs=(), include_intercept=True, legend_extra=(), ols_label=None):
    fig, mp, leg_ax, axes = grid_axes(meta, names, include_intercept=include_intercept)
    for j, ax in mp.items():
        for yref, lab in refs:
            ax.axhline(yref, color=REF, lw=1.0, ls=(0, (4, 3)) if lab else "-", zorder=1)
        if ols_key is not None:
            ax.plot(NS, pc_series(percoef, ols_key, "ols", j), color=OLS_COLOR, lw=LW, marker="s",
                    ms=MS - 1.5, mec="white", mew=0.5, zorder=6)
        for k, e in enumerate(EPS):
            c = EPS_COLORS[k]
            if lo_key:
                ax.fill_between(NS, pc_series(percoef, lo_key, "vfl_bc", j, e),
                                pc_series(percoef, hi_key, "vfl_bc", j, e), color=c,
                                alpha=BAND_ALPHA, lw=0, zorder=2)
            eps_line(ax, pc_series(percoef, key, "vfl_bc", j, e), k, meta, ms=MS - 1)
        style(ax, logy=True)
        log_y_ticks(ax, *ylim)
        panel_title(ax, meta, names, j)
        ax.set_xlim(75, 135000)
    for ax in axes[1]:
        ax.set_xlabel("n (log)")
    for ax in axes[:, 0]:
        ax.set_ylabel(ylabel)
    extra = [RIDGE_HANDLE] + [Line2D([], [], color=REF, lw=1.0, ls=(0, (4, 3)), label=lab)
                              for _y, lab in refs if lab] + list(legend_extra)
    leg_ax.legend(handles=eps_handles(ols_label, "bootstrap 95% CI" if lo_key else None, extra),
                  loc="center left", fontsize=8, borderaxespad=0)
    fig.suptitle(title, x=0.02, ha="left", y=1.02, fontsize=12, fontweight="bold", color=INK)
    fig.text(0.02, 0.975, subtitle, fontsize=8.5, color=INK2, ha="left")
    fig.subplots_adjust(left=0.075, right=0.995, top=0.88, bottom=0.08, wspace=0.12, hspace=0.42)
    row_labels(fig, axes)
    save(fig, outdir, fname)


def fig_B_relerr(percoef, meta, names, outdir, fname, reps):
    idx = [j for j in range(1, len(names)) if abs(meta["beta_true"][j]) >= 0.1]
    r_idx = [j for j in idx if j < meta["d_R_block"]]; o_idx = [j for j in idx if j >= meta["d_R_block"]]
    ncol = max(len(r_idx) + 1, len(o_idx))
    fig, axes = plt.subplots(2, ncol, figsize=(2.55 * ncol + 0.6, 6.2), sharex=True, sharey=True)
    mp = {j: axes[0, c] for c, j in enumerate(r_idx)}
    mp.update({j: axes[1, c] for c, j in enumerate(o_idx)})
    spare = [a for a in axes.ravel() if a not in mp.values()]
    for a in spare:
        a.axis("off")
    for j, ax in mp.items():
        ax.axhline(0.10, color=REF, lw=1.0, ls=(0, (4, 3)), zorder=1)
        ax.plot(NS, pc_series(percoef, "relerr_median", "ols", j), color=OLS_COLOR, lw=LW, marker="s",
                ms=MS - 1.5, mec="white", mew=0.5, zorder=6)
        for k, e in enumerate(EPS):
            eps_line(ax, pc_series(percoef, "relerr_median", "vfl_bc", j, e), k, meta, ms=MS - 1)
        style(ax, logy=True)
        log_y_ticks(ax, 0.002, 1.5)
        panel_title(ax, meta, names, j)
        ax.set_xlim(75, 135000)
        rawb = meta["beta_raw"][j - 1]
        rtxt = f"{rawb:,.0f}" if abs(rawb) >= 10 else f"{rawb:.3f}"
        ax.text(0.04, 0.04, f"raw β = {rtxt}\n($ of y per unit of x)", transform=ax.transAxes,
                ha="left", va="bottom", fontsize=7.2, color=INK2)
    for ax in axes[1]:
        ax.set_xlabel("n (log)")
    for ax in axes[:, 0]:
        ax.set_ylabel("median |β̂ − β| / |β|")
    spare[0].legend(handles=eps_handles("Non-private OLS", None, [RIDGE_HANDLE, TEN_PCT]),
                    loc="center left", fontsize=8)
    fig.suptitle(f"Relative error per coefficient (|β| ≥ 0.1 only), R² = {meta['r2']:g}", x=0.02,
                 ha="left", y=1.02, fontsize=12, fontweight="bold")
    fig.text(0.02, 0.975, f"Median over {reps} runs. Identical on the raw and standardized scales "
             "(the rescaling cancels), so it is unit-free; unstable for coefficients near zero",
             fontsize=8.5, color=INK2)
    fig.subplots_adjust(left=0.1, right=0.995, top=0.88, bottom=0.08, wspace=0.12, hspace=0.42)
    row_labels(fig, axes)
    save(fig, outdir, fname)


def fig_B_raw_view(percoef, meta, names, outdir, fname, n_show=10000, eps_show=1.0):
    p = len(names); idx = list(range(1, p))
    k = int(np.nonzero(EPS == eps_show)[0][0])
    c = EPS_COLORS[k]
    rows = {j: lookup(percoef, n=n_show, eps=eps_show, method="vfl_bc", j=j) for j in idx}
    raw_rmse = np.array([rows[j]["raw_rmse"] for j in idx])
    raw_b = np.abs(np.array([meta["beta_raw"][j - 1] for j in idx]))
    std_rmse = np.array([rows[j]["rmse_std"] for j in idx])
    std_b = np.abs(np.array([meta["beta_true"][j] for j in idx]))
    eff = np.array([rows[j]["eff_ratio"] for j in idx])
    x = np.arange(len(idx))
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6), gridspec_kw={"wspace": 0.28})
    for ax, (vals, bvals, ttl, ylab) in zip(axes[:2], [
            (raw_rmse, raw_b, "Raw scale: dollars of y per raw unit of x_j",
             "RMSE and |β| (raw units, log)"),
            (std_rmse, std_b, "Standardized scale: SDs of y per SD of x_j",
             "RMSE and |β| (standardized, log)")]):
        ax.vlines(x, np.minimum(vals, bvals), np.maximum(vals, bvals), color=GRID, lw=1.2, zorder=1)
        ax.plot(x, bvals, ls="none", marker="o", ms=6.5, mfc="white", mec=OLS_COLOR, mew=1.4, zorder=3,
                label="|true β_j|")
        ax.plot(x, vals, ls="none", marker="o", ms=6.5, mfc=c, mec="white", mew=0.8, zorder=4,
                label=f"RMSE_j, VFL corrected ({eps_label(eps_show)}, n = {fmt_n(n_show)})")
        style(ax, logx=False, grid_x=False)
        ax.set_yscale("log")
        ax.set_xticks(x); ax.set_xticklabels([names[j] for j in idx], rotation=55, ha="right", fontsize=8)
        ax.set_title(ttl, loc="left", fontsize=10)
        ax.set_ylabel(ylab)
        span = np.log10(max(vals.max(), bvals.max()) / min(vals.min(), bvals.min()))
        ax.text(0.02, 0.97, f"values span {span:.1f} orders of magnitude", transform=ax.transAxes,
                fontsize=8, color=INK2, va="top")
        ax.legend(loc="lower right", fontsize=7.8)
    pooled = float(np.sqrt(np.mean(raw_rmse ** 2)))
    bin_idx = [i for i, j in enumerate(idx) if meta["types"][j - 1] == "binary"]
    share = float(np.sum(raw_rmse[bin_idx] ** 2) / np.sum(raw_rmse ** 2))
    axes[0].text(0.02, 0.915, f"one pooled raw RMSE = {pooled:,.0f};\n{100 * share:.0f}% of its square comes "
                 f"from the {len(bin_idx)} binary coefficients", transform=axes[0].transAxes, fontsize=8,
                 color=INK2, va="top")
    axes[0].set_ylim(1e-4, 1e7)
    axes[1].set_ylim(1e-3, 1e0)
    ax = axes[2]
    ax.axhline(1.0, color=REF, lw=1.0)
    ax.plot(x, eff, ls="none", marker="o", ms=6.5, mfc=c, mec="white", mew=0.8, zorder=4)
    style(ax, logx=False, grid_x=False)
    ax.set_xticks(x); ax.set_xticklabels([names[j] for j in idx], rotation=55, ha="right", fontsize=8)
    ax.set_title("Efficiency ratio RMSE_DP,j / SE_OLS,j (scale-free)", loc="left", fontsize=10)
    ax.set_ylabel("efficiency ratio")
    ax.set_ylim(0, max(eff.max() * 1.25, 1.5))
    ax.text(len(idx) - 0.6, 1.0, "OLS", fontsize=7.8, color=INK2, va="bottom", ha="right")
    fig.suptitle(f"Why raw-scale RMSE cannot compare coefficients (R² = {meta['r2']:g}, "
                 f"{eps_label(eps_show)}, n = {n_show:,})", x=0.02, ha="left", y=1.04, fontsize=12,
                 fontweight="bold")
    fig.text(0.02, 0.975, "Same estimates in all three panels. Raw RMSE_j = y_sd·RMSE_j(std)/s_j, so it "
             "tracks the units of x_j (income in dollars vs a 0/1 flag), not estimation quality.",
             fontsize=8.5, color=INK2)
    save(fig, outdir, fname)


def fig_C(percoef, meta, names, outdir, fname, key, ylabel, title, subtitle, reps, ylim,
          include_intercept=True, nominal=None, chance=None):
    fig, mp, leg_ax, axes = grid_axes(meta, names, include_intercept=include_intercept)
    for j, ax in mp.items():
        if nominal is not None:
            tol = Z975 * np.sqrt(nominal * (1 - nominal) / reps)
            ax.axhspan(nominal - tol, min(nominal + tol, 1.0), color=GRID, alpha=0.9, lw=0, zorder=0.5)
            ax.axhline(nominal, color=REF, lw=1.0, zorder=1)
        if chance is not None:
            ax.axhline(chance, color=REF, lw=1.0, ls=(0, (4, 3)), zorder=1)
        ax.plot(NS, pc_series(percoef, key, "ols", j), color=OLS_COLOR, lw=LW, marker="s", ms=MS - 1.5,
                mec="white", mew=0.5, zorder=6)
        for k, e in enumerate(EPS):
            eps_line(ax, pc_series(percoef, key, "vfl_bc", j, e), k, meta, ms=MS - 1)
        style(ax)
        ax.set_ylim(*ylim)
        panel_title(ax, meta, names, j)
        ax.set_xlim(75, 135000)
    for ax in axes[1]:
        ax.set_xlabel("n (log)")
    for ax in axes[:, 0]:
        ax.set_ylabel(ylabel)
    extra = []
    first = mp[min(mp)]
    if nominal is not None:
        extra.append(Patch(facecolor=GRID, lw=0, label=f"nominal {nominal:.2f} ± MC error ({reps} runs)"))
    if chance is not None:
        extra.append(Line2D([], [], color=REF, lw=1.0, ls=(0, (4, 3)), label="chance (0.5)"))
    ols_lab = "Non-private OLS (classical CI)" if nominal is not None else "Non-private OLS"
    leg_ax.legend(handles=eps_handles(ols_lab, None, [RIDGE_HANDLE] + extra), loc="center left",
                  fontsize=8, borderaxespad=0)
    fig.suptitle(title, x=0.02, ha="left", y=1.02, fontsize=12, fontweight="bold")
    fig.text(0.02, 0.975, subtitle, fontsize=8.5, color=INK2)
    fig.subplots_adjust(left=0.075, right=0.995, top=0.88, bottom=0.08, wspace=0.12, hspace=0.42)
    row_labels(fig, axes)
    save(fig, outdir, fname)


def fig_A_r2_compare(all_overall, all_meta, outdir):
    """Figure A for each r2 side by side (shared y), bias-corrected lines + OLS floor."""
    r2s = sorted(all_overall)
    fig, axes = plt.subplots(1, len(r2s), figsize=(4.6 * len(r2s), 4.3), sharey=True)
    for ax, r2 in zip(axes, r2s):
        ov = all_overall[r2]; meta = all_meta[r2]
        beta = np.array(meta["beta_true"])
        ax.axhline(np.sqrt(np.mean(beta[1:] ** 2)), color=REF, lw=1.0, ls=(0, (4, 3)), zorder=1)
        ax.fill_between(NS, series(ov, "rmse_p025", "ols"), series(ov, "rmse_p975", "ols"),
                        color=OLS_COLOR, alpha=BAND_ALPHA * 0.8, lw=0)
        ax.plot(NS, series(ov, "rmse_median", "ols"), color=OLS_COLOR, lw=LW, marker="s", ms=MS - 0.5,
                mec="white", mew=0.6, zorder=6)
        ax.plot(NS, series(ov, "rmse_median", "ols_unclipped"), color=INK2, lw=1.2, ls=(0, (1, 1.6)),
                zorder=6)
        for k, e in enumerate(EPS):
            ax.fill_between(NS, series(ov, "rmse_p025", "vfl_bc", e), series(ov, "rmse_p975", "vfl_bc", e),
                            color=EPS_COLORS[k], alpha=BAND_ALPHA, lw=0)
            eps_line(ax, series(ov, "rmse_median", "vfl_bc", e), k, meta)
        style(ax, logy=True)
        log_y_ticks(ax, 1e-3, 1.0)
        ax.set_xlim(75, 135000)
        ax.set_title(f"R² = {r2:g}  (noise sd {np.sqrt(1 - r2):.2f}, ‖β‖ = {np.linalg.norm(beta):.2f})",
                     loc="left", fontsize=10)
        ax.set_xlabel("Intersection size n (log)")
    axes[0].set_ylabel("RMSE of slopes vs true β (standardized, log)")
    h = [Line2D([], [], color=EPS_COLORS[k], lw=LW, marker="o", ms=MS, mec="white", mew=0.6,
                label=eps_label(e)) for k, e in enumerate(EPS)]
    h += [Line2D([], [], color=OLS_COLOR, lw=LW, marker="s", ms=MS - 0.5, mec="white", mew=0.6,
                 label="Non-private OLS"),
          Line2D([], [], color=INK2, lw=1.2, ls=(0, (1, 1.6)), label="OLS, unclipped"),
          Line2D([], [], color=REF, lw=1.0, ls=(0, (4, 3)), label="β̂ = 0"),
          RIDGE_HANDLE,
          Patch(facecolor="#5598e7", alpha=BAND_ALPHA + 0.07, lw=0, label="2.5–97.5% of runs")]
    fig.legend(handles=h[:5], loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.075), fontsize=8.5,
               columnspacing=2.2)
    fig.legend(handles=h[5:], loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.135), fontsize=8.5,
               columnspacing=2.2)
    fig.suptitle("VFL bias-corrected coefficient RMSE (median over runs) at three signal strengths",
                 x=0.02, ha="left", y=1.03, fontsize=12, fontweight="bold")
    save(fig, outdir, "figA_r2_comparison")


# =============================================================================
# Driver
# =============================================================================

def write_csv(rows, path):
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            vals = []
            for k in keys:
                v = r.get(k, "")
                if v is None:
                    v = ""
                if isinstance(v, float):
                    v = f"{v:.6g}"
                vals.append(str(v))
            f.write(",".join(vals) + "\n")


def run_one(r2, reps, outdir, plots_only, main):
    tag = f"r2_{r2:g}"
    npz = os.path.join(outdir, f"raw_estimates_{tag}.npz")
    if plots_only:
        z = np.load(npz, allow_pickle=True)
        out = {k: z[k] for k in z.files if k != "meta"}
        meta = json.loads(str(z["meta"]))
        pop = sc.Population(r2=r2, seed=POP_SEED)
    else:
        print(f"Simulating r2 = {r2} with {reps} replicates", flush=True)
        out, meta, pop = simulate(r2, reps)
        np.savez_compressed(npz, meta=json.dumps(meta), **out)
    reps = out["beta_ols"].shape[1]
    names, blocks = coef_labels(pop)
    meta["names"] = names
    overall, percoef, ratio = compute_metrics(out, meta, names, blocks,
                                              np.random.default_rng([BASE_SEED, 7, int(r2 * 100)]))
    write_csv(overall, os.path.join(outdir, f"summary_overall_{tag}.csv"))
    write_csv(percoef, os.path.join(outdir, f"summary_per_coef_{tag}.csv"))
    write_csv(ratio, os.path.join(outdir, f"correction_ratio_{tag}.csv"))
    kn = key_numbers(overall, meta)
    with open(os.path.join(outdir, f"key_numbers_{tag}.json"), "w") as f:
        json.dump({"meta": meta, "key_numbers": kn}, f, indent=1, default=float)

    fig_A(overall, meta, outdir, tag, reps)
    fig_A2(ratio, overall, meta, outdir, tag)
    if main:
        fig_B_metric(percoef, meta, names, outdir, f"figB1_efficiency_ratio_{tag}", "eff_ratio",
                     "eff_ratio_ci_lo", "eff_ratio_ci_hi", "RMSE_DP,j / SE_OLS,j",
                     f"Efficiency ratio per coefficient: DP error in units of that coefficient's own OLS "
                     f"standard error, R² = {meta['r2']:g}",
                     "SE_OLS,j = empirical sd of non-private OLS over the same replicates. 1 = as good "
                     "as OLS. Black line = OLS's own RMSE/SE (above 1 only where clipping bias shows).",
                     (0.5, 50), ols_key="eff_ratio", refs=[(1.0, None), (2.0, "2× OLS")],
                     ols_label="Non-private OLS (RMSE/SE)")
        fig_B_metric(percoef, meta, names, outdir, f"figB2_standardized_error_{tag}", "rmse_std",
                     "rmse_std_ci_lo", "rmse_std_ci_hi", "RMSE_j (SD y per SD x_j)",
                     f"Standardized-coefficient error per coefficient, R² = {meta['r2']:g}",
                     "RMSE_j of β̂_j·s_j/s_y vs the true standardized coefficient; common log y-axis "
                     "across panels. The protocol estimates on this scale (public standardization).",
                     (1e-3, 1.0), ols_key="rmse_std", ols_label="Non-private OLS")
        fig_B_relerr(percoef, meta, names, outdir, f"figB3_relative_error_{tag}", reps)
        fig_B_raw_view(percoef, meta, names, outdir, f"figB4_raw_vs_standardized_{tag}")
        fig_C(percoef, meta, names, outdir, f"figC1_coverage_{tag}", "coverage", "95% Wald coverage",
              f"Coverage of 95% Wald intervals per coefficient, R² = {meta['r2']:g}",
              "VFL: first-order variance (sampling + mechanism) centred at the bias-corrected β̂; "
              "coverage of the true (generating) β.", reps, (0.0, 1.03), nominal=0.95)
        fig_C(percoef, meta, names, outdir, f"figC2_sign_recovery_{tag}", "sign_rate",
              "P(sign β̂_j = sign β_j)", f"Sign recovery per coefficient, R² = {meta['r2']:g}",
              f"Share of {reps} runs in which the estimated slope has the true sign.", reps,
              (0.3, 1.02), include_intercept=False, chance=0.5)
    return overall, percoef, ratio, meta, kn


def print_tables(overall, ratio, meta, kn):
    print(f"\n=== r2 = {meta['r2']} ===")
    print("sigma per eps:", dict(zip(EPS.tolist(), np.round(meta["sigma"], 2).tolist())))
    print("n_ridge_off:", dict(zip(EPS.tolist(), np.round(meta["n_ridge_off"]).tolist())))
    print("beta_clip - beta_true (slopes):",
          np.round(np.array(meta["beta_clip"]) - np.array(meta["beta_true"]), 4))
    hdr = "n      " + " ".join(f"{'eps' + format(e, 'g'):>18s}" for e in EPS) + f"{'OLS':>16s}"
    print("\nmedian per-run slope RMSE (pooled RMSE)  [ratio to OLS median]")
    print(hdr)
    for n in NS:
        cells = []
        for e in EPS:
            v = lookup(overall, n=int(n), eps=float(e), method="vfl_bc")
            cells.append(f"{v['rmse_median']:.4f}({v['rmse_pooled']:.4f})[{v['ratio_to_ols_median']:.1f}]")
        o = lookup(overall, n=int(n), method="ols")
        print(f"{n:<7d}" + " ".join(f"{c:>18s}" for c in cells) + f"{o['rmse_median']:>8.4f}({o['rmse_pooled']:.4f})")
    print("\nn within 2x OLS floor:", json.dumps(kn["n_within_2x_floor"]))
    print("gate:", json.dumps(kn["gate"]))
    print("\ncorrected/uncorrected pooled ratio [CI] (ridge active *)")
    for n in NS:
        cells = []
        for e in EPS:
            r = lookup(ratio, n=int(n), eps=float(e))
            cells.append(f"{r['ratio_pooled']:.3f}[{r['ci_lo']:.3f},{r['ci_hi']:.3f}]{'*' if r['ridge_active'] else ' '}")
        print(f"{n:<7d}" + " ".join(cells))
    print("\nmean slope coverage VFL bc (OLS):")
    for n in NS:
        cells = [f"{lookup(overall, n=int(n), eps=float(e), method='vfl_bc')['coverage_mean_slopes']:.3f}"
                 for e in EPS]
        o = lookup(overall, n=int(n), method="ols")
        print(f"{n:<7d}" + " ".join(f"{c:>7s}" for c in cells) + f"   OLS {o['coverage_mean_slopes']:.3f}")
    print("\nOLS clipped vs unclipped pooled RMSE:")
    for n in NS:
        o = lookup(overall, n=int(n), method="ols"); u = lookup(overall, n=int(n), method="ols_unclipped")
        print(f"{n:<7d} clipped {o['rmse_pooled']:.5f} unclipped {u['rmse_pooled']:.5f}  "
              f"clipped-vs-clip-target {o['rmse_clip_target_pooled']:.5f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--reps-sens", type=int, default=200, help="replicates for r2 = 0.3, 0.9")
    ap.add_argument("--r2-sens", type=float, nargs="*", default=[0.3, 0.9])
    ap.add_argument("--out", default=os.path.join(HERE, "..", "reports", "coef_accuracy"))
    ap.add_argument("--plots-only", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    all_overall, all_meta, all_kn = {}, {}, {}
    ov, pc, ra, meta, kn = run_one(0.5, a.reps, a.out, a.plots_only, main=True)
    print_tables(ov, ra, meta, kn)
    all_overall[0.5], all_meta[0.5], all_kn[0.5] = ov, meta, kn
    for r2 in a.r2_sens:
        ov2, _, ra2, meta2, kn2 = run_one(r2, a.reps_sens, a.out, a.plots_only, main=False)
        print_tables(ov2, ra2, meta2, kn2)
        all_overall[r2], all_meta[r2], all_kn[r2] = ov2, meta2, kn2
    if len(all_overall) > 1:
        fig_A_r2_compare(all_overall, all_meta, a.out)
    with open(os.path.join(a.out, "results.json"), "w") as f:
        json.dump({"config": {"ns": NS.tolist(), "eps": EPS.tolist(), "delta": DELTA,
                              "rho_star": RHO_STAR, "ell_frac": ELL_FRAC, "pop_seed": POP_SEED,
                              "base_seed": BASE_SEED, "n_boot": N_BOOT},
                   "by_r2": {str(r2): {"meta": all_meta[r2], "key_numbers": all_kn[r2]}
                             for r2 in all_overall}}, f, indent=1, default=float)


if __name__ == "__main__":
    main()
