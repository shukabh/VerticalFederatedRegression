#!/usr/bin/env python3
"""
bias_correction_study.py -- does the zero-budget bias correction (paper Sec. 7, eq. (bc)) matter,
especially when the DP noise sigma is large?

Everything runs on the sufficient statistics through experiments/sim_core.py (read-only):
fixed ridge lambda = max{0, 2 rho* sigma sqrt(p) - n ell} (eq. lambda_rule), Psi = I,
beta_bc = beta~ - sigma^2 M(P~) beta~, release gate rho_hat >= 1 (Remark 7.2).

  E1   value map over n x eps. Conditional mode: fixed dataset (5 data seeds), >= 10,000
       antithetic noise pairs (E, f) / (-E, -f) per seed; bias, bias^2 share, MSE ratio bc/unc
       with a Monte-Carlo CI, max_j |bias_j|/sd_j, against the protocol target beta_lambda and
       against sample OLS. Run for ell = 0.8 lambda_min(pop) (main) and ell = 0 (sensitivity).
  FULL full mode (redraw data + noise, 1,000 reps per cell): coverage of 95% Wald intervals
       (sim_core.first_order_cov) against the TRUE beta, with and without correction; also the
       sampling and shrinkage terms for E2.
  E1s  stress test for the rule threshold: oracle ridge set to hit rho_lambda in [1, 6], and
       plain OLS (lambda = 0) at n chosen to hit rho_0 in [1, 6].
  E2   decomposition of the error vs true beta (shrinkage, mechanism bias^2, mechanism
       variance, sampling) -- assembled from E1 + FULL in the report stage.
  E3   averaging K releases of the same data -- computed in the E1 stage from the independent
       "+" draws (direct, K in {1,5,20,100}) and from the precise bias/variance (analytic).
  E4   fixed lambda vs the original adaptive gate (sim_core.adaptive_gate), same noise draws.

    python experiments/bias_correction_study.py                 # all stages, sequential
    python experiments/bias_correction_study.py --stage e1      # one stage (e1|full|stress|e4)
    python experiments/bias_correction_study.py --stage report  # merge + figures + tables
    python experiments/bias_correction_study.py --quick         # smoke test (tiny sizes)

Stage outputs go to experiments/_work/bias_correction/ (git-ignored); the merged results.json,
CSV, figures (PNG 200 dpi + PDF) go to reports/bias_correction/.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import sim_core as sc  # noqa: E402

OUT = os.path.join(ROOT, "reports", "bias_correction")
WORK = os.path.join(ROOT, "experiments", "_work", "bias_correction")

N_GRID = [500, 1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000]
EPS_GRID = [0.5, 1.0, 2.0, 4.0, 8.0]
K_GRID = [1, 5, 20, 100]
DELTA = 1e-5
RHO_STAR = 2.0
ELL_FRAC = 0.8
GATE = 1.0
BASE_SEED = 20260930
Z95 = 1.959963984540054

SIZES = {
    "full": dict(seeds=5, pairs=10_000, full_reps=1000, e4_seeds=5, e4_pairs=1000,
                 stress_seeds=5, stress_pairs=10_000, signal_seeds=5, signal_pairs=5000),
    "quick": dict(seeds=2, pairs=400, full_reps=40, e4_seeds=1, e4_pairs=60,
                  stress_seeds=1, stress_pairs=400, signal_seeds=1, signal_pairs=200),
}

# E4: grid cells in the ridge regime and around its onset (0.25 <= rho_0(pop) <= 5)
E4_RHO_RANGE = (0.25, 5.0)
# E1s: (eps, n) cells with rho_0 < 1 so every rho_lambda target is reachable by a ridge
STRESS_RIDGE_CELLS = [(0.5, 4000), (1.0, 2000), (4.0, 500)]
STRESS_RIDGE_TARGETS = [1.25, 1.4, 1.5, 1.6, 1.75, 2.0, 2.5, 3.0, 4.0, 6.0]
STRESS_OLS_EPS = [1.0, 4.0]
STRESS_OLS_TARGETS = [1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0, 6.0]

# E1b: signal strength. c -> k c scales beta_hat by k with the design, sigma and noise fixed;
# the correction's own fluctuation grows with ||beta_hat|| while the f-noise does not.
SIGNAL_CELLS = [(1.0, 2000), (1.0, 8000), (1.0, 16000), (1.0, 64000)]
SIGNAL_K = [1, 2, 4, 8]

POP = sc.Population(seed=0)
DR, DO, P = POP.d_R_block, POP.d_O, POP.p
BETA_TRUE = POP.beta_true
ELL_MAIN = ELL_FRAC * POP.lam_min_pop
ELLS = {"main": ELL_MAIN, "zero": 0.0}
SIGMA = {e: POP.sigma_for(e, DELTA)[0] for e in EPS_GRID}
RHO_SWITCH = RHO_STAR / ELL_FRAC      # rho_0 below which the fixed ridge is on (pop value)


# =============================================================================
# helpers
# =============================================================================

def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, float) and not np.isfinite(o):
        return None if np.isnan(o) else ("inf" if o > 0 else "-inf")
    return o


def save(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(jsonable(obj), fh, indent=1)


def load(path):
    with open(path) as fh:
        return json.load(fh)


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def dataset(n, idx, stream=1):
    rng = np.random.default_rng([BASE_SEED, stream, n, idx])
    Z, y, clip = POP.draw(n, rng)
    G, c, yty = sc.stats(Z, y)
    return G, c, yty, clip


def rho0_of(G, sigma):
    return float(np.linalg.eigvalsh(G)[0] / (2 * sigma * np.sqrt(P)))


def ridge_target(G, c, lam):
    return np.linalg.solve(G + lam * np.eye(P), c)


def draw_pairs(G, c, yty, sigma, lam, m, rng, chunk=5000, adaptive=False):
    """m antithetic pairs: sim_core.release gives (G+E, c+f); the partner is (G-E, c-f).
    Fixed-lambda solve + correction on both; optionally the adaptive gate on the same draws."""
    parts = []
    for s0 in range(0, m, chunk):
        k = min(chunk, m - s0)
        Gt, ct, _ = sc.release(G, c, yty, DR, sigma, k, rng)
        Gm, cm = 2.0 * G[None] - Gt, 2.0 * c[None] - ct
        bp, bcp, _ = sc.solve(Gt, ct, lam, sigma, DR)
        bm, bcm, _ = sc.solve(Gm, cm, lam, sigma, DR)
        rp, rm = sc.rho_hat(Gt, lam, sigma), sc.rho_hat(Gm, lam, sigma)
        d = dict(bp=bp, bcp=bcp, bm=bm, bcm=bcm, keep=(rp >= GATE) & (rm >= GATE))
        if adaptive:
            d["abp"], d["abcp"], d["lamp"] = sc.adaptive_gate(Gt, ct, sigma, DR, DO)
            d["abm"], d["abcm"], d["lamm"] = sc.adaptive_gate(Gm, cm, sigma, DR, DO)
        parts.append(d)
    return {key: np.concatenate([d[key] for d in parts]) for key in parts[0]}


def err_stats(ep, em, t):
    """Bias (antithetic pair means), per-draw sd, per-pair squared error vs target t."""
    m = len(ep)
    pm = 0.5 * (ep + em) - t
    bias = pm.mean(0)
    se = pm.std(0, ddof=1) / np.sqrt(m)
    sd = np.vstack([ep, em]).std(0, ddof=1)
    a = 0.5 * (((ep - t) ** 2).sum(1) + ((em - t) ** 2).sum(1))
    return dict(bias=bias, se=se, sd=sd, a=a, bsq=float(bias @ bias - se @ se),
                bsq_raw=float(bias @ bias), mse=float(a.mean()), trV=float((sd ** 2).sum()))


def paired_ratio(a_num, a_den):
    """mean(a_num)/mean(a_den) over paired units, delta-method SE."""
    R = a_num.mean() / a_den.mean()
    se = (a_num - R * a_den).std(ddof=1) / (np.sqrt(len(a_num)) * a_den.mean())
    return float(R), float(se)


def compare(ep_u, em_u, ep_c, em_c, t):
    U, C = err_stats(ep_u, em_u, t), err_stats(ep_c, em_c, t)
    R, se = paired_ratio(C["a"], U["a"])
    tn = float(np.linalg.norm(t))
    # split of the MSE gain 1 - ratio into bias removal and variance change (MSE = bias^2 + tr Var)
    out = dict(mse_ratio=R, mse_ratio_se=se, tnorm=tn, var_ratio=C["trV"] / U["trV"],
               gain_bias_part=(max(U["bsq"], 0.0) - max(C["bsq"], 0.0)) / U["mse"],
               gain_var_part=(U["trV"] - C["trV"]) / U["mse"])
    for tag, S in (("unc", U), ("bc", C)):
        bsq = max(S["bsq"], 0.0)
        out[tag] = dict(bias_norm=float(np.sqrt(bsq)), bias_norm_raw=float(np.sqrt(S["bsq_raw"])),
                        bias_mcres=float(np.sqrt(S["se"] @ S["se"])),
                        rel_bias=float(np.sqrt(bsq) / tn), mse=S["mse"], trV=S["trV"],
                        bias_sq=bsq, bias_share=bsq / S["mse"],
                        max_bias_over_sd=float(np.max(np.abs(S["bias"]) / S["sd"])),
                        max_z=float(np.max(np.abs(S["bias"]) / S["se"])),
                        mc_floor_over_sd=float(np.max(S["se"] / S["sd"])),
                        bias=S["bias"], bias_se=S["se"], sd=S["sd"])
    return out, U, C


SCALARS = ["bias_norm", "bias_norm_raw", "bias_mcres", "rel_bias", "mse", "trV", "bias_sq",
           "bias_share", "max_bias_over_sd", "max_z", "mc_floor_over_sd"]


def agg_compare(L):
    """Average per-seed summaries; combine the per-seed Monte-Carlo SEs of the ratio."""
    S = len(L)
    r = np.array([x["mse_ratio"] for x in L])
    se = np.array([x["mse_ratio_se"] for x in L])
    out = dict(mse_ratio=float(r.mean()), mse_ratio_se=float(np.sqrt((se ** 2).sum()) / S),
               mse_ratio_seed_min=float(r.min()), mse_ratio_seed_max=float(r.max()),
               mse_ratio_seeds=r.tolist(), tnorm=float(np.mean([x["tnorm"] for x in L])))
    for k in ("var_ratio", "gain_bias_part", "gain_var_part"):
        out[k] = float(np.mean([x[k] for x in L]))
    for tag in ("unc", "bc"):
        out[tag] = {k: float(np.mean([x[tag][k] for x in L])) for k in SCALARS}
    return out


def strip_vectors(cmp):
    return {k: ({kk: vv for kk, vv in v.items() if kk not in ("bias", "bias_se", "sd")}
                if isinstance(v, dict) else v) for k, v in cmp.items()}


def e3_seed_stats(D, t, U, C):
    """Averaging K independent releases: independent '+' draws only (the antithetic partners
    are not independent). Bias from the precise pair means, variance from the '+' draws."""
    keep = D["keep"]
    bp, bcp = D["bp"][keep], D["bcp"][keep]
    out = dict(bsq_unc=max(U["bsq"], 0.0), bsq_bc=max(C["bsq"], 0.0),
               trV_unc=float(bp.var(0, ddof=1).sum()), trV_bc=float(bcp.var(0, ddof=1).sum()),
               direct={})
    for K in K_GRID:
        g = len(bp) // K
        if g < 5:
            continue
        au = ((bp[: g * K].reshape(g, K, P).mean(1) - t) ** 2).sum(1)
        ac = ((bcp[: g * K].reshape(g, K, P).mean(1) - t) ** 2).sum(1)
        R, se = paired_ratio(ac, au)
        out["direct"][K] = dict(mse_unc=float(au.mean()), mse_bc=float(ac.mean()),
                                ratio=R, ratio_se=se, groups=g)
    return out


def k_cross(bsq_u, bsq_c, tv_u, tv_c, r):
    """Smallest K >= 1 with (bsq_c + tv_c/K) <= r (bsq_u + tv_u/K)."""
    num, den = tv_c - r * tv_u, r * bsq_u - bsq_c
    if num <= 0:
        return 1.0
    if den <= 0:
        return float("inf")
    return max(1.0, num / den)


def e3_aggregate(L):
    keys = ["bsq_unc", "bsq_bc", "trV_unc", "trV_bc"]
    m = {k: float(np.mean([x[k] for x in L])) for k in keys}
    Ks = np.unique(np.round(np.logspace(0, 6, 61)).astype(int))
    m["analytic_K"] = Ks.tolist()
    m["analytic_ratio"] = [(m["bsq_bc"] + m["trV_bc"] / K) / (m["bsq_unc"] + m["trV_unc"] / K)
                           for K in Ks]
    m["analytic_ratio_at"] = {K: (m["bsq_bc"] + m["trV_bc"] / K) / (m["bsq_unc"] + m["trV_unc"] / K)
                              for K in K_GRID}
    m["K_cross_10pct"] = k_cross(m["bsq_unc"], m["bsq_bc"], m["trV_unc"], m["trV_bc"], 0.90)
    m["K_cross_1pct"] = k_cross(m["bsq_unc"], m["bsq_bc"], m["trV_unc"], m["trV_bc"], 0.99)
    m["K_bias_eq_var"] = m["trV_unc"] / m["bsq_unc"] if m["bsq_unc"] > 0 else float("inf")
    direct = {}
    for K in K_GRID:
        rows = [x["direct"][K] for x in L if K in x["direct"]]
        if rows:
            direct[K] = dict(ratio=float(np.mean([r["ratio"] for r in rows])),
                             ratio_se=float(np.sqrt(sum(r["ratio_se"] ** 2 for r in rows)) / len(rows)),
                             mse_unc=float(np.mean([r["mse_unc"] for r in rows])),
                             mse_bc=float(np.mean([r["mse_bc"] for r in rows])),
                             groups=int(sum(r["groups"] for r in rows)))
    m["direct"] = direct
    return m


# =============================================================================
# E1 (+E3): conditional value map
# =============================================================================

def stage_e1(sz):
    t0 = time.time()
    cells = {ell: [] for ell in ELLS}
    for n in N_GRID:
        data = [dataset(n, i) for i in range(sz["seeds"])]
        for ei, eps in enumerate(EPS_GRID):
            sigma = SIGMA[eps]
            for ell_name, ell in ELLS.items():
                lam = sc.fixed_lambda(sigma, P, n, ell, RHO_STAR)
                per_lam, per_ols, per_e3, rho0s, rholams, keeps = [], [], [], [], [], []
                bl_list, contraction = [], []
                for i, (G, c, yty, clip) in enumerate(data):
                    rng = np.random.default_rng([BASE_SEED, 10, n, i, ei])   # same noise for both ells
                    D = draw_pairs(G, c, yty, sigma, lam, sz["pairs"], rng)
                    k = D["keep"]
                    b_ols, b_lam = np.linalg.solve(G, c), ridge_target(G, c, lam)
                    args = (D["bp"][k], D["bm"][k], D["bcp"][k], D["bcm"][k])
                    cl, U, C = compare(*args, b_lam)
                    co, _, _ = compare(*args, b_ols)
                    per_lam.append(cl); per_ols.append(co)
                    per_e3.append(e3_seed_stats(D, b_lam, U, C))
                    V = np.cov(D["bp"][k], rowvar=False)
                    A = np.eye(P) - sigma ** 2 * sc.bias_op(np.linalg.inv(G + lam * np.eye(P)), DR)
                    Vb = np.cov(D["bcp"][k], rowvar=False)
                    contraction.append(dict(pred=float(np.trace(A @ V @ A.T) / np.trace(V)),
                                            actual=float(np.trace(Vb) / np.trace(V))))
                    rho0s.append(rho0_of(G, sigma))
                    rholams.append(float(np.linalg.eigvalsh(G + lam * np.eye(P))[0] / (2 * sigma * np.sqrt(P))))
                    keeps.append(float(k.mean()))
                    bl_list.append(dict(shrink_sq=float(np.sum((b_lam - b_ols) ** 2)),
                                        samp_sq=float(np.sum((b_ols - BETA_TRUE) ** 2))))
                cell = dict(n=n, eps=eps, sigma=sigma, ell=ell_name, lam=lam,
                            rho0=float(np.mean(rho0s)), rho0_seeds=rho0s,
                            rho_lam=float(np.mean(rholams)), gate_pass=float(np.mean(keeps)),
                            pairs_per_seed=sz["pairs"], seeds=sz["seeds"],
                            vs_lam=agg_compare(per_lam), vs_ols=agg_compare(per_ols),
                            e3=e3_aggregate(per_e3),
                            var_ratio_contraction_pred=float(np.mean([x["pred"] for x in contraction])),
                            var_ratio_plus_draws=float(np.mean([x["actual"] for x in contraction])),
                            shrink_sq_cond=float(np.mean([b["shrink_sq"] for b in bl_list])),
                            samp_sq_cond=float(np.mean([b["samp_sq"] for b in bl_list])),
                            per_seed_vs_lam=[strip_vectors(x) for x in per_lam],
                            bias_vec_seed0=dict(unc=per_lam[0]["unc"]["bias"], bc=per_lam[0]["bc"]["bias"],
                                                se_unc=per_lam[0]["unc"]["bias_se"],
                                                se_bc=per_lam[0]["bc"]["bias_se"],
                                                sd_unc=per_lam[0]["unc"]["sd"]))
                cells[ell_name].append(cell)
                v = cell["vs_lam"]
                log(f"E1 ell={ell_name:4s} n={n:6d} eps={eps:3.1f} rho0={cell['rho0']:7.2f} lam={lam:8.1f} "
                    f"ratio={v['mse_ratio']:.4f}+-{v['mse_ratio_se']:.4f} share_unc={v['unc']['bias_share']:.4f} "
                    f"gain bias/var={v['gain_bias_part']:.4f}/{v['gain_var_part']:.4f} "
                    f"relbias unc/bc={v['unc']['rel_bias']:.2e}/{v['bc']['rel_bias']:.2e} gate={cell['gate_pass']:.4f}")
    log(f"E1 done in {time.time() - t0:.0f}s")
    return cells


# =============================================================================
# FULL mode: coverage vs true beta (+ E2 sampling / shrinkage terms)
# =============================================================================

def beta_clip_population(n_rows=4_000_000, chunk=400_000):
    """Large-sample limit of OLS on the clipped design (clipping moves it slightly off beta_true)."""
    path = os.path.join(WORK, "beta_clip_pop.json")
    if os.path.exists(path):
        return np.array(load(path)["beta_clip_pop"])
    rng = np.random.default_rng([BASE_SEED, 99])
    G, c = np.zeros((P, P)), np.zeros(P)
    for _ in range(n_rows // chunk):
        Z, y, _ = POP.draw(chunk, rng)
        G += Z.T @ Z; c += Z.T @ y
    b = np.linalg.solve(G, c)
    save(dict(beta_clip_pop=b, n_rows=n_rows), path)
    return b


def stage_full(sz):
    t0 = time.time()
    b_clip = beta_clip_population()
    R = sz["full_reps"]
    cells = {ell: [] for ell in ELLS}
    for n in N_GRID:
        Gs, cs, ys = np.empty((R, P, P)), np.empty((R, P)), np.empty(R)
        for r in range(R):
            Gs[r], cs[r], ys[r], _ = dataset(n, r, stream=2)
        b_ols = np.linalg.solve(Gs, cs[..., None])[..., 0]
        lmin = np.linalg.eigvalsh(Gs)[:, 0]
        for ei, eps in enumerate(EPS_GRID):
            sigma = SIGMA[eps]
            rng = np.random.default_rng([BASE_SEED, 20, n, ei])
            E, f, e = sc.release(np.zeros((P, P)), np.zeros(P), 0.0, DR, sigma, R, rng)
            Gt, ct, yt = Gs + E, cs + f, ys + e
            for ell_name, ell in ELLS.items():
                lam = sc.fixed_lambda(sigma, P, n, ell, RHO_STAR)
                b, bc, Pm = sc.solve(Gt, ct, lam, sigma, DR)
                keep = sc.rho_hat(Gt, lam, sigma) >= GATE
                cov_u, _ = sc.first_order_cov(Gt, ct, yt, b, Pm, sigma, n, DR)
                cov_c, _ = sc.first_order_cov(Gt, ct, yt, bc, Pm, sigma, n, DR)
                se_u = np.sqrt(np.einsum("rii->ri", cov_u))
                se_c = np.sqrt(np.einsum("rii->ri", cov_c))
                b_lam = np.linalg.solve(Gs + lam * np.eye(P)[None], cs[..., None])[..., 0]
                k = keep
                cell = dict(n=n, eps=eps, sigma=sigma, ell=ell_name, lam=lam, reps=R,
                            gate_pass=float(k.mean()),
                            rho0=float(np.mean(lmin / (2 * sigma * np.sqrt(P)))))
                for tag, est, se in (("unc", b, se_u), ("bc", bc, se_c)):
                    est, se = est[k], se[k]
                    cov_true = np.abs(est - BETA_TRUE) <= Z95 * se
                    cov_clip = np.abs(est - b_clip) <= Z95 * se
                    emp_sd = (est - BETA_TRUE).std(0, ddof=1)
                    cell[tag] = dict(
                        cover_true=cov_true.mean(0), cover_true_mean=float(cov_true.mean()),
                        cover_true_slopes=float(cov_true[:, 1:].mean()),
                        cover_true_min=float(cov_true.mean(0).min()),
                        cover_clip=cov_clip.mean(0), cover_clip_mean=float(cov_clip.mean()),
                        se_calib=(se.mean(0) / emp_sd),            # plug-in SE / empirical SD
                        se_calib_mean=float(np.mean(se.mean(0) / emp_sd)),
                        mean_se=float(se.mean()),
                        mse_true=float(np.mean(np.sum((est - BETA_TRUE) ** 2, 1))),
                        mse_mech=float(np.mean(np.sum((est - b_lam[k]) ** 2, 1))),
                        bias_true=(est - BETA_TRUE).mean(0),
                        bias_true_norm=float(np.linalg.norm((est - BETA_TRUE).mean(0))))
                cell["shrink_sq"] = float(np.mean(np.sum((b_lam - b_ols) ** 2, 1)))
                cell["samp_sq"] = float(np.mean(np.sum((b_ols - BETA_TRUE) ** 2, 1)))
                cell["samp_sq_clip"] = float(np.mean(np.sum((b_ols - b_clip) ** 2, 1)))
                cell["width_ratio_bc_unc"] = float(np.mean(se_c[k] / se_u[k]))
                au = np.sum((b[k] - BETA_TRUE) ** 2, 1); ac = np.sum((bc[k] - BETA_TRUE) ** 2, 1)
                cell["mse_ratio_true"], cell["mse_ratio_true_se"] = paired_ratio(ac, au)
                cells[ell_name].append(cell)
                log(f"FULL ell={ell_name:4s} n={n:6d} eps={eps:3.1f} rho0={cell['rho0']:7.2f} "
                    f"cover unc/bc={cell['unc']['cover_true_mean']:.3f}/{cell['bc']['cover_true_mean']:.3f} "
                    f"(clip-pop {cell['unc']['cover_clip_mean']:.3f}/{cell['bc']['cover_clip_mean']:.3f}) "
                    f"se_calib={cell['unc']['se_calib_mean']:.3f} gate={cell['gate_pass']:.3f}")
        log(f"FULL n={n} done ({time.time() - t0:.0f}s)")
    return dict(cells=cells, beta_clip_pop=b_clip, beta_true=BETA_TRUE)


# =============================================================================
# E1s: stress test for the rule threshold
# =============================================================================

def stage_stress(sz):
    t0 = time.time()
    out = dict(ridge=[], ols=[])
    for eps, n in STRESS_RIDGE_CELLS:
        ei = EPS_GRID.index(eps)
        sigma = SIGMA[eps]
        data = [dataset(n, i) for i in range(sz["stress_seeds"])]
        for ti, rt in enumerate(STRESS_RIDGE_TARGETS):
            per, keeps, rl = [], [], []
            for i, (G, c, yty, _) in enumerate(data):
                lam = max(0.0, 2 * rt * sigma * np.sqrt(P) - np.linalg.eigvalsh(G)[0])   # oracle
                rl.append(float(np.linalg.eigvalsh(G + lam * np.eye(P))[0] / (2 * sigma * np.sqrt(P))))
                rng = np.random.default_rng([BASE_SEED, 30, n, i, ei, ti])
                D = draw_pairs(G, c, yty, sigma, lam, sz["stress_pairs"], rng)
                k = D["keep"]
                keeps.append(float(k.mean()))
                if k.sum() < 10:
                    continue
                cl, _, _ = compare(D["bp"][k], D["bm"][k], D["bcp"][k], D["bcm"][k], ridge_target(G, c, lam))
                per.append(cl)
            if not per:
                out["ridge"].append(dict(eps=eps, n=n, rho_target=rt, rho_lam=float(np.mean(rl)),
                                         gate_pass=float(np.mean(keeps)), vs_lam=None))
                log(f"STRESS ridge eps={eps} n={n} rho_lam={rt:4.2f}: gate pass {np.mean(keeps):.4f}, too few pairs")
                continue
            a = agg_compare(per)
            out["ridge"].append(dict(eps=eps, n=n, rho_target=rt, rho_lam=float(np.mean(rl)),
                                     gate_pass=float(np.mean(keeps)), vs_lam=a))
            log(f"STRESS ridge eps={eps} n={n} rho_lam={rt:4.2f} ratio={a['mse_ratio']:.4f}+-{a['mse_ratio_se']:.4f} "
                f"share={a['unc']['bias_share']:.4f} gate={np.mean(keeps):.4f}")
    for eps in STRESS_OLS_EPS:
        ei = EPS_GRID.index(eps)
        sigma = SIGMA[eps]
        for ti, rt in enumerate(STRESS_OLS_TARGETS):
            n = int(round(rt * 2 * sigma * np.sqrt(P) / POP.lam_min_pop))
            per, keeps, r0 = [], [], []
            for i in range(sz["stress_seeds"]):
                G, c, yty, _ = dataset(n, i, stream=4)
                rng = np.random.default_rng([BASE_SEED, 31, n, i, ei])
                D = draw_pairs(G, c, yty, sigma, 0.0, sz["stress_pairs"], rng)
                k = D["keep"]
                keeps.append(float(k.mean()))
                if k.sum() < 10:
                    continue
                cl, _, _ = compare(D["bp"][k], D["bm"][k], D["bcp"][k], D["bcm"][k], np.linalg.solve(G, c))
                per.append(cl); r0.append(rho0_of(G, sigma))
            if not per:
                out["ols"].append(dict(eps=eps, n=n, rho_target=rt, rho0=None, gate_pass=0.0, vs_ols=None))
                log(f"STRESS ols eps={eps} n={n} rho_target={rt}: gate never passed")
                continue
            a = agg_compare(per)
            out["ols"].append(dict(eps=eps, n=n, rho_target=rt, rho0=float(np.mean(r0)),
                                   gate_pass=float(np.mean(keeps)), vs_ols=a))
            log(f"STRESS ols eps={eps} n={n} rho0={np.mean(r0):4.2f} ratio={a['mse_ratio']:.4f}+-{a['mse_ratio_se']:.4f} "
                f"share={a['unc']['bias_share']:.4f} gate={np.mean(keeps):.4f}")
    log(f"STRESS done in {time.time() - t0:.0f}s")
    return out


# =============================================================================
# E1b: signal-strength sensitivity (mechanism check, not a recalibrated population)
# =============================================================================

def stage_signal(sz):
    t0 = time.time()
    out = []
    for eps, n in SIGNAL_CELLS:
        ei = EPS_GRID.index(eps)
        sigma = SIGMA[eps]
        lam = sc.fixed_lambda(sigma, P, n, ELL_MAIN, RHO_STAR)
        data = [dataset(n, i) for i in range(sz["signal_seeds"])]
        for k in SIGNAL_K:
            per, norms, r0 = [], [], []
            for i, (G, c, yty, _) in enumerate(data):
                ck = k * c
                rng = np.random.default_rng([BASE_SEED, 50, n, i, ei])      # same noise for every k
                D = draw_pairs(G, ck, k * k * yty, sigma, lam, sz["signal_pairs"], rng)
                kp = D["keep"]
                cl, _, _ = compare(D["bp"][kp], D["bm"][kp], D["bcp"][kp], D["bcm"][kp], ridge_target(G, ck, lam))
                per.append(cl); norms.append(float(np.linalg.norm(np.linalg.solve(G, ck))))
                r0.append(rho0_of(G, sigma))
            a = agg_compare(per)
            out.append(dict(eps=eps, n=n, k=k, lam=lam, rho0=float(np.mean(r0)),
                            beta_ols_norm=float(np.mean(norms)), vs_lam=a))
            log(f"SIGNAL eps={eps} n={n} k={k} |beta|={np.mean(norms):.2f} rho0={np.mean(r0):.2f} "
                f"ratio={a['mse_ratio']:.4f}+-{a['mse_ratio_se']:.4f} gain bias/var={a['gain_bias_part']:.4f}/"
                f"{a['gain_var_part']:.4f}")
    log(f"SIGNAL done in {time.time() - t0:.0f}s")
    return out


# =============================================================================
# E4: fixed lambda vs the adaptive gate
# =============================================================================

def e4_cells():
    cells = []
    for eps in EPS_GRID:
        for n in N_GRID:
            r = n * POP.lam_min_pop / (2 * SIGMA[eps] * np.sqrt(P))
            if E4_RHO_RANGE[0] <= r <= E4_RHO_RANGE[1]:
                cells.append((eps, n))
    return cells


def stage_e4(sz):
    t0 = time.time()
    out = []
    for eps, n in e4_cells():
        ei = EPS_GRID.index(eps)
        sigma = SIGMA[eps]
        lam = sc.fixed_lambda(sigma, P, n, ELL_MAIN, RHO_STAR)
        acc = {k: [] for k in ("fx_lam", "fx_ols", "ad_tgt", "ad_ols", "ad_vs_fx")}
        lam_ad, lam_ad_cv, ridge_frac, lam_star, modes, rho0s, keeps = [], [], [], [], [], [], []
        for i in range(sz["e4_seeds"]):
            G, c, yty, _ = dataset(n, i)
            rng = np.random.default_rng([BASE_SEED, 40, n, i, ei])
            D = draw_pairs(G, c, yty, sigma, lam, sz["e4_pairs"], rng, chunk=500, adaptive=True)
            k = D["keep"]
            b_ols = np.linalg.solve(G, c)
            ad_t, _, ls = sc.adaptive_gate(G[None], c[None], sigma, DR, DO)      # gate on clean G
            mode = "O" if np.linalg.eigvalsh(G[:DR, :DR])[0] / (2 * sigma * np.sqrt(P)) >= 2.0 else "I"
            fx = (D["bp"][k], D["bm"][k], D["bcp"][k], D["bcm"][k])
            ad = (D["abp"][k], D["abm"][k], D["abcp"][k], D["abcm"][k])
            c1, _, Cf = compare(*fx, ridge_target(G, c, lam))
            c2, _, Cfo = compare(*fx, b_ols)
            c3, _, _ = compare(*ad, ad_t[0])
            c4, _, Cao = compare(*ad, b_ols)
            acc["fx_lam"].append(c1); acc["fx_ols"].append(c2)
            acc["ad_tgt"].append(c3); acc["ad_ols"].append(c4)
            R, se = paired_ratio(Cao["a"], Cfo["a"])       # adaptive-bc vs fixed-bc, both vs OLS
            acc["ad_vs_fx"].append((R, se))
            la = np.concatenate([D["lamp"][k], D["lamm"][k]])
            lam_ad.append(float(la.mean())); lam_ad_cv.append(float(la.std() / la.mean()))
            ridge_frac.append(float(np.mean(la > 1.5e-3)))
            lam_star.append(float(ls[0])); modes.append(mode); rho0s.append(rho0_of(G, sigma))
            keeps.append(float(k.mean()))
        r = np.array([x[0] for x in acc["ad_vs_fx"]]); s = np.array([x[1] for x in acc["ad_vs_fx"]])
        cell = dict(eps=eps, n=n, sigma=sigma, lam_fixed=lam, rho0=float(np.mean(rho0s)),
                    lam_adaptive_mean=float(np.mean(lam_ad)), lam_adaptive_cv=float(np.mean(lam_ad_cv)),
                    lam_adaptive_clean=float(np.mean(lam_star)), adaptive_ridge_frac=float(np.mean(ridge_frac)),
                    adaptive_mode=modes, gate_pass=float(np.mean(keeps)),
                    pairs_per_seed=sz["e4_pairs"], seeds=sz["e4_seeds"],
                    fixed_vs_lam=agg_compare(acc["fx_lam"]), fixed_vs_ols=agg_compare(acc["fx_ols"]),
                    adaptive_vs_target=agg_compare(acc["ad_tgt"]), adaptive_vs_ols=agg_compare(acc["ad_ols"]),
                    adaptive_bc_over_fixed_bc_vs_ols=dict(ratio=float(r.mean()),
                                                          se=float(np.sqrt((s ** 2).sum()) / len(s))))
        for key in ("fixed_vs_lam", "adaptive_vs_target"):
            u, b = cell[key]["unc"]["bias_norm"], cell[key]["bc"]["bias_norm"]
            cell[key]["residual_over_bias"] = b / u if u > 0 else float("nan")
        out.append(cell)
        log(f"E4 eps={eps} n={n} rho0={cell['rho0']:.2f} mode={modes[0]} lam fx={lam:.1f} ad={cell['lam_adaptive_mean']:.1f} "
            f"(cv {cell['lam_adaptive_cv']:.2f}) resid/bias fx={cell['fixed_vs_lam']['residual_over_bias']:.3f} "
            f"ad={cell['adaptive_vs_target']['residual_over_bias']:.3f} ratio fx={cell['fixed_vs_lam']['mse_ratio']:.4f} "
            f"ad={cell['adaptive_vs_target']['mse_ratio']:.4f} ({time.time() - t0:.0f}s)")
    log(f"E4 done in {time.time() - t0:.0f}s")
    return out


# =============================================================================
# report stage: merge, tables, figures
# =============================================================================

C_BC, C_UNC = "#2a78d6", "#eb6834"
C_AQUA, C_YELLOW, C_MAGENTA = "#1baf7a", "#eda100", "#e87ba4"
RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]
INK, INK2, GRID_C, SPINE, NEUTRAL = "#0b0b0b", "#52514e", "#e6e6e3", "#c8c8c4", "#a3a29d"
EPS_COLOR = dict(zip(EPS_GRID, RAMP))
K_COLOR = dict(zip(K_GRID, [RAMP[0], RAMP[2], RAMP[3], RAMP[4]]))
BL = r"$\hat{\beta}_\lambda$"
R0 = r"$\rho_0$"


def plot_style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8.5, "axes.titlesize": 9,
        "axes.labelsize": 8.5, "legend.fontsize": 7.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "text.color": INK, "axes.labelcolor": INK, "axes.titlecolor": INK,
        "xtick.color": SPINE, "ytick.color": SPINE, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "axes.edgecolor": SPINE, "axes.linewidth": 0.8,
        "axes.grid": True, "axes.grid.which": "major", "grid.color": GRID_C, "grid.linewidth": 0.6,
        "grid.linestyle": "-", "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False,
        "lines.linewidth": 1.5, "lines.markersize": 5, "lines.markeredgewidth": 0,
        "legend.frameon": False, "legend.labelcolor": INK,
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "figure.dpi": 100, "mathtext.fontset": "dejavusans"})
    return plt


def log_x(ax, ticks=None):
    """Log x-axis; on narrow ranges a few plainly labelled ticks (no colliding minor labels)."""
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
    ax.set_xscale("log")
    if ticks is not None:
        ax.xaxis.set_major_locator(FixedLocator(ticks))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.xaxis.set_minor_formatter(NullFormatter())


def savefig(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=200, bbox_inches="tight")
    fig.savefig(os.path.join(OUT, name + ".pdf"), bbox_inches="tight")


def ridge_marker(ax, label=True, y_text=0.97, va="top"):
    ax.axvline(RHO_SWITCH, color=INK2, lw=0.8, ls=(0, (3, 3)), zorder=1)
    if label:
        kw = dict(transform=ax.get_xaxis_transform(), va=va, color=INK2, fontsize=7)
        ax.text(RHO_SWITCH / 1.12, y_text, "ridge on", ha="right", **kw)
        ax.text(RHO_SWITCH * 1.12, y_text, r"$\lambda = 0$", ha="left", **kw)


def by_eps(cells, eps):
    return sorted([c for c in cells if c["eps"] == eps], key=lambda c: c["rho0"])


def fig_mse_ratio(plt, e1):
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6))
    panels = [(e1["main"], "vs_lam", r"(a) vs protocol target $\hat{\beta}_\lambda$   ($\ell = 0.8\,\lambda_{min}$)"),
              (e1["main"], "vs_ols", r"(b) vs sample OLS $\hat{\beta}$   ($\ell = 0.8\,\lambda_{min}$)"),
              (e1["zero"], "vs_lam", r"(c) vs $\hat{\beta}_\lambda$,  $\ell = 0$ (ridge always on)")]
    for pi, (ax, (cells, key, title)) in enumerate(zip(axes, panels)):
        for eps in EPS_GRID:
            cs = by_eps(cells, eps)
            x = np.array([c["rho0"] for c in cs])
            y = np.array([c[key]["mse_ratio"] for c in cs])
            s = np.array([c[key]["mse_ratio_se"] for c in cs])
            ax.fill_between(x, y - Z95 * s, y + Z95 * s, color=EPS_COLOR[eps], alpha=0.18, lw=0)
            ax.plot(x, y, "-o", color=EPS_COLOR[eps], label=f"ε = {eps:g}")
        ax.axhline(1.0, color=INK2, lw=0.9, zorder=1)
        ax.set_xscale("log")
        ax.set_xlabel(r"$\rho_0 = \lambda_{min}(G)\,/\,(2\sigma\sqrt{p})$")
        ax.set_title(title, loc="left")
        if pi < 2:
            ridge_marker(ax, label=(pi == 0), y_text=0.04, va="bottom")
    axes[0].set_ylabel("MSE(corrected) / MSE(uncorrected)")
    lo = min(axes[0].get_ylim()[0], axes[2].get_ylim()[0])
    for ax in (axes[0], axes[2]):
        ax.set_ylim(lo, 1.012)
    axes[2].legend(loc="lower right")
    fig.tight_layout()
    savefig(fig, "fig1_mse_ratio")
    plt.close(fig)


def fig_decomposition(plt, e1, full):
    fig, axes = plt.subplots(1, 5, figsize=(14, 3.3), sharey=True)
    fm = {(c["n"], c["eps"]): c for c in full["cells"]["main"]}
    for ax, eps in zip(axes, EPS_GRID):
        cs = by_eps(e1["main"], eps)
        x = np.array([c["rho0"] for c in cs])
        shrink = np.array([fm[(c["n"], eps)]["shrink_sq"] for c in cs])
        samp = np.array([fm[(c["n"], eps)]["samp_sq"] for c in cs])
        mb = np.array([c["vs_lam"]["unc"]["bias_sq"] for c in cs])
        mv = np.array([c["vs_lam"]["unc"]["trV"] for c in cs])
        pos = shrink > 0
        ax.plot(x[pos], shrink[pos], "-o", color=C_AQUA,
                label=r"(i) ridge shrinkage $\|\hat{\beta}_\lambda - \hat{\beta}\|^2$")
        ax.plot(x, mb, "-o", color=C_UNC, label="(ii) mechanism bias² (what the correction removes)")
        ax.plot(x, mv, "-o", color=C_YELLOW, label=r"(iii) mechanism variance tr Var$(\tilde{\beta})$")
        ax.plot(x, samp, "-o", color=C_MAGENTA, label=r"(iv) sampling $\|\hat{\beta} - \beta_{true}\|^2$")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_title(f"ε = {eps:g}   (σ = {SIGMA[eps]:.0f})", loc="left")
        ax.set_xlabel(R0)
        ridge_marker(ax, y_text=0.04, va="bottom")
    axes[0].set_ylabel("squared error (sum over 11 coefficients)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    savefig(fig, "fig2_decomposition")
    plt.close(fig)


def _getk(d, K):
    return d.get(str(K), d.get(K)) if isinstance(d, dict) else None


def fig_releases(plt, e1):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    ax = axes[0]
    cs = by_eps(e1["main"], 1.0)
    x = np.array([c["rho0"] for c in cs])
    for K in K_GRID:
        ya = np.array([_getk(c["e3"]["analytic_ratio_at"], K) for c in cs])
        ax.plot(x, ya, "-", color=K_COLOR[K], label=f"K = {K}")
        dk = [_getk(c["e3"]["direct"], K) for c in cs]
        ok = np.array([d is not None for d in dk])
        if ok.any():
            yd = np.array([d["ratio"] for d in dk if d is not None])
            sd = np.array([d["ratio_se"] for d in dk if d is not None])
            ax.errorbar(x[ok], yd, yerr=Z95 * sd, fmt="o", color=K_COLOR[K], ms=4, lw=1, capsize=0)
    ax.axhline(1.0, color=INK2, lw=0.9, zorder=1)
    ax.set_xscale("log")
    ax.set_xlabel(r"$\rho_0$   ($\varepsilon = 1$)")
    ax.set_ylabel("MSE(corrected avg) / MSE(uncorrected avg)")
    ax.set_title(r"(a) average of K releases, vs $\hat{\beta}_\lambda$  (lines: formula; dots: direct)", loc="left")
    ridge_marker(ax, y_text=0.99, va="top")
    ax.legend(loc="lower right")
    ax = axes[1]
    cells = sorted(e1["main"], key=lambda c: c["rho0"])
    x = np.array([c["rho0"] for c in cells])
    kbv = np.array([c["e3"]["K_bias_eq_var"] for c in cells], dtype=float)
    k10 = np.array([c["e3"]["K_cross_10pct"] for c in cells], dtype=float)
    ax.plot(x, kbv, "o", color=C_AQUA, ms=4.5,
            label="bias² = variance in the uncorrected average  (K = tr Var / ‖bias‖²)")
    ax.plot(x, k10, "o", color=C_YELLOW, ms=4.5, label="correction cuts the average's MSE by ≥ 10%")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$\rho_0$   (all $\varepsilon$ pooled)")
    ax.set_ylabel("number of averaged releases K")
    ax.set_title("(b) how many releases before the correction matters", loc="left")
    ridge_marker(ax, y_text=0.04, va="bottom")
    ax.legend(loc="upper left")
    fig.tight_layout()
    savefig(fig, "fig3_multiple_releases")
    plt.close(fig)


def fig_coverage(plt, full):
    fig, axes = plt.subplots(1, 5, figsize=(14, 3.3), sharey=True)
    for ax, eps in zip(axes, EPS_GRID):
        cs = by_eps(full["cells"]["main"], eps)
        x = np.array([c["rho0"] for c in cs])
        ax.axhline(0.95, color=INK2, lw=0.9, zorder=1)
        ax.plot(x, [c["unc"]["cover_true_mean"] for c in cs], "-o", color=C_UNC, label="uncorrected, vs true β")
        ax.plot(x, [c["bc"]["cover_true_mean"] for c in cs], "-o", color=C_BC, ms=3.5,
                label="bias-corrected, vs true β")
        ax.plot(x, [c["unc"]["cover_clip_mean"] for c in cs], "-", color=NEUTRAL, lw=1.1,
                label="uncorrected, vs clipped-population β (large-n limit of the clipped design)")
        ax.set_xscale("log")
        ax.set_title(f"ε = {eps:g}", loc="left")
        ax.set_xlabel(R0)
        ridge_marker(ax, y_text=0.04, va="bottom")
    axes[0].set_ylabel("coverage of 95% Wald CIs\n(mean over 11 coefficients)")
    axes[0].set_ylim(0, 1.02)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    savefig(fig, "fig4_coverage")
    plt.close(fig)


def fig_bias(plt, e1):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.7))
    cells = e1["main"]
    x = np.array([c["rho0"] for c in cells])
    o = np.argsort(x)
    ax = axes[0]
    for tag, col, lab in (("unc", C_UNC, "uncorrected"), ("bc", C_BC, "bias-corrected")):
        y = np.array([c["vs_lam"][tag]["bias_norm_raw"] / c["vs_lam"]["tnorm"] for c in cells])
        ax.plot(x[o], y[o], "o", color=col, label=lab, ms=4)
    res = np.array([c["vs_lam"]["unc"]["bias_mcres"] / c["vs_lam"]["tnorm"] for c in cells])
    ax.plot(x[o], res[o], "-", color=NEUTRAL, lw=1.2, label="Monte-Carlo resolution")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$\rho_0$   (all $\varepsilon$ pooled)")
    ax.set_ylabel(r"$\|E[\tilde{\beta}] - \hat{\beta}_\lambda\| \,/\, \|\hat{\beta}_\lambda\|$")
    ax.set_title("(a) relative mechanism bias", loc="left")
    ridge_marker(ax, y_text=0.04, va="bottom")
    ax.legend(loc="upper right")
    ax = axes[1]
    for tag, col, lab in (("unc", C_UNC, "uncorrected"), ("bc", C_BC, "bias-corrected")):
        y = np.array([c["vs_lam"][tag]["max_bias_over_sd"] for c in cells])
        ax.plot(x[o], y[o], "o", color=col, label=lab, ms=4)
    fl = np.array([c["vs_lam"]["unc"]["mc_floor_over_sd"] for c in cells])
    ax.plot(x[o], 2 * fl[o], "-", color=NEUTRAL, lw=1.2, label="2 × Monte-Carlo SE")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$\rho_0$   (all $\varepsilon$ pooled)")
    ax.set_ylabel(r"$\max_j\ |\mathrm{bias}_j|\,/\,\mathrm{sd}_j$   (one release)")
    ax.set_title("(b) worst coefficient's bias in sd units", loc="left")
    ridge_marker(ax, y_text=0.04, va="bottom")
    ax.legend(loc="upper right")
    fig.tight_layout()
    savefig(fig, "fig5_bias")
    plt.close(fig)


def fig_e4(plt, e4):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.7))
    cs = sorted(e4, key=lambda c: c["rho0"])
    x = np.array([c["rho0"] for c in cs])
    ticks = [0.25, 0.5, 1, 2, 4]
    for ax, key, title in ((axes[0], "fixed_vs_lam", r"(a) fixed $\lambda$ (eq. lambda_rule): bias vs $\hat{\beta}_\lambda$"),
                           (axes[1], "adaptive_vs_target", "(b) adaptive gate: bias vs its clean-data target")):
        for tag, col, lab in (("unc", C_UNC, "uncorrected"), ("bc", C_BC, "bias-corrected")):
            y = np.array([c[key][tag]["bias_norm_raw"] for c in cs])
            ax.plot(x, y, "o", color=col, label=lab, ms=4)
        res = np.array([c[key]["unc"]["bias_mcres"] for c in cs])
        ax.plot(x, res, "-", color=NEUTRAL, lw=1.2, label="Monte-Carlo resolution")
        log_x(ax, ticks); ax.set_yscale("log")
        ax.set_xlabel(R0)
        ax.set_ylabel(r"$\|$bias$\|$")
        ax.set_title(title, loc="left")
        ax.legend(loc="lower left")
    ylo = min(axes[0].get_ylim()[0], axes[1].get_ylim()[0]); yhi = max(axes[0].get_ylim()[1], axes[1].get_ylim()[1])
    axes[0].set_ylim(ylo, yhi); axes[1].set_ylim(ylo, yhi)
    ax = axes[2]
    for key, col, lab in (("fixed_vs_lam", C_AQUA, r"fixed $\lambda$"), ("adaptive_vs_target", C_YELLOW, "adaptive gate")):
        y = np.array([c[key]["mse_ratio"] for c in cs]); s = np.array([c[key]["mse_ratio_se"] for c in cs])
        ax.errorbar(x, y, yerr=Z95 * s, fmt="o", color=col, label=lab, ms=4, lw=1, capsize=0)
    ax.axhline(1.0, color=INK2, lw=0.9, zorder=1)
    log_x(ax, ticks)
    ax.set_xlabel(R0)
    ax.set_ylabel("MSE(corrected) / MSE(uncorrected)")
    ax.set_title("(c) value of the correction vs own target", loc="left")
    ax.legend(loc="lower right")
    fig.tight_layout()
    savefig(fig, "fig6_fixed_vs_adaptive")
    plt.close(fig)


def fig_stress(plt, st):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.7), sharey=True)
    ticks = [1.25, 1.5, 2, 3, 4, 6]
    ax = axes[0]
    for eps, n in STRESS_RIDGE_CELLS:
        rows = sorted([r for r in st["ridge"] if r["eps"] == eps and r["n"] == n and r["vs_lam"]],
                      key=lambda r: r["rho_lam"])
        x = np.array([r["rho_lam"] for r in rows]); y = np.array([r["vs_lam"]["mse_ratio"] for r in rows])
        s = np.array([r["vs_lam"]["mse_ratio_se"] for r in rows])
        ax.fill_between(x, y - Z95 * s, y + Z95 * s, color=EPS_COLOR[eps], alpha=0.18, lw=0)
        ax.plot(x, y, "-o", color=EPS_COLOR[eps], label=f"ε = {eps:g}, n = {n:,}")
    ax.axhline(1.0, color=INK2, lw=0.9, zorder=1)
    ax.axvline(RHO_STAR, color=INK2, lw=0.8, ls=(0, (3, 3)))
    ax.text(RHO_STAR * 1.03, 0.04, r"$\rho^* = 2$", transform=ax.get_xaxis_transform(), color=INK2,
            fontsize=7, va="bottom")
    log_x(ax, ticks)
    ax.set_xlabel(r"$\rho_\lambda = \lambda_{min}(G + \lambda I)\,/\,(2\sigma\sqrt{p})$   (oracle ridge)")
    ax.set_ylabel("MSE(corrected) / MSE(uncorrected)")
    ax.set_title(r"(a) ridge regime, $\lambda$ set to hit $\rho_\lambda$; gate $\hat{\rho} \geq 1$", loc="left")
    ax.legend(loc="lower right")
    ax = axes[1]
    for eps in STRESS_OLS_EPS:
        rows = sorted([r for r in st["ols"] if r["eps"] == eps and r["vs_ols"]], key=lambda r: r["rho0"])
        x = np.array([r["rho0"] for r in rows]); y = np.array([r["vs_ols"]["mse_ratio"] for r in rows])
        s = np.array([r["vs_ols"]["mse_ratio_se"] for r in rows])
        ax.fill_between(x, y - Z95 * s, y + Z95 * s, color=EPS_COLOR[eps], alpha=0.18, lw=0)
        ax.plot(x, y, "-o", color=EPS_COLOR[eps], label=f"ε = {eps:g}")
    ax.axhline(1.0, color=INK2, lw=0.9, zorder=1)
    log_x(ax, ticks)
    ax.set_xlabel(r"$\rho_0$   ($\lambda = 0$, n chosen to hit $\rho_0$)")
    ax.set_title(r"(b) no ridge (plain OLS); gate $\hat{\rho} \geq 1$", loc="left")
    ax.legend(loc="lower right")
    fig.tight_layout()
    savefig(fig, "fig7_stress_threshold")
    plt.close(fig)


def fig_signal(plt, sig):
    fig, ax = plt.subplots(1, 1, figsize=(5.6, 3.6))
    cells = sorted({(r["eps"], r["n"]) for r in sig}, key=lambda t: t[1])
    cols = [RAMP[0], RAMP[2], RAMP[3], RAMP[4]]
    for col, (eps, n) in zip(cols, cells):
        rows = sorted([r for r in sig if r["eps"] == eps and r["n"] == n], key=lambda r: r["k"])
        x = np.array([r["beta_ols_norm"] for r in rows]); y = np.array([r["vs_lam"]["mse_ratio"] for r in rows])
        s = np.array([r["vs_lam"]["mse_ratio_se"] for r in rows])
        ax.fill_between(x, y - Z95 * s, y + Z95 * s, color=col, alpha=0.18, lw=0)
        ax.plot(x, y, "-o", color=col, label=f"n = {n:,}  ($\\rho_0$ = {rows[0]['rho0']:.1f}, $\\lambda$ = {rows[0]['lam']:.0f})")
    ax.axhline(1.0, color=INK2, lw=0.9, zorder=1)
    log_x(ax, [0.75, 1.5, 3, 6])
    ax.set_xlabel(r"$\|\hat{\beta}\|$   (signal scaled by k = 1, 2, 4, 8; $\varepsilon$ = 1)")
    ax.set_ylabel("MSE(corrected) / MSE(uncorrected), vs " + BL)
    ax.set_title("value of the correction vs signal strength", loc="left")
    ax.legend(loc="upper left")
    fig.tight_layout()
    savefig(fig, "fig8_signal_strength")
    plt.close(fig)


def fmt_ratio(v):
    return f"{v['mse_ratio']:.4f} ± {Z95 * v['mse_ratio_se']:.4f}"


def tables(res):
    e1, full = res["E1"]["main"], res["FULL"]["cells"]["main"]
    fm = {(c["n"], c["eps"]): c for c in full}
    lines = ["| ε | n | ρ₀ | λ | rel. bias unc → bc (vs β̂_λ) | bias² share unc | "
             "MSE ratio bc/unc vs β̂_λ (95% CI) | gain from bias / from variance | MSE ratio vs OLS β̂ | "
             "max|b_j|/sd_j unc → bc | coverage unc / bc (true β) |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in sorted(e1, key=lambda c: (c["eps"], c["n"])):
        v, o, f = c["vs_lam"], c["vs_ols"], fm[(c["n"], c["eps"])]
        lines.append(f"| {c['eps']:g} | {c['n']:,} | {c['rho0']:.2f} | {c['lam']:.0f} | "
                     f"{v['unc']['rel_bias']:.2e} → {v['bc']['rel_bias']:.1e} | {v['unc']['bias_share']:.4f} | "
                     f"{fmt_ratio(v)} | {v['gain_bias_part']:.4f} / {v['gain_var_part']:.4f} | "
                     f"{o['mse_ratio']:.4f} | {v['unc']['max_bias_over_sd']:.3f} → "
                     f"{v['bc']['max_bias_over_sd']:.3f} | {f['unc']['cover_true_mean']:.3f} / {f['bc']['cover_true_mean']:.3f} |")
    return "\n".join(lines)


def write_csv(res):
    path = os.path.join(OUT, "e1_value_map.csv")
    fm = {(c["n"], c["eps"], c["ell"]): c for ell in ELLS for c in res["FULL"]["cells"][ell]}
    cols = ["ell", "eps", "n", "sigma", "lam", "rho0", "rho_lam", "gate_pass",
            "relbias_unc", "relbias_bc", "bias_share_unc", "bias_share_bc", "mse_ratio_vs_lam",
            "mse_ratio_vs_lam_ci95", "gain_bias_part", "gain_var_part", "var_ratio", "mse_ratio_vs_ols", "mse_ratio_vs_ols_ci95", "max_bias_sd_unc",
            "max_bias_sd_bc", "mc_floor_sd", "K_cross_10pct", "K_bias_eq_var",
            "cover_true_unc", "cover_true_bc", "cover_clip_unc", "cover_clip_bc", "se_calib_unc",
            "mse_true_unc", "mse_true_bc", "shrink_sq", "samp_sq", "mech_bias_sq_unc", "mech_var_unc"]
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for ell in ELLS:
            for c in sorted(res["E1"][ell], key=lambda c: (c["eps"], c["n"])):
                v, o, f = c["vs_lam"], c["vs_ols"], fm[(c["n"], c["eps"], ell)]
                w.writerow([ell, c["eps"], c["n"], f"{c['sigma']:.4f}", f"{c['lam']:.2f}", f"{c['rho0']:.4f}",
                            f"{c['rho_lam']:.4f}", f"{c['gate_pass']:.5f}",
                            f"{v['unc']['rel_bias']:.4e}", f"{v['bc']['rel_bias']:.4e}",
                            f"{v['unc']['bias_share']:.5f}", f"{v['bc']['bias_share']:.6f}",
                            f"{v['mse_ratio']:.5f}", f"{Z95 * v['mse_ratio_se']:.5f}",
                            f"{v['gain_bias_part']:.5f}", f"{v['gain_var_part']:.5f}", f"{v['var_ratio']:.5f}",
                            f"{o['mse_ratio']:.5f}", f"{Z95 * o['mse_ratio_se']:.5f}",
                            f"{v['unc']['max_bias_over_sd']:.4f}", f"{v['bc']['max_bias_over_sd']:.4f}",
                            f"{v['unc']['mc_floor_over_sd']:.4f}", f"{c['e3']['K_cross_10pct']:.4g}",
                            f"{c['e3']['K_bias_eq_var']:.4g}",
                            f"{f['unc']['cover_true_mean']:.4f}", f"{f['bc']['cover_true_mean']:.4f}",
                            f"{f['unc']['cover_clip_mean']:.4f}", f"{f['bc']['cover_clip_mean']:.4f}",
                            f"{f['unc']['se_calib_mean']:.4f}",
                            f"{f['unc']['mse_true']:.5e}", f"{f['bc']['mse_true']:.5e}",
                            f"{f['shrink_sq']:.5e}", f"{f['samp_sq']:.5e}",
                            f"{v['unc']['bias_sq']:.5e}", f"{v['unc']['trV']:.5e}"])
    return path


def e2_table(res):
    fm = {(c["n"], c["eps"]): c for c in res["FULL"]["cells"]["main"]}
    rows = []
    for c in sorted(res["E1"]["main"], key=lambda c: (c["eps"], c["n"])):
        f = fm[(c["n"], c["eps"])]
        comp = dict(shrink=f["shrink_sq"], mech_bias_sq=c["vs_lam"]["unc"]["bias_sq"],
                    mech_bias_sq_bc=c["vs_lam"]["bc"]["bias_sq"], mech_var=c["vs_lam"]["unc"]["trV"],
                    mech_var_bc=c["vs_lam"]["bc"]["trV"], sampling=f["samp_sq"],
                    total_full_unc=f["unc"]["mse_true"], total_full_bc=f["bc"]["mse_true"])
        comp["sum_components"] = comp["shrink"] + comp["mech_bias_sq"] + comp["mech_var"] + comp["sampling"]
        comp["cross_terms"] = comp["total_full_unc"] - comp["sum_components"]
        comp["mech_bias_share_of_total"] = comp["mech_bias_sq"] / comp["total_full_unc"]
        rows.append(dict(eps=c["eps"], n=c["n"], rho0=c["rho0"], lam=c["lam"], **comp))
    return rows


def stage_report(res):
    plt = plot_style()
    res["E2"] = e2_table(res)
    fig_mse_ratio(plt, res["E1"])
    fig_decomposition(plt, res["E1"], res["FULL"])
    fig_releases(plt, res["E1"])
    fig_coverage(plt, res["FULL"])
    fig_bias(plt, res["E1"])
    if "E4" in res:
        fig_e4(plt, res["E4"])
    if "E1s" in res:
        fig_stress(plt, res["E1s"])
    if "E1b" in res:
        fig_signal(plt, res["E1b"])
    csv_path = write_csv(res)
    print(tables(res))
    print("wrote", csv_path)


# =============================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", default="all", choices=["all", "e1", "full", "stress", "signal", "e4", "report"])
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    global WORK, OUT
    size = "quick" if args.quick else "full"
    sz = SIZES[size]
    if args.quick:
        WORK = os.path.join(WORK, "quick"); OUT = os.path.join(OUT, "quick")
    os.makedirs(WORK, exist_ok=True)
    stages = ["e1", "full", "stress", "signal", "e4", "report"] if args.stage == "all" else [args.stage]
    runners = dict(e1=stage_e1, full=stage_full, stress=stage_stress, signal=stage_signal, e4=stage_e4)
    for st in stages:
        if st in runners:
            log(f"stage {st} ({size})")
            save(dict(result=runners[st](sz), sizes=sz), os.path.join(WORK, f"{st}.json"))
        else:
            res = dict(config=dict(
                sizes=sz, n_grid=N_GRID, eps_grid=EPS_GRID, K_grid=K_GRID, delta=DELTA, rho_star=RHO_STAR,
                ell_main=ELL_MAIN, ell_frac=ELL_FRAC, gate=GATE, base_seed=BASE_SEED,
                population=dict(d_R=POP.d_R, d_O=POP.d_O, p=P, d_R_block=DR, r2=POP.r2, seed=POP.seed,
                                lam_min_pop=POP.lam_min_pop, B_R=POP.B_R, B_O=POP.B_O, B_y=POP.B_y,
                                beta_true=BETA_TRUE),
                sigma={str(e): SIGMA[e] for e in EPS_GRID},
                delta_rep=POP.sigma_for(1.0, DELTA)[1], rho_switch_pop=RHO_SWITCH,
                stress_ridge_cells=STRESS_RIDGE_CELLS, signal_cells=SIGNAL_CELLS, signal_k=SIGNAL_K,
                e4_cells=e4_cells()))
            for key, fname in (("E1", "e1"), ("FULL", "full"), ("E1s", "stress"), ("E1b", "signal"), ("E4", "e4")):
                pth = os.path.join(WORK, f"{fname}.json")
                if os.path.exists(pth):
                    res[key] = load(pth)["result"]
            stage_report(res)
            save(res, os.path.join(OUT, "results.json"))
            log("report written to " + OUT)


if __name__ == "__main__":
    main()
