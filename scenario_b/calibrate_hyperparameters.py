"""
calibrate_hyperparameters.py — everything the protocol needs fixed BEFORE it runs.

Produces dp_params.json holding:
  * the committed clip bounds (B_R, B_O, B_y) and the DP budget (eps, delta)
  * the Scenario-B joint sensitivity Delta_2 and the noise scale sigma
  * Phase-0 (BFV) parameters: plaintext modulus, ring dimension, bins, degrees, batching
  * Phase-1 (CKKS) parameters: batch size, depth, scaling bits
  * a feasibility pre-check on the convergence condition rho > 1

WHY BOUNDS ARE INPUTS, NOT MEASUREMENTS
---------------------------------------
The (eps, delta) guarantee requires the sensitivity to be fixed independently of the
data. Reading B_y = max|y| off the realized sample makes the bound itself a function of
the data, which leaks and voids the guarantee. So (B_R, B_O, B_y) are COMMITTED here and
the data is CLIPPED to them; the clip rate is reported as a utility diagnostic, never
used to re-tune the bounds.

Usage
-----
    python calibrate_hyperparameters.py --data_dir vfl_B --eps 1.0 --delta 1e-5 \
        --B_R 3.0 --B_O 3.0 --B_y 20.0
    # omit the bounds to get a suggestion from quantiles (then commit them explicitly)
"""
from __future__ import annotations

import argparse, json, math, os
import numpy as np
import pandas as pd


# ============================================================================
# Clipping (enforces the committed bounds)
# ============================================================================

def apply_standardization(X, center, scale):
    """z = (x - center) / scale, applied column-wise. A FIXED affine map."""
    return (np.asarray(X, dtype=float) - np.asarray(center, dtype=float)) \
           / np.asarray(scale, dtype=float)


def standardization_block(X_R, X_O, y_O, mode, source,
                          center_R=None, scale_R=None, center_O=None, scale_O=None,
                          center_y=None, scale_y=None):
    """Build the standardization spec.

    mode   : "none" | "scale" | "center_scale"
    source : "committed"    centers/scales supplied by the caller (public constants,
                            agency reference metadata, a prior cycle) -> DP-VALID.
             "data_suggest" computed from THIS sample -> NOT DP-valid; exploration only.

    A third route, "dp", estimates the moments under their own budget eps_std and
    composes: eps_total = eps_std + eps_gram. It needs crude public ranges per variable
    and a separate mechanism, so it is documented rather than silently half-implemented.

    Centering forces an intercept. In aligned coordinates R's intercept column is
    exactly the selection vector b (1 on matched rows, 0 elsewhere), so R adds it
    locally inside its own clean block A and O never learns anything new.
    """
    d_R, d_O = X_R.shape[1], X_O.shape[1]
    if mode == "none":
        return {"enabled": False, "add_intercept": False}

    def _pick(given, computed):
        if given is not None:
            return [float(v) for v in given]
        return [float(v) for v in np.atleast_1d(computed)]

    if mode == "scale":
        cR = [0.0] * d_R; cO = [0.0] * d_O; cy = 0.0
    else:
        cR = _pick(center_R, X_R.mean(axis=0))
        cO = _pick(center_O, X_O.mean(axis=0))
        cy = float(center_y if center_y is not None else y_O.mean())
    sR = _pick(scale_R, X_R.std(axis=0, ddof=1))
    sO = _pick(scale_O, X_O.std(axis=0, ddof=1))
    sy = float(scale_y if scale_y is not None else y_O.std(ddof=1))
    eps_ = 1e-12
    sR = [v if abs(v) > eps_ else 1.0 for v in sR]
    sO = [v if abs(v) > eps_ else 1.0 for v in sO]
    sy = sy if abs(sy) > eps_ else 1.0
    return {"enabled": True, "mode": mode, "source": source,
            "dp_valid": (source == "committed"),
            "add_intercept": (mode == "center_scale"),
            "center_R": cR, "scale_R": sR,
            "center_O": cO, "scale_O": sO,
            "center_y": cy, "scale_y": sy}


def clip_rows(X, bound):
    """Scale any row with ||x||_2 > bound down to exactly bound. Returns (Xc, rate)."""
    X = np.asarray(X, dtype=float)
    nrm = np.linalg.norm(X, axis=1)
    hit = nrm > bound
    Xc = X.copy()
    if hit.any():
        Xc[hit] = X[hit] * (bound / nrm[hit])[:, None]
    return Xc, float(hit.mean())


def clip_scalar(y, bound):
    """Clip |y| to bound. Returns (yc, rate)."""
    y = np.asarray(y, dtype=float)
    hit = np.abs(y) > bound
    return np.clip(y, -bound, bound), float(hit.mean())


# ============================================================================
# Sensitivity and the Gaussian mechanism
# ============================================================================

def joint_sensitivity_B(B_R, B_O, B_y):
    """Scenario-B joint L2 sensitivity of the stacked O-dependent release
    (B, C, c_O, c_R, y^T y) when one matched record changes:

        Delta_2^2 = (B_O^2 + B_R^2)(B_O^2 + B_y^2) + B_y^4.

    No B_R^4 term: the block A = X_R^{I T} X_R^I is R's own data, computed locally and
    never noised. Its absence is a structural check on the corrected protocol.
    """
    return float(math.sqrt((B_O**2 + B_R**2) * (B_O**2 + B_y**2) + B_y**4))


def _Phi(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _log_Phi(x):
    """log of the standard normal CDF, stable in the far-left tail.

    Needed because B(sigma) contains e^eps * Phi(-a-b): for large eps, e^eps
    overflows while Phi(-a-b) underflows, and only their product is finite. We
    therefore evaluate that product as exp(eps + log Phi(-a-b)).
    """
    if x > -20.0:
        v = _Phi(x)
        if v > 0.0:
            return math.log(v)
    z = -x                                        # asymptotic expansion, z large
    return (-0.5 * z * z - math.log(z) - 0.5 * math.log(2.0 * math.pi)
            + math.log1p(-1.0 / z**2 + 3.0 / z**4))


def analytic_gaussian_sigma(Delta, eps, delta, tol=1e-12):
    """Balle & Wang (ICML 2018) analytic Gaussian mechanism: the SMALLEST sigma with

        Phi(Delta/(2 sigma) - eps sigma/Delta) - e^eps Phi(-Delta/(2 sigma) - eps sigma/Delta) <= delta.

    Strictly tighter than the classical Delta sqrt(2 ln(1.25/delta))/eps bound, which is
    only valid for eps < 1 anyway. B(sigma) is decreasing, so we bracket then bisect.
    """
    if Delta <= 0:
        return 0.0

    def B(sigma):
        a = Delta / (2.0 * sigma)
        b = eps * sigma / Delta
        log_tail = eps + _log_Phi(-a - b)         # e^eps * Phi(-a-b), in log space
        return _Phi(a - b) - (math.exp(log_tail) if log_tail < 700.0 else math.inf)

    hi = max(Delta / eps, 1e-12)
    while B(hi) > delta:
        hi *= 2.0
        if hi > 1e12:
            raise RuntimeError("sigma search diverged; check eps/delta")
    lo = hi / 2.0
    while B(lo) <= delta:          # walk down until lo violates, so the root is bracketed
        hi = lo
        lo /= 2.0
        if lo < 1e-12:
            return hi
    while hi - lo > tol * max(1.0, hi):
        mid = 0.5 * (lo + hi)
        if B(mid) <= delta:
            hi = mid
        else:
            lo = mid
    return hi


def classical_gaussian_sigma(Delta, eps, delta):
    return float(Delta * math.sqrt(2.0 * math.log(1.25 / delta)) / eps)


# ============================================================================
# Phase-0 (BFV) parameter selection
# ============================================================================

def _is_prime(n):
    if n < 2: return False
    for q in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % q == 0:
            return n == q
    d, r = n - 1, 0
    while d % 2 == 0:
        d //= 2; r += 1
    for a in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        x = pow(a, d, n)
        if x in (1, n - 1): continue
        for _ in range(r - 1):
            x = x * x % n
            if x == n - 1: break
        else:
            return False
    return True


def psi_plaintext_modulus(n_R, n_O, ring_dim, n_hash, delta_psi, max_bits=60):
    """Smallest prime t with t = 1 (mod 2*ring_dim) (BFV SIMD packing) such that the
    PSI false-positive probability stays below delta_psi.

    A false match needs two distinct identifiers to collide in field value inside a
    shared bin; the expected count is ~ n_R * n_O * n_hash / (ring_dim * t).
    """
    t_min = n_R * n_O * n_hash / (ring_dim * delta_psi)
    step = 2 * ring_dim
    k = max(1, int(math.ceil((t_min - 1) / step)))
    while True:
        t = k * step + 1
        if t.bit_length() > max_bits:
            raise RuntimeError(f"no prime <= {max_bits} bits meets delta_psi={delta_psi}; "
                               f"widen the field via CRT (two coprime primes)")
        if _is_prime(t):
            return t
        k += 1


# ============================================================================
# Main
# ============================================================================

def calibrate(data_dir, eps, delta, B_R=None, B_O=None, B_y=None,
              delta_psi=1e-6, n_hash=3, d_cap=64, ring_dim_bfv=16384,
              quantile=0.99, rho_target=2.0, sigma_prior=None,
              standardize="none", scale_source="data_suggest"):
    meta = pd.read_csv(os.path.join(data_dir, "metadata.csv")).iloc[0]
    if str(meta["scenario"]) != "B":
        raise ValueError("this calibration targets Scenario B (O holds y)")
    n_R, n_O = int(meta["n_R"]), int(meta["n_O"])
    d_R, d_O = int(meta["d_R"]), int(meta["d_O"])
    p = d_R + d_O

    X_R = pd.read_csv(os.path.join(data_dir, "X_R.csv")).to_numpy()[:, 1:]
    X_O = pd.read_csv(os.path.join(data_dir, "X_O.csv")).to_numpy()[:, 1:]
    y_O = pd.read_csv(os.path.join(data_dir, "y_O.csv"))["y"].to_numpy()

    # ---- standardization (applied BEFORE bounds/clipping) -------------------
    std = standardization_block(X_R, X_O, y_O, standardize, scale_source)
    if std["enabled"]:
        X_R = apply_standardization(X_R, std["center_R"], std["scale_R"])
        X_O = apply_standardization(X_O, std["center_O"], std["scale_O"])
        y_O = (y_O - std["center_y"]) / std["scale_y"]
        if std["add_intercept"]:
            d_R += 1; p += 1          # R's intercept column lives inside its clean A

    # ---- suggest bounds only if the user did not commit them ----------------
    suggested = {
        "B_R": float(np.quantile(np.linalg.norm(X_R, axis=1), quantile)),
        "B_O": float(np.quantile(np.linalg.norm(X_O, axis=1), quantile)),
        "B_y": float(np.quantile(np.abs(y_O), quantile)),
    }
    committed = (B_R is not None and B_O is not None and B_y is not None)
    B_R = suggested["B_R"] if B_R is None else float(B_R)
    B_O = suggested["B_O"] if B_O is None else float(B_O)
    B_y = suggested["B_y"] if B_y is None else float(B_y)
    B_R_rows = B_R                      # bound actually used to clip R's feature rows
    if std["enabled"] and std["add_intercept"]:
        # the appended intercept column contributes 1 to the squared row norm
        B_R = float(math.sqrt(1.0 + B_R_rows**2))

    # ---- clip rates (diagnostic only) ---------------------------------------
    _, rate_R = clip_rows(X_R, B_R_rows)
    _, rate_O = clip_rows(X_O, B_O)
    _, rate_y = clip_scalar(y_O, B_y)

    # ---- sensitivity and sigma ---------------------------------------------
    Delta2 = joint_sensitivity_B(B_R, B_O, B_y)
    sigma = analytic_gaussian_sigma(Delta2, eps, delta)
    sigma_cls = classical_gaussian_sigma(Delta2, eps, delta)

    # ---- Phase 0 (BFV) ------------------------------------------------------
    t = psi_plaintext_modulus(n_R, n_O, ring_dim_bfv, n_hash, delta_psi)
    cuckoo_cap = int(0.9 * ring_dim_bfv)
    n_batches = int(math.ceil(n_R / cuckoo_cap))
    n_window = int(math.ceil(math.log2(d_cap))) + 1

    # ---- Phase 1 (CKKS) -----------------------------------------------------
    batch = 1 << int(math.ceil(math.log2(max(n_O, 8))))
    n_chunks = 1 if n_O <= batch else int(math.ceil(n_O / batch))
    n_ct_out = d_O * (d_O + 1) // 2 + d_R * d_O + d_R + d_O + 1

    # ---- convergence feasibility -------------------------------------------
    noise_edge = 2.0 * sigma * math.sqrt(p)
    lam_cert = noise_edge * rho_target          # full-ridge, data-independent certificate
    need_n = None
    if sigma_prior:                             # optional prior on lambda_min(Sigma)
        need_n = noise_edge * rho_target / float(sigma_prior)

    params = {
        "scenario": "B",
        "standardization": std,
        "dims": {"n_R": n_R, "n_O": n_O, "d_R": d_R, "d_O": d_O, "p": p},
        "dp": {
            "eps": eps, "delta": delta,
            "B_R": B_R, "B_O": B_O, "B_y": B_y,
            "bounds_committed": committed,
            "bounds_suggested_from_quantile": suggested,
            "quantile_used_for_suggestion": quantile,
            "clip_rate": {"X_R": rate_R, "X_O": rate_O, "y": rate_y},
            "B_R_rows": B_R_rows,
            "Delta2": Delta2,
            "sigma": sigma,
            "sigma_classical": sigma_cls,
            "sigma_ratio_classical_over_analytic": sigma_cls / sigma if sigma > 0 else None,
            "mechanism": "single-draw symmetric Gaussian (each independent entry ~ N(0,sigma^2))",
        },
        "phase0_bfv": {
            "plaintext_modulus": t, "ring_dim": ring_dim_bfv, "num_bins": ring_dim_bfv,
            "n_hash": n_hash, "d_cap": d_cap, "n_window": n_window,
            "mult_depth": 4, "cuckoo_capacity": cuckoo_cap, "n_batches": n_batches,
            "delta_psi": delta_psi,
        },
        "phase1_ckks": {
            "batch_size": batch, "mult_depth": 2, "scale_bits": 50,
            "n_column_chunks": n_chunks, "ciphertexts_O_to_R": n_ct_out,
        },
        "convergence": {
            "rho_target": rho_target,
            "noise_edge_2sigma_sqrt_p": noise_edge,
            "lambda_full_ridge_certificate": lam_cert,
            "min_n_given_sigma_prior": need_n,
            "note": "rho = lambda_min(G_lambda)/(2 sigma sqrt p); O-block ridge is capped "
                    "at lambda_min(A)/(2 sigma sqrt p), checkable by R after PSI.",
        },
    }
    return params


def main():
    ap = argparse.ArgumentParser(description="Calibrate DP and HE hyperparameters.")
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--eps", type=float, default=1.0)
    ap.add_argument("--delta", type=float, default=1e-5)
    ap.add_argument("--B_R", type=float, default=None, help="committed ||x_R||_2 bound")
    ap.add_argument("--B_O", type=float, default=None, help="committed ||x_O||_2 bound")
    ap.add_argument("--B_y", type=float, default=None, help="committed |y| bound")
    ap.add_argument("--delta_psi", type=float, default=1e-6)
    ap.add_argument("--ring_dim_bfv", type=int, default=16384)
    ap.add_argument("--d_cap", type=int, default=64)
    ap.add_argument("--rho_target", type=float, default=2.0)
    ap.add_argument("--sigma_prior", type=float, default=None,
                    help="prior on lambda_min(Sigma) to get a minimum-n estimate")
    ap.add_argument("--standardize", default="none",
                    choices=["none", "scale", "center_scale"],
                    help="'scale' divides by committed scales; 'center_scale' also "
                         "centers and adds an intercept to R's clean block")
    ap.add_argument("--scale_source", default="data_suggest",
                    choices=["committed", "data_suggest"],
                    help="'data_suggest' derives scales from THIS sample and is NOT "
                         "DP-valid (exploration only); commit public constants instead")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    P = calibrate(a.data_dir, a.eps, a.delta, a.B_R, a.B_O, a.B_y,
                  delta_psi=a.delta_psi, ring_dim_bfv=a.ring_dim_bfv,
                  d_cap=a.d_cap, rho_target=a.rho_target, sigma_prior=a.sigma_prior,
                  standardize=a.standardize, scale_source=a.scale_source)
    out = a.out or os.path.join(a.data_dir, "dp_params.json")
    with open(out, "w") as fh:
        json.dump(P, fh, indent=2)

    dp, p0, p1, cv = P["dp"], P["phase0_bfv"], P["phase1_ckks"], P["convergence"]
    print(f"[calib] dims {P['dims']}")
    st = P["standardization"]
    if st["enabled"]:
        print(f"[calib] standardize: mode={st['mode']} source={st['source']} "
              f"intercept={st['add_intercept']}")
        if not st["dp_valid"]:
            print("[calib] WARNING scales derived from this sample: the (eps,delta) "
                  "guarantee is NOT valid. Commit public scales (--scale_source committed).")
    if not dp["bounds_committed"]:
        print(f"[calib] WARNING: bounds not committed; using 0.99 "
              f"quantiles of the data -> the (eps,delta) guarantee is NOT valid until you "
              f"commit them explicitly with --B_R/--B_O/--B_y.")
    print(f"[calib] bounds  B_R={dp['B_R']:.4g} B_O={dp['B_O']:.4g} B_y={dp['B_y']:.4g}"
          f"   clip rates X_R={dp['clip_rate']['X_R']:.3%} X_O={dp['clip_rate']['X_O']:.3%} "
          f"y={dp['clip_rate']['y']:.3%}")
    print(f"[calib] Delta_2={dp['Delta2']:.4g}  sigma={dp['sigma']:.4g} (analytic)  "
          f"vs {dp['sigma_classical']:.4g} (classical, {dp['sigma_ratio_classical_over_analytic']:.2f}x)")
    print(f"[calib] BFV  t={p0['plaintext_modulus']} ring={p0['ring_dim']} "
          f"batches={p0['n_batches']} W={p0['n_window']} D_cap={p0['d_cap']}")
    print(f"[calib] CKKS batch={p1['batch_size']} chunks={p1['n_column_chunks']} "
          f"ct(O->R)={p1['ciphertexts_O_to_R']}")
    print(f"[calib] convergence: 2*sigma*sqrt(p)={cv['noise_edge_2sigma_sqrt_p']:.4g}  "
          f"full-ridge certificate lambda>={cv['lambda_full_ridge_certificate']:.4g}")
    print(f"[calib] wrote {out}")


if __name__ == "__main__":
    main()
