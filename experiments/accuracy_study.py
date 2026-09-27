"""
accuracy_study.py -- coefficient accuracy of the Scenario-B private VFL-OLS protocol.

Compares, per synthetic configuration (cohort size n = n_intersect):

  (a) beta_true           generator coefficients
  (b) beta_GT             OLS on the TRUE matches (sigma.csv), raw data
  (c) beta_plain          plaintext linkage (join on id in the clear) + OLS with y_O
  (d) beta_clip           OLS on the clipped data the protocol consumes (its real target)
  (e) protocol, sigma~0   dp.sigma = 1e-9: isolates PSI + HE correctness
  (f) protocol, eps grid  beta_ridge and beta_bc; >= 1 full socket run per eps through
                          run_protocol.py, plus Monte-Carlo noise repetitions that reuse
                          the audited PSI output and call the Phase-4/6 logic directly
                          (phase1_common.draw_noise / assemble / solve_and_correct).

Everything the parties run is the user's unmodified code in scenario_b/ (plus the
stand-in modules psi_common / phase1_common / he_backend).

    python experiments/accuracy_study.py                         # full study (~15 min)
    python experiments/accuracy_study.py --quick                 # smoke test
    python experiments/accuracy_study.py --openfhe_python /path/to/py3.12/bin/python

Outputs go to reports/accuracy/ (report, CSV/JSON, PNGs, socket-run logs); generated
datasets and per-run artifacts go to experiments/_work/ (git-ignored).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCEN = os.path.join(ROOT, "scenario_b")


# ---------------------------------------------------------------------------
# Audited party R: the unmodified party_r.run() with phase1_common.local_A wrapped
# so that the PSI output (Xdot_R in O's row order) is dumped for verification.
# Invoked as a subprocess:  accuracy_study.py --audited-r DATA BACKEND PORT DUMP
# ---------------------------------------------------------------------------
if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "--audited-r":
    _data, _backend, _port, _dump = sys.argv[2:6]
    sys.path.insert(0, SCEN)
    import party_r  # noqa: E402
    import phase1_common as _p1  # noqa: E402
    _orig = _p1.local_A

    def _capture(Xdot):
        np.save(_dump, np.asarray(Xdot, dtype=float))
        return _orig(Xdot)

    _p1.local_A = _capture
    party_r.run(_data, _backend, "127.0.0.1", int(_port))
    sys.exit(0)


import pandas as pd  # noqa: E402

sys.path.insert(0, SCEN)
import phase1_common as p1  # noqa: E402
import prepare_data  # noqa: E402
from calibrate_hyperparameters import (clip_rows, clip_scalar,  # noqa: E402
                                       joint_sensitivity_B, analytic_gaussian_sigma)

EPS_GRID = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
SIGMA_ZERO = 1e-9
DELTA = 1e-5
D_R = D_O = 5
NOISE_STD = 0.1
SEED = 42
B_R = float(np.sqrt(D_R))          # U[0,1] rows have norm < sqrt(d): never clipped
B_O = float(np.sqrt(D_O))
B_Y = 5.0                          # committed a priori (see report)

# name, n_R, n_O, n_intersect, required-by-brief?
CONFIGS = [
    ("n200", 250, 2000, 200, True),
    ("n400", 500, 4000, 400, True),        # Problem-3 base configuration
    ("n1000", 1250, 4000, 1000, True),
    ("n4000", 5000, 8000, 4000, True),
    ("n16000", 20000, 32000, 16000, False),  # extension: where does DP noise stop dominating?
    ("n64000", 80000, 80000, 64000, False),
]

LOG_CHUNKS = []


# ---------------------------------------------------------------------------
# process helpers
# ---------------------------------------------------------------------------

def _log(title, text):
    LOG_CHUNKS.append(f"\n{'=' * 100}\n### {title}\n{'=' * 100}\n{text.rstrip()}\n")


def run_cmd(cmd, title, cwd=SCEN, timeout=3600):
    t0 = time.time()
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    out = (r.stdout or "") + (("\n[stderr]\n" + r.stderr) if r.stderr.strip() else "")
    _log(f"{title}   (rc={r.returncode}, {time.time() - t0:.1f}s)\n$ " +
         " ".join(os.path.relpath(c, ROOT) if os.path.isabs(c) and c.startswith(ROOT) else c
                  for c in cmd), out)
    if r.returncode != 0:
        raise RuntimeError(f"{title} failed (rc={r.returncode}):\n{out[-3000:]}")
    return out, time.time() - t0


def calibrate_cli(data_dir, eps, py=sys.executable):
    cmd = [py, "calibrate_hyperparameters.py", "--data_dir", data_dir, "--eps", str(eps),
           "--delta", str(DELTA), "--B_R", repr(B_R), "--B_O", repr(B_O), "--B_y", repr(B_Y),
           "--standardize", "none"]
    run_cmd(cmd, f"calibrate {os.path.basename(data_dir)} eps={eps}")
    return json.load(open(os.path.join(data_dir, "dp_params.json")))


def set_sigma(data_dir, sigma):
    path = os.path.join(data_dir, "dp_params.json")
    P = json.load(open(path))
    P["dp"]["sigma"] = float(sigma)
    P["dp"]["note_sigma_override"] = "sigma overridden by accuracy_study.py (PSI/HE check)"
    json.dump(P, open(path, "w"), indent=2)
    return P


def collect_run(data_dir, dest):
    os.makedirs(dest, exist_ok=True)
    for f in ("beta_private.csv", "run_summary.json", "dp_params.json"):
        shutil.copy(os.path.join(data_dir, f), os.path.join(dest, f))
    bp = pd.read_csv(os.path.join(data_dir, "beta_private.csv"))
    summ = json.load(open(os.path.join(data_dir, "run_summary.json")))
    return {"beta_ridge": bp["beta_ridge"].to_numpy(),
            "beta_bc": bp["beta_bias_corrected"].to_numpy(), "summary": summ}


def socket_run(data_dir, backend, port, title, py=sys.executable):
    for f in ("beta_private.csv", "run_summary.json"):
        p = os.path.join(data_dir, f)
        if os.path.exists(p):
            os.remove(p)
    out, dt = run_cmd([py, "run_protocol.py", "--data_dir", data_dir, "--backend", backend,
                       "--port", str(port)], title)
    if "=== verification ===" not in out:
        raise RuntimeError(f"{title}: run_protocol did not reach verification")
    return out, dt


def audited_socket_run(data_dir, backend, port, dump, title, py=sys.executable):
    """party_r (wrapped to dump Xdot) + unmodified party_o.py over loopback."""
    for f in ("beta_private.csv", "run_summary.json"):
        p = os.path.join(data_dir, f)
        if os.path.exists(p):
            os.remove(p)
    t0 = time.time()
    r = subprocess.Popen([py, os.path.abspath(__file__), "--audited-r", data_dir, backend,
                          str(port), dump], cwd=SCEN, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True)
    time.sleep(3.0)
    o = subprocess.run([py, "party_o.py", "--data_dir", data_dir, "--backend", backend,
                        "--port", str(port)], cwd=SCEN, capture_output=True, text=True)
    r_out, _ = r.communicate(timeout=3600)
    txt = ("--- party R (audited) ---\n" + r_out + "\n--- party O ---\n" + o.stdout +
           (("\n[O stderr]\n" + o.stderr) if o.stderr.strip() else ""))
    _log(f"{title}   (R rc={r.returncode}, O rc={o.returncode}, {time.time() - t0:.1f}s)", txt)
    if r.returncode or o.returncode:
        raise RuntimeError(f"{title} failed:\n{txt[-3000:]}")
    return txt, time.time() - t0


# ---------------------------------------------------------------------------
# data + references
# ---------------------------------------------------------------------------

def load_dataset(d):
    XR = pd.read_csv(os.path.join(d, "X_R.csv"))
    XO = pd.read_csv(os.path.join(d, "X_O.csv"))
    D = {
        "ids_R": XR["id"].astype(np.int64).to_numpy(),
        "X_R": XR.to_numpy()[:, 1:].astype(float),
        "ids_O": XO["id"].astype(np.int64).to_numpy(),
        "X_O": XO.to_numpy()[:, 1:].astype(float),
        "y_O": pd.read_csv(os.path.join(d, "y_O.csv"))["y"].to_numpy().astype(float),
        "y": pd.read_csv(os.path.join(d, "y.csv"))["y"].to_numpy().astype(float),
        "sig": pd.read_csv(os.path.join(d, "sigma.csv")),
        "beta_true": pd.read_csv(os.path.join(d, "beta_true.csv"))["beta"].to_numpy(),
    }
    return D


def ols(Z, y):
    return np.linalg.solve(Z.T @ Z, Z.T @ y)


def references(D, P):
    r_idx = D["sig"]["researcher_idx"].to_numpy()
    o_idx = D["sig"]["org_idx"].to_numpy()
    # (b) ground truth: true matches, raw data (same oracle as run_protocol.py)
    beta_gt = ols(np.hstack([D["X_R"][r_idx], D["X_O"][o_idx]]), D["y"])
    # (c) plaintext linkage: join on id in the clear
    L = pd.DataFrame({"id": D["ids_R"], "r": np.arange(len(D["ids_R"]))}).merge(
        pd.DataFrame({"id": D["ids_O"], "o": np.arange(len(D["ids_O"]))}), on="id")
    found = set(zip(L["r"].tolist(), L["o"].tolist()))
    truth = set(zip(r_idx.tolist(), o_idx.tolist()))
    tp = len(found & truth)
    beta_plain = ols(np.hstack([D["X_R"][L["r"]], D["X_O"][L["o"]]]), D["y_O"][L["o"]])
    # (d) clipped target: exactly what the parties feed the protocol
    dp = P["dp"]
    XRc, rate_R = clip_rows(D["X_R"], dp.get("B_R_rows", dp["B_R"]))
    XOc, rate_O = clip_rows(D["X_O"], dp["B_O"])
    yc, rate_y = clip_scalar(D["y_O"], dp["B_y"])
    beta_clip = ols(np.hstack([XRc[r_idx], XOc[o_idx]]), yc[o_idx])
    return {
        "beta_true": D["beta_true"], "beta_gt": beta_gt, "beta_plain": beta_plain,
        "beta_clip": beta_clip,
        "linkage": {"found": len(found), "true": len(truth), "tp": tp,
                    "precision": tp / max(len(found), 1), "recall": tp / max(len(truth), 1)},
        "clip_rate": {"X_R": rate_R, "X_O": rate_O, "y_all_O_rows": rate_y,
                      "y_matched": float(np.mean(np.abs(D["y_O"][o_idx]) > dp["B_y"]))},
        "max_abs_y_matched": float(np.abs(D["y_O"][o_idx]).max()),
        "XRc": XRc, "XOc": XOc, "yc": yc, "r_idx": r_idx, "o_idx": o_idx,
    }


def psi_audit(Xdot_psi, ref, n_O):
    Xdot_true = np.zeros((n_O, ref["XRc"].shape[1]))
    Xdot_true[ref["o_idx"]] = ref["XRc"][ref["r_idx"]]
    b_psi = np.any(Xdot_psi != 0, axis=1)          # R rows are U[0,1]: never all-zero
    b_true = np.zeros(n_O, dtype=bool); b_true[ref["o_idx"]] = True
    return {
        "n_matched_psi": int(b_psi.sum()), "n_true": int(b_true.sum()),
        "matched_set_equal": bool(np.array_equal(b_psi, b_true)),
        "false_matches": int((b_psi & ~b_true).sum()),
        "missed_matches": int((~b_psi & b_true).sum()),
        "rows_misaligned": int(np.any(Xdot_psi != Xdot_true, axis=1).sum()),
        "alignment_exact": bool(np.array_equal(Xdot_psi, Xdot_true)),
    }, b_psi.astype(float)


def phase_stats(Xdot, b, XOc, yc):
    """The noise-free Phase-1 quantities (what O's encrypted inner products compute)."""
    return {"A": p1.local_A(Xdot), "B": XOc.T @ (b[:, None] * XOc), "C": Xdot.T @ XOc,
            "c_R": Xdot.T @ yc, "c_O": XOc.T @ (b * yc), "yty": float(b @ (yc * yc))}


def monte_carlo(st, sigma, reps, rho_target, rng):
    d_R, d_O = st["C"].shape
    out = {k: [] for k in ("ridge", "bc", "lam", "psi", "rho", "rho0")}
    for _ in range(reps):
        nz = p1.draw_noise(d_R, d_O, sigma, rng)       # the noise party_o.py draws
        Gt, ct = p1.assemble(st["A"], st["B"] + nz.E_B, st["C"] + nz.E_C,
                             st["c_R"] + nz.f_R, st["c_O"] + nz.f_O)
        s = p1.solve_and_correct(Gt, ct, d_R, d_O, sigma, mode="auto", rho_target=rho_target)
        out["ridge"].append(s.beta_ridge); out["bc"].append(s.beta_bc)
        out["lam"].append(s.lam); out["psi"].append(s.Psi_mode)
        out["rho"].append(s.rho); out["rho0"].append(s.rho0)
    for k in ("ridge", "bc", "lam", "rho", "rho0"):
        out[k] = np.asarray(out[k])
    out["psi"] = np.asarray(out["psi"])
    return out


def mask_matrix(p, d_R):
    chi = np.ones((p, p)); chi[:d_R, :d_R] = 0.0
    return chi


def first_order_dp_cov(G, beta, sigma, d_R):
    """Cov of the first-order DP error P f - P E beta (lambda = 0), via Lemma 1."""
    p = G.shape[0]
    P = np.linalg.inv(G)
    chi = mask_matrix(p, d_R)
    PiO = np.diag([0.0] * d_R + [1.0] * (p - d_R))
    b2 = beta ** 2
    S = np.eye(p) + chi * np.outer(beta, beta) + np.diag(chi @ b2) - PiO @ np.diag(b2)
    return sigma ** 2 * P @ S @ P


def err_metrics(est, ref, d_R):
    e = est - ref[None, :]
    nrm = np.linalg.norm(e, axis=1)
    return {
        "rmse": float(np.sqrt(np.mean(e ** 2))),
        "rmse_R": float(np.sqrt(np.mean(e[:, :d_R] ** 2))),
        "rmse_O": float(np.sqrt(np.mean(e[:, d_R:] ** 2))),
        "maxabs_mean": float(np.mean(np.max(np.abs(e), axis=1))),
        "maxabs_median": float(np.median(np.max(np.abs(e), axis=1))),
        "l2_median": float(np.median(nrm)),
        "rel_mean": float(np.mean(nrm) / np.linalg.norm(ref)),
        "rel_median": float(np.median(nrm) / np.linalg.norm(ref)),
    }


def bias_metrics(est, target):
    R = est.shape[0]
    bias = est.mean(axis=0) - target
    se = est.std(axis=0, ddof=1) / np.sqrt(R)
    z = bias / np.where(se > 0, se, np.inf)
    return {"bias": bias.tolist(), "se": se.tolist(), "sd": est.std(axis=0, ddof=1).tolist(),
            "bias_l2": float(np.linalg.norm(bias)), "se_l2": float(np.linalg.norm(se)),
            "max_abs_z": float(np.max(np.abs(z))), "n_abs_z_gt3": int(np.sum(np.abs(z) > 3))}


# ---------------------------------------------------------------------------
# main study
# ---------------------------------------------------------------------------

def study(args):
    work = os.path.abspath(args.work_dir)
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(work, exist_ok=True); os.makedirs(out_dir, exist_ok=True)
    configs = [c for c in CONFIGS if (not args.configs or c[0] in args.configs)]
    eps_grid = args.eps
    port = args.port
    results = {"settings": {
        "eps_grid": eps_grid, "delta": DELTA, "d_R": D_R, "d_O": D_O, "noise_std": NOISE_STD,
        "seed": SEED, "B_R": B_R, "B_O": B_O, "B_y": B_Y, "reps": args.reps,
        "sigma_zero": SIGMA_ZERO, "rho_target": 2.0, "configs": [c[:4] for c in configs]},
        "configs": {}}

    for name, n_R, n_O, n, required in configs:
        print(f"\n######## config {name}: n_R={n_R} n_O={n_O} n={n} ########", flush=True)
        dd = os.path.join(work, "data", name)
        prepare_data.prepare(n_R, D_R, n_O, D_O, n, NOISE_STD, SEED, dd, verbose=False)
        D = load_dataset(dd)
        C = {"n_R": n_R, "n_O": n_O, "n": n, "required": required, "socket": {}, "eps": {}}

        # ---- (e) sigma ~ 0 : audited socket run (plaintext) --------------------
        P = calibrate_cli(dd, 1.0)
        ref = references(D, P)
        C["ref"] = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                    for k, v in ref.items() if k in ("beta_true", "beta_gt", "beta_plain",
                                                      "beta_clip", "linkage", "clip_rate",
                                                      "max_abs_y_matched")}
        set_sigma(dd, SIGMA_ZERO)
        port += 1
        dump = os.path.join(work, "psi", f"{name}_plaintext_Xdot.npy")
        os.makedirs(os.path.dirname(dump), exist_ok=True)
        _, dt = audited_socket_run(dd, "plaintext", port, dump,
                                   f"{name}: sigma={SIGMA_ZERO} audited socket run (plaintext)")
        run0 = collect_run(dd, os.path.join(work, "runs", name, "sigma0_plaintext"))
        audit, b_psi = psi_audit(np.load(dump), ref, n_O)
        C["psi_audit_plaintext"] = audit
        C["sigma0_plaintext"] = {
            "time_s": dt, "n_matched": run0["summary"]["n_matched"],
            "Psi": run0["summary"]["Psi"], "lambda": run0["summary"]["lambda"],
            "max_abs_ridge_minus_clip": float(np.max(np.abs(run0["beta_ridge"] - ref["beta_clip"]))),
            "max_abs_bc_minus_clip": float(np.max(np.abs(run0["beta_bc"] - ref["beta_clip"]))),
            "beta_bc": run0["beta_bc"].tolist()}
        print(f"  sigma~0 plaintext: PSI {audit}  max|bc-clip|="
              f"{C['sigma0_plaintext']['max_abs_bc_minus_clip']:.2e}", flush=True)
        if name == "n400":   # Problem-3 log: sigma~0 through run_protocol.py as well
            port += 1
            socket_run(dd, "plaintext", port, f"{name}: sigma={SIGMA_ZERO} run_protocol.py (plaintext)")

        # ---- (e) sigma ~ 0 with real HE (OpenFHE), base config only -----------
        if args.openfhe_python and name == args.openfhe_config:
            port += 1
            dump_o = os.path.join(work, "psi", f"{name}_openfhe_Xdot.npy")
            _, dto = audited_socket_run(dd, "openfhe", port, dump_o,
                                        f"{name}: sigma={SIGMA_ZERO} audited socket run (OPENFHE)",
                                        py=args.openfhe_python)
            runo = collect_run(dd, os.path.join(work, "runs", name, "sigma0_openfhe"))
            audit_o, _ = psi_audit(np.load(dump_o), ref, n_O)
            C["psi_audit_openfhe"] = audit_o
            C["sigma0_openfhe"] = {
                "time_s": dto, "n_matched": runo["summary"]["n_matched"],
                "max_abs_ridge_minus_clip": float(np.max(np.abs(runo["beta_ridge"] - ref["beta_clip"]))),
                "max_abs_bc_minus_clip": float(np.max(np.abs(runo["beta_bc"] - ref["beta_clip"]))),
                "max_abs_bc_minus_plaintext_sigma0": float(np.max(np.abs(runo["beta_bc"] - run0["beta_bc"])))}
            print(f"  sigma~0 OPENFHE: PSI {audit_o}  max|bc-clip|="
                  f"{C['sigma0_openfhe']['max_abs_bc_minus_clip']:.2e}", flush=True)

        # ---- reused PSI output -> noise-free Phase-1 statistics ----------------
        Xdot_psi = np.load(dump)
        st = phase_stats(Xdot_psi, b_psi, ref["XOc"], ref["yc"])
        G, c = p1.assemble(st["A"], st["B"], st["C"], st["c_R"], st["c_O"])
        beta_hat_psi = np.linalg.solve(G, c)
        C["noise_free_from_psi_minus_clip"] = float(np.max(np.abs(beta_hat_psi - ref["beta_clip"])))
        C["lambda_min_G"] = float(np.linalg.eigvalsh(G)[0])
        C["lambda_min_A"] = float(np.linalg.eigvalsh(st["A"])[0])
        bt, bg, bcl = ref["beta_true"], ref["beta_gt"], ref["beta_clip"]
        C["sampling"] = {"l2_gt_minus_true": float(np.linalg.norm(bg - bt)),
                         "rmse_gt_vs_true": float(np.sqrt(np.mean((bg - bt) ** 2))),
                         "maxabs_gt_vs_true": float(np.max(np.abs(bg - bt))),
                         "rmse_theory": float(NOISE_STD * np.sqrt(np.trace(np.linalg.inv(G)) / G.shape[0])),
                         "l2_clip_minus_gt": float(np.linalg.norm(bcl - bg)),
                         "max_abs_plain_minus_gt": float(np.max(np.abs(ref["beta_plain"] - bg)))}

        # ---- (f) eps grid: one socket run per eps + Monte Carlo ---------------
        rng = np.random.default_rng([SEED, n, 17])
        for eps in eps_grid:
            P = calibrate_cli(dd, eps)
            sigma = P["dp"]["sigma"]
            rho_t = P["convergence"]["rho_target"]
            E = {"sigma": sigma, "Delta2": P["dp"]["Delta2"]}
            if not args.skip_socket:
                port += 1
                _, dt = socket_run(dd, "plaintext", port, f"{name}: eps={eps} run_protocol.py (plaintext)")
                sr = collect_run(dd, os.path.join(work, "runs", name, f"eps{eps}_plaintext"))
                E["socket"] = {"time_s": dt, "n_matched": sr["summary"]["n_matched"],
                               "Psi": sr["summary"]["Psi"], "lambda": sr["summary"]["lambda"],
                               "rho_hat": sr["summary"]["rho_hat"],
                               "beta_ridge": sr["beta_ridge"].tolist(),
                               "beta_bc": sr["beta_bc"].tolist(),
                               "rmse_ridge_vs_gt": float(np.sqrt(np.mean((sr["beta_ridge"] - bg) ** 2))),
                               "rmse_bc_vs_gt": float(np.sqrt(np.mean((sr["beta_bc"] - bg) ** 2)))}
            if args.openfhe_python and name == args.openfhe_config and eps == 1.0:
                port += 1
                _, dto = socket_run(dd, "openfhe", port, f"{name}: eps={eps} run_protocol.py (OPENFHE)",
                                    py=args.openfhe_python)
                so = collect_run(dd, os.path.join(work, "runs", name, f"eps{eps}_openfhe"))
                E["socket_openfhe"] = {"time_s": dto, "n_matched": so["summary"]["n_matched"],
                                       "Psi": so["summary"]["Psi"], "lambda": so["summary"]["lambda"],
                                       "beta_bc": so["beta_bc"].tolist(),
                                       "rmse_bc_vs_gt": float(np.sqrt(np.mean((so["beta_bc"] - bg) ** 2)))}
            t0 = time.time()
            mc = monte_carlo(st, sigma, args.reps, rho_t, rng)
            E["mc_time_s"] = time.time() - t0
            E["frac_ridge"] = float(np.mean(mc["lam"] > 0))
            E["frac_full_ridge"] = float(np.mean(mc["psi"] == "I"))
            E["frac_O_ridge"] = float(np.mean(mc["psi"] == "O"))
            E["lam_median"] = float(np.median(mc["lam"]))
            E["rho0_median"] = float(np.median(mc["rho0"]))
            E["rho0_clean"] = float(C["lambda_min_G"] / (2 * sigma * np.sqrt(D_R + D_O)))
            E["rho_ceiling_O_clean"] = float(C["lambda_min_A"] / (2 * sigma * np.sqrt(D_R + D_O)))
            for est in ("ridge", "bc"):
                X = mc[est]
                E[est] = {"vs_gt": err_metrics(X, bg, D_R), "vs_true": err_metrics(X, bt, D_R),
                          "vs_clip": err_metrics(X, bcl, D_R), "bias_vs_clip": bias_metrics(X, bcl),
                          "mean": X.mean(axis=0).tolist()}
                if "socket" in E:   # where the single socket draw falls in the MC distribution
                    e_mc = np.sqrt(np.mean((X - bg) ** 2, axis=1))
                    E["socket"][f"rmse_pct_in_mc_{est}"] = float(
                        np.mean(e_mc <= E["socket"][f"rmse_{est}_vs_gt"]) * 100)
            # Theorem-1 prediction of the lambda=0 bias and first-order DP spread
            Pm = np.linalg.inv(G)
            E["theory_bias_ols"] = (sigma ** 2 * p1.M(Pm, D_R) @ bcl).tolist()
            cov1 = first_order_dp_cov(G, bcl, sigma, D_R)
            E["theory_rmse_first_order"] = float(np.sqrt(np.trace(cov1) / G.shape[0]))
            C["eps"][str(eps)] = E
            print(f"  eps={eps:<5} sigma={sigma:8.3f} ridge={E['frac_ridge']:.2f} full={E['frac_full_ridge']:.2f} "
                  f"RMSE(bc vs GT)={E['bc']['vs_gt']['rmse']:.4f} (ridge {E['ridge']['vs_gt']['rmse']:.4f}) "
                  f"theory1={E['theory_rmse_first_order']:.4f} samp={C['sampling']['rmse_gt_vs_true']:.4f}",
                  flush=True)
        results["configs"][name] = C

    # ---- B_y sensitivity on one config (MC only) --------------------------------
    if args.by_config in results["configs"]:
        name = args.by_config
        cfg = [c for c in configs if c[0] == name][0]
        dd = os.path.join(work, "data", name)
        D = load_dataset(dd)
        Xdot_psi = np.load(os.path.join(work, "psi", f"{name}_plaintext_Xdot.npy"))
        b_psi = np.any(Xdot_psi != 0, axis=1).astype(float)
        r_idx = D["sig"]["researcher_idx"].to_numpy(); o_idx = D["sig"]["org_idx"].to_numpy()
        XOc, _ = clip_rows(D["X_O"], B_O)
        bg = np.asarray(results["configs"][name]["ref"]["beta_gt"])
        rows = []
        rng = np.random.default_rng([SEED, 99])
        for By in args.by_grid:
            yc, _ = clip_scalar(D["y_O"], By)
            st = phase_stats(Xdot_psi, b_psi, XOc, yc)
            XRc, _ = clip_rows(D["X_R"], B_R)
            bclip = ols(np.hstack([XRc[r_idx], XOc[o_idx]]), yc[o_idx])
            Delta = joint_sensitivity_B(B_R, B_O, By)
            for eps in eps_grid:
                sigma = analytic_gaussian_sigma(Delta, eps, DELTA)
                mc = monte_carlo(st, sigma, max(200, args.reps // 4), 2.0, rng)
                rows.append({"B_y": By, "eps": eps, "Delta2": Delta, "sigma": sigma,
                             "clip_rate_y_matched": float(np.mean(np.abs(D["y_O"][o_idx]) > By)),
                             "rmse_clip_vs_gt": float(np.sqrt(np.mean((bclip - bg) ** 2))),
                             "rmse_bc_vs_gt": err_metrics(mc["bc"], bg, D_R)["rmse"],
                             "rmse_ridge_vs_gt": err_metrics(mc["ridge"], bg, D_R)["rmse"],
                             "frac_ridge": float(np.mean(mc["lam"] > 0))})
        results["by_sensitivity"] = {"config": name, "rows": rows}

    with open(os.path.join(out_dir, "results.json"), "w") as fh:
        json.dump(results, fh, indent=1)
    with open(os.path.join(out_dir, "socket_runs.log"), "w") as fh:
        fh.write("Logs of every socket run and calibration performed by experiments/accuracy_study.py\n")
        fh.write("".join(LOG_CHUNKS))
    return results


# ---------------------------------------------------------------------------
# tables, figures, report
# ---------------------------------------------------------------------------

def summary_rows(R):
    rows = []
    for name, C in R["configs"].items():
        for eps, E in C["eps"].items():
            for est in ("ridge", "bc"):
                m = E[est]
                rows.append({
                    "config": name, "n": C["n"], "n_R": C["n_R"], "n_O": C["n_O"],
                    "eps": float(eps), "sigma": E["sigma"], "estimator": est,
                    "rmse_vs_gt": m["vs_gt"]["rmse"], "rmse_R_vs_gt": m["vs_gt"]["rmse_R"],
                    "rmse_O_vs_gt": m["vs_gt"]["rmse_O"], "maxabs_vs_gt": m["vs_gt"]["maxabs_mean"],
                    "rel_vs_gt": m["vs_gt"]["rel_mean"], "rmse_vs_true": m["vs_true"]["rmse"],
                    "maxabs_vs_true": m["vs_true"]["maxabs_mean"], "rel_vs_true": m["vs_true"]["rel_mean"],
                    "bias_l2_vs_clip": m["bias_vs_clip"]["bias_l2"],
                    "bias_se_l2": m["bias_vs_clip"]["se_l2"],
                    "bias_max_abs_z": m["bias_vs_clip"]["max_abs_z"],
                    "sampling_rmse_gt_vs_true": C["sampling"]["rmse_gt_vs_true"],
                    "ratio_dp_to_sampling": m["vs_gt"]["rmse"] / C["sampling"]["rmse_gt_vs_true"],
                    "frac_ridge": E["frac_ridge"], "frac_full_ridge": E["frac_full_ridge"],
                    "lam_median": E["lam_median"], "rho0_clean": E["rho0_clean"],
                    "theory_rmse_first_order": E["theory_rmse_first_order"],
                    "socket_rmse_vs_gt": (E.get("socket") or {}).get(f"rmse_{est}_vs_gt"),
                    "socket_Psi": (E.get("socket") or {}).get("Psi"),
                })
    return pd.DataFrame(rows)


PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
MARKERS = ["o", "s", "D", "^", "v", "P", "X", "*"]
INK, INK2, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"


def _style(ax):
    ax.set_facecolor(SURF)
    ax.grid(True, which="major", color=GRID, linewidth=0.8, linestyle="-")
    ax.grid(False, which="minor")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.xaxis.label.set_color(INK2); ax.yaxis.label.set_color(INK2)
    ax.set_axisbelow(True)


def make_figures(R, df, out_dir, rep_cfg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": SURF,
                         "savefig.facecolor": SURF})
    comma = FuncFormatter(lambda v, _: f"{v:,.0f}" if v >= 1 else f"{v:g}")
    figs = []
    names = list(R["configs"].keys())
    eps_grid = R["settings"]["eps_grid"]

    # Fig 1: DP error / sampling error vs eps, one line per cohort (bias-corrected)
    fig, ax = plt.subplots(figsize=(7.6, 4.6), dpi=150)
    _style(ax)
    for i, name in enumerate(names):
        d = df[(df.config == name) & (df.estimator == "bc")].sort_values("eps")
        ax.plot(d.eps, d.ratio_dp_to_sampling, color=PAL[i], lw=2, marker=MARKERS[i], ms=6.5,
                mec=SURF, mew=1.5, label=f"n = {R['configs'][name]['n']:,}", solid_capstyle="round")
    ax.axhline(1.0, color=INK2, lw=1)
    ax.annotate("DP error = sampling error", (eps_grid[0], 1.0), xytext=(0, 4),
                textcoords="offset points", fontsize=8, color=INK2)
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xticks(eps_grid); ax.set_xticklabels([f"{e:g}" for e in eps_grid])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlabel("privacy budget ε  (δ = 1e-5)")
    ax.set_ylabel("RMSE(β_bc − β_GT) / RMSE(β_GT − β_true)")
    ax.set_title("Private-estimate error relative to sampling error", color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0),
              labelcolor=INK2, title="matched n", title_fontsize=8)
    fig.tight_layout()
    p = os.path.join(out_dir, "fig1_error_ratio_vs_eps.png"); fig.savefig(p); plt.close(fig)
    figs.append(p)

    # Fig 2: RMSE vs n per eps: MC points + first-order theory extrapolation + sampling
    rep_name = rep_cfg[0]
    _, k = crossover(R, rep_name)
    ngrid = np.logspace(2, 9, 200)
    fig, ax = plt.subplots(figsize=(7.6, 5.0), dpi=150)
    _style(ax)
    for i, eps in enumerate(eps_grid):
        d = df[(df.eps == eps) & (df.estimator == "bc")].sort_values("n")
        sigma = float(d.sigma.iloc[0])
        n_gate = 4.0 * sigma * np.sqrt(D_R + D_O) / k["lambda_min_Sigma"]
        g = ngrid[ngrid >= n_gate]
        ax.plot(g, sigma * k["k_dp1"] / g, color=PAL[i], lw=1, alpha=0.6)
        ax.plot(d.n, d.rmse_vs_gt, color=PAL[i], lw=2, marker=MARKERS[i], ms=6.5, mec=SURF, mew=1.5,
                label=f"ε = {eps:g}")
    ax.plot(ngrid, k["k_s"] / np.sqrt(ngrid), color=INK, lw=1)
    ns = [R["configs"][c]["n"] for c in names]
    ax.plot(ns, [R["configs"][c]["sampling"]["rmse_gt_vs_true"] for c in names], color=INK, lw=0,
            marker="o", ms=5, label="sampling (β_GT − β_true)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(1e2, 1e9)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}" if v < 1e6 else f"{v:.0e}"))
    ax.set_xlabel("matched cohort size n")
    ax.set_ylabel("coefficient RMSE vs β_GT")
    ax.set_title("Bias-corrected private estimate: error vs cohort size", color=INK, fontsize=11, loc="left")
    ax.text(0.01, 0.02, "markers: Monte Carlo (≥ 1 socket run each).  Thin lines: first-order theory with\n"
            f"G = nΣ and β of {rep_name}, drawn from the n where the ρ gate picks λ = 0; black line: σ_y·√(tr Σ⁻¹/(np))",
            transform=ax.transAxes, fontsize=7, color=MUTED, va="bottom")
    ax.legend(frameon=False, fontsize=8, loc="upper right", ncol=1, labelcolor=INK2)
    fig.tight_layout()
    p = os.path.join(out_dir, "fig2_rmse_vs_n.png"); fig.savefig(p); plt.close(fig)
    figs.append(p)

    # Fig 3: per-coefficient, representative config
    name, eps_list = rep_cfg
    C = R["configs"][name]
    p_ = len(C["ref"]["beta_gt"])
    labels = [f"R{j}" for j in range(D_R)] + [f"O{j}" for j in range(D_O)]
    fig, axes = plt.subplots(1, len(eps_list), figsize=(7.6, 4.4), dpi=150, sharey=True)
    axes = np.atleast_1d(axes)
    for ax, eps in zip(axes, eps_list):
        _style(ax)
        E = C["eps"][str(eps)]
        x = np.arange(p_)
        for kk, (est, col, off) in enumerate((("ridge", PAL[0], -0.17), ("bc", PAL[1], 0.17))):
            mean = np.asarray(E[est]["mean"]); sd = np.asarray(E[est]["bias_vs_clip"]["sd"])
            ax.errorbar(x + off, mean, yerr=sd, fmt=MARKERS[kk], color=col, ms=5.5, mec=SURF, mew=1.2,
                        elinewidth=1.5, capsize=0, label=f"β_{est}: MC mean ± 1 SD")
        ax.scatter(x, C["ref"]["beta_gt"], marker="_", s=200, color=INK, lw=2, label="β_GT", zorder=5)
        ax.scatter(x, C["ref"]["beta_true"], marker="x", s=28, color=INK2, lw=1.2, label="β_true", zorder=5)
        ax.axhline(0, color=AXIS, lw=1)
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(f"ε = {eps:g}   σ = {E['sigma']:.1f}, ridge in {E['frac_ridge']:.0%} of draws",
                     fontsize=9, color=INK, loc="left")
    axes[0].set_ylabel("coefficient value")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=8, labelcolor=INK2)
    fig.suptitle(f"Per-coefficient estimates, n = {C['n']:,}  ({R['settings']['reps']} noise draws per ε)",
                 color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    p = os.path.join(out_dir, "fig3_per_coefficient.png"); fig.savefig(p); plt.close(fig)
    figs.append(p)
    return figs


def per_coef_table(R, name, eps):
    C = R["configs"][name]; E = C["eps"][str(eps)]
    labels = [f"R{j}" for j in range(D_R)] + [f"O{j}" for j in range(D_O)]
    df = pd.DataFrame({
        "coef": labels, "beta_true": C["ref"]["beta_true"], "beta_GT": C["ref"]["beta_gt"],
        "beta_plain": C["ref"]["beta_plain"], "beta_clip": C["ref"]["beta_clip"],
        "sigma0_protocol_bc": C["sigma0_plaintext"]["beta_bc"],
        "socket_ridge": (E.get("socket") or {}).get("beta_ridge", [np.nan] * len(labels)),
        "socket_bc": (E.get("socket") or {}).get("beta_bc", [np.nan] * len(labels)),
        "mc_mean_ridge": E["ridge"]["mean"], "mc_bias_ridge": E["ridge"]["bias_vs_clip"]["bias"],
        "mc_se_ridge": E["ridge"]["bias_vs_clip"]["se"],
        "mc_mean_bc": E["bc"]["mean"], "mc_bias_bc": E["bc"]["bias_vs_clip"]["bias"],
        "mc_se_bc": E["bc"]["bias_vs_clip"]["se"], "mc_sd_bc": E["bc"]["bias_vs_clip"]["sd"],
        "theory_bias_ols": E["theory_bias_ols"],
    })
    return df


def crossover(R, ref_cfg):
    """First-order extrapolation (lambda = 0 regime) of the n where DP error = sampling error."""
    C = R["configs"][ref_cfg]
    beta = np.asarray(C["ref"]["beta_clip"])
    p = D_R + D_O
    Sig = np.eye(p) / 12.0 + np.ones((p, p)) / 4.0          # E[z z^T], z ~ U[0,1]^p
    P1 = np.linalg.inv(Sig)
    chi = mask_matrix(p, D_R)
    PiO = np.diag([0.0] * D_R + [1.0] * D_O)
    S = np.eye(p) + chi * np.outer(beta, beta) + np.diag(chi @ beta ** 2) - PiO @ np.diag(beta ** 2)
    k_dp1 = np.sqrt(np.trace(P1 @ S @ P1) / p)               # RMSE_dp = sigma * k_dp1 / n
    k_s = NOISE_STD * np.sqrt(np.trace(P1) / p)              # RMSE_s  = k_s / sqrt(n)
    lmin = np.linalg.eigvalsh(Sig)[0]
    rows = []
    for eps, E in C["eps"].items():
        sigma = E["sigma"]
        n_star = (sigma * k_dp1 / k_s) ** 2
        n_gate = 2.0 * 2.0 * sigma * np.sqrt(p) / lmin       # rho(0) >= 2 on the clean Gram
        n_10 = sigma * k_dp1 / (0.10 * np.linalg.norm(beta)) * np.sqrt(p)
        rows.append({"eps": float(eps), "sigma": sigma, "n_gate_lambda0": n_gate,
                     "n_rel10": max(n_10, n_gate), "n_star_dp_eq_sampling": max(n_star, n_gate),
                     "n_star_dp_eq_0.3_sampling": max(n_star / 0.09, n_gate)})
    return pd.DataFrame(rows), {"k_dp1": k_dp1, "k_s": k_s, "lambda_min_Sigma": lmin}


def fmt(x, nd=4):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "–"
    if isinstance(x, (int, np.integer)):
        return f"{x:,}"
    ax = abs(x)
    if ax != 0 and (ax < 1e-3 or ax >= 1e5):
        return f"{x:.2e}"
    return f"{x:.{nd}f}"


def md_table(df, cols, headers=None, nd=4):
    headers = headers or cols
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(fmt(r[c], nd) if not isinstance(r[c], str) else r[c]
                                     for c in cols) + " |")
    return "\n".join(out)


def write_outputs(R, out_dir, rep_cfg, rep_eps_fig):
    df = summary_rows(R)
    df.to_csv(os.path.join(out_dir, "results_summary.csv"), index=False)
    name, eps = rep_cfg
    pc = per_coef_table(R, name, eps)
    pc.to_csv(os.path.join(out_dir, f"per_coefficient_{name}_eps{eps:g}.csv"), index=False)
    figs = make_figures(R, df, out_dir, (name, rep_eps_fig))
    xo, xo_k = crossover(R, name)
    xo.to_csv(os.path.join(out_dir, "crossover_extrapolation.csv"), index=False)
    return df, pc, xo, xo_k, figs


def write_report(R, a):
    df, pc, xo, xo_k, figs = write_outputs(R, a.out_dir, (a.rep_config, a.rep_eps),
                                           sorted({1.0, a.rep_eps}))
    print(df[df.estimator == "bc"][["config", "eps", "sigma", "rmse_vs_gt", "ratio_dp_to_sampling",
                                    "frac_ridge", "frac_full_ridge"]].to_string())
    print(xo.to_string())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work_dir", default=os.path.join(ROOT, "experiments", "_work"))
    ap.add_argument("--out_dir", default=os.path.join(ROOT, "reports", "accuracy"))
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--eps", type=float, nargs="+", default=EPS_GRID)
    ap.add_argument("--configs", nargs="*", default=None)
    ap.add_argument("--skip_socket", action="store_true")
    ap.add_argument("--openfhe_python", default=None,
                    help="python interpreter with openfhe-python installed (e.g. a 3.12 venv)")
    ap.add_argument("--openfhe_config", default="n400")
    ap.add_argument("--by_config", default="n4000")
    ap.add_argument("--by_grid", type=float, nargs="+", default=[3.0, 5.0, 10.0])
    ap.add_argument("--rep_config", default="n4000")
    ap.add_argument("--rep_eps", type=float, default=8.0)
    ap.add_argument("--port", type=int, default=46000)
    ap.add_argument("--quick", action="store_true", help="2 small configs, 2 eps, 100 reps")
    ap.add_argument("--report_only", action="store_true", help="rebuild outputs from results.json")
    a = ap.parse_args()
    if a.quick:
        a.configs = a.configs or ["n200", "n400"]; a.eps = [1.0, 8.0]; a.reps = 100
        a.rep_config = "n400"; a.rep_eps = 8.0; a.by_config = "n400"
    if a.report_only:
        R = json.load(open(os.path.join(a.out_dir, "results.json")))
    else:
        R = study(a)
    write_report(R, a)


if __name__ == "__main__":
    main()
