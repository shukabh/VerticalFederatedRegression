"""
party_r.py — Researcher R (Scenario B: O holds y). Socket SERVER. Run this FIRST.

Phase 0: cuckoo-hashes its identifiers, sends encrypted windowed powers, decrypts O's
         masked replies and recovers (I, sigma, b, Xdot_R); builds the CLEAN local
         block A, which is never transmitted and never noised.
Phase 1: encrypts only b and Xdot_R, receives O's DP-noised aggregates, splices in A,
         runs the rho gate (O-block ridge, escalating to full ridge), solves, and
         applies the zero-budget bias correction.

    python party_r.py --data_dir vfl_B --backend openfhe
"""
from __future__ import annotations

import argparse, json, os, pickle, socket, time
import numpy as np
import pandas as pd

import psi_common as pc
import phase1_common as p1
import he_backend as hb
from calibrate_hyperparameters import clip_rows, apply_standardization


def run(data_dir, backend, host, port):
    P = json.load(open(os.path.join(data_dir, "dp_params.json")))
    dp, p0c, p1c = P["dp"], P["phase0_bfv"], P["phase1_ckks"]
    t, NB, NH = p0c["plaintext_modulus"], p0c["num_bins"], p0c["n_hash"]
    W, CAP = p0c["n_window"], p0c["cuckoo_capacity"]

    df = pd.read_csv(os.path.join(data_dir, "X_R.csv"))
    ids_R = df["id"].astype(np.int64).tolist()
    X_R = df.to_numpy()[:, 1:].astype(float)
    n_R, d_R = X_R.shape
    d_O = P["dims"]["d_O"]
    n_O = P["dims"]["n_O"]

    std = P.get("standardization", {"enabled": False})
    if std.get("enabled"):
        X_R = apply_standardization(X_R, std["center_R"], std["scale_R"])
        print(f"[R] standardized (mode={std['mode']}, intercept={std['add_intercept']})")
    X_R, rR = clip_rows(X_R, dp.get("B_R_rows", dp["B_R"]))   # committed bound on rows
    d_R_raw = d_R
    print(f"[R] n_R={n_R} d_R={d_R}  clip rate X_R={rR:.3%}")

    id_to_k = {}
    for k, x in enumerate(ids_R):
        id_to_k.setdefault(int(x), k)

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, port)); srv.listen(1)
    print(f"[R] listening on {host}:{port} — start party_o.py ...")
    conn, addr = srv.accept()
    print(f"[R] O connected from {addr}")

    # =================== PHASE 0 : PSI (BFV) ============================
    t0 = time.time()
    bfv = hb.bfv_owner(plain_modulus=t, mult_depth=p0c["mult_depth"],
                       n_slots=NB, ring_dim=p0c["ring_dim"], backend=backend)
    slots = bfv.n_slots()
    if slots < NB:
        raise RuntimeError(f"[R] backend gives {slots} slots < num_bins {NB}")
    pc.send_msg(conn, bfv.public_blob())

    rng = np.random.default_rng(0)
    batches = pc.batch_ids(ids_R, CAP)
    tables = []
    for bi, chunk in enumerate(batches):
        tb = pc.build_cuckoo_table(chunk, rng, NB, NH)
        if tb is None:
            raise RuntimeError(f"[R] cuckoo insertion failed on batch {bi}")
        tables.append(tb)
    pc.send_msg(conn, pickle.dumps({"n_batches": len(batches), "n_window": W}))
    meta_o = pickle.loads(pc.recv_msg(conn))
    alpha, D = meta_o["alpha"], meta_o["D"]
    print(f"[R] cuckoo: {len(batches)} batch(es); O will send {alpha} partition(s), D={D}")

    sigma_map = {}
    for bi, tb in enumerate(tables):
        yv = [pc.DUMMY_Y] * NB
        for slot, x in tb.items():
            yv[slot] = pc.id_to_field(int(x), t)
        wins = [bfv.ct_bytes(bfv.encrypt([pow(v, 1 << i, t) if v else 0 for v in yv]))
                for i in range(W)]
        pc.send_msg(conn, pickle.dumps(wins))
        slot_to_k = {s_: id_to_k[int(x)] for s_, x in tb.items()}
        for _ in range(alpha):
            s_ser, q_ser = pickle.loads(pc.recv_msg(conn))
            sdec = bfv.decrypt(bfv.ct_from_bytes(s_ser), NB)
            qdec = bfv.decrypt(bfv.ct_from_bytes(q_ser), NB)
            for slot, k in slot_to_k.items():
                if sdec[slot] % t == 0 and k not in sigma_map:
                    sigma_map[k] = qdec[slot] % t         # O's canonical row index
        print(f"[R] PSI batch {bi+1}/{len(tables)}: total matched {len(sigma_map)}")

    b = np.zeros(n_O); Xdot = np.zeros((n_O, d_R))
    for k, j in sigma_map.items():
        if 0 <= j < n_O:
            b[j] = 1.0; Xdot[j, :] = X_R[k, :]
    if std.get("enabled") and std.get("add_intercept"):
        # in aligned coordinates R's intercept column IS b (1 on matched rows, 0 else),
        # so it lives inside R's own clean block A; O learns nothing new from it
        Xdot = np.hstack([b[:, None], Xdot]); d_R = d_R_raw + 1
    A = p1.local_A(Xdot)                                   # CLEAN, LOCAL, never sent
    print(f"[R] Phase 0 done: n={int(b.sum())} matched ({time.time()-t0:.1f}s)")

    # =================== PHASE 1 : aggregation (CKKS) ===================
    t1 = time.time()
    ckks = hb.ckks_owner(mult_depth=p1c["mult_depth"], scale_bits=p1c["scale_bits"],
                         batch_size=p1c["batch_size"], backend=backend)
    if ckks.n_slots() < n_O:
        raise RuntimeError(f"[R] CKKS slots {ckks.n_slots()} < n_O {n_O}; chunk columns")
    pc.send_msg(conn, ckks.public_blob())
    pc.send_msg(conn, pickle.dumps({"d_R": d_R, "d_O": d_O, "n_O": n_O}))
    pc.send_msg(conn, ckks.ct_bytes(ckks.encrypt(b)))      # Enc(b)
    pc.send_msg(conn, pickle.dumps(                        # Enc(Xdot_R), column-packed
        [ckks.ct_bytes(ckks.encrypt(Xdot[:, j])) for j in range(d_R)]))
    agg = pickle.loads(pc.recv_msg(conn))
    conn.close(); srv.close()

    val = lambda blob: ckks.decrypt_slot0(ckks.ct_from_bytes(blob))
    B = np.zeros((d_O, d_O))
    for (j, k), blob in agg["B"].items():
        B[j, k] = B[k, j] = val(blob)                      # symmetric noise mirrored
    C = np.zeros((d_R, d_O))
    for (j, k), blob in agg["C"].items():
        C[j, k] = val(blob)
    c_O = np.array([val(x) for x in agg["cO"]])
    c_R = np.array([val(x) for x in agg["cR"]])
    yty = val(agg["yty"])
    print(f"[R] Phase 1 aggregates decrypted ({time.time()-t1:.1f}s)")

    # =================== assemble, gate, solve, correct =================
    Gt, ct = p1.assemble(A, B, C, c_R, c_O)                # clean A spliced in
    sigma = dp["sigma"]
    sol = p1.solve_and_correct(Gt, ct, d_R, d_O, sigma, mode="auto",
                               rho_target=P["convergence"]["rho_target"])
    lam_A = float(np.linalg.eigvalsh(A)[0])
    ceil_O = lam_A / (2 * sigma * np.sqrt(d_R + d_O))
    print(f"[R] sigma={sigma:.4g}  O-block ridge ceiling rho<={ceil_O:.3g}  "
          f"-> Psi={sol.Psi_mode} lambda={sol.lam:.4g} rho_hat={sol.rho:.2f}"
          f"{'  (escalated to full ridge)' if sol.Psi_mode=='I' else ''}")
    if sol.rho < P["convergence"]["rho_target"]:
        print("[R] WARNING rho below target even after escalation — expansion may be invalid")

    out = pd.DataFrame({
        "party": ["R"] * d_R + ["O"] * d_O,
        "feature_idx": list(range(d_R)) + list(range(d_O)),
        "beta_ridge": sol.beta_ridge,
        "beta_bias_corrected": sol.beta_bc,
    })
    # ---- back-transform to the ORIGINAL scale (post-processing: free) ------
    if std.get("enabled"):
        bb = np.asarray(sol.beta_bc, dtype=float)
        icpt_std = float(bb[0]) if std["add_intercept"] else 0.0
        off = 1 if std["add_intercept"] else 0
        bR_s, bO_s = bb[off:d_R], bb[d_R:]
        sy, cy = std["scale_y"], std["center_y"]
        sR_, cR_ = np.array(std["scale_R"]), np.array(std["center_R"])
        sO_, cO_ = np.array(std["scale_O"]), np.array(std["center_O"])
        bR_o, bO_o = sy * bR_s / sR_, sy * bO_s / sO_
        icpt = cy + sy * icpt_std - float(bR_o @ cR_) - float(bO_o @ cO_)
        orig = pd.DataFrame({
            "party": ["intercept"] + ["R"] * len(bR_o) + ["O"] * len(bO_o),
            "feature_idx": [-1] + list(range(len(bR_o))) + list(range(len(bO_o))),
            "beta_original_scale": np.concatenate([[icpt], bR_o, bO_o]),
        })
        orig.to_csv(os.path.join(data_dir, "beta_original_scale.csv"), index=False)
        print(f"[R] back-transformed to original scale -> beta_original_scale.csv "
              f"(intercept {icpt:.4f})")
    out.to_csv(os.path.join(data_dir, "beta_private.csv"), index=False)
    json.dump({"n_matched": int(b.sum()), "sigma": sigma, "lambda": sol.lam,
               "Psi": sol.Psi_mode, "rho_hat": sol.rho, "yty": yty,
               "backend": backend},
              open(os.path.join(data_dir, "run_summary.json"), "w"), indent=2)
    print(f"[R] wrote beta_private.csv and run_summary.json  (y^T y = {yty:.4f})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Party R — PSI receiver + solver/corrector")
    ap.add_argument("--data_dir", default="vfl_B")
    ap.add_argument("--backend", default="openfhe", choices=["openfhe", "plaintext"])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=65432)
    a = ap.parse_args()
    run(a.data_dir, a.backend, a.host, a.port)
