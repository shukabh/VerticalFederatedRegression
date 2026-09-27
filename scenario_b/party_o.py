"""
party_o.py — Organization O (Scenario B: O holds y). Socket CLIENT.

Phase 0: evaluates its own plaintext membership/label polynomials at R's ENCRYPTED
         query points and returns masked ciphertexts. Learns only n = |I|.
Phase 1: computes every O-dependent sufficient-statistic block as a depth-1 encrypted
         inner product, injects DP noise INSIDE the ciphertext, returns the blocks.

O never decrypts anything and never sees A, beta, or the match set.

    python party_o.py --data_dir vfl_B --backend openfhe
"""
from __future__ import annotations

import argparse, json, os, pickle, socket, time
import numpy as np
import pandas as pd

import psi_common as pc
import phase1_common as p1
import he_backend as hb
from calibrate_hyperparameters import clip_rows, clip_scalar, apply_standardization


def power_from_windows(d, windows, be):
    """Reconstruct Enc(y^d) as a balanced product of window ciphertexts (ct x ct)."""
    facs = [windows[i] for i in range(d.bit_length()) if (d >> i) & 1]
    while len(facs) > 1:
        nxt = [be.mul_ct(facs[j], facs[j + 1]) for j in range(0, len(facs) - 1, 2)]
        if len(facs) % 2:
            nxt.append(facs[-1])
        facs = nxt
    return facs[0]


def run(data_dir, backend, host, port):
    P = json.load(open(os.path.join(data_dir, "dp_params.json")))
    dp, p0c, p1c = P["dp"], P["phase0_bfv"], P["phase1_ckks"]
    t, NB, NH, DCAP = (p0c["plaintext_modulus"], p0c["num_bins"],
                       p0c["n_hash"], p0c["d_cap"])

    ids_O = pd.read_csv(os.path.join(data_dir, "X_O.csv"))["id"].astype(np.int64).to_numpy()
    X_O = pd.read_csv(os.path.join(data_dir, "X_O.csv")).to_numpy()[:, 1:].astype(float)
    y_O = pd.read_csv(os.path.join(data_dir, "y_O.csv"))["y"].to_numpy().astype(float)
    n_O = len(ids_O)

    # standardize first (fixed affine map), then enforce the COMMITTED bounds;
    # without the clipping the (eps,delta) claim is void
    std = P.get("standardization", {"enabled": False})
    if std.get("enabled"):
        X_O = apply_standardization(X_O, std["center_O"], std["scale_O"])
        y_O = (y_O - std["center_y"]) / std["scale_y"]
        print(f"[O] standardized (mode={std['mode']})")
    X_O, rO = clip_rows(X_O, dp["B_O"])
    y_O, ry = clip_scalar(y_O, dp["B_y"])
    print(f"[O] n_O={n_O}  clip rates: X_O={rO:.3%}  y={ry:.3%}")

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((host, port))
    print(f"[O] connected to R at {host}:{port}")

    # =================== PHASE 0 : PSI (BFV) ============================
    t0 = time.time()
    bfv = hb.bfv_from_public(pc.recv_msg(s), backend=backend)
    meta_r = pickle.loads(pc.recv_msg(s))
    n_batches, W = meta_r["n_batches"], meta_r["n_window"]

    # preprocessing over O's OWN plaintext data (amortizable, no crypto)
    P_layers, L_layers, D, alpha, coll = pc.bin_and_interpolate(ids_O, t, NB, NH, DCAP)
    if D > (1 << (W - 1)):
        raise RuntimeError(f"[O] need degree {D} but windows cover {1 << (W-1)}")
    if coll:
        print(f"[O] WARNING {coll} within-bin field collisions dropped (widen field via CRT)")
    pc.send_msg(s, pickle.dumps({"alpha": alpha, "D": D}))
    print(f"[O] PSI preprocessing: D={D} alpha={alpha} ({time.time()-t0:.1f}s)")

    rng = np.random.default_rng()
    for bi in range(n_batches):
        windows = [bfv.ct_from_bytes(w) for w in pickle.loads(pc.recv_msg(s))]
        powers = {d: power_from_windows(d, windows, bfv) for d in range(1, D + 1)}
        for a in range(alpha):
            # coefficients stay PLAINTEXT (ct x pt) — saves a level vs encrypting them
            ctP = bfv.mul_pt(powers[1], P_layers[a][1])
            for d in range(2, D + 1):
                ctP = bfv.add(ctP, bfv.mul_pt(powers[d], P_layers[a][d]))
            ctP = bfv.add_pt(ctP, P_layers[a][0])
            ctL = bfv.mul_pt(powers[1], L_layers[a][1]) if D > 1 else None
            for d in range(2, D):
                ctL = bfv.add(ctL, bfv.mul_pt(powers[d], L_layers[a][d]))
            ctL = bfv.add_pt(ctL, L_layers[a][0]) if ctL is not None else \
                  bfv.mul_pt(powers[1], [0] * NB)
            r1 = rng.integers(1, t, NB).tolist()
            r2 = rng.integers(1, t, NB).tolist()
            ct_s = bfv.mul_pt(ctP, r1)                       # 0 iff match
            ct_q = bfv.add(bfv.mul_pt(ctP, r2), ctL)         # label iff match
            pc.send_msg(s, pickle.dumps((bfv.ct_bytes(ct_s), bfv.ct_bytes(ct_q))))
        print(f"[O] PSI batch {bi+1}/{n_batches} answered")
    print(f"[O] Phase 0 done ({time.time()-t0:.1f}s)")

    # =================== PHASE 1 : aggregation (CKKS) ===================
    t1 = time.time()
    ckks = hb.ckks_from_public(pc.recv_msg(s), backend=backend)
    m1 = pickle.loads(pc.recv_msg(s))
    d_R, d_O = m1["d_R"], m1["d_O"]
    ct_b = ckks.ct_from_bytes(pc.recv_msg(s))
    ct_xdot = [ckks.ct_from_bytes(w) for w in pickle.loads(pc.recv_msg(s))]
    sigma = dp["sigma"]

    # single-draw symmetric Gaussian: EVERY independent entry ~ N(0, sigma^2)
    nz = p1.draw_noise(d_R, d_O, sigma, np.random.default_rng())

    def inner(ct, w):
        return ckks.inner_product(ct, np.asarray(w, dtype=float))

    out = {"B": {}, "C": {}, "cO": [], "cR": [], "yty": None}
    for j in range(d_O):                                   # B: b-masked self-Gram
        for k in range(j, d_O):
            ct = ckks.add_scalar(inner(ct_b, X_O[:, j] * X_O[:, k]), nz.E_B[j, k])
            out["B"][(j, k)] = ckks.ct_bytes(ct)
    for j in range(d_R):                                   # C: Xdot_R already zero off-match
        for k in range(d_O):
            ct = ckks.add_scalar(inner(ct_xdot[j], X_O[:, k]), nz.E_C[j, k])
            out["C"][(j, k)] = ckks.ct_bytes(ct)
    for k in range(d_O):                                   # c_O: b-masked
        out["cO"].append(ckks.ct_bytes(
            ckks.add_scalar(inner(ct_b, X_O[:, k] * y_O), nz.f_O[k])))
    for j in range(d_R):                                   # c_R: the central cross-term
        out["cR"].append(ckks.ct_bytes(
            ckks.add_scalar(inner(ct_xdot[j], y_O), nz.f_R[j])))
    out["yty"] = ckks.ct_bytes(                            # for RSS / inference
        ckks.add_scalar(inner(ct_b, y_O * y_O), nz.e_yty))

    pc.send_msg(s, pickle.dumps(out))
    del nz                                                 # noise never disclosed
    n_ct = len(out["B"]) + len(out["C"]) + d_O + d_R + 1
    print(f"[O] Phase 1 done: {n_ct} noised ciphertexts sent ({time.time()-t1:.1f}s)")
    s.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Party O — PSI sender + CKKS aggregator")
    ap.add_argument("--data_dir", default="vfl_B")
    ap.add_argument("--backend", default="openfhe", choices=["openfhe", "plaintext"])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=65432)
    a = ap.parse_args()
    run(a.data_dir, a.backend, a.host, a.port)
