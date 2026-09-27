"""
d5_implementation_leaks.py -- leaks outside the DP accounting, in the code as shipped.

 1. calibrate_hyperparameters.py --scale_source committed does NOT commit anything: it
    computes centres/scales from the sample and labels them dp_valid=True.
 2. dp_params.json (read by party_r.py) always contains O-data statistics: 0.99-quantiles
    of ||x_O|| and |y|, clip rates of X_O and y, and (when standardising) O's column means/SDs.
 3. PSI: O sends (D, alpha) = (max bin load, #partitions) of its simple-hashed ID set in the
    clear. This is a function of O's ID set, not only of n_O.
"""
import json, subprocess, sys, os
import numpy as np, pandas as pd
SB = "/home/user/VerticalFederatedRegression/scenario_b"
W = "/home/user/VerticalFederatedRegression/independent_review/_work/d5"
sys.path.insert(0, SB)
import psi_common as pc

if not os.path.exists(f"{W}/X_O.csv"):
    subprocess.run([sys.executable, "prepare_data.py", "--n_R", "500", "--d_R", "5", "--n_O", "4000",
                    "--d_O", "5", "--n_intersect", "400", "--seed", "42", "--out_dir", W],
                   cwd=SB, check=True, capture_output=True)
subprocess.run([sys.executable, "calibrate_hyperparameters.py", "--data_dir", W, "--eps", "1",
                "--delta", "1e-5", "--B_R", "3", "--B_O", "3", "--B_y", "3", "--standardize",
                "center_scale", "--scale_source", "committed", "--out", f"{W}/dp_params.json"],
               cwd=SB, check=True, capture_output=True)
P = json.load(open(f"{W}/dp_params.json")); s = P["standardization"]
XO = pd.read_csv(f"{W}/X_O.csv").to_numpy()[:, 1:]; y = pd.read_csv(f"{W}/y_O.csv")["y"].to_numpy()
print("1. --scale_source committed  ->  source =", s["source"], ", dp_valid =", s["dp_valid"])
print("   centre_O equals O's sample means:", np.allclose(s["center_O"], XO.mean(0)),
      "| scale_O equals O's sample SDs:", np.allclose(s["scale_O"], XO.std(0, ddof=1)))
print("   centre_y/scale_y equal mean/SD of O's y:", np.isclose(s["center_y"], y.mean()),
      np.isclose(s["scale_y"], y.std(ddof=1)))
print("2. O-data statistics present in the dp_params.json that R loads:")
print("   bounds_suggested_from_quantile =",
      {k: round(v, 3) for k, v in P["dp"]["bounds_suggested_from_quantile"].items()})
print("   clip_rate =", P["dp"]["clip_rate"])
print("   centre_O =", np.round(s["center_O"], 4).tolist())

# 3. (D, alpha) depends on O's ID set
NB, NH = 16384, 3


def max_load(ids):
    load = np.zeros(NB, dtype=int)
    for x in ids:
        for sl in {pc.bin_hash(i, int(x), NB) for i in range(NH)}:
            load[sl] += 1
    return load.max()


rng = np.random.default_rng(3)
base = rng.integers(10**12, 10**15, 4000)
loads = [max_load(np.concatenate([base[:-1], rng.integers(10**12, 10**15, 1)])) for _ in range(30)]
vals, cnt = np.unique(loads, return_counts=True)
print("3. max bin load D over 30 O-sets that differ from each other in ONE identifier:",
      dict(zip(vals.tolist(), cnt.tolist())))
