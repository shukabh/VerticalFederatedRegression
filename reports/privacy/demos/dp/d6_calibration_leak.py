"""
D6 — dp_params.json is a non-private release of both parties' raw data.

calibrate_hyperparameters.calibrate() reads X_R, X_O and y_O in ONE process and writes, into
the single file both party_r.py and party_o.py load:
  * bounds_suggested_from_quantile  (0.99 quantiles of ||x_R||, ||x_O||, |y|)  -- ALWAYS
  * clip_rate                        (exact fraction of rows beyond each bound) -- ALWAYS
  * standardization centers/scales   (exact means/SDs) when standardize != none, and
    dp_valid=True if --scale_source committed even though no constants can be passed.
This script runs the real calibrate() on Definition-4 neighbours and shows what differs.

Run: TMPDIR=<scratch> python d6_calibration_leak.py     (~2 s)
"""
import json, os, sys, tempfile
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "../../../../scenario_b")))
import calibrate_hyperparameters as ch  # noqa: E402

rng = np.random.default_rng(5)
n_R, n_O, d_R, d_O, n = 300, 2000, 4, 3, 200
X_R = rng.standard_normal((n_R, d_R)); X_O = rng.standard_normal((n_O, d_O))
y_O = X_O @ np.array([1.0, -0.5, 0.3]) + rng.standard_normal(n_O)
ids_R = np.arange(n_R) + 10**12; ids_O = np.r_[ids_R[:n], np.arange(n_O - n) + 2 * 10**12]
matched_row = 7                                    # O-row 7 is a matched person (ids_O[:n] shared)


def write(d, X_R, X_O, y_O):
    os.makedirs(d, exist_ok=True)
    pd.DataFrame(np.column_stack([ids_R, X_R])).rename(columns={0: "id"}).to_csv(f"{d}/X_R.csv", index=False)
    pd.DataFrame(np.column_stack([ids_O, X_O])).rename(columns={0: "id"}).to_csv(f"{d}/X_O.csv", index=False)
    pd.DataFrame({"id": ids_O, "y": y_O}).to_csv(f"{d}/y_O.csv", index=False)
    pd.DataFrame([dict(scenario="B", n_R=n_R, n_O=len(y_O), d_R=d_R, d_O=d_O)]).to_csv(f"{d}/metadata.csv", index=False)


def diff(a, b, path=""):
    out = []
    if isinstance(a, dict):
        for k in a:
            out += diff(a[k], b.get(k), f"{path}.{k}" if path else k)
    elif isinstance(a, list):
        if not np.allclose(a, b, rtol=0, atol=0):
            out.append((path, a, b))
    elif isinstance(a, float) and a != b:
        out.append((path, a, b))
    return out


with tempfile.TemporaryDirectory() as tmp:
    y1 = y_O.copy(); y2 = y_O.copy(); y2[matched_row] = 9.0      # one matched person's outcome changes
    write(f"{tmp}/D", X_R, X_O, y1); write(f"{tmp}/Dp", X_R, X_O, y2)

    print("=== 1. bounds COMMITTED (B_R=B_O=B_y=3), standardize=none  -> the 'DP-valid' configuration")
    P = ch.calibrate(f"{tmp}/D", 1.0, 1e-5, 3.0, 3.0, 3.0)
    Q = ch.calibrate(f"{tmp}/Dp", 1.0, 1e-5, 3.0, 3.0, 3.0)
    for k, a, b in diff(P, Q):
        print(f"  differs between neighbours: {k:40s} {a!s:>22.22s} -> {b!s:>22.22s}")
    print(f"  bounds_committed={P['dp']['bounds_committed']}: R reads O's exact count of |y|>B_y: "
          f"{P['dp']['clip_rate']['y']*n_O:.0f} people (incl. UNMATCHED ones); O reads R's clip rate "
          f"{P['dp']['clip_rate']['X_R']:.3%} and 0.99-quantile of ||x_R|| {P['dp']['bounds_suggested_from_quantile']['B_R']:.4f}")

    print("\n=== 2. bounds NOT committed (documented 'suggest, then commit' workflow)")
    P = ch.calibrate(f"{tmp}/D", 1.0, 1e-5); Q = ch.calibrate(f"{tmp}/Dp", 1.0, 1e-5)
    for k in ("B_y", "Delta2", "sigma"):
        print(f"  {k:6s}: D={P['dp'][k]:.6f}  D'={Q['dp'][k]:.6f}   (public sigma itself is a function of the data)")

    print("\n=== 3. standardize=center_scale, --scale_source committed (no way to pass constants)")
    P = ch.calibrate(f"{tmp}/D", 1.0, 1e-5, 3.0, 3.0, 3.0, standardize="center_scale", scale_source="committed")
    Q = ch.calibrate(f"{tmp}/Dp", 1.0, 1e-5, 3.0, 3.0, 3.0, standardize="center_scale", scale_source="committed")
    s, t = P["standardization"], Q["standardization"]
    print(f"  dp_valid={s['dp_valid']} (warning suppressed)   center_y={s['center_y']:.12f}  mean(y_O)={y1.mean():.12f}")
    rec = n_O * (t["center_y"] - s["center_y"])
    print(f"  R recovers the neighbour's change exactly: n_O*(center_y'-center_y) = {rec:.10f};"
          f"  true y'-y = {y2[matched_row]-y1[matched_row]:.10f}")

    print("\n=== 4. next data cycle adds ONE person to O's register; calibration re-run")
    x_new, y_new = rng.standard_normal(d_O), 1.2345678
    ids_O = np.r_[ids_O, 3 * 10**12]
    write(f"{tmp}/D2", X_R, np.vstack([X_O, x_new]), np.r_[y1, y_new])
    P2 = ch.calibrate(f"{tmp}/D2", 1.0, 1e-5, 3.0, 3.0, 3.0, standardize="center_scale", scale_source="committed")
    s2 = P2["standardization"]
    y_rec = (n_O + 1) * s2["center_y"] - n_O * s["center_y"]
    x_rec = (n_O + 1) * np.array(s2["center_O"]) - n_O * np.array(s["center_O"])
    print(f"  y_new recovered = {y_rec:.6f} (true {y_new});  max |x_O,new error| = {np.max(np.abs(x_rec-x_new)):.1e}")
