"""
prepare_data.py -- generate a Scenario-B dataset AND the y_O.csv the parties expect.

generate_vfl_data.py (used unmodified) writes y.csv with the response for the
n_intersect MATCHED records only, but party_o.py and calibrate_hyperparameters.py
read y_O.csv: one response per O record (n_O rows, O's shuffled row order, column
"y"), because in Scenario B the organization holds y for its whole population.

This adapter writes, in addition to the generator's files:

    y_O.csv   (n_O x 2)   [id, y]   in X_O.csv row order

  * matched O rows:   y_O[sigma] = y       (exactly the generator's y.csv values)
  * unmatched O rows: y = x_R_latent @ beta_R + x_O @ beta_O + eps,
        x_R_latent ~ U[0,1]^{d_R} drawn fresh (these people are not in R's cohort,
        so their R-side covariates are unobserved "latent" values from the same
        distribution), eps ~ N(0, noise_std^2), same beta_R/beta_O/noise_std.

Choice/assumption: the unmatched responses come from the SAME model as the matched
ones, so y_O is one population. They never enter any statistic of the protocol
(every O-side aggregate is masked by b or by Xdot_R, which are zero off the match),
but they do matter for (i) data-derived suggestions in calibrate_hyperparameters.py
(quantile bounds, standardization scales) and (ii) the clip rate of y reported by
party_o.py, both of which look at all n_O rows. The extra draws use an independent
RNG stream (seed -> SeedSequence([seed, 7919])), so the generator's own stream --
and hence every file it writes -- is identical to calling it directly.

    python prepare_data.py --n_R 500 --d_R 5 --n_O 4000 --d_O 5 \
        --n_intersect 400 --noise_std 0.1 --seed 42 --out_dir vfl_B
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

import generate_vfl_data as gvd


def make_y_O(data: "gvd.VFLDataset", noise_std: float, seed) -> np.ndarray:
    ss = np.random.SeedSequence([int(seed), 7919]) if seed is not None \
        else np.random.SeedSequence()
    rng = np.random.default_rng(ss)
    y_O = np.empty(data.n_O)
    unmatched = np.ones(data.n_O, dtype=bool)
    unmatched[data.sigma] = False
    m = int(unmatched.sum())
    x_R_latent = rng.uniform(0.0, 1.0, size=(m, data.d_R))
    eps = rng.normal(0.0, noise_std, size=m)
    y_O[unmatched] = x_R_latent @ data.beta_R + data.X_O[unmatched] @ data.beta_O + eps
    y_O[data.sigma] = data.y
    return y_O


def prepare(n_R, d_R, n_O, d_O, n_intersect, noise_std=0.1, seed=42, out_dir="vfl_B",
            verbose=True):
    data = gvd.generate_vfl_data(n_R, d_R, n_O, d_O, n_intersect, "B", noise_std, seed)
    paths = gvd.save_vfl_data(data, out_dir, noise_std=noise_std, seed=seed)
    y_O = make_y_O(data, noise_std, seed)
    p = os.path.join(out_dir, "y_O.csv")
    pd.DataFrame({"id": data.ids_O, "y": y_O}).astype({"id": np.int64}).to_csv(p, index=False)
    paths["y_O.csv"] = p
    # sanity: y_O agrees with y.csv on the matched rows, in sigma order
    assert np.array_equal(y_O[data.sigma], data.y)
    if verbose:
        print(f"Saved {len(paths)} files to: {os.path.abspath(out_dir)}/")
        for name, path in paths.items():
            print(f"  {name:<22} {pd.read_csv(path).shape}")
        print(f"  |y| over matched rows: max {np.abs(data.y).max():.3f}; "
              f"over all O rows: max {np.abs(y_O).max():.3f}")
    return data, paths


def _args():
    ap = argparse.ArgumentParser(description="Scenario-B data + y_O.csv adapter")
    ap.add_argument("--n_R", type=int, default=500)
    ap.add_argument("--d_R", type=int, default=5)
    ap.add_argument("--n_O", type=int, default=4000)
    ap.add_argument("--d_O", type=int, default=5)
    ap.add_argument("--n_intersect", type=int, default=400)
    ap.add_argument("--noise_std", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", default="vfl_B")
    return ap.parse_args()


if __name__ == "__main__":
    a = _args()
    prepare(a.n_R, a.d_R, a.n_O, a.d_O, a.n_intersect, a.noise_std, a.seed, a.out_dir)
