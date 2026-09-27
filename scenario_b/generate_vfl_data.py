"""
generate_vfl_data.py
--------------------
Synthetic data generator for the privacy-preserving VFL protocol.

Each record is assigned a unique large integer ID (simulating a hashed SIN).
The intersection size n_intersect controls how many of R's IDs overlap with O's.
O's records are randomly shuffled so matches are not at predictable positions.

Usage (CLI)
-----------
    python generate_vfl_data.py \
        --n_R 500 --d_R 10 --n_O 5000 --d_O 15 \
        --n_intersect 400 --scenario B --noise_std 0.1 --seed 42

Usage (import)
--------------
    from generate_vfl_data import generate_and_save
    paths = generate_and_save(
        n_R=500, d_R=10, n_O=5000, d_O=15,
        n_intersect=400, scenario='B', seed=42,
    )

ID structure
------------
  n_intersect shared IDs   — appear in both X_R and X_O (the matched cohort)
  n_R - n_intersect R-only IDs — appear only in X_R (researcher records with no O match)
  n_O - n_intersect O-only IDs — appear only in X_O (O records with no R match)
  Total unique IDs: n_R + n_O - n_intersect

Output files
------------
    X_R.csv          (n_R x 1+d_R)      [id, R_feat_0, ...]
    X_O.csv          (n_O x 1+d_O)      [id, O_feat_0, ...]   shuffled row order
    y.csv            (n   x 2)          [id, y]               n = n_intersect
    sigma.csv        (n   x 3)          [researcher_idx, org_idx, id]
    X_R_aligned.csv  (n_O x 1+d_R)     [org_id, R_feat_0, ...]   dXR
    b.csv            (n_O x 2)          [org_id, b]
    y_aligned.csv    (n_O x 2)          [org_id, y_aligned]   Scenario A only
    beta_true.csv    (d_R+d_O x 3)      [party, feature_idx, beta]
    metadata.csv     (1   x *)          generation parameters

Assumptions
-----------
- Covariates drawn i.i.d. from Uniform[0, 1].
- n_intersect of R's records are randomly selected as the matched cohort.
- Response: y = X_R_I @ beta_R + X_O_I @ beta_O + eps, eps ~ N(0, noise_std^2),
  over the n_intersect matched records only.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Dict, Literal, Optional

import numpy as np
import pandas as pd


# ------------------------------------------------------------------ #
# Data container
# ------------------------------------------------------------------ #

@dataclass
class VFLDataset:
    # Full covariate matrices
    X_R:     np.ndarray   # (n_R, d_R)  — R's records in original order
    X_O:     np.ndarray   # (n_O, d_O)  — O's records in shuffled order

    # IDs
    ids_R:   np.ndarray   # (n_R,)  all researcher IDs
    ids_O:   np.ndarray   # (n_O,)  all organization IDs (shuffled order)

    # Matched submatrices (size n = n_intersect)
    X_R_I:      np.ndarray   # (n, d_R)
    X_O_I:      np.ndarray   # (n, d_O)
    ids_matched: np.ndarray  # (n,)  shared IDs of matched records

    # Protocol-ready aligned quantities (indexed to O's shuffled row order)
    X_R_dot: np.ndarray   # (n_O, d_R)  dXR: matched R rows at O positions, zeros elsewhere
    b:       np.ndarray   # (n_O,)      binary selection vector
    y_dot:   np.ndarray   # (n_O,)      dy: matched y values at O positions (Scenario A)

    # Response (over matched cohort only)
    y:       np.ndarray   # (n,)

    # Ground-truth matching
    I_R:     np.ndarray   # (n,)  indices into X_R of matched researcher records
    sigma:   np.ndarray   # (n,)  indices into X_O of matched records

    # Ground-truth coefficients
    beta_R:  np.ndarray   # (d_R,)
    beta_O:  np.ndarray   # (d_O,)

    scenario:    str
    n:           int   # = n_intersect
    n_R:         int
    n_O:         int
    d_R:         int
    d_O:         int

    def __repr__(self) -> str:
        return (
            f"VFLDataset(scenario={self.scenario!r}, "
            f"n_R={self.n_R}, d_R={self.d_R}, "
            f"n_O={self.n_O}, d_O={self.d_O}, "
            f"n_intersect={self.n})"
        )


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def _unique_ids(rng: np.random.Generator, count: int) -> np.ndarray:
    """
    Return `count` unique random integers in [10^12, 10^15),
    simulating truncated hashed identifiers.
    """
    lo, hi = 10**12, 10**15
    ids: set = set()
    while len(ids) < count:
        batch = rng.integers(lo, hi, size=(count - len(ids)) + 128)
        ids.update(batch.tolist())
    return np.array(sorted(ids)[:count], dtype=np.int64)


# ------------------------------------------------------------------ #
# Generator
# ------------------------------------------------------------------ #

def generate_vfl_data(
    n_R:          int,
    d_R:          int,
    n_O:          int,
    d_O:          int,
    n_intersect:  int,
    scenario:     Literal['A', 'B'],
    noise_std:    float = 0.1,
    seed:         Optional[int] = None,
) -> VFLDataset:
    """
    Generate a synthetic VFL dataset in memory.

    Parameters
    ----------
    n_R          : Number of researcher records.
    d_R          : Number of researcher features.
    n_O          : Number of organization records; must satisfy n_O >= n_intersect.
    d_O          : Number of organization features.
    n_intersect  : Number of matched records (intersection size n = |I|);
                   must satisfy 1 <= n_intersect <= n_R.
    scenario     : 'A' — y held by R;  'B' — y held by O.
    noise_std    : Std dev of additive Gaussian noise on y.
    seed         : Optional random seed.

    Returns
    -------
    VFLDataset
    """
    if scenario not in ('A', 'B'):
        raise ValueError(f"scenario must be 'A' or 'B', got {scenario!r}.")
    if not (1 <= n_intersect <= n_R):
        raise ValueError(
            f"n_intersect ({n_intersect}) must be in [1, n_R={n_R}]."
        )
    if n_O < n_intersect:
        raise ValueError(
            f"n_O ({n_O}) must be >= n_intersect ({n_intersect})."
        )

    rng = np.random.default_rng(seed)
    n   = n_intersect

    # ---- IDs -------------------------------------------------------- #
    # Total unique IDs needed: n shared + (n_R - n) R-only + (n_O - n) O-only
    #                        = n_R + n_O - n
    all_ids      = _unique_ids(rng, n_R + n_O - n)
    ids_shared   = all_ids[:n]                  # (n,)       in both R and O
    ids_R_only   = all_ids[n: n_R]              # (n_R - n,) only in R
    ids_O_only   = all_ids[n_R: n_R + n_O - n]  # (n_O - n,) only in O

    # ---- Covariates ------------------------------------------------- #
    X_R     = rng.uniform(0.0, 1.0, size=(n_R, d_R))
    X_O_pre = rng.uniform(0.0, 1.0, size=(n_O, d_O))

    # ---- Assign IDs to R ------------------------------------------- #
    # Randomly select n of R's records as matched; the rest are unmatched.
    I_R             = np.sort(rng.choice(n_R, size=n, replace=False))
    unmatched_R_idx = np.setdiff1d(np.arange(n_R), I_R)

    ids_R                   = np.empty(n_R, dtype=np.int64)
    ids_R[I_R]              = ids_shared
    ids_R[unmatched_R_idx]  = ids_R_only

    # ---- Assign IDs to O and shuffle -------------------------------- #
    # O's pre-shuffle order: matched records first, then unmatched.
    ids_O_pre = np.concatenate([ids_shared, ids_O_only])   # (n_O,)

    perm_O  = rng.permutation(n_O)
    X_O     = X_O_pre[perm_O]
    ids_O   = ids_O_pre[perm_O]

    # ---- Ground-truth matching -------------------------------------- #
    id_to_org_idx = {int(v): i for i, v in enumerate(ids_O)}
    sigma         = np.array([id_to_org_idx[int(ids_shared[k])]
                               for k in range(n)], dtype=np.int64)

    X_R_I        = X_R[I_R]
    X_O_I        = X_O[sigma]
    ids_matched  = ids_shared.copy()

    # ---- Aligned matrices ------------------------------------------- #
    b = np.zeros(n_O, dtype=np.float64)
    b[sigma] = 1.0

    X_R_dot = np.zeros((n_O, d_R), dtype=np.float64)
    X_R_dot[sigma] = X_R_I

    # ---- Response --------------------------------------------------- #
    beta_R = rng.standard_normal(d_R)
    beta_O = rng.standard_normal(d_O)
    eps    = rng.normal(0.0, noise_std, size=n)
    y      = X_R_I @ beta_R + X_O_I @ beta_O + eps

    y_dot = np.zeros(n_O, dtype=np.float64)
    if scenario == 'A':
        y_dot[sigma] = y

    return VFLDataset(
        X_R=X_R, X_O=X_O,
        ids_R=ids_R, ids_O=ids_O,
        X_R_I=X_R_I, X_O_I=X_O_I,
        ids_matched=ids_matched,
        X_R_dot=X_R_dot, b=b, y_dot=y_dot,
        y=y, I_R=I_R, sigma=sigma,
        beta_R=beta_R, beta_O=beta_O,
        scenario=scenario, n=n,
        n_R=n_R, n_O=n_O, d_R=d_R, d_O=d_O,
    )


# ------------------------------------------------------------------ #
# Save to CSV
# ------------------------------------------------------------------ #

def save_vfl_data(
    data:      VFLDataset,
    out_dir:   str,
    noise_std: float = 0.1,
    seed:      Optional[int] = None,
) -> Dict[str, str]:
    """
    Write all arrays in a VFLDataset to labelled CSV files.

    Returns a dict mapping filename -> full path for every file written.
    """
    os.makedirs(out_dir, exist_ok=True)
    paths: Dict[str, str] = {}
    d = data

    def _save(df: pd.DataFrame, name: str) -> None:
        path = os.path.join(out_dir, name)
        df.to_csv(path, index=False)
        paths[name] = path

    # X_R  (n_R x 1+d_R)
    _save(
        pd.DataFrame(
            np.column_stack([d.ids_R, d.X_R]),
            columns=["id"] + [f"R_feat_{j}" for j in range(d.d_R)],
        ).astype({"id": np.int64}),
        "X_R.csv",
    )

    # X_O  (n_O x 1+d_O)  — shuffled row order
    _save(
        pd.DataFrame(
            np.column_stack([d.ids_O, d.X_O]),
            columns=["id"] + [f"O_feat_{j}" for j in range(d.d_O)],
        ).astype({"id": np.int64}),
        "X_O.csv",
    )

    # y  (n x 2)
    _save(
        pd.DataFrame({"id": d.ids_matched, "y": d.y}).astype({"id": np.int64}),
        "y.csv",
    )

    # sigma  (n x 3)
    _save(
        pd.DataFrame({
            "researcher_idx": d.I_R,
            "org_idx":        d.sigma,
            "id":             d.ids_matched,
        }).astype({"id": np.int64}),
        "sigma.csv",
    )

    # X_R_aligned  (n_O x 1+d_R)
    _save(
        pd.DataFrame(
            np.column_stack([d.ids_O, d.X_R_dot]),
            columns=["org_id"] + [f"R_feat_{j}" for j in range(d.d_R)],
        ).astype({"org_id": np.int64}),
        "X_R_aligned.csv",
    )

    # b  (n_O x 2)
    _save(
        pd.DataFrame({"org_id": d.ids_O, "b": d.b}).astype({"org_id": np.int64}),
        "b.csv",
    )

    # y_aligned  (n_O x 2)  — Scenario A only
    if d.scenario == 'A':
        _save(
            pd.DataFrame({"org_id": d.ids_O, "y_aligned": d.y_dot})
            .astype({"org_id": np.int64}),
            "y_aligned.csv",
        )

    # beta_true  (d_R+d_O x 3)
    _save(
        pd.DataFrame({
            "party":       ["R"] * d.d_R + ["O"] * d.d_O,
            "feature_idx": list(range(d.d_R)) + list(range(d.d_O)),
            "beta":        np.concatenate([d.beta_R, d.beta_O]),
        }),
        "beta_true.csv",
    )

    # metadata  (1 x *)
    _save(
        pd.DataFrame([{
            "scenario":    d.scenario,
            "n_R":         d.n_R,
            "d_R":         d.d_R,
            "n_O":         d.n_O,
            "d_O":         d.d_O,
            "n_intersect": d.n,
            "noise_std":   noise_std,
            "seed":        seed if seed is not None else "None",
            "y_holder":    "R" if d.scenario == "A" else "O",
        }]),
        "metadata.csv",
    )

    return paths


# ------------------------------------------------------------------ #
# Combined entry point
# ------------------------------------------------------------------ #

def generate_and_save(
    n_R:          int,
    d_R:          int,
    n_O:          int,
    d_O:          int,
    n_intersect:  int,
    scenario:     Literal['A', 'B'],
    noise_std:    float = 0.1,
    seed:         Optional[int] = None,
    out_dir:      Optional[str] = None,
) -> Dict[str, str]:
    """Generate data and save to CSV. Returns dict of written paths."""
    if out_dir is None:
        seed_tag = str(seed) if seed is not None else "rand"
        out_dir  = f"vfl_{scenario}_n{n_intersect}_{seed_tag}"

    data  = generate_vfl_data(
        n_R, d_R, n_O, d_O, n_intersect, scenario, noise_std, seed
    )
    paths = save_vfl_data(data, out_dir, noise_std=noise_std, seed=seed)

    print(f"Saved {len(paths)} files to: {os.path.abspath(out_dir)}/")
    for name, path in paths.items():
        shape = pd.read_csv(path).shape
        print(f"  {name:<22} {shape}")

    return paths


# ------------------------------------------------------------------ #
# CLI
# ------------------------------------------------------------------ #

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate synthetic VFL data.")
    p.add_argument("--n_R",         type=int,   required=True)
    p.add_argument("--d_R",         type=int,   required=True)
    p.add_argument("--n_O",         type=int,   required=True)
    p.add_argument("--d_O",         type=int,   required=True)
    p.add_argument("--n_intersect", type=int,   required=True,
                   help="Intersection size (1 <= n_intersect <= n_R).")
    p.add_argument("--scenario",    type=str,   required=True, choices=["A", "B"])
    p.add_argument("--noise_std",   type=float, default=0.1)
    p.add_argument("--seed",        type=int,   default=None)
    p.add_argument("--out_dir",     type=str,   default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    generate_and_save(
        n_R=args.n_R,
        d_R=args.d_R,
        n_O=args.n_O,
        d_O=args.d_O,
        n_intersect=args.n_intersect,
        scenario=args.scenario,
        noise_std=args.noise_std,
        seed=args.seed,
        out_dir=args.out_dir,
    )