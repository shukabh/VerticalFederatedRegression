"""Inputs for the validation runs: the worked toy example and synthetic cohorts.

Synthetic cohorts use the simulation population of experiments/sim_core.py (d_R = 4, d_O = 6, the paper's
example bounds B_R = 5.59, B_O = 6.24, B_y = 2.89 come from it): matched individuals carry R and O features
and y from one draw; O's register also holds individuals outside R's cohort, R's cohort holds individuals
outside O's scope.  Identifiers are random 9-digit strings.
"""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.normpath(os.path.join(HERE, '..', '..', 'experiments'))

# ---------------------------------------------------------------- toy example (experiments/toy_example.py)
TOY_R = [("104", 0.5, -0.2), ("221", -0.4, 0.8), ("317", 0.9, 0.1)]
TOY_O = [("088", 0.3, 1.0, -0.5), ("104", -0.6, 1.0, 0.7), ("150", 0.1, 1.0, -0.2),
         ("221", 0.8, 0.0, 0.4), ("399", -0.2, 0.0, 0.9)]
TOY_TAU = {"150": 1, "221": 3, "088": 4, "399": 5, "104": 6}      # the dummy "7e5" takes position 2


def toy():
    data_R = dict(ids=[r[0] for r in TOY_R], X=np.array([r[1:] for r in TOY_R]))
    data_O = dict(ids=[o[0] for o in TOY_O], X=np.array([o[1:3] for o in TOY_O]), y=np.array([o[3] for o in TOY_O]))
    bounds = dict(B_R=float(np.sqrt(3.0)), B_O=float(np.sqrt(2.0)), B_y=1.0)
    return data_R, data_O, bounds


def population(seed=0):
    sys.path.insert(0, EXP)
    import sim_core as sc
    return sc.Population(seed=seed)


def _ids(rng, k, taken):
    out = []
    while len(out) < k:
        c = f"{int(rng.integers(10 ** 8, 10 ** 9))}"
        if c not in taken:
            taken.add(c)
            out.append(c)
    return out


def synthetic(n_bar_O, n_scope, n_R, n, seed=0, pop=None):
    """O's scope has n_scope real rows (padded by O to n_bar_O), R's cohort n_R rows, n matched."""
    assert n <= min(n_scope, n_R)
    rng = np.random.default_rng(seed)
    pop = pop or population()
    taken = set()
    ids_m = _ids(rng, n, taken)
    ids_O_only = _ids(rng, n_scope - n, taken)
    ids_R_only = _ids(rng, n_R - n, taken)
    Zm, ym, _ = pop.draw(n, rng)                    # intercept | 4 R features | 6 O features (clipped)
    Zo, yo, _ = pop.draw(n_scope - n, rng)
    Zr, _, _ = pop.draw(n_R - n, rng)
    dR = pop.d_R
    XR = np.vstack([Zm[:, 1:1 + dR], Zr[:, 1:1 + dR]])
    ids_R = ids_m + ids_R_only
    XO = np.vstack([Zm[:, 1 + dR:], Zo[:, 1 + dR:]])
    yO = np.concatenate([ym, yo])
    ids_O = ids_m + ids_O_only
    pR = rng.permutation(n_R)
    pO = rng.permutation(n_scope)
    data_R = dict(ids=[ids_R[i] for i in pR], X=XR[pR])
    data_O = dict(ids=[ids_O[i] for i in pO], X=XO[pO], y=yO[pO])
    truth = dict(matched=set(ids_m))
    bounds = dict(B_R=pop.B_R, B_O=pop.B_O, B_y=pop.B_y)
    return data_R, data_O, truth, bounds


def plaintext_stats(data_R, data_O, B_R, B_O, B_y):
    """G, c, y'y over the matched records, with both parties' clipping, in the protocol's coordinates."""
    Omap = {i: k for k, i in enumerate(data_O['ids'])}
    XR = np.asarray(data_R['X'], float)
    lim = np.sqrt(B_R ** 2 - 1)
    nr = np.linalg.norm(XR, axis=1)
    XRc = XR * np.minimum(1.0, lim / np.maximum(nr, 1e-300))[:, None]
    XO = np.asarray(data_O['X'], float)
    no = np.linalg.norm(XO, axis=1)
    XOc = XO * np.minimum(1.0, B_O / np.maximum(no, 1e-300))[:, None]
    yc = np.clip(np.asarray(data_O['y'], float), -B_y, B_y)
    rows, ys, matched = [], [], []
    for k, i in enumerate(data_R['ids']):
        if i in Omap:
            j = Omap[i]
            rows.append(np.concatenate([[1.0], XRc[k], XOc[j]]))
            ys.append(yc[j])
            matched.append(i)
    Z = np.array(rows)
    y = np.array(ys)
    return dict(G=Z.T @ Z, c=Z.T @ y, yy=float(y @ y), n=len(y), matched=matched, Z=Z, y=y)
