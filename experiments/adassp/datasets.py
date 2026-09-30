"""
datasets.py -- Python ports of the per-dataset preparation scripts of Wang (2018),
<https://github.com/yuxiangw/optimal_dp_linear_regression> (data/<name>/generatedata_<name>.m),
followed by the preprocessing of code/exp_uci.m.

Wang's repository has no licence file, so NOTHING from it is copied into this repository: the raw
files are read at run time from a local clone (default /home/user/yuxiangw/optimal_dp_linear_regression,
override with the environment variable WANG_REPO), and every step below is re-implemented from
the MATLAB source.

Per dataset, generatedata_<name>.m does (in this order)
    read the file -> select columns -> (sometimes) transform y, e.g. log(y) or log(y+1)
    -> shuffle (irrelevant here: we draw our own folds) -> centre the features and divide by
    std (N-1) (0/0 = NaN for constant columns) -> centre y (full data) -> 10-fold cvpartition.
exp_uci.m then does
    data(isnan(data)) = 0;  y = y / max|y|;  X = zscore(X);  X = X ./ ||row||_2.
Because zscore is invariant to the per-column affine map done by generatedata, the only
generatedata steps that matter are the column choice, the y transform, the y centring and the NaN
handling of constant columns; they are all reproduced exactly.

MATLAB reader quirks that matter (each one was checked against the published trivial-predictor
MSE, which depends on y only: mean over folds of mean(y_test^2)):
  * concrete, yacht: generatedata deletes the last column (`data(:, end) = []`) before taking y as
    the last column. In the published run that deletion evidently removed an extra, empty column
    produced by textread (trailing whitespace / CR line endings): the published trivial MSE is
    reproduced only with y = compressive strength (concrete: 0.12739 vs 0.1274) and
    y = log(residuary resistance) (yacht: 0.10516 vs 0.1053); deleting a real column would give
    0.0391 and 0.2572. We therefore keep all real columns for these two files.
  * energy: 10 real columns; deleting the last (cooling load) leaves y = heating load (0.23518 vs
    0.2352), as the script says.
  * machine: csvread pads the two short rows (vendor code fused with MYCT by the upstream sed) with
    a trailing 0; reproduced.

The four "largedata" sets (buzz, houseelectric, slice, song) need downloads and are not used;
keggdirected / keggundirected are not part of the published 29-dataset results and are skipped.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass

import numpy as np
import scipy.io as sio

WANG_REPO = os.environ.get("WANG_REPO", "/home/user/yuxiangw/optimal_dp_linear_regression")
DATA = os.path.join(WANG_REPO, "data")

# published dataset axis of code/exp_results.mat (ASCII order of the data folders at the time)
PUBLISHED_ORDER = ("3droad airfoil autompg autos bike breastcancer concrete concreteslump "
                   "elevators energy fertility forest gas housing kin40k machine parkinsons "
                   "pendulum pol protein pumadyn32nm servo skillcraft sml solar stock "
                   "tamielectric wine yacht").split()
PUBLISHED_METHODS = ["trivial", "non-private", "SSP", "ObjPert", "OPS", "AdaOPS", "AdaSSP",
                     "OPS-concentrated", "OPS-diffused", "OPS-balanced", "OPS-worstcase"]


# -----------------------------------------------------------------------------
# MATLAB-like readers
# -----------------------------------------------------------------------------

def _lines(path):
    with open(path, "r", newline=None) as fh:          # universal newlines: \n, \r\n and \r
        return [ln for ln in fh.read().splitlines() if ln.strip()]


def csvread(path):
    """csvread: comma separated numbers; empty fields are 0, short rows are zero-padded."""
    rows = [[float(t) if t.strip() else 0.0 for t in ln.split(",")] for ln in _lines(path)]
    w = max(len(r) for r in rows)
    return np.array([r + [0.0] * (w - len(r)) for r in rows])


def wsread(path):
    """textread / dlmread on whitespace separated numeric files (all rows equal length)."""
    rows = [[float(t) for t in ln.split()] for ln in _lines(path)]
    w = {len(r) for r in rows}
    assert len(w) == 1, (path, w)
    return np.array(rows)


def matload(path):
    return np.asarray(sio.loadmat(path)["data"], float)


# -----------------------------------------------------------------------------
# generatedata_<name>.m ports: return (features, y) BEFORE standardization/centring,
# with y already transformed (log etc.) as in the script.
# -----------------------------------------------------------------------------

def _p(*parts):
    return os.path.join(DATA, *parts)


def _raw_3droad():
    d = csvread(_p("3droad", "3D_spatial_network.txt"))
    return d[:, :-1], d[:, -1]


def _raw_airfoil():
    d = wsread(_p("airfoil", "airfoil_self_noise.dat.txt"))
    return d[:, :-1], d[:, -1]


def _raw_autompg():
    d = wsread(_p("autompg", "auto-mpg.txt"))
    return d[:, 1:], d[:, 0]


def _raw_autos():
    d = csvread(_p("autos", "imports-85.data.txt"))
    return d[:, :-1], np.log(d[:, -1])


def _raw_bike():
    d = csvread(_p("bike", "hour.csv"))[:, 1:]
    return d[:, :-1], np.log(d[:, -1])


def _raw_breastcancer():
    d = csvread(_p("breastcancer", "wpbc.data.txt"))
    cols = [0, 1] + list(range(3, d.shape[1] - 1))           # MATLAB [1:2, 4:end-1]
    return d[:, cols], d[:, 2]


def _raw_concrete():
    d = wsread(_p("concrete", "Concrete_Data.txt"))           # 9 real columns (see module doc)
    return d[:, :-1], d[:, -1]


def _raw_concreteslump():
    d = csvread(_p("concreteslump", "slump_test.data.txt"))
    d = d[:, :-3]                                             # data(:, end-2:end) = []
    return d[:, :-1], d[:, -1]


def _raw_elevators():
    d = np.vstack([csvread(_p("elevators", "elevators.data")),
                   csvread(_p("elevators", "elevators.test"))])
    return d[:, :-1], np.log(d[:, -1])


def _raw_energy():
    d = wsread(_p("energy", "energy.txt"))[:, :-1]            # data(:, end) = []
    return d[:, :-1], d[:, -1]


def _raw_fertility():
    d = csvread(_p("fertility", "fertility_Diagnosis.txt"))
    k = d.shape[1]
    cols = list(range(k - 2)) + [k - 1]                       # [1:end-2, end]
    return d[:, cols], d[:, k - 2]


def _raw_forest():
    d = csvread(_p("forest", "forestfires.csv"))
    return d[:, :-1], np.log(d[:, -1] + 1)


def _raw_gas():
    rows = []
    for j in range(1, 11):
        for ln in _lines(_p("gas", f"batch{j}.dat")):
            tok = [t for t in re.split(r"[ :;]+", ln.strip()) if t != ""][:258]
            rows.append([float(t) for t in tok])
    d = np.array(rows)
    d = d[d[:, 0] == 1]                                       # class 1 (ethanol)
    d = d[:, 1::2]                                            # data(:, 2:2:258)
    return d[:, 1:], np.log(d[:, 0] + 1)


def _raw_housing():
    d = wsread(_p("housing", "housing.txt"))
    return d[:, :-1], d[:, -1]


def _raw_kin40k():
    d = matload(_p("kin40k", "kin40k_all.mat"))
    return d[:, :-1], d[:, -1]


def _raw_machine():
    d = csvread(_p("machine", "machine.data.txt"))[:, :-1]    # data(:, end) = []
    return d[:, :-1], np.log(d[:, -1])


def _raw_parkinsons():
    d = wsread(_p("parkinsons", "parkinsons.txt"))
    y = d[:, 5]
    d = np.delete(d, [4, 5], axis=1)
    return d, y


def _raw_pendulum():
    d = matload(_p("pendulum", "pendulum_all.mat"))
    return d[:, :-1], d[:, -1]


def _raw_pol():
    d = matload(_p("pol", "pol_all.mat"))
    return d[:, :-1], d[:, -1]


def _raw_protein():
    d = csvread(_p("protein", "CASP.csv"))
    return d[:, 1:], np.log(d[:, 0] + 1)


def _raw_pumadyn32nm():
    d = matload(_p("pumadyn32nm", "pumadyn32nm_all.mat"))
    return d[:, :-1], d[:, -1]


def _raw_servo():
    d = csvread(_p("servo", "servo.data.txt"))
    return d[:, :-1], np.log(d[:, -1])


def _raw_skillcraft():
    d = wsread(_p("skillcraft", "SkillCraft.txt"))
    cols = list(range(12)) + list(range(13, d.shape[1]))      # [1:12, 14:end]
    return d[:, cols], np.log(d[:, 12])


def _raw_sml():
    d = wsread(_p("sml", "sml.txt"))
    cols = list(range(11)) + list(range(12, d.shape[1]))      # [1:11, 13:end]
    return d[:, cols], d[:, 11]


def _raw_solar():
    d = wsread(_p("solar", "flare.data2.txt"))
    return d[:, :-1], d[:, -1]


def _raw_stock():
    d = wsread(_p("stock", "stock.txt"))
    cols = [0, 1, 2] + list(range(4, d.shape[1]))             # [1:3, 5:end]
    return d[:, cols], d[:, 3]


def _raw_tamielectric():
    d = csvread(_p("tamielectric", "eb.arff"))
    k = d.shape[1]
    d = d[:, list(range(k - 2)) + [k - 1]]                    # [1:end-2, end]
    return d[:, [0] + list(range(2, d.shape[1]))], d[:, 1]


def _raw_wine():
    d = wsread(_p("wine", "winequality-red.csv"))
    k = d.shape[1]
    return d[:, list(range(k - 2)) + [k - 1]], d[:, k - 2]    # y = alcohol, as in the script


def _raw_yacht():
    d = wsread(_p("yacht", "yacht_hydrodynamics.data.txt"))   # 7 real columns (see module doc)
    return d[:, :-1], np.log(d[:, -1])


RAW = {name: globals()[f"_raw_{name}"] for name in PUBLISHED_ORDER}

# datasets skipped, with the reason
SKIPPED = {
    "buzz": "largedata: needs a download from UCI (wget_largedata.sh); no published result",
    "houseelectric": "largedata: needs a download; no published result",
    "slice": "largedata: needs a download; no published result",
    "song": "largedata: needs a download; no published result",
    "keggdirected": "not in the published 29-dataset exp_results.mat (added to the repo later)",
    "keggundirected": "not in the published 29-dataset exp_results.mat (added to the repo later)",
}


# -----------------------------------------------------------------------------
# generatedata standardization + exp_uci preprocessing
# -----------------------------------------------------------------------------

def _std1(a):
    return a.std(axis=0, ddof=1)


def generatedata_finish(F, y):
    """Centre features, divide by std (0/0 -> NaN as in MATLAB), centre y; returns data=[F, y]."""
    F = F - F.mean(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        F = F / _std1(F)
    y = y - y.mean()
    return np.column_stack([F, y])


def exp_uci_preprocess(data):
    """code/exp_uci.m: NaN -> 0; y / max|y|; zscore(X) (sigma 0 -> 1); unit-norm rows."""
    data = np.where(np.isnan(data), 0.0, data)
    X, y = data[:, :-1], data[:, -1]
    y = y / np.max(np.abs(y))
    mu, sd = X.mean(0), _std1(X)
    sd = np.where(sd == 0, 1.0, sd)
    X = (X - mu) / sd
    nr = np.sqrt((X**2).sum(1))
    X = X / np.where(nr == 0, 1.0, nr)[:, None]
    return X, y


@dataclass
class Dataset:
    name: str
    X: np.ndarray          # n x d, rows unit norm
    y: np.ndarray          # n, |y| <= 1

    @property
    def n(self):
        return self.X.shape[0]

    @property
    def d(self):
        return self.X.shape[1]


_CACHE: dict = {}


def load(name) -> Dataset:
    if name not in _CACHE:
        F, y = RAW[name]()
        X, yy = exp_uci_preprocess(generatedata_finish(F, y))
        _CACHE[name] = Dataset(name, X, yy)
    return _CACHE[name]


def kfold(n, k=10, seed=0):
    """Random k-fold partition (our replacement for MATLAB cvpartition(m,'k',k)): fold sizes
    differ by at most one, like cvpartition. Returns a list of test-index arrays."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    return [np.sort(perm[i::k]) for i in range(k)]


def published():
    """Wang's code/exp_results.mat -> (err, std) arrays (method, eps, dataset)."""
    m = sio.loadmat(os.path.join(WANG_REPO, "code", "exp_results.mat"))
    return m["results_err"], m["results_std"]


EPS_LIST = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10]
DELTA = 1e-6

if __name__ == "__main__":
    err, _ = published()
    print(f"{'dataset':14s} {'n':>7s} {'d':>4s}  trivial ours / published   max|row norm-1|")
    for i, nm in enumerate(PUBLISHED_ORDER):
        ds = load(nm)
        tr = float(np.mean([np.mean(ds.y[te] ** 2) for te in kfold(ds.n)]))
        print(f"{nm:14s} {ds.n:7d} {ds.d:4d}  {tr:.5f} / {err[0, 0, i]:.5f} "
              f"({tr / err[0, 0, i]:.4f})   {np.abs(np.linalg.norm(ds.X, axis=1) - 1).max():.1e}")
