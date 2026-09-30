# AdaSSP vs the revised VFL protocol

Benchmark of the revised Scenario-B protocol against AdaSSP (Wang 2018) on Wang's UCI regression
datasets, with Wang's preprocessing, 10-fold cross-validation and test MSE.

## Run it locally

Requirements: Python 3.9+, `numpy`, `scipy`, `matplotlib`.

```bash
# 1. this repository, on the working branch
git clone -b claude/friendly-heisenberg-8yh85u https://github.com/shukabh/VerticalFederatedRegression
cd VerticalFederatedRegression
pip install numpy scipy matplotlib

# 2. Wang's repository, for the raw data only (it has no licence, so nothing is copied from it)
git clone https://github.com/yuxiangw/optimal_dp_linear_regression ../optimal_dp_linear_regression
export WANG_REPO=../optimal_dp_linear_regression        # Windows PowerShell: $env:WANG_REPO="..\optimal_dp_linear_regression"

# 3. run
python experiments/adassp/run_benchmark.py --quick      # 3 small datasets, 3 values of eps: ~3 s
python experiments/adassp/run_benchmark.py              # 9 datasets, 10 values of eps, R = 50: ~15 s
python experiments/adassp/run_benchmark.py --all        # all 29 datasets of Wang's published run: ~2 min
```

Outputs (default `reports/adassp/`, change with `--out`):

| file | content |
|---|---|
| `results.csv` | one row per dataset × ε × method: test MSE, standard error and 95% CI over folds, middle 95% of individual runs, ratio to non-private, Wang's published MSE where it exists, VFL gate pass rate, mean λ |
| `compare.csv` | paired comparisons of ours against AdaSSP, matched AdaSSP and the uncorrected estimate: mean difference, 95% CI half-width, verdict (better / worse / no difference) |
| `summary.md` | tables at ε = 0.1, 1, 10, counts of paired verdicts, and a check of our AdaSSP against Wang's published numbers |
| `fig_mse_vs_eps.png` | test MSE against ε, one panel per dataset |

Uncertainty: a run is one CV fold × one noise draw. The folds are the independent units, so a
method's 95% CI uses the t distribution over the 10 fold means (error bars in the figure); the shaded
bands are the middle 95% of individual runs. Fold-to-fold variation is shared by all methods, so
per-method intervals overlap even when one method is consistently better: compare methods with the
paired tests in `compare.csv`.

### Intersection-size sweep

```bash
python experiments/adassp/sweep_n.py            # bike, elevators, pol, protein: ~20 s
```

Test MSE against the matched-set size n on real data, for ours and AdaSSP at ε ∈ {0.5, 1, 2, 4}.
For each fold the training folds are shuffled once and the first n rows form the matched set
(nested across n); the held-out fold is the test set. Results are relative to non-private at the
same n. Outputs in `reports/adassp/sweep/`: `sweep.csv`, `crossings.csv` (n needed to come within
10% / 25% of non-private), `summary.md`, `fig_sweep_n.png` and a slide version `fig_sweep_n_deck.png`.
kin40k is left out: its linear fit is no better than predicting zero.

Useful options: `--datasets bike pol`, `--eps 0.5 1 2`, `--R 100` (noise draws per fold),
`--d-r 3` (features held by R; default ⌊d/2⌋), `--rho-star 2.5` (ridge constant), `--seed 1`.
`python experiments/adassp/datasets.py` checks the data loaders against Wang's published
trivial-predictor MSE.

## What is compared

| method | privacy guarantee | notes |
|---|---|---|
| Trivial, Non-private | none | θ = 0 and (XᵀX + I)⁻¹Xᵀy, Wang's references |
| SSP | whole record (x, y) | Wang's plain sufficient-statistics perturbation |
| AdaSSP (published) | whole record (x, y) | Wang's algorithm and noise constants; a **stronger** guarantee than ours |
| AdaSSP (matched) | ours: replace-one of (x_O, y), x_R fixed | AdaSSP's adaptive ridge under our adjacency and accountant, A exact. Idealized: its private λ_min step needs XᵀX in the clear, which nobody holds in VFL |
| VFL, ours | ours | revised protocol: fixed ridge λ = 2ρ*σ√p, bias correction, release gate ρ̂ ≥ 1 (a failed draw releases nothing, θ = 0) |
| VFL, uncorrected | ours | same release and ridge, no bias correction |

Setup details: R holds the first ⌊d/2⌋ features, O holds the rest and y. Wang's joint row
normalization gives ‖x_R‖, ‖x_O‖, |y| ≤ 1, so the committed bounds are B_R = B_O = B_y = 1 and
Δ_rep = √10. δ = 10⁻⁶ throughout. As in Wang's code, the scaling y / max|y| uses the data
maximum (a non-private preprocessing step shared by every method).

## Files

- `datasets.py`: Python ports of Wang's per-dataset preparation scripts and of `exp_uci.m`
- `methods.py`: the estimators above, batched over noise draws; VFL parts call `experiments/sim_core.py`
- `run_benchmark.py`: cross-validation driver, CSV, summary and figure
