# Plan: bias correction, coefficient-accuracy graphics, AdaSSP comparison

This plan responds to three comments on the Scenario B work:

1. The protocol is correct, but does the bias correction matter when σ itself may be large?
2. A figure comparing the VFL β (with bias correction) to the true β (e.g. RMSE) against the
   intersection size on a log scale, for several ε, with confidence intervals showing the
   variation across runs. In practice the coefficients will be on different scales, so they
   should also be compared separately.
3. Compare the method with AdaSSP from Wang (UAI 2018,
   <https://github.com/yuxiangw/optimal_dp_linear_regression>) on its benchmark datasets.

## What we already know (comment 1)

The accuracy study (`reports/accuracy/results.json`) already contains the answer in rough form.
It used the user's original modules, whose ridge is adaptive. Errors are measured against the
protocol's target, the clipped OLS.

| Regime | Cells | MSE change from the correction | Share of MSE that is bias, without correction |
|---|---|---|---|
| ρ₀ < ~2: full ridge forced (σ large relative to λ_min(G)) | 26 cells, e.g. n = 4,000 at ε ≤ 4 | **0 to −6.6%** (slightly worse) | 88–99.8%, almost all of it deterministic ridge shrinkage |
| ρ₀ ≈ 2.7–4, no ridge | n = 4,000 at ε = 8; n = 16,000 at ε = 2; n = 64,000 at ε = 0.5 | **+5 to +11%** | 0.9–1.4% |
| ρ₀ ≈ 6–7 | n = 16,000 at ε = 4; n = 64,000 at ε = 1 | +1.4 to +2% | 0.3% |
| ρ₀ > 10 | remaining large-n cells | < 1% | ≤ 0.2% |

(ρ₀ = λ_min(G)/(2σ√p) on the clean Gram; "no ridge" means λ = 0 in every draw.)

This is what the theory predicts:
- The mechanism bias is O(σ²), while its standard deviation is O(σ). So bias/sd ∝ 1/ρ.
- When σ is large (small ρ), the ridge is forced. Its deterministic shrinkage, −λP_λΨβ̂, then
  swamps both the mechanism bias and the variance. The correction deliberately doesn't touch
  that shrinkage, so it is immaterial there, and the noisy plug-in `M(P̃)` can make things
  slightly worse.
- The correction matters only in a window around ρ ≈ 2–6, and when several releases are
  averaged. Averaging reduces variance but leaves the bias.

The experiments below confirm this with the **revised** estimator (fixed λ from public inputs),
on a finer ρ grid, with coverage and multi-release views.

## Workstream 0: shared harness (prerequisite)

- **`experiments/sim_core.py`:** the revised estimator in numpy.
  - replace-one Δ (the closed form), analytic-Gaussian σ, and a discrete-Gaussian draw;
  - symmetric masked noise on the O-dependent blocks, with `A` clean;
  - fixed λ = max{0, 2ρ*σ√p − nℓ} and the bias correction `BiasOp`;
  - baselines: non-private OLS, the revised protocol without correction, and the original
    adaptive gate.

  The HE and PSI layers are exact to about 1e-10 (already verified), so the Monte Carlo can skip
  them. A few socket runs will spot-check agreement.
- **A realistic synthetic generator.** The current generator uses U[0,1] features and R² ≈ 0.99,
  which is unrealistic and makes the sampling error tiny. The new one will have:
  - correlated features (Toeplitz or factor covariance) on **heterogeneous scales**: income-like
    log-normal, age-like, binary, and counts;
  - an intercept;
  - coefficients of mixed magnitude;
  - noise tuned to R² ∈ {0.3, 0.6, 0.9}.

  Standardization uses public (population) constants, as the protocol requires. Bounds are
  committed from the known generating distribution.
- **One Monte Carlo loop.** Each replicate redraws the data and the matching, then the DP noise.
  This captures sampling and privacy variation together.

## Workstream 1: does the bias correction matter? (comment 1)

**E1 — where the correction helps.** A grid of n × ε chosen to sweep ρ from about 0.5 to 50, with
5,000 noise draws per cell (antithetic pairs, for precise bias estimates). Per cell:
- the MSE ratio of corrected to uncorrected estimates;
- the bias share of MSE, with and without correction;
- coverage of nominal 95% intervals per coefficient, with and without correction.

The intervals use the first-order variance `s²P + mechanism block` from the paper.

**E2 — decomposition when σ is large.** Split the error into three parts:
1. the deterministic shrinkage `−λP_λΨβ̂`;
2. the mechanism bias;
3. the variance.

Plot them against ρ, marking the ρ* that forces the ridge. This shows directly that the correction
acts only on part 2, and that part 1 dominates when σ is large.

**E3 — multiple releases.** Average K ∈ {1, 5, 20, 100} independent releases, as in meta-analysis
or repeated yearly releases. Report where the correction becomes necessary (roughly K ≳ (sd/bias)²).

**E4 — fixed vs adaptive λ.** Compare the revised fixed-λ rule with the original adaptive gate,
tying back to the Theorem 2 finding.

**Outcome.** A figure and a short paper subsection, "When does the correction matter?", with a
recommendation:
- apply the correction only when ρ̂ ≥ ρ_bc (the threshold is set by E1), or always if it is never
  harmful;
- always report ρ̂;
- rewrite §7.7 of the paper with the measured window.

## Workstream 2: coefficient-accuracy figures (comment 2)

**Figure A — RMSE vs intersection size.**
- x-axis: n on a log scale (100, 250, 500, 1k, 2.5k, 5k, 10k, 25k, 50k, 100k).
- y-axis (log): RMSE of β̂ against the true (generating) β, on the **standardized scale**.
- One line per ε ∈ {0.5, 1, 2, 4, 8}. Solid lines are corrected; dashed lines are uncorrected.
- A black reference line for non-private OLS on the true matches: the sampling floor, which is
  also what plaintext linkage plus plaintext OLS gives.
- Bands:
  - shaded: the 2.5–97.5% range of per-run RMSE across R = 200 replicates (the run-to-run
    variation);
  - thin: a bootstrap 95% CI for the mean RMSE.

**Figure B — each coefficient on its own scale.** Small multiples, one panel per coefficient,
grouped into R block and O block, with x = n (log) and one line per ε. Two scale-free metrics:
- **Efficiency ratio** `RMSE_DP,j / SE_OLS,j`: the extra error measured in that coefficient's own
  OLS standard errors.
- **Standardized coefficient error**: errors in `β_j·s_j/s_y` units, meaning SDs of y per SD of
  x_j. The protocol estimates on this scale anyway, because both parties standardize with public
  constants.

Raw relative error |β̂_j − β_j|/|β_j| is shown only for coefficients well away from zero, since it
is unstable near zero. Theory predicts that R-block coefficients degrade with tr(S⁻¹); the panels
will show whether they do.

**Figure C — coverage and sign recovery per coefficient.** Coverage of the 95% intervals and the
rate at which each coefficient's sign is recovered, against n, for each ε.

The same figures will be repeated on two or three real datasets from Workstream 3. There the "true
β" is the full-data non-private OLS.

## Workstream 3: comparison with AdaSSP (comment 3)

**What Wang's repo contains**
- MATLAB code: `adassp.m`, `suffstats_perturb.m` (SSP), `ObjPert.m`, `adaops.m`, `noisySGD.m`,
  and `exp_uci.m`.
- 36 UCI regression datasets:
  - 6 as ready `.mat` files: elevators, kin40k, pendulum, pol, pumadyn32nm, kegg;
  - the rest as raw CSV/TXT, with a small MATLAB script per dataset to prepare it;
  - 4 large ones via a download script.
- Published results in `code/exp_results.mat`.
- **No license file.** I will reimplement AdaSSP from the paper and read the data at run time,
  rather than copying their code or data into this repo.

**AdaSSP, as implemented**
- Data are preprocessed so that B_X = B_Y = 1: X is z-scored and each row normalized to unit ℓ₂
  norm, and y is divided by max|y|.
- ε is split into three equal parts:
  - one to privately estimate λ_min(XᵀX), giving `λ = max(0, η − λ̃_min)`;
  - one to add symmetric Gaussian noise to XᵀX (with `+I`);
  - one to add Gaussian noise to Xᵀy.
- Noise scale: `√log(6/δ)·B²/(ε/3)`.
- There is no bias correction.
- Evaluation: held-out prediction MSE by cross-validation, for ε ∈ {0.01, …, 10}, δ = 1e-6.

**Setup**
1. Port AdaSSP and SSP to Python and **validate against `exp_results.mat`** on the datasets
   available locally.
2. Port the per-dataset preparation scripts for about 10 datasets spanning n ≈ 300 to 50k, plus
   the 6 `.mat` sets.
3. Simulate a vertical split: a random half of the columns goes to R and the rest to O (5
   random splits per dataset). R's cohort is a random subset of rows of size n (the intersection
   sweep); O holds every row. The PSI is exact, so we use the true matches.
4. Use Wang's preprocessing and δ = 1e-6. With B_R = B_O = 1 (valid, because ‖x_R‖, ‖x_O‖ ≤ ‖x‖ = 1)
   and B_y = 1, all methods see the same data.

**Methods**
- trivial predictor (predicts 0);
- non-private OLS;
- SSP;
- **AdaSSP as published** (central curator);
- **AdaSSP recalibrated** to our adjacency and accountant;
- **VFL revised protocol** (fixed λ, with correction);
- VFL without correction;
- VFL with the original adaptive gate.

**Metrics**
- Wang's metric: test MSE over 10 folds, redrawn in Python with fixed seeds.
- Coefficient RMSE against full-data OLS, on the standardized scale.
- Relative efficiency MSE/MSE_OLS.

**Plots**
- Per-dataset plots of MSE against ε, in Wang's format.
- A summary across datasets: a performance profile, or the median ratio against AdaSSP.
- Intersection-size curves on the larger datasets.

**Fairness caveats.** These will be stated in the write-up.
- **The guarantees differ.** Published AdaSSP protects the whole record (x, y) against the
  analyst. Our protocol protects O's side, (x_O, y), with R's features known to R, and leaves
  `A` un-noised. Hence two comparisons:
  - "as published", as the reference point readers know;
  - "matched guarantee", which isolates the estimator (private adaptive λ vs public fixed λ plus
    correction).
- **AdaSSP's private λ_min has no direct VFL analogue.** Nobody sees G in the clear. The closest
  VFL analogue computes λ from G̃, which is free but adaptive; that is the original gate, and it
  is included.
- **Calibration conventions differ.** AdaSSP's noise constants, its symmetric-noise convention
  (off-diagonal variance halved) and its adjacency differ from ours. The matched comparison puts
  everything on one accountant.
- **Loose bounds.** B_R = B_O = 1 is loose by up to √2 against the joint constraint
  ‖x_R‖² + ‖x_O‖² ≤ 1. A tighter joint sensitivity can be derived later.
- **Non-private preprocessing.** z-scoring with sample statistics is not DP. It is the
  benchmark's convention and is kept for comparability.

**Expected result.** Under the matched guarantee, VFL should do as well as or better than central
SSP. That is because `A` is clean and our sensitivity has no `B_R⁴` term. Against AdaSSP, the
difference comes down to the λ rule. Published AdaSSP gives a stronger guarantee, so beating it
would not be a like-for-like claim.

## Deliverables and order

| Step | Output | Rough effort |
|---|---|---|
| WS0 | `experiments/sim_core.py`, new generator, validation against socket runs | small |
| WS1 | `experiments/bias_correction_study.py`, figures E1–E4, paper subsection | medium |
| WS2 | `experiments/coef_accuracy_figures.py`, Figures A–C (synthetic, then real data) | medium |
| WS3 | `experiments/adassp/` (port, dataset loaders, runner), validation against `exp_results.mat`, comparison figures and table | largest |
| Paper | new section "Empirical evaluation"; §7.7 updated; README | small |

## Decisions (defaults used unless changed)

1. **"True β"** means the generating β. The OLS-on-true-matches floor is drawn as a reference
   line.
2. **Bands** show the per-run 2.5–97.5% range, plus a bootstrap CI for the mean.
3. **AdaSSP** is reported both as published and under the matched guarantee.
4. **Datasets:** about 16 first (the 6 `.mat` sets plus about 10 ported), then all 36 if useful.
5. **"Our method"** is the revised protocol (fixed λ plus correction). The original adaptive gate
   is kept as a comparison.
