# Evaluation of protocol v3.1: Sections 12.5–12.8 rerun

These results replace the "(earlier mechanism)" subsections 12.5–12.8 of `paper/main.tex`. The draft
text is `paper/drafts/eval_v31.tex`. Every number below is regenerated from the JSON files in this
directory by `experiments/v31_eval/tables.py`, which also writes `key_numbers.md` (all tables) and
`tables_v31.tex` (LaTeX table bodies).

## What was run

A plaintext simulation of the exact v3.1 mechanism, built from the implementation's own modules:

- **Calibration.** `implementation/vfr/dp.py` `calibrate` with the implementation's CKKS constants. It
  gives σ₁, σ₂, σ_yy, the joint or separate y'y mode, the tight zCDP conversion and the δ slack.
- **Stage 1.** `solve.masked_noise` on (triu B, C) with s.d. σ₁, then `solve.stage1_rule`.
- **Stage 2.** Noise with s.d. σ₂ on every O-dependent entry, the gate ϑ̂ ≥ 1 and the solve. At
  R² = 0.5 every release also runs `solve.stage2_infer` (bootstrap, Wald and bias-aware intervals).
- **Data.** `vfr/data.population()` (`experiments/sim_core.Population`) and the per-party transforms
  of `bench/utility_runs.transform`.

Evaluation settings. They are stated as the evaluation's settings, not as protocol defaults:

- δ = 1e-9, except δ = 1e-6 in the AdaSSP comparison (12.7), as in Wang's published evaluation.
- f₁ = 0.1, ϑ* = 2.5, κ_Ψ = 2, α = 0.05 with α_L = α_U = α/20.
- y'y goes in stage 2 when it is free, otherwise in its own release with f_y = 0.1. This is the
  implementation's behaviour.
- Bias correction off.
- ε ∈ {0.5, 1, 2, 4, 8}.
- κ = 40, 40, 40, 44, 48 for ε = 0.5, …, 8. This keeps the slack (1 + e^ε)γ negligible. With κ = 40,
  Calibrate refuses ε = 8 at δ = 1e-9. The derived-κ rule that `dp.py` is getting (κ = 41 at ε = 4,
  46 at ε = 8) changes σ₂ by 0.1–0.15%.

The calibration reproduces Table `tab:noff_sim` exactly: σ₁ = 2746, 1411, 729, 380, 202 and
σ₂ = 1022, 525, 271, 141, 75, so n_off = 81k, 42k, 22k, 11k, 6.0k.

## Files

| file | content |
|---|---|
| `sim_accuracy.json` | 12.5. Per (R², n, ε): RMSE quantiles against the true β, OLS floor, ridge share, λ quantiles, Ψ choice, gate, prediction MSE, interval coverage at R² = 0.5, effect of the correction against the true β. Also per-coefficient efficiency ratios, crossings, and the stage-1 check at n = m·n_off |
| `bias_study.json` | 12.6. Conditional study per (n, ε): bias² share, max \|bias_j\|/sd_j, MSE ratios of the corrected and uncorrected estimates against θ_λ and against the sample OLS, check of Theorem 10.2 |
| `adassp_benchmark.json` | 12.7. 29 datasets × 5 ε × 10 folds × 5 arms (plus a replace-one recalibrated AdaSSP for check 32): fold-level test MSE on a common y scale, own non-private baselines, paired tests, verdict counts, AdaSSP accounting |
| `sweep_n.json` | 12.8. bike, elevators, pol, protein: ratio to own and to a common non-private baseline against n, and crossings |
| `encrypted_spot.json` | Encrypted spot check (full OpenFHE protocol) in the ridge-on regime: n = 10,000, ε = 2 |
| `key_numbers.md`, `tables_v31.tex` | Generated tables |

Figures: `paper/figures/{sim_rmse_vs_n,sim_efficiency_ratio,bias_mse_ratio,adassp_mse_vs_eps,sweep_n}_v31.png`.

Reproduce (4 cores; about 25 min in total):

```bash
python experiments/v31_eval/sim_accuracy.py run && python experiments/v31_eval/sim_accuracy.py summarize
python experiments/v31_eval/bias_study.py
python experiments/v31_eval/real_data.py benchmark --delta 1e-6     # WANG_REPO must point at Wang's repository
python experiments/v31_eval/real_data.py sweep --delta 1e-9
python experiments/v31_eval/encrypted_spot.py                        # optional, OpenFHE, about 35 s per run
python experiments/v31_eval/figures.py && python experiments/v31_eval/tables.py
```

Raw per-replicate arrays go to `experiments/_work/v31_eval/` (git-ignored).

## Headline results

### 12.5 Coefficient accuracy against n (synthetic population, δ = 1e-9)

- **Ridge switch-off matches Section 9.5.** Share of stage-1 releases with λ = 0 (200 cohorts per point):

  | n | 0.6·n_off | 0.8·n_off | 1.0·n_off | 1.2·n_off | 1.4·n_off |
  |---|---|---|---|---|---|
  | share with λ = 0 | 0.00 | 0.00 | 0.03–0.05 | 0.94–0.99 | 1.00 |

  Below the threshold the median λ is 1.5–1.6 times the linear prediction g* + r₁ − nℓ, because
  λ_min(Ĝ₁) is biased downward.
- **Median RMSE of the slopes, R² = 0.5, ratio to the OLS floor in brackets:**

  | n | ε = 0.5 | ε = 1 | ε = 2 | ε = 4 | ε = 8 |
  |---|---|---|---|---|---|
  | 1e3 | 0.242 (10.8) | 0.239 (10.7) | 0.232 (10.4) | 0.222 (9.9) | 0.203 (9.0) |
  | 1e4 | 0.213 (29.8) | 0.184 (25.7) | 0.131 (18.3) | 0.043 (6.0) | 0.012 (1.7) |
  | 1e5 | 0.014 (5.3) | 0.0076 (2.9) | 0.0045 (1.7) | 0.0033 (1.3) | 0.0028 (1.1) |

  The earlier mechanism gave 0.092 (12.3), 0.044 (5.8), 0.025 (3.4), 0.014 (1.9) and 0.010 (1.3) at
  1e4.
- **Twice the OLS floor.** Reached at n ≈ 7.3k (ε = 8), 23k (ε = 4) and 79k (ε = 2); not reached by
  1e5 at ε ≤ 1. The earlier mechanism reached it at 3.2k, 9.4k and 34k. The factor of 2.3–2.5 matches
  (σ₂/σ_earlier)² = 2.1–2.5.
- **R² = 0.9.** 23k (ε = 8) and 51k (ε = 4), against 13k and 30k before.
- **Prediction.** Test MSE comes within 10% of non-private at n ≈ 0.71·n_off for every ε, which is
  4.2k–58k. That is 1.7–5.2 times sooner than twice-the-floor accuracy on the coefficients.
- **Inference (R² = 0.5).**
  - The Wald interval for θ_λ covers 0.945–0.965 in all 70 (n, ε) cells (target 0.955).
  - The bias-aware interval for β, with no public bound on ‖β‖, is unbounded while the ridge is on
    and L̂ = 0. Once the ridge is off it covers the true β 0.930–0.964 and the clipped-population β
    0.945–0.962.
  - The gate refused no release.

### 12.6 Does the bias correction matter? (δ = 1e-9)

- **The mechanism bias is negligible.** At most 0.048% of the MSE around θ_λ, and at most 0.044 of a
  coefficient's s.d., in all 45 cells. The earlier mechanism gave 0.75% and 0.14. The stage-1 ridge
  holds ϑ_λ at 7.9–9.2 while it is on, and the ridge is off only from ϑ₀ ≈ 9.6.
- **Theorem 10.2 is confirmed.** The simulated bias is 0.997–1.010 of the predicted bias wherever
  ϑ₀ ≥ 1.
- **Against θ_λ the correction lowers the MSE by at most 0.55%** (earlier: 7–12%), almost all of it
  from lower variance.
- **Against the sample OLS** it costs up to 1.1% while the ridge is on.
- **Against the true β** it changes the MSE by −0.4% to +1.7%.
- **Averaging.** The bias would matter only after averaging at least about 2,100 releases (earlier:
  130).
- **Real data.** −0.01% to +0.39% change in test MSE over 29 datasets (earlier: −3.7% to +2.3%).

### 12.7 Benchmark against AdaSSP (δ = 1e-6, 29 datasets, 10 folds)

Arms:

- **(a)** AdaSSP with its published constants.
- **(b)** AdaSSP matched: our adjacency and accountant, A exact.
- **(c)** v3.1 with Wang's joint row normalization. Idealized, because no party can normalize a whole
  row in the vertical setting.
- **(d)** v3.1 with per-party block normalization.
- **(e)** v3.1 with per-party standardization.

Each arm's MSE is reported as a ratio to its own non-private fit, (Z'Z + I)⁻¹Z'y, and as an absolute
MSE on Wang's y/max|y| scale.

- **The mechanism.** With identical preprocessing, (c) has significantly higher test MSE than (a) on
  20 of 29 datasets at ε = 1, and lower on 6. The median gap is +28%. Against (b) it is higher on 18
  and lower on 8 (+15%). The earlier mechanism beat (b) on 27 of 29.
  - The cause is the ridge. v3.1's median λ is 3.2–5.8 times that of (b) and 4.7–12 times that of (a)
    at ε = 1: housing 786 / 188 / 127, protein 559 / 150 / 101.
  - v3.1 is ahead only where there is little linear signal: forest, fertility, pendulum, pumadyn32nm,
    solar, tamielectric.
- **The preprocessing.** (d) and (c) differ significantly on only 12 of 29 datasets (absolute MSE,
  ε = 1). The differences come from the non-private fits, for example elevators, where block
  normalization's non-private fit is 2.6 times worse than Wang's.
- **The implementable arm.** (d) is worse than (a) on 19 of 29 at ε = 1 and better on 5. The median
  gap is +31% at ε = 1 and +12% at ε = 8.
- **(e) is worse than (d)** on 23–25 datasets.
- **Check 32.**
  - Wang's constants are valid with room under add/remove adjacency: the tight conversion gives
    ε = 0.62–0.78 × nominal.
  - Under replace-one adjacency they give 0.97–1.28 × nominal.
  - Recalibrating AdaSSP to exactly (ε, δ) under replace-one raises its test MSE by a median 2–3%
    (range −5% to +28%). The published column therefore overstates AdaSSP's advantage only slightly.

### 12.8 Intersection size on real data (δ = 1e-9)

Matched records needed to come within 10% of the arm's own non-private fit. The first number is
v3.1 with block normalization; the number in brackets is AdaSSP with its published constants.

| dataset | n available | ε = 0.5 | ε = 1 | ε = 2 | ε = 4 | ε = 8 |
|---|---|---|---|---|---|---|
| bike | 13,903 | >14k (>14k) | >14k (9.1k) | >14k (4.4k) | >14k (2.3k) | 8.3k (1.1k) |
| elevators | 13,279 | >13k (>13k) | >13k (>13k) | >13k (>13k) | 8.9k (>13k) | 4.7k (8.7k) |
| pol | 12,000 | >12k (>12k) | >12k (8.5k) | >12k (4.3k) | >12k (2.2k) | >12k (1.0k) |
| protein | 36,584 | 20k (4.6k) | 10k (2.1k) | 5.2k (1.0k) | 2.6k (0.4k) | 1.3k (0.2k) |

- On protein, v3.1 needs 4.4–7.8 times AdaSSP's records. The earlier mechanism needed 1.4–2.1 times
  (protein: 5.4k, 2.8k, 1.4k, 619 for ε = 0.5–4).
- The elevators entries are relative to block normalization's own non-private fit, which is 2.8 times
  worse than standardized. Relative to a common standardized baseline, block-normalized v3.1 never
  comes within 10% on elevators.
- The ridge stays on throughout: Section 9.5 puts n_off for protein in the hundreds of thousands.

### Encrypted spot check

Full OpenFHE protocol in the ridge-on regime: n = 10,000 matched records in a 12,000-row register
(padded to 16,384), ε = 2, δ = 1e-9, n_off = 21.6k. Ten runs in two batches of about 30 s each: the
first is in `encrypted_spot_batch1.json`, the second (with the raw decrypted noise) in
`encrypted_spot.json`.

- λ was 8,077–9,296 in every run, inside the simulated 95% range of 7,659–10,187.
- The Wald interval covered θ_λ for 105 of 110 coefficients.
- Second batch: the decrypted noise standardizes to s.d. 0.98 (stage 2, 372 entries) and 0.98
  (stage 1, 306 entries), with means 0.06 and 0.02 and max |z| of 2.6 and 3.3. A was exact.
- Against the simulated distribution given each run's λ, 109 of 110 coefficients lie within 2.5 s.d.
  One, in the first batch, is at 3.85 s.d.; that batch did not record its raw noise. A larger
  encrypted batch is flagged as a `\tocheck` in the draft.

## Where v3.1 looks worse than the paper currently claims

- **Coefficients.** v3.1 needs about 2.3–2.5 times more matched records for coefficient accuracy than
  the earlier-mechanism numbers in the paper, and much more while the stage-1 ridge is on. That is the
  case up to 0.8–1.2·n_off, between 6k and 81k on the synthetic population.
- **Against AdaSSP.** The current text says the private VFL estimate beats AdaSSP's adaptive ridge on
  27 of 29 datasets under the same guarantee. Under v3.1 it is worse than AdaSSP (matched) on 17–21 of
  29 datasets for every ε in {0.5, …, 8}.
- **Real-data intersection size.** The abstract's "within 10% of non-private at 1.4k–11k matched
  records for ε ∈ {1, 2}" becomes 5.2k–10k on protein. Bike and pol do not get within 10% for ε ≤ 4
  at the 12k–14k records available.
- **Bias correction.** It no longer reduces variance by 7–12%; it is now irrelevant, which supports
  leaving it off.
