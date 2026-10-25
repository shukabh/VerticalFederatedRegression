# Two named settings of protocol v3.1: inference (I) and prediction (P)

The paper evaluates the same protocol under two settings of its statistical parameters. Both were chosen on the
synthetic population only, by the rule pre-registered in `settings_choice.md` (written and committed before any
selection, validation or real-data run; the selection result is appended to it). Every number below is regenerated
from the JSON files in this directory by `experiments/v31_settings/tables_settings.py`, which also writes
`key_numbers.md` (all tables) and `tables_v31s.tex` (LaTeX table bodies). The previous setting's results are in
`reports/v31_eval/` and are not rerun.

## The settings

| | previous (reference) | I (inference) | P (prediction) |
|---|---|---|---|
| stage-1 radius r₁ | union bound | exact quantile | exact quantile |
| r₁/σ₁ at p = 11 | 11.53 | 7.42 | 4.94 |
| f₁ | 0.1 | 0.25 | 0.2 |
| ϑ* | 2.5 | 2.5 | 0.5 |
| κ_Ψ | 2 | 2 | 2 |
| α_L, α_U | 0.0025, 0.0025 | 0.0025, 0.0025 | 0.5, unused |
| intervals | yes | yes | no |
| n_off, simulation population, ε = 1 | 41.7k | 27.0k | 11.3k |

The exact radius is σ₁ times an upper 99% confidence bound on the (1 − α_L) quantile of λ_max of the unit stage-1
noise matrix (symmetric, zero on A, N(0, 1) on triu B and C), from Monte Carlo draws with at least 2,500 expected
exceedances (1e6 draws at α_L = 0.0025), a fixed public seed, and a cache in `implementation/vfr/r1_quantiles.json`
(`vfr/solve.py`, `r1_quantile_exact`). Both settings keep the gate ϑ̂ ≥ 1. The interval guarantee holds only under I.
In code: `dp.Agreement(..., **dp.setting('I'))`, or `run_protocol.py --setting P`.

## Files

| file | content |
|---|---|
| `settings_choice.md` | the pre-registered rule, and the selection result appended after the run |
| `selection.json`, `settings.json` | criteria for all 44 candidates (R² = 0.3, 0.5, 0.9), n_off, per-ε curves; the chosen values |
| `sim_I.json`, `sim_P.json` | 13.5 under I and P: per (R², n, ε) RMSE, prediction MSE, ridge share, λ, gate; coverage at R² = 0.5 (I); per-coefficient efficiency ratios; crossings; the switch-off check at 0.6–1.4·n_off |
| `bias_check_I.json` | 13.6 under I: conditional bias study, 45 cells, 5,000 antithetic pairs per (dataset, λ) |
| `adassp_benchmark.json` | 13.7: 29 datasets × 5 ε × 10 folds; arms (a), (b), (c) I/P, (d) I/P, (e) I; fold-level test MSE; paired tests against each other and against the previous setting's fold values |
| `sweep_n.json` | 13.8: bike, elevators, pol, protein; (d) under I and P, AdaSSP; ratio to own and to a common non-private fit against n; crossings; gate refusals |
| `encrypted_spot.json` | full OpenFHE runs, 5 under I (ε = 2) and 5 under P (ε = 1), n = 10,000 in a 16,384-row register, with the raw decrypted noise |
| `key_numbers.md`, `tables_v31s.tex` | generated tables |

Figures: `paper/figures/{sim_rmse_vs_n,sim_pred_vs_n,sim_efficiency_ratio,adassp_mse_vs_eps,sweep_n}_v31s.png`.

Reproduce (4 cores; about 2 hours, most of it the Monte Carlo radius for gas, p = 129):

```bash
python experiments/v31_settings/select_settings.py run && python experiments/v31_settings/select_settings.py choose
python experiments/v31_settings/sim_settings.py run --setting I && python experiments/v31_settings/sim_settings.py summarize --setting I
python experiments/v31_settings/sim_settings.py run --setting P && python experiments/v31_settings/sim_settings.py summarize --setting P
python experiments/v31_settings/bias_check.py
python experiments/v31_settings/real_settings.py benchmark --delta 1e-6    # WANG_REPO must point at Wang's repository
python experiments/v31_settings/real_settings.py sweep --delta 1e-9
python experiments/v31_settings/encrypted_settings.py                      # OpenFHE, about 30 s per run
python experiments/v31_settings/figures_settings.py && python experiments/v31_settings/tables_settings.py
```

Raw per-replicate arrays go to `experiments/_work/v31_settings/` (git-ignored).

## Key numbers

### Selection (synthetic population only)

- I: f₁ = 0.25 minimizes the mean log ratio of slope RMSE to least squares (1.6719; flat over 0.2–0.3; same choice
  at R² = 0.3 and 0.9; previous setting 1.8006).
- P: (ϑ*, α_L, f₁) = (0.5, 0.5, 0.2) minimizes the mean ratio of test MSE to least squares (1.3017; previous setting
  1.4045, I 1.3538). It lies on the grid edge in ϑ* and α_L; the grid was not extended, as pre-registered. Its gate
  refused no release on the population.

### 13.5 Synthetic population (δ = 1e-9, R² = 0.5, 200 cohorts per n; same cohorts and unit noise as the previous run)

- **Coverage under I.** Wald interval for θ_λ: 0.945–0.966 in all 70 (n, ε) cells (previous 0.945–0.965; target
  0.955). Bias-aware interval for β with the ridge off in every release (27 cells): true β 0.930–0.964,
  clipped-population β 0.946–0.962. No regime where coverage fails. The gate refused no release (I or P, any R²).
- **Switch-off.** n_off under I = 53k, 27k, 14k, 7.3k, 3.9k for ε = 0.5, …, 8 (previous 81k, 42k, 22k, 11k,
  6.0k). Share of releases with λ = 0: 3–4% at n_off, 95–99.5% at 1.2·n_off (previous 3–5% and 94–99%). Under P
  the switch-off is late: 0% at n_off, 4–10% at 1.2·n_off, 56–70% at 1.4·n_off.
- **Validity ratio held by the ridge:** median 5.0–5.2 under I (previous 7.9–9.2), 2.9–3.0 under P.
- **Coefficients (I vs previous).** Better with the ridge on (n = 1e4: 0.197, 0.154, 0.079, 0.023 against 0.213,
  0.184, 0.131, 0.043 for ε = 0.5–4). Slightly worse once it is off (σ₂ up 9.5%): twice the floor at 8.0k, 27k, 88k
  (ε = 8, 4, 2) against 7.3k, 23k, 79k; ratio at 1e5 3.2 and 5.8 at ε = 1, 0.5 against 2.9 and 5.3.
- **Prediction within 10% of non-private:** P 23k, 12k, 6.2k, 3.6k, 2.0k; I 38k, 20k, 10k, 5.3k, 2.9k; previous
  58k, 30k, 15k, 8.0k, 4.2k (ε = 0.5, …, 8).

### 13.6 Bias under I

Squared mechanism bias at most 0.19% of the MSE around θ_λ (previous 0.048%); largest |bias_j|/sd_j 0.078
(previous 0.044); the correction lowers the MSE against θ_λ by 1.4–1.7% with the ridge on; Theorem 10.2 holds
(0.994–1.017); K ≥ 490 releases (previous 2,100). Still negligible.

### 13.7 AdaSSP benchmark (δ = 1e-6, 29 datasets, 10 folds, 50 draws)

Verdicts are datasets where the v3.1 arm has significantly lower / higher absolute test MSE (paired over folds);
gaps are the median over the 29 datasets of the absolute-MSE ratio minus 1, against the same AdaSSP draws. The
previous setting's verdicts are from its earlier run (`reports/v31_eval/`).

| arm | vs | ε = 1 verdict | ε = 1 gap | ε = 8 verdict | ε = 8 gap |
|---|---|---|---|---|---|
| (d) block norm, I | (a) AdaSSP published | 5 / 20 | +25% | 1 / 21 | +11% |
| (d) block norm, P | (a) | 5 / 20 | +36% | 2 / 22 | +81% |
| (d) block norm, previous | (a) | 5 / 19 | +28% | 1 / 21 | +12% |
| (c) joint norm, I | (a) | 5 / 23 | +19% | 0 / 22 | +11% |
| (c) joint norm, P | (a) | 4 / 22 | +8.5% | 0 / 26 | +7.3% |
| (c) joint norm, previous | (a) | 6 / 20 | +28% | 0 / 23 | +13% |
| (c) joint norm, I | (b) AdaSSP matched | 6 / 18 | +12% | 2 / 20 | +6.3% |
| (c) joint norm, P | (b) | 6 / 11 | +0.8% | 3 / 12 | +1.1% |
| (c) joint norm, previous | (b) | 8 / 18 | +17% | 3 / 20 | +8.2% |

- **Setting I** improves on the previous setting on 18–24 datasets (worse on at most 4) at every ε, by a median of
  1–3%, but the verdict against AdaSSP is unchanged. Its ridge is 2.8× AdaSSP (matched) (previous 4.2×), 4.1×
  AdaSSP (published). The gate refused nothing.
- **Setting P with joint normalization** (idealized: no party can normalize a whole row) matches AdaSSP (matched)
  to within about 1% in median test MSE at every ε; its ridge is 1.3× AdaSSP (matched); gate refusals 1.8% on
  average. It beats (c) under I on 20–21 datasets at every ε.
- **Setting P with block normalization** fails: the gate refused 65% of releases on average over the 29 datasets
  and all ε (at least a quarter on 25 datasets, all of them on elevators, gas and solar); it passes only on 3droad,
  kin40k and tamielectric (ridge off) and pol. It is worse than (d) under I on 16–20 datasets.
- (e) standardization under I is worse than (d) under I on 18–25 datasets. AdaSSP was rerun with new draws; its
  test MSE changed by a median of 0.0% (at most 4.7%) from the earlier run.

### 13.8 Real-data sweep (δ = 1e-9): records to come within 10% of own non-private

| dataset | n available | ε = 0.5 | ε = 1 | ε = 2 | ε = 4 | ε = 8 |
|---|---|---|---|---|---|---|
| bike, I / P / previous (AdaSSP) | 13,903 | >14k / >14k / >14k (>14k) | >14k / >14k / >14k (9.1k) | >14k / >14k / >14k (4.5k) | 13.6k / >14k / >14k (2.3k) | 7.3k / >14k / 8.3k (1.2k) |
| elevators | 13,279 | >13k / >13k / >13k (>13k) | >13k / >13k / >13k (>13k) | >13k / >13k / >13k (>13k) | 8.0k / >13k / 8.9k (>13k) | 4.1k / >13k / 4.7k (8.8k) |
| pol | 12,000 | >12k / >12k / >12k (>12k) | >12k / >12k / >12k (8.5k) | >12k / >12k / >12k (4.3k) | >12k / 12.0k / >12k (2.2k) | >12k / 6.4k / >12k (1.0k) |
| protein | 36,584 | 16k / >37k / 20k (4.5k) | 8.2k / >37k / 10k (2.2k) | 4.1k / >37k / 5.2k (992) | 2.1k / >37k / 2.6k (427) | 985 / 32k / 1.3k (166) |

**Setting P fails on block-normalized real data.** While the ridge is on, its gate refuses 96–100% of releases on
bike, 100% on elevators and 82–97% on protein (at most 2% on pol, where P beats I). A refused release outputs θ = 0,
so P's test MSE stays near that of predicting the mean (3.9, 1.6 and 1.34 times non-private at the largest n). The
cause: with ϑ* = 0.5 the gate needs r₁ ≳ σ₂√p plus the extra downward shift of λ_min from the larger stage-2 noise.
On the synthetic population (Δ_G ≈ 0.9Δ) P has σ₁ = 1.79σ₂ and r₁ = 8.9σ₂; under block normalization Δ_G/Δ ≈ 0.34
(protein), so σ₁ ≈ 0.67σ₂ and r₁ ≈ 3.1σ₂ < σ₂√10. The selection population never entered this regime.

### Encrypted spot check (OpenFHE, n = 10,000, register 12,000 rows padded to 16,384)

- I at ε = 2: λ 3,162–3,729 in 5/5 runs, inside the simulated 95% range 2,578–4,232; θ inside the simulated 95%
  range for 53/55 coefficients, within 2.5 s.d. of the simulation given λ for 54/55 (max 2.91); Wald coverage 54/55.
- P at ε = 1: λ 3,890–4,897 in 5/5 runs, inside 2,197–5,543; 53/55 and 54/55 (max 2.58); no refusals.
- Raw decrypted noise, 510 stage-1 and 620 stage-2 entries: pooled s.d. 1.00, mean 0.00 (KS p = 0.79); s.d.
  0.96–1.08 per setting and stage; max |z| 3.23; A exact.
