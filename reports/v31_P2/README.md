# P2: the re-selected prediction setting of protocol v3.1

P2 replaces setting P. P was chosen on the standardized synthetic population only, and on block-normalized real data its
gate refused most releases (`reports/v31_settings`). P2 was chosen by a new rule, pre-registered in
`settings_choice_P2.md` (commit 0d7e7a0, before any selection run; the result is appended there). The rule minimizes the
mean test-MSE ratio to non-private least squares over a standardized and a block-normalized synthetic population, subject
to a gate-refusal rate of at most 1% in every cell. The protocol is unchanged: the gate stays at ϑ̂ ≥ 1. In the paper
draft (`paper/drafts/p2.tex`) P2 is called **setting P**, and the old P is called **P₀**.

Every number below is regenerated from the JSON files in this directory by `experiments/v31_P2/tables_P2.py`, which writes
`key_numbers.md` (all tables). Setting I, the old P, AdaSSP and the previous setting were not rerun. Their values come from
`reports/v31_settings` and `reports/v31_eval`. Where P2 shares their noise streams, recomputing I on the same path
reproduces the published values to 1e-14.

## The setting

| | I (inference) | **P2 (prediction)** | old P (P₀, replaced) |
|---|---|---|---|
| radius r₁ | exact quantile | exact quantile | exact quantile |
| r₁/σ₁ at p = 11 | 7.42 | **4.44** | 4.94 |
| f₁ | 0.25 | **0.25** | 0.2 |
| ϑ* | 2.5 | **1.5** | 0.5 |
| κ_Ψ | 2 | 2 | 2 |
| α_L, α_U | 0.0025, 0.0025 | **0.75**, unused | 0.5, unused |
| intervals | yes | no | no |
| σ₁, σ₂ (simulation population, ε = 1) | 892, 575 | 892, 575 | 998, 557 |
| n_off (simulation population, ε = 1) | 27.0k | 16.2k | 11.3k |

Preset line for `implementation/vfr/dp.py` (`SETTINGS`), to apply:

```python
    'P': dict(f1=0.25, theta_star=1.5, kappa_Psi=2.0, alpha=0.05, alpha_L=0.75, alpha_U=0.0025, r1_mode='exact',
              intervals=False),
```

The radius cache `implementation/vfr/r1_quantiles.json` now holds α_L = 0.75 (and 0.05) for every (d_A, d_O) of the 29
datasets and of the simulation population.

## Files

| file | content |
|---|---|
| `settings_choice_P2.md` | the pre-registered rule, the population check, and the appended selection result |
| `selection_P2.json`, `setting_P2.json` | criteria for all 140 candidates (grid and the one-time α_L extension), robustness at R² = 0.3 and 0.9, n_off, per-ε curves; the chosen values |
| `sim_P2.json` | validation on the 13.5 cohorts and noise: P2, I and the old P on both populations, R² ∈ {0.3, 0.5, 0.9}; crossings, refusals, the validity ratio held, the switch-off of P2 at 0.6–1.4·n_off |
| `adassp_benchmark_P2.json` | 13.7: arms (c) and (d) under P2 (fold-level test MSE, refusals, λ, σ₁/σ₂). The published arms are read in, and paired verdicts and gaps are computed against them |
| `sweep_n_P2.json` | 13.8: (d) under P2 on bike, elevators, pol and protein, with the published I, old P, previous and AdaSSP curves |
| `encrypted_P2.json` | 5 full OpenFHE runs of P2 at ε = 1 (n = 10,000; 12,000-row register padded to 16,384), with the raw decrypted noise |
| `key_numbers.md` | generated tables |

Figures: `paper/figures/{sim_pred_vs_n,adassp_mse_vs_eps,sweep_n}_v31p2.png`. Paper draft: `paper/drafts/p2.tex`. It
has 39 replacements, each quoting its exact old text. Apply it with `experiments/v31_P2/apply_p2_draft.py`.

Reproduce (4 cores, about 45 minutes; the radii at α_L = 0.75 for gas take a few minutes):

```bash
python experiments/v31_P2/select_P2.py run && python experiments/v31_P2/select_P2.py choose
python experiments/v31_P2/select_P2.py run --ext && python experiments/v31_P2/select_P2.py choose   # edge rule
python experiments/v31_P2/sim_P2.py run && python experiments/v31_P2/sim_P2.py summarize
python experiments/v31_P2/real_P2.py benchmark --delta 1e-6      # WANG_REPO must point at Wang's repository
python experiments/v31_P2/real_P2.py sweep --delta 1e-9
python experiments/v31_P2/encrypted_P2.py --reps 5 --eps 1       # OpenFHE, about 30 s per run
python experiments/v31_P2/figures_P2.py && python experiments/v31_P2/tables_P2.py
```

Raw per-cell files go to `experiments/_work/v31_P2/` (git-ignored).

## Key numbers

### Selection (synthetic only; R² = 0.5 decides; 140 cells × 200 cohorts)

- **Populations.** (a) Standardized, as in 13.5: Δ = 86.2, Δ_G = 77.2, σ₁/σ₂ = 1.79 at f₁ = 0.2 and 1.55 at 0.25.
  (b) Block-normalized: B_R² = 2, B_O = 1, B_y = 2.89, Δ = 10.2, Δ_G = 2.83, y'y released separately, so σ₁/σ₂ =
  0.53 at f₁ = 0.2 and 0.46 at 0.25.
  - Under block normalization the real data range from 0.29 (solar) to 0.96 (pol) at f₁ = 0.2, median 0.65; protein is
    0.67. So (b) is harsher than protein, and 8 of the 29 datasets lie below it.
- **Grid stage (112 candidates).** All 140 cells are binding and 56 candidates are feasible. The best is (ϑ*, α_L, f₁) =
  (1.5, 0.5, 0.25), tied within 0.1% with (1.25, 0.05, 0.2); the tie goes to the larger ϑ*. It lies on the α_L edge, so
  α_L = 0.75 was added once, as pre-registered.
- **Final (140 candidates, 68 feasible): P2 = (1.5, 0.75, 0.25).**
  - C = 1.2409 (std 1.3186, bn 1.1632); I gives 1.2744, the old P 1.2861 and the previous setting 1.3058.
  - P2's largest refusal in any cell is 0 on (a) and 1.0% on (b), which is 2 of 200 and exactly at the bound.
  - **P2 lies on the edge of the extended grid in α_L. As pre-registered, it is kept.**
- **The constraint binds.** Six ϑ* = 1.25 candidates are up to 0.28% better, but refuse 2–15% in 7–27 cells of (b).
  - No ϑ* = 0.5 candidate is feasible: each refuses 9–100% in some cell of (b).
  - The old P refuses 37% of (b) releases on average.
- **Robustness (not used).** At R² = 0.3 and 0.9, P2 is first by C on both populations and is feasible.
  - At R² = 0.3 the tie rule would pick (1.5, 0.5, 0.25) instead, which is within 0.1%.

### Synthetic validation (13.5 cohorts and noise, δ = 1e-9, R² = 0.5)

| records for test MSE within 10% of non-private | ε = 0.5 | 1 | 2 | 4 | 8 |
|---|---|---|---|---|---|
| (a) standardized: P2 | 26k | 13.7k | 7.1k | 4.0k | 2.1k |
| (a) I | 38k | 20k | 10.0k | 5.3k | 2.9k |
| (a) old P | 23k | 12.0k | 6.2k | 3.6k | 2.0k |
| (b) block-normalized: P2 | 9.4k | 4.8k | 2.5k | 1.5k | 729 |
| (b) I | 14.7k | 7.5k | 4.0k | 2.1k | 1.1k |
| (b) old P (gate refuses 98–100% while its ridge is on) | 14.2k | 7.2k | 4.5k | 2.2k | 984 |

- **Records needed.** P2 needs 1.3–1.5× fewer records than I on (a) and 1.4–1.6× fewer on (b).
  - On (a) the old P needed 5–13% fewer than P2.
  - At R² = 0.9 (a), P2 needs 1.04–1.20× fewer records than I; at R² = 0.3, 1.3–1.4× fewer.
- **Gate refusals of P2.** None on (a) at any R². On (b), at most 1.0% in any cell and 0.06% on average, at every R².
- **Switch-off of P2 on (a).** The share of releases with the ridge off is 0–1% at n_off, 46–63% at 1.2·n_off and
  98–99% at 1.4·n_off.
  - At 0.8·n_off, λ is 2.0–2.3× the linear prediction.
  - n_off = 31.5k, 16.2k, 8.4k, 4.4k and 2.3k for ε = 0.5 to 8. In closed form, n_off = (11.5Δ + 8.9Δ_G)/(ℓ√(2ρ_run)).
- **Validity ratio held by the ridge.** P2 holds a median of 3.4–3.5 on (a) and 2.0 on (b). I holds 5.0–5.2 and 3.2;
  the old P held 2.9–3.0 and 1.1–1.2.

### 13.7 AdaSSP benchmark (δ = 1e-6, 29 datasets, 10 folds, 50 draws)

Verdict: significantly lower / higher absolute test MSE, paired over folds. Gap: median over the datasets of the
absolute-MSE ratio − 1.

| arm | vs | ε = 1 verdict | ε = 1 gap | ε = 8 verdict | ε = 8 gap |
|---|---|---|---|---|---|
| (d) block norm, **P2** | (a) AdaSSP published | 5 / 20 | **+13.6%** | 3 / 15 | **+4.3%** |
| (d) block norm, I | (a) | 5 / 20 | +25.1% | 1 / 21 | +10.7% |
| (d) block norm, old P | (a) | 5 / 20 | +35.7% | 2 / 22 | +81.2% |
| (d) block norm, **P2** | (b) AdaSSP matched | 5 / 19 | +6.3% | 9 / 15 | +1.4% |
| (c) joint norm, **P2** | (a) | 5 / 22 | +8.8% | 0 / 25 | +7.8% |
| (c) joint norm, I | (a) | 5 / 23 | +18.9% | 0 / 22 | +11.3% |
| (c) joint norm, old P | (a) | 4 / 22 | +8.5% | 0 / 26 | +7.3% |
| (c) joint norm, **P2** | (b) | 6 / 18 | +4.9% | 1 / 18 | +2.6% |
| (c) joint norm, I | (b) | 6 / 18 | +11.6% | 2 / 20 | +6.3% |
| (c) joint norm, old P | (b) | 6 / 11 | +0.8% | 3 / 12 | +1.1% |

- **Against I, under block normalization:** P2 is better on 18–24 datasets and worse on at most 2, at every ε, by a
  median of 3–7%. Its ridge is 0.6× I's.
- **Against the old P:** better on 18–20 datasets under block normalization. Worse on 18–20 under joint normalization,
  by a median of 0.3–2.9%.
- **Ridge under joint normalization:** 1.6–1.85× AdaSSP (matched); I 2.6–2.9×, the old P 1.1–1.4×.
- **Gate refusals of P2, (d) block normalization:**
  - 0.21% on average over the 29 datasets and all ε, against 65% for the old P.
  - More than 1% on at most 3 datasets at any ε; none at 25% or more.
  - Worst: solar, 7.2% at ε = 1, where σ₁/σ₂ = 0.25; then servo, 1.8% at ε = 8.
- **Gate refusals of P2, (c) joint normalization:** 0.03% on average; at most 2.4% (servo, ε = 8).

### 13.8 Real-data sweep (δ = 1e-9, block normalization): records to come within 10% of own non-private

| dataset (n available) | arm | ε = 0.5 | ε = 1 | ε = 2 | ε = 4 | ε = 8 |
|---|---|---|---|---|---|---|
| bike (13,903) | **P2** | >14k | >14k | >14k | **8.6k** | **4.5k** |
| | I / old P / previous | >14k / >14k / >14k | >14k / >14k / >14k | >14k / >14k / >14k | 13.6k / >14k / >14k | 7.3k / >14k / 8.3k |
| | AdaSSP | >14k | 9.1k | 4.5k | 2.3k | 1.2k |
| elevators (13,279) | **P2** | >13k | >13k | **9.5k** | **5.1k** | **2.5k** |
| | I / old P / previous | >13k / >13k / >13k | >13k / >13k / >13k | >13k / >13k / >13k | 8.0k / >13k / 8.9k | 4.1k / >13k / 4.7k |
| | AdaSSP | >13k | >13k | >13k | >13k | 8.8k |
| pol (12,000) | **P2** | >12k | >12k | >12k | >12k | **10.5k** |
| | I / old P / previous | >12k / >12k / >12k | >12k / >12k / >12k | >12k / >12k / >12k | >12k / 12.0k / >12k | >12k / 6.4k / >12k |
| | AdaSSP | >12k | 8.5k | 4.3k | 2.2k | 1.0k |
| protein (36,584) | **P2** | **10k** | **5.0k** | **2.5k** | **1.2k** | **599** |
| | I / old P / previous | 16k / >37k / 20k | 8.2k / >37k / 10k | 4.1k / >37k / 5.2k | 2.1k / >37k / 2.6k | 985 / 32k / 1.3k |
| | AdaSSP | 4.5k | 2.2k | 992 | 427 | 166 |

- **Gate refusals of P2:** at most 0.3% on bike and 1.0% on elevators at any n and ε; none on pol or protein. The old P
  refused 82–100% on bike, elevators and protein.
- **Against I:** P2 needs 1.6–1.7× fewer records on protein. AdaSSP still needs 2.2–3.6× fewer records than P2 there.

### Encrypted spot check (OpenFHE, P2 at ε = 1)

- **Setup.** n = 10,000 matched records in a register of 12,000 rows padded to 16,384. The matched design's n_off is 16.4k,
  so the ridge is on. r₁ = 3,964 from the cached exact quantile.
- **λ.** 4,924–6,708 in 5/5 runs, inside the simulated 95% range of 4,837–7,855. No release was refused (the simulated
  refusal rate is 0).
- **θ.** 53/55 coefficients lie inside the simulated 95% range, and 54/55 within 2.5 s.d. of the simulation given λ
  (largest 2.59).
- **Raw decrypted noise.** 255 stage-1 and 310 stage-2 entries. Standardized, they have s.d. 0.954 and 0.975, and mean
  −0.10 overall (2.5 standard errors), with KS p = 0.27 and 0.09 and max |z| 3.07. A is exact.
  - Pooled with the 10 earlier runs of I and the old P (1,695 entries): mean −0.03, s.d. 0.99, KS p = 0.46.

## Caveats

- **Grid edge.** P2 lies on the edge of the extended grid in α_L (0.75, so r₁ is the 25% quantile of λ_max). The rule
  allowed only one extension, so it is kept.
- **The refusal bound is met exactly, and the constraint binds.** P2 refuses 2/200 releases in one block-normalized
  cell. ϑ* = 1.25 candidates would be up to 0.3% better on the populations but fail the bound.
- **The noise-ratio range does not cover all datasets.** Population (b) gives σ₁/σ₂ = 0.53 at f₁ = 0.2, harsher than the
  0.67 of protein. The selection still does not cover the 8 datasets below it: solar refuses up to 7.2% under P2.
- **P2 does not close the gap to AdaSSP.** Under block normalization it is worse than AdaSSP (published) on 20 of 29
  datasets at ε = 1 (19 against matched), though its gaps are about half those of I.
- **The old P is better where its gate passed.** That is under joint normalization, on the standardized population
  (5–13% fewer records), and on pol.
- **Noise mean in the P2 encrypted runs.** The five runs have a mean of −0.10 (2.5 standard errors). It is consistent
  with chance when pooled with the earlier runs, but the encrypted checks are still small (5 runs per setting).
