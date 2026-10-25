# Re-selection of the prediction setting (P2): pre-registered rule

Written on 2026-10-25, before any selection run for P2, any validation run of P2, and any real-data run of P2. The
rule below decides P2. The selection result will be appended at the end of this file without changing the rule.
Scripts: `experiments/v31_P2/` (selection: `select_P2.py`).

## Why

Setting P (r₁ exact, ϑ* = 0.5, α_L = 0.5, f₁ = 0.2, no intervals) was chosen on the synthetic population under
standardization only (`reports/v31_settings/settings_choice.md`). On block-normalized real data its gate refused
82–100% of releases on protein, bike and elevators, and 65% on average over Wang's 29 datasets. With ϑ* < 1 the
gate passes only if r₁ is large relative to σ₂√p. The selection population had σ₁ = 1.79σ₂; block normalization
gives Δ_G/Δ ≈ 0.34 and σ₁ ≈ 0.67σ₂ on protein. P2 replaces P. Setting I is unchanged.

## What does not change

- The protocol: two releases; the stage-1 rule (λ_I, λ_O, κ_Ψ, r₁, g* = 2ϑ*σ₂√p); the stage-2 gate ϑ̂ ≥ 1, where a
  refused release outputs θ = 0 and still spends the budget; the solve; bias correction off.
- Calibration by the implementation (`vfr/dp.py` `calibrate`), as in `reports/v31_settings`: κ = 40, 40, 40, 44, 48
  for ε = 0.5, 1, 2, 4, 8; y'y in stage 2 when it is free, otherwise its own release with f_y = 0.1 (the
  implementation releases it whether or not intervals are on).
- P2 is a prediction setting: r₁ is the exact radius (σ₁ times the upper 99% confidence bound on the (1 − α_L)
  quantile of λ_max of the unit stage-1 noise, `vfr/solve.py` `r1_quantile_exact`), κ_Ψ = 2, intervals off (α_L
  only sizes r₁). The privacy guarantee does not depend on the setting: the noise follows from (ε, δ, f₁) alone.
- P2's parameters are passed to the scripts directly. The `SETTINGS` presets in `vfr/dp.py` are not read for P2.

## Selection populations

Both are built from the simulation population `vfr/data.population()` (`experiments/sim_core.Population`, seed 0,
p = 11, d_R = 4 plus R's intercept, d_O = 6). Every cohort is one draw of features and outcome, preprocessed both
ways (the same records in both populations).

**(a) Standardized ("std").** Public standardization constants and the population's committed 99.5% bounds, as in
Section 13.5: B_R = 5.591 (intercept included), B_O = 6.242. Exactly `Population.draw`.

**(b) Block-normalized ("bn").** Each party scales its own standardized feature block of every row to unit norm:
R's 4 features and O's 6 features separately. So B_R² = 2 with R's intercept (B_R = 1.414) and B_O = 1, exactly.
The outcome is the same standardized y in both populations; B_y is the population's 99.5% bound on |y|, the same
value in (a) and (b).

Sensitivity and noise ratio at the decisive R² = 0.5 (δ = 1e-9; the ratio is the same at every ε up to the scale
search's rounding):

| | B_R | B_O | B_y | Δ | Δ without y'y | Δ_G | Δ_G/Δ | y'y | ℓ = λ_min(Σ) |
|---|---|---|---|---|---|---|---|---|---|
| (a) std | 5.591 | 6.242 | 2.888 | 86.18 | 86.18 | 77.20 | 0.896 | free (in stage 2) | 0.599 |
| (b) bn | 1.414 | 1 | 2.888 | 10.17 | 10.00 | 2.83 | 0.278 | own release, f_y = 0.1 | 0.105 |

σ₁/σ₂ by f₁:

| f₁ | 0.05 | 0.1 | 0.2 | 0.25 | 0.35 | 0.45 | 0.55 |
|---|---|---|---|---|---|---|---|
| (a) std | 3.91 | 2.69 | 1.79 | 1.55 | 1.22 | 0.99 | 0.81 |
| (b) bn | 1.17 | 0.80 | 0.53 | 0.46 | 0.35 | 0.28 | 0.23 |

**Check against real data.** At f₁ = 0.2, (b) gives σ₁/σ₂ = 0.53. Protein under block normalization gives 0.67
(Δ_G/Δ = 0.333 with y'y free). (b) is lower because its B_y = 2.89 makes y'y not free, so stage 2 is calibrated to
Δ without y'y with the share 1 − f₁ − f_y. Over Wang's 29 datasets the ratio at f₁ = 0.2 under block normalization
ranges from 0.29 (solar) to 0.96 (pol), median 0.65. Eight datasets lie below 0.53: solar 0.29, elevators 0.42,
forest 0.42, 3droad 0.45, gas 0.50, housing 0.51, skillcraft 0.51 and stock 0.52. pumadyn32nm (0.535) and kin40k
(0.540) are just above. These ratios come from the bounds and calibration already recorded in
`reports/v31_settings/adassp_benchmark.json` (B_y from the public reference rows). No outcome or error of any real
dataset was read, and the rule does not use them. (a) and (b) together span 0.53–1.79 at f₁ = 0.2, which brackets
protein and the median dataset. (b) is on the harsh side of 0.67, so the constraint below is, if anything, stricter
than protein needs. The eight datasets below 0.53 lie outside the span, solar furthest. We keep (b) as defined.

**Robustness only (not used for the choice):** both populations at R² = 0.3 and R² = 0.9 (B_y = 2.837 and 3.072;
Δ_G/Δ under (b) 0.286 and 0.252).

Test MSE is the population's: the design as each population preprocesses it, with unclipped y. The moments come
from 4 million rows: for (a) the moments of Section 13.5 (`experiments/v31_eval/sim_accuracy.py` `pop_moments`); for
(b) the same computation with its own seed (`experiments/v31_P2/p2common.py` `pop_moments`). The release and the
non-private fit use clipped y.

## Design

- δ = 1e-9, ε ∈ {0.5, 1, 2, 4, 8}.
- n ∈ {100, 250, 500, 1k, 2.5k, 5k, 7.5k, 10k, 15k, 25k, 35k, 50k, 75k, 100k} (the grid of 13.5), **200 cohorts
  per n**.
- Seeds: `SeedSequence([20261024, 5250, round(100 R²), n])`, spawned into one data stream and one noise stream per
  ε. These are independent of the 13.5 validation seeds and of the first selection (tag 5150).
- Common random numbers: per (R², n, ε), one set of unit stage-1 and stage-2 noise draws, scaled by each candidate's
  σ₁ and σ₂, and shared by both populations. Candidates differ only through their parameters.
- Every candidate runs the stage-1 rule, the gate ϑ̂ ≥ 1, and the solve. A refused release gives θ = 0.
- A **selection cell** is (population, ε, n) at R² = 0.5: 2 × 5 × 14 = 140 cells.
- Not read for the choice: any of Wang's 29 UCI datasets, protein, or any other real data.

## Grid

All candidates use the exact r₁, κ_Ψ = 2, and intervals off.

- ϑ* ∈ {0.5, 0.75, 1, 1.25, 1.5, 2, 2.5}
- α_L ∈ {0.0025, 0.05, 0.25, 0.5}
- f₁ ∈ {0.1, 0.2, 0.25, 0.35}

That is 7 × 4 × 4 = 112 candidates. Setting I, the old P and the previous setting are computed on the same cells
for reference. They are not candidates.

## Rule

For candidate c and cell k, let **ρ(c, k)** be the share of the 200 releases that the gate refuses. Let **m(c, k)**
be the mean population test MSE of the 200 releases (θ = 0 when refused) divided by the mean population test MSE of
non-private least squares on the same 200 cohorts.

1. **Constraint.** A cell k is *binding* if at least one candidate in the grid has ρ(c, k) ≤ 0.01. Candidate c is
   *feasible* if ρ(c, k) ≤ 0.01 in every binding cell (at most 2 refusals out of 200). If no candidate is feasible,
   the candidates violating the fewest binding cells are treated as feasible.
2. **Objective.** C(c) = mean of m(c, k) over the 140 cells, with both populations weighted equally.
3. **Choice.** P2 is the feasible candidate with the smallest C.
4. **Ties.** Feasible candidates with C ≤ 1.001 × C_min (within 0.1% of the best) are tied. Among tied candidates,
   choose the larger ϑ*, then the smaller α_L (the larger radius), then the f₁ closest to 0.25, then the smaller
   f₁.
5. **Grid edges: extend once.** If the chosen candidate lies on an edge of the grid in a parameter, the grid is
   extended once beyond that edge:
   - ϑ* = 0.5 → add 0.25; ϑ* = 2.5 → add 3 and 3.5;
   - α_L = 0.0025 → add 0.0005; α_L = 0.5 → add 0.75;
   - f₁ = 0.1 → add 0.05; f₁ = 0.35 → add 0.45 and 0.55.

   The new values are crossed with all values of the other parameters, including any other axis extended in the
   same step. Steps 1–4 are then applied again on the union: binding cells, feasibility, objective and ties are all
   recomputed over the union, on the same cells, cohorts and noise. There is no second extension. If the new choice
   lies on an edge of the extended grid, it is kept and reported as being on that edge.

**Reported for every candidate:** C, C by population, the largest and the mean ρ over the cells by population, the
number of binding cells violated, and the share of releases with the ridge on. The same quantities are reported at
R² = 0.3 and 0.9, both populations, and whether the choice would change there. These do not decide.

## What happens next, fixed now

The setting is not changed after any of the following results, whatever they show.

1. **Synthetic validation (13.5 design).** P2, I and the old P on both populations; R² ∈ {0.3, 0.5, 0.9}; the n
   grid above; 200 cohorts per n. The seeds and unit noise are those of `experiments/v31_settings/sim_settings.py`
   (the 13.5 cohorts), so on (a) P2 is paired with the published I and P runs. Reported: test-MSE ratio to non-private
   least squares, the n from which it stays within 10% (and 1%), gate refusals, ridge share, and the stage-1
   switch-off of P2 at 0.6–1.4·n_off on (a).
2. **AdaSSP benchmark (13.7).** The same 29 datasets, folds, 10% public reference rows, δ = 1e-6, and 50 draws per
   fold as `reports/v31_settings`. Arms (c) joint normalization and (d) block normalization are rerun under P2, with
   the unit-noise streams of the earlier run (paired with I and the old P). (a) AdaSSP published, (b) AdaSSP
   matched, and I and the old P under (c) and (d) are taken from `reports/v31_settings/adassp_benchmark.json` and
   not rerun. Reported: paired verdicts, median gaps, gate refusals, and ridge ratios.
3. **Real-data sweep (13.8).** bike, elevators, pol, protein; δ = 1e-9; (d) under P2 with the earlier seeds.
   Reported: records needed to come within 10% of own non-private, next to I, the old P, the previous setting and
   AdaSSP (from `reports/v31_settings/sweep_n.json`), and gate refusals.
4. **Encrypted spot check.** 5 full OpenFHE runs of P2 at ε = 1. The data are the synthetic dataset of the earlier
   check (n = 10,000 matched in a 12,000-row register padded to 16,384; never more than 16,384 rows). Recorded: the
   raw decrypted noise. Checked: λ and θ against the plaintext simulation's 95% range on the same data.

## Result of the selection (appended after the run; the rule above is unchanged)

Run: `select_P2.py run` (140 cells × 200 cohorts at R² = 0.5, plus R² = 0.3 and 0.9), then `choose`. The grid choice
lay on an edge, so `run --ext` and `choose` again, as the rule requires. Full output: `selection_P2.json`. The chosen
values: `setting_P2.json`. Exact radius at p = 11 (d_A = 5, d_O = 6), as r₁/σ₁: 7.415 at α_L = 0.0025, 6.303 at 0.05,
5.468 at 0.25, 4.940 at 0.5, and 4.442 at 0.75. These are upper 99% bounds, from 1e6 draws at 0.0025 and 2e5 draws
otherwise.

**Grid (112 candidates).** Every one of the 140 cells is binding: setting ϑ* ≥ 2 refuses nothing anywhere. 56
candidates are feasible. The best feasible is (ϑ*, α_L, f₁) = (1.5, 0.5, 0.25) with C = 1.2426. (1.25, 0.05, 0.2), at
C = 1.2437, is within 0.1%, and the tie goes to the larger ϑ*. The choice lies on the upper edge in α_L, so α_L =
0.75 was added, crossed with all ϑ* and f₁ (28 candidates).

**Extended grid (140 candidates; final).** 68 are feasible.

**P2 = (ϑ* = 1.5, α_L = 0.75, f₁ = 0.25)**, with r₁ exact (4.442σ₁ at p = 11), κ_Ψ = 2, intervals off and the gate
kept. C = 1.2409 (std 1.3186, bn 1.1632). Its largest refusal rate in any cell is 0 on (a) and 0.010 on (b), which is
2 of 200 and exactly at the bound; the mean over the (b) cells is 0.0006. (1.5, 0.75, 0.2) is within 0.1% (C =
1.2421), and the tie goes to the f₁ closest to 0.25.

**P2 lies on the edge of the extended grid in α_L (0.75).** As pre-registered, it is kept, with no second extension.
The radius at α_L = 0.75 is the 25% quantile of λ_max. Since ϑ* = 1.5 > 1, the gate's margin comes from g*, not from
r₁ (Section 9 of the paper). The gain from extending was 0.14% in C over the grid choice.

Top ten feasible candidates (R² = 0.5; refusal in the (b) cells, largest and mean; (a) refused nothing for all ten):

| rank | ϑ* | α_L | f₁ | C | C (a) std | C (b) bn | largest refusal (b) | mean refusal (b) |
|---|---|---|---|---|---|---|---|---|
| 1 | 1.5 | 0.75 | 0.25 | **1.2409** | 1.3186 | 1.1632 | 0.010 | 0.0006 |
| 2 | 1.5 | 0.75 | 0.2 | 1.2421 | 1.3216 | 1.1626 | 0.000 | 0.0000 |
| 3 | 1.5 | 0.5 | 0.25 | 1.2426 | 1.3211 | 1.1641 | 0.010 | 0.0004 |
| 4 | 1.25 | 0.05 | 0.2 | 1.2437 | 1.3271 | 1.1602 | 0.010 | 0.0009 |
| 5 | 1.5 | 0.5 | 0.2 | 1.2441 | 1.3245 | 1.1637 | 0.000 | 0.0000 |
| 6 | 1.5 | 0.25 | 0.25 | 1.2444 | 1.3238 | 1.1650 | 0.005 | 0.0001 |
| 7 | 1.5 | 0.25 | 0.2 | 1.2462 | 1.3276 | 1.1648 | 0.000 | 0.0000 |
| 8 | 1.25 | 0.75 | 0.1 | 1.2471 | 1.3343 | 1.1598 | 0.000 | 0.0000 |
| 9 | 1.5 | 0.05 | 0.35 | 1.2472 | 1.3253 | 1.1691 | 0.010 | 0.0011 |
| 10 | 1 | 0.25 | 0.1 | 1.2473 | 1.3377 | 1.1569 | 0.005 | 0.0003 |

By ϑ* (20 candidates each):

| ϑ* | feasible | best feasible (α_L, f₁): C | best overall (α_L, f₁): C, largest refusal (b) |
|---|---|---|---|
| 0.5 | 0 | — | (0.0025, 0.1): 1.2500, 0.090 |
| 0.75 | 1 | (0.0025, 0.1): 1.2524 | (0.5, 0.1): 1.2439, 0.130 |
| 1 | 3 | (0.25, 0.1): 1.2473 | (0.05, 0.2): 1.2414, 0.145 |
| 1.25 | 7 | (0.05, 0.2): 1.2437 | (0.75, 0.2): 1.2374, 0.045 |
| 1.5 | 17 | (0.75, 0.25): 1.2409 | (0.75, 0.25): 1.2409, 0.010 |
| 2 | 20 | (0.75, 0.25): 1.2530 | (0.75, 0.25): 1.2530, 0.000 |
| 2.5 | 20 | (0.75, 0.25): 1.2647 | (0.75, 0.25): 1.2647, 0.000 |

**The constraint binds.** Six infeasible candidates, all with ϑ* = 1.25, have a lower C than P2, by at most 0.28%
(best: (1.25, 0.75, 0.2) at C = 1.2374). They refuse 2–15% of releases in 7–27 cells of (b). No candidate with ϑ* =
0.5 is feasible: on (b) every one refuses up to 9–100% of releases in some cell.

**Reference settings on the same 140 cells:**

| setting | C | C (a) std | C (b) bn | largest refusal (b) | mean refusal (b) |
|---|---|---|---|---|---|
| I | 1.2744 | 1.3553 | 1.1935 | 0 | 0 |
| old P | 1.2861 | 1.3036 | 1.2687 | 1.000 | 0.367 |
| previous | 1.3058 | 1.4057 | 1.2059 | 0 | 0 |

On (a), the old P stays 1.1% better than P2 (1.3036 against 1.3186). On (b), its gate refuses 37% of releases on
average and all of them in some cells. P2 is 2.6% better than I on (a) and 2.5% on (b).

**Robustness (not used).**
- R² = 0.3: P2 is first by C (1.1012; I 1.1139, old P 1.1221) and feasible. Under the tie rule the choice there would
  be (1.5, 0.5, 0.25), which is within 0.1% and has the smaller α_L.
- R² = 0.9: P2 is first by C (2.8138; I 3.0724, old P 3.0005) and feasible. The rule applied there picks P2 itself.

P2's largest refusal on (b) is 0.005 at R² = 0.3 and 0.010 at R² = 0.9. It refuses nothing on (a) at any R².

**n_off of P2**, eq. (noff), ε = 0.5, 1, 2, 4, 8:
- (a): 31.5k, 16.2k, 8.4k, 4.4k, 2.3k (I: 52.5k, 27.0k, 13.9k, 7.3k, 3.9k; old P: 22.0k, 11.3k, 5.8k, 3.0k, 1.6k)
- (b), ℓ = 0.105: 16.0k, 8.2k, 4.2k, 2.2k, 1.2k

In closed form at p = 11, n_off = (11.5Δ + 8.9Δ_G)/(ℓ√(2ρ_run)) under P2, against (19.1Δ + 14.8Δ_G) under I.

The setting is now fixed. Validation, real-data and encrypted runs follow as stated above.
