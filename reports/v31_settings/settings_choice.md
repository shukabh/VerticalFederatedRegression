# Choice of the evaluation settings I and P: pre-registered rule

Written on 2026-10-24, after the implementation of the exact stage-1 radius and before any run of the selection
study, of the validation, or of any real dataset under a new setting. The rule below decides the settings; the
selection result is appended at the end without changing the rule. Script:
`experiments/v31_settings/select_settings.py`.

## What does not change

- The protocol: two releases, the stage-1 rule (λ_I, λ_O, κ_Ψ), the stage-2 gate ϑ̂ ≥ 1 (a refused release
  outputs nothing, so θ = 0, and still spends the budget), the solve, bias correction off.
- Calibration by the implementation (`vfr/dp.py` `calibrate`), κ = 40, 40, 40, 44, 48 for ε = 0.5, 1, 2, 4, 8,
  y'y in stage 2 when it is free.
- Both settings are settings of the same protocol. Neither changes the privacy guarantee: ε, δ, the sensitivity
  and the noise follow from (ε, δ, f₁) alone, and r₁, ϑ*, κ_Ψ and α_L are applied by R to released statistics.

## Selection data (synthetic only)

- Population: `vfr/data.population()` (`experiments/sim_core.Population`, seed 0), p = 11, d_R = 4, d_O = 6.
  R² = 0.5 decides. R² = 0.3 and 0.9 are reported as a robustness check and are not used.
- δ = 1e-9, ε ∈ {0.5, 1, 2, 4, 8}.
- n ∈ {100, 250, 500, 1k, 2.5k, 5k, 7.5k, 10k, 15k, 25k, 35k, 50k, 75k, 100k} (the grid of 13.5), 200 cohorts per
  n, drawn with seeds (tag 5150) independent of the validation runs (13.5 seeds).
- Common random numbers: per (n, ε) one set of unit stage-1 and stage-2 noise draws, scaled by each candidate's
  σ₁ and σ₂.
- None of the 29 UCI datasets, nor protein, is read for the choice.

## Setting I (inference)

Fixed: r₁ exact, i.e. σ₁ times an upper 99% confidence bound on the (1 − α_L) quantile of λ_max of the unit
stage-1 noise matrix, from at least 1e6 Monte Carlo draws at α_L = 0.0025 (`vfr/solve.py`); ϑ* = 2.5; κ_Ψ = 2;
α = 0.05 with α_L = α_U = α/20; intervals on.

Free: f₁ ∈ {0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50}.

Criterion: C_I(f₁) = mean over the 70 (n, ε) cells of log(RMSE_DP / RMSE_OLS), where RMSE_DP is the pooled RMSE
of the 10 slopes against the true coefficients over the 200 cohorts (θ = 0 for a refused release) and RMSE_OLS is
that of non-private least squares on the same cohorts. **I takes the f₁ that minimizes C_I.** n_off of Section 9.5
is reported for each f₁ but does not decide.

## Setting P (prediction)

Fixed: r₁ exact (as above, at the candidate's α_L), κ_Ψ = 2, the gate, intervals off (P gives up the interval
guarantee; α_L only sizes the stage-1 radius).

Grid (36 candidates): ϑ* ∈ {0.5, 1, 1.5, 2.5} × α_L ∈ {0.0025, 0.25, 0.5} × f₁ ∈ {0.1, 0.2, 0.35}.

Criterion: C_P = mean over the 70 (n, ε) cells of (mean test MSE of the release over the 200 cohorts) / (mean test
MSE of non-private least squares on the same cohorts). Test MSE is the population's: clipped design, unclipped y,
from the population moments of 13.5 (4 million rows). A refused release counts with θ = 0. **P takes the candidate
that minimizes C_P.** The gate-refusal rate of every candidate is reported.

If the minimizer of either criterion lies on an edge of its grid, it is reported as such and the grid is not
extended.

## What happens next, fixed now

1. Validation of I on the synthetic population as in 13.5 (200 cohorts per n, n from 1e2 to 1e5, R² ∈ {0.3, 0.5,
   0.9}, inference at R² = 0.5 with B = 2000 antithetic pairs and no public bound on ‖β‖): Wald coverage for
   θ_λ, bias-aware coverage for β where the ridge is off, the switch-off against n_off at 0.6–1.4·n_off (200
   cohorts each), the slope RMSE and the records needed for twice the floor. The same for P's prediction curve.
   The settings are not changed after seeing these results, whatever the coverage.
2. Bias check under I (13.6 design, shortened).
3. AdaSSP benchmark (13.7): same 29 datasets, folds, δ = 1e-6, 50 draws per fold; arms (a), (b), (c) and (d)
   under I and P, (e) under I; the previous setting from the existing results.
4. Real-data sweep (13.8): bike, elevators, pol, protein at δ = 1e-9; (d) under I and P, and AdaSSP.
5. Encrypted spot check: 5–10 OpenFHE runs, register ≤ 16,384 rows, I and P each at one ε with the ridge on.

## Result of the selection (appended after the run; the rule above is unchanged)

Run: `select_settings.py run --reps 200` then `choose`; full output in `selection.json`, the chosen values in
`settings.json`. Exact radius at p = 11 (d_A = 5, d_O = 6): r₁/σ₁ = 7.415 at α_L = 0.0025 (union bound 11.529),
5.468 at α_L = 0.25 (8.988), 4.940 at α_L = 0.5 (8.298); upper 99% confidence bounds from 1e6, 2e5 and 2e5 draws.

**Setting I: f₁ = 0.25** (r₁ exact, ϑ* = 2.5, κ_Ψ = 2, α = 0.05, α_L = α_U = 0.0025, intervals on).

| f₁ | 0.20 | 0.25 | 0.30 | 0.35 | 0.40 | 0.45 | 0.50 | previous (union, 0.1) |
|---|---|---|---|---|---|---|---|---|
| C_I, R² = 0.5 (decides) | 1.6743 | **1.6719** | 1.6762 | 1.6851 | 1.6978 | 1.7140 | 1.7336 | 1.8006 |
| C_I, R² = 0.3 | 1.3759 | 1.3754 | 1.3803 | 1.3895 | 1.4020 | 1.4177 | 1.4364 | |
| C_I, R² = 0.9 | 2.4048 | 2.4017 | 2.4057 | 2.4152 | 2.4289 | 2.4464 | 2.4677 | |
| n_off at ε = 1 | 27.8k | 27.0k | 26.6k | 26.5k | 26.6k | 26.9k | 27.3k | 41.7k |

The criterion is flat between 0.2 and 0.3 and agrees at R² = 0.3 and 0.9. n_off alone would favour 0.35 (as
Section 9.5 said), but the larger σ₂ of a larger f₁ costs accuracy once the ridge is off. The gate refused no
release for any f₁. n_off under I: 52.5k, 27.0k, 13.9k, 7.3k, 3.9k for ε = 0.5, 1, 2, 4, 8 (previous: 81.2k,
41.7k, 21.6k, 11.2k, 6.0k).

**Setting P: ϑ* = 0.5, α_L = 0.5, f₁ = 0.2** (r₁ exact, κ_Ψ = 2, intervals off, the gate kept).

C_P at R² = 0.5 (the gate-refusal rate, largest over the 70 cells, in brackets where it is not 0):

| ϑ* \ α_L, f₁ | 0.0025, 0.1 | 0.0025, 0.2 | 0.0025, 0.35 | 0.25, 0.1 | 0.25, 0.2 | 0.25, 0.35 | 0.5, 0.1 | 0.5, 0.2 | 0.5, 0.35 |
|---|---|---|---|---|---|---|---|---|---|
| 0.5 | 1.3420 | 1.3157 | 1.3075 | 1.3259 | 1.3046 | 1.3028 (0.01) | 1.3214 | **1.3017** | 1.3025 (0.02) |
| 1 | 1.3517 | 1.3266 | 1.3172 | 1.3362 | 1.3152 | 1.3100 | 1.3318 | 1.3121 | 1.3083 |
| 1.5 | 1.3610 | 1.3375 | 1.3282 | 1.3462 | 1.3260 | 1.3202 | 1.3420 | 1.3229 | 1.3181 |
| 2.5 | 1.3784 | 1.3579 | 1.3508 | 1.3648 | 1.3474 | 1.3429 | 1.3609 | 1.3445 | 1.3407 |

The previous setting gives C_P = 1.4045 and I gives 1.3538. The chosen P refused no release in any cell (largest
mean refusal rate on the grid: 0.001). **P lies on the grid edge in ϑ* (0.5) and α_L (0.5).** As pre-registered,
the grid is not extended; the top five candidates are within 0.5% of each other in C_P. At R² = 0.3 the same
candidate is best; at R² = 0.9 it is third (3.631 against 3.560 for f₁ = 0.35), so the best f₁ for P depends on
the signal strength. I's f₁ = 0.25 is best at all three R².
n_off under P: 22.0k, 11.3k, 5.8k, 3.1k, 1.6k for ε = 0.5, 1, 2, 4, 8.
