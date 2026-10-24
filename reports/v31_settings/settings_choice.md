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
