# Check item #9: the bias-aware interval with a public variance bound

Paper check item #9 (`paper/main.tex`, the `\tocheck` after "public bound $\bar\sigma^2$"). With the data-driven
residual variance s~^2 the bias-aware interval for beta is not guaranteed when clipping binds: the clipped errors are
heteroskedastic, and s~^2 estimates an average of the conditional variances, not their maximum.

**Decision (approved).** The bias-aware interval that the protocol reports as *guaranteed* uses a public bound
sigma_bar^2 >= max_i Var(y~_i | X), an Agreement parameter fixed before the run from public information: B_y^2 by
default (valid with no model), or a declared bound on the pre-clipping error variance, admissible when only y is
clipped (clipping is 1-Lipschitz). Proofs Part B, Proposition S10b. The s~^2 interval is still reported, labelled as
valid only under the working model. The Wald interval for theta_lambda is unchanged.

## Implementation

* `vfr/dp.py`: `Agreement.sigma_bar2` (None = B_y^2); `dp.sigma_bar2_of(ag)` validates it (finite, in (0, B_y^2]) and is
  called by `dp.alpha_split`, so O refuses an ill-formed value in Setup; R checks that the field matches.
* `vfr/solve.py`: `stage2_infer(..., sigma_bar2=None)` returns `half_beta_guaranteed` / `V_guaranteed` (sigma_bar^2)
  and `half_beta_model` / `V_model` (s~^2); `half_beta` and `V` remain aliases of the working-model values so that
  earlier results reproduce. No change in the random-number consumption, the estimate or the Wald interval.
* `vfr/party_r.py`, `run_protocol.py` (`--sigma_bar2`; prints both intervals). Spec `docs/protocol_v31.tex`
  (Algorithm 6, Setup list, parameter table) and `implementation/DEVIATIONS.md` row A26.
* Tests: `implementation/tests/test_sigmabar.py` (validation, both intervals, unchanged draws, and a fixed-design Monte
  Carlo with binding clipping where the guaranteed interval covers beta* and the working-model one does not).

## Simulation (`experiments/v31_sigmabar/sim_sigmabar.py`, `tables_sigmabar.py`)

Same design as `experiments/v31_settings/sim_settings.py`: setting I (exact r_1, f_1 = 0.25, theta* = 2.5,
alpha_L = alpha_U = 0.0025), delta = 1e-9, sigma from Calibrate's rho split with each population's B_y, a fresh pair
of releases per cohort, `solve.stage2_infer` with 2,000 antithetic bootstrap pairs, no public bound on ||beta||.
1,000 cohorts per cell; eps = 1 (n = 5k, 25k, 50k, 100k) and eps = 8 (n = 1k, 2.5k, 10k, 25k, 100k). All three
intervals come from the same release and bootstrap.

Populations (features, feature bounds and slopes of `sim_core.Population(r2=0.5, seed=0)`):

| population | errors before clipping | B_y | y clipped | mean / max Var(y~ given X) |
|---|---|---|---|---|
| base | homoskedastic, variance 0.5 | 2.89 (99.5% quantile) | 0.5% | 0.50 / 0.50 |
| yclip | homoskedastic, variance 0.5 | 1.27 (80% quantile) | 20% | 0.35 / 0.44 |
| hetero | s.d. proportional to 0.3 + abs(z) of O's first (lognormal) feature, mean variance 0.5 | 1.51 (90% quantile) | 10% | 0.29 / 1.87 |

Targets: the projection coefficient beta* = G^-1 X~' E[y~ | X] of each cohort (the estimand of Proposition S10b;
closed form for Gaussian errors) and the population least-squares coefficient of the clipped data (4M rows; the
target of the paper's FWS comparison). Coverage over the ten slopes; an unbounded interval counts as covering.

## Findings (full tables in `key_numbers.md`, raw summaries in `sim_sigmabar.json`)

* **Guaranteed interval (sigma_bar^2 = B_y^2)** covers every slope in every cell (lowest single slope 0.956 for beta*,
  0.957 for the population coefficient). It is wide where the sampling variance dominates: median half-width ratio to
  the working-model interval 1.26-1.46 (base, eps = 1, ridge off) and 2.3-3.7 (base, eps = 8), close to
  sqrt(B_y^2 / s~^2) = 4.1 at n = 100k; 1.4-1.9 (yclip) and 1.6-2.4 (hetero) at eps = 8.
* **Working-model interval (s~^2)** covers on base and yclip (0.936-0.977 per slope where finite), but on hetero it
  under-covers the slope of the variance-driving feature at eps = 8: 0.923 (n = 10k), 0.894 (25k) and 0.829 (100k) for beta*;
  0.908, 0.867 and 0.788 for the population coefficient. At eps = 1, where the mechanism noise dominates, it covers.
* **Declared bound (sigma_bar^2 = 0.5)**, admissible on base and yclip: coverage 0.936-0.985 per slope, the same width
  as the working-model interval on base and at most 14% wider on yclip.
* Ridge on: both intervals are unbounded while L_hat = 0, and otherwise dominated by the bias bound (within 3%).
* The Wald interval for theta_lambda covers 0.947-0.962 in every cell (unchanged).

## Reproduce

    python experiments/v31_sigmabar/sim_sigmabar.py run --reps 1000 --workers 4   # cells cached in experiments/_work/v31_sigmabar
    python experiments/v31_sigmabar/sim_sigmabar.py summarize
    python experiments/v31_sigmabar/tables_sigmabar.py

Paper draft: `paper/drafts/sigmabar.tex` (merge with `experiments/v31_more/apply_more_draft.py --draft ...`).
