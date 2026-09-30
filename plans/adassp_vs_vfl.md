# AdaSSP (Wang 2018) vs our VFL protocol

AdaSSP is described here from its paper and from `adassp.m` in
<https://github.com/yuxiangw/optimal_dp_linear_regression>. "Ours" is the revised Scenario-B
protocol in `paper/main.tex`.

Both methods are **sufficient-statistics perturbation** (SSP). Each adds Gaussian noise to the
Gram matrix and the moment vector, then solves a ridge-regularized system. The DP machinery is
the same. The difference is what surrounds it.

| | **AdaSSP** (Wang, UAI 2018) | **Our VFL protocol** (Scenario B, revised) |
|---|---|---|
| Data setting | One curator already holds the joined data `(X, y)` | Two parties with vertically partitioned data. R has `X_R` (researcher cohort); O has `X_O` and `y` (national register). The data are never pooled. |
| Record linkage | None needed; the rows are already joined | **Private** linkage by labeled PSI (BFV). R learns the match set and the alignment; O learns only public sizes. |
| Who computes the statistics | The trusted curator, in the clear | O, under R's CKKS key (depth-1 homomorphic inner products). Nobody ever sees the joint Gram in the clear. |
| Trust / threat model | The curator is trusted; the adversary is whoever sees the output | Honest-but-curious parties. R both takes part and receives the output. O never sees R's data. R sees only noised, sanitized aggregates. |
| Unit of privacy (adjacency) | One whole record `(x, y)` | One matched individual's O-side data `(x_O, y)`, with `x_R` fixed (replace-one). R's features are R's own data. Membership in the intersection is revealed to R by design. |
| What is noised | All of `XᵀX`, all of `Xᵀy`, and `λ_min(XᵀX)` | Only the O-dependent blocks `B, C, c_R, c_O, yᵀy`. R's block `A = X_Rᵀ X_R` is **exact** and never leaves R. |
| Sensitivity | Separate: `B_X²` for `XᵀX` and `B_X B_Y` for `Xᵀy`, each calibrated on its own share of ε | One joint replace-one sensitivity for the stacked O-dependent release. There is **no `B_R⁴` term**, because `A` is not released. |
| Budget split | ε/3 for λ_min, ε/3 for `XᵀX`, ε/3 for `Xᵀy` | All of ε goes to one release, with a single σ from the analytic Gaussian mechanism. λ costs nothing. |
| Noise | Continuous Gaussian. Symmetric matrix noise `(G+Gᵀ)/2`, whose off-diagonal variance is half the diagonal. Scale `√log(6/δ)·B²/(ε/3)`. | Discrete Gaussian from a CSPRNG. Single-draw symmetric noise (every entry has variance σ²), carried identically in every CKKS slot, then flooded and sanitized. |
| Ridge λ | Adaptive: `λ = max(0, η − λ̃_min)`, where `λ̃_min` is a **private** estimate costing ε/3 | Fixed from public inputs before decryption, `λ = max{0, 2ρ*σ√p − nℓ}`. A release gate ρ̂ ≥ 1 is post-processing. (A private λ_min is not available in VFL, because no one holds the Gram in the clear.) |
| Bias handling | None. The estimator keeps the ridge bias and the bias from matrix inversion | A **zero-budget bias correction** `β̂ − σ²M(P̃)β̂`, from Theorems 1–2 (conditional, second order) |
| Theory | Near-minimax prediction and estimation rates that adapt to λ_min | Leading-order finite-noise bias with an explicit validity condition, and a sensitivity closed form. No optimality result yet. |
| Bounds / preprocessing | Code: z-score X, normalize rows to unit norm, divide y by max\|y\|, giving `B_X = B_Y = 1` | Committed public bounds `B_R, B_O, B_y` with clipping. Each party standardizes with **public** constants; no data-derived statistic is exchanged. |
| Composition | One release at total ε | A zCDP **ledger** at O caps cumulative privacy loss across runs, retraining and inference queries |
| Cryptography | None | BFV (PSI), CKKS (aggregation), fresh re-encryption and noise flooding (circuit privacy), 128-bit parameters |
| Cost | `O(nd²)` in plaintext | PSI with depth `O(log D)`, plus `O(p²)` encrypted inner products over `n_O` slots, split into chunks |
| When to use it | A single data holder wants to publish a DP regression | Two data holders who **cannot legally pool** their data want a joint regression with an output-privacy guarantee |

## Our selling point

AdaSSP answers the question "given a dataset, how do I release a differentially private
regression?" It assumes the joined dataset already sits with a trusted curator. In the setting
that motivates this work, a researcher's cohort and a national register, that joined dataset is
exactly what the law forbids anyone to hold.

Our protocol delivers the same kind of estimator, sufficient-statistics perturbation with a
ridge, without the data ever being pooled. Records are linked privately, the cross-party
statistics are computed under encryption, and the organization keeps control of the privacy
budget from end to end: it sets σ, it injects the noise, and it keeps the ledger.

The vertical split also brings a utility advantage over central SSP:
- R's own block `A` is exact rather than noised.
- The joint sensitivity has no `B_R⁴` term.
- No budget is spent choosing λ.
- The remaining finite-noise bias is removed at no privacy cost.

So, measured against the protection O's individuals receive, the estimator can be *less* noisy
than applying AdaSSP to a pooled table.

The trade-off is a narrower guarantee. What is protected is O's side of each matched record, not
the whole record, and R learns which of its cohort appear in the register. A fair comparison
therefore reports AdaSSP both as published (the stronger guarantee) and recalibrated to our
adjacency (the matched guarantee). The second isolates the estimator choice: a public fixed λ
with bias correction, against a private adaptive λ.
