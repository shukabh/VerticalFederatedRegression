# Independent critique vs the earlier reviews

The independent reviewer read only the paper and `scenario_b/`, without the earlier reports or
the git history. Its report is [`critique.md`](critique.md), with demos in `demos/`. This page
compares its findings with the earlier reviews (`reports/code_review.md` and
`reports/privacy/`) and records what was re-checked.

## The main update: circuit privacy is critical, not "high, magnitude uncertain"

The earlier crypto report (C1) flagged missing circuit privacy but called the per-record
leakage "genuinely uncertain". The independent review settles it, and the orchestrator
re-ran the demo (`demos/d1_ckks_circuit_privacy.py`, real OpenFHE; identical output):

- `add_scalar` is `EvalAdd(ct, float(z))` (`he_backend.py:248`). It changes only the
  ciphertext's c0 component, so **c1 of O's reply does not depend on the DP noise at all**
  (T3: byte-identical for different z).
- c1 is a deterministic function of R's own ciphertext, the evaluation keys and **O's
  plaintext columns**. A DP adversary knows every record but one. R can therefore recompute c1
  for each candidate dataset and compare.
- Result: R identifies the true dataset in **100% of 20 trials**, for a matched record *and*
  for an **unmatched** one. An ε=1 view would cap R at 73%.
  - Unmatched records leak because `b` masks them only in the message: `Enc(b)`'s c1 is
    random everywhere, so O's values for every row enter c1.
- Adding a fresh `Enc(z)`, as the paper's eq. (5) does, re-randomises c1 (T5: the byte test
  fails in 20/20). The error polynomial still depends on O's data, so **noise flooding** (or
  an equivalent sanitisation) is still needed. That residual channel is argued, not
  demonstrated: the `d2` demo that `d1`'s output mentions is not in `demos/`.

Consequence: in the code as it stands, under honest-but-curious, **neither Definition 1 (input
privacy of O against R) nor Definition 3 (output DP) holds for R's actual view.** This applies
to O's whole register, not only the matched cohort.

**Fix** (a protocol change, moderate cost):
1. O adds a fresh encryption of the noise, not a plaintext scalar.
2. It then floods the error with a statistically hiding term (for example, adding an
   encryption of zero with error about 2^κ times the data-dependent error, κ≈40), or
   switches to a decryption procedure that reveals only the value.
3. Budget the extra modulus bits, and fold in the paper's item (iv) so that one
   finite-precision-safe distribution serves as both the DP noise and the flooding noise.

The same reasoning applies to O's BFV PSI replies (crypto report C4).

## Agreement (found independently by both)

| Independent | Earlier | Topic |
|---|---|---|
| C2 | review #1, DP D2 | Eq. (6) is add-one but Def. 4 is replace-one. The independent review puts the understatement at "up to ~1.5×" for its bounds; the earlier closed form gives 1–2× across bounds (close to 2 when `B_R` dominates). Both agree the fix is to recalibrate to replace-one. |
| C3 | review #2, DP D3 | `--scale_source committed` computes scales from the data and marks the run DP-valid. |
| C4 | review #2, DP D1, A11 | `dp_params.json` gives R statistics of O's data; calibration needs a trusted third party. |
| C5 | accuracy study | At the target cohort sizes the estimates are dominated by shrinkage. The independent review says raw ε=1 gives about 100% relative error until n≈50k; the accuracy study finds 86–99% at n ≤ 4,000 and says n≈36k is needed for 10% at ε=1. |
| C9 | crypto C5, review #11 | PSI reveals `D` and `α` (O's max bin load). |
| Credit | review "checked and correct" | The single-draw bias operator including `−P diag(P) Π_O` is right, keeping `A` local is right, and the Balle–Wang σ is right. |

## New in the independent review

- **C6: releasing `yᵀy` in the same budget inflates σ.** `yᵀy` isn't needed for β̃, yet it
  contributes the `B_y⁴` term. With the accuracy study's bounds (`B_R=B_O=√5, B_y=5`),
  Δ₂ = √925 = 30.4 with it and √300 = 17.3 without. That makes σ **1.76× larger** than the
  point estimate needs, which shifts every cohort-size threshold by the same factor. Fix:
  release `yᵀy` under a separate budget, or only when inference is requested.
- **C7: Theorem 1's "bias" is a conditional expectation.** With Gaussian E,
  `E[(G+E)⁻¹(c+f)]` does not exist (d4 part C: the running means do not converge at ρ=0.5).
  The theorem should be stated for the conditional or truncated mean on the event where the
  ρ-gate holds. This is a wording fix.
- **C8: the ρ-gate chooses λ for validity, not MSE.** It over-shrinks compared with the
  interior-λ optimum of Prop. 1. Standardizing with committed scales is the largest single
  utility lever (d6).

## Disagreement, resolved

- **Does the bias correction survive a λ chosen from G̃?**
  - Independent (d4 part D): yes. The residual is 0.0014 with the adaptive λ against 0.0010
    with a fixed λ, SE 0.001.
  - Earlier (`reports/checks/theorem1_and_adaptive_ridge_check.py`): no. The residual is
    3–5× the corrected bias.

  Both are right in their own regime. The independent test uses heavy shrinkage
  (λ≈1,430, `‖β̂_λ‖=0.41` against 2.03 for OLS), where the effect is below its Monte Carlo
  resolution. The earlier test has λ comparable to `λ_min(G)`, where it dominates. Theorem 2's
  fixed-λ assumption is violated in both. How much that matters grows as the ridge becomes
  moderate.
- **Balle–Wang tail accuracy.** The independent review finds the σ correct to 1e-12 at
  typical parameters. The earlier DP report (D9) finds under-noising only at extreme settings
  (δ ≤ 1e-8 with ε=20). The two are consistent.

## In the earlier reviews but not the independent one

These still stand:
- There is no privacy ledger or composition across runs (D4).
- The linkage (membership, `n`, and the row index, with no O-side permutation) is not covered
  by DP (D5, crypto C7).
- The published β gives R's own cohort no protection, and fitted values are not
  post-processing (D7).
- The O-block ridge breaks down near its ceiling (review M1).
- `DUMMY_ROOT` false matches (M2).
- The λ search overshoots, with no λ=0 branch (M3).
- Label loss when D=1 (#7).
- `n_O` is capped at one ciphertext (#6).
- Unkeyed hash-to-field for low-entropy IDs (crypto C6).
- Floating-point Gaussian and non-cryptographic RNG (D8, C8).
- `pickle` on peer data and no TLS (out of scope, but fix before deployment).
