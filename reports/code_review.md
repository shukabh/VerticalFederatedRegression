# Review: Scenario B implementation vs. the working draft

Reviewed against *Privacy-Preserving Vertical Federated Linear Regression when the
Organization Holds the Response* (working draft, 31 Aug 2026).

Files reviewed (user's originals, unmodified, in `scenario_b/`):
`party_r.py`, `party_o.py`, `run_protocol.py`, `calibrate_hyperparameters.py`,
`generate_vfl_data.py`.

**Not reviewed: they were not provided.** `psi_common.py` (cuckoo hashing, binning,
polynomial interpolation, hash-to-field), `phase1_common.py` (noise draw, assembly,
the ρ gate, the bias correction) and `he_backend.py` (OpenFHE wrapper). These hold
most of the mathematics. The stand-ins in `scenario_b/` were written from the call
sites so that the pipeline can run. They are **not** the user's code, so any result
that depends on them reflects the stand-in, not the original.

## Verdict

The orchestration follows the corrected data flow in Section 4:

- `A` is built locally from `Xdot` and never sent or noised (`party_r.py:105`).
- R sends only the public key, `Enc(b)` and `Enc(Xdot_R)` (`party_r.py:114-118`).
- O computes the five blocks as depth-1 inner products. It adds noise to every
  O-dependent block, and the noise on `B` is symmetric (`party_o.py:109-131`).
- σ comes from the analytic Gaussian mechanism applied to eq. (6).
- R runs the Remark 4 gate and applies the zero-budget correction.

The intercept is placed in R's clean block as `b`, with `B_R → sqrt(1+B_R²)` in the
sensitivity. That is correct, and so is the back-transform to the original scale.

There are two intentional deviations from the draft, and both are fine:

- **PSI.** The code uses Chen–Laine–Rindal labeled PSI (cuckoo hashing, windowed
  powers, partitioned polynomials, label = O's row index) rather than Appendix B.
  It is more efficient and gives R the same output (I, σ).
- **Noise injection.** O adds the noise as a plaintext scalar rather than as
  `Enc(z)`, which gives the same distribution.

The problems are listed below, most serious first.

## Findings

| # | Severity | Where | Finding |
|---|---|---|---|
| 1 | **High** (privacy; paper and code) | paper Def. 4 vs eq. (6); `calibrate_hyperparameters.py:112-121` | **The sensitivity doesn't match the adjacency.** Definition 4 is *replace* one matched individual's `(x_O, y)` with `I` fixed. Eq. (6) is the *add/remove* bound. Under replacement, `ΔB = xxᵀ − x'x'ᵀ`, `ΔC = x_R(x − x')ᵀ`, and so on, which is larger. Numerically maximised (`reports/checks/replacement_sensitivity_check.py`): with `B_R=B_O=B_y=1`, Δ = √10 against eq. (6)'s √5, so **σ is √2 too small**. With `B_R=B_O=3, B_y=20`, the ratio is 1.001 because the `y²` term dominates and is the same under both adjacencies. Always-valid bounds: `2·Δ₍₆₎` (triangle inequality), or the termwise `sqrt(2B_O⁴ + 4B_R²B_O² + 4B_O²B_y² + 4B_R²B_y² + B_y⁴)`. The DP report gives a tighter closed form. |
| 2 | **High** (privacy) | `calibrate_hyperparameters.py:243-245, 248, 64-83, 257-274, 299-316` | **Calibration needs a party that sees both datasets, and it leaks what it computes.** `calibrate()` loads `X_R`, `X_O` and `y_O` in one process. It then writes data-derived values into `dp_params.json`, which *both* parties read: 0.99-quantile bounds, clip rates, and standardization centers and scales. So R receives statistics of O's data outside the DP budget, and O receives R's feature means and SDs. Separately, `--scale_source committed` offers no way to pass the constants: `_pick(None, computed)` quietly falls back to the data values while setting `dp_valid=True`, which also suppresses the warning. **Fix:** split calibration by party. R commits `B_R` and its own scaling; O commits `B_O`, `B_y` and σ. Share only those public constants. |
| 3 | **High** (validity; paper and stand-in) | paper Thm 2 + Remark 4; `party_r.py:137` (`mode="auto"`) | **Choosing λ and Ψ from the noisy Gram breaks the "unbiased through O(σ⁴)" claim.** Theorem 2 assumes `λΨ` is fixed in advance. The gate picks λ from `λ_min(G̃)`, which depends on `E` at first order, so it adds an O(σ²) bias that the correction does not remove. In Monte Carlo (ρ≈2, `reports/checks/theorem1_and_adaptive_ridge_check.py`), fixed λ leaves a residual of about 5e-4 after correction. The adaptive gate leaves about 7e-2, **3–5× larger than the bias being corrected**. This only matters when the ridge is active; when λ=0 with high probability there's no issue. **Fix:** choose λ from quantities that don't depend on the noise. For example, use the certificate `λ = ρ_target·2σ√p` (already computed in calibration) for full ridge, or anything based only on `A`, σ and p. Alternatively, extend Theorem 2 to cover a λ that depends on the data. |
| 4 | Medium (privacy) | `party_o.py:42-45, 69` | **O never permutes its own records.** PSI labels are O's row indices in file order, and R learns each matched person's index (`party_r.py:94`). If O's file is sorted by region, registration date or ID, the index reveals that attribute. Appendix B requires a secret permutation τ, but here only the data generator shuffles. **Fix:** O shuffles `(ids_O, X_O, y_O)` with a secret permutation right after loading. |
| 5 | Medium (security; depends on `he_backend.py`) | `party_o.py:94-96, 112-131` | **Circuit privacy.** R holds the secret key and decrypts ciphertexts that O computed from its plaintext data. Nothing visible floods the noise or reduces to the last modulus level. The DP noise goes into the *message*, not the RLWE error. The crypto-layer report covers what this can leak. Also check that `inner_product` replicates the total into every slot. If other slots hold partial sums, then differences between adjacent slots expose single records, and the DP noise cancels out. |
| 6 | Medium (scalability) | `party_r.py:112-113`; `calibrate_hyperparameters.py:288-289` | **`n_O` is limited to one CKKS ciphertext.** OpenFHE allows at most 65,536 slots. A national agency's `n_O` is in the millions. `n_chunks` is dead code: `batch` is always ≥ `n_O`, so it is always 1. **Fix:** split each column into chunks and add up the per-chunk inner products. |
| 7 | Low (bug) | `party_o.py:87-91` | **Labels are lost when D == 1.** `ctL` becomes `Enc(0)` and the constant label `L_layers[a][0]` is never added, so every match decodes to O-row 0. This happens only for tiny `n_O`, where no bin has two items. **Fix:** `ctL = bfv.add_pt(bfv.mul_pt(powers[1], [0]*NB), L_layers[a][0])`. |
| 8 | Low (tests) | `run_protocol.py:32-33`; `party_r.py:140` | **The verification only checks the match count, not the alignment.** A labeling error that gets the count right would pass. There is also no noise-free exactness test, and σ=0 raises `ZeroDivisionError` at `party_r.py:140`. With `center_scale`, the intercept model's slopes are compared with no-intercept OLS. |
| 9 | Low (security) | `party_o.py:77, 92-93, 110` | The PSI masks and the DP noise use numpy PCG64, which is not a CSPRNG, and the Gaussian is floating-point (Mironov 2012). The draft already lists the latter as item (iv). Use `secrets` for the masks and a discrete or snapped Gaussian for the noise. |
| 10 | Low (security) | `party_r.py:75, 89, 119`; `party_o.py:65, 79, 103, 106` | `pickle.loads` runs on bytes received from the peer, which allows arbitrary code execution, and the channel is unauthenticated with no TLS. The adversarial report covers this. |
| 11 | Low | `party_o.py:74` | O sends R `alpha` and `D`, which depend on how O's IDs fall into bins. Fix both from public parameters (`d_cap` and `n_O`). |
| 12 | Low | `run_protocol.py:14` | `time.sleep(2.0)` before starting O creates a race if R's imports take longer. O should retry its connection. |
| 13 | Low (incomplete) | `party_r.py:131-177` | The following are not implemented: `RSS̃ = ỹᵀy − c̃ᵀG̃⁻¹c̃` (only `yᵀy` is saved), the separate report of the deterministic shrinkage `−λP_λΨβ̂` that Prop. 1 recommends, and any budget ledger across runs. |
| 14 | Low (tooling) | `generate_vfl_data.py:287-291` | The generator writes `y.csv` (matched rows only), but `party_o.py:44` and `calibrate_hyperparameters.py:245` read `y_O.csv`. `scenario_b/prepare_data.py` fills this in. |

## Checked and correct

- **Theorem 1 / Lemma 1.** The bias operator `M(P)` equals the exact second moment
  `E[PEPE]/σ²` for the masked single-draw mechanism, to about 1e-20. The check sums
  over the basis of free entries. In Monte Carlo, the correction removes roughly 90%
  of the O(σ²) bias at ρ≈2, and the remainder is O(σ⁴) plus Monte Carlo error.
  The p=1 degenerate check holds.
- **Analytic Gaussian.** It matches Balle–Wang Thm 8. Bracketing then bisecting
  returns the conservative endpoint. `_Phi` uses `1+erf`, which loses accuracy
  below about −8, so it only matters for δ ≲ 1e-13; `0.5*erfc(-x/√2)` is safer.
- **PSI false-positive sizing.** `t ≥ n_R·n_O·n_hash/(NB·δ_psi)` is the right
  count for pairs that share a bin. The windowing condition `D ≤ 2^(W−1)` is correct.
- **Data-flow ledger.** Messages match Section 4.4, except that the PSI runs in the
  opposite direction from Appendix B.

## Notes on the draft itself

1. Eq. (6) vs Definition 4 (finding 1). Either change the adjacency or change the
   bound.
2. Theorem 2 vs the adaptive gate in Remark 4 (finding 3).
3. **Appendix B.**
   - Evaluating Horner's rule literally under BFV (`acc ← acc·x_j + Enc(c_k)`)
     multiplies the noise by about `p` at each of the `n_R` steps. The sum-of-powers
     form `Σ_k Enc(c_k)·(x_j^k mod p)` grows it only additively. The text says
     "n_R scalar-ciphertext multiplications", which fits the sum of powers, but the
     section is titled "Horner".
   - Notation clashes: τ is used for both the permutation and the mask, and `b_j`
     for both the selection vector and the identifier.
   - The implementation uses a different PSI (labeled, cuckoo-hashed), so the
     appendix should describe that one.
4. **Remark 2.** It treats `‖PE‖_op < 1` as equivalent to spectral radius < 1. The
   Neumann series converges if and only if the spectral radius is below 1, which
   `−G ≺ E ≺ G` captures. The operator-norm condition is sufficient but not
   equivalent, because `PE` is not symmetric.
5. **Eq. (6) and the symmetric B block.** It uses the full Frobenius norm of `ΔB`,
   although only the upper triangle is released. That is conservative, and it
   partly offsets finding 1.
