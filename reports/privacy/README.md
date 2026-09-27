# Privacy risks — consolidated (honest-but-curious model)

The threat model is **honest-but-curious**: R and O follow the protocol exactly but analyse
everything they legitimately receive. This page merges the three detailed reports and the
code review, removes duplicates, and ranks what applies under that model.

| Detailed report | Lens |
|---|---|
| [`dp_layer.md`](dp_layer.md) (D1–D10) | sensitivity, calibration, composition, output privacy |
| [`crypto_layer.md`](crypto_layer.md) (C1–C11) | HE and PSI: slots, circuit privacy, parameters, randomness |
| [`adversarial_deployment.md`](adversarial_deployment.md) (A1–A14) | deviating parties and deployment; mostly out of scope, kept for the record |
| [`../code_review.md`](../code_review.md) | implementation vs the draft |

Caveat: `psi_common.py`, `phase1_common.py` and `he_backend.py` were not provided. Items that
depend on them are marked **verify**.

**One run at realistic bounds leaks little** (D10: per-record noise-to-signal is about 25–300,
and whole-cohort reconstruction gives R² ≤ 0.09 at ε=8). The real exposure comes from the
systemic issues below. Each one either makes the (ε, δ) claim false or leaves something that
DP never covered.

## In scope, ranked

| # | Severity | Risk | Who learns what | Fix | Refs |
|---|---|---|---|---|---|
| 1 | **Critical** | **The calibration file leaks raw-data statistics.** `calibrate_hyperparameters.py` reads both datasets in one process and writes quantiles, clip rates and (when standardizing) means and SDs into the `dp_params.json` both parties load. When bounds are "suggested", σ itself also depends on the data. | R learns O's register statistics, unmatched people included, outside the budget. O learns R's cohort statistics. Across two calibration cycles, a new person's `y` and `x_O` were recovered exactly. | Split calibration by party: R commits `B_R` and its own scaling, O commits `B_O`, `B_y` and σ. Share only those constants. Refuse to run unless `bounds_committed` is set. | D1, D3, A11, review #2 |
| 2 | **Critical — verify** | **CKKS slot layout with broadcast noise.** `add_scalar` adds the noise to every slot, and R, who holds the secret key, can decrypt every slot. If any slot is not the full total (a partial sum from rotate-and-add, or a zero from masking), R subtracts slots and gets the **noise-free** statistic. In the partial-sum case it also gets single records. | R gets exact `B`, `C`, `c_R`, `c_O` and `yᵀy`, so the DP guarantee on them is void. | Decrypt all slots of one returned block in a real run. Either confirm a full cyclic sum (every slot = total), or mask to slot 0 and add the noise to slot 0 only. | C2, D6, review #5 |
| 3 | **High** | **The sensitivity doesn't match the adjacency.** Eq. (6) is add/remove, but Def. 4 is replace-one. The correct Δ is 1–2× eq. (6): √2 when the bounds are equal, close to 2 when `B_R` dominates (many R features, or the intercept), and about 1 when `B_y` dominates. | A claimed ε=1 actually delivers 1.47–1.96. | Use the replace-one closed form (drop-in `joint_sensitivity_B_replace` in `dp_layer.md`; checked by brute force and independently in `../checks/`), or change Def. 4 to add/remove. | D2, review #1 |
| 4 | **High** | **`--scale_source committed` computes the scales from the data anyway**, sets `dp_valid=True` and suppresses the warning. Neither party checks `dp_valid` or `bounds_committed`. | Same as #1. | Add a CLI path for real committed constants, and fail closed. | D3, review #2 |
| 5 | **High** | **No privacy ledger.** Every run, re-run or retrain draws fresh noise, and nothing records the cumulative ε. | At ε=1 per run: 10 runs give ε 3.6, or 5.4 under Def. 4. 100 runs give 14.4, or 22.7. | O keeps a budget ledger per dataset and per researcher, and refuses runs beyond a cap. | D4, A4 |
| 6 | **High** | **No circuit privacy.** O's returned BFV and CKKS ciphertexts are not flooded, mod-switched or re-randomised. The DP noise goes into the message, not the RLWE error, and the error depends on O's operands. | R holds the secret key and can examine the error. How much this leaks per record is uncertain, but IND-CPA gives O no protection against the key holder. | Noise flooding with a statistical security parameter, or at least modulus switching to the last level. Budget the extra modulus. | C1, C3, C4 |
| 7 | **High** | **DP doesn't cover the linkage.** R learns exactly which of its cohort are in O's register, plus `n`. It also learns each match's row index in O's file, and `party_o.py` applies no secret permutation (only the generator shuffles). | R. In the paper's own benefits example, membership *is* the outcome. | O shuffles its rows with a secret permutation. Treat membership as released information and document it. Consider protecting `n`. | D5, C7, review #4 |
| 8 | Medium | **HE parameters vs the HE Standard.** The BFV ring is fixed at 16384 while `t` (up to 60 bits) and the depth vary. The CKKS ring at `n_O ≤ 8192` is marginal for depth 2 at scale 50. | Security below 128 bits in some configurations. | Assert 128-bit parameters in `he_backend` and let the ring scale up. | C11 |
| 9 | Medium | **Identifier handling in the PSI.** The hash to the field is unkeyed, so low-entropy IDs such as SINs can be enumerated. `t` can be as small as about 2^29. Within-bin collisions are silently dropped. `alpha` and `D` reveal O's bin loads. | Mostly R (and anyone holding the IDs). | Keyed hash or OPRF. Fix `alpha` and `D` from public parameters. Report collisions. | C5, C6, review #11 |
| 10 | Medium | **Published model and fitted values.** A published β̃ gives R's own cohort no protection, because `A` is exact. Fitted values need `x_O`, so they are not post-processing of β̃. | Readers of a publication. | Say so in the paper. If R's cohort must also be protected, noise `A` too. | D7 |
| 11 | Low | **Randomness.** Masks and noise come from PCG64, and the Gaussian is float64. A naive discrete Gaussian under encryption also leaks (lattice artefacts). | R. | `secrets` for masks. A discrete or snapped Gaussian designed for the encrypted setting. | C8, D8, A14 |
| 12 | Low | **Analytic Gaussian tail.** `_Phi` returns 0 below x ≈ −8.5, which under-noises only at extreme δ (1.26× at δ=1e-8, ε=20). | — | Use `0.5*erfc(-x/√2)`. | D9 |
| 13 | Low | **Oracle files and logs.** The generator writes the true matching and `y` into the shared data folder, and both parties log statistics. | Anyone with filesystem or log access. | Keep simulation oracles out of party folders. Trim the logs. | A12 |

Not a privacy issue, but relevant: choosing λ from the noisy Gram (Remark 4) breaks the
bias-correction guarantee of Theorem 2 when the ridge is active (review #3).

## Out of scope under honest-but-curious

These need a party to deviate. They are kept because they show how much rests on the
assumption.

- **Malicious R amplifies its inputs** (A1, A2, A5, S1). O cannot check `b ∈ {0,1}` or
  `‖x_R,i‖ ≤ B_R` under encryption. Scaled one-hot probes recover any O row's exact
  `(x_O, y)` in one run. Across about `n_O/d_R` runs they recover the whole register.
- **R enumerates IDs in the PSI** (A3): it can query arbitrary IDs for membership and row index.
- **Malicious O** (A6, A7, A8): fabricated aggregates, poisoned labels, selective failure.
  R cannot detect any of these.

**Engineering hygiene** (outside the cryptographic model, but fix before any real deployment):
- `pickle.loads` on the peer's bytes allows arbitrary code execution (A9, C9).
- There is no TLS or authentication, so public keys can be substituted (A10).
- There is no DoS protection: no timeouts, unbounded messages, and a start-up race (A13).
