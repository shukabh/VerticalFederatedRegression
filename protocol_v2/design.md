# Scenario B, revised protocol

Diagram: [`protocol_v2.pdf`](protocol_v2.pdf) / [`protocol_v2.png`](protocol_v2.png). The TikZ
source is [`protocol_v2.tex`](protocol_v2.tex); build it with `pdflatex protocol_v2.tex`.

This revision keeps the draft's structure: PSI, then R's aggregation under its own CKKS key,
then O's noise, then R's solve with its exact local block `A`. It fixes every critical and high
risk found in the reviews (`reports/privacy/README.md`, `independent_review/`), under the same
**honest-but-curious** model. The one exception is membership, which the draft releases on
purpose; see [What it does not fix](#what-it-does-not-fix).

## Risks and fixes

| # | Risk (source) | Fix | Cost |
|---|---|---|---|
| 1 | **Circuit privacy (critical).** `EvalAdd(ct, float(z))` leaves c1 of O's reply independent of the noise, so R distinguishes neighbouring datasets with certainty, unmatched records included (independent review, d1). O's BFV PSI replies aren't sanitized either. | CKKS: O adds a **fresh `Enc_pk(z)`**, which re-randomises c1, then **floods the error** with a Gaussian calibrated so the error channel is (ε_e, δ_e)-DP per record, then switches to the last modulus level. BFV: **statistical flooding** (κ = 40) plus a modulus switch. | CKKS: a few extra modulus bits, plus a small ε_e in the budget. BFV: about κ + log B_err ≈ 60–70 more modulus bits. Ring 16384 still fits for t ≲ 2^40; larger t needs 32768. |
| 2 | **Calibration leak (critical) and the `committed` bug (high).** One process reads both datasets and writes data-derived statistics into the shared `dp_params.json`. | **Per-party commitments.** R standardizes with its own constants, which never leave R, and commits `B_R`. O standardizes with **public** constants (or DP-released ones under a separate budget) and commits `B_O` and `B_y`. Only public constants cross. Refuse to run otherwise. | None. Some utility, if the public constants are cruder than sample estimates. |
| 3 | **Sensitivity (high).** Eq. (6) is add-one but Def. 4 is replace-one. | σ from **Δ_rep** (below). | σ rises by 1–2× Δ₍₆₎: +4% at the accuracy study's bounds, up to +41% at equal bounds, close to 2× when `B_R` dominates. |
| 4 | **No privacy ledger (high).** | O keeps a **zCDP ledger** per (dataset, researcher) and refuses a run if ρ_used + ρ_run > ρ_cap, with ρ_run = Δ_rep²/2σ² + ρ_err. | Caps the number of re-runs, which is the point. |
| 5 | **Row index and PSI metadata (high, partial).** Labels were file row indices, and `D` and `α` depended on O's IDs. | O applies a **secret uniform permutation τ** before the PSI, so labels are τ-positions. It pads every bin to a **public** `D` and `α` computed from `n_O`, and uses dummy root `t−1` for the calibrated `t` (fixes review M2). | PSI evaluation grows to the public degree: about 2–6× at `d_cap = 64`. A max-load quantile from `n_O` is tighter. |
| 6 | **λ chosen from the noise (validity).** Theorem 2 needs a fixed λ, but Remark 4 picked it from G̃. There is also the O-block ceiling blow-up (M1). | **Fix λ and Ψ from public inputs before decrypting:** `λ = max{0, ρ*·2σ√p − n·ℓ}` with Ψ = I. Here ℓ is a committed public lower bound on λ_min(G/n) for standardized data; ℓ = 0 gives the always-valid certificate. The check ρ̂ ≥ 1 on G̃ is post-processing, and the theorems are stated conditional on it (independent review C7). | Over-shrinks when ℓ is conservative. |
| 7 | **Floating-point noise and weak RNG (medium; folded in).** | Discrete Gaussian `N_Z(0, σ²)` (Canonne–Kamath–Steinke) from a CSPRNG, for both the DP noise and the masks r₁, r₂. | None. |

Also changed: `Enc(Ẋ_R)` is sent in column chunks of at most the slot count, which lifts the
`n_O ≤ 65,536` cap. The channel is mutually authenticated TLS with structured serialization and
no `pickle`.

## Protocol

**Phase 0: setup and commitments.** Only public constants cross.
1. **R:**
   - standardizes `X_R` with its own constants, kept local;
   - clips rows to `B_R` (intercept included);
   - generates BFV and CKKS keys at 128-bit, with modulus headroom for flooding.

   R → O: `pk`, `evk`, `B_R`, `d_R`.
2. **O:**
   - standardizes `X_O` and `y` with public constants;
   - clips to `B_O` and `B_y`;
   - computes `Δ_rep` and `σ = AnalyticGauss(Δ_rep, ε_msg, δ)`;
   - checks the ledger.

   O → R: `σ`, `B_O`, `B_y`, `d_O`, `n_O`, the public `D` and `α`, and the hash key `K`.

**Phase 1: private record linkage** (BFV labeled PSI).
1. **O:** applies a secret permutation τ to its rows and uses τ-positions as labels. It simple-hashes `H_K(ID_O)`, pads bins to the public `D` and `α`, and uses dummy root `t−1`.
2. **R:** cuckoo-hashes `H_K(ID_R)` and sends encrypted windowed powers `Enc(y^{2^i})`.
3. **O:**
   - evaluates `P(y)` and `L(y)` per partition (ciphertext × plaintext);
   - draws masks r₁, r₂ from a CSPRNG and forms `s = r₁P(y)`, `q = r₂P(y) + L(y)`;
   - applies `Sanitize_BFV` to both and sends them.
4. **R:** decrypts to get `I = {k : s_k = 0}` and `k ↦ j = q_k`, where j is a uniformly random
   τ-position. It builds `b` and `Ẋ_R` in τ-order and computes `A = Ẋ_Rᵀ Ẋ_R` locally.

**Phase 2: encrypted aggregation and DP noise** (CKKS, depth 1).
1. **R → O:** `Enc(b)` and `Enc(Ẋ_R)` in column chunks.
2. **O:**
   - computes `Enc(B)`, `Enc(C)`, `Enc(c_R)`, `Enc(c_O)`, `Enc(yᵀy)` by inner products (EvalSum over
     the batch, so the total sits in every slot);
   - draws one discrete-Gaussian `z` per released entry (symmetric on `B`), the **same value in
     every slot**;
   - sets `ct ← Sanitize_CKKS(ct ⊞ Enc_pk(z))`;
   - updates the ledger and sends.

**Phase 3: solve at R** (post-processing only).
1. Fix λ and Ψ from public inputs.
2. Decrypt slot 0, and assemble `G̃ = [[A, C̃], [C̃ᵀ, B̃]]` and `c̃ = [c̃_R; c̃_O]`.
3. Compute `β̃_λ = (G̃ + λΨ)⁻¹ c̃` and `β̂_bc = β̃_λ − σ² M(P̃_λ) β̃_λ` (Theorem 2, given ρ̂ ≥ 1).
4. Release `β̂_bc`, together with ρ̂, the shrinkage `−λP̃_λΨβ̃_λ`, and `RSS̃ = ỹᵀy − c̃ᵀβ̃_λ` floored at zero.

## Parameters

**Sensitivity (replace-one, `x_R` fixed).** The released vector is `(triu B, C, c_R, c_O, yᵀy)`.
The closed form was derived in `reports/privacy/dp_layer.md`, where `joint_sensitivity_B_replace`
is a drop-in replacement:

```
Δ_rep² = ½(2B_O² + B_R² + B_y²)² + 2B_R²B_y²    if |B_R² − B_y²| ≤ 2B_O²
       = 4B_R²(B_O² + B_y²)                      if  B_R² − B_y² ≥ 2B_O²
       = max_t F(c*(t), t)                       otherwise (then Δ_rep ≈ Δ₍₆₎)
```

It always satisfies Δ₍₆₎ ≤ Δ_rep ≤ 2Δ₍₆₎. Numerical checks (`reports/checks/`):

| Bounds | Δ₍₆₎ | Δ_rep |
|---|---|---|
| `B_R = B_O = B_y = 1` | √5 | √10 |
| `B_R = 3.16`, `B_O = B_y = 3` | 20.6 | 29.4 |
| `B_R = B_O = √5`, `B_y = 5` (accuracy study) | 30.4 | 31.6 |

**Keeping `yᵀy` is free under replace-one.** The worst case swaps `(x_O, y)` for `(−x_O, −y)`,
which leaves `y²` unchanged. Numerically, Δ_rep is identical with or without `yᵀy` for all three
bound settings above (`protocol_v2/check_sensitivity_no_yty.py`). The independent review's C6
(dropping `yᵀy` shrinks σ by 1.76×) holds only under add-one adjacency.

**CKKS flooding.** After the fresh `Enc_pk(z)`:
- c1 is computationally uniform (RLWE, with O's fresh randomness).
- R's decryption returns `Δ·(s + z) + e_data + e_fresh + e_fl`.
- The only data-dependent term is `e_data`. It depends on O's plaintext columns through R's
  known encryption error, the key-switching error and rounding. Let `S_err` be its per-record
  ℓ₂ sensitivity; this is a bound on how much one record can move it.
- O adds `e_fl ~ D_{Z^N, τ}` with `ρ_err = S_err² / 2τ²` (zCDP). This is the DP-based flooding
  analysis of Li, Micciancio, Schultz and Sorrell (CRYPTO 2022), applied here to the
  evaluator-to-decryptor direction.

Because `S_err` is roughly the size of the error, τ needs only a few bits above it, so the
decoded perturbation `e_fl/Δ` stays far below σ. Classical statistical flooding (2^40 × the
error) would need a scale beyond 2^60 to keep precision, which the 64-bit OpenFHE build does not
provide.

**Why the noise must be the same in every slot.** EvalSum replicates the total into all N/2
slots. If each slot carried independent noise, R could average them and remove it. So `z` must
be one value shared by every slot. The flooding term does perturb each slot independently, but
only by about `τ/Δ`, which is tiny. Averaging it away recovers `s + z`, never `s`.

**BFV flooding.** BFV decrypts exactly (the rounding removes the error), so plain statistical
flooding works: `‖e_fl‖ ≈ 2^40·B_err`, followed by `ModSwitch₀`. The cost is only modulus bits.

**Ledger.** zCDP adds up across channels and runs:
- ρ_run = Δ_rep²/2σ² + ρ_err.
- ρ_cap is set by O's governance.
- Convert to (ε, δ) with ε = ρ + 2√(ρ ln(1/δ)).

## Privacy argument (sketch)

- **O's view:** public constants and ciphertexts under R's key. That view can be simulated from
  public sizes (IND-CPA; OpenFHE enforces 128-bit parameters). This is unchanged from the draft.
- **R's view in Phase 1:**
  - After flooding, the replies depend only on their decrypted values: `s = 0` on matches and
    uniform otherwise; `q` = a τ-position on matches and uniform otherwise.
  - That can be simulated from `(I, σ_τ)`, the declared PSI output, where τ makes the positions
    uniform.
  - `D` and `α` are public.
- **R's view in Phase 2:**
  - c1 is uniform.
  - The message is `s + z`, a Gaussian mechanism with Δ_rep.
  - The error channel is DP through flooding. It also covers O's unmatched records, which enter
    `e_data` through `Enc(b)`'s error but not the message.
  - The whole transcript is therefore (ε, δ)-DP with respect to one O record, with the budget
    tracked by the ledger.
- **Phase 3:** post-processing.

The proof obligations are listed below.

## What it does not fix

- **Membership and n.** R still learns which of its cohort appear in O's register (Def. 4 makes
  this a release). A membership-private variant exists, but it gives up the draft's key idea:
  - Use circuit-PSI (e.g. OPPRF with a secret-shared equality test) so that neither party
    learns `b`, and compute the Gram in shares.
  - Then `A` depends on O's membership data and must be noised too. The adjacency becomes
    add/remove a person from O's register, so the sensitivity gains a `B_R⁴` term, `n` becomes a
    DP count, and the O-block-ridge advantage is gone.
  - Worth it only when membership itself is sensitive, as in a benefits register.
- **Utility at small cohorts** (independent review C5) is fundamental. It is mitigated by
  committed standardization (the largest lever), by choosing λ through an MSE rule on public
  inputs, and by larger ε or cohorts.
- **Malicious parties** are out of scope. O cannot check `Enc(b)` or `Enc(Ẋ_R)` without
  zero-knowledge proofs.
- **Proof obligations:**
  - a bound on `S_err` for the CKKS error channel;
  - conditional-expectation statements of Theorems 1–2;
  - zCDP accounting for the discrete Gaussian.

## Mapping to the code

| File | Change |
|---|---|
| `calibrate_hyperparameters.py` | Split into `calibrate_r.py` (`B_R`, private standardization) and `calibrate_o.py` (`B_O`, `B_y`, public constants, Δ_rep, σ, ledger check). Remove all data-derived output. |
| `phase1_common.py` | Replace-one `joint_sensitivity_B`. Discrete-Gaussian `draw_noise` from `secrets`. A deterministic λ rule replaces `select_ridge`/`auto_lambda`. Release the shrinkage and `RSS̃`. |
| `he_backend.py` | `add_scalar` → `add_fresh_encryption(z)` + `sanitize()` (flooding + `ModSwitch` to level 0) for CKKS and BFV. `ct_from_bytes` and public blobs without `pickle`. |
| `psi_common.py` | Keyed `H_K`. `DUMMY_ROOT = p − 1` per call. Pad to public `D` and `α`. Count and report collisions. |
| `party_o.py` | τ permutation on load. `secrets` for masks. Sanitize the PSI replies. Ledger file. Fix the `D == 1` label bug. |
| `party_r.py` | Chunked `Enc(Ẋ_R)`. λ fixed before decryption. Report ρ̂, the shrinkage and `RSS̃`. |
