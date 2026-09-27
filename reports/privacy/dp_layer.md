# DP / output-privacy layer review: Scenario B (O holds y)

**Threat model: honest-but-curious (semi-honest).** R and O run the protocol exactly as specified, on their genuine inputs. Each then analyses everything it legitimately receives: messages, files, ciphertexts it can decrypt, and its own outputs. Choosing one's own inputs, such as R's cohort and feature set within the committed bounds, is not a deviation. Risks that need a party to deviate are unranked and appear only in the final "Out of scope under semi-honest" section.

**Scope.** This review covers the differential-privacy and statistical-disclosure layer of the working draft (Aug 31 2026: Def. 4, §5, §6.4 Remark 4, §7) and of `scenario_b/` (`calibrate_hyperparameters.py`, `party_r.py`, `party_o.py`, `run_protocol.py`, `generate_vfl_data.py`). The HE and PSI cryptography is only covered where it decides what a party sees.

**Not available:** `phase1_common.py` (noise draw, solve/correct), `he_backend.py` and `psi_common.py`. Items that depend on them are marked *verify*.

**Evidence.** All numbers come from the scripts in `reports/privacy/demos/dp/`. Each script has a `dN_output.txt` with its output, and each runs in 1–25 s.

---

> **Update after the original modules were provided:** D6 (slot case) does not apply to the user's `he_backend.py`. Under OpenFHE every returned slot holds total + noise, and OpenFHE enforces 128-bit parameters. See `../code_review.md` ("The three imported modules") and `../checks/originals/`.

## Executive summary

- **A curious R, and a curious O, read the other side's raw-data statistics straight from `dp_params.json` (D1).** `calibrate_hyperparameters.py` reads both parties' raw data in one process and writes exact data statistics into the one file both parties load:
  - always: 0.99-quantiles and clip rates;
  - when standardizing: exact means and SDs;
  - with suggested bounds: a data-derived σ.

  Adjacent datasets give different files, so ε = ∞ for either party's view. Across two calibration cycles, R recovers a newly added person's y and x_O exactly. Running the calibration at all also requires someone who sees both raw datasets.
- **The noise is calibrated to the wrong adjacency (D2).** Eq. (6) is the add/remove sensitivity, but Def. 4 is replace-one. The exact replace-one sensitivity is between 1× and 2× eq. (6):
  - √2× when the bounds are balanced (√10 vs √5 at B = 1);
  - up to 2× when R's bound dominates (many R features, or R's intercept).

  A claimed ε = 1 actually delivers ε ≈ 1.5–2.0; ε = 8 delivers 12–17. The closed form and a 25-line exact algorithm are below.
- **"DP-valid" labels are unreliable, and nobody checks them (D3).** `--scale_source committed` still computes scales from the data but is labelled `dp_valid=True`. The documented workflow "commits" bounds that were read off the data. Neither party reads either flag.
- **Honest re-runs compose without limit (D4).** There is no ledger anywhere, and `party_o.py` answers every connection with fresh noise. Ten runs at ε = 1 give ε ≈ 3.6 (5.4 under Def. 4). A hundred runs at ε = 4 let R reconstruct individual y at R² ≈ 0.42.
- **DP does not cover the most sensitive disclosure (D5).** An honest R learns exactly which of its cohort are in O's register, with no noise. It may choose that cohort. In the paper's own example (a benefits agency, y = benefit receipt), register membership *is* the sensitive attribute. R also learns each match's row position in O's file, and n_O.
- **Noise channels:**
  - CKKS: R holds sk and decrypts every slot and error polynomial of what it receives. There is no sanitization, and slot contents are unverified (D6).
  - Floating point: the float64 / PCG64 noise is masked under openfhe but exposed on the plaintext backend. The paper's proposed discrete-Gaussian fix, done naively under encryption, recovered all 16 binary outcomes in 14 of 20 trials (D8).
- **Other issues:**
  - The paper's "fitted values inherit DP" and "released model is DP" claims are too broad (D7).
  - The analytic-Gaussian tail numerics under-noise for very small δ or very large ε (D9).
- **Single-run practical risk at realistic bounds is low (D10).** The per-record noise-to-signal ratio is 25–300. A worst-case targeted attack on a binary y reaches 54–61% accuracy at ε = 8, and whole-cohort reconstruction reaches R² ≤ 0.09. In practice, the dangers are the non-DP channels (D1, D5, D6) and repetition (D4).
- **Out of scope under semi-honest:** a deviating R (out-of-bound Xdot, non-binary b, nonzero off-match rows) extracts exact records in one run. See the final section.

| Rank | ID | Title | Severity | Who extracts it (semi-honest) |
|---|---|---|---|---|
| 1 | D1 | `dp_params.json` is a non-private release of both parties' raw data | **critical** | R (O's register stats), O (R's cohort stats); the calibrator sees all |
| 2 | D2 | Sensitivity uses add/remove, not Def. 4 replace-one: σ too small by 1–2× | **high** | R (weaker guarantee on every release) |
| 3 | D3 | Data-dependent bounds/scales labelled DP-valid; parties never check | **high** | R (and O via R-side constants) |
| 4 | D4 | No privacy ledger; honest re-runs compose without limit | **high** | R |
| 5 | D5 | Not covered by DP: membership (PSI output), n, n_O, O's row index | **high** | R |
| 6 | D6 | CKKS output channel: slot hygiene, decryption error, no sanitization (*verify*) | medium (high if slot check fails) | R (holds sk) |
| 7 | D7 | Non-DP downstream outputs; over-broad post-processing claims | medium | whoever receives fitted values / β̃ |
| 8 | D8 | Floating-point Gaussian, non-CSPRNG, unverified `draw_noise`; naive discrete fix leaks | low (now) | R |
| 9 | D9 | `analytic_gaussian_sigma` tail inaccuracy (tiny δ / large ε) | low | R |
| 10 | D10 | Practical single-run reconstruction at the operating points (informational) | low | R |
| — | S1 | Deviating R: unenforced B_R, binary b, zero-off-match Xdot | out of scope | — |

---

## D1 — `dp_params.json` is a non-private release of both parties' raw data

- **Severity:** critical.
- **Who extracts it:**
  - A curious R reads O's statistics.
  - A curious O reads R's statistics.
  - Whoever runs `calibrate()` sees both raw datasets.

  This is inside the model: the file is a legitimate protocol input to both parties, which only have to read it.
- **Location:** `calibrate_hyperparameters.py:243-245` (loads X_R, X_O, y_O together); `:248` (standardization from data); `:257-261` (quantiles always computed); `:272-274` (clip rates); `:299-335` (all written to the params dict); `:368-370` (one output file). Consumers: `party_r.py:26`, `party_o.py:37`.

**Mechanism.** Whoever runs `calibrate()` must hold X_R, X_O and y (all n_O rows of y, not only matched ones). That breaks Def. 1 input privacy before the protocol starts. The output file is then read by both parties. Whatever the flags, it contains:

- `bounds_suggested_from_quantile`: the 0.99-quantiles of ‖x_R‖, ‖x_O‖ and |y|;
- `clip_rate`: the exact fraction of rows beyond each bound.

Depending on the mode, it also contains:

- exact centers and scales (means and SDs of X_R, X_O, y);
- with suggested bounds, B_y, Δ₂ and σ themselves, all functions of the data.

What each party gets:
- **Curious R:** exact statistics of O's whole register, unmatched people included.
- **Curious O:** R's cohort statistics: `center_R`, `scale_R`, the X_R clip rate, the ‖x_R‖ quantile, plus d_R and n_R from `dims`.

**Evidence** (`d6_calibration_leak.py`):
- **Bounds committed, no standardization.** This is the configuration the docstring calls DP-valid. Two Def.-4 neighbours (one matched person's y changed) produce different files: `B_y` quantile 3.8652 → 3.8761 and `clip_rate.y` 0.0495 → 0.0500. That is a deterministic distinguisher, so ε = ∞ for R's view.
- In that run, R reads that exactly 99 of O's people have |y| > B_y, and O reads R's clip rate (4.667%) and ‖x_R‖ quantile (3.3323).
- **Suggested bounds.** The public σ moves with the data (107.72 → 108.01).
- **`center_scale`.** `center_y` equals mean(y_O) to 12 digits. n_O·Δcenter_y recovers the neighbour's change exactly (7.2050708019). If calibration is re-run in a later cycle where O's register gained one person, R recovers that person's y (1.234568 vs 1.2345678) and x_O (error 4e-15).

**Mitigation.**
- Split calibration by party. Each party computes its own clip rates and quantiles locally and prints them only to its own console.
- The shared file may contain only public constants: ε, δ, B_R, B_O, B_y, the standardization constants taken from *public* reference metadata, σ, and HE parameters derived from public sizes.
- Delete `clip_rate` and `bounds_suggested_from_quantile` from `params["dp"]`.
- If data-driven bounds or scales are wanted, estimate them privately with a separate budget, and add ε_std to the ledger (zCDP: ρ_total = ρ_std + ρ_gram). Use a DP quantile (e.g. the exponential-mechanism quantile) and DP mean/SD with crude public ranges.
- O recomputes σ from its own copy of (ε, δ, bounds) and refuses a mismatching file.

**Confidence:** high. The real `calibrate()` was run.

## D2 — Sensitivity is add/remove; Def. 4 is replace-one

- **Severity:** high.
- **Who extracts it:** R. Every release gives R a weaker guarantee than claimed.
- **Location:** paper §5.2 eq. (6) vs Def. 4; `calibrate_hyperparameters.py:112-121` (`joint_sensitivity_B`), used at `:277`.

**Mechanism.** Def. 4 fixes the intersection and x_R, and neighbours differ in one matched person's (x_O, y). The released vector is [triu(B), vec(C), c_R, c_O, yᵀy], with one N(0, σ²) draw per entry (`party_o.py:116-131`). Replacing (x_O, y) = (u, y) by (v, y′), with ‖x_R‖ = r ≤ B_R, gives

```
‖Δ‖² = ‖triu(uuᵀ − vvᵀ)‖² + r²‖u − v‖² + r²(y − y′)² + ‖uy − vy′‖² + (y² − y′²)².
```

Every term except the triu one is invariant under a joint rotation of u and v. Also, ‖triu(M)‖² = ½(‖M‖_F² + ‖diag M‖²) ≤ ‖M‖_F², with equality when M is diagonal, which a rotation achieves. So at ‖u‖ = ‖v‖ = B_O, cos∠(u, v) = c, y = B_y and y′ = t:

```
F(c,t) = 2B_O⁴(1−c²) + 2B_R²B_O²(1−c) + B_R²(B_y−t)² + B_O²(B_y²+t²) − 2B_O²·c·B_y·t + (B_y²−t²)²
Δ_rep² = max_{c∈[−1,1], t∈[−B_y,B_y]} F(c,t)
```

F is concave in c, with c*(t) = clip(−(B_R² + B_y·t)/(2B_O²), −1, 1). This gives closed forms in two of the three regimes.

| regime | Δ_rep² | extremal pair |
|---|---|---|
| balanced, \|B_R² − B_y²\| ≤ 2B_O² | **½(2B_O² + B_R² + B_y²)² + 2B_R²B_y²** | y′ = −y, cos∠ = (B_y² − B_R²)/(2B_O²) |
| R-dominant, B_R² − B_y² ≥ 2B_O² | **4B_R²(B_O² + B_y²)** | x_O′ = −x_O, y′ = −y (C and c_R double) |
| y-dominant, B_y² − B_R² > 2B_O² | 1-D max of the piecewise quartic F(c*(t), t) over t | ratio to eq. (6) ≈ 1 |

The bounds Δ₆ ≤ Δ_rep ≤ 2Δ₆ always hold:
- the lower bound because replacing a record with (0, 0) is equivalent to removing it for every released block;
- the upper bound by the triangle inequality.

At B_R = B_O = B_y = 1 this gives ½·16 + 2 = 10, so Δ_rep = √10 vs √5, confirming established item 1.

**Evidence** (`d1_replacement_sensitivity.py`):
- An assumption-free brute force over the actual triu layout agrees with the exact algorithm on all 12 bound triples (gap 0).
- The two closed forms are exact on 8651/8651 and 5690/5690 random triples in their regimes.
- Ratio Δ_rep/Δ₆ by regime:
  - balanced: 1.07–1.65 (median 1.414);
  - R-dominant: 1.42–2.00 (median 1.945);
  - y-dominant: 1.000–1.41 (median 1.001).
- Effective ε delivered by the code's σ under Def. 4 (δ = 1e-5):

| bounds | ratio | ε claimed → actual |
|---|---|---|
| all 1 | 1.414 | 0.5→0.73, 1→1.47, 4→5.99, 8→12.26 |
| generator data, raw U[0,1], d_R = 10, d_O = 15 | 1.353 | 1→1.40, 8→11.60 |
| standardized + intercept, d_R = 10, d_O = 15 | 1.469 | 1→1.53, 8→12.86 |
| R-heavy, standardized, d_R = 50, d_O = 5 | 1.841 | 1→1.96, 8→17.16 |
| docstring example (3, 3, 20) | 1.001 | unchanged; yᵀy is 95.6% of Δ² here |

**Intercept.** Under `center_scale`, R's intercept column is b. The intercept row of C is Σx_O and c_R[0] is Σy, both pure O-side sums. The code's B_R = √(1 + B_R_rows²) (`:266-269`) correctly covers them for eq. (6). Under replacement, the same substitution is right, because F is increasing in r. But the intercept pushes the calibration towards the R-dominant regime. At B_O = B_y = 1 the ratio is 1.414 without it, 1.54 at B_R_rows = 1 and 1.92 at B_R_rows = 4.

**Mitigation.** Replace `joint_sensitivity_B` with the exact replace-one value (the code is in the fix list). If a closed form is preferred, a slightly loose but always valid choice is Δ = min(2Δ₆, √U), where U = ½(2B_O² + 2B_y² + B_R²)² if B_R² ≤ 2(B_O² + B_y²), and 4B_R²(B_O² + B_y²) otherwise. Also correct eq. (6) in the paper, or change Def. 4 to add/remove, which would then have to cover n and A (see D5).

**Confidence:** high.

## D3 — Data-dependent bounds and scales labelled DP-valid; parties never check

- **Severity:** high.
- **Who extracts it:** R. The data-derived centers and scales flow into its view and into the published intercept.
- **Location:** `calibrate_hyperparameters.py:19-23` (the documented "omit the bounds to get a suggestion … then commit them" workflow); `:64-67` (`_pick` falls back to data); `:83` (`dp_valid = (source == "committed")`); `:357-360` (no CLI to pass constants); `:378-380` (warning keyed on `dp_valid`); `party_o.py` and `party_r.py` (no check of `bounds_committed` / `dp_valid`).

**Mechanism.** Honest parties following the documented workflow end up with a DP-invalid run that is reported as valid.
- With `--scale_source committed`, the centers and scales are still computed from the data, yet the run is labelled `dp_valid=True` and the warning is suppressed. This confirms established item 2; demonstrated in D1.
- `bounds_committed` only means "passed on the CLI". The documented workflow reads the quantiles off the data and then "commits" them, so the bounds, Δ and σ are functions of the data regardless of the flag.
- Clipping happens after a data-derived standardization. Changing one record then moves the center and scale and therefore *every* standardized row, so no per-record sensitivity argument applies. (With genuinely public constants, standardize-then-clip is fine.)
- Neither party reads either flag. A DP-invalid run produces the same `beta_private.csv` and `run_summary.json` as a valid one, and the back-transform (`party_r.py:154-171`) folds the data-derived cy and sy into the published intercept.

**Mitigation.**
- Add `--center_R/--scale_R/--center_O/--scale_O/--center_y/--scale_y` (or a public-constants JSON).
- Set `dp_valid` only when every constant was supplied, and raise if `source=="committed"` and any is missing.
- In `party_o.py`, before Phase 1: `assert P["dp"]["bounds_committed"] and P["standardization"].get("dp_valid", True)`.
- Record both flags in `run_summary.json`.
- Remove the "suggest then commit" advice, or make it a separate DP-quantile step with its own budget.

**Confidence:** high.

## D4 — No privacy ledger; honest re-runs compose without limit

- **Severity:** high.
- **Who extracts it:** R, simply by running the protocol again. Retraining is legitimate; paper §6.6 anticipates "repeated releases, retraining", and §7(iii) lists the ledger as future work.
- **Location:** `party_o.py:110` (`np.random.default_rng()` per connection); there is no state, counter or budget file anywhere in `scenario_b/`.

**Mechanism.** Every honest run releases a fresh Gaussian perturbation of the same statistic vector. k runs are exactly one Gaussian release with sensitivity √k·Δ, and averaging them is sufficient. The same applies across different researchers, cohorts or feature sets that touch the same O individuals. There is no single budget owner.

The yᵀy entry is *not* a separate composition: it sits inside the joint Δ. When B_y dominates, though, it dominates Δ (95.6% of Δ² in the docstring example), which inflates σ for everything.

**Evidence** (`d3_composition_repeat.py`; bounds 1, δ = 1e-5):
- At ε = 1 per run:
  - k = 2 gives ε ≈ 1.47 (2.15 under Def. 4);
  - k = 10 gives 3.62 (5.41);
  - k = 100 gives 14.4 (22.7).
- At ε = 4 per run, k = 10 gives 16.1 (25.5).
- Reconstruction (`d4_reconstruction.py` §4, n = d_R = 20):
  - ε = 1 per run: R² of y goes 0.00 → 0.10 → 0.39 at k = 1, 100, 1000;
  - ε = 4 per run: 0.01 → 0.13 → 0.42 → 0.76 at k = 1, 10, 100, 1000.

**Mitigation.**
- Keep a persistent O-side accountant (zCDP: ρ_run = Δ²/(2σ²)) keyed to the register. The budget is per O-individual and global across researchers, since any O record can be matched.
- Refuse a run when Σρ would exceed ρ_total, and log every release.
- Serve exact repeats of a query from a cache of the *noised output*. That is post-processing and free. Never reuse the noise z across *different* queries, because that enables differencing.
- Every λ, Ψ, bias correction and column subset R needs is post-processing of one release, so none of them should trigger a new run.

**Confidence:** high.

## D5 — Not covered by DP: membership, n, n_O, O's row index

- **Severity:** high.
- **Who extracts it:** R. Def. 4 declares I to be R's "legitimate PSI output", so this is inside the protocol and outside DP by definition.
- **Location:** paper Def. 4, §4.2 ("without R learning X_O or y beyond what the released β̂ implies"), Protocol 2 (τ permutation); `party_r.py:88-95` (R recovers `O's canonical row index`); `party_o.py:69` (labels built from `ids_O` in file order, with no τ visible).

**Mechanism.**
- An honest R learns, exactly and with no noise, which of its identifiers are in O's database. That holds for every member of a genuine cohort.
- Picking its cohort is R's own input choice, not a deviation. The "cohort" can therefore be a list of targets. With n = 1, R learns a single person's membership outright, plus that person's noised record.
- In the paper's own example (O a national agency, y benefit receipt), if O's register is a benefits register then membership is y. More generally it reveals being a client of O.
- R also learns n_O (`dp_params["dims"]`, needed to size Xdot) and each match's row position in O's file. Paper Protocol 2 requires O to apply a uniform permutation τ first. `party_o.py` does not; it may happen in `psi_common` (*verify*). If O's extract is sorted by date, region or outcome, the position leaks that attribute.
- None of this is bounded by ε.

**Mitigation.**
- State plainly that DP covers attribute disclosure for already-matched people only.
- If membership is sensitive, move to PSI-with-computation (circuit-PSI / PSI-sum). There, R does not learn I or b in the clear, b stays secret-shared or encrypted, and n and A are noised. This means Def. 4 becomes add/remove over O's register, and A (which depends on I) must also be protected.
- At minimum:
  - pre-register R's cohort with a legal basis;
  - cap how small n may be;
  - apply τ in `party_o.py` before `bin_and_interpolate`, with random labels rather than file row indices.

**Confidence:** high for the membership point; medium for the row-index point (`psi_common` not seen).

## D6 — CKKS output channel: slot hygiene, decryption error, no sanitization (*verify*)

- **Severity:** medium, but high if the slot check below fails. That case would be a complete bypass by an honest R.
- **Who extracts it:** R. It holds sk, legitimately decrypts everything it receives, and knows the randomness of its own encryptions.
- **Location:** `party_o.py:112-131` (`inner_product` then `add_scalar`, sent as is); `party_r.py:122` (`decrypt_slot0`); paper §7(iv).

**Mechanism.** The DP argument models R's view as s + z. What R actually holds is the full output ciphertext, sk, its own encryption randomness and every slot, which exposes three things:
1. **Slot hygiene.** If `add_scalar` adds z only to slot 0 while `inner_product` leaves the sum, or partial rotate-and-sum windows, in other slots, those slots are released without noise. If z is added to all slots but slot k holds a partial sum, then slot k minus slot k+1 cancels z and yields a single record's term.
2. **Decryption error.** e_ckks is a data-dependent function of rescale rounding of O's plaintext operands. Slot 0 hides it under z (σ ≫ e). The other slots and coefficients do not.
3. **No sanitization.** Without circuit privacy (re-randomization plus noise flooding), the output ciphertext components are a deterministic function of R's known ciphertexts and O's plaintext operands.

None of this needs R to deviate.

**Mitigation.**
- Mask the result to one slot and add independent noise wherever O-data can appear.
- Add a fresh Enc_pk(0) with flooding noise, and mod-switch to the last level before sending.
- Include the CKKS error bound in the sensitivity: Δ′ = Δ + 2√m·e_max.
- Add an integration test that decrypts *all* slots and asserts they carry nothing but s₀ + z (up to e).

**Confidence:** low–medium (`he_backend` not seen).

## D7 — Non-DP downstream outputs and over-broad post-processing claims

- **Severity:** medium.
- **Who extracts it:** whoever receives fitted values or the published β̃.
- **Location:** paper §5.1 ("every downstream function of β̃ — including fitted values — inherits (ε, δ)-DP"), §4.4; `party_r.py:147-176`.

**Mechanism.**
- **Fitted values and residuals.** For *training* individuals these use x_O,i and y_i, which are private inputs, not functions of the release, so post-processing does not apply. If O computes them honestly and hands them to R, R learns x_O,iᵀβ̃_O, and hence x_O,i.
- **R's cohort.** A is exact, so a published β̃ gives R's cohort members no DP protection: a change in x_R,i moves β̃ through un-noised A.
- **O's unmatched records.** They are outside Def. 4. They are protected only cryptographically, and D1, D5 and D6 show where that protection fails even under semi-honest behaviour.

**Mitigation.**
- Restrict the fitted-values claim to new, non-training data. Any per-individual output computed by O counts as a new release.
- State the scope of the DP guarantee as: matched individuals' O-attributes, semi-honest parties, single run.
- If β̃ is to be published, R needs its own noise on A and c_R, with adjacency over full records.

**Confidence:** high.

## D8 — Floating-point Gaussian, non-cryptographic PRNG, unverified noise draw

- **Severity:** low for the current code; the paper's naive fix would be critical.
- **Who extracts it:** R.
- **Location:** `party_o.py:110` (numpy PCG64, float64, via the unseen `phase1_common.draw_noise`); `run_protocol.py:8` (default `--backend plaintext`); paper §7(iv).

**Mechanism.**
- The DP proof assumes exact real-valued Gaussian noise from true randomness. Float64 samplers have data-dependent support gaps (Mironov 2012; Jin et al., S&P 2022).
- Under openfhe these gaps (ulp ≈ 1.4e-14 at 100) are far below the CKKS error (≈1e-12 to 1e-8), so they are masked. Under the plaintext backend, the default in `run_protocol.py`, they are not. That backend presumably offers no input privacy either (*verify*), so it must never touch real data.
- PCG64 is not a CSPRNG, though R never sees raw outputs, so state recovery is impractical.
- *Verify* three things in `draw_noise`:
  - every released coordinate has variance ≥ σ² (a (G + Gᵀ)/2 symmetrization would under-noise the off-diagonals of B by √2);
  - it uses the passed OS-seeded `rng`, not a fixed seed (`party_r.py:66` shows a `default_rng(0)` pattern in the codebase);
  - it draws fresh noise per run.
- **The paper's proposed fix is dangerous under encryption.** O cannot snap the encrypted s to a grid. Adding noise on a grid γ to an un-snapped s releases (s + e) mod γ with no noise, to an honest R. In `d7_lattice_noise_leak.py` (n = 16, binary y, known x_R, σ = 8.3), this recovered every y_i exactly in 14 of 20 trials, against 0 of 20 with continuous noise.

**Mitigation.**
- Sample from a CSPRNG (ChaCha20/AES-CTR from `os.urandom`) with an exact discrete-Gaussian sampler (Canonne–Kamath–Steinke 2020; e.g. OpenDP), at the CKKS plaintext scale.
- Add the noise as an integer-coefficient plaintext, so no float is ever encoded.
- Account for e_max in Δ (D6).
- Never use a grid coarser than the CKKS resolution.

**Confidence:** medium.

## D9 — `analytic_gaussian_sigma` tail numerics

- **Severity:** low.
- **Who extracts it:** R, via a smaller σ than intended.
- **Location:** `calibrate_hyperparameters.py:124-141` (`_Phi` = ½(1 + erf); `_log_Phi` uses it down to x = −20).

**Mechanism.** The criterion itself is exactly Balle–Wang Thm 8, and the bracketing and bisection are correct: they return the side with δ(σ) ≤ δ. But 1 + erf(x/√2) cancels catastrophically in the left tail.

**Evidence** (`d2_analytic_gaussian_check.py`, against `scipy.special.log_ndtr`):
- Relative error of `_Phi`: 2e-6 at x = −7, 1.8% at −8, 30% at −8.25, and exactly 0 at x ≤ −8.5.
- σ matches the reference within 1e-5 for δ ≥ 1e-12 and ε ≤ 8.
- Under-noised cases:
  - ε = 20, δ = 1e-8: δ_actual = 1.26δ;
  - δ = 1e-15 to 1e-17: up to 2.4δ;
  - δ = 1e-20: σ is 7–10% low and δ_actual = 260–2300δ.

**Mitigation.**
- Use `_Phi = lambda x: 0.5*math.erfc(-x/math.sqrt(2))` (or `scipy.special.ndtr`) and `_log_Phi = scipy.special.log_ndtr`.
- After bisection, re-verify δ(σ) with the accurate profile and multiply σ by (1 + 1e-9).

**Confidence:** high.

## D10 — Practical reconstruction at the operating points (informational)

- **Severity:** low.
- **Who extracts it:** R.
- **Location:** paper §6.6; `d4_reconstruction.py`.

**Setup.** A curious R knows X_R exactly and runs a Bayes-optimal linear attack on c̃_R and C̃. Bounds are public 0.99-quantiles, and σ comes from the code. y depends only on X_O, so any recovery is record-level.

**Evidence:**
- **Per-record noise-to-signal** (d_R = 20, d_O = 5): σ = 242, 129, 37 and 21 at ε = 0.5, 1, 4 and 8. One record is 0.003–0.039 noise SDs in c_R. The noise is dominated by the B_O⁴ and B_O²B_y² terms, which an attacker cannot turn against y.
- **Targeted worst case.** R gives the target a dedicated feature column, which is a legal input within B_R. Binary-y accuracy at ε = 8 is 0.54–0.61 (0.57 for d_R = 20, d_O = 5), and 0.51–0.52 at ε = 1.
- **Whole cohort** (n = d_R ∈ {20, 50}): at ε = 8, R²(y) ≤ 0.06 with d_O = 5 and ≤ 0.09 with d_O = 1. Binary-y AUC ≤ 0.60. At ε ≤ 1, everything is ≈ 0.

**Reading.** Per run, the calibrated mechanism is conservative against this attack class. The worst-case μ (1.67 claimed, 2.8–3.1 true at ε = 8) is attained only by record pairs that differ in the B block. Honest re-runs (D4) erase this margin.

The same σ also sets utility. The entry-level noise of 20–240 requires n in the thousands for useful estimates, and it puts small cohorts in the regime where the paper's own Remark 2 expansion fails. That is a validity issue: established item 3 notes that the λ/Ψ gate is privacy-neutral post-processing but breaks Theorem 2.

**Confidence:** medium–high. This covers linear attacks only, and bounds at the distribution quantiles.

---

## Checked and sound (semi-honest)

- A un-noised is correct under Def. 4.
- The bias correction and the ρ/λ/Ψ gate use only released quantities plus A, so they are free post-processing.
- The joint (not per-block) calibration is correct, and yᵀy is inside it.
- Single-draw symmetric noise on triu(B) matches a triu-based sensitivity. The B_O⁴ term is tight for add/remove, and 2B_O⁴ for replacement.
- The intercept is folded in correctly for eq. (6).
- The analytic-Gaussian criterion and root-finding are right (see D9 for the tail numerics).
- O clips x_O and y itself. R's own clipping of x_R (`party_r.py:42`) is sufficient when R is honest.
- Fresh noise per run is correct. Cached noise across different queries would enable exact differencing.

## Prioritized fix list

1. **D1/D3: stop the calibration leak and gate validity.** Split calibration per party. The shared file carries public constants only; drop `clip_rate` and `bounds_suggested_from_quantile`. Add CLI flags for public centers and scales, set `dp_valid` only when all are given, and have `party_o.py` refuse to run unless `bounds_committed` and `dp_valid` are true and it has recomputed σ itself.
2. **D2: calibrate to the Def. 4 (replace-one) sensitivity.** Corrected formula:

   ```
   Δ_rep² = ½(2B_O² + B_R² + B_y²)² + 2B_R²B_y²      if |B_R² − B_y²| ≤ 2B_O²
          = 4B_R²(B_O² + B_y²)                        if  B_R² − B_y² ≥ 2B_O²
          = max_t F(c*(t), t)  (exact 1-D, below)     if  B_y² − B_R² > 2B_O²
   ```

   Here Δ₆ ≤ Δ_rep ≤ 2Δ₆. With R's intercept, B_R = √(1 + B_R_rows²). The ratio to eq. (6) is √2 at equal bounds and approaches 2 when B_R dominates. Drop-in code, verified against brute force (`d1_replacement_sensitivity.py`):

   ```python
   def joint_sensitivity_B_replace(B_R, B_O, B_y):
       r, o, B = float(B_R), float(B_O), float(B_y)
       if o == 0:
           return math.sqrt(max(4*r*r*B*B, B**4))
       F = lambda c, t: (2*o**4*(1-c*c) + 2*r*r*o*o*(1-c) + r*r*(B-t)**2
                         + o*o*(B*B+t*t) - 2*o*o*c*B*t + (B*B-t*t)**2)
       cs = lambda t: min(1.0, max(-1.0, -(r*r + B*t) / (2*o*o)))
       cand = [-B, B, 0.0] + ([(2*o*o - r*r)/B, (-2*o*o - r*r)/B] if B > 0 else [])
       for co in ([4, 0, 2*r*r + 2*o*o - 3*B*B, -r*r*B],
                  [4, 0, 2*r*r + 2*o*o - 4*B*B,  2*B*(o*o - r*r)],
                  [4, 0, 2*r*r + 2*o*o - 4*B*B, -2*B*(r*r + o*o)]):
           cand += [z.real for z in np.roots(co) if abs(z.imag) < 1e-9]
       return math.sqrt(max(F(cs(t), t) for t in (min(B, max(-B, t)) for t in cand)))
   ```

   Also fix eq. (6) in the paper, or redefine adjacency.
3. **D4: add an O-side zCDP ledger.** Make it persistent and global across researchers and cohorts. Serve exact repeats from a cache of noised outputs. Retraining, λ, Ψ and column subsets are post-processing of one release.
4. **D5: scope the guarantee honestly.** If register membership is sensitive, move to PSI-with-computation with a noised n. At minimum, pre-register cohorts, set a minimum n, and apply τ with random labels in `party_o.py`.
5. **D6: sanitize outputs.** Verify `he_backend` slot semantics first. Then mask to one slot, re-randomize with a flooding Enc(0), add the CKKS error to Δ, and add an all-slots decryption test.
6. **D8: use a CSPRNG and an exact discrete Gaussian at the plaintext scale.** Add it as an integer plaintext, verify `draw_noise` (variance ≥ σ² on every released entry, fresh OS-seeded rng), do not use a coarse-grid noise, and keep the plaintext backend away from real data.
7. **D9: fix the tail numerics.** Use `erfc` / `log_ndtr` in `analytic_gaussian_sigma`, with a post-check of δ(σ).
8. **D7: correct the paper's post-processing claims.** Restrict fitted values to non-training data, and give R's cohort its own mechanism if β̃ is published.

---

## Out of scope under semi-honest (not ranked)

**S1 — A deviating R can void the sensitivity assumptions, and O cannot detect it.** The assumptions are ‖x_R,i‖ ≤ B_R, b ∈ {0,1}^{n_O} and Xdot = 0 off the match set. O sees only Enc(b) and Enc(Xdot) and checks none of them (`party_o.py:103-131`). R also generates the CKKS parameters (`party_r.py:110`) and declares d_R.

`d5_unenforced_bounds.py`, a plaintext replay of `party_o.py`'s arithmetic, shows what a deviating R gets:
- With a column scaled by M = 10⁴ and pointed at 10 rows (5 of them unmatched), R reads (x_O, y) for those rows to within 0.017 in one run. The actual μ is ≥ 1130 vs the claimed 0.27.
- b = M·e_i yields y_i² and x_O,i·y_i.
- Xdot = M on all rows yields register-wide sums.

This does not affect the semi-honest analysis, but it means §5.1's "the privacy budget stays under O's control" holds only while R is honest. If the model is ever widened, the remedies are:
- zero-knowledge well-formedness proofs from R;
- or O-side homomorphic clipping;
- plus O-validated CKKS parameters and a cap on d_R.

## Reproducing

```
cd reports/privacy/demos/dp
python d1_replacement_sensitivity.py   # D2  (~25 s)
python d2_analytic_gaussian_check.py   # D9
python d3_composition_repeat.py        # D4
python d4_reconstruction.py            # D4, D10 (~7 s)
python d5_unenforced_bounds.py         # S1 (out of scope)
TMPDIR=<scratch> python d6_calibration_leak.py   # D1, D3
python d7_lattice_noise_leak.py        # D8
```
