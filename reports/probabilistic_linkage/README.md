# Probabilistic record linkage through the OPRF-PSI: validity assessment (2026-10-27)

This note asks whether the protocol's OPRF + labeled-PSI alignment can link records on noisy quasi-identifiers
(name, date of birth, postcode) instead of a clean shared ID. The threat model is honest-but-curious. The full
argument is in `report.pdf` (source `report.tex`); the experiment code is in `experiments/probabilistic_linkage/`.
No existing file was changed.

## Verdict: partially valid

- **Valid: several deterministic, error-tolerant keys per record with an OR rule.** Example keys: nickname-canonical
  name + DOB, NYSIIS/Soundex/Metaphone variants, DOB with day and month unordered, name + postcode. The keys go
  through today's OPRF-PSI with these changes:
  1. a uniqueness filter at O, with random dummies replacing removed keys;
  2. one PSI instance per key type;
  3. a 1:1 resolution step at R that replaces the injectivity abort.

  The DP guarantee is unchanged, provided two conditions hold: no DP-protected variable enters a key, and the
  final alignment gives each of O's positions at most one of R's rows.
- **Not preserved: the disclosure guarantee G4 in its current content.** R learns the key-level match function:
  agreement patterns, positions of *non-cohort* people who share a key, and cross-key profiles of a row.
- **Statistically, false matches attenuate R's own coefficients by about (1 - π)**, where π is the false-match
  rate. O's coefficients are nearly unbiased, because y travels with x_O. The bias-aware interval does not cover
  this bias. The tolerable π shrinks as n and ε grow.
- **Not viable in this design:**
  - LSH keys (MinHash or Bloom-filter bands): they are low-entropy by design, so the uniqueness filter removes them
    or they flood R with candidates.
  - Private Fellegi–Sunter scoring of field agreement: it needs circuit-PSI with secure scoring and resolution,
    which is a different protocol.

## Key numbers

Synthetic world: O holds 10^6 rows; R holds 50,000 records, 40,000 of them in scope. Corruption is realistic:
nicknames, typos, surname change, DOB swaps, moves, missing values, twins and households. Scripts:
`run_linkage.py synthetic`, `febrl`, `scale`.

| linkage | precision | recall | non-cohort positions R learns |
|---|---|---|---|
| exact match on clean QIs (K=1) | 99.99% | 78.3% | 0 |
| multikey, 7 keys, uniqueness filter, ≥1 key | 99.31% | 97.0% | 360 |
| multikey, ≥2 keys | 99.97% | 87.7% | 360 |
| LSH (8 MinHash bands + birth year), filter | 77.9% | 72.4% | 11,777 |
| FS on per-field tokens (labels carry tokens; leaks agreement vectors) | 99.33% | 97.0% | 360 |

- **MNAR world** (corruption doubled for y below the median): exact QIs recall 62.8%; multikey recall 88.7%.
- **FEBRL4** (5,000 + 5,000, heavy corruption): exact QIs recall 42.6%; multikey 79.6% at precision 1.0;
  LSH 72.2% at precision 0.991.
- **Register size** (`scale.json`): multikey precision falls slowly with register size: 99.9% at 10^5, 99.3% at 10^6, 98.3% at 3·10^6 rows (recall 97.2% to 96.5%). On 10^6 rows,
  LSH without the filter returns 607,878 candidate positions for 5,000 queries.

Regression on the linked rows (`tab_regression.tex`):
- **Multikey:** the least-squares ratio for R's slopes is 1.004, against 1.007 with perfect linkage; both are within
  the sampling noise of one cohort. Setting I at ε=1 gives coefficient RMSE 0.021 against 0.020, and the
  bias-aware coverage of R's slopes stays at 0.93–0.96.
- **LSH:** R's ratio falls to 0.766, and the coverage of R's slopes falls to 0.42 at ε=1 and 0.001 at ε=4.

Controlled sweep (`run_sweep.py`; n0 = 40,000, 100 cohorts per cell):
- **Formula against simulation:** R's least-squares coefficient ratio follows the contamination formula
  θ*(π) = ((1-π)M + πM_⊥)^{-1}((1-π)m + πm_⊥) to three decimals: 0.992, 0.950, 0.897 at π = 1%, 5%, 10%.
- **Coverage of R's slopes by the bias-aware interval:**

  | ε | π = 0 | 0.5% | 1% | 2% | 5% | 10% | 20% |
  |---|---|---|---|---|---|---|---|
  | 1 | 0.95 | 0.96 | 0.95 | 0.94 | 0.87 | 0.70 | 0.41 |
  | 4 | 0.95 | 0.92 | 0.91 | 0.84 | 0.54 | 0.30 | 0.04 |

- **Wald interval for θ_λ:** coverage stays near 0.95 in every cell, but its estimand is the contaminated one.
- **30% misses:**
  - at random: no bias; n falls to about 28k, close to n_off = 27k at ε=1, and the ridge turns on in about 80% of
    releases;
  - MNAR, with miss probability ∝ e^{-y}: every coefficient is biased (R's ratio 0.84, O's 0.84).

Cost, from the repo's refreshed cost model at n̄_O ≈ 10^6 (`cost.json`; extrapolations beyond K=1):

| layout | queries | K | O wall time | traffic |
|---|---|---|---|---|
| today's protocol | 20k | 1 | 85 s | 0.46 GB |
| one PSI per key type | 20k | 7 | 7.5 min | 0.9 GB |
| one joint PSI over all keys | 20k | 7 | 15.8 min | 1.5 GB |
| one PSI per key type | 100k | 7 | 19 min | 2.8 GB |
| one joint PSI over all keys | 100k | 7 | 3.2 h | 18 GB |

The joint layout grows like K², because both the cuckoo tables and the partitions per bin grow with K; the
per-type layout grows like K. Use one instance per key type.

## What would have to be added to the protocol

1. Public key definitions, normalisation and nickname tables, and a public K. R pads its query to K keys per
   padded record.
2. K PSI instances, one per key type. O applies a uniqueness filter and refills removed or missing keys with random
   dummies, so the PSI sizes stay public.
3. A resolution step at R in place of the injectivity abort: FS weights with u estimated from R's own file, then
   greedy 1:1 matching, dropping ties. The final alignment must be injective on O's positions. With m R rows per
   position, the sensitivity becomes up to m·Δ.
4. A restated G4 (R learns the key-level match function L_K), accepted in the disclosure review.
5. Linkage-error reporting. The intervals are conditional on the linkage, and the bias-aware β interval does not
   cover linkage bias.

## Open problems

- A trustworthy false-match-rate estimate without access to O's data, and intervals that remain valid under
  linkage error. A moment correction needs π and a DP release of O's scope means.
- Hiding agreement patterns while keeping R's exact A. This needs circuit-PSI.
- A deviating R that spends its K·n_R^max query budget on low-entropy keys.
- Non-exchangeable false matches, e.g. within households or between twins.

## Method notes

- **Why plaintext simulation is exact.** The OPRF is injective on its domain up to negligible collisions, and the
  labeled PSI returns, for each queried item, the label of the identical item O inserted. Its output on the keys is
  therefore a dictionary lookup, up to the PSI's correctness failures (≤ δ_psi + p_pass ≤ 2^-40), and
  `linkage.psi_output` computes exactly that.
- **Regression path.** The regression uses the repo's plaintext path, as in `experiments/large/sim_large.py`:
  scommon stage 1, v31_eval stage 2, and `vfr/solve.stage2_infer` for the intervals. Calibration uses
  `_calibrate_plain`; OpenFHE is not installed here.
- **Data.**
  - Name frequencies, corruption rates and household structure are modelling assumptions.
  - Quasi-identifiers are independent of the regression variables, except in the MNAR world.
  - FEBRL4 CSVs are read from an installed `recordlinkage` package and are not copied into the repo.
- **Dependencies.** The scripts need `jellyfish` (phonetic codes) and `matplotlib` (figure), installed for this
  work. For FEBRL4, the `recordlinkage` package supplies the CSVs.
- **Files.**
  - `synthetic_s1.json`, `synthetic_s2_mnar2.json`, `febrl.json`, `scale.json`, `sweep.json`, `cost.json`: results.
  - `tab_*.tex`, `numbers.tex`, `fig_sweep.pdf`: generated by `experiments/probabilistic_linkage/make_assets.py`.
