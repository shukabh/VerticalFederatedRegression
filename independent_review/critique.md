# Independent review — Scenario B (O holds y): corrected VFL-OLS protocol

Reviewer: independent cryptographer / DP-statistics expert. Inputs: the working-draft paper
(`paper_scenarioB.txt/.pdf`) and the implementation in `scenario_b/` only. All claims below
that are quantitative are backed by a script in `independent_review/demos/` (outputs saved
alongside, `*.out`). Threat model as intended by the authors: honest-but-curious R and O.

---

## 1. Overall verdict

**The privacy claim is not sound as it stands, and it fails in the very place the paper does
not look: the object R actually receives.** The DP analysis reasons about the *decrypted scalar*
`s + z`, but R's view is a full CKKS ciphertext whose error/structure depends deterministically
on O's private data, and R holds the secret key. In the code the DP noise is added with
`EvalAdd(ct, float(z))` (a plaintext-scalar add, `he_backend.py:248`), which does not even
re-randomise the ciphertext: R recovers O's data-dependent part exactly and can distinguish
adjacent O-datasets with probability 1 (demo `d1`, 100% vs the ~73% an ε=1 view would permit).
So Definition 1 (input privacy for R's view of O) and Definition 3 (output DP) both fail on the
transcript, and neither the paper's eq. (5) (fresh `Enc(z)`) fixes it without noise flooding —
which the paper defers to future work (item iv) and the code never implements. **The theory,
by contrast, is largely sound.** The single-draw finite-noise bias operator of Theorem 1/2
(including the non-obvious `−P diag(P)Π_O` single-draw correction) matches Monte-Carlo to
0.6–1.7% at ρ=4–8 and matches the code (demo `d4`); the Balle–Wang σ is implemented correctly
(demo `d7`); the "keep A local" correction and the joint-sensitivity calibration are right. The
remaining defects are: a genuine adjacency/sensitivity mismatch that understates ε by up to ~1.5×
(demo `d3`); a standardization path that silently voids ε,δ (demo `d5`); O-statistics leaked to R
through the shared params file (demo `d5`); and a privacy/utility trade-off that is not viable at
the small-to-moderate cohorts the paper explicitly targets (demo `d6`).

---

## 2. Critique table

| ID | Title | Area | Severity | Fixability | Fix effort |
|----|-------|------|----------|-----------|-----------|
| C1 | HE circuit-privacy gap: DP noise doesn't protect the ciphertext R decrypts | crypto | **critical** | protocol change (noise flooding / re-encryption) | high |
| C2 | Sensitivity uses add-one; Def. 4 declares replace-one → ε understated up to ~1.5× | DP | **high** | fixable in place | low |
| C3 | `--scale_source committed` derives standardization from the sample, stamps `dp_valid=True` | implementation/DP | **high** | fixable in place | low |
| C4 | `dp_params.json` (read by R) carries O-data statistics; calibration is a trusted-3rd-party step | linkage/impl | **high** | fixable in place | medium |
| C5 | Shrinkage-dominated, near-useless estimates at target cohort sizes | utility | **high** | fundamental (mitigable) | medium |
| C6 | Bundling yᵀy (B_y⁴ term) inflates σ ~2× for the point estimate | DP/utility | medium | protocol change (split budget) | low |
| C7 | Theorem 1 "bias" is a conditional expectation; unconditional mean doesn't exist | theory | medium | fixable in place (wording) | low |
| C8 | ρ-gate picks λ for validity, not MSE — over-shrinks vs Prop. 1's interior λ | utility/impl | medium | fixable in place | low |
| C9 | PSI reveals (D, α) = max bin load of O's ID set to R, beyond n | linkage | low | fixable in place | low |

---

## 3. Critiques in detail

### C1 — HE circuit privacy: the DP noise does not protect the object R receives (CRITICAL, crypto)

**What is wrong.** Phase 4/5 (paper §4.3, eq. (5)) has O add DP noise "in the encrypted domain"
and send the ciphertexts to R, who holds `sk`. The DP analysis (paper §5, §6) treats the release
as the decrypted scalars `˜s = s + z`. But R's *view* is the ciphertext, and a semantically-secure
HE ciphertext is **not circuit-private**: the noise/error polynomial and (for a plaintext-scalar
add) the second component carry information about O's plaintext inputs beyond `s + z`.

In the code the injection is `add_scalar`, i.e. `EvalAdd(ct, float(z))`
(`he_backend.py:248`, plaintext mirror `:185`; called at `party_o.py:305,309,313,316,318`). A
plaintext-scalar add touches only `c0`. The whole `c1` component of O's reply is a *deterministic*
function of R's own `Enc(b)`/`Enc(Ẋ_R)` and O's plaintext columns — independent of `z`.

**Why it matters (who learns what).** R is the DP adversary. R knows all of O's data except one
record (the DP model), holds `sk`, and generated the input ciphertexts. R recomputes O's
deterministic circuit on each candidate dataset with its own keys and compares `c1` byte-for-byte;
the DP noise `z` never entered `c1`, so R identifies the true dataset with certainty. This means
the transcript R receives is **not (ε,δ)-DP** (Def. 3 fails), and R's view is **not simulatable
from R's input and the agreed output** (Def. 1 fails for R's view of O's data). The plaintext
value `s + z` is fine; the *ciphertext handed to R* is not.

**Evidence (`demos/d1_ckks_circuit_privacy.py`, real OpenFHE):**
- T3: `c1(reply)` is identical for two different noise draws `z, z'` → noise lives only in `c0`.
- T4: over 20 trials with fresh DP noise each time, R distinguishes adjacent O-datasets in
  **100%** of trials — for a **matched** record (b=1, the Def. 4 adjacency) *and* for an
  **unmatched** record (b=0), whose data a correct mechanism would hide entirely. A true
  (ε=1,δ)-DP view caps success at (eᵉ+δ)/(1+eᵉ) ≈ 73%.
- T5: replacing `add_scalar` with a fresh `Enc(z)` (paper eq. (5) literally) re-randomises `c1`,
  so the byte test is defeated (0/20) — but this only closes the `c1` channel, not the error
  channel. Standard BFV/BGV/CKKS ciphertexts remain non-circuit-private: the residual decryption
  error in O's reply is a function of O's inputs, readable by R with `sk`.

**The fix and its cost.** Circuit privacy is required, not optional:
1. **Noise flooding.** Add encrypted DP noise whose *ciphertext error* statistically drowns the
   data-dependent error: the smoothing/flooding term must exceed the honest error by a factor
   ~2^κ (κ≈40) *and* carry the DP Gaussian. This is exactly the paper's deferred item (iv), but it
   is a prerequisite for Defs 1&3, not a precision nicety. Cost: larger CKKS coefficient modulus
   (more RNS limbs → bigger ciphertexts, slower), and the DP/flooding noise must be reconciled in
   one distribution (discrete/snapped Gaussian with a flood tail). This raises the effective σ
   floor, worsening utility (already marginal — see C5).
2. **Re-encryption / key-switch to a fresh key**, or **bootstrap**, before release — removes the
   data-dependent error but is heavier and still needs DP noise added afterwards.
3. Alternatively, have O compute and release **only** the noised scalars through a circuit-private
   functionality (e.g., a small MPC reveal), so R literally receives `s + z` and nothing else.

**Confidence: very high** for the as-implemented break (empirically demonstrated); **high** that
eq. (5) alone is insufficient (this is the textbook non-circuit-privacy of RLWE HE; the DP-over-HE
literature requires flooding for exactly this reason). If fixed via flooding, it creates the new
problem in C5 (utility) by adding to σ, and the flood distribution must itself be finite-precision
safe. This is the single most important finding.

---

### C2 — Sensitivity is for add-one; the declared adjacency is replace-one (HIGH, DP)

**What is wrong.** Def. 4 (paper §3) fixes the intersection and says two datasets "differ in a
single matched individual's O-side data, i.e. in one pair (x_O,i, y_i)" — **replacement** with n
held fixed. But eq. (6) (paper §5.2; `calibrate_hyperparameters.py:112`, `phase1_common.py:371`)
sets `ΔB = x_O x_Oᵀ, ΔC = x_R x_Oᵀ, Δc_O = x_O y, Δc_R = x_R y, Δ(yᵀy) = y²` — the contribution of
**adding/removing** one record. Replacement changes each block by a *difference of two* such terms
(e.g. `ΔB = x xᵀ − x' x'ᵀ`), whose ℓ₂ norm is larger.

**Why it matters.** The Gaussian σ is calibrated to the too-small add-one Δ₂, so the true ε is
larger than the nominal one — the (ε,δ) guarantee is quantitatively wrong.

**Evidence (`demos/d3_sensitivity_replacement.py`).** Maximising the ℓ₂ norm of the *released*
vector `[triu(B), C, c_R, c_O, yᵀy]` under replacement (random restarts + Nelder–Mead, plus
closed-form witnesses), against eq. (6):

| B_R | B_O | B_y | Δ eq.(6) | Δ replace | ratio | nominal ε | actual ε (replace) |
|----:|----:|----:|---------:|----------:|------:|----------:|-------------------:|
| 1.0 | 1.0 | 1.0 | 2.236 | 3.162 | 1.41 | 1.00 | **1.47** |
| 3.3 | 3.3 | 2.6 | 20.74 | 30.41 | 1.47 | 1.00 | **1.52** |
| 2.0 | 2.0 | 1.0 | 6.403 | 9.618 | 1.50 | 1.00 | **1.57** |

Closed-form witness at B=1 (any d_O≥2): x'⊥x, y'=−y gives `Δ = √10 = 3.162` vs eq. (6) `√5`, a
factor √2. General bound `Δ_replace ≤ 2·Δ_add`.

**The fix and its cost.** Either (a) keep replace-one and use the correct replace sensitivity —
tightest is a per-block maximisation, or the safe closed form `Δ²_replace = 2·[(B_O²+B_R²)(B_O²+B_y²)+B_y⁴]`
minus the diagonal-double-count savings; simplest safe choice: multiply the squared sensitivity in
`joint_sensitivity_B` by 2 (√2 on σ). Or (b) switch Def. 4 to add/remove-one — but then n changes
between adjacent datasets and n is released to R, so add/remove adjacency would additionally require
protecting n (conflicts with the stated PSI output), which is worse. Recommended: (a), tight replace
bound. Cost: σ up by √2–~1.25× → more noise, worse utility (compounds C5). **Confidence: high**
(explicit witnesses + numeric maximisation).

---

### C3 — `--scale_source committed` is not committed; it voids ε,δ silently (HIGH, implementation/DP)

**What is wrong.** `standardization_block` (`calibrate_hyperparameters.py:42`) sets
`dp_valid = (source == "committed")` (`:83`) but, when the caller does not pass explicit
centres/scales, computes them from the sample: `_pick(center_R, X_R.mean(axis=0))` etc.
(`:72–77`). Crucially the CLI has **no flags** to supply committed constants (`:353–360` only
expose `--standardize` and `--scale_source`), so `--scale_source committed` *always* uses this
sample's means/SDs yet stamps them DP-valid.

**Why it matters.** Standardizing by data-derived moments makes the released statistics (and the
sensitivity) a function of the private data, breaking the (ε,δ) claim — while the tool reports it
as valid and prints no warning (the warning at `:378` only fires for `data_suggest`).

**Evidence (`demos/d5_implementation_leaks.py`).** With `--standardize center_scale
--scale_source committed`: `dp_valid=True`, yet `center_O` equals O's exact sample means,
`scale_O` equals O's exact sample SDs, and `center_y/scale_y` equal O's y mean/SD.

**The fix and its cost.** Add `--center_R/--scale_R/--center_O/--scale_O/--center_y/--scale_y`
CLI args; make `standardization_block` raise (or force `dp_valid=False`) if `source=="committed"`
but any value is missing. Zero runtime cost. Alternatively implement the documented `"dp"` route
(estimate moments under a separate budget ε_std and compose). **Confidence: very high** (direct).

---

### C4 — `dp_params.json` leaks O statistics to R; calibration is a trusted-third-party step (HIGH, linkage/impl)

**What is wrong.** `calibrate_hyperparameters.py` reads X_R, X_O and y_O in one process
(`:243–245`) and writes `dp_params.json`, which `party_r.py` loads at startup (`party_r.py:26`).
That file contains non-DP functions of O's raw data: `bounds_suggested_from_quantile` (0.99
quantiles of ‖x_O‖ and |y|, `:257–261`), `clip_rate` for X_O and y (`:271–274`), and — when
standardizing — O's per-column means and SDs (`center_O`, `scale_O`, `center_y`, `scale_y`).

**Why it matters.** R learns O's covariate norms, response scale, tail (clip rate) and per-feature
moments outside any DP accounting — an input/linkage-privacy leak of O's distribution. Separately,
the calibration as shipped is a single trusted process holding both raw datasets, which no party is
allowed to be; in a real deployment each party must commit bounds over its own data.

**Evidence (`demos/d5`).** The loaded params contain `bounds_suggested_from_quantile =
{B_R:3.14, B_O:3.16, B_y:2.51}`, `clip_rate = {X_O:0.031, y:0.0015}`, and `center_O =
[0.494, 0.503, 0.502, 0.507, 0.508]` — all exact O-sample statistics.

**The fix and its cost.** Split calibration per party: O commits (B_O, B_y) and any O-side scales
over O's data and publishes only the *committed constants*; R commits B_R. Strip
`bounds_suggested_from_quantile`, O `clip_rate`, and O moments from the file R receives (keep them
in an O-private diagnostics file). Compute σ from committed public bounds only. Cost: a little
plumbing; removes a genuine leak. **Confidence: high.**

---

### C5 — Not viable at the target cohort sizes: shrinkage-dominated estimates (HIGH, utility)

**What is wrong.** The paper targets "the small-to-moderate matched-cohort regime typical of
research-data-centre use" (§6.6). In that regime the validity gate (Remark 2, ρ>1) forces a large
ridge λ (to lift λ_min(G_λ) above the noise edge 2σ√p), and Prop. 1's deterministic shrinkage
`−λ P_λ Ψ β̂` dominates. The bias correction (§6.5) removes only the O(σ²) *mechanism* bias, not
the shrinkage. So the deliverable is a heavily shrunk estimate, and the correction is irrelevant.

**Why it matters.** The headline "zero-budget bias correction" fixes the negligible part while the
estimate is swamped by shrinkage. The estimator is not usable at the sizes claimed.

**Evidence (`demos/d6_utility_sweep.py`, authors' own algebra; and `d4` part D).** Relative error
`‖β_bias-corrected − β_OLS‖ / ‖β_OLS‖`, median over 60 noise draws, d_R=d_O=5, δ=1e-5:

| design | ε | n | σ | ridge | λ | rel.err |
|:--|--:|--:|--:|:--:|--:|--:|
| raw | 1 | 400 | 154 | I | 3276 | **0.99** |
| raw | 1 | 2000 | 154 | I | 3276 | **0.94** |
| raw | 1 | 10000 | 154 | I | 2184 | 0.73 |
| raw | 1 | 50000 | 154 | O | 0 | 0.12 |
| std | 1 | 400 | 77 | I | 1456 | 0.83 |
| std | 1 | 2000 | 77 | O | 0 | 0.28 |
| std | 4 | 2000 | 22 | O | 0 | 0.08 |

At ε=1 with raw moments the estimate is ~100% off until n≈50k. `d4` part D shows, in the
noise-dominated regime, the shrinkage bias `‖β̂_λ−β̂_OLS‖=1.645` vs `‖β̂_OLS‖=2.03` (80% shrink),
while the mechanism-bias correction moves the estimate by only ~0.008.

**The fix and its cost.** This is close to **fundamental** — σ is O(1) in n but its constant is
large, so utility needs n≫σ². Mitigations (all recommended, none free):
(i) **standardization with genuinely public/committed scales** (C3) — the single biggest lever
(raw→std cuts required n by ~10×); (ii) choose λ to minimise estimated MSE, not just clear ρ (C8);
(iii) drop yᵀy from the estimator's release (C6); (iv) state plainly that the method needs
moderate-to-large matched cohorts (thousands to tens of thousands at ε=1), contradicting the
current framing. **Confidence: high.** The authors should reframe §6.6 honestly.

---

### C6 — Bundling yᵀy inflates σ for the point estimate (MEDIUM, DP/utility)

**What is wrong.** yᵀy is released in the same stacked Gaussian mechanism as the Gram/moment blocks
(`party_o.py:317`), contributing the `+B_y⁴` term to Δ² (eq. (6)). yᵀy is needed only for
downstream RSS/inference (§6.5), not for β̂. When B_y is a dominant bound, that term inflates σ for
the estimator that does not use it.

**Evidence (`demos/d6`, "no-yty" columns).** Dropping yᵀy cuts σ from 154→75 (raw) and 77→69 (std),
roughly halving noise on the estimate in the raw case.

**The fix and its cost.** Release yᵀy under a separate budget (compose, e.g. split ε), or only when
inference is requested; calibrate the Gram/moment σ without B_y⁴. Cost: an extra budget line if
inference is needed; strictly better estimator when it is not. **Confidence: high.**

---

### C7 — Theorem 1's "bias" is a conditional expectation (MEDIUM, theory)

**What is wrong.** The proof (App. A, Steps 1–3) expands on the event `‖PE‖<1` but then evaluates
`E[EPE]` as an *unconditional* Gaussian second moment. On the complement (G+E near-singular) β̃ has
no finite mean, so `E[β̃]` does not exist unconditionally; the stated bias is really
`E[β̃ | ˜G≻0, ‖PE‖<1] − β̂` (a truncated/conditional expectation).

**Why it matters.** The equality `Bias = E[EPE]β̂ + O(σ⁴)` needs the difference between conditional
and unconditional `E[H²]` to be O(σ⁴). That holds only because the bad event is exponentially
unlikely under the ρ-gate; it should be stated, not assumed.

**Evidence (`demos/d4_theorem_checks.py`).** Part A: with antithetic conditioning at ρ=4,8 the
empirical bias matches the operator M to 1.7%/0.6% and clearly rejects the GOE variant — the
formula is correct as a conditional statement. Part C: unconditionally at ρ=0.5, the running mean
of β̃₀ does not settle (−1.13, +4.68, +1.79 over 10³–4·10⁵ draws; max |β̃₀|~3.8·10⁵) — no finite mean.

**The fix and its cost.** State Theorem 1/2 for the truncated estimator (or condition on ˜G≻0 and
bound the truncation error via Remark 2's tail), and note the correction `β̂_bc` is applied to the
gated β̃. Wording only. **Confidence: high.**

---

### C8 — The ρ-gate optimises validity, not MSE (MEDIUM, utility/impl)

**What is wrong.** `select_ridge`/`auto_lambda` (`phase1_common.py:446–482`) pick the *smallest* λ
with ρ_λ≥target (default 2), i.e. the minimal λ for a valid Neumann expansion. But Prop. 1 says the
MSE-optimal λ is an interior point trading shrinkage against mechanism variance. The code never
computes that; in the noise-dominated regime it lands at a λ far larger than MSE-optimal
(over-shrinking, C5).

**Evidence.** `d6`: forced full ridge with λ up to 3276 gives rel.err ~0.99. `d4` D: the shrinkage
term is two orders larger than the mechanism bias the code targets.

**The fix and its cost.** After gating for validity, search λ≥λ_gate minimising an estimated MSE
`‖shrinkage(λ)‖² + trace of the (post-processing) variance proxy`, all from released quantities
(free). Report the deterministic shrinkage separately as the paper already recommends. Cost: a 1-D
search; can only help. **Confidence: medium-high** (the estimated-variance proxy needs the
companion note's SE decomposition to be precise).

---

### C9 — PSI reveals O's max bin load (D, α) to R (LOW, linkage)

**What is wrong.** O sends `{alpha, D}` to R (`party_o.py:261`), where D = max simple-hash bin load
of O's ID multiset and α = ⌈max_load/D_cap⌉ (`psi_common.py:243–245`). These are functions of O's
ID *set*, not only n_O.

**Why it matters.** A mild leak beyond the cardinality n that Def. 2 permits. In practice D is
nearly determined by n_O for a fixed hash, so the marginal leak is small.

**Evidence (`demos/d5`).** Over 30 O-ID-sets differing pairwise in one identifier, D=7 in all 30
(stable), confirming the leak is small but nonzero and data-dependent.

**The fix and its cost.** Pad D and α to public worst-case values derived from n_O alone (a
standard cuckoo/simple-hash overflow bound) so the message is a function of n_O only. Cost:
a few extra padded ciphertexts. **Confidence: medium** (low severity; fix is standard).

---

## 4. Prioritized roadmap

**Minimal set before the privacy claim can be defended:**

1. **C1 — circuit privacy.** Replace `add_scalar` in-ciphertext with a mechanism that releases only
   the noised value: encrypted DP noise + noise flooding (κ≈40) reconciled into one
   finite-precision-safe distribution, or re-encryption/MPC reveal. Without this, Defs 1 and 3 fail
   against R regardless of the DP math. (Also fold in the deferred item iv here.)
2. **C2 — correct the sensitivity to the replace-one adjacency of Def. 4** (tight replace bound, or
   ×2 on Δ²). Recalibrate σ. Non-negotiable for the stated (ε,δ).
3. **C3 + C4 — close the standardization/DP hole and the metadata leak.** Require actually-committed
   public scales/bounds; strip O-derived statistics (quantiles, clip rates, moments) from what R
   receives; split calibration per party.
4. **C7 — restate Theorems 1/2 as conditional (truncated) results** and tie them to the ρ-gate.

**Nice-to-haves (utility and honesty):**

5. **C5 + C8 + C6 — make the estimator usable and the claims honest:** default to
   committed-scale standardization; choose λ by estimated MSE, not just validity; release yᵀy under
   a separate budget; and reframe §6.6 to state the true cohort sizes needed (thousands–tens of
   thousands at ε=1), since the method is *not* viable in the small-cohort regime as currently claimed.
6. **C9 — pad the PSI (D, α) message to a public function of n_O.**

---

## 5. What the authors got right (credit where due)

- **The single-draw bias operator is correct and non-obvious.** The `−P diag(P)Π_O` term that
  distinguishes the deployed single-draw symmetric Gaussian from the GOE convention is subtle, and
  it is right: `d4` confirms the operator (and its code in `phase1_common._bias_operator`) against
  antithetic Monte-Carlo to <2% at ρ=4–8, cleanly rejecting the GOE variant. The Schur-complement
  reading of the trace weights (eq. (9)) is a genuinely useful diagnostic.
- **"Keep A local" is the correct fix and is internally consistent.** A being R's clean block makes
  the top-left Gram-noise block vanish and removes the B_R⁴ term from the sensitivity — a real
  structural consistency check, and the code faithfully never noises or transmits A.
- **The Balle–Wang analytic Gaussian σ is implemented correctly**, including the far-tail
  log-Φ stabilization needed for large ε (`d7`: matches an independent scipy root-find to 1e-12).
- **Joint (stacked) sensitivity calibration** rather than naive per-block composition is the right
  call and is correctly implemented (one σ, one draw per released coordinate).
- **The bias correction is genuine zero-budget post-processing and survives data-dependent λ.**
  `d4` D shows it removes the O(σ²) bias whether λ is fixed or chosen from the released ˜G.
- **Input privacy for O's view of R's data is sound**: O only ever holds ciphertexts under R's key,
  so by IND-CPA its view is simulatable — the asymmetry is entirely on R's side (C1).
- The paper is **honest about the shrinkage/OLS gap** (Prop. 1) and **flags the CKKS error channel**
  (item iv) — the criticism is that the code doesn't implement the fix and that the channel is a
  prerequisite, not a refinement.
