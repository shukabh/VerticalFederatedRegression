# Cryptographic-layer risk report — Scenario B (HE + PSI)

Scope: the homomorphic-encryption aggregation (CKKS) and the private set intersection
(BFV, Chen–Laine–Rindal-style labeled PSI) of the two-party vertical-federated OLS
protocol. Threat model under review is the paper's own: **semi-honest** — both parties
follow the protocol but infer everything they can from the messages they receive
(Def. 1), with **linkage privacy** for O (Def. 2) and **(ε,δ)-DP** on the released model
(Def. 3–4). R holds the CKKS/BFV secret key and is the only decryptor.

Files reviewed (read-only): `scenario_b/party_r.py`, `party_o.py`,
`calibrate_hyperparameters.py`, `run_protocol.py`, `generate_vfl_data.py`. The imported
`psi_common.py`, `phase1_common.py`, `he_backend.py` were **not** provided; where a finding
depends on them it is stated as conditional and the condition is made checkable.
Demos: `reports/privacy/demos/crypto/` (`slot_sum_check.py`, `psi_params_check.py`), run
with TenSEAL 0.3.18 / SEAL and numpy.

---

## Executive summary

The protocol's *statistical* privacy story (DP on the O-dependent Gram/moment blocks) is
undermined by the *cryptographic* delivery of those blocks. Two issues dominate:

- **C2 (critical).** The DP noise is added with a slot-broadcast `add_scalar`, and the
  aggregates are read by a party that holds `sk`. Reading "slot 0" is a *convention*, not
  an *enforcement*: R can decrypt every slot. Whenever a slot other than slot 0 holds a
  data-independent value (a partial sum, a zero from masking, or — as in the visible code —
  the broadcast noise scalar itself), R recovers the **noise-free** aggregate by a slot
  subtraction, in which the DP noise cancels. Demonstrated end-to-end. This voids the
  (ε,δ) guarantee for B, C, c_O, c_R and yᵀy against R, who is precisely the DP adversary.
  *Caveat (orchestrator):* the one slot layout that is safe with a broadcast `add_scalar`
  is a full cyclic sum that leaves the same total in **every** slot (demo row `cyclic`).
  Whether the unseen `he_backend.inner_product` does that is the open question. Check it by
  decrypting all slots of one returned block.

- **C1 (high).** Neither phase applies **circuit privacy** (noise flooding / modulus
  switching / re-randomisation) before handing ciphertexts to the key holder. The returned
  ciphertext's error term is a deterministic function of O's plaintext operands and O's
  encoding; IND-CPA says nothing about hiding it from the key holder. The paper itself lists
  "close the CKKS decryption-error channel" as open work (§7 item iv), so this is a known,
  in-scope gap, not an accident.

Supporting findings: PSI labels are O's raw row indices with **no O-side permutation in the
code** (C7); PSI metadata (`alpha`, `D`, `n_batches`) leaks bin structure and coarse set
sizes outside the paper's Def.-2 accounting (C5); plaintext modulus / identifier-entropy
sizing is weak for low-entropy IDs (C6); DP noise and PSI masks use PCG64 with a
floating-point Gaussian (C8); public keys and pickled control data cross an unauthenticated
socket, and `pickle.loads` on peer data is a remote-code-execution vector (C9); the
parameters need a 128-bit re-check against the HE Standard (C11); and the paper's Appendix-B
"encrypted Horner evaluation" is unsound under BFV noise growth at scale — the
implementation silently replaced it with a binned windowed-powers evaluation whose leakage
the paper never analyses (C10).

---

## C1 — No circuit privacy: key holder can read operand-dependent ciphertext error

- **Severity:** high
- **Inside semi-honest threat model?** Yes. R inferring O's inputs from received
  ciphertexts is the definition of a semi-honest attack (Def. 1). The paper acknowledges
  the channel (§7, item iv).
- **Location:** CKKS aggregation `party_o.py:112–131` (`inner`, `add_scalar`, block
  returns); BFV replies `party_o.py:83–96`. No flooding/mod-switch/re-randomisation anywhere
  in the visible O code before `send_msg`. Paper §4.3 Phase 5, §5.1 (returns `Enc(s+z)` with
  no sanitisation).
- **Mechanism / what leaks to whom.** In CKKS, O computes each block as
  `plaintext-encode(X_O column) × Enc(b or Ẋ_R)`, then rescales and sums. The resulting
  ciphertext `(c0, c1)` decrypts as `⟨plaintext⟩ + e`, where the LWE error `e` is a
  deterministic function of O's plaintext operand (its magnitude scales with
  ‖X_O column‖ and with the rescale/level path) plus R's own encryption noise, which R
  knows. A semi-honest R holding `sk` does not merely learn the rounded value: it can
  inspect `e = c0 + c1·s − encode(value)` and read operand-dependent quantities (column
  norms, and in the worst case finer structure) that the *ideal functionality* (the DP-noised
  aggregate) does not reveal. Standard HE security (IND-CPA) protects a ciphertext from a
  party *without* `sk`; it says nothing about what the key holder learns from `e`. In BFV the
  decrypted value is exact (noise rounds away), but the same residual-noise channel exists in
  the ciphertext and leaks the structure/weight of O's membership and label polynomials to R.
- **Evidence.** Reasoning above (the operand-dependence of CKKS error is the subject of
  Li–Micciancio 2021 and the reason OpenFHE ships a noise-flooding decrypt mode). The
  concrete, *already-usable* consequence of the missing sanitisation is demonstrated under
  C2 (slot subtraction recovers noise-free values); C1 is the broader statement that even the
  intended single slot is not sanitised.
- **Mitigation with cost.** Before returning, O should (i) modulus-switch each ciphertext to
  the **lowest** remaining level (drops most of the operand-dependent high-order noise and
  shrinks payloads), then (ii) **noise-flood**: add an encryption of zero whose error
  exceeds the worst-case circuit error by a statistical security factor 2^λ (λ≈40).
  OpenFHE exposes this for CKKS as a noise-flooding decryption/execution mode (e.g.
  `EXEC_NOISE_FLOODING` / the `NOISE_FLOODING_*` config) and for BGV/BFV as re-randomisation
  by adding `Enc(0)` with large noise. Cost: flooding consumes ~λ bits of the modulus budget,
  which for a fixed ring dimension eats into the multiplicative depth or forces a larger ring
  (see C11); for CKKS it also costs precision. This is real but standard.
- **Confidence:** high that the sanitisation is absent and required; medium on how much
  per-record (vs aggregate-norm) information a semi-honest R can practically extract from `e`
  alone — the decisive leak is C2, which needs no noise analysis.

## C2 — DP noise broadcast to all slots + key holder reads all slots ⇒ DP defeated

- **Severity:** critical
- **Inside semi-honest threat model?** Yes. R holds `sk` and decrypts; nothing stops it
  decrypting slots other than the one it "should" read.
- **Location:** `party_o.py:118,122,126,129,131` — every block is
  `ckks.add_scalar(inner(...), noise)`; `add_scalar` adds the scalar to **every** slot (task
  fact 2). `party_r.py:122` reads only slot 0 via `ckks.decrypt_slot0`, but that is a
  convention — R has `sk`. `inner_product`/`EvalSum` live in the unseen `he_backend.py`.
- **Mechanism / what leaks to whom.** Two independent sub-cases, both giving R the
  **noise-free** aggregate (so full DP loss, per released statistic, to R):
  1. If `inner_product` is a rotate-and-add "dot/sum" (the standard TenSEAL/OpenFHE idiom),
     the non-zero slots hold **partial sums**. `slot_j − slot_{j+1}` is one record's
     contribution `b_j · X_O[i,j]·X_O[i,k]` (or `b_j·X_O·y`, etc.), and the broadcast DP
     scalar cancels in the difference. R reads **individual O records**.
  2. Even if `inner_product` masks perfectly to slot 0 (other slots = 0), `add_scalar`
     then writes the DP scalar `z` into **every** slot. R reads any slot `j≠0`, gets `z`,
     and subtracts: `slot0 − z = ` the noise-free aggregate. So in this case the broadcast
     defeats DP on its own. The only safe layout with a broadcast is a full cyclic sum that
     leaves the same total in every slot (demo row `cyclic`: `slot0 − slot[2048] = 0`), and
     that depends on the unseen `he_backend`.
- **Evidence.** `demos/crypto/slot_sum_check.py` (SEAL-CKKS, N=8192). Output
  (`slot_sum_check.out`), n_O=12, 5 matched:
  - `suffix`: `slot_j − slot_{j+1}` equals `b_j·w_j` to ≤4e-7 (per-record recovery); and a
    slot outside the summation window holds exactly `z`, so `slot0 − that slot = 1.383072` =
    the noise-free aggregate `b·w`.
  - `masked+bcast` (mask to slot0, then broadcast `z`): `slot0 − far_slot = 1.383073` =
    noise-free `b·w` — confirms sub-case 2 straight from the visible code.
  - `masked+slot0` (mask **and** put `z` in slot 0 only): other slots ≈0, `slot0 = b·w + z`.
    No noise-free path. This is the fix.
- **Mitigation with cost.** (a) In `he_backend`, make `inner_product` reduce to a single
  informative slot (multiply by a one-hot `e_0` plaintext, or a full cyclic sum so no slot
  is a partial sum). (b) Encode the DP noise as a **vector** that is non-zero only in slot 0
  (or add it before the masking so it is masked with the value), never with a broadcast
  `add_scalar`. (c) Ideally combine with C1 flooding/re-randomisation so no slot's *error*
  leaks either. Cost: the one-hot mask is one extra CKKS multiplicative level (depth 2→3),
  which raises the modulus and forces ring ≥ 16384 (see C11); otherwise negligible.
- **Confidence:** high that the leak occurs whenever any slot differs from the total
  (sub-cases 1 and 2). The visible code does not settle which layout `he_backend` produces.
  A full cyclic sum (every slot = total) is safe with the broadcast. Severity is critical
  unless that is confirmed. To confirm which applies,
  decrypt **all** slots of one returned block in a test run and check for partial sums —
  do not trust `decrypt_slot0`.

## C3 — CKKS approximate decryption (Li–Micciancio IND-CPA^D): direction of the risk

- **Severity:** medium (this is the lens through which C1/C2 are the concrete leaks)
- **Inside semi-honest threat model?** Yes.
- **Location:** conceptual; applies to all CKKS returns `party_o.py:118–131`, decrypt
  `party_r.py:122`.
- **Mechanism / what leaks to whom.** IND-CPA^D (Li–Micciancio) breaks when *decryption
  results of adversarially chosen ciphertexts are exposed to someone*, because the CKKS
  decryption error reveals secret-dependent information. Here the direction matters: **only R
  decrypts, and R is also the party trying to learn O's data.** So (i) the classic
  IND-CPA^D attack of feeding decryptions back to a *different* party does **not** arise
  between R and O — nothing is decrypted by a non-key-holder, and O never sees a decryption;
  (ii) but this is worse for O's privacy, not better: R is simultaneously key holder and
  adversary, so the IND-CPA guarantee that would protect O's inputs from an outsider gives O
  **no** protection against R. The approximate-decryption error is exactly the operand-
  dependent channel of C1. The paper's own remedy (§7 item iv: discrete/snapped Gaussian)
  addresses the *DP-noise* precision, not this channel; the channel needs flooding (C1).
- **Evidence.** Reasoning; consequence realised in C2's demo.
- **Mitigation with cost.** Same as C1 (flooding + mod-switch). No non-key-holder decryption
  is introduced, so no multiparty-decryption smudging is needed.
- **Confidence:** high on the direction; the practical severity rides on C1/C2.

## C4 — No circuit privacy in the BFV PSI replies

- **Severity:** low–medium
- **Inside semi-honest threat model?** Yes.
- **Location:** `party_o.py:83–96` (`mul_pt`/`add`/`add_pt` build `ct_s = r1·P(x)` and
  `ct_q = r2·P(x) + L(x)`), returned without mod-switch or flooding.
- **Mechanism / what leaks to whom.** BFV is exact, so the *decrypted* values are clean
  (`0` on match, uniform mask otherwise; label on match). But the ciphertext noise after the
  `ct×pt` chain is a function of O's polynomial coefficients (its ID/label set) and O's
  random masks; a key-holding R can inspect it, learning coarse structure such as the
  effective degree used per bin (already sent as `D`) or the weight of O's coefficient
  vectors. R also decrypts **all** `NB` slots, but R itself populated every bin (its own
  cuckoo table + dummy 0s), so no *extra* bins are exposed; the one edge is empty bins filled
  with `id=0` (`party_r.py:81–84` via `pc.DUMMY_Y`/`pow(0,·)`), which probes whether O has an
  identifier whose field value is 0.
- **Evidence.** Reasoning from the code path; no separate demo (BFV value channel is exact
  and correct).
- **Mitigation with cost.** Re-randomise each reply by adding `Enc(0)` with fresh large
  noise and mod-switch to the last level before sending; cheap relative to the polynomial
  evaluation. Avoid the literal `id=0` sentinel for empty bins (use a random non-root
  filler).
- **Confidence:** medium.

## C5 — PSI metadata leaks bin structure and coarse set sizes (outside Def. 2)

- **Severity:** low
- **Inside semi-honest threat model?** Yes, but not covered by the paper's Def. 2 accounting
  ("O learns only n").
- **Location:** `party_o.py:69,74` sends `{"alpha","D"}`; `party_r.py:74` sends
  `{"n_batches","n_window"}`; read at `party_r.py:76`, `party_o.py:66`.
- **Mechanism / what leaks to whom.** `alpha` and `D` are functions of O's **max bin load**
  under simple hashing, i.e. of `n_O` and O's specific ID→bin distribution; they flow O→R.
  `n_batches = ceil(n_R / cuckoo_cap)` (cap ≈ 0.9·16384 = 14745) flows R→O and leaks R's
  cohort size to within one batch (~15k). Neither breaks the intersection secrecy (only R
  decrypts, so O still learns nothing about I or n — in fact **less** than Def. 2's budget of
  `n`), but the paper's security analysis does not mention either quantity, and `alpha`/`D`
  expose O's hashing regime.
- **Evidence.** `demos/crypto/psi_params_check.py` part 2: for `n_O`=5·10³/5·10⁴/10⁶ the max
  simple-hashing load is ~7 / ~24 / ~240, giving `D`≈7 / 24 / 60 and `alpha`=1/1/4 — a direct
  read on O's scale. `P(D=1)` is ~0 except for tiny `n_O`.
- **Mitigation with cost.** Fix `D` and `alpha` to public worst-case constants for the agreed
  `(n_O, NB, n_hash)` rather than deriving them from realised loads; pad every bin to degree
  `D`. Fix `n_batches` to a public cap on `n_R`. Cost: some wasted computation on padded bins.
- **Confidence:** high on the mechanism; the privacy impact is minor because `n_O` is already
  sent in the clear (`party_o.py:103`, `party_r.py:36`).

## C6 — Plaintext-modulus / identifier-entropy sizing and false-positive semantics

- **Severity:** medium
- **Inside semi-honest threat model?** Partly: correctness/false-positive is a utility issue;
  the low-entropy-ID point is a linkage-privacy issue.
- **Location:** `calibrate_hyperparameters.py:208–225` (`psi_plaintext_modulus`),
  `party_r.py:84` and `pc.id_to_field(int(x), t)` (identifiers reduced into `Z_t`),
  `party_o.py:73` (within-bin field collisions dropped).
- **Mechanism / what leaks to whom.** (i) The modulus is sized so the **expected** number of
  false matches ≤ `delta_psi`, via `t_min = n_R·n_O·n_hash/(ring·delta_psi)`. That is an
  expectation bound, not a high-probability bound, and it silently drops collisions
  (`coll` warning at `party_o.py:73`), so a within-bin field collision makes a genuine O
  record **unmatchable** — a data-dependent, load-dependent correctness loss. (ii)
  Identifiers are mapped into `Z_t` with `t` as small as ~2²⁹ for moderate cohorts; real
  identifiers (a 9-digit SIN is ~2^29.9, Luhn-valid ~2^26.6) are then reduced mod `t`, and
  the map `id_to_field` is **unkeyed** (public `t`). Anyone who can observe a party's bin
  contents, or who compromises the transcript, can enumerate a ~2²⁷ identifier space offline
  and invert the mapping — there is no PRF/OPRF keying the identifiers.
- **Evidence.** `demos/crypto/psi_params_check.py` part 1: `t` bit-length is 29 / 36 / 42 /
  48 / 53 bits for (n_R,n_O) from (500,5k) up to (10⁶,4·10⁷), all with expected-FP≈1e-6;
  identifier-space sizes printed alongside.
- **Mitigation with cost.** Key the identifier encoding with an OPRF (or at least a keyed
  hash whose key neither party can enumerate) before reduction into `Z_t`; size `t` (or use
  CRT over two primes, as the code's own error message suggests) so distinct identifiers do
  not collide with high probability, and treat the FP bound as tail, not mean. Cost: an OPRF
  round; wider field / CRT increases BFV cost (see C11).
- **Confidence:** medium (the collision-drop and unkeyed-map behaviours are in the visible
  code; the exact `id_to_field` is in unseen `psi_common.py`).

## C7 — PSI labels are O's raw row indices with no O-side permutation in the code

- **Severity:** medium
- **Inside semi-honest threat model?** Yes.
- **Location:** `party_r.py:94` `sigma_map[k] = qdec[slot] % t` ("O's canonical row index"),
  used at `party_r.py:98–100` to index `b`, `Xdot`. O builds the label layers in
  `party_o.py:69` (`bin_and_interpolate`, labels = row index) and applies **no permutation**
  — contrast paper Appendix B "Pre-processing (O): O applies a uniform random permutation τ so
  that recovered indices are semantically inert." The only shuffle is in the **data
  generator** (`generate_vfl_data.py:202–204`, `perm_O`), which is not part of the protocol.
- **Mechanism / what leaks to whom.** R recovers, for each of the `n` matches, O's physical
  row index in `X_O`. That index is needed to align to O's aggregate ordering (it *is* the
  σ output), so it is R's legitimate output — **provided** the row order carries no meaning.
  Appendix B guarantees that with τ; the implementation delegates it to whoever produced
  `X_O.csv`. If O's storage order is meaningful (enrolment date, region blocks, a sort key),
  R learns that ordering for every matched record, beyond the abstract alignment σ. R learns
  nothing about *non-matched* O rows (uniform payloads), and O learns nothing (only R
  decrypts).
- **Evidence.** Absence of any permutation in `party_o.py`; the shuffle lives only in the
  generator. Reasoning.
- **Mitigation with cost.** O must apply its own secret permutation τ to rows before building
  the label polynomials and un-permute nothing (R only ever needs the permuted index, which
  it also uses to read the aggregates — so O must build the Phase-1 aggregates in the **same**
  permuted order). Effectively free (a relabelling), but it must be O's secret and consistent
  across both phases.
- **Confidence:** high that the code omits τ; the impact depends on whether O's real row
  order is meaningful.

## C8 — Non-cryptographic RNG (PCG64) and floating-point Gaussian for DP noise and PSI masks

- **Severity:** medium
- **Inside semi-honest threat model?** Yes (a semi-honest R that can predict/recover O's
  noise defeats DP; the float-Gaussian leak is an established DP soundness bug).
- **Location:** PSI masks `party_o.py:77,92–93` (`np.random.default_rng()` → PCG64,
  `rng.integers(1,t,NB)` for `r1,r2`); DP noise `party_o.py:110`
  `p1.draw_noise(d_R,d_O,sigma, np.random.default_rng())`. (task fact 5)
- **Mechanism / what leaks to whom.** PCG64 is not a CSPRNG: its internal state is
  recoverable from a modest run of outputs. For the **DP noise**, if R can recover the
  generator state it can subtract the exact noise and fully de-privatise the aggregates
  (this is independent of, and additional to, C2). Separately, sampling a *floating-point*
  Gaussian and adding it exposes the Mironov-2012 low-bit leakage: the released value's
  fractional bits reveal that the noise is not the ideal distribution, enabling reconstruction
  attacks — this holds even with a perfect RNG. The PSI masks `r1,r2` are lower-risk (R never
  observes them in the clear; on matches `P=0` so `r1·P=0` regardless), but should still be
  drawn securely.
- **Evidence.** Reasoning; PCG64 state-recovery and Mironov float-Gaussian are established
  results. Note the two `default_rng()` calls are unseeded (OS entropy), so state is not
  trivially known, but PCG64 is still not designed to resist state recovery, and the
  float-Gaussian issue is unconditional.
- **Mitigation with cost.** Use a CSPRNG (`secrets`/OS DRBG) to seed noise generation, and a
  **discrete/snapped Gaussian** (the paper's §7 item iv), sampled and added as an integer at
  the CKKS encoding scale so the low-bit channel is closed. Cost: a discrete-Gaussian sampler;
  negligible runtime.
- **Confidence:** medium–high (float-Gaussian is unconditional; PCG64 exploitability against
  DP depends on additional access, but the fix is cheap and standard).

## C9 — Unauthenticated socket + `pickle.loads` on peer data (RCE)

- **Severity:** high (as an implementation/deployment issue); **not** a semi-honest
  input-privacy violation by itself.
- **Inside semi-honest threat model?** No — a network MITM and malicious deserialization are
  active/out-of-model — but it is a concrete security defect worth flagging.
- **Location:** raw TCP with no TLS/auth (`party_r.py:50–55`, `party_o.py:58–60`); public
  keys sent in the clear (`party_r.py:64,114`); `pickle.loads` on data received from the peer
  at `party_r.py:75,89,119` and `party_o.py:65,79,103,106`.
- **Mechanism / what leaks to whom.** `pickle.loads` on attacker-controlled bytes is
  arbitrary code execution: a compromised peer or a network MITM can run code on the other
  party's host. The channel is also unauthenticated and unencrypted, so a passive eavesdropper
  reads all plaintext metadata (`alpha`, `D`, `n_batches`, `dims`, `n_O`) and an active one can
  substitute the public key. Encrypted payloads stay confidential, but key substitution would
  let a MITM read them.
- **Evidence.** Direct from the code (pickle over sockets throughout).
- **Mitigation with cost.** Replace `pickle` with a schema'd, non-executable serialization
  (protobuf/JSON + explicit ciphertext (de)serialization from the HE library); run the
  channel over mutually authenticated TLS; pin/attest the public key out of band. Cost: a
  serialization refactor; standard.
- **Confidence:** high.

## C10 — Appendix B "encrypted Horner evaluation" is unsound under BFV noise; implementation diverges and is unanalysed

- **Severity:** medium
- **Inside semi-honest threat model?** N/A directly — this is a soundness/analysis gap in
  the paper vs the code.
- **Location:** Paper Appendix B, Step 2 ("O evaluates Enc(P(x_j)) with x_j via n_R
  scalar-ciphertext multiplications"). Implementation: `party_r.py:84` sends encrypted
  **windowed powers** `Enc(x^{2^i})`; `party_o.py:25–33` (`power_from_windows`) rebuilds
  powers by balanced `ct×ct` products; `party_o.py:83–96` evaluates the bin polynomials as
  `Σ_d pt_coeff_d · Enc(x^d)`.
- **Mechanism.** A literal Horner evaluation of a degree-`n_R` polynomial is a chain of
  `n_R` sequential multiplications by the encrypted `x`; in BFV that is multiplicative depth
  ≈ `n_R`, whose noise growth is far beyond any practical modulus for `n_R` in the hundreds
  or thousands — Appendix B as written does not run at scale. The implementation is the
  Chen–Laine–Rindal fix: **cuckoo/simple hashing into bins** caps the per-bin degree at
  `D ≤ d_cap = 64`, and **windowed powers** give the powers at depth ≈ `log2(D)` (≤ 6, the
  `n_window` bound at `party_o.py:70–71`), with coefficients kept plaintext
  (`ct×pt`, `party_o.py:82`). So the code is sound where the paper is not — but its leakage
  surface (bin loads → `alpha`/`D`, cuckoo capacity → `n_batches`, hashing regime; C5/C7)
  is different from Appendix B's and is **not** covered by the paper's Def.-2 analysis. The
  paper's τ step (Appendix B pre-processing) is also dropped in the code (C7).
- **Evidence.** Depth arithmetic above; `demos/crypto/psi_params_check.py` part 3 shows the
  BFV modulus budget even for the shallow binned circuit is tight (see C11), let alone a
  depth-`n_R` Horner chain.
- **Mitigation with cost.** Update Appendix B to describe the binned windowed-powers
  construction actually implemented, and extend the linkage-privacy analysis to the bin-load
  and metadata leakage it introduces (C5), plus reinstate O's permutation τ (C7). No runtime
  cost; it is a spec/analysis correction.
- **Confidence:** high on the unsoundness of literal Horner at scale and on the divergence.

## C11 — Parameter security to re-check against the HE Standard

- **Severity:** medium
- **Inside semi-honest threat model?** Yes — input privacy rests on the HE parameters
  meeting 128-bit security.
- **Location:** BFV `calibrate_hyperparameters.py:208–225,317–322`
  (`ring_dim_bfv=16384` fixed, `mult_depth=4`, plaintext modulus up to `max_bits=60`);
  CKKS `calibrate_hyperparameters.py:287–289,323–326` (`batch_size = next pow2 ≥ n_O` becomes
  the ring, `mult_depth=2`, `scale_bits=50`).
- **Mechanism.** HE-Standard 128-bit classical (ternary secret) caps total `log2(QP)` at
  109 / 218 / 438 / 881 bits for ring 4096 / 8192 / 16384 / 32768. (i) **BFV:** the ring is
  hard-coded to 16384 regardless of `t` and depth. A depth-4 circuit with a 45–60-bit
  plaintext modulus needs a coefficient modulus that a rough budget puts near or above the
  438-bit ceiling for N=16384 — i.e. large-cohort runs (which drive `t` up, see C6) may fall
  below 128-bit or fail OpenFHE param-gen, since the ring cannot grow. (ii) **CKKS:** the ring
  equals `batch_size`, derived from `n_O`. A depth-2, scale-2⁵⁰ CKKS chain needs ≈
  60+50+50+60 ≈ 220 bits, which **exceeds** the 218-bit ceiling at N=8192. So any run with
  `n_O ≤ 8192` (ring 8192) is marginally sub-128-bit; `n_O > 8192` (ring ≥ 16384) is fine.
  The C2 one-hot-mask fix adds a level (≈ +50 bits → ~270), keeping the CKKS requirement at
  ring ≥ 16384.
- **Evidence.** `demos/crypto/psi_params_check.py` part 3: BFV `t≈2^45/2^60` → rough
  `log2(QP)` ~465 / ~540 bits vs 438 allowed at N=16384 ("EXCEEDS"); CKKS depth-2 scale-50 ~220
  vs 218 at N=8192.
- **Mitigation with cost.** Do not fix the ring independently of `(t, depth)`; let the HE
  library choose N to hit 128-bit for the actual circuit, or raise BFV to N=32768 for large
  `t`/depth and enforce CKKS ring ≥ 16384. Verify with the library's own 128-bit param-gen
  (unseen `he_backend.py`), and re-verify after adding flooding (C1) and the mask level (C2),
  which both enlarge the modulus. Cost: larger rings ≈ 2× per doubling in compute/bandwidth.
- **Confidence:** medium — the budget figures are heuristic and the real modulus selection is
  in the unseen backend; the CKKS N=8192 edge and the "ring fixed while t/depth vary" design
  are concrete and worth an explicit 128-bit assertion in `he_backend`.

---

## Prioritized fix list

1. **C2 (critical).** Stop broadcasting the DP noise: add it to slot 0 only (as a vector) and
   force `inner_product` to leave a single informative slot; verify by decrypting *all* slots
   of a returned block. Until fixed, DP on O's aggregates is void against R.
2. **C1 / C3 (high).** Add circuit privacy on every O→R ciphertext: modulus-switch to the last
   level then noise-flood (OpenFHE noise-flooding mode for CKKS; `Enc(0)`+large-noise
   re-randomisation for BFV). Budget the extra modulus into C11.
3. **C9 (high, deployment).** Remove `pickle.loads` on socket data; run over authenticated
   TLS; pin the public key.
4. **C11 (medium).** Assert 128-bit security in `he_backend` for the actual `(N, t, depth,
   scale)` including the C1/C2 additions; do not fix the ring while `t`/depth vary; enforce
   CKKS ring ≥ 16384.
5. **C7 (medium).** Reinstate O's secret row permutation τ (protocol, not the data
   generator), consistent across PSI and Phase 1.
6. **C6 (medium).** Key the identifier encoding (OPRF/keyed hash) and size `t`/CRT so distinct
   IDs don't collide; treat the false-positive bound as a tail, not a mean.
7. **C8 (medium).** CSPRNG-seed the DP noise and use a discrete/snapped Gaussian (paper §7
   item iv).
8. **C10 (medium, spec).** Rewrite Appendix B to the binned windowed-powers construction that
   is implemented, and extend the linkage-privacy analysis to its metadata (C5) and τ (C7).
9. **C5 (low).** Fix `alpha`/`D`/`n_batches` to public constants; pad bins.
10. **C4 (low–medium).** Re-randomise BFV replies; drop the literal `id=0` empty-bin sentinel.
