# Deviations from the specification, choices made, and points that look wrong or underspecified

Reference documents: the v3.1 specification (`protocol_v31.tex`), the corrected algorithms of
`paper/algorithms.tex` / `paper/main.tex` (followed where they differ from the specification), and the proof
review (`proofs/crypto/03a, 03b, 06, 07`). Items are grouped by phase. "Spec" = protocol_v31.tex, "paper" =
paper/algorithms.tex.

## A. Deliberate deviations (with reason)

| # | Where | Spec / paper | Implementation | Why |
|---|---|---|---|---|
| A1 | Link, reply | Spec: "modulus-switch; reply". Paper: SanitizeBFV (tocheck) | SanitizeBFV as in the paper: modulus switch to 2 limbs, + a switched fresh `Enc_pk(0)` under R's BFV key, + flooding noise `e ~ D_{Z^N, 2^40 B_err}` added to c0, drawn with the exact sampler | You asked for it; G4 needs circuit privacy. `B_err` is a worst-case l2 bound of the rounding terms of the switch (ternary s: `2(sqrt(N)/2 + N^1.5/2) + 1`, about 2^21); the `+1` covers the scaled evaluation noise, which we measured at 2^-72 with the 7-limb chain but did **not** bound analytically (heuristic). Terms that are functions of the decrypted plaintext are not flooded (R learns the plaintext anyway). |
| A2 | Link, masks | `r1 * P(w)`, `r2 * P(w) + L(w)` as ciphertext-plaintext products | The fresh masks are folded into the plaintext coefficients (`Sc = r1 * Pc`, `Qc = r2 * Pc + Lc`, mod t) | Same values, one plaintext-multiplication level less. With separate mask products the BFV noise exceeded `Q/2t` at depth 4. |
| A3 | Link, sizes | D and alpha "public functions of (n_bar_O, h, N_bins, d_cap)" | `D = min(d_cap, B)`, `alpha = ceil(B/D)` with B the smallest load such that `N_bins * P[Binomial(n_bar_O h, 1/N_bins) > B] <= delta_psi = 2^-40` | Makes them data independent (the old prototype used the observed maximum load). Duplicate item values are never put in the same partition. |
| A4 | Keygen | "threshold key generation with a common random a" | Own implementation of the one-round Mouchet et al. shares on OpenFHE's DCRTPoly: `p0_P = -a s_P + e_P` and rotation-key shares in OpenFHE's hybrid layout `b_j = -a_j psi^{-1}(s_P) + e + P g_j s_P`; every `a`, `a_j` is expanded from a CRS seed with SHAKE-256 | So R sends only the b-parts (20 MB instead of 40 MB). The CRS seed is drawn by O and sent in the public parameters (enough under honest-but-curious). O sums the shares and holds the joint pk and joint rotation keys; O also sends its pk share to R (the paper's table lists it; R does not use it). Secrets are uniform ternary; errors are OpenFHE's discrete Gaussian (sigma 3.19; the bound uses 3.2). |
| A5 | Encoding / scales | `Ecd(z) = round(Omega tau^{-1}(z))` with one scale Omega | R encodes at `Omega_U = 2^68` (default for circuit A; `q2` or any `2^log2_Omega_U` as options), O encodes the masks at `Omega_u = q1` and its data at `Omega_V`; the final scale `Omega_f = Omega_U Omega_V Omega_u /(q1 q2)` is tracked exactly (double-double) and R decodes with the exact rational value | The proofs' model (M2) allows separate scales. This is what makes circuit A "as written" possible with OpenFHE's 59-bit scaling primes at `Omega_f ~ 2^62.8-63.2` (see section C). Encoding and decoding are our own (double-double FFT, exact rounding), not OpenFHE's (fixed per-level scales, double precision). |
| A6 | Calibrate | spec B_tot ~ 2^25.3 / Omega ~ 2^61.5 at the example | B_tot computed by a port of `btot.py`, generalized to (i) separate scales, (ii) OpenFHE's actual gadget: digit products and P of the context, `varsigma_up = 2 k_j` (OpenFHE's fast base extension returns digits in `[0, k_j Q~_j)`, uncentred) and ModDown rounding in `[0, k_P)` instead of `[-1/2, 1/2]`; T8 kept for circuit C | Faithful to the implemented key switching. Effect: B_tot = 2^26.6 (stage 2, example bounds) instead of 2^25.3, and Omega_f = 2^62.78 instead of 2^61.5. The generalized formulas reproduce btot.py exactly when the proof's settings are plugged in (test). |
| A7 | Calibrate | spec form of delta | Paper's corrected form: `gamma = n_rel(delta_B + 2^-kappa(1 + 2^-80) + 2^-97)`, `delta_DP = delta_run - (1 + e^eps) gamma`; sensitivity `Delta_enc = sqrt(2/N) Omega_f Delta + Delta_x + sqrt(N)` (centred mechanism); `sigma_c^2 = ceil(Delta_enc^2 / (2 rho))` (integer, as the exact sampler needs); admissible scales on a grid of step 2^(1/32), each exactly representable as a double | Paper corrections / proof review. kappa = 40, delta_B = 2^-50 (proof review's choices; paper tocheck). |
| A8 | Modulus chain | "two-prime q0 of about 70-73 bits" | `q0 = p0 p1` = 60 + 60 bits = 2^119 (chain [60, 60, 59, 59] + P [60, 60], log QP = 358 < 438) | OpenFHE requires `firstModSize >= scalingModSize` and scaling primes below 2^60. The specification's own condition gives 81 bits at n_bar_O = 4096 and 89 bits at 1M (see B1), so a 119-bit q0 is needed anyway at 1M margins; it costs no security margin here. |
| A9 | Aggregate | "modulus-switch M1, M2, My to q0" | no-op | After the two rescales the ciphertext is already at q0 = p0 p1 (as the proofs note). |
| A10 | Release | — | R sends a `release_request(stage)`; O answers each stage once, charges the ledger when it releases | Message-flow detail; the releases are non-adaptive. |
| A11 | Release debug | "eta = 0 by a clearly guarded debug flag" | O refuses `debug_no_dp` unless the environment variable `VFR_DEBUG_NO_DP=1` is also set; `run_protocol.py` has no switch for it; only tests set it | As requested. The test-only helpers that expose c0' and c1' s_R fingerprints are guarded the same way. |
| A12 | Release erasure | "erase c0', u, e0, e1, eta" | They live only inside one native call and go out of scope; memory is not zeroized | Zeroization is out of scope for this prototype. |
| A13 | Sampler | constant-time exact sampler (obligation O2) | Exact CKS20 sampler (big integers, ChaCha20 CSPRNG), **not** constant-time | (CT) is open in the proofs. |
| A14 | Solve | spec Algorithm 6 | Paper's corrected Algorithm 6: one-sided r1, closed-form lambda_O (Lemma S3(c)) instead of bisection, bootstrap truncation at 1/2, corrected `U_yy = min{yy^ + z_{1-alpha_U/2} sigma_yy + B_y^2 sqrt(n ln(2/alpha_U)/2), n B_y^2}`; antithetic pairs on; B = 2000 draws | Paper tocheck items; B is not fixed anywhere. alg6.py's `U_spec` is also returned. |
| A15 | Ledger | cap and composition unspecified | rho additive across releases and runs, statistical slack `(1 + e^eps) gamma` added to delta, cap `(eps_cap, delta_cap)` checked through the same tight conversion; OPRF quota counter per (researcher, project) | Minimal explicit choice; the spec leaves it open (paper tocheck). |
| A16 | Separate y'y release | f_y has no default | f_y = 0.1 | Only used in separate mode (not triggered by the example bounds, which are in joint mode). |
| A17 | Circuit C | restructured circuit (proof sketched) | Implemented as an option: EvalSum at the top level, our own stochastic rescale (`(c - r)/q` with `P[r = r0 - q] = r0/q`) for both rescales, one final rescale per release; OpenFHE's ModDown inside key switching stays deterministic | The ported bound for C divides the key-switching error by q2 and does not need a stochastic ModDown. Circuit A with Omega_U = 2^68 is the default (complete proof). |

## B. Things in the spec or paper that look wrong or underspecified

1. **q0 size.** The spec says q0 is "about 70-73 bits". Its own Calibrate condition
   `q0 > 2(Omega S_max + 2^kappa B_tot + 10 sigma_c)` with `S_max = n_bar_O max(r, o, w, ...)` gives 81 bits at
   n_bar_O = 4096 and 89 bits at 1M (circuit A, eps = 1); the sharper condition (Q) of the proofs gives 74-82 bits.
2. **Fast base conversion in the error bound.** The proof's `varsigma_up <= k_j` assumes centred digits; OpenFHE's
   ApproxModUp returns uncentred digits (`varsigma_up = 2 k_j`) and its ApproxModDown rounding is in `[0, k_P)`, not
   `[-1/2, 1/2]`. With OpenFHE these add about 1.3 bits to B_tot (A6).
3. **R's encoding scale.** With `Omega_U = q2` (the proofs' example), the term T1 (R's encryption error times O's
   data) is proportional to `Omega_V / q2` and grows with n_bar_O; at n_bar_O = 1M and eps = 4 it dominates and
   Calibrate needs Omega_f = 2^70.3 (q0 then needs 96.5 bits). Encoding R's vectors at `Omega_U ~ 2^68` makes B_tot
   and Omega_f independent of n_bar_O (2^62.75 at eps = 1, 2^64.6 at eps = 4, all sizes). **Decided (2026-10-24):
   circuit A with Omega_U = 2^68 is the default** (`log2_Omega_U='auto'`; `q2` keeps the old choice).
4. **PSI parameters are unspecified, and flooding constrains them.** SanitizeBFV needs `q' > 2t(t_eta 2^kappa B_err + ...)`
   after the switch (2 limbs of 60 bits for t ~ 2^32), and the data-dependent evaluation noise must be negligible
   after the switch, which forces a long BFV chain: 7 limbs (420 bits) at N_B = 2^14 with D = 64. `B_err` itself has
   no worst-case derivation for the BFV evaluation (A1).
5. **PSI item length vs delta_psi at 1M.** Items live in Z_t with t ~ 2^32. The expected number of false matches is
   about `n_R^max h n_bar_O / (N_bins t)`: 8.5e-5 per run at n_bar_O = 1M, n_R^max = 2000 (4e-4 at n_R^max = 10k), far
   above 2^-40. R aborts on ambiguous or out-of-range labels, but a false match with an in-range label is
   undetectable. Larger t or multi-slot items (APSI style) are needed for 1M.
6. **b and xdot_0 are the same vector.** The spec uploads `{b, xdot_0, ..., xdot_dR}`; the intercept column xdot_0
   equals b (1 on matches, 0 elsewhere). We follow the spec; dropping one saves 1/(d_R + 2) of the upload (61 MB at 1M).
7. **delta_s per release** in the ledger is not defined; Calibrate returns rho_s only (we charge `slack / n_rel`).
8. **"Erase"** and **constant time** are stated as requirements without an implementation model (A12, A13).
9. **ChooseBounds** is not specified (paper tocheck); bounds are inputs of the agreement here.
10. **kappa at large eps.** With the corrected delta accounting the statistical slack `(1 + e^eps) gamma` is 0.7% of
   delta at eps = 1, 10% at eps = 4 and more than delta at eps = 8 (delta = 1e-9, kappa = 40), where Calibrate
   refuses. kappa = 44 at eps = 4 (Omega_f 2^68.7) and kappa = 48 at eps = 8 (Omega_f 2^73.6 at 4k to 2^77.3 at
   1M) work; kappa should be a function of (eps, delta).

## C. The CKKS scale (known issue)

Measured / computed here (N = 2^14, d_R = 4, d_O = 6, example bounds, delta = 1e-9, kappa = 40):

| option | log2 Omega_f needed (eps 1 / 4) at 4k, 1M | chain | EvalSum, 1 thread | proof status |
|---|---|---|---|---|
| A as written, Omega_U = q2 (implemented option, `--log2_Omega_U q2`) | 62.78 / 64.72 at 4k; 63.19 / 70.28 at 1M | [60,60,59,59] + P[60,60], 358 bits | 0.24 s | complete (model M2); bound generalized |
| A, Omega_U = 2^68 (implemented, **default**) | 62.75 / 64.62 at every size | same | same | same as above |
| C restructured (implemented option) | 53.66 / 55.59 at 4k; 54.06 / 61.16 at 1M | same | 0.23 s (at the top level) | sketched in the proofs |
| OpenFHE composite scaling (not used; benchmarked) | native 2^64 | 8 Q limbs [45,45,32x6] + P[2], 376 bits, dnum 4 | 0.73 s (3.1x) | rescale by a product of primes; bound to redo |
| 128-bit native build (not used; benchmarked) | native 2^64 | [90,64,65] + P[119], 337 bits | 0.77 s (3.3x) | bound to redo for the 128-bit gadget |

Conclusion: the 60-bit prime cap is not a cap on the scale once the scales are decoupled from the primes and
tracked exactly, so circuit A runs as written in the 64-bit build at no extra cost; composite scaling and a 128-bit
build cost about 3x in the dominant key-switching operations.
