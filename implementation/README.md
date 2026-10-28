# Protocol v3.1: implementation

Two-party vertical federated ridge regression with threshold CKKS and lattice DP releases (honest-but-curious),
implemented end to end: R and O run as separate OS processes that exchange explicit, structured messages over a
socket, and both log a transcript of every message. Validated at small scale; benchmarked at 1k-64k register rows,
and run end to end at 100k, 300k and 1M register rows (n_R^max = 20,000 and 100,000; 78 runs; `reports/large/`): about
3 minutes per run at 1M on 4 cores, O's peak memory 2.3 GB, 475 MB of traffic. With O's OPRF evaluation now parallel
(A25) a run at 1M takes 2.0 minutes (O: 91 s; `reports/eng/`).

Deviations, choices and open points: [DEVIATIONS.md](DEVIATIONS.md). Results: `bench/results/` (validation.json,
benchmarks.md, cost_model.md, scale_options.json).

## Versions (as built and tested here)

| component | version |
|---|---|
| OpenFHE | 1.5.1 (tag v1.5.1, commit 1306d14f), built from source, 64-bit native integers, OpenMP; installed to /opt/openfhe-1.5.1. A 128-bit build (/opt/openfhe-1.5.1-n128) is used only by `bench/alt_scaling.cpp` |
| Python | 3.11.15; numpy 2.4.6, scipy 1.17.1, cryptography 41.0.7 (ChaCha20 CSPRNG), pybind11 3.1.0, pytest 9.1.1 |
| libsodium | 1.0.18 (Ubuntu libsodium23), via ctypes: ristretto255 for the OPRF |
| OpenSSL | 3.0.13 (libcrypto: SHAKE-256 for the seeded polynomials) |
| compiler | gcc 13.3.0 (needs libquadmath for the encoder's twiddle factors) |
| machine | 4-core Intel Xeon @ 2.8 GHz, 15 GB RAM |

The PyPI wheel `openfhe==1.5.1.0.22.4` is built for CPython 3.10 only and does not expose polynomial-level access,
which the release step needs (OpenFHE's own multiparty decryption adds its own floating-point smudging noise), so
the CKKS/BFV layer is a small C++ extension (`native/vfr_native.cpp`) on top of OpenFHE's DCRTPoly arithmetic,
NTTs, hybrid key switching and BFV evaluation.

## Build and run

```bash
scripts/build_openfhe.sh /opt/openfhe-1.5.1          # once (about 5 min on 4 cores)
python3 -m pip install numpy scipy cryptography pybind11 pytest
native/build.sh /opt/openfhe-1.5.1                   # builds vfr/vfr_native*.so
python3 -m pytest -q tests/                          # 107 tests, about 3 min (8 min on a loaded machine)
python3 run_protocol.py toy
python3 run_protocol.py synthetic --n_bar_O 4096 --n_R 500 --n 300   # [--eps 8] [--kappa auto|48] [--ledger_eps none]
#                                                     [--setting prev|I|P] [--r1_mode union|exact] [--f1 F] ...
python3 bench/validate.py                            # numbers in bench/results/validation.json
python3 bench/bench_micro.py --threads 1             # microbenchmarks (full parameters)
python3 bench/bench_e2e.py --sizes 1024 4096 16384 32768 65536 --threads 1 [--workers 4]
python3 bench/cost_model.py && python3 bench/scale_options.py && python3 bench/report_tables.py
python3 ../experiments/eng/refresh_cost_model.py         # re-measure the cost-model inputs (idle machine) and refit
```

## Layout and mapping to the specification

| phase (spec algorithm) | where | what |
|---|---|---|
| 0 Setup (Alg. 1, 7) | `vfr/dp.py` | `delta_rep` (exact enumeration over the three cubics, all regimes; Delta and Delta_noyy), `delta_G`, joint/separate mode, `rho_tight` (Canonne-Kamath-Steinke, rounded down), `eps_smudge` (eps'(kappa, delta') of the hockey-stick smudging lemma, rounded up), `choose_kappa` (derived flooding parameter: max(kappa_min, kappa_run, kappa_L), with the hockey-stick kappa_L rule by default), `calibrate` (paper's Algorithm 7: gamma with (1+e^eps), Delta_x of the centred mechanism, integer sigma_c^2, smallest admissible scale with sigma_c >= max(2^kappa B_tot / sqrt(2 pi), 2^42) (tested exactly, with B_tot rounded up), q0 check (diagnostic (Q) with the proved decoding offset), log QP against the 128-bit bound for N; the run's ledger cost; raises `CalibrationRefused`), `Ledger` (cap per base register; hockey-stick accounting with three sums rho, eps', Gamma and the share xi_L = 1e-3 of eps_L for smudging, or the earlier statistical-distance accounting with `xi=None`; 'filter' rule for adaptive runs, 'fixed' rule for runs fixed in advance; the full cost of a run is charged once, at admission, under a file lock, and never refunded; exact fractions, capacities rounded down, no tolerance; OPRF quota) |
| | `vfr/btot.py` | worst-case, data-independent B_tot (port of the proofs' btot.py, generalized to separate scales and OpenFHE's gadget; reproduces the proofs' table) and Delta_x |
| | `native: CKKSCore.pk_share / make_pk / rotkey_share / make_rotkeys` | threshold keygen with a CRS (SHAKE-256): `p0 = -a s + e` with `s = s_R + s_O`; joint rotation keys for the 13 shifts 2^i in OpenFHE's hybrid layout; no relinearization or conjugation key |
| 1 Link (Alg. 2) | `vfr/oprf.py` | 2HashDH OPRF, RFC 9497 mode 0, ristretto255-SHA512 over libsodium; passes the RFC test vectors; fresh key k per run; exactly n_R^max blinded queries |
| | `vfr/oprf.py` `evaluate_batch`, `blind_evaluate_batch`; `native: oprf_evaluate_batch, oprf_blind_evaluate_batch` | O's OPRF over its register and its answer to the blinded queries, batched and parallel over the items (OpenMP, the run's threads), the same libsodium calls and key; bit-identical to the item-by-item functions (DEVIATIONS A25) |
| | `vfr/psi.py`, `native: BFVCore, psi_build_polys` | labeled PSI: 64-bit items as F = 2 field elements (felts) of Z_t, felt f of bin b in slot f N_bins + b (ciphertext f); cuckoo hashing at R (h = 3, N_bins = 16384, k balanced tables with total failure bound <= delta_psi; random walk with an augmenting-path fallback), O's membership/label polynomials per slot and partition (label felts tau(j) + H_f(F'_k(id)) mod t); a match needs all felts zero and consistent label felts; Setup checks the false-match, cuckoo and SanitizeBFV conditions (`PSIParams.check`), windowed powers under R's own BFV key, fresh masks r1, r2, SanitizeBFV (switch to 2 limbs + Enc_pk(0) + exact discrete-Gaussian flooding with 2^40 B_err); R recovers I and pi, computes A exactly |
| 2 Upload (Alg. 3) | `native: encrypt_seeded / convert` | R: `c0 = -a s_R + e + Ecd(v)`, `a = PRG(seed)`, sends (c0, seed) (seeded form, half size); O: `c0 <- c0 - a s_O` |
| 3 Aggregate (Alg. 4) | `vfr/aggregate.py` | circuit A as written (`T_e = EvalSum(Rescale(sum_h U_e[h] * Ecd(V_e[h])))`, `P_e = Rescale(T_e * Ecd(u_slot))`, M1 = sum over (triu B, C), M2 = all stage-2 entries, My if separate); circuit C as an option; optional entry-parallel threads |
| 4 Release (Alg. 5) | `vfr/aggregate.release_share`, `native: rerand_partial_decrypt` | fresh ternary u and errors e0, e1; `c0' = c0 + p0 u + e0`, `c1' = c1 + a u + e1`; eta exact `N_Z(0, sigma_c^2)^N` from the budgeted sampler (`vfr/sampler.py`: a fixed public number of candidate trials in fixed-width limb arithmetic, failure probability <= `sampler_xi` = 2^-80 per release, charged in gamma; DEVIATIONS A13, A24); `h_O = c0' + c1' s_O + eta mod q0`; sends only (c1', h_O) (the ledger was charged at admission); each release is held until a deadline fixed before the first one from public parameters (`Agreement.release_pad`, default 'auto': 2 x the calibrated time of one release), and refused if its computation overruns it (DEVIATIONS A24) |
| | `native: combine / decode_i128` | R: `v = h_O + c1' s_R mod q0`, CRT to int128, decode with the exact rational final scale |
| 5 Solve and infer (Alg. 6) | `vfr/solve.py` | corrected Algorithm 6 (port of alg6.py with the paper's corrections); stage-1 radius r_1 by the union bound (`r1_mode='union'`) or the exact Monte Carlo quantile (`'exact'`, `r1_quantile_exact`, cached in `vfr/r1_quantiles.json`); `intervals=False` for estimate-only settings |
| parties | `vfr/party_r.py`, `vfr/party_o.py`, `vfr/protocol.py`, `vfr/wire.py` | two processes, one socket; JSON header + binary fields, no pickle; per-message transcript (type, direction, bytes per field, SHA-256); each party also returns local phase marks (wall time, process CPU time over all threads, peak RSS; `common.mark`) |
| inputs | `vfr/data.py` | toy example; synthetic cohorts from `experiments/sim_core.Population` (d_R = 4, d_O = 6) |
| preprocessing (Setup, `Preprocess`) | `bench/utility_runs.py` (`transform`) | each party transforms its own columns before the protocol with constants from a public reference split: standardize, optional winsorizing / rank-Gaussian, optional block unit norm; bounds = 99.5% quantiles on the transformed reference data. It is local computation, so it runs before `protocol.run` |

Statistical settings (`vfr/dp.py`, `SETTINGS`; DEVIATIONS A18-A19). The Agreement defaults are the previous
evaluation setting `'prev'` (union-bound r_1, f_1 = 0.1, theta* = 2.5, kappa_Psi = 2, alpha_L = alpha_U = alpha/20).
The paper's two settings of the same protocol are `'I'` (inference: exact r_1, f_1 = 0.25, theta* = 2.5,
alpha_L = alpha_U = 0.0025, the interval guarantee) and `'P'` (prediction: exact r_1, f_1 = 0.25, theta* = 1.5,
alpha_L = 0.75, no intervals); both keep the gate theta_hat >= 1 and kappa_Psi = 2. They were chosen on the synthetic
population only: I by the rule pre-registered in `reports/v31_settings/settings_choice.md`, P by the rule in
`reports/v31_P2/settings_choice_P2.md`, which adds a block-normalized population and a gate-refusal bound of 1%.
`'P0'` (theta* = 0.5, alpha_L = 0.5, f_1 = 0.2) is the first prediction setting, kept only to reproduce earlier
results: its gate refuses most releases on block-normalized real data, where Delta_G is small against Delta. Pass them as
`dp.Agreement(..., **dp.setting('I'))` or `run_protocol.py --setting I`. Both parties check every statistical field
of the agreement; O refuses an ill-formed setting (intervals on with alpha_L + alpha_U >= alpha).

Parameters (`vfr/common.py`, the paper's evaluation settings): CKKS N = 2^14, chain q = [p0 60, p1 59, p2 59, p3 59]
bits with q0 = p0 p1 (2^119), q1 = p2, q2 = p3, special primes P = 2 x 60 bits, dnum = 2, log QP = 357 (128-bit
secure for N = 2^14; OpenFHE's check, repeated by Calibrate). Circuit A (as written) with R encoding at Omega_U = 2^68 (the default, `log2_Omega_U='auto'`; decided
2026-10-24), masks at Omega_u = q1 and O's data at Omega_V from Calibrate; the final scale
Omega_f = Omega_U Omega_V Omega_u / (q1 q2) is tracked exactly. With the derived flooding parameter (`kappa='auto'`:
kappa = max(40, kappa_run, kappa_L), zeta = 0.1; single-run ledger) kappa = 40 / 40 / 40 / 41 / 46 and Omega_f =
2^61.8 / 2^62.75 / 2^63.7 / 2^65.6 / 2^71.6-2^71.9 for eps = 0.5 / 1 / 2 / 4 / 8 at n_bar_O from 4k to 1M
(delta = 1e-9); at a fixed kappa = 40, eps = 4 needs 2^64.6 and eps = 8 is refused (slack > delta). An agreement may
name a ledger cap (`ledger_eps`, `ledger_delta`, `ledger_zeta`, `ledger_xi`); `run_protocol.py` uses (10, 1e-6) for its
persistent demo ledger. With the hockey-stick ledger (xi_L = 1e-3, the default) the ledger's kappa_L is 22 / 21 / 19 /
17 / 15 and does not raise kappa; the earlier statistical-distance ledger (`ledger_xi=None`, `--ledger_xi none`) raises
it to 48 / 46 / 44 / 42 / 46. `--log2_Omega_U q2` selects the
earlier choice Omega_U = q2. The benchmarks in `bench/results/` were measured with Omega_U = q2; the modulus chain
and the operations are the same, so the timings carry over. BFV N_B = 2^14,
t = 4293918721, 2 felts per item (64-bit items; `PSIParams.felts`), 7 x 60-bit limbs, reply switched to 2 limbs,
SanitizeBFV flooding kappa_psi = 40. delta_B = 2^-50, delta_psi = 2^-40. At 1M rows and n_R^max = 20,000 the
false-match bound p_pass is 2^-67.5 and R's two cuckoo tables fail with probability at most 2^-43.4 (DEVIATIONS A22, A23).

## Tests (`tests/`, 107 passing)

- `test_primitives.py`: RFC 9497 vectors; exact sampler (chi-square at sigma^2 = 3, moments and low bits at
  sigma = 2^55); DeltaRep against the closed forms, the paper's (1, 1, 5) example and a random search; rho_run =
  0.01496; the generalized B_tot reproduces the proofs' table (2^25.34 for A, 2^17.56 for C'); Calibrate floods;
  `round_published` rounds to the stated precision relative to sigma.
- `test_ckks_psi.py`: encoder round trip at scales up to 2^61.5; seeded upload and key conversion (and failure
  with a wrong share or seed); EvalSum with joint keys = trace; circuits A, A with Omega_U = 2^68, and C reproduce
  the plaintext statistics with eta = 0; the randomness-dependent error measured from two independent key sets
  stays more than 3 bits below the worst-case B_tot; fresh re-randomization; labeled PSI exact recovery with two felts
  through BFV and SanitizeBFV; forced near collisions (one felt alone, or two felts from two different items of a
  partition, is not a match; a full 64-bit collision with a consistent label is); the repeated-position check with
  one and two felts; the PSI parameter check (two felts meet delta_psi up to 1M rows and n_R^max = 100,000, one felt
  is refused, a one-limb reply fails the switch condition); balanced cuckoo tables; the cuckoo insertion's augmenting-path fallback
  (forced by a cut-off walk) returns an assignment exactly when one exists (checked against scipy's bipartite
  matching), and the walk alone is unchanged at the evaluation's load.
- `test_protocol.py` (two processes): toy example (I = {104, 221}, pi(104) = 6, pi(221) = 3; B, C, c_R, c_O, y'y
  equal toy.json to < 1e-10); synthetic cohort n_bar_O = 4096, n = 300: exact match set and alignment, noise-off
  G and c equal to 1e-11 relative; transcript checks (allowed message types only; both transcripts agree frame by
  frame; a release carries exactly two ring elements mod q0; Setup refuses one-felt PSI items before charging the ledger; OPRF count = n_R^max; the bytes of c0' never appear in
  anything R receives; the bytes of R's would-be share c1' s_R never appear in anything O receives); noise on: the
  stage-2 estimate, both intervals, V_mech, RSS correction equal alg6.py fed the same decoded statistics; the debug
  flag is refused without the test-only environment variable; O's ledger holds the run's full cost, charged once at
  admission.
- `test_calibrate.py`: the derived kappa (40, 40, 40, 41, 46 for eps = 0.5 ... 8 at delta = 1e-9) is the smallest
  admissible and floods; eps = 8 is feasible at 1M rows; a fixed kappa = 40 at eps = 8 is refused; refusals for a
  chain above the security bound, a too small q0 and a too large delta_B; the hockey-stick kappa_L (22, 21, 19, 17, 15
  at the cap (10, 1e-6), xi_L = 1e-3), with kappa then equal to the single-run kappa, and the statistical-distance
  kappa_L (46 at eps = 1) with `ledger_xi=None`; under either accounting the ledger filter is exhausted by rho, not by
  slack; at a fixed kappa = 40 the statistical-distance ledger converts the slack at the cap's eps and refuses after two
  runs, the hockey-stick ledger after 102; a single-run ledger is filled exactly; the floor sigma_c >= 2^42, and
  Calibrate raises the scale to it when it binds; the flooding test is conservative (a case that the double-precision
  test passes at the boundary is refused, and every release floods in exact rational arithmetic); the q0 diagnostic
  uses the proved decoding offset (d_max = 2^30.3 at 1M rows with n_R^max = 2,000, 2^38.6 if every row may match).
- `test_ledger.py`: eps'(kappa, delta') equals the tight values of the composition proof and is below the closed form;
  rho_tight rounds down and is tight to 1e-9; the capacities are rounded down and the delta budget adds up (Corollary
  5.4); the filter and the fixed rule test without tolerance (a run that fills a capacity exactly is admitted, the
  smallest excess refused); the hockey-stick fixed rule; the smudging budget can bind for a run below its kappa_L; a
  run's full cost is charged at admission, is visible to every handle on the file and is never refunded; older ledger
  files are refused; two stale handles, and four processes racing on one file, admit exactly what the cap allows.
- `test_stage1_radius.py`: the exact r_1 gives the one-sided failure rate alpha_L on 400k fresh noise draws
  (alpha_L = 0.0025 and 0.25; rate within 3 s.d. above and at least 0.8 alpha_L); the order-statistic upper bound
  covers the true quantile at its confidence level; the union bound is still the default and matches the formula;
  the agreement's alpha split, intervals flag and named settings; a two-process run threads `r1_mode='exact'`,
  alpha_L = 0.25 and `intervals=False` through both parties, and O refuses an ill-formed setting.
- `test_oprf_parallel.py`: the batched, parallel OPRF evaluation (native, OpenMP) is bit-identical to the
  item-by-item reference at 1, 2, 4 and 7 threads, on the RFC 9497 vectors and on R's blinded queries (R's finalized
  outputs equal O's direct evaluation); invalid points, the identity element and a short key are refused.
- `test_release_timing.py`: release padding: the hold ends at the deadline whatever the work before it took; with a
  fixed deadline (1.5 s) and with 'auto', every release takes exactly the deadline at O (to 10 ms) and R's latency of
  the second release lies in [D, D + 0.25 s]; an overrun (deadline 0.1 ms) is refused, never sent, and the run's
  cost stays charged; release_pad=None restores the earlier behaviour; ill-formed values are refused in Setup.
- `test_noise.py`: 40 + 20 releases of one packed ciphertext: pooled s.d. of (decoded - true) within 6% of
  sigma_s from Calibrate, per-entry s.d. within [0.4, 1.9] sigma_s.
