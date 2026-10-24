# Protocol v3.1: implementation

Two-party vertical federated ridge regression with threshold CKKS and lattice DP releases (honest-but-curious),
implemented end to end: R and O run as separate OS processes that exchange explicit, structured messages over a
socket, and both log a transcript of every message. Validated at small scale; benchmarked at 1k-64k register rows
and extrapolated to 100k-1M by a fitted cost model. **Nothing was run at n_bar_O >= 100k.**

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
python3 -m pytest -q tests/                          # 29 tests, about 2.5 min
python3 run_protocol.py toy
python3 run_protocol.py synthetic --n_bar_O 4096 --n_R 500 --n 300
python3 bench/validate.py                            # numbers in bench/results/validation.json
python3 bench/bench_micro.py --threads 1             # microbenchmarks (full parameters)
python3 bench/bench_e2e.py --sizes 1024 4096 16384 32768 65536 --threads 1 [--workers 4]
python3 bench/cost_model.py && python3 bench/scale_options.py && python3 bench/report_tables.py
```

## Layout and mapping to the specification

| phase (spec algorithm) | where | what |
|---|---|---|
| 0 Setup (Alg. 1, 7) | `vfr/dp.py` | `delta_rep` (exact enumeration over the three cubics, all regimes; Delta and Delta_noyy), `delta_G`, joint/separate mode, `rho_tight` (Canonne-Kamath-Steinke), `calibrate` (paper's corrected form: gamma with (1+e^eps), Delta_x of the centred mechanism, integer sigma_c^2, smallest admissible scale with sigma_c >= 2^kappa B_tot / sqrt(2 pi), q0 check), `Ledger` (cap, per-release charge, OPRF quota) |
| | `vfr/btot.py` | worst-case, data-independent B_tot (port of the proofs' btot.py, generalized to separate scales and OpenFHE's gadget; reproduces the proofs' table) and Delta_x |
| | `native: CKKSCore.pk_share / make_pk / rotkey_share / make_rotkeys` | threshold keygen with a CRS (SHAKE-256): `p0 = -a s + e` with `s = s_R + s_O`; joint rotation keys for the 13 shifts 2^i in OpenFHE's hybrid layout; no relinearization or conjugation key |
| 1 Link (Alg. 2) | `vfr/oprf.py` | 2HashDH OPRF, RFC 9497 mode 0, ristretto255-SHA512 over libsodium; passes the RFC test vectors; fresh key k per run; exactly n_R^max blinded queries |
| | `vfr/psi.py`, `native: BFVCore, psi_build_polys` | labeled PSI: cuckoo hashing at R (h = 3, N_bins = 16384), O's membership/label polynomials per bin and partition (labels tau(j) + H(F'_k(id)) mod t), windowed powers under R's own BFV key, fresh masks r1, r2, SanitizeBFV (switch to 2 limbs + Enc_pk(0) + exact discrete-Gaussian flooding with 2^40 B_err); R recovers I and pi, computes A exactly |
| 2 Upload (Alg. 3) | `native: encrypt_seeded / convert` | R: `c0 = -a s_R + e + Ecd(v)`, `a = PRG(seed)`, sends (c0, seed) (seeded form, half size); O: `c0 <- c0 - a s_O` |
| 3 Aggregate (Alg. 4) | `vfr/aggregate.py` | circuit A as written (`T_e = EvalSum(Rescale(sum_h U_e[h] * Ecd(V_e[h])))`, `P_e = Rescale(T_e * Ecd(u_slot))`, M1 = sum over (triu B, C), M2 = all stage-2 entries, My if separate); circuit C as an option; optional entry-parallel threads |
| 4 Release (Alg. 5) | `vfr/aggregate.release_share`, `native: rerand_partial_decrypt` | fresh ternary u and errors e0, e1; `c0' = c0 + p0 u + e0`, `c1' = c1 + a u + e1`; eta exact `N_Z(0, sigma_c^2)^N` (big integers, `vfr/sampler.py`); `h_O = c0' + c1' s_O + eta mod q0`; ledger charge; sends only (c1', h_O) |
| | `native: combine / decode_i128` | R: `v = h_O + c1' s_R mod q0`, CRT to int128, decode with the exact rational final scale |
| 5 Solve and infer (Alg. 6) | `vfr/solve.py` | corrected Algorithm 6 (port of alg6.py with the paper's corrections) |
| parties | `vfr/party_r.py`, `vfr/party_o.py`, `vfr/protocol.py`, `vfr/wire.py` | two processes, one socket; JSON header + binary fields, no pickle; per-message transcript (type, direction, bytes per field, SHA-256) |
| inputs | `vfr/data.py` | toy example; synthetic cohorts from `experiments/sim_core.Population` (d_R = 4, d_O = 6) |

Parameters (`vfr/common.py`): CKKS N = 2^14, chain q = [p0 60, p1 60, p2 59, p3 59] bits with q0 = p0 p1,
q1 = p2, q2 = p3, special primes P = 2 x 60 bits, dnum = 2, log QP = 358 (128-bit secure for N = 2^14, OpenFHE's
check). Circuit A (as written) with R encoding at Omega_U = 2^68 (the default, `log2_Omega_U='auto'`; decided
2026-10-24), masks at Omega_u = q1 and O's data at Omega_V from Calibrate; the final scale
Omega_f = Omega_U Omega_V Omega_u / (q1 q2) is tracked exactly: 2^61.8 / 2^62.75 / 2^63.7 / 2^64.6 for
eps = 0.5 / 1 / 2 / 4 at every n_bar_O from 4k to 1M (delta = 1e-9, kappa = 40). `--log2_Omega_U q2` selects the
earlier choice Omega_U = q2. The benchmarks in `bench/results/` were measured with Omega_U = q2; the modulus chain
and the operations are the same, so the timings carry over. BFV N_B = 2^14,
t = 4293918721, 7 x 60-bit limbs, reply switched to 2 limbs. kappa = 40, delta_B = 2^-50, delta_psi = 2^-40.

## Tests (`tests/`, 29 passing)

- `test_primitives.py`: RFC 9497 vectors; exact sampler (chi-square at sigma^2 = 3, moments and low bits at
  sigma = 2^55); DeltaRep against the closed forms, the paper's (1, 1, 5) example and a random search; rho_run =
  0.01496; the generalized B_tot reproduces the proofs' table (2^25.34 for A, 2^17.56 for C'); Calibrate floods.
- `test_ckks_psi.py`: encoder round trip at scales up to 2^61.5; seeded upload and key conversion (and failure
  with a wrong share or seed); EvalSum with joint keys = trace; circuits A, A with Omega_U = 2^68, and C reproduce
  the plaintext statistics with eta = 0; the randomness-dependent error measured from two independent key sets
  stays more than 3 bits below the worst-case B_tot; fresh re-randomization; labeled PSI exact recovery.
- `test_protocol.py` (two processes): toy example (I = {104, 221}, pi(104) = 6, pi(221) = 3; B, C, c_R, c_O, y'y
  equal toy.json to < 1e-10); synthetic cohort n_bar_O = 4096, n = 300: exact match set and alignment, noise-off
  G and c equal to 1e-11 relative; transcript checks (allowed message types only; both transcripts agree frame by
  frame; a release carries exactly two ring elements mod q0; OPRF count = n_R^max; the bytes of c0' never appear in
  anything R receives; the bytes of R's would-be share c1' s_R never appear in anything O receives); noise on: the
  stage-2 estimate, both intervals, V_mech, RSS correction equal alg6.py fed the same decoded statistics; the debug
  flag is refused without the test-only environment variable.
- `test_noise.py`: 40 + 20 releases of one packed ciphertext: pooled s.d. of (decoded - true) within 6% of
  sigma_s from Calibrate, per-entry s.d. within [0.4, 1.9] sigma_s.
