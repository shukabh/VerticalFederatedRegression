# Key numbers: budgeted sampler and release timing (check item T8)

Source: `timing.json` (written by `experiments/eng/release_timing_budget.py`, 2026-10-27). Machine: 4-core Xeon,
**shared with other jobs during the measurement: load average 16-23** (not idle). All timings are wall clock.

## Budget (public, from `sampler.budget`; xi_T = 2^-80 per call)

| call | sigma | N | b (t = 2^b) | p_lo | T | T / N | J_v | k_max | xi_bound | coin bytes |
|---|---|---|---|---|---|---|---|---|---|---|
| release, largest sigma_c^2 of the 1M runs (Pbn, eps 4, y'y release) | 2^68.37 (sigma^2 = 2^136.7) | 2^14 | 68 | 0.4457 | 39,088 | 2.39 | 69 | 29 | 2^-80.90 | 20.0 MB |
| SanitizeBFV flooding | 2^61 | 2^14 | 61 | 0.4805 | 36,189 | 2.21 | 69 | 29 | 2^-80.92 | 18.5 MB |

xi_bound = Chernoff-KL bound on P[Bin(T, p_lo) < N] (<= xi_T/2) + T (e^-J_v + 1/k_max! + (J_v + 1 + k_max) 2^-128).

Charge: gamma = n_rel (delta_B + 2^-kappa (1 + 2^-80) + 2^-97 + xi_T); hockey-stick ledger slack
n_rel (delta_B + delta' + xi_T). At n_rel = 2 the added 2^-79 is 2^-30 of the delta_B term (2^-49).
PSI: 2 alpha F calls per cuckoo table; at 1M rows (alpha = 5, F = 2, 1 table) 20 calls, 2^-75.7 <= delta_psi = 2^-40.

## Cost (interleaved, same process)

| | plain CKS20 + residues | budgeted (incl. residues) | ratio |
|---|---|---|---|
| release, sigma_c = 2^68.4, N = 2^14 (40 reps) | median 0.451 s (sd 0.057) | median 0.244 s (sd 0.023) | 0.54 |
| flooding, sigma_psi = 2^61, N_B = 2^14 (20 reps) | median 0.490 s | median 0.225 s | 0.46 |

The budgeted sampler does ~2.4x more candidate trials than needed on average, but in fixed-width numpy limb
arithmetic instead of Python big integers, so it is about twice as fast as the sampler it replaces.

## Budget failures (empirical)

| | calls | failures | undecided trials | accepted A: mean / min / max | min surplus A - N | min surplus in s.d. | acceptance rate vs p_lo |
|---|---|---|---|---|---|---|---|
| release sigma | 1,000 | 0 | 0 | 17,422 / 17,066 / 17,726 | 682 | 6.9 | 0.44570 vs 0.44572 |
| flooding sigma | 300 | 0 | 0 | 17,387 / 17,121 / 17,696 | 737 | 7.8 | 0.48046 vs 0.48052 |

p_lo (Poisson-summation lower bound) is tight: the empirical acceptance rate matches it to 4 digits.

## Value independence of the work (evidence for assumption (W))

Time of one call's T trials on crafted coins (same budget and shapes), medians:

| coins | run 1 (30 calls) | run 2 (100 calls) |
|---|---|---|
| random | 0.150 s | 0.123 s |
| every v = 0 | 0.141 s | 0.129 s |
| every v = J_v - 1 (largest \|y\|, none accepted) | 0.149 s | 0.122 s |
| every u = 0 | 0.142 s | 0.126 s |
| max relative difference of medians | 6.3% | 5.3% |

Per-call s.d. 0.015 s (10-12%); the ordering of the kinds is not reproducible between the runs. The plain CKS20
sampler, per value, by |eta| / sigma: median 9.7 us (< 1), 12.1 us (1-2), 16.5 us (2-3), 25.6 us (> 3), i.e. its
time does depend on the value.

## Padded release (deadline D, defence in depth)

D = 2 x (worst of 3 budgeted samplings 0.320 s + partial decryption 0.063 s) = 0.767 s, calibrated at load ~20.

| | releases | busy mean / sd / max | overruns |
|---|---|---|---|
| "idle" (load ~22 from other jobs) | 300 | 0.332 / 0.059 / 0.805 s | 1 (max 1.05 D) |
| plus 4 CPU-bound processes | 100 | 1.48 / 0.39 / 2.94 s | 100 |

Overruns now depend on machine load only (under (W)), not on eta, so they are neither charged nor needed for the
theorem; they cost availability, not privacy. The earlier measurement (`reports/eng/release_timing.json`, plain
sampler, idle machine) had D = 0.49 s and 0/1000 idle, 4/200 loaded.
