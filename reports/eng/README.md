# Engineering items (2026-10-26): parallel OPRF, refreshed cost model, padded release timing, full-record SSP cost

Machine: 4-core Xeon @ 2.8 GHz, 15 GB, idle unless stated. Scripts are in `experiments/eng/`; raw logs and run
records in `experiments/_work/eng/` (git-ignored). No encrypted protocol was run above 1M register rows.

## 1. Parallel OPRF (`oprf_speed.json`, `oprf_speed.py`)

O's OPRF over its register and its answer to R's blinded queries run in the native layer, in parallel over the
items with OpenMP, with the same libsodium calls and key as the item-by-item code (DEVIATIONS A25). Outputs were
compared in full: identical at every size and thread count (also `implementation/tests/test_oprf_parallel.py`).

| n_bar_O | Python, 1 thread | native, 1 thread | native, 4 threads | speedup vs Python |
|---|---|---|---|---|
| 106,496 | 7.78 s | 6.77 s | 1.85 s | 4.2x |
| 1,007,616 | 72.7 s | 67.5 s | 17.7 s | 4.1x |

The 20,000 blinded queries: 1.25 s -> 0.32 s.

## 2. Full runs with the parallel OPRF (`run_large_eng.json`, `run_large_eng.py`)

One run each of cells A100k_Istd_e1 and A1M_Istd_e1 of `reports/large` (setting I, eps = 1, n_R^max = 20,000, 4 threads
per party, 4 parallel entries, release padding on). O's total was re-measured by a full run, not by substitution.

| | 100k | 1M |
|---|---|---|
| O total (s), earlier runs (single-threaded OPRF, mean of 3) | 34.5 | 151.9 |
| O total (s), parallel OPRF | 28.6 | 91.1 |
| of which OPRF (s) | 2.4 (was 8.4) | 18.6 (was 79.8) |
| run end to end (s) | 36 (was 42) | 120 (was 179) |
| refreshed model, parallel OPRF (s) | 25.9 | 85.3 |
| peak memory O / R (MB), traffic (MB) | 762 / 363, 133 | 2,333 / 1,733, 475 |

Both runs linked exactly; slot-noise s.d. 0.993-1.008. Release deadline 0.49 s (100k) and 0.51 s (1M); R saw the
second release 2 ms after it (the first also waits for O's aggregation).

## 3. Refreshed cost model (`model_check.json`, `model_check.py`; inputs in `implementation/bench/results/`)

`refresh_cost_model.py` re-ran `bench_micro.py` at 1 and 4 threads, `experiments/large/agg_speed.py` at
(threads, entries) = (1, 1), (4, 4), (1, 4), and the interpolation at the actual partition fill, and refitted
`cost_model.py` (`micro_t1.json`, `micro_t4.json`, `agg_speed.json`, `polys_fill.json`, `cost_model.json/.md`;
`micro_t1_logN15.json` was not re-measured). Changes to the model:
- PSI interpolation at the actual fill: membership polynomials (always degree D) from an empty-partition measurement,
  label interpolation scaled by E[sum_a n_a^2] under Binomial(n_bar_O h, 1/N_bins) bin loads; on 4 threads the static
  OpenMP schedule's imbalance (partition 0 is the fullest). Measured / model: 0.88-0.97 (1 thread), 0.91-0.98
  (4 threads) at 100k-1M.
- Aggregation speedup of the configuration that is run, 4 threads x 4 entries: 2.83 (one thread per entry: 3.51).
- `predict_run`: per-phase wall time of a run in the large-run configuration, with the OPRF of the code that ran
  (`oprf='python'` for the 78 runs, `'native'` now) and optional release padding; each run's own dimensions
  (3droad: d_R = 1, d_O = 2); binning fitted at 200,000 rows.

Measured / predicted for the 78 runs of `reports/large` (same code as those runs):

| stage | O total | gated phases at O (>= 2 s) | misses (> 25%) |
|---|---|---|---|
| 1 (100k) | 1.01-1.07 | 0.96-1.07 | R's OPRF 1.25, 1.27 |
| 2 (300k) | 1.03-1.12 | 0.95-1.15 | R's OPRF 1.29, 1.30 |
| 3 (1M) | 0.98-1.04 | 0.86-1.13; upload conversion 1.11-1.36 | upload conversion 1.36; R's OPRF 1.31 |
| 4 (1M, n_R^max = 100k) | 0.99-1.07 | 0.91-1.17; upload conversion 1.23-1.40 | upload conversion 1.27-1.40; R's OPRF 1.27 |
| 5 (3droad) | 0.95-1.06 | 0.91-1.09 | R's OPRF 1.27 |

Earlier: 0.59-0.80 against the published model and 0.94-1.25 against the model with re-measured inputs
(`reports/large/README.md`). The upload conversion at 1M remains the one phase at O outside 25%: the isolated
microbenchmark misses the memory traffic of 738 MB of ciphertexts. R's OPRF is measured as CPU time between phase
marks. Note: `experiments/large/lcommon.py`'s 'published' inputs read `implementation/bench/results/`, which now
hold the refreshed values; the stage JSONs keep the values computed at the time.

## 4. Padded release timing (`release_timing.json`, `release_timing.py`)

Design (DEVIATIONS A24; `Agreement.release_pad = 'auto'`, `release_pad_margin = 2`): after the aggregation O fixes
D = 2 x (worst of 3 timed exact samplings of N = 2^14 values at the run's largest sigma_c^2, with an independent
generator, + one re-randomized partial decryption with eta = 0, discarded). Each release is held until D after O takes
up the request and is sent with D; a release whose computation overruns D is refused (never sent late; the run's
cost stays charged). Outside an overrun, R's latency is D plus scheduling jitter, independent of eta, so the timing
channel adds at most P[overrun] to the statistical distance.

At the largest sigma_c of the 1M runs (2^68.5): D = 0.486 s.

| condition | releases | busy time mean +- s.d. | max / D | overruns |
|---|---|---|---|---|
| idle | 1,000 | 0.229 +- 0.023 s | 0.68 | 0 (P <= 0.3% at 95%) |
| 4 CPU-bound processes competing, D calibrated idle | 200 | 0.294 +- 0.064 s | 1.07 | 4 |

P[overrun] is measured, not bounded, and not charged to delta. The flooding of the PSI replies is not padded.

## 5. Full-record SSP cost, measured (`fullrecord.json`, `fullrecord_bench.py`)

Benchmark path (not a protocol mode): R encrypts and O converts the 10 extra vectors b x_j x_k; O aggregates 63 vs 77
entries under encryption (the 14 entries of triu A without n against a vector of ones); one release of each packed
ciphertext; a full v3.1 run as the baseline. Single thread, n_R^max = 2,000, d_R = 4, d_O = 6. A noiseless test
decryption returns triu A to 1e-11.

| n_bar_O | aggregation v3.1 -> full | x (model) | whole run x (model) | extra upload | traffic x (model) |
|---|---|---|---|---|---|
| 16,384 | 7.35 -> 8.89 s | 1.21 (1.22) | 1.08 (1.16) | 10.0 MB (10.0) | 1.15 (1.19) |
| 65,536 | 10.21 -> 11.74 s | 1.15 (1.22) | 1.09 (1.16) | 40.0 MB (40.0) | 1.46 (1.55) |

The whole-run ratio is (v3.1 run + measured extra work) / v3.1 run, with v3.1 runs of 21.2 and 33.2 s; the model's
traffic ratios used a one-felt v3.1 baseline (two felts now). The releases cost the same. The U-side bound of the
product vectors in B_tot was not re-derived (only re-evaluated with the larger entry counts).
