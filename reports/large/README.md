# Large-scale evaluation of protocol v3.1 (register sizes 100k, 300k and 1M rows)

This evaluation ran the full OpenFHE implementation 78 times on the production path:
- two processes over a socket, with exact discrete-Gaussian noise inside O's decryption share;
- a single-run ledger in a fresh file for each run;
- 4 OpenMP threads per party, and 4 parallel entries in O's aggregation;
- implementation at commit b6c613c (64-bit PSI items, two felts per item).

Parameters:
- N = 2^14, modulus chain [60,59,59,59] + P[60,60];
- δ = 1e-9 and κ = auto;
- settings I and P (`vfr/dp.py`), κ_Ψ = 2, no bias correction;
- PSI with h = 3, δ_psi = 2^-40, F = 2 felts per item.

Each cell was run 3 times on one fixed dataset. The plaintext comparisons use the same data and the σ of the runs.

Scripts are in `experiments/large/`:
- `run_large.py` runs the encrypted runs;
- `summarize_large.py` checks each stage and its gates;
- `micro_large.py` and `agg_speed.py` re-measure the cost-model inputs;
- `sim_large.py` simulates at large n;
- `tables_large.py` writes `key_numbers.md` and `summary.json`;
- `figures_large.py` draws `paper/figures/{cost,noise,accuracy}_large.png`.

Raw per-run records are in `experiments/_work/large/*.jsonl` (git-ignored).

## Cells

| stage | cells | data |
|---|---|---|
| 1, 2, 3 (regime A) | n̄_O = 106,496 / 303,104 / 1,007,616 (100k / 300k / 1M real rows, padded to a multiple of 8,192); n_R^max = 20,000 (cohort of 15,000, 12,000 matched); {I std, P std, P block-normalized} × ε ∈ {1, 4} | synthetic population (`vfr/data.py`), standardized with public constants (std) or block-normalized (bn, the P2 population (b)) |
| 4 (regime B) | n̄_O = 1,007,616; n_R^max = 100,000 (cohort of 100,000); 60,000 or 95,000 matched; setting I std; ε ∈ {0.5, 1} | synthetic, std |
| 5 (3droad) | O's register = 304,413 rows (padded to 311,296); 10% reference, 20% test; n_R^max = 20,000, 12,000 matched; R holds feature 1 plus the intercept, O holds features 2 and 3 and y; block normalization; {I, P} × ε ∈ {1, 4} | 3droad |

## Gates (fixed in `summarize_large.py` before the runs) and outcome

- Cost: every phase that takes at least 2 s, and O's total, must lie within 25% of the cost model.
- PSI must be exact, both the match set and every position.
- λ must lie inside the simulated 95% range in all but at most max(1, 10%) of the runs.
- At least 85% of the estimates must lie inside the simulated 95% ranges, with |z| < 4 against the law given λ.
- The noise s.d. must be within 3% of 1.

| stage | PSI | λ | estimates | noise s.d. | cost vs published model | cost vs re-measured model |
|---|---|---|---|---|---|---|
| 1 (100k) | 18/18 exact | 18/18 inside | 0.965 inside, max \|z\| 3.42 | 0.995–1.003 | **fail**: O total 0.66–0.69 of prediction | **pass**: O total 1.12–1.19, every phase within 25% |
| 2 (300k) | 18/18 | 18/18 | 0.939, 3.23 | 0.995–1.007 | **fail**: 0.73–0.80 | O total 1.15–1.25; **2 phases in 2 of 6 cells miss** (see below) |
| 3 (1M) | 18/18 | 18/18 | 0.975, 2.51 | 0.997–1.002 | **fail**: 0.74–0.78 | O total 1.10–1.17; **upload conversion misses in 5 of 6 cells** (see below) |
| 4 (regime B) | 12/12 | 12/12 | | 0.996–1.005 | 0.70–0.76 | 1.11–1.20 |
| 5 (3droad) | 12/12 | 12/12 | | 0.994–1.007 | 0.59–0.66 | 0.94–1.05 (the model's entry count is that of p = 11, not 4) |

**The published cost model failed the 25% gate at every size, because it over-predicts.** The coordinator decided that a result below the model is not a safety failure. Under that decision, the inputs were re-measured on the idle machine and the gate was judged against the re-measured model. Both comparisons are kept.

The re-measurement found three things:
1. The microbenchmarks behind `bench/results/micro_t{1,4}.json` run 1.4–2.2× faster today than in the committed files.
2. The interpolation term assumes full partitions. The actual fill is 0.15 / 0.43 / 0.58 at 100k / 300k / 1M.
3. The published aggregation speedup (3.96) was measured with one thread per entry. The runs use 4 OpenMP threads × 4 entries, for which the measured speedup is 3.13.

Two further adjustments apply to the re-measured model:
- The single-threaded OPRF and binning phases use the single-thread inputs.
- The structure of the model is unchanged otherwise.

Remaining misses against the re-measured model, all too slow and all in small phases:
- **300k:** PSI preprocessing at 1.27 and 1.26, and PSI evaluation at 1.30, in two cells. In one of these cells every phase ran 5–8% slower, including the single-threaded OPRF.
- **1M:** O's upload conversion takes 4.8 s against 3.6 s re-measured (1.36). The published model predicted 4.8 s exactly. The isolated microbenchmark misses the memory traffic of 738 MB of ciphertexts.

I continued to the later stages with these misses recorded, because O's total stays within 25% and memory stays near 2.3 GB per party. **Read strictly, stages 2 and 3 fail the re-measured cost gate.**

## Cost (setting I, ε = 1; seconds, wall time, 4 threads)

| | 100k | 300k | 1M | 1M, n_R^max = 100k | 3droad 311k |
|---|---|---|---|---|---|
| O total, measured | **34.5** | **56.6** | **151.9** | **270.5** | **56.4** |
| O total, published model | 50.2 | 77.0 | 193.9 | 357.3 | 85.1 |
| O total, re-measured model | 29.0 | 49.2 | 130.1 | 225.0 | 53.8 |
| run, end to end | 42 | 67 | 179 | 313 | 65 |
| O CPU (all threads) | 78 | 109 | 256 | 550 | 97 |
| peak memory O / R (MB) | 761 / 362 | 990 / 660 | 2,332 / 1,732 | 2,337 / 1,790 | 901 / 439 |
| traffic (MB; equals the model) | 132.8 | 204.8 | 474.9 | 756.2 | 155.3 |

At 1M (n_R^max = 20,000), O's time divides as follows:

| phase | seconds |
|---|---|
| OPRF of the whole register (single-threaded) | 80 |
| PSI binning and interpolation | 19 |
| aggregation | 18 |
| PSI evaluation | 15 |
| SanitizeBFV | 12 |
| upload conversion | 5 |

R's upload is 369 MB of the 475 MB of traffic. At n_R^max = 100,000, the PSI evaluation (74 s) and SanitizeBFV (62 s) grow with the 10 cuckoo tables, and the traffic grows to 756 MB.

The paper's earlier extrapolation (one felt, n_R^max = 2,000) was:
- O's CPU time of 52 / 99 / 260 s on one core;
- 76 s on four cores at 1M, which assumed a parallel OPRF that is not implemented;
- 88–422 MB of traffic.

## PSI

- 78 of 78 runs were exact: the match set and every position agreed with the plaintext join.
- The cuckoo fallback never ran, and no felt-wise false match occurred.

Bounds at 1M and n_R^max = 20,000:

| event | probability |
|---|---|
| a false match passes both tests (can abort or change the output) | 2^-67.5 |
| a false match goes undetected | 2^-79.6 |
| any felt matches (APSI's rule alone; inconsistent labels are dropped) | 2^-35.5 |
| cuckoo failure, 2 tables | 2^-43.4 |
| cuckoo failure, 10 tables, at n_R^max = 100k | 2^-41.1 |

## Noise

- Pooled over all 78 runs, the slot noise has 3.04 million values. Its mean is −0.0007 (z = −1.2) and its s.d. is 0.9997.
- Over the released entries:
  - stage 1: 3,450 values, mean +0.003, s.d. 1.000, KS p = 0.83;
  - stage 2: 4,224 values, mean +0.007, s.d. 1.012, KS p = 0.41.
- Per-run KS p-values range from 0.008 upward, across about 200 tests.

## Statistics against simulation; coverage

- **λ:** inside the simulated 95% range in 78 of 78 runs. Under I at 12k matched, ε = 1, λ was 11.3k–11.8k against a range of 9.9k–12.9k.
- **Wald coverage of θ_λ under I:**
  - encrypted runs: 344 of 354 coefficients (0.972);
  - 400 plaintext releases per cell on the same data: 0.941–0.968.

## Accuracy

**Synthetic, at 12k matched (regime A):**

| setting | ε | population prediction error / least squares |
|---|---|---|
| I, std | 1 | 1.27 (ridge on) |
| I, std | 4 | 1.003 |
| P, std | 1 | 1.13 |
| P, bn | 1 | 1.01 |

The encrypted runs match the simulation; see the Cells table in `key_numbers.md`.

**Large-n simulation:** with 100 cohorts per n, from 10^5 to 10^6 matched records, prediction comes within 0.5% of least squares at every ε ≥ 0.5. Coefficient RMSE relative to least squares falls to 1.45 (ε = 0.5) and 1.15 (ε = 1) at 10^6 under I.

**Regime B (n_R^max = 100k, setting I):**
- The ridge was off in all 12 encrypted runs.
- In simulation, the ridge is on in 17% of releases at 60k and ε = 0.5. n_off is 52.5k, and the switch-off lies within its 1.2× factor. The ridge is off in every release elsewhere.
- The bias-aware interval for β was finite in every run.

| matched | ε | median β half-width | β coverage (sim) | prediction error / LS | RMSE / LS |
|---|---|---|---|---|---|
| 60k | 0.5 | 0.055 | 0.969 | 1.012 | 9.1 |
| 60k | 1 | 0.029 | 0.941 | 1.003 | 4.7 |
| 95k | 0.5 | 0.034 | 0.954 | 1.005 | 4.7 |
| 95k | 1 | 0.018 | 0.949 | 1.001 | 2.6 |

**3droad (block normalization):**
- The ridge was off in every release.
- Test MSE relative to the arm's own non-private fit: 0.9995–1.0014 across all cells.
- The signal is weak: the non-private test MSE is 0.973 against 0.994 for predicting the mean, because block normalization reduces R's single feature to its sign.
- Wald coverage under I: 23 of 24 (encrypted), 0.958–0.968 (simulated).

## Files

- `stage{1..5}.json`: per-stage summaries, gates, cost (published and re-measured), statistics.
- `micro_large.json` and `micro_t{1,4}_remeasured.json`: re-measured inputs.
- `sim_large.json`: the large-n simulation.
- `key_numbers.md` and `summary.json`: the key tables.
