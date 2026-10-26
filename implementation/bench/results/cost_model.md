# Cost model (single thread unless stated; N = 2^14, d_R = 4, d_O = 6, |E| = 63, 2 releases)

Inputs measured 2026-10-26 00:12 (1 thread) and 2026-10-26 00:12 (4 threads) on the idle machine. Fitted: binning 12.6 us/row; R's BFV keygen 0.12 s. Measured 4-core speedups: aggregation 2.83x in the run configuration (4 threads x 4 parallel entries; 3.51x with one thread per entry), PSI polynomials 3.99x, BFV powers 2.28x (OpenFHE OpenMP), O's OPRF 3.85x (batched, OpenMP). The PSI interpolation is evaluated at the actual partition fill.

## Validation against the measured end-to-end runs (single thread; item-by-item OPRF at the time)

| n_bar_O | aggregate meas/pred (s) | PSI eval meas/pred | sanitize meas/pred | O OPRF self meas/pred | upload meas/pred | comm meas/pred (MB) | wall (s) |
|---|---|---|---|---|---|---|---|
| 1024 | 14.7 / 8.7 | 1.2 / 0.6 | 0.9 / 0.6 | 0.1 / 0.1 | 0.18 / 0.10 | 48.4 / 48.4 | 22.4 |
| 4096 | 14.7 / 8.7 | 2.0 / 1.0 | 0.9 / 0.6 | 0.5 / 0.3 | 0.19 / 0.10 | 48.4 / 48.4 | 24.5 |
| 16384 | 14.6 / 9.1 | 3.1 / 1.8 | 0.9 / 0.6 | 1.7 / 1.3 | 0.37 / 0.20 | 53.1 / 53.1 | 27.5 |
| 32768 | 16.8 / 9.8 | 4.6 / 2.6 | 0.9 / 0.6 | 3.4 / 2.6 | 0.70 / 0.40 | 60.9 / 60.9 | 33.5 |
| 65536 | 19.8 / 11.3 | 6.3 / 3.9 | 0.9 / 0.6 | 6.3 / 5.1 | 1.30 / 0.81 | 72.9 / 72.9 | 42.5 |

## Extrapolation

O wall 4 cores: predict_run (4 OpenMP threads per party, 4 parallel entries, batched parallel OPRF, release padding with margin 2). The large runs of reports/large/ are compared with the model by experiments/eng/model_check.py (reports/eng/model_check.json).

| n_bar_O | n_R^max | m | D, alpha | O CPU 1 core (s) | O wall 4 cores (s) | R CPU (s) | comm (MB) | O memory (MB) | aggregate N=2^14 / 2^15 (s) |
|---|---|---|---|---|---|---|---|---|---|
| 100,000 | 2,000 | 13 | 64, 1 | 35 | 16 | 2 | 101 | 289 | 13 / 48 |
| 100,000 | 20,000 | 13 | 64, 1 | 47 | 22 | 5 | 129 | 289 | 13 / 48 |
| 300,000 | 2,000 | 37 | 64, 2 | 65 | 29 | 4 | 175 | 520 | 22 / 68 |
| 300,000 | 20,000 | 37 | 64, 2 | 79 | 37 | 7 | 205 | 520 | 22 / 68 |
| 1,000,000 | 2,000 | 123 | 64, 5 | 169 | 72 | 10 | 439 | 1325 | 53 / 142 |
| 1,000,000 | 20,000 | 123 | 64, 5 | 189 | 85 | 14 | 475 | 1325 | 53 / 142 |

### Breakdown at n_bar_O = 1M, n_R^max = 2000 (single thread, seconds)

- O.psi.oprf_self: 67.1
- O.agg.encode: 37.9
- O.psi.eval: 13.4
- O.psi.binning: 12.6
- O.psi.polys: 11.7
- R.upload_encrypt: 8.8
- O.agg.evalsum: 8.0
- O.agg.mul_add: 7.0
- O.psi.sanitize: 5.8
- O.upload_convert: 3.6
- R.psi.bfv: 0.9
- O.keygen: 0.6
- O.release: 0.5
- O.agg.rescale_mask: 0.4
- R.keygen: 0.4
- R.psi.oprf: 0.3
- O.psi.oprf_requests: 0.1
- R.release: 0.0

### O wall time per phase at n_bar_O = 1M, n_R^max = 20,000 (4 cores, seconds)

- O.aggregate: 18.8
- O.oprf_self: 17.4
- O.psi_preprocess: 17.3
- O.psi_eval: 13.7
- O.psi_sanitize: 11.6
- O.upload_convert: 3.9
- O.release: 1.5
- O.keygen: 0.5
- O.oprf_requests: 0.3

Communication at 1M:

- R->O keygen (pk + rotation-key shares): 20.0 MB
- O->R keygen (pk share): 0.5 MB
- R->O OPRF: 0.1 MB
- O->R OPRF: 0.1 MB
- R->O BFV keys: 14.0 MB
- R->O PSI query: 24.5 MB
- O->R PSI reply: 10.0 MB
- R->O upload: 369.0 MB
- O->R releases: 1.0 MB

Memory at O at 1M:

- uploaded_cts: 738 MB
- rotation_keys: 39 MB
- psi_tables: 161 MB
- bfv_powers: 112 MB
- register: 275 MB
