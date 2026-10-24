# Cost model (single thread unless stated; N = 2^14, d_R = 4, d_O = 6, |E| = 63, 2 releases)

Fitted: binning 22.9 us/row; R's BFV keygen 0.05 s. Measured 4-core speedups: aggregation 3.96x (parallel entries), PSI polynomials 4.04x, BFV powers 2.21x (OpenFHE OpenMP).

## Validation against the measured end-to-end runs (single thread)

| n_bar_O | aggregate meas/pred (s) | PSI eval meas/pred | sanitize meas/pred | O OPRF self meas/pred | upload meas/pred | comm meas/pred (MB) | wall (s) |
|---|---|---|---|---|---|---|---|
| 1024 | 14.7 / 16.3 | 1.2 / 0.9 | 0.9 / 1.0 | 0.1 / 0.1 | 0.18 / 0.16 | 48.4 / 48.4 | 22.4 |
| 4096 | 14.7 / 16.3 | 2.0 / 1.7 | 0.9 / 1.0 | 0.5 / 0.4 | 0.19 / 0.16 | 48.4 / 48.4 | 24.5 |
| 16384 | 14.6 / 17.0 | 3.1 / 3.0 | 0.9 / 1.0 | 1.7 / 1.5 | 0.37 / 0.32 | 53.1 / 53.1 | 27.5 |
| 32768 | 16.8 / 18.4 | 4.6 / 4.3 | 0.9 / 1.0 | 3.4 / 3.0 | 0.70 / 0.65 | 60.9 / 60.9 | 33.5 |
| 65536 | 19.8 / 21.3 | 6.3 / 6.4 | 0.9 / 1.0 | 6.3 / 6.0 | 1.30 / 1.29 | 72.9 / 72.9 | 42.5 |

## Extrapolation (not run)

| n_bar_O | n_R^max | m | D, alpha | O CPU 1 core (s) | O wall 4 cores (s) | O wall 16 cores (s) | R CPU (s) | comm (MB) | O memory (MB) | aggregate N=2^14 / 2^15 (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| 100,000 | 2,000 | 13 | 64, 1 | 52 | 17 | 9 | 3 | 88 | 273 | 25 / 48 |
| 100,000 | 10,000 | 13 | 64, 1 | 53 | 18 | 10 | 5 | 88 | 273 | 25 / 48 |
| 300,000 | 2,000 | 37 | 64, 2 | 99 | 30 | 13 | 6 | 161 | 488 | 42 / 68 |
| 300,000 | 10,000 | 37 | 64, 2 | 99 | 31 | 14 | 8 | 161 | 488 | 42 / 68 |
| 1,000,000 | 2,000 | 123 | 64, 5 | 260 | 76 | 28 | 17 | 422 | 1244 | 102 / 142 |
| 1,000,000 | 10,000 | 123 | 64, 5 | 261 | 77 | 29 | 18 | 422 | 1244 | 102 / 142 |

### Breakdown at n_bar_O = 1M, n_R^max = 2000 (single thread, seconds)

- O.psi.oprf_self: 91.1
- O.agg.encode: 75.9
- O.psi.binning: 22.9
- O.psi.polys: 20.9
- O.agg.evalsum: 15.0
- R.upload_encrypt: 14.9
- O.psi.eval: 11.7
- O.agg.mul_add: 10.6
- O.upload_convert: 5.0
- O.psi.sanitize: 4.9
- O.keygen: 0.9
- O.release: 0.7
- R.psi.bfv: 0.7
- O.agg.rescale_mask: 0.6
- R.keygen: 0.6
- R.psi.oprf: 0.4
- O.psi.oprf_requests: 0.1
- R.release: 0.1

Communication at 1M:

- R->O keygen (pk + rotation-key shares): 20.0 MB
- O->R keygen (pk share): 0.5 MB
- R->O OPRF: 0.1 MB
- O->R OPRF: 0.1 MB
- R->O BFV keys: 14.0 MB
- R->O PSI query: 12.3 MB
- O->R PSI reply: 5.0 MB
- R->O upload: 369.0 MB
- O->R releases: 1.0 MB

Memory at O at 1M:

- uploaded_cts: 738 MB
- rotation_keys: 39 MB
- psi_tables: 81 MB
- bfv_powers: 112 MB
- register: 275 MB
