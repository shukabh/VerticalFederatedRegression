# Benchmarks (4-core Xeon @ 2.8 GHz, 15 GB; N = 2^14 CKKS, N_B = 2^14 BFV; d_R = 4, d_O = 6, n_R^max = 2000)

## End to end, per phase (seconds; circuit A; DP noise on; single thread per party)

| n_bar_O | m | PSI D, alpha | O: OPRF self | O: PSI preproc | O: PSI eval | O: sanitize | R: upload enc | O: convert | O: aggregate | O: aggregate, 4 threads | O: 2 releases | R: 2 releases | R: solve | wall | wall (4 thr agg) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1024 | 1 | 11, 1 | 0.10 | 0.06 | 1.22 | 0.90 | 0.14 | 0.04 | 14.71 | 3.99 | 0.68 | 0.10 | 0.03 | 22.4 | 11.8 |
| 4096 | 1 | 16, 1 | 0.48 | 0.14 | 1.99 | 0.91 | 0.13 | 0.06 | 14.74 | 4.05 | 0.73 | 0.11 | 0.04 | 24.5 | 13.3 |
| 16384 | 2 | 26, 1 | 1.74 | 0.51 | 3.11 | 0.92 | 0.25 | 0.12 | 14.56 | 4.35 | 0.72 | 0.12 | 0.04 | 27.5 | 16.4 |
| 32768 | 4 | 36, 1 | 3.40 | 1.07 | 4.55 | 0.94 | 0.51 | 0.18 | 16.83 | 4.76 | 0.68 | 0.10 | 0.04 | 33.5 | 20.7 |
| 65536 | 8 | 51, 1 | 6.28 | 1.84 | 6.29 | 0.87 | 0.98 | 0.32 | 19.76 | 5.41 | 0.64 | 0.11 | 0.04 | 42.5 | 29.4 |

Keygen: R 1.9 s (rotation-key shares), O 2.0 s (shares + joint keys); setup/calibration 0.15 s.

## Messages (bytes, from the transcript)

| message (R view) | 1024 | 4096 | 16384 | 32768 | 65536 |
|---|---|---|---|---|---|
| recv:public_params | 0.002 MB | 0.002 MB | 0.002 MB | 0.002 MB | 0.002 MB |
| sent:keygen_R | 20.000 MB | 20.000 MB | 20.000 MB | 20.000 MB | 20.000 MB |
| recv:keygen_O | 0.500 MB | 0.500 MB | 0.500 MB | 0.500 MB | 0.500 MB |
| sent:oprf_request | 0.061 MB | 0.061 MB | 0.061 MB | 0.061 MB | 0.061 MB |
| recv:oprf_response | 0.061 MB | 0.061 MB | 0.061 MB | 0.061 MB | 0.061 MB |
| sent:psi_keys | 14.007 MB | 14.007 MB | 14.007 MB | 14.007 MB | 14.007 MB |
| sent:psi_query | 8.757 MB | 8.757 MB | 10.509 MB | 12.260 MB | 12.260 MB |
| recv:psi_response | 1.003 MB | 1.003 MB | 1.003 MB | 1.003 MB | 1.003 MB |
| sent:upload | 3.001 MB | 3.001 MB | 6.002 MB | 12.003 MB | 24.006 MB |
| sent:release_request | 0.000 MB | 0.000 MB | 0.000 MB | 0.000 MB | 0.000 MB |
| recv:release | 1.000 MB | 1.000 MB | 1.000 MB | 1.000 MB | 1.000 MB |
| sent:done | 0.000 MB | 0.000 MB | 0.000 MB | 0.000 MB | 0.000 MB |
| total | 48.4 MB | 48.4 MB | 53.1 MB | 60.9 MB | 72.9 MB |

## Microbenchmarks, 1 thread(s) (median, seconds)

| operation | time |
|---|---|
| ckks_pk_share | 0.01034 |
| ckks_rotkey_share | 0.5665 |
| ckks_make_rotkeys | 0.3385 |
| ckks_encode_dd | 0.0098 |
| ckks_encrypt_seeded_R | 0.02014 |
| ckks_convert_O | 0.006837 |
| ckks_mul_plain_L0 | 0.0009978 |
| ckks_add_L0 | 0.0003646 |
| ckks_rescale_det_L0 | 0.005639 |
| ckks_rescale_stoch_L0 | 0.00802 |
| ckks_rotate_L1 | 0.01532 |
| ckks_rotate_L0 | 0.02276 |
| ckks_evalsum_L1 | 0.2379 |
| ckks_evalsum_L0 | 0.2251 |
| ckks_mul_plain_L1 | 0.0006643 |
| ckks_rescale_det_L1 | 0.003833 |
| dg_sample_N | 0.3492 |
| dg_residues_N | 0.005079 |
| ckks_rerand_partial_dec | 0.008146 |
| ckks_combine_R | 0.0411 |
| ckks_decode_dd_R | 0.007316 |
| oprf_server_evaluate_per_item | 9.112e-05 |
| oprf_client_blind_per_item | 9.168e-05 |
| oprf_server_blind_eval_per_item | 7.45e-05 |
| oprf_client_finalize_per_item | 0.000113 |
| bfv_encrypt | 0.03762 |
| bfv_compute_powers_D64 | 7.397 |
| bfv_eval_partition_D64 | 0.8618 |
| bfv_flood_sample_N | 0.4119 |
| bfv_sanitize_excl_sampling | 0.07748 |
| bfv_decrypt | 0.04006 |
| psi_build_polys_1M_shape | 20.93 |

| size | bytes |
|---|---|
| size_ckks_ct_L0_bytes | 1,048,576 |
| size_seeded_upload_bytes | 524,320 |
| size_release_bytes | 524,288 |
| size_rotkey_share_bytes | 20,447,232 |
| size_bfv_pk_relin_bytes | 14,686,008 |
| size_bfv_query_ct_bytes | 1,836,485 |
| size_bfv_reply_ct_bytes | 525,755 |
| (sigma_c of the release, log2) | 65.72 |

## Microbenchmarks, 4 thread(s) (median, seconds)

| operation | time |
|---|---|
| ckks_pk_share | 0.01022 |
| ckks_rotkey_share | 0.5999 |
| ckks_make_rotkeys | 0.3016 |
| ckks_encode_dd | 0.008715 |
| ckks_encrypt_seeded_R | 0.01762 |
| ckks_convert_O | 0.006472 |
| ckks_mul_plain_L0 | 0.0008998 |
| ckks_add_L0 | 0.0004027 |
| ckks_rescale_det_L0 | 0.005281 |
| ckks_rescale_stoch_L0 | 0.007579 |
| ckks_rotate_L1 | 0.008451 |
| ckks_rotate_L0 | 0.01006 |
| ckks_evalsum_L1 | 0.1154 |
| ckks_evalsum_L0 | 0.1401 |
| ckks_mul_plain_L1 | 0.0006171 |
| ckks_rescale_det_L1 | 0.003776 |
| dg_sample_N | 0.3472 |
| dg_residues_N | 0.005018 |
| ckks_rerand_partial_dec | 0.008021 |
| ckks_combine_R | 0.04133 |
| ckks_decode_dd_R | 0.00746 |
| oprf_server_evaluate_per_item | 9.863e-05 |
| oprf_client_blind_per_item | 0.0001118 |
| oprf_server_blind_eval_per_item | 0.0001357 |
| oprf_client_finalize_per_item | 0.0001198 |
| bfv_encrypt | 0.02266 |
| bfv_compute_powers_D64 | 3.348 |
| bfv_eval_partition_D64 | 0.3872 |
| bfv_flood_sample_N | 0.3781 |
| bfv_sanitize_excl_sampling | 0.04678 |
| bfv_decrypt | 0.04189 |
| psi_build_polys_1M_shape | 5.174 |

## CKKS at N = 2^15 (1 thread)

| operation | time |
|---|---|
| ckks_pk_share | 0.02226 |
| ckks_rotkey_share | 1.238 |
| ckks_make_rotkeys | 1.254 |
| ckks_encode_dd | 0.02134 |
| ckks_encrypt_seeded_R | 0.04267 |
| ckks_convert_O | 0.0141 |
| ckks_mul_plain_L0 | 0.003646 |
| ckks_add_L0 | 0.00216 |
| ckks_rescale_det_L0 | 0.01209 |
| ckks_rescale_stoch_L0 | 0.01682 |
| ckks_rotate_L1 | 0.0343 |
| ckks_rotate_L0 | 0.04632 |
| ckks_evalsum_L1 | 0.5469 |
| ckks_evalsum_L0 | 0.6482 |
| ckks_mul_plain_L1 | 0.003275 |
| ckks_rescale_det_L1 | 0.008318 |
| dg_sample_N | 0.8794 |
| dg_log2_sigma_c | 67.04 |
| dg_residues_N | 0.02201 |
| ckks_rerand_partial_dec | 0.03146 |
| ckks_combine_R | 0.08887 |
| ckks_decode_dd_R | 0.01923 |

## Scale options: OpenFHE-only operation costs (1 thread)

| backend | Q limbs (bits) | P limbs | log QP | native scale | ct-pt mult | rescale | rotation | EvalSum (13 rot.) |
|---|---|---|---|---|---|---|---|---|
| 64-bit build FIXEDMANUAL | 4 (60,60,59,60) | 2 | 358 | 2^59.0 | 1.60 ms | 3.75 ms | 20.1 ms | 0.305 s |
| 64-bit build COMPOSITE | 8 (45,45,32,32,33,32,32,32) | 2 | 376 | 2^64.0 | 3.48 ms | 1.15 ms | 54.4 ms | 0.738 s |
| 64-bit build COMPOSITE | 12 (27,27,26,22,18,20,22,21,21,20,21,21) | 5 | 415 | 2^60.7 | 5.59 ms | 1.96 ms | 67.5 ms | 1.001 s |
| 128-bit build FIXEDMANUAL | 3 (90,64,65) | 1 | 337 | 2^64.0 | 3.82 ms | 19.59 ms | 48.7 ms | 0.705 s |
| 128-bit build FIXEDMANUAL | 4 (60,60,59,60) | 1 | 357 | 2^59.0 | 5.20 ms | 24.91 ms | 104.4 ms | 1.120 s |

| circuit | n_bar_O | eps | log2 Omega_f | log2 B_tot (stage 1, 2) | sigma_2 (slot) | q0 needed: spec / (Q) (bits) |
|---|---|---|---|---|---|---|
| A | 4,096 | 1.0 | 62.78 | 26.46, 26.62 | 525.0 | 81.1 / 74.1 |
| A | 4,096 | 4.0 | 64.72 | 26.53, 26.69 | 141.8 | 83.0 / 76.0 |
| A | 100,000 | 1.0 | 62.88 | 26.56, 26.72 | 525.0 | 85.8 / 78.7 |
| A | 100,000 | 4.0 | 65.16 | 26.97, 27.13 | 141.8 | 88.0 / 81.0 |
| A | 1,000,000 | 1.0 | 63.19 | 26.88, 27.04 | 525.0 | 89.4 / 82.4 |
| A | 1,000,000 | 4.0 | 70.28 | 32.09, 32.25 | 141.8 | 96.5 / 89.5 |
| A (Omega_U=2^68) | 4,096 | 1.0 | 62.75 | 26.43, 26.59 | 525.0 | 81.0 / 74.1 |
| A (Omega_U=2^68) | 4,096 | 4.0 | 64.62 | 26.43, 26.59 | 141.8 | 82.9 / 75.9 |
| A (Omega_U=2^68) | 100,000 | 1.0 | 62.75 | 26.43, 26.59 | 525.0 | 85.6 / 78.6 |
| A (Omega_U=2^68) | 100,000 | 4.0 | 64.62 | 26.43, 26.59 | 141.8 | 87.5 / 80.5 |
| A (Omega_U=2^68) | 1,000,000 | 1.0 | 62.75 | 26.43, 26.59 | 525.0 | 89.0 / 81.9 |
| A (Omega_U=2^68) | 1,000,000 | 4.0 | 64.62 | 26.43, 26.59 | 141.8 | 90.8 / 83.8 |
| C | 4,096 | 1.0 | 53.66 | 17.49, 17.5 | 525.0 | 71.9 / 65.0 |
| C | 4,096 | 4.0 | 55.59 | 17.55, 17.56 | 141.8 | 73.9 / 66.9 |
| C | 100,000 | 1.0 | 53.75 | 17.58, 17.6 | 525.0 | 76.6 / 69.6 |
| C | 100,000 | 4.0 | 56.06 | 17.96, 18.02 | 141.8 | 79.0 / 71.9 |
| C | 1,000,000 | 1.0 | 54.06 | 17.87, 17.92 | 525.0 | 80.3 / 73.3 |
| C | 1,000,000 | 4.0 | 61.16 | 22.97, 23.13 | 141.8 | 87.4 / 80.3 |
