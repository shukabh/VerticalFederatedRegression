# Intersection-size sweep on real data

Command: `experiments/adassp/sweep_n.py`  
10-fold CV; nested random subsets of the training folds; R = 40 noise draws per fold and n; δ = 1e-06; ρ* = 2. Ratios are test MSE ÷ non-private test MSE at the same n.

## Matched records needed to come within 10% of non-private

VFL, ours (AdaSSP published in brackets). Log-linear interpolation of the mean curve.

| dataset | n available | ε = 0.5 | ε = 1 | ε = 2 | ε = 4 |
|---|---:|---:|---:|---:|---:|
| bike | 15,641 | > 16k (15k) | 11k (7.5k) | 5.9k (3.8k) | 3.1k (1.8k) |
| elevators | 14,939 | > 15k (> 15k) | > 15k (> 15k) | > 15k (> 15k) | > 15k (14k) |
| pol | 13,500 | > 14k (> 14k) | 10k (7.0k) | 5.3k (3.5k) | 2.8k (1.7k) |
| protein | 41,157 | 5.4k (3.7k) | 2.8k (1.7k) | 1.4k (784) | 619 (295) |

## Matched records needed to come within 25% of non-private

VFL, ours (AdaSSP published in brackets). Log-linear interpolation of the mean curve.

| dataset | n available | ε = 0.5 | ε = 1 | ε = 2 | ε = 4 |
|---|---:|---:|---:|---:|---:|
| bike | 15,641 | 10k (7.3k) | 5.5k (3.8k) | 2.8k (1.9k) | 1.5k (908) |
| elevators | 14,939 | > 15k (> 15k) | > 15k (> 15k) | > 15k (14k) | 12k (7.4k) |
| pol | 13,500 | 9.6k (7.0k) | 5.0k (3.5k) | 2.6k (1.7k) | 1.4k (801) |
| protein | 41,157 | 717 (555) | 301 (196) | 100 (100) | 100 (100) |

## Ratio at selected n (VFL, ours / AdaSSP published)

| dataset | n | ε = 0.5 | ε = 1 | ε = 2 | ε = 4 |
|---|---:|---:|---:|---:|---:|
| bike | 1,000 | 2.715 / 2.467 | 2.195 / 1.925 | 1.721 / 1.474 | 1.384 / 1.218 |
| bike | 10,000 | 1.257 / 1.168 | 1.113 / 1.065 | 1.043 / 1.021 | 1.015 / 1.007 |
| bike | 15,641 | 1.146 / 1.094 | 1.057 / 1.032 | 1.021 / 1.010 | 1.007 / 1.003 |
| elevators | 1,000 | 3.505 / 3.444 | 3.118 / 3.005 | 2.794 / 2.633 | 2.458 / 2.232 |
| elevators | 10,000 | 2.295 / 2.114 | 1.952 / 1.727 | 1.598 / 1.386 | 1.317 / 1.170 |
| elevators | 14,939 | 2.076 / 1.891 | 1.719 / 1.518 | 1.399 / 1.240 | 1.187 / 1.087 |
| pol | 1,000 | 2.099 / 1.986 | 1.857 / 1.702 | 1.587 / 1.412 | 1.334 / 1.194 |
| pol | 10,000 | 1.238 / 1.160 | 1.101 / 1.060 | 1.038 / 1.020 | 1.013 / 1.006 |
| pol | 13,500 | 1.163 / 1.107 | 1.065 / 1.038 | 1.023 / 1.012 | 1.008 / 1.004 |
| protein | 1,000 | 1.223 / 1.194 | 1.163 / 1.137 | 1.117 / 1.088 | 1.079 / 1.052 |
| protein | 10,000 | 1.068 / 1.050 | 1.042 / 1.028 | 1.023 / 1.015 | 1.013 / 1.007 |
| protein | 41,157 | 1.021 / 1.015 | 1.011 / 1.007 | 1.006 / 1.004 | 1.003 / 1.001 |
