# Diagnostics: where the VFL protocol loses to AdaSSP

10-fold CV at full n, R = 30 noise draws per fold, delta = 1e-06. Test MSE / non-private.

## 1. Ridge ablation

| dataset | ε | AdaSSP (published) | VFL, ℓ = 0 (protocol) | VFL, oracle ℓ | VFL, no ridge (median) |
|---|---:|---:|---:|---:|---:|
| housing | 1 | 1.852 | 2.031 | 2.026 | 27 |
| housing | 10 | 1.084 | 1.180 | 1.163 | 2.55 |
| wine | 1 | 1.686 | 1.858 | 1.835 | 9.85 |
| wine | 10 | 1.014 | 1.162 | 1.058 | 1.03 |
| elevators | 1 | 1.514 | 1.722 | 1.722 | 2.37 |
| elevators | 10 | 1.013 | 1.051 | 1.051 | 1.28 |
| bike | 1 | 1.031 | 1.057 | 1.057 | 1.33 |
| bike | 10 | 1.001 | 1.002 | 1.002 | 1.28 |
| pol | 1 | 1.037 | 1.065 | 1.065 | 1.19 |
| pol | 10 | 1.001 | 1.003 | 1.003 | 1.01 |

## 2. How well y fills its bound, and conditioning

| dataset | n | d | rms abs(y) | q99.5 abs(y) | max / q99.5 | λmin(XᵀX/n) | λmax/λmin |
|---|---:|---:|---:|---:|---:|---:|---:|
| elevators | 16,599 | 18 | 0.191 | 0.707 | 1.41 | 0 (rank-deficient) | ∞ |
| wine | 1,599 | 11 | 0.238 | 0.711 | 1.41 | 1.3e-02 | 2.2e+01 |
| bike | 17,379 | 17 | 0.328 | 1.000 | 1.00 | 0 (rank-deficient) | ∞ |
| housing | 506 | 13 | 0.335 | 1.000 | 1.00 | 6.9e-03 | 6.3e+01 |
| protein | 45,730 | 9 | 0.409 | 1.000 | 1.00 | 3.8e-04 | 1.6e+03 |
| pol | 15,000 | 26 | 0.587 | 1.000 | 1.00 | 1.1e-05 | 1.5e+04 |
