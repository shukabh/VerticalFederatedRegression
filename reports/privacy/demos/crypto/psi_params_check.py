"""Sizing checks for the PSI / HE parameters (risks C5, C6, C7 in crypto_layer.md).

1. Plaintext modulus t chosen by calibrate_hyperparameters.psi_plaintext_modulus for a
   range of (n_R, n_O), and how that compares with realistic identifier spaces.
2. Simulated simple-hashing bin loads for O (n_hash=3, NB=16384 bins) -> the max load,
   which is what the {alpha, D} metadata sent to R reveals, and whether D=1 occurs.
3. HE-Standard (128-bit classical, ternary secret) max log2(QP) per ring dimension vs a
   rough modulus requirement for the configured BFV / CKKS depths.
Pure numpy; runs in ~1-2 s.  The user's calibrate script is imported read-only.
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "scenario_b"))
from calibrate_hyperparameters import psi_plaintext_modulus  # noqa: E402

NB, NH, DCAP, RING = 16384, 3, 64, 16384

print("1) PSI plaintext modulus t (delta_psi=1e-6, n_hash=3, ring=16384)")
print(f"   {'n_R':>9} {'n_O':>10} {'t bits':>6} {'E[false pos]':>13}")
for n_R, n_O in [(500, 5_000), (5_000, 50_000), (20_000, 1_000_000),
                 (100_000, 10_000_000), (1_000_000, 40_000_000)]:
    try:
        t = psi_plaintext_modulus(n_R, n_O, RING, NH, 1e-6)
        efp = n_R * n_O * NH / (RING * t)
        print(f"   {n_R:>9} {n_O:>10} {t.bit_length():>6} {efp:>13.2e}")
    except RuntimeError as e:
        print(f"   {n_R:>9} {n_O:>10}   --  {str(e)[:60]}")
print("   identifier spaces: 9-digit SIN ~ 2^29.9 (Luhn-valid ~ 2^26.6); generator IDs < 10^15 ~ 2^49.8")
print("   -> for moderate sizes t is ~2^29: IDs larger than t are reduced mod t (or hashed);")
print("      an unkeyed map of a ~2^27 identifier space is enumerable offline.\n")

print("2) O's simple-hashing max bin load (what alpha/D reveal); 20 trials each")
rng = np.random.default_rng(7)
for n_O in [40, 200, 5_000, 50_000, 1_000_000]:
    loads = []
    for _ in range(20):
        bins = rng.integers(0, NB, size=(n_O, NH)).ravel()   # item placed in all 3 bins
        loads.append(np.bincount(bins, minlength=NB).max())
    loads = np.array(loads)
    alpha = np.ceil(loads / DCAP).astype(int)
    D = np.ceil(loads / alpha).astype(int)
    print(f"   n_O={n_O:>8}: max load {loads.min():>3}-{loads.max():<3} -> alpha {alpha.min()}-{alpha.max()},"
          f" D {D.min()}-{D.max()};  P(D=1)={np.mean(D == 1):.2f}")
print("   (assumes D = ceil(load/alpha); psi_common.bin_and_interpolate was not provided)\n")

print("3) HE-Standard 128-bit classical max log2(QP), ternary secret")
std = {4096: 109, 8192: 218, 16384: 438, 32768: 881}
print("   " + "  ".join(f"N={k}: {v}" for k, v in std.items()))
for tb in (30, 45, 60):
    # heuristic BFV budget: ~ (log t + log N + 10) bits per ct x ct level, the same per
    # full-range plaintext multiply, plus ~60 bits base and ~60 bits of key-switch primes
    per = tb + math.log2(RING) + 10
    need = 60 + 3 * per + 2 * per + 60     # 3 ct x ct (D<=64 windows) + P_d and r1 pt-mults
    print(f"   BFV t~2^{tb}: rough log2(QP) need ~{need:.0f} bits vs 438 allowed at N=16384 "
          f"-> {'OK' if need <= 438 else 'EXCEEDS: N=32768 needed or security < 128'}")
ckks = 60 + 2 * 50 + 60
print(f"   CKKS depth 2, scale 2^50: log2(QP) ~ {ckks} (60 + 2x50 + 60 special) -> needs N>=16384"
      f" (218 max at N=8192); depth 3 (one-hot mask fix) ~{ckks + 50} -> still N=16384")
