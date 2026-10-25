"""Worst-case reply noise of the PSI evaluation in the textbook BFV model (crypto proofs, Part A, Proposition on B_psi):
per-level recursion E_{l+1} = F(E_l, E_l) for the windowed products, plaintext-coefficient evaluation, then the switch
q_B -> q'.  Parameters: implementation/vfr/psi.py (N_B = 2^14, t = 4293918721, d_cap = 64, 7 x 60-bit limbs, 2 limbs
after the switch); errors with sup-norm at most B_e = 27 (OpenFHE's table sampler at sigma = 3.19 saturates at 27).
E_rl (relinearization) and E_rns (an additive allowance for the RNS variant's approximate scaling) are model parameters."""
import math
from fractions import Fraction as Fr
N, t, D, B_e = 2 ** 14, 4293918721, 64, 27
logqB, logqp = 7 * 60, 2 * 60
q = 2 ** logqB
W = math.ceil(math.log2(D)) + 1
depth = math.ceil(math.log2(W))
kk = Fr(N + 3, 2)
def F(E, E2, E_rl, E_rns):
    return (Fr(N) * t * t / 4 + t * (Fr(N) * t / 4 + Fr(1, 2)) + Fr(N) * t / 2 * (E + E2) + Fr(t) * N / q * E * E2
            + Fr(N) * t * t * kk + Fr(t) * N * kk * (E + E2) + Fr(1 + N + N * N, 2) + E_rl + E_rns)
for E_rl_bits, E_rns_bits in ((26, 0), (90, 90)):
    E = Fr(B_e * (2 * N + 1))
    print(f'E_rl = 2^{E_rl_bits}, E_rns = 2^{E_rns_bits}:  E_0 = 2^{math.log2(E):.2f}', end='')
    for l in range(depth):
        E = F(E, E, Fr(2) ** E_rl_bits, Fr(2) ** E_rns_bits)
        print(f', E_{l+1} = 2^{math.log2(E):.2f}', end='')
    E_eval = D * (Fr(N) * t / 2 * E + t * (Fr(N) * t / 4 + Fr(1, 2))) + Fr(t) * (D + 1) / 2
    w_eval = math.sqrt(N) * float(E_eval) / 2.0 ** (logqB - logqp)
    Delta_half = q // t // 2
    print(f'\n   evaluation noise E_eval = 2^{math.log2(E_eval):.2f} (decryption needs < Delta/2 = 2^{math.log2(Delta_half):.1f});'
          f' after the switch ||w_eval||_2 <= sqrt(N) E_eval q\'/q_B = 2^{math.log2(w_eval):.2f}')
print(f'windows W = {W}, product depth = {depth}')
