"""Exact discrete Gaussian sampling on Z (Canonne, Kamath and Steinke 2020, Algorithms 1-3).

Only integer arithmetic (arbitrary-precision Python ints) and uniform random bits from a CSPRNG are used;
sigma^2 must be rational (here: a positive integer or a Fraction).  The output is exactly N_Z(0, sigma^2).
The CSPRNG is ChaCha20 keyed from os.urandom (via the `cryptography` package), buffered.

Not constant-time: the paper's (CT) requirement (fixed iteration counts, fixed-width limbs, padded batch
duration) is an open obligation (O2) and is not implemented here; see DEVIATIONS.md.
"""
from __future__ import annotations

import os
from fractions import Fraction

import numpy as np
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms


class CSPRNG:
    def __init__(self, key: bytes | None = None, chunk: int = 1 << 20):
        self.key = key or os.urandom(32)
        self.nonce = os.urandom(16)
        self._enc = Cipher(algorithms.ChaCha20(self.key, self.nonce), mode=None).encryptor()
        self.chunk = chunk
        self.buf = b''
        self.pos = 0

    def bytes(self, n: int) -> bytes:
        if self.pos + n > len(self.buf):
            self.buf = self.buf[self.pos:] + self._enc.update(b'\0' * max(self.chunk, n))
            self.pos = 0
        out = self.buf[self.pos:self.pos + n]
        self.pos += n
        return out

    def randbits(self, k: int) -> int:
        if k <= 0:
            return 0
        nb = (k + 7) // 8
        x = int.from_bytes(self.bytes(nb), 'little')
        return x >> (8 * nb - k)

    def randbelow(self, n: int) -> int:
        """uniform in {0, ..., n-1} by rejection."""
        if n <= 0:
            raise ValueError
        k = n.bit_length()
        while True:
            x = self.randbits(k)
            if x < n:
                return x

    def uniform_mod_array(self, q: int, size: int) -> np.ndarray:
        """uniform residues in [0, q) for q < 2^63 (vectorised rejection)."""
        bits = q.bit_length()
        mask = (1 << bits) - 1
        out = np.empty(size, dtype=np.uint64)
        filled = 0
        while filled < size:
            need = (size - filled) * 2 + 16
            x = np.frombuffer(self.bytes(8 * need), dtype=np.uint64) & np.uint64(mask)
            x = x[x < np.uint64(q)]
            k = min(len(x), size - filled)
            out[filled:filled + k] = x[:k]
            filled += k
        return out


def bernoulli(rng: CSPRNG, num: int, den: int) -> int:
    """Bernoulli(num/den), 0 <= num <= den."""
    return 1 if rng.randbelow(den) < num else 0


def bernoulli_exp_le1(rng: CSPRNG, num: int, den: int) -> int:
    """Bernoulli(exp(-gamma)) for gamma = num/den in [0, 1] (CKS20 Algorithm 1)."""
    k = 1
    while True:
        if not bernoulli(rng, num, den * k):
            break
        k += 1
    return k & 1


def bernoulli_exp(rng: CSPRNG, num: int, den: int) -> int:
    """Bernoulli(exp(-gamma)) for rational gamma = num/den >= 0."""
    while num > den:
        if not bernoulli_exp_le1(rng, 1, 1):
            return 0
        num -= den
    return bernoulli_exp_le1(rng, num, den)


def discrete_laplace(rng: CSPRNG, t: int) -> int:
    """P[X = x] proportional to exp(-|x|/t) for an integer t >= 1 (CKS20 Algorithm 2 with s = 1)."""
    while True:
        u = rng.randbelow(t)
        if not bernoulli_exp(rng, u, t):
            continue
        v = 0
        while bernoulli_exp(rng, 1, 1):
            v += 1
        x = u + t * v
        b = rng.randbits(1)
        if b == 1 and x == 0:
            continue
        return -x if b else x


def isqrt_floor(n: int) -> int:
    import math
    return math.isqrt(n)


def discrete_gaussian(rng: CSPRNG, sigma2) -> int:
    """Exact N_Z(0, sigma2) for rational sigma2 > 0 (CKS20 Algorithm 3)."""
    s2 = Fraction(sigma2)
    if s2 <= 0:
        raise ValueError
    num, den = s2.numerator, s2.denominator
    t = isqrt_floor(num // den) + 1                     # floor(sigma) + 1
    while True:
        y = discrete_laplace(rng, t)
        # gamma = (|y| - sigma^2/t)^2 / (2 sigma^2) = (|y| t den - num)^2 / (2 num t^2 den)
        a = abs(y) * t * den - num
        g_num = a * a
        g_den = 2 * num * t * t * den
        if bernoulli_exp(rng, g_num, g_den):
            return y


def sample_vector(sigma2, n: int, rng: CSPRNG | None = None) -> list:
    rng = rng or CSPRNG()
    return [discrete_gaussian(rng, sigma2) for _ in range(n)]


def residues(values, moduli) -> np.ndarray:
    """(len(moduli), len(values)) uint64 array of values mod q (Python ints, any sign)."""
    out = np.empty((len(moduli), len(values)), dtype=np.uint64)
    for i, q in enumerate(moduli):
        out[i] = np.array([v % q for v in values], dtype=np.uint64)
    return out
