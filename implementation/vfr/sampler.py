"""Exact discrete Gaussian sampling on Z (Canonne, Kamath and Steinke 2020, Algorithms 1-3).

Only integer arithmetic (arbitrary-precision Python ints) and uniform random bits from a CSPRNG are used;
sigma^2 must be rational (here: a positive integer or a Fraction).  The output is exactly N_Z(0, sigma^2).
The CSPRNG is ChaCha20 keyed from os.urandom (via the `cryptography` package), buffered.

The functions `discrete_gaussian` and `sample_vector` are the plain CKS20 sampler: exact, but not constant time
(its loops run a value-dependent number of times).  The protocol uses the budgeted sampler of the second half of
this module (`budget`, `sample_residues`; DEVIATIONS A24): an exact rejection sampler run for a FIXED number T of
candidate trials, every trial doing the same fixed-width array work, so that its operation count does not depend
on the values it outputs; the only value-dependent event is the budget failure, whose probability is bounded by
a public xi_T (default 2^-80) that the ledger charges.
"""
from __future__ import annotations

import math
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


# =============================================================================================
# Budgeted exact sampler: a fixed work budget of T candidate trials (DEVIATIONS A24; proofs, Part A,
# Section "Release timing")
# =============================================================================================
#
# Proposal (one trial, for an integer variance S = sigma^2 and t = 2^b ~ sigma): a sign bit s, u uniform on
# [0, t), and v geometric with P[v = k] = (1 - e^-1) e^-k; m = u + t v, and y = m if s = 0, y = -(m + 1) if s = 1
# (a bijection onto Z, so q(y) = (1 - e^-1) e^-v(y) / (2t) with v(y) = floor(m / t)).  Acceptance: Bernoulli
# (exp(-g)) with g = y^2 / (2S) - v + c and c = C / 2^r >= S / (2 t^2), so that g >= (t v / sigma - sigma / t)^2 / 2
# >= 0.  q(y) exp(-g(y)) is proportional to exp(-y^2 / (2S)), so an accepted candidate is exactly N_Z(0, S).
#
# Every coin is a uniform U in [0, 1) of which the first r = 128 bits R are drawn; a comparison U < p is decided
# exactly by R against floor(2^r p) unless R equals it (a tie, probability 2^-r).
#   v:      v = #{1 <= j <= J_v : U < e^-j} (inversion); undecided if v = J_v (probability e^-J_v) or on a tie.
#   accept: g = n + f with n = floor(g), f in [0, 1); Bernoulli(e^-n) by one coin against floor(2^r e^-n) (a table),
#           and Bernoulli(e^-f) by CKS20 Algorithm 1 with exactly k_max coins A_k = [U_k < f / k] (thresholds
#           floor(floor(2^r f) / k)); K = first k with A_k = 0, accept iff K is odd; undecided if all k_max coins are
#           1 (probability f^k_max / k_max! <= 1 / k_max!) or on a tie.
# floor(2^r g) is computed exactly in 32-bit limbs (numpy uint64 arrays), as is y mod q for the residues.
#
# Fixed work: all T trials are computed with the same sequence of array operations on arrays of fixed shape, the
# surplus is discarded and the first n accepted values are returned.  The number of accepted trials A is
# Binomial(T, p) and independent of the values returned (rejection sampling).  The call FAILS (raises
# SamplerBudgetExhausted) iff some trial is undecided or A < n.  Coupled with the unbudgeted exact sampler on the
# same coins (with the remaining coins and trials drawn afresh), the two return the same values unless the call
# fails, and P[fail] <= xi_bound <= xi, from `budget` (exact binomial Chernoff bound for A < n plus the union bound
# over the trials).

RBITS = 128                      # bits of every uniform coin (four 32-bit limbs)
XI_DEFAULT = 2.0 ** -80          # failure probability per call (one vector of n values) that `budget` meets
_MASK = np.uint64(0xFFFFFFFF)
_S32 = np.uint64(32)


class SamplerBudgetExhausted(RuntimeError):
    """The budgeted sampler's fixed work budget did not suffice (probability at most xi per call)."""


def _exp_table(r: int = RBITS, P: int = 768):
    """[floor(2^r e^-j) for j = 0, 1, ..., J] with J the first index at which it is 0, each value certified exact:
    e^-1 lies between consecutive partial sums of its alternating series, the powers are bounded with directed
    rounding at P bits, and both bounds must give the same floor (otherwise an error is raised)."""
    s = Fraction(0)
    term = Fraction(1)
    sums = []
    for k in range(0, 160):
        s += term if k % 2 == 0 else -term
        term /= (k + 1)
        sums.append(s)
    lo_f, hi_f = sums[157], sums[158]                 # odd k: below e^-1; even k: above
    assert lo_f < hi_f
    Lo = (lo_f.numerator << P) // lo_f.denominator                # floor
    Hi = -((-(hi_f.numerator << P)) // hi_f.denominator)          # ceil
    tab = [1 << r]
    lo_j, hi_j = 1 << P, 1 << P
    while True:
        lo_j = (lo_j * Lo) >> P
        hi_j = -((-(hi_j * Hi)) >> P)
        a, b = lo_j >> (P - r), hi_j >> (P - r)
        if a != b:
            raise ArithmeticError('exp table not certified; raise P')
        tab.append(a)
        if a == 0:
            return tab


EXP_TABLE = _exp_table()          # EXP_TABLE[j] = floor(2^128 e^-j); EXP_TABLE[-1] = 0
J_ZERO = len(EXP_TABLE) - 1       # floor(2^128 e^-j) = 0 for every j >= J_ZERO (= 89)


def _limbs(x: int, L: int):
    if x < 0 or x >> (32 * L):
        raise OverflowError
    return [(x >> (32 * i)) & 0xFFFFFFFF for i in range(L)]


def _table_limbs(vals, L: int = 4) -> np.ndarray:
    """(L, len(vals)) uint64 array of 32-bit limbs; values >= 2^(32L) are stored as 0 (callers special-case them)."""
    out = np.zeros((L, len(vals)), dtype=np.uint64)
    for j, x in enumerate(vals):
        if x < (1 << (32 * L)):
            out[:, j] = _limbs(x, L)
    return out


_EXP_LIMBS = _table_limbs(EXP_TABLE)


def _mul(A: np.ndarray, B) -> np.ndarray:
    """Product of limb arrays (schoolbook): A (La, n), B (Lb, n) or a list of Lb constant limbs.  Every step fits in
    64 bits: (2^32 - 1) + (2^32 - 1)^2 + (2^32 - 1) = 2^64 - 1."""
    La, n = A.shape
    Lb = len(B)
    out = np.zeros((La + Lb, n), dtype=np.uint64)
    for i in range(La):
        carry = np.zeros(n, dtype=np.uint64)
        ai = A[i]
        for j in range(Lb):
            t = out[i + j] + ai * np.uint64(B[j]) + carry if not isinstance(B, np.ndarray) else out[i + j] + ai * B[j] + carry
            out[i + j] = t & _MASK
            carry = t >> _S32
        out[i + Lb] = carry
    return out


def _shr(A: np.ndarray, s: int, L: int) -> np.ndarray:
    """floor(A / 2^s), first L limbs."""
    q, sh = divmod(s, 32)
    La, n = A.shape
    z = np.zeros(n, dtype=np.uint64)
    get = lambda i: A[i] if 0 <= i < La else z
    out = np.empty((L, n), dtype=np.uint64)
    for i in range(L):
        if sh == 0:
            out[i] = get(i + q)
        else:
            out[i] = (get(i + q) >> np.uint64(sh)) | ((get(i + q + 1) << np.uint64(32 - sh)) & _MASK)
    return out


def _shl(A: np.ndarray, s: int, L: int) -> np.ndarray:
    """A * 2^s in L limbs (the caller sizes L so that nothing is lost)."""
    q, sh = divmod(s, 32)
    La, n = A.shape
    z = np.zeros(n, dtype=np.uint64)
    get = lambda i: A[i] if 0 <= i < La else z
    out = np.empty((L, n), dtype=np.uint64)
    for i in range(L):
        if sh == 0:
            out[i] = get(i - q)
        else:
            out[i] = ((get(i - q) << np.uint64(sh)) & _MASK) | (get(i - q - 1) >> np.uint64(32 - sh))
    return out


def _pad(A: np.ndarray, L: int) -> np.ndarray:
    if A.shape[0] >= L:
        return A[:L]
    return np.vstack([A, np.zeros((L - A.shape[0], A.shape[1]), dtype=np.uint64)])


def _sub(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """A - B for A >= B (same number of limbs)."""
    out = np.empty_like(A)
    borrow = np.zeros(A.shape[1], dtype=np.uint64)
    base = np.uint64(1 << 32)
    for i in range(A.shape[0]):
        d = A[i] + base - B[i] - borrow
        out[i] = d & _MASK
        borrow = np.uint64(1) - (d >> _S32)
    return out


def _add_small(A: np.ndarray, c: np.ndarray, at: int = 0) -> np.ndarray:
    """A + c 2^(32 at) for a vector c < 2^32 (carries propagated through every limb)."""
    out = A.copy()
    carry = c.astype(np.uint64)
    for i in range(at, A.shape[0]):
        t = out[i] + carry
        out[i] = t & _MASK
        carry = t >> _S32
    return out


def _add_const(A: np.ndarray, x: int) -> np.ndarray:
    cl = _limbs(x, A.shape[0])
    out = np.empty_like(A)
    carry = np.zeros(A.shape[1], dtype=np.uint64)
    for i in range(A.shape[0]):
        t = A[i] + np.uint64(cl[i]) + carry
        out[i] = t & _MASK
        carry = t >> _S32
    return out


def _sub_small(A: np.ndarray, c: np.ndarray, at: int) -> np.ndarray:
    """A - c 2^(32 at) for a vector c < 2^32 (the caller guarantees a non-negative result)."""
    out = A.copy()
    borrow = c.astype(np.uint64)
    base = np.uint64(1 << 32)
    for i in range(at, A.shape[0]):
        d = out[i] + base - borrow
        out[i] = d & _MASK
        borrow = np.uint64(1) - (d >> _S32)
    return out


def _cmp(A: np.ndarray, B) -> tuple:
    """(A < B, A == B) for limb arrays of the same length (B may be a list of constant limbs)."""
    L = A.shape[0]
    lt = A[L - 1] < B[L - 1]
    eq = A[L - 1] == B[L - 1]
    for i in range(L - 2, -1, -1):
        lt = lt | (eq & (A[i] < B[i]))
        eq = eq & (A[i] == B[i])
    return lt, eq


_M16 = np.uint64(0xFFFF)
_S16 = np.uint64(16)
_DIV_S = 28                      # shift of the reciprocal: exact for numerators < 2^22 and divisors k < 64


def _div_small(A: np.ndarray, k: int) -> np.ndarray:
    """floor(A / k) for 1 <= k < 64, by long division with 16-bit digits (every partial value < k 2^16 < 2^22),
    each digit step by a multiplication with M = ceil(2^28 / k) and a shift instead of a hardware division (whose
    latency can depend on its operands).  Exact: M k - 2^28 < k <= 2^(28 - 22), so floor(x M / 2^28) = floor(x / k)
    for every x < 2^22 (Granlund and Montgomery 1994, Theorem 4.2); x M < 2^50."""
    if not 1 <= k < 64:
        raise ValueError('divisor out of range')
    M = np.uint64(-(-(1 << _DIV_S) // k))
    kk = np.uint64(k)
    sh = np.uint64(_DIV_S)
    out = np.empty_like(A)
    rem = np.zeros(A.shape[1], dtype=np.uint64)
    for i in range(A.shape[0] - 1, -1, -1):
        hi_d, lo_d = A[i] >> _S16, A[i] & _M16
        cur = (rem << _S16) | hi_d
        qh = (cur * M) >> sh
        rem = cur - qh * kk
        cur = (rem << _S16) | lo_d
        ql = (cur * M) >> sh
        rem = cur - ql * kk
        out[i] = (qh << _S16) | ql
    return out


def _as_int_variance(sigma2) -> int:
    fr = Fraction(sigma2)
    if fr.denominator != 1 or fr <= 0:
        raise ValueError('the budgeted sampler needs a positive integer variance sigma^2')
    return int(fr)


def _log_binom_lower_tail(T: int, k: int, p: float) -> float:
    """log of the Chernoff bound exp(-T KL(k/T || p)) on P[Binomial(T, p) <= k], for k < T p (0 otherwise)."""
    a = k / T
    if a >= p:
        return 0.0
    kl = (a * math.log(a / p) if a > 0 else 0.0) + (1 - a) * math.log((1 - a) / (1 - p))
    return -T * kl


_BUDGET_CACHE: dict = {}


def budget(sigma2, n: int, xi: float = XI_DEFAULT) -> dict:
    """The public work budget of one call that returns n values of N_Z(0, sigma2) (sigma2 a positive integer):
    t = 2^b, C (c = C / 2^128), the lower bound p_lo on the acceptance probability, the number of trials T, J_v,
    k_max, the limb sizes, and xi_bound >= P[the call fails], with xi_bound <= xi.  A function of (sigma2, n, xi).

    p_lo = (1 - e^-1) sqrt(2 pi S) e^-c / (2t) <= P[accept] since sum_y exp(-y^2/(2S)) >= sqrt(2 pi S) (Poisson
    summation).  Allocation of xi: P[Binomial(T, p_lo) < n] <= xi / 2 (Chernoff-KL bound), T e^-J_v <= xi / 8,
    T / k_max! <= xi / 8, and the ties T (J_v + 1 + k_max) 2^-128 <= xi / 8."""
    S = _as_int_variance(sigma2)
    n = int(n)
    key = (S, n, float(xi))
    if key in _BUDGET_CACHE:
        return dict(_BUDGET_CACHE[key])
    if n < 1 or not (0 < xi < 1):
        raise ValueError('n >= 1 and 0 < xi < 1 are needed')
    r = RBITS
    best = None
    b0 = max(0, (S.bit_length() - 1) // 2)
    for b in sorted({max(0, b0 - 1), b0, b0 + 1, b0 + 2}):
        C = -(-(S << r) // (1 << (2 * b + 1)))          # ceil(2^r S / (2 t^2)): c = C / 2^r >= S / (2 t^2)
        logp = (math.log1p(-math.exp(-1.0)) + 0.5 * math.log(2 * math.pi) + 0.5 * math.log(S)
                - C / 2.0 ** r - (b + 1) * math.log(2.0))
        if best is None or logp > best[2]:
            best = (b, C, logp)
    b, C, logp = best
    p_lo = math.exp(logp) * (1 - 1e-9)                   # covers the floating-point error many times over
    if not 0 < p_lo < 1:
        raise ValueError('acceptance bound out of range')
    target = math.log(xi / 2)
    lo = hi = max(n, math.ceil(n / p_lo))
    while _log_binom_lower_tail(hi, n - 1, p_lo) > target:
        hi *= 2
    while lo < hi:                                        # smallest T with the bound <= xi / 2
        mid = (lo + hi) // 2
        if _log_binom_lower_tail(mid, n - 1, p_lo) <= target:
            hi = mid
        else:
            lo = mid + 1
    T = lo
    J_v = max(1, math.ceil(math.log(8 * T / xi)))
    if J_v >= J_ZERO:
        raise ValueError('xi too small for the 128-bit coins')
    k_max = 1
    while T / math.factorial(k_max) > xi / 8:
        k_max += 1
    if k_max >= 64:
        raise ValueError('xi too small for the reciprocal division (k_max < 64)')
    p_tie = T * (J_v + 1 + k_max) * 2.0 ** -r
    if p_tie > xi / 8:
        raise ValueError('xi too small for the 128-bit coins')
    p_bin = math.exp(_log_binom_lower_tail(T, n - 1, p_lo))
    xi_bound = p_bin + T * math.exp(-J_v) + T / math.factorial(k_max) + p_tie
    assert xi_bound <= xi
    bits_y = b + (J_v + 1).bit_length() + 1             # |y| <= t (J_v + 1) < 2^bits_y
    Ly = -(-bits_y // 32)
    P_rec = (r - 1) + 2 * bits_y
    out = dict(sigma2=S, n=n, xi=float(xi), xi_bound=xi_bound, b=b, C=C, p_lo=p_lo, T=T, J_v=J_v, k_max=k_max,
               bits_y=bits_y, Ly=Ly, P_rec=P_rec, log2_xi_bound=math.log2(xi_bound),
               coin_bytes=4 * T * (1 + Ly + 8 + 4 * k_max), trials_per_value=T / n)
    _BUDGET_CACHE[key] = out
    return dict(out)


def coin_limbs(bud: dict) -> int:
    """Number of 32-bit coin limbs per trial: sign, u, the coin of v, the coin of Bernoulli(e^-n), k_max coins."""
    return 1 + bud['Ly'] + 4 + 4 + 4 * bud['k_max']


def _trials(bud: dict, rng: CSPRNG | None = None, raw: np.ndarray | None = None):
    """Run the T trials of one call.  Returns (|y| limbs (Ly, T), sign (T,), accepted (T,), undecided (T,)).
    Every operation is on arrays of a fixed shape; nothing branches on a value.  `raw` (coin_limbs x T uint64
    array of 32-bit values) replaces the coins drawn from rng (tests)."""
    S, T, b, C = bud['sigma2'], bud['T'], bud['b'], bud['C']
    J_v, km, Ly, r = bud['J_v'], bud['k_max'], bud['Ly'], RBITS
    nl = coin_limbs(bud)
    if raw is None:
        raw = np.frombuffer(rng.bytes(4 * nl * T), dtype='<u4').reshape(nl, T).astype(np.uint64)
    T = raw.shape[1]
    sgn = raw[0] & np.uint64(1)
    Y = raw[1:1 + Ly].copy()
    for i in range(Ly):                                  # u: b uniform bits
        nb = min(32, max(0, b - 32 * i))
        Y[i] &= np.uint64((1 << nb) - 1)
    RV, RA = raw[1 + Ly:5 + Ly], raw[5 + Ly:9 + Ly]
    RK = raw[9 + Ly:].reshape(km, 4, T)
    # v by inversion: v = #{j <= J_v : U < e^-j}
    v = np.zeros(T, dtype=np.uint64)
    bad = np.zeros(T, dtype=bool)
    for j in range(1, J_v + 1):
        lt, eq = _cmp(RV, _EXP_LIMBS[:, j])
        v += lt
        bad |= eq
    bad |= v >= np.uint64(J_v)
    # |y| = (u + t v) + s, with t = 2^b and u < t
    li, sh = divmod(b, 32)
    vv = v << np.uint64(sh)
    Y[li] |= vv & _MASK
    if li + 1 < Ly:
        Y[li + 1] |= vv >> _S32
    Y = _add_small(Y, sgn)
    # G = floor(2^r g) = floor(2^(r-1) y^2 / S) - v 2^r + C, exactly
    Y2 = _mul(Y, Y)
    LS = -(-S.bit_length() // 32)
    rinv = (1 << bud['P_rec']) // S
    LQ = -(-(bud['P_rec'] - (S.bit_length() - 1) + 1) // 32)
    qh = _shr(_mul(Y2, _limbs(rinv, -(-rinv.bit_length() // 32))), 2 * bud['bits_y'], LQ)   # Q - 1 or Q
    LX = max(2 * Ly + 5, LQ + LS)
    X = _shl(Y2, r - 1, LX)
    diff = _sub(X, _pad(_mul(qh, _limbs(S, LS)), LX))
    lt, _ = _cmp(diff, _limbs(S, LX))
    LG = max(LQ, -(-C.bit_length() // 32), 5) + 1
    G = _add_const(_add_small(_pad(qh, LG), (~lt).astype(np.uint64)), C)
    G = _sub_small(G, v, r // 32)
    F = G[:4]
    over = np.any(G[5:] != 0, axis=0)
    nc = np.where(over | (G[4] >= np.uint64(J_ZERO)), np.uint64(J_ZERO), G[4]).astype(np.intp)
    # Bernoulli(e^-n): one coin against floor(2^r e^-n) (n = 0 always accepts)
    lt, eq = _cmp(RA, _EXP_LIMBS[:, nc])
    n0 = nc == 0
    acc = n0 | lt
    bad |= (~n0) & eq
    # Bernoulli(e^-f), CKS20 Algorithm 1 with exactly k_max coins
    alive = np.ones(T, dtype=bool)
    lead = np.zeros(T, dtype=np.int64)
    for k in range(1, km + 1):
        lt, eq = _cmp(RK[k - 1], _div_small(F, k))
        alive &= lt
        lead += alive
        bad |= eq
    bad |= lead == km
    acc &= (lead & 1) == 0
    return Y, sgn, acc, bad


def _mod_q(Y: np.ndarray, q: int) -> np.ndarray:
    """Y mod q for a limb array Y (Ly, n) and q < 2^63, by a reciprocal with one correction (all fixed width)."""
    Ly = Y.shape[0]
    P = 32 * Ly
    rq = (1 << P) // q
    LQ = Ly + 1
    qh = _shr(_mul(Y, _limbs(rq, -(-rq.bit_length() // 32))), P, LQ)   # floor(Y / q) - 1 or floor(Y / q)
    rem = _sub(_pad(Y, LQ + 2), _pad(_mul(qh, _limbs(q, 2)), LQ + 2))
    r64 = rem[0] | (rem[1] << _S32)                      # < 2q < 2^64
    qq = np.uint64(q)
    return r64 - qq * (r64 >= qq).astype(np.uint64)


def run_budgeted(sigma2, n: int, rng: CSPRNG | None = None, xi: float = XI_DEFAULT) -> dict:
    """One call with its diagnostics: the selected |y| limbs and signs, the number of accepted trials, the failure
    flag.  Does not raise; `sample_residues` and `sample_budgeted` do."""
    rng = rng or CSPRNG()
    bud = budget(sigma2, n, xi)
    Y, sgn, acc, bad = _trials(bud, rng)
    A = int(np.count_nonzero(acc))
    failed = bool(bad.any()) or A < bud['n']
    idx = np.flatnonzero(acc)[:bud['n']]
    return dict(budget=bud, Y=Y[:, idx], sign=sgn[idx], accepted=A, failed=failed, n_undecided=int(bad.sum()))


def sample_residues(sigma2, n: int, moduli, rng: CSPRNG | None = None, xi: float = XI_DEFAULT) -> np.ndarray:
    """n exact samples of N_Z(0, sigma2) from the budgeted sampler, returned only as residues: a
    (len(moduli), n) uint64 array (moduli < 2^63).  Raises SamplerBudgetExhausted on failure (probability at most
    budget(sigma2, n, xi)['xi_bound'] <= xi)."""
    out = run_budgeted(sigma2, n, rng, xi)
    if out['failed']:
        raise SamplerBudgetExhausted(f"budgeted sampler failed (T = {out['budget']['T']}, accepted {out['accepted']})")
    Y, neg = out['Y'], out['sign'] == 1
    res = np.empty((len(moduli), out['budget']['n']), dtype=np.uint64)
    for i, q in enumerate(moduli):
        q = int(q)
        if not 1 < q < (1 << 63):
            raise ValueError('modulus out of range')
        rm = _mod_q(Y, q)
        res[i] = np.where(neg & (rm != 0), np.uint64(q) - rm, rm)
    return res


def limbs_to_ints(Y: np.ndarray, sign: np.ndarray) -> list:
    """Signed Python integers from |y| limbs and signs (diagnostics and tests only)."""
    vals = []
    for j in range(Y.shape[1]):
        a = 0
        for i in range(Y.shape[0] - 1, -1, -1):
            a = (a << 32) | int(Y[i, j])
        vals.append(-a if sign[j] else a)
    return vals


def sample_budgeted(sigma2, n: int, rng: CSPRNG | None = None, xi: float = XI_DEFAULT) -> list:
    """The same samples as Python integers (tests and diagnostics; the protocol uses sample_residues)."""
    out = run_budgeted(sigma2, n, rng, xi)
    if out['failed']:
        raise SamplerBudgetExhausted('budgeted sampler failed')
    return limbs_to_ints(out['Y'], out['sign'])
