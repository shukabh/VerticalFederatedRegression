"""2HashDH OPRF, RFC 9497 mode 0 (OPRF), suite ristretto255-SHA512, over libsodium (ctypes).

Group operations (hash-to-group map, scalar multiplication, scalar inversion and reduction) are libsodium's
vetted ristretto255 implementation; the protocol framing (expand_message_xmd, DSTs, Finalize) follows the
RFC and is checked against its test vectors (tests/test_oprf.py).

  client (R):  blind(x) -> (r, alpha = r * H1(x));  finalize(x, r, beta) = SHA512(len|x|len|r^-1 beta|"Finalize")
  server (O):  blind_evaluate(k, alpha) = k * alpha;  evaluate(k, x) = the same output computed directly
The 64-byte output is split by the protocol into an item part F_k (bytes 0..31) and a key part F'_k (32..63).

O's OPRF over its whole register and its answer to R's blinded queries are batched: evaluate_batch and
blind_evaluate_batch run the same libsodium calls (hash-to-group, scalar multiplication, point validation) in the
native layer (native/vfr_native.cpp), in parallel over the items with OpenMP, and return outputs bit-identical to
the item-by-item functions below (tests/test_oprf_parallel.py).  The key is the same for every item; each thread
only reads it.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import hashlib
import os

_lib_path = ctypes.util.find_library('sodium') or 'libsodium.so.23'
_sodium = ctypes.CDLL(_lib_path)
if _sodium.sodium_init() < 0:
    raise RuntimeError('libsodium initialisation failed')
for _fn in ('crypto_core_ristretto255_from_hash', 'crypto_scalarmult_ristretto255',
            'crypto_core_ristretto255_scalar_invert', 'crypto_core_ristretto255_scalar_reduce',
            'crypto_core_ristretto255_is_valid_point', 'crypto_core_ristretto255_scalar_random'):
    if not hasattr(_sodium, _fn):
        raise RuntimeError(f'libsodium lacks {_fn} (need >= 1.0.18)')

SUITE = b'ristretto255-SHA512'
MODE = 0
CONTEXT = b'OPRFV1-' + bytes([MODE]) + b'-' + SUITE
DST_H2G = b'HashToGroup-' + CONTEXT
DST_H2S = b'HashToScalar-' + CONTEXT
DST_DKP = b'DeriveKeyPair' + CONTEXT
L_ORDER = 2 ** 252 + 27742317777372353535851937790883648493


def libsodium_version() -> str:
    _sodium.sodium_version_string.restype = ctypes.c_char_p
    return _sodium.sodium_version_string().decode()


def i2osp(x: int, n: int) -> bytes:
    return x.to_bytes(n, 'big')


def expand_message_xmd(msg: bytes, dst: bytes, len_in_bytes: int) -> bytes:
    """RFC 9380 section 5.3.1 with SHA-512 (b_in_bytes = 64, s_in_bytes = 128)."""
    b_in, s_in = 64, 128
    ell = -(-len_in_bytes // b_in)
    if ell > 255 or len(dst) > 255:
        raise ValueError
    dst_prime = dst + i2osp(len(dst), 1)
    z_pad = b'\0' * s_in
    msg_prime = z_pad + msg + i2osp(len_in_bytes, 2) + i2osp(0, 1) + dst_prime
    b0 = hashlib.sha512(msg_prime).digest()
    b1 = hashlib.sha512(b0 + i2osp(1, 1) + dst_prime).digest()
    out = [b1]
    for i in range(2, ell + 1):
        prev = bytes(x ^ y for x, y in zip(b0, out[-1]))
        out.append(hashlib.sha512(prev + i2osp(i, 1) + dst_prime).digest())
    return b''.join(out)[:len_in_bytes]


def hash_to_group(x: bytes) -> bytes:
    u = expand_message_xmd(x, DST_H2G, 64)
    p = ctypes.create_string_buffer(32)
    _sodium.crypto_core_ristretto255_from_hash(p, u)
    return p.raw


def hash_to_scalar(x: bytes, dst: bytes = DST_H2S) -> bytes:
    u = expand_message_xmd(x, dst, 64)
    s = ctypes.create_string_buffer(32)
    _sodium.crypto_core_ristretto255_scalar_reduce(s, u)
    return s.raw


def scalarmult(k: bytes, P: bytes) -> bytes:
    Q = ctypes.create_string_buffer(32)
    if _sodium.crypto_scalarmult_ristretto255(Q, k, P) != 0:
        raise ValueError('identity element')
    return Q.raw


def scalar_inv(k: bytes) -> bytes:
    r = ctypes.create_string_buffer(32)
    if _sodium.crypto_core_ristretto255_scalar_invert(r, k) != 0:
        raise ValueError('zero scalar')
    return r.raw


def random_scalar() -> bytes:
    r = ctypes.create_string_buffer(32)
    _sodium.crypto_core_ristretto255_scalar_random(r)
    return r.raw


def is_valid_point(P: bytes) -> bool:
    return len(P) == 32 and _sodium.crypto_core_ristretto255_is_valid_point(P) == 1


def derive_key_pair(seed: bytes, info: bytes) -> bytes:
    derive_input = seed + i2osp(len(info), 2) + info
    for counter in range(256):
        sk = hash_to_scalar(derive_input + i2osp(counter, 1), DST_DKP)
        if int.from_bytes(sk, 'little') != 0:
            return sk
    raise ValueError('DeriveKeyPairError')


def keygen() -> bytes:
    """fresh OPRF key k (a uniformly random non-zero scalar), one per run."""
    return random_scalar()


def blind(x: bytes, r: bytes | None = None):
    r = r or random_scalar()
    return r, scalarmult(r, hash_to_group(x))


def blind_evaluate(k: bytes, alpha: bytes) -> bytes:
    if not is_valid_point(alpha):
        raise ValueError('invalid blinded element')
    return scalarmult(k, alpha)


def finalize(x: bytes, r: bytes, beta: bytes) -> bytes:
    if not is_valid_point(beta):
        raise ValueError('invalid evaluated element')
    n = scalarmult(scalar_inv(r), beta)
    return hashlib.sha512(i2osp(len(x), 2) + x + i2osp(len(n), 2) + n + b'Finalize').digest()


def evaluate(k: bytes, x: bytes) -> bytes:
    n = scalarmult(k, hash_to_group(x))
    return hashlib.sha512(i2osp(len(x), 2) + x + i2osp(len(n), 2) + n + b'Finalize').digest()


def _native():
    from . import vfr_native as vn
    return vn


def evaluate_batch(k: bytes, xs, threads: int = 0) -> list:
    """[evaluate(k, x) for x in xs], computed in parallel (threads <= 0: the OpenMP setting, vfr_native.set_threads)."""
    xs = list(xs)
    out = _native().oprf_evaluate_batch(k, xs, int(threads))
    return [out[64 * i:64 * i + 64] for i in range(len(xs))]


def blind_evaluate_batch(k: bytes, blinded: bytes, threads: int = 0) -> bytes:
    """b''.join(blind_evaluate(k, alpha_i)) over the 32-byte elements of `blinded`, computed in parallel; raises
    ValueError if any element is not a valid ristretto255 point."""
    return _native().oprf_blind_evaluate_batch(k, blinded, int(threads))
