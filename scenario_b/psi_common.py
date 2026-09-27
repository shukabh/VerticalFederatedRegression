"""
psi_common.py -- Phase-0 (labeled PSI) helpers shared by party_r.py and party_o.py.

##############################################################################
#  STAND-IN written from call sites in party_r.py/party_o.py; the user's      #
#  original module was not available. Replace with the original.             #
##############################################################################

What the call sites require (and what this file implements)
-----------------------------------------------------------
  send_msg(sock, bytes) / recv_msg(sock)
      8-byte big-endian length prefix + payload.
  batch_ids(ids, cap) -> list of chunks (each <= cap ids)
  build_cuckoo_table(chunk, rng, NB, NH) -> {slot: id}, or None on failure
      R's side of Chen-Laine-Rindal (CLR) labeled PSI: every R identifier is
      placed in exactly ONE of its NH candidate bins h_0(x) .. h_{NH-1}(x).
  DUMMY_Y
      field value R encrypts in empty cuckoo slots (0; id_to_field never
      returns 0, and party_r.py short-circuits `pow(0, .) -> 0`).
  id_to_field(id, t) -> int in [1, t-1]
      the identifier's value in Z_t (a keyed hash, independent of the bin hash).
  bin_and_interpolate(ids_O, t, NB, NH, DCAP) -> (P_layers, L_layers, D, alpha, collisions)
      O's side: O simple-hashes each identifier into ALL of its (distinct)
      candidate bins. Each bin's items are split into `alpha` partitions of size
      <= DCAP (balanced: partition a gets items[a::alpha]).  For partition a:
        P_layers[a][d]  (d = 0..D)   length-NB list: degree-d coefficient (mod t)
                        of the monic polynomial whose roots are that partition's
                        field values in each bin (empty partition -> P = 1);
        L_layers[a][d]  (d = 0..D-1) length-NB list: degree-d coefficient of the
                        interpolation polynomial through (field value, label),
                        label = O's 0-based ROW INDEX in its file order (party_r
                        reads it back as sigma);  empty partition -> L = 0.
      D = max partition size over all bins/partitions (>= 1).
      collisions = number of O items dropped because two DIFFERENT rows landed
      in the same bin with the same field value (all rows at such a value are
      dropped, so a colliding bin never returns an ambiguous label).

Hash functions are deterministic and shared by both parties: BLAKE2b keyed by
the hash index (bins) or by a separate key (field value).
"""
from __future__ import annotations

import hashlib
import struct
from typing import Dict, Iterable, List, Optional, Sequence

DUMMY_Y = 0

_BIN_KEYS = [f"vfl-psi-bin-{i}".encode() for i in range(64)]
_FIELD_KEY = b"vfl-psi-field-v1"


# ---------------------------------------------------------------------------
# Framing
# ---------------------------------------------------------------------------

def send_msg(sock, data: bytes) -> None:
    """Send one length-prefixed message."""
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise TypeError(f"send_msg expects bytes, got {type(data).__name__}")
    sock.sendall(struct.pack(">Q", len(data)) + bytes(data))


def _recv_exact(sock, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(min(n - len(buf), 1 << 20))
        if not chunk:
            raise ConnectionError(f"socket closed after {len(buf)}/{n} bytes")
        buf.extend(chunk)
    return bytes(buf)


def recv_msg(sock) -> bytes:
    """Receive one length-prefixed message."""
    (n,) = struct.unpack(">Q", _recv_exact(sock, 8))
    return _recv_exact(sock, n)


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------

def _id_bytes(x) -> bytes:
    return int(x).to_bytes(16, "big", signed=True)


def bin_hash(i: int, x, NB: int) -> int:
    """i-th cuckoo/simple hash of identifier x into [0, NB)."""
    h = hashlib.blake2b(_id_bytes(x), digest_size=8, key=_BIN_KEYS[i]).digest()
    return int.from_bytes(h, "big") % NB


def candidate_bins(x, NB: int, NH: int) -> List[int]:
    return [bin_hash(i, x, NB) for i in range(NH)]


def id_to_field(x, t: int) -> int:
    """Identifier -> nonzero element of Z_t (never equals DUMMY_Y = 0)."""
    h = hashlib.blake2b(_id_bytes(x), digest_size=16, key=_FIELD_KEY).digest()
    return int.from_bytes(h, "big") % (t - 1) + 1


# ---------------------------------------------------------------------------
# R side: batching + cuckoo hashing
# ---------------------------------------------------------------------------

def batch_ids(ids: Sequence, cap: int) -> List[list]:
    """Split R's identifiers into chunks of at most `cap` (one BFV query each)."""
    cap = int(cap)
    if cap < 1:
        raise ValueError("cap must be >= 1")
    ids = list(ids)
    return [ids[i:i + cap] for i in range(0, len(ids), cap)] or [[]]


def build_cuckoo_table(chunk: Iterable, rng, NB: int, NH: int,
                       max_kicks: int = 100_000) -> Optional[Dict[int, int]]:
    """Random-walk cuckoo insertion with NH hash functions into NB bins.

    Returns {slot: id} with every (distinct) id of `chunk` in one of its
    candidate bins, or None if some insertion exceeds `max_kicks` evictions.

    calibrate_hyperparameters.py sets cuckoo_capacity = 0.9 * NB, close to the
    ~0.918 load threshold of 3-hash cuckoo hashing: at that load single insertions
    can need thousands of evictions (a 1000-kick limit failed 2 of 3 trials at
    14745 ids / 16384 bins; 20000 succeeded 3/3 in ~0.6 s), hence the generous cap.
    """
    table: Dict[int, int] = {}
    for x0 in dict.fromkeys(int(v) for v in chunk):      # dedupe, keep order
        cur = x0
        placed = False
        for _ in range(max_kicks):
            cands = candidate_bins(cur, NB, NH)
            for s in cands:
                if s not in table:
                    table[s] = cur
                    placed = True
                    break
            if placed:
                break
            s = cands[int(rng.integers(len(cands)))]     # evict a random occupant
            table[s], cur = cur, table[s]
        if not placed:
            return None
    return table


# ---------------------------------------------------------------------------
# O side: simple hashing, partitioning, polynomial interpolation (mod t)
# ---------------------------------------------------------------------------

def _poly_from_roots(roots: Sequence[int], t: int) -> List[int]:
    """Coefficients (ascending degree) of prod (x - r) mod t; monic."""
    c = [1]
    for r in roots:
        nr = (-r) % t
        new = [0] * (len(c) + 1)
        for k, ck in enumerate(c):
            new[k] = (new[k] + ck * nr) % t
            new[k + 1] = (new[k + 1] + ck) % t
        c = new
    return c


def _synthetic_div(Q: Sequence[int], r: int, t: int) -> List[int]:
    """Q(x) / (x - r) for monic Q with Q(r) = 0; ascending coefficients."""
    deg = len(Q) - 1
    out = [0] * deg
    acc = 0
    for k in range(deg, 0, -1):
        acc = (Q[k] + acc * r) % t
        out[k - 1] = acc
    return out


def _interpolate(xs: Sequence[int], ys: Sequence[int], t: int) -> List[int]:
    """Lagrange interpolation mod prime t; returns len(xs) ascending coeffs."""
    m = len(xs)
    if m == 0:
        return []
    Q = _poly_from_roots(xs, t)
    L = [0] * m
    for xi, yi in zip(xs, ys):
        Qi = _synthetic_div(Q, xi, t)                    # prod_{j != i} (x - x_j)
        den = 0
        for c in reversed(Qi):                           # Horner: Qi(xi)
            den = (den * xi + c) % t
        w = yi * pow(den, t - 2, t) % t
        for k in range(m):
            L[k] = (L[k] + w * Qi[k]) % t
    return L


def bin_and_interpolate(ids_O: Sequence, t: int, NB: int, NH: int, DCAP: int):
    """O's plaintext preprocessing for CLR-style labeled PSI.

    Returns (P_layers, L_layers, D, alpha, collisions); see module docstring.
    """
    n_O = len(ids_O)
    if n_O >= t:
        raise ValueError(f"labels (row indices < {n_O}) do not fit in Z_t, t={t}")
    bins: List[Dict[int, int]] = [dict() for _ in range(NB)]
    poisoned: List[set] = [set() for _ in range(NB)]
    collisions = 0
    for j, x in enumerate(ids_O):
        fv = id_to_field(int(x), t)
        for s in set(candidate_bins(int(x), NB, NH)):   # distinct bins only
            b = bins[s]
            if fv in poisoned[s]:
                collisions += 1
                continue
            if fv in b:
                if b[fv] != j:                           # different row, same value
                    del b[fv]
                    poisoned[s].add(fv)
                    collisions += 2
                continue
            b[fv] = j

    max_load = max((len(b) for b in bins), default=0)
    alpha = max(1, -(-max_load // DCAP))                 # ceil
    D = max(1, -(-max_load // alpha))

    P_layers = [[[0] * NB for _ in range(D + 1)] for _ in range(alpha)]
    L_layers = [[[0] * NB for _ in range(D)] for _ in range(alpha)]
    for s, b in enumerate(bins):
        items = list(b.items())
        for a in range(alpha):
            part = items[a::alpha]
            Pa, La = P_layers[a], L_layers[a]
            if not part:
                Pa[0][s] = 1                             # P = 1: never zero
                continue
            xs = [fv for fv, _ in part]
            ys = [lab for _, lab in part]
            pc = _poly_from_roots(xs, t)
            for d, c in enumerate(pc):
                Pa[d][s] = c
            lc = _interpolate(xs, ys, t)
            for d, c in enumerate(lc):
                La[d][s] = c
    return P_layers, L_layers, D, alpha, collisions
