"""
psi_common.py  —  Phase 0 shared primitives (pure Python, NO TenSEAL import)

Everything that is *not* homomorphic lives here so that the TenSEAL parties
(phase0_r.py, phase0_o.py) and the plaintext validator (phase0_validate.py)
run byte-identical hashing, binning, interpolation and alignment logic. What
I can validate without TenSEAL is therefore exactly what the encrypted path
executes on the non-crypto steps.

Fixes folded in (from the code review):
  * lagrange_coeffs / bin nodes are DEDUP-GUARDED  -> no crash on duplicate
    field values within a bin (duplicate ID or field collision).
  * id_to_field excludes DUMMY_ROOT from its range (reduce mod p-1).
  * cuckoo capacity is exposed so callers can BATCH when n_R > ~0.9*NUM_BINS.
"""

from __future__ import annotations

import hashlib
import socket
import struct
import pickle
from typing import Dict, List, Tuple


# ==============================================================================
# Shared constants  (must match across all three scripts)
# ==============================================================================

PLAIN_MODULUS  = 4_293_918_721          # prime, 1 (mod 2*16384): BFV SIMD
POLY_DEGREE    = 16_384
NUM_BINS       = POLY_DEGREE
COEFF_MOD_BITS = [60, 60, 60, 60, 60, 60, 60]   # 420-bit, ~7 ct x ct budget

N_HASH     = 3
D_CAP      = 64
N_WINDOW   = 7                          # ceil(log2(D_CAP)) + 1 -> covers y^1..y^64
DUMMY_ROOT = PLAIN_MODULUS - 1          # padding root; excluded from id_to_field
DUMMY_Y    = 0                          # query value packed into empty cuckoo slots

CUCKOO_LOAD = 0.9                       # 3-hash random-walk threshold (~0.918)

HOST = "127.0.0.1"
PORT = 65432


def cuckoo_capacity(num_bins: int = NUM_BINS) -> int:
    """Max R IDs per single query vector before cuckoo insertion fails."""
    return int(CUCKOO_LOAD * num_bins)


# ==============================================================================
# Length-prefixed socket messaging
# ==============================================================================

def send_msg(sock, data: bytes) -> None:
    sock.sendall(struct.pack(">Q", len(data)))
    for i in range(0, len(data), 64 << 20):
        sock.sendall(data[i:i + (64 << 20)])


def recv_msg(sock):
    raw = _recvall(sock, 8)
    if not raw:
        return None
    return _recvall(sock, struct.unpack(">Q", raw)[0])


def _recvall(sock, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(min(n - len(buf), 64 << 20))
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


# ==============================================================================
# Hashing  (identical on both parties)
# ==============================================================================

def bin_hash(i: int, x: int, num_bins: int = NUM_BINS) -> int:
    """i-th bin-assignment hash, shared by R and O."""
    return int(hashlib.sha256(f"bin{i}|{int(x)}".encode()).hexdigest(), 16) % num_bins


def id_to_field(x, p: int = PLAIN_MODULUS) -> int:
    """Map an ID into Z_p, EXCLUDING DUMMY_ROOT (= p-1).

    Reducing mod (p-1) yields a value in [0, p-2], so a real query point can
    never coincide with the dummy padding root and spuriously match padding.
    All polynomial arithmetic still uses the prime modulus p.
    """
    h = int(hashlib.sha256(str(int(x)).encode()).hexdigest(), 16) % (p - 1)
    return h                                    # in [0, p-2], never p-1


# ==============================================================================
# Plaintext polynomial arithmetic over Z_p  (ascending coefficient lists)
# ==============================================================================

def poly_from_roots_mod(roots, p):
    coeffs = [1]
    for r in roots:
        new = [0] * (len(coeffs) + 1)
        for i, c in enumerate(coeffs):
            new[i + 1] = (new[i + 1] + c) % p
            new[i]     = (new[i]     - r * c) % p
        coeffs = new
    return coeffs


def polymul_mod(a, b, p):
    out = [0] * (len(a) + len(b) - 1)
    for i, ai in enumerate(a):
        if ai:
            for j, bj in enumerate(b):
                out[i + j] = (out[i + j] + ai * bj) % p
    return out


def divide_by_root(coeffs, r, p):
    """Synthetic division coeffs / (x - r); exact when r is a root."""
    n = len(coeffs) - 1
    q = [0] * n
    q[n - 1] = coeffs[n]
    for i in range(n - 1, 0, -1):
        q[i - 1] = (coeffs[i] + r * q[i]) % p
    return q


def poly_eval_mod(coeffs, x, p):
    val = 0
    for c in reversed(coeffs):
        val = (val * x + c) % p
    return val


def _dedup_nodes(points):
    """Keep-first on duplicate x-nodes. Returns (unique_points, n_collisions).

    A collision here means two DISTINCT O-rows share a field value inside one
    bin partition (duplicate ID, or a genuine field collision). Un-guarded
    Lagrange interpolation would divide by zero; we drop the later row and
    report the count so the caller can log it.
    """
    seen = {}
    out  = []
    coll = 0
    for (x, y) in points:
        if x in seen:
            coll += 1
            continue
        seen[x] = y
        out.append((x, y))
    return out, coll


def lagrange_coeffs(points, p):
    """Interpolate L with L(x_i)=y_i through DISTINCT nodes (dedup-guarded)."""
    points, _ = _dedup_nodes(points)
    n = len(points)
    if n == 0:
        return []
    xs   = [pt[0] for pt in points]
    full = poly_from_roots_mod(xs, p)               # prod (x - x_i)
    out  = [0] * n
    for (xi, yi) in points:
        qi    = divide_by_root(full, xi, p)
        denom = poly_eval_mod(qi, xi, p)            # prod_{k!=i}(x_i - x_k) != 0
        scale = (yi * pow(denom, -1, p)) % p
        for d in range(n):
            out[d] = (out[d] + scale * qi[d]) % p
    return out


# ==============================================================================
# Cuckoo hashing (R side)  +  batching
# ==============================================================================

def build_cuckoo_table(ids, rng, num_bins: int = NUM_BINS,
                       n_hash: int = N_HASH, max_kicks: int = 5000):
    """Place each ID into exactly one of its n_hash candidate bins.
    Returns dict slot -> id, or None if insertion fails (load too high)."""
    table: Dict[int, int] = {}
    for x in ids:
        cur = int(x)
        for _ in range(max_kicks):
            placed = False
            for i in range(n_hash):
                s = bin_hash(i, cur, num_bins)
                if s not in table:
                    table[s] = cur
                    placed = True
                    break
            if placed:
                break
            i = int(rng.integers(n_hash))
            s = bin_hash(i, cur, num_bins)
            cur, table[s] = table[s], cur
        else:
            return None
    return table


def batch_ids(ids, cap: int) -> List[list]:
    """Split R's IDs into batches that each fit one cuckoo table."""
    return [list(ids[i:i + cap]) for i in range(0, len(ids), cap)]


# ==============================================================================
# O-side preprocessing: bin, partition, interpolate membership + label polys
# ==============================================================================

def bin_and_interpolate(ids_O, p: int = PLAIN_MODULUS,
                        num_bins: int = NUM_BINS, n_hash: int = N_HASH,
                        d_cap: int = D_CAP):
    """
    Simple-hash O's IDs into all candidate bins, partition each bin into
    degree-D chunks, and interpolate per chunk:
        P (membership): real roots + dummy padding, degree D
        L (label):      L(field_value_j) = j  (O's row index in X_O.csv order)

    Returns (P_layers, L_layers, D, alpha, n_collisions) where
        P_layers[a][d][slot]  is the degree-d coefficient of partition a, bin slot
        L_layers[a][d][slot]  likewise (d < D)
    The label j is O's ROW INDEX in the CSV. Because X_O.csv is already emitted
    in O's shuffled order by the generator (perm_O == tau), j is directly the
    canonical Phase-1 index; in production O would apply tau here.
    """
    bins: List[List[Tuple[int, int]]] = [[] for _ in range(num_bins)]
    for j, x in enumerate(ids_O):
        x  = int(x)
        fv = id_to_field(x, p)
        seen = set()
        for i in range(n_hash):
            s = bin_hash(i, x, num_bins)
            if s not in seen:
                bins[s].append((fv, j))
                seen.add(s)

    max_load = max((len(b) for b in bins), default=1)
    D        = min(d_cap, max_load) if max_load else 1
    alpha    = (max_load + D - 1) // D if D else 1

    # dummy padding polynomials (x - DUMMY_ROOT)^t
    dummy_pows = [[1]]
    for _ in range(1, D + 1):
        dummy_pows.append(polymul_mod(dummy_pows[-1], [(-DUMMY_ROOT) % p, 1], p))

    P_layers = [[[0] * num_bins for _ in range(D + 1)] for _ in range(alpha)]
    L_layers = [[[0] * num_bins for _ in range(D)]     for _ in range(alpha)]

    total_coll = 0
    for slot in range(num_bins):
        items = bins[slot]
        for a in range(alpha):
            chunk = items[a * D:(a + 1) * D]
            chunk, coll = _dedup_nodes(chunk)       # guard: distinct field vals
            total_coll += coll
            n_real = len(chunk)

            if n_real == 0:
                P = dummy_pows[D]
            else:
                P_real = poly_from_roots_mod([fv for fv, _ in chunk], p)
                P = polymul_mod(P_real, dummy_pows[D - n_real], p) \
                    if n_real < D else P_real
            for d in range(D + 1):
                P_layers[a][d][slot] = P[d]

            if n_real:
                L = lagrange_coeffs(chunk, p)        # degree n_real-1
                for d in range(n_real):
                    L_layers[a][d][slot] = L[d]

    return P_layers, L_layers, D, alpha, total_coll


def eval_layers_plaintext(P_layers, L_layers, D, alpha, slot, y, p):
    """Plaintext mirror of the HE evaluation at a single (slot, query y):
    returns list over partitions of (P(y), L(y)) with P,L = sum_d coeff[d] y^d.
    HE path computes the identical sum via encrypted windowed powers y^d."""
    out = []
    for a in range(alpha):
        Pv = 0
        for d in range(D + 1):
            Pv = (Pv + P_layers[a][d][slot] * pow(y, d, p)) % p
        Lv = 0
        for d in range(D):
            Lv = (Lv + L_layers[a][d][slot] * pow(y, d, p)) % p
        out.append((Pv, Lv))
    return out


# ==============================================================================
# HE helper (used by the TenSEAL parties only; pure-list, no ts import here)
# ==============================================================================

def power_from_windows(d: int, windows):
    """Reconstruct Enc(y^d) as a balanced product of window ciphertexts."""
    facs = [windows[i] for i in range(d.bit_length()) if (d >> i) & 1]
    while len(facs) > 1:
        nxt = [facs[j] * facs[j + 1] for j in range(0, len(facs) - 1, 2)]
        if len(facs) % 2:
            nxt.append(facs[-1])
        facs = nxt
    return facs[0]


# ==============================================================================
# Alignment output helpers
# ==============================================================================

def selection_vector(sigma: Dict[int, int], n_O: int):
    """b in {0,1}^{n_O}: b[j] = 1 iff O-row j is matched."""
    b = [0] * n_O
    for j in sigma.values():
        if 0 <= j < n_O:
            b[j] = 1
    return b
