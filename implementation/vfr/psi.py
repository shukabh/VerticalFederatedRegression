"""Labeled PSI on OPRF outputs (Chen-Laine-Rindal style), adapted from scenario_b/psi_common.py.

Items and labels
  OPRF output y = F(id) (64 bytes): item part y[0:32] (F_k), key part y[32:64] (F'_k).
  item value  = SHA-256("vfr-item" | y[0:32]) mod (t - 1)   -> in [0, t-2], never the padding root t-1
  bin_i(item) = SHA-256("vfr-bin" | i | y[0:32]) mod N_bins, i < h
  label key   = SHA-256("vfr-label" | y[32:64]) mod t
  label       = (tau(j) + key) mod t                         (tau(j) is O's 1-based shuffled position)

Public sizes: N_bins, h, the bin degree D and the number of partitions alpha are public functions of
(n_bar_O, h, N_bins, d_cap, delta_psi), never of the data (the old prototype used the observed maximum
load, which is data dependent).
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import numpy as np
from scipy.stats import binom

SIGMA_FLOOR = 2.0 ** 42          # floor on the flooding s.d. (as dp.SIGMA_C_FLOOR)
T_DEFAULT = 4293918721          # prime, = 1 mod 2^20 (BFV batching for N_B <= 2^19), < 2^32
DUMMY_Y = 0


@dataclass
class PSIParams:
    N_B: int = 16384            # BFV ring dimension
    t: int = T_DEFAULT
    h: int = 3                  # cuckoo hash functions
    d_cap: int = 64             # max degree of a partition polynomial
    delta_psi: float = 2.0 ** -40
    bfv_depth: int = 7          # OpenFHE BFV multiplicative depth (sets the modulus chain: 7 x 60 bits)
    reply_towers: int = 2       # SanitizeBFV: modulus switch to this many towers
    kappa: int = 40             # flooding parameter for SanitizeBFV
    cuckoo_load: float = 0.9

    @property
    def n_bins(self):
        return self.N_B

    def sizes(self, n_bar_O: int):
        """(D, alpha, load bound) from public sizes: the bin load is Binomial(n_bar_O h, 1/N_bins);
        choose B with N_bins * P[load > B] <= delta_psi, then D = min(d_cap, B), alpha = ceil(B / D)."""
        n = n_bar_O * self.h
        k = np.arange(0, n + 1)
        ok = binom.logsf(k, n, 1.0 / self.n_bins) + math.log(self.n_bins) <= math.log(self.delta_psi)
        B = int(k[np.argmax(ok)]) if ok.any() else n          # smallest B with N_bins P[load > B] <= delta_psi
        B = max(B, 1)
        D = min(self.d_cap, B)
        alpha = -(-B // D)
        return D, alpha, B

    def windows(self, D: int) -> int:
        return max(1, math.ceil(math.log2(D))) + 1 if D > 1 else 1

    def B_err(self) -> float:
        """Worst-case l2 bound of the error of a reply after the modulus switch, excluding terms that are
        functions of the decrypted plaintext (which R learns anyway): rounding e0 + e1 s of the switched
        evaluation result and of the switched Enc_pk(0) (each <= sqrt(N)/2 + N^1.5/2, ternary s), plus 1
        for the scaled evaluation noise (measured 2^-72 at the default chain; not a proven bound)."""
        N = self.N_B
        return 2 * (math.sqrt(N) / 2 + N ** 1.5 / 2) + 1.0

    def flood_sigma2(self) -> int:
        """sigma_psi^2 with sigma_psi = max(2^kappa_psi B_err, 2^42): above the requirement
        max(2^kappa_psi B_psi / sqrt(2 pi), 2^42) of SanitizeBFV (the floor makes the smudging lemma's factor
        1 + 2^-80 hold; at the defaults sigma_psi = 2^61 and the floor does not bind)."""
        return math.ceil(max(2.0 ** self.kappa * self.B_err(), SIGMA_FLOOR) ** 2)


def _h(*parts: bytes) -> bytes:
    return hashlib.sha256(b'|'.join(parts)).digest()


def item_value(y: bytes, t: int) -> int:
    return int.from_bytes(_h(b'vfr-item', y[:32]), 'little') % (t - 1)


def bins_of(y: bytes, h: int, n_bins: int):
    return [int.from_bytes(_h(b'vfr-bin', bytes([i]), y[:32])[:8], 'little') % n_bins for i in range(h)]


def label_key(y: bytes, t: int) -> int:
    return int.from_bytes(_h(b'vfr-label', y[32:])[:8], 'little') % t


# ------------------------------------------------------------------------------------------- O side
def bin_items(oprf_outputs, positions, pp: PSIParams, n_bar_O: int):
    """O: hash every row into its h bins and assign it to a partition (no duplicate item value inside a
    partition).  Returns the (alpha, N_bins, D) item/label arrays, the counts and the public sizes."""
    D, alpha, B = pp.sizes(n_bar_O)
    t, nb = pp.t, pp.n_bins
    bins = [[] for _ in range(nb)]
    for y, pos in zip(oprf_outputs, positions):
        x = item_value(y, t)
        lab = (pos + label_key(y, t)) % t
        for b in sorted(set(bins_of(y, pp.h, nb))):
            bins[b].append((x, lab))
    xs = np.zeros((alpha, nb, D), dtype=np.uint64)
    ys = np.zeros((alpha, nb, D), dtype=np.uint64)
    counts = np.zeros((alpha, nb), dtype=np.int32)
    n_dup = 0
    for b, items in enumerate(bins):
        if len(items) > alpha * D:
            raise RuntimeError('PSI bin overflow (probability <= delta_psi): abort the run')
        seen = [set() for _ in range(alpha)]
        for x, lab in items:
            # first partition that has room and does not already hold the same item value
            for a in range(alpha):
                if counts[a, b] < D and x not in seen[a]:
                    xs[a, b, counts[a, b]] = x
                    ys[a, b, counts[a, b]] = lab
                    counts[a, b] += 1
                    seen[a].add(x)
                    break
            else:
                raise RuntimeError('PSI: duplicate item values exhaust the partitions of a bin')
            if any(x in s_ for s_ in seen[:a]):
                n_dup += 1
    return dict(xs=xs, ys=ys, counts=counts, D=D, alpha=alpha, B=B, n_dup=n_dup,
                max_load=max(len(x) for x in bins))


def sender_tables(oprf_outputs, positions, pp: PSIParams, n_bar_O: int):
    """O's per-run preprocessing.  oprf_outputs[j] = F(id_j), positions[j] = tau(j) (1-based).
    Returns (Pc, Lc) layers (alpha, D+1, N_bins) and (alpha, D, N_bins) and the public sizes."""
    import time
    t0 = time.time()
    bi = bin_items(oprf_outputs, positions, pp, n_bar_O)
    t1 = time.time()
    from . import vfr_native as vn
    Pc, Lc = vn.psi_build_polys(bi['xs'], bi['ys'], bi['counts'], pp.t)
    return dict(Pc=Pc, Lc=Lc, D=bi['D'], alpha=bi['alpha'], B=bi['B'], n_dup=bi['n_dup'], max_load=bi['max_load'],
                t_bin=t1 - t0, t_polys=time.time() - t1)


def masked_layers(Pc, Lc, rng_bytes, t: int):
    """Fold fresh masks r1, r2 (uniform in Z_t^*, one per bin and partition) into the coefficients:
    Sc = r1 * Pc, Qc = r2 * Pc + Lc (mod t)."""
    alpha, D1, nb = Pc.shape
    out = []
    for a in range(alpha):
        r1 = _uniform_nonzero(rng_bytes, nb, t)
        r2 = _uniform_nonzero(rng_bytes, nb, t)
        P = Pc[a].astype(np.uint64)
        Sc = (P * r1[None, :]) % np.uint64(t)                 # products < 2^64 since t < 2^32
        Qc = (P * r2[None, :]) % np.uint64(t)
        Qc[:D1 - 1] = (Qc[:D1 - 1] + Lc[a]) % np.uint64(t)
        out.append((Sc, Qc))
    return out


def _uniform_nonzero(rng, n, t):
    x = rng.uniform_mod_array(t - 1, n) + np.uint64(1)
    return x.astype(np.uint64)


# ------------------------------------------------------------------------------------------- R side
def cuckoo(items_bins, n_bins: int, h: int, rng, max_kicks: int = 2000):
    """items_bins[i] = list of h candidate bins of item i.  Returns dict bin -> item index, or None."""
    table = {}
    for i in range(len(items_bins)):
        cur = i
        for _ in range(max_kicks):
            placed = False
            for b in items_bins[cur]:
                if b not in table:
                    table[b] = cur
                    placed = True
                    break
            if placed:
                break
            b = items_bins[cur][int(rng.integers(h))]
            cur, table[b] = table[b], cur
        else:
            return None
    return table


def receiver_batches(oprf_outputs, pp: PSIParams, rng):
    """R: split its (padded) items into cuckoo tables of at most cuckoo_load * N_bins items each."""
    cap = int(pp.cuckoo_load * pp.n_bins)
    t = pp.t
    batches = []
    for start in range(0, len(oprf_outputs), cap):
        chunk = list(range(start, min(start + cap, len(oprf_outputs))))
        cand = [bins_of(oprf_outputs[i], pp.h, pp.n_bins) for i in chunk]
        tab = cuckoo(cand, pp.n_bins, pp.h, rng)
        if tab is None:
            raise RuntimeError('cuckoo hashing failed')
        q = np.full(pp.n_bins, DUMMY_Y, dtype=np.uint64)
        slot_item = {}
        for b, li in tab.items():
            i = chunk[li]
            q[b] = item_value(oprf_outputs[i], t)
            slot_item[b] = i
        batches.append(dict(query=q, slot_item=slot_item))
    return batches


def window_values(q: np.ndarray, W: int, t: int):
    """w^(2^i) mod t for i < W, slot-wise."""
    out = []
    cur = q.astype(np.uint64) % np.uint64(t)
    for i in range(W):
        out.append(cur.copy())
        cur = np.array([(int(v) * int(v)) % t for v in cur], dtype=np.uint64)   # exact (Python ints)
    return out


class PSIOutputRejected(RuntimeError):
    """R rejects the PSI output: the run aborts (O observes the abort)."""


def check_injective(found):
    """R: reject an output in which two identifiers receive the same position.  An honest O assigns distinct
    positions tau(j), so a repeated position can only come from a false match; aborting keeps it out of pi."""
    seen = {}
    for i, pos in sorted(found.items()):
        if pos in seen:
            raise PSIOutputRejected(f'PSI output is not injective: items {seen[pos]} and {i} both received '
                                    f'position {pos}; abort the run')
        seen[pos] = i


def receiver_decode(S_list, Q_list, slot_item, oprf_outputs, t: int, n_bar_O: int):
    """From the decrypted replies of each partition, recover {item index: position}.  Flags ambiguities."""
    found, ambiguous = {}, set()
    for S, Q in zip(S_list, Q_list):
        for b, i in slot_item.items():
            if int(S[b]) == 0:
                pos = (int(Q[b]) - label_key(oprf_outputs[i], t)) % t
                if i in found and found[i] != pos:
                    ambiguous.add(i)
                found[i] = pos
    bad = {i for i, p in found.items() if not (1 <= p <= n_bar_O)}
    return found, ambiguous, bad
