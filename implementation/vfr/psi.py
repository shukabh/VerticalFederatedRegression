"""Labeled PSI on OPRF outputs (Chen-Laine-Rindal style), adapted from scenario_b/psi_common.py.

Items and labels (F = felts, the number of field elements per item, APSI's "felts per item"; default 2)
  OPRF output y = F(id) (64 bytes): item part y[0:32] (F_k), key part y[32:64] (F'_k).
  felt 0      = SHA-256("vfr-item" | y[0:32]) mod (t - 1)          -> in [0, t-2], never the padding root t-1
  felt f >= 1 = SHA-256("vfr-item" | f | y[0:32]) mod (t - 1)
  bin_i(item) = SHA-256("vfr-bin" | i | y[0:32]) mod N_bins, i < h   (one bin for all felts of the item)
  label key f = SHA-256("vfr-label" | y[32:64]) mod t for f = 0, SHA-256("vfr-label" | f | y[32:64]) mod t otherwise
  label felt f = (tau(j) + key_f) mod t                            (tau(j) is O's 1-based shuffled position)

Slot layout: felt f of bin b sits in slot f * N_bins + b of a vector of F * N_bins slots, i.e. in ciphertext f
(N_bins = N_B slots each).  Every slot carries its own membership and label polynomial, so O interpolates felt f of
the items of bin b (and their label felt f) in slot (f, b); the items of one partition hold distinct values in every
felt.  R queries felt f of its item in slot (f, b) and declares a match in a partition only if (i) the masked
membership value is 0 in all F slots of the bin and (ii) the F label felts decode to the same position
(label_f - key_f(x) mod t equal for all f).  A true match always passes both.  A non-member can pass (i) with felts
from different items of the partition (as in APSI); such a cross-item match decodes F unrelated positions, so (ii)
rejects it except with probability <= t^-(F-1) per event, and R treats it as no match (no abort, nothing for O to
observe).  An event that passes both is what can still abort the run or, if its position is in range and not
repeated, change the output; its probability p_pass is bounded exactly by `PSIParams.false_match` and must be at
most delta_psi (Setup refuses otherwise, `PSIParams.check`).

Public sizes: N_bins, h, F, the bin degree D, the number of partitions alpha and the number of cuckoo tables are
public functions of (n_bar_O, n_R^max, h, N_bins, d_cap, delta_psi), never of the data (the old prototype used the
observed maximum load, which is data dependent).
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from fractions import Fraction

import numpy as np
from scipy.stats import binom

SIGMA_FLOOR = 2.0 ** 42          # floor on the flooding s.d. (as dp.SIGMA_C_FLOOR)
T_DEFAULT = 4293918721          # prime, = 1 mod 2^20 (BFV batching for N_B <= 2^19), < 2^32
DUMMY_Y = 0


class PSIParamsRefused(ValueError):
    """The PSI parameters do not meet their failure-probability or SanitizeBFV requirements: Setup refuses."""


def hall_bound(n: int, m: int, h: int) -> float:
    """Upper bound on the probability that no assignment of n items with h uniform candidate bins among m exists
    (Hall's condition and a union bound; crypto proofs, Lemma "The cuckoo-hashing slack"; every term summed, as in
    experiments/proofs_check/tv1_cuckoo.py)."""
    tot = 0.0
    for s in range(2, n + 1):
        lt = (math.lgamma(n + 1) - math.lgamma(s + 1) - math.lgamma(n - s + 1)
              + math.lgamma(m + 1) - math.lgamma(s) - math.lgamma(m - s + 2) + h * s * math.log((s - 1) / m))
        tot += math.exp(min(lt, 700.0))
    return tot


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
    cuckoo_load: float = 0.9    # hard cap on the load of one cuckoo table (the table count also meets delta_psi)
    felts: int = 2              # field elements per item (64-bit items at t ~ 2^32)
    t_eta: float = 10.0         # SanitizeBFV: tail factor of the flooding noise in the switch condition
    sampler_xi: float = 2.0 ** -80   # failure bound of one budgeted flooding sampling call (N_B values; A24)

    @property
    def n_bins(self):
        return self.N_B

    @property
    def n_slots(self):
        """Length of the slot axis of a query and of O's coefficient layers: F ciphertexts of N_B slots."""
        return self.felts * self.N_B

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

    def tables(self, n_R_max: int):
        """(number of cuckoo tables k, items per table at most, p_cuckoo bound): the smallest k with
        ceil(n_R^max / k) <= cuckoo_load N_bins and k * hall_bound(ceil(n_R^max / k), N_bins, h) <= delta_psi
        (R's queries are split into k balanced tables).  If no k up to the cap meets delta_psi, the smallest k
        allowed by the load cap is returned with its (too large) bound, and `check` refuses."""
        cap = int(self.cuckoo_load * self.n_bins)
        k0 = max(1, -(-n_R_max // cap))
        first = None
        for k in range(k0, max(k0, n_R_max) + 1):
            per = -(-n_R_max // k)
            p = k * hall_bound(per, self.n_bins, self.h)
            if first is None:
                first = (k, per, p)
            if p <= self.delta_psi:
                return k, per, p
            if per <= 1:
                break
        return first

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

    def sanitize_check(self, moduli) -> dict:
        """SanitizeBFV's requirements for the BFV chain `moduli` (the first reply_towers limbs are kept):
        flooding sigma_psi >= max(2^kappa_psi B_psi / sqrt(2 pi), 2^42), and the modulus-switch condition
        q'/(2t) > t_eta sigma_psi + B_psi + t (B_psi = B_err).  Exact integer comparisons (sigma_psi^2 is an
        integer; both sides squared where needed)."""
        qp = math.prod(int(q) for q in list(moduli)[:self.reply_towers])
        s2 = self.flood_sigma2()
        B = Fraction(self.B_err())
        need = max(Fraction(2) ** self.kappa * B, Fraction(SIGMA_FLOOR))     # sigma_psi >= need (>= the / sqrt(2 pi) form)
        flood_ok = Fraction(s2) >= need * need
        sig_up = Fraction(math.isqrt(s2) + 1)                                # >= sigma_psi
        rhs = Fraction(self.t_eta) * sig_up + B + self.t
        switch_ok = Fraction(qp, 2 * self.t) > rhs
        return dict(flood_ok=bool(flood_ok), switch_ok=bool(switch_ok), log2_qprime=math.log2(qp),
                    log2_sigma_psi=0.5 * math.log2(s2), log2_switch_margin=float(math.log2(Fraction(qp, 2 * self.t) / rhs)))

    def false_match(self, n_bar_O: int, n_R_max: int) -> dict:
        """Exact upper bounds (random-oracle model for the hashes) on the false-match events of one run.

        Per query and partition with c items (c <= D, sum over the partitions of a bin <= B, the public load bound):
        each felt of the query is uniform on Z_{t-1} up to the bias of reducing a 256-bit hash, so it hits one of
        the c distinct values of that felt with probability <= c u_item, u_item = (floor(2^256/(t-1)) + 1)/2^256,
        independently across felts.  Hence
          p_felt = n_R^max S_F u_item^F,  S_F = max sum_a c_a^F = floor(B/D) D^F + (B mod D)^F
        bounds the probability that some query has all F membership values 0 in some partition (with felts from any
        items of the partition); p_item = n_R^max B u_item^F the same with all F felts from one item.  Such an event
        passes R's label-consistency test only if F - 1 independent label keys of the query hit given values:
          p_pass = p_felt u_key^(F-1),  u_key = (floor(2^256/t) + 1)/2^256,
        and only an event that passes can abort the run (ambiguous, out of range, repeated position) or change the
        output: p_silent = p_pass n_bar_O u_key (its position must also be in range).  With F = 1 there is no
        consistency test and p_pass = p_felt.  Returns floats and log2 values; `check` requires p_pass <= delta_psi."""
        D, alpha, B = self.sizes(n_bar_O)
        F = self.felts
        u_item = Fraction(2 ** 256 // (self.t - 1) + 1, 2 ** 256)
        u_key = Fraction(2 ** 256 // self.t + 1, 2 ** 256)
        full, rem = divmod(B, D)
        S_F = full * D ** F + rem ** F
        p_felt = n_R_max * S_F * u_item ** F
        p_item = n_R_max * B * u_item ** F
        p_pass = p_felt * u_key ** (F - 1)
        p_silent = p_pass * n_bar_O * u_key
        mean_load = Fraction(n_bar_O * self.h, self.n_bins)
        p_mean = n_R_max * mean_load * u_item ** F            # the paper's n_R^max n_bar_O h / (N_bins (t-1)^F)

        def lg(x):
            return float(math.log2(x)) if x > 0 else float('-inf')
        return dict(felts=F, D=D, alpha=alpha, B=B, S_F=S_F, p_felt=float(p_felt), p_item=float(p_item),
                    p_pass=float(p_pass), p_silent=float(p_silent), p_mean_formula=float(p_mean),
                    log2_p_felt=lg(p_felt), log2_p_item=lg(p_item), log2_p_pass=lg(p_pass), log2_p_silent=lg(p_silent),
                    log2_p_mean_formula=lg(p_mean), item_bits=F * math.log2(self.t - 1),
                    ok=bool(p_pass <= Fraction(self.delta_psi)))

    def check(self, n_bar_O: int, n_R_max: int, moduli=None) -> dict:
        """Setup's test of the PSI parameters for the run's public sizes: the false matches that can abort the run
        or change its output (p_pass <= delta_psi), the cuckoo-hashing failure of R's tables (<= delta_psi), the
        bin overflow (<= delta_psi by the choice of B in `sizes`) and, if the BFV moduli are given, SanitizeBFV's
        flooding and modulus-switch conditions.  Returns a dict with 'ok' and 'reasons'."""
        reasons = []
        if self.felts < 1 or int(self.felts) != self.felts:
            raise PSIParamsRefused('felts must be a positive integer')
        fm = self.false_match(n_bar_O, n_R_max)
        if not fm['ok']:
            reasons.append(f"false matches: p_pass = 2^{fm['log2_p_pass']:.1f} > delta_psi with {self.felts} felt(s) "
                           f"per item (raise felts)")
        k, per, pc = self.tables(n_R_max)
        if pc > self.delta_psi:
            reasons.append(f'cuckoo hashing: {k} table(s) of {per} items fail with probability up to 2^{math.log2(pc):.1f}'
                           f' > delta_psi')
        out = dict(false_match=fm, tables=k, per_table=per, p_cuckoo=pc,
                   log2_p_cuckoo=math.log2(pc) if pc > 0 else float('-inf'))
        # SanitizeBFV draws 2 alpha F vectors of N_B flooding values per cuckoo table with the budgeted sampler; the
        # timing of the replies depends on the flooding noise only if one of these calls fails, which aborts the run
        # (probability <= p_sampler, added to the statistical distance of G4's second layer; DEVIATIONS A24)
        D_, alpha_, _ = self.sizes(n_bar_O)
        n_calls = 2 * alpha_ * self.felts * k
        p_sampler = n_calls * float(self.sampler_xi)
        out.update(sampler_calls=n_calls, p_sampler=p_sampler, log2_p_sampler=math.log2(p_sampler))
        if not (0 < self.sampler_xi <= 2.0 ** -40) or p_sampler > self.delta_psi:
            reasons.append(f'SanitizeBFV sampler: {n_calls} calls fail with probability up to '
                           f'2^{math.log2(p_sampler):.1f} > delta_psi (lower sampler_xi)')
        if moduli is not None:
            sc = self.sanitize_check(moduli)
            out['sanitize'] = sc
            if not sc['flood_ok']:
                reasons.append('SanitizeBFV: flooding s.d. below max(2^kappa_psi B_psi / sqrt(2 pi), 2^42)')
            if not sc['switch_ok']:
                reasons.append("SanitizeBFV: q'/(2t) <= t_eta sigma_psi + B_psi + t after the modulus switch")
        out['ok'] = not reasons
        out['reasons'] = reasons
        return out


def _h(*parts: bytes) -> bytes:
    return hashlib.sha256(b'|'.join(parts)).digest()


def item_felts(y: bytes, t: int, felts: int):
    """The F field elements of the item of OPRF output y, each in [0, t-2]."""
    out = [int.from_bytes(_h(b'vfr-item', y[:32]), 'little') % (t - 1)]
    for f in range(1, felts):
        out.append(int.from_bytes(_h(b'vfr-item', bytes([f]), y[:32]), 'little') % (t - 1))
    return out


def item_value(y: bytes, t: int) -> int:
    """Felt 0 of the item (the whole item when felts = 1)."""
    return item_felts(y, t, 1)[0]


def bins_of(y: bytes, h: int, n_bins: int):
    return [int.from_bytes(_h(b'vfr-bin', bytes([i]), y[:32])[:8], 'little') % n_bins for i in range(h)]


def label_key(y: bytes, t: int, f: int = 0) -> int:
    """Label key of felt f, uniform on Z_t up to a relative bias of t / 2^256 (the whole hash is reduced)."""
    parts = (b'vfr-label', y[32:]) if f == 0 else (b'vfr-label', bytes([f]), y[32:])
    return int.from_bytes(_h(*parts), 'little') % t


# ------------------------------------------------------------------------------------------- O side
def _greedy_bin(rows, X, alpha: int, D: int):
    """Place the items `rows` of one bin (in this order) into partitions: the first partition with room whose items
    differ from the item in every felt.  Returns (partition, index) per row and the number of rows displaced."""
    F = X.shape[1]
    seen = [[set() for _ in range(F)] for _ in range(alpha)]
    cnt = [0] * alpha
    out, n_dup = [], 0
    for r in rows:
        x = [int(v) for v in X[r]]
        for a in range(alpha):
            if cnt[a] < D and all(x[f] not in seen[a][f] for f in range(F)):
                out.append((a, cnt[a]))
                cnt[a] += 1
                for f in range(F):
                    seen[a][f].add(x[f])
                break
        else:
            raise RuntimeError('PSI: duplicate felt values exhaust the partitions of a bin')
        if a > 0 and any(any(x[f] in seen[a2][f] for f in range(F)) for a2 in range(a)):
            n_dup += 1
    return out, n_dup


def bin_items(oprf_outputs, positions, pp: PSIParams, n_bar_O: int):
    """O: hash every row into its h bins and assign it to a partition, in row order: the k-th item of a bin goes to
    partition k // D unless an item already there shares one of its felt values, in which case the bin is placed by
    the first-fit rule of `_greedy_bin` (rare: a felt collision inside a partition has probability ~ D^2 / 2t).
    Returns the (alpha, F * N_bins, D) item/label arrays (slot f * N_bins + b holds felt f of bin b), the counts and
    the public sizes."""
    D, alpha, B = pp.sizes(n_bar_O)
    t, nb, F, h = pp.t, pp.n_bins, pp.felts, pp.h
    n = len(oprf_outputs)
    X = np.empty((n, F), dtype=np.uint64)
    L = np.empty((n, F), dtype=np.uint64)
    BIN = np.empty((n, h), dtype=np.int64)
    for j, (y, pos) in enumerate(zip(oprf_outputs, positions)):
        X[j] = item_felts(y, t, F)
        L[j] = [(int(pos) + label_key(y, t, f)) % t for f in range(F)]
        BIN[j] = bins_of(y, h, nb)
    Bs = np.sort(BIN, axis=1)
    keep = np.ones(Bs.shape, dtype=bool)
    keep[:, 1:] = Bs[:, 1:] != Bs[:, :-1]                      # an item enters each of its distinct bins once
    rows = np.broadcast_to(np.arange(n)[:, None], Bs.shape)[keep]
    bins = Bs[keep]
    order = np.argsort(bins, kind='stable')                    # within a bin, row order
    rows, bins = rows[order], bins[order]
    load = np.bincount(bins, minlength=nb)
    max_load = int(load.max()) if n else 0
    if max_load > alpha * D:
        raise RuntimeError('PSI bin overflow (probability <= delta_psi): abort the run')
    start = np.concatenate([[0], np.cumsum(load)[:-1]])
    rank = np.arange(len(bins)) - start[bins]
    part, idx = rank // D, rank % D
    # felt collisions inside a (bin, partition): redo those bins with the first-fit rule
    bad_bins = set()
    for f in range(F):
        key = np.lexsort((X[rows, f], part, bins))
        same = (bins[key][1:] == bins[key][:-1]) & (part[key][1:] == part[key][:-1]) & \
               (X[rows, f][key][1:] == X[rows, f][key][:-1])
        bad_bins.update(int(b) for b in bins[key][1:][same])
    n_dup = 0
    for b in sorted(bad_bins):
        sel = np.nonzero(bins == b)[0]
        placed, nd = _greedy_bin(rows[sel], X, alpha, D)
        n_dup += nd
        part[sel] = [p for p, _ in placed]
        idx[sel] = [i for _, i in placed]
    xs = np.zeros((alpha, F * nb, D), dtype=np.uint64)
    ys = np.zeros((alpha, F * nb, D), dtype=np.uint64)
    counts = np.zeros((alpha, F * nb), dtype=np.int32)
    cnt = np.zeros((alpha, nb), dtype=np.int32)
    np.add.at(cnt, (part, bins), 1)
    for f in range(F):
        xs[part, f * nb + bins, idx] = X[rows, f]
        ys[part, f * nb + bins, idx] = L[rows, f]
        counts[:, f * nb:(f + 1) * nb] = cnt
    return dict(xs=xs, ys=ys, counts=counts, D=D, alpha=alpha, B=B, n_dup=n_dup, n_bins_redone=len(bad_bins),
                max_load=max_load)


def sender_tables(oprf_outputs, positions, pp: PSIParams, n_bar_O: int):
    """O's per-run preprocessing.  oprf_outputs[j] = F(id_j), positions[j] = tau(j) (1-based).
    Returns (Pc, Lc) layers (alpha, D+1, F N_bins) and (alpha, D, F N_bins) and the public sizes."""
    import time
    t0 = time.time()
    bi = bin_items(oprf_outputs, positions, pp, n_bar_O)
    t1 = time.time()
    from . import vfr_native as vn
    Pc, Lc = vn.psi_build_polys(bi['xs'], bi['ys'], bi['counts'], pp.t)
    return dict(Pc=Pc, Lc=Lc, D=bi['D'], alpha=bi['alpha'], B=bi['B'], n_dup=bi['n_dup'], max_load=bi['max_load'],
                n_bins_redone=bi['n_bins_redone'], t_bin=t1 - t0, t_polys=time.time() - t1)


def masked_layers(Pc, Lc, rng_bytes, t: int):
    """Fold fresh masks r1, r2 (uniform in Z_t^*, one per slot and partition, so independent across the felts of a
    bin) into the coefficients: Sc = r1 * Pc, Qc = r2 * Pc + Lc (mod t)."""
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


def felt_slice(M, f: int, N_B: int):
    """The columns of a (rows, F N_B) layer that belong to ciphertext f (felt f), as a contiguous array."""
    return np.ascontiguousarray(M[:, f * N_B:(f + 1) * N_B])


def _uniform_nonzero(rng, n, t):
    x = rng.uniform_mod_array(t - 1, n) + np.uint64(1)
    return x.astype(np.uint64)


# ------------------------------------------------------------------------------------------- R side
def cuckoo(items_bins, n_bins: int, h: int, rng, max_kicks: int = 2000, stats: dict | None = None):
    """items_bins[i] = list of h candidate bins of item i.  Returns dict bin -> item index, or None.

    Random-walk insertion with at most max_kicks evictions per item.  If the walk exceeds its step limit, the
    partial assignment it holds is completed by augmenting paths (bipartite matching between items and their
    candidate bins), which finds an assignment whenever one exists (Berge); None is returned only when no
    assignment exists.  When the walk succeeds, the result and the use of rng are as without the fallback.
    If `stats` is a dict, stats['fallback'] records whether the fallback ran."""
    table = {}
    if stats is not None:
        stats['fallback'] = False
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
            if stats is not None:
                stats['fallback'] = True
            # cur is homeless and items i+1, ... are not yet inserted; every other item holds a distinct bin
            for u in [cur] + list(range(i + 1, len(items_bins))):
                if not _augment(table, items_bins, u):
                    return None
            return table
    return table


def _augment(table: dict, items_bins, u: int) -> bool:
    """Place item u by an augmenting path: breadth-first search over items, from u, along candidate bins; a bin
    held by item v leads to v.  On success the path is flipped (each item on it moves to the next bin) and True is
    returned; if no free bin is reachable, the table is unchanged and False is returned.  Once an item cannot be
    placed it cannot be placed after later augmentations either, so no assignment of all items exists."""
    parent = {}                       # bin -> (item that reaches it, bin that item currently holds or None)
    holds = {v: b for b, v in table.items()}
    frontier, seen_items = [u], {u}
    while frontier:
        nxt = []
        for v in frontier:
            for b in items_bins[v]:
                if b in parent:
                    continue
                parent[b] = v
                if b not in table:    # free bin: flip the path back to u
                    while True:
                        w = parent[b]
                        prev = holds.get(w) if w != u else None
                        table[b] = w
                        if prev is None:
                            return True
                        b = prev
                w = table[b]
                if w not in seen_items:
                    seen_items.add(w)
                    nxt.append(w)
        frontier = nxt
    return False


def receiver_batches(oprf_outputs, pp: PSIParams, rng):
    """R: split its (padded) items into pp.tables(len(items)) balanced cuckoo tables (sizes differ by at most one,
    each at most cuckoo_load * N_bins) and build each table's query: felt f of the item in bin b goes to slot
    f * N_bins + b; empty bins query DUMMY_Y and are ignored at decoding."""
    n = len(oprf_outputs)
    k, _, _ = pp.tables(n)
    t, nb, F = pp.t, pp.n_bins, pp.felts
    batches = []
    bounds = [n * j // k for j in range(k + 1)]
    for j in range(k):
        chunk = list(range(bounds[j], bounds[j + 1]))
        cand = [bins_of(oprf_outputs[i], pp.h, nb) for i in chunk]
        stats = {}
        tab = cuckoo(cand, nb, pp.h, rng, stats=stats)
        if tab is None:
            raise RuntimeError('cuckoo hashing failed')
        q = np.full(F * nb, DUMMY_Y, dtype=np.uint64)
        slot_item = {}
        for b, li in tab.items():
            i = chunk[li]
            for f, v in enumerate(item_felts(oprf_outputs[i], t, F)):
                q[f * nb + b] = v
            slot_item[b] = i
        batches.append(dict(query=q, slot_item=slot_item, fallback=stats.get('fallback', False)))
    return batches


def window_values(q: np.ndarray, W: int, t: int):
    """w^(2^i) mod t for i < W, slot-wise (exact: values < t < 2^32, so squares fit in 64 bits)."""
    if t >= 2 ** 32:
        raise ValueError('window_values needs t < 2^32')
    out = []
    cur = q.astype(np.uint64) % np.uint64(t)
    for i in range(W):
        out.append(cur.copy())
        cur = (cur * cur) % np.uint64(t)
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


def receiver_decode(S_list, Q_list, slot_item, oprf_outputs, t: int, n_bar_O: int, felts: int = 1,
                    n_bins: int | None = None, stats: dict | None = None):
    """From the decrypted replies of each partition (each of length felts * n_bins; felt f of bin b in slot
    f * n_bins + b), recover {item index: position}.  A partition matches item i only if its membership value is 0
    in every felt slot of i's bin and the label felts decode to one position (label_f - key_f mod t equal for all f);
    a partition that passes the first test but not the second is a cross-item or colliding false match and is not
    a match (counted in stats['felt_inconsistent']).  Returns (found, ambiguous, bad): ambiguous = items matched in
    two partitions with different positions, bad = positions outside [1, n_bar_O]."""
    if n_bins is None:
        n_bins = len(S_list[0]) // felts
    found, ambiguous = {}, set()
    n_incons = n_partial = 0
    keys = {}
    for S, Q in zip(S_list, Q_list):
        for b, i in slot_item.items():
            zero = [int(S[f * n_bins + b]) == 0 for f in range(felts)]
            if not all(zero):
                n_partial += any(zero)
                continue
            if i not in keys:
                keys[i] = [label_key(oprf_outputs[i], t, f) for f in range(felts)]
            pos = {(int(Q[f * n_bins + b]) - keys[i][f]) % t for f in range(felts)}
            if len(pos) != 1:
                n_incons += 1
                continue
            pos = pos.pop()
            if i in found and found[i] != pos:
                ambiguous.add(i)
            found[i] = pos
    bad = {i for i, p in found.items() if not (1 <= p <= n_bar_O)}
    if stats is not None:
        stats['felt_inconsistent'] = stats.get('felt_inconsistent', 0) + n_incons
        stats['felt_partial'] = stats.get('felt_partial', 0) + n_partial
    return found, ambiguous, bad
