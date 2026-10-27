"""Plaintext simulation of what the OPRF-PSI computes on derived keys, and R's post-PSI resolution.

What the PSI outputs.  With keys fed to the OPRF instead of identifiers, the labeled PSI gives R, for each of its
queried keys, whether O inserted the same key and, if so, the label: O's shuffled position tau(j) of the row that
produced it.  F_k is a PRF and the PSI is correct except with probability <= delta_psi + p_pass (paper, Appendix on
PSI), so the output is exactly a dictionary lookup of R's keys in O's key -> position map.  `psi_output` computes
that lookup.  Positions are represented by O's row index; tau is a uniform relabelling and changes nothing below.

Multiplicity on O's side.  The labeled PSI holds one label per item: identical items in one bin go to different
partitions and R then sees one key matched with two positions, which the implementation treats as an abort
(psi.receiver_decode 'ambiguous').  So O must make each key value unique before inserting it.  Policies:
  'unique'  O drops every key value that occurs in more than one of its rows (a uniqueness filter, local to O);
  'multi'   (ablation, NOT supported by the current PSI) the label carries every position with that key value.

Resolution at R (post-processing of R's PSI output, no further interaction):
  candidate pairs (i, t) with the set of key types that matched -> binary pattern gamma;
  Fellegi-Sunter weights by EM on the candidate patterns (2-class latent model, conditional independence);
  accept pairs with posterior >= `p_accept`, then 1:1 by descending weight; a record or position whose best
  weight is tied between two candidates is dropped (ambiguous).
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np


def psi_output(keys_O, keys_R, policy='unique'):
    """keys_*: list (per row) of K-lists of key strings or None.
    Returns cand: dict (i, t) -> bitmask of matched key types; and counters."""
    K = max(len(k) for k in keys_O[:10])
    index = [defaultdict(list) for _ in range(K)]
    n_items = 0
    for t, ks in enumerate(keys_O):
        for k, v in enumerate(ks):
            if v is not None:
                index[k][v].append(t)
                n_items += 1
    n_dropped = 0
    if policy == 'unique':
        for k in range(K):
            dup = [v for v, ts in index[k].items() if len(ts) > 1]
            n_dropped += sum(len(index[k][v]) for v in dup)
            for v in dup:
                del index[k][v]
    cand = defaultdict(int)
    n_queries = n_hits = 0
    for i, ks in enumerate(keys_R):
        for k, v in enumerate(ks):
            if v is None:
                continue
            n_queries += 1
            ts = index[k].get(v)
            if ts:
                n_hits += 1
                for t in ts:
                    cand[(i, t)] |= 1 << k
    return dict(cand), dict(K=K, O_items=n_items, O_items_after_filter=n_items - n_dropped,
                            O_items_dropped_nonunique=n_dropped, R_queries=n_queries, R_key_hits=n_hits)


def u_from_file(tokens):
    """Random-pair agreement rate of each key type / field within one party's own file: u_k = sum_v c_v (c_v - 1) /
    (n (n - 1)) over the values v of type k (c_v = count, n = records with the value present), floored at 1e-9.
    It estimates P(agree | non-match) and needs no data from the other party (R computes it on its cohort, or both
    on public reference data)."""
    from collections import Counter
    K = len(tokens[0])
    u = []
    for k in range(K):
        c = Counter(ks[k] for ks in tokens if ks[k] is not None)
        n = sum(c.values())
        pairs = sum(v * (v - 1) for v in c.values())
        u.append(max(pairs / max(n * (n - 1), 1), 1e-9))
    return np.array(u)


def em_fs(G, iters=200, tol=1e-9, m0=0.9, u0=None, p0=0.5, u_fixed=None):
    """EM for the 2-class Fellegi-Sunter model on binary patterns G (n, K).  Returns m, u, prior p.
    u_fixed: keep u at these values (estimated from random pairs, u_from_file) and fit only m and p; on candidate
    sets produced by blocking, which are nearly all matches, a free u otherwise absorbs the true matches with
    unusual patterns."""
    G = np.asarray(G, float)
    n, K = G.shape
    m = np.full(K, m0)
    u = np.full(K, 0.1) if u0 is None else np.asarray(u0, float)
    if u_fixed is not None:
        u = np.clip(np.asarray(u_fixed, float), 1e-9, 1 - 1e-9)
    p = p0
    ll_old = -np.inf
    for _ in range(iters):
        lm = G @ np.log(m) + (1 - G) @ np.log(1 - m) + np.log(p)
        lu = G @ np.log(u) + (1 - G) @ np.log(1 - u) + np.log(1 - p)
        mx = np.maximum(lm, lu)
        ll = float(np.sum(mx + np.log(np.exp(lm - mx) + np.exp(lu - mx))))
        g = 1.0 / (1.0 + np.exp(lu - lm))
        p = float(np.clip(g.mean(), 1e-6, 1 - 1e-6))
        m = np.clip((g @ G) / max(g.sum(), 1e-12), 1e-4, 1 - 1e-4)
        if u_fixed is None:
            u = np.clip(((1 - g) @ G) / max((1 - g).sum(), 1e-12), 1e-4, 1 - 1e-4)
        if ll - ll_old < tol * abs(ll):
            break
        ll_old = ll
    if u_fixed is None and m.mean() < u.mean():   # label switching: the match class has the higher agreement
        m, u, p = u, m, 1 - p
    return m, u, p


def fs_weights(G, m, u):
    G = np.asarray(G, float)
    return G @ np.log2(m / u) + (1 - G) @ np.log2((1 - m) / (1 - u))


def resolve(pairs, w, post, p_accept=0.5, rule='greedy'):
    """pairs: (n, 2) int array of (i, t); w: weights; post: posterior match probabilities.
    rule 'greedy': accept by descending weight, 1:1, drop records/positions whose top weight is tied.
    rule 'unambiguous': keep only pairs whose record and position have exactly one accepted candidate."""
    ok = post >= p_accept
    P, W = pairs[ok], w[ok]
    if len(P) == 0:
        return np.zeros((0, 2), int)
    if rule == 'unambiguous':
        ci = np.bincount(P[:, 0]); ct = np.bincount(P[:, 1])
        keep = (ci[P[:, 0]] == 1) & (ct[P[:, 1]] == 1)
        return P[keep]
    # ties at the top: for each record (and each position), if its two largest weights are equal, drop it
    best_i, best_t = {}, {}
    tied_i, tied_t = set(), set()
    for (i, t), x in zip(P.tolist(), W.tolist()):
        for d, tied, key in ((best_i, tied_i, i), (best_t, tied_t, t)):
            b = d.get(key)
            if b is None or x > b + 1e-12:
                d[key] = x
                tied.discard(key)
            elif abs(x - b) <= 1e-12:
                tied.add(key)
    order = np.argsort(-W, kind='stable')
    used_i, used_t, out = set(), set(), []
    for k in order:
        i, t = int(P[k, 0]), int(P[k, 1])
        if i in used_i or t in used_t or i in tied_i or t in tied_t:
            continue
        used_i.add(i); used_t.add(t); out.append((i, t))
    return np.array(out, int).reshape(-1, 2)


def link_keys(keys_O, keys_R, policy='unique', p_accept=0.5, rule='greedy', u_fixed='file', min_keys=1, cache=None):
    """u_fixed='file': u from R's own file (u_from_file); None: free EM.  min_keys: a candidate needs at least this
    many matched key types.  cache: dict reused across calls on the same files (PSI output)."""
    if cache is not None and ('psi', policy) in cache:
        cand, cnt = cache[('psi', policy)]
    else:
        cand, cnt = psi_output(keys_O, keys_R, policy)
        if cache is not None:
            cache[('psi', policy)] = (cand, cnt)
    if not cand:
        return np.zeros((0, 2), int), cnt, {}
    pairs = np.array(list(cand.keys()), int)
    masks = np.array(list(cand.values()), np.int64)
    K = cnt['K']
    G = ((masks[:, None] >> np.arange(K)[None, :]) & 1).astype(float)
    if isinstance(u_fixed, str) and u_fixed == 'file':
        if cache is not None and 'u' in cache:
            uf = cache['u']
        else:
            uf = u_from_file(keys_R)
            if cache is not None:
                cache['u'] = uf
    else:
        uf = u_fixed
    m, u, p = em_fs(G, u_fixed=uf)
    w = fs_weights(G, m, u)
    lr = w * np.log(2) + np.log(p / (1 - p))
    post = 1 / (1 + np.exp(-np.clip(lr, -700, 700)))
    post = np.where(G.sum(1) >= min_keys, post, 0.0)
    acc = resolve(pairs, w, post, max(p_accept, 1e-300), rule)
    info = dict(m=m.tolist(), u=u.tolist(), prior=p, candidate_pairs=int(len(pairs)))
    return acc, cnt, dict(info, pairs=pairs, G=G, post=post)


def link_fields(fields_O, fields_R, cand_pairs, p_accept=0.5, rule='greedy', u_fixed=None):
    """FS variant: candidates from a blocking PSI (cand_pairs), then per-field agreement vectors (what R would learn
    from per-field OPRF tokens carried in the label), EM weights, resolution."""
    from keys import agreement
    G = np.array([agreement(fields_R[i], fields_O[t]) for i, t in cand_pairs], float)
    m, u, p = em_fs(G, u_fixed=u_fixed)
    w = fs_weights(G, m, u)
    lr = w * np.log(2) + np.log(p / (1 - p))
    post = 1 / (1 + np.exp(-np.clip(lr, -700, 700)))
    acc = resolve(cand_pairs, w, post, p_accept, rule)
    return acc, dict(m=m.tolist(), u=u.tolist(), prior=p, candidate_pairs=int(len(cand_pairs)))


def evaluate(acc, true_pos, n_overlap, cohort_true_rows=None):
    """Precision / recall of accepted pairs against the truth (true_pos[i] = O row of R record i, -1 if none)."""
    if len(acc) == 0:
        return dict(accepted=0, correct=0, precision=float('nan'), recall=0.0, false_rate=float('nan'))
    corr = true_pos[acc[:, 0]] == acc[:, 1]
    out = dict(accepted=int(len(acc)), correct=int(corr.sum()), precision=float(corr.mean()),
               recall=float(corr.sum() / n_overlap), false_rate=float(1 - corr.mean()),
               false_with_outsider=int(((true_pos[acc[:, 0]] < 0)).sum()),
               false_with_insider=int(((true_pos[acc[:, 0]] >= 0) & ~corr).sum()))
    return out


def leakage(pairs, true_pos):
    """What R learns from the PSI beyond (I, pi|_I) of exact linkage: candidate positions that are not the querying
    record's true row, and how many of them are rows of people outside the cohort (non-cohort individuals whose
    existence and key values R learns)."""
    if len(pairs) == 0:
        return {}
    true_rows = set(true_pos[true_pos >= 0].tolist())
    wrong = true_pos[pairs[:, 0]] != pairs[:, 1]
    pos_all = set(pairs[:, 1].tolist())
    pos_wrong = set(pairs[wrong, 1].tolist())
    noncohort = pos_wrong - true_rows
    per_rec = np.bincount(pairs[:, 0])
    per_pos = np.bincount(pairs[:, 1])
    return dict(candidate_pairs=int(len(pairs)), wrong_pairs=int(wrong.sum()),
                positions_revealed=len(pos_all), positions_not_true_match=len(pos_wrong),
                positions_of_noncohort_people=len(noncohort),
                records_with_multiple_candidates=int((per_rec > 1).sum()),
                positions_with_multiple_candidates=int((per_pos > 1).sum()),
                max_candidates_per_record=int(per_rec.max()), max_candidates_per_position=int(per_pos.max()))
