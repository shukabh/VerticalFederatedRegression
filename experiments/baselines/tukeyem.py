"""TukeyEM (Amin, Joseph, Ribero and Vassilvitskii, ICLR 2023, "Easy differentially private linear regression"),
reimplemented.

Reference code: github.com/google-research/google-research, dp_regression/tukey.py (repository commit
ba728e288c51126eac25e316b11dfee94d68c301; tukey.py unchanged since the 2022 release apart from headers).  The mechanism:

  1. shuffle the n rows and split them into m disjoint groups of floor(n/m) rows (the remainder is dropped); fit OLS
     (no added intercept; the design carries its own) on each group: m models in R^p;
  2. per coordinate, sort the m model values (after a 1e-10 uniform perturbation that breaks ties); the volume of the
     region of approximate Tukey depth >= i is the product over coordinates of the width between the i-th smallest
     and the i-th largest value;
  3. propose-test-release with eps/2: t = floor(floor(m/2)/2); k^ = distance_to_unsafety(log volumes, eps/2, delta, t)
     (a binary search for a lower bound on the Hamming distance, in models, to a dataset on which the restricted
     exponential mechanism is unsafe); pass iff k^ + Lap(1/(eps/2)) > log(1/(2 delta)) / (eps/2); on failure output 0;
  4. otherwise sample a depth from the exponential mechanism with eps/2 restricted to depth >= t (probability of depth
     d proportional to the volume of exact depth d times exp((eps/2) d)), then a point uniformly from the region of
     that exact depth.
The output is (eps, delta)-DP for datasets that differ in one record: the record changes one of the m models, and the
PTR distance is a Hamming distance between the multisets of models.  It needs no bound on the features or the labels.

Two differences from the reference code, deliberately (both reproducible with flags, and the validation runs both):
  * PTR noise: their dp_tukey draws np.random.laplace(scale=split_epsilon), i.e. Laplace with scale eps/2, whereas the
    test against the threshold log(1/(2 delta))/(eps/2) needs scale 1/(eps/2) (the distance has sensitivity 1).  We use
    1/(eps/2) (laplace_scale='inverse', the default; 'reference' reproduces their code).
  * Depth: their restricted_racing_sample_depth returns a depth relative to projections[:, t-1:-(t-1)], but dp_tukey
    passes it to sample_exact with projections[:, t:-t], so the point is drawn one depth level deeper than the level
    the exponential mechanism chose (and, at the deepest level, from an empty region; numpy's legacy uniform then
    returns a point outside it instead of failing).  We sample from the chosen level (depth_mode='chosen', the default;
    'reference' reproduces their code).
  * distance_to_unsafety: when the set of depths compared with depth t - k - 1 is empty (m a multiple of 4 and
    k = t - 2), the reference code fails (np.min of an empty array); we treat the check as passed, since no volume ratio
    can be violated.

Everything else follows their code line by line (distance_to_unsafety is a direct port, recursion included), but
vectorized: the m OLS fits are one batched solve (lstsq for a singular group, as sklearn's LinearRegression does).
"""
from __future__ import annotations

import math

import numpy as np


def multiple_regressions(X, y, m, rng):
    """m OLS models on a random partition (tukey.multiple_regressions).  Raises if floor(n/m) < p."""
    n, p = X.shape
    b = n // m
    if b < p:
        raise RuntimeError(f'{m} models require {m * p} points, but there are only {n}')
    order = rng.permutation(n)
    Xb = X[order[:b * m]].reshape(m, b, p)
    yb = y[order[:b * m]].reshape(m, b)
    G = np.einsum('mbi,mbj->mij', Xb, Xb)
    c = np.einsum('mbi,mb->mi', Xb, yb)
    models = np.empty((m, p))
    cond_ok = np.linalg.cond(G) < 1e12
    if cond_ok.any():
        models[cond_ok] = np.linalg.solve(G[cond_ok], c[cond_ok][..., None])[..., 0]
    for i in np.where(~cond_ok)[0]:                       # rank-deficient group: minimum-norm least squares
        models[i] = np.linalg.lstsq(Xb[i], yb[i], rcond=None)[0]
    return models


def perturb_and_sort(models, rng):
    """(p, m) matrix: each coordinate's m values plus 1e-10 U(0,1), sorted."""
    M = models.T
    return np.sort(M + 1e-10 * rng.random(M.shape), axis=1)


def log_measure_geq_all_depths(proj):
    max_depth = int(np.ceil(proj.shape[1] / 2))
    diff = proj[:, ::-1] - proj
    with np.errstate(divide='ignore'):
        return np.sum(np.log(diff[:, :max_depth]), axis=0)


def distance_to_unsafety(log_volumes, epsilon, delta, t, k_low, k_high):
    """Direct port of tukey.distance_to_unsafety (binary search, recursive)."""
    k = int((k_low + k_high) / 2)
    log_vol_ytk = log_volumes[t - k - 2]
    log_vols_ytk_gs = log_volumes[t + k + 1:len(log_volumes) - 1]
    log_epsilon_terms = (epsilon / 2) * np.arange(1, len(log_vols_ytk_gs) + 1)
    log_threshold = np.log(delta / (8 * np.exp(epsilon)))
    # an empty comparison set (no depth beyond t + k + 1; it happens when m is a multiple of 4 and k = t - 2) has
    # nothing to violate: treated as passing the check.  The reference code raises on np.min of an empty array here.
    gaps = (log_vol_ytk - log_vols_ytk_gs) - log_epsilon_terms
    if gaps.size and np.min(gaps) <= log_threshold:
        if k_low >= k_high - 1:
            return k_high
        new_k_low = k_low + int((k_high - k_low) / 2)
        return distance_to_unsafety(log_volumes, epsilon, delta, t, new_k_low, k_high)
    if k_high > k_low + 1:
        new_k_high = k_high - int((k_high - k_low) / 2)
        return distance_to_unsafety(log_volumes, epsilon, delta, t, k_low, new_k_high)
    return k_low


def racing_sample(log_terms, rng):
    return int(np.argmin(np.log(np.log(1.0 / rng.uniform(size=log_terms.shape))) - log_terms))


def restricted_depth_sample(proj, epsilon, restricted_depth, rng):
    proj = proj[:, (restricted_depth - 1):-(restricted_depth - 1)]
    at_least = np.exp(log_measure_geq_all_depths(proj))
    exact = at_least.copy()
    exact[:-1] = at_least[:-1] - at_least[1:]
    depths = np.arange(restricted_depth, restricted_depth + len(at_least))
    with np.errstate(divide='ignore'):
        log_terms = np.log(exact) + epsilon * depths
    return 1 + racing_sample(log_terms, rng)


def _u(rng, lo, hi):
    """U(lo, hi) as lo + (hi - lo) U(0, 1): the formula of numpy's legacy uniform, which the reference code calls
    (it also accepts hi < lo, which numpy's Generator.uniform refuses)."""
    return lo + (hi - lo) * rng.uniform()


def _sample_geq_1d(depth, proj, rng):
    return _u(rng, proj[depth - 1], proj[-depth])


def _sample_exact_1d(depth, proj, rng):
    left_low, left_high = proj[depth - 1], proj[depth]
    right_low, right_high = proj[-(depth + 1)], proj[-depth]
    ml, mr = left_high - left_low, right_high - right_low
    if rng.uniform() < ml / (ml + mr):
        return left_low + rng.uniform() * ml
    return right_low + rng.uniform() * mr


def sample_exact(depth, proj, rng):
    """A point of exactly the given depth in proj (tukey.sample_exact).  Widths of empty regions (negative, once the
    depth reaches the middle) are set to zero, which the reference code leaves to log() of a negative number."""
    d = proj.shape[0]
    with np.errstate(divide='ignore', invalid='ignore'):
        w_gt = np.maximum(proj[:, -(depth + 1)] - proj[:, depth], 0.0)
        w_ge = np.maximum(proj[:, -depth] - proj[:, depth - 1], 0.0)
        lg_gt = np.log(w_gt)
        lg_exact = np.log(np.maximum(w_ge - w_gt, 0.0))
    left = np.zeros(d)
    left[1:] = np.cumsum(lg_gt)[:-1]
    right = np.zeros(d)
    right[:-1] = (np.cumsum(lg_gt[::-1])[::-1])[1:]
    j = racing_sample(lg_exact + left + right, rng)
    out = np.zeros(d)
    for i in range(j):
        out[i] = _sample_geq_1d(depth + 1, proj[i], rng)
    out[j] = _sample_exact_1d(depth, proj[j], rng)
    for i in range(j + 1, d):
        out[i] = _sample_geq_1d(depth, proj[i], rng)
    return out


def ptr_threshold(epsilon, delta):
    return math.log(1 / (2 * delta)) / (epsilon / 2)


def dp_tukey(models, epsilon, delta, rng, laplace_scale='inverse', depth_mode='chosen'):
    """(epsilon, delta)-DP TukeyEM from the m models.  Returns (theta, info); theta = 0 if the PTR test fails.
    laplace_scale: 'inverse' (PTR noise Lap(1/(eps/2)), default) or 'reference' (Lap(eps/2), as the reference code).
    depth_mode: 'chosen' (sample from the depth the exponential mechanism chose, default) or 'reference' (the reference
    code's sample_exact(depth, projections[:, t:-t]), one level deeper than the chosen one: the depth it returns is
    relative to projections[:, t-1:-(t-1)])."""
    m, p = models.shape
    proj = perturb_and_sort(models, rng)
    max_depth = int(m / 2)
    log_volumes = log_measure_geq_all_depths(proj)
    t = int(max_depth / 2)
    eps2 = epsilon / 2
    info = dict(m=m, t=t, threshold=ptr_threshold(epsilon, delta))
    if t < 2:                                             # the reference code would index out of range
        info.update(distance=None, passed=False, reason='t < 2')
        return np.zeros(p), info
    dist = distance_to_unsafety(log_volumes, eps2, delta, t, -1, t - 1)
    scale = 1.0 / eps2 if laplace_scale == 'inverse' else eps2
    passed = dist + rng.laplace(scale=scale) > info['threshold']
    info.update(distance=int(dist), passed=bool(passed))
    if not passed:
        return np.zeros(p), info
    depth = restricted_depth_sample(proj, eps2, t, rng)          # relative to proj[:, t-1:-(t-1)]
    info['depth'] = int(depth + t - 1)
    if depth_mode == 'reference':
        return sample_exact(depth, proj[:, t:-t], rng), info
    return sample_exact(depth + t - 1, proj, rng), info


def tukey_em(X, y, epsilon, delta, m, rng, laplace_scale='inverse', depth_mode='chosen'):
    models = multiple_regressions(X, y, m, rng)
    return dp_tukey(models, epsilon, delta, rng, laplace_scale, depth_mode)
