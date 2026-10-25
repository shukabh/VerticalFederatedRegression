"""Replace-one sensitivity of the FULL-RECORD release (baseline 1: vertical full-record SSP).

Baseline 1 aggregates the whole Gram matrix securely (Karr et al., Hall et al., Gascon et al.) and then adds noise to
EVERY released entry, R's own block A included, under replace-one adjacency of a whole record (x_R, x_O, y).  This
module derives the l2 sensitivity of that release; it is the analogue of Proposition prop:sens (implementation/vfr/dp.py,
delta_rep) with R's row allowed to change too.

Record.  w = (u, b, y): u in R^{d_u} are R's features without the intercept (||u||^2 <= R_u), b = x_O in R^{d_O}
(||b||^2 <= o = B_O^2), y with y^2 <= w = B_y^2.  With an intercept (iota = 1) R's row is (1, u) and R_u = B_R^2 - 1
(B_R covers the intercept, as in v3.1); without one (Wang's preprocessing, arm (c)) R's row is u and R_u = B_R^2.

Release.  Stage 2 releases triu of the second-moment matrix of (1, u, b, y) without its constant (0, 0) entry n:
the first moments iota (sum u, sum b, sum y), and triu(sum w w^T) = (triu A_u, C_ub, triu B, c_u, c_O, y'y).  In
separate y'y mode the y'y entry is dropped (it gets its own release, sensitivity w, as in v3.1).  Stage 1 releases the
same without anything that involves y.  Every entry gets noise of the same s.d.

Sensitivity.  Replacing w by w' changes the release by iota (u - u', b - b', y - y') and by triu(w w^T - w' w'^T).
Write n_k = ||w_k||^2, n'_k = ||w'_k||^2 and g_k = <w_k, w'_k> per block k in {u, b, y}.  For any symmetric block,
||triu(M)||^2 <= ||M||_F^2, so

  D^2 <= U = iota sum_k (n_k + n'_k - 2 g_k) + sum_k (n_k^2 + n'_k^2 - 2 g_k^2)
             + sum_{k<l} (n_k n_l + n'_k n'_l - 2 g_k g_l),                                     (*)

with |g_k| <= sqrt(n_k n'_k).  (*) is exact for the scalar block y (g_y = y y') and attained for a block of dimension
>= 2: in two coordinates, for any norms and inner product, x = sqrt(n)(cos a, sin a), x' = sqrt(n')(cos b, sin b) can
be chosen with x x^T - x' x'^T diagonal (tan 2b = -n sin 2t / (n cos 2t - n'), t = a - b fixed by the inner product),
and then triu loses nothing.  For a block of dimension 1, g_k = +-sqrt(n_k n'_k) is imposed, which is again exact.  So
sup D^2 = sup U over the norms, y, y' and the admissible g.  U is a concave quadratic in (g_u, g_b) for fixed norms and
(y, y'), maximized exactly.  For a block of dimension >= 2 the admissible set of g grows with the norms and every other
term has a nonnegative coefficient, so the sup sits at both norms on the bound (_sup_U_fast); what remains, (y, y') and
the norms of 1-dimensional blocks, is a dense grid refined by bounded local search.  delta_full(..., slow=True) searches
all six variables instead (a check of the reduction).  The value is rounded up by a relative 1e-12, as delta_rep is.

Checks (python experiments/baselines/sens_full.py check): (i) with R's row held fixed (u = u', n_u = R_u) the same
maximization reproduces dp.delta_rep and dp.delta_G; (ii) a direct maximization of D^2 over actual vectors (d_u, d_O in
{1, 2, 3}) from many random starts never exceeds U and attains it; (iii) the closed form at unit bounds without an
intercept.
"""
from __future__ import annotations

import itertools
import json
import math
import os
import sys
from functools import lru_cache

import numpy as np
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, '..', '..'))
SAFEGUARD = 1e-12


# ------------------------------------------------------------------------------------------- the bound U
def _inner_max(nu, nu2, nb, nb2, y, y2, iota, with_y, with_yy, du1, db1):
    """max over admissible (g_u, g_b) of U, vectorized over arrays of (nu, nu2, nb, nb2, y, y2).
    U = const + L (g_u + g_b) - 2 g_u^2 - 2 g_b^2 - 2 g_u g_b, with L = -2 (iota + y y') (y terms only if with_y)."""
    yy_ = y * y2 if with_y else 0.0 * y
    const = (iota * (nu + nu2 + nb + nb2) + nu ** 2 + nu2 ** 2 + nb ** 2 + nb2 ** 2 + nu * nb + nu2 * nb2)
    if with_y:
        const = const + iota * (y - y2) ** 2 + nu * y ** 2 + nu2 * y2 ** 2 + nb * y ** 2 + nb2 * y2 ** 2
        if with_yy:
            const = const + (y ** 2 - y2 ** 2) ** 2
    L = -2.0 * (iota + yy_)
    hu, hb = np.sqrt(nu * nu2), np.sqrt(nb * nb2)

    def Q(gu, gb):
        return const + L * (gu + gb) - 2 * gu ** 2 - 2 * gb ** 2 - 2 * gu * gb

    cands = []
    gu_set = [-hu, hu] if du1 else None
    gb_set = [-hb, hb] if db1 else None
    if du1 and db1:
        for a in gu_set:
            for b in gb_set:
                cands.append(Q(a, b))
    elif du1:                                   # g_u in {+-hu}; g_b continuous: argmax = clip((L - 2 g_u) / 4)
        for a in gu_set:
            b = np.clip((L - 2 * a) / 4, -hb, hb)
            cands.append(Q(a, b))
    elif db1:
        for b in gb_set:
            a = np.clip((L - 2 * b) / 4, -hu, hu)
            cands.append(Q(a, b))
    else:                                       # box: interior stationary point, else the best edge
        a0 = b0 = L / 6
        inside = (np.abs(a0) <= hu) & (np.abs(b0) <= hb)
        cands.append(np.where(inside, Q(a0, b0), -np.inf))
        for a in (-hu, hu):
            b = np.clip((L - 2 * a) / 4, -hb, hb)
            cands.append(Q(a, b))
        for b in (-hb, hb):
            a = np.clip((L - 2 * b) / 4, -hu, hu)
            cands.append(Q(a, b))
    return np.max(np.stack(cands), axis=0)


def _sup_U(Ru, o, By, iota, with_y, with_yy, du1, db1, fix_u=False, grid=9, ygrid=41, refine=24):
    """sup of U.  fix_u: R's row held fixed (u = u' with ||u||^2 = R_u), the v3.1 adjacency (check (i))."""
    gn_u = np.linspace(0, Ru, grid)
    gn_b = np.linspace(0, o, grid)
    gy = np.linspace(-By, By, ygrid) if with_y else np.zeros(1)

    def U_at(nu, nu2, nb, nb2, y, y2):
        if fix_u:                               # g_u = n_u = R_u: only g_b free
            gu = np.full_like(nb, Ru)
            hb = np.sqrt(nb * nb2)
            yy_ = y * y2 if with_y else 0 * y
            L = -2.0 * (iota + yy_)
            const = (iota * (nb + nb2) + nb ** 2 + nb2 ** 2 + Ru * nb + Ru * nb2 - 0.0)
            if with_y:
                const = const + iota * (y - y2) ** 2 + Ru * (y - y2) ** 2 + nb * y ** 2 + nb2 * y2 ** 2
                if with_yy:
                    const = const + (y ** 2 - y2 ** 2) ** 2
            # U = const - 2 g_b^2 + g_b (L - 2 R_u) ... the u-b cross term is R_u n_b + R_u n_b' - 2 R_u g_b
            vals = []
            if db1:
                for b in (-hb, hb):
                    vals.append(const + L * b - 2 * b ** 2 - 2 * Ru * b)
            else:
                b = np.clip((L - 2 * Ru) / 4, -hb, hb)
                vals.append(const + L * b - 2 * b ** 2 - 2 * Ru * b)
            del gu
            return np.max(np.stack(vals), axis=0)
        return _inner_max(nu, nu2, nb, nb2, y, y2, iota, with_y, with_yy, du1, db1)

    if fix_u:
        mesh = np.meshgrid(gn_b, gn_b, gy, gy, indexing='ij')
        nb, nb2, y, y2 = [m.ravel() for m in mesh]
        vals = U_at(None, None, nb, nb2, y, y2)
        X = np.stack([nb, nb2, y, y2], 1)
    else:
        rows, vals_l = [], []
        for nu in gn_u:
            mesh = np.meshgrid(gn_u, gn_b, gn_b, gy, gy, indexing='ij')
            nu2, nb, nb2, y, y2 = [m.ravel() for m in mesh]
            nu_ = np.full_like(nu2, nu)
            vals_l.append(U_at(nu_, nu2, nb, nb2, y, y2))
            rows.append(np.stack([nu_, nu2, nb, nb2, y, y2], 1))
        vals = np.concatenate(vals_l)
        X = np.concatenate(rows)
    best = float(vals.max())
    lo = np.array(([0, 0, 0, 0] if not fix_u else [0, 0]) + [-By, -By])
    hi = np.array(([Ru, Ru, o, o] if not fix_u else [o, o]) + [By, By])
    if not with_y:
        lo[-2:] = 0.0
        hi[-2:] = 0.0
    order = np.argsort(-vals)
    seen = []
    for i in order:
        x0 = X[i]
        if any(np.allclose(x0, s) for s in seen):
            continue
        seen.append(x0)
        if len(seen) > refine:
            break
        f = (lambda z: -float(U_at(*[np.atleast_1d(v) for v in z])[0])) if not fix_u else \
            (lambda z: -float(U_at(None, None, *[np.atleast_1d(v) for v in z])[0]))
        bnds = list(zip(lo, hi))
        for meth in ('L-BFGS-B', 'Powell'):
            r = minimize(f, x0, method=meth, bounds=bnds, options=dict(maxiter=4000))
            best = max(best, -float(r.fun))
    return best


def _sup_U_fast(Ru, o, By, iota, with_y, with_yy, du1, db1, ygrid=None, ngrid=21, refine=None):
    """sup of U using its monotonicity: for a block of dimension >= 2 the admissible set of g grows with the norms
    and every other term of U has a nonnegative coefficient, so U is nondecreasing in n_k and n'_k and the sup sits
    at n_k = n'_k = bound.  Only the norms of 1-dimensional blocks and (y, y') are searched (dense grid, then local
    refinement)."""
    one_d = du1 or db1
    ygrid = ygrid or (61 if one_d else 401)
    refine = refine or (16 if one_d else 8)
    gy = np.linspace(-By, By, ygrid) if with_y else np.zeros(1)
    gnu = np.linspace(0, Ru, ngrid) if du1 else np.array([Ru])
    gnb = np.linspace(0, o, ngrid) if db1 else np.array([o])
    mesh = np.meshgrid(gnu, gnu, gnb, gnb, gy, gy, indexing='ij')
    if not du1:
        mesh[0][...] = Ru; mesh[1][...] = Ru
    if not db1:
        mesh[2][...] = o; mesh[3][...] = o
    X = np.stack([m.ravel() for m in mesh], 1)
    vals = _inner_max(*X.T, iota, with_y, with_yy, du1, db1)
    best = float(vals.max())
    free = [du1, du1, db1, db1, with_y, with_y]
    lo = np.array([0, 0, 0, 0, -By, -By], float)
    hi = np.array([Ru, Ru, o, o, By, By], float)
    idx = [i for i in range(6) if free[i]]
    if not idx:
        return best
    order = np.argsort(-vals)[:refine * 20]
    seen = []
    for i in order:
        x0 = X[i]
        if any(np.allclose(x0, s_, atol=1e-9) for s_ in seen):
            continue
        seen.append(x0)
        if len(seen) > refine:
            break

        def f(zf, x0=x0):
            z = x0.copy()
            z[idx] = zf
            return -float(_inner_max(*[np.atleast_1d(v) for v in z], iota, with_y, with_yy, du1, db1)[0])
        for meth in ('L-BFGS-B', 'Powell'):
            r = minimize(f, x0[idx], method=meth, bounds=list(zip(lo[idx], hi[idx])), options=dict(maxiter=4000))
            best = max(best, -float(r.fun))
    return best


@lru_cache(maxsize=None)
def delta_full(B_R, B_O, B_y, intercept, d_u, d_O, slow=False):
    """(Delta, Delta_noyy, Delta_G) of the full-record release: stage 2 with and without y'y, and stage 1.
    B_R covers the intercept when intercept=True (as in v3.1).  Rounded up by a relative 1e-12.
    slow=True searches all six variables (no monotonicity argument), as a check."""
    iota = 1.0 if intercept else 0.0
    Ru = B_R ** 2 - (1.0 if intercept else 0.0)
    o = B_O ** 2
    du1, db1 = d_u == 1, d_O == 1
    sup = _sup_U if slow else _sup_U_fast
    D2 = sup(Ru, o, B_y, iota, True, True, du1, db1)
    D2n = sup(Ru, o, B_y, iota, True, False, du1, db1)
    DG2 = sup(Ru, o, B_y, iota, False, False, du1, db1)
    up = 1 + SAFEGUARD
    return math.sqrt(D2) * up, math.sqrt(D2n) * up, math.sqrt(DG2) * up


def delta_fixed_R(B_R, B_O, B_y, intercept, d_O):
    """The same maximization with R's row fixed (the v3.1 adjacency): must equal dp.delta_rep / dp.delta_G."""
    iota = 1.0 if intercept else 0.0
    Ru = B_R ** 2 - (1.0 if intercept else 0.0)
    o = B_O ** 2
    D2 = _sup_U(Ru, o, B_y, iota, True, True, False, d_O == 1, fix_u=True)
    D2n = _sup_U(Ru, o, B_y, iota, True, False, False, d_O == 1, fix_u=True)
    DG2 = _sup_U(Ru, o, B_y, iota, False, False, False, d_O == 1, fix_u=True)
    return math.sqrt(D2), math.sqrt(D2n), math.sqrt(DG2)


# ------------------------------------------------------------------------------------------- direct check
def release_change(u, b, y, u2, b2, y2, intercept, with_y=True, with_yy=True):
    """The exact change of the full-record release (vector) for actual records."""
    w = np.concatenate([u, b, [y]]) if with_y else np.concatenate([u, b])
    w2 = np.concatenate([u2, b2, [y2]]) if with_y else np.concatenate([u2, b2])
    M = np.outer(w, w) - np.outer(w2, w2)
    iu = np.triu_indices(len(w))
    v = M[iu]
    if with_y and not with_yy:                        # drop the y'y entry (last diagonal)
        keep = ~((iu[0] == len(w) - 1) & (iu[1] == len(w) - 1))
        v = v[keep]
    if intercept:
        v = np.concatenate([v, w - w2])
    return v


def direct_max(B_R, B_O, B_y, intercept, d_u, d_O, with_y, with_yy, starts=300, seed=0):
    """Maximize ||release change|| over actual vectors by bounded local search from random starts."""
    rng = np.random.default_rng(seed)
    Ru = B_R ** 2 - (1.0 if intercept else 0.0)
    ru, ro = math.sqrt(Ru), B_O

    def unpack(z):
        i = 0
        out = []
        for d, r in ((d_u, ru), (d_O, ro), (d_u, ru), (d_O, ro)):
            v = z[i:i + d]
            i += d
            nv = np.linalg.norm(v)
            out.append(v * min(1.0, r / nv) if nv > 0 else v)
        y, y2 = np.clip(z[i], -B_y, B_y), np.clip(z[i + 1], -B_y, B_y)
        return out[0], out[1], y, out[2], out[3], y2

    def f(z):
        u, b, y, u2, b2, y2 = unpack(z)
        v = release_change(u, b, y, u2, b2, y2, intercept, with_y, with_yy)
        return -float(v @ v)

    best = 0.0
    dim = 2 * (d_u + d_O) + 2
    for _ in range(starts):
        z0 = np.concatenate([rng.standard_normal(d_u) * ru, rng.standard_normal(d_O) * ro,
                             rng.standard_normal(d_u) * ru, rng.standard_normal(d_O) * ro,
                             rng.uniform(-B_y, B_y, 2)])
        r = minimize(f, z0, method='Powell', options=dict(maxiter=20000, xtol=1e-10, ftol=1e-14))
        best = max(best, -r.fun)
        assert len(z0) == dim
    return best


def check(out_path=None):
    sys.path.insert(0, os.path.join(ROOT, 'implementation'))
    from vfr import dp
    res = dict(fixed_R_vs_delta_rep=[], direct_vs_U=[], closed_form_unit=None)
    # (i) R fixed reproduces v3.1's sensitivities, every regime
    for (BR, BO, By) in [(1, 1, 1), (math.sqrt(2), 1, 2.89), (math.sqrt(2), 1, 1.2), (5.59, 6.24, 2.89), (1, 1, 5),
                         (math.sqrt(2), 1, 4.0), (3.0, 1.0, 0.5)]:
        D, Dn = dp.delta_rep(BR, BO, By)
        DG = dp.delta_G(BR, BO)
        f = delta_fixed_R(BR, BO, By, True, 3)
        res['fixed_R_vs_delta_rep'].append(dict(bounds=[BR, BO, By], delta_rep=[D, Dn, DG], ours=list(f),
                                                max_rel_diff=max(abs(f[0] / D - 1), abs(f[1] / Dn - 1), abs(f[2] / DG - 1))))
    # (ii) direct maximization over vectors never exceeds U and attains it
    for (BR, BO, By, ic) in [(1, 1, 1, False), (math.sqrt(2), 1, 2.0, True), (math.sqrt(2), 1, 1.0, True),
                             (2.0, 1.5, 3.0, True)]:
        for du, dO in [(1, 2), (2, 2), (3, 3), (1, 3)]:
            D, Dn, DG = delta_full(BR, BO, By, ic, du, dO)
            got = [math.sqrt(direct_max(BR, BO, By, ic, du, dO, True, True, starts=120, seed=1)),
                   math.sqrt(direct_max(BR, BO, By, ic, du, dO, True, False, starts=120, seed=2)),
                   math.sqrt(direct_max(BR, BO, By, ic, du, dO, False, False, starts=120, seed=3))]
            res['direct_vs_U'].append(dict(bounds=[BR, BO, By], intercept=ic, d_u=du, d_O=dO, U=[D, Dn, DG],
                                           direct=got, ratio=[g / u for g, u in zip(got, (D, Dn, DG))]))
            print(res['direct_vs_U'][-1], flush=True)
    # (iii) unit bounds, no intercept, d >= 2: Delta^2 = 32/3 at c_u = c_b = 1/3, y' = -y (derived by hand)
    D, Dn, DG = delta_full(1.0, 1.0, 1.0, False, 3, 3)
    res['closed_form_unit'] = dict(Delta=D, Delta_sq=D * D, expected_sq=32 / 3, Delta_G=DG, Delta_G_sq=DG * DG,
                                   v31_Delta_sq=dp.delta_rep(1.0, 1.0, 1.0)[0] ** 2, v31_Delta_G=dp.delta_G(1.0, 1.0))
    if out_path:
        with open(out_path, 'w') as fh:
            json.dump(res, fh, indent=1)
    return res


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'check':
        r = check(os.path.join(ROOT, 'reports', 'baselines', 'sens_full_check.json'))
        print(json.dumps(r['fixed_R_vs_delta_rep'], indent=1))
        print(r['closed_form_unit'])
