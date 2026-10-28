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
the norms of 1-dimensional blocks, is maximized EXACTLY (delta_full): by the enumeration of Theorem S10c (Part B) when
both blocks have dimension >= 2 (sup2_closed: t = y y' at +-w, at the breakpoints, or at a real root of a cubic), and
by that of Lemma S10d (docs/proofs/stats_s7_s10.tex) when one block has dimension 1 (sup2_dim1: five faces of the box
of the scalar block and (y, y'), vertices, edge stationary points and interior stationary points from resultants,
with exact rational arithmetic; needs sympy).  The value is rounded up by a relative 1e-12, as delta_rep is.  The
earlier numerical value (a dense grid refined by bounded local search, which the reports up to reports/baselines
used) is kept as delta_full_grid; delta_full_grid(..., slow=True) searches all six variables (a check of the
reduction).  The two agree to 1e-13 on every bound vector checked (reports/baselines_v2).

Checks (python experiments/baselines/sens_full.py check): (i) with R's row held fixed (u = u', n_u = R_u) the same
maximization reproduces dp.delta_rep and dp.delta_G; (ii) a direct maximization of D^2 over actual vectors (d_u, d_O in
{1, 2, 3}) from many random starts never exceeds U and attains it; (iii) the closed form at unit bounds without an
intercept; (iv)-(vi) check_exact: exact vs grid, a dimension-1 value never below a dense 4-variable search, and never
above Theorem S10c's conservative value.  Writes reports/baselines_v2/sens_full_check.json.
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
def delta_full_grid(B_R, B_O, B_y, intercept, d_u, d_O, slow=False):
    """The earlier NUMERICAL value (grid + bounded local search, rounded up by a relative 1e-12): a lower bound on the
    supremum up to the safeguard.  Kept as a cross-check of the exact enumeration (delta_full); the numbers in
    reports/baselines (before reports/baselines_v2) used it.  slow=True searches all six variables."""
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


# ------------------------------------------------------------------------------------------- exact: blocks of dim >= 2
def _H_pieces(Ru, o):
    """H(a) = max over the box |g_u| <= R_u, |g_O| <= o of h (Theorem S10c), as pieces (lo, hi, coeffs in |a|)."""
    m1, m2 = min(Ru, o), max(Ru, o)
    return [(0.0, 3 * m1, [0.0, 0.0, 2.0 / 3.0]),
            (3 * m1, m1 + 2 * m2, [-1.5 * m1 * m1, m1, 0.5]),
            (m1 + 2 * m2, math.inf, [-2 * (m1 * m1 + m2 * m2 + m1 * m2), 2 * (m1 + m2)])]


def _H(a, Ru, o):
    x = abs(a)
    for lo, hi, c in _H_pieces(Ru, o):
        if x <= hi:
            return float(np.polynomial.polynomial.polyval(x, c))
    raise AssertionError


def sup2_closed(B_R, B_O, B_y, intercept, stage=2, joint=True):
    """Theorem S10c (blocks u and x_O of dimension >= 2): Delta^2 = max_{|t| <= w} Psi_2(t), Psi_2 a piecewise quartic
    in t = y y'; exact enumeration of t = +-w, the breakpoints |iota + t| in {3 m1, m1 + 2 m2} and the real roots of the
    cubic Psi_2' on each piece (the style of dp.delta_rep).  stage=1: K + H(iota).  Returns (Delta^2, t*)."""
    Pn = np.polynomial.polynomial
    iota = 1.0 if intercept else 0.0
    Ru = B_R ** 2 - iota
    o = B_O ** 2
    w = B_y ** 2
    K = iota * (2 * Ru + 2 * o) + 2 * Ru ** 2 + 2 * o ** 2 + 2 * Ru * o
    if stage == 1:
        return K + _H(iota, Ru, o), 0.0
    base = Pn.polyadd([0.0, -2 * iota, (iota + Ru + o) / w], [(iota + Ru + o) * w])
    if joint:
        base = Pn.polyadd(base, [w * w, 0.0, -2.0, 0.0, 1.0 / (w * w)])        # (w - t^2/w)^2
    base = Pn.polyadd(base, [K])
    cands = {-w, w}
    for lo, hi, c in _H_pieces(Ru, o):
        for sgn in (1.0, -1.0):                       # |iota + t| = sgn (iota + t) on this piece
            lin = [sgn * iota, sgn]
            Hp, powt = [0.0], [1.0]
            for ck in c:
                Hp = Pn.polyadd(Hp, Pn.polymul([ck], powt))
                powt = Pn.polymul(powt, lin)
            tot = Pn.polyadd(base, Hp)
            for bnd in (lo, hi):
                if math.isfinite(bnd):
                    tb = sgn * bnd - iota
                    if -w <= tb <= w:
                        cands.add(tb)
            for r in (np.roots(Pn.polyder(tot)[::-1]) if len(tot) > 1 else []):
                if abs(r.imag) < 1e-9 * max(1, abs(r.real)):
                    t = float(r.real)
                    if -w <= t <= w and lo - 1e-12 <= sgn * (iota + t) <= hi + 1e-12:
                        cands.add(t)
    vals = {t: float(Pn.polyval(t, base)) + _H(iota + t, Ru, o) for t in cands}
    tbest = max(vals, key=vals.get)
    return vals[tbest], tbest


# ------------------------------------------------------------------------------------------- exact: one block of dim 1
# Lemma S10d (docs/proofs/stats_s7_s10.tex).  R's block u is a scalar (|u| <= r = sqrt(R_u)); x_O has dimension >= 2
# (||x_O||^2 <= o).  With p = u, q = u' and x_O at its bound (monotonicity), the bracket (*) is exact and equals
#   U = iota[(p - q)^2 + (y - y')^2] + 2 iota o + 2 o^2 + o(p^2 + q^2 + y^2 + y'^2) + (p^2 - q^2)^2 + (p y - q y')^2
#       + (y^2 - y'^2)^2 [joint mode] + G(iota + p q + y y'),   G(a) = a^2/2 (|a| <= 2o), 2o|a| - 2o^2 (|a| > 2o),
# (terms in y dropped at stage 1).  U is convex in lambda^2 along (p, q) -> lambda (p, q) and along
# (y, y') -> lambda (y, y'), so some maximizer has (p, q) = 0 or max(|p|, |q|) = r, and (y, y') = 0 or
# max(|y|, |y'|) = B_y; by the symmetries (p, q, y, y') -> (q, p, y', y), (-p, -q, y, y'), (p, q, -y, -y') it lies on
# one of five faces.  On each face U is piecewise polynomial in at most two free variables, with pieces cut by the
# lines a = +-2o; its maximum is attained at a vertex, at a stationary point on an edge (a univariate polynomial) or
# at an interior stationary point of a piece (a common root of two bivariate polynomials, from their resultant).
# All are enumerated exactly (rational coefficients, exact real-root isolation); a positive-dimensional stationary
# set raises.  Needs sympy (and its mpmath).

def _sp():
    import mpmath
    import sympy
    return sympy, mpmath


def _face_setup(iota, o, with_y, with_yy):
    sp, _ = _sp()
    p, q, y, z = sp.symbols('p q y z', real=True)
    a = iota + p * q + (y * z if with_y else 0)
    base = iota * (p - q) ** 2 + 2 * iota * o + 2 * o * o + o * (p * p + q * q) + (p * p - q * q) ** 2
    if with_y:
        base += iota * (y - z) ** 2 + o * (y * y + z * z) + (p * y - q * z) ** 2
        if with_yy:
            base += (y * y - z * z) ** 2
    pieces = {'mid': sp.expand(base + a * a / 2), 'hi': sp.expand(base + 2 * o * a - 2 * o * o),
              'lo': sp.expand(base - 2 * o * a - 2 * o * o)}
    return (p, q, y, z), a, pieces


def U_dim1(vals, iota, o, with_y, with_yy):
    """U of Lemma S10d at a point (u, u', y, y'), in mpmath precision (the true U, with the true G)."""
    _, mp = _sp()
    P, Q, Y, Z = [mp.mpf(v) for v in vals]
    io, oo = mp.mpf(iota), mp.mpf(o)
    a = io + P * Q + (Y * Z if with_y else 0)
    G = a * a / 2 if abs(a) <= 2 * oo else 2 * oo * abs(a) - 2 * oo * oo
    v = io * (P - Q) ** 2 + 2 * io * oo + 2 * oo * oo + oo * (P * P + Q * Q) + (P * P - Q * Q) ** 2 + G
    if with_y:
        v += io * (Y - Z) ** 2 + oo * (Y * Y + Z * Z) + (P * Y - Q * Z) ** 2
        if with_yy:
            v += (Y * Y - Z * Z) ** 2
    return v


def _roots_exact(expr, var, lo, hi):
    """Real roots in [lo, hi] of a univariate polynomial with rational coefficients (exact isolation, then 60 digits).
    None if the polynomial is identically zero."""
    sp, mp = _sp()
    P = sp.Poly(expr, var)
    if P.is_zero:
        return None
    if P.degree() <= 0:
        return []
    lo_, hi_ = mp.mpf(sp.N(lo, 60)), mp.mpf(sp.N(hi, 60))
    out = []
    for rt in P.real_roots():
        v = mp.mpf(sp.N(rt, 60))
        if lo_ <= v <= hi_:
            out.append(v)
    return out


def _roots_mp(coeffs, lo, hi):
    """Real roots in [lo, hi] of a polynomial with mpmath coefficients (highest first); None if identically zero."""
    _, mp = _sp()
    scale = max([abs(c) for c in coeffs] + [mp.mpf(0)])
    if scale == 0:
        return None
    while coeffs and abs(coeffs[0]) <= mp.mpf(10) ** -45 * scale:
        coeffs = coeffs[1:]
    if len(coeffs) <= 1:
        return []
    out = []
    for r_ in mp.polyroots(coeffs, maxsteps=1000, extraprec=400):
        if abs(mp.im(r_)) <= mp.mpf(10) ** -30 * max(1, abs(r_)):
            v = mp.re(r_)
            tol = mp.mpf(10) ** -30 * max(1, abs(v))
            if lo - tol <= v <= hi + tol:
                out.append(min(max(v, lo), hi))
    return out


def _face_candidates(fixed, free, bnd, a_expr, pieces, o):
    """Candidate maximizers on one face: vertices, stationary points on edges, interior stationary points."""
    sp, mp = _sp()
    a_f = sp.expand(a_expr.subs(fixed))
    polys = {k: sp.expand(e.subs(fixed)) for k, e in pieces.items()}
    lo = {v: -bnd[v] for v in free}
    hi = {v: bnd[v] for v in free}

    def M(x):
        return mp.mpf(sp.N(x, 60))

    if not free:
        return [{}]
    if len(free) == 1:
        s = free[0]
        pts = [M(lo[s]), M(hi[s])]
        for c in (2 * o, -2 * o):                                        # breakpoints of G
            pts += _roots_exact(a_f - c, s, lo[s], hi[s]) or []
        for e in polys.values():
            pts += _roots_exact(sp.diff(e, s), s, lo[s], hi[s]) or []
        return [{s: v} for v in pts]
    s, t = free
    cands = []
    for vs in (lo[s], hi[s]):                                            # vertices of the box
        for vt in (lo[t], hi[t]):
            cands.append({s: M(vs), t: M(vt)})
    for fv, ov in ((s, t), (t, s)):                                      # box edges
        for val in (lo[fv], hi[fv]):
            for c in (2 * o, -2 * o):                                    # where a line a = +-2o meets the edge
                for r_ in _roots_exact(a_f.subs(fv, val) - c, ov, lo[ov], hi[ov]) or []:
                    cands.append({fv: M(val), ov: r_})
            for e in polys.values():
                for r_ in _roots_exact(sp.diff(e.subs(fv, val), ov), ov, lo[ov], hi[ov]) or []:
                    cands.append({fv: M(val), ov: r_})
    ca, cs, ct = a_f.subs({s: 0, t: 0}), sp.diff(a_f, s), sp.diff(a_f, t)  # a is affine on every face
    assert not (cs.free_symbols | ct.free_symbols)
    for c in (2 * o, -2 * o):                                            # the lines a = +-2o inside the box
        if ct != 0:                                                      # (U is continuous: either piece)
            tl = (c - ca - cs * s) / ct
            for r_ in _roots_exact(sp.diff(sp.expand(polys['mid'].subs(t, tl)), s), s, lo[s], hi[s]) or []:
                tv = M(tl.subs(s, sp.Float(r_, 60)))
                if M(lo[t]) <= tv <= M(hi[t]):
                    cands.append({s: r_, t: tv})
        elif cs != 0:
            sv = (c - ca) / cs
            if lo[s] <= sv <= hi[s]:
                for r_ in _roots_exact(sp.diff(sp.expand(polys['mid'].subs(s, sv)), t), t, lo[t], hi[t]) or []:
                    cands.append({s: M(sv), t: r_})
    for k, e in polys.items():                                           # interior stationary points of each piece
        fs, ft = sp.expand(sp.diff(e, s)), sp.expand(sp.diff(e, t))
        res = sp.expand(sp.resultant(fs, ft, t))
        if res == 0:
            raise RuntimeError(f'positive-dimensional stationary set on piece {k}: gcd {sp.gcd(fs, ft)}')
        for sv in _roots_exact(res, s, lo[s], hi[s]) or []:
            for f in (ft, fs):
                cf = [mp.mpf(sp.N(cc.subs(s, sp.Float(sv, 60)), 60)) for cc in sp.Poly(f, t).all_coeffs()]
                rts = _roots_mp(cf, M(lo[t]), M(hi[t]))
                if rts is not None:
                    cands += [{s: sv, t: tv} for tv in rts]
                    break
            else:                                                        # both partials vanish on the line s = sv,
                cands.append({s: sv, t: mp.mpf(0)})                      # so U is constant along it
    return cands


def sup2_dim1(Ru, o, B_y, intercept, with_y=True, with_yy=True):
    """Lemma S10d: the exact supremum of U when R's block u has dimension 1 and x_O dimension >= 2 (swap R_u and o for
    the converse).  Ru, o, B_y: floats or Fractions (converted exactly).  u ranges over [-r, r] with r the smallest
    double >= sqrt(R_u), a superset of its range, so the value is an upper bound exceeding the supremum by a relative
    O(2^-52) at most; the float is rounded up by 2^-50.  Returns (sup U, maximizer (u, u', y, y'))."""
    sp, mp = _sp()
    mp.mp.dps = 60
    iota = sp.Integer(1 if intercept else 0)
    Ru, o, B = sp.Rational(Ru), sp.Rational(o), sp.Rational(B_y)
    rf = math.nextafter(math.sqrt(float(Ru)), math.inf)
    while sp.Rational(rf) ** 2 < Ru:
        rf = math.nextafter(rf, math.inf)
    r = sp.Rational(rf)
    (p, q, y, z), a_expr, pieces = _face_setup(iota, o, with_y, with_yy)
    bnd = {p: r, q: r, y: B, z: B}
    if with_y:
        faces = [({p: r, y: B}, [q, z]), ({q: r, y: B}, [p, z]), ({p: 0, q: 0, y: B}, [z]),
                 ({p: r, y: 0, z: 0}, [q]), ({p: 0, q: 0, y: 0, z: 0}, [])]
    else:
        faces = [({p: r, y: 0, z: 0}, [q]), ({p: 0, q: 0, y: 0, z: 0}, [])]
    best, arg = None, None
    for fixed, free in faces:
        for c in _face_candidates(fixed, free, bnd, a_expr, pieces, o):
            vals = [mp.mpf(sp.N(fixed[v], 60)) if v in fixed else c.get(v, mp.mpf(0)) for v in (p, q, y, z)]
            val = U_dim1(vals, iota, o, with_y, with_yy)
            if best is None or val > best:
                best, arg = val, vals
    return float(best) * (1 + 2.0 ** -50), tuple(float(v) for v in arg)


# ------------------------------------------------------------------------------------------- the reported values
METHODS = {'closed': "Theorem S10c (blocks of dimension >= 2): exact enumeration over t = y y'",
           'dim1': 'Lemma S10d (one block of dimension 1): exact enumeration over the faces'}


@lru_cache(maxsize=None)
def delta_full_info(B_R, B_O, B_y, intercept, d_u, d_O):
    """The exact sensitivities, the method used and the maximizers (for the reports)."""
    if d_u >= 2 and d_O >= 2:
        D2, t2 = sup2_closed(B_R, B_O, B_y, intercept, 2, True)
        D2n, t2n = sup2_closed(B_R, B_O, B_y, intercept, 2, False)
        DG2, _ = sup2_closed(B_R, B_O, B_y, intercept, 1)
        method, arg = 'closed', dict(t_joint=t2, t_separate=t2n)
    elif min(d_u, d_O) == 1 and max(d_u, d_O) >= 2:
        from fractions import Fraction
        Ru = Fraction(B_R) ** 2 - int(intercept)
        o = Fraction(B_O) ** 2
        if d_O == 1:                                  # U is symmetric in the blocks u and x_O
            Ru, o = o, Ru
        D2, a2 = sup2_dim1(Ru, o, B_y, intercept, True, True)
        D2n, a2n = sup2_dim1(Ru, o, B_y, intercept, True, False)
        DG2, aG = sup2_dim1(Ru, o, B_y, intercept, False, False)
        method, arg = 'dim1', dict(joint=a2, separate=a2n, stage1=aG, swapped=d_O == 1)
    else:
        raise NotImplementedError("R's block and O's block both of dimension 1: not needed by any dataset")
    up = 1 + SAFEGUARD
    return dict(Delta=math.sqrt(D2) * up, Delta_noyy=math.sqrt(D2n) * up, Delta_G=math.sqrt(DG2) * up,
                method=method, argmax=arg)


def delta_full(B_R, B_O, B_y, intercept, d_u, d_O):
    """(Delta, Delta_noyy, Delta_G) of the full-record release: stage 2 with and without y'y, and stage 1, EXACT:
    Theorem S10c when both blocks have dimension >= 2, Lemma S10d when one has dimension 1.  B_R covers the intercept
    when intercept=True (as in v3.1).  Rounded up by a relative 1e-12, as dp.delta_rep."""
    r = delta_full_info(float(B_R), float(B_O), float(B_y), bool(intercept), int(d_u), int(d_O))
    return r['Delta'], r['Delta_noyy'], r['Delta_G']


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


def brute_dim1(Ru, o, B_y, iota, with_y, with_yy, n=29, seed=0, starts=40):
    """A dense grid over (u, u', y, y') in the box plus local refinement of the best points: a lower bound on sup U
    for a 1-dimensional block u (independent of the face reduction of Lemma S10d)."""
    r = math.sqrt(Ru)
    gp = np.linspace(-r, r, n)
    gy = np.linspace(-B_y, B_y, n) if with_y else np.zeros(1)
    P, Q, Y, Z = [m.ravel() for m in np.meshgrid(gp, gp, gy, gy, indexing='ij')]

    def U(P, Q, Y, Z):
        a = iota + P * Q + (Y * Z if with_y else 0.0)
        G = np.where(np.abs(a) <= 2 * o, a * a / 2, 2 * o * np.abs(a) - 2 * o * o)
        v = iota * (P - Q) ** 2 + 2 * iota * o + 2 * o * o + o * (P * P + Q * Q) + (P * P - Q * Q) ** 2 + G
        if with_y:
            v = v + iota * (Y - Z) ** 2 + o * (Y * Y + Z * Z) + (P * Y - Q * Z) ** 2
            if with_yy:
                v = v + (Y * Y - Z * Z) ** 2
        return v

    vals = U(P, Q, Y, Z)
    best = float(vals.max())
    bnds = [(-r, r), (-r, r), (-B_y, B_y), (-B_y, B_y)] if with_y else [(-r, r), (-r, r), (0, 0), (0, 0)]
    rng = np.random.default_rng(seed)
    x0s = [np.array([P[i], Q[i], Y[i], Z[i]]) for i in np.argsort(-vals)[:starts]]
    x0s += [np.array([rng.uniform(lo, hi) for lo, hi in bnds]) for _ in range(starts)]
    for x0 in x0s:
        rr = minimize(lambda x: -float(U(*[np.atleast_1d(v) for v in x])[0]), x0, method='L-BFGS-B', bounds=bnds)
        best = max(best, -float(rr.fun))
    return best


def check_exact(seed=0, n_random=24):
    """Checks of the exact values: (iv) exact >= the earlier grid value / (1 + 1e-12) (the grid evaluates feasible
    points) and their difference; (v) for a 1-dimensional block, the exact value never below a dense 4-variable grid
    with local refinement, which does not use the face reduction; (vi) for a 1-dimensional block, the exact value at
    most the conservative closed form of Theorem S10c."""
    rng = np.random.default_rng(seed)
    cases = [(1.0, 1.0, 1.0, False), (math.sqrt(2), 1.0, 1.7188210326789277, True),
             (math.sqrt(2), 1.0, 3.4330469390257967, True), (5.59, 6.24, 2.89, True), (2.0, 1.5, 3.0, True),
             (3.0, 0.5, 1.0, True), (1.0, 1.0, 5.0, False), (1.2, 2.5, 0.4, True), (0.5, 3.0, 4.0, False)]
    for _ in range(n_random):
        ic = bool(rng.integers(2))
        cases.append((float(rng.uniform(1.0 if ic else 0.3, 6)), float(rng.uniform(0.3, 6)), float(rng.uniform(0.3, 6)),
                      ic))
    out = dict(exact_vs_grid=[], dim1_vs_brute=[], dim1_vs_closed=[])
    for (BR, BO, By, ic) in cases:
        for du, dO in ((3, 3), (1, 3), (3, 1)):
            ex = delta_full(BR, BO, By, ic, du, dO)
            gr = delta_full_grid(BR, BO, By, ic, du, dO)
            out['exact_vs_grid'].append(dict(bounds=[BR, BO, By], intercept=ic, d_u=du, d_O=dO, exact=list(ex),
                                             grid=list(gr), rel=[e / g - 1 for e, g in zip(ex, gr)]))
        iota = 1.0 if ic else 0.0
        Ru, o = BR ** 2 - iota, BO ** 2
        ex = delta_full(BR, BO, By, ic, 1, 3)
        br = [math.sqrt(brute_dim1(Ru, o, By, iota, wy, wyy)) for wy, wyy in ((True, True), (True, False), (False, False))]
        out['dim1_vs_brute'].append(dict(bounds=[BR, BO, By], intercept=ic, exact=list(ex), brute=br,
                                         rel=[e / b - 1 for e, b in zip(ex, br)]))
        cl = [math.sqrt(sup2_closed(BR, BO, By, ic, 2, True)[0]), math.sqrt(sup2_closed(BR, BO, By, ic, 2, False)[0]),
              math.sqrt(sup2_closed(BR, BO, By, ic, 1)[0])]
        out['dim1_vs_closed'].append(dict(bounds=[BR, BO, By], intercept=ic, exact=list(ex), closed=cl,
                                          rel=[e / c - 1 for e, c in zip(ex, cl)]))
    rels = [x for r in out['exact_vs_grid'] for x in r['rel']]
    out['summary'] = dict(
        n_cases=len(cases),
        exact_over_grid_min=min(rels), exact_over_grid_max=max(rels),
        dim1_exact_over_brute_min=min(x for r in out['dim1_vs_brute'] for x in r['rel']),
        dim1_exact_over_closed_max=max(x for r in out['dim1_vs_closed'] for x in r['rel']),
        dim1_strictly_below_closed=sum(any(x < -1e-9 for x in r['rel']) for r in out['dim1_vs_closed']))
    return out


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
    res.update(check_exact())
    if out_path:
        with open(out_path, 'w') as fh:
            json.dump(res, fh, indent=1)
    return res


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'check':
        os.makedirs(os.path.join(ROOT, 'reports', 'baselines_v2'), exist_ok=True)
        r = check(os.path.join(ROOT, 'reports', 'baselines_v2', 'sens_full_check.json'))
        print(json.dumps(r['fixed_R_vs_delta_rep'], indent=1))
        print(r['closed_form_unit'])
        print(json.dumps(r['summary'], indent=1))
