"""Closed form (exact enumeration) of the full-record replace-one sensitivity, Proposition prop:sens_full of the paper,
for blocks u (R's features without the intercept, dim d_u, ||u||^2 <= R_u) and x_O (dim d_O, ||x_O||^2 <= o) of
dimension >= 2, and scalar y (y^2 <= w).  Proved in the stats proofs (Part B):
  Delta_full^2 = max_{t in [-w, w]} Psi(t),
  Psi(t) = K + [(iota + R_u + o)(w + t^2/w) - 2 iota t]  (stage 2)  + (w - t^2/w)^2  (joint mode)  + H(iota + t)  (t := 0 at stage 1)
  K = iota (2 R_u + 2 o) + 2 R_u^2 + 2 o^2 + 2 R_u o,
  H(a) = 2a^2/3 if |a| <= 3 m1;  a^2/2 + |a| m1 - 3 m1^2/2 if 3 m1 < |a| <= m1 + 2 m2;  2|a|(m1 + m2) - 2(m1^2 + m2^2 + m1 m2) beyond,
  m1 = min(R_u, o), m2 = max(R_u, o).
Psi is a piecewise quartic in t; its maximum is attained at t = +-w, at a breakpoint |iota + t| in {3 m1, m1 + 2 m2}, or at a
real root of the cubic Psi' on a piece, which this script enumerates (as DeltaRep does for prop:sens).  If a block has
dimension 1 the same value is a valid, conservative upper bound (a 1-dimensional configuration embeds in 2 dimensions).
The script compares the enumeration with experiments/baselines/sens_full.delta_full_grid (the earlier grid + local search; sens_full.delta_full is now this enumeration, and Lemma S10d for a block of dimension 1).
"""
import math, os, sys
import numpy as np
from numpy.polynomial import polynomial as Pn
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'baselines'))
import sens_full as sf

def H_pieces(Ru, o):
    m1, m2 = min(Ru, o), max(Ru, o)
    # each piece: (lo, hi, polynomial coefficients in |a|, low->high)
    return [(0.0, 3 * m1, [0.0, 0.0, 2.0 / 3.0]),
            (3 * m1, m1 + 2 * m2, [-1.5 * m1 * m1, m1, 0.5]),
            (m1 + 2 * m2, math.inf, [-2 * (m1 * m1 + m2 * m2 + m1 * m2), 2 * (m1 + m2)])]

def H(a, Ru, o):
    x = abs(a)
    for lo, hi, c in H_pieces(Ru, o):
        if x <= hi:
            return Pn.polyval(x, c)

def delta2_closed(B_R, B_O, B_y, intercept, stage=2, joint=True):
    iota = 1.0 if intercept else 0.0
    Ru = B_R ** 2 - iota; o = B_O ** 2; w = B_y ** 2
    K = iota * (2 * Ru + 2 * o) + 2 * Ru ** 2 + 2 * o ** 2 + 2 * Ru * o
    if stage == 1:
        return K + H(iota, Ru, o)
    # base polynomial in t (low -> high), without H
    base = Pn.polyadd([0.0, -2 * iota, (iota + Ru + o) / w], [(iota + Ru + o) * w])
    if joint:
        base = Pn.polyadd(base, [w * w, 0.0, -2.0, 0.0, 1.0 / (w * w)])     # (w - t^2/w)^2
    base = Pn.polyadd(base, [K])
    cands = {-w, w}
    for lo, hi, c in H_pieces(Ru, o):
        for sgn in (1.0, -1.0):          # |iota + t| = sgn (iota + t) on this piece
            # H(iota + t) as polynomial in t: c(sgn(iota + t))
            lin = [sgn * iota, sgn]
            Hp = [0.0]
            powt = [1.0]
            for k, ck in enumerate(c):
                Hp = Pn.polyadd(Hp, Pn.polymul([ck], powt))
                powt = Pn.polymul(powt, lin)
            tot = Pn.polyadd(base, Hp)
            # t-range of the piece for this sign
            for bnd in (lo, hi):
                if math.isfinite(bnd):
                    tb = sgn * bnd - iota
                    if -w <= tb <= w: cands.add(tb)
            for r in np.roots(Pn.polyder(tot)[::-1]) if len(tot) > 1 else []:
                if abs(r.imag) < 1e-9 * max(1, abs(r.real)):
                    t = r.real
                    if -w <= t <= w and lo - 1e-12 <= sgn * (iota + t) <= hi + 1e-12:
                        cands.add(t)
    def Psi(t):
        v = Pn.polyval(t, base) + H(iota + t, Ru, o)
        return v
    vals = {t: Psi(t) for t in cands}
    tbest = max(vals, key=vals.get)
    return vals[tbest], tbest

def check(cases):
    worst = 0.0
    for (BR, BO, By, ic) in cases:
        D, Dn, DG = sf.delta_full_grid(BR, BO, By, ic, 3, 3)
        c2, t2 = delta2_closed(BR, BO, By, ic, 2, True)
        c2n, _ = delta2_closed(BR, BO, By, ic, 2, False)
        c1 = delta2_closed(BR, BO, By, ic, 1)
        rel = max(abs(math.sqrt(c2) - D) / D, abs(math.sqrt(c2n) - Dn) / Dn, abs(math.sqrt(c1) - DG) / DG)
        worst = max(worst, rel)
        print(f'B=({BR},{BO},{By}) intercept={ic}: closed Delta={math.sqrt(c2):.6f} (t*={t2:+.4f}), grid {D:.6f}; '
              f'no yy {math.sqrt(c2n):.6f} vs {Dn:.6f}; stage 1 {math.sqrt(c1):.6f} vs {DG:.6f}')
    print('max relative difference closed vs grid+local search:', f'{worst:.2e}')
    return worst

if __name__ == '__main__':
    print('unit bounds, no intercept: Delta^2 =', delta2_closed(1, 1, 1, False)[0], '(32/3 =', 32 / 3, ');',
          'stage 1:', delta2_closed(1, 1, 1, False, 1), '(6)')
    rng = np.random.default_rng(0)
    cases = [(1.0, 1.0, 1.0, False), (5.59, 6.24, 2.89, True), (1.0, 1.0, 5.0, False), (1.0, 1.0, 5.0, True),
             (1.4142, 1.0, 2.89, True), (3.0, 0.5, 1.0, True), (0.5, 3.0, 4.0, False)]
    for _ in range(12):                  # with an intercept ||(1, u)|| <= B_R needs B_R >= 1
        ic = bool(rng.integers(2))
        cases.append((float(rng.uniform(1.0 if ic else 0.3, 6)), float(rng.uniform(0.3, 6)), float(rng.uniform(0.3, 6)), ic))
    check(cases)
    # dimension-1 block: the closed form is an upper bound
    for (BR, BO, By, ic) in [(1.4142, 1.0, 2.89, True), (5.59, 6.24, 2.89, True), (2.0, 1.5, 3.0, True)]:
        D1, Dn1, DG1 = sf.delta_full_grid(BR, BO, By, ic, 1, 2)
        print(f'd_u=1 B=({BR},{BO},{By}): grid {D1:.6f} <= closed {math.sqrt(delta2_closed(BR,BO,By,ic)[0]):.6f};'
              f' no yy {Dn1:.6f} <= {math.sqrt(delta2_closed(BR,BO,By,ic,2,False)[0]):.6f}; stage 1 {DG1:.6f} <= {math.sqrt(delta2_closed(BR,BO,By,ic,1)):.6f}')
