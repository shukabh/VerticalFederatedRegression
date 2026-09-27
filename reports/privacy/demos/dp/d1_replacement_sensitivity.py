"""
D1 — l2 sensitivity of the Scenario-B release under Definition 4 (replace one matched
individual's O-side data (x_O, y) -> (x_O', y'); x_R and the intersection fixed).

Released vector (what party_o.py actually noises, one N(0, sigma^2) draw per entry):
    [ triu(B) , vec(C) , c_R , c_O , y'y ]
with B = sum x_O x_O^T, C = sum x_R x_O^T, c_R = sum x_R y, c_O = sum x_O y, y'y = sum y^2.

What this script does
  1. brute force: maximise ||phi(x_R,x_O,y) - phi(x_R,x_O',y')||^2 over the ACTUAL layout
     (triu, not Frobenius) with multistart BFGS, no structural assumptions;
  2. exact 1-D algorithm (closed-form candidates, derivation in dp_layer.md, D1);
  3. regime closed forms; eq. (6) (add/remove); 2*eq.(6) (triangle bound);
  4. sigma inflation factor and the EFFECTIVE eps of the code's sigma under Definition 4,
     using calibrate_hyperparameters.analytic_gaussian_sigma itself.

Run:  python d1_replacement_sensitivity.py      (~25 s)
"""
import math, os, sys
import numpy as np
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "../../../../scenario_b")))
from calibrate_hyperparameters import joint_sensitivity_B, analytic_gaussian_sigma  # noqa: E402


# ---------------------------------------------------------------- the release map
def release(xR, xO, y):
    iu = np.triu_indices(len(xO))
    return np.concatenate([np.outer(xO, xO)[iu], np.outer(xR, xO).ravel(),
                           xR * y, xO * y, [y * y]])


def brute_force_sq(BR, BO, By, dR=2, dO=3, starts=10, seed=0):
    """No assumptions: optimise over two O-records in the balls, x_R on its sphere."""
    rng = np.random.default_rng(seed)
    xR = np.zeros(dR); xR[0] = BR          # F is increasing in ||x_R||; direction irrelevant

    def to_ball(z, B):
        n = np.linalg.norm(z) + 1e-15
        return z / n * B * np.tanh(n)

    def negF(p):
        u, v = to_ball(p[:dO], BO), to_ball(p[dO:2 * dO], BO)
        y, yp = By * np.tanh(p[-2]), By * np.tanh(p[-1])
        return -np.sum((release(xR, u, y) - release(xR, v, yp)) ** 2)

    best = 0.0
    for _ in range(starts):
        r = minimize(negF, rng.normal(size=2 * dO + 2) * 2.0, method="BFGS")
        best = max(best, -r.fun)
    return best


# ---------------------------------------------------------------- exact 1-D algorithm
def delta_replace_sq(BR, BO, By):
    """Exact sup of the squared l2 change under replacement adjacency.

    Reduction (verified against brute_force_sq): ||x_R||=B_R, ||x_O||=||x_O'||=B_O, y=B_y,
    y'=t in [-B_y,B_y], cos(x_O,x_O')=c.  After rotating x_O,x_O' so that
    x_O x_O^T - x_O' x_O'^T is diagonal (the triu norm then equals the Frobenius norm):
      F(c,t) = 2o^4(1-c^2) + 2r^2 o^2(1-c) + r^2(B-t)^2 + o^2(B^2+t^2) - 2 o^2 c B t + (B^2-t^2)^2
    F is concave in c with c*(t) = clip(-(r^2+Bt)/(2 o^2), -1, 1); on each piece F(c*(t),t)
    is a quartic in t, so the max over t is attained at an endpoint, a breakpoint, or a
    real root of one of three cubics. Evaluate all candidates and take the max.
    """
    r, o, B = float(BR), float(BO), float(By)
    if o == 0:
        return max(4 * r * r * B * B, B ** 4) if B > 0 else 0.0   # only c_R and y'y move

    def F(c, t):
        return (2 * o**4 * (1 - c * c) + 2 * r * r * o * o * (1 - c) + r * r * (B - t) ** 2
                + o * o * (B * B + t * t) - 2 * o * o * c * B * t + (B * B - t * t) ** 2)

    def cstar(t):
        return min(1.0, max(-1.0, -(r * r + B * t) / (2 * o * o)))

    cand = [-B, B, 0.0]
    if B > 0:
        cand += [(2 * o * o - r * r) / B, (-2 * o * o - r * r) / B]
    cubics = [
        [4, 0, 2 * r * r + 2 * o * o - 3 * B * B, -r * r * B],                  # interior c
        [4, 0, 2 * r * r + 2 * o * o - 4 * B * B, 2 * B * (o * o - r * r)],      # c = -1
        [4, 0, 2 * r * r + 2 * o * o - 4 * B * B, -2 * B * (r * r + o * o)],     # c = +1
    ]
    for co in cubics:
        for z in np.roots(co):
            if abs(z.imag) < 1e-9:
                cand.append(z.real)
    best = 0.0
    for t in cand:
        if -B - 1e-12 <= t <= B + 1e-12:
            t = min(B, max(-B, t))
            best = max(best, F(cstar(t), t))
    return best


def delta_replace(BR, BO, By):
    return math.sqrt(delta_replace_sq(BR, BO, By))


# ---------------------------------------------------------------- closed forms
def eq6_sq(BR, BO, By):
    return (BO**2 + BR**2) * (BO**2 + By**2) + By**4


def balanced_sq(BR, BO, By):      # regime |B_R^2-B_y^2| <= 2B_O^2 (and B_y not dominant)
    return 0.5 * (2 * BO**2 + BR**2 + By**2) ** 2 + 2 * BR**2 * By**2


def r_dominant_sq(BR, BO, By):    # regime B_R^2 >= B_y^2 + 2 B_O^2
    return 4 * BR**2 * (BO**2 + By**2)


def rot_relaxed_sq(BR, BO, By):   # always-valid closed-form upper bound (see D1)
    W2 = BO**2 + By**2
    return 0.5 * (2 * W2 + BR**2) ** 2 if BR**2 <= 2 * W2 else 4 * BR**2 * W2


def eff_eps(sigma, Delta, delta):
    """eps actually delivered by noise sigma for sensitivity Delta (same delta)."""
    lo, hi = 1e-6, 200.0
    for _ in range(200):
        mid = math.sqrt(lo * hi)
        if analytic_gaussian_sigma(Delta, mid, delta) > sigma:
            lo = mid
        else:
            hi = mid
    return hi


def main():
    print("=== 1. exact algorithm vs assumption-free brute force (layout: triu B, C, c_R, c_O, y'y)")
    tests = [(1, 1, 1), (3, 3, 3), (3, 3, 20), (5, 1, 1), (1, 1, 5), (2, 1, 1), (1, 3, 1),
             (math.sqrt(11), math.sqrt(15), 3.4)]
    rng = np.random.default_rng(7)
    tests += [tuple(np.round(np.exp(rng.uniform(-1, 2.5, 3)), 3)) for _ in range(4)]
    worst = 0.0
    for b in tests:
        bf = brute_force_sq(*b)
        ex = delta_replace_sq(*b)
        worst = max(worst, (bf - ex) / ex)
        print(f"  B_R,B_O,B_y={tuple(round(float(v),3) for v in b)!s:28s} brute={bf:12.4f} exact={ex:12.4f}"
              f"  eq6={eq6_sq(*b):12.4f}  ratio Delta_rep/Delta_6={math.sqrt(ex/eq6_sq(*b)):.4f}")
    print(f"  max relative excess brute over exact: {worst:.2e}  (<=0 means exact is the sup)")

    print("\n=== 2. regime closed forms vs exact on a random grid")
    ok_bal = ok_rd = n_bal = n_rd = 0
    rng = np.random.default_rng(3)
    for _ in range(20000):
        BR, BO, By = np.exp(rng.uniform(-2, 3, 3))
        ex = delta_replace_sq(BR, BO, By)
        if abs(BR**2 - By**2) <= 2 * BO**2 and By**2 <= 4 * BR**2 + 2 * BO**2:
            n_bal += 1; ok_bal += abs(balanced_sq(BR, BO, By) / ex - 1) < 1e-9
        if BR**2 >= By**2 + 2 * BO**2:
            n_rd += 1; ok_rd += abs(r_dominant_sq(BR, BO, By) / ex - 1) < 1e-9
        assert ex <= rot_relaxed_sq(BR, BO, By) * (1 + 1e-12) and ex <= 4 * eq6_sq(BR, BO, By) * (1 + 1e-12)
        assert ex >= eq6_sq(BR, BO, By) * (1 - 1e-12)
    print(f"  balanced formula exact in {ok_bal}/{n_bal} draws of its regime")
    print(f"  R-dominant formula exact in {ok_rd}/{n_rd} draws of its regime")
    print("  eq6 <= Delta_rep^2 <= min(4*eq6, rot-relaxed bound): held on all 20000 draws")

    print("\n=== 3. representative bounds: sigma inflation and effective eps of the code's sigma")
    reps = [
        ("normalised, all bounds 1", 1.0, 1.0, 1.0),
        ("docstring example (3,3,20)", 3.0, 3.0, 20.0),
        ("generator data raw U[0,1], d_R=10,d_O=15", math.sqrt(10), math.sqrt(15), 4.0),
        ("std+intercept d_R=10,d_O=15 (B_R=sqrt(1+4.1^2))", math.sqrt(1 + 4.1**2), 4.9, 2.6),
        ("R-heavy std d_R=50,d_O=5 (+intercept)", math.sqrt(1 + 8.4**2), 3.2, 2.6),
        ("same, no intercept", 8.4, 3.2, 2.6),
    ]
    delta = 1e-5
    print(f"  delta={delta}")
    print(f"  {'case':50s} {'D6':>8s} {'Drep':>8s} {'ratio':>6s} | eps_claimed -> eps_actual (Def.4)")
    for name, BR, BO, By in reps:
        d6 = joint_sensitivity_B(BR, BO, By)
        dr = delta_replace(BR, BO, By)
        effs = []
        for eps in (0.5, 1.0, 4.0, 8.0):
            s = analytic_gaussian_sigma(d6, eps, delta)
            effs.append(f"{eps:g}->{eff_eps(s, dr, delta):.2f}")
        print(f"  {name:50s} {d6:8.3f} {dr:8.3f} {dr/d6:6.3f} | " + "  ".join(effs))

    print("\n=== 4. intercept: code sets B_R = sqrt(1 + B_R_rows^2); effect on ratio for B_O=B_y=1")
    for rows in (0.0, 0.5, 1.0, 2.0, 4.0):
        BR = math.sqrt(1 + rows**2)
        print(f"  B_R_rows={rows:3.1f}  B_R={BR:.3f}  Delta_6={joint_sensitivity_B(BR,1,1):.3f}"
              f"  Delta_rep={delta_replace(BR,1,1):.3f}  ratio={delta_replace(BR,1,1)/joint_sensitivity_B(BR,1,1):.3f}")


    print("\n=== 5. range of Delta_rep/Delta_6 per regime (20000 random bound triples, log-uniform)")
    rng = np.random.default_rng(4)
    stats = {"balanced": [], "R-dominant": [], "y-dominant": []}
    for _ in range(20000):
        BR, BO, By = np.exp(rng.uniform(-3, 3, 3))
        k = ("balanced" if abs(BR**2 - By**2) <= 2 * BO**2 else
             "R-dominant" if BR**2 > By**2 else "y-dominant")
        stats[k].append(math.sqrt(delta_replace_sq(BR, BO, By) / eq6_sq(BR, BO, By)))
    for k, v in stats.items():
        v = np.array(v)
        print(f"  {k:10s}: n={len(v):5d}  min={v.min():.3f}  median={np.median(v):.3f}  max={v.max():.3f}")


if __name__ == "__main__":
    main()
