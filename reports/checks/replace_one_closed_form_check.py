# Run: python3 reports/checks/replace_one_closed_form_check.py  (numpy + scipy; several minutes)
# Validate the replace-one closed form (DP report) against brute force on random bounds.
import numpy as np
from scipy.optimize import minimize, minimize_scalar
rng = np.random.default_rng(11)
def closed(BR,BO,By):
    R,O,Y = BR**2,BO**2,By**2
    if abs(R-Y) <= 2*O: return np.sqrt(0.5*(2*O+R+Y)**2 + 2*R*Y)
    if R-Y >= 2*O:      return np.sqrt(4*R*(O+Y))
    def F(t):
        c = np.clip(-(R + By*t)/(2*O), -1, 1)
        return 2*O*O*(1-c*c) + 2*R*O*(1-c) + R*(By-t)**2 + O*(Y+t*t) - 2*O*c*By*t + (Y-t*t)**2
    ts = np.linspace(-By, By, 4001)
    return np.sqrt(max(F(t) for t in ts))
def sq(v, BR, BO, By, dO):
    x, xp = v[:dO], v[dO:2*dO]
    x = x*min(1, BO/max(np.linalg.norm(x),1e-12)); xp = xp*min(1, BO/max(np.linalg.norm(xp),1e-12))
    y, yp = np.clip(v[2*dO], -By, By), np.clip(v[2*dO+1], -By, By)
    dB = np.outer(x,x)-np.outer(xp,xp); iu = np.triu_indices(dO)
    return np.sum(dB[iu]**2) + BR**2*np.sum((x-xp)**2) + np.sum((x*y-xp*yp)**2) + BR**2*(y-yp)**2 + (y*y-yp*yp)**2
def brute(BR,BO,By,dO=3,starts=80):
    b=0
    for _ in range(starts):
        v0 = rng.normal(size=2*dO+2)*max(BO,By,BR)
        r = minimize(lambda v: -sq(v,BR,BO,By,dO), v0, method="Nelder-Mead", options={"maxiter":3000,"xatol":1e-9,"fatol":1e-11})
        b = max(b, -r.fun)
    return np.sqrt(b)
worst = 0
cases = [(5,1,1),(4,1,2),(1,1,5),(0.5,2,6),(np.sqrt(5),np.sqrt(5),5)] + [tuple(rng.uniform(0.3,5,3)) for _ in range(10)]
for BR,BO,By in cases:
    R,O,Y = BR**2,BO**2,By**2
    reg = 1 if abs(R-Y)<=2*O else (2 if R-Y>=2*O else 3)
    cf, bf = closed(BR,BO,By), brute(BR,BO,By)
    worst = max(worst, (bf-cf)/bf)
    print(f"B_R={BR:.2f} B_O={BO:.2f} B_y={By:.2f} regime={reg}: closed={cf:.4f} brute={bf:.4f} rel(brute-closed)={(bf-cf)/bf:+.1e}")
print("max relative shortfall of closed form:", f"{worst:+.2e}", "(<= 0 means the closed form is never below the brute-force sup)")
