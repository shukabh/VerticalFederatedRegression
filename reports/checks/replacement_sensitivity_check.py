# Run: python3 reports/checks/replacement_sensitivity_check.py   (numpy + scipy; a few minutes)
# Replacement-adjacency sensitivity of the stacked O-release (B, C, c_R, c_O, y'y)
# vs. eq. (6) (add/remove). One matched record's (x_O, y) -> (x_O', y'); x_R fixed.
import numpy as np
from scipy.optimize import minimize
rng = np.random.default_rng(0)
def sq(v, BR, BO, By, dO):
    xR = BR  # only ||x_R|| enters
    x, xp = v[:dO], v[dO:2*dO]
    x  = x  * min(1, BO/ max(np.linalg.norm(x), 1e-12))
    xp = xp * min(1, BO/ max(np.linalg.norm(xp),1e-12))
    y, yp = np.clip(v[2*dO], -By, By), np.clip(v[2*dO+1], -By, By)
    dB = np.outer(x,x)-np.outer(xp,xp)
    iu = np.triu_indices(dO)                      # released: upper triangle of B
    return (np.sum(dB[iu]**2) + xR**2*np.sum((x-xp)**2) + np.sum((x*y-xp*yp)**2)
            + xR**2*(y-yp)**2 + (y*y-yp*yp)**2)
for (BR,BO,By,dO) in [(1,1,1,3),(3,3,20,4),(3.16,3,3,5)]:
    best = 0
    for _ in range(100):
        v0 = rng.normal(size=2*dO+2)*max(BO,By)
        r = minimize(lambda v: -sq(v,BR,BO,By,dO), v0, method="Nelder-Mead",
                     options={"maxiter":4000,"xatol":1e-10,"fatol":1e-12})
        best = max(best, -r.fun)
    eq6 = (BO**2+BR**2)*(BO**2+By**2)+By**4
    print(f"B_R={BR} B_O={BO} B_y={By}: eq(6) Delta={np.sqrt(eq6):.3f}   replacement Delta~{np.sqrt(best):.3f}   ratio {np.sqrt(best/eq6):.3f}")
