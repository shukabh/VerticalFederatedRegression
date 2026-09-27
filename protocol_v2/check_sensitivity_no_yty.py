# Run: python3 protocol_v2/check_sensitivity_no_yty.py  (numpy + scipy; a few minutes)
# Replace-one sensitivity of (triu B, C, c_R, c_O) WITHOUT y'y, vs with it, vs eq.(6).
import numpy as np
from scipy.optimize import minimize
rng = np.random.default_rng(0)
def sq(v, BR, BO, By, dO, yty):
    x, xp = v[:dO], v[dO:2*dO]
    x = x*min(1, BO/max(np.linalg.norm(x),1e-12)); xp = xp*min(1, BO/max(np.linalg.norm(xp),1e-12))
    y, yp = np.clip(v[2*dO], -By, By), np.clip(v[2*dO+1], -By, By)
    dB = np.outer(x,x)-np.outer(xp,xp); iu = np.triu_indices(dO)
    s = np.sum(dB[iu]**2) + BR**2*np.sum((x-xp)**2) + np.sum((x*y-xp*yp)**2) + BR**2*(y-yp)**2
    return s + ((y*y-yp*yp)**2 if yty else 0)
def best(BR,BO,By,dO,yty,starts=120):
    b=0
    for _ in range(starts):
        v0 = rng.normal(size=2*dO+2)*max(BO,By)
        r = minimize(lambda v: -sq(v,BR,BO,By,dO,yty), v0, method="Nelder-Mead", options={"maxiter":3000,"xatol":1e-9,"fatol":1e-11})
        b = max(b, -r.fun)
    return np.sqrt(b)
for BR,BO,By in [(np.sqrt(5),np.sqrt(5),5),(1,1,1),(3.16,3,3)]:
    eq6 = np.sqrt((BO**2+BR**2)*(BO**2+By**2)+By**4)
    w = best(BR,BO,By,3,True); wo = best(BR,BO,By,3,False)
    tw = np.sqrt(2*BO**4 + 4*BO**2*(BR**2+By**2) + 4*BR**2*By**2)
    print(f"B_R={BR:.3g} B_O={BO:.3g} B_y={By}: eq6={eq6:.2f}  replace+yty={w:.2f}  replace-no-yty={wo:.2f}  termwise-bound-no-yty={tw:.2f}  gain(no yty vs replace+yty)={w/wo:.2f}x")
