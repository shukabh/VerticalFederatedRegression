# select_ridge picks the O-block ridge whenever lambda_min(A)/(2 sigma sqrt p) >= target.
# Near that ceiling the O-block lambda needed explodes (lambda_min(G + lam Pi_O) -> lambda_min(A)
# only as lam -> inf), and auto_lambda returns its cap (1e12) if the target is never reached.
import sys, numpy as np
sys.path.insert(0, "/home/user/VerticalFederatedRegression/scenario_b")
import phase1_common as p1
rng = np.random.default_rng(0)
dR, dO, n = 3, 3, 4000
X = rng.uniform(size=(n, dR + dO)); beta = rng.normal(size=dR + dO)
y = X @ beta + 0.1 * rng.normal(size=n)
G, c = X.T @ X, X.T @ y
p = dR + dO; lamA = np.linalg.eigvalsh(G[:dR, :dR])[0]
bols = np.linalg.solve(G, c)
for margin in (1.5, 1.05, 1.01, 1.0):
    sigma = lamA / (2 * np.sqrt(p) * 2.0 * margin)       # ceiling = 2.0 * margin
    mode, lam, rho = p1.select_ridge(G, dR, dO, sigma, target=2.0)
    Psi = p1.ridge_selector(dR, dO, mode); br = np.linalg.solve(G + lam * Psi, c)
    lamI = p1.auto_lambda(G, np.eye(p), sigma, p, 2.0); bI = np.linalg.solve(G + lamI * np.eye(p), c)
    print(f"ceiling={2*margin:.2f}: chosen {mode} lam={lam:.3g} rho={rho:.3f} "
          f"|beta_O|/|ols_O|={np.linalg.norm(br[dR:])/np.linalg.norm(bols[dR:]):.3f}   "
          f"(full ridge would use lam={lamI:.3g}, |beta_O| ratio {np.linalg.norm(bI[dR:])/np.linalg.norm(bols[dR:]):.3f}, "
          f"|beta_R| ratio {np.linalg.norm(bI[:dR])/np.linalg.norm(bols[:dR]):.3f})")
