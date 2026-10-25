"""Leading-order bias of the plug-in residual sum of squares that the FWS comparison uses,
   RSS_hat = yy_hat - c_hat' G_hat^{-1} c_hat   (G_hat = G + E masked symmetric, c_hat = c + f, yy_hat = y'y + g),
   E[RSS_hat | T] = RSS - sigma^2 (tr P + beta' C_chi(P) beta) + O(sigma^4),  P = G^{-1}, beta = P c,
with C_chi(P) = sigma^-2 E[E P E] = chi o P + diag(chi p) - Pi_O diag(P) (Lemma bias.contraction), conditional on the
truncation event T = {||G^-1/2 E G^-1/2|| <= 1/2}.  Monte Carlo check with antithetic pairs on a random G.
"""
import numpy as np
rng = np.random.default_rng(3)
p, dA = 11, 5
X = rng.standard_normal((4000, p)) @ np.diag(np.linspace(0.5, 1.5, p))
G = X.T @ X; beta0 = rng.standard_normal(p); y = X @ beta0 + rng.standard_normal(4000)
c = X.T @ y; yy = y @ y; P = np.linalg.inv(G); beta = P @ c; RSS = yy - c @ beta
chi = np.ones((p, p)); chi[:dA, :dA] = 0
PiO = np.diag([0.0] * dA + [1.0] * (p - dA))
C = chi * P + np.diag(chi @ np.diag(P)) - PiO @ np.diag(np.diag(P))
iu = np.triu_indices(p); m = chi[iu] > 0
Gm12 = np.linalg.inv(np.linalg.cholesky(G))
for sigma in (20.0, 40.0, 80.0):
    pred = -sigma ** 2 * (np.trace(P) + beta @ C @ beta)
    acc, cnt = [], 0
    for _ in range(40000):
        z = np.zeros((p, p)); v = np.zeros(len(iu[0])); v[m] = rng.standard_normal(m.sum()) * sigma
        z[iu] = v; E = z + np.triu(z, 1).T
        f = rng.standard_normal(p) * sigma
        for sgn in (1, -1):
            M = Gm12 @ (sgn * E) @ Gm12.T
            if np.max(np.abs(np.linalg.eigvalsh(M))) > 0.5: continue
            ch = c + sgn * f
            acc.append(yy - ch @ np.linalg.solve(G + sgn * E, ch) - RSS)
    acc = np.array(acc)
    print(f'sigma={sigma:5.1f}: simulated bias {acc.mean():9.3f} +- {acc.std()/np.sqrt(len(acc)):.3f}; '
          f'leading term {pred:9.3f}; ratio {acc.mean()/pred:.4f}; theta_0 = {np.linalg.eigvalsh(G)[0]/(2*sigma*np.sqrt(p)):.1f}')
