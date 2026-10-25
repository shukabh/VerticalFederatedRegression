"""Second-reader check of the book's Appendix B results at the evaluation setting (circuit A, Omega_U = 2^68):
  * B_tot with OpenFHE's uncentred gadget (lem:btot.gadget): varsigma_up = 2 max k_j, varsigma_down = 2 k_P;
  * the centre-sensitivity remainder eps_x of lem:btot.sensZ (must be <= 1 for the '+1' in Delta_x);
  * the decoding-offset bound d_bar (prop:btot.offset / lem:agg.offset), slot domain, and its coefficient-domain
    form, which serves as d_max in the no-wraparound condition (Q) (prop:agg.wrap);
  * a brute-force check of the offset bound and of the centre sensitivity at toy size (N = 16), exact rationals.
Run from the repository root:  python3 experiments/proofs_check/check_btot_offset.py
"""
import math, random, sys
from fractions import Fraction
sys.path.insert(0, 'implementation')
from vfr import btot as bt, dp
from vfr.common import CKKSParams, make_ckks, gadget_constants

ck = make_ckks(CKKSParams())
qs, gad = list(ck.qs), gadget_constants(ck)
q0, q1, q2 = qs[0] * qs[1], qs[2], qs[3]
N = 2 ** 14
print('chain bits', [round(math.log2(q), 3) for q in qs], 'P bits', round(math.log2(gad['P']), 3), gad)

def evaluate(n_bar, eps, f1=0.1, n_R_max=2000):
    ag = dp.Agreement(n_R_max=n_R_max, n_bar_O=n_bar, d_R=4, d_O=6, B_R=5.59, B_O=6.24, B_y=2.89,
                      eps_run=eps, f1=f1, kappa=40)
    cal = dp.calibrate(ag, N, q1, q2, q0, gad)
    OmU, Omu, OmV, Omf = float(cal['Omega_U']), float(cal['Omega_u']), float(cal['Omega_V']), cal['Omega_f']
    cp = bt.CircuitParams(N=N, n_bar=n_bar, d_R=4, d_O=6, B_R=5.59, B_O=6.24, B_y=2.89, q1=q1, q2=q2,
                          Omega_U=OmU, Omega_V=OmV, Omega_u=Omu, **gad)
    r, o, w = 5.59 ** 2, 6.24 ** 2, 2.89 ** 2
    m = -(-n_bar // (N // 2))
    out = dict(log2_Omega_f=math.log2(Omf))
    for s in (1, 2):
        nE = bt.entry_counts(4, 6, True)[s]
        Xi = bt.gamma_E(cp, s, True)
        nuU = bt.nu_U(cp, s, True)
        zeta = math.sqrt(o * o + r * o) if s == 1 else math.sqrt(o * o + r * (o + w) + o * w + w * w)
        Bt = bt.B_tot(cp, s, True, 'A')
        Delta = cal['releases'][str(s)]['Delta']
        Dx = bt.delta_x(cp, s, True, Delta)
        q12 = q1 * q2
        eps_x = Omu / q12 * N ** 1.5 * math.sqrt(nE) / (2 * math.sqrt(2)) + N ** 2.5 * nE / (8 * q12)
        for nmatch in (n_R_max, n_bar):          # App. B uses n_R^max, Ch. 9 uses n_max = n_bar (conservative)
            A = (N * math.sqrt(m) / (2 * math.sqrt(2)) * (math.sqrt(nmatch) * nuU / OmV + math.sqrt(n_bar * Xi) / OmU)
                 + math.sqrt(nE) * m * N ** 2 / (8 * OmU * OmV))
            dbar = A + N * math.sqrt(nE) / (2 * math.sqrt(2) * Omu) * (nmatch * zeta + A) + N / (2 * math.sqrt(2) * Omf)
            rbar = A * OmU * OmV
            dcoef = (math.sqrt(2 / N) * Omu / q12 * rbar
                     + math.sqrt(N) / (2 * q12) * math.sqrt(nE) * (OmU * OmV * nmatch * zeta + rbar))
            out[(s, nmatch)] = dict(dbar=dbar, log2_dmax_coef=math.log2(dcoef))
        out[s] = dict(log2_Btot=math.log2(Bt), log2_Dx=math.log2(Dx), log2_eps_x=math.log2(eps_x),
                      sigma_slot=cal['releases'][str(s)]['sigma_slot'])
    out['q0_need_bits'] = cal['q0_need_bits']
    return out

for n_bar in (4096, 16384, 10 ** 6):
    for eps in (1.0, 8.0):
        try:
            res = evaluate(n_bar, eps)
        except dp.CalibrationRefused as e:
            print(n_bar, eps, 'refused', e); continue
        print(f'n_bar={n_bar:>7} eps={eps}: log2 Omega_f={res["log2_Omega_f"]:.2f}  q0 need 2^{res["q0_need_bits"]:.1f}')
        for s in (1, 2):
            r_ = res[s]
            print(f'   stage {s}: log2 B_tot={r_["log2_Btot"]:.2f} log2 Delta_x={r_["log2_Dx"]:.2f} '
                  f'log2 eps_x={r_["log2_eps_x"]:.2f} sigma_slot={r_["sigma_slot"]:.1f}')
            for nm in (2000, n_bar):
                d = res[(s, nm)]
                print(f'      n_match<={nm:>7}: d_bar={d["dbar"]:.3e} (2^{math.log2(d["dbar"]):.1f}); '
                      f'coefficient-domain d_max=2^{d["log2_dmax_coef"]:.1f}')

# ---------------- brute force at toy size: exact rationals, N = 16 ----------------
def negacyclic_mul(a, b):
    n = len(a); c = [Fraction(0)] * n
    for i in range(n):
        if a[i] == 0: continue
        for j in range(n):
            k = i + j
            if k < n: c[k] += a[i] * b[j]
            else: c[k - n] -= a[i] * b[j]
    return c

def slots(f, n):                     # tau(f)_k = f(zeta^{5^k}), k < n/2
    import cmath
    out = []
    for k in range(n // 2):
        h = pow(5, k, 2 * n)
        z = cmath.exp(1j * math.pi * h / n)
        out.append(sum(float(f[i]) * z ** i for i in range(n)))
    return out

def tau_inv(zs, n):                  # real polynomial with given slots (inverse of slots), float
    import cmath
    coef = [0.0] * n
    for k, zk in enumerate(zs):
        h = pow(5, k, 2 * n)
        for i in range(n):
            coef[i] += 2 * (zk * cmath.exp(-1j * math.pi * h * i / n)).real / n
    return coef

def ecd(zs, Om, n):
    return [Fraction(round(c * Om)) for c in tau_inv(zs, n)]

def trace(f, n):
    return [Fraction(n, 2) * f[0] if i == 0 else (Fraction(n, 2) * f[n // 2] if i == n // 2 else Fraction(0))
            for i in range(n)]

def centre(U, V, Om, n, ks):         # U[e], V[e]: real vectors of length n/2 (one chunk); returns mu_hat
    OmU, OmV, Omu, q1_, q2_ = Om
    tot = [Fraction(0)] * n
    for e in range(len(U)):
        y = trace(negacyclic_mul(ecd(U[e], OmU, n), ecd(V[e], OmV, n)), n)
        mu_ = ecd([1.0 if k == ks[e] else 0.0 for k in range(n // 2)], Omu, n)
        p = negacyclic_mul(y, mu_)
        tot = [t + pi / (q1_ * q2_) for t, pi in zip(tot, p)]
    return tot

random.seed(1)
n = 16; Om = (2 ** 10, 2 ** 9, 2 ** 10, 2 ** 10, 2 ** 9)   # Omega_U, Omega_V, Omega_u, q1, q2 (decoupled)
OmU, OmV, Omu, q1_, q2_ = Om
Omf = Fraction(OmU * OmV * Omu, q1_ * q2_)
worst_ratio_off, worst_ratio_sens = 0.0, 0.0
for trial in range(30):
    nE = 3; ks = [0, 1, 2]
    U = [[random.choice([0.0, 1.0]) for _ in range(n // 2)] for _ in range(nE)]
    V = [[random.uniform(-1, 1) for _ in range(n // 2)] for _ in range(nE)]
    mu_hat = centre(U, V, Om, n, ks)
    mu = [Fraction(round(c)) for c in mu_hat]
    sl = slots(mu, n)
    z = [sum(U[e][t] * V[e][t] for t in range(n // 2)) for e in range(nE)]
    off = math.sqrt(sum((sl[ks[e]].real / float(Omf) - z[e]) ** 2 for e in range(nE)))
    # bound (prop:btot.offset) with m = 1, n_match = n/2, nu_U = sqrt(nE), Xi = nE, zeta = sqrt(nE)
    nmatch, nuU, Xi, zeta = n // 2, math.sqrt(nE), nE, math.sqrt(nE)
    A = n / (2 * math.sqrt(2)) * (math.sqrt(nmatch) * nuU / OmV + math.sqrt(n // 2 * Xi) / OmU) + math.sqrt(nE) * n ** 2 / (8 * OmU * OmV)
    dbar = A + n * math.sqrt(nE) / (2 * math.sqrt(2) * Omu) * (nmatch * zeta + A) + n / (2 * math.sqrt(2) * float(Omf))
    worst_ratio_off = max(worst_ratio_off, off / dbar)
    # centre sensitivity: change one row t* of V
    t = random.randrange(n // 2)
    V2 = [list(v) for v in V]
    for e in range(nE): V2[e][t] = random.uniform(-1, 1)
    mu2 = [Fraction(round(c)) for c in centre(U, V2, Om, n, ks)]
    diff = math.sqrt(sum(float(a - b) ** 2 for a, b in zip(mu, mu2)))
    Dl = math.sqrt(sum((U[e][t] * (V[e][t] - V2[e][t])) ** 2 for e in range(nE)))
    nuV = math.sqrt(sum((V[e][t] - V2[e][t]) ** 2 for e in range(nE)))
    q12 = q1_ * q2_
    Dx = (math.sqrt(n / 2) * Omu / q12 * (OmV * nuV + math.sqrt(n) * OmU * nuU)
          + math.sqrt(n) / 2 * OmU * OmV / q12 * math.sqrt(nE) * Dl
          + n ** 1.5 / (4 * q12) * math.sqrt(nE) * (OmV * nuV + math.sqrt(n) * OmU * nuU) + 1)
    eps_x = Omu / q12 * n ** 1.5 * math.sqrt(nE) / (2 * math.sqrt(2)) + n ** 2.5 * nE / (8 * q12)
    bound = math.sqrt(2 / n) * float(Omf) * Dl + Dx - 1 + eps_x + math.sqrt(n)
    worst_ratio_sens = max(worst_ratio_sens, diff / bound)
print(f'toy N=16, decoupled scales: max offset/d_bar = {worst_ratio_off:.3f}, max centre change/bound = {worst_ratio_sens:.3f}'
      ' (both must be <= 1)')
