"""(1) A proved bound on TV_1(sigma) = TV(discrete Gaussian D_{Z,sigma}, round(N(0, sigma^2))), the constant of the coupling
of the stage-1 radius (book lem:rel.slot, cor:inf.transfer), which the sources left numerical:
      TV_1(sigma) <= phi(1)/(12 sigma^2) + V/(48 sigma^3) + eps(sigma),
  V = total variation of |phi''| over R for the standard normal density (= 2(phi(0) + 2*2 phi(sqrt 3))),
  eps(sigma) = sum_{j>=1} exp(-2 pi^2 sigma^2 j^2) (Poisson summation; <= 1.0001 exp(-2 pi^2 sigma^2) for sigma >= 1).
  Proof: TV = (1/2) sum_k |p_k - q_k|; |p_k - phi_sigma(k)| sums to |1 - sum_k phi_sigma(k)| = 2 eps; |q_k - phi_sigma(k)|
  <= sup_{[k-1/2,k+1/2]} |phi_sigma''| / 24 (Taylor, the linear term integrates to 0); and sum_k sup_{I_k} g <= int g + TV(g).
  The script checks the bound against an exact high-precision evaluation of TV_1 on a grid of sigma.
(2) A proved bound on the probability that no cuckoo assignment exists (Hall's condition), for n items with h independent
  uniform candidate bins among m:  p_noassign <= sum_{s=2}^n C(n,s) C(m,s-1) ((s-1)/m)^{hs}.
"""
import math
from mpmath import mp, mpf, exp, sqrt, pi, erf, quad, inf
mp.dps = 50
phi = lambda x: exp(-x * x / 2) / sqrt(2 * pi)
# total variation of |phi''| (standard normal): local maxima of |phi''(x)| = phi(x)|x^2-1| at x=0 and x=+-sqrt(3), zeros at +-1
V = 2 * (phi(0) + 2 * phi(sqrt(3)) * 2)
c2 = phi(1) / 12            # = (1/2)(1/24) int |phi''| = (1/48) 4 phi(1)
c3 = V / 48
print('int|phi_1\'\'| = 4 phi(1) =', float(4 * phi(1)), ' check:', float(quad(lambda x: abs(phi(x) * (x * x - 1)), [-inf, -1, 0, 1, inf])))
print(f'TV_1(sigma) <= {float(c2):.6f}/sigma^2 + {float(c3):.6f}/sigma^3 + eps(sigma)')

def tv1_exact(s):
    s = mpf(s); K = int(40 * s) + 5
    Z = sum(exp(-mpf(k) ** 2 / (2 * s * s)) for k in range(-K, K + 1))
    Phi = lambda x: (1 + erf(x / (s * sqrt(2)))) / 2
    tot = mpf(0)
    for k in range(-K, K + 1):
        p = exp(-mpf(k) ** 2 / (2 * s * s)) / Z
        q = Phi(k + mpf(1) / 2) - Phi(k - mpf(1) / 2)
        tot += abs(p - q)
    tail = 2 * (1 - Phi(K + mpf(1) / 2))
    return (tot + tail) / 2

worst = 0
for s in [1, 1.5, 2, 3, 5, 10, 20, 50, 100]:
    ex = tv1_exact(s)
    eps = sum(exp(-2 * pi ** 2 * mpf(s) ** 2 * j * j) for j in range(1, 6))
    bd = c2 / mpf(s) ** 2 + c3 / mpf(s) ** 3 + eps
    worst = max(worst, float(ex / bd))
    print(f'sigma={s:>5}: sigma^2 TV_1 = {float(ex * s * s):.6f}; bound * sigma^2 = {float(bd * s * s):.6f}; ratio = {float(ex/bd):.4f}')
print('max exact/bound =', round(worst, 4), '(must be <= 1)')
s_c = mpf(2) ** mpf(65.29)
print('N TV_1 at N=2^14, sigma_c=2^65.29: log2 <=', float(mp.log(2 ** 14 * (c2 / s_c ** 2 + c3 / s_c ** 3), 2)))

# ---------------- cuckoo: Hall union bound
def hall_bound(n, m, h):
    tot = 0.0
    for s in range(2, n + 1):
        lt = (math.lgamma(n + 1) - math.lgamma(s + 1) - math.lgamma(n - s + 1)
              + math.lgamma(m + 1) - math.lgamma(s) - math.lgamma(m - s + 2) + h * s * math.log((s - 1) / m))
        tot += math.exp(min(lt, 700.0))   # every term is summed: the terms rise again for linear s
    return tot
for n in (2000, 4096, 8192):
    b = hall_bound(n, 16384, 3)
    print(f'n={n} items, m=16384 bins, h=3: P[no assignment] <= {b:.3e} (2^{math.log2(b):.1f})')
lo, hi = 2000, 14745
while hi - lo > 1:
    mid = (lo + hi) // 2
    if hall_bound(mid, 16384, 3) <= 2.0 ** -40: lo = mid
    else: hi = mid
print(f'largest n with bound <= 2^-40 at m=16384, h=3: {lo} (load {lo/16384:.3f})')
