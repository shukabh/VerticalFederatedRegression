import sys, math
sys.path.insert(0, "/home/user/VerticalFederatedRegression/scenario_b")
import openfhe as ofhe
import he_backend as hb
def logq(cc):
    m = cc.GetModulus()
    try: return math.log2(float(m))
    except Exception: return str(m)[:40]
for bits, depth in [(29,4),(48,4),(59,4),(59,8)]:
    # smallest prime t = 1 mod 32768 with the given bit length
    k = (1 << (bits-1)) // 32768 + 1
    while True:
        t = k*32768+1
        if all(t % q for q in range(3, 2000, 2)) and pow(2, t-1, t) == 1: break
        k += 1
    for rd in (16384, None):
        try:
            be = hb.bfv_owner(plain_modulus=t, mult_depth=depth, ring_dim=rd, backend="openfhe")
            print(f"t~2^{bits} depth={depth} ring_req={rd}: ring={be.cc.GetRingDimension()} log2Q={logq(be.cc):.1f}")
        except Exception as e:
            print(f"t~2^{bits} depth={depth} ring_req={rd}: REFUSED ({str(e).splitlines()[0][:110]})")
print("default security level:", ofhe.CCParamsBFVRNS().GetSecurityLevel())
