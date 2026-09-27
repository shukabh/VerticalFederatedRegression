"""
d1_ckks_circuit_privacy.py -- does O's Phase-4 reply reveal more than the DP-noised aggregate?

Reproduces EXACTLY the calls of he_backend._OpenFHECKKS / party_o.py:
    R:  ckks_owner(mult_depth=2, scale_bits=50, batch_size=2^ceil(log2 n_O)); Enc(b)
    O:  ct = EvalInnerProduct(Enc(b), w, batch);  ct = EvalAdd(ct, z)   (add_scalar)
with w = X_O[:,k]*y (the c_O statistic) and z ~ N(0, sigma^2) the DP noise.

Tests
  T1  every slot of the reply holds the same value (no partial-sum leak through EvalSum)
  T2  O's evaluation is deterministic (no fresh randomness besides z)
  T3  the c1 polynomial of the reply does not depend on z at all
  T4  DP-distinguishing attack: R knows every O record except one, whose (x,y) is one
      of two adjacent candidates. R recomputes O's circuit on each candidate with its
      own public keys and compares c1 byte-for-byte. Run for a MATCHED record (b=1:
      the adjacency of Def. 4) and for an UNMATCHED record (b=0: a person outside the
      intersection whose data should not influence R's view at all).
  T5  the same attack when O instead adds a FRESH encryption Enc(pk, z) (what paper
      eq. (5) literally says). c1 is then re-randomised, so the byte test fails; the
      remaining error-polynomial channel still needs noise flooding (see d2).
"""
import sys, time, pickle
import numpy as np
import openfhe as o

rng = np.random.default_rng(1)
n_O = 3000
batch = 1 << int(np.ceil(np.log2(n_O)))
sigma = 85.0                       # order of the calibrated sigma for the default data

# ---------------- R: key generation and upload (as party_r.py) ----------------
p = o.CCParamsCKKSRNS(); p.SetMultiplicativeDepth(2); p.SetScalingModSize(50)
p.SetBatchSize(batch)
cc = o.GenCryptoContext(p)
for f in (o.PKE, o.KEYSWITCH, o.LEVELEDSHE, o.ADVANCEDSHE):
    cc.Enable(f)
kp = cc.KeyGen(); cc.EvalMultKeyGen(kp.secretKey); cc.EvalSumKeyGen(kp.secretKey)
print(f"ring dim {cc.GetRingDimension()}  batch {batch}  n_O {n_O}")

b = np.zeros(n_O); matched = rng.choice(n_O, 400, replace=False); b[matched] = 1
ct_b = cc.Encrypt(kp.publicKey, cc.MakeCKKSPackedPlaintext([float(v) for v in b]))

# ---------------- O: its private column --------------------------------------
x = rng.uniform(0, 1, n_O); y = rng.normal(2, 1, n_O)
w = x * y                                              # c_O column for one feature


def O_reply(ct, wvec, z, fresh=False):
    pt = cc.MakeCKKSPackedPlaintext([float(v) for v in wvec])
    out = cc.EvalInnerProduct(ct, pt, batch)
    if fresh:
        zc = cc.Encrypt(kp.publicKey, cc.MakeCKKSPackedPlaintext([float(z)] * batch))
        return cc.EvalAdd(out, zc)
    return cc.EvalAdd(out, float(z))                   # == he_backend.add_scalar


def c1_bytes(ct):
    """Serialize a ciphertext whose BOTH elements are ct's c1 -> bytes identify c1."""
    t = ct.Clone(); e = t.GetElements(); t.SetElements([e[1], e[1]])
    return o.Serialize(t, o.BINARY)


def slots(ct, n):
    d = cc.Decrypt(ct, kp.secretKey); d.SetLength(n)
    return np.array([v.real for v in d.GetCKKSPackedValue()])


# T1
z = rng.normal(0, sigma)
ct_out = O_reply(ct_b, w, z)
v = slots(ct_out, batch)
true = float(b @ w)
print(f"\nT1 slot spread: max|slot_i - slot_0| = {np.max(np.abs(v - v[0])):.2e}; "
      f"slot0 - (s+z) = {v[0] - (true + z):.2e}  -> all slots carry s+z (no partial sums)")

# T2
a1 = o.Serialize(O_reply(ct_b, w, z), o.BINARY)
a2 = o.Serialize(O_reply(ct_b, w, z), o.BINARY)
print(f"T2 two evaluations with same inputs byte-identical: {a1 == a2}")

# T3
z2 = rng.normal(0, sigma)
print(f"T3 c1(reply | z) == c1(reply | z') for z={z:.2f}, z'={z2:.2f}: "
      f"{c1_bytes(O_reply(ct_b, w, z)) == c1_bytes(O_reply(ct_b, w, z2))}")


# T4 / T5 : distinguishing attack on adjacent datasets
def attack(idx, trials, fresh=False):
    correct = 0
    for t in range(trials):
        # two adjacent datasets differ only in record idx: (x,y) -> (x',y')
        w0 = w.copy(); w1 = w.copy()
        w1[idx] = rng.uniform(0, 1) * rng.normal(2, 1)
        truth = rng.integers(2)
        wt = w0 if truth == 0 else w1
        reply = O_reply(ct_b, wt, rng.normal(0, sigma), fresh=fresh)   # fresh DP noise each run
        # R's test: recompute the circuit on candidate 0 with zero noise, compare c1
        c1r = c1_bytes(reply)
        guess = 0 if c1_bytes(O_reply(ct_b, w0, 0.0, fresh=False)) == c1r else 1
        correct += (guess == truth)
    return correct / trials


T = 20
i_m = int(matched[0]); i_u = int(np.setdiff1d(np.arange(n_O), matched)[0])
t0 = time.time()
acc_m = attack(i_m, T)
acc_u = attack(i_u, T)
print(f"\nT4 add_scalar (as implemented), {T} trials each, fresh DP noise every trial:")
print(f"   matched record   (b=1): R identifies the true dataset in {acc_m:.0%} of trials")
print(f"   UNMATCHED record (b=0): R identifies the true dataset in {acc_u:.0%} of trials")
print(f"   (an (eps,delta)-DP view would cap the success rate at "
      f"(e^eps+delta)/(1+e^eps) = {np.e/(1+np.e):.0%} for eps=1)")
hits = sum(c1_bytes(O_reply(ct_b, w, rng.normal(0, sigma), fresh=True))
           == c1_bytes(O_reply(ct_b, w, 0.0)) for _ in range(T))
print(f"T5 fresh Enc(z) instead of add_scalar: c1 of the reply equals R's recomputation "
      f"(correct candidate) in {hits}/{T} trials -> byte test defeated; c1 is "
      f"re-randomised, but the error-polynomial channel remains (see d2)")
print(f"({time.time()-t0:.0f}s)")
