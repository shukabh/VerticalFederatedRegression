"""Defensive check for risk C2 (crypto_layer.md): what does an encrypted inner product
leave in the slots OTHER than slot 0, which the key holder R can also decrypt?

he_backend.inner_product was not provided, so three plausible summation routines are
compared on SEAL-CKKS (via tenseal.sealapi):

  (a) "suffix" : rotate-and-add with steps size/2, ..., 2, 1 where size = next pow2 >= n
                 (what TenSEAL's CKKSVector.dot / sum does)
  (b) "cyclic" : rotate-and-add over ALL N/2 slots (log2(N/2) rotations) -> total in
                 every slot
  (c) "masked" : (a) followed by a one-hot plaintext e_0 multiply (one extra level)

In every case O then adds the DP noise z as a scalar to all slots, as party_o.py does
with ckks.add_scalar.  Run time: a few seconds.
"""
import numpy as np
import tenseal.sealapi as sa

N = 8192
SLOTS = N // 2
SCALE = 2.0 ** 40

parms = sa.EncryptionParameters(sa.SCHEME_TYPE.CKKS)
parms.set_poly_modulus_degree(N)
parms.set_coeff_modulus(sa.CoeffModulus.Create(N, [60, 40, 40, 60]))
ctx = sa.SEALContext(parms, True, sa.SEC_LEVEL_TYPE.TC128)
kg = sa.KeyGenerator(ctx)
pk = sa.PublicKey(); kg.create_public_key(pk)
gk = sa.GaloisKeys(); kg.create_galois_keys(gk)
enc, dec = sa.Encryptor(ctx, pk), sa.Decryptor(ctx, kg.secret_key())
ev, cod = sa.Evaluator(ctx), sa.CKKSEncoder(ctx)


def encode(vec, ref=None, scale=SCALE):
    v = np.zeros(SLOTS); v[:len(vec)] = vec
    pt = sa.Plaintext(); cod.encode(v.tolist(), scale, pt)
    if ref is not None:
        ev.mod_switch_to_inplace(pt, ref.parms_id())
    return pt


def decrypt_all(ct):
    pt = sa.Plaintext(); dec.decrypt(ct, pt)
    return np.array(cod.decode_double(pt))


def rot_add(ct, steps):
    for k in steps:
        r = sa.Ciphertext(); ev.rotate_vector(ct, k, gk, r); ev.add_inplace(ct, r)
    return ct


def add_scalar(ct, z):                       # z in every slot, like ckks.add_scalar
    pt = sa.Plaintext(); cod.encode(float(z), ct.scale, pt)
    ev.mod_switch_to_inplace(pt, ct.parms_id()); ev.add_plain_inplace(ct, pt)
    return ct


rng = np.random.default_rng(1)
n_O = 12
b = (rng.random(n_O) < 0.5).astype(float)     # R's selection vector (R encrypts)
w = rng.uniform(0, 1, n_O)                    # stand-in for an O column product (plaintext)
z = rng.normal(0, 5.0)                        # O's DP noise draw
truth = b @ w + z

ct_b = sa.Ciphertext(); enc.encrypt(encode(b), ct_b)


def product():
    ct = sa.Ciphertext(); ev.multiply_plain(ct_b, encode(w, ct_b), ct)
    ev.rescale_to_next_inplace(ct)
    return ct


size = 1 << int(np.ceil(np.log2(n_O)))
variants = {}
variants["suffix"] = add_scalar(rot_add(product(), [size >> i for i in range(1, int(np.log2(size)) + 1)]), z)
variants["cyclic"] = add_scalar(rot_add(product(), [1 << i for i in range(int(np.log2(SLOTS)))]), z)
ct = rot_add(product(), [size >> i for i in range(1, int(np.log2(size)) + 1)])
e0 = encode([1.0], ct); ev.multiply_plain_inplace(ct, e0); ev.rescale_to_next_inplace(ct)
ct.scale = SCALE                              # SEAL: normalise scale bookkeeping
variants["masked+bcast"] = add_scalar(ct, z)     # mask, but z still broadcast to all slots

def add_slot0(ct, z):                          # z in slot 0 ONLY
    pt = sa.Plaintext(); v = np.zeros(SLOTS); v[0] = z
    cod.encode(v.tolist(), ct.scale, pt)
    ev.mod_switch_to_inplace(pt, ct.parms_id()); ev.add_plain_inplace(ct, pt)
    return ct

ct = rot_add(product(), [size >> i for i in range(1, int(np.log2(size)) + 1)])
e0 = encode([1.0], ct); ev.multiply_plain_inplace(ct, e0); ev.rescale_to_next_inplace(ct)
ct.scale = SCALE
variants["masked+slot0"] = add_slot0(ct, z)    # the correct fix

print(f"n_O={n_O}  matched={int(b.sum())}  b.w+z = {truth:.6f}\n")
print(f"true noise-free aggregate b.w = {b @ w:.6f}   DP noise z = {z:.6f}\n")
for name, c in variants.items():
    s = decrypt_all(c)
    d = s[:n_O] - s[1:n_O + 1]                # slot j - slot j+1 : z cancels
    err = np.max(np.abs(d - b * w))
    far = s[SLOTS // 2]                       # a slot far outside the data window
    print(f"[{name:12s}] slot0={s[0]:10.6f}  slot[{SLOTS//2}]={far:10.6f}  "
          f"slot0 - slot[{SLOTS//2}]={s[0]-far:10.6f}  "
          f"max|(s_j - s_j+1) - b_j w_j|={err:8.2e}")
    if name == "suffix":
        print("          slot_j - slot_j+1 :", np.round(d[:8], 6))
        print("          b_j * w_j         :", np.round((b * w)[:8], 6))
print("""
Reading:
 suffix       : slot differences equal b_j*w_j (one record per slot pair, z cancels) AND any
                slot outside the summation window holds exactly z, so slot0 - that slot is
                the NOISE-FREE aggregate b.w.  DP is void against R.
 cyclic       : total replicated in every slot, no per-record structure in the real parts.
 masked+bcast : partial sums gone, but slots 1.. hold z alone -> noise-free b.w again.
 masked+slot0 : correct fix; other slots ~0, slot0 = b.w + z.
 Residual error-driven content (imaginary parts, ciphertext structure) is C1's subject.""")
