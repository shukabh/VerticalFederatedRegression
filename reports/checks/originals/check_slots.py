# Does the ORIGINAL he_backend's CKKS inner_product + add_scalar leave every slot = total + z?
import sys, numpy as np
sys.path.insert(0, "/home/user/VerticalFederatedRegression/scenario_b")
import he_backend as hb
rng = np.random.default_rng(3)
for n_O, batch in [(12, 16), (3000, 4096)]:
    R = hb.ckks_owner(mult_depth=2, scale_bits=50, batch_size=batch, backend="openfhe")
    O = hb.ckks_from_public(R.public_blob(), backend="openfhe")
    b = (rng.random(n_O) < 0.4).astype(float); w = rng.random(n_O)
    ct_b = O.ct_from_bytes(R.ct_bytes(R.encrypt(b)))
    z = float(rng.normal(0, 10))
    out = O.add_scalar(O.inner_product(ct_b, w), z)
    back = R.ct_from_bytes(O.ct_bytes(out))
    pt = R.cc.Decrypt(back, R.sk); pt.SetLength(batch)
    slots = np.array([v.real for v in pt.GetCKKSPackedValue()])
    tot = float(b @ w)
    print(f"n_O={n_O} batch={batch} ring={R.cc.GetRingDimension()}  total+z={tot+z:.6f}  "
          f"slot0={slots[0]:.6f}  max|slot_j - slot0|={np.abs(slots-slots[0]).max():.2e}  "
          f"slot0-slot[last]={slots[0]-slots[-1]:.2e}")
