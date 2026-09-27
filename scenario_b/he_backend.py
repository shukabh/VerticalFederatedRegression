"""
he_backend.py — one API, two backends.

    backend="openfhe"    real BFV / CKKS via the OpenFHE Python bindings
    backend="plaintext"  exact numpy simulation with identical semantics

Both expose the same surface, so party_r.py / party_o.py are written once and the
backend is a flag. The plaintext backend lets the whole socket protocol be validated
end-to-end (alignment, masking, DP noise, bias correction) without paying HE cost;
the OpenFHE backend is the real thing.

Phase 0 uses BFV (exact mod-t arithmetic: zero test + exact label recovery).
Phase 1 uses CKKS (approximate reals: Gram / moment inner products).
R owns both secret keys and never sends them; O receives only a public bundle.
"""
from __future__ import annotations

import pickle
import numpy as np

try:
    import openfhe as ofhe
    _HAVE_OPENFHE = True
except Exception:                                    # pragma: no cover
    _HAVE_OPENFHE = False


def _insert_key(fn, blob):
    """Insert a deserialized eval key into OpenFHE's global key map.

    In the real two-process deployment O's map is empty and this simply loads the
    key. When R and O run in the SAME process (tests), the map already holds R's
    key under the same tag and OpenFHE refuses the insert; that refusal is benign
    because the key present is the very one being loaded, so we ignore it.
    """
    try:
        fn(blob, ofhe.BINARY)
    except RuntimeError as e:
        if "key vector for the given keyTag" not in str(e):
            raise


# ============================================================================
# Phase 0 — BFV
# ============================================================================

class _PlainBFV:
    """Exact simulation: a 'ciphertext' is an int64 vector mod t."""
    kind = "plaintext"

    def __init__(self, t, n_slots):
        self.t, self._slots = int(t), int(n_slots)

    # -- construction ---------------------------------------------------
    @classmethod
    def create_owner(cls, plain_modulus, mult_depth, n_slots, **kw):
        return cls(plain_modulus, n_slots)

    @classmethod
    def from_public(cls, blob, **kw):
        d = pickle.loads(blob)
        return cls(d["t"], d["slots"])

    def public_blob(self):
        return pickle.dumps({"t": self.t, "slots": self._slots})

    def n_slots(self):
        return self._slots

    # -- ops -------------------------------------------------------------
    def _vec(self, v):
        a = np.zeros(self._slots, dtype=object)
        a[:len(v)] = [int(x) % self.t for x in v]
        return a

    def encrypt(self, vec):      return self._vec(vec)
    def decrypt(self, ct, length):
        return [int(x) % self.t for x in ct[:length]]
    def mul_pt(self, ct, vec):   return (ct * self._vec(vec)) % self.t
    def mul_ct(self, a, b):      return (a * b) % self.t
    def add(self, a, b):         return (a + b) % self.t
    def add_pt(self, ct, vec):   return (ct + self._vec(vec)) % self.t

    def ct_bytes(self, ct):      return pickle.dumps(ct)
    def ct_from_bytes(self, blob): return pickle.loads(blob)


class _OpenFHEBFV:
    kind = "openfhe"

    def __init__(self, cc, pk, sk=None):
        self.cc, self.pk, self.sk = cc, pk, sk

    @classmethod
    def create_owner(cls, plain_modulus, mult_depth, n_slots=None, ring_dim=None, **kw):
        p = ofhe.CCParamsBFVRNS()
        p.SetPlaintextModulus(int(plain_modulus))
        p.SetMultiplicativeDepth(int(mult_depth))
        if ring_dim:
            p.SetRingDim(int(ring_dim))
        cc = ofhe.GenCryptoContext(p)
        for f in (ofhe.PKE, ofhe.KEYSWITCH, ofhe.LEVELEDSHE):
            cc.Enable(f)
        kp = cc.KeyGen()
        cc.EvalMultKeyGen(kp.secretKey)              # ct x ct for window powers
        return cls(cc, kp.publicKey, kp.secretKey)

    @classmethod
    def from_public(cls, blob, **kw):
        d = pickle.loads(blob)
        cc = ofhe.DeserializeCryptoContextString(d["cc"], ofhe.BINARY)
        _insert_key(ofhe.DeserializeEvalMultKeyString, d["mk"])
        pk = ofhe.DeserializePublicKeyString(d["pk"], ofhe.BINARY)
        return cls(cc, pk, None)

    def public_blob(self):
        return pickle.dumps({
            "cc": ofhe.Serialize(self.cc, ofhe.BINARY),
            "pk": ofhe.Serialize(self.pk, ofhe.BINARY),
            "mk": ofhe.SerializeEvalMultKeyString(ofhe.BINARY),
        })

    def n_slots(self):
        return self.cc.GetRingDimension()

    def _pt(self, vec):
        return self.cc.MakePackedPlaintext([int(x) for x in vec])

    def encrypt(self, vec):
        return self.cc.Encrypt(self.pk, self._pt(vec))

    def decrypt(self, ct, length):
        d = self.cc.Decrypt(ct, self.sk)
        d.SetLength(int(length))
        return list(d.GetPackedValue())

    def mul_pt(self, ct, vec):   return self.cc.EvalMult(ct, self._pt(vec))
    def mul_ct(self, a, b):      return self.cc.EvalMult(a, b)
    def add(self, a, b):         return self.cc.EvalAdd(a, b)
    def add_pt(self, ct, vec):   return self.cc.EvalAdd(ct, self._pt(vec))

    def ct_bytes(self, ct):      return ofhe.Serialize(ct, ofhe.BINARY)
    def ct_from_bytes(self, blob):
        return ofhe.DeserializeCiphertextString(blob, ofhe.BINARY)


# ============================================================================
# Phase 1 — CKKS
# ============================================================================

class _PlainCKKS:
    """Exact simulation: a 'ciphertext' is a float vector; slot 0 carries results."""
    kind = "plaintext"

    def __init__(self, n_slots):
        self._slots = int(n_slots)

    @classmethod
    def create_owner(cls, mult_depth, scale_bits, batch_size, **kw):
        return cls(batch_size)

    @classmethod
    def from_public(cls, blob, **kw):
        return cls(pickle.loads(blob)["slots"])

    def public_blob(self):
        return pickle.dumps({"slots": self._slots})

    def n_slots(self):
        return self._slots

    def _vec(self, v):
        a = np.zeros(self._slots, dtype=float)
        a[:len(v)] = np.asarray(v, dtype=float)
        return a

    def encrypt(self, vec):        return self._vec(vec)
    def decrypt_slot0(self, ct):   return float(ct[0])

    def inner_product(self, ct, vec):
        out = np.zeros(self._slots, dtype=float)
        out[0] = float(np.dot(ct, self._vec(vec)))
        return out

    def add_scalar(self, ct, s):
        out = ct.copy(); out[0] += float(s); return out

    def ct_bytes(self, ct):        return pickle.dumps(ct)
    def ct_from_bytes(self, blob): return pickle.loads(blob)


class _OpenFHECKKS:
    kind = "openfhe"

    def __init__(self, cc, pk, sk=None, batch=None):
        self.cc, self.pk, self.sk, self.batch = cc, pk, sk, batch

    @classmethod
    def create_owner(cls, mult_depth, scale_bits, batch_size, ring_dim=None, **kw):
        p = ofhe.CCParamsCKKSRNS()
        p.SetMultiplicativeDepth(int(mult_depth))
        p.SetScalingModSize(int(scale_bits))
        p.SetBatchSize(int(batch_size))
        if ring_dim:
            p.SetRingDim(int(ring_dim))
        cc = ofhe.GenCryptoContext(p)
        for f in (ofhe.PKE, ofhe.KEYSWITCH, ofhe.LEVELEDSHE, ofhe.ADVANCEDSHE):
            cc.Enable(f)
        kp = cc.KeyGen()
        cc.EvalMultKeyGen(kp.secretKey)
        cc.EvalSumKeyGen(kp.secretKey)               # rotate-and-add for inner products
        return cls(cc, kp.publicKey, kp.secretKey, int(batch_size))

    @classmethod
    def from_public(cls, blob, **kw):
        d = pickle.loads(blob)
        cc = ofhe.DeserializeCryptoContextString(d["cc"], ofhe.BINARY)
        _insert_key(ofhe.DeserializeEvalMultKeyString, d["mk"])
        _insert_key(ofhe.DeserializeEvalAutomorphismKeyString, d["ak"])
        pk = ofhe.DeserializePublicKeyString(d["pk"], ofhe.BINARY)
        return cls(cc, pk, None, d["batch"])

    def public_blob(self):
        return pickle.dumps({
            "cc": ofhe.Serialize(self.cc, ofhe.BINARY),
            "pk": ofhe.Serialize(self.pk, ofhe.BINARY),
            "mk": ofhe.SerializeEvalMultKeyString(ofhe.BINARY),
            "ak": ofhe.SerializeEvalAutomorphismKeyString(ofhe.BINARY),
            "batch": self.batch,
        })

    def n_slots(self):
        return self.batch

    def encrypt(self, vec):
        return self.cc.Encrypt(self.pk, self.cc.MakeCKKSPackedPlaintext(
            [float(x) for x in vec]))

    def decrypt_slot0(self, ct):
        d = self.cc.Decrypt(ct, self.sk)
        d.SetLength(1)
        return float(d.GetCKKSPackedValue()[0].real)

    def inner_product(self, ct, vec):
        pt = self.cc.MakeCKKSPackedPlaintext([float(x) for x in vec])
        return self.cc.EvalInnerProduct(ct, pt, self.batch)

    def add_scalar(self, ct, s):
        return self.cc.EvalAdd(ct, float(s))         # ct + scalar: no level matching

    def ct_bytes(self, ct):        return ofhe.Serialize(ct, ofhe.BINARY)
    def ct_from_bytes(self, blob):
        return ofhe.DeserializeCiphertextString(blob, ofhe.BINARY)


# ============================================================================
# Factories
# ============================================================================

def _pick(backend):
    if backend == "openfhe" and not _HAVE_OPENFHE:
        raise RuntimeError("backend='openfhe' but the openfhe module is not importable")
    return backend


def bfv_owner(plain_modulus, mult_depth, n_slots=None, ring_dim=None, backend="openfhe"):
    cls = _OpenFHEBFV if _pick(backend) == "openfhe" else _PlainBFV
    return cls.create_owner(plain_modulus=plain_modulus, mult_depth=mult_depth,
                            n_slots=n_slots, ring_dim=ring_dim)


def bfv_from_public(blob, backend="openfhe"):
    cls = _OpenFHEBFV if _pick(backend) == "openfhe" else _PlainBFV
    return cls.from_public(blob)


def ckks_owner(mult_depth, scale_bits, batch_size, ring_dim=None, backend="openfhe"):
    cls = _OpenFHECKKS if _pick(backend) == "openfhe" else _PlainCKKS
    return cls.create_owner(mult_depth=mult_depth, scale_bits=scale_bits,
                            batch_size=batch_size, ring_dim=ring_dim)


def ckks_from_public(blob, backend="openfhe"):
    cls = _OpenFHECKKS if _pick(backend) == "openfhe" else _PlainCKKS
    return cls.from_public(blob)
