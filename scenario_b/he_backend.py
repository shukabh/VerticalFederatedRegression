"""
he_backend.py -- homomorphic-encryption backends for Phase 0 (BFV) and Phase 1 (CKKS).

##############################################################################
#  STAND-IN written from call sites in party_r.py/party_o.py; the user's      #
#  original module was not available. Replace with the original.             #
##############################################################################

Factory functions (as called by the party files)
------------------------------------------------
  bfv_owner(plain_modulus, mult_depth, n_slots, ring_dim, backend)   # R: holds sk
  bfv_from_public(blob, backend)                                     # O: public ctx
  ckks_owner(mult_depth, scale_bits, batch_size, backend)            # R: holds sk
  ckks_from_public(blob, backend)                                    # O: public ctx

Methods used
------------
  BFV  owner : n_slots, public_blob, encrypt(list[int]), decrypt(ct, n) -> list[int],
               ct_bytes, ct_from_bytes
  BFV  public: ct_from_bytes, ct_bytes, mul_ct(ct, ct), mul_pt(ct, list[int]),
               add(ct, ct), add_pt(ct, list[int])
  CKKS owner : n_slots, public_blob, encrypt(vec), decrypt_slot0(ct), ct_bytes,
               ct_from_bytes
  CKKS public: ct_from_bytes, ct_bytes, inner_product(ct, w) (sum_i ct_i w_i in
               slot 0), add_scalar(ct, s)

Backends
--------
  "plaintext"  NO ENCRYPTION. A "ciphertext" is the plaintext vector itself,
               pickled on the wire. BFV arithmetic is exact modulo t using Python
               ints (object arrays; t may be up to 60 bits, so int64 products would
               overflow). CKKS arithmetic is exact float64 (no CKKS approximation
               error is simulated). Use for functional testing only.
  "openfhe"    Real RLWE encryption via openfhe-python (BFVrns with batching;
               CKKSrns with EvalInnerProduct/EvalSum). The owner generates the key
               pair, relinearization (eval-mult) keys and, for CKKS, the EvalSum
               rotation keys, and serializes {crypto context, public key, eval keys}
               into public_blob. The secret key never leaves the owner object.
               openfhe-python wheels exist for CPython 3.10 (Ubuntu 22.04 build) and
               >= 3.12 (Ubuntu 24.04 build) but not 3.11.
"""
from __future__ import annotations

import pickle
from typing import Sequence

import numpy as np

BACKENDS = ("plaintext", "openfhe")


def _check_backend(backend):
    if backend not in BACKENDS:
        raise ValueError(f"unknown HE backend {backend!r}; choose one of {BACKENDS}")


# ============================================================================
# plaintext backend (no security; exact arithmetic)
# ============================================================================

class _PlainBFV:
    """Slot-wise arithmetic mod t on length-n_slots vectors of Python ints."""

    def __init__(self, t, n_slots, ring_dim, mult_depth):
        self.t = int(t)
        self._n = int(max(n_slots, 1))
        self.ring_dim = int(ring_dim)
        self.mult_depth = int(mult_depth)

    # -- helpers --
    def _vec(self, values) -> np.ndarray:
        v = np.zeros(self._n, dtype=object)
        vals = [int(x) % self.t for x in values]
        if len(vals) > self._n:
            raise ValueError(f"{len(vals)} values > {self._n} slots")
        v[:len(vals)] = vals
        return v

    # -- common API --
    def n_slots(self):
        return self._n

    def public_blob(self):
        return pickle.dumps({"scheme": "BFV", "backend": "plaintext", "t": self.t,
                             "n_slots": self._n, "ring_dim": self.ring_dim,
                             "mult_depth": self.mult_depth})

    def ct_bytes(self, ct):
        return pickle.dumps(ct, protocol=pickle.HIGHEST_PROTOCOL)

    def ct_from_bytes(self, blob):
        return pickle.loads(blob)

    # -- owner --
    def encrypt(self, values):
        return self._vec(values)

    def decrypt(self, ct, n=None):
        n = self._n if n is None else int(n)
        return [int(x) for x in ct[:n]]

    # -- evaluator --
    def mul_ct(self, a, b):
        return (a * b) % self.t

    def mul_pt(self, ct, values):
        return (ct * self._vec(values)) % self.t

    def add(self, a, b):
        return (a + b) % self.t

    def add_pt(self, ct, values):
        return (ct + self._vec(values)) % self.t


class _PlainCKKS:
    def __init__(self, batch_size, mult_depth, scale_bits):
        self._n = int(batch_size)
        self.mult_depth = int(mult_depth)
        self.scale_bits = int(scale_bits)

    def _vec(self, values) -> np.ndarray:
        values = np.asarray(values, dtype=float).ravel()
        if values.size > self._n:
            raise ValueError(f"{values.size} values > {self._n} slots")
        v = np.zeros(self._n)
        v[:values.size] = values
        return v

    def n_slots(self):
        return self._n

    def public_blob(self):
        return pickle.dumps({"scheme": "CKKS", "backend": "plaintext",
                             "batch_size": self._n, "mult_depth": self.mult_depth,
                             "scale_bits": self.scale_bits})

    def ct_bytes(self, ct):
        return pickle.dumps(ct, protocol=pickle.HIGHEST_PROTOCOL)

    def ct_from_bytes(self, blob):
        return pickle.loads(blob)

    def encrypt(self, values):
        return self._vec(values)

    def decrypt_slot0(self, ct):
        return float(ct[0])

    def inner_product(self, ct, w):
        # EvalInnerProduct semantics: the full sum lands in (every) slot
        return np.full(self._n, float(np.dot(ct, self._vec(w))))

    def add_scalar(self, ct, s):
        return ct + float(s)


# ============================================================================
# openfhe backend
# ============================================================================

def _insert_keys(fn, blob, o):
    """Deserialize eval keys into OpenFHE's static key map. When owner and evaluator
    live in the SAME process (tests), the map already holds these keys under the
    same tag and OpenFHE refuses to overwrite; that case is benign and ignored."""
    try:
        fn(blob, o.BINARY)
    except RuntimeError as e:
        if "for the given keyTag" not in str(e) and "already" not in str(e):
            raise


def _ofhe():
    try:
        import openfhe  # noqa: F401
    except Exception as e:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "backend 'openfhe' needs openfhe-python (pip install openfhe; wheels exist "
            "for CPython 3.10 and >= 3.12, not 3.11). Import failed: %r" % (e,)) from e
    return openfhe


class _OfheBFV:
    def __init__(self, cc, pk, sk, t, n_slots):
        self.o = _ofhe()
        self.cc, self.pk, self.sk = cc, pk, sk
        self.t = int(t)
        self._n = int(n_slots)

    @classmethod
    def owner(cls, t, mult_depth, n_slots, ring_dim):
        o = _ofhe()
        params = o.CCParamsBFVRNS()
        params.SetPlaintextModulus(int(t))
        params.SetMultiplicativeDepth(int(mult_depth))
        if ring_dim:
            params.SetRingDim(int(ring_dim))
        cc = o.GenCryptoContext(params)
        for feat in (o.PKESchemeFeature.PKE, o.PKESchemeFeature.KEYSWITCH,
                     o.PKESchemeFeature.LEVELEDSHE):
            cc.Enable(feat)
        keys = cc.KeyGen()
        cc.EvalMultKeyGen(keys.secretKey)                 # relinearization for ct x ct
        slots = cc.GetRingDimension()
        return cls(cc, keys.publicKey, keys.secretKey, t, slots)

    @classmethod
    def from_public(cls, blob):
        o = _ofhe()
        d = pickle.loads(blob)
        cc = o.DeserializeCryptoContextString(d["cc"], o.BINARY)
        pk = o.DeserializePublicKeyString(d["pk"], o.BINARY)
        _insert_keys(o.DeserializeEvalMultKeyString, d["emk"], o)
        return cls(cc, pk, None, d["t"], d["n_slots"])

    def _centered(self, values):
        t, h = self.t, self.t // 2
        out = []
        for x in values:
            x = int(x) % t
            out.append(x - t if x > h else x)
        return out

    def _pt(self, values):
        return self.cc.MakePackedPlaintext(self._centered(values))

    def n_slots(self):
        return self._n

    def public_blob(self):
        o = self.o
        return pickle.dumps({
            "scheme": "BFV", "backend": "openfhe", "t": self.t, "n_slots": self._n,
            "cc": o.Serialize(self.cc, o.BINARY),
            "pk": o.Serialize(self.pk, o.BINARY),
            "emk": o.SerializeEvalMultKeyString(o.BINARY, self.sk.GetKeyTag()),
        })

    def ct_bytes(self, ct):
        return self.o.Serialize(ct, self.o.BINARY)

    def ct_from_bytes(self, blob):
        return self.o.DeserializeCiphertextString(blob, self.o.BINARY)

    def encrypt(self, values):
        return self.cc.Encrypt(self.pk, self._pt(values))

    def decrypt(self, ct, n=None):
        if self.sk is None:
            raise RuntimeError("public context cannot decrypt")
        n = self._n if n is None else int(n)
        pt = self.cc.Decrypt(self.sk, ct)
        pt.SetLength(n)
        return [int(v) for v in pt.GetPackedValue()[:n]]

    def mul_ct(self, a, b):
        return self.cc.EvalMult(a, b)

    def mul_pt(self, ct, values):
        return self.cc.EvalMult(ct, self._pt(values))

    def add(self, a, b):
        return self.cc.EvalAdd(a, b)

    def add_pt(self, ct, values):
        return self.cc.EvalAdd(ct, self._pt(values))


class _OfheCKKS:
    def __init__(self, cc, pk, sk, batch_size):
        self.o = _ofhe()
        self.cc, self.pk, self.sk = cc, pk, sk
        self._n = int(batch_size)

    @classmethod
    def owner(cls, mult_depth, scale_bits, batch_size):
        o = _ofhe()
        params = o.CCParamsCKKSRNS()
        params.SetMultiplicativeDepth(int(mult_depth))
        params.SetScalingModSize(int(scale_bits))
        params.SetBatchSize(int(batch_size))
        cc = o.GenCryptoContext(params)
        for feat in (o.PKESchemeFeature.PKE, o.PKESchemeFeature.KEYSWITCH,
                     o.PKESchemeFeature.LEVELEDSHE, o.PKESchemeFeature.ADVANCEDSHE):
            cc.Enable(feat)
        keys = cc.KeyGen()
        cc.EvalMultKeyGen(keys.secretKey)
        cc.EvalSumKeyGen(keys.secretKey)                  # rotations for EvalSum
        return cls(cc, keys.publicKey, keys.secretKey, batch_size)

    @classmethod
    def from_public(cls, blob):
        o = _ofhe()
        d = pickle.loads(blob)
        cc = o.DeserializeCryptoContextString(d["cc"], o.BINARY)
        pk = o.DeserializePublicKeyString(d["pk"], o.BINARY)
        _insert_keys(o.DeserializeEvalMultKeyString, d["emk"], o)
        _insert_keys(o.DeserializeEvalAutomorphismKeyString, d["eak"], o)
        return cls(cc, pk, None, d["batch_size"])

    def _vals(self, values):
        v = np.zeros(self._n)
        values = np.asarray(values, dtype=float).ravel()
        if values.size > self._n:
            raise ValueError(f"{values.size} values > {self._n} slots")
        v[:values.size] = values
        return v.tolist()

    def n_slots(self):
        return self._n

    def public_blob(self):
        o = self.o
        return pickle.dumps({
            "scheme": "CKKS", "backend": "openfhe", "batch_size": self._n,
            "cc": o.Serialize(self.cc, o.BINARY),
            "pk": o.Serialize(self.pk, o.BINARY),
            "emk": o.SerializeEvalMultKeyString(o.BINARY, self.sk.GetKeyTag()),
            "eak": o.SerializeEvalAutomorphismKeyString(o.BINARY, self.sk.GetKeyTag()),
        })

    def ct_bytes(self, ct):
        return self.o.Serialize(ct, self.o.BINARY)

    def ct_from_bytes(self, blob):
        return self.o.DeserializeCiphertextString(blob, self.o.BINARY)

    def encrypt(self, values):
        return self.cc.Encrypt(self.pk, self.cc.MakeCKKSPackedPlaintext(self._vals(values)))

    def decrypt_slot0(self, ct):
        if self.sk is None:
            raise RuntimeError("public context cannot decrypt")
        pt = self.cc.Decrypt(self.sk, ct)
        pt.SetLength(1)
        return float(pt.GetRealPackedValue()[0])

    def inner_product(self, ct, w):
        pt = self.cc.MakeCKKSPackedPlaintext(self._vals(w))
        return self.cc.EvalInnerProduct(ct, pt, self._n)

    def add_scalar(self, ct, s):
        return self.cc.EvalAdd(ct, float(s))


# ============================================================================
# factories
# ============================================================================

def bfv_owner(plain_modulus, mult_depth, n_slots, ring_dim, backend="plaintext"):
    _check_backend(backend)
    if backend == "plaintext":
        return _PlainBFV(plain_modulus, max(int(n_slots), int(ring_dim or 0)),
                         ring_dim, mult_depth)
    return _OfheBFV.owner(plain_modulus, mult_depth, n_slots, ring_dim)


def bfv_from_public(blob, backend="plaintext"):
    _check_backend(backend)
    if backend == "plaintext":
        d = pickle.loads(blob)
        return _PlainBFV(d["t"], d["n_slots"], d["ring_dim"], d["mult_depth"])
    return _OfheBFV.from_public(blob)


def ckks_owner(mult_depth, scale_bits, batch_size, backend="plaintext"):
    _check_backend(backend)
    if backend == "plaintext":
        return _PlainCKKS(batch_size, mult_depth, scale_bits)
    return _OfheCKKS.owner(mult_depth, scale_bits, batch_size)


def ckks_from_public(blob, backend="plaintext"):
    _check_backend(backend)
    if backend == "plaintext":
        d = pickle.loads(blob)
        return _PlainCKKS(d["batch_size"], d["mult_depth"], d["scale_bits"])
    return _OfheCKKS.from_public(blob)
