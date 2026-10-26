"""The parallel (native, OpenMP) OPRF evaluation of O's register and of R's blinded queries is bit-identical to the
item-by-item reference implementation in vfr/oprf.py (RFC 9497 vectors included), at every thread count."""
import json
import os

import pytest

from vfr import oprf

HERE = os.path.dirname(os.path.abspath(__file__))


def _ids(n):
    # register-like identifiers of varying length, plus the padding dummies O adds, an empty and a long input
    xs = [str(i).encode() for i in range(n)] + [f'dummy:{i}'.encode() for i in range(50)]
    return xs + [b'', os.urandom(300), bytes(range(256)) * 4]


@pytest.mark.parametrize('threads', [1, 2, 4, 7])
def test_evaluate_batch_bit_identical(threads):
    k = oprf.keygen()
    xs = _ids(2000)
    ref = [oprf.evaluate(k, x) for x in xs]
    assert oprf.evaluate_batch(k, xs, threads) == ref


def test_evaluate_batch_rfc_vectors_and_default_threads():
    V = json.load(open(os.path.join(HERE, 'data', 'rfc9497_ristretto255_sha512_oprf.json')))[0]
    sk = oprf.derive_key_pair(bytes.fromhex(V['seed']), bytes.fromhex(V['keyInfo']))
    xs = [bytes.fromhex(t['Input']) for t in V['vectors']]
    assert [o.hex() for o in oprf.evaluate_batch(sk, xs)] == [t['Output'] for t in V['vectors']]
    blinded = b''.join(bytes.fromhex(t['BlindedElement']) for t in V['vectors'])
    ev = oprf.blind_evaluate_batch(sk, blinded, 4)
    assert [ev[32 * i:32 * i + 32].hex() for i in range(len(xs))] == [t['EvaluationElement'] for t in V['vectors']]
    assert oprf.evaluate_batch(sk, []) == []


@pytest.mark.parametrize('threads', [1, 4])
def test_blind_evaluate_batch_bit_identical(threads):
    k = oprf.keygen()
    xs = _ids(500)
    bl = [oprf.blind(x) for x in xs]
    blob = b''.join(a for _, a in bl)
    ev = oprf.blind_evaluate_batch(k, blob, threads)
    assert ev == b''.join(oprf.blind_evaluate(k, a) for _, a in bl)
    # and R's finalized outputs equal O's direct evaluation of the same identifiers
    outs = [oprf.finalize(x, r, ev[32 * i:32 * i + 32]) for i, (x, (r, _)) in enumerate(zip(xs, bl))]
    assert outs == oprf.evaluate_batch(k, xs, threads)


def test_blind_evaluate_batch_rejects_invalid_points():
    k = oprf.keygen()
    good = oprf.blind(b'1')[1]
    bad = b'\xff' * 32                                   # not a canonical ristretto255 encoding
    with pytest.raises(ValueError):
        oprf.blind_evaluate_batch(k, good + bad + good, 4)
    with pytest.raises(ValueError):
        oprf.blind_evaluate_batch(k, good + b'\0' * 32, 2)   # the identity element is refused, as in blind_evaluate
    with pytest.raises(ValueError):
        oprf.blind_evaluate_batch(k, good[:31], 1)
    with pytest.raises(ValueError):
        oprf.evaluate_batch(k[:31], [b'1'])
