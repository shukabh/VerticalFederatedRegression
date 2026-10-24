"""Explicit, structured (non-executable) messages between R and O, and the transcript of what crosses.

A message is   8-byte big-endian frame length | 4-byte header length | JSON header | blob bytes...
The JSON header holds the message type, a JSON-able `meta` dict and, for every binary field, its name,
byte length and (for numpy arrays) dtype and shape.  No pickle anywhere.

Each party wraps its socket in a Channel that appends one record per message to its Transcript:
sequence number, direction, message type, total bytes, per-field byte counts and a SHA-256 of the frame.
For tests the raw frames can be kept (keep_raw) to scan for forbidden content.
"""
from __future__ import annotations

import hashlib
import json
import socket
import struct
import time

import numpy as np


def encode_msg(mtype: str, meta: dict | None = None, blobs: dict | None = None) -> bytes:
    meta = meta or {}
    blobs = blobs or {}
    fields, parts = [], []
    for name, val in blobs.items():
        if isinstance(val, np.ndarray):
            arr = np.ascontiguousarray(val)
            b = arr.tobytes()
            fields.append(dict(name=name, len=len(b), dtype=str(arr.dtype), shape=list(arr.shape)))
        elif isinstance(val, (bytes, bytearray)):
            b = bytes(val)
            fields.append(dict(name=name, len=len(b)))
        else:
            raise TypeError(f'blob {name}: unsupported type {type(val)}')
        parts.append(b)
    header = json.dumps(dict(type=mtype, meta=meta, fields=fields), sort_keys=True).encode()
    return struct.pack('>I', len(header)) + header + b''.join(parts)


def decode_msg(frame: bytes):
    hl = struct.unpack('>I', frame[:4])[0]
    header = json.loads(frame[4:4 + hl])
    off = 4 + hl
    blobs = {}
    for f in header['fields']:
        b = frame[off:off + f['len']]
        off += f['len']
        if 'dtype' in f:
            blobs[f['name']] = np.frombuffer(b, dtype=np.dtype(f['dtype'])).reshape(f['shape'])
        else:
            blobs[f['name']] = b
    if off != len(frame):
        raise ValueError('trailing bytes in frame')
    return header['type'], header['meta'], blobs, header['fields']


class Transcript:
    def __init__(self, party: str, keep_raw: bool = False):
        self.party = party
        self.records = []
        self.keep_raw = keep_raw
        self.raw = []

    def log(self, direction, mtype, frame, fields):
        rec = dict(seq=len(self.records), party=self.party, dir=direction, type=mtype, bytes=len(frame),
                   fields={f['name']: f['len'] for f in fields}, sha256=hashlib.sha256(frame).hexdigest()[:16],
                   time=time.time())
        self.records.append(rec)
        if self.keep_raw:
            self.raw.append((direction, mtype, frame))

    def summary(self):
        out = {}
        for r in self.records:
            k = (r['dir'], r['type'])
            out.setdefault(k, [0, 0])
            out[k][0] += 1
            out[k][1] += r['bytes']
        return {f'{d}:{t}': dict(count=c, bytes=b) for (d, t), (c, b) in out.items()}


class Channel:
    def __init__(self, sock: socket.socket, transcript: Transcript):
        self.sock = sock
        self.tr = transcript

    def send(self, mtype: str, meta: dict | None = None, blobs: dict | None = None):
        frame = encode_msg(mtype, meta, blobs)
        self.sock.sendall(struct.pack('>Q', len(frame)))
        view = memoryview(frame)
        for i in range(0, len(frame), 1 << 24):
            self.sock.sendall(view[i:i + (1 << 24)])
        hl = struct.unpack('>I', frame[:4])[0]
        self.tr.log('sent', mtype, frame, json.loads(frame[4:4 + hl])['fields'])

    def recv(self, expect: str | None = None):
        n = struct.unpack('>Q', self._recvall(8))[0]
        frame = self._recvall(n)
        mtype, meta, blobs, fields = decode_msg(frame)
        self.tr.log('recv', mtype, frame, fields)
        if expect is not None and mtype != expect:
            raise RuntimeError(f'expected {expect}, got {mtype}: {meta}')
        return mtype, meta, blobs

    def _recvall(self, n):
        buf = bytearray()
        while len(buf) < n:
            chunk = self.sock.recv(min(n - len(buf), 1 << 24))
            if not chunk:
                raise ConnectionError('peer closed')
            buf.extend(chunk)
        return bytes(buf)
