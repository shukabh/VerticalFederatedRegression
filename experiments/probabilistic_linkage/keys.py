"""Match keys derived from quasi-identifiers.

A *key* is a string; a record yields at most K keys, one per key type, each domain-separated by its type index so
that keys of different types never collide.  In the protocol every key would be fed to the OPRF exactly as the
identifier is today; the OPRF is injective on its domain up to a negligible collision probability, so two parties'
keys meet in the PSI iff the strings are equal.

Schemes
  exact      1 key:  first | last | sex | dob                   (deterministic exact match on the clean QIs)
  multikey   7 keys (OR rule), each tolerant to a different error:
               k1 canon-first | last | dob                       nicknames (via the public nickname table)
               k2 nysiis(canon-first) | nysiis(last) | dob        spelling / typo in names
               k3 soundex(last) | initial | sex | year | {day, month} unordered   day-month swap, first-name typo
               k4 canon-first | last | postcode                   any DOB error
               k5 metaphone(canon-first) | metaphone(last) | year | month         day typo
               k6 canon-first | sex | dob | postcode               surname change
               k7 last | sex | dob | postcode                     first-name error or missing
  lsh        b = 8 band keys of a MinHash (24 hash functions, r = 3 per band) over the character bigrams of
             'canon-first last', each band key combined with the birth year; similar names collide in at least one
             band with probability 1 - (1 - J^3)^8 for bigram Jaccard similarity J.
Field tokens for the Fellegi-Sunter variant are in `fields()`.
"""
from __future__ import annotations

import re
from functools import lru_cache

import jellyfish
import numpy as np

import names as NM

_nonalpha = re.compile(r'[^a-z]')


@lru_cache(maxsize=None)
def norm(s: str) -> str:
    return _nonalpha.sub('', s.lower()) if s else ''


@lru_cache(maxsize=None)
def canon(s: str) -> str:
    s = norm(s)
    return NM.CANON.get(s, s)


@lru_cache(maxsize=None)
def soundex(s):
    return jellyfish.soundex(s) if s else ''


@lru_cache(maxsize=None)
def nysiis(s):
    return jellyfish.nysiis(s) if s else ''


@lru_cache(maxsize=None)
def metaphone(s):
    return jellyfish.metaphone(s) if s else ''


def _dob(y, m, d):
    return f'{y}-{m:02d}-{d:02d}' if y > 0 else None


def multikey(first, last, sex, year, month, day, post):
    fc, ln = canon(first), norm(last)
    dob = _dob(year, month, day)
    ks = [None] * 7
    if fc and ln and dob:
        ks[0] = f'1|{fc}|{ln}|{dob}'
        ks[1] = f'2|{nysiis(fc)}|{nysiis(ln)}|{dob}'
    if ln and fc and year > 0 and sex >= 0:
        a, b = sorted((month, day))
        ks[2] = f'3|{soundex(ln)}|{fc[0]}|{sex}|{year}|{a}|{b}'
    if fc and ln and post:
        ks[3] = f'4|{fc}|{ln}|{post}'
    if fc and ln and year > 0:
        ks[4] = f'5|{metaphone(fc)}|{metaphone(ln)}|{year}|{month}'
    if fc and dob and post and sex >= 0:
        ks[5] = f'6|{fc}|{sex}|{dob}|{post}'
    if ln and dob and post and sex >= 0:
        ks[6] = f'7|{ln}|{sex}|{dob}|{post}'
    return ks


def exact(first, last, sex, year, month, day, post):
    dob = _dob(year, month, day)
    f, l = norm(first), norm(last)
    return [f'0|{f}|{l}|{sex}|{dob}' if (f and l and dob and sex >= 0) else None]


class MinHashLSH:
    def __init__(self, n_hash=24, r=3, seed=7):
        assert n_hash % r == 0
        self.r, self.b = r, n_hash // r
        rng = np.random.default_rng(seed)
        nb = 28 * 28
        p = 2 ** 31 - 1
        a = rng.integers(1, p, n_hash); c = rng.integers(0, p, n_hash)
        self.H = ((a[:, None] * np.arange(nb)[None, :] + c[:, None]) % p).astype(np.int64)

    @staticmethod
    def bigrams(s):
        c = [27] + [26 if ch == ' ' else ord(ch) - 97 for ch in s] + [27]    # a-z = 0..25, space 26, boundary 27
        return [c[i] * 28 + c[i + 1] for i in range(len(c) - 1)]

    @lru_cache(maxsize=None)
    def signature(self, s):
        ids = self.bigrams(s)
        return tuple(self.H[:, ids].min(axis=1).tolist())


_LSH = MinHashLSH()


def lsh(first, last, sex, year, month, day, post):
    fc, ln = canon(first), norm(last)
    if not (fc and ln and year > 0):
        return [None] * _LSH.b
    sig = _LSH.signature(fc + ' ' + ln)
    r = _LSH.r
    return [f'L{j}|{"/".join(map(str, sig[j * r:(j + 1) * r]))}|{year}' for j in range(_LSH.b)]


SCHEMES = dict(exact=exact, multikey=multikey, lsh=lsh)
K_OF = dict(exact=1, multikey=7, lsh=_LSH.b)


def keys_for(P, scheme):
    f = SCHEMES[scheme]
    return [f(P.first[i], P.last[i], int(P.sex[i]), int(P.year[i]), int(P.month[i]), int(P.day[i]), P.post[i])
            for i in range(len(P))]


# ------------------------------------------------------------------------------------------ field comparison (FS)
FIELD_NAMES = ['first', 'first_phon', 'last', 'last_soundex', 'dob', 'year', 'md_unordered', 'sex', 'post']


def fields(P, i):
    """Per-field tokens of record i (None = missing).  In the FS variant these would be OPRF tokens carried in the
    PSI label, so that R can test equality of each field of a candidate pair; equality is all R learns per field."""
    fc, ln = canon(P.first[i]), norm(P.last[i])
    y, m, d = int(P.year[i]), int(P.month[i]), int(P.day[i])
    return [norm(P.first[i]) or None, nysiis(fc) or None, ln or None, soundex(ln) or None,
            _dob(y, m, d), y if y > 0 else None, tuple(sorted((m, d))) if y > 0 else None,
            int(P.sex[i]) if P.sex[i] >= 0 else None, P.post[i] or None]


def agreement(fa, fb):
    """Binary agreement vector; a missing value on either side counts as disagreement."""
    return [int(a is not None and a == b) for a, b in zip(fa, fb)]
