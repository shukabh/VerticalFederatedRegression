"""Synthetic people with quasi-identifiers (QIs), regression variables, and two noisy copies (R's cohort, O's register).

World
  * persons 0 .. n_scope-1 form O's study scope; R's cohort is n_overlap persons drawn from the scope plus
    n_R - n_overlap persons outside it (people O does not hold);
  * households of 1-4 people share a postcode and (mostly) a surname; a fraction of births are twins (same DOB and
    surname, different first name) -- the hard negatives of real linkage;
  * QIs: first name (Zipf-Mandelbrot over a head of common names plus a synthetic tail), surname (same, longer tail),
    sex, date of birth (1935-2005), postcode;
  * regression variables: the repo's synthetic population (vfr/data.population(), sim_core.Population; d_R = 4,
    d_O = 6, R^2 = 0.5), standardized and clipped as in the protocol's 'std' preprocessing.  The QIs are drawn
    independently of the regression variables, except that with `mnar > 0` the corruption rates of a person's
    records rise when the person's outcome is below the median (linkage errors that depend on y).

Corruption (independently per copy; R's copy is noisier than O's, rates in RATES_R / RATES_O): first-name nickname
or spelling variant, typo (substitute / insert / delete / transpose one letter), missing; surname typo, change
(marriage, women aged 22+), missing; DOB day-month swap, one-digit typo in day or year, missing; postcode change
(moved) or typo; sex missing.  The rates are modelling assumptions in the range reported for administrative data,
not estimates for any particular register.
"""
from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass, field

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, HERE)
import names as NM  # noqa: E402

LETTERS = np.array(list('abcdefghijklmnopqrstuvwxyz'))

RATES_R = dict(first_variant=0.08, first_typo=0.03, first_missing=0.01,
               last_typo=0.03, last_change=0.08, last_missing=0.01,
               dob_swap=0.02, dob_typo=0.02, dob_missing=0.005,
               post_moved=0.15, post_typo=0.01, sex_missing=0.005)
RATES_O = dict(first_variant=0.03, first_typo=0.01, first_missing=0.002,
               last_typo=0.01, last_change=0.02, last_missing=0.002,
               dob_swap=0.005, dob_typo=0.005, dob_missing=0.001,
               post_moved=0.03, post_typo=0.003, sex_missing=0.001)


def zipf_mandelbrot(n, q, s):
    k = np.arange(1, n + 1, dtype=float)
    w = 1.0 / (k + q) ** s
    return w / w.sum()


@dataclass
class Tables:
    male: list
    female: list
    surnames: list
    p_male: np.ndarray
    p_female: np.ndarray
    p_surname: np.ndarray


def tables(n_surname_tail=60_000):
    male = NM.MALE + [s for s in NM.synthetic_first(500) if s not in NM.MALE]
    female = NM.FEMALE + [s for s in NM.synthetic_first(500)[::-1] if s not in NM.FEMALE]
    sur = list(dict.fromkeys(NM.SURNAMES + NM.synthetic_surnames(n_surname_tail)))
    return Tables(male, female, sur,
                  zipf_mandelbrot(len(male), 8.0, 1.25), zipf_mandelbrot(len(female), 8.0, 1.2),
                  zipf_mandelbrot(len(sur), 20.0, 1.0))


def postcode_str(i):
    """A Canadian-looking postal code for index i (letter digit letter digit letter digit)."""
    L = 'abceghjklmnprstvxy'
    out = []
    for k in range(3):
        out.append(L[i % len(L)]); i //= len(L)
        out.append(str(i % 10)); i //= 10
    return ''.join(out)


@dataclass
class People:
    first: np.ndarray        # object arrays of str ('' = missing)
    last: np.ndarray
    sex: np.ndarray          # 0 / 1, -1 missing
    year: np.ndarray         # -1 missing
    month: np.ndarray
    day: np.ndarray
    post: np.ndarray
    pid: np.ndarray          # true person id

    def take(self, idx):
        return People(*(getattr(self, f)[idx] for f in ('first', 'last', 'sex', 'year', 'month', 'day', 'post', 'pid')))

    def copy(self):
        return self.take(np.arange(len(self.pid)))

    def __len__(self):
        return len(self.pid)


def make_people(n, rng, T: Tables, twin_rate=0.015, household_surname=0.85):
    """n persons in households; returns People (clean) and household ids."""
    sizes = rng.choice([1, 2, 3, 4], size=n, p=[0.30, 0.30, 0.20, 0.20])
    hh = np.repeat(np.arange(len(sizes)), sizes)[:n]
    first_of_hh = np.r_[True, hh[1:] != hh[:-1]]
    sex = rng.integers(0, 2, n)
    fm = rng.choice(len(T.male), n, p=T.p_male)
    ff = rng.choice(len(T.female), n, p=T.p_female)
    first = np.where(sex == 1, np.array(T.male, dtype=object)[fm], np.array(T.female, dtype=object)[ff])
    sur_idx = rng.choice(len(T.surnames), n, p=T.p_surname)
    hh_sur = sur_idx[np.flatnonzero(first_of_hh)][hh]          # household's surname
    keep_hh = rng.random(n) < household_surname
    last = np.array(T.surnames, dtype=object)[np.where(keep_hh | first_of_hh, hh_sur, sur_idx)]
    n_post = max(1000, n // 20)
    post_idx = rng.integers(0, n_post, len(sizes))[hh]
    post = np.array([postcode_str(int(i) * 7919 % (18 ** 3 * 1000)) for i in post_idx], dtype=object)
    year = rng.integers(1935, 2006, n)
    month = rng.integers(1, 13, n)
    day = rng.integers(1, 29, n)
    # twins: a non-first household member copies the previous member's DOB and surname
    twin = (~first_of_hh) & (rng.random(n) < twin_rate / 0.7)
    for i in np.flatnonzero(twin):
        year[i], month[i], day[i], last[i] = year[i - 1], month[i - 1], day[i - 1], last[i - 1]
    return People(first, last, sex, year, month, day, post, np.arange(n)), hh


def _typo(s, rng):
    if len(s) < 2:
        return s
    op = rng.integers(0, 4)
    k = int(rng.integers(0, len(s)))
    c = LETTERS[rng.integers(0, 26)]
    if op == 0:
        return s[:k] + c + s[k + 1:]
    if op == 1:
        return s[:k] + c + s[k:]
    if op == 2:
        return s[:k] + s[k + 1:]
    k = min(k, len(s) - 2)
    return s[:k] + s[k + 1] + s[k] + s[k + 2:]


def corrupt(P: People, rates: dict, rng, T: Tables, mult=None):
    """A corrupted copy of P.  mult: per-record multiplier of every rate (MNAR), default 1."""
    Q = P.copy()
    n = len(Q)
    m = np.ones(n) if mult is None else np.asarray(mult, float)

    def hit(key):
        return rng.random(n) < np.minimum(1.0, rates[key] * m)

    # first name
    for i in np.flatnonzero(hit('first_variant')):
        f = Q.first[i]
        formal = NM.CANON.get(f, f)
        vs = [formal] + NM.NICKNAMES.get(formal, [])
        vs = [v for v in vs if v != f]
        if vs:
            Q.first[i] = vs[int(rng.integers(0, len(vs)))]
    for i in np.flatnonzero(hit('first_typo')):
        Q.first[i] = _typo(Q.first[i], rng)
    Q.first[hit('first_missing')] = ''
    # surname
    for i in np.flatnonzero(hit('last_typo')):
        Q.last[i] = _typo(Q.last[i], rng)
    age = 2025 - Q.year
    ch = hit('last_change') & (Q.sex == 0) & (age >= 22)
    if ch.any():
        new = rng.choice(len(T.surnames), int(ch.sum()), p=T.p_surname)
        Q.last[ch] = np.array(T.surnames, dtype=object)[new]
    Q.last[hit('last_missing')] = ''
    # date of birth
    sw = hit('dob_swap') & (Q.day <= 12)
    Q.month[sw], Q.day[sw] = Q.day[sw].copy(), Q.month[sw].copy()
    for i in np.flatnonzero(hit('dob_typo')):
        if rng.random() < 0.5:
            d = Q.day[i]
            Q.day[i] = int(min(28, max(1, (d // 10) * 10 + int(rng.integers(0, 10)) or 1)))
        else:
            y = Q.year[i]
            Q.year[i] = int((y // 10) * 10 + int(rng.integers(0, 10)))
    miss = hit('dob_missing')
    Q.year[miss] = -1; Q.month[miss] = -1; Q.day[miss] = -1
    # postcode, sex
    mv = np.flatnonzero(hit('post_moved'))
    Q.post[mv] = [postcode_str(int(rng.integers(0, 10 ** 7))) for _ in mv]
    for i in np.flatnonzero(hit('post_typo')):
        s = list(Q.post[i]); k = int(rng.integers(0, len(s)))
        s[k] = str(rng.integers(0, 10)) if s[k].isdigit() else str(LETTERS[rng.integers(0, 26)])
        Q.post[i] = ''.join(s)
    Q.sex[hit('sex_missing')] = -1
    return Q


@dataclass
class World:
    O: People                 # O's register copy (scope only), in O's row order
    R: People                 # R's cohort copy
    in_scope_R: np.ndarray    # R record i is a scope person (has a true O row)
    true_pos: np.ndarray      # R record i -> O row index of the same person (-1 if none)
    ZR: np.ndarray            # per person (indexed by pid): R's features (n_persons, d_R), clipped std
    ZO: np.ndarray            # O's features (n_persons, d_O)
    y: np.ndarray             # outcome (unclipped, std scale)
    meta: dict = field(default_factory=dict)


def make_world(n_scope=1_000_000, n_R=50_000, n_overlap=40_000, seed=1, rates_R=None, rates_O=None, mnar=0.0,
               T: Tables | None = None):
    rng = np.random.default_rng(seed)
    T = T or tables()
    n_persons = n_scope + (n_R - n_overlap)
    P, hh = make_people(n_persons, rng, T)
    # regression variables per person
    sys.path.insert(0, os.path.join(ROOT, 'experiments', 'large'))
    import lcommon as lc
    pop = lc.population()
    X = pop._features(n_persons, rng)
    zR, zO, Zs = lc._transform(pop, X, 'std')
    y = Zs @ pop.beta_std + pop.noise_sd * rng.standard_normal(n_persons)
    mult = None
    if mnar > 0:
        mult = 1.0 + mnar * (y < np.median(y))
    # O's copy: the scope persons, shuffled into O's row order
    scope = rng.permutation(n_scope)
    O = corrupt(P.take(scope), rates_O or RATES_O, rng, T, None if mult is None else mult[scope])
    # R's cohort: n_overlap scope persons + outsiders
    ov = rng.choice(n_scope, n_overlap, replace=False)
    outside = np.arange(n_scope, n_persons)
    coh = rng.permutation(np.r_[ov, outside])
    R = corrupt(P.take(coh), rates_R or RATES_R, rng, T, None if mult is None else mult[coh])
    pos_of_pid = np.full(n_persons, -1)
    pos_of_pid[scope] = np.arange(n_scope)
    true_pos = pos_of_pid[coh]
    return World(O=O, R=R, in_scope_R=true_pos >= 0, true_pos=true_pos, ZR=zR, ZO=zO, y=y,
                 meta=dict(n_scope=n_scope, n_R=n_R, n_overlap=n_overlap, seed=seed, mnar=mnar, n_households=int(hh.max() + 1)))


def name_stats(T: Tables):
    """Head mass and collision probabilities of the name distributions (for the report)."""
    def st(p):
        ps = np.sort(p)[::-1]
        return dict(top1=float(ps[0]), top10=float(ps[:10].sum()), top100=float(ps[:100].sum()),
                    collision=float((p ** 2).sum()), entropy_bits=float(-(p * np.log2(p)).sum()), types=int(len(p)))
    return dict(male=st(T.p_male), female=st(T.p_female), surname=st(T.p_surname))


if __name__ == '__main__':
    import json
    print(json.dumps(name_stats(tables()), indent=1))
