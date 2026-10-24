"""Public parameter sets and the entry map shared by both parties."""
from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction


@dataclass
class CKKSParams:
    logN: int = 14
    depth: int = 3              # chain [p0 (first), p1, p2, p3]; q0 = p0 p1 (two primes), q1 = p2, q2 = p3
    scaling_mod_size: int = 59  # OpenFHE 64-bit build: scaling primes < 2^60
    first_mod_size: int = 60
    dnum: int = 2
    sec128: bool = True

    def as_dict(self):
        return dict(logN=self.logN, depth=self.depth, scaling_mod_size=self.scaling_mod_size,
                    first_mod_size=self.first_mod_size, dnum=self.dnum, sec128=self.sec128)


def make_ckks(cp: CKKSParams):
    from . import vfr_native as vn
    return vn.CKKSCore(cp.logN, cp.depth, cp.scaling_mod_size, cp.first_mod_size, cp.dnum, cp.sec128)


def gadget_constants(ck):
    """Key-switching constants at the EvalSum level of circuit A (after one rescale: towers p0 p1 p2) and of
    circuit C (top level: p0 p1 p2 p3), as OpenFHE's hybrid key switching partitions them."""
    qs, ps, alpha = list(ck.qs), list(ck.ps), ck.alpha

    def digits(k):
        ds = [qs[i:min(i + alpha, k)] for i in range(0, k, alpha)]
        return ds

    dA, dC = digits(len(qs) - 1), digits(len(qs))
    prod = lambda xs: math.prod(xs)
    return dict(Qtilde_max_A=float(max(prod(d) for d in dA)), Qtilde_max_C=float(max(prod(d) for d in dC)),
                P=float(prod(ps)), k_digit_A=max(len(d) for d in dA), k_digit_C=max(len(d) for d in dC),
                dnum_A=len(dA), dnum_C=len(dC), k_P=len(ps))


def dd(x) -> tuple:
    """(hi, lo) double-double of an int, Fraction or float."""
    F = Fraction(x)
    hi = float(F)
    lo = float(F - Fraction(hi))
    return (hi, lo)


def entries(d_R: int, d_O: int, joint: bool):
    """Global entry list.  Each entry: name, U (index of R's uploaded vector: 0 = b, 1 + j = xdot_j),
    V (how O forms its vector), releases it belongs to, slot (= index in the list, shared by all releases)."""
    out = []
    for j in range(d_O):
        for k in range(j, d_O):
            out.append(dict(name=f'B_{j}{k}', kind='B', U=0, V=('xx', j, k), rel={1, 2}))
    for j in range(d_R + 1):
        for k in range(d_O):
            out.append(dict(name=f'C_{j}{k}', kind='C', U=1 + j, V=('x', k), rel={1, 2}))
    for j in range(d_R + 1):
        out.append(dict(name=f'cR_{j}', kind='cR', U=1 + j, V=('y',), rel={2}))
    for k in range(d_O):
        out.append(dict(name=f'cO_{k}', kind='cO', U=0, V=('xy', k), rel={2}))
    out.append(dict(name='yy', kind='yy', U=0, V=('yy',), rel={2} if joint else {'y'}))
    for i, e in enumerate(out):
        e['slot'] = i
    return out


def release_entries(ents, stage):
    return [e for e in ents if stage in e['rel']]


def stage_key(s):
    return 'y' if s == 'y' else int(s)
