"""Worst-case, data-independent bound B_tot on every flooded error term (Obligation O1).

Port of the proof review's scratch/btot.py (crypto/03a_error_model.tex, 03b_error_bound.tex), generalized
in four ways that the implementation needs, all inside the proof's own model (M1)-(M5):

  * separate encoding scales Omega_U (R), Omega_V (O's data), Omega_u (masks) and primes q1, q2
    (model (M2) allows this; the final scale is Omega_f = Omega_U Omega_V Omega_u / (q1 q2));
  * the actual gadget of OpenFHE's hybrid key switching: digit products Q~_j, special modulus P and the
    fast (approximate) base conversions.  OpenFHE's ApproxModUp returns digits in [0, k_j Q~_j), so
    ||D~_j||_inf <= varsigma_up Q~_j / 2 with varsigma_up = 2 k_j; its ApproxModDown returns
    floor(c/P) - u with u in [0, k_P), so the ModDown rounding lies in [0, k_P) and we scale the
    proof's [-1/2, 1/2] rounding terms by varsigma_down = 2 k_P;
  * variant 'C' (EvalSum before the first rescale, stochastic rounding, one final rescale per release)
    uses the formulas of btot.py with the mask factor Omega_u/q1 and the key-switch division by q2 made
    explicit, plus a T8 term (btot.py omits it for C; we keep it, it is < 1);
  * Delta_x (Corollary sensZ) with separate scales.

Every quantity is an l2 norm of a coefficient vector.  B_tot is a function of public parameters only.

Full-record circuit (CircuitParams.fullrecord=True; the full-record SSP baseline, Lemma S10e of
docs/proofs/stats_s7_s10.tex).  Stages 1 and 2 also release triu A without the constant n: n_A = d_R(d_R+3)/2 more
entries, the d_R first moments sum_t b_t x_j,t (R's existing vector U = b xdot_j) and the d_R(d_R+1)/2 products
sum_t b_t x_j,t x_k,t (R's extra vectors U = b x_j x_k), each against O's vector of ones (V = 1 on every row).
Re-derived, not just re-evaluated: in the error decomposition (Proposition decomp of Part A) R's vectors enter only
the centre, never an error term, so B_tot changes only through |E| (+ n_A) and the row bound Gamma_E (+ n_A, from
V = 1).  The U-side bounds (B_R for the first moments, B_R^2 for the products) enter nu_U, hence Delta_x and the
decoding offset, and zeta_E (+ r^2).  The adjacency of the full-record release replaces R's row as well, which adds
cross terms to Delta_x (delta_x below).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import log, log2, sqrt


@dataclass
class CircuitParams:
    N: int
    n_bar: int                 # padded register size
    d_R: int
    d_O: int
    B_R: float
    B_O: float
    B_y: float
    q1: int                    # prime removed by the second rescale (mask level)
    q2: int                    # prime removed by the first rescale
    Omega_U: float             # R's encoding scale
    Omega_V: float             # O's encoding scale for the data vectors
    Omega_u: float             # O's encoding scale for the one-hot masks
    Qtilde_max_A: float        # largest digit product at the EvalSum level, circuit A (after one rescale)
    Qtilde_max_C: float        # same at the top level (circuit C)
    P: float                   # special modulus
    k_digit_A: int             # max primes per digit at the EvalSum level (A)
    k_digit_C: int
    dnum_A: int                # digits at the EvalSum level (A)
    dnum_C: int
    k_P: int                   # number of special primes
    sigma_e: float = 3.2       # error s.d. (OpenFHE uses 3.19; 3.2 is an upper bound)
    delta_B: float = 2.0 ** -50
    n_ev: int = 12
    v_s: float = 4 / 3         # s = s_R + s_O, both uniform ternary
    v_u: float = 2 / 3
    varsigma_up: float | None = None    # override (None: 2 k_digit, OpenFHE's ApproxModUp)
    varsigma_down: float | None = None  # override (None: 2 k_P, OpenFHE's ApproxModDown)
    fullrecord: bool = False   # the full-record circuit (triu A released too; whole-record adjacency)

    @property
    def L(self):               # number of rotations in EvalSum (log2 of the slot count)
        return int(round(log2(self.N // 2)))

    @property
    def m(self):
        return -(-self.n_bar // (self.N // 2))

    @property
    def r(self):
        return self.B_R ** 2

    @property
    def o(self):
        return self.B_O ** 2

    @property
    def w(self):
        return self.B_y ** 2


def n_A(d_R: int) -> int:
    """Entries of triu A without the constant n (full-record circuit): d_R first moments + d_R(d_R+1)/2 products."""
    return d_R * (d_R + 3) // 2


def entry_counts(d_R: int, d_O: int, joint: bool, fullrecord: bool = False):
    nG = d_O * (d_O + 1) // 2 + (d_R + 1) * d_O
    n2 = nG + (d_R + 1) + d_O + (1 if joint else 0)
    a = n_A(d_R) if fullrecord else 0
    return {1: nG + a, 2: n2 + a, 'y': 1}


def gamma_E(cp: CircuitParams, stage, joint: bool, ones: bool = True):
    """Lemma data: sum_e V_{e,t}^2 <= Gamma_E for every row t.  Full-record circuit: + n_A for O's vectors of ones
    (ones=False leaves them out: they never change between neighbours, so nu_V does not count them)."""
    o, w, dR1 = cp.o, cp.w, cp.d_R + 1
    a = n_A(cp.d_R) if (cp.fullrecord and ones and stage in (1, 2)) else 0
    if stage == 1:
        return o * o + dR1 * o + a
    if stage == 2:
        return o * o + dR1 * (o + w) + o * w + (w * w if joint else 0.0) + a
    return w * w  # separate y'y release


def nu_U(cp: CircuitParams, stage, joint: bool):
    """sqrt(sum_e U_max,e^2): 1 for entries that use b, B_R for entries that use xdot (conservative, as btot.py).
    Full-record circuit: + d_R B_R^2 (first moments, U = b xdot_j) + d_R(d_R+1)/2 B_R^4 (products, |b x_j x_k| <= B_R^2)."""
    nB, nC = cp.d_O * (cp.d_O + 1) // 2, (cp.d_R + 1) * cp.d_O
    extra = 0.0
    if cp.fullrecord and stage in (1, 2):
        extra = cp.d_R * cp.B_R ** 2 + cp.d_R * (cp.d_R + 1) // 2 * cp.B_R ** 4
    if stage == 1:
        return sqrt(nB * 1 + nC * cp.B_R ** 2 + extra)
    if stage == 2:
        n_b = nB + cp.d_O + (1 if joint else 0)
        n_x = nC + (cp.d_R + 1)
        return sqrt(n_b + n_x * cp.B_R ** 2 + extra)
    return 1.0


def nu_V_max(cp: CircuitParams, stage, joint: bool):
    """sqrt(sum_e V_max,e^2), V_max,e = max_t |V_{e,t}|: o for triu B, B_O for C, B_y for c_R, B_O B_y for c_O, w for
    y'y, 1 for O's vectors of ones (full-record circuit).  Enters only the whole-record adjacency of delta_x."""
    o, w, dR1, dO = cp.o, cp.w, cp.d_R + 1, cp.d_O
    v = dO * (dO + 1) // 2 * o * o + dR1 * dO * o + (n_A(cp.d_R) if cp.fullrecord else 0)
    if stage == 1:
        return sqrt(v)
    if stage == 2:
        return sqrt(v + dR1 * w + dO * o * w + (w * w if joint else 0.0))
    return w


def zeta_E(cp: CircuitParams, stage, joint: bool):
    """Largest contribution of one row to the statistic, sqrt(sup_t sum_e (U_{e,t} V_{e,t})^2) (crypto proofs,
    docs/crypto_proofs.tex, Section 4, before Proposition 4.5, with r = B_R^2).  Full-record circuit: + r^2, since
    sum over triu(xdot xdot^T) without (0, 0) of (x_j x_k)^2 <= ||xdot||^4 - 1 < r^2."""
    r, o, w = cp.B_R ** 2, cp.o, cp.w
    a = r * r if (cp.fullrecord and stage in (1, 2)) else 0.0
    if stage == 1:
        return sqrt(o * o + r * o + a)
    if stage == 2:
        return sqrt(o * o + r * (o + w) + o * w + (w * w if joint else 0.0) + a)
    return w  # separate y'y release


def offset_bounds(cp: CircuitParams, stage, joint: bool, n_max: int) -> dict:
    """Public bound on the decoding offset (crypto proofs, docs/crypto_proofs.tex, Proposition 4.5) for at most
    n_max matched rows: d_bar (slot domain, on the slots of the entries) and d_max (coefficient domain, l2 norm of
    Zhat_s(D) - Omega_f tau^{-1}(z_s(D))), the d_max of the no-wraparound condition (Q).  Public parameters only."""
    N, m = cp.N, cp.m
    nE = entry_counts(cp.d_R, cp.d_O, joint, cp.fullrecord)[stage]
    Gam, nuU, zeta = gamma_E(cp, stage, joint), nu_U(cp, stage, joint), zeta_E(cp, stage, joint)
    OmU, OmV, Omu, q12 = cp.Omega_U, cp.Omega_V, cp.Omega_u, cp.q1 * cp.q2
    Omf = OmU * OmV * Omu / q12
    A = (N * sqrt(m) / (2 * sqrt(2)) * (sqrt(n_max) * nuU / OmV + sqrt(cp.n_bar * Gam) / OmU)
         + sqrt(nE) * m * N ** 2 / (8 * OmU * OmV))
    d_bar = A + N * sqrt(nE) / (2 * sqrt(2) * Omu) * (n_max * zeta + A) + N / (2 * sqrt(2) * Omf)
    r_bar = OmU * OmV * A
    d_max = (sqrt(2 / N) * Omu / q12 * r_bar
             + sqrt(N) / (2 * q12) * sqrt(nE) * (OmU * OmV * n_max * zeta + r_bar))
    return dict(d_bar=d_bar, d_max=d_max)


def bounds(cp: CircuitParams, stage, joint: bool, variant: str = 'A', fresh: bool = True):
    """Return the dict of term bounds B1..B8 (Proposition terms of 03b_error_bound.tex, generalized)."""
    N, L, sig_e, v_s, v_u = cp.N, cp.L, cp.sigma_e, cp.v_s, cp.v_u
    nE = entry_counts(cp.d_R, cp.d_O, joint, cp.fullrecord)[stage]
    Gam = gamma_E(cp, stage, joint)
    m = cp.m
    t = log(cp.n_ev / cp.delta_B)
    tt = 1 + sqrt(2 * t)
    lnN = log(2 * N * cp.n_ev / cp.delta_B)
    sq = sqrt(2 / N)
    mu = cp.Omega_u / cp.q1                       # main factor of masking + second rescale
    vsU = (2 * cp.k_digit_A if variant == 'A' else 2 * cp.k_digit_C) if cp.varsigma_up is None else cp.varsigma_up
    vsD = 2 * cp.k_P if cp.varsigma_down is None else cp.varsigma_down   # ModDown rounding in [0, k_P)
    QtP = (cp.Qtilde_max_A if variant == 'A' else cp.Qtilde_max_C) / cp.P
    dnum = cp.dnum_A if variant == 'A' else cp.dnum_C
    v_rk = 2 * sig_e ** 2                         # e_rk = e_rk,R + e_rk,O
    T = {}
    # T1: R's encryption errors times O's data, traced and masked
    mV = cp.Omega_V * sq * sqrt(cp.n_bar * Gam) + sqrt(nE * m * N) / 2
    T['T1'] = sq * mu * (N / 2) / cp.q2 * sig_e * sqrt(2) * mV * tt
    s2 = sqrt(v_s) * (sqrt(N) + sqrt(2 * t))      # ||s|| bound
    scan = sqrt(2 * v_s * N * lnN)                 # canonical-norm bound of s
    epk_can = sqrt(2 * 2 * sig_e ** 2 * N * lnN)
    epk2 = sqrt(2) * sig_e * (sqrt(N) + sqrt(2 * t))
    rd_det = (N / 2) * sum(2 ** ((L - 1 - i) / 2) for i in range(L))
    if variant == 'A':
        T['T2'] = sq * mu * (N / 2) * (sqrt(nE / 2) + sqrt(v_s) * sqrt(2 * nE) * (sqrt(N) / 2) * tt)
        T['T3'] = sq * mu * sqrt(v_rk) * sqrt(nE * L * dnum) * vsU * N ** 1.5 / 2 * QtP * tt
        T['T4'] = sq * mu * sqrt(nE) * (rd_det + sqrt(v_s) * L * N ** 1.5 / 2 * tt) * vsD
        T['T5'] = nE * sqrt(N) / 2 + sqrt(v_s) * nE * (N / 2) * tt
    elif variant == 'C':
        T['T2'] = sq * mu * (sqrt(nE) * N / 2 + 0.5 * sqrt(N) * scan * (sqrt(nE) + sqrt(2 * t)))
        T['T3+T4'] = (sq * sqrt(v_rk) * sqrt(nE * L * dnum) * vsU * N ** 1.5 / 2 * QtP * tt
                      + sq * sqrt(nE) * (rd_det + sqrt(v_s) * L * N ** 1.5 / 2 * tt) * vsD) * mu / cp.q2
        T['T5'] = 0.5 * (sqrt(N) + sqrt(N) * s2 + sqrt(2 * t) * (1 + scan))
    else:
        raise ValueError(variant)
    # T6: fresh-encryption error e_pk u + e0 + e1 s of the re-randomization
    T['T6'] = (sqrt(N * (v_u * epk2 ** 2 + sig_e ** 2 + sig_e ** 2 * s2 ** 2))
               + sqrt(2 * t * (v_u * epk_can ** 2 + sig_e ** 2 + sig_e ** 2 * scan ** 2))) if fresh else 0.0
    T['T7'] = sqrt(N) / 2
    # T8: spill of the errors through the mask rounding r_u
    tp = 1 + sqrt(2 * (t + log(nE * L)))
    eks = sqrt(v_rk) * dnum * N * QtP * vsU / 2 * tp + vsD * (sqrt(N) / 2 + sqrt(v_s) * (N / 2) * tp)
    if variant == 'C':
        eks = eks / cp.q2
    T['T8'] = N / (2 * cp.q1) * (nE * sqrt(N / 2) * (cp.q1 / cp.Omega_u) * (T['T1'] + T['T2'])
                                 + nE * (N / 2) * eks)
    return T


def B_tot(cp: CircuitParams, stage, joint: bool, variant: str = 'A') -> float:
    return float(sum(bounds(cp, stage, joint, variant).values()))


def epsilon_x(cp: CircuitParams, stage, joint: bool) -> float:
    """The leftover of Lemma sensZdec (Part A), covered by the '+1' of Delta_x when at most 1; doubled for the
    whole-record adjacency of the full-record circuit (Lemma S10e)."""
    N = cp.N
    nE = entry_counts(cp.d_R, cp.d_O, joint, cp.fullrecord)[stage]
    q12 = cp.q1 * cp.q2
    e = cp.Omega_u / q12 * N ** 1.5 * sqrt(nE) / (2 * sqrt(2)) + N ** 2.5 * nE / (8 * q12)
    return 2 * e if cp.fullrecord else e


def delta_x(cp: CircuitParams, stage, joint: bool, Delta_s: float) -> float:
    """Extra sensitivity of the centre f_s (Corollary sensZ), separate scales, '+1' for omitted terms.
    Full-record circuit (Lemma S10e): the neighbour also changes R's row, so the slot value of each entry gains
    Omega_U Delta U_e tau(r_V')_l* + sum_l tau(delta r_U)_l tau(m_V')_l, which adds Omega_U nu_dU + sqrt(N) Omega_V nu_Vmax
    (nu_dU = 2 nu_U) to the bracket Omega_V nu_V + sqrt(N) Omega_U nu_U; Delta_s is the full-record sensitivity."""
    N = cp.N
    nE = entry_counts(cp.d_R, cp.d_O, joint, cp.fullrecord)[stage]
    nuV = 2 * sqrt(gamma_E(cp, stage, joint, ones=False))
    nuU = nu_U(cp, stage, joint)
    q12 = cp.q1 * cp.q2
    K = cp.Omega_V * nuV + sqrt(N) * cp.Omega_U * nuU
    if cp.fullrecord:
        K += cp.Omega_U * 2 * nuU + sqrt(N) * cp.Omega_V * nu_V_max(cp, stage, joint)
    main = sqrt(N / 2) * (cp.Omega_u / q12) * K
    spill = (sqrt(N) / 2) * (cp.Omega_U * cp.Omega_V / q12) * sqrt(nE) * Delta_s \
        + N ** 1.5 / (4 * q12) * sqrt(nE) * K
    return main + spill + 1.0


def table(cp: CircuitParams, joint: bool, variant: str):
    out = {}
    for st in ([1, 2] if joint else [1, 2, 'y']):
        T = bounds(cp, st, joint, variant)
        out[st] = {k: (log2(v) if v > 0 else float('-inf')) for k, v in T.items()}
        out[st]['B_tot'] = log2(sum(T.values()))
    return out
