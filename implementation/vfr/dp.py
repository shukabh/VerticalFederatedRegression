"""Phase 0 calibration: replace-one sensitivities, joint/separate y'y mode, tight zCDP conversion,
Calibrate (scale chosen so that the DP noise floods the worst-case error), and O's ledger.

Follows paper/algorithms.tex (Algorithm 7, Calibrate): the statistical slack is
gamma = n_rel (delta_B + 2^-kappa (1 + 2^-80) + 2^-97) and delta_DP = delta_run - (1 + e^eps) gamma; the encoded
sensitivity includes Delta_x (centred mechanism, exact final scale); sigma_c^2 is the integer
ceil(Delta_enc^2 / (2 rho)) so that the exact sampler gets a rational variance.

The flooding parameter kappa is derived unless an integer is given (kappa='auto', the default):
kappa = max(kappa_min, kappa_run, kappa_L), where kappa_run is the smallest integer with
(1 + e^eps_run) gamma <= zeta delta_run, and kappa_L (only when the agreement names a ledger cap) is the smallest
integer with gamma <= (rho_run / rho_L) Gamma_L, so that the ledger is exhausted by rho, never by slack.
Calibrate refuses (CalibrationRefused) when no kappa meets the rule, when no scale floods, when q0 is too small,
or when the modulus chain exceeds the security bound for N.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass, field
from fractions import Fraction

import numpy as np
from scipy.optimize import brentq, minimize_scalar

from . import btot as bt

# =============================================================================================
# DeltaRep (Proposition sens / Part B Theorem S1): exact enumeration in every regime
# =============================================================================================


def _F_poly(r, o, w, By, region, with_yy=True):
    """F(c*(t), t) as a numpy polynomial in t for one region of c*(t).
    F(c,t) = 2o^2(1-c^2) + 2ro(1-c) + r(By-t)^2 + o(w+t^2) - 2 o c By t + (w-t^2)^2."""
    P = np.polynomial.Polynomial
    tpol = P([0.0, 1.0])
    base = r * (By - tpol) ** 2 + o * (w + tpol ** 2)
    if with_yy:
        base = base + (w - tpol ** 2) ** 2
    if region == 'interior':
        c = -(r + By * tpol) / (2 * o)
    elif region == 'minus':
        c = P([-1.0])
    else:
        c = P([1.0])
    return base + 2 * o * o * (1 - c ** 2) + 2 * r * o * (1 - c) - 2 * o * c * By * tpol


def _F_val(r, o, w, By, t, with_yy=True):
    c = min(1.0, max(-1.0, -(r + By * t) / (2 * o))) if o > 0 else 0.0
    v = 2 * o * o * (1 - c * c) + 2 * r * o * (1 - c) + r * (By - t) ** 2 + o * (w + t * t) - 2 * o * c * By * t
    return v + ((w - t * t) ** 2 if with_yy else 0.0)


def _sup_F(r, o, w, By, with_yy=True):
    if By == 0:
        return _F_val(r, o, w, By, 0.0, with_yy)
    lo_c, hi_c = (-2 * o - r) / By, (2 * o - r) / By        # interior iff lo_c < t < hi_c
    regions = [('plus', -By, min(By, lo_c)), ('interior', max(-By, lo_c), min(By, hi_c)),
               ('minus', max(-By, hi_c), By)]
    cands = {-By, By}
    for name, a, b in regions:
        if a > b:
            continue
        cands.update([a, b])
        dpol = _F_poly(r, o, w, By, name, with_yy).deriv()
        for z in dpol.roots():
            if abs(z.imag) < 1e-9 * max(1.0, abs(z.real)) and a - 1e-12 <= z.real <= b + 1e-12:
                cands.add(float(min(max(z.real, a), b)))
    return max(_F_val(r, o, w, By, t, with_yy) for t in cands)


def delta_rep(B_R, B_O, B_y, safeguard=1e-12):
    """Replace-one l2 sensitivity of (triu B, C, c_R, c_O, y'y) and of the release without y'y.
    Exact enumeration over the stationary points of F(c*(t), t) (three cubics) and the region boundaries;
    rounded up by a relative `safeguard`.  Returns (Delta, Delta_noyy)."""
    r, o, w = B_R ** 2, B_O ** 2, B_y ** 2
    D2 = _sup_F(r, o, w, B_y, True)
    D2n = _sup_F(r, o, w, B_y, False)
    return math.sqrt(D2) * (1 + safeguard), math.sqrt(D2n) * (1 + safeguard)


def delta_rep_closed(B_R, B_O, B_y):
    """Closed forms of regimes 1 and 2 (for cross-checking); None in regime 3."""
    r, o, w = B_R ** 2, B_O ** 2, B_y ** 2
    if abs(r - w) <= 2 * o:
        return math.sqrt(0.5 * (2 * o + r + w) ** 2 + 2 * r * w)
    if r - w >= 2 * o:
        return math.sqrt(4 * r * (o + w))
    return None


def delta_G(B_R, B_O):
    r, o = B_R ** 2, B_O ** 2
    return (2 * o + r) / math.sqrt(2) if r <= 2 * o else 2 * math.sqrt(r * o)


# =============================================================================================
# tight zCDP -> (eps, delta) conversion (Canonne, Kamath and Steinke 2020); port of
# experiments/calibration_protocol.py
# =============================================================================================


def log_delta_cks(rho, eps):
    f = lambda t: ((a := 1 + np.exp(t)) - 1) * (a * rho - eps) - np.log(a - 1) + a * np.log1p(-1 / a)
    return minimize_scalar(f, bounds=(-15, 15), method="bounded", options={"xatol": 1e-11}).fun


def rho_tight(eps, delta):
    return brentq(lambda r: log_delta_cks(r, eps) - np.log(delta), 1e-12, 50 * eps, xtol=1e-15)


def zcdp_to_eps(rho, delta):
    """Smallest eps such that rho-zCDP implies (eps, delta)-DP under the same conversion."""
    return brentq(lambda e: log_delta_cks(rho, e) - np.log(delta), 1e-9, 1e3, xtol=1e-12)


# =============================================================================================
# Calibrate
# =============================================================================================


@dataclass
class Agreement:
    """What both parties agree on in Setup (Algorithm 1, line 1) plus the public sizes.  The defaults are the
    previous evaluation setting ('prev' in SETTINGS), not part of the protocol; the paper's two evaluation settings,
    inference (I) and prediction (P), are applied with Agreement(..., **setting('I')) or setting('P')."""
    n_R_max: int
    n_bar_O: int
    d_R: int
    d_O: int
    B_R: float                     # covers R's intercept: ||xdot|| <= B_R
    B_O: float
    B_y: float
    eps_run: float = 1.0
    delta_run: float = 1e-9
    f1: float = 0.1
    f_y: float = 0.1               # share of a separate y'y release (the protocol has no default; our choice)
    theta_star: float = 2.5
    alpha: float = 0.05
    alpha_L: float | None = None   # stage-1 error level; None: alpha / 20
    alpha_U: float | None = None   # error level of U_yy; None: alpha / 20
    r1_mode: str = 'union'         # stage-1 radius: 'union' (union bound) or 'exact' (Monte Carlo quantile, solve.py)
    intervals: bool = True         # False: estimate only, no interval guarantee (alpha_CI need not be positive)
    B_beta_pub: float = float('inf')
    kappa_Psi: float = 2.0
    kappa: int | str = 'auto'      # flooding parameter: an integer, or 'auto' (derived by Calibrate)
    zeta: float = 0.1              # kappa='auto': slack (1 + e^eps) gamma at most zeta delta_run
    kappa_min: int = 40            # kappa='auto': statistical security floor
    delta_B: float = 2.0 ** -50
    t_eta: float = 10.0            # q0 margin for the noise, in units of sigma_c (correctness 2N exp(-t_eta^2/2))
    lambda_sec: int = 128          # security level of the modulus chain (log QP bound for N)
    # ledger cap of the base register (public).  None: a single-run ledger whose cap is this run's budget.
    ledger_eps: float | None = None
    ledger_delta: float | None = None
    ledger_zeta: float = 0.1       # share of delta_L reserved for the statistical slack of all runs
    ledger_mode: str = 'filter'    # 'filter' (adaptively chosen runs) or 'fixed' (runs fixed in advance)
    variant: str = 'A'             # 'A' circuit as written; 'C' restructured (proof sketched only)
    # R's encoding scale: 'auto' = 2^68 for circuit A (decided: keeps B_tot and Omega_f independent of n_bar_O),
    # q2 for circuit C; None = q2 (the prime of the first rescale); a number = 2^log2_Omega_U
    log2_Omega_U: float | str | None = 'auto'
    researcher: str = 'R'
    project: str = 'demo'


LOG2_OMEGA_U_A = 68.0   # R's encoding scale for circuit A

# Named evaluation settings of the statistical parameters (paper, Section "Parameters").  None of them is part
# of the protocol.  'prev' is the earlier evaluation setting (the Agreement defaults).  'I' (inference) keeps the
# interval guarantee with the exact stage-1 radius and a larger stage-1 share; 'P' (prediction) is tuned for test
# error only and gives up the interval guarantee.  I and P were chosen on the synthetic population only, by the
# rule pre-registered in reports/v31_settings/settings_choice.md.
SETTINGS = {
    'prev': dict(f1=0.1, theta_star=2.5, kappa_Psi=2.0, alpha=0.05, alpha_L=None, alpha_U=None, r1_mode='union',
                 intervals=True),
    'I': dict(f1=0.25, theta_star=2.5, kappa_Psi=2.0, alpha=0.05, alpha_L=0.0025, alpha_U=0.0025, r1_mode='exact',
              intervals=True),
    'P': dict(f1=0.2, theta_star=0.5, kappa_Psi=2.0, alpha=0.05, alpha_L=0.5, alpha_U=0.0025, r1_mode='exact',
              intervals=False),
}


def setting(name: str) -> dict:
    """The Agreement fields of a named evaluation setting (a copy)."""
    if name not in SETTINGS:
        raise ValueError(f'unknown setting {name!r}; known: {sorted(SETTINGS)}')
    return dict(SETTINGS[name])


def alpha_split(ag: 'Agreement'):
    """(alpha_L, alpha_U, alpha_CI) of an agreement.  alpha_L and alpha_U default to alpha / 20.  With intervals
    on, alpha = alpha_CI + alpha_L + alpha_U needs alpha_CI > 0; with intervals off alpha_L only sizes the stage-1
    radius and may be any value in (0, 1)."""
    aL = ag.alpha / 20 if ag.alpha_L is None else float(ag.alpha_L)
    aU = ag.alpha / 20 if ag.alpha_U is None else float(ag.alpha_U)
    if not (0 < aL < 1 and 0 < aU < 1):
        raise ValueError('alpha_L and alpha_U must lie in (0, 1)')
    if ag.r1_mode not in ('union', 'exact'):
        raise ValueError(f"r1_mode must be 'union' or 'exact', not {ag.r1_mode!r}")
    aCI = ag.alpha - aL - aU
    if ag.intervals and not aCI > 0:
        raise ValueError('with intervals on, alpha_L + alpha_U must be below alpha')
    return aL, aU, aCI


def resolve_log2_Omega_U(log2_Omega_U, variant):
    """'auto' -> 2^68 for circuit A and q2 (None) for circuit C; anything else is returned unchanged."""
    if log2_Omega_U == 'auto':
        return LOG2_OMEGA_U_A if variant == 'A' else None
    return log2_Omega_U


def scales_from_primes(q1: int, q2: int, log2_Omega_U=None):
    """Masks at Omega_u = q1 (exact prime); R encodes at an exactly representable 2^log2_Omega_U, or at
    Omega_U = q2 if None.  O's data scale Omega_V is chosen by Calibrate; Omega_f = Omega_U Omega_V Omega_u / (q1 q2)."""
    OmU = Fraction(q2) if log2_Omega_U is None else Fraction(admissible_scale(log2_Omega_U))
    return OmU, Fraction(q1)


def admissible_scale(log2_omega: float) -> float:
    """A scale exactly representable as a double: 2^e times a 20-bit odd-ish mantissa."""
    e = math.floor(log2_omega) - 20
    mant = round(2 ** (log2_omega - e))
    return float(mant) * 2.0 ** e


class CalibrationRefused(RuntimeError):
    """Calibrate found no admissible derived parameters; O refuses the run."""


# Largest log2(QP) at 128-bit classical security for uniform ternary secrets, per ring dimension
# (Homomorphic Encryption Security Standard; OpenFHE stdlatticeparms, HEStd_ternary / HEStd_128_classic).
LOG2_QP_MAX = {128: {1024: 27, 2048: 54, 4096: 109, 8192: 218, 16384: 438, 32768: 881, 65536: 1747,
                     131072: 3523}}


def slack_per_release(kappa: int, delta_B: float) -> float:
    """Statistical distance charged per release: delta_B + 2^-kappa (1 + 2^-80) + 2^-97 (Lemma smudge)."""
    return delta_B + 2.0 ** -kappa * (1 + 2.0 ** -80) + 2.0 ** -97


def _kappa_for_slack(budget: float, n_rel: int, delta_B: float, kappa_max: int = 256):
    """Smallest integer kappa >= 1 with n_rel * slack_per_release(kappa) <= budget, or None."""
    for k in range(1, kappa_max + 1):
        if n_rel * slack_per_release(k, delta_B) <= budget:
            return k
    return None


def ledger_cap(ag: Agreement):
    """(eps_L, delta_L, zeta_L, mode) of the base register.  Without an explicit cap the ledger holds this one
    run: cap = (eps_run, delta_run), checked with the exact (non-adaptive) rule."""
    if ag.ledger_eps is None:
        return ag.eps_run, ag.delta_run, ag.zeta, 'fixed'
    if ag.ledger_delta is None:
        raise ValueError('ledger_delta must be given with ledger_eps')
    return float(ag.ledger_eps), float(ag.ledger_delta), float(ag.ledger_zeta), ag.ledger_mode


def choose_kappa(ag: Agreement, n_rel: int):
    """Resolve the flooding parameter.  Returns (kappa, info).  An integer agreement.kappa is used as given;
    'auto' gives kappa = max(kappa_min, kappa_run, kappa_L) (see the module docstring)."""
    if not (isinstance(ag.kappa, str) and ag.kappa == 'auto'):
        return int(ag.kappa), dict(rule='fixed')
    if not 0 < ag.zeta < 1:
        raise ValueError('zeta must lie in (0, 1)')
    k_run = _kappa_for_slack(ag.zeta * ag.delta_run / (1 + math.exp(ag.eps_run)), n_rel, ag.delta_B)
    if k_run is None:
        raise CalibrationRefused(f'no kappa keeps (1 + e^eps) gamma <= zeta delta_run: n_rel (1 + e^eps)'
                                 f' (delta_B + 2^-97) alone exceeds it; lower delta_B')
    info = dict(rule='auto', zeta=ag.zeta, kappa_min=ag.kappa_min, kappa_run=k_run, kappa_L=None)
    ks = [ag.kappa_min, k_run]
    if ag.ledger_eps is not None:
        eps_L, delta_L, zeta_L, _ = ledger_cap(ag)
        rho_L = rho_tight(eps_L, (1 - zeta_L) * delta_L)
        Gamma_L = zeta_L * delta_L / (1 + math.exp(eps_L))
        rho_low = rho_tight(ag.eps_run, (1 - ag.zeta) * ag.delta_run)   # rho_run >= rho_low once kappa >= kappa_run
        k_L = _kappa_for_slack(rho_low / rho_L * Gamma_L, n_rel, ag.delta_B)
        if k_L is None:
            raise CalibrationRefused('the ledger slack share of this run is below n_rel (delta_B + 2^-97); '
                                     'lower delta_B or the ledger eps')
        info.update(kappa_L=k_L, rho_L=rho_L, Gamma_L=Gamma_L)
        ks.append(k_L)
    return max(ks), info


def calibrate(ag: Agreement, N: int, q1: int, q2: int, q0: int, gadget: dict, log2_grid_step=1 / 32,
              log2_start=40.0):
    """Algorithm 7, Calibrate.  `gadget` holds the key-switching constants of the context (public).
    q0 is the product of the last-level primes; log QP = log2(q0 q1 q2) + log2(P) is checked against the
    security bound for N.  Returns a JSON-able dict with kappa, Omega_f, sigma_c^2 (integers), rho_s, sigma_s,
    B_tot, Delta_x and the q0 and log QP checks.  Raises CalibrationRefused when the parameters are not admissible."""
    Delta, Delta_noyy = delta_rep(ag.B_R, ag.B_O, ag.B_y)
    DG = delta_G(ag.B_R, ag.B_O)
    Dy = ag.B_y ** 2
    joint = (Delta - Delta_noyy) <= 1e-9 * Delta
    n_rel = 2 if joint else 3
    log2_QP = math.log2(q0) + math.log2(q1) + math.log2(q2) + math.log2(gadget['P'])
    QP_max = LOG2_QP_MAX.get(ag.lambda_sec, {}).get(N)
    if QP_max is None:
        raise CalibrationRefused(f'no security bound tabulated for lambda_sec = {ag.lambda_sec}, N = {N}')
    if log2_QP > QP_max:
        raise CalibrationRefused(f'log QP = {log2_QP:.1f} exceeds the {ag.lambda_sec}-bit bound {QP_max} for N = {N}')
    kappa, kappa_info = choose_kappa(ag, n_rel)
    gamma = n_rel * slack_per_release(kappa, ag.delta_B)
    delta_DP = ag.delta_run - (1 + math.exp(ag.eps_run)) * gamma
    if delta_DP <= 0:
        raise CalibrationRefused(f"delta_run too small for the statistical slack at kappa = {kappa}")
    rho_run = rho_tight(ag.eps_run, delta_DP)
    rho = {1: ag.f1 * rho_run}
    if joint:
        rho[2] = rho_run - rho[1]
        sens = {1: DG, 2: Delta}
    else:
        rho['y'] = ag.f_y * rho_run
        rho[2] = rho_run - rho[1] - rho['y']
        sens = {1: DG, 2: Delta_noyy, 'y': Dy}
    stages = list(sens.keys())
    OmU, Omu = scales_from_primes(q1, q2, resolve_log2_Omega_U(ag.log2_Omega_U, ag.variant))

    def cparams(Omega_V):
        return bt.CircuitParams(N=N, n_bar=ag.n_bar_O, d_R=ag.d_R, d_O=ag.d_O, B_R=ag.B_R, B_O=ag.B_O, B_y=ag.B_y,
                                q1=q1, q2=q2, Omega_U=float(OmU), Omega_V=Omega_V, Omega_u=float(Omu),
                                delta_B=ag.delta_B, **gadget)

    chosen = None
    lg = log2_start
    while lg < 120:
        # target final scale 2^lg; O's data scale Omega_V (exactly representable), exact Omega_f from it
        OmV = admissible_scale(lg + math.log2(q1) + math.log2(q2) - math.log2(OmU) - math.log2(Omu))
        Om = float(OmU * Fraction(OmV) * Omu / (q1 * q2))
        cp = cparams(OmV)
        ok, rows = True, {}
        for s in stages:
            Bt = bt.B_tot(cp, s, joint, ag.variant)
            Dx = bt.delta_x(cp, s, joint, sens[s])
            Denc = math.sqrt(2 / N) * Om * sens[s] + Dx + math.sqrt(N)
            sig2 = math.ceil(Fraction(Denc) ** 2 / (2 * Fraction(rho[s])))     # integer sigma_c^2
            sig = math.sqrt(sig2)
            need = max(2.0 ** kappa * Bt / math.sqrt(2 * math.pi), 2.0 ** 20)   # Theorem output: both floors
            rows[s] = dict(B_tot=Bt, Delta_x=Dx, Delta_enc=Denc, sigma_c2=int(sig2), sigma_c=sig,
                           rho=rho[s], rho_exact=Denc ** 2 / (2 * sig2), flood_needed=need,
                           sigma_slot=sig * math.sqrt(N / 2) / Om, Delta=sens[s])
            ok &= sig >= need
        if ok:
            chosen = (Om, OmV, rows)
            break
        lg += log2_grid_step
    if chosen is None:
        raise CalibrationRefused(f"no admissible scale below 2^{lg:.0f} floods at kappa = {kappa}: the error bound "
                                 "grows as fast as the scale (raise Omega_U or lower kappa)")
    Om, OmV, rows = chosen
    r, o, w = ag.B_R ** 2, ag.B_O ** 2, ag.B_y ** 2
    S_max = ag.n_bar_O * max(r, o, w, math.sqrt(r * o), math.sqrt(r * w), math.sqrt(o * w))
    sig_max = max(v['sigma_c'] for v in rows.values())
    B_max = max(v['B_tot'] for v in rows.values())
    q0_need = 2 * (Om * S_max + 2.0 ** kappa * B_max + ag.t_eta * sig_max)
    nE = bt.entry_counts(ag.d_R, ag.d_O, joint)
    q0_need_Q = max(2 * (Om * 2 * nE[s] / N * S_max + 2.0 ** 35 + 0.5 + rows[s]['B_tot']
                         + ag.t_eta * rows[s]['sigma_c']) for s in stages)
    if q0 <= q0_need:
        # the log QP a chain with a large enough q0 would have, for the operator
        QP_need = log2_QP - math.log2(q0) + math.log2(q0_need)
        verdict = 'within' if QP_need <= QP_max else 'above'
        raise CalibrationRefused(f"q0 = 2^{math.log2(q0):.1f} too small: need 2^{math.log2(q0_need):.1f}; a chain with "
                                 f"that q0 has log QP = {QP_need:.1f}, {verdict} the bound {QP_max} for N = {N}")
    return dict(Delta=Delta, Delta_noyy=Delta_noyy, Delta_G=DG, Delta_y=Dy, joint=joint, n_rel=n_rel,
                gamma=gamma, slack=(1 + math.exp(ag.eps_run)) * gamma,
                slack_fraction=(1 + math.exp(ag.eps_run)) * gamma / ag.delta_run, delta_DP=delta_DP,
                rho_run=rho_run, Omega_f=Om, log2_Omega_f=math.log2(Om),
                Omega_U=(int(OmU) if OmU.denominator == 1 else float(OmU)), Omega_u=int(Omu), Omega_V=OmV,
                q0_bits=math.log2(q0), q0_need_bits=math.log2(q0_need), q0_need_Q_bits=math.log2(q0_need_Q),
                log2_QP=log2_QP, log2_QP_max=QP_max,
                releases={str(s): v for s, v in rows.items()}, variant=ag.variant, kappa=kappa,
                kappa_info=kappa_info, delta_B=ag.delta_B, entries={str(k): v for k, v in nE.items()})


# =============================================================================================
# Ledger (per base register; shared by all researchers) and OPRF quota
# =============================================================================================


class Ledger:
    """O's ledger for one base register, shared by all researchers and scopes (paper, Section 7.4).

    Cap (eps_cap, delta_cap) and the share zeta of delta_cap reserved for statistical slack.  zCDP costs add;
    the raw statistical slacks gamma of the runs add and are converted at the cap's eps:
      mode 'fixed'  (runs fixed in advance): admit iff
                    delta(rho_spent + rho_run, eps_cap) + (1 + e^eps_cap)(gamma_spent + gamma_run) <= delta_cap;
      mode 'filter' (runs chosen after seeing earlier outputs): admit iff rho_spent + rho_run <= rho_cap and
                    gamma_spent + gamma_run <= Gamma_cap, with rho_cap = rho_tight(eps_cap, (1 - zeta) delta_cap)
                    and Gamma_cap = zeta delta_cap / (1 + e^eps_cap)  (a privacy filter: Feldman and Zrnic 2021,
                    Theorem 1.1, at every Renyi order).
    Either way the joint view of all researchers is (eps_cap, delta_cap)-IND-CDP.  delta(rho, eps) is the
    conversion of Calibrate.  The cap is fixed when the ledger file is created."""

    VERSION = 2

    def __init__(self, path: str, eps_cap: float, delta_cap: float, zeta: float = 0.1, mode: str = 'filter',
                 quota: int = 100):
        if mode not in ('filter', 'fixed'):
            raise ValueError(mode)
        self.path = path
        if os.path.exists(path):
            with open(path) as fh:
                self.d = json.load(fh)
            if self.d.get('version') != self.VERSION:
                raise RuntimeError(f'{path}: ledger file of an older format; it cannot be converted')
            if (self.d['eps_cap'], self.d['delta_cap'], self.d['zeta'], self.d['mode']) != \
                    (eps_cap, delta_cap, zeta, mode):
                raise RuntimeError(f'{path}: the ledger cap differs from the agreement')
        else:
            self.d = dict(version=self.VERSION, eps_cap=eps_cap, delta_cap=delta_cap, zeta=zeta, mode=mode,
                          rho_cap=rho_tight(eps_cap, (1 - zeta) * delta_cap),
                          gamma_cap=zeta * delta_cap / (1 + math.exp(eps_cap)),
                          rho_spent=0.0, gamma_spent=0.0, releases=[], oprf_runs={}, quota=quota)

    def save(self):
        with open(self.path, 'w') as fh:
            json.dump(self.d, fh, indent=1)

    def admits(self, rho_add: float, gamma_add: float) -> bool:
        rho = self.d['rho_spent'] + rho_add
        gam = self.d['gamma_spent'] + gamma_add
        if self.d['mode'] == 'filter':
            tol = 1e-12
            return rho <= self.d['rho_cap'] * (1 + tol) and gam <= self.d['gamma_cap'] * (1 + tol)
        slack = (1 + math.exp(self.d['eps_cap'])) * gam
        if slack >= self.d['delta_cap']:
            return False
        if rho <= 0:
            return True
        # delta(rho, eps_cap) + slack <= delta_cap, with a relative tolerance for a run that fills the cap exactly
        return log_delta_cks(rho, self.d['eps_cap']) <= math.log(self.d['delta_cap'] - slack) + 1e-9

    def check_run(self, rho_run, gamma_run, researcher, project):
        key = f"{researcher}|{project}"
        if self.d['oprf_runs'].get(key, 0) >= self.d['quota']:
            return False, 'OPRF quota exceeded'
        if not self.admits(rho_run, gamma_run):
            return False, 'ledger cap would be exceeded'
        return True, 'ok'

    def log_oprf_run(self, researcher, project):
        key = f"{researcher}|{project}"
        self.d['oprf_runs'][key] = self.d['oprf_runs'].get(key, 0) + 1
        self.save()

    def charge(self, stage, rho, gamma):
        """Charge one release: its zCDP cost and its raw statistical slack gamma_s."""
        self.d['rho_spent'] += rho
        self.d['gamma_spent'] += gamma
        self.d['releases'].append(dict(stage=str(stage), rho=rho, gamma=gamma))
        self.save()
