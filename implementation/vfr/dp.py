"""Phase 0 calibration: replace-one sensitivities, joint/separate y'y mode, tight zCDP conversion,
Calibrate (scale chosen so that the DP noise floods the worst-case error), and O's ledger.

Follows paper/algorithms.tex (Algorithm 7, corrected): the delta slack has the factor (1 + e^eps) and the
2^-80, 2^-97 terms; the encoded sensitivity includes Delta_x (centred mechanism, exact final scale);
sigma_c^2 is the integer ceil(Delta_enc^2 / (2 rho)) so that the exact sampler gets a rational variance.
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
    """What both parties agree on in Setup (Algorithm 1, line 1) plus the public sizes."""
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
    f_y: float = 0.1               # share of a separate y'y release (no default in the spec; our choice)
    theta_star: float = 2.5
    alpha: float = 0.05
    B_beta_pub: float = float('inf')
    kappa_Psi: float = 2.0
    kappa: int = 40
    delta_B: float = 2.0 ** -50
    variant: str = 'A'             # 'A' circuit as written; 'C' restructured (proof sketched only)
    # R's encoding scale: 'auto' = 2^68 for circuit A (decided: keeps B_tot and Omega_f independent of n_bar_O),
    # q2 for circuit C; None = q2 (the prime of the first rescale); a number = 2^log2_Omega_U
    log2_Omega_U: float | str | None = 'auto'
    researcher: str = 'R'
    project: str = 'demo'


LOG2_OMEGA_U_A = 68.0   # R's encoding scale for circuit A


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


def calibrate(ag: Agreement, N: int, q1: int, q2: int, q0: int, gadget: dict, log2_grid_step=1 / 32,
              log2_start=40.0):
    """Algorithm 7, Calibrate.  `gadget` holds the key-switching constants of the context (public).
    Returns a JSON-able dict with Omega_f, sigma_c^2 (integers), rho_s, sigma_s, B_tot, Delta_x, q0 checks."""
    Delta, Delta_noyy = delta_rep(ag.B_R, ag.B_O, ag.B_y)
    DG = delta_G(ag.B_R, ag.B_O)
    Dy = ag.B_y ** 2
    joint = (Delta - Delta_noyy) <= 1e-9 * Delta
    n_rel = 2 if joint else 3
    gamma = n_rel * (ag.delta_B + 2.0 ** -ag.kappa * (1 + 2.0 ** -80) + 2.0 ** -97)
    delta_DP = ag.delta_run - (1 + math.exp(ag.eps_run)) * gamma
    if delta_DP <= 0:
        raise ValueError("delta_run too small for the statistical slack")
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
            need = 2.0 ** ag.kappa * Bt / math.sqrt(2 * math.pi)
            rows[s] = dict(B_tot=Bt, Delta_x=Dx, Delta_enc=Denc, sigma_c2=int(sig2), sigma_c=sig,
                           rho=rho[s], rho_exact=Denc ** 2 / (2 * sig2), flood_needed=need,
                           sigma_slot=sig * math.sqrt(N / 2) / Om, Delta=sens[s])
            ok &= sig >= need
        if ok:
            chosen = (Om, OmV, rows)
            break
        lg += log2_grid_step
    if chosen is None:
        raise RuntimeError("no admissible scale found")
    Om, OmV, rows = chosen
    r, o, w = ag.B_R ** 2, ag.B_O ** 2, ag.B_y ** 2
    S_max = ag.n_bar_O * max(r, o, w, math.sqrt(r * o), math.sqrt(r * w), math.sqrt(o * w))
    sig_max = max(v['sigma_c'] for v in rows.values())
    B_max = max(v['B_tot'] for v in rows.values())
    q0_need = 2 * (Om * S_max + 2.0 ** ag.kappa * B_max + 10 * sig_max)
    nE = bt.entry_counts(ag.d_R, ag.d_O, joint)
    q0_need_Q = max(2 * (Om * 2 * nE[s] / N * S_max + 2.0 ** 35 + 0.5 + rows[s]['B_tot'] + 10 * rows[s]['sigma_c'])
                    for s in stages)
    if q0 <= q0_need:
        raise RuntimeError(f"q0 = 2^{math.log2(q0):.1f} too small: need 2^{math.log2(q0_need):.1f}")
    return dict(Delta=Delta, Delta_noyy=Delta_noyy, Delta_G=DG, Delta_y=Dy, joint=joint, n_rel=n_rel,
                gamma=gamma, delta_DP=delta_DP, rho_run=rho_run, Omega_f=Om, log2_Omega_f=math.log2(Om),
                Omega_U=(int(OmU) if OmU.denominator == 1 else float(OmU)), Omega_u=int(Omu), Omega_V=OmV, q0_bits=math.log2(q0), q0_need_bits=math.log2(q0_need),
                q0_need_Q_bits=math.log2(q0_need_Q),
                releases={str(s): v for s, v in rows.items()}, variant=ag.variant, kappa=ag.kappa,
                delta_B=ag.delta_B, entries={str(k): v for k, v in nE.items()})


# =============================================================================================
# Ledger (per base register; shared by all researchers) and OPRF quota
# =============================================================================================


class Ledger:
    """rho is additive under (non-adaptive) composition; the cap is (eps_cap, delta_cap), checked through
    the same tight conversion.  The cap and the composition rule are not fixed by the specification
    (paper tocheck); this is a minimal, explicit choice."""

    def __init__(self, path: str, eps_cap: float = 10.0, delta_cap: float = 1e-6, quota: int = 100):
        self.path = path
        if os.path.exists(path):
            with open(path) as fh:
                self.d = json.load(fh)
        else:
            self.d = dict(eps_cap=eps_cap, delta_cap=delta_cap, rho_spent=0.0, gamma_spent=0.0, releases=[],
                          oprf_runs={}, quota=quota)

    def save(self):
        with open(self.path, 'w') as fh:
            json.dump(self.d, fh, indent=1)

    def eps_if(self, rho_add, slack_add):
        """eps of everything charged so far plus (rho_add, slack_add) at the cap's delta.  `slack` is the
        statistical delta of each run, (1 + e^eps_run) gamma (Calibrate), which is subtracted from delta_cap."""
        rho = self.d['rho_spent'] + rho_add
        dl = self.d['delta_cap'] - (self.d['gamma_spent'] + slack_add)
        if dl <= 0:
            return float('inf')
        return 0.0 if rho <= 0 else zcdp_to_eps(rho, dl)

    def check_run(self, rho_run, slack, researcher, project):
        key = f"{researcher}|{project}"
        if self.d['oprf_runs'].get(key, 0) >= self.d['quota']:
            return False, 'OPRF quota exceeded'
        if self.eps_if(rho_run, slack) > self.d['eps_cap']:
            return False, 'ledger cap would be exceeded'
        return True, 'ok'

    def log_oprf_run(self, researcher, project):
        key = f"{researcher}|{project}"
        self.d['oprf_runs'][key] = self.d['oprf_runs'].get(key, 0) + 1
        self.save()

    def charge(self, stage, rho, slack):
        self.d['rho_spent'] += rho
        self.d['gamma_spent'] += slack
        self.d['releases'].append(dict(stage=str(stage), rho=rho, slack=slack))
        self.save()
