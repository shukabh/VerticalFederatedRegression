"""Phase 0 calibration: replace-one sensitivities, joint/separate y'y mode, tight zCDP conversion,
Calibrate (scale chosen so that the DP noise floods the worst-case error), and O's ledger.

Follows paper/algorithms.tex (Algorithm 7, Calibrate): the statistical slack is
gamma = n_rel (delta_B + 2^-kappa (1 + 2^-80) + 2^-97) and delta_DP = delta_run - (1 + e^eps) gamma; the encoded
sensitivity includes Delta_x (centred mechanism, exact final scale); sigma_c^2 is the integer
ceil(Delta_enc^2 / (2 rho)) so that the exact sampler gets a rational variance.

The flooding parameter kappa is derived unless an integer is given (kappa='auto', the default):
kappa = max(kappa_min, kappa_run, kappa_L), where kappa_run is the smallest integer with
(1 + e^eps_run) gamma <= zeta delta_run, and kappa_L (only when the agreement names a ledger cap) makes the ledger
run out of zCDP budget before it runs out of the budget that the smudging step uses.  With the hockey-stick ledger
(the default, ledger_xi = xi_L > 0; composition proof, Theorem 5.2, Definition 5.3, Proposition 5.5) kappa_L is the
smallest integer with n_rel eps'(kappa, delta') <= (rho~_run / rho_L) E'_L, E'_L = xi_L eps_L / 2; with the earlier
statistical-distance ledger (ledger_xi = None) it is the smallest integer with gamma <= (rho~_run / rho_L) Gamma_L.
Calibrate refuses (CalibrationRefused) when no kappa meets the rule, when no scale floods, when q0 is too small,
or when the modulus chain exceeds the security bound for N.

Rounding: every capacity of the ledger (rho_L, E'_L, Gamma_L) is rounded down and every cost is rounded up, the
ledger keeps its sums as exact fractions, and its tests have no tolerance.
"""
from __future__ import annotations

import contextlib
import fcntl
import json
import math
import os
import time
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
    """log delta(rho, eps), evaluated at the order a the minimizer finds.  Every order a > 1 gives an upper bound on
    the infimum, so the value returned is an upper bound on log delta(rho, eps) up to floating-point rounding."""
    f = lambda t: ((a := 1 + np.exp(t)) - 1) * (a * rho - eps) - np.log(a - 1) + a * np.log1p(-1 / a)
    return minimize_scalar(f, bounds=(-15, 15), method="bounded", options={"xatol": 1e-11}).fun


# Margin on log delta that covers the floating-point error of evaluating the objective (about 1e-15).  rho_tight
# leaves twice this margin, so a run that fills a cap exactly is still admitted by a test that keeps this margin.
_LOG_MARGIN = 1e-13
_REL = 1e-12            # relative margin by which capacities are rounded down and smudging costs rounded up


def _down(x: float) -> float:
    """The largest float strictly below x."""
    return math.nextafter(x, -math.inf)


def _up(x: float) -> float:
    """The smallest float strictly above x."""
    return math.nextafter(x, math.inf)


def _ceil_float(fr: Fraction) -> float:
    """The smallest float >= the rational fr."""
    f = float(fr)
    return f if Fraction(f) >= fr else _up(f)


def rho_tight(eps, delta):
    """Largest rho with delta(rho, eps) <= delta under the tight conversion, rounded DOWN: the value returned
    satisfies log_delta_cks(rho, eps) <= log(delta) - 2 * _LOG_MARGIN, and log_delta_cks is an upper bound on the
    true log delta, so delta(rho, eps) <= delta holds for the value returned (never a root slightly above it)."""
    target = math.log(delta) - 2 * _LOG_MARGIN
    g = lambda r: log_delta_cks(r, eps) - target
    r = brentq(g, 1e-12, 50 * eps, xtol=1e-15)
    while g(r) > 0:                                  # brentq may return a point just above the root: step down
        r *= 1 - 1e-14
    return r


def zcdp_to_eps(rho, delta):
    """Smallest eps such that rho-zCDP implies (eps, delta)-DP under the same conversion."""
    return brentq(lambda e: log_delta_cks(rho, e) - np.log(delta), 1e-9, 1e3, xtol=1e-12)


def eps_smudge(kappa: int, delta_p: float) -> float:
    """eps'(kappa, delta') of the hockey-stick smudging lemma (composition proof, Lemma 5.1): the smallest eps with
    delta(pi 4^-kappa, eps) <= delta' under the tight conversion, returned rounded UP.

    delta(rho', eps) <= delta' iff eps >= g(a) for some order a > 1, where
        g(a) = a rho' + (ln(1/delta') - ln a) / (a - 1) - ln(1 + 1/(a - 1)),
    so eps' = max(0, inf_a g(a)), and g at ANY order is a valid eps' (never too small).  The minimizer's order is used,
    and the result is raised by a relative 1e-12 of the terms of g, which covers the floating-point error of evaluating
    it.  At delta' = 2^-100 this is 2.02e-11 at kappa = 40, against the closed form
    2 sqrt(pi ln(1/delta')) 2^-kappa + pi 4^-kappa = 2.68e-11."""
    rho = math.pi * 4.0 ** -kappa
    L = -math.log(delta_p)

    def terms(t):                                    # order a = 1 + e^t, written without cancellation for large a
        am1 = math.exp(t)
        a = 1.0 + am1
        return a * rho, (L - math.log(a)) / am1, -math.log1p(1.0 / am1)

    t0 = 0.5 * math.log(max(L, 1.0) / rho)           # the optimum is near a = sqrt(ln(1/delta') / rho')
    r = minimize_scalar(lambda t: sum(terms(t)), bounds=(t0 - 20.0, t0 + 20.0), method="bounded",
                        options={"xatol": 1e-10})
    tt = terms(r.x)
    return max(0.0, sum(tt)) + _REL * sum(abs(v) for v in tt)


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
    # share xi_L of eps_L reserved for the smudging of the encryption error (hockey-stick ledger, composition proof
    # Section 5); None selects the earlier statistical-distance ledger, which converts all of the slack at eps_L
    ledger_xi: float | None = 1e-3
    ledger_delta_prime: float = 2.0 ** -100   # per-release hockey-stick slack delta' of the smudging step
    variant: str = 'A'             # 'A' circuit as written; 'C' restructured (proof sketched only)
    # R's encoding scale: 'auto' = 2^68 for circuit A (decided: keeps B_tot and Omega_f independent of n_bar_O),
    # q2 for circuit C; None = q2 (the prime of the first rescale); a number = 2^log2_Omega_U
    log2_Omega_U: float | str | None = 'auto'
    # release timing (paper, "Release timing"): O holds every release until a deadline fixed before the first one,
    # measured from the receipt of the request, so that R's observed latency does not depend on the noise values
    # drawn by the (not constant-time) exact sampler.  'auto': deadline = release_pad_margin x the calibrated time
    # of one release (O times RELEASE_PAD_REPS dummy samplings at the largest sigma_c^2 of the run, with an
    # independent generator, plus one re-randomized partial decryption whose output is discarded); a number: that
    # deadline in seconds; None: no padding (earlier behaviour).  A release whose computation overruns the deadline
    # is refused (O sends 'refuse'; the run's cost stays charged, as for any abort).
    release_pad: float | str | None = 'auto'
    release_pad_margin: float = 2.0
    researcher: str = 'R'
    project: str = 'demo'


LOG2_OMEGA_U_A = 68.0   # R's encoding scale for circuit A
RELEASE_PAD_REPS = 3    # dummy samplings timed by O for release_pad='auto'

# Named evaluation settings of the statistical parameters (paper, Section "Parameters").  None of them is part
# of the protocol.  'prev' is the earlier evaluation setting (the Agreement defaults).  'I' (inference) keeps the
# interval guarantee with the exact stage-1 radius and a larger stage-1 share; 'P' (prediction) is tuned for test
# error only and gives up the interval guarantee.  All were chosen on the synthetic population only: I by the rule
# pre-registered in reports/v31_settings/settings_choice.md, and P by the rule in
# reports/v31_P2/settings_choice_P2.md, which adds a gate-refusal constraint and a block-normalized population.
# 'P0' is the first prediction setting, whose gate refused most releases on block-normalized real data; it is
# kept only to reproduce the earlier results.
SETTINGS = {
    'prev': dict(f1=0.1, theta_star=2.5, kappa_Psi=2.0, alpha=0.05, alpha_L=None, alpha_U=None, r1_mode='union',
                 intervals=True),
    'I': dict(f1=0.25, theta_star=2.5, kappa_Psi=2.0, alpha=0.05, alpha_L=0.0025, alpha_U=0.0025, r1_mode='exact',
              intervals=True),
    'P': dict(f1=0.25, theta_star=1.5, kappa_Psi=2.0, alpha=0.05, alpha_L=0.75, alpha_U=0.0025, r1_mode='exact',
              intervals=False),
    'P0': dict(f1=0.2, theta_star=0.5, kappa_Psi=2.0, alpha=0.05, alpha_L=0.5, alpha_U=0.0025, r1_mode='exact',
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
    check_release_pad(ag)
    return aL, aU, aCI


def check_release_pad(ag: 'Agreement'):
    """release_pad must be None, 'auto' (with a margin of at least 1) or a positive number of seconds."""
    rp = ag.release_pad
    if rp is None:
        return
    if rp == 'auto':
        if not float(ag.release_pad_margin) >= 1.0:
            raise ValueError('release_pad_margin must be at least 1')
        return
    if isinstance(rp, bool) or not isinstance(rp, (int, float)) or not (0 < float(rp) < float('inf')):
        raise ValueError(f"release_pad must be None, 'auto' or a positive number of seconds, not {rp!r}")


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


SIGMA_C_FLOOR = 2.0 ** 42   # floor on sigma_c: the factor 1 + 2^-80 of the smudging lemma needs sigma_c >= 2.1 * 2^40


# Relative margin by which the flooding test rounds B_tot and the needed sigma_c up.  B_tot is a sum of positive
# terms evaluated in double precision (relative error about 1e-15), so a margin of 2^-40 covers it many times over.
_FLOOD_REL = 2.0 ** -40


def flood_needed(B_tot: float, kappa: int) -> float:
    """sigma_c needed for flooding, max(2^kappa B_tot / sqrt(2 pi), SIGMA_C_FLOOR), with B_tot and the result
    rounded up by the relative margin _FLOOD_REL (so the float value is never below the exact one)."""
    B_up = B_tot * (1 + _FLOOD_REL)
    return max(_up(2.0 ** kappa * B_up / math.sqrt(2 * math.pi) * (1 + _FLOOD_REL)), SIGMA_C_FLOOR)


def floods(sigma_c2: int, B_tot: float, kappa: int) -> bool:
    """The flooding condition sigma_c >= flood_needed(B_tot, kappa), tested exactly: the integer sigma_c^2 against
    the square of the rounded-up float, as rationals.  Floating-point error can only make this test refuse a case
    that exact arithmetic would admit, never the converse."""
    return Fraction(int(sigma_c2)) >= Fraction(flood_needed(B_tot, kappa)) ** 2


def slack_per_release(kappa: int, delta_B: float) -> float:
    """Statistical distance charged per release: delta_B + 2^-kappa (1 + 2^-80) + 2^-97 (Lemma smudge), computed
    exactly and rounded up to a float (in double precision 2^-kappa (1 + 2^-80) would round down to 2^-kappa)."""
    return _ceil_float(Fraction(delta_B) + Fraction(1, 2 ** kappa) * (1 + Fraction(1, 2 ** 80)) + Fraction(1, 2 ** 97))


def _kappa_for_slack(budget: float, n_rel: int, delta_B: float, kappa_max: int = 256):
    """Smallest integer kappa >= 1 with n_rel * slack_per_release(kappa) <= budget, or None."""
    for k in range(1, kappa_max + 1):
        if n_rel * slack_per_release(k, delta_B) <= budget:
            return k
    return None


def _kappa_for_smudging(budget: float, n_rel: int, delta_p: float, kappa_max: int = 256):
    """Smallest integer kappa >= 1 with n_rel * eps'(kappa, delta') <= budget, or None (Proposition 5.5)."""
    for k in range(1, kappa_max + 1):
        if n_rel * eps_smudge(k, delta_p) <= budget:
            return k
    return None


def ledger_cap(ag: Agreement) -> dict:
    """The public ledger parameters of the base register: eps, delta, zeta, mode, xi, delta_prime.  Without an
    explicit cap the ledger holds this one run: cap = (eps_run, delta_run), checked with the exact (non-adaptive) rule
    of the statistical-distance accounting, which is the accounting of the per-run theorem (xi = None)."""
    if ag.ledger_eps is None:
        return dict(eps=float(ag.eps_run), delta=float(ag.delta_run), zeta=float(ag.zeta), mode='fixed', xi=None,
                    delta_prime=None)
    if ag.ledger_delta is None:
        raise ValueError('ledger_delta must be given with ledger_eps')
    xi = None if ag.ledger_xi is None else float(ag.ledger_xi)
    return dict(eps=float(ag.ledger_eps), delta=float(ag.ledger_delta), zeta=float(ag.ledger_zeta),
                mode=ag.ledger_mode, xi=xi, delta_prime=None if xi is None else float(ag.ledger_delta_prime))


def ledger_capacities(eps_L: float, delta_L: float, zeta_L: float, xi_L: float | None = None) -> dict:
    """Capacities of a ledger, each rounded down.
    Statistical-distance accounting (xi_L None): rho_L = largest rho with delta(rho, eps_L) <= (1 - zeta_L) delta_L,
      Gamma_L = zeta_L delta_L / (1 + e^eps_L).
    Hockey-stick accounting (composition proof, Definition 5.3): E'_L = xi_L eps_L / 2,
      rho_L = largest rho with delta(rho, eps_I) <= e^-E'_L (1 - zeta_L) delta_L at eps_I = eps_L - 2 E'_L = (1 - xi_L) eps_L,
      Gamma'_L = zeta_L delta_L / (1 + e^(eps_L - E'_L))."""
    if not 0 < zeta_L < 1:
        raise ValueError('zeta_L must lie in (0, 1)')
    if xi_L is None:
        return dict(rho=rho_tight(eps_L, (1 - zeta_L) * delta_L), E=None,
                    Gamma=(zeta_L * delta_L / (1 + math.exp(eps_L))) * (1 - _REL))
    if not 0 < xi_L < 1:
        raise ValueError('xi_L must lie in (0, 1)')
    E = (xi_L * eps_L / 2) * (1 - _REL)
    eps_I = eps_L - 2 * E
    return dict(rho=rho_tight(eps_I, math.exp(-E) * (1 - zeta_L) * delta_L), E=E, eps_I=eps_I,
                Gamma=(zeta_L * delta_L / (1 + math.exp(eps_L - E))) * (1 - _REL))


def run_cost(n_rel: int, kappa: int, delta_B: float, rho_run: float, xi: float | None, delta_prime: float | None):
    """The cost of a run that the ledger charges at admission, each part rounded up.  Statistical-distance accounting:
    (rho_run, gamma = n_rel (delta_B + 2^-kappa (1 + 2^-80) + 2^-97)).  Hockey-stick accounting: (rho_run,
    eps' = n_rel eps'(kappa, delta'), gamma = n_rel (delta_B + delta'))."""
    if xi is None:
        return dict(rho=float(rho_run), eps_prime=0.0, gamma=_ceil_float(n_rel * Fraction(slack_per_release(kappa, delta_B))))
    return dict(rho=float(rho_run), eps_prime=n_rel * eps_smudge(kappa, delta_prime) * (1 + _REL),
                gamma=_ceil_float(n_rel * (Fraction(delta_B) + Fraction(delta_prime))))


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
        led = ledger_cap(ag)
        cap = ledger_capacities(led['eps'], led['delta'], led['zeta'], led['xi'])
        rho_low = rho_tight(ag.eps_run, (1 - ag.zeta) * ag.delta_run)   # rho_run >= rho_low once kappa >= kappa_run
        share = rho_low / cap['rho']                                     # this run's share of the ledger
        if led['xi'] is None:                                            # statistical-distance ledger
            k_L = _kappa_for_slack(share * cap['Gamma'], n_rel, ag.delta_B)
            if k_L is None:
                raise CalibrationRefused('the ledger slack share of this run is below n_rel (delta_B + 2^-97); '
                                         'lower delta_B or the ledger eps')
            info.update(kappa_L=k_L, ledger_rule='sd', rho_L=cap['rho'], Gamma_L=cap['Gamma'])
        else:                                                            # hockey-stick ledger (Proposition 5.5)
            if n_rel * (ag.delta_B + led['delta_prime']) > share * cap['Gamma']:
                raise CalibrationRefused("the ledger slack share of this run is below n_rel (delta_B + delta'); "
                                         'lower delta_B or the ledger eps')
            k_L = _kappa_for_smudging(share * cap['E'], n_rel, led['delta_prime'])
            if k_L is None:
                raise CalibrationRefused('no kappa keeps the smudging cost within the run\'s share of E\'_L')
            info.update(kappa_L=k_L, ledger_rule='hs', rho_L=cap['rho'], E_L=cap['E'], Gamma_L=cap['Gamma'],
                        xi_L=led['xi'], delta_prime=led['delta_prime'])
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
    gamma = _ceil_float(n_rel * Fraction(slack_per_release(kappa, ag.delta_B)))
    delta_DP = ag.delta_run - _up((1 + math.exp(ag.eps_run)) * gamma)
    if delta_DP <= 0:
        raise CalibrationRefused(f"delta_run too small for the statistical slack at kappa = {kappa}")
    delta_DP = _down(delta_DP)
    rho_run = rho_tight(ag.eps_run, delta_DP)
    rho = {1: ag.f1 * rho_run}
    if joint:
        sens = {1: DG, 2: Delta}
    else:
        rho['y'] = ag.f_y * rho_run
        sens = {1: DG, 2: Delta_noyy, 'y': Dy}
    # stage 2 takes the rest, rounded down so that the planned costs sum to at most rho_run exactly
    rest = Fraction(rho_run) - sum(Fraction(v) for v in rho.values())
    rho[2] = float(rest) if Fraction(float(rest)) <= rest else _down(float(rest))
    led = ledger_cap(ag)
    ledger_cost = run_cost(n_rel, kappa, ag.delta_B, rho_run, led['xi'], led['delta_prime'])
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
            need = flood_needed(Bt, kappa)                                      # Theorem output: both floors
            rows[s] = dict(B_tot=Bt, Delta_x=Dx, Delta_enc=Denc, sigma_c2=int(sig2), sigma_c=sig,
                           rho=rho[s], rho_exact=Denc ** 2 / (2 * sig2), flood_needed=need,
                           sigma_slot=sig * math.sqrt(N / 2) / Om, Delta=sens[s])
            ok &= floods(sig2, Bt, kappa)
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
    # diagnostic only: the sharper condition (Q) of the proofs, with the proved decoding offset d_max
    # (crypto proofs, Proposition 4.5) for at most n_max = min(n_R^max, n_bar_O) matched rows
    n_max = min(ag.n_R_max, ag.n_bar_O)
    cp = cparams(OmV)
    for s in stages:
        rows[s]['d_max'] = bt.offset_bounds(cp, s, joint, n_max)['d_max']
    q0_need_Q = max(2 * (Om * 2 * nE[s] / N * S_max + rows[s]['d_max'] + 0.5 + rows[s]['B_tot']
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
                n_max_offset=n_max, log2_d_max=max(math.log2(rows[s]['d_max']) for s in stages),
                log2_QP=log2_QP, log2_QP_max=QP_max,
                releases={str(s): v for s, v in rows.items()}, variant=ag.variant, kappa=kappa,
                kappa_info=kappa_info, delta_B=ag.delta_B, entries={str(k): v for k, v in nE.items()},
                ledger=led, ledger_cost=ledger_cost)


# =============================================================================================
# Ledger (per base register; shared by all researchers) and OPRF quota
# =============================================================================================


class Ledger:
    """O's ledger for one base register, shared by all researchers and scopes (paper, Section 7.4; composition proof,
    Definitions 2.3 and 5.3).

    Parameters: the cap (eps_cap, delta_cap), the share zeta of delta_cap reserved for statistical slack, and, for the
    hockey-stick accounting (the default), the share xi of eps_cap reserved for the smudging of the encryption error
    and the per-release constant delta_prime.  xi=None selects the earlier statistical-distance accounting.

    A run's cost (Calibrate's 'ledger_cost') is charged IN FULL, for all its planned releases, when the run is
    admitted, under an exclusive file lock: the state is re-read from disk, tested and written back atomically, so
    two concurrently admitted runs, in one process or in several, can never exceed the cap together.  Nothing is
    refunded when a run is aborted or a release is never requested (the composition theorem is stated for this rule;
    refunding the delta_B part would not be sound).  The sums are kept as exact fractions, the capacities are rounded
    down and the costs up, and the tests have no tolerance.

    Statistical-distance accounting, costs (rho, gamma) with gamma = n_rel (delta_B + 2^-kappa (1 + 2^-80) + 2^-97):
      mode 'fixed'  (runs fixed in advance): admit iff delta(rho+, eps_cap) + (1 + e^eps_cap) Gamma+ <= delta_cap;
      mode 'filter' (runs chosen after seeing earlier outputs): admit iff rho+ <= rho_cap and Gamma+ <= Gamma_cap.
    Hockey-stick accounting, costs (rho, eps', gamma) with eps' = n_rel eps'(kappa, delta'), gamma = n_rel (delta_B + delta'):
      mode 'fixed':  admit iff e^E+ delta(rho+, eps_cap - 2 E+) + (1 + e^(eps_cap - E+)) Gamma+ <= delta_cap;
      mode 'filter': admit iff rho+ <= rho_cap, E+ <= E_cap and Gamma+ <= Gamma_cap.
    Here x+ is the sum after adding the run, and the capacities are those of ledger_capacities.  Either way the joint
    view of all researchers is (eps_cap, delta_cap)-IND-CDP.  The cap is fixed when the ledger file is created."""

    VERSION = 3

    def __init__(self, path: str, eps_cap: float, delta_cap: float, zeta: float = 0.1, mode: str = 'filter',
                 quota: int = 100, xi: float | None = 1e-3, delta_prime: float | None = 2.0 ** -100):
        if mode not in ('filter', 'fixed'):
            raise ValueError(mode)
        if xi is None:
            delta_prime = None
        elif delta_prime is None or not 0 < delta_prime < 1:
            raise ValueError('delta_prime must lie in (0, 1) with the hockey-stick accounting')
        self.path = path
        params = dict(eps_cap=float(eps_cap), delta_cap=float(delta_cap), zeta=float(zeta), mode=mode,
                      xi=None if xi is None else float(xi), delta_prime=None if delta_prime is None else float(delta_prime))
        with self._locked():
            if os.path.exists(path):
                self._load()
                if any(self.d.get(k) != v for k, v in params.items()):
                    raise RuntimeError(f'{path}: the ledger cap differs from the agreement')
            else:
                cap = ledger_capacities(eps_cap, delta_cap, zeta, xi)
                self.d = dict(version=self.VERSION, **params, accounting='sd' if xi is None else 'hs',
                              rho_cap=cap['rho'], E_cap=cap['E'], gamma_cap=cap['Gamma'],
                              rho_spent='0', eps_spent='0', gamma_spent='0', runs=[], oprf_runs={}, quota=quota)
                self._save()

    # -------------------------------------------------------------------------------- storage and locking
    @contextlib.contextmanager
    def _locked(self):
        """Exclusive lock on the ledger (a sidecar lock file, so that the atomic replace of the ledger file itself
        does not invalidate the lock)."""
        with open(self.path + '.lock', 'a') as lf:
            fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lf.fileno(), fcntl.LOCK_UN)

    def _load(self):
        with open(self.path) as fh:
            d = json.load(fh)
        if d.get('version') != self.VERSION:
            raise RuntimeError(f'{self.path}: ledger file of an older format; it cannot be converted')
        self.d = d

    def _save(self):
        tmp = f'{self.path}.tmp{os.getpid()}'
        with open(tmp, 'w') as fh:
            json.dump(self.d, fh, indent=1)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, self.path)

    def save(self):
        with self._locked():
            self._save()

    def refresh(self):
        """Re-read the state from disk (another process may have admitted runs)."""
        with self._locked():
            self._load()

    def spent(self) -> dict:
        """The sums charged so far, as exact fractions."""
        return {k: Fraction(self.d[f'{k}_spent']) for k in ('rho', 'eps', 'gamma')}

    # -------------------------------------------------------------------------------- the admission test
    def admits(self, cost: dict) -> bool:
        """Would a run of this cost be admitted on the state held in memory?  (admit_run re-reads the state first.)"""
        sp = self.spent()
        rho = sp['rho'] + Fraction(cost['rho'])
        E = sp['eps'] + Fraction(cost.get('eps_prime', 0.0))
        gam = sp['gamma'] + Fraction(cost['gamma'])
        hs = self.d['accounting'] == 'hs'
        if self.d['mode'] == 'filter':
            ok = rho <= Fraction(self.d['rho_cap']) and gam <= Fraction(self.d['gamma_cap'])
            return ok and (not hs or E <= Fraction(self.d['E_cap']))
        eps_cap, delta_cap = self.d['eps_cap'], self.d['delta_cap']
        E_up = _ceil_float(E)
        eps_eff = _down(eps_cap - 2 * E_up) if hs else eps_cap
        slack = _up((1 + math.exp(eps_cap - E_up if hs else eps_cap)) * _ceil_float(gam))
        if slack >= delta_cap or eps_eff <= 0:
            return False
        if rho == 0:
            return True
        lhs = log_delta_cks(_ceil_float(rho), eps_eff) + (E_up if hs else 0.0)
        return lhs <= math.log(_down(delta_cap - slack)) - _LOG_MARGIN

    def check_run(self, cost: dict, researcher, project):
        """Test the quota and the cap on the state held in memory, without charging anything."""
        key = f"{researcher}|{project}"
        if self.d['oprf_runs'].get(key, 0) >= self.d['quota']:
            return False, 'OPRF quota exceeded'
        if not self.admits(cost):
            return False, 'ledger cap would be exceeded'
        return True, 'ok'

    def admit_run(self, cost: dict, researcher, project, meta: dict | None = None):
        """Admission: under the lock, re-read the state, test the quota and the cap, and charge the run's full cost
        (all planned releases) at once.  Returns (admitted, reason).  Nothing is refunded later."""
        with self._locked():
            self._load()
            ok, why = self.check_run(cost, researcher, project)
            if ok:
                sp = self.spent()
                self.d['rho_spent'] = str(sp['rho'] + Fraction(cost['rho']))
                self.d['eps_spent'] = str(sp['eps'] + Fraction(cost.get('eps_prime', 0.0)))
                self.d['gamma_spent'] = str(sp['gamma'] + Fraction(cost['gamma']))
                self.d['runs'].append(dict(researcher=researcher, project=project, time=time.time(),
                                           rho=cost['rho'], eps_prime=cost.get('eps_prime', 0.0),
                                           gamma=cost['gamma'], **(meta or {})))
                self._save()
        return ok, why

    def log_oprf_run(self, researcher, project):
        with self._locked():
            self._load()
            key = f"{researcher}|{project}"
            self.d['oprf_runs'][key] = self.d['oprf_runs'].get(key, 0) + 1
            self._save()
