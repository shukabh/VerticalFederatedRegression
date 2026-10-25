"""Validate the reimplementations of FWS and TukeyEM against the reference code (not vendored; cloned into the
scratchpad).  The reference code needs numpy < 2 (FWS assigns size-1 arrays to scalars) and scikit-learn (TukeyEM), so
this script runs in a separate virtual environment, isolated:

    python -m venv $SCRATCH/venv_ref && $SCRATCH/venv_ref/bin/pip install numpy==1.26.4 scipy==1.11.4 scikit-learn==1.4.2
    $SCRATCH/venv_ref/bin/python -I experiments/baselines/validate_ref.py

Writes reports/baselines/validation_reference.json.
  FWS:     their hybrid_bootstrap_OLS and our fws_core on the same inputs and the same random numbers (identical
           draws required), and the coverage of their full pipeline against ours on their own simulation;
  TukeyEM: their dp_regression/tukey.py and our tukeyem.py driven by the same random-number calls (legacy numpy
           RNG through an adapter), with the reference Laplace scale: identical models, PTR distances, depths and
           outputs required; the deterministic pieces on random inputs; and the pass rates under the corrected scale.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402

import fws  # noqa: E402
import tukeyem as te  # noqa: E402

GR = os.environ.get('GR_REF_DIR', '/tmp/claude-0/-home-user-VerticalFederatedRegression/efa2e795-381b-5bb8-8206-13c9bd0ddd5e/scratchpad/ext/gr')   # local sparse clone (not vendored)
GR_COMMIT = 'ba728e288c51126eac25e316b11dfee94d68c301'


def load_tukey_ref():
    import sklearn.linear_model  # noqa: F401  (their module uses sklearn.linear_model after `import sklearn`)
    spec = importlib.util.spec_from_file_location('tukey_ref', os.path.join(GR, 'dp_regression', 'tukey.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class LegacyRNG:
    """Adapter: our Generator-style calls replayed on numpy's legacy global RNG in the reference code's order."""

    def random(self, shape):
        return np.random.rand(*shape)

    def uniform(self, low=0.0, high=1.0, size=None):
        return np.random.uniform(low, high, size)

    def laplace(self, scale):
        return np.random.laplace(scale=scale)

    def permutation(self, n):
        order = np.arange(n)
        np.random.shuffle(order)
        return order


def tukey_checks(trials=60):
    ref = load_tukey_ref()
    out = dict(deterministic=[], replay=[], pass_rates=[])
    rng = np.random.default_rng(0)
    # deterministic pieces on random sorted projections
    worst_lv, mism_dist = 0.0, 0
    for _ in range(200):
        p, m = int(rng.integers(2, 12)), int(rng.integers(20, 400))
        proj = np.sort(rng.standard_normal((p, m)) * rng.uniform(0.1, 3, (p, 1)), axis=1)
        a, b = ref.log_measure_geq_all_depths(proj), te.log_measure_geq_all_depths(proj)
        worst_lv = max(worst_lv, float(np.max(np.abs(a - b))))
        t = int(int(m / 2) / 2)
        if t >= 2:
            for eps, delta in ((0.5, 1e-6), (2.0, 1e-6), (4.0, 1e-5)):
                mism_dist += int(ref.distance_to_unsafety(a, eps, delta, t, -1, t - 1) !=
                                 te.distance_to_unsafety(b, eps, delta, t, -1, t - 1))
    out['deterministic'] = dict(max_abs_diff_log_volumes=worst_lv, distance_mismatches=mism_dist)
    # full replay with the reference Laplace scale
    import io
    import contextlib
    mism, maxdiff, npass = 0, 0.0, 0
    for tr in range(trials):
        n, d = int(rng.integers(10000, 30000)), int(rng.integers(2, 8))
        X = np.column_stack([rng.standard_normal((n, d)), np.ones(n)])
        y = X @ rng.standard_normal(d + 1) + rng.standard_normal(n)
        m = int(rng.choice([200, 500, 1000]))
        eps = float(rng.choice([1.0, 2.0, 8.0]))
        np.random.seed(100 + tr)
        models_ref = ref.multiple_regressions(X, y, m)
        with contextlib.redirect_stdout(io.StringIO()):
            th_ref = ref.dp_tukey(models_ref, eps, 1e-6)
        np.random.seed(100 + tr)
        models = te.multiple_regressions(X, y, m, LegacyRNG())
        th, info = te.dp_tukey(models, eps, 1e-6, LegacyRNG(), laplace_scale='reference', depth_mode='reference')
        maxdiff = max(maxdiff, float(np.max(np.abs(models - models_ref))), float(np.max(np.abs(th - th_ref))))
        mism += int(not np.allclose(th, th_ref, rtol=1e-9, atol=1e-12))
        npass += int(info['passed'])
    out['replay'] = dict(trials=trials, mismatches=mism, max_abs_diff=maxdiff, passed=npass)
    # pass rates, reference vs corrected Laplace scale, on one synthetic problem (n = 22,000, d = 10, as their demo)
    n, d = 22000, 10
    X = np.column_stack([rng.standard_normal((n, d)), np.ones(n)])
    y = X @ rng.standard_normal(d + 1) * 10 + 10 * rng.standard_normal(n)
    for m in (250, 500, 1000, 2000):
        for eps in (0.5, 1.0, np.log(3)):
            r, err = {}, {}
            for sc, dm in (('reference', 'reference'), ('inverse', 'chosen')):
                g = np.random.default_rng(5)
                runs = [te.tukey_em(X, y, eps, 1e-5, m, g, sc, dm) for _ in range(40)]
                r[sc] = float(np.mean([o[1]['passed'] for o in runs]))
                bols = np.linalg.lstsq(X, y, rcond=None)[0]
                ok = [o[0] for o in runs if o[1]['passed']]
                err[sc] = float(np.median([np.linalg.norm(t - bols) for t in ok])) if ok else None
            out['pass_rates'].append(dict(m=m, eps=float(eps), delta=1e-5, reference_code=r['reference'],
                                          corrected=r['inverse'], median_error_reference_code=err['reference'],
                                          median_error_corrected=err['inverse']))
    return out


def main():
    res = dict(fws=dict(commit=fws.FWS_COMMIT, repo='github.com/ceciliaferrando/PB-DP-CIs',
                        replay_max_rel_diff=fws.check_against_reference(),
                        coverage=fws.check_coverage_like_reference()),
               tukeyem=dict(commit=GR_COMMIT, repo='github.com/google-research/google-research (dp_regression)',
                            **tukey_checks()),
               numpy=np.__version__)
    path = os.path.join(ROOT, 'reports', 'baselines', 'validation_reference.json')
    with open(path, 'w') as fh:
        json.dump(res, fh, indent=1, default=float)
    print(json.dumps(res, indent=1, default=float))


if __name__ == '__main__':
    main()
