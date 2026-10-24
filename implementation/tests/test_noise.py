"""With the noise on, the empirical s.d. of (decoded - true) per entry over repeated releases matches the
slot-domain sigma_s = sigma_c sqrt(N/2) / Omega_f from Calibrate.  Each release draws fresh (u, e0, e1) and a
fresh exact discrete Gaussian eta on Z^N."""
import numpy as np
import pytest

from helpers import Pipeline, tau_vectors, true_entries
from vfr import common as C
from vfr import data, dp

R_REL = 40


@pytest.mark.parametrize('variant', ['A'])
def test_release_noise_sd_matches_calibration(variant):
    n_bar = 4096
    dR_, dO_, truth, b = data.synthetic(n_bar, 3500, 500, 300, seed=21)
    ag = dp.Agreement(n_R_max=600, n_bar_O=n_bar, d_R=4, d_O=6, variant=variant, **b)
    perm = np.random.default_rng(0).permutation(n_bar)[:len(dO_['ids'])] + 1
    tau = {i: int(p) for i, p in zip(dO_['ids'], perm)}
    vecs, XO, yt = tau_vectors(dR_, dO_, tau, ag)
    ps = data.plaintext_stats(dR_, dO_, b['B_R'], b['B_O'], b['B_y'])
    tru = true_entries(ps['G'], ps['c'], ps['yy'], ag.d_R, ag.d_O)
    pl = Pipeline(ag, C.CKKSParams(), vecs, XO, yt)
    for stage, reps in ((2, R_REL), (1, R_REL // 2)):
        sig = pl.cal['releases'][str(stage)]['sigma_slot']
        names = [e['name'] for e in C.release_entries(pl.ents, stage)]
        rows = []
        for _ in range(reps):
            vals = pl.release(stage)[0]
            rows.append([vals[k] - tru[k] for k in names])
        D = np.array(rows)
        pooled = np.sqrt((D ** 2).mean())
        assert abs(pooled / sig - 1) < 0.06, (stage, pooled, sig)
        assert abs(D.mean()) < 4 * sig / np.sqrt(D.size)
        per = D.std(axis=0, ddof=0) / sig
        assert per.min() > 0.4 and per.max() < 1.9, (per.min(), per.max())
