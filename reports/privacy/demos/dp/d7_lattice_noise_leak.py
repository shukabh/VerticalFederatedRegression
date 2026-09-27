"""
Report item D8 — the paper's proposed fix (Sec. 7 iv: "discrete/snapped Gaussian to close the CKKS
decryption-error channel") is dangerous if done naively under encryption.

O cannot snap/round the ENCRYPTED statistic s to a grid. If it adds discrete-Gaussian noise
gamma*K (K integer) to an un-snapped s, R decrypts out = s + gamma*K + e_ckks and
    out mod gamma = (s + e_ckks) mod gamma,
i.e. the low-order digits of s are released WITHOUT noise. With R's own features known
exactly, that is a subset-sum constraint on O's data. Here: n=16 matched people with binary
y (e.g. benefit receipt), one R column x_R, statistic c_R = x_R . y, sigma from the code.

Also prints the float granularity numbers relevant to Mironov-type attacks on the current
continuous float64 sampler.

Run: python d7_lattice_noise_leak.py     (~3 s)
"""
import itertools, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "../../../../scenario_b")))
import calibrate_hyperparameters as ch  # noqa: E402

rng = np.random.default_rng(99)
n = 16
sigma = ch.analytic_gaussian_sigma(ch.joint_sensitivity_B(1.0, 1.0, 1.0), 1.0, 1e-5)
gamma = 2.0 ** -8          # grid of the discrete Gaussian
ckks_err = 1e-10           # decryption error SD (order of 2^-33; depends on parameters)
tol = 1e-8

Y = np.array(list(itertools.product([0.0, 1.0], repeat=n)))       # all 65536 candidates


def circ(a):
    r = np.mod(a, gamma)
    return np.minimum(r, gamma - r)


print(f"sigma={sigma:.3f} (eps=1, bounds 1), grid gamma={gamma}, CKKS error SD={ckks_err:g}")
for mech in ("continuous Gaussian (current code)", "discrete Gaussian on grid, s not snapped"):
    hits, sizes = 0, []
    for trial in range(20):
        xR = rng.uniform(0, 1, n)
        y = (rng.uniform(size=n) < 0.3).astype(float)
        s = xR @ y
        if mech.startswith("continuous"):
            out = s + sigma * rng.standard_normal() + ckks_err * rng.standard_normal()
        else:
            out = s + gamma * np.round(sigma / gamma * rng.standard_normal()) + ckks_err * rng.standard_normal()
        cand = Y[(circ(out - Y @ xR) < tol) & (np.abs(out - Y @ xR) < 6 * sigma)]
        sizes.append(len(cand))
        hits += int(len(cand) == 1 and np.array_equal(cand[0], y))
    print(f"  {mech:42s}: exact recovery of all {n} y_i in {hits}/20 trials; "
          f"consistent candidates per trial: median {int(np.median(sizes))}")

print("\nfloat granularity (for Mironov / Jin et al. 2022 style attacks on the float64 sampler):")
print(f"  ulp of a released value ~100: {np.spacing(100.0):.1e}; ~1000: {np.spacing(1000.0):.1e}")
print(f"  CKKS decryption error at scale 2^50, depth 2 is typically 1e-12..1e-8 -> float artefacts are masked")
print(f"  under the openfhe backend but NOT under --backend plaintext (run_protocol.py's default).")
