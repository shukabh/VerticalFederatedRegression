# psi_common: DUMMY_ROOT is fixed from PLAIN_MODULUS, but the protocol runs mod the
# calibrated t. Then id_to_field(x, t) can equal DUMMY_ROOT mod t, and every bin padded
# with dummy roots reports that ID as a member (P(y) = 0) with a garbage label.
import sys, numpy as np
sys.path.insert(0, "/home/user/VerticalFederatedRegression/scenario_b")
import psi_common as pc
from calibrate_hyperparameters import psi_plaintext_modulus
t_base = psi_plaintext_modulus(500, 4000, 16384, 3, 1e-6)
print(f"calibrated t for n_R=500, n_O=4000: {t_base} (PLAIN_MODULUS={pc.PLAIN_MODULUS}); "
      f"DUMMY_ROOT mod t = {pc.DUMMY_ROOT % t_base}, t-1 = {t_base-1}")
print(f"  -> P(false match per R id) = 1/(t-1) = {1/(t_base-1):.2e}; for n_R=500: {500/(t_base-1):.2e} "
      f"(the calibration budgets delta_psi = 1e-6 for everything)")
# demonstrate with a small prime so we can find a colliding ID quickly
p = 1_000_003
d0 = pc.DUMMY_ROOT % p
x = next(i for i in range(10**12, 10**12 + 50 * p) if pc.id_to_field(i, p) == d0)
ids_O = np.arange(2 * 10**12, 2 * 10**12 + 200)            # O's IDs; x is NOT among them
P, L, D, alpha, _ = pc.bin_and_interpolate(ids_O, p, 1024, 3, 64)
slot = pc.bin_hash(0, x, 1024)
Pv, Lv = pc.eval_layers_plaintext(P, L, D, alpha, slot, pc.id_to_field(x, p), p)[0]
print(f"small-prime demo p={p}: ID {x} not in O, bin {slot} load={sum(1 for j in ids_O for i in range(3) if pc.bin_hash(i,j,1024)==slot)}, "
      f"D={D}: P(y)={Pv} (0 => reported as a match), label L(y)={Lv}")
