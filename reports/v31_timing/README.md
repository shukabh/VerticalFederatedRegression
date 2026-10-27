# Release timing with a fixed work budget (paper check item #6, T8)

## The problem

R observes when each release arrives. The exact CKS20 sampler loops a value-dependent number of times, so the
time O takes depends on the noise eta. Until now O padded each release to a deadline D and refused an overrun,
but the overrun probability was only measured (0/1000 idle, 4/200 under contention), not bounded, and not charged
to delta; and the SanitizeBFV flooding samples of the PSI replies were not covered.

## What changed

**Budgeted exact sampler** (`implementation/vfr/sampler.py`: `budget`, `sample_residues`). For an integer variance
S and n values it runs a fixed number T of candidate trials, every one of them, with the same fixed-width work:

- proposal: sign bit s, u uniform on [0, 2^b), v geometric(e^-1) by inversion of one 128-bit coin against the
  certified table floor(2^128 e^-j); y = u + 2^b v or -(u + 2^b v + 1) (a bijection onto Z);
- acceptance Bernoulli(exp(-g)), g = y^2/(2S) - v + c >= 0: Bernoulli(e^-floor(g)) by one coin against the table,
  Bernoulli(e^-frac(g)) by CKS20 Algorithm 1 with exactly k_max coins; floor(2^128 g) and y mod q are computed
  exactly in 32-bit limbs (numpy uint64), with no hardware division and no branch on a value;
- the first n accepted values are returned; the call fails (and the release is refused / the run aborts) iff some
  trial is undecided (v reaches J_v, all k_max coins are 1, or a coin equals its threshold) or fewer than n trials
  are accepted.

T, J_v, k_max are public functions of (S, n, xi_T) chosen so that P[fail] <= xi_T = 2^-80 (Chernoff-KL bound on
the binomial lower tail for the acceptances, union bound for the rest). Outside the failure event the output is
exactly the unbudgeted exact sampler's output on the same coins, and the operation trace depends only on public
parameters and the number A of accepted trials, which is independent of the values returned.

**Charge.** `Agreement.sampler_xi` (default 2^-80, checked by both parties) is added per release to the statistical
slack: gamma = n_rel (delta_B + 2^-kappa (1 + 2^-80) + 2^-97 + xi_T), and in the hockey-stick ledger
gamma_L = n_rel (delta_B + delta' + xi_T) (`dp.slack_per_release`, `dp.run_cost`, `dp.choose_kappa`,
`dp.calibrate`). Calibrate publishes each release's budget (`releases[s]['sampler']`) and R checks it.

**PSI.** SanitizeBFV draws its flooding vectors with the same sampler (`party_o`), and `PSIParams.check` charges
`2 alpha F x tables x sampler_xi` (2^-75.7 at 1M rows) against delta_psi, as part of G4's second layer (not delta:
the Link phase has the same law under neighbouring datasets). A failure aborts the run (R handles the refusal).

**Deadline kept as defence in depth.** `release_pad` is unchanged (calibrated now with the budgeted sampler). It is
no longer what the theorem rests on: under the cost assumption (W) an overrun depends on machine load, not on eta.

## The guarantee (docs/proofs, Part A, Section 5.5; spec Theorem 1; paper draft `paper/drafts/timing.tex`)

Lemmas 5.15-5.17: the ideal sampler is exact and its acceptances carry no information; coupling; failure bound.
Assumption 5.18 (W): O's time is a function of the operation trace (operations and public shapes) and of machine
noise independent of the secrets, not of operand values. Lemma 5.20: R's view with the release timing is within
statistical distance xi_T of one with exact noise and a timing independent of the noise and of D.
Corollary 5.21: Theorem 1 holds for the view including arrival times with slack gamma + n_rel xi_T; Part C,
Remark 5.9: the same charge in the ledger.

**What is an assumption, honestly.** (W) is not proved for CPython/numpy: numpy loops are assumed free of
value-dependent fast paths; the gather from the 2.9 kB exp table has value-dependent addresses (a cache channel on
O's machine); the allocator; and OpenFHE's samplers for (u, e0, e1) inside the native re-randomization were not
analysed. Evidence: crafted extreme coins give the same call time within noise (`key_numbers.md`).

## Measurements (`timing.json`, `key_numbers.md`; machine shared with other jobs, load 16-23)

- Budget at the 1M runs' largest sigma_c^2 (2^136.7) and N = 2^14: T = 39,088 = 2.39 N, J_v = 69, k_max = 29,
  xi_bound = 2^-80.9; flooding (sigma_psi = 2^61): T = 36,189.
- Cost: 0.244 s per release noise vector against 0.451 s for the plain sampler (ratio 0.54); flooding 0.225 vs 0.490
  s (0.46). The fixed budget costs nothing net: limb arithmetic in numpy is faster than Python big integers.
- No budget failure in 1,000 release-size and 300 flooding-size calls; the smallest surplus of accepted trials was
  682 (6.9 s.d.) and 737; the acceptance rate matches p_lo to 4 digits.
- Deadline D = 0.77 s; 1/300 overruns at load ~22 and 100/100 with four extra CPU hogs: availability, not privacy.

## Tests

`implementation/tests/test_sampler_budget.py` (limb arithmetic and every decision against a big-integer reference
on the same coins, adversarial coins included; certified exp table; exact reciprocal division; pmf and moments;
residues; the budget bound against scipy's exact binomial tail; failure raises) and new cases in
`test_release_timing.py` (budget published and charged; a sampler failure refuses the release; a failure in the
PSI aborts the run; PSIParams.check charges the flooding calls).

Full suite (`cd implementation && python -m pytest -q`): 130 tests; 129 passed in 40.7 min on the loaded machine, the one failure (a test that used sampler_xi = 2^-200, below what 128-bit coins support) was fixed by making Calibrate refuse such a value (`CalibrationRefused`) and the affected files re-run.

Reproduce: `python experiments/eng/release_timing_budget.py` (about 20 min on a loaded 4-core machine).
