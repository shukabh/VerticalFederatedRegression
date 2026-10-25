# Baselines: vertical full-record SSP, FWS intervals, TukeyEM

Three baselines added to the evaluation of protocol v3.1 (settings I and P = the re-selected P2 of
`reports/v31_P2`). Paper text: `paper/drafts/baselines.tex` (drop-in blocks for `paper/main.tex`). All numbers below are
regenerated from the JSON files in this directory by `experiments/baselines/tables_base.py`, which writes
`key_numbers.md` (every table). Everything is a plaintext simulation of the mechanism; the encrypted protocol was not run.

| baseline | question | needs the joined table? |
|---|---|---|
| 1. Vertical full-record SSP: secure aggregation of the whole Gram matrix (Karr et al., Hall et al., Gascón et al.), noise on every block including R's `A`, replace-one of a whole record; otherwise v3.1 (two releases, same accountant, stage-1 rule, gate, settings I and P) | what does keeping `A` exact buy? | no: runs in the vertical setting with block normalization (idealized with joint normalization, like arm (c)) |
| 2. Ferrando–Wang–Sheldon parametric-bootstrap intervals for DP OLS, on our own stage-2 release and on a single full-budget release | how do our intervals compare, estimand by estimand? | no: post-processing of statistics the protocol releases |
| 3. TukeyEM (Amin et al. 2023): least squares on m disjoint subsets, exponential mechanism on approximate Tukey depth after a PTR test; no bounds | what do clipping and per-party preprocessing cost? | yes: each subset fit uses whole rows (idealized, central) |

## Files

| file | content |
|---|---|
| `sens_full_check.json` | checks of the full-record sensitivity (Proposition `prop:sens_full`): R's row fixed reproduces `dp.delta_rep`/`dp.delta_G` to 1e-12; direct maximization over vectors (d_R, d_O ∈ {1,2,3}) attains and never exceeds it; 32/3 at unit bounds |
| `r1_full_quantiles.json` | exact radii q (upper 99% bound on the (1 − α_L) quantile of λ_max) of the full-record noise matrix (no zero block on A), same Monte Carlo rule as `vfr/solve.py`; 200,000 draws for p > 64 (gas only) |
| `fullrecord_sim.json` | baseline 1 on the 13.5 population (standardized and block-normalized), δ = 1e-9, R² = 0.5, 200 cohorts per n, same cohorts and unit noise as `reports/v31_P2/sim_P2.json` |
| `fullrecord_benchmark.json` | baseline 1 on the 29 datasets (13.7 design, δ = 1e-6): fold values, verdicts, gaps, refusals, noise ratios |
| `fws_sim.json`, `fws_real.json` | baseline 2: coverage and widths per method × estimand, synthetic (13.5 cohorts and releases of setting I) and real data (3droad, protein, bike, elevators) |
| `tukey_benchmark.json` | baseline 3 on the 29 datasets: test MSE for m = ⌊n/(kp)⌋, k ∈ {1.5, 2, 3, 5, 10, 25} (primary k = 2), PTR pass rates, verdicts |
| `validation_reference.json` | reimplementations against the reference code (see below) |
| `latex_rows.json` | table bodies used in the paper draft |
| `key_numbers.md` | generated tables |

Figures: `paper/figures/{fullrecord_sim,fullrecord_real,fws_coverage,tukey}_base.png`.

## Reproduce (4 cores, about 2 hours, most of it the radii for gas and the interval bootstraps)

```bash
python experiments/baselines/sens_full.py check
python experiments/baselines/real_full.py radii
python experiments/baselines/sim_full.py run && python experiments/baselines/sim_full.py summarize
python experiments/baselines/real_full.py benchmark          # WANG_REPO must point at Wang's repository
python experiments/baselines/sim_fws.py run && python experiments/baselines/sim_fws.py summarize
python experiments/baselines/real_fws.py run && python experiments/baselines/real_fws.py summarize
python experiments/baselines/real_tukey.py benchmark
$SCRATCH/venv_ref/bin/python -I experiments/baselines/validate_ref.py   # numpy 1.26, scikit-learn; see the script
python experiments/baselines/figures_base.py && python experiments/baselines/tables_base.py && python experiments/baselines/tables_base.py latex
python experiments/baselines/apply_baselines_draft.py --out /tmp/main_baselines.tex   # splice the draft into a copy
```

Raw per-cell files go to `experiments/_work/baselines/` (git-ignored). The implementation's radius cache
`implementation/vfr/r1_quantiles.json` is read but never written (`bcommon` disables its writer).

## Third-party code (cloned into the scratchpad, read, not vendored)

| code | commit | use |
|---|---|---|
| github.com/ceciliaferrando/PB-DP-CIs (`DPCIs-OLS.py`) | `f0d2bc6e9707501dfc91b231cdd9779bca866e3b` | FWS reimplemented in `experiments/baselines/fws.py`. Same inputs and random numbers: identical bootstrap draws (max relative difference 8.9e-16). Their simulation (N = 1,000, d = 4, ε = 5, 400 trials): coverage 0.958 (theirs) vs 0.950 (ours), median width 0.581 vs 0.585 |
| github.com/google-research/google-research (`dp_regression/tukey.py`) | `ba728e288c51126eac25e316b11dfee94d68c301` | TukeyEM reimplemented in `experiments/baselines/tukeyem.py`. With the reference code's random-number calls replayed: identical models, PTR distances and outputs in 60 of 60 trials (max abs difference 1.3e-13); deterministic pieces identical on 200 random inputs |

Their code needs numpy < 2 (FWS assigns size-1 arrays to scalars) and scikit-learn (TukeyEM), so the validation runs in
a separate virtual environment (numpy 1.26.4, scipy 1.11.4, scikit-learn 1.4.2) with `python -I`.

Deliberate deviations of our TukeyEM from the reference code (each reproducible with a flag; the validation replays the
reference behaviour exactly):
1. **PTR noise scale.** The reference draws `np.random.laplace(scale=epsilon/2)`; the test against the threshold
   ln(1/(2δ))/(ε/2) needs scale 1/(ε/2). We use 1/(ε/2).
2. **Depth offset.** The reference samples from `projections[:, t:-t]` with a depth that is relative to
   `projections[:, t-1:-(t-1)]`, i.e. one level deeper than the level the exponential mechanism chose (and from an
   empty region at the deepest level). We sample from the chosen level.
3. **Empty comparison set in the PTR check** (m a multiple of 4, k = t − 2): the reference raises on `np.min` of an
   empty array; we treat the check as passed (nothing can be violated).

On the reference demo (n = 22,000, d = 10, δ = 1e-5) deviations 1–2 leave the PTR pass rates unchanged and change the
median error of the released model by 0.1–5%.

FWS on our statistics: FWS release s² through a separate mechanism; we use the plug-in (ŷy₂ − ĉ₂ᵀβ̂)/(n − p) from the
released y'y, floored at 0.1 as FWS floor theirs, and FWS's PSD projection of Q̂ = Ĝ/n (eigenvalues ≤ 0 → 0.1). The
bootstrap uses our noise law (σ₂ on the O-dependent entries, zero on A). Level 0.955 = 1 − α_CI, as our intervals.

## Key numbers

### Baseline 1: vertical full-record SSP (what keeping A exact buys)

**Sensitivity** (Proposition `prop:sens_full`, `experiments/baselines/sens_full.py`). The release adds triu(A) without
the constant n; a whole record (x_R, x_O, y) is replaced. Bound by triu ≤ Frobenius per block, attained in two
coordinates; the bracket is nondecreasing in the block norms (for blocks of dimension ≥ 2), so only (y, y') and the
norms of 1-dimensional blocks are searched.

| bounds | Δ v3.1 → full | Δ without y'y | Δ_G (stage 1) |
|---|---|---|---|
| unit, no intercept (arm (c), d_R, d_O ≥ 2) | √10 = 3.162 → √(32/3) = 3.266 (+3.3%) | same | 2.121 → √6 = 2.449 (+15.5%) |
| simulation population, standardized (5.59, 6.24, 2.89) | 86.2 → 92.7 (+7.5%) | same (y'y free) | 77.2 → 85.8 (+11.2%) |
| block normalization (√2, 1, 2.89) | 10.17 → 10.26 (+0.9%) | **10.00 → 10.00 (0%)** | 2.83 → 3.27 (+15.5%) |

Exact radius at p = 11: q₀.₉₉₇₅ = 7.42 → 7.87 (setting I), q₀.₂₅ = 4.44 → 4.87 (setting P). On the 29 datasets (fold 0,
ε = 1): σ₁ +12–16% and σ₂ +0–7% under block normalization (y'y separate, so σ₂ is set by Δ without y'y); σ₁ +0–16%,
σ₂ +0–3% under joint normalization.

**Synthetic population** (δ = 1e-9, R² = 0.5, 13.5 cohorts and unit noise; v3.1 reproduces `sim_P2` exactly):

| matched records, v3.1 / full-record | ε = 1, I | ε = 1, P | ε = 8, I | ε = 8, P |
|---|---|---|---|---|
| n_off (standardized) | 27.0k / 30.2k | 16.2k / 18.4k | 3.9k / 4.3k | 2.3k / 2.6k |
| test MSE within 10% of non-private (standardized) | 20.1k / 22.1k | 13.7k / 15.8k | 2.9k / 3.4k | 2.1k / 2.2k |
| median slope RMSE ≤ 2× OLS floor (standardized) | >100k / >100k | >100k / >100k | 8.0k / 9.4k | 8.0k / 9.4k |
| test MSE within 10% (block-normalized) | 7.5k / 7.9k | 4.8k / 5.1k | 1.1k / 1.2k | 729 / 767 |

Standardized: n_off +12% (I) / +13% (P); prediction within 10% needs 1.07–1.16× the records; slopes at 2× the floor
1.17–1.18× (ε = 4, 8; at ε = 2 v3.1 reaches it at 88k, full-record not by 100k). Block-normalized: 1.03–1.07×.

**29 datasets** (δ = 1e-6, 10 folds, 50 draws; v3.1 recomputed on the published streams to 9e-15). Paired verdicts
(full-record significantly lower / higher absolute test MSE) and median gap:

| full-record vs v3.1, same preprocessing and setting | ε = 0.5 | ε = 1 | ε = 2 | ε = 4 | ε = 8 |
|---|---|---|---|---|---|
| (d) block norm, I | 1 / 21 (+0.4%) | 1 / 23 (+0.7%) | 2 / 23 (+1.0%) | 1 / 24 (+0.8%) | 0 / 24 (+0.6%) |
| (d) block norm, P | 2 / 19 (+0.6%) | 1 / 20 (+0.9%) | 2 / 22 (+1.2%) | 0 / 23 (+0.8%) | 0 / 23 (+0.7%) |
| (c) joint norm, I (idealized) | 7 / 21 (+1.0%) | 5 / 21 (+1.6%) | 3 / 22 (+1.7%) | 2 / 23 (+1.1%) | 0 / 24 (+1.1%) |
| (c) joint norm, P (idealized) | 7 / 20 (+1.3%) | 5 / 21 (+1.5%) | 5 / 22 (+1.7%) | 3 / 21 (+1.0%) | 1 / 23 (+1.0%) |

Largest gaps at ε = 1 (block norm): energy +5.1% (I) / +7.3% (P), sml +4.1% / +6.3%. The full-record ridge is 8–10%
(block) and 14–17% (joint) larger (median). Where full-record is lower (joint norm, ε ≤ 1) the datasets have little
linear signal (fertility, forest, pendulum, pumadyn32nm, solar, tamielectric; plus concreteslump at ε = 0.5). Gate
refusals of full-record (d) P: 0.14% on average at ε = 1, at most 2.4% (solar). Against AdaSSP (published), full-record
(d) I: 5 / 20 at ε = 1, median gap +25.9% (v3.1 (d) I: +25.1%).

**Verdict:** keeping A exact buys about 1% of test MSE on real data (consistently: significant on 19–24 of 29), and
7–18% of the records needed on standardized synthetic data, where R's bound is large; under block normalization σ₂ does
not change, so only the noise on A and part of Δ_G are saved.

### Baseline 2: FWS intervals against v3.1's intervals (setting I, δ = 1e-9)

Synthetic population (13.5 cohorts and releases; our half-widths reproduce the published ones to 1e-14), coverage of the
ten slopes (target 0.955) and median half-width:

| ε, n | ridge | Wald for θ_λ | bias-aware for β | FWS on stage 2, for β | FWS one release, for β |
|---|---|---|---|---|---|
| 1, 1,000 | on | 0.955 (0.057) | unbounded | 1.000 (10.5) | 1.000 (10.6) |
| 1, 7,500 | on | 0.959 (0.056) | unbounded | 0.972 (0.40) | 0.965 (0.28) |
| 1, 10,000 | on | 0.957 (0.056) | unbounded | 0.960 (0.21) | 0.951 (0.17) |
| 1, 25,000 | on | 0.960 (0.057) | 1.000 (0.30) | 0.957 (0.070) | 0.949 (0.061) |
| 1, 100,000 | off | 0.958 (0.017) | 0.959 (0.018) | 0.958 (0.018) | 0.958 (0.015) |
| 8, 500 | on | 0.963 (0.057) | unbounded | 1.000 (8.8) | 1.000 (8.1) |
| 8, 1,000 | on | 0.953 (0.056) | unbounded | 0.980 (0.58) | 0.972 (0.34) |
| 8, 2,500 | on | 0.954 (0.057) | 1.000 (98% finite; 1.21) | 0.951 (0.11) | 0.948 (0.093) |
| 8, 100,000 | off | 0.955 (0.0024) | 0.958 (0.0054) | 0.956 (0.0053) | 0.949 (0.0052) |

- Wald covers θ_λ 0.946–0.967 in all 70 cells. FWS covers θ_λ only 0.57–0.86 where it is informative (not its target).
- Ridge off (27 cells): FWS on our release = our bias-aware interval (median half-widths within 0.7%; coverage of the
  clipped-population β 0.943–0.958 vs 0.946–0.962). FWS with one release is 3–13% narrower (no stage-1 spend).
- Ridge on (41 cells): bias-aware unbounded while L̂ = 0 (all releases at n ≤ 10k for ε = 1, n ≤ 1k for ε = 8). FWS
  finite and covering (0.951–1.000) but informative only from ϑ₀ = λ_min(G)/(2σ₂√p) ≈ 1 (≈ 6.4k at ε = 1, 0.9k at
  ε = 8, about a quarter of n_off); below it its half-width is 9.5–11.8 (Q̂ projected to PSD in 93–100% of releases).
- Real data (block norm; 200 cohorts per n from the rows outside the reference set; reference = their least-squares
  fit): Wald covers θ_λ 0.938–0.978 everywhere. 3droad (ridge off in 11 of 15 cells): FWS = bias-aware again (0.088 vs
  0.089 at n = 5k, ε = 1), coverage 0.945–0.970. Protein and bike (ridge on everywhere, ϑ₀ ≪ 1): bias-aware unbounded,
  FWS finite but uninformative (half-width 2.1–29) and 0.929/0.940 in two cells. Elevators is run but not reported in
  the paper: its block-normalized design is nearly singular (a population coefficient of 5.4e4).

**Estimands.** θ_λ (our Wald) has no sampling error; β (FWS and our bias-aware) does. FWS has no ridge, so it pays
variance instead of a bound on ‖β‖: it is the better interval for β between ϑ₀ ≈ 1 and n_off, and equal with the ridge off.

### Baseline 3: TukeyEM (ε, δ = 1e-6, same folds; m = ⌊n/(2p)⌋)

- PTR threshold ln(1/(2δ))/(ε/2) = 26.2 at ε = 1, 3.3 at ε = 8; distance ≤ t − 1, t = ⌊⌊m/2⌋/2⌋ → needs m ≳ 110 (ε = 1)
  and m ≳ 20 (ε = 8). 21 of 29 datasets have m < 110, 12 have m < 20.
- **Where it fails:** at ε ≤ 2 the PTR test fails in every run on 25 of 29 datasets and passes in every run on kin40k,
  protein, tamielectric, 3droad (n/p ≥ 3,556). On the 25 the distance bound is −1 even where m allows more (pol,
  elevators, bike: m = 222–386), because fits on 2p rows disagree. At ε = 8 it also passes on pol, elevators, bike,
  airfoil (98%); it never passes on the other 21 (fertility, concreteslump, autos, servo, breastcancer, machine,
  yacht, autompg, housing, forest, stock, pendulum, energy, concrete, solar, wine, gas, skillcraft, sml, parkinsons,
  pumadyn32nm).
- Where it passes: within 0.4% of its own non-private fit on kin40k, tamielectric, 3droad; protein +12% (ε = 1) / +8%
  (ε = 8); at ε = 8 bike +1.2%, airfoil +6%, elevators +15%; pol passes at ε = 8 but predicts worse than the mean
  (0.983 vs 0.345).
- Against v3.1 (d) under P: higher on 25, lower on 2 at ε = 1 (+26%); higher on 24, lower on 3 at ε = 8 (+31%). Under
  I: 25 / 1 (+18%) and 24 / 3 (+30%). Lower on 3droad (unclipped z-scored fit 0.02649 vs block 0.02689), pumadyn32nm
  (no signal; predicting the mean wins) and, at ε = 8, elevators (block normalization's own fit 2.6× worse). The
  best-k oracle changes the counts against v3.1 by at most one dataset.

### Every arm against v3.1 (d) under I and P

See the last section of `key_numbers.md` (and `verdicts_vs_v31d.json`): both bases (absolute test MSE, ratio to own
fit), all ε. At ε = 1 / 8, absolute basis:

| arm | vs v3.1 (d) I, ε = 1 | ε = 8 | vs v3.1 (d) P, ε = 1 | ε = 8 |
|---|---|---|---|---|
| full-record (d), I | 1 / 23 (+0.7%) | 0 / 24 (+0.6%) | 1 / 21 (+5.5%) | 0 / 23 (+4.9%) |
| full-record (d), P | 22 / 1 (−3.8%) | 23 / 0 (−3.7%) | 1 / 20 (+0.9%) | 0 / 23 (+0.7%) |
| full-record (c), I (idealized) | 8 / 5 (−0.8%) | 6 / 7 (+0.1%) | 2 / 18 (+3.2%) | 4 / 19 (+3.7%) |
| full-record (c), P (idealized) | 19 / 4 (−6.6%) | 13 / 3 (−1.8%) | 6 / 12 (+0.7%) | 6 / 10 (+0.6%) |
| TukeyEM (idealized) | 1 / 25 (+17.9%) | 3 / 24 (+30.1%) | 2 / 25 (+26.2%) | 3 / 24 (+31.3%) |

FWS has no test-MSE arm (it is an interval method on our own release).

## Open items (also \tocheck items in the draft)

- TukeyEM's adjacency and PTR noise scale are taken from the reference code and the PTR threshold; the paper of Amin et
  al. was not reachable from this environment (arxiv.org and openreview.net do not resolve). Confirm both, and the
  choice of m.
- The full-record sensitivity is computed numerically after the monotonicity reduction; a closed form is not derived.
- An FWS-type interval for β from the unregularized stage-2 solve, gated by λ_min(Ĝ₂) ≥ 2σ₂√p, would give finite valid
  intervals where the bias-aware interval is unbounded; not part of the protocol, not proved for our noise structure.
- FWS's s² is the plug-in from the released y'y (biased at order σ₂²); whether our bootstrap correction changes FWS's
  coverage in the ridge-on cells is not checked.

