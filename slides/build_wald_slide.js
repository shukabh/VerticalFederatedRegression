// Build a one-slide explainer of the Wald interval for the ridge target: the bootstrap that computes
// V_mech, the closed form it estimates, and the interval.
// Requires: npm install pptxgenjs jszip react-icons react react-dom sharp
// Run from the repo root:  node slides/build_wald_slide.js
// Display equations are LaTeX images: after editing slides/equations.json, run  python slides/make_equations.py  first.
const fa = require("react-icons/fa");
const { P, C, HEAD, newDeck, writeDeck } = require("./deck_lib");
const { INK, SLATE, MIST, EMBER, EMBER_L, MUTED } = C;

(async () => {
  const { pres, h } = await newDeck("How the Wald interval is computed", { check: fa.FaCheckCircle });
  const { title, numCircle, card, txt, latex } = h;
  const s = pres.addSlide();
  title(s, "How the Wald interval is computed", "V_mech, the covariance due to the DP noise, by parametric bootstrap at R: free post-processing");

  const head = (n, t, x, col) => {
    numCircle(s, n, x + 0.25, 1.95, 0.55, col, 16);
    txt(s, t, { x: x + 0.95, y: 1.98, w: 3.4, h: 0.5, fontFace: HEAD, fontSize: 17, bold: true, color: INK });
  };

  // 1. bootstrap
  card(s, 0.6, 1.75, 4.4, 4.5, MIST);
  head(1, "Simulate the noise", 0.6, INK);
  txt(s, "For b = 1, …, B, around R's own release:", { x: 0.85, y: 2.6, w: 4.0, h: 0.35, fontSize: 13, color: MUTED });
  latex(s, "w_boot", 0.85, 3.0, { size: 13, maxW: 3.95 });
  latex(s, "w_vmech", 0.85, 4.0, { size: 14, maxW: 3.95 });
  latex(s, "w_trunc", 0.85, 4.8, { size: 13, maxW: 3.95 });
  txt(s, "Without this truncation the bootstrap mean and covariance do not exist. Use antithetic pairs (E*_b, f*_b) and (−E*_b, −f*_b).",
    { x: 0.85, y: 5.25, w: 3.95, h: 0.9, fontSize: 12, color: MUTED });

  // 2. closed form
  card(s, 5.2, 1.75, 4.4, 4.5, MIST);
  head(2, "What it estimates", 5.2, SLATE);
  txt(s, "First order in the noise:", { x: 5.45, y: 2.6, w: 4.0, h: 0.35, fontSize: 13, color: MUTED });
  latex(s, "w_fo", 5.45, 3.0, { size: 12.5, maxW: 3.95 });
  latex(s, "w_v0", 5.45, 3.5, { size: 16, maxW: 3.95 });
  latex(s, "w_K", 5.45, 4.05, { size: 12.5, maxW: 3.95 });
  txt(s, [{ text: "χ_ij = 0 if i and j are both in R's block A (never noised), else 1. ", options: { breakLine: true } },
    { text: "I: noise f on ĉ_2.  K(θ): noise E on Ĝ_2, times θ.", options: { breakLine: true } },
    { text: "R can use this closed form instead of the bootstrap: same first-order guarantee, no Monte Carlo error." }],
    { x: 5.45, y: 4.55, w: 3.95, h: 1.6, fontSize: 12, color: MUTED, paraSpaceAfter: 4 });

  // 3. the interval
  card(s, 9.8, 1.75, 2.9, 4.5, EMBER_L);
  head(3, "The interval", 9.8, EMBER);
  latex(s, "w_ci", 10.05, 2.75, { size: 17, maxW: 2.45 });
  txt(s, "α_CI = α − α_L − α_U = 0.045, so z = 2.005 (nominal 95.5%)", { x: 10.05, y: 3.5, w: 2.45, h: 0.75, fontSize: 12.5 });
  txt(s, "Target", { x: 10.05, y: 4.3, w: 2.45, h: 0.35, fontSize: 13, bold: true, color: EMBER });
  latex(s, "w_target", 10.05, 4.7, { size: 15, maxW: 2.45 });
  txt(s, "a fixed function of the data, so only the noise enters: no sampling variance.", { x: 10.05, y: 5.2, w: 2.45, h: 0.95, fontSize: 12, color: MUTED });

  card(s, 0.6, 6.45, 12.1, 0.75, "F7F9FC");
  txt(s, "First order: relative error O(1/ϑ_0) + O(B^{−1/2}). In simulation V_mech is within 2% of the true noise s.d., and the intervals cover 0.944–0.962 (target 0.955). The interval for β adds sampling variance, V = V_mech + s̃² P̃ Ĝ_2 P̃, and is widened for the ridge bias.",
    { x: 0.85, y: 6.5, w: 11.6, h: 0.65, fontSize: 12, valign: "middle" });

  s.addNotes("How R computes the Wald interval for the ridge target. Step 1: R knows the noise distribution exactly, so it simulates fresh noise around its own stage-2 release: symmetric noise on the B and C entries (none on its exact block A) and noise on c-hat, solves the ridge each time, and takes the sample covariance of the draws. Draws where the simulated noise is too large relative to the ridged matrix are discarded; without this truncation the bootstrap moments do not exist. Step 2: to first order the estimate's error is P (f - E theta), so the bootstrap estimates sigma_2^2 P (I + K(theta)) P, where I comes from the noise on c and K(theta) from the noise on G times the coefficients. R may use this closed form directly. Step 3: the interval is the estimate plus or minus z times the square root of the diagonal of V_mech, with z = 2.005 because 0.005 of the error budget goes to the two auxiliary bounds. It targets theta_lambda, a fixed function of the data, so only the mechanism noise matters. The interval for beta adds the sampling variance and is widened for the ridge bias.");

  await writeDeck(pres, P("slides", "wald_interval.pptx"));
})();
