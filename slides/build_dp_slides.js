// Build a three-slide explainer of the DP accounting: the sensitivity Delta_s of each release, how
// sigma_s and rho_s follow from it and the budget, and where the noise lives in the CKKS ciphertext.
// Requires: npm install pptxgenjs jszip react-icons react react-dom sharp
// Run from the repo root:  node slides/build_dp_slides.js
// Display equations are LaTeX images: after editing slides/equations.json, run  python slides/make_equations.py  first.
const fa = require("react-icons/fa");
const { P, C, HEAD, newDeck, writeDeck } = require("./deck_lib");
const { INK, SLATE, MIST, EMBER, EMBER_L, MUTED, WHITE } = C;

(async () => {
  const { pres, h } = await newDeck("The DP accounting: Δ_s, σ_s and ρ_s", {
    shield: fa.FaUserShield, ruler: fa.FaRulerHorizontal, scale: fa.FaBalanceScale,
    sliders: fa.FaSlidersH, zoom: fa.FaSearchPlus, water: fa.FaWater,
  });
  const { title, circleIcon, numCircle, card, txt, bullets, latex, tableOpts, hdr } = h;
  const head = (s, icon, t, x, y, col) => {
    circleIcon(s, icon, x + 0.25, y + 0.2, 0.5, col);
    txt(s, t, { x: x + 0.9, y: y + 0.22, w: 4.9, h: 0.45, fontFace: HEAD, fontSize: 17, bold: true, color: INK });
  };
  const strip = (s, t) => {
    card(s, 0.6, 6.55, 12.1, 0.65, "F7F9FC");
    txt(s, t, { x: 0.85, y: 6.57, w: 11.6, h: 0.61, fontSize: 12.5, valign: "middle" });
  };

  // ---------- 1. sensitivity ----------
  {
    const s = pres.addSlide();
    title(s, "Δ_s: how far one person can move a release", "DP protects the people in O's register from R, the only party that sees decrypted output");

    card(s, 0.6, 1.75, 5.6, 4.65, MIST);
    head(s, "shield", "Who is protected", 0.6, 1.75, INK);
    txt(s, "Neighbours D ∼ D′: one matched person's (x_O, y) is replaced by any other value within the bounds. Their R row ẋ and the match set I stay fixed.",
      { x: 0.85, y: 2.6, w: 5.1, h: 0.8, fontSize: 13 });
    bullets(s, ["R already knows ẋ and I (the linkage reveals I by design: outside DP).",
      "O sees only ciphertexts (encryption, not DP)."],
      { x: 0.85, y: 3.45, w: 5.1, h: 0.85, fontSize: 12.5, paraSpaceAfter: 3 });
    txt(s, "What release s contains", { x: 0.85, y: 4.35, w: 5.1, h: 0.3, fontSize: 13.5, bold: true, color: INK });
    s.addTable([hdr(["s", "Stage", "Entries released"]),
      ["1", "stage 1", "triu B, C"],
      ["2", "stage 2", "triu B, C, c_R, c_O, yᵀy"]],
    Object.assign(tableOpts([0.5, 1.0, 3.6]), { x: 0.85, y: 4.7, fontSize: 12.5, rowH: 0.32, fill: { color: WHITE } }));
    txt(s, "A = ẊᵀẊ is never released: R computes it itself, and it is the same in both neighbours.",
      { x: 0.85, y: 5.8, w: 5.1, h: 0.5, fontSize: 12, color: MUTED });

    card(s, 6.4, 1.75, 6.3, 4.65, MIST);
    head(s, "ruler", "The sensitivity Δ_s", 6.4, 1.75, SLATE);
    latex(s, "d_sens", 6.65, 2.62, { size: 15 });
    txt(s, "With ẋ fixed and x = x_O, a replacement (x, y) → (x′, y′) moves:",
      { x: 6.65, y: 3.12, w: 5.85, h: 0.3, fontSize: 13 });
    latex(s, "d_moves", 6.65, 3.55, { size: 11.5 });
    txt(s, "Every entry gets the same noise, so the squared changes add up over the blocks of a release. Stage 1 has only the first two blocks, so Δ_G < Δ. Only the upper triangle of the symmetric B is released, so each off-diagonal change counts once.",
      { x: 8.9, y: 3.55, w: 3.6, h: 1.3, fontSize: 12, color: MUTED });
    txt(s, "Maximizing over the bounds, with (r, o, w) = (B_R², B_O², B_y²):", { x: 6.65, y: 4.98, w: 5.85, h: 0.3, fontSize: 13 });
    latex(s, "d_closed", 6.65, 5.33, { size: 11.5 });
    txt(s, "Toy: B_R² = 3, B_O² = 2, B_y² = 1 → Δ = √38 = 6.16, Δ_G = √24.5 = 4.95.",
      { x: 6.65, y: 5.95, w: 5.85, h: 0.3, fontSize: 12, color: MUTED });

    strip(s, "Clipping is what makes Δ finite, and Δ does not depend on n: more records make the signal bigger, not the noise.");
    s.addNotes("Differential privacy here protects the people in O's register, and the adversary is R, because R is the only party that ever sees a decrypted output. O only sees ciphertexts, so encryption protects O's view. Two datasets are neighbours if one matched person's O-side features and outcome are replaced by any other values inside the clipping bounds, with that person's R row and the match set fixed. R already knows both; the match set is revealed by the linkage by design and is outside the DP guarantee. The protocol makes two releases: stage 1 with the upper triangle of B and C, and stage 2 with everything. A is never released because R computes it. The sensitivity Delta_s is the largest change one person can make to the whole vector of release s, in l2 norm. Replacing x and y moves each block as shown; since every entry gets noise of the same size, the squared changes add up. Maximizing over the bounds gives the closed forms. In the toy example Delta is root 38 and Delta_G is root 24.5; both were checked by brute-force maximization. Delta depends only on the bounds, not on n.");
  }

  // ---------- 2. sigma and rho ----------
  {
    const s = pres.addSlide();
    title(s, "σ_s and ρ_s: one equation ties them to Δ_s", "Δ_s is fixed by the bounds and ρ_s by the budget, so the noise σ_s follows");

    card(s, 0.6, 1.75, 12.1, 1.0, EMBER_L);
    latex(s, "d_key", 0.95, 1.885, { size: 20 });
    txt(s, [
      { text: "Δ_s", options: { bold: true, color: EMBER } }, { text: "   sensitivity: how far one person can move release s", options: { breakLine: true } },
      { text: "ρ_s", options: { bold: true, color: EMBER } }, { text: "   zCDP cost: the share of the budget release s spends", options: { breakLine: true } },
      { text: "σ_s", options: { bold: true, color: EMBER } }, { text: "   noise s.d. added to every entry of release s" }],
    { x: 6.4, y: 1.8, w: 6.1, h: 0.9, fontSize: 13.5, valign: "middle" });

    card(s, 0.6, 2.95, 5.9, 3.45, MIST);
    head(s, "scale", "Why the equation holds", 0.6, 2.95, INK);
    txt(s, "R sees ẑ_s = z_s + N(0, σ_s² I): independent noise with the same s.d. on every entry. For neighbours D ∼ D′ and any order a > 1, the Rényi divergence is",
      { x: 0.85, y: 3.8, w: 5.45, h: 0.75, fontSize: 13 });
    latex(s, "d_renyi", 0.85, 4.65, { size: 10.5, maxW: 5.4 });
    txt(s, "ρ-zCDP means this is at most aρ for every a > 1, so release s costs exactly ρ_s = Δ_s²/(2σ_s²). Costs add: ρ_run = ρ₁ + ρ₂. They add across runs too, and O's ledger caps the total.",
      { x: 0.85, y: 5.2, w: 5.45, h: 1.0, fontSize: 12.5 });

    card(s, 6.7, 2.95, 6.0, 3.45, MIST);
    head(s, "sliders", "From the run's (ε, δ) to the noise", 6.7, 2.95, SLATE);
    const step = (n, label, y) => {
      numCircle(s, n, 6.95, y, 0.36, SLATE, 12);
      txt(s, label, { x: 7.45, y: y + 0.02, w: 5.0, h: 0.3, fontSize: 13, bold: true, color: INK });
    };
    step(1, "Convert (ε, δ) to a zCDP budget", 3.72);
    latex(s, "zcdp_conv", 7.45, 4.07, { size: 11.5 });
    txt(s, "Tight conversion: ρ_run = 0.01497 at ε = 1, δ = 10⁻⁹.",
      { x: 7.45, y: 4.35, w: 5.05, h: 0.25, fontSize: 11, color: MUTED });
    step(2, "Split it between the two stages", 4.72);
    latex(s, "d_split", 7.45, 5.07, { size: 11.5 });
    step(3, "Set the noise", 5.38);
    latex(s, "c_sigma", 7.45, 5.74, { size: 11.5 });
    txt(s, "σ₁ ≈ 2.4σ₂ on purpose: stage 1 only sets the ridge, and its 10% share raises σ₂ by just 5%.",
      { x: 9.45, y: 5.7, w: 3.05, h: 0.6, fontSize: 11, color: MUTED });

    strip(s, "Toy example (ε = 1, δ = 10⁻⁹): Δ_G = 4.95, ρ₁ = 0.0015, σ₁ = 90.45; Δ = 6.16, ρ₂ = 0.0135, σ₂ = 37.55. Everything after the releases (λ, the gate, θ̃, V_mech, the intervals) uses only ẑ₁, ẑ₂, A and public numbers: post-processing, at no extra privacy cost.");
    s.addNotes("The three quantities are tied by one equation: rho_s equals Delta_s squared over two sigma_s squared. Delta_s comes from the clipping bounds and rho_s from the privacy budget, so sigma_s follows. Why: R sees each released entry plus independent Gaussian noise of the same standard deviation. The Renyi divergence of order a between the two Gaussians for neighbouring datasets is a times the squared change over two sigma squared, which is at most a times Delta squared over two sigma squared. That is exactly the definition of rho-zCDP with rho equal to Delta squared over two sigma squared. zCDP costs add, across the two stages and across runs; O keeps a ledger and refuses to decrypt past the cap. To set the noise: convert the run's epsilon and delta to a zCDP budget rho_run; the protocol uses the tight conversion of Canonne, Kamath and Steinke, which allows about 27 percent more rho than the simple bound. Split it, ten percent to stage 1 and ninety to stage 2, and set each sigma from its own sensitivity and share. Stage 1 is deliberately noisier: it only has to give a rough lower bound to set the ridge, and its share costs stage 2 only five percent in noise. Everything computed afterwards is post-processing and costs nothing more.");
  }

  // ---------- 3. where the noise lives ----------
  {
    const s = pres.addSlide();
    title(s, "Where the noise actually lives", "O adds integer noise η to the ciphertext's coefficients; R decodes it as Gaussian noise on each entry");

    const box = (n, t, x, w, col) => {
      card(s, x, 1.75, w, 2.0, MIST);
      numCircle(s, n, x + 0.22, 1.92, 0.45, col, 14);
      txt(s, t, { x: x + 0.82, y: 1.95, w: w - 1.0, h: 0.4, fontFace: HEAD, fontSize: 15, bold: true, color: INK });
    };
    const arrow = (x, top, bottom) => {
      s.addShape(pres.shapes.LINE, { x, y: 2.75, w: 0.6, h: 0, line: { color: MUTED, width: 2, endArrowType: "triangle" } });
      txt(s, top, { x: x - 0.15, y: 2.38, w: 0.9, h: 0.3, fontSize: 11, color: MUTED, align: "center" });
      txt(s, bottom, { x: x - 0.15, y: 2.83, w: 0.9, h: 0.3, fontSize: 11, color: MUTED, align: "center" });
    };
    box(1, "Statistic in the slots", 0.6, 3.5, INK);
    txt(s, "Release s is a vector z_s with one entry per slot. One person moves it by at most Δ_s in ℓ2.",
      { x: 0.85, y: 2.5, w: 3.05, h: 1.1, fontSize: 12.5 });
    arrow(4.18, "encode", "×Ω, τ⁻¹");
    box(2, "Coefficients: O adds η", 4.85, 3.65, SLATE);
    latex(s, "d_eta", 5.07, 2.45, { size: 12 });
    latex(s, "d_enc", 5.07, 2.8, { size: 12 });
    latex(s, "d_rho_lat", 5.07, 3.15, { size: 12 });
    txt(s, "exact for the discrete Gaussian", { x: 5.07, y: 3.43, w: 3.3, h: 0.25, fontSize: 11, color: MUTED });
    arrow(8.58, "decode", "÷Ω, τ");
    box(3, "What R decodes", 9.25, 3.45, EMBER);
    latex(s, "d_slot", 9.47, 2.5, { size: 14 });
    txt(s, "Noise with s.d. σ_s on every released entry: the picture of the previous slide.",
      { x: 9.47, y: 2.95, w: 3.05, h: 0.7, fontSize: 12.5 });

    card(s, 0.6, 3.95, 5.6, 2.45, MIST);
    head(s, "zoom", "Why the slot picture is enough", 0.6, 3.95, INK);
    latex(s, "d_subst", 0.85, 4.8, { size: 14 });
    txt(s, "With N = 2^14 and Ω ≈ 2^61.5 the extra term is below 10⁻¹⁴. Sampling the discrete Gaussian exactly, with integers, makes the guarantee exact rather than approximate.",
      { x: 0.85, y: 5.45, w: 5.1, h: 0.85, fontSize: 12.5 });

    card(s, 6.4, 3.95, 6.3, 2.45, MIST);
    head(s, "water", "η also hides the CKKS error", 6.4, 3.95, SLATE);
    latex(s, "smudge", 6.65, 4.7, { size: 11, maxW: 5.8 });
    txt(s, "The error e depends on the data and on O's key share, so it must be hidden (κ = 40). Since σ_c,s = σ_s·Ω·√(2/N), O raises Ω until this holds:",
      { x: 6.65, y: 5.13, w: 5.85, h: 0.45, fontSize: 11.5 });
    latex(s, "scale", 6.65, 5.6, { size: 10.5, maxW: 5.8 });
    txt(s, "Price: the DP itself gets δ_DP = δ_run − n_rel(δ_B + 2^−κ).", { x: 6.65, y: 6.08, w: 5.85, h: 0.25, fontSize: 11, color: MUTED });

    strip(s, "How the σ's drive the statistics: σ₁ sets the radius r₁ = σ₁(2√p + 2√ln(1/α_L)) and so λ; σ₂ sets the ridge target 2ϑ*σ₂√p, the gate and V_mech ∝ σ₂². Records needed for a given coefficient accuracy grow like σ₂² ∝ Δ²/ρ.");
    s.addNotes("The slot picture is how to think about it, but O does not add Gaussian noise to the slots. The statistic is encoded: scaled by Omega and mapped through the inverse canonical embedding, which shrinks l2 norms by root 2 over N. So in the coefficient domain the sensitivity is root 2 over N times Omega times Delta_s, plus root N for rounding. O adds an integer vector eta, each coefficient a discrete Gaussian with standard deviation sigma_c,s, and the cost is exactly Delta_enc squared over two sigma_c squared. When R decodes, the noise on each slot has standard deviation sigma_c times root N over 2, divided by Omega. Substituting, sigma_s equals Delta_s over root two rho_s up to a term of order 10 to the minus 14, so the slot picture is accurate; the discrete Gaussian, sampled exactly with integers, makes it rigorous. The same eta also hides the CKKS approximation error, which depends on the data and on O's key share: by the smudging lemma the error is hidden up to 2 to the minus 40 if sigma_c is at least 2 to the 40 times the worst-case error bound over root 2 pi. Since sigma_s is fixed by the budget, O meets this by choosing the scale Omega large enough, about 2 to the 61.5 for the circuit as specified. The cost is a small deduction from delta. Finally, sigma_1 drives the ridge choice and sigma_2 drives the gate and the confidence intervals; the records needed for coefficient accuracy grow like sigma_2 squared, so like Delta squared over rho.");
  }

  await writeDeck(pres, P("slides", "dp_accounting.pptx"));
})();
