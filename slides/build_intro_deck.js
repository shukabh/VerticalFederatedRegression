// Build the self-contained introduction deck: motivation, building blocks, the protocol, its
// guarantees, parameter choice and results. Written for a team meeting the material for the first time.
// Requires: npm install pptxgenjs jszip react-icons react react-dom sharp
// Run from the repo root:  node slides/build_intro_deck.js
// Display equations are LaTeX images: after editing slides/equations.json, run  python slides/make_equations.py  first.
const fs = require("fs");
const fa = require("react-icons/fa");
const { P, readCsv, mdTables, C, HEAD, BODY, newDeck, writeDeck } = require("./deck_lib");
const { INK, SLATE, MIST, LINE, EMBER, EMBER_L, TEAL, AMBER, TEXT, MUTED, WHITE } = C;
const RBLUE = "2A78D6", RBLUE_L = "E3EEFB";          // R's colour in the protocol figure

// Results: AdaSSP benchmark and real-data sweep (experiments/adassp), calibration reports
const ADA9 = readCsv(P("reports", "adassp", "results.csv"));
const ADA_ALL = readCsv(P("reports", "adassp", "all", "results.csv"));
const ADA_CMP = readCsv(P("reports", "adassp", "all", "compare.csv"));
const SWEEP_X = readCsv(P("reports", "adassp", "sweep", "crossings.csv"));
const sec = (tables, prefix) => tables[Object.keys(tables).find((k) => k.startsWith(prefix))];
const CAL = mdTables(P("reports", "calibration", "heuristic.md"));        // experiments/calibration_heuristic.py
const PROT = mdTables(P("reports", "calibration", "protocol.md"));        // experiments/calibration_protocol.py
const ERRBARS = {};
const adaGet = (rows, ds, eps, m) => rows.find((r) => r.dataset === ds && Number(r.eps) === eps && r.method === m);

(async () => {
  const { pres, icons, h } = await newDeck("Regression across a register and a cohort, without pooling", {
    user: fa.FaUserGraduate, bank: fa.FaLandmark, link: fa.FaLink, lock: fa.FaLock, unlock: fa.FaUnlockAlt,
    shield: fa.FaShieldAlt, eye: fa.FaEyeSlash, key: fa.FaKey, check: fa.FaCheckCircle, chart: fa.FaChartLine,
    flask: fa.FaFlask, db: fa.FaDatabase, book: fa.FaBook, balance: fa.FaBalanceScale, ledger: fa.FaClipboardList,
    hourglass: fa.FaHourglassHalf, calc: fa.FaCalculator, layers: fa.FaLayerGroup, cogs: fa.FaCogs,
    random: fa.FaRandom, warn: fa.FaExclamationTriangle, upload: fa.FaUpload, ruler: fa.FaRulerCombined,
    users: fa.FaUsers, puzzle: fa.FaPuzzlePiece, flag: fa.FaFlagCheckered, search: fa.FaSearch,
  });
  const { title, circleIcon, numCircle, card, txt, bullets, chip, img, latex, tableOpts, hdr, divider } = h;
  const party = (s, who, x, y) => chip(s, who === "R" ? "R" : who === "O" ? "O" : "R + O", x, y, who === "both" ? 0.75 : 0.42,
    who === "R" ? RBLUE : who === "O" ? EMBER : SLATE);

  // ======================= Title =======================
  {
    const s = pres.addSlide(); s.background = { color: INK };
    s.addShape(pres.shapes.OVAL, { x: 7.75, y: 1.55, w: 3.3, h: 3.3, fill: { color: SLATE, transparency: 35 }, line: { color: "8FA6C4", width: 1.5 } });
    s.addShape(pres.shapes.OVAL, { x: 9.35, y: 1.55, w: 3.3, h: 3.3, fill: { color: EMBER, transparency: 45 }, line: { color: "E7A77C", width: 1.5 } });
    s.addText("R", { x: 8.1, y: 2.85, w: 0.8, h: 0.7, fontFace: HEAD, fontSize: 30, bold: true, color: WHITE, margin: 0, isTextBox: true });
    s.addText("O", { x: 11.45, y: 2.85, w: 0.8, h: 0.7, fontFace: HEAD, fontSize: 30, bold: true, color: WHITE, margin: 0, isTextBox: true });
    s.addImage({ data: icons.lock, x: 10.15, y: 2.95, w: 0.5, h: 0.5 });
    s.addText("Regression across a register and a cohort, without pooling", { x: 0.8, y: 1.45, w: 6.7, h: 2.1,
      fontFace: HEAD, fontSize: 36, bold: true, color: WHITE, margin: 0, valign: "top", isTextBox: true });
    s.addText("Privacy-preserving vertical federated linear regression: motivation, protocol and results", { x: 0.8, y: 3.65, w: 6.6, h: 0.9,
      fontFace: BODY, fontSize: 18, color: "CAD5E6", margin: 0, valign: "top", isTextBox: true });
    s.addText("Internal training  ·  October 2026", { x: 0.8, y: 6.3, w: 7, h: 0.4, fontFace: BODY, fontSize: 13, color: "8FA6C4", margin: 0, isTextBox: true });
    s.addNotes("An introduction for people meeting this material for the first time. A researcher and a national statistical agency want to fit a linear regression on the people they have in common, where the agency holds the outcome. Neither may share its records. We build the protocol up from three standard tools (differential privacy, homomorphic encryption and private set intersection), state what it guarantees, show how to choose its parameters, and look at the results so far.");
  }

  // ======================= Agenda =======================
  {
    const s = pres.addSlide(); title(s, "What we will cover");
    const items = [
      ["The problem", "Why join a research cohort to an agency register, and what must stay private"],
      ["Building blocks", "Differential privacy, homomorphic encryption, private set intersection"],
      ["The protocol", "Six phases, from setup to confidence intervals"],
      ["Guarantees", "What each party learns, and what is proven"],
      ["Choosing parameters", "From a privacy budget to noise, ridge and the sample size needed"],
      ["Results", "Accuracy in simulation and on real data, and what is still to run"],
    ];
    items.forEach(([hd, b], i) => {
      const x = 0.6 + (i % 3) * 4.1, y = 1.55 + Math.floor(i / 3) * 2.65;
      card(s, x, y, 3.9, 2.4);
      numCircle(s, i + 1, x + 0.3, y + 0.3, 0.7, i === 2 ? EMBER : INK, 20);
      txt(s, hd, { x: x + 0.3, y: y + 1.12, w: 3.35, h: 0.45, fontFace: HEAD, fontSize: 20, bold: true, color: INK });
      txt(s, b, { x: x + 0.3, y: y + 1.6, w: 3.35, h: 0.75, fontSize: 14, color: MUTED });
    });
    s.addNotes("Six parts. The building blocks are short primers: enough to follow the protocol, not a full course. The protocol and its guarantees are the core. The parameter and results parts show how the pieces trade off in practice.");
  }

  // ======================= 1. The problem =======================
  divider(1, "The problem", "Joint regression across data that cannot be pooled")
    .addNotes("Part 1: the setting, the data layout, what has to be computed, and the privacy requirements.");

  {
    const s = pres.addSlide(); title(s, "Two datasets, one question, no pooling");
    card(s, 0.6, 1.55, 5.6, 3.55); circleIcon(s, "user", 0.9, 1.85, 0.8, RBLUE);
    txt(s, "Researcher  R", { x: 1.9, y: 1.95, w: 4, h: 0.5, fontFace: HEAD, fontSize: 22, bold: true, color: INK });
    bullets(s, ["Smaller cohort with rich domain variables: health, education, surveys",
      "Wants to know whether these predict an administrative outcome",
      "Holds features X_R and identifiers"], { x: 0.95, y: 2.85, w: 5.0, h: 2.1 });
    card(s, 6.6, 1.55, 6.1, 3.55); circleIcon(s, "bank", 6.9, 1.85, 0.8, EMBER);
    txt(s, "Organization  O  (e.g. StatCan)", { x: 7.9, y: 1.97, w: 4.7, h: 0.5, fontFace: HEAD, fontSize: 20, bold: true, color: INK });
    bullets(s, ["Population register: tax filings, income, benefits",
      "Holds the outcome y and features X_O",
      "Cannot release records; law and governance bar direct sharing"], { x: 6.95, y: 2.85, w: 5.5, h: 2.1 });
    txt(s, "6–8 weeks", { x: 0.6, y: 5.45, w: 3.6, h: 0.9, fontFace: HEAD, fontSize: 40, bold: true, color: EMBER });
    txt(s, "of vetting and approvals today before a researcher can link a cohort to agency records in a Research Data Centre. Neither dataset alone answers the question.",
      { x: 4.3, y: 5.55, w: 8.4, h: 0.9, fontSize: 16, color: MUTED });
    s.addNotes("The motivating setting. A researcher's cohort and an agency's register cover overlapping people but different variables. Today, linking them means weeks of vetting and work inside a secure Research Data Centre. The goal is a protocol that answers the regression question directly, without either side handing over records.");
  }

  {
    const s = pres.addSlide(); title(s, "Same people, different variables", "Vertically partitioned data: overlapping rows, disjoint columns");
    const top = 2.1, rowH = 0.25;
    txt(s, "R: cohort (n_R rows)", { x: 0.8, y: top + 3 * rowH - 0.4, w: 3.2, h: 0.35, bold: true, color: RBLUE, fontSize: 14 });
    for (let i = 0; i < 12; i++) {
      const matched = i >= 3 && i < 11;
      s.addShape(pres.shapes.RECTANGLE, { x: 0.8, y: top + (i + 3) * rowH, w: 2.6, h: rowH - 0.04,
        fill: { color: matched ? "C9DCF5" : RBLUE_L }, line: { color: WHITE } });
    }
    txt(s, "X_R", { x: 0.8, y: top + 15 * rowH + 0.05, w: 2.6, h: 0.35, align: "center", color: RBLUE, bold: true, fontSize: 15 });
    txt(s, "O: register (n_O rows, n_O ≫ n_R)", { x: 4.3, y: top - 0.4, w: 4.2, h: 0.35, bold: true, color: EMBER, fontSize: 14 });
    for (let k = 0; k < 18; k++) {
      const matched = k >= 6 && k < 14;
      s.addShape(pres.shapes.RECTANGLE, { x: 4.3, y: top + k * rowH, w: 3.2, h: rowH - 0.04,
        fill: { color: matched ? "F3D2BC" : "F8EDE5" }, line: { color: WHITE } });
      s.addShape(pres.shapes.RECTANGLE, { x: 7.55, y: top + k * rowH, w: 0.5, h: rowH - 0.04,
        fill: { color: matched ? "E3A57B" : "F1D9C8" }, line: { color: WHITE } });
    }
    txt(s, "X_O", { x: 4.3, y: top + 18 * rowH + 0.05, w: 3.2, h: 0.35, align: "center", color: EMBER, bold: true, fontSize: 15 });
    txt(s, "y", { x: 7.55, y: top + 18 * rowH + 0.05, w: 0.5, h: 0.35, align: "center", color: EMBER, bold: true, fontSize: 15 });
    s.addShape(pres.shapes.RECTANGLE, { x: 3.55, y: top + 6 * rowH, w: 0.6, h: 8 * rowH - 0.04, fill: { color: INK, transparency: 85 }, line: { color: INK, width: 1, dashType: "dash" } });
    txt(s, "n\nmatched", { x: 3.45, y: top + 9.2 * rowH, w: 0.8, h: 0.8, align: "center", fontSize: 11, bold: true, color: INK });
    card(s, 8.7, 1.55, 4.0, 5.1);
    txt(s, "Goal", { x: 9.0, y: 1.8, w: 3.5, h: 0.4, fontFace: HEAD, fontSize: 20, bold: true, color: INK });
    bullets(s, ["Fit y on [X_R | X_O] over the n matched people",
      "Neither party sees the other's records",
      "O must not learn who is in R's cohort",
      "The released model must not expose any individual"], { x: 9.0, y: 2.35, w: 3.5, h: 4.1 });
    s.addNotes("Rows are people, columns are variables. The two tables overlap only on the matched people, and the register is far larger than the cohort. This layout is called vertical partitioning: the parties hold different columns for the same rows. The regression we want uses R's columns and O's columns together, on the matched rows only.");
  }

  {
    const s = pres.addSlide(); title(s, "What has to be computed", "Least squares needs only a few sums over the matched rows, and those sums span both parties");
    latex(s, "ols_I", 0.6, 1.78, { size: 26, maxW: 7 });
    const gx = 0.9, gy = 2.7, c = 1.55;
    const cell = (x, y, label, sub, fill, color = INK) => {
      s.addShape(pres.shapes.RECTANGLE, { x, y, w: c, h: c, fill: { color: fill }, line: { color: WHITE, width: 2 } });
      s.addText([{ text: label, options: { fontFace: HEAD, fontSize: 24, bold: true, color, breakLine: !!sub } },
        ...(sub ? [{ text: sub, options: { fontFace: BODY, fontSize: 14, color } }] : [])],
        { x, y, w: c, h: c, align: "center", valign: "middle", margin: 2, isTextBox: true });
    };
    // every block is a sum over the matched rows I, hence the superscript
    cell(gx, gy, "A^I", "X_R^{I⊤} X_R^I", "C9DCF5");
    cell(gx + c, gy, "C^I", "X_R^{I⊤} X_O^I", EMBER, WHITE);
    cell(gx, gy + c, "C^{I⊤}", "", EMBER, WHITE);
    cell(gx + c, gy + c, "B^I", "X_O^{I⊤} X_O^I", "F3D2BC");
    cell(gx + 2 * c + 0.4, gy, "c_R^I", "X_R^{I⊤} y^I", EMBER, WHITE);
    cell(gx + 2 * c + 0.4, gy + c, "c_O^I", "X_O^{I⊤} y^I", "F3D2BC");
    txt(s, "G^I = ZᵀZ", { x: gx, y: gy + 2 * c + 0.1, w: 2 * c, h: 0.4, align: "center", fontFace: HEAD, fontSize: 16, color: MUTED });
    txt(s, "c^I = Zᵀy^I", { x: gx + 2 * c + 0.4, y: gy + 2 * c + 0.1, w: c, h: 0.4, align: "center", fontFace: HEAD, fontSize: 16, color: MUTED });
    const lg = [["C9DCF5", "R's rows in I only: R computes it exactly, and it never leaves R"],
      ["F3D2BC", "O's rows π(I) only, as is (yᵀy)^I: depends on which rows match, not on how they pair"],
      [EMBER, "Cross-party: pairs R's row i with O's row π(i), so it depends on the bijection π"]];
    lg.forEach(([col, t], i) => {
      s.addShape(pres.shapes.RECTANGLE, { x: 6.7, y: 2.72 + i * 0.72, w: 0.4, h: 0.4, fill: { color: col }, line: { color: col } });
      txt(s, t, { x: 7.25, y: 2.7 + i * 0.72, w: 5.45, h: 0.65, fontSize: 14.5 });
    });
    card(s, 6.7, 4.95, 6.0, 1.85, EMBER_L);
    txt(s, [{ text: "I", options: { bold: true } }, { text: " = the cohort members found in O's register. " },
      { text: "π", options: { bold: true } }, { text: " = the bijection from I to O's matching rows; X_O^I and y^I list O's rows π(i) in the order of I. Everything that touches O's data is computed under encryption and released to R only with noise. From here on we drop the superscript I." }],
      { x: 6.95, y: 5.07, w: 5.55, h: 1.65, fontSize: 13.5, color: INK, valign: "middle" });
    s.addNotes("Ordinary least squares needs the Gram matrix G = ZᵀZ and the vector c = Zᵀy over the matched rows. These are sufficient statistics: once you have them, you never need the rows again. Every block is a sum over the matched set I, which is why each carries a superscript I. They differ in what they need to know about the matching. A^I uses only R's rows in I, so R computes it exactly and it never leaves R. B^I, c_O^I and the sum of squares (yᵀy)^I use only O's matched rows π(I): they depend on which rows match but not on how R's and O's rows pair up. C^I and c_R^I multiply R's row i by O's row π(i), so they depend on the bijection π itself; this is why the linkage step must give R the position of every match, not just the match set. The protocol computes all O-dependent sums under encryption and releases them to R only after adding noise. On later slides the superscript I is dropped.");
  }

  {
    const s = pres.addSlide(); title(s, "Three requirements");
    const items = [
      ["eye", "Input privacy", "O learns nothing about R's features. Its whole view can be simulated from public sizes.", RBLUE],
      ["link", "Cohort privacy", "O learns nothing about who is in R's cohort, not even its exact size.", TEAL],
      ["shield", "Output privacy", "Everything R receives is differentially private for each person's O-side data (x_O, y).", EMBER],
    ];
    items.forEach(([ic, hd, b, col], i) => {
      const x = 0.6 + i * 4.15;
      card(s, x, 1.55, 3.9, 3.45);
      circleIcon(s, ic, x + 0.3, 1.85, 0.85, col);
      txt(s, hd, { x: x + 0.3, y: 2.9, w: 3.3, h: 0.5, fontFace: HEAD, fontSize: 21, bold: true, color: INK });
      txt(s, b, { x: x + 0.3, y: 3.45, w: 3.35, h: 1.5, fontSize: 15 });
    });
    card(s, 0.6, 5.3, 12.1, 1.5, "F7F9FC");
    circleIcon(s, "users", 0.85, 5.62, 0.8, INK);
    txt(s, [{ text: "Honest-but-curious parties. ", options: { bold: true } },
      { text: "Both follow the protocol but analyse everything they receive. R and O each hold a share of the decryption key. By design, R learns which members of its cohort are in the register." }],
      { x: 1.9, y: 5.45, w: 10.6, h: 1.2, fontSize: 15.5, color: INK, valign: "middle" });
    s.addNotes("The three requirements. Input privacy protects R's data from O; cohort privacy protects the fact of who R is studying; output privacy protects the people in O's register from R, the analyst. The threat model is honest-but-curious, also called semi-honest: nobody deviates from the protocol, but everyone tries to learn from what they see. A researcher who deliberately deviated could break output privacy; covering that needs extra machinery, discussed at the end. One thing is disclosed by design: R learns which of its own cohort members appear in the register.");
  }

  {
    const s = pres.addSlide(); title(s, "Where this work sits");
    const rows = [
      hdr(["Approach", "Where the data sit", "Record linkage", "Output privacy", "Gap for this setting"]),
      ["Central DP regression (e.g. AdaSSP)", "One trusted curator holds the joined table", "Not needed", "Differential privacy", "Assumes the pooled data exist"],
      ["Federated learning with encryption", "Split across parties", "Assumed, or a separate step", "Usually none", "The model can leak individuals"],
      ["Private set intersection", "Split across parties", "Yes", "Not applicable", "Linkage only, no model"],
      [{ text: "This protocol", options: { bold: true, color: EMBER } }, "Split, never pooled", "Labeled PSI: O learns nothing", "DP on all R receives; budget enforced by O", { text: "Protects O's side; R learns which of its cohort match", options: { color: MUTED } }],
    ];
    s.addTable(rows, Object.assign(tableOpts([2.9, 2.5, 2.25, 2.25, 2.2]), { y: 1.6, rowH: 0.62, fontSize: 13.5 }));
    card(s, 0.6, 5.35, 12.1, 1.35, EMBER_L);
    txt(s, "The statistical engine is the classic one for DP regression: perturb the sufficient statistics, then solve a ridge regression. What is new is doing it without pooling, with R's own block kept exact.",
      { x: 0.9, y: 5.55, w: 11.5, h: 1.0, fontSize: 16.5, color: INK, valign: "middle" });
    s.addNotes("Central DP regression, such as AdaSSP by Wang (2018), assumes one curator already holds the joined table. Federated learning with homomorphic encryption keeps data split but usually releases an unprotected model. Private set intersection only does the linkage. This protocol combines private linkage, encrypted aggregation and differential privacy. Its guarantee is narrower than central DP in one respect: it protects O's side of each record, and R learns which of its cohort members match.");
  }

  // ======================= 2. Building blocks =======================
  divider(2, "Building blocks", "Differential privacy, homomorphic encryption, private set intersection")
    .addNotes("Part 2: three short primers, enough to follow the protocol.");

  {
    const s = pres.addSlide(); title(s, "Differential privacy in one slide", "A release is private if no single person's data changes it by much");
    latex(s, "dp_def", 0.6, 1.8, { size: 21, maxW: 6.6 });
    bullets(s, [
      { text: "D and D′ differ in one person's O-side data (x_O, y): replace-one adjacency", options: {} },
      "ε caps how much one person can shift the odds of any outcome (a factor e^ε)",
      "δ is a small slack; take δ ≤ 1/(10N) for a register of N people",
      "Post-processing is free: anything computed from a DP release stays DP",
    ], { x: 0.6, y: 2.55, w: 6.5, h: 2.6, fontSize: 15 });
    card(s, 0.6, 5.3, 6.5, 1.5, EMBER_L);
    txt(s, "The Gaussian mechanism", { x: 0.85, y: 5.4, w: 6, h: 0.35, fontSize: 15, bold: true, color: EMBER });
    latex(s, "gauss", 0.85, 5.82, { size: 17, maxW: 6.0 });
    latex(s, "sens", 0.85, 6.3, { size: 15, maxW: 6.0 });
    // two output densities, one sensitivity apart
    const xs = [], d0 = [], d1 = [];
    for (let v = -4; v <= 5.001; v += 0.25) { xs.push(v.toFixed(1)); d0.push(Math.exp(-v * v / 2)); d1.push(Math.exp(-(v - 1) * (v - 1) / 2)); }
    s.addChart(pres.charts.LINE, [{ name: "M(D)", labels: xs, values: d0 }, { name: "M(D′)", labels: xs, values: d1 }], {
      x: 7.4, y: 1.7, w: 5.3, h: 3.6, chartColors: [SLATE, EMBER], lineSize: 2.5, lineDataSymbol: "none", lineSmooth: true,
      showTitle: true, title: "Outputs on neighbouring datasets", titleFontFace: HEAD, titleFontSize: 14, titleColor: INK,
      showLegend: true, legendPos: "b", legendFontSize: 12, catAxisHidden: true, valAxisHidden: true,
      valGridLine: { style: "none" }, catGridLine: { style: "none" },
    });
    txt(s, "The two curves are shifted by the sensitivity Δ. The wider the noise relative to Δ, the harder it is to tell them apart.",
      { x: 7.6, y: 5.45, w: 5.0, h: 1.2, fontSize: 14, color: MUTED });
    s.addNotes("Differential privacy is a property of a randomized release M. For any two neighbouring datasets D and D', and any set of outcomes S, the probability of S changes by at most a factor e^epsilon, plus a small slack delta. Here neighbours differ in one person's O-side data, with R's features held fixed: replace-one adjacency. The Gaussian mechanism adds normal noise to a statistic f. Its scale is the sensitivity Delta, the most f can move between neighbours, times a multiplier s that depends only on (epsilon, delta). Two properties we use constantly: post-processing cannot weaken the guarantee, and costs of several releases add up (next slides).");
  }

  {
    const s = pres.addSlide(); title(s, "How much one person can move the statistics", "The sensitivity Δ sets the noise; it follows from public clipping bounds");
    txt(s, "Every row is clipped to public bounds: ‖x_R‖ ≤ B_R (with the intercept), ‖x_O‖ ≤ B_O and |y| ≤ B_y. Replacing one person's (x_O, y) changes B, C, c_R, c_O and yᵀy by at most Δ:",
      { x: 0.6, y: 1.75, w: 12.1, h: 0.75, fontSize: 15.5 });
    latex(s, "delta_rep", 0.6, 2.62, { size: 21, maxW: 9 });
    const rows = [hdr(["Bounds (B_R, B_O, B_y)", "Δ", "Comment"]),
      ["(1, 1, 1)", "3.16", "unit bounds"],
      ["(0.71, 0.71, 1)", "2.03", "one unit row bound split between the parties"],
      ["(5.59, 6.24, 2.89)", "86.2", "simulation population (standardized income, age, counts)"]];
    s.addTable(rows, Object.assign(tableOpts([2.6, 1.0, 4.1]), { y: 3.45, rowH: 0.44, fontSize: 13.5 }));
    card(s, 8.6, 3.45, 4.1, 3.3, EMBER_L);
    bullets(s, ["A = X_Rᵀ X_R is computed exactly by R and never released, so it adds nothing to Δ",
      "yᵀy is free when B_y² ≤ 2(B_R² + B_O²); otherwise it gets its own budget",
      "Δ grows with the squared bounds: tight bounds matter most"],
      { x: 8.85, y: 3.65, w: 3.7, h: 3.0, fontSize: 14 });
    txt(s, "Bounds are chosen from public reference data before any data are seen; values beyond them are clipped.",
      { x: 0.6, y: 5.45, w: 7.7, h: 0.8, fontSize: 13.5, italic: true, color: MUTED });
    s.addNotes("The released vector is (triu B, C, c_R, c_O, yT y). Replacing one person's O-side data, with R's row fixed, moves it by at most Delta in the l2 norm. With R, O, Y the squared bounds, the usual case has the closed form on the slide; it is the exact supremum when O has at least two features. Three practical consequences. Keeping A out of the release removes R's fourth-power term, which is why A stays exact at R. yT y adds nothing when B_y^2 <= 2(B_R^2 + B_O^2), which covers all our settings; otherwise it is released separately. And since Delta scales with the squares of the bounds, choosing tight bounds is the most powerful lever on accuracy. Splitting one row bound between the parties, B_R^2 = B_O^2 = 1/2, gives Delta = 2.03 instead of 3.16.");
  }

  {
    const s = pres.addSlide(); title(s, "Budgets add up, and O keeps the books", "Zero-concentrated DP (zCDP) makes the accounting simple");
    latex(s, "zcdp", 0.6, 1.8, { size: 22, maxW: 6.3 });
    latex(s, "zcdp_conv", 0.6, 2.75, { size: 19, maxW: 6.3 });
    bullets(s, ["Each release s costs ρ_s; the costs of a run add up",
      "The run's total converts to (ε, δ); the protocol uses the tight conversion, and the formula above is a close, simple bound",
      "Doubling ε roughly halves the noise σ"], { x: 0.6, y: 3.55, w: 6.3, h: 2.2, fontSize: 15 });
    // budget split bar
    txt(s, "One run = two releases", { x: 7.4, y: 1.8, w: 5.3, h: 0.4, fontFace: HEAD, fontSize: 17, bold: true, color: INK });
    s.addShape(pres.shapes.RECTANGLE, { x: 7.4, y: 2.35, w: 0.53, h: 0.7, fill: { color: SLATE }, line: { color: WHITE, width: 2 } });
    s.addShape(pres.shapes.RECTANGLE, { x: 7.93, y: 2.35, w: 4.77, h: 0.7, fill: { color: EMBER }, line: { color: WHITE, width: 2 } });
    txt(s, "Stage 2: all statistics, 90%", { x: 8.1, y: 2.35, w: 4.5, h: 0.7, fontSize: 14, bold: true, color: WHITE, valign: "middle" });
    txt(s, "Stage 1: B and C only, 10% of ρ, used to choose the ridge", { x: 7.4, y: 3.12, w: 5.3, h: 0.5, fontSize: 13, color: MUTED });
    card(s, 7.4, 3.9, 5.3, 2.85, EMBER_L);
    circleIcon(s, "ledger", 7.65, 4.12, 0.7, EMBER);
    txt(s, "Ledger at O", { x: 8.55, y: 4.2, w: 4, h: 0.45, fontSize: 17, bold: true, color: INK });
    txt(s, "One ledger per register, shared by all researchers. O refuses a run once the cap would be passed. Nothing can be decrypted without O's share, so the cap is enforced technically.",
      { x: 8.55, y: 4.7, w: 3.95, h: 1.95, fontSize: 14 });
    s.addNotes("zCDP measures privacy loss with a single number rho. A Gaussian release with sensitivity Delta and noise sigma costs Delta^2 / (2 sigma^2), and the costs of several releases simply add. At the end, the total converts to an (epsilon, delta) statement; the protocol uses the tight conversion of Canonne, Kamath and Steinke, and the simple bound on the slide is within a few tens of percent of it. A protocol run makes two releases: a small one (10% of the budget) used only to choose the ridge penalty, and the main one. O keeps a ledger per register, shared across researchers, because an individual can appear in many cohorts. Because R cannot decrypt anything without O's key share, O can enforce the cap simply by refusing.");
  }

  {
    const s = pres.addSlide(); title(s, "Homomorphic encryption", "Computing on ciphertexts. CKKS: approximate arithmetic on encrypted vectors of real numbers");
    bullets(s, ["A ciphertext holds a whole vector: N/2 slots (N = 2^14 gives 8,192)",
      "O can add ciphertexts, multiply them by its own plaintext numbers, and rotate slots to sum them",
      "Values are scaled by Ω and rounded; decryption returns them plus a small error e",
      "Security rests on the Ring-LWE problem"], { x: 0.6, y: 1.8, w: 6.2, h: 2.9, fontSize: 15 });
    card(s, 0.6, 4.85, 6.2, 1.9, "F7F9FC");
    txt(s, "Decryption with secret key s", { x: 0.85, y: 4.97, w: 5.8, h: 0.4, fontSize: 15, bold: true, color: SLATE });
    latex(s, "ckks_dec", 0.85, 5.5, { size: 21, maxW: 5.8 });
    txt(s, "A ciphertext is a pair (c_0, c_1) of large polynomials.", { x: 0.85, y: 6.2, w: 5.8, h: 0.4, fontSize: 13, italic: true, color: MUTED });
    card(s, 7.1, 1.8, 5.6, 4.95, EMBER_L);
    circleIcon(s, "key", 7.35, 2.05, 0.75, EMBER);
    txt(s, "Split key (threshold decryption)", { x: 8.3, y: 2.15, w: 4.3, h: 0.5, fontSize: 17, bold: true, color: INK });
    txt(s, "The secret key is the sum of two shares, one per party. Decrypting needs both:", { x: 7.35, y: 3.0, w: 5.1, h: 0.7, fontSize: 14.5 });
    latex(s, "thresh", 7.35, 3.85, { size: 16, maxW: 5.1 });
    txt(s, "So neither party can decrypt alone, and O can attach its noise to its own part of every decryption.",
      { x: 7.35, y: 5.35, w: 5.1, h: 1.2, fontSize: 14.5 });
    s.addNotes("CKKS is a homomorphic encryption scheme for approximate arithmetic. One ciphertext packs a vector of up to N/2 real numbers into slots. The operations we need are adding ciphertexts, multiplying a ciphertext by plaintext numbers that O holds, and rotating slots so that a vector can be summed into one slot. Values are encoded by multiplying by a large scale Omega and rounding, so decryption returns the value plus a small error. Decryption itself is c0 + c1 s. In the threshold variant the secret is split, s = s_R + s_O: R can compute c1 s_R, O can compute c0 + c1 s_O, and only together do they give the plaintext. That split is what lets O add privacy noise inside its share of the decryption.");
  }

  {
    const s = pres.addSlide(); title(s, "Private set intersection", "Linking without revealing. Labeled PSI (Chen, Laine and Rindal): R learns its matches, O learns nothing");
    const steps = [
      ["1", "Blind the identifiers", "R gets a keyed hash F_k(id) of each cohort id from O (an OPRF), without O seeing the ids. R pads to a public cap n_R^max."],
      ["2", "Match under encryption", "R sends encrypted queries; O evaluates polynomials built from its own hashed register and returns encrypted answers."],
      ["3", "Read the result", "R decrypts: which ids match, and a label for each match, here the row's position in O's secretly shuffled register."],
    ];
    steps.forEach(([n, hd, b], i) => {
      const x = 0.6 + i * 4.15;
      card(s, x, 1.85, 3.9, 3.0);
      numCircle(s, n, x + 0.3, 2.08, 0.65, i === 2 ? EMBER : INK, 18);
      txt(s, hd, { x: x + 1.1, y: 2.15, w: 2.7, h: 0.5, fontFace: HEAD, fontSize: 18, bold: true, color: INK });
      txt(s, b, { x: x + 0.3, y: 2.9, w: 3.35, h: 1.85, fontSize: 14.5 });
    });
    card(s, 0.6, 5.15, 5.95, 1.6, RBLUE_L);
    txt(s, "R learns", { x: 0.85, y: 5.27, w: 5.5, h: 0.4, fontSize: 15, bold: true, color: RBLUE });
    txt(s, "The match set I and each match's position in O's shuffled order: enough to line its data up with O's rows later.",
      { x: 0.85, y: 5.68, w: 5.5, h: 1.0, fontSize: 14 });
    card(s, 6.75, 5.15, 5.95, 1.6, EMBER_L);
    txt(s, "O learns", { x: 7.0, y: 5.27, w: 5.5, h: 0.4, fontSize: 15, bold: true, color: EMBER });
    txt(s, "Only the padded sizes n_R^max and n̄_O. Its shuffle is fresh for every run, so positions reveal nothing.",
      { x: 7.0, y: 5.68, w: 5.5, h: 1.0, fontSize: 14 });
    s.addNotes("Linkage uses labeled private set intersection in the style of Chen, Laine and Rindal (2018). First, an oblivious PRF: R obtains a keyed hash of each of its identifiers from O without revealing them, and pads its list to a public cap so O does not even learn the cohort size. Second, R sends encrypted queries (BFV encryption under R's own key), and O evaluates polynomials built from its hashed register. Third, R decrypts and learns which identifiers match and, for each match, a label. The label is the row's position in O's register after a secret random shuffle, which O draws fresh for every run. R needs these positions to line up its features with O's rows in the next phase. O learns only the public padded sizes.");
  }

  // ======================= 3. The protocol =======================
  divider(3, "The protocol", "Six phases, from setup to confidence intervals")
    .addNotes("Part 3: the protocol phase by phase. Blue is R, orange is O.");

  {
    const s = pres.addSlide(); title(s, "The protocol at a glance", "What each party does, and what crosses the wire");
    const ph = [
      ["0", "Setup", "Agree public parameters; key share s_R", "setup and key shares", "both", "Bounds → Δ, σ, Ω; ledger check; key share s_O"],
      ["1", "Link", "Learns matches and positions; computes A", "OPRF and PSI", "both", "Answers PSI over its padded register"],
      ["2", "Upload", "Encrypts its features in O's row order", "encrypted vectors", "R", "Converts them to the joint key"],
      ["3", "Aggregate", "", "", "", "Computes B, C, c_R, c_O, yᵀy under encryption"],
      ["4", "Release ×2", "Adds its share; decodes noisy statistics", "(c₁′, h_O)", "O", "Re-randomizes; adds noise in its decryption share"],
      ["5", "Solve", "Ridge from release 1; estimate and intervals from release 2", "", "", ""],
    ];
    const x0 = 1.75, cw = 1.82, gap = 0.04;
    txt(s, "R", { x: 0.6, y: 2.25, w: 1.0, h: 0.5, fontFace: HEAD, fontSize: 24, bold: true, color: RBLUE, align: "center" });
    txt(s, "O", { x: 0.6, y: 5.35, w: 1.0, h: 0.5, fontFace: HEAD, fontSize: 24, bold: true, color: EMBER, align: "center" });
    ph.forEach(([n, hd, r, msg, dir, o], i) => {
      const x = x0 + i * (cw + gap);
      numCircle(s, n, x + cw / 2 - 0.25, 1.55, 0.5, i === 4 ? EMBER : INK, 14);
      if (r) { card(s, x, 2.15, cw, 1.45, RBLUE_L); txt(s, r, { x: x + 0.1, y: 2.22, w: cw - 0.2, h: 1.35, fontSize: 12.5, color: INK }); }
      if (o) { card(s, x, 4.65, cw, 1.45, EMBER_L); txt(s, o, { x: x + 0.1, y: 4.72, w: cw - 0.2, h: 1.35, fontSize: 12.5, color: INK }); }
      if (msg) {
        const ax = x + cw / 2;
        if (dir === "R" || dir === "both") s.addShape(pres.shapes.LINE, { x: ax - (dir === "both" ? 0.12 : 0), y: 3.7, w: 0, h: 0.85, line: { color: RBLUE, width: 2, endArrowType: "triangle" } });
        if (dir === "O" || dir === "both") s.addShape(pres.shapes.LINE, { x: ax + (dir === "both" ? 0.12 : 0), y: 3.7, w: 0, h: 0.85, line: { color: EMBER, width: 2, beginArrowType: "triangle" } });
        txt(s, msg, { x: x + 0.02, y: 3.8, w: cw / 2 - 0.17, h: 0.7, fontSize: 10, color: MUTED, align: "right", valign: "middle" });
      }
      txt(s, hd, { x, y: 6.2, w: cw, h: 0.4, align: "center", fontFace: HEAD, fontSize: 14, bold: true, color: INK });
    });
    txt(s, "O never receives a decryption share from R; R never receives the part of a ciphertext that would expose O's key share.",
      { x: 0.6, y: 6.75, w: 12.1, h: 0.4, fontSize: 13, italic: true, color: MUTED });
    s.addNotes("The whole protocol in one picture. Phase 0: both parties agree public parameters and generate a split key. Phase 1: private linkage; R learns which of its cohort members match and where they sit in O's shuffled register, and computes its own block A. Phase 2: R uploads its features, encrypted, in O's row order. Phase 3: O computes the needed sums under encryption. Phase 4: O releases them in two rounds, adding noise inside its share of the decryption. Phase 5: R chooses the ridge from the first release, then computes the estimate and confidence intervals from the second. The next slides take each phase in turn.");
  }

  {
    const s = pres.addSlide(); title(s, "Phase 0: setup", "Everything that sets the noise is public and fixed before any data are touched");
    card(s, 0.6, 1.75, 5.95, 5.0);
    party(s, "both", 0.9, 1.97);
    txt(s, "Agreed in public", { x: 1.8, y: 1.92, w: 4.5, h: 0.45, fontFace: HEAD, fontSize: 19, bold: true, color: INK });
    bullets(s, ["Study scope Φ, on public attributes only (province, birth-year band)",
      "Caps: cohort n_R^max; register padded to n̄_O with dummy rows",
      "Clipping bounds B_R (with the intercept), B_O, B_y, from public reference data",
      "Budget (ε, δ); stage-1 share f_1 = 0.1; validity target ϑ* = 2.5; error level α"],
      { x: 0.95, y: 2.6, w: 5.35, h: 4.0, fontSize: 14.5 });
    card(s, 6.75, 1.75, 5.95, 2.7, EMBER_L);
    party(s, "O", 7.05, 1.97);
    txt(s, "Derived by O", { x: 7.65, y: 1.92, w: 4.5, h: 0.45, fontFace: HEAD, fontSize: 19, bold: true, color: INK });
    bullets(s, ["Sensitivities Δ and Δ_G from the bounds",
      "Noise σ_1, σ_2 and the CKKS scale Ω (Calibrate)",
      "Ledger check: refuse if the run would pass the cap",
      "A fresh secret shuffle τ of its rows and a fresh OPRF key"], { x: 7.05, y: 2.55, w: 5.4, h: 1.85, fontSize: 14 });
    card(s, 6.75, 4.65, 5.95, 2.1, "F7F9FC");
    party(s, "both", 7.05, 4.87);
    txt(s, "Split key generation", { x: 7.95, y: 4.82, w: 4.5, h: 0.45, fontFace: HEAD, fontSize: 19, bold: true, color: INK });
    txt(s, "Joint public key, secret s = s_R + s_O, and rotation keys only. Neither party can decrypt alone.",
      { x: 7.05, y: 5.45, w: 5.4, h: 1.2, fontSize: 14 });
    s.addNotes("Phase 0. The parties agree on everything that determines the noise before touching data: a study scope defined on public attributes, a cap on the cohort size, the padded register size, clipping bounds from public reference data, and the privacy budget. From these O derives the sensitivities, the noise scales and the CKKS scale, and checks its ledger. O draws a fresh secret shuffle of its rows and a fresh OPRF key for this run. Finally the parties run threshold key generation: one joint public key whose secret key is split between them.");
  }

  {
    const s = pres.addSlide(); title(s, "Phase 1: link", "Labeled PSI tells R which cohort members are in the register, and where");
    const steps = [
      ["R", "Pad the cohort identifiers with random dummies to exactly n_R^max"],
      ["both", "OPRF: R gets F_k(id) for each padded id; O evaluates exactly n_R^max calls"],
      ["R", "Cuckoo-hash the hashed ids; send encrypted queries under R's own key"],
      ["O", "Evaluate membership and label polynomials over its padded register; reply"],
      ["R", "Decrypt: match set I, and positions π(i) = τ(j) in O's shuffled order"],
      ["R", "Compute its own block A = Σ_{i∈I} ẋ_i ẋ_iᵀ exactly, locally"],
    ];
    steps.forEach(([who, t], i) => {
      const y = 1.75 + i * 0.78;
      party(s, who, 0.6, y + 0.12);
      txt(s, t, { x: 1.55, y: y + 0.05, w: 6.3, h: 0.7, fontSize: 15, valign: "middle" });
    });
    card(s, 8.3, 1.75, 4.4, 4.95, MIST);
    txt(s, "After phase 1", { x: 8.55, y: 1.95, w: 4, h: 0.45, fontFace: HEAD, fontSize: 19, bold: true, color: INK });
    bullets(s, [{ text: "R knows I and the positions τ(j) of its matches", options: { color: RBLUE, bold: true } },
      "R has A exactly; it never leaves R",
      { text: "O knows only n_R^max and n̄_O", options: { color: EMBER, bold: true } },
      "The positions are uniformly random, because τ is fresh every run"], { x: 8.55, y: 2.55, w: 3.95, h: 4.0, fontSize: 14.5 });
    s.addNotes("Phase 1 in detail. R pads its identifier list to the public cap, so O learns nothing from the count. The OPRF gives R a keyed hash of each identifier. R cuckoo-hashes these and sends encrypted queries, encrypted under R's own BFV key, so O cannot read them. O evaluates membership and label polynomials and replies. R decrypts: it learns which identifiers match and, for each, its position in O's shuffled register. R then computes its own block A exactly. O has learned only the public sizes.");
  }

  {
    const s = pres.addSlide(); title(s, "Phase 2: upload", "R lines its data up with O's rows, encrypts them, and sends them over");
    card(s, 0.6, 1.75, 6.0, 3.0, RBLUE_L);
    party(s, "R", 0.9, 1.97);
    txt(s, "Vectors over all n̄_O positions", { x: 1.5, y: 1.92, w: 5.0, h: 0.45, fontSize: 16, bold: true, color: INK });
    bullets(s, ["Selection b: 1 at a matched position, 0 elsewhere",
      "Features ẋ_j: R's clipped feature j (and the intercept) at matched positions, 0 elsewhere",
      "In O's shuffled order, so O can multiply them by its own columns position by position"],
      { x: 0.9, y: 2.5, w: 5.5, h: 2.2, fontSize: 14.5 });
    card(s, 6.85, 1.75, 5.85, 3.0, "F7F9FC");
    txt(s, "Why this hides everything from O", { x: 7.1, y: 1.92, w: 5.4, h: 0.45, fontSize: 16, bold: true, color: INK });
    txt(s, "O sees only ciphertexts, which look random without R's key share. Matched and unmatched positions are indistinguishable, so O cannot tell who is in the cohort.",
      { x: 7.1, y: 2.45, w: 5.4, h: 2.2, fontSize: 14.5 });
    card(s, 0.6, 5.05, 12.1, 1.7, EMBER_L);
    latex(s, "upload", 0.9, 5.25, { size: 18, maxW: 11.5 });
    txt(s, "R encrypts with its own share and a short seed (half the upload); O folds in its share to convert to the joint key.",
      { x: 0.9, y: 6.1, w: 11.5, h: 0.5, fontSize: 13.5, color: MUTED });
    s.addNotes("Phase 2. R builds vectors as long as O's padded register, in O's shuffled order: a selection vector with a 1 at each matched position, and one vector per feature with R's clipped value at matched positions and zeros elsewhere. The first feature is the intercept. R encrypts each vector with its own key share, sending only c0 and a short seed from which the random a can be regenerated; this halves the upload. O subtracts a times its own share, which turns the ciphertext into one under the joint key. To O, all of this looks random, so matched and unmatched positions cannot be told apart.");
  }

  {
    const s = pres.addSlide(); title(s, "Phase 3: aggregate", "O multiplies R's encrypted vectors by its own plaintext columns and sums");
    card(s, 0.6, 1.75, 7.1, 3.5, "F7F9FC");
    latex(s, "agg", 0.95, 1.95, { size: 22, maxW: 6.6, maxH: 2.5 });
    txt(s, "b: selection vector;  ẋ_R: R's features in O's order (zero off the match set)", { x: 0.95, y: 4.6, w: 6.6, h: 0.5, fontSize: 12.5, color: MUTED, italic: true });
    const pts = [["layers", "Sum chunks, then EvalSum", "A register longer than one ciphertext is split into chunks; slot sums use rotations."],
      ["random", "Pack: one entry per slot", "Every released number sits alone in its own slot of a packed ciphertext."],
      ["balance", "Two packed ciphertexts", "M_1 holds only B and C (for stage 1); M_2 holds every entry (stage 2)."]];
    pts.forEach(([ic, hd, b], i) => {
      const y = 1.75 + i * 1.18;
      circleIcon(s, ic, 8.0, y, 0.62, i === 2 ? EMBER : SLATE);
      txt(s, hd, { x: 8.8, y: y - 0.02, w: 3.9, h: 0.4, fontSize: 15, bold: true, color: INK });
      txt(s, b, { x: 8.8, y: y + 0.38, w: 3.9, h: 0.75, fontSize: 13, color: MUTED });
    });
    card(s, 0.6, 5.55, 12.1, 1.2, EMBER_L);
    txt(s, "Unmatched positions contribute zero, because b and ẋ_R are zero there. The result is the matched-row statistics, still encrypted. Multiplicative depth is 2.",
      { x: 0.9, y: 5.65, w: 11.5, h: 1.0, fontSize: 15.5, color: INK, valign: "middle" });
    s.addNotes("Phase 3. Each O-dependent statistic is an inner product between one of R's encrypted vectors and one of O's plaintext columns, summed over positions. Multiplying by the selection vector restricts O's data to the matched rows. O first sums the chunk products, then does one EvalSum per entry to collect the total into a slot. It then packs the entries so that each sits alone in its own slot: M1 holds only B and C, for the first release, and M2 holds everything, for the second. Keeping them separate means the first release can only depend on B and C.");
  }

  {
    const s = pres.addSlide(); title(s, "Phase 4: the release", "O adds the privacy noise inside its own share of the decryption");
    card(s, 0.6, 1.7, 7.2, 3.35, "F7F9FC");
    latex(s, "release", 0.9, 1.9, { size: 19, maxW: 6.7, maxH: 3.0 });
    const pts = [["lock", "c₀′ never leaves O", "Without it R cannot work out c₁′s_O, so O's key share stays hidden."],
      ["shield", "η does two jobs", "It is the DP noise on the statistics, and it hides every encryption error statistically (to 2⁻⁴⁰)."],
      ["random", "Fresh every time", "New u, e₀, e₁ and η for each release; O charges the ledger before sending."]];
    pts.forEach(([ic, hd, b], i) => {
      const y = 1.72 + i * 1.12;
      circleIcon(s, ic, 8.1, y, 0.62, i === 1 ? EMBER : SLATE);
      txt(s, hd, { x: 8.9, y: y - 0.02, w: 3.8, h: 0.4, fontSize: 15, bold: true, color: INK });
      txt(s, b, { x: 8.9, y: y + 0.38, w: 3.8, h: 0.72, fontSize: 13, color: MUTED });
    });
    card(s, 0.6, 5.3, 12.1, 1.45, EMBER_L);
    txt(s, [{ text: "What R ends up with: ", options: { bold: true } },
      { text: "the statistics plus discrete-Gaussian noise of the planned size, and a c₁′ that looks random. R's view can be simulated from the noisy statistics alone, which is exactly the DP guarantee." }],
      { x: 0.9, y: 5.4, w: 11.5, h: 1.25, fontSize: 15.5, color: INK, valign: "middle" });
    s.addNotes("Phase 4, the core of the protocol. O first re-randomizes the packed ciphertext with fresh randomness u, e0, e1, using the joint public key (p0, a). It then computes its share of the decryption, c0' + c1' s_O, and adds noise eta drawn from an exact discrete Gaussian on the integers. It sends c1' and this noisy share h_O, but never c0'. R adds c1' s_R and gets the encoded statistic, plus the encryption error e, plus eta. Two points matter. Because c0' is not sent, R cannot recover c1' s_O and hence O's key share. And eta is chosen wide enough to give the planned DP noise on the statistic and also to drown the encryption error statistically; that second job is what fixes the CKKS scale later. Fresh randomness every release stops R from differencing two releases.");
  }

  {
    const s = pres.addSlide(); title(s, "Why the noise must cover the whole ciphertext");
    txt(s, "100%", { x: 0.6, y: 1.55, w: 3.4, h: 1.2, fontFace: HEAD, fontSize: 64, bold: true, color: EMBER });
    txt(s, "of trials in which R picked the true dataset out of two candidates (20 of 20, real OpenFHE), when O added the noise only to the plaintext value.",
      { x: 0.6, y: 2.8, w: 3.7, h: 1.5, fontSize: 14.5 });
    txt(s, "73%", { x: 0.6, y: 4.35, w: 3.4, h: 1.0, fontFace: HEAD, fontSize: 48, bold: true, color: SLATE });
    txt(s, "the most that ε = 1 differential privacy allows.", { x: 0.6, y: 5.35, w: 3.6, h: 0.8, fontSize: 14.5 });
    card(s, 4.7, 1.55, 3.85, 4.75); txt(s, "What went wrong", { x: 4.95, y: 1.75, w: 3.4, h: 0.45, fontFace: HEAD, fontSize: 20, bold: true, color: INK });
    bullets(s, ["Adding noise to the plaintext changes only c₀",
      "c₁ stays a deterministic function of R's ciphertexts and O's data",
      "Holding a key, R can recompute c₁ for each candidate dataset and compare"], { x: 4.95, y: 2.3, w: 3.4, h: 3.9, fontSize: 14.5 });
    card(s, 8.85, 1.55, 3.85, 4.75, EMBER_L); txt(s, "How we avoid it", { x: 9.1, y: 1.75, w: 3.4, h: 0.45, fontFace: HEAD, fontSize: 20, bold: true, color: EMBER });
    bullets(s, ["O re-randomizes c₁ before releasing, so c₁′ looks random",
      "The noise sits in O's decryption share, so it covers everything R can compute",
      "The error terms are hidden statistically by the same noise"], { x: 9.1, y: 2.3, w: 3.4, h: 3.9, fontSize: 14.5 });
    txt(s, "Lesson: DP must hold for everything a party sees, not only for the decrypted number.", { x: 0.6, y: 6.55, w: 12.1, h: 0.45, fontSize: 15, italic: true, color: INK });
    s.addNotes("A cautionary experiment that motivates the release design. In a simpler design, O added the DP noise as a plaintext value to the ciphertext, and the key holder decrypted. The decrypted value was correctly noised, but the ciphertext's c1 part did not depend on the noise at all: it was a deterministic function of R's own ciphertexts and O's data. So R could recompute c1 for each candidate dataset and compare. In real OpenFHE runs R identified the true dataset 20 times out of 20, where epsilon = 1 would allow at most 73%. The protocol avoids this by re-randomizing the ciphertext, putting the noise into O's decryption share, and making that noise wide enough to hide the encryption error too.");
  }

  {
    const s = pres.addSlide(); title(s, "Two releases: pick the ridge, then estimate", "The ridge must not depend on the noise in the release it is applied to");
    const box = (x, y, w, hh, head, body, fill, col) => {
      card(s, x, y, w, hh, fill);
      txt(s, head, { x: x + 0.2, y: y + 0.12, w: w - 0.4, h: 0.4, fontSize: 15, bold: true, color: col });
      txt(s, body, { x: x + 0.2, y: y + 0.52, w: w - 0.4, h: hh - 0.6, fontSize: 13, color: INK });
    };
    const arrow = (x, y, w) => s.addShape(pres.shapes.LINE, { x, y, w, h: 0, line: { color: MUTED, width: 2, endArrowType: "triangle" } });
    box(0.6, 1.75, 2.75, 1.6, "Stage 1 (10%)", "Release of B and C only, noise σ_1", "DCE4EF", SLATE);
    arrow(3.4, 2.55, 0.45);
    box(3.9, 1.75, 2.75, 1.6, "Ĝ_1 = A ⊕ noisy B, C", "R adds its exact block A; finds a lower bound on λ_min(G)", MIST, INK);
    arrow(6.7, 2.55, 0.45);
    box(7.2, 1.75, 2.75, 1.6, "Choose (λ, Ψ)", "Ridge just large enough that ϑ ≥ 2.5; Ψ ridges only O's block if that suffices", MIST, INK);
    arrow(10.0, 2.55, 0.45);
    box(10.5, 1.75, 2.2, 1.6, "Stage 2 (90%)", "All statistics, noise σ_2; gate ϑ̂ ≥ 1", EMBER_L, EMBER);
    latex(s, "lam_rule", 0.6, 3.75, { size: 18, maxW: 12.1 });
    latex(s, "estimate", 0.6, 4.6, { size: 18, maxW: 12.1 });
    card(s, 0.6, 5.65, 12.1, 1.1, "F7F9FC");
    txt(s, "Because (λ, Ψ) comes from stage 1 and A only, it is independent of the stage-2 noise. That keeps the error analysis and the intervals valid. ϑ measures how far the noisy Gram matrix is from singular.",
      { x: 0.85, y: 5.72, w: 11.6, h: 0.95, fontSize: 14.5, color: INK, valign: "middle" });
    s.addNotes("The noisy Gram matrix can be badly conditioned, or even singular, when n is small relative to the noise, so the protocol adds a ridge. Choosing the ridge from the same noisy release would make it depend on the noise and break the error analysis. So the protocol spends 10% of the budget on a first release of B and C only. R combines it with its exact A, computes a lower confidence bound on the smallest eigenvalue, and picks the smallest ridge that makes the validity ratio theta at least 2.5. Psi says where the ridge goes: only on O's block if that is enough, otherwise on everything. The main release then uses the remaining 90%, a gate refuses to report if the matrix still looks too close to singular, and the estimate is the ridge solution on the second release.");
  }

  {
    const s = pres.addSlide(); title(s, "Phase 5: estimates and intervals", "Everything here is post-processing at R: it costs no extra privacy budget");
    txt(s, "Two targets", { x: 0.6, y: 1.75, w: 6, h: 0.4, fontFace: HEAD, fontSize: 18, bold: true, color: INK });
    latex(s, "estimands", 0.6, 2.25, { size: 19, maxW: 6.2 });
    txt(s, "θ_λ is what the protocol estimates directly (the ridge target on the clipped data); β is the least-squares coefficient.",
      { x: 0.6, y: 2.85, w: 6.2, h: 0.75, fontSize: 13.5, color: MUTED });
    card(s, 0.6, 3.75, 6.2, 3.0, "F7F9FC");
    txt(s, "Parametric bootstrap of the noise", { x: 0.85, y: 3.88, w: 5.8, h: 0.4, fontSize: 15, bold: true, color: SLATE });
    bullets(s, ["R knows the noise distribution exactly, so it simulates many fresh noise draws around its release",
      "This gives the mechanism's covariance V_mech, and corrects the residual sum of squares for the noise",
      "Add the usual sampling variance to get the total V"], { x: 0.85, y: 4.35, w: 5.75, h: 2.35, fontSize: 13.5 });
    card(s, 7.1, 1.75, 5.6, 2.35, EMBER_L);
    txt(s, "Interval for θ_λ (Wald)", { x: 7.35, y: 1.87, w: 5.1, h: 0.4, fontSize: 15, bold: true, color: EMBER });
    latex(s, "ci_theta", 7.35, 2.45, { size: 18, maxW: 5.1 });
    txt(s, "Covers the protocol's own target at the nominal level.", { x: 7.35, y: 3.3, w: 5.1, h: 0.6, fontSize: 13, color: MUTED });
    card(s, 7.1, 4.3, 5.6, 2.45, EMBER_L);
    txt(s, "Interval for β (bias-aware)", { x: 7.35, y: 4.42, w: 5.1, h: 0.4, fontSize: 15, bold: true, color: EMBER });
    latex(s, "ci_beta", 7.35, 4.95, { size: 15, maxW: 5.1 });
    txt(s, "Widened for the worst-case ridge bias (Armstrong, Kolesár and Kwon); needs a bound on ‖β‖.", { x: 7.35, y: 5.75, w: 5.1, h: 0.9, fontSize: 13, color: MUTED });
    s.addNotes("R now has a noisy release and a known noise distribution, so everything else is post-processing and costs no privacy. There are two targets. theta_lambda is the ridge solution on the true clipped statistics, which is what the protocol estimates without bias at first order. beta is the plain least-squares coefficient; the ridge pulls the estimate away from it by a bias that is bounded but not zero. A parametric bootstrap simulates fresh noise around the release to estimate the mechanism's covariance and to correct the residual sum of squares. The Wald interval targets theta_lambda. The bias-aware interval for beta widens the Wald interval to cover the worst-case ridge bias, which needs a bound on the norm of beta, either public or derived privately from stage 1. An optional second-order bias correction exists; it is off by default because its effect is small.");
  }

  {
    const s = pres.addSlide(); title(s, "The whole protocol on one page");
    img(s, P("slides", "assets", "protocol_v31.png"), 2.3, 1.2, 8.7, 6.1, 1963, 1867);
    txt(s, "Blue arrows: R to O. Orange arrows: O to R.", { x: 0.6, y: 1.3, w: 1.9, h: 1.2, fontSize: 12.5, italic: true, color: MUTED });
    s.addNotes("A reference figure with every message of the protocol. Rows are phases, the left column is R, the right column is O, and the middle column holds public or joint objects. Use it as a map when reading the specification.");
  }

  // ======================= 4. Guarantees =======================
  divider(4, "Guarantees", "What each party learns, and what is proven")
    .addNotes("Part 4: the guarantees, the status of their proofs, and one consequence for the encryption parameters.");

  {
    const s = pres.addSlide(); title(s, "What each party learns", "Against honest-but-curious parties");
    const rows = [hdr(["", "Learns", "Does not learn", "Why"]),
      [{ text: "O", options: { bold: true, color: EMBER } }, "The public sizes n_R^max and n̄_O", "R's features, who is in the cohort, or which of its rows match", "Ring-LWE in R's key share; PSI security; O never receives a decryption share from R"],
      [{ text: "R", options: { bold: true, color: RBLUE } }, "Its own data; which of its cohort match; the DP-noised statistics", "Anything about one person's O-side data beyond (ε, δ)-DP", "Noise in O's decryption share; the release can be simulated from the noisy statistics (SIM-CDP)"]];
    s.addTable(rows, Object.assign(tableOpts([0.7, 3.4, 3.8, 4.2]), { y: 1.75, rowH: [0.45, 1.0, 1.0], fontSize: 14 }));
    card(s, 0.6, 4.6, 5.95, 2.15, "F7F9FC");
    txt(s, "Assumed", { x: 0.85, y: 4.7, w: 5.5, h: 0.4, fontSize: 15, bold: true, color: SLATE });
    bullets(s, ["Both parties follow the protocol (honest-but-curious)", "Ring-LWE; security of the PSI and of threshold key generation",
      "Private, authenticated channels"], { x: 0.85, y: 5.15, w: 5.5, h: 1.55, fontSize: 14 });
    card(s, 6.75, 4.6, 5.95, 2.15, EMBER_L);
    txt(s, "Not covered", { x: 7.0, y: 4.7, w: 5.5, h: 0.4, fontSize: 15, bold: true, color: EMBER });
    bullets(s, ["A researcher who deviates, e.g. skips clipping: needs zero-knowledge input proofs",
      "Hiding match status from R: possible with circuit-PSI, at a higher sensitivity"], { x: 7.0, y: 5.15, w: 5.5, h: 1.55, fontSize: 14 });
    s.addNotes("The guarantees in one table. O's view can be simulated from public sizes, so it learns nothing else. R's view can be simulated from the noisy statistics alone; formally, R's view is (epsilon, delta)-SIM-CDP, a computational version of differential privacy. The assumptions are honest-but-curious behaviour and standard lattice cryptography. Two things are out of scope. A researcher who deviates, for example by not clipping its inputs, could break output privacy; preventing that needs zero-knowledge proofs that the uploads are well formed. And R learns which of its cohort members are in the register; hiding that is possible with circuit-PSI, but the sensitivity then grows because match status itself becomes private.");
  }

  {
    const s = pres.addSlide(); title(s, "What is proven", "A 47-page proofs appendix: cryptography and DP (part A), sensitivity and statistics (part B)");
    const st = (t) => ({ text: t, options: { bold: true, color: t.startsWith("First") ? AMBER : TEAL } });
    const A = [hdr(["Cryptography and DP", "Status"]),
      ["Sensitivity of the encoded statistic", st("Proven")],
      ["Noise hides the encryption error (smudging), explicit constants", st("Proven")],
      ["Worst-case encryption error B_tot ≈ 2^25.3; OpenFHE measures 6–8 bits below it", st("Proven")],
      ["zCDP accounting; ρ_run = 0.0150 at ε = 1, δ = 10⁻⁹", st("Proven")],
      ["Output privacy (SIM-CDP), by a sequence of hybrids", st("Proven")],
      ["Input and cohort privacy; correctness without wraparound", st("Proven")]];
    const B = [hdr(["Sensitivity and statistics", "Status"]),
      ["Replace-one sensitivity, exact for d_O ≥ 2", st("Proven")],
      ["When yᵀy is free", st("Proven")],
      ["Stage-1 ridge choice independent of the stage-2 noise", st("Proven")],
      ["Lower bound on λ_min(G) holds with probability ≥ 1 − α_L", st("Proven")],
      ["Interval coverage (simulated: 0.944–0.962 against 0.955)", st("First order")],
      ["Bootstrap correction of the residual sum of squares", st("First order")]];
    const topts = (x) => ({ x, y: 1.75, colW: [4.6, 1.3], fontFace: BODY, fontSize: 12, color: TEXT, border: { type: "solid", color: LINE, pt: 0.75 }, valign: "middle", margin: [3, 6, 3, 6], rowH: 0.52 });
    s.addTable(A, topts(0.6)); s.addTable(B, topts(6.8));
    card(s, 0.6, 5.85, 12.1, 0.95, "F7F9FC");
    txt(s, [{ text: "Still open:  ", options: { bold: true, color: EMBER } },
      { text: "proofs for the improved circuit; composition across adaptive runs; a constant-time exact noise sampler; exact modulus sizes; heteroskedasticity when clipping binds." }],
      { x: 0.85, y: 5.95, w: 11.6, h: 0.75, fontSize: 13.5, valign: "middle" });
    s.addNotes("Status of every claim. On the cryptographic side, everything needed for the main theorem is proven, including an explicit worst-case bound on the encryption error, which we checked against OpenFHE: the measured error sits 6 to 8 bits below the bound, as expected for a worst-case bound. On the statistical side, the sensitivity, the yT y criterion, the independence of the ridge choice and the eigenvalue bound are proven; interval coverage and the bootstrap correction are proven to first order, which is standard for this kind of analysis, and simulations confirm them. The open items are listed at the bottom; none of them affects the main guarantee for a single run.");
  }

  {
    const s = pres.addSlide(); title(s, "Why the CKKS scale has to be large", "Hiding the encryption error statistically costs about 40 extra bits");
    latex(s, "smudge", 0.6, 1.75, { size: 18, maxW: 7.0 });
    latex(s, "sigma_slot", 0.6, 2.6, { size: 18, maxW: 7.0 });
    txt(s, "The privacy budget fixes the noise σ_2 seen in the statistics. So the scale Ω must be large enough that η is also 2^40 times the error:",
      { x: 0.6, y: 3.15, w: 7.0, h: 0.8, fontSize: 14.5 });
    latex(s, "scale", 0.6, 4.05, { size: 16, maxW: 7.0 });
    txt(s, "Example: ε = 1, δ = 10⁻⁹, N = 2^14, simulation bounds (σ_2 = 525).", { x: 0.6, y: 4.95, w: 7.0, h: 0.4, fontSize: 12.5, italic: true, color: MUTED });
    const rows = [hdr(["Circuit", "B_tot", "log₂ Ω"]),
      ["As specified", "2^25.3", "61.5"],
      ["Sum before the first rescale; stochastic rounding; one final rescale", "2^21.1", "57.2"],
      ["Same, with R encoding at scale 2^60", "2^17.6", "53.7"]];
    s.addTable(rows, { x: 7.95, y: 1.75, colW: [3.0, 0.9, 0.85], fontFace: BODY, fontSize: 12.5, color: TEXT, border: { type: "solid", color: LINE, pt: 0.75 }, valign: "middle", margin: [3, 5, 3, 5], rowH: 0.45 });
    card(s, 7.95, 4.25, 4.75, 2.5, EMBER_L);
    txt(s, "2^14.0–2^15.7", { x: 8.2, y: 4.35, w: 4.3, h: 0.65, fontFace: HEAD, fontSize: 28, bold: true, color: EMBER });
    txt(s, "measured error in OpenFHE for the same circuit shape, against a worst-case bound of 2^21.9. O must calibrate from the worst-case bound, never from the observed error, which depends on the data.",
      { x: 8.2, y: 5.0, w: 4.3, h: 1.7, fontSize: 13 });
    txt(s, "Larger scales mean larger moduli; exact parameter sets are still to be fixed.", { x: 0.6, y: 5.6, w: 7.0, h: 0.8, fontSize: 13.5, color: INK });
    s.addNotes("This slide explains one concrete consequence of the proofs. The noise eta must be about 2^40 times the worst-case encryption error to hide it statistically (the smudging lemma). But the noise the analyst sees in the statistics, sigma_2, is fixed by the privacy budget, and it equals sigma_c sqrt(N/2) / Omega. So Omega has to grow until both hold: about 2^61.5 for the circuit as specified, for the example. Restructuring the circuit lowers the worst-case error and brings this to 2^57.2, or 2^53.7 if R also encodes at a larger scale. OpenFHE's measured error is well below the bound, but O must calibrate from the worst-case bound: an estimate from the actual ciphertexts would depend on the data and could leak through the noise level.");
  }

  // ======================= 5. Choosing parameters =======================
  divider(5, "Choosing parameters", "From a privacy budget to noise, ridge and the sample size needed")
    .addNotes("Part 5: how to go from a privacy budget to concrete noise levels, the ridge, and the number of matched records needed. All numbers here are for the full protocol.");

  {
    const s = pres.addSlide(); title(s, "From ε to σ and λ", "Every input is public and fixed before any data are seen");
    const steps = [
      ["δ", "cal_delta", "N = people O protects. Smaller δ costs little (next slide)"],
      ["Budget", "c_rho", "L = ln(1/δ). Two releases share ρ_run"],
      ["Δ", "cal_Delta", "(R, O, Y) = squared bounds; Δ_G: B and C only"],
      ["σ", "c_sigma", "f_1 = 0.1 is the stage-1 share"],
      ["λ", "c_lambda", "Planning value; ℓ = λ_min(G)/n from reference data, else 0"],
    ];
    const gap = 2.45;
    s.addShape(pres.shapes.LINE, { x: 0.95, y: 1.95, w: gap * 4, h: 0, line: { color: LINE, width: 2 } });
    steps.forEach(([hd, f, b], i) => {
      const x = 0.6 + i * gap;
      numCircle(s, i + 1, x, 1.6, 0.7, i === 3 ? EMBER : INK, 18);
      txt(s, hd, { x: x + 0.82, y: 1.72, w: 1.5, h: 0.45, fontFace: HEAD, fontSize: 18, bold: true, color: INK });
      card(s, x, 2.5, 2.3, 2.15);
      latex(s, f, x + 0.15, 2.65, { size: 15, maxW: 2.0, maxH: 0.92 });
      txt(s, b, { x: x + 0.15, y: 3.65, w: 2.0, h: 0.95, fontSize: 11.5, color: MUTED });
    });
    const t1 = sec(PROT, "1.");
    const rows = [hdr(["σ_2 / Δ   δ \\ ε", ...t1[0].slice(1)])];
    t1.slice(1).forEach((r) => rows.push([`δ = ${r[0].replace(/^1e-0?/, "10^−")}`, ...r.slice(1)]));
    s.addTable(rows, Object.assign(tableOpts([1.3, 0.83, 0.83, 0.83, 0.83, 0.83, 0.83, 0.83]), { y: 5.0, rowH: 0.36, fontSize: 12 }));
    card(s, 7.9, 4.95, 4.8, 1.8, EMBER_L);
    txt(s, "Planning formula", { x: 8.15, y: 5.05, w: 4.4, h: 0.35, fontSize: 14, bold: true, color: EMBER });
    latex(s, "plan2", 8.15, 5.42, { size: 16, maxW: 4.4, maxH: 0.75 });
    txt(s, "Uses the simple zCDP bound, so it overstates σ_2 by 6–33%. O computes the exact σ for each release.",
      { x: 8.15, y: 6.2, w: 4.4, h: 0.5, fontSize: 11.5, color: MUTED });
    s.addNotes("The calibration chain, from experiments/calibration_protocol.py (reports/calibration/protocol.md). Pick delta from the number of people protected. Convert (epsilon, delta) to a zCDP budget with the tight conversion. Compute the sensitivities from the bounds. Split the budget between the two releases: 10% for stage 1, 90% for stage 2, which gives the two noise scales. The ridge at run time comes from stage 1; for planning, approximate the smallest eigenvalue of G by n times ell, where ell comes from public reference data, or 0 if none exists. The table gives sigma_2 per unit of sensitivity; for example at epsilon = 1 and delta = 1e-9 it is about 6 times Delta. The planning formula uses the simple conversion and is conservative.");
  }

  {
    const s = pres.addSlide(); title(s, "From σ to the matched records you need", "Two scaling laws, exact at first order: one pilot figure carries over to any ε");
    const laws = [
      ["Prediction", "law_pred", "The ridge scales with σ, so the excess test error depends on n only through n / σ. Doubling ε roughly halves the records."],
      ["Coefficients", "law_coef", "Once the ridge is off, DP error / sampling error ≈ σ²(1+‖β‖²) tr(Σ⁻²) / (n s² tr(Σ⁻¹)). Doubling ε cuts the records by about 3.5×."],
    ];
    laws.forEach(([hd, f, b], i) => {
      const x = 0.6 + i * 6.15;
      card(s, x, 1.55, 5.95, 2.05, i === 0 ? MIST : EMBER_L);
      txt(s, hd, { x: x + 0.3, y: 1.7, w: 2.6, h: 0.4, fontFace: HEAD, fontSize: 18, bold: true, color: INK });
      latex(s, f, x + 2.9, 1.74, { size: 22, w: 2.85, maxW: 2.85, align: "r" });
      txt(s, b, { x: x + 0.3, y: 2.3, w: 5.4, h: 1.2, fontSize: 13.5, color: TEXT });
    });
    const t2 = sec(CAL, "2."), t3 = sec(CAL, "3.");
    const sim = (r2, e) => t2.find((r) => r[0] === r2 && r[1] === e);
    const real = (ds, e) => t3.find((r) => r[0] === ds && r[1] === e);
    const rows = [hdr(["Target", "Setting", "First-order calculator", "Transfer (× σ or × σ²)", "Observed"])];
    [["bike", "2"], ["pol", "4"], ["protein", "4"]].forEach(([ds, e]) => {
      const r = real(ds, e);
      rows.push(["Prediction within 10%", `${ds}, ε = ${e} (from ε = 1)`, r[3], r[4], { text: r[5], options: { bold: true } }]);
    });
    [["0.5", "4"], ["0.5", "8"], ["0.3", "1"]].forEach(([r2, e]) => {
      const r = sim(r2, e);
      rows.push(["Coefficients within 2× OLS", `R² = ${r2}, ε = ${e} (from ε = 2)`, r[3], r[5], { text: r[6], options: { bold: true } }]);
    });
    s.addTable(rows, Object.assign(tableOpts([2.75, 3.5, 2.0, 2.25, 1.6]), { y: 3.85, rowH: 0.34, fontSize: 12.5 }));
    txt(s, "Checked on the experiments of part 6 (one release; prediction on real data, coefficients in simulation). The laws carry over to the protocol's σ_2. They break down for ill-conditioned designs, at high R² where clipping bias dominates, and at small n.",
      { x: 0.6, y: 6.55, w: 12.1, h: 0.65, fontSize: 11.5, italic: true, color: MUTED });
    s.addNotes("Two rules of thumb to translate a noise level into a sample size. For prediction, the excess test error depends on n only through n divided by sigma, so the records needed scale with sigma. For coefficients, once the ridge is off, the DP error relative to the sampling error scales with sigma squared over n, so the records needed scale with sigma squared. In practice: measure the records needed once, at one epsilon, on a pilot or public data, and transfer. The table checks both laws against the experiments; transferred predictions land within about 10% of what was observed, except at small n.");
  }

  {
    const s = pres.addSlide(); title(s, "What each choice costs", "Defaults: δ = 10⁻⁹, f_1 = 0.1, ϑ* = 2.5, tight bounds, one run per budget");
    const t5 = sec(PROT, "5.");
    const rename = { "bounds 2x closer to the data (no extra clipping)": "Bounds 2× closer to the data (no extra clipping)",
      "split a unit row bound (B_R^2 = B_O^2 = 1/2 instead of B_R = B_O = 1)": "Split one unit row bound between the parties",
      "double eps (eps in [0.25, 4])": "Double ε", "delta from 1e-6 to 1e-9": "δ from 10⁻⁶ to 10⁻⁹",
      "stage-1 share 0.2 instead of 0.1": "Stage-1 share 0.2 instead of 0.1",
      "one budget shared by 5 runs (zCDP: rho / 5)": "One budget shared by 5 runs (zCDP)",
      "one budget shared by 5 runs (split eps and delta)": "One budget shared by 5 runs (naive split of ε, δ)" };
    const fx = (v) => "× " + v.replace(" to ", "–");
    const rows = [hdr(["Change", "σ_2", "Records for prediction", "Records for coefficients"])];
    t5.slice(1).forEach((r) => rows.push([{ text: rename[r[0]] || r[0], options: { bold: true } }, fx(r[1]), fx(r[2]), fx(r[3])]));
    s.addTable(rows, Object.assign(tableOpts([5.9, 1.9, 2.15, 2.15]), { y: 1.65, rowH: 0.33, fontSize: 12.5 }));
    const t4 = sec(PROT, "4.");
    txt(s, "Worked example: the simulation population, δ = 10⁻⁹ (Δ = 86.2, Δ_G = 77.2, p = 11, ℓ = 0.494)", { x: 0.6, y: 4.62, w: 12.1, h: 0.4, fontFace: HEAD, fontSize: 14.5, bold: true, color: INK });
    const ex = [hdr(["ε", "ρ_run", "σ_1", "σ_2", "λ at n = 10k", "Ridge off from n", "…if ℓ were known exactly"])];
    t4.slice(1).forEach((r) => ex.push([r[0], r[1], r[2], r[3], r[5], { text: r[6], options: { bold: true } }, { text: r[7], options: { color: MUTED } }]));
    s.addTable(ex, Object.assign(tableOpts([0.9, 1.3, 1.3, 1.3, 1.9, 2.3, 3.1]), { y: 5.08, rowH: 0.3, fontSize: 12 }));
    s.addNotes("The knobs and what they buy. Tighter bounds are by far the most powerful: twice as close cuts sigma by 4 and the records needed for coefficients by 16. Splitting one row bound between the parties cuts sigma by about a third. Doubling epsilon roughly halves sigma; going from delta = 1e-6 to 1e-9 costs 20-32%. Sharing one budget over several runs is much cheaper with zCDP than with a naive split. In the worked example, note the last two columns: because the ridge is chosen from a noisy first release, it stays on until n is about three times larger than it would if the eigenvalue bound were known exactly. That is the price of not knowing the joint design in advance.");
  }

  // ======================= 6. Results =======================
  divider(6, "Results", "Accuracy in simulation and on real data, and what is still to run")
    .addNotes("Part 6: results. Read the setup slide first: it says exactly what the experiments simulate.");

  {
    const s = pres.addSlide(); title(s, "How we evaluated");
    const cards = [
      ["db", "Simulation population", "Features on real-world scales (income, age, binary indicators, counts), correlated. Public standardization constants and 99.5% bounds. R² ∈ {0.3, 0.5, 0.9}; n from 100 to 100k; 200 replicates."],
      ["book", "Real data", "29 UCI regression datasets from Wang's AdaSSP study, with his preprocessing; 10-fold cross-validation; δ = 10⁻⁶."],
      ["balance", "Comparisons", "Non-private least squares (the floor); predicting zero (the trivial baseline); AdaSSP, central DP on pooled data."],
    ];
    cards.forEach(([ic, hd, b], i) => {
      const x = 0.6 + i * 4.15;
      card(s, x, 1.45, 3.9, 2.95);
      circleIcon(s, ic, x + 0.3, 1.68, 0.7, SLATE);
      txt(s, hd, { x: x + 1.15, y: 1.76, w: 2.65, h: 0.5, fontFace: HEAD, fontSize: 17, bold: true, color: INK });
      txt(s, b, { x: x + 0.3, y: 2.55, w: 3.35, h: 1.8, fontSize: 13.5 });
    });
    const t3 = sec(PROT, "3.");
    const rng = (rows) => { const v = rows.flatMap((r) => r.slice(1).map(Number)); return `${Math.min(...v).toFixed(2)}–${Math.max(...v).toFixed(2)}`; };
    card(s, 0.6, 4.6, 12.1, 2.25, EMBER_L);
    txt(s, "What the experiments simulate", { x: 0.85, y: 4.7, w: 11.6, h: 0.4, fontSize: 15, bold: true, color: EMBER });
    bullets(s, [`One DP release of the statistics (analytic-Gaussian σ), then the ridge solve. The protocol's two releases give a σ_2 that is ${rng(t3.slice(1))}× larger at the same (ε, δ).`,
      "The ridge was set from a known eigenvalue bound in simulation, and from ℓ = 0 on real data. The protocol learns it from stage 1 instead (previous slide).",
      "The optional bias correction was applied; on real data it changes test MSE by −3.7% to +2.3%. Reruns under the full protocol are next."],
      { x: 0.85, y: 5.12, w: 11.6, h: 1.65, fontSize: 13.5, paraSpaceAfter: 4 });
    s.addNotes("The evaluation has a simulation part, where the true coefficients are known, and a real-data part on 29 datasets from Wang's AdaSSP study. The box at the bottom matters for interpreting everything that follows. The experiments simulate one DP release followed by the ridge solve. The full protocol splits the budget into two releases, so its main release is about 11-16% noisier at the same (epsilon, delta), and it learns the ridge from stage 1. In the simulation the ridge used the true eigenvalue bound, which is optimistic; on real data the experiments used no bound at all (ell = 0), so the ridge never switched off there, whereas the protocol switches it off once stage 1 shows enough signal. The experiments also applied the optional bias correction, whose effect is a few percent. A rerun under the exact protocol is the next step.");
  }

  {
    const s = pres.addSlide(); title(s, "Linkage and aggregation are exact", "A prototype of these two steps, run end to end over sockets (single key, no release step)");
    const stats = [["0", "misaligned rows", "PSI recovered exactly the true match set in all 6 test cohorts, plaintext and OpenFHE"],
      ["≈10⁻¹⁰", "max |β − plaintext answer|", "noise switched off: encryption adds no error to the regression"],
      ["≈13 s", "end-to-end on OpenFHE", "n = 400 matched out of n_O = 4,000 records (OpenFHE 1.5.1)"]];
    stats.forEach(([v, l, d], i) => {
      const x = 0.6 + i * 4.15;
      card(s, x, 1.85, 3.9, 2.35);
      txt(s, v, { x: x + 0.3, y: 1.98, w: 3.4, h: 0.85, fontFace: HEAD, fontSize: 38, bold: true, color: i === 1 ? EMBER : INK });
      txt(s, l, { x: x + 0.3, y: 2.85, w: 3.4, h: 0.4, fontSize: 15, bold: true, color: SLATE });
      txt(s, d, { x: x + 0.3, y: 3.27, w: 3.35, h: 0.85, fontSize: 13, color: MUTED });
    });
    txt(s, "Cost model of the full protocol, per run", { x: 0.6, y: 4.45, w: 8, h: 0.45, fontFace: HEAD, fontSize: 17, bold: true, color: INK });
    chip(s, "Benchmarks to be run", 10.6, 4.5, 2.1, SLATE);
    const rows = [hdr(["Step", "Communication", "O's computation", "Rounds"]),
      ["Link", "n_R^max OPRF calls; PSI query and reply", "Preprocessing linear in n̄_O; per-query evaluation", "2–3"],
      ["Upload", "(d_R + 2)·m half-size ciphertexts", "(d_R + 2)·m key conversions", "1"],
      ["Aggregate", "None", "|ℰ|·m ciphertext–plaintext products; |ℰ|·log₂(N/2) rotations", "0"],
      ["Release (×2)", "Two ring elements each", "Re-randomization, partial decryption, N exact noise samples", "1 each"]];
    s.addTable(rows, Object.assign(tableOpts([1.6, 3.5, 5.6, 1.4]), { y: 4.95, rowH: 0.34, fontSize: 12 }));
    txt(s, "m = ⌈n̄_O/(N/2)⌉ chunks; |ℰ| = number of released entries. O's work grows with the padded register n̄_O, so the public scope Φ is the main lever on cost.",
      { x: 0.6, y: 6.75, w: 12.1, h: 0.45, fontSize: 11.5, italic: true, color: MUTED });
    s.addNotes("What has been run on real cryptography so far. A prototype of the linkage and aggregation steps ran end to end over sockets, with both a plaintext backend and OpenFHE. Linkage recovered exactly the right matches with the right alignment, and with the noise switched off the regression agreed with the plaintext answer to about 1e-10. That prototype used a single CKKS key and did not include the threshold release step. The full protocol is specified and proven but not yet implemented; the table gives its cost model. The key fact for planning is that O's work is linear in the padded register size, which is why the study scope matters.");
  }

  {
    const s = pres.addSlide(); title(s, "Accuracy against intersection size", "Simulation, R² = 0.5, one release per run");
    img(s, P("reports", "coef_accuracy", "figA_rmse_vs_n_r2_0.5.png"), 0.4, 1.55, 8.5, 5.75, 1904, 1224);
    card(s, 9.1, 1.6, 3.65, 5.2);
    txt(s, "Within 2× of non-private least squares at", { x: 9.35, y: 1.8, w: 3.2, h: 0.7, fontSize: 15, bold: true, color: INK });
    const cross = [["ε = 8", "n ≈ 3.2k"], ["ε = 4", "n ≈ 9.4k"], ["ε = 2", "n ≈ 34k"], ["ε = 1", "n ≈ 105k"]];
    cross.forEach(([e, n], i) => {
      txt(s, e, { x: 9.35, y: 2.55 + i * 0.5, w: 1.2, h: 0.45, fontSize: 15, color: MUTED });
      txt(s, n, { x: 10.6, y: 2.55 + i * 0.5, w: 2.0, h: 0.45, fontSize: 15, bold: true, color: EMBER });
    });
    bullets(s, ["Small n: the ridge pulls the estimate towards 0", "Then the DP error falls like 1/n and merges into the sampling floor",
      "Run-to-run spread is about ×2.5–3"], { x: 9.35, y: 4.65, w: 3.2, h: 2.1, fontSize: 13.5 });
    s.addNotes("Median error of the standardized slopes over 200 runs, each with new data and a new DP release, against the non-private floor. At small n the ridge dominates and the estimate is pulled towards zero (the dashed line). Once the ridge switches off, the DP error falls like 1/n, faster than the sampling error's 1/sqrt(n), and the curves merge into the floor. The crossings on the right are where the private estimate comes within twice the non-private error. Remember the setup caveats: one release, ridge from a known eigenvalue bound, so the full protocol needs somewhat more data.");
  }

  {
    const s = pres.addSlide(); title(s, "Where the error comes from when σ is large");
    img(s, P("reports", "bias_correction", "fig2_decomposition.png"), 0.4, 1.25, 12.5, 3.15, 2785, 700);
    card(s, 0.6, 4.55, 5.95, 2.25);
    txt(s, "Error anatomy", { x: 0.85, y: 4.68, w: 5.5, h: 0.4, fontSize: 16, bold: true, color: INK });
    bullets(s, ["Heavy ridge: 72–90% of the error is shrinkage, 11–31% noise variance", "Bias from the noise itself: at most 0.75% of the error",
      "At large n, clipping shifts the target slightly (any DP method pays this)"], { x: 0.85, y: 5.12, w: 5.5, h: 1.6, fontSize: 13.5, paraSpaceAfter: 4 });
    card(s, 6.75, 4.55, 5.95, 2.25, EMBER_L);
    txt(s, "So what?", { x: 7.0, y: 4.68, w: 5.5, h: 0.4, fontSize: 16, bold: true, color: EMBER });
    bullets(s, ["The ridge, not the noise, limits accuracy at small n: tighter bounds and more matched records help most",
      "A second-order bias correction exists but changes little, so it is optional and off by default"],
      { x: 7.0, y: 5.12, w: 5.5, h: 1.6, fontSize: 13.5, paraSpaceAfter: 4 });
    s.addNotes("The error split into its parts, for five privacy levels, as the validity ratio rho_0 grows (left to right means more data relative to noise). When sigma is large relative to the data, the ridge's shrinkage dominates the error. The bias that the noise itself causes, through inverting a noisy matrix, is tiny. That is why the optional bias correction barely matters, and why the protocol leaves it off by default. The practical message: at small n, accuracy is limited by the ridge, so tighter bounds and more matched records are what help.");
  }

  {
    const s = pres.addSlide(); title(s, "Intervals that hold up", "Simulation of the release and the solve step, 2,000 replications per scenario");
    const rows = [hdr(["Scenario", "n", "Ridge", "Wald interval for θ_λ (target 0.955)", "Bias-aware interval for β (target ≥ 0.95)"]),
      ["Ridge off", "4,000", "λ = 0", "0.946–0.958", "0.945–0.968"],
      ["Ridge on, whole matrix", "1,500", "Ψ = I, median λ = 477", "0.948–0.960", "1.000 with a public bound on ‖β‖; unbounded without"],
      ["Ridge on, O's block only", "3,000", "Ψ = Π_O, median λ = 439", "0.944–0.962", "unbounded without a public bound"],
      ["Small n, public bound", "200", "Ψ = I, median λ = 1,021", "0.946–0.962", "1.000"]];
    s.addTable(rows, Object.assign(tableOpts([2.6, 0.9, 2.5, 2.9, 3.2]), { y: 1.75, rowH: 0.55, fontSize: 13 }));
    card(s, 0.6, 4.8, 5.95, 1.95, "F7F9FC");
    txt(s, "Reading the table", { x: 0.85, y: 4.92, w: 5.5, h: 0.4, fontSize: 15, bold: true, color: SLATE });
    bullets(s, ["Ranges are over the 11 coefficients", "The Wald interval hits its target with or without the ridge",
      "The β interval is conservative: worst-case bias, and a cautious variance"], { x: 0.85, y: 5.35, w: 5.5, h: 1.35, fontSize: 13.5 });
    card(s, 6.75, 4.8, 5.95, 1.95, EMBER_L);
    txt(s, "For contrast", { x: 7.0, y: 4.92, w: 5.5, h: 0.4, fontSize: 15, bold: true, color: EMBER });
    txt(s, "A naive Wald interval around a ridge estimate, read as an interval for β, covered only 42–81% when no eigenvalue bound was known. Naming the target matters.",
      { x: 7.0, y: 5.35, w: 5.5, h: 1.35, fontSize: 13.5 });
    s.addNotes("Coverage of the two intervals, from a simulation of the protocol's release and solve-and-infer step (stage 1, ridge choice, stage 2, bootstrap, intervals), 2,000 replications per scenario. The Wald interval for theta_lambda reaches its 95.5% target in every scenario, including with an active ridge. The bias-aware interval for beta always covers, but it is conservative, and without a public bound on the norm of beta it can be unbounded when the ridge is on. For contrast, in an earlier study a naive interval around a ridge estimate, interpreted as covering beta, covered only 42 to 81%: the ridge bias is not small. The lesson is to be explicit about which target an interval is for.");
  }

  {
    const s = pres.addSlide(); title(s, "Real data against central DP", "Test MSE ÷ non-private at ε = 1, δ = 10⁻⁶, 10-fold cross-validation");
    const rows = [hdr(["Dataset", "n", "d", "Predict zero", "AdaSSP (pooled data)", "This protocol"])];
    rows[0][5].options.fill = { color: EMBER };
    ["housing", "wine", "airfoil", "concrete", "bike", "elevators", "pol", "kin40k", "protein"].forEach((ds) => {
      const np = adaGet(ADA9, ds, 1, "nonprivate");
      const r = (m) => Number(adaGet(ADA9, ds, 1, m).ratio_to_nonprivate);
      const a = r("adassp").toFixed(2), v = r("vfl").toFixed(2);       // bold only a strict win at the precision shown
      rows.push([ds, Number(np.n).toLocaleString("en-US"), np.d, { text: `${r("trivial").toFixed(2)}×`, options: { color: MUTED } },
        { text: `${a}×`, options: +a < +v ? { bold: true } : {} }, { text: `${v}×`, options: +v < +a ? { bold: true } : {} }]);
    });
    s.addTable(rows, Object.assign(tableOpts([1.8, 1.2, 0.7, 1.7, 2.3, 2.0]), { y: 1.75, rowH: 0.4, fontSize: 13 }));
    card(s, 10.5, 1.75, 2.2, 1.85, EMBER_L);
    txt(s, "1.0× = as good as non-private; lower is better; bold = better private method.", { x: 10.65, y: 1.88, w: 1.9, h: 1.6, fontSize: 12.5, color: INK });
    txt(s, "AdaSSP needs one curator with the pooled data, and protects the whole record. At nominal ε = 1 it is actually more private (ε ≈ 0.82), so the comparison favours it. Here the protocol's R holds about half the features.",
      { x: 0.6, y: 5.95, w: 12.1, h: 0.85, fontSize: 12.5, italic: true, color: MUTED });
    s.addNotes("Nine of the 29 datasets, at epsilon = 1. Each number is test MSE divided by the non-private MSE, so 1.00 means no loss. Predicting zero is the floor any useful method must beat. AdaSSP is the standard central DP regression; it needs all the data pooled with one trusted curator, so it is not available in our setting, but it shows what a central method achieves. Its guarantee is also stronger, and at the same nominal epsilon it is in fact more private than our release, so this comparison is conservative for us. The protocol is close to AdaSSP on the large datasets and somewhat worse on the small ones.");
  }

  {
    const s = pres.addSlide(); title(s, "Across privacy levels", "Test MSE ÷ non-private on three datasets of increasing size; bars: 95% CI over the 10 folds");
    const EPS = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10];
    const SER = [["trivial", "Predict zero", "B8BEC8"], ["adassp", "AdaSSP (pooled data)", "1BAF7A"], ["vfl", "This protocol", "2A78D6"]];
    ["housing", "elevators", "protein"].forEach((ds, k) => {
      const n = Number(adaGet(ADA9, ds, 1, "vfl").n).toLocaleString("en-US");
      const data = SER.map(([m, name]) => ({ name, labels: EPS.map(String), values: EPS.map((e) => Number(adaGet(ADA9, ds, e, m).ratio_to_nonprivate)) }));
      const ctitle = `${ds}  (n = ${n})`;
      ERRBARS[ctitle] = Object.fromEntries(SER.filter(([m]) => m !== "trivial").map(([m, name]) => [name, EPS.map((e) => {
        const r = adaGet(ADA9, ds, e, m);
        return (Number(r.ci95_hi) - Number(r.test_mse)) / Number(adaGet(ADA9, ds, e, "nonprivate").test_mse);
      })]));
      s.addChart(pres.charts.LINE, data, {
        x: 0.6 + k * 4.1, y: 1.6, w: 3.95, h: 3.3,
        chartColors: SER.map((r) => r[2]), lineSize: 2, lineDataSymbol: "circle", lineDataSymbolSize: 5,
        showTitle: true, title: ctitle, titleFontFace: HEAD, titleFontSize: 13, titleColor: INK,
        showLegend: false, valAxisMinVal: 1, valAxisLabelFormatCode: "0.0",
        showValAxisTitle: k === 0, valAxisTitle: "MSE ÷ non-private", valAxisTitleFontSize: 10, valAxisTitleColor: MUTED,
        showCatAxisTitle: true, catAxisTitle: "ε", catAxisTitleFontSize: 10, catAxisTitleColor: MUTED,
        catAxisLabelFontSize: 9, valAxisLabelFontSize: 9, catAxisLabelColor: MUTED, valAxisLabelColor: MUTED,
        valGridLine: { color: "E5E7EB", size: 0.5 }, catGridLine: { style: "none" },
      });
    });
    SER.forEach(([, name, col], i) => {
      const x = 3.3 + i * 2.6;
      s.addShape(pres.shapes.LINE, { x, y: 5.12, w: 0.4, h: 0, line: { color: col, width: 2.5 } });
      txt(s, name, { x: x + 0.5, y: 4.98, w: 2.0, h: 0.3, fontSize: 12, color: TEXT });
    });
    const N = new Set(ADA_CMP.map((r) => r.dataset)).size;
    const E3 = [0.1, 1, 10];
    const verdicts = (e) => {
      const v = ADA_CMP.filter((r) => r.vs === "adassp" && Number(r.eps) === e).map((r) => r.verdict);
      return `${v.filter((x) => x === "better").length} / ${v.filter((x) => x === "worse").length}`;
    };
    const loseToZero = (m, e) => {
      const ds = [...new Set(ADA_ALL.map((r) => r.dataset))];
      return ds.filter((d) => Number(adaGet(ADA_ALL, d, e, m).test_mse) > Number(adaGet(ADA_ALL, d, e, "trivial").test_mse)).length;
    };
    const trows = [hdr([`Of ${N} datasets`, "ε = 0.1", "ε = 1", "ε = 10"]),
      ["Protocol better / worse than AdaSSP", ...E3.map(verdicts)],
      ["Protocol worse than predicting zero", ...E3.map((e) => String(loseToZero("vfl", e)))],
      ["AdaSSP worse than predicting zero", ...E3.map((e) => String(loseToZero("adassp", e)))]];
    s.addTable(trows, Object.assign(tableOpts([3.3, 1.0, 1.0, 1.0]), { y: 5.4, rowH: 0.34, fontSize: 12.5 }));
    txt(s, "Better / worse: paired over folds, 95% CI; the rest are ties.", { x: 0.6, y: 6.85, w: 6.3, h: 0.3, fontSize: 11, italic: true, color: MUTED });
    card(s, 7.3, 5.4, 5.4, 1.55, EMBER_L);
    txt(s, "From ε ≈ 0.5 the central method is ahead, by a median 4% at ε = 1, and the gap shrinks as n grows. At very small ε both are often no better than predicting zero.",
      { x: 7.5, y: 5.5, w: 5.0, h: 1.35, fontSize: 13.5, color: INK, valign: "middle" });
    s.addNotes("Native charts from reports/adassp/results.csv; verdicts over all 29 datasets from reports/adassp/all/compare.csv. Each curve is test MSE relative to non-private, as epsilon goes from 0.01 to 10. At very small epsilon both private methods are often worse than predicting zero, so wins there mean little. From epsilon around 0.5, central AdaSSP is ahead, and the gap shrinks with more data: on protein, with 45k rows, the two are almost identical. That gap is the price of not pooling, plus our looser per-party bounds.");
  }

  {
    const SWF = P("reports", "adassp", "sweep", "fig_sweep_n_deck.png");
    const hdrPng = fs.readFileSync(SWF);
    const [FIGW, FIGH] = [hdrPng.readUInt32BE(16), hdrPng.readUInt32BE(20)];
    const s = pres.addSlide(); title(s, "How many matched records on real data?", "Test MSE ÷ non-private at the same n; nested subsets of the training folds; bars: 95% CI over folds");
    img(s, SWF, 0.4, 1.6, 7.4, 5.75, FIGW, FIGH);
    txt(s, "Matched records to come within 10% of non-private", { x: 8.05, y: 1.65, w: 4.7, h: 0.7, fontFace: HEAD, fontSize: 16, bold: true, color: INK });
    txt(s, "This protocol (AdaSSP, pooled data)", { x: 8.05, y: 2.35, w: 4.7, h: 0.3, fontSize: 12, italic: true, color: MUTED });
    const EPS4 = [0.5, 1, 2, 4];
    const nfmt = (v) => (v < 1000 ? String(Math.round(v)) : v < 10000 ? (v / 1000).toFixed(1) + "k" : Math.round(v / 1000) + "k");
    const cross = (ds, e, m) => {
      const r = SWEEP_X.find((x) => x.dataset === ds && Number(x.threshold) === 1.1 && Number(x.eps) === e && x.method === m);
      return r.n_cross === "" ? `>${Math.floor(Number(r.n_max) / 1000)}k` : nfmt(Number(r.n_cross));
    };
    const rows = [hdr(["", ...EPS4.map((e) => `ε = ${e}`)])];
    ["bike", "elevators", "pol", "protein"].forEach((ds) =>
      rows.push([ds, ...EPS4.map((e) => ({ text: [{ text: cross(ds, e, "vfl"), options: { bold: true, breakLine: true } },
        { text: `(${cross(ds, e, "adassp")})`, options: { color: MUTED, fontSize: 11 } }] }))]));
    s.addTable(rows, { x: 8.05, y: 2.7, colW: [1.1, 0.9, 0.9, 0.9, 0.9], fontFace: BODY, fontSize: 12.5, color: TEXT,
      border: { type: "solid", color: LINE, pt: 0.75 }, valign: "middle", align: "center", margin: [2, 3, 2, 3], rowH: [0.36, 0.56, 0.56, 0.56, 0.56] });
    bullets(s, ["Doubling ε roughly halves the records needed",
      "About 1.5–2× the records the central method needs",
      "elevators never gets within 10% at the sizes available (up to 14.9k)"], { x: 8.05, y: 5.5, w: 4.7, h: 1.5, fontSize: 13.5 });
    s.addNotes("The real-data counterpart of the simulation curves. For four datasets with at least 15k rows and a real linear signal, we took nested subsets of the training folds as the matched set and measured test error. The table gives the number of matched records needed to come within 10% of non-private. Doubling epsilon roughly halves it, as the scaling law predicts, and the protocol needs about 1.5 to 2 times the records of the central method. 'Within 10%' is not equally hard on every dataset: on protein, predicting zero is already only 1.4 times worse than non-private.");
  }

  // ======================= Limits and summary =======================
  {
    const s = pres.addSlide(); title(s, "Limits and open items");
    card(s, 0.6, 1.55, 5.95, 5.2);
    circleIcon(s, "warn", 0.9, 1.8, 0.7, SLATE);
    txt(s, "By design", { x: 1.8, y: 1.88, w: 4.5, h: 0.5, fontFace: HEAD, fontSize: 19, bold: true, color: INK });
    bullets(s, ["Honest-but-curious parties only; a deviating researcher needs zero-knowledge input proofs",
      "R learns which of its cohort are in the register",
      "Exact-identifier linkage; linear models; model-based variance",
      "O's work grows with the padded register, so the scope Φ matters"], { x: 0.95, y: 2.6, w: 5.35, h: 4.0, fontSize: 14.5 });
    card(s, 6.75, 1.55, 5.95, 5.2, EMBER_L);
    circleIcon(s, "hourglass", 7.05, 1.8, 0.7, EMBER);
    txt(s, "Still to do", { x: 7.95, y: 1.88, w: 4.5, h: 0.5, fontFace: HEAD, fontSize: 19, bold: true, color: INK });
    bullets(s, ["Implement the full protocol in OpenFHE and measure its cost",
      "Rerun the experiments under the protocol's two-release rule",
      "Fix exact parameter sets for the larger CKKS scale",
      "Close the open proof items: improved circuit, adaptive composition, constant-time sampler",
      "A fairer real-data comparison: preprocessing each party can do on its own"], { x: 7.1, y: 2.6, w: 5.35, h: 4.0, fontSize: 14.5 });
    s.addNotes("Limits by design: the guarantee assumes both parties follow the protocol; R learns match status; linkage is on exact identifiers; the model is linear with a model-based variance; and O's cost is linear in its padded register. Still to do: implement and benchmark the full protocol, rerun the experiments under its exact two-release rule, fix concrete encryption parameters for the larger scale, close the remaining proof items, and redo the real-data comparison with preprocessing that each party can do on its own data, since Wang's pipeline normalizes whole rows across both parties' features.");
  }

  {
    const s = pres.addSlide(); s.background = { color: INK };
    s.addText("Summary", { x: 0.6, y: 0.45, w: 12, h: 0.8, fontFace: HEAD, fontSize: 34, bold: true, color: WHITE, margin: 0, isTextBox: true });
    const take = ["A researcher and an agency can fit a joint regression without pooling: linkage by labeled PSI, sums under split-key encryption, DP noise added inside O's decryption share",
      "O learns only public sizes; R gets (ε, δ)-DP statistics. The main guarantees are proven, for honest-but-curious parties",
      "Tight public bounds and enough matched records drive accuracy: within 2× of non-private at 3k–105k records in simulation, within 10% at about 1.4k–11k on real data for ε = 1–2",
      "Close to central DP on large datasets; about 1.5–2× the records on real data"];
    take.forEach((t, i) => {
      numCircle(s, i + 1, 0.6, 1.55 + i * 1.32, 0.6, EMBER, 17);
      s.addText(t, { x: 1.45, y: 1.55 + i * 1.32, w: 6.0, h: 1.2, fontFace: BODY, fontSize: 15, color: WHITE, margin: 0, valign: "top", isTextBox: true });
    });
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 7.85, y: 1.5, w: 4.85, h: 5.2, fill: { color: "1F3050" }, line: { color: "1F3050" }, rectRadius: 0.08 });
    s.addText("To read next", { x: 8.15, y: 1.7, w: 4.3, h: 0.5, fontFace: HEAD, fontSize: 22, bold: true, color: "E7A77C", margin: 0, isTextBox: true });
    const nxt = ["The protocol specification (algorithms 1–7)", "The proofs appendix: parts A and B",
      "reports/calibration: from ε to σ, λ and sample size", "reports/adassp: the real-data benchmark and sweep"];
    s.addText(nxt.map((t, i) => ({ text: t, options: { bullet: true, breakLine: i < nxt.length - 1, paraSpaceAfter: 10 } })),
      { x: 8.15, y: 2.35, w: 4.3, h: 4.2, fontFace: BODY, fontSize: 15, color: WHITE, margin: 0, valign: "top", isTextBox: true });
    s.addNotes("Four things to remember. The protocol combines three standard tools so that a researcher and an agency can fit a regression without pooling. Its guarantees are proven for honest-but-curious parties. Accuracy is driven mostly by tight bounds and the number of matched records. And compared with central DP on pooled data, the price of not pooling is moderate on large datasets. The documents on the right go deeper.");
  }

  await writeDeck(pres, P("slides", "vfl_intro_deck.pptx"), ERRBARS);
})();
