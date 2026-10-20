// Build the worked-example deck: the protocol of the introduction deck, step by step, on a toy dataset
// (R: 3 records x 2 features; O: 5 records x 2 features and y). Every number comes from
// reports/toy_example/toy.json, written by experiments/toy_example.py.
// Requires: npm install pptxgenjs jszip react-icons react react-dom sharp
// Run from the repo root:  node slides/build_toy_deck.js
const fs = require("fs");
const fa = require("react-icons/fa");
const { P, C, HEAD, BODY, newDeck, writeDeck } = require("./deck_lib");
const { INK, SLATE, MIST, LINE, EMBER, EMBER_L, TEAL, AMBER, TEXT, MUTED, WHITE } = C;
const RBLUE = "2A78D6", RBLUE_L = "E3EEFB", BLUE_BLK = "C9DCF5", O_BLK = "F3D2BC", HILITE = "FFF4C2";

const T = JSON.parse(fs.readFileSync(P("reports", "toy_example", "toy.json"), "utf8"));
const num = (v, d = 2) => (v < 0 && Math.abs(v) >= 0.5 * 10 ** -d ? "−" : "") + Math.abs(v).toFixed(d);
const int = (v) => (v < 0 ? "−" : "") + Math.abs(v).toLocaleString("en-US");
const NAMES = ["intercept", "education", "health", "age", "benefits"];

(async () => {
  const { pres, icons, h } = await newDeck("A worked example of the protocol", {
    user: fa.FaUserGraduate, bank: fa.FaLandmark, lock: fa.FaLock, key: fa.FaKey, eye: fa.FaEye,
    search: fa.FaSearch, layers: fa.FaLayerGroup, check: fa.FaCheckCircle, warn: fa.FaExclamationTriangle,
    shield: fa.FaShieldAlt, users: fa.FaUsers, chart: fa.FaChartLine, ruler: fa.FaRulerCombined,
  });
  const { title, circleIcon, numCircle, card, txt, bullets, chip, tableOpts, hdr } = h;
  const phase = (s, label, col = INK) => chip(s, label, 10.7, 0.45, 2.0, col);
  // a small matrix as a table: rows of numbers, optional row/column labels and per-cell fills
  const mat = (s, M, x, y, { cw = 0.75, rh = 0.32, fs = 12, d = 2, rowLab, colLab, fill, labW = 1.1 } = {}) => {
    const rows = [];
    if (colLab) rows.push([...(rowLab ? [""] : []), ...colLab.map((c) => ({ text: c, options: { bold: true, color: MUTED, fontSize: fs - 1 } }))]);
    M.forEach((r, i) => rows.push([...(rowLab ? [{ text: rowLab[i], options: { bold: true, color: MUTED, fontSize: fs - 1 } }] : []),
      ...r.map((v, j) => {
        const f = fill ? fill(i, j) : null;
        return { text: typeof v === "string" ? v : num(v, d), options: Object.assign({ align: "center" }, f ? { fill: { color: f[0] }, color: f[1] || TEXT } : {}) };
      })]));
    s.addTable(rows, { x, y, colW: [...(rowLab ? [labW] : []), ...M[0].map(() => cw)], rowH: rh, fontFace: BODY, fontSize: fs, color: TEXT,
      border: { type: "solid", color: LINE, pt: 0.75 }, valign: "middle", margin: [1, 3, 1, 3] });
  };
  const gFill = (i, j) => (i < 3 && j < 3 ? [BLUE_BLK] : i >= 3 && j >= 3 ? [O_BLK] : [EMBER, WHITE]);
  const S1 = T.stage1, S2 = T.stage2, SC = T.scale, TOY = T.toy;
  const matched = (id) => T.I.includes(id);

  // ======================= Title =======================
  {
    const s = pres.addSlide(); s.background = { color: INK };
    s.addText("A worked example of the protocol", { x: 0.8, y: 1.5, w: 7.5, h: 1.6, fontFace: HEAD, fontSize: 38, bold: true, color: WHITE, margin: 0, valign: "top", isTextBox: true });
    s.addText("Three cohort records, five register records: every step of the protocol with its numbers", { x: 0.8, y: 3.2, w: 7.2, h: 0.9,
      fontFace: BODY, fontSize: 18, color: "CAD5E6", margin: 0, valign: "top", isTextBox: true });
    s.addText("Companion to the introduction deck  ·  October 2026", { x: 0.8, y: 6.3, w: 7, h: 0.4, fontFace: BODY, fontSize: 13, color: "8FA6C4", margin: 0, isTextBox: true });
    // a 3-row and a 5-row table, two rows linked
    for (let i = 0; i < 3; i++) s.addShape(pres.shapes.RECTANGLE, { x: 8.9, y: 2.3 + i * 0.55, w: 1.2, h: 0.45, fill: { color: i < 2 ? RBLUE : "3D5A80" }, line: { color: INK } });
    for (let k = 0; k < 5; k++) s.addShape(pres.shapes.RECTANGLE, { x: 11.0, y: 1.75 + k * 0.55, w: 1.5, h: 0.45, fill: { color: [1, 3].includes(k) ? EMBER : "7A4A2C" }, line: { color: INK } });
    [[0, 1], [1, 3]].forEach(([i, k]) => s.addShape(pres.shapes.LINE, { x: 10.1, y: 2.52 + i * 0.55, w: 0.9, h: (1.97 + k * 0.55) - (2.52 + i * 0.55), line: { color: "E7A77C", width: 1.5, dashType: "dash" } }));
    s.addText("R", { x: 8.9, y: 4.05, w: 1.2, h: 0.5, align: "center", fontFace: HEAD, fontSize: 22, bold: true, color: WHITE, margin: 0, isTextBox: true });
    s.addText("O", { x: 11.0, y: 4.55, w: 1.5, h: 0.5, align: "center", fontFace: HEAD, fontSize: 22, bold: true, color: WHITE, margin: 0, isTextBox: true });
    s.addNotes("A companion to the introduction deck. We run the protocol by hand on a toy dataset: a researcher R with three cohort records and two features, and an organization O with five register records, two features and the outcome. Every number on these slides is produced by experiments/toy_example.py, so the example can be rerun and checked.");
  }

  // ======================= How to read =======================
  {
    const s = pres.addSlide(); title(s, "How to read this example");
    const cards = [
      ["layers", "Real steps, toy sizes", "Every phase of the protocol, in order, with the numbers each step produces.", SLATE],
      ["eye", "Shown in the clear", "We print values the protocol keeps encrypted, so you can check each step. A party only ever sees what its own boxes say.", RBLUE],
      ["warn", "Too small to be useful", "Two people match and the model has five coefficients: least squares has no unique answer, and the privacy noise dwarfs the data. Watch how the protocol fails safe; the last slides repeat the steps with 20,000 people.", EMBER],
    ];
    cards.forEach(([ic, hd, b, col], i) => {
      const x = 0.6 + i * 4.15;
      card(s, x, 1.55, 3.9, 3.55);
      circleIcon(s, ic, x + 0.3, 1.8, 0.75, col);
      txt(s, hd, { x: x + 0.3, y: 2.75, w: 3.35, h: 0.5, fontFace: HEAD, fontSize: 19, bold: true, color: INK });
      txt(s, b, { x: x + 0.3, y: 3.3, w: 3.35, h: 1.75, fontSize: 14 });
    });
    card(s, 0.6, 5.4, 12.1, 1.35, "F7F9FC");
    txt(s, "Notation", { x: 0.85, y: 5.5, w: 3, h: 0.35, fontSize: 14, bold: true, color: SLATE });
    txt(s, "I: R's cohort members found in O's register.   π: the bijection sending each i ∈ I to its row in O's register.   τ: O's secret shuffle of its rows.   Budget: ε = 1, δ = 10⁻⁹.",
      { x: 0.85, y: 5.88, w: 11.6, h: 0.75, fontSize: 14 });
    s.addNotes("Three things to keep in mind. The steps are the real protocol steps, applied to tiny tables. We show values that the protocol keeps encrypted, so that each step can be checked against the answer computed in the clear; each party only ever sees what its own boxes on the slides say. And the example is deliberately too small to be useful: with two matched people and five coefficients, least squares has no unique solution, and at epsilon = 1 the privacy noise is about 30 times the size of the statistics. That is the honest situation with two people, and it shows how the protocol degrades: it shrinks the estimate towards zero instead of leaking. The last slides run the same steps on 20,000 people.");
  }

  // ======================= The data =======================
  {
    const s = pres.addSlide(); title(s, "The data", "Already scaled so that every value lies in [−1, 1]");
    card(s, 0.6, 1.75, 5.2, 3.4, RBLUE_L);
    circleIcon(s, "user", 0.85, 1.95, 0.6, RBLUE);
    txt(s, "Researcher R: 3 records × 2 features", { x: 1.6, y: 2.0, w: 4.1, h: 0.45, fontFace: HEAD, fontSize: 16, bold: true, color: INK });
    const Rrows = [hdr(["id", "education x_R1", "health x_R2"]), ...T.R_rows.map(([id, a, b]) => [id, num(a, 1), num(b, 1)])];
    s.addTable(Rrows, { x: 0.85, y: 2.75, colW: [0.9, 1.8, 1.9], rowH: 0.4, fontFace: BODY, fontSize: 14, color: TEXT, border: { type: "solid", color: LINE, pt: 0.75 }, valign: "middle", align: "center" });
    card(s, 6.1, 1.75, 6.6, 3.4, EMBER_L);
    circleIcon(s, "bank", 6.35, 1.95, 0.6, EMBER);
    txt(s, "Organization O: 5 records × 2 features, plus y", { x: 7.1, y: 2.0, w: 5.5, h: 0.45, fontFace: HEAD, fontSize: 16, bold: true, color: INK });
    const Orows = [hdr(["id", "age x_O1", "benefits x_O2", "income y"]), ...T.O_rows.map(([id, a, b, y]) => [id, num(a, 1), num(b, 1), num(y, 1)])];
    s.addTable(Orows, { x: 6.35, y: 2.75, colW: [0.9, 1.55, 1.85, 1.75], rowH: 0.4, fontFace: BODY, fontSize: 14, color: TEXT, border: { type: "solid", color: LINE, pt: 0.75 }, valign: "middle", align: "center" });
    card(s, 0.6, 5.45, 12.1, 1.3, "F7F9FC");
    txt(s, [{ text: "Goal: ", options: { bold: true } },
      { text: "regress income on education, health, age and benefits, plus an intercept, over the people in both tables. Neither party knows yet who that is: identifiers 104 and 221 appear in both, 317 only in R's cohort." }],
      { x: 0.85, y: 5.55, w: 11.6, h: 1.1, fontSize: 15, valign: "middle" });
    s.addNotes("The researcher holds three records with education and a health score; the organization holds five register records with age, a benefits indicator and income, the outcome. All values are already scaled to [-1, 1], which will also serve as the public clipping bounds. The regression has five coefficients: an intercept and R's two features (R's columns), plus O's two features. Identifiers 104 and 221 are in both tables; 317 is only in R's cohort. Neither party knows this yet: finding it out privately is phase 1.");
  }

  // ======================= The answer in the clear =======================
  {
    const s = pres.addSlide(); title(s, "The answer we are after", "Computed here by pooling the data, only to check the protocol's steps against it");
    txt(s, `Matched set I = {${T.I.join(", ")}}, so n = ${T.I.length}. Rows of Z = [X_R^I | X_O^I] and y^I:`, { x: 0.6, y: 1.7, w: 8.2, h: 0.45, fontSize: 15 });
    mat(s, T.Z.map((r, i) => [...r, T.yI[i]]), 0.6, 2.25, { rowLab: T.I, colLab: [...NAMES, "income"], cw: 1.08, labW: 0.6, d: 1, fs: 12.5,
      fill: (i, j) => (j < 3 ? [BLUE_BLK] : j < 5 ? [O_BLK] : [O_BLK]) });
    txt(s, "Gram matrix G^I = ZᵀZ", { x: 0.6, y: 3.45, w: 5, h: 0.4, fontFace: HEAD, fontSize: 16, bold: true, color: INK });
    mat(s, T.G, 0.6, 3.9, { rowLab: NAMES, colLab: NAMES, cw: 1.08, labW: 1.05, fs: 12, fill: gFill });
    txt(s, "c^I = Zᵀy^I", { x: 7.65, y: 3.45, w: 2.5, h: 0.4, fontFace: HEAD, fontSize: 16, bold: true, color: INK });
    mat(s, T.c.map((v, i) => [v]), 7.65, 3.9, { colLab: ["c"], cw: 0.9, fs: 12, fill: (i) => (i < 3 ? [EMBER, WHITE] : [O_BLK]) });
    txt(s, `yᵀy = ${num(T.yy)}`, { x: 7.65, y: 6.0, w: 2.5, h: 0.4, fontSize: 14, bold: true });
    card(s, 9.1, 2.25, 3.6, 4.5, EMBER_L);
    txt(s, "Read the colours as in the introduction", { x: 9.3, y: 2.37, w: 3.2, h: 0.6, fontSize: 14, bold: true, color: EMBER });
    bullets(s, ["Blue A^I: R's rows only", "Light orange B^I, c_O^I: O's matched rows", "Orange C^I, c_R^I: pairs R's row i with O's row π(i)",
      { text: `G has rank ${T.rankG} < 5: with two people, least squares has no unique solution. The ridge will matter.`, options: { bold: true } }],
      { x: 9.3, y: 3.05, w: 3.25, h: 3.6, fontSize: 13.5 });
    s.addNotes("This is what pooling would give, and what the protocol must approximate without pooling. Two people match, so Z has two rows and five columns, and G = ZT Z is a 5 by 5 matrix of rank 2. Least squares has no unique solution, so a ridge is unavoidable here; in real use n is in the thousands. The colours are those of the introduction deck: R's own block in blue, O's matched-row block in light orange, and the cross-party blocks, which depend on how R's rows pair with O's rows, in dark orange. We will see each of these numbers again, computed by the protocol.");
  }

  // ======================= Phase 0 =======================
  {
    const s = pres.addSlide(); title(s, "Phase 0: setup", "Everything that sets the noise is public and fixed before any data are touched"); phase(s, "Phase 0 · R + O", SLATE);
    card(s, 0.6, 1.75, 5.0, 5.0);
    txt(s, "Agreed in public", { x: 0.85, y: 1.9, w: 4.5, h: 0.45, fontFace: HEAD, fontSize: 18, bold: true, color: INK });
    bullets(s, [`Caps: R pads its identifiers to n_R^max = ${T.n_R_max}; O pads its register to n̄_O = ${T.n_O_bar}`,
      "Bounds: every value in [−1, 1], so B_R = √3 (two features and the intercept), B_O = √2, B_y = 1",
      `Budget ε = ${T.eps}, δ = 10⁻⁹; stage-1 share f_1 = ${T.f1}; target ϑ* = ${T.theta_star}; α = ${T.alpha}`,
      "Split key: s = s_R + s_O, neither share known to the other party"], { x: 0.85, y: 2.45, w: 4.55, h: 4.2, fontSize: 14.5 });
    card(s, 5.85, 1.75, 6.85, 3.35, EMBER_L);
    txt(s, "Derived by O", { x: 6.1, y: 1.9, w: 6.3, h: 0.45, fontFace: HEAD, fontSize: 18, bold: true, color: INK });
    const rows = [
      ["Sensitivity", `Δ² = (2·2 + 3 + 1)²/2 + 2·3·1 = 38`, `Δ = ${num(T.Delta)}`],
      ["Stage 1 only (B, C)", "Δ_G = (2·2 + 3)/√2", `Δ_G = ${num(T.Delta_G)}`],
      ["Budget", "tight zCDP conversion of (1, 10⁻⁹)", `ρ_run = ${num(T.rho, 4)}`],
      ["Stage-1 noise", "Δ_G / √(2 · 0.1 · ρ_run)", `σ_1 = ${num(T.sigma1, 1)}`],
      ["Stage-2 noise", "Δ / √(2 · 0.9 · ρ_run)", `σ_2 = ${num(T.sigma2, 1)}`],
    ];
    s.addTable(rows, { x: 6.1, y: 2.45, colW: [1.65, 3.45, 1.25], rowH: 0.42, fontFace: BODY, fontSize: 12.5, color: TEXT, border: { type: "solid", color: LINE, pt: 0.75 }, valign: "middle", fill: { color: WHITE } });
    card(s, 5.85, 5.35, 6.85, 1.4, "F7F9FC");
    txt(s, [{ text: "Already visible: ", options: { bold: true, color: EMBER } },
      { text: `each released number will carry noise of about ±${num(T.sigma2, 0)}, while the statistics are about 1. With two people, privacy must win.` }],
      { x: 6.1, y: 5.45, w: 6.4, h: 1.2, fontSize: 14.5, valign: "middle" });
    s.addNotes("Phase 0. The parties fix the caps, the clipping bounds and the budget before touching data. With every value in [-1, 1], R's row including the intercept has norm at most sqrt(3), O's at most sqrt(2), and |y| at most 1. In the notation of the introduction, R = 3, O = 2 and Y = 1, so Delta^2 = 38 and Delta = 6.16. yT y is free here because Y = 1 is at most 2(R + O) = 10. The tight conversion of (1, 1e-9) gives rho = 0.0150; 10% goes to stage 1, 90% to stage 2, giving sigma_1 = 90.5 and sigma_2 = 37.5. Compare that with statistics of size about 1: with two people, the released numbers will be almost pure noise, as they must be.");
  }

  // ======================= Phase 1: link =======================
  {
    const s = pres.addSlide(); title(s, "Phase 1: link", "Labeled PSI: R learns which of its identifiers match, and where those rows sit in O's shuffled register"); phase(s, "Phase 1 · R + O", SLATE);
    card(s, 0.6, 1.75, 4.6, 3.65, RBLUE_L);
    txt(s, "R's padded identifiers", { x: 0.85, y: 1.88, w: 4.2, h: 0.4, fontSize: 15, bold: true, color: RBLUE });
    const rids = [...T.R_rows.map((r) => r[0]), T.R_dummy];
    s.addTable([hdr(["id", "F_k(id)", "result"]), ...rids.map((id) => [id === T.R_dummy ? `${id} (dummy)` : id, T.oprf[id],
      { text: T.psi[id] ? `match → position ${T.psi[id]}` : "no match", options: T.psi[id] ? { bold: true, color: RBLUE } : { color: MUTED } }])],
      { x: 0.85, y: 2.35, colW: [1.25, 1.1, 1.85], rowH: 0.4, fontFace: BODY, fontSize: 13, color: TEXT, border: { type: "solid", color: LINE, pt: 0.75 }, valign: "middle", fill: { color: WHITE } });
    card(s, 5.45, 1.75, 7.25, 3.65, EMBER_L);
    txt(s, "O's padded register, after its secret shuffle τ", { x: 5.7, y: 1.88, w: 6.8, h: 0.4, fontSize: 15, bold: true, color: EMBER });
    const oids = [...T.O_rows.map((r) => r[0]), T.O_dummy];
    s.addTable([hdr(["id", "F_k(id)", "position τ", "note"]), ...oids.map((id) => [id === T.O_dummy ? `${id} (dummy)` : id, T.oprf[id], String(T.tau[id]),
      { text: id === T.O_dummy ? "x_O = 0, y = 0" : "", options: { color: MUTED } }])],
      { x: 5.7, y: 2.35, colW: [1.4, 1.1, 1.3, 2.9], rowH: 0.4, fontFace: BODY, fontSize: 13, color: TEXT, border: { type: "solid", color: LINE, pt: 0.75 }, valign: "middle", fill: { color: WHITE } });
    card(s, 0.6, 5.65, 6.0, 1.1, RBLUE_L);
    txt(s, [{ text: "R learns: ", options: { bold: true, color: RBLUE } }, { text: `π(104) = ${T.psi["104"]}, π(221) = ${T.psi["221"]}; 317 is not in the register.` }], { x: 0.85, y: 5.72, w: 5.6, h: 0.95, fontSize: 14.5, valign: "middle" });
    card(s, 6.75, 5.65, 5.95, 1.1, EMBER_L);
    txt(s, [{ text: "O learns: ", options: { bold: true, color: EMBER } }, { text: `${T.n_R_max} queries and its own ${T.n_O_bar} rows; not which matched, nor that R has 3 people.` }], { x: 7.0, y: 5.72, w: 5.55, h: 0.95, fontSize: 14.5, valign: "middle" });
    txt(s, "F_k(id): O's keyed hash (OPRF), truncated here for display.", { x: 0.6, y: 7.0, w: 8, h: 0.3, fontSize: 11, italic: true, color: MUTED });
    s.addNotes("Phase 1. R pads its three identifiers with a dummy to the public cap of four. Through the OPRF it obtains O's keyed hash of each identifier without revealing it. O pads its register with a dummy row to six rows and shuffles them with a secret permutation tau, fresh for this run. Labeled PSI then tells R which of its hashed identifiers appear in O's register and, for each, a label: the row's position after the shuffle. So R learns that 104 sits at position 6 and 221 at position 3, which defines the bijection pi, and that 317 is not in the register. O learns only the padded sizes.");
  }

  // ======================= Phase 1: A =======================
  {
    const s = pres.addSlide(); title(s, "Phase 1: R computes its own block", "A uses only R's rows in I, so R computes it exactly; it never leaves R"); phase(s, "Phase 1 · R", RBLUE);
    txt(s, "R's rows ẋ_i = (1, x_R1, x_R2) for i ∈ I", { x: 0.6, y: 1.75, w: 6, h: 0.4, fontFace: HEAD, fontSize: 16, bold: true, color: INK });
    mat(s, T.I.map((id) => [1, ...T.R_rows.find((r) => r[0] === id).slice(1)]), 0.6, 2.25, { rowLab: T.I.map((i) => `ẋ_${i}`), colLab: NAMES.slice(0, 3), cw: 1.15, labW: 0.9, d: 1, fs: 13, fill: () => [BLUE_BLK] });
    txt(s, "A = ẋ_104 ẋ_104ᵀ + ẋ_221 ẋ_221ᵀ", { x: 0.6, y: 3.45, w: 6, h: 0.4, fontFace: HEAD, fontSize: 16, bold: true, color: INK });
    mat(s, T.A, 0.6, 3.95, { rowLab: NAMES.slice(0, 3), colLab: NAMES.slice(0, 3), cw: 1.15, labW: 1.1, fs: 13, fill: () => [BLUE_BLK] });
    card(s, 6.9, 1.75, 5.8, 2.6, "F7F9FC");
    txt(s, "Check", { x: 7.15, y: 1.88, w: 5.3, h: 0.4, fontSize: 15, bold: true, color: SLATE });
    txt(s, "A equals the blue block of G on slide 4: the protocol gets R's part exactly, with no encryption and no noise.",
      { x: 7.15, y: 2.32, w: 5.35, h: 1.9, fontSize: 14.5 });
    card(s, 6.9, 4.6, 5.8, 2.15, EMBER_L);
    txt(s, "Why it matters here", { x: 7.15, y: 4.72, w: 5.3, h: 0.4, fontSize: 15, bold: true, color: EMBER });
    txt(s, `A has rank 2 (two people, three columns), so its smallest eigenvalue is 0. That will decide where the ridge goes in stage 1.`,
      { x: 7.15, y: 5.15, w: 5.35, h: 1.5, fontSize: 14.5 });
    s.addNotes("R now knows which of its people matched, so it computes its own block A directly from its two matched rows, each with a leading 1 for the intercept. A is exact, never encrypted, never noised and never sent. With two people and three columns, A has rank 2: its smallest eigenvalue is zero. Keep that in mind for the ridge choice in stage 1.");
  }

  // ======================= Phase 2: upload =======================
  {
    const s = pres.addSlide(); title(s, "Phase 2: upload", "R writes its data into O's shuffled order, zero everywhere it has no match, and encrypts it"); phase(s, "Phase 2 · R → O", RBLUE);
    const pos = [1, 2, 3, 4, 5, 6];
    const hot = (j) => [3, 6].includes(j + 1);
    const vecs = [["b (selection)", T.b], ["ẋ_0 (intercept)", T.X[0]], ["ẋ_1 (education)", T.X[1]], ["ẋ_2 (health)", T.X[2]]];
    mat(s, vecs.map((v) => v[1]), 0.6, 1.85, { rowLab: vecs.map((v) => v[0]), colLab: pos.map((p) => `pos ${p}`), cw: 0.95, labW: 1.75, d: 1, fs: 13, rh: 0.42,
      fill: (i, j) => (hot(j) ? [HILITE] : null) });
    txt(s, "Positions 3 and 6 are π(221) and π(104). ẋ_0 equals b: the intercept is 1 exactly where there is a match.", { x: 0.6, y: 4.1, w: 7.5, h: 0.6, fontSize: 13.5, color: MUTED, italic: true });
    card(s, 8.55, 1.85, 4.15, 2.75, EMBER_L);
    txt(s, "What O receives", { x: 8.8, y: 1.98, w: 3.7, h: 0.4, fontSize: 15, bold: true, color: EMBER });
    ["b", "ẋ_0", "ẋ_1", "ẋ_2"].forEach((nm, i) => {
      circleIcon(s, "lock", 8.8, 2.48 + i * 0.5, 0.36, SLATE);
      txt(s, `Enc(${nm}): looks random`, { x: 9.3, y: 2.5 + i * 0.5, w: 3.3, h: 0.35, fontSize: 13 });
    });
    card(s, 0.6, 4.95, 12.1, 1.8, "F7F9FC");
    h.latex(s, "upload", 0.9, 5.15, { size: 18, maxW: 11.5 });
    txt(s, "R encrypts each vector with its own share; O folds in its share to move it under the joint key. Every position looks the same to O, so it cannot see that 3 and 6 are special.",
      { x: 0.9, y: 5.95, w: 11.5, h: 0.7, fontSize: 13.5, color: MUTED });
    s.addNotes("Phase 2. R builds vectors as long as O's padded register, in O's shuffled order. The selection vector b has a 1 at positions 3 and 6, the matched rows. The feature vectors hold R's values at those positions: 221's values at position 3, 104's at position 6, and zeros elsewhere. The intercept vector equals b. R encrypts each vector under its own key share, with a seed for the random part, and sends it; O subtracts a times its own share to obtain an encryption under the joint key. O sees four ciphertexts that look random, so it learns nothing about which positions matter.");
  }

  // ======================= Phase 3: aggregate =======================
  {
    const s = pres.addSlide(); title(s, "Phase 3: aggregate", "O multiplies R's encrypted vectors by its own columns, position by position, and sums"); phase(s, "Phase 3 · O", EMBER);
    const pos = [1, 2, 3, 4, 5, 6];
    const hot = (j) => [3, 6].includes(j + 1);
    txt(s, "O's columns in its shuffled order", { x: 0.6, y: 1.7, w: 6, h: 0.4, fontSize: 15, bold: true, color: EMBER });
    mat(s, T.Ocol, 0.6, 2.15, { rowLab: ["age x_O1", "benefits x_O2", "income y"], colLab: pos.map((p, j) => `${p}: ${T.order[j] === T.O_dummy ? "dummy" : T.order[j]}`), cw: 0.95, labW: 1.6, d: 1, fs: 12.5, rh: 0.4,
      fill: (i, j) => (hot(j) ? [HILITE] : null) });
    card(s, 0.6, 3.95, 7.3, 2.8, "F7F9FC");
    txt(s, "Two of the sums, worked out (other positions multiply by 0)", { x: 0.85, y: 4.07, w: 6.9, h: 0.4, fontSize: 14, bold: true, color: SLATE });
    const c11 = T.Cm[1][0], b11 = T.Bm[0][0];
    txt(s, [{ text: "C_11 = Σ_t ẋ_1,t · age_t = (−0.4)(0.8) + (0.5)(−0.6) = ", options: {} }, { text: num(c11), options: { bold: true } },
      { text: "  (cross-party: pairs R's row with O's row)", options: { color: MUTED, fontSize: 12.5, breakLine: true } },
      { text: "B_11 = Σ_t b_t · age_t² = (0.8)² + (−0.6)² = ", options: {} }, { text: num(b11), options: { bold: true } },
      { text: "  (O's matched rows only)", options: { color: MUTED, fontSize: 12.5 } }],
      { x: 0.85, y: 4.55, w: 6.9, h: 2.1, fontSize: 15, paraSpaceAfter: 14 });
    card(s, 8.15, 3.95, 4.55, 2.8, EMBER_L);
    txt(s, "All results, still encrypted", { x: 8.4, y: 4.07, w: 4.1, h: 0.4, fontSize: 14, bold: true, color: EMBER });
    txt(s, [`B: ${num(T.Bm[0][0])}, ${num(T.Bm[0][1])}, ${num(T.Bm[1][1])}`, `C: ${T.Cm.map((r) => r.map((v) => num(v)).join(", ")).join("; ")}`,
      `c_R: ${T.cR.map((v) => num(v)).join(", ")}`, `c_O: ${T.cO.map((v) => num(v)).join(", ")}`, `yᵀy: ${num(T.yy)}`].map((t, i, a) => ({ text: t, options: { breakLine: i < a.length - 1 } })),
      { x: 8.4, y: 4.52, w: 4.15, h: 1.55, fontSize: 13 });
    txt(s, "Identical to the pooled answer.", { x: 8.4, y: 6.15, w: 4.1, h: 0.45, fontSize: 13.5, bold: true, color: TEAL });
    s.addNotes("Phase 3. O has its own columns in the shuffled order, including the dummy row of zeros. Each needed sum is an inner product between one of R's encrypted vectors and one of O's plaintext columns. Positions where R wrote zeros contribute nothing, so only positions 3 and 6, the matched rows, survive. Two examples are worked out. C_11 pairs R's education with O's age row by row, so it depends on the bijection pi. B_11 needs only O's matched rows. The full list matches the pooled answer exactly; the only error in practice is CKKS's tiny approximation error.");
  }

  // ======================= Phase 3: packing =======================
  {
    const s = pres.addSlide(); title(s, "Phase 3: packing the results", "Each released number sits alone in its own slot of a packed ciphertext"); phase(s, "Phase 3 · O", EMBER);
    const strip = (slots, y, label, sub, fillOf) => {
      txt(s, label, { x: 0.6, y, w: 6, h: 0.4, fontFace: HEAD, fontSize: 16, bold: true, color: INK });
      txt(s, sub, { x: 6.6, y: y + 0.03, w: 6.1, h: 0.4, fontSize: 13, color: MUTED, align: "right" });
      const w = 12.1 / Math.max(15, slots.length + 1);
      slots.forEach(([k, v], i) => {
        const [f, col] = fillOf(k);
        s.addShape(pres.shapes.RECTANGLE, { x: 0.6 + i * w, y: y + 0.5, w: w - 0.04, h: 0.95, fill: { color: f }, line: { color: WHITE, width: 1 } });
        txt(s, [{ text: k.replace(/^([BC])_(\d)(\d)$/, "$1_{$2$3}").replace(/^c_(R|O)(\d)$/, "c_{$1$2}"), options: { bold: true, breakLine: true, fontSize: 11.5 } }, { text: num(v), options: { fontSize: 12 } }],
          { x: 0.6 + i * w, y: y + 0.55, w: w - 0.04, h: 0.85, align: "center", color: col, valign: "middle" });
      });
      if (slots.length < 15) txt(s, "0 …", { x: 0.6 + slots.length * w, y: y + 0.5, w: 12.1 - slots.length * w, h: 0.95, fontSize: 14, color: MUTED, valign: "middle", align: "center" });
    };
    const fillOf = (k) => (k.startsWith("B") || k.startsWith("c_O") || k.startsWith("y") ? [O_BLK, TEXT] : [EMBER, WHITE]);
    strip(T.slots1, 1.75, "M_1: for stage 1", "B and C only (9 slots); the other 8,183 slots hold 0", fillOf);
    strip(T.slots2, 3.75, "M_2: for stage 2", "all 15 O-dependent numbers; yᵀy included, since it is free", fillOf);
    card(s, 0.6, 5.75, 12.1, 1.0, "F7F9FC");
    txt(s, "Two separate ciphertexts, so the first release can only ever depend on B and C. Slots run 1 to N/2 = 8,192 in the real scheme.",
      { x: 0.85, y: 5.82, w: 11.6, h: 0.85, fontSize: 14.5, valign: "middle" });
    s.addNotes("O moves every result into its own slot of a packed ciphertext. M1 carries only the nine entries of B (upper triangle) and C, for the first release, which is used to choose the ridge. M2 carries all fifteen O-dependent numbers, including c_R, c_O and yT y, for the main release. Keeping them separate means stage 1 cannot depend on anything but B and C. A real ciphertext has 8,192 slots; the rest are zero.");
  }

  // ======================= Phase 4: release algebra =======================
  {
    const s = pres.addSlide(); title(s, "Phase 4: the release, on one number", "Toy arithmetic: one integer instead of a polynomial, modulus q = 1,000,003, no security"); phase(s, "Phase 4 · O → R", EMBER);
    const rows = [hdr(["", "Step", "Formula", "Value"]),
      ["", "The statistic", "z = C_11", num(TOY.z)],
      ["", "Encode at scale Ω = 1000", "Ecd(z) = round(Ω z)", int(TOY.ecd)],
      ["", "Aggregated ciphertext (key s = s_R + s_O = 3 + 5)", "c_0 = −a s + e + Ecd(z),  c_1 = a", `(${int(TOY.c0)}, ${int(TOY.c1)})`],
      [{ text: "O", options: { bold: true, color: EMBER } }, "Re-randomize (u = 1, e_0 = 1, e_1 = −2)", "c_0′ = c_0 + p_0 u + e_0,  c_1′ = c_1 + a u + e_1", `(${int(TOY.c0p)}, ${int(TOY.c1p)})`],
      [{ text: "O", options: { bold: true, color: EMBER } }, "Add noise inside its share", `h_O = c_0′ + c_1′ s_O + η,  η = ${int(TOY.eta)}`, int(TOY.hO)],
      [{ text: "O", options: { bold: true, color: EMBER } }, "Send; keep c_0′ secret", "(c_1′, h_O)", `(${int(TOY.c1p)}, ${int(TOY.hO)})`],
      [{ text: "R", options: { bold: true, color: RBLUE } }, "Add its share", "v = h_O + c_1′ s_R", int(TOY.v)],
      [{ text: "R", options: { bold: true, color: RBLUE } }, "…which equals", `Ecd(z) + e_tot + η = ${int(TOY.ecd)}${[TOY.e_tot, TOY.eta].map((t) => (t < 0 ? " − " : " + ") + Math.abs(t).toLocaleString("en-US")).join("")}`, int(TOY.v)],
      [{ text: "R", options: { bold: true, color: RBLUE } }, "Decode", "v / Ω", { text: num(TOY.decoded), options: { bold: true } }]];
    s.addTable(rows, { x: 0.6, y: 1.7, colW: [0.45, 4.3, 4.85, 2.5], rowH: 0.42, fontFace: BODY, fontSize: 13, color: TEXT, border: { type: "solid", color: LINE, pt: 0.75 }, valign: "middle" });
    card(s, 0.6, 6.15, 12.1, 0.95, EMBER_L);
    txt(s, `R gets ${num(TOY.z)} plus noise of ${num((TOY.eta + TOY.e_tot) / TOY.Omega)}: the noise η/Ω has standard deviation σ_2 = ${num(T.sigma2, 1)}. Without c_0′, R cannot strip η off, and it never sees O's share.`,
      { x: 0.85, y: 6.22, w: 11.6, h: 0.8, fontSize: 14.5, valign: "middle" });
    s.addNotes("The release algebra on a single number, with toy values so the arithmetic can be followed; real ciphertexts are polynomials of degree 16,384 with much larger moduli. The aggregated ciphertext (c0, c1) encrypts C_11 = -0.62 at scale 1000 under the joint key s = 3 + 5. O re-randomizes it with fresh u, e0, e1, adds its share c1' s_O plus noise eta, and sends only c1' and h_O. R adds its own share and gets Ecd(z) plus a small encryption error plus eta. Decoding gives -0.62 plus noise of about -64: the noise eta / Omega is drawn with standard deviation sigma_2 = 37.5. R cannot remove eta, because that would need c0', which O keeps.");
  }

  // ======================= Stage 1 =======================
  {
    const s = pres.addSlide(); title(s, "Stage 1: choosing the ridge", `A first release of B and C only, with noise σ_1 = ${num(T.sigma1, 1)}`); phase(s, "Phases 4–5 · O → R", EMBER);
    const keys = T.slots1.map((x) => x[0]);
    const trueV = T.slots1.map((x) => x[1]);
    const iu = [[0, 0], [0, 1], [1, 1]];
    const rel = [...iu.map(([j, k]) => S1.Bhat[j][k]), ...S1.Chat.flat()];
    const lab = (k) => k.replace(/^([BC])_(\d)(\d)$/, "$1_{$2$3}");
    s.addTable([hdr(["", ...keys.map(lab)]), [{ text: "true", options: { bold: true, color: MUTED } }, ...trueV.map((v) => num(v))],
      [{ text: "released", options: { bold: true, color: EMBER } }, ...rel.map((v) => ({ text: num(v, 1), options: { bold: true } }))]],
      { x: 0.6, y: 1.75, colW: [1.1, ...keys.map(() => 1.22)], rowH: 0.38, fontFace: BODY, fontSize: 13, color: TEXT, border: { type: "solid", color: LINE, pt: 0.75 }, valign: "middle", align: "center" });
    const rows = [
      ["R forms Ĝ_1 = A ⊕ noisy B, C; its smallest eigenvalue", `λ_min(Ĝ_1) = ${num(S1.lmin1, 1)}`],
      ["Radius for the stage-1 noise", `r_1 = σ_1(2√5 + 2√ln 400) = ${num(S1.r1, 1)}`],
      ["Target for stage 2", `g* = 2 · 2.5 · σ_2 · √5 = ${num(S1.g, 1)}`],
      ["Ridge on everything", `λ_I = max{0, g* + r_1 − λ_min(Ĝ_1)} = ${num(S1.lam_I, 1)}`],
      ["Ridge on O's block only", "λ_O = ∞: it can never lift λ_min above λ_min(A) = 0"],
      [{ text: "Choice", options: { bold: true } }, { text: `Ψ = I, λ = ${num(S1.lam, 0)}`, options: { bold: true, color: EMBER } }]];
    s.addTable(rows, { x: 0.6, y: 3.15, colW: [5.2, 6.9], rowH: 0.42, fontFace: BODY, fontSize: 13.5, color: TEXT, border: { type: "solid", color: LINE, pt: 0.75 }, valign: "middle" });
    card(s, 0.6, 5.9, 12.1, 0.85, EMBER_L);
    txt(s, "The ridge is huge because the noise is huge relative to two people's data. It is chosen before stage 2, so it is independent of the noise there.",
      { x: 0.85, y: 5.95, w: 11.6, h: 0.75, fontSize: 14.5, valign: "middle" });
    s.addNotes("Stage 1 spends 10% of the budget on a release of B and C only, with sigma_1 = 90.5. The released numbers bear no resemblance to the true ones: B_11 is 1.00 but released as -0.5. R adds its exact block A and computes the smallest eigenvalue of the noisy matrix, which is negative. The rule then asks for a ridge large enough that, even after the stage-1 noise (radius r_1) and with margin for the stage-2 noise (target g*), the matrix stays safely invertible. A ridge on O's block alone cannot work here, because R's own block A has rank 2 and smallest eigenvalue 0. So the ridge goes on everything, with lambda of about 1,487.");
  }

  // ======================= Stage 2 =======================
  {
    const s = pres.addSlide(); title(s, "Stage 2: the main release", `All fifteen O-dependent numbers, with noise σ_2 = ${num(T.sigma2, 1)}`); phase(s, "Phase 4 · O → R", EMBER);
    const G2 = S2.G2, c2 = S2.c2;
    const rel = [G2[3][3], G2[3][4], G2[4][4], G2[0][3], G2[0][4], G2[1][3], G2[1][4], G2[2][3], G2[2][4], ...c2.slice(0, 3), c2[3], c2[4], S2.yy2];
    const keys = T.slots2.map((x) => x[0]), trueV = T.slots2.map((x) => x[1]);
    const lab = (k) => k.replace(/^([BC])_(\d)(\d)$/, "$1_{$2$3}").replace(/^c_(R|O)(\d)$/, "c_{$1$2}");
    [[0, 8], [8, 15]].forEach(([a, b], g) => {
      s.addTable([hdr(["", ...keys.slice(a, b).map(lab)]), [{ text: "true", options: { bold: true, color: MUTED } }, ...trueV.slice(a, b).map((v) => num(v))],
        [{ text: "released", options: { bold: true, color: EMBER } }, ...rel.slice(a, b).map((v) => ({ text: num(v, 1), options: { bold: true } }))]],
        { x: 0.6, y: 1.75 + g * 1.45, colW: [1.1, ...keys.slice(a, b).map(() => 1.3)], rowH: 0.38, fontFace: BODY, fontSize: 13, color: TEXT, border: { type: "solid", color: LINE, pt: 0.75 }, valign: "middle", align: "center" });
    });
    card(s, 0.6, 4.85, 5.95, 1.9, "F7F9FC");
    txt(s, "Gate", { x: 0.85, y: 4.97, w: 5.5, h: 0.4, fontSize: 15, bold: true, color: SLATE });
    txt(s, `ϑ̂ = λ_min(Ĝ_2 + λI) / (2σ_2√5) = ${num(S2.gate)} ≥ 1, so R may report an estimate.`, { x: 0.85, y: 5.4, w: 5.5, h: 1.25, fontSize: 14.5 });
    card(s, 6.75, 4.85, 5.95, 1.9, EMBER_L);
    txt(s, "What R holds now", { x: 7.0, y: 4.97, w: 5.5, h: 0.4, fontSize: 15, bold: true, color: EMBER });
    txt(s, "Its exact A, and fifteen numbers that are almost pure noise. Nothing about any one person in O's register can be read from them beyond what ε = 1 allows.",
      { x: 7.0, y: 5.4, w: 5.5, h: 1.25, fontSize: 14 });
    s.addNotes("Stage 2 releases all fifteen O-dependent numbers from M2, with the remaining 90% of the budget, so sigma_2 = 37.5. Again the released numbers are dominated by noise: yT y is 0.65 but released as 77.6. The gate checks that the ridged matrix is safely invertible; with lambda around 1,500 it is, so R may report an estimate. Note what R now holds: its exact block A and fifteen noisy numbers. That is all it ever learns about O's data.");
  }

  // ======================= Phase 5: estimate =======================
  {
    const s = pres.addSlide(); title(s, "Phase 5: the estimate and its intervals", "θ̃ = (Ĝ_2 + λI)⁻¹ ĉ_2, with a parametric bootstrap for the noise"); phase(s, "Phase 5 · R", RBLUE);
    const zq = T.z;
    const rows = [hdr(["Coefficient", "Protocol estimate θ̃", `Wald interval θ̃ ± ${num(zq)}·sd`, "Ridge target θ_λ (true)", "Covers?"])];
    NAMES.forEach((nm, j) => {
      const lo = S2.theta[j] - zq * S2.sd[j], hi = S2.theta[j] + zq * S2.sd[j];
      const cov = lo <= S2.theta_lam[j] && S2.theta_lam[j] <= hi;
      rows.push([nm, num(S2.theta[j], 3), `[${num(lo, 3)}, ${num(hi, 3)}]`, num(S2.theta_lam[j], 4), { text: cov ? "yes" : "no", options: { bold: true, color: cov ? TEAL : EMBER } }]);
    });
    s.addTable(rows, Object.assign(tableOpts([2.2, 2.4, 3.1, 2.7, 1.7]), { y: 1.75, rowH: 0.4, fontSize: 13.5 }));
    card(s, 0.6, 4.4, 5.95, 2.35, "F7F9FC");
    txt(s, "Reading it", { x: 0.85, y: 4.52, w: 5.5, h: 0.4, fontSize: 15, bold: true, color: SLATE });
    bullets(s, [`λ ≈ ${num(S1.lam, 0)} dwarfs G (entries about 1), so θ_λ ≈ 0 and θ̃ ≈ ĉ_2 / λ: noise, shrunk`,
      "The intervals are honest about it: they are as wide as the estimates, and cover the target",
      "The least-squares β does not exist here (G has rank 2)"], { x: 0.85, y: 4.95, w: 5.55, h: 1.75, fontSize: 13.5 });
    card(s, 6.75, 4.4, 5.95, 2.35, EMBER_L);
    txt(s, "The protocol fails safe", { x: 7.0, y: 4.52, w: 5.5, h: 0.4, fontSize: 15, bold: true, color: EMBER });
    txt(s, "With two people, any answer that told R something about them would break privacy. So the estimate is shrunk to almost nothing, and the intervals say so.",
      { x: 7.0, y: 4.95, w: 5.5, h: 1.7, fontSize: 14.5 });
    s.addNotes("The estimate is the ridge solution on the second release. Because lambda is so large compared with the entries of G, the true ridge target theta_lambda is essentially zero, and the protocol's estimate is the noisy c vector divided by lambda: small numbers of size about 0.03 that carry no information. The bootstrap captures exactly this: the Wald intervals have half-width about 0.05 and cover the target. The least-squares coefficient beta is not defined with two people, and the bias-aware interval for beta would be unbounded without a public bound on its norm. This is the protocol failing safe: when the data are too few for the budget, it returns a heavily shrunk estimate with honest intervals rather than leaking.");
  }

  // ======================= Views =======================
  {
    const s = pres.addSlide(); title(s, "What each party saw");
    card(s, 0.6, 1.55, 5.95, 4.3, EMBER_L);
    circleIcon(s, "bank", 0.85, 1.8, 0.7, EMBER);
    txt(s, "O", { x: 1.75, y: 1.9, w: 4, h: 0.5, fontFace: HEAD, fontSize: 22, bold: true, color: INK });
    bullets(s, [`${T.n_R_max} blinded OPRF queries and encrypted PSI queries`, "Four upload ciphertexts that look random",
      `The padded sizes: ${T.n_R_max} queries, ${T.n_O_bar} rows`, { text: "Not: which rows matched, R's features, or that R has three people", options: { bold: true } },
      { text: "Not: any decryption share from R", options: { bold: true } }], { x: 0.85, y: 2.75, w: 5.45, h: 3.0, fontSize: 15 });
    card(s, 6.75, 1.55, 5.95, 4.3, RBLUE_L);
    circleIcon(s, "user", 7.0, 1.8, 0.7, RBLUE);
    txt(s, "R", { x: 7.9, y: 1.9, w: 4, h: 0.5, fontFace: HEAD, fontSize: 22, bold: true, color: INK });
    bullets(s, ["Its own data, and its exact block A", "That 104 and 221 are in the register, at positions 6 and 3; that 317 is not",
      `Two noisy releases (σ_1 = ${num(T.sigma1, 1)}, σ_2 = ${num(T.sigma2, 1)}) and the estimate θ̃`,
      { text: "Not: O's rows, O's key share, or the c_0′ that would remove the noise", options: { bold: true } }], { x: 7.0, y: 2.75, w: 5.45, h: 3.0, fontSize: 15 });
    card(s, 0.6, 6.1, 12.1, 0.75, "F7F9FC");
    txt(s, "Nobody ever computed O's parts of G and c in the clear: the pooled answer on slide 4 exists only in this deck.", { x: 0.85, y: 6.15, w: 11.6, h: 0.65, fontSize: 14.5, valign: "middle", italic: true });
    s.addNotes("A recap of the two views. O handled R's queries and ciphertexts but never decrypted anything and never received a decryption share from R, so it learned only the padded sizes. R learned its match status, as the protocol discloses by design, computed its own block exactly, and received two heavily noised releases. It never saw O's rows, O's key share, or the part of the ciphertext that would let it strip off the noise.");
  }

  // ======================= Scale-up =======================
  {
    const s = pres.addSlide(); title(s, `The same steps with ${SC.n.toLocaleString("en-US")} people`, "Same bounds, same budget, so the same noise: σ_1 = 90.5 and σ_2 = 37.5 per released number");
    txt(s, "Matched people drawn from a population like the toy's: R's features and age uniform on [−1, 1], benefits 0 or 1, income linear in all four plus noise, clipped to [−1, 1].",
      { x: 0.6, y: 1.7, w: 12.1, h: 0.65, fontSize: 14.5 });
    const rows = [hdr(["Coefficient", "True β", "Least squares, pooled (non-private)", "Protocol θ̃", `Wald interval ± ${num(T.z)}·sd`])];
    NAMES.forEach((nm, j) => rows.push([nm, num(SC.beta_true[j], 2), num(SC.ols[j], 3), { text: num(SC.theta[j], 3), options: { bold: true } }, `± ${num(T.z * SC.sd[j], 3)}`]));
    s.addTable(rows, Object.assign(tableOpts([2.2, 1.6, 3.5, 2.2, 2.6]), { y: 2.5, rowH: 0.4, fontSize: 13.5 }));
    card(s, 0.6, 5.1, 5.95, 1.65, "F7F9FC");
    txt(s, "Stage 1 now", { x: 0.85, y: 5.2, w: 5.5, h: 0.4, fontSize: 15, bold: true, color: SLATE });
    txt(s, `λ_min(Ĝ_1) = ${int(Math.round(SC.lmin1))} > g* + r_1 = ${int(Math.round(SC.g + SC.r1))}, so λ = 0: no ridge. Gate ϑ̂ = ${num(SC.gate, 1)}.`,
      { x: 0.85, y: 5.62, w: 5.5, h: 1.05, fontSize: 14.5 });
    card(s, 6.75, 5.1, 5.95, 1.65, EMBER_L);
    txt(s, "Why it works now", { x: 7.0, y: 5.2, w: 5.5, h: 0.4, fontSize: 15, bold: true, color: EMBER });
    txt(s, `The noise per number is unchanged, but the statistics grew with n (G_11 ≈ ${int(Math.round(SC.G11))}). The estimate lands within 0.01 of pooled least squares.`,
      { x: 7.0, y: 5.62, w: 5.5, h: 1.05, fontSize: 14.5 });
    s.addNotes("The same protocol, the same bounds and the same budget, so exactly the same noise per released number. The only change is the number of matched people: 20,000 instead of 2. Now the statistics are in the thousands, the stage-1 eigenvalue comfortably exceeds the threshold, the rule picks no ridge at all, and the gate passes easily. The protocol's estimate is within 0.01 of pooled, non-private least squares on every coefficient, with intervals of about plus or minus 0.01 to 0.02. Differential privacy hides individuals, and with enough people the aggregate signal comes through.");
  }

  // ======================= Takeaways =======================
  {
    const s = pres.addSlide(); s.background = { color: INK };
    s.addText("What the example shows", { x: 0.6, y: 0.45, w: 12, h: 0.8, fontFace: HEAD, fontSize: 34, bold: true, color: WHITE, margin: 0, isTextBox: true });
    const take = ["Linkage gives R its matches and their positions in O's shuffled register; O learns only the padded sizes",
      "R writes its data into O's order with zeros elsewhere, so O's encrypted sums pick out exactly the matched rows",
      "O adds the noise inside its share of the decryption: R sees each statistic only with noise of the planned size",
      "With two people the noise dominates; the ridge shrinks the estimate to almost nothing, and the intervals say so",
      `With ${SC.n.toLocaleString("en-US")} people the same steps, with the same noise, recover least squares to within 0.01`];
    take.forEach((t, i) => {
      numCircle(s, i + 1, 0.6, 1.6 + i * 1.05, 0.6, EMBER, 17);
      s.addText(t, { x: 1.45, y: 1.62 + i * 1.05, w: 11.2, h: 0.9, fontFace: BODY, fontSize: 17, color: WHITE, margin: 0, valign: "top", isTextBox: true });
    });
    s.addNotes("Five things to take away. Linkage defines the bijection between R's matched rows and O's rows without telling O anything. R's zero-padded vectors in O's order make the encrypted sums select exactly the matched rows. The noise sits in O's decryption share, so R only ever sees noisy statistics. With too few people the protocol fails safe. And with enough people the same steps, with the same noise, give essentially the least-squares answer.");
  }

  await writeDeck(pres, P("slides", "vfl_toy_example.pptx"));
})();
