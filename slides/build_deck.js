// Build the Scenario-B VFL deck.
// Requires: npm install pptxgenjs react-icons react react-dom sharp
// Run from the repo root:  node slides/build_deck.js
const path = require("path");
const pptxgen = require("pptxgenjs");
const React = require("react");
const ReactDOMServer = require("react-dom/server");
const sharp = require("sharp");
const fa = require("react-icons/fa");

const ROOT = path.resolve(__dirname, "..");
const P = (...p) => path.join(ROOT, ...p);

// ---------- palette & type ----------
const INK = "14213D";     // dominant dark
const SLATE = "3D5A80";   // secondary
const MIST = "EEF2F7";    // light panels
const LINE = "C9D3E0";
const EMBER = "C45A0C";   // accent (matches the protocol figure's "new/changed" orange)
const EMBER_L = "F6E3D6";
const TEAL = "1B8A7A";
const TEXT = "1F2937";
const MUTED = "5B6576";
const WHITE = "FFFFFF";
const HEAD = "Cambria";
const BODY = "Calibri";

const W = 13.333, H = 7.5;

async function iconPng(Icon, color = "#FFFFFF", size = 256) {
  const svg = ReactDOMServer.renderToStaticMarkup(React.createElement(Icon, { color, size: String(size) }));
  const buf = await sharp(Buffer.from(svg)).png().toBuffer();
  return "image/png;base64," + buf.toString("base64");
}

(async () => {
  const icons = {};
  const need = {
    user: fa.FaUserGraduate, bank: fa.FaLandmark, link: fa.FaLink, lock: fa.FaLock,
    shield: fa.FaShieldAlt, secret: fa.FaUserSecret, chart: fa.FaChartLine, flask: fa.FaFlask,
    db: fa.FaDatabase, key: fa.FaKey, warn: fa.FaExclamationTriangle, check: fa.FaCheckCircle,
    search: fa.FaSearch, cogs: fa.FaCogs, book: fa.FaBook, table: fa.FaTable, clock: fa.FaClock,
    layers: fa.FaLayerGroup, random: fa.FaRandom, balance: fa.FaBalanceScale, eye: fa.FaEyeSlash,
    ledger: fa.FaClipboardList, hourglass: fa.FaHourglassHalf, code: fa.FaCode, calc: fa.FaCalculator,
  };
  for (const [k, I] of Object.entries(need)) icons[k] = await iconPng(I);

  const pres = new pptxgen();
  pres.layout = "LAYOUT_WIDE";
  pres.author = "VFL Scenario B project";
  pres.title = "Privacy-Preserving Vertical Federated Linear Regression";

  // ---------- helpers ----------
  const title = (s, t, sub) => {
    s.addText(t, { x: 0.6, y: 0.35, w: 12.1, h: 0.8, fontFace: HEAD, fontSize: 32, bold: true,
      color: INK, margin: 0, isTextBox: true });
    if (sub) s.addText(sub, { x: 0.6, y: 1.12, w: 12.1, h: 0.45, fontFace: BODY, fontSize: 16,
      color: MUTED, italic: true, margin: 0, isTextBox: true });
  };
  const circleIcon = (s, key, x, y, d, fill = INK) => {
    s.addShape(pres.shapes.OVAL, { x, y, w: d, h: d, fill: { color: fill }, line: { color: fill } });
    const pad = d * 0.24;
    s.addImage({ data: icons[key], x: x + pad, y: y + pad, w: d - 2 * pad, h: d - 2 * pad });
  };
  const numCircle = (s, n, x, y, d, fill = EMBER, fs = 16) => {
    s.addShape(pres.shapes.OVAL, { x, y, w: d, h: d, fill: { color: fill }, line: { color: fill } });
    s.addText(String(n), { x, y, w: d, h: d, align: "center", valign: "middle", fontFace: BODY,
      fontSize: fs, bold: true, color: WHITE, margin: 0, isTextBox: true });
  };
  const card = (s, x, y, w, h, fill = MIST) =>
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, fill: { color: fill }, line: { color: fill }, rectRadius: 0.08 });
  const txt = (s, t, o) => s.addText(t, Object.assign({ fontFace: BODY, fontSize: 15, color: TEXT, margin: 0,
    valign: "top", isTextBox: true }, o));
  const bullets = (s, items, o) => s.addText(items.map((it, i) => ({
    text: typeof it === "string" ? it : it.text,
    options: Object.assign({ bullet: true, breakLine: i < items.length - 1, paraSpaceAfter: 6 },
      typeof it === "string" ? {} : it.options || {}),
  })), Object.assign({ fontFace: BODY, fontSize: 15, color: TEXT, margin: 0, valign: "top", isTextBox: true }, o));
  const chip = (s, t, x, y, w, fill, color = WHITE) => {
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h: 0.32, fill: { color: fill }, line: { color: fill }, rectRadius: 0.08 });
    s.addText(t, { x, y, w, h: 0.32, align: "center", valign: "middle", fontFace: BODY, fontSize: 11, bold: true,
      color, margin: 0, isTextBox: true });
  };
  const img = (s, file, x, y, maxW, maxH, pxW, pxH) => {
    const r = pxW / pxH;
    let w = maxW, h = maxW / r;
    if (h > maxH) { h = maxH; w = maxH * r; }
    s.addImage({ path: file, x: x + (maxW - w) / 2, y, w, h });
    return { w, h };
  };
  const placeholder = (s, x, y, w, h, head, body) => {
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, fill: { color: "FAFBFD" },
      line: { color: SLATE, width: 1.25, dashType: "dash" }, rectRadius: 0.08 });
    circleIcon(s, "hourglass", x + w / 2 - 0.35, y + h / 2 - 0.95, 0.7, SLATE);
    s.addText(head, { x: x + 0.3, y: y + h / 2 - 0.15, w: w - 0.6, h: 0.45, align: "center", fontFace: BODY,
      fontSize: 16, bold: true, color: INK, margin: 0, isTextBox: true });
    s.addText(body, { x: x + 0.4, y: y + h / 2 + 0.3, w: w - 0.8, h: h / 2 - 0.45, align: "center", fontFace: BODY,
      fontSize: 12.5, color: MUTED, margin: 0, valign: "top", isTextBox: true });
  };
  const tableOpts = (colW) => ({ x: 0.6, colW, fontFace: BODY, fontSize: 12.5, color: TEXT, border: { type: "solid", color: LINE, pt: 0.75 },
    valign: "middle", margin: [3, 6, 3, 6] });
  const hdr = (cells) => cells.map((c) => ({ text: c, options: { bold: true, color: WHITE, fill: { color: INK } } }));
  const divider = (n, t, sub) => {
    const s = pres.addSlide(); s.background = { color: INK };
    numCircle(s, n, 0.9, 2.55, 1.3, EMBER, 36);
    s.addText(t, { x: 2.6, y: 2.45, w: 9.5, h: 0.9, fontFace: HEAD, fontSize: 40, bold: true, color: WHITE, margin: 0, isTextBox: true });
    s.addText(sub, { x: 2.6, y: 3.35, w: 9.5, h: 0.6, fontFace: BODY, fontSize: 18, color: "CAD5E6", margin: 0, isTextBox: true });
    return s;
  };

  // ======================= 1. Title =======================
  {
    const s = pres.addSlide(); s.background = { color: INK };
    // Venn motif: two parties, private intersection
    s.addShape(pres.shapes.OVAL, { x: 7.75, y: 1.55, w: 3.3, h: 3.3, fill: { color: SLATE, transparency: 35 }, line: { color: "8FA6C4", width: 1.5 } });
    s.addShape(pres.shapes.OVAL, { x: 9.35, y: 1.55, w: 3.3, h: 3.3, fill: { color: EMBER, transparency: 45 }, line: { color: "E7A77C", width: 1.5 } });
    s.addText("R", { x: 8.1, y: 2.85, w: 0.8, h: 0.7, fontFace: HEAD, fontSize: 30, bold: true, color: WHITE, margin: 0, isTextBox: true });
    s.addText("O", { x: 11.45, y: 2.85, w: 0.8, h: 0.7, fontFace: HEAD, fontSize: 30, bold: true, color: WHITE, margin: 0, isTextBox: true });
    s.addImage({ data: icons.lock, x: 10.15, y: 2.95, w: 0.5, h: 0.5 });
    s.addText("Privacy-Preserving Vertical Federated Linear Regression", { x: 0.8, y: 1.6, w: 6.6, h: 1.9,
      fontFace: HEAD, fontSize: 38, bold: true, color: WHITE, margin: 0, valign: "top", isTextBox: true });
    s.addText("When the organization holds the response: protocol, privacy analysis and evaluation", { x: 0.8, y: 3.6, w: 6.6, h: 0.9,
      fontFace: BODY, fontSize: 18, color: "CAD5E6", margin: 0, valign: "top", isTextBox: true });
    s.addText("Working draft  ·  September 2026", { x: 0.8, y: 6.3, w: 7, h: 0.4, fontFace: BODY, fontSize: 13, color: "8FA6C4", margin: 0, isTextBox: true });
    s.addNotes("Two-party linear regression between a researcher and a national statistical agency, where the agency holds the outcome. We cover the problem, what we found when reviewing the draft protocol, the revised protocol, and results so far.");
  }

  // ======================= 2. Divider: problem =======================
  divider(1, "The problem", "Joint regression across data that cannot be pooled")
    .addNotes("Part 1: motivation, data layout, goal and privacy requirements.");

  // ======================= 3. Motivation =======================
  {
    const s = pres.addSlide(); title(s, "Two datasets, one question, no pooling");
    card(s, 0.6, 1.55, 5.6, 3.55); circleIcon(s, "user", 0.9, 1.85, 0.8, SLATE);
    txt(s, "Researcher  R", { x: 1.9, y: 1.95, w: 4, h: 0.5, fontFace: HEAD, fontSize: 22, bold: true, color: INK });
    bullets(s, ["Smaller cohort with rich domain variables: health, education, surveys",
      "Wants to know whether these predict an administrative outcome",
      "Holds features X_R and identifiers"], { x: 0.95, y: 2.85, w: 5.0, h: 2.1 });
    card(s, 6.6, 1.55, 6.1, 3.55); circleIcon(s, "bank", 6.9, 1.85, 0.8, EMBER);
    txt(s, "Organization  O  (e.g. StatCan)", { x: 7.9, y: 1.95, w: 4.7, h: 0.5, fontFace: HEAD, fontSize: 22, bold: true, color: INK });
    bullets(s, ["Population register: tax filings, income, benefits",
      "Holds the outcome y and features X_O",
      "Cannot release records; law and governance bar direct sharing"], { x: 6.95, y: 2.85, w: 5.5, h: 2.1 });
    txt(s, "6–8 weeks", { x: 0.6, y: 5.45, w: 3.4, h: 0.9, fontFace: HEAD, fontSize: 44, bold: true, color: EMBER });
    txt(s, "of vetting and approvals today before a researcher can link a cohort to agency records in a Research Data Centre. Neither dataset alone answers the question.",
      { x: 4.0, y: 5.55, w: 8.7, h: 0.9, fontSize: 16, color: MUTED });
    s.addNotes("The motivating setting: a researcher cohort and an agency register cover overlapping people but different variables. Today, linking them takes weeks of vetting in an RDC.");
  }

  // ======================= 4. Vertical partitioning =======================
  {
    const s = pres.addSlide(); title(s, "Same people, different variables", "Vertically partitioned data: overlapping rows, disjoint columns");
    const top = 2.1, rowH = 0.25;   // O's first row; R's rows align with O's matched block
    txt(s, "R: cohort (n_R rows)", { x: 0.8, y: top + 3 * rowH - 0.4, w: 3.2, h: 0.35, bold: true, color: SLATE, fontSize: 14 });
    for (let i = 0; i < 12; i++) {
      const matched = i >= 3 && i < 11;
      s.addShape(pres.shapes.RECTANGLE, { x: 0.8, y: top + (i + 3) * rowH, w: 2.6, h: rowH - 0.04,
        fill: { color: matched ? "D6E0EE" : "EEF2F7" }, line: { color: WHITE } });
    }
    txt(s, "X_R", { x: 0.8, y: top + 15 * rowH + 0.05, w: 2.6, h: 0.35, align: "center", color: SLATE, bold: true, fontSize: 14 });
    txt(s, "O: register (n_O rows, n_O ≫ n_R)", { x: 4.3, y: top - 0.4, w: 4.2, h: 0.35, bold: true, color: EMBER, fontSize: 14 });
    for (let k = 0; k < 18; k++) {
      const matched = k >= 6 && k < 14;
      s.addShape(pres.shapes.RECTANGLE, { x: 4.3, y: top + k * rowH, w: 3.2, h: rowH - 0.04,
        fill: { color: matched ? "F3D2BC" : "F8EDE5" }, line: { color: WHITE } });
      s.addShape(pres.shapes.RECTANGLE, { x: 7.55, y: top + k * rowH, w: 0.5, h: rowH - 0.04,
        fill: { color: matched ? "E3A57B" : "F1D9C8" }, line: { color: WHITE } });
    }
    txt(s, "X_O", { x: 4.3, y: top + 18 * rowH + 0.05, w: 3.2, h: 0.35, align: "center", color: EMBER, bold: true, fontSize: 14 });
    txt(s, "y", { x: 7.55, y: top + 18 * rowH + 0.05, w: 0.5, h: 0.35, align: "center", color: EMBER, bold: true, fontSize: 14 });
    s.addShape(pres.shapes.RECTANGLE, { x: 3.55, y: top + 6 * rowH, w: 0.6, h: 8 * rowH - 0.04, fill: { color: INK, transparency: 85 }, line: { color: INK, width: 1, dashType: "dash" } });
    txt(s, "n\nmatched", { x: 3.45, y: top + 9.2 * rowH, w: 0.8, h: 0.8, align: "center", fontSize: 11, bold: true, color: INK });
    // right side text
    card(s, 8.7, 1.55, 4.0, 5.1);
    txt(s, "Goal", { x: 9.0, y: 1.8, w: 3.5, h: 0.4, fontFace: HEAD, fontSize: 20, bold: true, color: INK });
    bullets(s, ["Fit y on [X_R | X_O] over the n matched individuals",
      "Neither party sees the other's raw data",
      "O must not learn which of its records are in R's cohort",
      "The released model must not expose individuals"], { x: 9.0, y: 2.35, w: 3.5, h: 4.1 });
    s.addNotes("Rows are people, columns are variables. The two tables overlap only on the matched individuals; the register is far larger than the cohort.");
  }

  // ======================= 5. Joint OLS =======================
  {
    const s = pres.addSlide(); title(s, "What has to be computed", "OLS on the matched join needs a Gram matrix that spans both parties");
    txt(s, "β̂ = (ZᵀZ)⁻¹ Zᵀy,    Z = [ X_Rᴵ | X_Oᴵ ]", { x: 0.6, y: 1.75, w: 7, h: 0.6, fontFace: HEAD, fontSize: 24, color: INK });
    const gx = 0.9, gy = 2.7, c = 1.55;
    const cell = (x, y, label, sub, fill, color = INK, sup) => {
      s.addShape(pres.shapes.RECTANGLE, { x, y, w: c, h: c, fill: { color: fill }, line: { color: WHITE, width: 2 } });
      const head = [{ text: label, options: { fontFace: HEAD, fontSize: 24, bold: true, color, breakLine: !sup } }];
      if (sup) head.push({ text: sup, options: { fontFace: HEAD, fontSize: 24, bold: true, color, superscript: true, breakLine: true } });
      s.addText([...head,
                 { text: sub, options: { fontFace: BODY, fontSize: 11, color } }],
        { x, y, w: c, h: c, align: "center", valign: "middle", margin: 2, isTextBox: true });
    };
    cell(gx, gy, "A", "X_Rᵀ X_R", "D6E0EE");
    cell(gx + c, gy, "C", "X_Rᵀ X_O", EMBER, WHITE);
    cell(gx, gy + c, "C", "", EMBER, WHITE, "T");
    cell(gx + c, gy + c, "B", "X_Oᵀ X_O", "F3D2BC");
    cell(gx + 2 * c + 0.4, gy, "c_R", "X_Rᵀ y", EMBER, WHITE);
    cell(gx + 2 * c + 0.4, gy + c, "c_O", "X_Oᵀ y", "F3D2BC");
    txt(s, "ZᵀZ", { x: gx, y: gy + 2 * c + 0.1, w: 2 * c, h: 0.4, align: "center", fontFace: HEAD, fontSize: 16, color: MUTED });
    txt(s, "Zᵀy", { x: gx + 2 * c + 0.4, y: gy + 2 * c + 0.1, w: c, h: 0.4, align: "center", fontFace: HEAD, fontSize: 16, color: MUTED });
    // legend
    const lg = [["D6E0EE", "R alone: exact, never leaves R"], ["F3D2BC", "O's data, but only on matched rows (needs b)"], [EMBER, "Cross-party: needs both parties' data"]];
    lg.forEach(([col, t], i) => {
      s.addShape(pres.shapes.RECTANGLE, { x: 6.7, y: 2.75 + i * 0.62, w: 0.4, h: 0.4, fill: { color: col }, line: { color: col } });
      txt(s, t, { x: 7.25, y: 2.78 + i * 0.62, w: 5.4, h: 0.4, fontSize: 15 });
    });
    card(s, 6.7, 4.8, 6.0, 1.6, EMBER_L);
    txt(s, "Only A is purely R's. Everything that touches O's data is computed under R's encryption key and noised by O before R sees it.",
      { x: 6.95, y: 5.0, w: 5.5, h: 1.3, fontSize: 16, color: INK });
    s.addNotes("The Gram matrix splits into blocks. A is the researcher's own; B and c_O are the organization's data restricted to matched rows; C and c_R are genuinely cross-party.");
  }

  // ======================= 6. Threat model =======================
  {
    const s = pres.addSlide(); title(s, "Three guarantees, honest-but-curious parties");
    const items = [
      ["eye", "Input privacy", "Each party's view can be simulated from its own input and the agreed output. Neither sees the other's raw data.", SLATE],
      ["link", "Linkage privacy", "O learns nothing about which of its records match R's cohort (only public sizes).", TEAL],
      ["shield", "Output privacy", "What R receives is differentially private for each matched person's O-side data (x_O, y), with R's features held fixed (replace-one).", EMBER],
    ];
    items.forEach(([ic, h, b, col], i) => {
      const x = 0.6 + i * 4.15;
      card(s, x, 1.6, 3.9, 3.6);
      circleIcon(s, ic, x + 0.3, 1.9, 0.85, col);
      txt(s, h, { x: x + 0.3, y: 2.95, w: 3.3, h: 0.5, fontFace: HEAD, fontSize: 21, bold: true, color: INK });
      txt(s, b, { x: x + 0.3, y: 3.5, w: 3.35, h: 1.6, fontSize: 14.5 });
    });
    card(s, 0.6, 5.55, 12.1, 1.25, "F7F9FC");
    circleIcon(s, "key", 0.85, 5.78, 0.8, INK);
    txt(s, "Parties follow the protocol but analyse everything they receive. R holds all decryption keys and learns the match set by design; O controls the privacy budget.",
      { x: 1.9, y: 5.8, w: 10.6, h: 0.85, fontSize: 15.5, color: INK, valign: "middle" });
    s.addNotes("Threat model: semi-honest. Three guarantees: cryptographic input privacy, linkage privacy for the organization, and differential privacy on what the researcher receives.");
  }

  // ======================= 7. Positioning =======================
  {
    const s = pres.addSlide(); title(s, "Where this work sits");
    const rows = [
      hdr(["Approach", "Where the data sit", "Record linkage", "Output privacy", "Gap for our setting"]),
      ["Central DP regression (AdaSSP, Wang 2018)", "One trusted curator holds the joined table", "Not needed", "DP", "Assumes the pooled data exist"],
      ["VFL with HE (e.g. Yang et al. 2019)", "Split across parties", "Assumed or separate PSI", "Usually none", "Model can leak individuals"],
      ["Private set intersection (Chen–Laine–Rindal)", "Split across parties", "Yes", "Not applicable", "Linkage only, no model"],
      [{ text: "This work", options: { bold: true, color: EMBER } }, "Split; never pooled", "Labeled PSI, private to O", "DP, budget controlled by O", { text: "Narrower guarantee: protects O's side; R learns membership", options: { color: MUTED } }],
    ];
    s.addTable(rows, Object.assign(tableOpts([3.0, 2.55, 2.1, 2.05, 2.4]), { y: 1.65, rowH: 0.62, fontSize: 13.5 }));
    card(s, 0.6, 5.35, 12.1, 1.35, EMBER_L);
    txt(s, "Same statistical engine as AdaSSP (perturb the sufficient statistics, then solve a ridge), delivered without pooling, and with R's own block kept exact.",
      { x: 0.9, y: 5.55, w: 11.5, h: 1.0, fontSize: 16.5, color: INK, valign: "middle" });
    s.addNotes("Central DP regression assumes a pooled table; VFL-with-HE work usually lacks output privacy; PSI gives linkage only. We combine private linkage, encrypted aggregation and DP.");
  }

  // ======================= 8. Divider: our work =======================
  divider(2, "What we did", "Reviewed the draft, found what breaks, and rebuilt it")
    .addNotes("Part 2: the work we did and what we found.");

  // ======================= 9. Timeline =======================
  {
    const s = pres.addSlide(); title(s, "From draft to revised protocol");
    const steps = [
      ["search", "Review", "Draft paper and code checked line by line"],
      ["cogs", "Run", "End-to-end over sockets: plaintext and real OpenFHE"],
      ["secret", "Attack", "Three privacy reviews plus an independent critique"],
      ["layers", "Redesign", "Revised protocol, TikZ figure, rewritten paper"],
      ["chart", "Simulate", "Accuracy and bias-correction studies"],
      ["balance", "Benchmark", "AdaSSP on Wang's UCI datasets (set up, not yet run)"],
    ];
    const y = 2.55, gap = 2.02;
    s.addShape(pres.shapes.LINE, { x: 1.05, y: y + 0.45, w: gap * 5, h: 0, line: { color: LINE, width: 2 } });
    steps.forEach(([ic, h, b], i) => {
      const x = 0.6 + i * gap;
      circleIcon(s, ic, x, y, 0.9, i === 5 ? SLATE : (i === 2 ? EMBER : INK));
      txt(s, `${i + 1}. ${h}`, { x: x - 0.25, y: y + 1.15, w: 1.9, h: 0.45, fontFace: HEAD, fontSize: 17, bold: true, color: INK });
      txt(s, b, { x: x - 0.25, y: y + 1.65, w: 1.85, h: 1.5, fontSize: 13.5, color: MUTED });
    });
    chip(s, "Next", 0.6 + 5 * gap - 0.2, y - 0.6, 1.3, SLATE);
    s.addNotes("The sequence of work: review, running the implementation, adversarial privacy reviews, redesign, simulations, and the AdaSSP benchmark, which is set up but not yet run.");
  }

  // ======================= 10. Critical finding =======================
  {
    const s = pres.addSlide(); title(s, "Critical finding: the noise did not protect the ciphertext");
    txt(s, "100%", { x: 0.6, y: 1.65, w: 3.4, h: 1.2, fontFace: HEAD, fontSize: 64, bold: true, color: EMBER });
    txt(s, "of trials in which R identified the true dataset from O's reply: matched and unmatched records alike (real OpenFHE, 20/20).",
      { x: 0.6, y: 2.9, w: 3.6, h: 1.4, fontSize: 14.5 });
    txt(s, "73%", { x: 0.6, y: 4.35, w: 3.4, h: 1.0, fontFace: HEAD, fontSize: 48, bold: true, color: SLATE });
    txt(s, "the most that ε = 1 differential privacy allows.", { x: 0.6, y: 5.35, w: 3.6, h: 0.8, fontSize: 14.5 });
    card(s, 4.7, 1.65, 3.85, 4.6); txt(s, "Why", { x: 4.95, y: 1.85, w: 3.4, h: 0.45, fontFace: HEAD, fontSize: 20, bold: true, color: INK });
    bullets(s, ["The draft added the noise as a plaintext scalar, which changes only c₀",
      "c₁ is a deterministic function of R's own ciphertexts, keys and O's data",
      "R, who holds the key, recomputes c₁ for each candidate dataset and compares"], { x: 4.95, y: 2.4, w: 3.4, h: 3.7, fontSize: 14.5 });
    card(s, 8.85, 1.65, 3.85, 4.6, EMBER_L); txt(s, "Fix", { x: 9.1, y: 1.85, w: 3.4, h: 0.45, fontFace: HEAD, fontSize: 20, bold: true, color: EMBER });
    bullets(s, ["Add a fresh encryption of the noise, which re-randomizes c₁",
      "Switch to the last modulus level, then flood the error term",
      "BFV linkage replies: statistical flooding; CKKS: DP-calibrated flooding"], { x: 9.1, y: 2.4, w: 3.4, h: 3.7, fontSize: 14.5 });
    s.addNotes("The most important finding. The DP argument reasoned about the decrypted value, but R holds the key and sees the whole ciphertext. With plaintext-scalar noise, the c1 component is independent of the noise.");
  }

  // ======================= 11. Review scorecard =======================
  {
    const s = pres.addSlide(); title(s, "Review scorecard");
    const sev = (t, col) => ({ text: t, options: { bold: true, color: col } });
    const rows = [
      hdr(["Finding", "Severity", "Status in the revised protocol"]),
      ["Plaintext-scalar noise leaks through the ciphertext", sev("Critical", EMBER), "Fixed: fresh encryption plus flooding"],
      ["Calibration step shares raw-data statistics", sev("Critical", EMBER), "Fixed: per-party commitments, public constants only"],
      ["Add/remove sensitivity vs replace-one adjacency (true ε up to ~2× the claim)", sev("High", "B7791F"), "Fixed: replace-one Δ, closed form"],
      ["No privacy ledger across runs", sev("High", "B7791F"), "Fixed: zCDP ledger at O"],
      ["PSI reveals O's row order and bin loads", sev("High", "B7791F"), "Fixed: secret permutation, public padding"],
      ["Adaptive λ cancels the bias correction", sev("Validity", SLATE), "Fixed: λ set from public inputs"],
      ["CKKS slot leak; HE parameter security", sev("Checked", TEAL), "Not an issue with OpenFHE"],
      ["R learns which of its cohort are in the register", sev("By design", MUTED), "Stated in Definition 4"],
    ];
    s.addTable(rows, Object.assign(tableOpts([6.0, 1.5, 4.6]), { y: 1.5, rowH: 0.56, fontSize: 13.5 }));
    s.addNotes("Summary of findings across the code review, three privacy reviews and an independent critique, with how the revised protocol handles each.");
  }

  // ======================= 12. Contributions =======================
  {
    const s = pres.addSlide(); title(s, "Contributions");
    const items = [
      ["Corrected data flow", "R's Gram block A is computed locally, is never sent and never noised. The sensitivity loses its B_R⁴ term."],
      ["Revised protocol", "Sanitized replies, replace-one DP, zCDP ledger, per-party calibration, hardened PSI, ridge fixed from public inputs."],
      ["Theory", "Finite-noise bias as a conditional expectation, a zero-budget correction, and a closed-form replace-one sensitivity."],
      ["Evaluation", "End-to-end runs, simulation studies of accuracy and of the correction, and a benchmark against AdaSSP."],
    ];
    items.forEach(([h, b], i) => {
      const x = 0.6 + (i % 2) * 6.15, y = 1.6 + Math.floor(i / 2) * 2.6;
      card(s, x, y, 5.95, 2.35);
      numCircle(s, i + 1, x + 0.3, y + 0.3, 0.7, i === 1 ? EMBER : INK, 20);
      txt(s, h, { x: x + 1.2, y: y + 0.38, w: 4.5, h: 0.5, fontFace: HEAD, fontSize: 21, bold: true, color: INK });
      txt(s, b, { x: x + 1.2, y: y + 0.95, w: 4.5, h: 1.3, fontSize: 15 });
    });
    s.addNotes("Four contributions: the corrected data flow, the revised protocol, the theory, and the evaluation.");
  }

  // ======================= 13. Divider: methodology =======================
  divider(3, "Methodology", "The revised protocol, phase by phase")
    .addNotes("Part 3: how the revised protocol works.");

  // ======================= 14. Protocol at a glance =======================
  {
    const s = pres.addSlide(); title(s, "The protocol at a glance");
    const ph = [["0", "Setup and commitments", "Only public constants cross"], ["1", "Private linkage", "BFV labeled PSI; R learns the match set"],
      ["2", "Encrypted aggregation", "CKKS, depth 1; DP noise added by O"], ["3", "Solve at R", "Fixed ridge, bias correction; post-processing"]];
    ph.forEach(([n, h, b], i) => {
      const y = 1.7 + i * 1.3;
      numCircle(s, n, 0.6, y, 0.7, i === 2 ? EMBER : INK, 20);
      txt(s, h, { x: 1.5, y: y + 0.02, w: 5.0, h: 0.45, fontFace: HEAD, fontSize: 20, bold: true, color: INK });
      txt(s, b, { x: 1.5, y: y + 0.5, w: 5.0, h: 0.5, fontSize: 15, color: MUTED });
    });
    txt(s, "Orange in the figure marks what changed from the draft.", { x: 0.6, y: 6.8, w: 6, h: 0.35, fontSize: 12, italic: true, color: MUTED });
    img(s, P("protocol_v2", "protocol_v2.png"), 7.2, 0.3, 5.6, 6.95, 1817, 2394);
    s.addNotes("Four phases. The figure on the right is the full message flow with the risks each step fixes.");
  }

  // ======================= 15. Setup & linkage =======================
  {
    const s = pres.addSlide(); title(s, "Phases 0–1: commitments and private linkage");
    card(s, 0.6, 1.55, 5.95, 5.2);
    numCircle(s, 0, 0.9, 1.8, 0.6, INK, 16);
    txt(s, "Setup and commitments", { x: 1.7, y: 1.83, w: 4.6, h: 0.5, fontFace: HEAD, fontSize: 20, bold: true, color: INK });
    bullets(s, [
      { text: "R: standardizes with its own constants (kept), clips rows to B_R, generates BFV and CKKS keys" },
      { text: "O: standardizes with public constants, clips to B_O and B_y" },
      { text: "O computes Δ_rep and σ, and refuses the run if the ledger would exceed its cap" },
      { text: "Only public constants cross: no quantiles, clip rates or sample moments", options: { bold: true } },
    ], { x: 0.95, y: 2.6, w: 5.35, h: 4.0, fontSize: 14.5 });
    card(s, 6.75, 1.55, 5.95, 5.2);
    numCircle(s, 1, 7.05, 1.8, 0.6, INK, 16);
    txt(s, "Labeled PSI with alignment", { x: 7.85, y: 1.83, w: 4.6, h: 0.5, fontFace: HEAD, fontSize: 20, bold: true, color: INK });
    bullets(s, [
      "O applies a secret permutation τ, so labels are random positions",
      "Keyed hashing; bins padded to a public degree D and count α",
      "R cuckoo-hashes its IDs and sends encrypted windowed powers",
      "O evaluates membership and label polynomials, masks, and sanitizes",
      "R learns I and the aligned positions; O learns only public sizes",
    ], { x: 7.1, y: 2.6, w: 5.35, h: 4.0, fontSize: 14.5 });
    s.addNotes("Phase 0 exchanges only committed public constants. Phase 1 is labeled PSI in the Chen–Laine–Rindal style, hardened with a secret permutation, public padding and flooded replies.");
  }

  // ======================= 16. Aggregation =======================
  {
    const s = pres.addSlide(); title(s, "Phase 2: encrypted aggregation", "O multiplies R's encrypted vectors by its own plaintext columns (depth 1)");
    card(s, 0.6, 1.8, 7.1, 3.35, "F7F9FC");
    const eq = [
      "Enc(B_jk)   = Σᵢ Enc(bᵢ) · X_O,ij · X_O,ik",
      "Enc(C_jk)   = Σᵢ Enc(Ẋ_R,ij) · X_O,ik",
      "Enc(c_O,k) = Σᵢ Enc(bᵢ) · X_O,ik · yᵢ",
      "Enc(c_R,j) = Σᵢ Enc(Ẋ_R,ij) · yᵢ",
    ];
    s.addText(eq.map((e, i) => ({ text: e, options: { breakLine: i < eq.length - 1 } })),
      { x: 0.95, y: 2.0, w: 6.6, h: 2.3, fontFace: HEAD, fontSize: 19, color: INK, margin: 0, valign: "top", paraSpaceAfter: 10, isTextBox: true });
    txt(s, "b: selection vector;  Ẋ_R: R's features in O's order (zero off the match set)", { x: 0.95, y: 4.45, w: 6.6, h: 0.5, fontSize: 12.5, color: MUTED, italic: true });
    circleIcon(s, "lock", 8.1, 1.85, 0.75, SLATE);
    txt(s, "A = Ẋ_Rᵀ Ẋ_R stays with R", { x: 9.05, y: 1.95, w: 3.7, h: 0.5, fontSize: 17, bold: true, color: INK });
    txt(s, "Exact, never transmitted, never noised.", { x: 9.05, y: 2.45, w: 3.7, h: 0.5, fontSize: 14, color: MUTED });
    circleIcon(s, "random", 8.1, 3.3, 0.75, EMBER);
    txt(s, "One noise draw per entry", { x: 9.05, y: 3.4, w: 3.7, h: 0.5, fontSize: 17, bold: true, color: INK });
    txt(s, "z is the same in every CKKS slot. Independent noise per slot could be averaged away.", { x: 9.05, y: 3.9, w: 3.7, h: 1.0, fontSize: 14, color: MUTED });
    card(s, 0.6, 5.5, 12.1, 1.2, EMBER_L);
    txt(s, "Each released entry:  ct ← Sanitize_CKKS( ct ⊞ Enc_pk(z) ),   z ~ discrete Gaussian N_ℤ(0, σ²) from a CSPRNG",
      { x: 0.9, y: 5.62, w: 11.5, h: 0.95, fontSize: 16.5, color: INK, valign: "middle" });
    s.addNotes("The four O-dependent blocks are depth-one inner products. The DP noise is a single value per released entry, replicated across slots, added by fresh encryption and then sanitized.");
  }

  // ======================= 17. DP calibration =======================
  {
    const s = pres.addSlide(); title(s, "Calibrating the noise, and keeping count");
    txt(s, "Replace-one sensitivity", { x: 0.6, y: 1.55, w: 6.2, h: 0.45, fontFace: HEAD, fontSize: 20, bold: true, color: INK });
    txt(s, "The adjacency replaces one matched person's (x_O, y). The draft used the add/remove bound, which understates Δ.",
      { x: 0.6, y: 2.05, w: 6.2, h: 0.8, fontSize: 14.5, color: MUTED });
    const rows = [hdr(["(B_R, B_O, B_y)", "Δ_add", "Δ_rep", "Ratio"]),
      ["(1, 1, 1)", "2.24", "3.16", "1.41"], ["(3.16, 3, 3)", "20.6", "29.4", "1.43"],
      ["(√5, √5, 5)", "30.4", "31.6", "1.04"], ["Simulation population", "58.2", "86.2", "1.48"]];
    s.addTable(rows, Object.assign(tableOpts([2.3, 1.35, 1.45, 1.1]), { y: 2.95, rowH: 0.46, fontSize: 13 }));
    txt(s, "Δ_add ≤ Δ_rep ≤ 2Δ_add. Releasing yᵀy costs nothing under replace-one.", { x: 0.6, y: 5.55, w: 6.2, h: 0.6, fontSize: 13, italic: true, color: MUTED });
    card(s, 7.2, 1.55, 5.5, 2.35);
    circleIcon(s, "calc", 7.45, 1.8, 0.7, SLATE);
    txt(s, "σ from the analytic Gaussian", { x: 8.35, y: 1.88, w: 4.2, h: 0.45, fontSize: 17, bold: true, color: INK });
    txt(s, "Smallest σ meeting (ε, δ) for the whole stacked release (Balle–Wang). Discrete Gaussian at the CKKS resolution.",
      { x: 8.35, y: 2.38, w: 4.15, h: 1.4, fontSize: 14 });
    card(s, 7.2, 4.15, 5.5, 2.55, EMBER_L);
    circleIcon(s, "ledger", 7.45, 4.4, 0.7, EMBER);
    txt(s, "zCDP ledger at O", { x: 8.35, y: 4.48, w: 4.2, h: 0.45, fontSize: 17, bold: true, color: INK });
    txt(s, "ρ_run = Δ²/2σ² + ρ_err. O refuses a run once the cumulative spend would pass its cap. Re-runs, retraining and inference queries all count.",
      { x: 8.35, y: 4.98, w: 4.15, h: 1.6, fontSize: 14 });
    s.addNotes("The draft's sensitivity bound was for add/remove adjacency; our adjacency is replace-one, which needs up to twice the Δ. The ledger makes re-runs compose under an explicit cap.");
  }

  // ======================= 18. Circuit privacy =======================
  {
    const s = pres.addSlide(); title(s, "Sanitizing O's replies (circuit privacy)", "R holds the key, so every reply must reveal nothing beyond its decrypted value");
    const steps = [
      ["1", "Fresh encryption", "ct ⊞ Enc_pk(z) makes c₁ computationally uniform: it no longer depends on O's data."],
      ["2", "Modulus switch", "Drop to the last level, which scales down the data-dependent error."],
      ["3", "Flood the error", "BFV: statistical flooding with N·2^−κ ≤ 2^−40. CKKS: DP-calibrated flooding with width ς = S_err / √(2ρ_err)."],
    ];
    steps.forEach(([n, h, b], i) => {
      const x = 0.6 + i * 4.15;
      card(s, x, 1.95, 3.9, 3.0);
      numCircle(s, n, x + 0.3, 2.2, 0.65, EMBER, 18);
      txt(s, h, { x: x + 1.1, y: 2.28, w: 2.7, h: 0.5, fontFace: HEAD, fontSize: 19, bold: true, color: INK });
      txt(s, b, { x: x + 0.3, y: 3.05, w: 3.35, h: 1.8, fontSize: 14.5 });
    });
    card(s, 0.6, 5.25, 6.0, 1.45, "F7F9FC");
    txt(s, "Why DP-calibrated flooding for CKKS", { x: 0.85, y: 5.38, w: 5.6, h: 0.4, fontSize: 15, bold: true, color: INK });
    txt(s, "Classical 2⁴⁰ flooding would swamp CKKS precision. A DP-sized flood needs only a few bits above the error; its cost joins the ledger.",
      { x: 0.85, y: 5.8, w: 5.6, h: 0.85, fontSize: 13.5, color: MUTED });
    card(s, 6.85, 5.25, 5.85, 1.45, EMBER_L);
    txt(s, "Open proof obligation", { x: 7.1, y: 5.38, w: 5.4, h: 0.4, fontSize: 15, bold: true, color: EMBER });
    txt(s, "An explicit bound on S_err, how much one record can change the ciphertext error, for the exact encoding and key switching.",
      { x: 7.1, y: 5.8, w: 5.4, h: 0.85, fontSize: 13.5, color: INK });
    s.addNotes("Three sanitization steps. The CKKS flooding is sized as a DP mechanism in the style of Li, Micciancio, Schultz and Sorrell, so it costs a small ledger entry instead of 40 bits of precision.");
  }

  // ======================= 19. Solve & bias correction =======================
  {
    const s = pres.addSlide(); title(s, "Phase 3: fixed ridge and zero-budget bias correction");
    card(s, 0.6, 1.55, 6.0, 5.15, "F7F9FC");
    const lines = [
      ["Ridge, fixed before decryption", "λ = max{ 0, 2ρ*σ√p − nℓ },  Ψ = I"],
      ["Release gate", "ρ̂ = λ_min(G̃ + λI) / (2σ√p) ≥ 1"],
      ["Private ridge estimate", "β̃_λ = (G̃ + λI)⁻¹ c̃"],
      ["Bias-corrected estimate", "β̂_bc = β̃_λ − σ² M(P̃_λ) β̃_λ"],
    ];
    lines.forEach(([h, e], i) => {
      txt(s, h, { x: 0.9, y: 1.8 + i * 1.2, w: 5.5, h: 0.4, fontSize: 14, bold: true, color: SLATE });
      txt(s, e, { x: 0.9, y: 2.2 + i * 1.2, w: 5.5, h: 0.55, fontFace: HEAD, fontSize: 18, color: INK });
    });
    card(s, 6.85, 1.55, 5.85, 3.1);
    txt(s, "Theorem (conditional bias)", { x: 7.1, y: 1.75, w: 5.4, h: 0.45, fontFace: HEAD, fontSize: 18, bold: true, color: INK });
    txt(s, "E[ β̃ | ℰ_η ] − β̂ = σ² M(P) β̂ + O(σ⁴) + O(exp(−p(ηρ−1)²/2))", { x: 7.1, y: 2.3, w: 5.45, h: 0.6, fontFace: HEAD, fontSize: 15.5, color: INK });
    txt(s, "M(P) = (P² + tr(P)P − P diag(P)) Π_O + (P Π_O P + tr(Π_O P) P) Π_R", { x: 7.1, y: 3.0, w: 5.45, h: 0.6, fontFace: HEAD, fontSize: 13.5, color: MUTED });
    txt(s, "The plain mean does not exist, hence the conditioning on ℰ_η.", { x: 7.1, y: 3.8, w: 5.45, h: 0.7, fontSize: 13, italic: true, color: MUTED });
    card(s, 6.85, 4.9, 5.85, 1.8, EMBER_L);
    txt(s, "Why λ must not depend on the noise", { x: 7.1, y: 5.02, w: 5.4, h: 0.4, fontSize: 15, bold: true, color: EMBER });
    txt(s, "Chosen from G̃, λ moves with the noise. The correction then removed none of the bias (1.0–1.6× left); with fixed λ it removes 87–98%.",
      { x: 7.1, y: 5.45, w: 5.45, h: 1.15, fontSize: 13.5, color: INK });
    s.addNotes("R fixes the ridge from public inputs, decrypts, splices in its exact block, solves, and applies the second-order bias correction. The theorem is stated conditionally because the unconditional mean does not exist.");
  }

  // ======================= 20. Evaluation design =======================
  {
    const s = pres.addSlide(); title(s, "How we evaluate");
    const cards = [
      ["db", "Simulation population", "Features on real-world scales (income, age, binary indicators, counts), correlated. Public standardization constants; committed 99.5% bounds; R² ∈ {0.3, 0.5, 0.9}."],
      ["flask", "Monte Carlo", "n from 100 to 100k matched records; ε ∈ {0.5, 1, 2, 4, 8}, δ = 10⁻⁵; 200 replicates, each with new data and a new DP release."],
      ["chart", "Metrics", "RMSE against the true β (standardized); per-coefficient efficiency ratio; interval coverage; MSE ratio corrected/uncorrected."],
      ["book", "Benchmark", "Wang's 36 UCI regression datasets; AdaSSP as published and under our guarantee; test MSE, as in Wang (2018)."],
    ];
    cards.forEach(([ic, h, b], i) => {
      const x = 0.6 + (i % 2) * 6.15, y = 1.55 + Math.floor(i / 2) * 2.65;
      card(s, x, y, 5.95, 2.4);
      circleIcon(s, ic, x + 0.3, y + 0.3, 0.75, i === 3 ? EMBER : SLATE);
      txt(s, h, { x: x + 1.25, y: y + 0.38, w: 4.5, h: 0.45, fontFace: HEAD, fontSize: 19, bold: true, color: INK });
      txt(s, b, { x: x + 1.25, y: y + 0.9, w: 4.5, h: 1.45, fontSize: 14 });
    });
    s.addNotes("Simulations use a known population so the true beta, public constants and committed bounds are all data-independent. The benchmark uses Wang's datasets.");
  }

  // ======================= 21. Divider: results =======================
  divider(4, "Results", "What we have so far, and what is still to run")
    .addNotes("Part 4: results. Slides marked as placeholders are still to be run.");

  // ======================= 22. Correctness =======================
  {
    const s = pres.addSlide(); title(s, "The cryptographic layer is exact", "Draft protocol, run end-to-end over sockets with plaintext and real OpenFHE backends");
    const stats = [["0", "misaligned rows", "PSI recovered exactly the true match set in all 6 cohorts, plaintext and OpenFHE"],
      ["≈10⁻¹⁰", "max |β − β_OLS|", "with the noise switched off: encryption adds no error"],
      ["13.8 s", "end-to-end on OpenFHE", "n_O = 4,000 records, n = 400 matched (plaintext backend: 3.2 s)"]];
    stats.forEach(([v, l, d], i) => {
      const x = 0.6 + i * 4.15;
      card(s, x, 1.9, 3.9, 3.6);
      txt(s, v, { x: x + 0.3, y: 2.15, w: 3.4, h: 1.1, fontFace: HEAD, fontSize: 48, bold: true, color: i === 1 ? EMBER : INK });
      txt(s, l, { x: x + 0.3, y: 3.3, w: 3.4, h: 0.45, fontSize: 16, bold: true, color: SLATE });
      txt(s, d, { x: x + 0.3, y: 3.8, w: 3.35, h: 1.5, fontSize: 14, color: MUTED });
    });
    txt(s, "All error comes from the DP noise and the ridge it forces. The revised protocol's sanitization has not been implemented in OpenFHE yet (see the placeholder slide).",
      { x: 0.6, y: 5.85, w: 12.1, h: 0.8, fontSize: 14.5, italic: true, color: MUTED });
    s.addNotes("Correctness of linkage and aggregation: exact alignment and 1e-10 agreement with plaintext OLS when noise is off.");
  }

  // ======================= 23. Figure A =======================
  {
    const s = pres.addSlide(); title(s, "Accuracy against intersection size");
    img(s, P("reports", "coef_accuracy", "figA_rmse_vs_n_r2_0.5.png"), 0.4, 1.3, 8.5, 5.9, 1904, 1224);
    card(s, 9.1, 1.4, 3.65, 5.4);
    txt(s, "Within 2× of non-private OLS at", { x: 9.35, y: 1.6, w: 3.2, h: 0.7, fontSize: 15, bold: true, color: INK });
    const cross = [["ε = 8", "n ≈ 3.2k"], ["ε = 4", "n ≈ 9.4k"], ["ε = 2", "n ≈ 34k"], ["ε = 1", "n ≈ 105k"]];
    cross.forEach(([e, n], i) => {
      txt(s, e, { x: 9.35, y: 2.35 + i * 0.5, w: 1.2, h: 0.45, fontSize: 15, color: MUTED });
      txt(s, n, { x: 10.6, y: 2.35 + i * 0.5, w: 2.0, h: 0.45, fontSize: 15, bold: true, color: EMBER });
    });
    bullets(s, ["Small n: the ridge shrinks β̂ towards 0", "Then the DP error falls like 1/n and merges into the OLS floor",
      "Run-to-run spread is about ×2.5–3"], { x: 9.35, y: 4.5, w: 3.2, h: 2.2, fontSize: 13.5 });
    s.addNotes("Median RMSE over 200 runs, with the 2.5–97.5% band across runs and the non-private OLS floor. R-squared 0.5.");
  }

  // ======================= 24. Key numbers table =======================
  {
    const s = pres.addSlide(); title(s, "Key numbers", "Median RMSE of the standardized slopes against the true β (ratio to the OLS floor), R² = 0.5");
    const rows = [hdr(["n", "ε = 0.5", "ε = 1", "ε = 2", "ε = 4", "ε = 8", "OLS floor"]),
      ["1,000", "0.231 (9.9×)", "0.213 (9.1×)", "0.181 (7.7×)", "0.138 (5.9×)", "0.078 (3.3×)", "0.023"],
      ["10,000", "0.092 (12×)", "0.044 (5.8×)", "0.025 (3.4×)", "0.014 (1.9×)", "0.010 (1.3×)", "0.0075"],
      ["100,000", "0.0087 (3.4×)", "0.0051 (2.0×)", "0.0034 (1.4×)", "0.0029 (1.2×)", "0.0026 (1.0×)", "0.0025"]];
    s.addTable(rows, Object.assign(tableOpts([1.3, 1.8, 1.8, 1.8, 1.8, 1.8, 1.8]), { y: 1.85, rowH: 0.55, fontSize: 14 }));
    txt(s, "n at which VFL comes within 2× of the OLS floor", { x: 0.6, y: 4.3, w: 8, h: 0.45, fontFace: HEAD, fontSize: 18, bold: true, color: INK });
    const rows2 = [hdr(["R²", "ε = 0.5", "ε = 1", "ε = 2", "ε = 4", "ε = 8"]),
      ["0.3", "≈240k*", "75k", "21k", "7.1k", "1.9k"], ["0.5", "≈380k*", "≈105k*", "34k", "9.4k", "3.2k"], ["0.9", ">100k", ">100k", "66k", "30k", "13k"]];
    s.addTable(rows2, Object.assign(tableOpts([1.3, 1.8, 1.8, 1.8, 1.8, 1.8]), { y: 4.85, rowH: 0.42, fontSize: 13.5 }));
    txt(s, "* extrapolated beyond n = 100k", { x: 0.6, y: 6.65, w: 6, h: 0.35, fontSize: 12, italic: true, color: MUTED });
    s.addNotes("Numbers behind the figure. A stronger signal lowers the OLS floor, but the DP part is set by the committed bounds, so the ratio is largest at high R-squared.");
  }

  // ======================= 25. Scale-free comparison =======================
  {
    const s = pres.addSlide(); title(s, "Comparing coefficients on different scales");
    img(s, P("reports", "coef_accuracy", "figB1_efficiency_ratio_r2_0.5.png"), 0.4, 1.25, 12.5, 4.1, 3094, 1325);
    const pts = [["Raw scale", "Coefficients span 5.6 orders of magnitude (≈$0.2 per $ income vs ≈$10k per binary). A pooled raw RMSE is 87% driven by two binary coefficients."],
      ["Efficiency ratio", "RMSE_DP,j / SE_OLS,j: unit-free. A ratio r means the release is about as accurate as OLS on n/r² records."],
      ["Result", "Once the ridge is off, DP costs about the same for every coefficient (±15%); O-side coefficients run ~10% higher."]];
    pts.forEach(([h, b], i) => {
      const x = 0.6 + i * 4.15;
      txt(s, h, { x, y: 5.5, w: 3.9, h: 0.4, fontSize: 15, bold: true, color: i === 1 ? EMBER : INK });
      txt(s, b, { x, y: 5.9, w: 3.9, h: 1.3, fontSize: 13, color: MUTED });
    });
    s.addNotes("Per-coefficient view. Raw RMSE is meaningless across coefficients because of units; the efficiency ratio compares each coefficient against its own sampling error.");
  }

  // ======================= 26. Bias correction =======================
  {
    const s = pres.addSlide(); title(s, "Does the bias correction matter?");
    img(s, P("reports", "bias_correction", "fig1_mse_ratio.png"), 0.4, 1.2, 12.5, 3.85, 2291, 705);
    const st = [["−7 to −12%", "MSE vs the protocol target when the ridge is on. The correction also shrinks noise; it never raised this MSE (90 settings)."],
      ["+0.3 to +4.3%", "MSE vs OLS under a moderate ridge: the correction restores shrinkage that noise had offset."],
      ["≥ 130", "averaged releases before the bias alone would matter; one release: at most 0.75% of MSE."]];
    st.forEach(([v, b], i) => {
      const x = 0.6 + i * 4.15;
      txt(s, v, { x, y: 5.15, w: 3.9, h: 0.7, fontFace: HEAD, fontSize: 30, bold: true, color: i === 0 ? EMBER : INK });
      txt(s, b, { x, y: 5.85, w: 3.9, h: 1.35, fontSize: 13, color: MUTED });
    });
    s.addNotes("Answer to the reviewer: not as a bias correction; the removed bias is tiny. It is still worth applying as a free variance reduction. Rule: apply to every release that passes the gate.");
  }

  // ======================= 27. Error anatomy & inference =======================
  {
    const s = pres.addSlide(); title(s, "Where the error comes from when σ is large");
    img(s, P("reports", "bias_correction", "fig2_decomposition.png"), 0.4, 1.25, 12.5, 3.3, 2785, 700);
    card(s, 0.6, 4.75, 5.95, 2.0);
    txt(s, "Error anatomy", { x: 0.85, y: 4.9, w: 5.5, h: 0.4, fontSize: 16, bold: true, color: INK });
    bullets(s, ["Heavy ridge: 72–90% of the error is shrinkage, 11–31% noise variance", "Mechanism bias²: at most 0.75% anywhere"],
      { x: 0.85, y: 5.35, w: 5.5, h: 1.3, fontSize: 14 });
    card(s, 6.75, 4.75, 5.95, 2.0, EMBER_L);
    txt(s, "Inference", { x: 7.0, y: 4.9, w: 5.5, h: 0.4, fontSize: 16, bold: true, color: EMBER });
    bullets(s, ["λ = 0: 95% intervals cover 0.94–0.96; SEs within 1–3% of the truth", "Ridge on: coverage collapses. Clipping shifts the target at large n (every DP method pays this)"],
      { x: 7.0, y: 5.35, w: 5.5, h: 1.3, fontSize: 14 });
    s.addNotes("When sigma is large, the ridge's shrinkage dominates. Intervals are reliable only when the ridge is off, and at large n clipping to the committed bounds moves the target.");
  }

  // ======================= 28. AdaSSP placeholder =======================
  {
    const s = pres.addSlide(); title(s, "Benchmark against AdaSSP (Wang 2018)", "Test MSE at ε = 1, δ = 10⁻⁶ (efficiency vs non-private in brackets)");
    chip(s, "Not yet run", 11.0, 1.15, 1.7, SLATE);
    const tbd = { text: "to be done", options: { color: "9AA5B4", italic: true } };
    const rows = [hdr(["Dataset", "n", "d", "Non-private", "AdaSSP (published)", "AdaSSP (matched)", "VFL, ours", "VFL, uncorrected"])];
    // n, d from Wang's preprocessed data (experiments/adassp/datasets.py)
    [["housing", "506", "13"], ["wine", "1,599", "11"], ["airfoil", "1,503", "5"], ["concrete", "1,030", "8"], ["bike", "17,379", "17"],
     ["elevators", "16,599", "18"], ["pol", "15,000", "26"], ["kin40k", "40,000", "8"], ["protein", "45,730", "9"]].forEach(([d, n, k]) =>
      rows.push([d, n, k, tbd, tbd, tbd, tbd, tbd]));
    s.addTable(rows, Object.assign(tableOpts([1.4, 1.0, 0.6, 1.55, 1.95, 1.9, 1.75, 1.95]), { y: 1.75, rowH: 0.4, fontSize: 13 }));
    txt(s, "Matched = AdaSSP's algorithm under our adjacency and accountant (A exact); it isolates the λ rule. Published AdaSSP protects the whole record, a stronger guarantee.",
      { x: 0.6, y: 6.2, w: 12.1, h: 0.7, fontSize: 13, italic: true, color: MUTED });
    s.addNotes("Placeholder: fill in once the AdaSSP benchmark is run (loaders are in experiments/adassp). Also add the per-dataset MSE-vs-epsilon grid and the cross-dataset summary.");
  }

  // ======================= 29. Remaining placeholders =======================
  {
    const s = pres.addSlide(); title(s, "Still to run");
    placeholder(s, 0.6, 1.55, 5.95, 5.15, "Intersection-size sweep on real data",
      "Graphic to be done: test MSE vs matched cohort size n (log) for bike, elevators, pol, kin40k and protein, one panel per dataset, lines for ε ∈ {0.5, 1, 2, 4}; AdaSSP and non-private for reference.");
    txt(s, "Cost of the revised protocol in OpenFHE", { x: 6.85, y: 1.6, w: 5.8, h: 0.45, fontFace: HEAD, fontSize: 18, bold: true, color: INK });
    const tbd = { text: "tbd", options: { color: "9AA5B4", italic: true } };
    const rows = [hdr(["n_O", "d", "PSI (s)", "Agg. (s)", "Traffic (MB)", "Flood cost"]),
      ["4,000", "10", tbd, tbd, tbd, tbd], ["32,000", "10", tbd, tbd, tbd, tbd], ["65,536", "20", tbd, tbd, tbd, tbd], ["10⁶ (chunked)", "20", tbd, tbd, tbd, tbd]];
    s.addTable(rows, { x: 6.85, y: 2.15, colW: [1.35, 0.5, 0.85, 0.9, 1.1, 1.15], fontFace: BODY, fontSize: 11.5, color: TEXT,
      border: { type: "solid", color: LINE, pt: 0.75 }, valign: "middle", margin: [3, 4, 3, 4], rowH: 0.48 });
    card(s, 6.85, 4.85, 5.85, 1.85, EMBER_L);
    txt(s, "Also pending", { x: 7.1, y: 4.98, w: 5.4, h: 0.4, fontSize: 15, bold: true, color: EMBER });
    bullets(s, ["Proof of the CKKS error-sensitivity bound S_err", "Finite-precision accounting for the discrete Gaussian", "Standard errors for the ridge regime"],
      { x: 7.1, y: 5.4, w: 5.4, h: 1.25, fontSize: 13.5 });
    s.addNotes("Placeholders for results that need new runs: real-data intersection sweeps, and runtime and communication of the revised protocol once implemented in OpenFHE.");
  }

  // ======================= 30. Conclusions =======================
  {
    const s = pres.addSlide(); s.background = { color: INK };
    s.addText("Takeaways and next steps", { x: 0.6, y: 0.45, w: 12, h: 0.8, fontFace: HEAD, fontSize: 34, bold: true, color: WHITE, margin: 0, isTextBox: true });
    const take = ["The revised protocol closes the critical and high gaps under honest-but-curious parties",
      "Accuracy reaches 2× of non-private OLS at n ≈ 3k–100k, depending on ε",
      "The bias correction is a small, free variance reduction; shrinkage dominates when σ is large"];
    take.forEach((t, i) => {
      numCircle(s, i + 1, 0.6, 1.65 + i * 1.35, 0.65, EMBER, 18);
      s.addText(t, { x: 1.5, y: 1.62 + i * 1.35, w: 5.2, h: 1.1, fontFace: BODY, fontSize: 17, color: WHITE, margin: 0, valign: "top", isTextBox: true });
    });
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 7.2, y: 1.5, w: 5.5, h: 5.2, fill: { color: "1F3050" }, line: { color: "1F3050" }, rectRadius: 0.08 });
    s.addText("Next", { x: 7.5, y: 1.7, w: 5, h: 0.5, fontFace: HEAD, fontSize: 22, bold: true, color: "E7A77C", margin: 0, isTextBox: true });
    const nxt = ["Finish the AdaSSP benchmark and real-data sweeps", "Implement the revised protocol in OpenFHE and measure its cost",
      "Prove the S_err bound; complete finite-precision accounting", "Standard errors; membership-private variant for sensitive registers"];
    s.addText(nxt.map((t, i) => ({ text: t, options: { bullet: true, breakLine: i < nxt.length - 1, paraSpaceAfter: 10 } })),
      { x: 7.5, y: 2.35, w: 4.95, h: 4.2, fontFace: BODY, fontSize: 16, color: WHITE, margin: 0, valign: "top", isTextBox: true });
    s.addNotes("Three takeaways and four next steps.");
  }

  const out = P("slides", "vfl_scenarioB_deck.pptx");
  await pres.writeFile({ fileName: out });
  console.log("wrote", out);
})();
