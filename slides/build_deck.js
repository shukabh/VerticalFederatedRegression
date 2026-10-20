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
const fs = require("fs");
function readCsv(file) {
  const [head, ...rows] = fs.readFileSync(file, "utf8").trim().split(/\r?\n/);
  const keys = head.split(",");
  return rows.map((r) => Object.fromEntries(r.split(",").map((v, i) => [keys[i], v])));
}
// AdaSSP benchmark outputs of experiments/adassp/run_benchmark.py
const ADA9 = readCsv(P("reports", "adassp", "results.csv"));
const ADA_ALL = readCsv(P("reports", "adassp", "all", "results.csv"));
const ADA_CMP = readCsv(P("reports", "adassp", "all", "compare.csv"));
const SWEEP_X = readCsv(P("reports", "adassp", "sweep", "crossings.csv"));   // experiments/adassp/sweep_n.py
// Tables of reports/calibration/heuristic.md (experiments/calibration_heuristic.py), keyed by the "## " heading
function mdTables(file) {
  const out = {};
  let head = "";
  for (const line of fs.readFileSync(file, "utf8").split(/\r?\n/)) {
    if (line.startsWith("## ")) { head = line.slice(3).trim(); continue; }
    if (!line.startsWith("|") || /^\|[-|: ]+\|$/.test(line)) continue;
    const cells = line.slice(1, -1).split("|").map((c) => c.trim());
    (out[head] = out[head] || []).push(cells);
  }
  return out;
}
const CAL = mdTables(P("reports", "calibration", "heuristic.md"));
const calSec = (prefix) => CAL[Object.keys(CAL).find((k) => k.startsWith(prefix))];
// Error bars for native charts, keyed by chart title; added to the chart XML after writing (pptxgenjs has no API for them)
const ERRBARS = {};
const adaGet = (rows, ds, eps, m) => rows.find((r) => r.dataset === ds && Number(r.eps) === eps && r.method === m);

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
    s.addText("When the organization holds the response: protocol, proofs and evaluation", { x: 0.8, y: 3.6, w: 6.6, h: 0.9,
      fontFace: BODY, fontSize: 18, color: "CAD5E6", margin: 0, valign: "top", isTextBox: true });
    s.addText("Working draft  ·  October 2026", { x: 0.8, y: 6.3, w: 7, h: 0.4, fontFace: BODY, fontSize: 13, color: "8FA6C4", margin: 0, isTextBox: true });
    s.addNotes("Two-party linear regression between a researcher and a national statistical agency, where the agency holds the outcome. We cover the problem, what we found when reviewing the draft protocol, the first revision (v2) and its evaluation, and the second revision (v3.1) with its proofs.");
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
    txt(s, "Only A is purely R's. Everything that touches O's data is computed under encryption and noised by O before R sees it.",
      { x: 6.95, y: 5.0, w: 5.5, h: 1.3, fontSize: 16, color: INK });
    s.addNotes("The Gram matrix splits into blocks. A is the researcher's own; B and c_O are the organization's data restricted to matched rows; C and c_R are genuinely cross-party.");
  }

  // ======================= 6. Threat model =======================
  {
    const s = pres.addSlide(); title(s, "Three guarantees, honest-but-curious parties");
    const items = [
      ["eye", "Input privacy", "Each party's view can be simulated from its own input and the agreed output. Neither sees the other's raw data.", SLATE],
      ["link", "Linkage privacy", "O learns nothing about which of its records match R's cohort (only public sizes).", TEAL],
      ["shield", "Output privacy", "What R receives is differentially private (SIM-CDP) for each matched person's O-side data (x_O, y), with R's features held fixed (replace-one).", EMBER],
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
    txt(s, "Parties follow the protocol but analyse everything they receive. In v3.1, R and O each hold a share of the decryption key (in v2, R held all of it). R learns the match set by design; O controls the privacy budget.",
      { x: 1.9, y: 5.8, w: 10.6, h: 0.85, fontSize: 15.5, color: INK, valign: "middle" });
    s.addNotes("Threat model: semi-honest. Three guarantees: cryptographic input privacy, linkage privacy for the organization, and differential privacy on what the researcher receives. The guarantee is computational (SIM-CDP), and it holds only if R follows the protocol: a malicious R could skip clipping. v3.1 splits the decryption key between R and O.");
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

  // ======================= 8. Divider: our work, with the timeline =======================
  {
    const s = pres.addSlide(); s.background = { color: INK };
    numCircle(s, 2, 0.9, 1.15, 1.3, EMBER, 36);
    s.addText("What we did", { x: 2.6, y: 1.05, w: 9.5, h: 0.9, fontFace: HEAD, fontSize: 40, bold: true, color: WHITE, margin: 0, isTextBox: true });
    s.addText("Reviewed the draft and rebuilt it; had the rebuild refereed, then rebuilt and proved it", { x: 2.6, y: 1.95, w: 10.2, h: 0.6, fontFace: BODY, fontSize: 18, color: "CAD5E6", margin: 0, isTextBox: true });
    const steps = [
      ["search", "Review", "Draft paper and code checked line by line"],
      ["cogs", "Run", "End-to-end over sockets: plaintext and OpenFHE"],
      ["secret", "Attack", "Three privacy reviews and a critique"],
      ["layers", "Redesign", "First revision (v2) and rewritten paper"],
      ["chart", "Simulate", "Accuracy and bias-correction studies"],
      ["balance", "Benchmark", "AdaSSP on 29 of Wang's UCI datasets"],
      ["book", "Referee", "Journal-style review; key claims re-checked"],
      ["check", "Prove", "Second revision (v3.1) with a proofs appendix"],
    ];
    const y = 3.6, gap = 1.56;
    s.addShape(pres.shapes.LINE, { x: 1.0, y: y + 0.36, w: gap * 7, h: 0, line: { color: "3D5A80", width: 2 } });
    steps.forEach(([ic, h, b], i) => {
      const x = 0.6 + i * gap;
      circleIcon(s, ic, x, y, 0.72, i === 2 || i === 7 ? EMBER : SLATE);
      s.addText(`${i + 1}. ${h}`, { x: x - 0.1, y: y + 0.92, w: 1.58, h: 0.42, fontFace: HEAD, fontSize: 13.5, bold: true, color: WHITE, margin: 0, isTextBox: true });
      s.addText(b, { x: x - 0.1, y: y + 1.38, w: 1.42, h: 1.5, fontFace: BODY, fontSize: 12.5, color: "CAD5E6", margin: 0, valign: "top", isTextBox: true });
    });
    s.addNotes("Part 2. The sequence of work: review, running the implementation, adversarial privacy reviews, the first redesign (v2), simulations, the AdaSSP benchmark, a journal-style referee review of the paper, and the second redesign (v3.1) with its proofs.");
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
      "BFV linkage replies: statistical flooding; CKKS: DP-calibrated flooding",
      { text: "v3.1 goes further: the key is split, and O's decryption share carries the noise", options: { bold: true } }], { x: 9.1, y: 2.4, w: 3.4, h: 3.7, fontSize: 14.5 });
    s.addNotes("The most important finding. The DP argument reasoned about the decrypted value, but R holds the key and sees the whole ciphertext. With plaintext-scalar noise, the c1 component is independent of the noise.");
  }

  // ======================= 11. Review scorecard =======================
  {
    const s = pres.addSlide(); title(s, "Review scorecard", "Findings on the draft, how the first revision (v2) handled them, and where they stand in v3.1");
    const sev = (t, col) => ({ text: t, options: { bold: true, color: col } });
    const weak = (t) => ({ text: t, options: { color: "9AA5B4", italic: true } });   // v2 fix the referee found incomplete
    const rows = [
      hdr(["Finding on the draft", "Severity", "First revision (v2)", "Now (v3.1)"]),
      ["Plaintext-scalar noise leaks through the ciphertext", sev("Critical", EMBER), weak("Fresh encryption plus flooding"), "Split key; O's decryption share carries the noise (proven)"],
      ["Calibration step shares raw-data statistics", sev("Critical", EMBER), "Per-party commitments, public constants only", "Unchanged"],
      ["Add/remove sensitivity vs replace-one adjacency (true ε up to ~2× the claim)", sev("High", "B7791F"), "Replace-one Δ, closed form", "Δ proven exact; yᵀy released only where it is free"],
      ["No privacy ledger across runs", sev("High", "B7791F"), weak("zCDP ledger at O"), "(ρ, δ) ledger per register; flooding costs only δ"],
      ["PSI reveals O's row order and bin loads", sev("High", "B7791F"), "Secret permutation, public padding", "Also OPRF-based; R's cohort capped and padded"],
      ["Adaptive λ cancels the bias correction", sev("Validity", SLATE), weak("λ set from public inputs"), "λ from a stage-1 release, independent of stage 2 (proven)"],
      ["CKKS slot leak; HE parameter security", sev("Checked", TEAL), "Not an issue with OpenFHE", "CKKS scale must rise to ≈2^61.5 (proofs)"],
      ["R learns which of its cohort are in the register", sev("By design", MUTED), "Stated in Definition 4", "Unchanged"],
    ];
    s.addTable(rows, Object.assign(tableOpts([4.0, 1.25, 3.0, 3.85]), { y: 1.75, rowH: 0.5, fontSize: 12.5 }));
    txt(s, "Grey: fixes that the referee later found incomplete (Part 5).", { x: 0.6, y: 6.5, w: 8, h: 0.3, fontSize: 11, italic: true, color: MUTED });
    s.addNotes("Findings across the code review, three privacy reviews and an independent critique. The v2 column is the first revision; the referee review of the paper found three of those fixes incomplete (grey): the fresh-encryption argument does not hold against a key-holding R, the ledger charged the flooding cost once per run instead of per reply, and the public lower bound ℓ behind the fixed λ cannot be committed in practice. The v3.1 column is the second revision, covered in Part 5.");
  }

  // ======================= 12. Contributions =======================
  {
    const s = pres.addSlide(); title(s, "Contributions");
    const items = [
      ["Corrected data flow", "R's Gram block A is computed locally, is never sent and never noised. The sensitivity loses its B_R⁴ term."],
      ["Protocol v3.1", "Split decryption key with the noise in O's share, replace-one DP, (ρ, δ) ledger, OPRF-based PSI, two-stage ridge, valid intervals."],
      ["Proofs", "Output privacy (SIM-CDP) with a worst-case error bound, the exact replace-one sensitivity, and interval coverage to first order."],
      ["Evaluation", "End-to-end runs, simulation studies of accuracy and of the bias correction, and a benchmark against AdaSSP."],
    ];
    items.forEach(([h, b], i) => {
      const x = 0.6 + (i % 2) * 6.15, y = 1.6 + Math.floor(i / 2) * 2.6;
      card(s, x, y, 5.95, 2.35);
      numCircle(s, i + 1, x + 0.3, y + 0.3, 0.7, i === 1 ? EMBER : INK, 20);
      txt(s, h, { x: x + 1.2, y: y + 0.38, w: 4.5, h: 0.5, fontFace: HEAD, fontSize: 21, bold: true, color: INK });
      txt(s, b, { x: x + 1.2, y: y + 0.95, w: 4.5, h: 1.3, fontSize: 15 });
    });
    s.addNotes("Four contributions: the corrected data flow, the protocol (now v3.1), its proofs, and the evaluation. The finite-noise bias expansion is still in the paper, but the referee judged it modest and the correction is now optional, so it is no longer listed as a headline contribution.");
  }

  // ======================= 13. Divider: methodology =======================
  divider(3, "Methodology", "The first revision (v2), phase by phase")
    .addNotes("Part 3: how the first revision (v2) works. Every result in Part 4 uses its release mechanism. Part 5 covers what the second revision (v3.1) changes.");

  // ======================= 14. Protocol at a glance =======================
  {
    const s = pres.addSlide(); title(s, "Protocol v2 at a glance");
    const ph = [["0", "Setup and commitments", "Only public constants cross"], ["1", "Private linkage", "BFV labeled PSI; R learns the match set"],
      ["2", "Encrypted aggregation", "CKKS, depth 1; DP noise added by O"], ["3", "Solve at R", "Fixed ridge, bias correction; post-processing"]];
    ph.forEach(([n, h, b], i) => {
      const y = 1.7 + i * 1.3;
      numCircle(s, n, 0.6, y, 0.7, i === 2 ? EMBER : INK, 20);
      txt(s, h, { x: 1.5, y: y + 0.02, w: 5.0, h: 0.45, fontFace: HEAD, fontSize: 20, bold: true, color: INK });
      txt(s, b, { x: 1.5, y: y + 0.5, w: 5.0, h: 0.5, fontSize: 15, color: MUTED });
    });
    txt(s, "Orange in the figure marks what changed from the draft. Part 5 shows what v3.1 changes.", { x: 0.6, y: 6.8, w: 6.4, h: 0.35, fontSize: 12, italic: true, color: MUTED });
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
    txt(s, "Δ_add ≤ Δ_rep ≤ 2Δ_add. Adding yᵀy costs nothing in these rows, but not always: it is free if B_y² ≤ 2(B_R² + B_O²). At (1, 1, 5), Δ = 26.1 with yᵀy and 14.1 without.",
      { x: 0.6, y: 5.45, w: 6.2, h: 0.9, fontSize: 13, italic: true, color: MUTED });
    card(s, 7.2, 1.55, 5.5, 2.35);
    circleIcon(s, "calc", 7.45, 1.8, 0.7, SLATE);
    txt(s, "σ from the analytic Gaussian", { x: 8.35, y: 1.88, w: 4.2, h: 0.45, fontSize: 17, bold: true, color: INK });
    txt(s, "Smallest σ meeting (ε, δ) for the whole stacked release (Balle–Wang). Discrete Gaussian at the CKKS resolution.",
      { x: 8.35, y: 2.38, w: 4.15, h: 1.4, fontSize: 14 });
    card(s, 7.2, 4.15, 5.5, 2.55, EMBER_L);
    circleIcon(s, "ledger", 7.45, 4.4, 0.7, EMBER);
    txt(s, "Privacy ledger at O", { x: 8.35, y: 4.48, w: 4.2, h: 0.45, fontSize: 17, bold: true, color: INK });
    txt(s, "O refuses a run once the spend would pass its cap; re-runs and inference queries count. v2 charged ρ_err once per run, though one record moves the error of all 63 replies. v3.1 tracks (ρ, δ): flooding costs only δ.",
      { x: 8.35, y: 4.98, w: 4.15, h: 1.65, fontSize: 13 });
    s.addNotes("The draft's sensitivity bound was for add/remove adjacency; our adjacency is replace-one, which needs up to twice the Δ. The proofs show the closed form is exact for d_O ≥ 2. yᵀy is free whenever B_y² ≤ 2(B_R² + B_O²), which covers every row here, and never free when B_y² > 3(B_R² + B_O²); the referee's counterexample (1, 1, 5) gives Δ = 26.07 with yᵀy against 14.14 without. So yᵀy is released only where it is free, otherwise under its own budget. The ledger makes re-runs compose under an explicit cap; the v2 ledger under-counted the error channel, which v3.1 removes by flooding statistically and charging only δ.");
  }

  // ======================= 18. Circuit privacy =======================
  {
    const s = pres.addSlide(); title(s, "Sanitizing O's replies in v2 (circuit privacy)", "R holds the key, so every reply must reveal nothing beyond its decrypted value");
    const steps = [
      ["1", "Fresh encryption", "ct ⊞ Enc_pk(z) makes c₁ computationally uniform: it no longer depends on O's data."],
      ["2", "Modulus switch", "Drop to the last level, which scales down the data-dependent error."],
      ["3", "Flood the error", "BFV: statistical flooding with N·2^−κ ≤ 2^−40. CKKS: DP-calibrated flooding with width ς = S_err / √(2ρ_err)."],
    ];
    steps.forEach(([n, h, b], i) => {
      const x = 0.6 + i * 4.15;
      card(s, x, 1.85, 3.9, 2.5);
      numCircle(s, n, x + 0.3, 2.07, 0.65, EMBER, 18);
      txt(s, h, { x: x + 1.1, y: 2.15, w: 2.7, h: 0.5, fontFace: HEAD, fontSize: 19, bold: true, color: INK });
      txt(s, b, { x: x + 0.3, y: 2.85, w: 3.35, h: 1.45, fontSize: 14.5 });
    });
    card(s, 0.6, 4.6, 5.3, 2.4, "F7F9FC");
    txt(s, "What v2 relied on", { x: 0.85, y: 4.75, w: 4.9, h: 0.4, fontSize: 15, bold: true, color: INK });
    txt(s, "A flood sized as a DP mechanism, only a few bits above the error, to save CKKS precision; and the fresh encryption to hide c₁ from the key holder.",
      { x: 0.85, y: 5.2, w: 4.85, h: 1.25, fontSize: 13.5, color: MUTED });
    card(s, 6.15, 4.6, 6.55, 2.4, EMBER_L);
    txt(s, "What the referee and the proofs found", { x: 6.4, y: 4.75, w: 6.1, h: 0.4, fontSize: 15, bold: true, color: EMBER });
    bullets(s, ["S_err is as large as the whole error, and its cost was charged once, not per reply",
      "Decrypting, R sees a hint on the fresh randomness: not a plain RLWE reduction",
      "v3.1 floods the full error inside O's decryption share. Proven, at a CKKS scale of 2^61.5 (2^53.7 with circuit changes)"],
      { x: 6.4, y: 5.2, w: 6.1, h: 1.7, fontSize: 13, paraSpaceAfter: 4 });
    s.addNotes("Three sanitization steps in v2. The CKKS flooding was sized as a DP mechanism in the style of Li, Micciancio, Schultz and Sorrell, so it would cost a small ledger entry instead of 40 bits of precision. The referee found three problems: the rotations, rescaling and modulus switch all depend on the data-dependent c1, so S_err is of the order of the whole error; R, who holds the key, learns a noisy hint on the fresh-encryption randomness, which needs a Hint-RLWE-type assumption; and the ledger charged the error channel once instead of once per reply. v3.1 instead splits the key and floods the whole error statistically (2^-40) inside O's decryption share. The proofs bound the worst-case error at about 2^25.3 for the circuit as specified, which needs a CKKS scale of about 2^61.5 (the v3.1 spec had assumed 2^52.5); restructuring the circuit brings this to 2^57.2, and having R encode at 2^60 brings it to 2^53.7.");
  }

  // ======================= 19. Solve & bias correction =======================
  {
    const s = pres.addSlide(); title(s, "Phase 3: fixed ridge and zero-budget bias correction");
    card(s, 0.6, 1.55, 6.0, 5.15, "F7F9FC");
    const lines = [
      ["Ridge, fixed before decryption", "λ = max{ 0, 2ρ*σ√p − nℓ },  Ψ = I"],
      ["Release gate", "ρ̂ = λ_min(G̃ + λI) / (2σ√p) ≥ 1"],
      ["Private ridge estimate", "β̃_λ = (G̃ + λI)⁻¹ c̃"],
      ["Bias correction (optional in v3.1; off by default)", "β̂_bc = β̃_λ − σ² M(P̃_λ) β̃_λ"],
    ];
    lines.forEach(([h, e], i) => {
      txt(s, h, { x: 0.9, y: 1.8 + i * 1.2, w: 5.5, h: 0.4, fontSize: 14, bold: true, color: SLATE });
      txt(s, e, { x: 0.9, y: 2.2 + i * 1.2, w: 5.5, h: 0.55, fontFace: HEAD, fontSize: 18, color: INK });
    });
    card(s, 6.85, 1.55, 5.85, 2.8);
    txt(s, "Theorem (conditional bias)", { x: 7.1, y: 1.72, w: 5.4, h: 0.45, fontFace: HEAD, fontSize: 18, bold: true, color: INK });
    txt(s, "E[ β̃ | ℰ_η ] − β̂ = σ² M(P) β̂ + O(σ⁴) + O(exp(−p(ηρ−1)²/2))", { x: 7.1, y: 2.25, w: 5.45, h: 0.6, fontFace: HEAD, fontSize: 15.5, color: INK });
    txt(s, "M(P) = (P² + tr(P)P − P diag(P)) Π_O + (P Π_O P + tr(Π_O P) P) Π_R", { x: 7.1, y: 2.92, w: 5.45, h: 0.6, fontFace: HEAD, fontSize: 13.5, color: MUTED });
    txt(s, "The plain mean does not exist, hence the conditioning on ℰ_η.", { x: 7.1, y: 3.65, w: 5.45, h: 0.6, fontSize: 13, italic: true, color: MUTED });
    card(s, 6.85, 4.55, 5.85, 2.15, EMBER_L);
    txt(s, "Why λ must not depend on the noise", { x: 7.1, y: 4.67, w: 5.4, h: 0.4, fontSize: 15, bold: true, color: EMBER });
    txt(s, "Chosen from G̃, λ moves with the noise, and the correction removed none of the bias (1.0–1.6× left); with fixed λ it removes 87–98%. But ℓ needs the joint Gram, so in practice ℓ = 0. v3.1 picks λ from a separate stage-1 release instead.",
      { x: 7.1, y: 5.1, w: 5.45, h: 1.5, fontSize: 13, color: INK });
    s.addNotes("R fixes the ridge from public inputs, decrypts, splices in its exact block, solves, and applies the second-order bias correction. The theorem is stated conditionally because the unconditional mean does not exist. The referee pointed out that ℓ must lower-bound the smallest eigenvalue of the joint Gram, which depends on cross-covariances neither party knows, so the feasible choice is ℓ = 0, under which the ridge never switches off. v3.1 spends a small budget share on a stage-1 release and chooses (λ, Ψ) from it and from A only; the proofs show this choice is independent of the stage-2 noise, so the conditional analysis still applies. The bias correction is optional in v3.1 and off by default.");
  }

  // ======================= 20. Evaluation design =======================
  {
    const s = pres.addSlide(); title(s, "How we evaluate");
    const cards = [
      ["db", "Simulation population", "Features on real-world scales (income, age, binary indicators, counts), correlated. Public standardization constants; committed 99.5% bounds; R² ∈ {0.3, 0.5, 0.9}."],
      ["flask", "Monte Carlo", "n from 100 to 100k matched records; ε ∈ {0.5, 1, 2, 4, 8}, δ = 10⁻⁵; 200 replicates, each with new data and a new DP release."],
      ["chart", "Metrics", "RMSE against the true β (standardized); per-coefficient efficiency ratio; interval coverage; MSE ratio corrected/uncorrected."],
      ["book", "Benchmark", "29 UCI datasets from Wang's published run, with his preprocessing; AdaSSP as published and under our guarantee; 10-fold test MSE."],
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
  divider(4, "Results", "Measured with the v2 release; what is still to run")
    .addNotes("Part 4: results. All of them use the v2 release mechanism; v3.1 changes the ridge rule and the inference, and its runs are still to come. Slides marked as placeholders are still to be run.");

  // ======================= 22. Correctness, and the cost still to measure =======================
  {
    const s = pres.addSlide(); title(s, "The cryptographic layer is exact", "Draft protocol, run end-to-end over sockets with plaintext and real OpenFHE backends");
    const stats = [["0", "misaligned rows", "PSI recovered exactly the true match set in all 6 cohorts, plaintext and OpenFHE"],
      ["≈10⁻¹⁰", "max |β − plaintext answer|", "noise off, against the plaintext ridge target at the same λ (10⁻³): encryption adds no error"],
      ["≈13 s", "end-to-end on OpenFHE", "n = 400 matched out of n_O = 4,000 records (OpenFHE 1.5.1)"]];
    stats.forEach(([v, l, d], i) => {
      const x = 0.6 + i * 4.15;
      card(s, x, 1.75, 3.9, 2.45);
      txt(s, v, { x: x + 0.3, y: 1.9, w: 3.4, h: 0.85, fontFace: HEAD, fontSize: 38, bold: true, color: i === 1 ? EMBER : INK });
      txt(s, l, { x: x + 0.3, y: 2.78, w: 3.4, h: 0.4, fontSize: 15, bold: true, color: SLATE });
      txt(s, d, { x: x + 0.3, y: 3.2, w: 3.35, h: 0.9, fontSize: 13, color: MUTED });
    });
    txt(s, "Cost of protocol v3.1 in OpenFHE", { x: 0.6, y: 4.5, w: 8, h: 0.45, fontFace: HEAD, fontSize: 18, bold: true, color: INK });
    chip(s, "To be run", 11.0, 4.55, 1.7, SLATE);
    const tbd = { text: "tbd", options: { color: "9AA5B4", italic: true } };
    const rows = [hdr(["Register size n_O", "d", "PSI time (s)", "Aggregation time (s)", "Traffic (MB)", "Release overhead"]),
      ["4,000", "10", tbd, tbd, tbd, tbd], ["32,000", "10", tbd, tbd, tbd, tbd], ["65,536", "20", tbd, tbd, tbd, tbd], ["10⁶ (chunked)", "20", tbd, tbd, tbd, tbd]];
    s.addTable(rows, Object.assign(tableOpts([2.3, 0.8, 1.9, 2.4, 2.0, 2.7]), { y: 5.05, rowH: 0.34, fontSize: 12.5 }));
    s.addNotes("Correctness of linkage and aggregation: exact alignment, and 1e-10 agreement with the plaintext answer when the noise is off. That answer is the clipped-data ridge target at the λ the run chose (10^-3, the old solver's floor), not plain OLS; the two differ by up to 1e-4 because of that floor. All error comes from the DP noise and the ridge it forces. These runs used the draft protocol. The table is a placeholder: v3.1's release (split key, noise in O's decryption share) is not yet implemented in OpenFHE, and the proofs show its CKKS scale must be about 2^61.5, or 2^53.7 with the improved circuit.");
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
    s.addNotes("Numbers behind the figure. A stronger signal lowers the OLS floor much more than it raises the DP error (which grows only through the norm of beta), so the ratio is largest at high R-squared.");
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
    const s = pres.addSlide(); title(s, "Does the bias correction matter?", "Barely: it acts mostly as extra shrinkage, so v3.1 makes it optional and off by default");
    img(s, P("reports", "bias_correction", "fig1_mse_ratio.png"), 0.4, 1.65, 12.5, 3.4, 2291, 705);
    const st = [["−7 to −12%", "MSE vs the protocol target when the ridge is on. The correction also shrinks noise; it never raised this MSE (90 settings)."],
      ["+0.3 to +4.3%", "MSE vs OLS under a moderate ridge: the correction restores shrinkage that noise had offset."],
      ["≥ 130", "averaged releases before the bias alone would matter; one release: at most 0.75% of MSE."]];
    st.forEach(([v, b], i) => {
      const x = 0.6 + i * 4.15;
      txt(s, v, { x, y: 5.15, w: 3.9, h: 0.7, fontFace: HEAD, fontSize: 30, bold: true, color: i === 0 ? EMBER : INK });
      txt(s, b, { x, y: 5.85, w: 3.9, h: 1.35, fontSize: 13, color: MUTED });
    });
    s.addNotes("Answer to the reviewer: not as a bias correction; the removed bias is tiny. The variance reduction is mostly extra shrinkage of the noise, which helps against the ridge target but hurts slightly against OLS or the true beta. On real data it is worse than no correction on 19 of 29 datasets at eps = 1 (slide on AdaSSP across privacy levels). So v3.1 keeps it as optional post-processing, off by default.");
  }

  // ======================= 27. Error anatomy & inference =======================
  {
    const s = pres.addSlide(); title(s, "Where the error comes from when σ is large");
    img(s, P("reports", "bias_correction", "fig2_decomposition.png"), 0.4, 1.25, 12.5, 3.15, 2785, 700);
    card(s, 0.6, 4.55, 5.95, 2.25);
    txt(s, "Error anatomy", { x: 0.85, y: 4.68, w: 5.5, h: 0.4, fontSize: 16, bold: true, color: INK });
    bullets(s, ["Heavy ridge: 72–90% of the error is shrinkage, 11–31% noise variance", "Mechanism bias²: at most 0.75% anywhere",
      "At large n, clipping shifts the target (every DP method pays this)"],
      { x: 0.85, y: 5.12, w: 5.5, h: 1.6, fontSize: 13.5, paraSpaceAfter: 4 });
    card(s, 6.75, 4.55, 5.95, 2.25, EMBER_L);
    txt(s, "Inference", { x: 7.0, y: 4.68, w: 5.5, h: 0.4, fontSize: 16, bold: true, color: EMBER });
    bullets(s, ["v2, λ = 0: 95% intervals cover 0.94–0.96", "v2, ridge on: coverage collapses (42–81% at ℓ = 0)",
      { text: "v3.1: Wald intervals for the ridge target cover 0.944–0.962 (target 0.955) with the ridge on; the β intervals over-cover", options: { bold: true } }],
      { x: 7.0, y: 5.12, w: 5.5, h: 1.6, fontSize: 13.5, paraSpaceAfter: 4 });
    s.addNotes("When sigma is large, the ridge's shrinkage dominates. Under v2, intervals were reliable only when the ridge was off; with the feasible ℓ = 0 the ridge never switches off and Wald coverage against the true beta is 42–81% even at n = 128k. At large n clipping to the committed bounds also moves the target. v3.1 targets the ridge coefficient θ_λ with a Wald interval whose variance comes from a parametric bootstrap, plus a bias-aware interval for β (Armstrong, Kolesár and Kwon). In the proofs' simulations (2000 replications per scenario) the Wald intervals cover 0.944–0.962 against a target of 0.955 with the ridge on; the β intervals over-cover and are uninformative without a public bound on the norm of beta in two of three ridge scenarios. Coverage is proven to first order.");
  }

  // ======================= 28. AdaSSP benchmark at eps = 1 =======================
  {
    const s = pres.addSlide(); title(s, "Benchmark against AdaSSP (Wang 2018)", "Test MSE at ε = 1, δ = 10⁻⁶, 10-fold CV. Ratio to non-private in brackets; bold = best private method");
    const PRIV = ["adassp", "adassp_matched", "vfl", "vfl_uncorrected"];
    const f3 = (v) => Number(v).toPrecision(3);
    const rows = [hdr(["Dataset", "n", "d", "Non-private", "AdaSSP (published)", "AdaSSP (matched)", "VFL, ours", "VFL, uncorrected"])];
    rows[0][6].options.fill = { color: EMBER };
    ["housing", "wine", "airfoil", "concrete", "bike", "elevators", "pol", "kin40k", "protein"].forEach((ds) => {
      const np = adaGet(ADA9, ds, 1, "nonprivate");
      const vals = PRIV.map((m) => Number(adaGet(ADA9, ds, 1, m).test_mse));
      const best = Math.min(...vals);
      rows.push([ds, Number(np.n).toLocaleString("en-US"), np.d, f3(np.test_mse),
        ...PRIV.map((m, k) => ({ text: `${f3(vals[k])} (${(vals[k] / Number(np.test_mse)).toFixed(2)}×)`,
          options: vals[k] === best ? { bold: true, color: INK } : {} }))]);
    });
    s.addTable(rows, Object.assign(tableOpts([1.4, 1.0, 0.6, 1.4, 2.0, 1.95, 1.8, 1.95]), { y: 1.75, rowH: 0.4, fontSize: 13 }));
    txt(s, "Published AdaSSP is more private than our release, not less: at nominal ε = 1 it is μ = 0.198 Gaussian DP (ε ≈ 0.82) under whole-record adjacency, against our μ = 0.237 (ε = 1.00) under the weaker O-side adjacency. Its noise is lower because of Wang's composition and the joint row bound (Δ² = 4.5, ours 10). Matched = AdaSSP's adaptive λ under our guarantee; idealized, since its λ_min step needs XᵀX in the clear.",
      { x: 0.6, y: 5.85, w: 12.1, h: 1.1, fontSize: 12.5, italic: true, color: MUTED });
    s.addNotes("From reports/adassp/results.csv. Our protocol beats the matched AdaSSP on every dataset here and is within a few percent of published AdaSSP at large n; published AdaSSP is ahead on the small datasets at eps = 1. Our AdaSSP re-implementation reproduces Wang's published MSE (median ratio 1.000 over 29 datasets and 10 values of eps). The earlier caveat that Wang's constants might overstate AdaSSP's advantage ran the wrong way: in Gaussian DP, which is exact for composed Gaussian mechanisms, published AdaSSP at nominal (1, 1e-6) is mu = 0.198, i.e. (0.82, 1e-6)-DP under whole-record replace-one, while our release is mu = 0.237, exactly (1.00, 1e-6), under O-side replace-one. Its per-entry noise (11.85 vs our 13.36) is lower because Gaussian composition is tighter than Wang's basic composition, and because our sensitivity uses separate block bounds (Delta^2 = 10) where the data satisfy a joint row bound (Delta^2 = 4.5).");
  }

  // ======================= 29. AdaSSP across eps =======================
  {
    const s = pres.addSlide(); title(s, "Across privacy levels", "Test MSE ÷ non-private, three datasets of increasing size; bars: 95% CI of the mean over the 10 folds");
    const EPS = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10];
    const SER = [["trivial", "Trivial (θ = 0)", "B8BEC8"], ["adassp", "AdaSSP (published)", "1BAF7A"],
                 ["adassp_matched", "AdaSSP (matched)", "EDA100"], ["vfl", "VFL, ours", "2A78D6"]];
    ["housing", "elevators", "protein"].forEach((ds, k) => {
      const n = Number(adaGet(ADA9, ds, 1, "vfl").n).toLocaleString("en-US");
      const data = SER.map(([m, name]) => ({ name, labels: EPS.map(String),
        values: EPS.map((e) => Number(adaGet(ADA9, ds, e, m).ratio_to_nonprivate)) }));
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
      const x = 2.6 + i * 2.35;
      s.addShape(pres.shapes.LINE, { x, y: 5.12, w: 0.4, h: 0, line: { color: col, width: 2.5 } });
      txt(s, name, { x: x + 0.5, y: 4.98, w: 1.8, h: 0.3, fontSize: 12, color: TEXT });
    });
    const N = new Set(ADA_CMP.map((r) => r.dataset)).size;
    const E3 = [0.1, 1, 10];
    const verdicts = (vs, e) => {
      const v = ADA_CMP.filter((r) => r.vs === vs && Number(r.eps) === e).map((r) => r.verdict);
      return `${v.filter((x) => x === "better").length} / ${v.filter((x) => x === "worse").length}`;
    };
    const trows = [hdr([`Ours better / worse (of ${N})`, "ε = 0.1", "ε = 1", "ε = 10"]),
      ["vs AdaSSP (published)", ...E3.map((e) => verdicts("adassp", e))],
      ["vs AdaSSP (matched)", ...E3.map((e) => verdicts("adassp_matched", e))],
      ["Bias-corrected vs uncorrected", ...E3.map((e) => verdicts("vfl_uncorrected", e))]];
    s.addTable(trows, Object.assign(tableOpts([3.0, 1.05, 1.05, 1.05]), { y: 5.4, rowH: 0.35, fontSize: 13 }));
    txt(s, "Paired over folds, 95% CI; the rest show no significant difference.", { x: 0.6, y: 6.92, w: 6.2, h: 0.3, fontSize: 11, italic: true, color: MUTED });
    card(s, 7.2, 5.35, 5.5, 1.85, EMBER_L);
    txt(s, [{ text: "From ε ≈ 0.5 published AdaSSP wins (median 4% lower MSE at ε = 1); the gap shrinks as n grows. Read the wins with care:", options: { breakLine: true } },
      { text: "at ε = 0.1 both often lose to θ = 0 (ours on 13 of 29, AdaSSP on 15)", options: { bullet: true, breakLine: true } },
      { text: "the matched baseline uses only ¾ of its μ budget, so 27 / 0 overstates the fixed ridge", options: { bullet: true } }],
      { x: 7.4, y: 5.45, w: 5.15, h: 1.7, fontSize: 12.5, color: INK, valign: "middle", paraSpaceAfter: 2 });
    s.addNotes("Native charts from reports/adassp/results.csv; paired verdicts from reports/adassp/all/compare.csv (all 29 datasets of Wang's published run, R = 50 draws per fold; a verdict needs the 95% CI of the paired fold difference to exclude 0). Two caveats from the referee, both re-checked on our results: at eps = 0.1 our release is worse than predicting zero on 13 of 29 datasets (AdaSSP on 15; at eps = 0.01, 21 and 23), so the small-eps wins compare estimators that are mostly no better than zero. And the matched AdaSSP spends eps/3 on lambda_min by basic composition: its total is mu = 0.178 of the 0.237 available, and its noise is 1.49x ours, whereas a Gaussian-DP-optimal split (10% of mu^2 on lambda_min) would cost only about 5% extra noise. The bias correction changes test MSE by -3.7% to +2.3% across all eps: it helps at small eps and slightly hurts at large eps. Paired fold t-tests are anti-conservative because training folds overlap, and there is no multiplicity correction.");
  }

  // ======================= 29. Intersection-size sweep on real data =======================
  {
    const SWF = P("reports", "adassp", "sweep", "fig_sweep_n_deck.png");
    const hdrPng = fs.readFileSync(SWF);                       // PNG IHDR: width, height at bytes 16-23
    const [FIGW, FIGH] = [hdrPng.readUInt32BE(16), hdrPng.readUInt32BE(20)];
    const s = pres.addSlide(); title(s, "Intersection size on real data", "Test MSE ÷ non-private at the same n; nested subsets of the training folds; bars: 95% CI over folds");
    img(s, SWF, 0.4, 1.6, 7.4, 5.75, FIGW, FIGH);
    txt(s, "Matched records to come within 10% of non-private", { x: 8.05, y: 1.65, w: 4.7, h: 0.7, fontFace: HEAD, fontSize: 16, bold: true, color: INK });
    txt(s, "Ours (AdaSSP published)", { x: 8.05, y: 2.35, w: 4.7, h: 0.3, fontSize: 12, italic: true, color: MUTED });
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
      "Ours needs about 1.5–2× the records AdaSSP needs for the same cost",
      "elevators stays above 10% at every size available (up to 14.9k)"], { x: 8.05, y: 5.5, w: 4.7, h: 1.5, fontSize: 13.5 });
    s.addNotes("Real-data analogue of Figure A. From reports/adassp/sweep (experiments/adassp/sweep_n.py): bike, elevators, pol and protein, the four of Wang's datasets with at least 15k rows and a real linear signal (kin40k has none: non-private is no better than predicting zero). For each fold, the training folds are shuffled once and the first n rows are the matched set; the held-out fold is the test set. Crossings are log-linear interpolations of the mean curve; '>15k' means never within 10% at the sizes available.");
  }

  // ======================= Calibration in practice (1/3): the chain =======================
  {
    const s = pres.addSlide(); title(s, "Calibration in practice: from ε to σ and λ", "Every input is public and committed before any data are seen");
    const steps = [
      ["δ", "δ ≤ 1/(10N)", "N = people O protects. 10⁻⁵ → 10⁻⁸ raises σ by only 25–50%"],
      ["Budget", "one release: (ε, δ)\nk releases: ρ = (√(L+ε) − √L)² / k", "L = ln(1/δ); zCDP adds up across releases"],
      ["Δ", "Δ² = (2O + R + Y)²/2 + 2RY", "R, O, Y = B_R², B_O², B_y² (usual case, exact). yᵀy is free if Y ≤ 2(R + O)"],
      ["σ", "σ = Δ · s(ε, δ)\nor Δ / √(2ρ) for k releases", "s: analytic-Gaussian multiplier (table below)"],
      ["λ", "λ = max{0, 2ρ*σ√p − nℓ}", "ρ* = 2.5; ℓ public, else 0; ridge off from n = 2ρ*σ√p / ℓ. v3.1: λ from stage 1"],
    ];
    const gap = 2.45;
    s.addShape(pres.shapes.LINE, { x: 0.95, y: 1.95, w: gap * 4, h: 0, line: { color: LINE, width: 2 } });
    steps.forEach(([h, f, b], i) => {
      const x = 0.6 + i * gap;
      numCircle(s, i + 1, x, 1.6, 0.7, i === 3 ? EMBER : INK, 18);
      txt(s, h, { x: x + 0.82, y: 1.72, w: 1.5, h: 0.45, fontFace: HEAD, fontSize: 18, bold: true, color: INK });
      card(s, x, 2.5, 2.3, 2.15);
      txt(s, f, { x: x + 0.15, y: 2.62, w: 2.0, h: 0.95, fontFace: HEAD, fontSize: 13, color: INK });
      txt(s, b, { x: x + 0.15, y: 3.6, w: 2.0, h: 1.0, fontSize: 11, color: MUTED });
    });
    const t1 = calSec("1.");                                    // first table: s(eps, delta) by delta
    const rows = [hdr(["σ/Δ  δ \\ ε", ...t1[0].slice(1).map((h) => h.replace("ε = ", ""))])];
    t1.slice(1).filter((r) => ["1e-05", "1e-06", "1e-08"].includes(r[0])).forEach((r) =>
      rows.push([`δ = ${r[0].replace("1e-0", "10⁻")}`, ...r.slice(1)]));
    s.addTable(rows, Object.assign(tableOpts([1.05, 0.86, 0.86, 0.86, 0.86, 0.86, 0.86, 0.86]), { y: 5.0, rowH: 0.36, fontSize: 12 }));
    card(s, 7.9, 4.95, 4.8, 1.75, EMBER_L);
    txt(s, "Planning formula", { x: 8.15, y: 5.05, w: 4.4, h: 0.35, fontSize: 14, bold: true, color: EMBER });
    txt(s, "σ ≈ Δ (√(2 ln(1/δ)) − 1) / ε^0.92", { x: 8.15, y: 5.42, w: 4.4, h: 0.45, fontFace: HEAD, fontSize: 17, color: INK });
    txt(s, "Within −8% to +3% of exact for ε ∈ [0.1, 8], δ ∈ [10⁻⁸, 10⁻⁵]. For planning only: O computes the exact σ for each release.",
      { x: 8.15, y: 5.92, w: 4.4, h: 0.75, fontSize: 11.5, color: MUTED });
    s.addNotes("The calibration chain, from experiments/calibration_heuristic.py (reports/calibration/heuristic.md). The privacy budget is the only real decision; the rest follows from public, committed constants. The zCDP route is needed when several releases share one budget; for a single release the analytic Gaussian gives a 13-36% smaller sigma at delta = 1e-6. The zCDP-to-(eps, delta) conversion shown here is the simple one; v3.1 uses the tighter Canonne-Kamath-Steinke conversion. The Delta formula is the exact supremum in the usual regime (proven for d_O >= 2); yT y adds nothing when B_y^2 <= 2(B_R^2 + B_O^2). The lambda rule is v2's: since a public l is rarely available, v3.1 chooses lambda from a small stage-1 release instead, so the ridge-off sizes here are a v2 figure.");
  }

  // ======================= Calibration in practice (2/3): scaling laws =======================
  {
    const s = pres.addSlide(); title(s, "From σ to the matched records you need", "Two scaling laws, exact at first order; one pilot figure carries over to any ε");
    const laws = [
      ["Prediction", "n ∝ σ ∝ Δ / ε", "With ℓ = 0 the ridge is proportional to σ, so the excess test error depends on n only through n / σ. Doubling ε roughly halves the records."],
      ["Coefficients", "n ∝ σ² ∝ (Δ / ε)²", "Once the ridge is off, DP error / sampling error ≈ σ²(1+‖β‖²) tr(Σ⁻²) / (n s² tr(Σ⁻¹)). Doubling ε cuts the records by about 3.5×."],
    ];
    laws.forEach(([h, f, b], i) => {
      const x = 0.6 + i * 6.15;
      card(s, x, 1.55, 5.95, 2.05, i === 0 ? MIST : EMBER_L);
      txt(s, h, { x: x + 0.3, y: 1.7, w: 2.6, h: 0.4, fontFace: HEAD, fontSize: 18, bold: true, color: INK });
      txt(s, f, { x: x + 2.9, y: 1.68, w: 2.85, h: 0.45, fontFace: HEAD, fontSize: 20, bold: true, color: i === 0 ? SLATE : EMBER, align: "right" });
      txt(s, b, { x: x + 0.3, y: 2.3, w: 5.4, h: 1.2, fontSize: 13.5, color: TEXT });
    });
    const t2 = calSec("2."), t3 = calSec("3.");
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
    txt(s, "Prediction: real-data sweep; coefficients: simulation. Breaks down for ill-conditioned designs (use the calculator's eigen-sums), at high R² where clipping bias dominates, and at small n. Flooding noise and the discrete Gaussian are not modelled.",
      { x: 0.6, y: 6.6, w: 12.1, h: 0.6, fontSize: 11.5, italic: true, color: MUTED });
    s.addNotes("Prediction targets come from the real-data sweep (reports/adassp/sweep), coefficient targets from the Monte Carlo (reports/coef_accuracy). The first-order calculator is coef_ratio / pred_ratio in experiments/calibration_heuristic.py; it needs the design's second moments, the coefficients and the residual variance from public reference data or a pilot.");
  }

  // ======================= Calibration in practice (3/3): knobs and a worked example =======================
  {
    const s = pres.addSlide(); title(s, "What each choice costs", "Defaults: δ = 10⁻⁶, ρ* = 2.5, ℓ from public data, tight bounds, k fixed in advance");
    const rows = [hdr(["Change", "σ", "Records for prediction", "Records for coefficients"]),
      ["Bounds 2× closer to the data (no extra clipping)", "× 1/4", "× 1/4", "× 1/16"],
      ["Double ε", "× 0.52–0.55", "× 0.52–0.55", "× 0.27–0.30"],
      ["δ from 10⁻⁶ to 10⁻⁸", "× 1.15–1.26", "× 1.15–1.26", "× 1.3–1.6"],
      ["Same budget over k = 5 releases instead of 1", "× 2.6–2.9", "× 2.6–2.9", "× 6.5–8.7"],
      ["Split a unit row bound, B_R² = B_O² = ½ (if little extra clipping)", "× 0.64", "× 0.64", "× 0.41"]];
    rows.slice(1).forEach((r) => { r[0] = { text: r[0], options: { bold: true } }; });
    s.addTable(rows, Object.assign(tableOpts([6.2, 1.6, 2.15, 2.15]), { y: 1.7, rowH: 0.4, fontSize: 13 }));
    txt(s, "Worked example: the simulation population at n = 10,000, δ = 10⁻⁶", { x: 0.6, y: 4.45, w: 12.1, h: 0.4, fontFace: HEAD, fontSize: 15, bold: true, color: INK });
    txt(s, "B_R = 5.59, B_O = 6.24, B_y = 2.89, p = 11, ℓ = 0.494, ρ* = 2.5", { x: 0.6, y: 4.83, w: 12.1, h: 0.3, fontSize: 12, italic: true, color: MUTED });
    const t4 = calSec("4.");
    const ex = [hdr(["ε", "Δ", "σ, one release", "zCDP cost ρ", "λ at n = 10k", "Ridge off from n", "σ, each of 5 releases"])];
    t4.slice(1).forEach((r) => ex.push([r[0], r[2], r[3], r[4], r[5], r[6], r[7]]));
    s.addTable(ex, Object.assign(tableOpts([1.0, 1.3, 1.9, 1.7, 1.8, 2.1, 2.3]), { y: 5.17, rowH: 0.3, fontSize: 12.5 }));
    s.addNotes("Multipliers from the s(eps, delta) table and the worked example in reports/calibration/heuristic.md. 'Bounds 2x closer' means the bounds were loose relative to the data, so tightening them adds no clipping; Delta is homogeneous of degree 2 in the bounds. Five releases: the sqrt(5) budget split times the looser zCDP conversion. Split bound: each party clips to half of a unit row bound, which needs no interaction; Delta^2 drops from 10 to 4.125 (B_y = 1), so sigma falls by 1.56x. The record multipliers assume the extra per-party clipping is negligible. The worked example prints Delta = 86.2; with the rounded bounds shown the exact value is 86.15.");
  }

  // ======================= Divider: second revision =======================
  divider(5, "Second revision: protocol v3.1", "A referee review, a redesign for honest-but-curious parties, and proofs")
    .addNotes("Part 5: a journal-style referee review of the paper, what we verified, the second revision (v3.1), and its proofs appendix (47 pages: cryptographic and DP guarantees, then sensitivity and statistical guarantees).");

  // ======================= Referee review and response =======================
  {
    const s = pres.addSlide(); title(s, "Referee review: findings and answers", "Recommendation: reject and resubmit. Every major comment accepted; the numerical claims re-checked");
    const v = (t, col) => ({ text: t, options: { bold: true, color: col } });
    const rows = [
      hdr(["Major comment", "Verdict", "In v3.1"]),
      ["Privacy theorem unproven: S_err unbounded; a hint to the key holder; no computational DP notion; error cost charged once; finite precision undefined",
        v("Accepted", INK), "Split key; full error flooded inside O's share; SIM-CDP; (ρ, δ) ledger; lattice quantization. Proven"],
      ["The semi-honest R is also the party DP protects against", v("Accepted", INK), "Stated: honest-but-curious only. Malicious R needs ZK input proofs (extension)"],
      ["Revised protocol neither implemented nor costed", v("Accepted", INK), { text: "Open: OpenFHE implementation is next", options: { color: EMBER, bold: true } }],
      ["Privacy levels ignore the error channel", v("Accepted", INK), "End-to-end (ρ, δ) accounting, tight zCDP conversion"],
      ["AdaSSP comparison mischaracterized", v("Verified", TEAL), "Published AdaSSP is more private (ε ≈ 0.82 vs 1.00); caveat replaced"],
      ["No valid inference under ℓ = 0 (coverage 42–81%)", v("Verified", TEAL), "Two-stage ridge; Wald interval for θ_λ, bias-aware interval for β"],
      ["Bias theory correct but of modest value", v("Partly", "B7791F"), "Correction optional, off by default"],
      ["yᵀy is not always free", v("Verified", TEAL), "Released only where free; exact criterion proven"],
    ];
    s.addTable(rows, Object.assign(tableOpts([5.2, 1.2, 5.7]), { y: 1.75, rowH: 0.5, fontSize: 12.5 }));
    s.addNotes("The referee (asked to review as for a top statistics journal) recommended reject and resubmit. We accepted every major comment and re-checked the numerical ones: the yT y counterexamples (Delta = 26.07 vs 14.14 at bounds (1, 1, 5)), the Gaussian-DP comparison with published AdaSSP (mu = 0.198 vs our 0.237), the 42–81% coverage under l = 0 from our own report, the matched baseline's unused budget, and the split-bound gain (Delta^2 = 4.125 vs 10). Comments 2 (malicious R) and 3 (implementation) are only partly answered: v3.1 is designed and proven for honest-but-curious parties, and it is not yet implemented.");
  }

  // ======================= v3.1 at a glance =======================
  {
    const s = pres.addSlide(); title(s, "Protocol v3.1 at a glance");
    const ph = [["key", "Split decryption key", "s = s_R + s_O. Neither party can decrypt alone; R uploads under s_R and O converts"],
      ["shield", "Noise in O's decryption share", "O sends (c₁′, h_O = c₀′ + c₁′s_O + η) and never c₀′. η floods the whole ciphertext error"],
      ["balance", "Two-stage ridge", "(λ, Ψ) from a small stage-1 release and A only, so it is independent of the stage-2 noise"],
      ["chart", "Valid intervals", "Parametric bootstrap; Wald interval for the ridge target θ_λ, bias-aware interval for β"],
      ["link", "Capped, padded linkage", "OPRF-based labeled PSI; R's cohort padded to n_R^max, O's register to n̄_O"]];
    ph.forEach(([ic, h, b], i) => {
      const y = 1.45 + i * 1.12;
      circleIcon(s, ic, 0.6, y, 0.62, i === 1 ? EMBER : INK);
      txt(s, h, { x: 1.4, y: y - 0.02, w: 4.5, h: 0.4, fontFace: HEAD, fontSize: 17, bold: true, color: INK });
      txt(s, b, { x: 1.4, y: y + 0.38, w: 4.5, h: 0.7, fontSize: 12.5, color: MUTED });
    });
    img(s, P("slides", "assets", "protocol_v31.png"), 6.75, 1.2, 6.2, 6.05, 1963, 1867);
    s.addNotes("The five changes from v2. The release in phase 4 is the core: O re-randomizes the packed result with fresh (u, e0, e1), adds an exact discrete-Gaussian η to its own decryption share, and sends only (c1', h_O). R adds c1' s_R and decodes. Because c0' never leaves O, R cannot recover c1' s_O, and η is wide enough to hide the whole error statistically, so R's view depends on the data only through the DP-noised statistic. Everything is analysed for honest-but-curious parties.");
  }

  // ======================= What is proven =======================
  {
    const s = pres.addSlide(); title(s, "What the proofs establish", "Proofs appendix: Part A, cryptography and DP (25 pp.); Part B, sensitivity and statistics (20 pp.)");
    const st = (t) => ({ text: t, options: { bold: true, color: t.startsWith("First") || t.startsWith("Partly") ? "B7791F" : TEAL } });
    const A = [hdr(["Part A: cryptography and DP", "Status"]),
      ["Encoding sensitivity √(2/N)", st("Proven")],
      ["Smudging lemma, explicit constants", st("Proven")],
      ["Worst-case error bound B_tot ≈ 2^25.3; OpenFHE measures 6–8 bits below it", st("Proven")],
      ["zCDP of the ideal mechanism; ρ_run = 0.01496 at ε = 1, δ = 10⁻⁹", st("Proven")],
      ["Output privacy (SIM-CDP) by hybrids, under RLWE and circular security", st("Proven")],
      ["Input and cohort privacy; correctness without wraparound", st("Proven")]];
    const B = [hdr(["Part B: sensitivity and statistics", "Status"]),
      ["Replace-one sensitivity, exact for d_O ≥ 2", st("Proven")],
      ["yᵀy free if B_y² ≤ 2(B_R² + B_O²), costly if > 3(B_R² + B_O²)", st("Proven")],
      ["Stage-1 ridge choice independent of the stage-2 noise", st("Proven")],
      ["Operator-norm radius r₁ (one-sided; ~1.55× conservative)", st("Proven")],
      ["Coverage: Wald remainder exact; plug-in variance; simulated 0.944–0.962", st("First order")],
      ["Bootstrap RSS correction; removes 87–100% of the bias", st("First order")]];
    s.addTable(A, { x: 0.6, y: 1.75, colW: [4.6, 1.3], fontFace: BODY, fontSize: 12, color: TEXT, border: { type: "solid", color: LINE, pt: 0.75 }, valign: "middle", margin: [3, 6, 3, 6], rowH: 0.52 });
    s.addTable(B, { x: 6.8, y: 1.75, colW: [4.6, 1.3], fontFace: BODY, fontSize: 12, color: TEXT, border: { type: "solid", color: LINE, pt: 0.75 }, valign: "middle", margin: [3, 6, 3, 6], rowH: 0.52 });
    card(s, 0.6, 5.85, 12.1, 0.95, "F7F9FC");
    txt(s, [{ text: "Still open:  ", options: { bold: true, color: EMBER } },
      { text: "proofs for the improved circuits; composition across adaptive runs; a constant-time exact sampler; modulus-chain sizes; the exact free-set boundary for yᵀy; heteroskedasticity when clipping binds; 16 citation details to check." }],
      { x: 0.85, y: 5.95, w: 11.6, h: 0.75, fontSize: 13, valign: "middle" });
    s.addNotes("Status of every claim in the v3.1 specification. Part A: the encoding sensitivity, the smudging lemma with explicit constants, an explicit 8-term worst-case bound on the ciphertext error (checked against OpenFHE, which measured 2^14 to 2^15.7 where the bound gives about 2^21.9), the zCDP accounting with the Canonne–Kamath–Steinke conversion, output privacy as a full hybrid proof, input and cohort privacy by simulators, and correctness. Part B: the exact replace-one sensitivity (the triu and Frobenius suprema coincide), when yT y is free, the operator-norm radius used by the gate, the independence of the stage-1 ridge choice from stage 2, the bound on the norm of beta, coverage of both intervals (first order: the plug-in variance; asymptotic: normality of the sampling term), and the bootstrap correction of the residual sum of squares.");
  }

  // ======================= What the proofs changed =======================
  {
    const s = pres.addSlide(); title(s, "What the proofs changed", "The CKKS scale was undersized, and several statistical steps need tightening");
    card(s, 0.6, 1.75, 5.6, 4.95, EMBER_L);
    txt(s, "61.5 bits", { x: 0.9, y: 1.9, w: 5.0, h: 1.0, fontFace: HEAD, fontSize: 54, bold: true, color: EMBER });
    txt(s, "CKKS scale needed to flood the worst-case error for the circuit as specified. The v3.1 spec assumed 52.5, which floods only the fresh-encryption error, at its mean.",
      { x: 0.9, y: 2.95, w: 5.05, h: 1.1, fontSize: 13.5, color: INK });
    const sc = [hdr(["Circuit", "B_tot", "log₂ Ω"]),
      ["As specified", "2^25.3", "61.5"],
      ["EvalSum before the first rescale, stochastic rounding, one final rescale", "2^21.1", "57.2"],
      ["Same, with R encoding at scale 2^60", "2^17.6", "53.7"]];
    s.addTable(sc, { x: 0.9, y: 4.2, colW: [3.2, 0.95, 0.85], fontFace: BODY, fontSize: 12, color: TEXT, border: { type: "solid", color: LINE, pt: 0.75 }, valign: "middle", margin: [3, 5, 3, 5], rowH: 0.42, fill: { color: WHITE } });
    txt(s, "Corrections to the specification", { x: 6.6, y: 1.75, w: 6.1, h: 0.45, fontFace: HEAD, fontSize: 18, bold: true, color: INK });
    bullets(s, [
      "Centre the mechanism at the noiseless circuit output; centring at the encoded input would need 67–71 bits",
      "Calibrate from the data-independent worst-case bound, never from OpenFHE's noise estimate",
      "The δ slack enters a single (ε, δ) statement with a factor (1 + e^ε)",
      "Define β as the coefficient on the clipped data; clipping breaks the homoskedastic variance",
      "Truncate the bootstrap (its moments do not exist otherwise); spend α_U once in U_yy",
      "The private bound on ‖β‖ is usually infinite while the ridge is on: the β interval then needs a public bound",
    ], { x: 6.6, y: 2.3, w: 6.1, h: 4.4, fontSize: 13.5, paraSpaceAfter: 7 });
    s.addNotes("The main finding of the crypto proofs: flooding the whole ciphertext error in l2 at 2^-40 needs B_tot of about 2^25.3 for stage 2 of the example circuit, hence a CKKS scale of about 2^61.5. Two circuit changes help: evaluate EvalSum before the first rescale, use stochastic rounding and a single final rescale (2^57.2); and additionally have R encode at scale 2^60 (2^53.7). The deterministic, data-dependent parts of the error (such as the mask spill) are part of the query: the mechanism must be centred at the noiseless circuit output and use that function's sensitivity, which rises by a relative 2^-33; flooding them instead would need 67 to 71 bits. On the statistics side: state the estimand on the clipped data, truncate the bootstrap or use the closed-form alternative, fix the double use of alpha_U, make the radius statement one-sided, and say whether coverage is conditional on passing the gate.");
  }

  // ======================= 30. Conclusions =======================
  {
    const s = pres.addSlide(); s.background = { color: INK };
    s.addText("Takeaways and next steps", { x: 0.6, y: 0.45, w: 12, h: 0.8, fontFace: HEAD, fontSize: 34, bold: true, color: WHITE, margin: 0, isTextBox: true });
    const take = ["v3.1 closes the critical gaps for honest-but-curious parties, with proofs; its CKKS scale must rise to about 2^61.5 (2^53.7 with circuit changes)",
      "Within 2× of non-private OLS at n ≈ 3k–100k (simulation); within 10% at 1.4k–11k records for ε = 1–2 (real data)",
      "Published AdaSSP is more private than our release and still wins from ε ≈ 0.5; we need about 1.5–2× its records",
      "The bias correction is a small effect, now optional; ridge shrinkage dominates when σ is large"];
    take.forEach((t, i) => {
      numCircle(s, i + 1, 0.6, 1.6 + i * 1.25, 0.6, EMBER, 17);
      s.addText(t, { x: 1.45, y: 1.6 + i * 1.25, w: 5.4, h: 1.1, fontFace: BODY, fontSize: 15.5, color: WHITE, margin: 0, valign: "top", isTextBox: true });
    });
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 7.2, y: 1.5, w: 5.5, h: 5.2, fill: { color: "1F3050" }, line: { color: "1F3050" }, rectRadius: 0.08 });
    s.addText("Next", { x: 7.5, y: 1.7, w: 5, h: 0.5, fontFace: HEAD, fontSize: 22, bold: true, color: "E7A77C", margin: 0, isTextBox: true });
    const nxt = ["Implement v3.1 in OpenFHE with the improved circuit, and measure its cost",
      "Rerun the evaluation under v3.1: two-stage ridge, intervals, split bounds",
      "Fairer AdaSSP comparison: equal μ, a Gaussian-DP budget split, preprocessing that VFL can do",
      "Close the open proof items; fold the corrections into a v3.2 specification",
      "Membership-private variant for sensitive registers"];
    s.addText(nxt.map((t, i) => ({ text: t, options: { bullet: true, breakLine: i < nxt.length - 1, paraSpaceAfter: 9 } })),
      { x: 7.5, y: 2.35, w: 4.95, h: 4.2, fontFace: BODY, fontSize: 15, color: WHITE, margin: 0, valign: "top", isTextBox: true });
    s.addNotes("Four takeaways and five next steps. The comparison with AdaSSP is now stated the right way round: published AdaSSP is mu = 0.198 Gaussian DP against our 0.237, so it is more private under a stronger adjacency and still ahead from eps of about 0.5. At very small eps both are often no better than predicting zero. Splitting one row bound between the parties cuts our sigma by 1.56x and should narrow the gap.");
  }

  const out = P("slides", "vfl_scenarioB_deck.pptx");
  const JSZip = require("jszip");
  const zip = await JSZip.loadAsync(await pres.write({ outputType: "nodebuffer" }));
  const lit = (vals) => `<c:numLit><c:formatCode>General</c:formatCode><c:ptCount val="${vals.length}"/>` +
    vals.map((v, i) => `<c:pt idx="${i}"><c:v>${v.toPrecision(4)}</c:v></c:pt>`).join("") + "</c:numLit>";
  let nBars = 0;
  for (const f of Object.keys(zip.files).filter((f) => /^ppt\/charts\/chart\d+\.xml$/.test(f))) {
    let xml = await zip.file(f).async("string");
    const key = Object.keys(ERRBARS).find((t) => xml.includes(t.replace(/&/g, "&amp;")));
    if (!key) continue;
    xml = xml.replace(/<c:ser>[\s\S]*?<\/c:ser>/g, (ser) => {
      const name = Object.keys(ERRBARS[key]).find((nm) => ser.includes(`<c:v>${nm}</c:v>`));
      if (!name) return ser;
      const col = (ser.match(/<a:srgbClr val="([0-9A-Fa-f]{6})"/) || [, "5B6576"])[1];
      const v = ERRBARS[key][name];
      nBars++;
      return ser.replace("<c:cat>", `<c:errBars><c:errDir val="y"/><c:errBarType val="both"/><c:errValType val="cust"/>` +
        `<c:noEndCap val="0"/><c:plus>${lit(v)}</c:plus><c:minus>${lit(v)}</c:minus>` +
        `<c:spPr><a:ln w="12700"><a:solidFill><a:srgbClr val="${col}"/></a:solidFill></a:ln></c:spPr></c:errBars><c:cat>`);
    });
    zip.file(f, xml);
  }
  fs.writeFileSync(out, await zip.generateAsync({ type: "nodebuffer", compression: "DEFLATE" }));
  console.log("error-bar series added:", nBars);
  console.log("wrote", out);
})();
