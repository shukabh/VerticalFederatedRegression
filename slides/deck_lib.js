// Shared pieces of the slide decks: palette, data readers, slide helpers, math formatting, and the
// post-processing that pptxgenjs cannot do itself (chart error bars, valid paragraph properties).
// Requires: npm install pptxgenjs jszip react-icons react react-dom sharp
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");
const JSZip = require("jszip");
const React = require("react");
const ReactDOMServer = require("react-dom/server");
const sharp = require("sharp");

const ROOT = path.resolve(__dirname, "..");
const P = (...p) => path.join(ROOT, ...p);

function readCsv(file) {
  const [head, ...rows] = fs.readFileSync(file, "utf8").trim().split(/\r?\n/);
  const keys = head.split(",");
  return rows.map((r) => Object.fromEntries(r.split(",").map((v, i) => [keys[i], v])));
}
// Markdown tables of a report, keyed by the "## " heading above them
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

// ---------- palette & type ----------
const C = {
  INK: "14213D",      // dominant dark
  SLATE: "3D5A80",    // secondary
  MIST: "EEF2F7",     // light panels
  LINE: "C9D3E0",
  EMBER: "C45A0C",    // accent; O's colour in the protocol figure
  EMBER_L: "F6E3D6",
  TEAL: "1B8A7A",
  AMBER: "B7791F",
  TEXT: "1F2937",
  MUTED: "5B6576",
  WHITE: "FFFFFF",
};
const HEAD = "Cambria", BODY = "Calibri";

// ---------- math in slide text ----------
// Every addText/addTable string is scanned for sub- and superscripts and split into runs:
//   X_R, c_O,k, Δ_{rep}  -> subscript      2^61.5, e^ε, n_R^max, x^{...} -> superscript
//   Unicode marks (ᵀ ⁻¹ ² ᴵ, ₀ ₁ ₂) and ρ* become real sub/superscript runs too.
const SUPC = { "⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4", "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9", "⁻": "−", "⁺": "+", "ᵀ": "T", "ᴵ": "I" };
const SUBC = { "₀": "0", "₁": "1", "₂": "2", "₃": "3", "₄": "4", "₅": "5", "₆": "6", "₇": "7", "₈": "8", "₉": "9" };
const IDC = "A-Za-z0-9α-ωΑ-Ωℓℤ";
const MATH_RX = new RegExp(`_\\{([^}]*)\\}|_([${IDC}]+(?:,[${IDC}]+)*)` +
  `|\\^\\{([^}]*)\\}|\\^([−+-]?(?:[0-9]+(?:\\.[0-9]+)?|[A-Za-zα-ωΑ-Ω]+))` +
  `|([${Object.keys(SUPC).join("")}]+)|([${Object.keys(SUBC).join("")}]+)|(?<=[ρϑθ])(\\*)`, "gu");
const PARA_KEYS = ["bullet", "paraSpaceAfter", "paraSpaceBefore", "align", "indentLevel", "lineSpacing", "lineSpacingMultiple"];
const hasMath = (t) => { MATH_RX.lastIndex = 0; return typeof t === "string" && MATH_RX.test(t); };
function mathRuns(text, opts) {
  const out = [];
  const push = (t, extra) => { if (t) out.push({ text: t, options: Object.assign({}, opts, extra) }); };
  let last = 0, m;
  MATH_RX.lastIndex = 0;
  while ((m = MATH_RX.exec(text))) {
    push(text.slice(last, m.index));
    if (m[1] !== undefined || m[2] !== undefined) push(m[1] ?? m[2], { subscript: true });
    else if (m[3] !== undefined || m[4] !== undefined) push(m[3] ?? m[4], { superscript: true });
    else if (m[5]) push([...m[5]].map((c) => SUPC[c]).join(""), { superscript: true });
    else if (m[6]) push([...m[6]].map((c) => SUBC[c]).join(""), { subscript: true });
    else push("*", { superscript: true });
    last = MATH_RX.lastIndex;
  }
  push(text.slice(last));
  return out.length ? out : [{ text: "", options: Object.assign({}, opts) }];
}
// Split text items into runs: paragraph options stay on the first run of each line, breakLine on the last.
function mathItems(items) {
  const res = [];
  items.forEach((it) => {
    const o = it.options || {}, run = {}, para = {};
    Object.entries(o).forEach(([k, v]) => { if (k !== "breakLine") (PARA_KEYS.includes(k) ? para : run)[k] = v; });
    const lines = String(it.text ?? "").split(/\r?\n/);
    lines.forEach((line, li) => {
      const pieces = mathRuns(line, run);
      Object.assign(pieces[0].options, para);
      if (li < lines.length - 1 || o.breakLine) pieces[pieces.length - 1].options.breakLine = true;
      res.push(...pieces);
    });
  });
  return res;
}
function mathText(t) {
  if (typeof t === "string") return hasMath(t) ? mathItems([{ text: t }]) : t;
  if (Array.isArray(t)) {
    const items = t.map((x) => (typeof x === "string" ? { text: x } : x));
    return items.some((x) => hasMath(x.text)) ? mathItems(items) : t;
  }
  return t;
}
const mathCell = (c) => {
  if (typeof c === "string") return hasMath(c) ? { text: mathItems([{ text: c }]) } : c;
  if (c && typeof c.text === "string") return hasMath(c.text) ? { text: mathItems([{ text: c.text }]), options: c.options } : c;
  if (c && Array.isArray(c.text)) return c.text.some((x) => hasMath(x.text)) ? { text: mathItems(c.text), options: c.options } : c;
  return c;
};

async function iconPng(Icon, color = "#FFFFFF", size = 256) {
  const svg = ReactDOMServer.renderToStaticMarkup(React.createElement(Icon, { color, size: String(size) }));
  const buf = await sharp(Buffer.from(svg)).png().toBuffer();
  return "image/png;base64," + buf.toString("base64");
}

// A 16:9 deck whose slides route all text through the math formatter, plus the slide helpers.
async function newDeck(title, iconSet) {
  const icons = {};
  for (const [k, I] of Object.entries(iconSet)) icons[k] = await iconPng(I);
  const EQ = JSON.parse(fs.readFileSync(P("slides", "assets", "eq", "manifest.json"), "utf8"));
  const pres = new pptxgen();
  const addSlide0 = pres.addSlide.bind(pres);
  pres.addSlide = (...a) => {
    const s = addSlide0(...a);
    const addText0 = s.addText.bind(s), addTable0 = s.addTable.bind(s);
    s.addText = (t, o) => addText0(mathText(t), o);
    s.addTable = (rows, o) => addTable0(rows.map((r) => r.map(mathCell)), o);
    return s;
  };
  pres.layout = "LAYOUT_WIDE";
  pres.author = "VFL Scenario B project";
  pres.title = title;
  const { INK, SLATE, MIST, LINE, EMBER, TEXT, MUTED, WHITE } = C;

  const h = {};
  h.title = (s, t, sub) => {
    s.addText(t, { x: 0.6, y: 0.35, w: 12.1, h: 0.8, fontFace: HEAD, fontSize: 32, bold: true,
      color: INK, margin: 0, isTextBox: true });
    if (sub) s.addText(sub, { x: 0.6, y: 1.12, w: 12.1, h: 0.45, fontFace: BODY, fontSize: 16,
      color: MUTED, italic: true, margin: 0, isTextBox: true });
  };
  h.circleIcon = (s, key, x, y, d, fill = INK) => {
    s.addShape(pres.shapes.OVAL, { x, y, w: d, h: d, fill: { color: fill }, line: { color: fill } });
    const pad = d * 0.24;
    s.addImage({ data: icons[key], x: x + pad, y: y + pad, w: d - 2 * pad, h: d - 2 * pad });
  };
  h.numCircle = (s, n, x, y, d, fill = EMBER, fs = 16) => {
    s.addShape(pres.shapes.OVAL, { x, y, w: d, h: d, fill: { color: fill }, line: { color: fill } });
    s.addText(String(n), { x, y, w: d, h: d, align: "center", valign: "middle", fontFace: BODY,
      fontSize: fs, bold: true, color: WHITE, margin: 0, isTextBox: true });
  };
  h.card = (s, x, y, w, hh, fill = MIST) =>
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h: hh, fill: { color: fill }, line: { color: fill }, rectRadius: 0.08 });
  h.txt = (s, t, o) => s.addText(t, Object.assign({ fontFace: BODY, fontSize: 15, color: TEXT, margin: 0,
    valign: "top", isTextBox: true }, o));
  h.bullets = (s, items, o) => s.addText(items.map((it, i) => ({
    text: typeof it === "string" ? it : it.text,
    options: Object.assign({ bullet: true, breakLine: i < items.length - 1, paraSpaceAfter: 6 },
      typeof it === "string" ? {} : it.options || {}),
  })), Object.assign({ fontFace: BODY, fontSize: 15, color: TEXT, margin: 0, valign: "top", isTextBox: true }, o));
  h.chip = (s, t, x, y, w, fill, color = WHITE) => {
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h: 0.32, fill: { color: fill }, line: { color: fill }, rectRadius: 0.08 });
    s.addText(t, { x, y, w, h: 0.32, align: "center", valign: "middle", fontFace: BODY, fontSize: 11, bold: true,
      color, margin: 0, isTextBox: true });
  };
  h.img = (s, file, x, y, maxW, maxH, pxW, pxH) => {
    const r = pxW / pxH;
    let w = maxW, hh = maxW / r;
    if (hh > maxH) { hh = maxH; w = maxH * r; }
    s.addImage({ path: file, x: x + (maxW - w) / 2, y, w, h: hh });
    return { w, h: hh };
  };
  h.placeholder = (s, x, y, w, hh, head, body) => {
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h: hh, fill: { color: "FAFBFD" },
      line: { color: SLATE, width: 1.25, dashType: "dash" }, rectRadius: 0.08 });
    h.circleIcon(s, "hourglass", x + w / 2 - 0.35, y + hh / 2 - 0.95, 0.7, SLATE);
    s.addText(head, { x: x + 0.3, y: y + hh / 2 - 0.15, w: w - 0.6, h: 0.45, align: "center", fontFace: BODY,
      fontSize: 16, bold: true, color: INK, margin: 0, isTextBox: true });
    s.addText(body, { x: x + 0.4, y: y + hh / 2 + 0.3, w: w - 0.8, h: hh / 2 - 0.45, align: "center", fontFace: BODY,
      fontSize: 12.5, color: MUTED, margin: 0, valign: "top", isTextBox: true });
  };
  // Display equation typeset by slides/make_equations.py (LaTeX, 10 pt): scaled to `size` pt, shrunk to fit maxW/maxH
  h.latex = (s, key, x, y, { size = 18, w = 0, maxW, maxH, align = "l" } = {}) => {
    const e = EQ[key], k = size / 10 / 72;
    const f = Math.min(1, maxW ? maxW / (e.w_pt * k) : 1, maxH ? maxH / (e.h_pt * k) : 1);
    const ew = e.w_pt * k * f, eh = e.h_pt * k * f;
    const ex = align === "c" ? x + (w - ew) / 2 : align === "r" ? x + w - ew : x;
    s.addImage({ path: P("slides", "assets", "eq", e.file), x: ex, y, w: ew, h: eh });
    return { w: ew, h: eh };
  };
  h.tableOpts = (colW) => ({ x: 0.6, colW, fontFace: BODY, fontSize: 12.5, color: TEXT, border: { type: "solid", color: LINE, pt: 0.75 },
    valign: "middle", margin: [3, 6, 3, 6] });
  h.hdr = (cells) => cells.map((c) => ({ text: c, options: { bold: true, color: WHITE, fill: { color: INK } } }));
  h.divider = (n, t, sub) => {
    const s = pres.addSlide(); s.background = { color: INK };
    h.numCircle(s, n, 0.9, 2.55, 1.3, EMBER, 36);
    s.addText(t, { x: 2.6, y: 2.45, w: 9.5, h: 0.9, fontFace: HEAD, fontSize: 40, bold: true, color: WHITE, margin: 0, isTextBox: true });
    s.addText(sub, { x: 2.6, y: 3.35, w: 9.5, h: 0.6, fontFace: BODY, fontSize: 18, color: "CAD5E6", margin: 0, isTextBox: true });
    return s;
  };
  return { pres, icons, h };
}

// Write the deck, adding what pptxgenjs cannot:
//  - error bars on native charts: errbars[chart title][series name] = half-widths
//  - valid paragraphs: pptxgenjs writes an <a:pPr> before every run of a multi-run paragraph, while
//    OOXML allows one, at the start; and PowerPoint's own subscript offset (-25%) instead of -40%.
async function writeDeck(pres, out, errbars = {}) {
  const zip = await JSZip.loadAsync(await pres.write({ outputType: "nodebuffer" }));
  const lit = (vals) => `<c:numLit><c:formatCode>General</c:formatCode><c:ptCount val="${vals.length}"/>` +
    vals.map((v, i) => `<c:pt idx="${i}"><c:v>${v.toPrecision(4)}</c:v></c:pt>`).join("") + "</c:numLit>";
  let nBars = 0;
  for (const f of Object.keys(zip.files).filter((f) => /^ppt\/charts\/chart\d+\.xml$/.test(f))) {
    let xml = await zip.file(f).async("string");
    const key = Object.keys(errbars).find((t) => xml.includes(t.replace(/&/g, "&amp;")));
    if (!key) continue;
    xml = xml.replace(/<c:ser>[\s\S]*?<\/c:ser>/g, (ser) => {
      const name = Object.keys(errbars[key]).find((nm) => ser.includes(`<c:v>${nm}</c:v>`));
      if (!name) return ser;
      const col = (ser.match(/<a:srgbClr val="([0-9A-Fa-f]{6})"/) || [, "5B6576"])[1];
      const v = errbars[key][name];
      nBars++;
      return ser.replace("<c:cat>", `<c:errBars><c:errDir val="y"/><c:errBarType val="both"/><c:errValType val="cust"/>` +
        `<c:noEndCap val="0"/><c:plus>${lit(v)}</c:plus><c:minus>${lit(v)}</c:minus>` +
        `<c:spPr><a:ln w="12700"><a:solidFill><a:srgbClr val="${col}"/></a:solidFill></a:ln></c:spPr></c:errBars><c:cat>`);
    });
    zip.file(f, xml);
  }
  const PPR = /<a:pPr\b[^>]*?(?:\/>|>[\s\S]*?<\/a:pPr>)/g;
  let nDup = 0;
  for (const f of Object.keys(zip.files).filter((f) => /^ppt\/slides\/slide\d+\.xml$/.test(f))) {
    let xml = await zip.file(f).async("string");
    xml = xml.replace(/<a:p>([\s\S]*?)<\/a:p>/g, (para, body) => {
      const firstRun = body.search(/<a:(r|br|fld)\b/);
      let kept = false;
      const clean = body.replace(PPR, (ppr, at) => {
        if (!kept && (firstRun < 0 || at < firstRun)) { kept = true; return ppr; }
        nDup++; return "";
      });
      return `<a:p>${clean}</a:p>`;
    }).replace(/baseline="-40000"/g, 'baseline="-25000"');
    zip.file(f, xml);
  }
  fs.writeFileSync(out, await zip.generateAsync({ type: "nodebuffer", compression: "DEFLATE" }));
  console.log(`error-bar series added: ${nBars}; duplicate paragraph properties removed: ${nDup}`);
  console.log("wrote", out);
}

module.exports = { ROOT, P, readCsv, mdTables, C, HEAD, BODY, newDeck, writeDeck };
