// Build a four-slide explainer of the two cryptographic building blocks as the protocol uses them:
// threshold CKKS (key generation, upload, and the noisy threshold decryption) and the OPRF-based labeled PSI.
// Requires: npm install pptxgenjs jszip react-icons react react-dom sharp
// Run from the repo root:  node slides/build_crypto_slides.js
// Display equations are LaTeX images: after editing slides/equations.json, run  python slides/make_equations.py  first.
const fa = require("react-icons/fa");
const { P, C, HEAD, newDeck, writeDeck } = require("./deck_lib");
const { INK, SLATE, MIST, EMBER, EMBER_L, MUTED, WHITE } = C;
const RBLUE = "2A78D6", RBLUE_L = "E3EEFB";          // R's colour in the protocol figure

(async () => {
  const { pres, h } = await newDeck("Threshold CKKS and OPRF-based PSI in the protocol", {
    lock: fa.FaLock, key: fa.FaKey, shield: fa.FaUserShield,
  });
  const { title, circleIcon, numCircle, card, txt, bullets, chip, latex, tableOpts } = h;
  const head = (s, icon, t, x, y, col) => {
    circleIcon(s, icon, x + 0.25, y + 0.2, 0.5, col);
    txt(s, t, { x: x + 0.9, y: y + 0.22, w: 4.9, h: 0.45, fontFace: HEAD, fontSize: 17, bold: true, color: INK });
  };
  const numHead = (s, n, t, x, y, col) => {
    numCircle(s, n, x + 0.22, y + 0.18, 0.45, col, 14);
    txt(s, t, { x: x + 0.82, y: y + 0.2, w: 3.0, h: 0.4, fontFace: HEAD, fontSize: 15, bold: true, color: INK });
  };
  const strip = (s, t) => {
    card(s, 0.6, 6.55, 12.1, 0.65, "F7F9FC");
    txt(s, t, { x: 0.85, y: 6.57, w: 11.6, h: 0.61, fontSize: 12.5, valign: "middle" });
  };

  // ---------- 1. threshold CKKS: keys and upload ----------
  {
    const s = pres.addSlide();
    title(s, "Threshold CKKS: one key, split between R and O", "Nobody ever holds s = s_R + s_O, so neither party can decrypt on its own");

    card(s, 0.6, 1.75, 5.9, 4.65, MIST);
    head(s, "lock", "CKKS in one box", 0.6, 1.75, INK);
    txt(s, "A ciphertext is a pair (c_0, c_1) of polynomials mod q. It packs a vector of N/2 = 8,192 real numbers (N = 2^14), each scaled by Ω and rounded.",
      { x: 0.85, y: 2.6, w: 5.45, h: 0.75, fontSize: 13 });
    txt(s, "Decryption with the secret key s:", { x: 0.85, y: 3.4, w: 5.45, h: 0.3, fontSize: 13 });
    latex(s, "ckks_dec", 0.85, 3.75, { size: 15 });
    txt(s, "What O can do with R's ciphertexts, holding no key:", { x: 0.85, y: 4.15, w: 5.45, h: 0.3, fontSize: 13, bold: true, color: INK });
    bullets(s, ["add them",
      "multiply them by its own plaintext columns (x_O, y)",
      "rotate slots, to sum a vector into one slot (EvalSum)"],
    { x: 0.85, y: 4.5, w: 5.45, h: 0.95, fontSize: 12.5, paraSpaceAfter: 2 });
    txt(s, "Depth 2 and no ciphertext × ciphertext products, so no relinearization key. Security rests on Ring-LWE: without s, c_0 looks uniformly random.",
      { x: 0.85, y: 5.55, w: 5.45, h: 0.7, fontSize: 12, color: MUTED });

    card(s, 6.7, 1.75, 6.0, 4.65, MIST);
    head(s, "key", "Joint key generation (setup)", 6.7, 1.75, SLATE);
    txt(s, "From a common random polynomial a, each party draws its own secret share and error, and publishes one polynomial:",
      { x: 6.95, y: 2.6, w: 5.55, h: 0.5, fontSize: 13 });
    latex(s, "he_keygen", 6.95, 3.15, { size: 13 });
    txt(s, "Rotation keys for slot shifts 2^i are made the same way. Each share stays with its owner.",
      { x: 6.95, y: 4.02, w: 5.55, h: 0.45, fontSize: 12, color: MUTED });
    txt(s, "Upload under the joint key, at half the size", { x: 6.95, y: 4.5, w: 5.55, h: 0.3, fontSize: 13, bold: true, color: INK });
    latex(s, "he_upload2", 6.95, 4.85, { size: 12, maxW: 5.5 });
    txt(s, "R sends c_0 and the short seed of a, not a itself. O's subtraction turns (c_0, a) into a ciphertext under s.",
      { x: 6.95, y: 5.45, w: 5.55, h: 0.75, fontSize: 12, color: MUTED });

    strip(s, "This is standard multiparty CKKS (Mouchet et al., 2021) with two parties and additive key shares. What is specific to us is how the decryption is used: next slide.");
    s.addNotes("CKKS encrypts vectors of real numbers: a ciphertext is a pair of polynomials, it holds 8,192 numbers at N = 2^14, and decryption is c0 + c1 s, which gives the encoded vector plus a small error. In our protocol O only adds ciphertexts, multiplies them by its own plaintext columns, and rotates slots to sum vectors, all without any key. The multiplicative depth is 2 and there are no ciphertext-by-ciphertext products, so no relinearization key is needed. The key is generated jointly, following Mouchet et al. 2021: from a common random a, R publishes -a s_R + e_R and O publishes -a s_O + e_O; the sum is a public key for s = s_R + s_O, a key that nobody holds. Rotation keys are made the same way. For the upload, R encrypts with its own share using a symmetric encryption whose random a comes from a seed, so it sends only c0 and the seed. O subtracts a times its share, which converts the ciphertext to the joint key. This is the standard multiparty construction with two parties.");
  }

  // ---------- 2. threshold decryption with the noise ----------
  {
    const s = pres.addSlide();
    title(s, "Threshold decryption, with the DP noise built in", "O adds its key share and the noise in one step; R finishes the decryption and sees only noisy statistics");

    card(s, 0.6, 1.75, 12.1, 1.0, EMBER_L);
    txt(s, "Decryption needs both shares:", { x: 0.9, y: 1.75, w: 3.5, h: 1.0, fontSize: 14, bold: true, color: INK, valign: "middle" });
    latex(s, "thresh", 4.5, 1.97, { size: 15 });

    card(s, 0.6, 2.95, 5.6, 2.45, "F7F9FC");
    txt(s, "Our release (phase 4)", { x: 0.85, y: 3.08, w: 5.1, h: 0.35, fontFace: HEAD, fontSize: 15, bold: true, color: INK });
    latex(s, "release", 0.85, 3.6, { size: 14.5, maxW: 5.1 });

    const pts = [
      ["Fresh u, e_0, e_1 every release", "The evaluated c_1 depends on O's plaintext columns; adding au + e_1 makes c_1′ a fresh RLWE sample that looks uniform."],
      ["Add η inside O's share", "η is the DP noise and also drowns the CKKS error e, up to 2⁻⁴⁰. It is fresh too, so two releases cannot be differenced."],
      ["Never send c_0′", "With c_0′, R would get c_1′s_O exactly and could solve for s_O, since c_1′ is invertible."],
    ];
    pts.forEach(([hd, b], i) => {
      const y = 2.95 + i * 0.83;
      numCircle(s, i + 1, 6.45, y + 0.03, 0.4, i === 1 ? EMBER : SLATE, 13);
      txt(s, hd, { x: 7.0, y, w: 5.7, h: 0.3, fontSize: 13.5, bold: true, color: INK });
      txt(s, b, { x: 7.0, y: y + 0.3, w: 5.7, h: 0.5, fontSize: 11.5, color: MUTED });
    });

    card(s, 0.6, 5.6, 5.95, 1.15, EMBER_L);
    txt(s, [{ text: "O learns nothing. ", options: { bold: true, color: EMBER } },
      { text: "It never receives a decryption share from R, so every ciphertext it holds stays hidden by Ring-LWE." }],
    { x: 0.85, y: 5.65, w: 5.5, h: 1.05, fontSize: 13, valign: "middle" });
    card(s, 6.75, 5.6, 5.95, 1.15, RBLUE_L);
    txt(s, [{ text: "R learns the noisy statistics. ", options: { bold: true, color: RBLUE } },
      { text: "Plus a c_1′ that looks random, so its view can be simulated from the noisy statistics alone: the DP guarantee." }],
    { x: 7.0, y: 5.65, w: 5.5, h: 1.05, fontSize: 13, valign: "middle" });
    s.addNotes("Decryption is linear in the key, so it splits: c0 + c1 s_O needs O's share and c1 s_R needs R's. Neither party alone can decrypt. In our release O does three things. It re-randomizes the packed ciphertext with fresh u, e0, e1 under the joint public key: the evaluated c1 is computed from public values and O's plaintext columns, and R chose the random a, so without this step c1 could reveal O's data; after it, c1' is a fresh Ring-LWE sample. Then O computes its share of the decryption and adds the noise eta inside it: eta is the differential-privacy noise on the statistics and also hides the CKKS approximation error statistically. Finally O sends c1' and the noisy share, never c0': with c0', R could compute c1' s_O exactly and recover O's key share. R adds c1' s_R and gets the statistics plus noise. O learns nothing because it never receives a decryption share. R's view can be simulated from the noisy statistics alone, which is what makes the DP guarantee hold for R's whole view.");
  }

  // ---------- 3. the OPRF ----------
  {
    const s = pres.addSlide();
    title(s, "The OPRF: a keyed hash that needs O", "R learns F_k(id) for its own ids; O never sees the ids, and R never sees the key k");

    card(s, 0.6, 1.75, 7.3, 3.7, MIST);
    latex(s, "ps_oprf", 0.9, 1.97, { size: 16 });
    txt(s, "2HashDH (RFC 9497); O's key k is fresh each run", { x: 3.45, y: 1.97, w: 4.35, h: 0.32, fontSize: 11.5, color: MUTED, valign: "middle" });
    const xR = 1.25, xO = 7.25;
    chip(s, "R", xR - 0.21, 2.55, 0.42, RBLUE);
    chip(s, "O", xO - 0.21, 2.55, 0.42, EMBER);
    [xR, xO].forEach((x) => s.addShape(pres.shapes.LINE, { x, y: 2.9, w: 0, h: 2.35, line: { color: "AAB6C6", width: 1.25 } }));
    s.addShape(pres.shapes.LINE, { x: xR, y: 3.45, w: xO - xR, h: 0, line: { color: RBLUE, width: 2, endArrowType: "triangle" } });
    txt(s, "α = H_1(id)^r, with r random", { x: xR, y: 3.1, w: xO - xR, h: 0.3, fontSize: 13, align: "center" });
    s.addShape(pres.shapes.LINE, { x: xR, y: 4.12, w: xO - xR, h: 0, flipH: true, line: { color: EMBER, width: 2, endArrowType: "triangle" } });
    txt(s, "β = α^k", { x: xR, y: 3.77, w: xO - xR, h: 0.3, fontSize: 13, align: "center" });
    txt(s, "R unblinds: β^{1/r} = H_1(id)^k. That gives the item F_k(id), used for matching, and the label key F′_k(id).",
      { x: xR + 0.2, y: 4.4, w: 5.6, h: 0.75, fontSize: 12.5 });

    card(s, 8.1, 1.75, 4.6, 3.7, MIST);
    head(s, "shield", "Why it is safe", 8.1, 1.75, SLATE);
    bullets(s, ["O sees α, a uniformly random group element: nothing about the id.",
      "R gets F_k only for ids it sends. Without k it cannot compute F_k for any other id.",
      "O answers exactly n_R^max queries per run (R pads with dummies) and logs them against a quota.",
      "A fresh k every run, so F_k values from different runs cannot be linked."],
    { x: 8.35, y: 2.6, w: 4.15, h: 2.7, fontSize: 12.5, paraSpaceAfter: 6 });

    card(s, 0.6, 5.65, 12.1, 1.1, "F7F9FC");
    txt(s, [{ text: "Why we need it: ", options: { bold: true } },
      { text: "identifiers such as social insurance numbers are a small, guessable space, so a plain hash could be inverted by trying every value. With the OPRF every evaluation needs O, so R cannot test ids offline, and the PSI messages carry no structure. Toy example: R holds 104, 221, 317, pads with dummy 9d1 to n_R^max = 4, and makes four calls." }],
    { x: 0.85, y: 5.7, w: 11.6, h: 1.0, fontSize: 12.5, valign: "middle" });
    s.addNotes("An oblivious pseudorandom function lets R compute a keyed hash of its identifiers under O's key without O seeing the identifiers and without R learning the key. We use 2HashDH, standardized in RFC 9497: hash the identifier into a group, raise it to the key k, and hash again. R blinds the group element with a random exponent r and sends alpha; O raises it to k; R removes r and applies the outer hash. The output is split into an item part for matching and a key part for opening labels. Alpha is uniformly random, so O learns nothing about the identifier. R only gets values for identifiers it sends, and cannot evaluate the function on anything else without k. R always sends exactly n_R^max queries, padding with dummies, and O logs them against a quota per researcher and project. We need this because identifiers are a small, guessable space: a plain hash could be inverted by brute force. The OPRF makes every membership test an online interaction with O. O draws a fresh key every run.");
  }

  // ---------- 4. labeled PSI ----------
  {
    const s = pres.addSlide();
    title(s, "Labeled PSI: who matches, and where", "R learns its match set I and positions π(i) = τ(j) in O's shuffled register; O learns only the padded sizes");

    const cols = [[EMBER_L, EMBER, "O prepares"], [MIST, SLATE, "R asks, O evaluates"], [RBLUE_L, RBLUE, "R decrypts"]];
    cols.forEach(([fill, col, hd], i) => {
      const x = 0.6 + i * 4.1;
      card(s, x, 1.75, 3.9, 2.85, fill);
      numHead(s, i + 1, hd, x, 1.75, col);
    });
    txt(s, "Pad the register to n̄_O with dummies and draw a fresh shuffle τ. Row j becomes the item F_k(id_j), labeled with its encrypted position. Hash the items into bins; per bin b:",
      { x: 0.85, y: 2.45, w: 3.45, h: 1.25, fontSize: 12 });
    latex(s, "ps_poly", 0.85, 3.85, { size: 11, maxW: 3.45 });
    txt(s, "R cuckoo-hashes its items, at most one per bin, encrypts them under its own BFV key, and sends their powers q, q², q⁴, … to keep the depth low. O evaluates every P_b and L_b under encryption and replies:",
      { x: 4.95, y: 2.45, w: 3.45, h: 1.45, fontSize: 12 });
    latex(s, "ps_reply", 4.95, 4.05, { size: 11, maxW: 3.45 });
    latex(s, "ps_decide", 9.05, 2.5, { size: 11, maxW: 3.45 });
    txt(s, "No match: P_b(q) ≠ 0 and the label value means nothing. The reply reveals nothing else about O's register.",
      { x: 9.05, y: 3.15, w: 3.45, h: 1.2, fontSize: 12 });

    card(s, 0.6, 4.8, 7.6, 1.95, "F7F9FC");
    txt(s, "Toy example", { x: 0.85, y: 4.9, w: 3, h: 0.3, fontSize: 13.5, bold: true, color: INK });
    s.addTable([
      [{ text: "O's padded row j", options: { bold: true, color: WHITE, fill: { color: INK } } },
        ...["150", "7e5", "221", "088", "399", "104"].map((t) => ({ text: t, options: { bold: true, color: WHITE, fill: { color: INK }, align: "center" } }))],
      [{ text: "position τ(j)", options: { bold: true } }, ...["1", "2", "3", "4", "5", "6"].map((t) => ({ text: t, options: { align: "center" } }))],
    ], Object.assign(tableOpts([1.75, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8]), { x: 0.85, y: 5.25, fontSize: 12, rowH: 0.3, fill: { color: WHITE } }));
    txt(s, "7e5 is O's dummy row. R queries 104, 221, 317 and its dummy 9d1: 104 matches with label 6 and 221 with label 3. So I = {104, 221}, π(104) = 6 and π(221) = 3.",
      { x: 0.85, y: 5.95, w: 7.1, h: 0.7, fontSize: 12 });

    card(s, 8.4, 4.8, 4.3, 1.95, MIST);
    txt(s, "What each party learns", { x: 8.65, y: 4.9, w: 3.9, h: 0.3, fontSize: 13.5, bold: true, color: INK });
    txt(s, [{ text: "O: ", options: { bold: true, color: EMBER } }, { text: "only n_R^max and n̄_O, both public.", options: { breakLine: true } },
      { text: "R: ", options: { bold: true, color: RBLUE } }, { text: "I and the positions, which are uniformly random because τ is fresh. R needs both: I for its block A, the positions to line up its upload." }],
    { x: 8.65, y: 5.25, w: 3.85, h: 1.4, fontSize: 12, paraSpaceAfter: 4 });
    s.addNotes("Labeled private set intersection, following Chen, Laine and Rindal and successors. O pads its register to the public size with dummy rows and draws a fresh secret shuffle tau. Each row becomes an item, its OPRF value, with a label: the row's position in the shuffle, encrypted under the row's label key. O hashes the items into bins and, for each bin, interpolates a membership polynomial whose roots are the items and a label polynomial that maps each item to its encrypted label. R cuckoo-hashes its own items so each bin holds at most one, encrypts them under its own BFV key, and sends powers of them so that O can evaluate high-degree polynomials at low depth. O evaluates both polynomials under encryption and replies. R decrypts: a zero membership value means a match, and then the label value decrypts, under the item's label key, to the row's position. In the toy example, 104 and 221 match with positions 6 and 3. O learns only the padded sizes. R learns the match set and the positions; the positions are uniformly random because the shuffle is fresh. R needs the match set to compute its own block A, and the positions to line up its data with O's rows in the upload.");
  }

  await writeDeck(pres, P("slides", "crypto_building_blocks.pptx"));
})();
